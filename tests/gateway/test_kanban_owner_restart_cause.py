"""A card attempt lost because its owner ended is recorded as such, at the loss, before it settles.

The owner binds the cause where it finds the managed worker lost (``owner_ending`` then), and
records it before the admission becomes terminal; boot recovery records only rows its predecessor
left running. A loss under a live owner — even one whose stop begins while the loss is being fenced
— is not an owner restart. Real SQLite stores and a real managed child; only the recorder is a spy.
"""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef

# A child that says hello, then answers its first frame with garbage: the owner loses it.
CHILD = """import json, os, sys, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
for line in sys.stdin.buffer:
    out.write(b'{"type": "bogus"}\\n')
    break
"""


def _lose_managed_worker(tmp_path, monkeypatch, *, ending_before, ending_during_fence=False):
    """Run one managed turn whose worker is lost; returns the spy's calls as admission statuses."""
    from gateway import session_managed_worker as managed
    from gateway import session_unattended_unknown as unattended
    from gateway import session_worker_reservation as reservation
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, get_session_admission

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD], **kw)
            if ending_before:
                # The stop signal reaches the owner while its worker runs (before the loss is found),
                # as when a service stop signals the owner and its worker together.
                runner._stop_requested_by_signal = True
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    calls = []
    monkeypatch.setattr(unattended, 'record_owner_restart', lambda db_, row: calls.append(
        get_session_admission(db_, admission_id=row['admission_id'])['status']))
    if ending_during_fence:
        original = reservation.lose_admission_worker

        def stop_begins_while_fencing(*args):
            runner._draining = True  # the owner's stop begins after the loss was found
            return original(*args)
        monkeypatch.setattr(reservation, 'lose_admission_worker', stop_begins_while_fencing)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
    with db:
        asyncio.run(scenario())
    return calls


@pytest.mark.platforms('linux')
def test_loss_while_owner_stopping_records_cause_before_settlement(tmp_path, monkeypatch):
    # Recorded once, while the admission is still only fenced ``unknown`` (never after terminal).
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=True) == ['unknown']


@pytest.mark.platforms('linux')
def test_loss_under_live_owner_records_nothing(tmp_path, monkeypatch):
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=False) == []


@pytest.mark.platforms('linux')
def test_stop_beginning_after_the_loss_is_found_records_nothing(tmp_path, monkeypatch):
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=False, ending_during_fence=True) == []


def test_boot_records_only_rows_the_predecessor_left_running(tmp_path, monkeypatch):
    from gateway import session_unattended_unknown as unattended
    from hermes_state import SessionDB
    from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                      get_session_admission, recover_session_inputs)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    card = {'board': 'b', 'task_id': 't', 'run_id': 1}
    ids = {}
    for name in ('running', 'fenced'):
        db.create_session(name, source='test')
    first = begin_runtime_epoch(db, instance_id='one')
    for name in ('running', 'fenced'):
        ids[name] = admit_session_input(db, epoch=first, principal_id='kanban', session_id=name,
                                        request_id=name, payload={'text': 'work', 'kanban_card_v1': card})['admission_id']
        claim_session_input(db, epoch=first, session_id=name)
    # The predecessor itself fenced one row (a worker lost under it), then died with the other running.
    db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?",
                                                (ids['fenced'],)))
    inflight = unattended.inflight_admissions(db)
    assert inflight == {ids['running']}
    second = begin_runtime_epoch(db, instance_id='two')
    recover_session_inputs(db, epoch=second)
    calls = []
    monkeypatch.setattr(unattended, 'record_owner_restart', lambda db_, row: calls.append(
        (row['admission_id'], get_session_admission(db_, admission_id=row['admission_id'])['status'])))
    assert unattended.settle_recovered_unattended(db, second, inflight) == 2
    assert calls == [(ids['running'], 'unknown')]  # before its terminal write, and only that row
    assert {get_session_admission(db, admission_id=i)['outcome'] for i in ids.values()} == {'interrupted'}
