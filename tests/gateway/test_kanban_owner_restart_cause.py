"""A card attempt lost because its owner ended carries that cause from its loss fence.

The owner binds the cause where it finds the managed worker lost (``owner_ending`` then) and commits
it with the ``unknown`` fence (``OWNER_RESTART``); boot recovery marks only card rows its predecessor
left running. A loss under a live owner — even one whose stop begins while the loss is being fenced
— is not an owner restart. Real SQLite stores and a real managed child.
"""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from hermes_state_runtime import OWNER_RESTART

# A child that says hello, then answers its first frame with garbage: the owner loses it.
CHILD = """import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
for line in sys.stdin.buffer:
    out.write(b'{"type": "bogus"}\\n')
    break
"""


def _lose_managed_worker(tmp_path, monkeypatch, *, ending_before, ending_during_fence=False):
    """Run one card turn whose managed worker is lost; returns its fenced ``(status, outcome)``."""
    from gateway import session_managed_worker as managed
    from gateway import session_worker_reservation as reservation
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, get_session_admission

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
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
    if ending_during_fence:
        original = reservation.lose_admission_worker

        def stop_begins_while_fencing(*args):
            runner._draining = True  # the owner's stop begins after the loss was found
            return original(*args)
        monkeypatch.setattr(reservation, 'lose_admission_worker', stop_begins_while_fencing)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s', request_id='turn',
                            payload={'text': 'go', 'kanban_card_v1': {'board': 'b', 'task_id': 't', 'run_id': 1}})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
        return row['admission_id']
    with db:
        lost = get_session_admission(db, admission_id=asyncio.run(scenario()))
    return lost['status'], lost['outcome']


@pytest.mark.platforms('linux')
def test_loss_while_owner_stopping_commits_the_cause_with_the_fence(tmp_path, monkeypatch):
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=True) == ('unknown', OWNER_RESTART)


@pytest.mark.platforms('linux')
def test_loss_under_live_owner_carries_no_cause(tmp_path, monkeypatch):
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=False) == ('unknown', None)


@pytest.mark.platforms('linux')
def test_stop_beginning_after_the_loss_is_found_carries_no_cause(tmp_path, monkeypatch):
    assert _lose_managed_worker(tmp_path, monkeypatch, ending_before=False, ending_during_fence=True) == ('unknown', None)


def test_boot_marks_only_card_rows_the_predecessor_left_running(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                      get_session_admission, recover_session_inputs)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    card = {'board': 'b', 'task_id': 't', 'run_id': 1}
    payloads = {'running': {'text': 'work', 'kanban_card_v1': card}, 'fenced': {'text': 'work', 'kanban_card_v1': card},
                'person': {'text': 'hello'}}
    ids = {}
    first = begin_runtime_epoch(db, instance_id='one')
    for name, payload in payloads.items():
        db.create_session(name, source='test')
        ids[name] = admit_session_input(db, epoch=first, principal_id='kanban', session_id=name,
                                        request_id=name, payload=payload)['admission_id']
        claim_session_input(db, epoch=first, session_id=name)
    # The predecessor itself fenced one row (a worker lost under it), then died with the others running.
    db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?",
                                                (ids['fenced'],)))
    recover_session_inputs(db, epoch=begin_runtime_epoch(db, instance_id='two'))
    fenced = {name: get_session_admission(db, admission_id=i) for name, i in ids.items()}
    assert {name: (row['status'], row['outcome']) for name, row in fenced.items()} == {
        'running': ('unknown', OWNER_RESTART), 'fenced': ('unknown', None), 'person': ('unknown', None)}
