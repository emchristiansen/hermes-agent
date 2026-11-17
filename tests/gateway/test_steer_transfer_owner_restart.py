"""A steer transfer survives its owner: at least once, never stranded (NC10-OWN-01/02).

- A transfer is ``terminal/steering`` bound to a claim. When its owner restarts before the claim
  settled it, only a transfer whose text its managed child confirmed consumed
  (``confirm_steer_transfer``) settles ``interrupted``; every other one is queued again in its own
  FIFO place and runs as its own turn once the lost claim is resolved: a duplicate, never a loss.
  The lost claim itself is never replayed.
- A refused ``prompt.resolve_unknown`` (the claim is still running) leaves the running agent's
  buffered steer and the transfer registration untouched: the running turn still drains it.

Real authority, SQLite store and owner recovery; only the finite execution is a fixture.
"""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import session_busy_ingress, session_finite
from gateway import session_managed_worker as managed
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import Submission
from gateway.session_worker_reservation import lose_admission_worker
import hermes_state_runtime as runtime
from hermes_state_runtime import (
    CONFIRMED_STEER_PREFIX, begin_runtime_epoch, claim_session_input, confirm_steer_transfer,
    find_session_admission, get_session_admission, recover_session_inputs, register_worker_execution,
    steer_session_input,
)
from tests.gateway.test_prompt_queued_controls import _call, _connection, _ledger, interactive
from tests.gateway.test_prompt_queued_transfer import BufferedAgent
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit


def _state(db, admission):
    row = get_session_admission(db, admission_id=admission.admission_id)
    return row['status'], row['outcome']


@pytest.mark.asyncio
@pytest.mark.parametrize('handover', ['confirmed', 'confirmation-reply-lost', 'confirmation-unwritten',
                                      'commit-then-error', 'not-consumed'])
async def test_a_transfer_the_lost_claim_never_settled_survives_an_owner_restart(tmp_path, monkeypatch, handover):
    """The child's correction transfer, its claim lost and its settlement write failed, then the owner
    restarts: confirmed consumed and recorded so, it never runs again; otherwise it runs once the lost
    turn is resolved, before the turn queued behind it. A confirmation whose own write failed is not
    recorded, so it runs again (a duplicate); one that committed before its reply was lost counts.
    The lost turn's own input never runs again."""
    db, authority = _authority(tmp_path, monkeypatch)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    executed, heads = [], []
    lost = asyncio.Event()
    correction_id = None

    async def execute(owner, ref, row):
        if row['request_id'] != 'head':
            executed.append(row['payload']['text'])
            return 'done'
        heads.append(row['payload']['text'])
        generation = row['generation']
        worker = managed.ManagedWorker(SimpleNamespace(pid=0))
        worker.kept['c1'] = managed.KeptCorrection('c1', correction_id)
        owner.adopted[ref.session_id] = (generation, worker)
        register_worker_execution(db, epoch=owner.epoch, execution_id='execution', session_id=ref.session_id,
                                  generation=generation, kind='compute', adoption_secret='test-only-secret')
        real_handover = runtime.steer_session_input
        if handover == 'commit-then-error':
            def commit_then_error(*args, **kwargs):
                real_handover(*args, **kwargs)
                raise OSError('hand-over reply lost after commit')
            monkeypatch.setattr(runtime, 'steer_session_input', commit_then_error)
        real_confirm = runtime.confirm_steer_transfer
        if handover in ('confirmation-reply-lost', 'confirmation-unwritten'):
            def confirmation_fails(*args, **kwargs):
                if handover == 'confirmation-reply-lost':
                    real_confirm(*args, **kwargs)
                raise sqlite3.OperationalError('database or disk is full')
            monkeypatch.setattr(runtime, 'confirm_steer_transfer', confirmation_fails)
        if handover != 'not-consumed':
            await managed.settle_consumed(owner, ref, row, worker, ['c1'])
        monkeypatch.setattr(runtime, 'steer_session_input', real_handover)
        monkeypatch.setattr(runtime, 'confirm_steer_transfer', real_confirm)
        lose_admission_worker(owner, row, {'epoch': owner.epoch, 'generation': generation,
                                           'execution_id': 'execution'})
        real_settle = runtime.settle_steer_transfers

        def fail_once(*args, **kwargs):
            monkeypatch.setattr(runtime, 'settle_steer_transfers', real_settle)
            raise sqlite3.OperationalError('database or disk is full')  # not retried by the store
        if handover != 'not-consumed':
            monkeypatch.setattr(runtime, 'settle_steer_transfers', fail_once)
        await managed._settle_transferred(owner, ref, row, worker)
        lost.set()
        raise managed.ManagedExecutionUnknown('child teardown complete')

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        head = await authority.submit(ACTOR, Submission('head', REF, {'text': 'ORIGINAL'}, 'queue'))
        correction = await authority.submit(ACTOR, Submission('correction:c1', REF, {'text': 'CORRECTION'}, 'queue'))
        correction_id = correction.admission_id
        follower = await authority.submit(ACTOR, Submission('follower', REF, {'text': 'FOLLOWER'}, 'queue'))
        monkeypatch.setattr(authority, '_schedule', schedule)
        schedule(REF)
        await asyncio.wait_for(lost.wait(), 10)
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        row = get_session_admission(db, admission_id=head.admission_id)
        assert row['status'] == 'unknown'
        if handover != 'not-consumed':
            assert _state(db, correction) == ('terminal', 'steering')  # still in transfer
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                                 _adapter_for_source=lambda source: None, session_store=SimpleNamespace())
        fresh = await initialize_session_authority(runner, profile_id='owned', instance_id='fresh-owner', db=db)
        fresh.sessions['s'] = LiveSession(authority.sessions['s'].source, 'route')
        recorded = handover in ('confirmed', 'confirmation-reply-lost')
        assert _state(db, correction) == (('terminal', 'interrupted') if recorded else ('queued', None))
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (CONFIRMED_STEER_PREFIX + correction_id,)) is None
        await fresh.resolve_unknown(ACTOR, REF, head.admission_id, row['generation'])
        await asyncio.wait_for(fresh.sessions['s'].task, 10)
    assert _state(db, head) == ('terminal', 'interrupted')
    assert _state(db, follower) == ('terminal', 'completed')
    assert _state(db, correction) == ('terminal', 'interrupted' if recorded else 'completed')
    assert heads == ['ORIGINAL']
    assert executed == (['FOLLOWER'] if recorded else ['CORRECTION', 'FOLLOWER'])
    assert not fresh.steer_transfers


@pytest.mark.parametrize('mark', [None, 'this-claim', 'other-claim'])
def test_an_owner_restart_settles_only_a_confirmed_transfer_interrupted(tmp_path, monkeypatch, mark):
    """The store rule itself: only a confirmation recorded for the transfer's own claim counts."""
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        for request_id in ('running', 'follow-up'):
            runtime.admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s',
                                        request_id=request_id, payload={'text': request_id})
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        follow = find_session_admission(db, principal_id='human', session_id='s', request_id='follow-up')
        steer_session_input(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                            generation=running['generation'])
        if mark is not None:
            assert confirm_steer_transfer(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                                          generation=running['generation'])
        if mark == 'other-claim':
            db._execute_write(lambda conn: conn.execute(
                'UPDATE state_meta SET value=? WHERE key=?',
                (runtime._json({'epoch': authority.epoch, 'generation': running['generation'] + 1}),
                 CONFIRMED_STEER_PREFIX + follow['admission_id'])))
        recover_session_inputs(db, epoch=begin_runtime_epoch(db, instance_id='next'))
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?',
                            (CONFIRMED_STEER_PREFIX + follow['admission_id'],)) is None
    expected = ('follow-up', 'terminal', 'interrupted') if mark == 'this-claim' else ('follow-up', 'queued', None)
    assert _ledger(db) == [('running', 'unknown', None), expected]


def test_a_confirmation_is_refused_for_a_row_no_longer_in_that_claims_transfer(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        runtime.admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s',
                                    request_id='running', payload={'text': 'running'})
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        assert not confirm_steer_transfer(db, epoch=authority.epoch, admission_id=running['admission_id'],
                                          generation=running['generation'])
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?',
                            (CONFIRMED_STEER_PREFIX + running['admission_id'],)) is None


@pytest.mark.asyncio
async def test_a_refused_resolution_of_a_running_claim_leaves_its_buffered_steer(tmp_path, monkeypatch):
    """``prompt.resolve_unknown`` of the still-started claim is refused (``stale_generation``) and has
    no effect: the agent keeps the accepted steer, the registration stays delivered, and the running
    turn drains it; it never runs again as its own turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(session_busy_ingress, 'steer_soon', lambda *args: None)
    agent = BufferedAgent()
    claimed, release = asyncio.Event(), asyncio.Event()
    ran, drained = [], []

    async def execute(owner, ref, row):
        ran.append(row['payload']['text'])
        owner.adopt_agent(ref.session_id, row['generation'], agent)
        if row['request_id'] == 'running':
            claimed.set()
            await asyncio.wait_for(release.wait(), 10)
            drained.append(agent._drain_pending_steer())
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        head = await _submit(authority, 'running')
        await asyncio.wait_for(claimed.wait(), 10)
        queued = await _submit(authority, 'correction', 'ACTIVE_CORRECTION')
        conn = _connection(authority)
        steer = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' not in steer, steer
        transfers = repr(authority.steer_transfers)
        try:
            refused = await _call(conn, 'prompt.resolve_unknown', admission_id=head.admission_id,
                                  execution_generation=authority.executing['s'])
            assert refused['error']['data']['reason'] == 'stale_generation', refused
            assert agent._pending_steer == 'ACTIVE_CORRECTION'
            assert repr(authority.steer_transfers) == transfers
        finally:
            release.set()
            await asyncio.wait_for(authority.sessions['s'].task, 10)
    assert drained == ['ACTIVE_CORRECTION']
    assert ran == ['running']
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('correction', 'terminal', 'steered')]
