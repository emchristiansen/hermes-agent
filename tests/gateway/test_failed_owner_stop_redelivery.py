"""A failed owner-stop settlement keeps its qualified no-tool message owed (MR-GATEWAY20).

Real submit/drain -> managed pre-spawn stop -> finalizer -> SQLite fallback -> healthy boot/drain.
Only policy/context expansion and the ordinary answering handler are inert local seams; no worker,
provider or tool is executed. Tool controls write evidence rows, never repeat their business effect.
"""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import session_authority as sa
from gateway import session_managed_worker as managed
from gateway import session_settlement_recovery as recovery
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from gateway.session_message_redelivery import REDELIVER_PREFIX
from hermes_state_runtime import CLAIM_MARK_PREFIX, RuntimeStoreError, get_session_admission
from tests.gateway.test_unknown_automation_settles import Peer, _runner


async def _queued_pair(tmp_path, monkeypatch):
    executed = []
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    db.create_session('s', source='telegram')
    first = await sa.initialize_session_authority(runner, profile_id='p', instance_id='first')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='inert')
    first.sessions['s'] = sa.LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    # Accept both before starting the drain; this is not a replacement for the claim or finalizer.
    first._schedule = lambda ref: None
    head = await first.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))
    follow = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    del first._schedule
    return store, runner, first, ref, source, (head.admission_id, follow.admission_id), executed


def _marker(db, admission_id):
    saved = db._read_one('SELECT value FROM state_meta WHERE key=?', (REDELIVER_PREFIX + admission_id,))
    return json.loads(saved[0]) if saved else None


def _stop_during_expansion(monkeypatch, first, *, tool=False, cause='Gateway shutting down', proof='valid'):
    # Select an inert managed policy for the first owner only. Its real pre-spawn producer is run.
    monkeypatch.setattr(managed, 'managed_policy', lambda owner, ref: SimpleNamespace(cwd=str(first.db.db_path.parent))
                        if owner is first else None)

    async def expand(owner, row, policy):
        owner.runner._draining = True
        if tool:
            owner.db.append_message('s', 'tool', 'inert already-recorded effect',
                                    tool_call_id='c', tool_name='terminal')
        key = CLAIM_MARK_PREFIX + row['admission_id']
        if proof == 'absent':
            owner.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (key,)))
        elif proof != 'valid':
            def write(conn):
                mark = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
                if proof == 'stale':
                    mark['generation'] += 1
                elif proof == 'boolean':
                    mark['mark'] = False
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(mark), key))
            owner.db._execute_write(write)
        first.pending_stops['s'] = row['generation']
        if cause:
            first.pending_stop_causes['s'] = (row['generation'], cause)
        return row['payload']['text'], None
    monkeypatch.setattr('gateway.session_context_refs.managed_turn_text', expand)


async def _healthy_drain(runner, source, ref):
    runner._draining = False
    second = await sa.initialize_session_authority(runner, profile_id='p', instance_id='healthy-next')
    second.sessions['s'] = sa.LiveSession(source, 's')
    second._schedule(ref)
    await second.sessions['s'].task
    if getattr(second, '_media_release_task', None):
        await second._media_release_task
    return second


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none', 'function', 'sqlite'])
@pytest.mark.parametrize('tool', [False, True], ids=['no-tool', 'tool-evidence'])
async def test_owner_stop_settlement_failure_preserves_head_and_follower(tmp_path, monkeypatch, caplog, failure, tool):
    from gateway import session_results
    store, runner, first, ref, source, ids, executed = await _queued_pair(tmp_path, monkeypatch)
    db = store._db
    _stop_during_expansion(monkeypatch, first, tool=tool)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0, 0))
    original = session_results.finish_result
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(kwargs['row']['admission_id'])
        raise sqlite3.OperationalError('private fixture terminal refusal')
    if failure == 'function':
        monkeypatch.setattr(session_results, 'finish_result', unavailable)
    elif failure == 'sqlite':
        db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER reject_terminal BEFORE UPDATE OF status ON session_admissions "
            "WHEN NEW.status='terminal' BEGIN SELECT RAISE(ABORT, 'private fixture terminal refusal'); END"))
    try:
        await first._drain(ref)
        if getattr(first, '_media_release_task', None):
            await first._media_release_task
        before = get_session_admission(db, admission_id=ids[0])
        owed = _marker(db, ids[0])
        assert before['status'] == ('terminal' if failure == 'none' else 'unknown')
        if tool:
            assert owed is None
        else:
            assert owed == {'epoch': first.epoch, 'generation': before['generation'], 'status': before['status']}
        if failure == 'function':
            assert attempts == [ids[0]] * 3
        elif failure == 'sqlite':
            assert sum('retrying in' in record.message for record in caplog.records) == 2
            assert any(isinstance(record.exc_info[1], sqlite3.IntegrityError)
                       for record in caplog.records if record.exc_info)
        monkeypatch.setattr(session_results, 'finish_result', original)
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER IF EXISTS reject_terminal'))
        await _healthy_drain(runner, source, ref)
        states = [get_session_admission(db, admission_id=aid)['status'] for aid in ids]
        if tool:
            assert executed == (['FOLLOWER'] if failure == 'none' else [])
            assert states == (['terminal', 'terminal'] if failure == 'none' else ['unknown', 'queued'])
        else:
            assert executed == ['HEAD', 'FOLLOWER'] and states == ['terminal', 'terminal']
        assert _marker(db, ids[0]) is None
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('proof,cause', [('absent', 'Gateway shutting down'), ('stale', 'Gateway shutting down'),
                                       ('boolean', 'Gateway shutting down'), ('valid', None),
                                       ('valid', 'Gateway restarting')])
async def test_failed_stop_requires_this_claims_proof_and_owner_cause(tmp_path, monkeypatch, proof, cause):
    store, runner, first, ref, source, ids, executed = await _queued_pair(tmp_path, monkeypatch)
    db = store._db
    _stop_during_expansion(monkeypatch, first, cause=cause, proof=proof)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0, 0))
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_terminal BEFORE UPDATE OF status ON session_admissions "
        "WHEN NEW.status='terminal' BEGIN SELECT RAISE(ABORT, 'private fixture terminal refusal'); END"))
    try:
        await first._drain(ref)
        eligible = proof == 'valid' and cause is not None
        assert bool(_marker(db, ids[0])) is eligible
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_terminal'))
        await _healthy_drain(runner, source, ref)
        assert executed == (['HEAD', 'FOLLOWER'] if eligible else [])
        assert [get_session_admission(db, admission_id=aid)['status'] for aid in ids] == (
            ['terminal', 'terminal'] if eligible else ['unknown', 'queued'])
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_proof', ['{', '[]'])
async def test_invalid_proof_refuses_fallback_without_claiming_persistence(tmp_path, monkeypatch, bad_proof):
    from hermes_state_runtime import claim_session_input
    store, _runner, first, _ref, _source, ids, executed = await _queued_pair(tmp_path, monkeypatch)
    db = store._db
    try:
        row = claim_session_input(db, epoch=first.epoch, session_id='s')
        first.pending_results[ids[0]] = {'result': {'interrupted': True, 'interrupt_message': 'Gateway shutting down'}}
        db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                                                   (bad_proof, CLAIM_MARK_PREFIX + ids[0])))
        with pytest.raises(RuntimeStoreError, match='storage_unavailable') as error:
            await recovery.recover_failed_settlement(first, row)
        assert error.value.field == 'claim_mark'
        assert get_session_admission(db, admission_id=ids[0])['status'] == 'started'
        assert _marker(db, ids[0]) is None and ids[0] in first.pending_results and executed == []
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_fallback_marker_write_failure_rolls_back_unknown_and_preserves_result(tmp_path, monkeypatch):
    from hermes_state_runtime import claim_session_input
    store, runner, first, ref, source, ids, executed = await _queued_pair(tmp_path, monkeypatch)
    db = store._db
    try:
        row = claim_session_input(db, epoch=first.epoch, session_id='s')
        first.pending_results[ids[0]] = {'result': {'interrupted': True, 'interrupt_message': 'Gateway shutting down'}}
        db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER reject_marker BEFORE INSERT ON state_meta "
            "WHEN NEW.key LIKE 'gateway.admission.redeliver.v1.%' "
            "BEGIN SELECT RAISE(ABORT, 'private fixture marker refusal'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='marker refusal'):
            recovery._mark_unknown(first, row)
        assert get_session_admission(db, admission_id=ids[0])['status'] == 'started'
        assert _marker(db, ids[0]) is None and ids[0] in first.pending_results
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_marker'))
        assert await recovery.recover_failed_settlement(first, row) == 'unknown'
        assert _marker(db, ids[0]) == {'epoch': first.epoch, 'generation': row['generation'], 'status': 'unknown'}
        await _healthy_drain(runner, source, ref)
        assert executed == ['HEAD', 'FOLLOWER']
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('fence', ['epoch', 'generation', 'terminal', 'already-unknown', 'completed-result'])
async def test_fallback_does_not_widen_existing_claim_result_or_unknown_fences(tmp_path, monkeypatch, fence):
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input
    store, _runner, first, _ref, _source, ids, executed = await _queued_pair(tmp_path, monkeypatch)
    db = store._db
    try:
        row = claim_session_input(db, epoch=first.epoch, session_id='s')
        first.pending_results[ids[0]] = {'result': {'interrupted': True, 'interrupt_message': 'Gateway shutting down'}}
        if fence == 'epoch':
            begin_runtime_epoch(db, instance_id='moved-owner')
        elif fence == 'generation':
            row = dict(row, generation=row['generation'] + 1)
        elif fence == 'terminal':
            settle_session_input(db, epoch=first.epoch, admission_id=ids[0], generation=row['generation'],
                                 outcome='completed', result={'result': {'final_response': 'COMMITTED'}})
        elif fence == 'already-unknown':
            db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (ids[0],)))
        else:
            first.pending_results[ids[0]] = {'result': {'completed': True, 'final_response': 'FINISHED'}}
        if fence in ('epoch', 'generation'):
            with pytest.raises(RuntimeStoreError, match='stale_' + fence):
                await recovery.recover_failed_settlement(first, row)
        else:
            assert await recovery.recover_failed_settlement(first, row) == ('terminal' if fence == 'terminal' else 'unknown')
        assert _marker(db, ids[0]) is None and executed == []
    finally:
        store.close_all_db_handles()
