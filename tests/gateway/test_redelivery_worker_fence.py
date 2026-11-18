"""A lost person turn whose claim still has a worker that could act is never queued again (MR12).

A worker's tool steps reach the transcript only through its owner, so "no tool row above the claim's
mark" proves nothing while a worker execution of that claim is registered, running or (after the
next owner's boot fence) ``unknown``. Such a turn is not redelivered, and its worker is never written
``terminal`` by the redelivery: that write would remove the claim fence (``unknown_execution``) that
keeps a second run from racing the first. Only a worker proven ended (its own ``execution.finish``)
leaves the turn redeliverable, and a turn that never had a worker still runs again, HEAD then
FOLLOWER (at least once; a duplicate answer is acceptable).

Real SessionStore SQLite, real owner epochs and boots, the real reservation/adoption/finish writers and
an inert private child process standing in for the worker (it reads stdin until EOF and does nothing
else). The handler only records which message ran. Each case prints one ``WITNESS`` JSON line.
"""
import json
import subprocess
import sys

import pytest

from gateway.session_contract import SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError, claim_session_input
from tests.gateway.test_person_message_redelivery import _drained, _notices, _owner
from tests.gateway.test_unknown_automation_settles import Peer, _person_head, _runner, _statuses

REDELIVER = 'gateway.admission.redeliver.v1.'


def _inert_child():
    return subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _hello(process):
    import os
    import psutil
    from hermes_cli.process_identity import worker_birth
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)),
            'ancestors': [os.getpid()]}


def _end_child(child):
    """The inert child exits on stdin EOF; nothing signals it."""
    child.stdin.close()
    return child.wait(timeout=30)


async def _claimed_head(tmp_path, monkeypatch, executed):
    """Owner 1 admits a person's HEAD and FOLLOWER, claims HEAD and writes its user row (no tool row)."""
    from gateway.session_controls import AuthorityConnection
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    db.create_session('s', source='telegram')
    first = await _owner(runner, 'owner-1')
    first._schedule = lambda ref: None
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    head_id = await _person_head(first, actor, ref)
    follower = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = claim_session_input(db, epoch=first.epoch, session_id='s')
    assert claimed['admission_id'] == head_id
    db.append_message('s', 'user', 'HEAD')
    return store, runner, first, claimed, follower.admission_id


def _reserve(first, claimed, child, *, state):
    """The real owner-side reservation; ``running``: then the real authenticated adoption; ``finished``:
    then the worker's own real ``execution.finish``."""
    from gateway.session_admission import admission_fingerprint
    from gateway.session_worker_reservation import reserve_admission_worker
    from hermes_state_runtime import adopt_worker_execution, mutate_worker_execution
    scope = reserve_admission_worker(first, admission_id=claimed['admission_id'], process=child,
                                     principal_id=claimed['principal_id'], hello=_hello(child))
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    if state in ('running', 'finished'):
        secret = admission_fingerprint(canonical_target=scope['session_id'],
                                       payload=identity | {'principal': claimed['principal_id']})
        adopt_worker_execution(first.db, epoch=first.epoch, execution_id=scope['execution_id'],
                               session_id=scope['session_id'], generation=scope['generation'], adoption_secret=secret)
    if state == 'finished':
        mutate_worker_execution(first.db, epoch=first.epoch, execution_id=scope['execution_id'],
                                session_id=scope['session_id'], generation=scope['generation'], sequence=1,
                                operation='execution.finish', payload={})
    return scope


def _worker(db, head_id):
    row = db._read_one('SELECT status, owner_epoch FROM worker_executions WHERE execution_id=?',
                       ('admission-worker:' + head_id,))
    return None if row is None else row[0]


def _marker(db, head_id):
    row = db._read_one('SELECT value FROM state_meta WHERE key=?', (REDELIVER + head_id,))
    return None if row is None else json.loads(row[0])


def _claim_probe(authority):
    try:
        return repr(claim_session_input(authority.db, epoch=authority.epoch, session_id='s'))
    except RuntimeStoreError as exc:
        return 'RuntimeStoreError:' + exc.reason


def _witness(case, **observed):
    print('WITNESS ' + json.dumps({'case': case, **observed}, sort_keys=True), flush=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['registered', 'running'])
async def test_a_lost_turns_worker_from_before_boot_is_neither_replayed_nor_retired(tmp_path, monkeypatch, state):
    """The worker exists before the next owner boots. The boot's fence makes it ``unknown``; nothing
    owes or runs HEAD again and nothing writes its worker ``terminal``. The independent FOLLOWER
    completes without acquiring HEAD's reservation or changing its uncertainty."""
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child()
    try:
        _reserve(first, claimed, child, state=state)
        before_boot = _worker(first.db, head_id)
        second = await _owner(runner, 'owner-2')
        after_boot = _worker(second.db, head_id)
        marker = _marker(second.db, head_id)
        await _drained(second)
        generation = second.db._read_one("SELECT runtime_generation FROM sessions WHERE id='s'")[0]
        worker_before_probe = dict(second.db._read_one(
            'SELECT * FROM worker_executions WHERE execution_id=?', ('admission-worker:' + head_id,)))
        claim = _claim_probe(second)
        assert dict(second.db._read_one('SELECT * FROM worker_executions WHERE execution_id=?',
                                       ('admission-worker:' + head_id,))) == worker_before_probe
        assert second.db._read_one("SELECT runtime_generation FROM sessions WHERE id='s'")[0] == generation
        observed = dict(worker_before_boot=before_boot, worker_after_boot=after_boot, marker_after_boot=marker,
                        executed=list(executed), worker_after_drain=_worker(second.db, head_id),
                        head=_statuses(second.db)[head_id], follower=_statuses(second.db)[follower_id],
                        claim_after_drain=claim,
                        will_run_again_notice=any('will run again' in n for n in _notices(second.db)))
    finally:
        observed_exit = _end_child(child)
        store.close_all_db_handles()
    _witness('pre_boot_worker_' + state, child_exit=observed_exit, **observed)
    assert observed['worker_before_boot'] == state
    assert observed['worker_after_boot'] == 'unknown'
    assert observed['marker_after_boot'] is None
    assert observed['executed'] == ['FOLLOWER']
    assert observed['worker_after_drain'] == 'unknown'
    assert observed['head'] == ('unknown', None)
    assert observed['follower'] == ('terminal', 'completed')
    assert observed['claim_after_drain'] == 'None'  # queue empty; HEAD remains unknown and unclaimed
    assert observed['will_run_again_notice'] is False


@pytest.mark.asyncio
async def test_a_worker_that_appears_after_the_redelivery_was_owed_refuses_the_requeue(tmp_path, monkeypatch):
    """owe -> worker appears -> requeue: the boot's fence owed the no-worker turn, then a worker row of
    that lost claim appears before the requeue. The requeue refuses: the marker, the ``unknown`` HEAD
    and the worker stay exactly as they are; nothing runs."""
    import gateway.session_message_redelivery as redelivery
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    real = redelivery.redeliver_recovered_messages
    seen = {}

    def worker_appears_then_requeue(db, epoch):
        seen['marker_before_requeue'] = _marker(db, head_id)
        db._execute_write(lambda c: c.execute(
            "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
            "VALUES(?,?,'compute',?,?,'running',?)",
            ('admission-worker:' + head_id, 's', first.epoch, claimed['generation'], '0' * 64)))
        seen['queued'] = real(db, epoch)
        return seen['queued']
    monkeypatch.setattr(redelivery, 'redeliver_recovered_messages', worker_appears_then_requeue)
    try:
        second = await _owner(runner, 'owner-2')
        monkeypatch.setattr(redelivery, 'redeliver_recovered_messages', real)
        await _drained(second)
        observed = dict(marker_before_requeue=seen['marker_before_requeue'], queued_by_requeue=seen['queued'],
                        marker_after_requeue=_marker(second.db, head_id), executed=list(executed),
                        worker_after_drain=_worker(second.db, head_id), head=_statuses(second.db)[head_id],
                        follower=_statuses(second.db)[follower_id], claim_after_drain=_claim_probe(second))
    finally:
        store.close_all_db_handles()
    _witness('owe_worker_appears_requeue', **observed)
    assert observed['marker_before_requeue'] == {'epoch': first.epoch, 'generation': claimed['generation'],
                                                 'status': 'unknown'}
    assert observed['queued_by_requeue'] == 0
    assert observed['marker_after_requeue'] == observed['marker_before_requeue']
    assert observed['executed'] == []
    assert observed['worker_after_drain'] == 'running'
    assert observed['head'] == ('unknown', None)
    assert observed['follower'] == ('queued', None)
    assert observed['claim_after_drain'] == 'RuntimeStoreError:unknown_execution'


@pytest.mark.asyncio
@pytest.mark.parametrize('worker', ['none', 'finished'])
async def test_a_lost_turn_with_no_worker_or_a_finished_one_still_runs_again(tmp_path, monkeypatch, worker):
    """Positive control: no worker ever reserved, or one that wrote its own ``execution.finish`` before
    the owner died (it can no longer act; a later write of it is refused). HEAD runs again, then FOLLOWER."""
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child() if worker == 'finished' else None
    try:
        if child is not None:
            _reserve(first, claimed, child, state='finished')
        before_boot = _worker(first.db, head_id)
        second = await _owner(runner, 'owner-2')
        await _drained(second)
        observed = dict(worker_before_boot=before_boot, executed=list(executed),
                        head=_statuses(second.db)[head_id], follower=_statuses(second.db)[follower_id],
                        will_run_again_notice=any('will run again' in n for n in _notices(second.db)))
    finally:
        child_exit = _end_child(child) if child is not None else None
        store.close_all_db_handles()
    _witness('positive_worker_' + worker, child_exit=child_exit, **observed)
    assert observed['worker_before_boot'] == (None if worker == 'none' else 'terminal')
    assert observed['executed'] == ['HEAD', 'FOLLOWER']
    assert observed['head'] == ('terminal', 'completed')
    assert observed['follower'] == ('terminal', 'completed')
    assert observed['will_run_again_notice'] is True


def _shutdown_capture():
    from gateway.run import _INTERRUPT_REASON_GATEWAY_SHUTDOWN
    return {'result': {'final_response': '', 'interrupted': True,
                       'interrupt_message': _INTERRUPT_REASON_GATEWAY_SHUTDOWN}, 'usage': {}}


@pytest.mark.asyncio
async def test_an_owner_stop_settlement_with_a_live_worker_owes_no_redelivery(tmp_path, monkeypatch):
    """Owner-stop settlement (``finish_result``) of a claim whose worker is still registered: no redelivery
    is owed (the settlement used to retire the worker first and then find it ended). The turn settles
    ``interrupted`` with the stop's own notice; FOLLOWER runs at the next owner, HEAD does not."""
    from gateway.session_results import finish_result
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child()
    try:
        _reserve(first, claimed, child, state='registered')
        settled, _ = finish_result(first.db, epoch=first.epoch, row=claimed, response='', outcome='interrupted',
                                   result=_shutdown_capture())
        marker = _marker(first.db, head_id)
        second = await _owner(runner, 'owner-2')
        await _drained(second)
        observed = dict(settled=(settled['status'], settled['outcome']), notice=settled.get('owner_stop_notice'),
                        marker_after_settlement=marker, executed=list(executed),
                        head=_statuses(second.db)[head_id], follower=_statuses(second.db)[follower_id])
    finally:
        child_exit = _end_child(child)
        store.close_all_db_handles()
    _witness('owner_stop_live_worker', child_exit=child_exit, **observed)
    assert observed['settled'] == ('terminal', 'interrupted')
    assert observed['marker_after_settlement'] is None
    assert 'will run again' not in (observed['notice'] or '')
    assert observed['executed'] == ['FOLLOWER']
    assert observed['head'] == ('terminal', 'interrupted')


@pytest.mark.asyncio
async def test_a_failed_owner_stop_fence_with_a_live_worker_owes_no_redelivery(tmp_path, monkeypatch):
    """The failed-settlement fallback (``_mark_unknown``) of an owner-stopped claim whose worker is still
    registered owes no HEAD replay; the next owner delivers FOLLOWER without retiring HEAD's worker."""
    from types import SimpleNamespace
    from gateway.session_settlement_recovery import _mark_unknown
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child()
    try:
        _reserve(first, claimed, child, state='registered')
        fence = SimpleNamespace(db=first.db, epoch=first.epoch, sessions={},
                                pending_results={head_id: _shutdown_capture()})
        fenced = _mark_unknown(fence, claimed)
        marker = _marker(first.db, head_id)
        second = await _owner(runner, 'owner-2')
        await _drained(second)
        observed = dict(fenced=fenced, marker_after_fence=marker, executed=list(executed),
                        worker_after_drain=_worker(second.db, head_id), head=_statuses(second.db)[head_id],
                        follower=_statuses(second.db)[follower_id])
    finally:
        child_exit = _end_child(child)
        store.close_all_db_handles()
    _witness('failed_owner_stop_live_worker', child_exit=child_exit, **observed)
    assert observed['fenced'] == 'unknown'
    assert observed['marker_after_fence'] is None
    assert observed['executed'] == ['FOLLOWER']
    assert observed['worker_after_drain'] == 'unknown'
    assert observed['head'] == ('unknown', None)
    assert observed['follower'] == ('terminal', 'completed')


@pytest.mark.asyncio
@pytest.mark.parametrize('proof', ['tool_intent_without_result', 'missing_mark', 'stale_mark', 'invalid_lineage'])
async def test_unproven_no_worker_turns_stay_excluded(tmp_path, monkeypatch, proof):
    """Exclusions in force with no worker at all: a tool call with no result yet, an absent or stale
    claim mark, and an unreadable claim scope (invalid lineage) are never queued again as HEAD.
    None of those HEAD exclusions revokes delivery of the independently accepted FOLLOWER."""
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    db = first.db
    if proof == 'tool_intent_without_result':
        db.append_message('s', 'assistant', None, tool_calls=[{'id': 'c1', 'type': 'function', 'function': {
            'name': 'terminal', 'arguments': '{"command": "touch effect"}'}}])
    elif proof == 'missing_mark':
        db._execute_write(lambda c: c.execute('DELETE FROM state_meta WHERE key=?',
                                              ('gateway.admission.claim_mark.v1.' + head_id,)))
    elif proof == 'stale_mark':
        db._execute_write(lambda c: c.execute(
            'UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps({'epoch': first.epoch, 'generation': claimed['generation'] + 7, 'mark': 0}),
             'gateway.admission.claim_mark.v1.' + head_id)))
    else:
        db._execute_write(lambda c: c.execute("UPDATE session_admissions SET lineage_json='{not json' "
                                              "WHERE admission_id=?", (head_id,)))
    try:
        second = await _owner(runner, 'owner-2')
        marker = _marker(second.db, head_id)
        await _drained(second)
        observed = dict(marker_after_boot=marker, executed=list(executed), head=_statuses(second.db)[head_id],
                        follower=_statuses(second.db)[follower_id])
    finally:
        store.close_all_db_handles()
    _witness('exclusion_' + proof, **observed)
    assert observed['marker_after_boot'] is None
    assert observed['executed'] == ['FOLLOWER']
    assert observed['head'] == ('unknown', None)
    assert observed['follower'] == ('terminal', 'completed')
