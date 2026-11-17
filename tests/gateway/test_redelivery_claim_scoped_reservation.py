"""A redelivered claim reserves its OWN worker execution; the earlier claim's stays fenced (MR21).

A person's no-tool HEAD whose first worker durably wrote ``execution.finish`` is queued again by the
next owner and claimed at a later generation with the SAME admission id. Its managed worker is
reserved under a claim-scoped execution id (``admission-worker:<id>:g<generation>``) instead of
colliding with the first claim's terminal row. Business-once stays keyed on the admission and its
transcript: a turn that reached a tool, or whose worker could still act, is never claimed again, so
no second reservation is reachable for it. The first claim's row, receipts and adoption digest are
only read; its late writes and adoption stay refused.

Real SessionStore SQLite, real owner epochs and boots, real claim/reservation/adoption/finish writers
and the real ``execute_managed`` consumer. Substituted seams, all inert: policy/context expansion
(``managed_turn_text``), the credential-bearing bootstrap frame builder (``_bootstrap``: no policy or
credential is resolved), the spawn/pipe transport (an EOF-only child stands in for the interpreter)
and the drain's execution stamp. Each case prints one ``WITNESS`` JSON line.
"""
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import session_managed_worker as managed
from gateway.session_contract import SessionRef
from gateway.session_worker_reservation import (_claim_execution_id, admission_worker_execution_ids,
                                                owner_worker_subject, reserve_admission_worker)
from hermes_state_runtime import (RuntimeStoreError, adopt_worker_execution, claim_session_input,
                                  mutate_worker_execution)
from tests.gateway.test_person_message_redelivery import _drained, _owner
from tests.gateway.test_redelivery_worker_fence import (_claim_probe, _claimed_head, _end_child, _hello,
                                                        _inert_child, _reserve, _witness)
from tests.gateway.test_unknown_automation_settles import _statuses


def _rows(db):
    with db._read_ctx() as conn:
        workers = [dict(r) for r in conn.execute(
            'SELECT execution_id, session_id, status, generation, owner_epoch, adoption_digest, last_sequence '
            'FROM worker_executions ORDER BY execution_id').fetchall()]
        receipts = [dict(r) for r in conn.execute(
            'SELECT execution_id, sequence, payload_digest, result_json FROM worker_receipts '
            'ORDER BY execution_id, sequence').fetchall()]
    return workers, receipts


def _refusal(call):
    try:
        call()
    except RuntimeStoreError as exc:
        return exc.reason
    return None


def _secret(scope, principal):
    from gateway.session_admission import admission_fingerprint
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    return admission_fingerprint(canonical_target=scope['session_id'], payload=identity | {'principal': principal})


class _BenignWorker:
    """The owner's side of a worker conversation, carried by an EOF-only child. On the bootstrap it
    authenticates with the real adoption writer; it answers with one no-tool result, and on the
    owner's ``finish`` writes its own real ``execution.finish`` and ends the child (stdin EOF)."""

    def __init__(self, authority, child):
        self.authority, self.child = authority, child
        self.stop = asyncio.Event()
        self.stop_cause = None
        self.secrets = []
        self.writer = SimpleNamespace(start=lambda: None)
        self.frames = asyncio.Queue()
        self.frames.put_nowait(_hello(child))
        self.scope = None
        self.closed = []
        self.child_exit = None

    async def next_frame(self, *args, **kwargs):
        return await self.frames.get()

    def send(self, frame):
        if frame.get('type') == 'finish':
            scope = self.scope
            mutate_worker_execution(self.authority.db, epoch=scope['epoch'], execution_id=scope['execution_id'],
                                    session_id=scope['session_id'], generation=scope['generation'], sequence=1,
                                    operation='execution.finish', payload={})
            self.child_exit = _end_child(self.child)
            self.frames.put_nowait({'type': 'finished'})
            return
        scope = frame['scope']
        adopt_worker_execution(self.authority.db, epoch=scope['epoch'], execution_id=scope['execution_id'],
                               session_id=scope['session_id'], generation=scope['generation'],
                               adoption_secret=_secret(scope, owner_worker_subject()))
        self.frames.put_nowait({'type': 'ready', 'pid': scope['pid']})
        self.frames.put_nowait({'type': 'result', 'result': {'final_response': 'ANSWER_HEAD_G' + str(scope['generation'])}})

    def answer_correction(self, frame):
        return False

    def answer_peer_message(self, frame):
        return False

    async def settle_peer_message(self, frame):
        return False

    def abandon_corrections(self, **kwargs):
        pass

    def close(self):
        if self.child.poll() is None:
            self.child_exit = _end_child(self.child)
        self.closed.append(self.child_exit)


async def _recovered_head(tmp_path, monkeypatch, executed):
    """Owner 1's HEAD worker reserves, adopts and durably finishes, then its inert child exits; owner 2
    boots, requeues the SAME admission and claims it again at a later generation."""
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    old = _inert_child()
    old_scope = _reserve(first, claimed, old, state='finished')
    old_exit = _end_child(old)
    before = _rows(first.db)
    second = await _owner(runner, 'owner-2')
    second._schedule = lambda ref: None
    again = claim_session_input(second.db, epoch=second.epoch, session_id='s')
    assert again['admission_id'] == claimed['admission_id']
    return store, runner, first, second, claimed, again, follower_id, old_scope, old_exit, before


@pytest.mark.asyncio
async def test_a_recovered_finished_head_completes_through_the_real_managed_caller(tmp_path, monkeypatch):
    """Benign managed success at generation 1 -> 2 through REAL execute_managed -> tracked_write ->
    reserve_admission_worker -> adoption -> execution.finish; then settlement, and FOLLOWER runs."""
    from gateway.session_results import finish_result
    executed = []
    store, _runner, _first, second, claimed, again, follower_id, old_scope, old_exit, before = \
        await _recovered_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child()
    worker = _BenignWorker(second, child)
    try:
        async def context(*args):
            return 'HEAD', None

        async def spawn(authority, ref, row):
            authority._managed_workers = {ref.session_id: worker}
            return worker, child

        def bootstrap(authority, ref, row, policy, scope, text=None):
            worker.scope = scope
            return {'version': 1, 'scope': scope, 'api_key': None, 'text': text}
        monkeypatch.setattr('gateway.session_context_refs.managed_turn_text', context)
        monkeypatch.setattr(managed, '_spawn_registered_worker', spawn)
        monkeypatch.setattr(managed, '_bootstrap', bootstrap)
        second.executing['s'] = again['generation']  # the drain's execution stamp for this claim
        answer = await managed.execute_managed(second, SessionRef('p', 's'), again, SimpleNamespace())
        pending = second.pending_results[head_id]
        after_execute = _rows(second.db)
        settled, _ = finish_result(second.db, epoch=second.epoch, row=again, response=answer, outcome='completed',
                                   result={'result': pending['result'], 'usage': pending['usage']})
        second.executing.pop('s', None)
        del second._schedule  # the real drain schedules FOLLOWER
        await _drained(second)
        observed = dict(old_generation=claimed['generation'], new_generation=again['generation'],
                        old_exit=old_exit, answer=answer, execution_id=pending['managed_exit']['execution_id'],
                        exit=pending['managed_exit']['exit']['status'],
                        returncode=pending['managed_exit']['exit']['returncode'],
                        settled=(settled['status'], settled['outcome']), executed=list(executed),
                        head=_statuses(second.db)[head_id], follower=_statuses(second.db)[follower_id],
                        workers_before=before[0], workers_after=after_execute[0],
                        old_receipts_before=[r for r in before[1] if r['execution_id'] == old_scope['execution_id']],
                        old_receipts_after=[r for r in after_execute[1]
                                            if r['execution_id'] == old_scope['execution_id']])
    finally:
        if child.poll() is None:
            _end_child(child)
        store.close_all_db_handles()
    _witness('managed_success_g1_g2', **observed)
    first_id, claim_id = admission_worker_execution_ids(head_id, again['generation'])
    assert old_scope['execution_id'] == first_id
    assert observed['old_exit'] == 0 and observed['new_generation'] > observed['old_generation']
    assert observed['answer'] == 'ANSWER_HEAD_G' + str(again['generation'])
    assert observed['execution_id'] == claim_id
    assert (observed['exit'], observed['returncode']) == ('exited', 0)
    old_before = [w for w in observed['workers_before'] if w['execution_id'] == first_id]
    old_after = [w for w in observed['workers_after'] if w['execution_id'] == first_id]
    assert old_before == old_after and old_after[0]['status'] == 'terminal'
    assert old_after[0]['generation'] == claimed['generation']
    assert observed['old_receipts_before'] == observed['old_receipts_after']
    new = [w for w in observed['workers_after'] if w['execution_id'] == claim_id]
    assert len(new) == 1 and new[0]['status'] == 'terminal' and new[0]['generation'] == again['generation']
    assert observed['settled'] == ('terminal', 'completed')
    assert observed['executed'] == ['FOLLOWER']  # HEAD ran on the managed path; the recorder saw FOLLOWER only
    assert observed['head'] == ('terminal', 'completed')
    assert observed['follower'] == ('terminal', 'completed')


@pytest.mark.asyncio
async def test_the_earlier_claims_late_writes_and_adoption_stay_refused(tmp_path, monkeypatch):
    """(d) After the generation-2 reservation: the generation-1 worker's late receipt write and late
    adoption are refused under either epoch; a stale-epoch adoption of the NEW execution is refused; a
    second reservation of the same claim is admission_conflict. The earlier row is unchanged."""
    executed = []
    store, _runner, first, second, claimed, again, _follower_id, old_scope, _old_exit, before = \
        await _recovered_head(tmp_path, monkeypatch, executed)
    child = _inert_child()
    other = None
    try:
        scope = reserve_admission_worker(second, admission_id=again['admission_id'], process=child,
                                         principal_id=again['principal_id'], hello=_hello(child))
        old = dict(execution_id=old_scope['execution_id'], session_id=old_scope['session_id'],
                   generation=old_scope['generation'])
        old_secret = _secret(old_scope, claimed['principal_id'])
        refusals = dict(
            old_write_old_epoch=_refusal(lambda: mutate_worker_execution(
                second.db, epoch=first.epoch, sequence=2, operation='execution.finish', payload={}, **old)),
            old_write_new_epoch=_refusal(lambda: mutate_worker_execution(
                second.db, epoch=second.epoch, sequence=2, operation='execution.finish', payload={}, **old)),
            old_adopt_new_epoch=_refusal(lambda: adopt_worker_execution(
                second.db, epoch=second.epoch, adoption_secret=old_secret, **old)),
            new_adopt_old_epoch=_refusal(lambda: adopt_worker_execution(
                second.db, epoch=first.epoch, execution_id=scope['execution_id'], session_id=scope['session_id'],
                generation=scope['generation'], adoption_secret=_secret(scope, again['principal_id']))))
        other = _inert_child()
        refusals['same_claim_second_reservation'] = _refusal(lambda: reserve_admission_worker(
            second, admission_id=again['admission_id'], process=other, principal_id=again['principal_id'],
            hello=_hello(other)))
        after = _rows(second.db)
        observed = dict(new_execution=scope['execution_id'], **refusals,
                        old_row_before=[w for w in before[0] if w['execution_id'] == old['execution_id']],
                        old_row_after=[w for w in after[0] if w['execution_id'] == old['execution_id']],
                        old_receipts_unchanged=[r for r in before[1] if r['execution_id'] == old['execution_id']]
                        == [r for r in after[1] if r['execution_id'] == old['execution_id']],
                        workers=len(after[0]))
    finally:
        _end_child(child)
        if other is not None:
            _end_child(other)
        store.close_all_db_handles()
    _witness('late_write_adoption_fences', **observed)
    assert observed['new_execution'] == admission_worker_execution_ids(again['admission_id'], again['generation'])[1]
    assert observed['old_write_old_epoch'] == 'stale_epoch'
    assert observed['old_write_new_epoch'] in ('stale_generation', 'stale_epoch')
    assert observed['old_adopt_new_epoch'] == 'stale_generation'
    assert observed['new_adopt_old_epoch'] == 'stale_epoch'
    assert observed['same_claim_second_reservation'] == 'admission_conflict'
    assert observed['old_row_before'] == observed['old_row_after']
    assert observed['old_receipts_unchanged'] is True
    assert observed['workers'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('first_claim', ['tool_committed_then_finished', 'registered', 'running'])
async def test_a_claim_that_reached_a_tool_or_could_still_act_gets_no_second_reservation(
        tmp_path, monkeypatch, first_claim):
    """(a) The generation-1 worker reached a tool and finished: business may have happened, so the turn
    is not queued again. (b) The generation-1 worker is registered/running (``unknown`` after the boot's
    fence): no replay, as before. Only the independent FOLLOWER runs. No second HEAD reservation
    appears, HEAD refuses any reservation, and probing the now-empty queue claims nothing."""
    executed = []
    store, runner, first, claimed, follower_id = await _claimed_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    db = first.db
    child = _inert_child()
    other = None
    try:
        if first_claim == 'tool_committed_then_finished':
            scope = _reserve(first, claimed, child, state='running')
            db.append_message('s', 'assistant', None, tool_calls=[{'id': 'c1', 'type': 'function', 'function': {
                'name': 'terminal', 'arguments': '{"command": "touch effect"}'}}])
            db.append_message('s', 'tool', 'done', tool_call_id='c1', tool_name='terminal')
            mutate_worker_execution(db, epoch=first.epoch, execution_id=scope['execution_id'],
                                    session_id=scope['session_id'], generation=scope['generation'], sequence=1,
                                    operation='execution.finish', payload={})
        else:
            _reserve(first, claimed, child, state=first_claim)
        second = await _owner(runner, 'owner-2')
        await _drained(second)
        other = _inert_child()
        reservation = _refusal(lambda: reserve_admission_worker(
            second, admission_id=head_id, process=other, principal_id=claimed['principal_id'], hello=_hello(other)))
        workers, receipts = _rows(second.db)
        generation = second.db._read_one("SELECT runtime_generation FROM sessions WHERE id='s'")[0]
        claim = _claim_probe(second)
        assert _rows(second.db) == (workers, receipts)
        assert second.db._read_one("SELECT runtime_generation FROM sessions WHERE id='s'")[0] == generation
        observed = dict(executed=list(executed), head=_statuses(second.db)[head_id],
                        follower=_statuses(second.db)[follower_id], claim=claim,
                        reservation=reservation, worker_ids=[w['execution_id'] for w in workers],
                        worker_states=[w['status'] for w in workers])
    finally:
        _end_child(child)
        if other is not None:
            _end_child(other)
        store.close_all_db_handles()
    _witness('no_second_reservation_' + first_claim, **observed)
    assert observed['executed'] == ['FOLLOWER']
    assert observed['head'] == ('unknown', None)
    assert observed['follower'] == ('terminal', 'completed')
    assert observed['claim'] == 'None'  # no queued work, not permission to reclaim unknown HEAD
    assert observed['reservation'] == 'producer_not_started'
    assert observed['worker_ids'] == [admission_worker_execution_ids(head_id, 0)[0]]
    assert observed['worker_states'] == ['terminal' if first_claim == 'tool_committed_then_finished' else 'unknown']


@pytest.mark.asyncio
async def test_recovery_readers_find_the_claim_scoped_row(tmp_path, monkeypatch):
    """Readers keyed on the admission still see the generation-2 execution: an unfinished generation-2
    worker makes the claim unresolved even after its row moved to another transcript (the id clause, not
    only the session/generation clause), so no redelivery is owed; reset's orphan discard leaves it, as
    any admission-linked execution; its outbox name is valid and distinct from generation 1's."""
    from agent.managed_worker import outbox_dir
    from gateway.session_message_redelivery import _unresolved_execution
    from hermes_state_runtime_workers import discard_orphan_workers_on_reset
    executed = []
    store, _runner, _first, second, claimed, again, _follower_id, old_scope, _old_exit, _before = \
        await _recovered_head(tmp_path, monkeypatch, executed)
    head_id = claimed['admission_id']
    child = _inert_child()
    try:
        scope = reserve_admission_worker(second, admission_id=head_id, process=child,
                                         principal_id=again['principal_id'], hello=_hello(child))
        second.db.create_session('moved', source='telegram')

        def probe(conn):
            row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (head_id,)).fetchone()
            in_place = _unresolved_execution(conn, row)
            conn.execute("UPDATE worker_executions SET session_id='moved' WHERE execution_id=?",
                         (scope['execution_id'],))
            moved = _unresolved_execution(conn, row)
            conn.execute("UPDATE worker_executions SET status='unknown' WHERE execution_id=?", (scope['execution_id'],))
            discard_orphan_workers_on_reset(conn, ['moved'])
            state = conn.execute('SELECT status FROM worker_executions WHERE execution_id=?',
                                 (scope['execution_id'],)).fetchone()[0]
            raise _Rollback(dict(in_place=in_place, moved=moved, after_reset=state))
        try:
            second.db._execute_write(probe)
        except _Rollback as done:
            probed = done.value
        outboxes = [outbox_dir(tmp_path, old_scope['execution_id']).name, outbox_dir(tmp_path, scope['execution_id']).name]
        observed = dict(execution_id=scope['execution_id'], **probed, outboxes=outboxes)
    finally:
        _end_child(child)
        store.close_all_db_handles()
    _witness('claim_scoped_readers', **observed)
    assert observed['execution_id'] == admission_worker_execution_ids(head_id, again['generation'])[1]
    assert observed['in_place'] is True
    assert observed['moved'] is True
    assert observed['after_reset'] == 'unknown'
    assert observed['outboxes'][0] != observed['outboxes'][1]


class _Rollback(Exception):
    def __init__(self, value):
        self.value = value


def test_claim_execution_id_rules():
    """The id choice itself: no earlier reservation -> the first-claim id; only terminal earlier-generation
    reservations -> this claim's id; any non-terminal, same-or-later or unreadable earlier generation ->
    admission_conflict. Another admission's id that merely shares the prefix is not this admission's."""
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE TABLE worker_executions(execution_id TEXT PRIMARY KEY, kind TEXT, status TEXT, generation, session_id TEXT DEFAULT '')")
    # The production reader now resolves exact persisted ownership and retained
    # admissions, rather than assuming numeric suffixes prove ownership.
    conn.execute('CREATE TABLE state_meta(key TEXT PRIMARY KEY, value TEXT)')
    conn.execute('CREATE TABLE session_admissions(admission_id TEXT PRIMARY KEY)')

    def case(rows, generation):
        conn.execute('DELETE FROM worker_executions')
        conn.executemany('INSERT INTO worker_executions(execution_id,kind,status,generation) VALUES(?,?,?,?)', rows)
        try:
            return _claim_execution_id(conn, 'a1', generation)
        except RuntimeStoreError as exc:
            return 'RuntimeStoreError:' + exc.reason
    first, _ = admission_worker_execution_ids('a1', 0)
    observed = dict(
        none=case([], 1),
        terminal_g1=case([(first, 'compute', 'terminal', 1)], 2),
        terminal_g1_g2=case([(first, 'compute', 'terminal', 1), (first + ':g2', 'compute', 'terminal', 2)], 3),
        same_generation=case([(first, 'compute', 'terminal', 2)], 2),
        claim_same_generation=case([(first, 'compute', 'terminal', 1), (first + ':g2', 'compute', 'terminal', 2)], 2),
        unknown_g1=case([(first, 'compute', 'unknown', 1)], 2),
        running_g1=case([(first, 'compute', 'running', 1)], 2),
        registered_g2=case([(first, 'compute', 'terminal', 1), (first + ':g2', 'compute', 'registered', 2)], 3),
        unreadable_generation=case([(first, 'compute', 'terminal', None)], 2),
        other_admission=case([(first + ':gx', 'compute', 'running', 1)], 2),
        child_row=case([(first + '/child:z', 'child', 'running', 1)], 2))
    _witness('claim_execution_id_rules', **observed)
    assert observed == dict(
        none=first, terminal_g1=first + ':g2', terminal_g1_g2=first + ':g3',
        same_generation='RuntimeStoreError:admission_conflict',
        claim_same_generation='RuntimeStoreError:admission_conflict',
        unknown_g1='RuntimeStoreError:admission_conflict', running_g1='RuntimeStoreError:admission_conflict',
        registered_g2='RuntimeStoreError:admission_conflict',
        unreadable_generation='RuntimeStoreError:admission_conflict',
        other_admission=first, child_row=first)
