"""Later accepted messages progress without replaying an unknown business claim (MR-DELIVERY22).

Essential: real authority admission/boot/drain, SQLite, reservation/adoption/persistence and store.
Incidental: EOF-only identity children, context expansion, scripted worker wire conversation;
no provider, recipient or protected business operation is invoked.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from gateway.session_worker_reservation import reserve_admission_worker, owner_worker_subject
from hermes_state_runtime import (
    CLAIM_MARK_PREFIX, RuntimeStoreError, adopt_worker_execution, claim_session_input,
    mutate_worker_execution,
)
from tests.gateway.test_managed_identity_collision import _forget_new_links, _rpc
from tests.gateway.test_managed_worker_turn_author import _policy
from tests.gateway.test_person_message_redelivery import _drained, _owner
from tests.gateway.test_redelivery_claim_scoped_reservation import _secret
from tests.gateway.test_redelivery_worker_fence import (
    _claimed_head, _end_child, _hello, _inert_child, _reserve, _worker,
)
from tests.gateway.test_unknown_automation_settles import _statuses


def _retained(db, admission_id, execution_id):
    with db._read_ctx() as conn:
        return {
            'admission': tuple(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                           (admission_id,)).fetchone()),
            'worker': tuple(conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                                        (execution_id,)).fetchone()),
            'receipts': [tuple(row) for row in conn.execute('SELECT * FROM worker_receipts WHERE execution_id=?',
                                                           (execution_id,))],
            'mark': tuple(conn.execute('SELECT * FROM state_meta WHERE key=?',
                                      (CLAIM_MARK_PREFIX + admission_id,)).fetchone()),
        }


def _late_refusals(db, first, current, old, principal):
    """Both old-authenticated and new-owner attempts hit real production fences."""
    reasons = []
    for epoch in (first.epoch, current.epoch):
        for operation in ('transcript.append', 'execution.finish'):
            with pytest.raises(RuntimeStoreError) as caught:
                mutate_worker_execution(db, epoch=epoch, execution_id=old['execution_id'],
                    session_id=old['session_id'], generation=old['generation'], sequence=2,
                    operation=operation, payload={'messages': [{'role': 'tool', 'content': 'LATE'}]}
                    if operation == 'transcript.append' else {})
            assert caught.value.reason in ('stale_epoch', 'stale_generation')
            reasons.append(caught.value.reason)
        with pytest.raises(RuntimeStoreError) as caught:
            adopt_worker_execution(db, epoch=epoch, execution_id=old['execution_id'],
                session_id=old['session_id'], generation=old['generation'], adoption_secret=_secret(old, principal))
        assert caught.value.reason in ('stale_epoch', 'stale_generation')
        reasons.append(caught.value.reason)
    return reasons


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True], ids=['mr23-links', 'retained-legacy'])
@pytest.mark.parametrize('tool', [False, True], ids=['no-transcript-tool', 'prior-tool-intent'])
async def test_follower_keeps_old_scope_unknown_and_byte_unchanged(tmp_path, monkeypatch, legacy, tool):
    executed = []
    store, runner, first, claimed, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    child = _inert_child()
    try:
        old = _reserve(first, claimed, child, state='running')
        mutate_worker_execution(first.db, epoch=first.epoch, execution_id=old['execution_id'],
            session_id=old['session_id'], generation=old['generation'], sequence=1,
            operation='usage.main', payload={'input_tokens': 1})
        if legacy:
            _forget_new_links(first.db, [(claimed['admission_id'], old)])
        if tool:
            first.db.append_message('s', 'assistant', None, tool_calls=[{
                'id': 'unknown-business', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}])
        second = await _owner(runner, 'owner-2')
        before = _retained(second.db, claimed['admission_id'], old['execution_id'])
        refusals = _late_refusals(second.db, first, second, old, claimed['principal_id'])
        await _drained(second)
        refusals += _late_refusals(second.db, first, second, old, claimed['principal_id'])
        assert executed == ['FOLLOWER']
        assert _statuses(second.db)[claimed['admission_id']] == ('unknown', None)
        assert _statuses(second.db)[follower] == ('terminal', 'completed')
        assert _retained(second.db, claimed['admission_id'], old['execution_id']) == before
        assert _worker(second.db, claimed['admission_id']) == 'unknown'
        assert child.poll() is None  # no fake physical exit/retirement was needed
        assert second._handle(SessionRef('p', 's')).execution_state == 'unknown'
        assert all(message.get('content') != 'LATE' for message in second.db.get_messages('s'))
        with pytest.raises(RuntimeStoreError) as caught:
            reserve_admission_worker(second, admission_id=claimed['admission_id'], process=child,
                principal_id=claimed['principal_id'], hello=_hello(child))
        assert caught.value.reason == 'producer_not_started'
        print('MR22_RETAINED', json.dumps({'legacy': legacy, 'tool': tool, 'executed': executed,
            'head': 'unknown', 'worker': 'unknown', 'old_scope_unchanged': True, 'late_refusals': refusals}), flush=True)
    finally:
        assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('blocker', ['orphan', 'foreign-link', 'current-epoch', 'running', 'started'])
async def test_claim_refuses_unproven_or_still_authorized_work(tmp_path, monkeypatch, blocker):
    executed = []
    store, runner, first, claimed, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    child = _inert_child()
    try:
        old = _reserve(first, claimed, child, state='running')
        second = await _owner(runner, 'owner-2')
        def corrupt_or_active(conn):
            if blocker == 'orphan':
                conn.execute('UPDATE worker_executions SET execution_id=? WHERE execution_id=?',
                             ('unlinked-orphan', old['execution_id']))
            elif blocker == 'foreign-link':
                from gateway.session_worker_reservation import _OWNER_KEY
                conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                             (json.dumps('another-admission'), _OWNER_KEY + old['execution_id']))
            elif blocker == 'current-epoch':
                conn.execute('UPDATE worker_executions SET owner_epoch=?', (second.epoch,))
            elif blocker == 'running':
                conn.execute("UPDATE worker_executions SET status='running'")
            else:
                conn.execute("UPDATE session_admissions SET status='started',owner_epoch=? WHERE admission_id=?",
                             (second.epoch, claimed['admission_id']))
        second.db._execute_write(corrupt_or_active)
        before = _statuses(second.db)
        try:
            result = claim_session_input(second.db, epoch=second.epoch, session_id='s', head=follower)
        except RuntimeStoreError as exc:
            assert exc.reason in ('unknown_execution', 'admission_conflict')
        else:
            assert result is None
        assert _statuses(second.db) == before
        assert executed == []
    finally:
        assert _end_child(child) == 0
        store.close_all_db_handles()


class _BenignWorker:
    """Wire-only substitute; persistence, adoption, outbox, bootstrap and finalizer stay real."""
    def __init__(self, child, authority, observations):
        self.child, self.authority, self.observations = child, authority, observations
        self.stop = asyncio.Event()
        self.stop_cause = None
        self.secrets = []
        self.transferred = []
        self.writer = SimpleNamespace(start=lambda: None)
        self.frames = asyncio.Queue()
        self.frames.put_nowait(_hello(child))
        self.scope = self.policy = self.session_id = self.worker = None
        self.ready = False
        self.store = None

    async def next_frame(self, *args, **kwargs):
        return await self.frames.get()

    def send(self, frame):
        from agent.managed_worker import outbox_dir
        from agent.runtime_session_store import RuntimeSessionStore
        if frame.get('type') == 'finish':
            self.store.finish()
            self.store.close()
            self.store = None
            assert _end_child(self.child) == 0
            self.frames.put_nowait({'type': 'finished'})
            return
        scope = frame['scope']
        assert frame['text'] == 'FOLLOWER' and frame['api_key'] is None
        assert self.authority._handle(SessionRef(self.authority.profile_id, 's')).execution_state == 'running'
        self.authority.check_approval_generation('s', scope['generation'])
        adopted = adopt_worker_execution(self.authority.db, epoch=scope['epoch'], execution_id=scope['execution_id'],
            session_id=scope['session_id'], generation=scope['generation'],
            adoption_secret=_secret(scope, owner_worker_subject()))
        path = outbox_dir(frame['home'], scope['execution_id'])
        self.store = RuntimeSessionStore(_rpc(self.authority.db), scope, path, adopted)
        self.store.append_messages_batch('s', [{'role': 'user', 'content': 'FOLLOWER'}])
        self.observations.append({'scope': scope, 'outbox': str(path), 'real_bootstrap': True})
        self.frames.put_nowait({'type': 'ready', 'pid': scope['pid']})
        self.frames.put_nowait({'type': 'result', 'result': {'final_response': 'ACK_FOLLOWER'}})

    def answer_correction(self, frame):
        return False

    def answer_peer_message(self, frame):
        return False

    async def settle_peer_message(self, frame):
        return False

    def abandon_corrections(self, **kwargs):
        pass

    def close(self):
        if self.store is not None:
            self.store.close()
        if self.child.poll() is None:
            assert _end_child(self.child) == 0


@pytest.mark.asyncio
async def test_follower_normal_drain_real_managed_bootstrap_store_and_settlement(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    executed = []
    observations = []
    store, runner, first, claimed, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    old_child, fresh_child = _inert_child(), _inert_child()
    worker = None
    try:
        old = _reserve(first, claimed, old_child, state='running')
        second = await _owner(runner, 'owner-2')
        second.profile_id = str(tmp_path)
        before = _retained(second.db, claimed['admission_id'], old['execution_id'])
        policy = _policy(tmp_path)
        assert policy.credential_ref is None and policy.config_secret_ref is None
        worker = _BenignWorker(fresh_child, second, observations)
        async def context(authority, row, policy):
            return row['payload']['text'], None
        async def spawn(authority, ref, row):
            authority._managed_workers = {ref.session_id: worker}
            return worker, fresh_child
        monkeypatch.setattr('gateway.session_context_refs.managed_turn_text', context)
        monkeypatch.setattr(managed, '_spawn_registered_worker', spawn)
        # Real validated bypass policy; only selection is substituted for this Telegram fixture.
        # execute_admission itself takes its ordinary managed branch (no handler/finalizer replacement).
        monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
        # No executing-stamp, claim, reader, bootstrap, store or finalizer substitution.
        await _drained(second)
        assert executed == []  # the in-process fallback handler was not used
        assert len(observations) == 1
        assert observations[0]['scope']['execution_id'] != old['execution_id']
        assert _statuses(second.db)[follower] == ('terminal', 'completed')
        assert _statuses(second.db)[claimed['admission_id']] == ('unknown', None)
        assert _retained(second.db, claimed['admission_id'], old['execution_id']) == before
        assert old_child.poll() is None and fresh_child.returncode == 0
        assert (Path(observations[0]['outbox']) / 'pending.json').exists()
        _late_refusals(second.db, first, second, old, claimed['principal_id'])
        print('MR22_MANAGED', json.dumps({'managed_delivered': ['FOLLOWER'], 'fallback_executed': executed,
            'head': 'unknown', 'old_worker': 'unknown',
            'follower': 'terminal/completed', 'real_bootstrap_store_finish_settlement': True}), flush=True)
    finally:
        if worker is not None:
            worker.close()
        for child in (old_child, fresh_child):
            if child.poll() is None:
                assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_same_drain_rechecks_preaccepted_follower_after_managed_loss(tmp_path, monkeypatch):
    """Real managed loss/unknown producer; no claim or settlement substitution.

    The existing loss writer revokes this own known-inert worker. MR22 does not
    invent terminality for a prior owner's unknown worker (covered above).
    """
    from gateway import session_managed_worker as managed
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_unknown_automation_settles import Peer, _person_head, _runner
    executed, observations, children = [], [], []
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    db.create_session('s', source='telegram')
    authority = await _owner(runner, 'owner-1')
    authority.profile_id = str(tmp_path)
    authority._schedule = lambda ref: None
    ref = SessionRef(authority.profile_id, 's')
    actor = AuthorityConnection(authority, Peer(), {'user_id': 'human'}).actor
    head = await _person_head(authority, actor, ref)
    follower = await authority.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    policy = _policy(tmp_path)

    class LostWorker(_BenignWorker):
        def send(self, frame):
            scope = frame['scope']
            assert frame['text'] == 'HEAD'
            adopt_worker_execution(db, epoch=scope['epoch'], execution_id=scope['execution_id'],
                session_id=scope['session_id'], generation=scope['generation'],
                adoption_secret=_secret(scope, owner_worker_subject()))
            self.frames.put_nowait({'type': 'error', 'reason': 'managed_worker_failed'})

    async def context(authority, row, policy):
        return row['payload']['text'], None

    async def spawn(owner, ref, row):
        child = _inert_child()
        children.append(child)
        worker_type = LostWorker if row['admission_id'] == head else _BenignWorker
        worker = worker_type(child, owner, observations)
        owner._managed_workers = {ref.session_id: worker}
        return worker, child

    monkeypatch.setattr('gateway.session_context_refs.managed_turn_text', context)
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
    monkeypatch.setattr(managed, '_spawn_registered_worker', spawn)
    try:
        del authority._schedule
        await _drained(authority)  # one schedule only, both admissions already committed
        assert _statuses(db)[head] == ('unknown', None)
        assert _statuses(db)[follower.admission_id] == ('terminal', 'completed')
        assert len(children) == 2 and all(child.returncode == 0 for child in children)
        assert len(observations) == 1 and executed == []
        assert db._read_one('SELECT status FROM worker_executions WHERE execution_id=?',
                            ('admission-worker:' + head,))[0] == 'terminal'
        print('MR22_SAME_DRAIN', json.dumps({'head': 'unknown', 'follower': 'terminal/completed',
            'schedules': 1, 'managed_consumer': True, 'old_loss_writer_unchanged': True}), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                assert _end_child(child) == 0
        store.close_all_db_handles()
