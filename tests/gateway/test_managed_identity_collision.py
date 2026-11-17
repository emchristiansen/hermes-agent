"""Legacy admission identity isolation through the importer and real managed consumers.

No provider/business workload. Real SQLite epochs, importer, claims, reservations,
adoption, worker persistence, bootstrap builder and settlement. Incidental externality:
EOF-only child + scripted worker transport and context expansion. Persistence RPC calls
the real writer directly, not a network/control-socket authentication test.
"""
import asyncio
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from agent.managed_worker import outbox_dir
from agent.runtime_session_store import RuntimeSessionStore
from gateway import session_managed_worker as managed
from gateway.session_contract import SessionRef
from gateway.session_worker_reservation import (
    _CLAIMS_KEY, _OWNER_KEY, admission_worker_rows, owner_worker_subject, reserve_admission_worker,
)
from hermes_state_runtime import (
    RuntimeStoreError, adopt_worker_execution, claim_session_input, import_legacy_session_admissions,
    mutate_worker_execution, settle_session_input,
)
from tests.gateway.test_managed_worker_turn_author import _policy
from tests.gateway.test_person_message_redelivery import _owner
from tests.gateway.test_redelivery_worker_fence import _end_child, _hello, _inert_child, _reserve
from tests.gateway.test_redelivery_claim_scoped_reservation import _secret
from tests.gateway.test_unknown_automation_settles import _runner


def _legacy(path, entries):
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE admissions(seq INTEGER, admission_id TEXT, target_session_id TEXT, '
                     'root TEXT, lineage TEXT, payload TEXT, status TEXT, outcome TEXT, generation INTEGER)')
        for seq, (admission_id, session_id) in enumerate(entries, 1):
            conn.execute('INSERT INTO admissions VALUES(?,?,?,?,?,?,?,?,?)',
                         (seq, admission_id, session_id, session_id, json.dumps([session_id]),
                          json.dumps({'text': 'benign'}), 'queued', None, None))


def _rpc(db):
    def persist(method, **args):
        assert method == 'worker.persist'
        return mutate_worker_execution(db, **{k: args[k] for k in (
            'epoch', 'execution_id', 'session_id', 'generation', 'sequence', 'operation', 'payload')})
    return persist


def _snapshot(db):
    with db._read_ctx() as conn:
        return ([tuple(row) for row in conn.execute('SELECT * FROM worker_executions ORDER BY execution_id')],
                [tuple(row) for row in conn.execute('SELECT * FROM worker_receipts ORDER BY execution_id, sequence')])


def _forget_new_links(db, scopes):
    """Private fixture: reproduce retained PRE-MR23 rows with no new ownership records."""
    def write(conn):
        for admission_id, scope in scopes:
            conn.execute('DELETE FROM state_meta WHERE key IN (?,?)',
                         (_CLAIMS_KEY + admission_id, _OWNER_KEY + scope['execution_id']))
    db._execute_write(write)


def _finish_legacy(authority, row, child, home):
    scope = _reserve(authority, row, child, state='running')
    legacy = home / 'worker-outboxes' / scope['execution_id'].replace(':', '-')
    with_store = RuntimeSessionStore(_rpc(authority.db), scope, legacy)
    try:
        with_store.finish()
    finally:
        with_store.close()
    assert _end_child(child) == 0
    return scope, legacy / 'pending.json'


@pytest.mark.asyncio
@pytest.mark.parametrize('other_id', ['a:g2', 'a-g2'])
@pytest.mark.parametrize('legacy_state', [False, True], ids=['new-links', 'retained-pre23'])
@pytest.mark.parametrize('retired_other', [False, True], ids=['live-ledger', 'terminal-tombstone'])
async def test_imported_collision_retries_through_real_bootstrap_and_outbox(
        tmp_path, monkeypatch, other_id, legacy_state, retired_other):
    store, runner = _runner(tmp_path, monkeypatch, [])
    children = []
    runtime_store = None
    try:
        db = store._db
        db.create_session('s', source='telegram')
        db.create_session('b', source='telegram')
        first = await _owner(runner, 'owner-1')
        first._schedule = lambda ref: None
        source = tmp_path / 'legacy.db'
        _legacy(source, [('a', 's'), (other_id, 'b')])
        assert import_legacy_session_admissions(db, epoch=first.epoch, source_path=source,
                                               principal_id='person', writers_drained=True) == 2
        other = claim_session_input(db, epoch=first.epoch, session_id='b')
        child = _inert_child()
        children.append(child)
        other_scope, other_journal = _finish_legacy(first, other, child, tmp_path)
        settle_session_input(db, epoch=first.epoch, admission_id=other_id,
                             generation=other['generation'], outcome='completed')
        if legacy_state:
            _forget_new_links(db, [(other_id, other_scope)])
        if retired_other:
            from hermes_state_mutation_retirement import retire_terminal_receipts
            db._execute_write(lambda conn: retire_terminal_receipts(conn, ['b']))
        target = claim_session_input(db, epoch=first.epoch, session_id='s')
        child = _inert_child()
        children.append(child)
        old_scope, old_journal = _finish_legacy(first, target, child, tmp_path)
        assert old_scope['execution_id'] == 'admission-worker:a'
        if legacy_state:
            _forget_new_links(db, [('a', old_scope)])
        journals = {str(p): p.read_bytes() for p in (other_journal, old_journal)}
        before = _snapshot(db)
        second = await _owner(runner, 'owner-2')
        # Production profile_id is the private profile HOME; the reused helper's
        # symbolic 'p' otherwise shares an outbox across parameterized cases.
        second.profile_id = str(tmp_path)
        second._schedule = lambda ref: None
        again = claim_session_input(db, epoch=second.epoch, session_id='s')
        assert again['admission_id'] == 'a' and again['generation'] == 2
        child = _inert_child()
        children.append(child)
        bootstrap = []

        class Worker:
            def __init__(self):
                self.stop = asyncio.Event()
                self.stop_cause = None
                self.secrets = []
                self.transferred = []
                self.writer = SimpleNamespace(start=lambda: None)
                self.frames = asyncio.Queue()
                self.frames.put_nowait(_hello(child))
                self.scope = self.policy = self.session_id = self.worker = None
                self.ready = False

            async def next_frame(self, *args, **kwargs):
                return await self.frames.get()

            def send(self, frame):
                nonlocal runtime_store
                if frame.get('type') == 'finish':
                    runtime_store.finish()
                    runtime_store.close()
                    runtime_store = None
                    assert _end_child(child) == 0
                    self.frames.put_nowait({'type': 'finished'})
                    return
                bootstrap.append(frame)
                scope = frame['scope']
                adopted = adopt_worker_execution(db, epoch=scope['epoch'], execution_id=scope['execution_id'],
                    session_id=scope['session_id'], generation=scope['generation'],
                    adoption_secret=_secret(scope, owner_worker_subject()))
                runtime_store = RuntimeSessionStore(_rpc(db), scope,
                    outbox_dir(frame['home'], scope['execution_id']), adopted)
                self.frames.put_nowait({'type': 'ready', 'pid': scope['pid']})
                self.frames.put_nowait({'type': 'result', 'result': {'final_response': 'benign identity answer'}})

            def answer_correction(self, frame):
                return False

            def answer_peer_message(self, frame):
                return False

            async def settle_peer_message(self, frame):
                return False

            def abandon_corrections(self, **kwargs):
                pass

            def close(self):
                if child.poll() is None:
                    assert _end_child(child) == 0

        worker = Worker()

        async def context(*args):
            return 'benign', None

        async def spawn(authority, ref, row):
            authority._managed_workers = {ref.session_id: worker}
            return worker, child

        monkeypatch.setattr('gateway.session_context_refs.managed_turn_text', context)
        monkeypatch.setattr(managed, '_spawn_registered_worker', spawn)
        policy = _policy(tmp_path)
        assert policy.credential_ref is None and policy.config_secret_ref is None
        second.executing['s'] = again['generation']
        answer = await managed.execute_managed(second, SessionRef(second.profile_id, 's'), again, policy)
        assert answer == 'benign identity answer'
        from gateway.session_results import finish_result
        pending = second.pending_results['a']
        settled, _ = finish_result(db, epoch=second.epoch, row=again, response=answer, outcome='completed',
                                  result={'result': pending['result'], 'usage': pending['usage']})
        assert (settled['status'], settled['outcome']) == ('terminal', 'completed')
        assert len(bootstrap) == 1 and bootstrap[0]['api_key'] is None
        scope = bootstrap[0]['scope']
        assert scope['execution_id'] not in (old_scope['execution_id'], other_scope['execution_id'])
        # Bootstrap uses the private home supplied by _policy. Verify preserved
        # old fixtures directly AND actual new store construction/finish above.
        assert all(Path(p).read_bytes() == contents for p, contents in journals.items())
        after = _snapshot(db)
        assert all(row in after[0] for row in before[0])
        assert all(row in after[1] for row in before[1])
        assert pending['managed_exit']['exit']['returncode'] == 0
        print('MR23_BOOTSTRAP', json.dumps({'other_id': other_id, 'retained_pre23': legacy_state,
              'execution_id': scope['execution_id'], 'old_rows_and_receipts_unchanged': True,
              'old_journals_unchanged': True, 'real_bootstrap': True, 'answer': answer}), flush=True)
    finally:
        if runtime_store is not None:
            runtime_store.close()
        for child in children:
            if child.poll() is None:
                assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('admission_id', ['../outside', 'a/child:b', 'C:\\evil', '😀' * 1024, 'x' * 1024, 'has\x00nul'])
async def test_full_imported_domain_reserves_portable_linked_execution(tmp_path, monkeypatch, admission_id):
    store, runner = _runner(tmp_path, monkeypatch, [])
    child = None
    box = None
    try:
        db = store._db
        db.create_session('s', source='telegram')
        authority = await _owner(runner, 'owner')
        authority._schedule = lambda ref: None
        source = tmp_path / 'legacy.db'
        _legacy(source, [(admission_id, 's')])
        assert import_legacy_session_admissions(db, epoch=authority.epoch, source_path=source,
                                               principal_id='person', writers_drained=True) == 1
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        assert row['admission_id'] == admission_id
        child = _inert_child()
        scope = _reserve(authority, row, child, state='running')
        path = outbox_dir(tmp_path, scope['execution_id'])
        assert path.is_relative_to(tmp_path / 'worker-outboxes')
        box = RuntimeSessionStore(_rpc(db), scope, path)
        box.finish()
        box.close()
        box = None
        with db._read_ctx() as conn:
            assert [r[0] for r in admission_worker_rows(conn, admission_id)] == [scope['execution_id']]
        # Even after terminal finish, same generation cannot mint another reservation.
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            reserve_admission_worker(authority, admission_id=admission_id, process=child,
                                     principal_id='person', hello=_hello(child))
    finally:
        if box is not None:
            box.close()
        if child is not None and child.poll() is None:
            assert _end_child(child) == 0
        store.close_all_db_handles()


def test_outbox_retains_exact_legacy_pending_journal_and_isolates_foreign_id(tmp_path):
    first, second = 'admission-worker:a:g2', 'admission-worker:a-g2'
    legacy = tmp_path / 'worker-outboxes' / first.replace(':', '-')
    legacy.mkdir(parents=True)
    original = json.dumps({'scope': {'execution_id': first}, 'pending': [{'original': 'retained'}]}).encode()
    (legacy / 'pending.json').write_bytes(original)
    assert outbox_dir(tmp_path, first) == legacy
    assert outbox_dir(tmp_path, second) != legacy
    assert (legacy / 'pending.json').read_bytes() == original
    # When the other spelling held the journal, the first spelling must use v2.
    (legacy / 'pending.json').write_text(json.dumps({'scope': {'execution_id': second}}))
    assert outbox_dir(tmp_path, second) == legacy
    assert outbox_dir(tmp_path, first) != legacy


def test_outbox_encoded_path_is_case_sensitive_portable_and_no_prefix_alias(tmp_path):
    keys = ['admission-worker:a:g2', 'admission-worker:A:g2', 'admission-worker:a-g2',
            'admission-worker:a:g2:g3', 'admission-worker:a:g2-g3', 'admission-worker:a', 'admission-worker:A']
    paths = [outbox_dir(tmp_path, key) for key in keys]
    assert len({str(path).casefold() for path in paths}) == len(keys)
    assert all(len(part) <= 200 for path in paths for part in path.relative_to(tmp_path).parts)
    assert all(not a.is_relative_to(b) and not b.is_relative_to(a)
               for index, a in enumerate(paths) for b in paths[index + 1:])


@pytest.mark.parametrize('kind', ['symlink', 'malformed', 'both-journals'])
def test_outbox_refuses_unsafe_or_ambiguous_retained_state(tmp_path, kind):
    key = 'admission-worker:a:g2'
    root = tmp_path / 'worker-outboxes'
    root.mkdir()
    legacy = root / 'admission-worker-a-g2'
    legacy.mkdir()
    if kind == 'symlink':
        (root / 'v2').symlink_to(tmp_path, target_is_directory=True)
        expected = 'unsafe_outbox'
    elif kind == 'malformed':
        (legacy / 'pending.json').write_text('not json')
        expected = 'invalid_outbox_identity'
    else:
        current = outbox_dir(tmp_path, key)
        current.mkdir(parents=True)
        (current / 'pending.json').write_text('{}')
        (legacy / 'pending.json').write_text(json.dumps({'scope': {'execution_id': key}}))
        expected = 'outbox_identity_conflict'
    with pytest.raises(ValueError, match=expected):
        outbox_dir(tmp_path, key)


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_state', [False, True], ids=['new-links', 'retained-pre23'])
async def test_queued_suffix_id_does_not_hide_prefix_retry_and_first_can_allocate_around_it(
        tmp_path, monkeypatch, legacy_state):
    from gateway.session_message_redelivery import _unresolved_execution
    from gateway.session_worker_reservation import _claim_execution_id
    store, runner = _runner(tmp_path, monkeypatch, [])
    children = []
    try:
        db = store._db
        db.create_session('s', source='telegram')
        db.create_session('b', source='telegram')
        first = await _owner(runner, 'owner-1')
        first._schedule = lambda ref: None
        source = tmp_path / 'legacy.db'
        _legacy(source, [('a', 's'), ('a:g2', 'b')])
        assert import_legacy_session_admissions(db, epoch=first.epoch, source_path=source,
                                               principal_id='person', writers_drained=True) == 2
        row = claim_session_input(db, epoch=first.epoch, session_id='s')
        child = _inert_child()
        children.append(child)
        original = _reserve(first, row, child, state='finished')
        assert _end_child(child) == 0
        second = await _owner(runner, 'owner-2')
        second._schedule = lambda ref: None
        again = claim_session_input(db, epoch=second.epoch, session_id='s')
        child = _inert_child()
        children.append(child)
        retry = _reserve(second, again, child, state='running')
        assert retry['execution_id'] == 'admission-worker:a:g2'
        if legacy_state:
            _forget_new_links(db, [('a', original), ('a', retry)])
        with db._read_ctx() as conn:
            current = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', ('a',)).fetchone()
            assert _unresolved_execution(conn, current)
            # A queued different ID must not hide this real active retry even though
            # it has the same spelling as that other ID's preferred first key.
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                _claim_execution_id(conn, 'a', 3)
        mutate_worker_execution(db, epoch=retry['epoch'], execution_id=retry['execution_id'],
                                session_id=retry['session_id'], generation=retry['generation'],
                                sequence=1, operation='execution.finish', payload={})
        assert _end_child(child) == 0
        before = _snapshot(db)
        other = claim_session_input(db, epoch=second.epoch, session_id='b')
        child = _inert_child()
        children.append(child)
        scope = _reserve(second, other, child, state='finished')
        assert scope['execution_id'] not in (original['execution_id'], retry['execution_id'])
        after = _snapshot(db)
        assert all(row in after[0] for row in before[0])
        assert all(row in after[1] for row in before[1])
        with db._read_ctx() as conn:
            assert {row[0] for row in admission_worker_rows(conn, 'a:g2')} == {scope['execution_id']}
    finally:
        for child in children:
            if child.poll() is None:
                assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['null-list', 'null-owner', 'missing-owner', 'duplicate-link'])
async def test_invalid_persisted_ownership_fails_closed(tmp_path, monkeypatch, damage):
    store, runner = _runner(tmp_path, monkeypatch, [])
    child = None
    try:
        db = store._db
        db.create_session('s', source='telegram')
        authority = await _owner(runner, 'owner')
        authority._schedule = lambda ref: None
        source = tmp_path / 'legacy.db'
        _legacy(source, [('a', 's')])
        assert import_legacy_session_admissions(db, epoch=authority.epoch, source_path=source,
                                               principal_id='person', writers_drained=True) == 1
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        child = _inert_child()
        scope = _reserve(authority, row, child, state='finished')
        before = _snapshot(db)
        def corrupt(conn):
            if damage == 'missing-owner':
                conn.execute('DELETE FROM state_meta WHERE key=?', (_OWNER_KEY + scope['execution_id'],))
            elif damage == 'duplicate-link':
                conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                             (json.dumps([scope['execution_id']] * 2), _CLAIMS_KEY + 'a'))
            else:
                key = _CLAIMS_KEY + 'a' if damage == 'null-list' else _OWNER_KEY + scope['execution_id']
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', ('null', key))
        db._execute_write(corrupt)
        with db._read_ctx() as conn:
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                admission_worker_rows(conn, 'a')
        assert _snapshot(db) == before
    finally:
        if child is not None and child.poll() is None:
            assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_terminal_retirement_keeps_linked_worker_receipts_readable(tmp_path, monkeypatch):
    from hermes_state_mutation_retirement import retire_terminal_receipts
    from hermes_state_terminal import WORKER_PREFIX
    store, runner = _runner(tmp_path, monkeypatch, [])
    child = None
    try:
        db = store._db
        db.create_session('s', source='telegram')
        authority = await _owner(runner, 'owner')
        authority._schedule = lambda ref: None
        source = tmp_path / 'legacy.db'
        _legacy(source, [('a', 's')])
        assert import_legacy_session_admissions(db, epoch=authority.epoch, source_path=source,
                                               principal_id='person', writers_drained=True) == 1
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        child = _inert_child()
        scope = _reserve(authority, row, child, state='finished')
        assert _end_child(child) == 0
        settle_session_input(db, epoch=authority.epoch, admission_id='a',
                             generation=row['generation'], outcome='completed')
        with db._read_ctx() as conn:
            before = dict(conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                                       (scope['execution_id'],)).fetchone())
            receipts = [dict(r) for r in conn.execute(
                'SELECT sequence,payload_digest,result_json FROM worker_receipts WHERE execution_id=?',
                (scope['execution_id'],))]
        db._execute_write(lambda conn: retire_terminal_receipts(conn, ['s']))
        with db._read_ctx() as conn:
            assert [tuple(r) for r in admission_worker_rows(conn, 'a')] == [(scope['execution_id'], 'terminal', 1)]
            archived = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                              (WORKER_PREFIX + scope['execution_id'],)).fetchone()[0])
        assert all(archived[key] == value for key, value in before.items())
        assert archived['receipts'] == receipts
    finally:
        if child is not None and child.poll() is None:
            assert _end_child(child) == 0
        store.close_all_db_handles()
