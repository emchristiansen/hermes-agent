"""Retained opaque roots are disjoint from real owner-minted child trees.

Real legacy admission import/reservation and child.reserve/read/finish consumers
produce all worker and receipt rows. No worker/admission rows are fabricated.
"""
from functools import partial
from pathlib import Path
import importlib.util
import json
import os
from types import SimpleNamespace

import psutil
import pytest

import hermes_state_runtime as rt
from hermes_cli.process_identity import worker_birth
from gateway.session_worker_child import delegation_limits, reserve_child_execution
from gateway.session_worker_reservation import owner_worker_subject
from tests.gateway.test_managed_identity_collision import _legacy
from tests.gateway.test_person_message_redelivery import _owner
from tests.gateway.test_redelivery_worker_fence import _end_child, _hello, _inert_child
from tests.gateway.test_unknown_automation_settles import _runner
from tests.hermes_state.test_runtime_worker_children import ledger as ledger_fixture

ledger = ledger_fixture


def _original_reserver():
    path = Path(__file__).resolve().parents[1] / 'fixtures' / 'retained_worker_reservation_13f.py'
    spec = importlib.util.spec_from_file_location('retained_original_reserver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reserve_admission_worker


def _child(db, scope, sequence, identity):
    handler = partial(reserve_child_execution, epoch=scope['epoch'], execution_id=scope['execution_id'],
                      identity=identity,
                      principal='person', limits=(8, 8), refused=None)
    result = rt.apply_worker_receipt(db, **scope, sequence=sequence, operation='child.reserve',
                                    payload={'secret': f'child-fixture-secret-{sequence:08d}'},
                                    handlers={'child.reserve': handler})
    assert 'child' in result
    return dict(epoch=scope['epoch'], **result['child'])


def _history(db, scope, sequence):
    rt.mutate_worker_execution(db, **scope, sequence=sequence, operation='transcript.append',
                              payload={'messages': [{'role': 'assistant', 'content': 'retained child output'}]})
    return rt.mutate_worker_execution(db, **scope, sequence=sequence + 1, operation='compression.history',
        payload=dict(target=scope['session_id'], include_ancestors=False, include_inactive=False,
                     repair_alternation=False, include_row_ids=False, include_compacted=False))


def _tree_snapshot(db, scopes):
    ids = json.dumps([scope['execution_id'] for scope in scopes])
    with db._read_ctx() as conn:
        workers = [dict(row) for row in conn.execute(
            'SELECT * FROM worker_executions WHERE execution_id IN (SELECT value FROM json_each(?)) ORDER BY execution_id', (ids,))]
        receipts = [dict(row) for row in conn.execute(
            'SELECT * FROM worker_receipts WHERE execution_id IN (SELECT value FROM json_each(?)) ORDER BY execution_id,sequence', (ids,))]
    return dict(workers=workers, receipts=receipts,
                sessions=[db.get_session(scope['session_id']) for scope in scopes],
                messages=[db.get_messages(scope['session_id']) for scope in scopes])


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup', ['settle', 'restart-discard', 'worker-finish'])
async def test_legacy_root_retirement_preserves_foreign_root_entire_real_subtree_and_receipts(tmp_path, monkeypatch, cleanup):
    store, runner = _runner(tmp_path, monkeypatch, [])
    processes = []
    try:
        db = store._db
        for sid in ['s', 'foreign']:
            db.create_session(sid, source='telegram')
        owner = await _owner(runner, 'retained-tree-owner')
        owner._schedule = lambda ref: None
        imported = tmp_path / 'legacy.db'
        _legacy(imported, [('a', 's'), ('a/child:b', 'foreign')])
        assert rt.import_legacy_session_admissions(db, epoch=owner.epoch, source_path=imported,
                                                  principal_id='person', writers_drained=True) == 2
        rows, scopes, identities = {}, {}, {}
        for sid in ['s', 'foreign']:
            row = rows[sid] = rt.claim_session_input(db, epoch=owner.epoch, session_id=sid)
            process = _inert_child()
            processes.append(process)
            reservation = _original_reserver()(owner, admission_id=row['admission_id'], process=process,
                principal_id=row['principal_id'], hello=_hello(process), worker_subject=owner_worker_subject())
            scopes[sid] = dict(epoch=owner.epoch, execution_id=reservation['execution_id'],
                               session_id=sid, generation=row['generation'])
            identities[sid] = dict(profile_id=owner.profile_id, pid=process.pid,
                                   birth=worker_birth(psutil.Process(process.pid)))
        own_child = _child(db, scopes['s'], 1, identities['s'])
        foreign_child = _child(db, scopes['foreign'], 1, identities['foreign'])
        foreign_grandchild = _child(db, foreign_child, 1, identities['foreign'])
        for scope, sequence in [(scopes['s'], 2), (scopes['foreign'], 2),
                                (own_child, 1), (foreign_child, 2), (foreign_grandchild, 1)]:
            assert _history(db, scope, sequence)['messages']
        if cleanup == 'restart-discard':
            owner = await _owner(runner, 'replacement-retained-tree-owner')
            owner._schedule = lambda ref: None
        foreign_tree = [scopes['foreign'], foreign_child, foreign_grandchild]
        before = _tree_snapshot(db, foreign_tree)
        if cleanup == 'settle':
            rt.settle_session_input(db, epoch=owner.epoch, admission_id=rows['s']['admission_id'],
                                    generation=rows['s']['generation'], outcome='completed')
        elif cleanup == 'restart-discard':
            rt.resolve_unknown_session_input(db, epoch=owner.epoch, admission_id=rows['s']['admission_id'],
                                             generation=rows['s']['generation'])
        else:
            rt.mutate_worker_execution(db, **scopes['s'], sequence=4, operation='execution.finish', payload={})
            from gateway.session_message_redelivery import _unresolved_execution
            with db._read_ctx() as conn:
                current = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                       (rows['s']['admission_id'],)).fetchone()
                assert not _unresolved_execution(conn, current), 'foreign child tree blocked a proven-finished root'
            rt.settle_session_input(db, epoch=owner.epoch, admission_id=rows['s']['admission_id'],
                                    generation=rows['s']['generation'], outcome='completed')
        assert _tree_snapshot(db, foreign_tree) == before
        assert db.get_session('foreign')['ended_at'] is None
        own = _tree_snapshot(db, [own_child])
        assert own['workers'][0]['status'] == 'terminal'
        assert own['sessions'][0]['end_reason'] == 'agent_close'
        assert any(json.loads(row['result_json']) == {'retired': True} for row in own['receipts'])
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch|stale_generation'):
            rt.mutate_worker_execution(db, **(own_child | {'epoch': owner.epoch}), sequence=3,
                operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': 'late'}]})
        rt.admit_session_input(db, epoch=owner.epoch, principal_id='person', session_id='s',
                               request_id='follower-after-retirement', payload={'text': 'FOLLOWER'})
        follower = rt.claim_session_input(db, epoch=owner.epoch, session_id='s')
        assert follower['request_id'] == 'follower-after-retirement'
        assert _tree_snapshot(db, foreign_tree) == before
    finally:
        for process in processes:
            assert _end_child(process) == 0
        store.close_all_db_handles()


@pytest.mark.parametrize('root', ['worker', 'worker/child:foreign'])
def test_opaque_root_spelling_changes_neither_child_capacity_nor_depth(ledger, root):
    db, epoch = ledger
    rt.register_worker_execution(db, epoch=epoch, execution_id='worker/child:foreign', session_id='other',
                                 generation=0, kind='compute', adoption_secret='retained-fixture')
    handler = partial(reserve_child_execution, epoch=epoch, execution_id=root,
                      identity={'profile_id': 'fixture', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=(1, 1), refused=None)
    sid = 'parent' if root == 'worker' else 'other'
    result = rt.apply_worker_receipt(db, epoch=epoch, execution_id=root, session_id=sid, generation=0,
        sequence=1, operation='child.reserve', payload={'secret': 'retained-fixture-child-secret'},
        handlers={'child.reserve': handler})
    assert 'child' in result


def test_root_policy_lookup_stops_at_actual_foreign_compute_root(ledger, monkeypatch):
    import gateway.session_policy as policy
    db, epoch = ledger
    root = 'worker/child:foreign'
    rt.register_worker_execution(db, epoch=epoch, execution_id=root, session_id='other', generation=0,
                                 kind='compute', adoption_secret='retained-fixture')
    configs = {'parent': {'delegation': {'max_concurrent_children': 1, 'max_spawn_depth': 1}},
               'other': {'delegation': {'max_concurrent_children': 3, 'max_spawn_depth': 4}}}
    monkeypatch.setattr(policy, 'policy_for_source', lambda runner, source:
                        SimpleNamespace(config=lambda owner: configs[source]))
    authority = SimpleNamespace(db=db, sessions={sid: SimpleNamespace(source=sid) for sid in configs},
                                logical_owner=lambda sid: sid, runner=object())
    assert delegation_limits(authority, root, 'other') == (3, 4)


@pytest.mark.parametrize('root', ['worker', 'worker/child:opaque'])
def test_finishing_compute_root_never_ends_its_session_by_spelling(tmp_path, root):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('root', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='root-owner')
        scope = dict(epoch=epoch, execution_id=root, session_id='root', generation=0)
        rt.register_worker_execution(db, **scope, kind='compute', adoption_secret='root-fixture')
        rt.mutate_worker_execution(db, **scope, sequence=1, operation='execution.finish', payload={})
        assert db.get_session('root')['ended_at'] is None
    finally:
        db.close()
