"""Retained proof and old completion cannot consume an independent admission (MR25/MR26).

Essential: private real SQLite, owner recovery, normal drain/resolve/publication,
projection/native control consumers and managed prompt-frame registration.
Incidental: inert handler, captured HEAD answer and EOF-only identity children;
remote response records a wire value, no provider/business operation or human.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from gateway.session_settlement_recovery import _mark_unknown
from gateway.session_worker_reservation import _CLAIMS_KEY, _OWNER_KEY
from hermes_state_runtime import RuntimeStoreError, claim_session_input, get_session_admission
from tests.gateway.test_delivery22_follower_scope import _late_refusals, _retained
from tests.gateway.test_person_message_redelivery import _drained, _owner
from tests.gateway.test_redelivery_worker_fence import _claimed_head, _end_child, _inert_child, _reserve
from tests.gateway.test_unknown_automation_settles import Peer, _statuses
from tools import clarify_gateway


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['foreign-owner', 'malformed-owner', 'missing-link', 'ambiguous-admission-link'])
async def test_retained_proof_refusal_preserves_follower_until_proof_repair(tmp_path, monkeypatch, damage):
    executed = []
    store, runner, first, head, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    child = _inert_child()
    try:
        old = _reserve(first, head, child, state='running')
        second = await _owner(runner, 'replacement-owner')
        keys = (_CLAIMS_KEY + head['admission_id'], _OWNER_KEY + old['execution_id'])
        saved = [tuple(second.db._read_one('SELECT key,value FROM state_meta WHERE key=?', (key,)))
                 for key in keys]
        def damage_proof(conn):
            if damage == 'ambiguous-admission-link':
                duplicate = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                              (head['admission_id'],)).fetchone())
                duplicate.update(admission_id='ambiguous-head-link', request_id='ambiguous-proof', seq=3)
                columns = ','.join(duplicate)
                values = ','.join('?' for _ in duplicate)
                conn.execute(f'INSERT INTO session_admissions({columns}) VALUES({values})', tuple(duplicate.values()))
            elif damage == 'missing-link':
                conn.execute('DELETE FROM state_meta WHERE key=?', (keys[0],))
            else:
                value = '{' if damage == 'malformed-owner' else json.dumps('foreign-admission')
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (value, keys[1]))
        second.db._execute_write(damage_proof)
        before = _retained(second.db, head['admission_id'], old['execution_id'])
        with pytest.raises(RuntimeStoreError) as refused:
            claim_session_input(second.db, epoch=second.epoch, session_id='s', head=follower)
        await _drained(second)
        states = _statuses(second.db)
        print('RETAINED_PROOF_REFUSAL', json.dumps({'damage': damage, 'states': states,
              'executed': executed, 'field': refused.value.field, 'detail': refused.value.detail}), flush=True)
        assert states[head['admission_id']] == ('unknown', None)
        assert states[follower] == ('queued', None)
        assert executed == []
        assert refused.value.field == 'worker_ownership' and refused.value.detail
        assert _retained(second.db, head['admission_id'], old['execution_id']) == before
        _late_refusals(second.db, first, second, old, head['principal_id'])
        # Repair only this private fixture's proof, then retry the still-accepted input.
        def restore(conn):
            if damage == 'ambiguous-admission-link':
                conn.execute('DELETE FROM session_admissions WHERE admission_id=?', ('ambiguous-head-link',))
            for key, value in saved:
                conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                             'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))
        second.db._execute_write(restore)
        await _drained(second)
        assert executed == ['FOLLOWER']
        assert _statuses(second.db)[follower] == ('terminal', 'completed')
        assert _statuses(second.db)[head['admission_id']] == ('unknown', None)
        assert _retained(second.db, head['admission_id'], old['execution_id']) == before
        _late_refusals(second.db, first, second, old, head['principal_id'])
        print('RETAINED_PROOF_RECOVERY', json.dumps({'damage': damage, 'states': _statuses(second.db),
              'executed': executed, 'old_scope_unchanged': True}), flush=True)
    finally:
        assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_queued_policy_conflict_still_refuses_only_that_input(tmp_path, monkeypatch):
    """Negative policy seam: a real claim transaction rejects this queued row twice.

    The owner policy guard is supplied; claim, retry, cancellation and drain are real.
    This must remain distinct from unreadable retained-worker proof.
    """
    executed, attempts = [], []
    store, _runner, owner, head, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    try:
        assert _mark_unknown(owner, head) == 'unknown'
        def rejected(conn):
            attempts.append(follower)
            raise RuntimeStoreError('admission_conflict', field='queued_policy', detail='fixture policy refusal')
        monkeypatch.setattr(owner, '_admission_gate', lambda *args, **kwargs: rejected)
        del owner._schedule
        await _drained(owner)
        assert attempts == [follower, follower]
        assert _statuses(owner.db)[head['admission_id']] == ('unknown', None)
        assert _statuses(owner.db)[follower] == ('terminal', 'cancelled')
        assert executed == []
        print('QUEUED_POLICY_REFUSAL', json.dumps({'attempts': attempts,
              'states': _statuses(owner.db), 'executed': executed}), flush=True)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['native-clarify', 'native-approval', 'remote-clarify', 'remote-approval'])
@pytest.mark.parametrize('resolve_head', [False, True], ids=['no-resolution', 'resolve-head'])
async def test_old_completion_preserves_follower_controls_and_both_outcomes(tmp_path, monkeypatch, kind, resolve_head):
    store, runner, owner, head, follower = await _claimed_head(tmp_path, monkeypatch, [])
    entered, release, registered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    frames, remote_answers = [], []
    approval_task, native_entry = None, None
    old_entry = clarify_gateway.register('old-control', 'private-old-route', 'Old inert question', ['yes'])
    live, ref = owner.sessions['s'], SessionRef('p', 's')
    live.controls.register_clarify('s', head['generation'], old_entry)
    route, prompt_id = 'private-follower-approval', 'follower-control'
    try:
        owner.pending_results[head['admission_id']] = {
            'result': {'final_response': 'ANSWER_HEAD', 'completed': True, 'messages': []}, 'usage': {}}
        assert _mark_unknown(owner, head) == 'unknown'
        async def follower_handler(event):
            assert event.text == 'FOLLOWER'
            entered.set()
            await release.wait()
            from gateway.session_results import execution_result
            execution_result.get()['result'] = {
                'final_response': 'ANSWER_FOLLOWER', 'completed': True, 'messages': []}
            return 'ANSWER_FOLLOWER'
        runner._handle_message = follower_handler
        live.event_stream.observers.add(frames.append)
        del owner._schedule
        owner._schedule(ref)
        await asyncio.wait_for(entered.wait(), 10)
        stamp = dict(live.event_stream.execution)
        generation = stamp['execution_generation']
        assert stamp['admission_id'] == follower
        connection = AuthorityConnection(owner, Peer(), {'user_id': 'human'})
        # Fixture attachment only; respond still checks real actor/capability/current generation.
        live.subscribers['private-test-attachment'] = connection.actor
        if kind == 'native-clarify':
            native_entry = clarify_gateway.register(prompt_id, 'private-follower-route', 'Inert question', ['yes'])
            owner.register_clarify('s', generation, native_entry)
        elif kind == 'native-approval':
            from tools.approval_gateway_wait import _await_gateway_decision
            loop = asyncio.get_running_loop()
            def notify(data):
                owner.register_approval('s', generation, route, data)
                loop.call_soon_threadsafe(registered.set)
            approval_task = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, route, notify,
                {'request_id': prompt_id, 'command': 'inert-fixture', 'description': 'No command executes'}))
            await asyncio.wait_for(registered.wait(), 10)
        else:
            from gateway.session_managed_worker import _prompt_frame
            worker = SimpleNamespace(respond=lambda *answer: remote_answers.append(answer))
            row = get_session_admission(owner.db, admission_id=follower)
            frame = ({'type': 'clarify', 'prompt_id': prompt_id, 'question': 'Inert remote question',
                      'choices': ['yes'], 'multi_select': False} if kind == 'remote-clarify' else
                     {'type': 'approval', 'data': {'request_id': prompt_id, 'command': 'inert-remote-fixture'}})
            with live.event_stream.lock:
                assert _prompt_frame(owner, ref, row, worker, frame)
        if resolve_head:
            await owner.resolve_unknown_as_owner(ref, head['admission_id'], head['generation'])
        # Try the actual responder before assertions, so a PRE run shows lost eligibility.
        params = {'session_id': 's', 'execution_generation': generation, 'prompt_id': prompt_id}
        if kind.endswith('clarify'):
            response = await connection.respond_clarify(ref, dict(params, answer='yes'))
        else:
            response = await connection.respond(ref, dict(params, choice='once'))
        witness = {'kind': kind, 'resolve_head': resolve_head, 'response': response,
                   'remote_answers': remote_answers, 'active_stamp': dict(live.event_stream.execution),
                   'states': _statuses(owner.db), 'frames': [f['params'] for f in frames]}
        print('OLD_COMPLETION_CONTROL', json.dumps(witness), flush=True)
        assert live.event_stream.execution == stamp
        assert response['status'] == 'resolved'
        if native_entry is not None:
            assert native_entry.event.is_set() and native_entry.response == 'yes'
        if approval_task is not None:
            actual = await asyncio.wait_for(asyncio.shield(approval_task), 10)
            assert actual['resolved'] and actual['choice'] == 'once'
        if kind.startswith('remote'):
            assert remote_answers == [(kind.split('-')[1], prompt_id, 'yes' if kind.endswith('clarify') else 'once')]
        assert owner.publish_execution('s', head['generation'], 'tool.start', {'tool_id': 'late-old'}) is False
        if resolve_head:
            assert old_entry.clarify_id not in live.controls.pending
            old_frames = [f['params'] for f in frames if f['params']['type'] == 'clarify.settled'
                          and f['params']['payload']['prompt_id'] == old_entry.clarify_id]
            assert len(old_frames) == 1
            assert old_frames[0]['admission_id'] == head['admission_id']
            assert old_frames[0]['execution_generation'] == head['generation']
        release.set()
        await _drained(owner)
        complete = {f['params']['payload']['admission_id']: f['params'] for f in frames
                    if f['params']['type'] == 'message.complete'}
        expected = {follower: ('ANSWER_FOLLOWER', generation)}
        if resolve_head:
            expected[head['admission_id']] = ('ANSWER_HEAD', head['generation'])
        assert set(complete) == set(expected)
        for admission_id, (answer, gen) in expected.items():
            frame = complete[admission_id]
            assert frame['admission_id'] == admission_id
            assert frame['execution_generation'] == gen
            assert frame['payload']['content'] == answer
            assert frame['authority_epoch'] == owner.epoch
            assert _statuses(owner.db)[admission_id] == ('terminal', 'completed')
        assert live.event_stream.execution == {}
        print('INDIVIDUAL_COMPLETIONS', json.dumps({'kind': kind, 'resolve_head': resolve_head,
              'complete': complete, 'states': _statuses(owner.db)}), flush=True)
    finally:
        from tools import approval
        approval.resolve_gateway_approval(route, 'deny', resolve_all=True)
        if approval_task is not None:
            await asyncio.wait_for(asyncio.shield(approval_task), 10)
        for entry in [old_entry, native_entry]:
            if entry is not None:
                clarify_gateway.resolve_gateway_clarify(entry.clarify_id, 'fixture cleanup')
                clarify_gateway.wait_for_response(entry.clarify_id, 0.01)
        release.set()
        await _drained(owner)
        store.close_all_db_handles()
