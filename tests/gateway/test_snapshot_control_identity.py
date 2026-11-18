"""Old-control cleanup belongs to its admission, not the running follower (MR-SCOPE28).

Real SQLite/claim/drain/resolve, API observers, replay and native control waiters.
The model handler is inert; the remote callback records the wire response only.
Subscriber attachment and captured HEAD answer are fixture seams, not live effects.
"""
import asyncio
import json
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from gateway.session_api_turn import observe_api_controls
from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from gateway.session_managed_worker import _prompt_frame
from gateway.session_settlement_recovery import _mark_unknown
from hermes_state_runtime import get_session_admission
from tests.gateway.test_person_message_redelivery import _drained
from tests.gateway.test_redelivery_worker_fence import _claimed_head
from tests.gateway.test_unknown_automation_settles import Peer, _statuses
from tools import approval, clarify_gateway


async def _register(owner, ref, row, kind, prompt_id, answers, entries, tasks):
    if kind == 'native-clarify':
        entry = clarify_gateway.register(prompt_id, prompt_id + '-route', 'Inert question', ['yes'])
        entries.append(entry)
        owner.register_clarify(ref.session_id, row['generation'], entry)
        return entry
    if kind == 'native-approval':
        from tools.approval_gateway_wait import _await_gateway_decision
        registered = asyncio.Event()
        loop = asyncio.get_running_loop()
        route = prompt_id + '-route'
        def notify(data):
            owner.register_approval(ref.session_id, row['generation'], route, data)
            loop.call_soon_threadsafe(registered.set)
        task = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, route, notify,
            {'request_id': prompt_id, 'command': 'inert-never-executed', 'description': 'No command runs'}))
        tasks.append((route, task))
        await asyncio.wait_for(registered.wait(), 10)
        return task
    worker = SimpleNamespace(respond=lambda *args: answers.append(args))
    frame = ({'type': 'clarify', 'prompt_id': prompt_id, 'question': 'Inert remote question',
              'choices': ['yes'], 'multi_select': False} if kind == 'remote-clarify' else
             {'type': 'approval', 'data': {'request_id': prompt_id, 'command': 'inert-never-executed'}})
    with owner.sessions[ref.session_id].event_stream.lock:
        assert _prompt_frame(owner, ref, row, worker, frame)
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize('snapshot_first', [False, True], ids=['complete-first', 'snapshot-first'])
@pytest.mark.parametrize('kind', ['native-clarify', 'native-approval', 'remote-clarify', 'remote-approval'])
async def test_old_control_cleanup_keeps_owning_identity_and_follower_response(tmp_path, monkeypatch, kind, snapshot_first):
    store, runner, owner, head, follower = await _claimed_head(tmp_path, monkeypatch, [])
    entered, release = asyncio.Event(), asyncio.Event()
    frames, old_api, new_api, answers, entries, tasks = [], [], [], [], [], []
    live, ref = owner.sessions['s'], SessionRef('p', 's')
    head_stamp = {'authority_epoch': head['owner_epoch'], 'execution_generation': head['generation'],
                  'admission_id': head['admission_id']}
    # The helper manually claimed HEAD; use that committed claim's genuine identity.
    live.event_stream.execution = dict(head_stamp)
    async def handler(event):
        assert event.text == 'FOLLOWER'
        entered.set()
        await release.wait()
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'ANSWER_FOLLOWER', 'completed': True, 'messages': []}
        return 'ANSWER_FOLLOWER'
    try:
        await _register(owner, ref, head, kind, 'head-control', answers, entries, tasks)
        owner.pending_results[head['admission_id']] = {
            'result': {'final_response': 'ANSWER_HEAD', 'completed': True, 'messages': []}, 'usage': {}}
        assert _mark_unknown(owner, head) == 'unknown'
        runner._handle_message = handler
        live.event_stream.observers.add(frames.append)
        epoch, seq = live.event_stream.watermark()
        del owner._schedule
        owner._schedule(ref)
        await asyncio.wait_for(entered.wait(), 10)
        follower_row = get_session_admission(owner.db, admission_id=follower)
        follower_stamp = dict(live.event_stream.execution)
        assert follower_stamp['admission_id'] == follower
        native = await _register(owner, ref, follower_row, kind, 'follower-control', answers, entries, tasks)
        connection = AuthorityConnection(owner, Peer(), {'user_id': 'human'})
        live.subscribers['private-fixture'] = connection.actor
        with ExitStack() as stack:
            stack.enter_context(observe_api_controls((owner, ref, head), lambda t, p: old_api.append((t, p))))
            if not snapshot_first:
                await owner.resolve_unknown_as_owner(ref, head['admission_id'], head['generation'])
            stack.enter_context(observe_api_controls((owner, ref, follower_row), lambda t, p: new_api.append((t, p))))
            if snapshot_first:
                # Snapshot already settled the old projection; resolving its durable result
                # must neither publish it twice nor remove the follower's native/remote waiter.
                assert 'head-control' not in live.controls.pending
                assert _statuses(owner.db)[head['admission_id']] == ('unknown', None)
                await owner.resolve_unknown_as_owner(ref, head['admission_id'], head['generation'])
            params = {'session_id': 's', 'execution_generation': follower_row['generation'], 'prompt_id': 'follower-control'}
            if kind.endswith('clarify'):
                response = await connection.respond_clarify(ref, dict(params, answer='yes'))
            else:
                response = await connection.respond(ref, dict(params, choice='once'))
            assert response['status'] == 'resolved'
            if kind == 'native-clarify':
                assert native.event.is_set() and native.response == 'yes'
            elif kind == 'native-approval':
                decision = await asyncio.wait_for(asyncio.shield(native), 10)
                assert decision['resolved'] and decision['choice'] == 'once'
            else:
                assert answers == [(kind.split('-')[1], 'follower-control', 'yes' if kind.endswith('clarify') else 'once')]
            assert live.event_stream.execution == follower_stamp
            assert owner.publish_execution('s', head['generation'], 'tool.start', {'tool_id': 'late-old'}) is False
            release.set()
            await _drained(owner)
        settled = [f['params'] for f in frames if f['params']['type'].endswith('.settled')]
        old = [f for f in settled if f['payload']['prompt_id'] == 'head-control']
        assert len(old) == 1
        assert {k: old[0][k] for k in head_stamp} == head_stamp
        assert old[0]['payload'] == {'prompt_id': 'head-control', 'execution_generation': head['generation']}
        assert [(t, p) for t, p in old_api if t.endswith('.settled')] == [(old[0]['type'], old[0]['payload'])]
        assert not any(p.get('prompt_id') == 'head-control' for _t, p in new_api)
        assert any(t.endswith('.request') and p['prompt_id'] == 'follower-control' for t, p in new_api)
        assert any(t.endswith('.settled') and p['prompt_id'] == 'follower-control' for t, p in new_api)
        replay = live.event_stream.since(epoch, seq)
        assert not replay['snapshot_required']
        assert [f for f in replay['events'] if f['type'].endswith('.settled')] == settled
        completed = [f['params'] for f in frames if f['params']['type'] == 'message.complete']
        assert {f['admission_id'] for f in completed} == {head['admission_id'], follower}
        assert all(f['admission_id'] == f['payload']['admission_id'] for f in completed)
        assert _statuses(owner.db)[head['admission_id']] == ('terminal', 'completed')
        assert _statuses(owner.db)[follower] == ('terminal', 'completed')
        assert live.event_stream.execution == {}
        print('SNAPSHOT_CONTROL_IDENTITY', json.dumps(dict(kind=kind, snapshot_first=snapshot_first,
            head_stamp=head_stamp, follower_stamp=follower_stamp, settled=settled, old_api=old_api,
            new_api=new_api, replay=replay, states=_statuses(owner.db))), flush=True)
    finally:
        for route, task in tasks:
            approval.resolve_gateway_approval(route, 'deny', resolve_all=True)
            await asyncio.wait_for(asyncio.shield(task), 10)
        for entry in entries:
            clarify_gateway.resolve_gateway_clarify(entry.clarify_id, 'fixture cleanup')
            clarify_gateway.wait_for_response(entry.clarify_id, 0.01)
        release.set()
        await _drained(owner)
        store.close_all_db_handles()


@pytest.mark.parametrize('cleanup', ['snapshot', 'complete', 'worker'])
def test_cleanup_keeps_captured_epoch_when_active_stamp_changes_in_place(cleanup):
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    events = SessionEvents()
    controls = PendingControls(events)
    owner = SimpleNamespace(sessions={'s': SimpleNamespace(controls=controls, event_stream=events)})
    ref = SessionRef('p', 's')
    old = {'authority_epoch': 7, 'execution_generation': 1, 'admission_id': 'head'}
    events.execution = dict(old)
    entry = SimpleNamespace(clarify_id='old', question='Inert?', choices=[], multi_select=False,
                            event=threading.Event())
    controls.register_clarify('s', 1, entry)
    epoch, seq = events.watermark()
    events.execution.update(authority_epoch=8, execution_generation=2, admission_id='follower')
    current = dict(events.execution)
    if cleanup == 'snapshot':
        assert controls.snapshot('s', 2) == ()
    elif cleanup == 'complete':
        controls.complete_execution('s', dict(old))
    else:
        assert _prompt_frame(owner, ref, {'generation': 1}, None, {'type': 'prompt_settled', 'prompt_id': 'old'})
    replay = events.since(epoch, seq)
    assert not replay['snapshot_required'] and len(replay['events']) == 1
    frame = replay['events'][0]
    assert {k: frame[k] for k in old} == old
    assert frame['type'] == 'clarify.settled' and frame['payload']['execution_generation'] == 1
    assert events.execution == current
    # Idempotent cleanup, including a delayed worker frame, produces no duplicate.
    assert _prompt_frame(owner, ref, {'generation': 1}, None, {'type': 'prompt_settled', 'prompt_id': 'old'})
    assert events.since(epoch, seq)['events'] == [frame]


def test_stale_worker_settlement_cannot_consume_reused_current_prompt_id():
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    events = SessionEvents()
    controls = PendingControls(events)
    owner = SimpleNamespace(sessions={'s': SimpleNamespace(controls=controls, event_stream=events)})
    ref = SessionRef('p', 's')
    events.execution = {'authority_epoch': 8, 'execution_generation': 2, 'admission_id': 'follower'}
    entry = SimpleNamespace(clarify_id='reused', question='Inert?', choices=[], multi_select=False,
                            event=threading.Event())
    controls.register_clarify('s', 2, entry)
    replies = []
    controls.remote_responders['reused'] = lambda *args: replies.append(args)
    epoch, seq = events.watermark()
    assert _prompt_frame(owner, ref, {'generation': 1}, None, {'type': 'prompt_settled', 'prompt_id': 'reused'})
    assert 'reused' in controls.pending and 'reused' in controls.remote_responders
    assert events.since(epoch, seq)['events'] == []
    assert controls.respond('s', 2, 'reused', {'answer': 'yes'}, kind='clarify')['status'] == 'resolved'
    assert replies == [('clarify', 'reused', 'yes')]
    settled = events.since(epoch, seq)['events']
    assert len(settled) == 1 and settled[0]['admission_id'] == 'follower'
    assert settled[0]['execution_generation'] == 2 and settled[0]['authority_epoch'] == 8
