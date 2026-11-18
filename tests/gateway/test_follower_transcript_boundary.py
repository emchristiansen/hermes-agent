"""Unknown work stays unknown, but its open transcript cannot become a new request.

Real authority claim/drain and SQLite -> SessionStore -> TurnRunner's history and
message preparation -> AIAgent -> loopback OpenAI request. No business tool runs.
The handler seam omits gateway delivery/UI and agent-cache construction only.
"""
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
from gateway.run_turn_runner import TurnRunner
from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import CLAIM_MARK_PREFIX, claim_session_input, get_session_admission, mutate_worker_execution
from tests.acp_adapter.test_failed_turn_closure import _LoopbackProvider
from tests.gateway.test_authority_unknown_resolution import Peer, _restarted_owner_with_unknown_head
from tests.gateway.test_delivery22_follower_scope import _late_refusals, _retained
from tests.gateway.test_person_message_redelivery import _owner
from tests.gateway.test_redelivery_worker_fence import _claimed_head, _end_child, _inert_child, _reserve


def _consumer(store, runner, provider, executed):
    async def consume(event):
        from gateway.session_results import execution_result
        from run_agent import AIAgent

        executed.append(event.text)
        turn = TurnRunner.__new__(TurnRunner)
        turn._runner = runner
        turn._ctx = SimpleNamespace(
            history=store.load_transcript('s'), channel_prompt=None, user_config={},
            session_id='s', session_key=None, persist_user_message=None,
            persist_user_timestamp=None, inbound_message_id=event.message_id, message=event.text)
        history, _observed, _media = turn._load_turn_history(SimpleNamespace(), False)
        turn._prepare_turn_message(history)
        # Check the actual message handed to the model, not just an earlier DB projection.
        assert turn._ctx.message == event.text
        agent = AIAgent(
            api_key='fixture-only', base_url=provider.base_url, provider='openai-compat',
            model='fixture-model', max_iterations=2, enabled_toolsets=[], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False, platform='cli')
        result = await asyncio.to_thread(
            agent.run_conversation, turn._ctx.message, conversation_history=history)
        assert result.get('final_response') == 'ok', result
        execution_result.get()['result'] = result
        return result['final_response']
    return consume


def _assert_request(provider, head_text):
    requests = [r for r in provider.requests if 'messages' in r]
    assert len(requests) == 1, requests
    sent = [m for m in requests[0]['messages'] if m['role'] != 'system']
    assert [m['content'] for m in sent if m['role'] == 'user'] == [head_text, 'FOLLOWER'], sent
    assert sent[-1] == {'role': 'user', 'content': 'FOLLOWER'}, sent
    assert any(m['role'] == 'assistant' and m.get('content') == PARTIAL_FAILED_TURN_NOTICE
               for m in sent[:-1]), sent
    assert all('display_kind' not in m for m in sent), sent
    print('FOLLOWER_REQUEST', json.dumps(sent), flush=True)


def _deny_boundary(db):
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_boundary BEFORE INSERT ON messages WHEN NEW.role='assistant' "
        "BEGIN SELECT RAISE(ABORT,'boundary failed'); END"))


@pytest.mark.asyncio
@pytest.mark.parametrize('resolution', ['none', 'failed', 'committed'])
@pytest.mark.parametrize('rotated', [False, True], ids=['logical', 'physical-tip'])
async def test_unknown_user_tail_is_not_merged_into_follower(tmp_path, monkeypatch, resolution, rotated):
    """No Discard, a rolled-back Discard, and committed Discard all isolate the NEW input."""
    monkeypatch.setenv('HERMES_DISABLE_PLUGINS', '1')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    executed = []
    store, authority, ref, head, follower = await _restarted_owner_with_unknown_head(
        tmp_path, monkeypatch, executed)
    db = authority.db
    target = 's-tip' if rotated else 's'
    if rotated:
        db.create_session(target, source='telegram', parent_session_id='s')
        db.end_session('s', 'compression')
    db.append_message(target, 'user', 'HEAD lost in restart')
    viewer = AuthorityConnection(authority, Peer(), {'user_id': 'human'})
    provider = _LoopbackProvider()
    try:
        authority._schedule = lambda ref: None
        if resolution != 'none':
            if resolution == 'failed':
                _deny_boundary(db)
            await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
            reply = await viewer.dispatch({'id': 2, 'method': 'prompt.resolve_unknown', 'params': {
                'session_id': 's', 'admission_id': head['admission_id'],
                'execution_generation': head['generation']}})
            assert ('error' in reply) == (resolution == 'failed'), reply
            if resolution == 'failed':
                db._execute_write(lambda conn: conn.execute('DROP TRIGGER deny_boundary'))
        before = get_session_admission(db, admission_id=head['admission_id'])
        provider.script.extend({'finish_reason': 'stop', 'content': 'ok'} for _ in range(4))
        authority.runner._handle_message = _consumer(store, authority.runner, provider, executed)
        del authority._schedule
        authority._schedule(ref)
        await authority.sessions['s'].task
        assert executed == ['FOLLOWER']
        _assert_request(provider, 'HEAD lost in restart')
        assert get_session_admission(db, admission_id=head['admission_id']) == before
        assert before['status'] == ('terminal' if resolution == 'committed' else 'unknown')
        current = get_session_admission(db, admission_id=follower.admission_id)
        assert (current['status'], current['outcome']) == ('terminal', 'completed')
        boundaries = [m for m in db.get_messages(target) if m.get('display_kind') == FAILED_TURN_DISPLAY_KIND]
        assert len(boundaries) == 1 and boundaries[0]['content'] == PARTIAL_FAILED_TURN_NOTICE
        if rotated:
            assert not any(m.get('display_kind') == FAILED_TURN_DISPLAY_KIND for m in db.get_messages('s'))
    finally:
        await asyncio.to_thread(provider.shutdown)
        await viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('tail', ['user', 'pending-call', 'tool'])
async def test_claim_boundary_preserves_old_private_worker_and_late_write_fences(tmp_path, monkeypatch, tail):
    """An old still-live physical child cannot adopt or write, and is not retired by isolation."""
    monkeypatch.setenv('HERMES_DISABLE_PLUGINS', '1')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    executed = []
    store, runner, first, head, follower = await _claimed_head(tmp_path, monkeypatch, executed)
    child, provider = _inert_child(), _LoopbackProvider()
    try:
        old = _reserve(first, head, child, state='running')
        mutate_worker_execution(first.db, epoch=first.epoch, execution_id=old['execution_id'],
            session_id='s', generation=old['generation'], sequence=1,
            operation='usage.main', payload={'input_tokens': 1})
        if tail != 'user':
            first.db.append_message('s', 'assistant', None, tool_calls=[{
                'id': 'old-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}])
        if tail == 'tool':
            first.db.append_message('s', 'tool', 'done', tool_call_id='old-call', tool_name='terminal')
        second = await _owner(runner, 'owner-2')
        before = _retained(second.db, head['admission_id'], old['execution_id'])
        _late_refusals(second.db, first, second, old, head['principal_id'])
        provider.script.extend({'finish_reason': 'stop', 'content': 'ok'} for _ in range(4))
        runner._handle_message = _consumer(store, runner, provider, executed)
        second._schedule(SessionRef('p', 's'))
        await second.sessions['s'].task
        assert executed == ['FOLLOWER']
        _assert_request(provider, 'HEAD')
        assert get_session_admission(second.db, admission_id=follower)['outcome'] == 'completed'
        assert _retained(second.db, head['admission_id'], old['execution_id']) == before
        _late_refusals(second.db, first, second, old, head['principal_id'])
        assert child.poll() is None
        assert second._handle(SessionRef('p', 's')).execution_state == 'unknown'
    finally:
        await asyncio.to_thread(provider.shutdown)
        assert _end_child(child) == 0
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_failed_boundary_rolls_back_normal_claim_and_prevents_inference(tmp_path, monkeypatch):
    """A real SQLite prerequisite failure protects launch; fixing storage resumes the SAME input."""
    executed = []
    store, authority, ref, head, follower = await _restarted_owner_with_unknown_head(
        tmp_path, monkeypatch, executed)
    db = authority.db
    db.append_message('s', 'user', 'HEAD')
    provider = _LoopbackProvider()
    # On the unfixed source launch is unexpectedly reached: keep the negative
    # control responsive so it fails the launch assertion, not provider retries.
    provider.script.extend({'finish_reason': 'stop', 'content': 'ok'} for _ in range(4))
    try:
        authority.runner._handle_message = _consumer(store, authority.runner, provider, executed)
        before = get_session_admission(db, admission_id=head['admission_id'])
        generation = db.get_session('s')['runtime_generation']
        messages = db.get_messages('s')
        _deny_boundary(db)
        authority._schedule(ref)
        with pytest.raises(sqlite3.IntegrityError, match='boundary failed'):
            await authority.sessions['s'].task
        assert executed == [] and provider.requests == []
        assert get_session_admission(db, admission_id=head['admission_id']) == before
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'queued'
        assert db.get_session('s')['runtime_generation'] == generation
        assert db.get_messages('s') == messages
        assert db._read_one('SELECT value FROM state_meta WHERE key=?',
                            (CLAIM_MARK_PREFIX + follower.admission_id,)) is None
        assert authority.executing == {} and authority.sessions['s'].event_stream.execution == {}
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER deny_boundary'))
        provider.script.extend({'finish_reason': 'stop', 'content': 'ok'} for _ in range(4))
        authority._schedule(ref)
        await authority.sessions['s'].task
        assert executed == ['FOLLOWER']
        _assert_request(provider, 'HEAD')
        assert get_session_admission(db, admission_id=head['admission_id']) == before
        assert get_session_admission(db, admission_id=follower.admission_id)['outcome'] == 'completed'
    finally:
        await asyncio.to_thread(provider.shutdown)
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_failed_later_claim_rolls_back_boundary_itself(tmp_path, monkeypatch):
    """A boundary cannot be published without the follower's successful claim transaction."""
    store, authority, _ref, head, follower = await _restarted_owner_with_unknown_head(tmp_path, monkeypatch, [])
    db = authority.db
    try:
        db.append_message('s', 'user', 'HEAD')
        messages = db.get_messages('s')
        before = get_session_admission(db, admission_id=head['admission_id'])
        db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER deny_claim BEFORE UPDATE OF status ON session_admissions "
            "WHEN NEW.status='started' BEGIN SELECT RAISE(ABORT,'claim failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='claim failed'):
            claim_session_input(db, epoch=authority.epoch, session_id='s')
        assert db.get_messages('s') == messages
        assert get_session_admission(db, admission_id=head['admission_id']) == before
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'queued'
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER deny_claim'))
        claimed = claim_session_input(db, epoch=authority.epoch, session_id='s')
        boundary = db.get_messages('s')[-1]
        assert boundary['display_kind'] == FAILED_TURN_DISPLAY_KIND
        assert claimed['transcript_mark'] >= boundary['id']
    finally:
        store.close_all_db_handles()
