"""A messaging route's internal wake steers into the session's running turn (Eric, 2026-11-04).

Producer entry: ``gateway.wake.deliver_wake`` on a wired messaging adapter, which commits the wake
through ``admit_route_wake`` -> ``SessionAuthority.admit_automation`` as a FIFO row
``{'text', 'native_text_v1': {..., 'automation': ...}}``. While a claim runs whose turn adopted a
real steer seam (``BufferedAgent``), the busy pass (``gateway.session_busy_ingress``) transfers that
row into the turn, which takes it at its next tool/result boundary
(``apply_pending_steer_to_tool_results``) framed as Hermes automation, never as the user's text.
An idle session still runs the wake as its own FIFO turn. Real GatewayRunner, DiscordAdapter,
SessionAuthority, admission ledger and drain over SQLite; the turn body is a stand-in.
"""
import asyncio

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from tests.gateway.test_prompt_queued_transfer import BufferedAgent
from tests.waiting import async_wait_until


def _boundary(agent):
    """The turn's tool/result boundary: the steer, if any, appended after a tool result."""
    messages = [{'role': 'assistant', 'tool_calls': []}, {'role': 'tool', 'content': 'fixture result'}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    return messages[-1]['content'] if messages[-1]['role'] == 'user' else None


async def _gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id=f's{len(sent)}')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    return runner, adapter, authority, source, key


def _rows(authority, runner, key):
    from hermes_state_runtime import list_session_admissions
    owner = runner.session_store.peek_session_id(key)
    return list_session_admissions(authority.db, session_id=owner, pending_only=False) if owner else []


def _assert_native_automation_row(row, wake):
    payload = row['payload']
    assert set(payload) == {'text', 'native_text_v1'}, payload
    assert payload['text'] == wake
    native = payload['native_text_v1']
    assert native.get('automation') is not None and not native.get('media'), native
    assert row['principal_id'].split(':')[0] == 'automation', row['principal_id']


@pytest.mark.asyncio
async def test_a_messaging_wake_to_a_busy_session_steers_into_the_running_turn(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_results import execution_result
    from gateway.wake import deliver_wake

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    running, boundary = asyncio.Event(), asyncio.Event()
    executed, taken = [], []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            owner = runner.session_store.peek_session_id(key)
            authority.adopt_agent(owner, authority.executing[owner], agent)  # the built, running agent
            running.set()
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    wake = '[Kanban task completed] t_1 result ready'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)  # accepted, no raise

    def wake_row():
        return next((r for r in _rows(authority, runner, key) if r['payload'].get('text') == wake), None)
    await async_wait_until(lambda: wake_row() is not None and (wake_row()['status'], wake_row()['outcome'])
                           == ('terminal', 'steering'), timeout=10, what='the wake to transfer to the running turn')
    _assert_native_automation_row(wake_row(), wake)
    boundary.set()
    await async_wait_until(lambda: [(r['status'], r['outcome']) for r in _rows(authority, runner, key)]
                           == [('terminal', 'completed'), ('terminal', 'steered')], timeout=10,
                           what='the turn to settle and the wake to settle steered')
    await async_wait_until(lambda: not runner._is_session_running(key), timeout=10, what='the session to release')
    assert executed == ['A'], 'the wake never ran as its own turn'
    [content] = taken
    assert content is not None and wake in content, 'the wake text, at the running turn boundary'
    assert '[DELIVERED MESSAGE — from Hermes automation' in content, content
    frame = content[content.index('[DELIVERED MESSAGE — from Hermes automation'):]
    assert wake in frame, 'the wake text sits inside the automation frame'
    assert 'a direct message from the user' not in content, 'never promoted to the user'


@pytest.mark.asyncio
async def test_a_messaging_wake_to_an_idle_session_runs_as_its_own_turn(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result
    from gateway.wake import deliver_wake

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    started, taken = [], []

    async def turn(event, source, quick_key, run_generation):
        started.append((event.text, executing_admission.get()))
        owner = runner.session_store.peek_session_id(key)
        authority.adopt_agent(owner, authority.executing[owner], agent)
        taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='hello', source=source, message_id='m1'))
    await async_wait_until(lambda: [r['status'] for r in _rows(authority, runner, key)] == ['terminal']
                           and not runner._is_session_running(key), timeout=10,
                           what='the first turn to settle and release the session')
    wake = '[Kanban task completed] t_2 result ready'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)
    await async_wait_until(lambda: [(r['status'], r['outcome']) for r in _rows(authority, runner, key)]
                           == [('terminal', 'completed'), ('terminal', 'completed')], timeout=10,
                           what='the wake to run and settle as its own turn')
    assert started == [('hello', True), (wake, True)], 'the idle wake ran as its own FIFO turn'
    _assert_native_automation_row(_rows(authority, runner, key)[1], wake)
    assert taken == [None, None], 'nothing was steered on an idle session'
