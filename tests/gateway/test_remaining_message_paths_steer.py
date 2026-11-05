"""The remaining message paths into a busy session steer (Eric, 2026-11-04).

(1) A messaging user's own text that reaches a busy session under the session authority BEFORE the
running turn adopted its agent: the messaging busy path has no built agent to steer into
(``_hm_busy_preempt`` sees the pending sentinel), so the text is admitted to the durable FIFO as a
native row (``{'text', 'native_text_v1'}``, no ``automation``). When the turn adopts its agent the busy
pass (``gateway.session_busy_ingress``) transfers it; the turn takes it at its next tool/result boundary
framed as the user's out-of-band message, with the origin the messaging busy path adds
(``_steer_text_with_origin``), never as a delivered automation message, and never runs it separately.

(2) A wake with NO session authority (legacy path): ``gateway.wake.deliver_wake`` falls back to the
adapter's ``handle_message``. When the runner's session state holds a running agent with the steer
seam, the busy handler that receives it steers it framed as Hermes automation, never interrupting and
never queueing it: the adapter's busy path when the adapter holds the running turn's guard (the normal
legacy turn, ``_handle_active_session_busy_message``), else the runner's
(``_hm_handle_running_session_message``). With no built agent, or another event already waiting, it
queues behind the turn as before.

Real GatewayRunner, DiscordAdapter, (in (1)) SessionAuthority, admission ledger and drain over SQLite;
the turn body is a stand-in; the boundary is the canonical ``apply_pending_steer_to_tool_results``.
"""
import asyncio
import json

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from tests.gateway.test_prompt_queued_transfer import BufferedAgent
from tests.waiting import async_wait_until

USER_MARKER = 'a direct message from the user'
AUTOMATION_MARKER = '[DELIVERED MESSAGE — from Hermes automation'


def _boundary(agent):
    """The turn's tool/result boundary: the steer, if any, appended after a tool result."""
    messages = [{'role': 'assistant', 'tool_calls': []}, {'role': 'tool', 'content': 'fixture result'}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    return messages[-1]['content'] if messages[-1]['role'] == 'user' else None


async def _gateway(monkeypatch, *, authority=True):
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
    owner = await initialize_session_authority(runner, profile_id='default', instance_id='owner') if authority else None
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    return runner, adapter, owner, source, key


def _rows(authority, runner, key):
    from hermes_state_runtime import list_session_admissions
    owner = runner.session_store.peek_session_id(key)
    return list_session_admissions(authority.db, session_id=owner, pending_only=False) if owner else []


@pytest.mark.asyncio
async def test_a_human_message_before_adoption_is_admitted_natively_and_steers_as_the_users_at_adoption(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session_results import execution_result

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    running, adopt_now, boundary = asyncio.Event(), asyncio.Event(), asyncio.Event()
    executed, taken, interrupts = [], [], []
    monkeypatch.setattr(agent, 'interrupt', lambda *a, **k: interrupts.append(a), raising=False)

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            running.set()
            await asyncio.wait_for(adopt_now.wait(), 30)  # the turn is still building its agent
            owner = runner.session_store.peek_session_id(key)
            authority.adopt_agent(owner, authority.executing[owner], agent)
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    assert runner._peek_session_state(key).turn.agent is _AGENT_PENDING_SENTINEL, 'no agent built yet'
    human = 'also check the staging logs'
    await adapter.handle_message(MessageEvent(text=human, source=source, message_id='B'))

    def human_row():
        return next((r for r in _rows(authority, runner, key) if r['payload'].get('text') == human), None)
    await async_wait_until(lambda: human_row() is not None and human_row()['status'] == 'queued', timeout=10,
                           what='the human text to be admitted to the FIFO before adoption')
    payload = human_row()['payload']
    assert set(payload) == {'text', 'native_text_v1'}, payload
    assert 'automation' not in payload['native_text_v1'] and not payload['native_text_v1'].get('media')
    assert human_row()['principal_id'].split(':')[0] == 'messaging', human_row()['principal_id']
    assert agent._pending_steer is None, 'nothing reached the turn before it had an agent'

    adopt_now.set()
    await async_wait_until(lambda: (human_row()['status'], human_row()['outcome']) == ('terminal', 'steering'),
                           timeout=10, what='adoption to transfer the human text to the running turn')
    boundary.set()
    await async_wait_until(lambda: [(r['status'], r['outcome']) for r in _rows(authority, runner, key)]
                           == [('terminal', 'completed'), ('terminal', 'steered')], timeout=10,
                           what='the turn to settle and the human text to settle steered')
    await async_wait_until(lambda: not runner._is_session_running(key), timeout=10, what='the session to release')
    assert executed == ['A'], 'the human text never ran as its own turn'
    assert interrupts == []
    [content] = taken
    assert content is not None and human in content, 'the human text, at the running turn boundary'
    assert USER_MARKER in content, content
    assert '[DELIVERED MESSAGE' not in content, 'the user text is never framed as a delivered message'
    assert 'Gateway message origin (JSON data' in content, 'the messaging busy path origin is kept'
    origin_line = content[content.index('Gateway message origin'):].splitlines()[1]
    origin = json.loads(origin_line)
    assert (origin['platform'], origin['chat_id'], origin['user_id']) == ('discord', '42', '42'), origin
    assert content.index(USER_MARKER) < content.index('Gateway message origin') < content.index(human)


@pytest.mark.asyncio
async def test_a_legacy_wake_steers_into_the_runners_running_agent_as_hermes_automation(monkeypatch):
    """No authority; the runner's session state holds the running turn's agent (steer seam) while the
    adapter holds no guard for it: the wake's adapter admission reaches the runner's busy branch."""
    from gateway.wake import deliver_wake

    runner, adapter, authority, source, key = await _gateway(monkeypatch, authority=False)
    assert authority is None and getattr(runner, 'session_authority', None) is None
    agent, interrupts, executed = BufferedAgent(), [], []
    monkeypatch.setattr(agent, 'interrupt', lambda *a, **k: interrupts.append(a), raising=False)

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        return 'never'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    runner._session_state(key).turn.agent = agent  # the built, running agent
    assert runner._is_session_running(key) and key not in adapter._active_sessions

    wake = '[Kanban task completed] t_9 result ready'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)  # accepted, no raise
    await async_wait_until(lambda: agent._pending_steer is not None and key not in adapter._active_sessions,
                           timeout=10, what='the wake to steer into the running agent and the handler to finish')
    assert interrupts == [], 'an internal wake never interrupts'
    assert executed == [], 'the wake never ran as its own turn'
    assert key not in adapter._pending_messages, 'never queued behind the turn'
    assert runner._peek_session_state(key).turn.agent is agent, 'the running turn kept its agent'
    content = _boundary(agent)
    assert content is not None and AUTOMATION_MARKER in content, content
    frame = content[content.index(AUTOMATION_MARKER):]
    assert wake in frame, 'the wake text sits inside the automation frame'
    assert USER_MARKER not in content, 'never promoted to the user'
    assert _boundary(agent) is None, 'delivered once'


@pytest.mark.asyncio
async def test_a_legacy_wake_with_no_running_agent_still_queues(monkeypatch):
    """No authority; the runner's turn is still building its agent (pending sentinel): the runner's busy
    branch has nothing to steer into and queues the wake behind the turn, as before."""
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.wake import deliver_wake

    runner, adapter, _, source, key = await _gateway(monkeypatch, authority=False)
    executed = []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    queued = asyncio.Event()
    original = runner._queue_or_replace_pending_event

    def queue(session_key, event):
        original(session_key, event)
        queued.set()
    monkeypatch.setattr(runner, '_queue_or_replace_pending_event', queue)
    state = runner._session_state(key)
    state.turn.agent = _AGENT_PENDING_SENTINEL

    wake = '[Kanban task completed] t_10 result ready'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)
    await asyncio.wait_for(queued.wait(), 10)
    assert executed == [], 'nothing ran while the turn holds the session'
    state.turn.agent = None  # the running turn ends: the queued wake runs as its own turn
    await async_wait_until(lambda: wake in executed, timeout=10, what='the queued wake to run as its own turn')
    assert executed == [wake]


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_built', [True, False])
async def test_a_legacy_wake_to_an_adapter_guarded_turn_steers_into_its_built_agent(monkeypatch, agent_built):
    """No authority; the turn runs through the adapter's own guard (the normal legacy turn). The adapter's
    busy path (``_handle_active_session_busy_message``) steers the internal wake into the built agent as
    Hermes automation's; before the agent is built it leaves it to the base adapter queue, to run as its
    own turn after the running one. It never interrupts."""
    from gateway.platforms.event import MessageEvent
    from gateway.wake import deliver_wake

    runner, adapter, _, source, key = await _gateway(monkeypatch, authority=False)
    agent, interrupts = BufferedAgent(), []
    monkeypatch.setattr(agent, 'interrupt', lambda *a, **k: interrupts.append(a), raising=False)
    running, finish, executed = asyncio.Event(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            if agent_built:
                runner._session_state(quick_key).turn.agent = agent  # the built, running agent
            running.set()
            await asyncio.wait_for(finish.wait(), 30)
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    assert key in adapter._active_sessions
    wake = '[Kanban task completed] t_11 result ready'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)  # accepted, no raise
    assert interrupts == [], 'an internal wake never interrupts'
    if agent_built:
        assert key not in adapter._pending_messages, 'never queued'
        content = _boundary(agent)
        finish.set()
        await async_wait_until(lambda: key not in adapter._active_sessions, timeout=10, what='the turn to end')
        assert executed == ['A'], 'never its own turn'
        assert AUTOMATION_MARKER in content and wake in content and USER_MARKER not in content
        return
    assert key in adapter._pending_messages and adapter._pending_messages[key].text == wake, 'queued behind'
    assert agent._pending_steer is None
    finish.set()
    await async_wait_until(lambda: wake in executed, timeout=10, what='the wake to run as its own turn')
    assert executed == ['A', wake]
