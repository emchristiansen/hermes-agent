"""The remaining message paths into a busy session steer (Eric, 2026-11-04).

(1) A messaging user's plain text admitted before the running turn adopted its agent steers into it at
adoption: the busy path marks it in memory (``_hm_steer_at_adoption``) before the row is written, since
the row cannot tell it from an explicit ``/queue`` or an A2A task, which keep their own turns (reviewer
GW-F01/GW-F02). Without a live mark (a drain, a restart) it keeps its own next turn, as before. A
post-admission hook that awaits while the turn adopts its agent leaves the text marked too, and the busy
pass steers it at once (reviewer ADM-02); a hook that consumes it still consumes it.

(2) A wake with NO session authority (legacy path): ``gateway.wake.deliver_wake`` falls back to the
adapter's ``handle_message``. When the runner's session state holds a running agent with the steer
seam, the busy handler that receives it steers it framed as Hermes automation, never interrupting and
never queueing it: the adapter's busy path when the adapter holds the running turn's guard (the normal
legacy turn, ``_handle_active_session_busy_message``), else the runner's
(``_hm_handle_running_session_message``). With no built agent, another event already waiting, or the
gateway draining (reviewer GW-F03), it queues behind the turn as before.

Real GatewayRunner, DiscordAdapter, (in (1)) SessionAuthority, admission ledger and drain over SQLite;
the turn body is a stand-in; the boundary is the canonical ``apply_pending_steer_to_tool_results``.
"""
import asyncio

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


async def _pre_adoption(monkeypatch, schedule):
    """Turn A runs and is still building its agent when the messaging user's plain text B arrives; A then
    adopts its agent, the adoption pass settles, and A crosses one tool/result boundary. ``schedule``:
    ``steer`` (the ordinary case), ``race`` (the adoption pass reads B's committed row before B's
    admission records its in-memory mark), ``drain_before_adoption`` and ``marks_lost`` (this owner's
    in-memory marks gone, as after a restart; a draining owner refuses B's admission itself). Returns what ran as turns, what
    A's boundary took, and B's row."""
    import threading
    from gateway.platforms.event import MessageEvent
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session_busy_ingress import _pass
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    running, adopt_now, adopted, boundary = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    executed, taken = [], []
    human = 'also check the staging logs'

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            running.set()
            await asyncio.wait_for(adopt_now.wait(), 30)  # the turn is still building its agent
            owner = runner.session_store.peek_session_id(key)
            authority.adopt_agent(owner, authority.executing[owner], agent)
            adopted.set()
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    loop, committed, release = asyncio.get_running_loop(), asyncio.Event(), threading.Event()
    if schedule == 'race':
        write = authority._admit_native_write

        def held_write(**kwargs):  # B's row commits, then its loop-side ``then`` waits for the test
            row = write(**kwargs)
            if kwargs['payload'].get('text') == human:
                loop.call_soon_threadsafe(committed.set)
                release.wait(30)
            return row
        monkeypatch.setattr(authority, '_admit_native_write', held_write)

    def row_b():
        return next((r for r in _rows(authority, runner, key) if r['payload'].get('text') == human), None)

    async def adoption_pass():
        await asyncio.wait_for(adopted.wait(), 10)
        sid = runner.session_store.peek_session_id(key)
        live = authority.sessions[sid]
        await _pass(authority, SessionRef(authority.profile_id, sid), live)
        if live.steering:
            await live.steering
        return live

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    assert runner._peek_session_state(key).turn.agent is _AGENT_PENDING_SENTINEL, 'no agent built yet'
    sending = asyncio.create_task(adapter.handle_message(MessageEvent(text=human, source=source, message_id='B')))
    if schedule == 'race':
        await asyncio.wait_for(committed.wait(), 10)
        adopt_now.set()
        await adoption_pass()
        assert row_b()['status'] == 'queued' and agent._pending_steer is None, 'the pass read B before its mark'
        release.set()
        await sending
        await async_wait_until(lambda: (row_b()['status'], row_b()['outcome']) == ('terminal', 'steering'),
                               timeout=10, what="the admission's own rerun of the pass to take B")
    else:
        await sending
        await async_wait_until(lambda: row_b() is not None, timeout=10, what='B admitted')
        runner._draining = schedule == 'drain_before_adoption'
        if schedule == 'marks_lost':
            authority.sessions[runner.session_store.peek_session_id(key)].adoption_steers.clear()
        adopt_now.set()
        await adoption_pass()
    runner._draining = False
    boundary.set()
    await async_wait_until(lambda: row_b()['outcome'] in ('steered', 'completed') and not runner._is_session_running(key),
                           timeout=15, what='B to settle')
    return executed, taken, row_b()


@pytest.mark.asyncio
@pytest.mark.parametrize('schedule', ['steer', 'race'])
async def test_a_human_message_before_adoption_steers_into_the_turn_at_adoption(monkeypatch, schedule):
    executed, [content], row = await _pre_adoption(monkeypatch, schedule)

    assert executed == ['A'], 'the human text never ran as its own turn'
    assert (row['status'], row['outcome']) == ('terminal', 'steered')
    assert content is not None and 'also check the staging logs' in content
    assert USER_MARKER in content and '[DELIVERED MESSAGE' not in content, 'the user\'s own text, never delivered'
    assert content.index('Gateway message origin') < content.index('also check the staging logs'), 'origin kept'


@pytest.mark.asyncio
@pytest.mark.parametrize('schedule', ['drain_before_adoption', 'marks_lost'])
async def test_a_human_message_before_adoption_keeps_its_own_turn_without_a_live_mark(monkeypatch, schedule):
    """A drain, or this owner's lost in-memory mark (a restart), leaves B its own next turn, as before."""
    executed, taken, row = await _pre_adoption(monkeypatch, schedule)

    assert taken == [None], 'nothing reached the running turn'
    assert executed == ['A', 'also check the staging logs']
    assert (row['status'], row['outcome']) == ('terminal', 'completed')


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


@pytest.mark.asyncio
@pytest.mark.parametrize('before_adopt', [False, True])
async def test_an_explicit_native_queue_keeps_its_next_turn(monkeypatch, before_adopt):
    """Reviewer GW-F01 schedule: ``/queue`` sent while A runs, before or after A adopted its agent."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_busy_ingress import _pass
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent, running, finish, executed = BufferedAgent(), asyncio.Event(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            if not before_adopt:
                sid = runner.session_store.peek_session_id(key)
                authority.adopt_agent(sid, authority.executing[sid], agent)
            running.set()
            await asyncio.wait_for(finish.wait(), 20)
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 10)
    await adapter.handle_message(MessageEvent(text='/queue only-after-A', source=source, message_id='Q'))
    await async_wait_until(lambda: len(_rows(authority, runner, key)) == 2, timeout=10, what='the /queue admission')
    sid = runner.session_store.peek_session_id(key)
    if before_adopt:
        authority.adopt_agent(sid, authority.executing[sid], agent)
    live = authority.sessions[sid]
    await _pass(authority, SessionRef(authority.profile_id, sid), live)
    if live.steering:
        await live.steering
    content = _boundary(agent)
    finish.set()
    await async_wait_until(lambda: executed == ['A', 'only-after-A'], timeout=15, what='the /queue text as its own turn')
    assert content is None


@pytest.mark.asyncio
async def test_a_native_a2a_task_keeps_its_own_execution(monkeypatch):
    """Reviewer GW-F02 schedule: a second A2A task in the same context, admitted before adoption."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_busy_ingress import _pass
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from plugins.platforms.a2a.adapter import A2AAdapter
    runner = GatewayRunner(GatewayConfig())
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner, adapter._loop = runner, asyncio.get_running_loop()
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='ctx-review', chat_type='dm', user_id='peer-review', user_name='peer-review')
    key = runner.session_store._generate_session_key(source)
    agent, running, finish, executed = BufferedAgent(), asyncio.Event(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if len(executed) == 1:
            running.set()
            await asyncio.wait_for(finish.wait(), 20)
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'job reply', 'messages': []}
        return 'job reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    def submit(text):
        return adapter._prepare_task({'message': {'role': 'user', 'contextId': 'ctx-review',
                                                  'parts': [{'kind': 'text', 'text': text}]}}, 'peer-review')
    await asyncio.to_thread(submit, 'job A')
    await asyncio.wait_for(running.wait(), 10)
    await asyncio.to_thread(submit, 'job B')
    await async_wait_until(lambda: len(_rows(authority, runner, key)) == 2, timeout=10, what='task B admission')
    sid = runner.session_store.peek_session_id(key)
    authority.adopt_agent(sid, authority.executing[sid], agent)
    live = authority.sessions[sid]
    await _pass(authority, SessionRef(authority.profile_id, sid), live)
    if live.steering:
        await live.steering
    content = _boundary(agent)
    finish.set()
    await async_wait_until(lambda: len(executed) == 2, timeout=15, what='task B to run as its own job')
    assert content is None and 'job B' in executed[1]


@pytest.mark.asyncio
@pytest.mark.parametrize('guarded', [False, True])
async def test_a_legacy_wake_during_a_drain_is_never_steered(monkeypatch, guarded):
    """Reviewer GW-F03 schedule: the gateway is already draining when the wake arrives."""
    from gateway.platforms.event import MessageEvent
    from gateway.wake import deliver_wake
    runner, adapter, _, source, key = await _gateway(monkeypatch, authority=False)
    agent, running, finish, handled = BufferedAgent(), asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        if event.text == 'A':
            runner._session_state(quick_key).turn.agent = agent
            running.set()
            await asyncio.wait_for(finish.wait(), 20)
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    if guarded:
        await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
        await asyncio.wait_for(running.wait(), 10)
    else:
        runner._session_state(key).turn.agent = agent
    runner._draining = True
    original = runner._hm_handle_running_session_message

    async def observe(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        finally:
            handled.set()
    monkeypatch.setattr(runner, '_hm_handle_running_session_message', observe)
    await deliver_wake(adapter, text='wake-during-drain', source=source)
    if not guarded:
        await asyncio.wait_for(handled.wait(), 10)
    content = _boundary(agent)
    finish.set()
    if not guarded:
        runner._session_state(key).turn.agent = None
    assert content is None


@pytest.mark.asyncio
async def test_a_mark_recorded_while_an_older_queue_read_is_in_flight_survives_it(monkeypatch):
    """The adoption pass reads the queue before B is admitted; B commits and records its mark while that
    read is still in flight (the authority adopted the agent; the runner still shows the pending sentinel,
    as in the window before ``_run_agent_track_agent`` polls). The older read must not erase B's mark."""
    import threading
    from gateway import session_busy_ingress
    from gateway.platforms.event import MessageEvent
    from gateway.session_results import execution_result

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    running, boundary = asyncio.Event(), asyncio.Event()
    executed, taken = [], []
    human = 'also check the staging logs'

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            running.set()
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    loop, read_taken, release = asyncio.get_running_loop(), asyncio.Event(), threading.Event()
    real_list, held = session_busy_ingress.list_session_admissions, []

    def listing(*args, **kwargs):  # the pass's first read after adoption: its snapshot, then held in flight
        rows = real_list(*args, **kwargs)
        if not held:
            held.append(rows)
            loop.call_soon_threadsafe(read_taken.set)
            release.wait(30)
        return rows

    def row_b():
        return next((r for r in _rows(authority, runner, key) if r['payload'].get('text') == human), None)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    monkeypatch.setattr(session_busy_ingress, 'list_session_admissions', listing)
    sid = runner.session_store.peek_session_id(key)
    authority.adopt_agent(sid, authority.executing[sid], agent)  # the runner still shows the sentinel
    await asyncio.wait_for(read_taken.wait(), 10)
    await adapter.handle_message(MessageEvent(text=human, source=source, message_id='B'))
    live = authority.sessions[sid]
    assert row_b()['status'] == 'queued' and row_b()['admission_id'] in live.adoption_steers, 'B marked'
    assert all(r['payload'].get('text') != human for r in held[0]), 'the in-flight read predates B'
    release.set()
    await async_wait_until(lambda: (row_b()['status'], row_b()['outcome']) == ('terminal', 'steering'),
                           timeout=10, what='the rerun to take B')
    boundary.set()
    await async_wait_until(lambda: row_b()['outcome'] in ('steered', 'completed') and not runner._is_session_running(key),
                           timeout=15, what='B to settle')
    assert executed == ['A'] and row_b()['outcome'] == 'steered'
    [content] = taken
    assert human in content and USER_MARKER in content


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_result', [None, {'action': 'handled'}], ids=['passes', 'consumes'])
async def test_a_human_message_whose_admission_hook_spans_adoption_steers_into_the_turn(monkeypatch, hook_result):
    """Reviewer ADM-02 schedule: B reaches the busy path while A has no agent (so ``_hm_busy_preempt``
    declines), then awaits a registered ``post_gateway_admission`` hook while A adopts and promotes its
    agent. A passing hook leaves B to steer into A at its next boundary; a consuming hook consumes B."""
    from types import SimpleNamespace
    from gateway.platforms.event import MessageEvent
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session_busy_ingress import _pass
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

    runner, adapter, authority, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    running, boundary, in_hook, resume_hook = (asyncio.Event() for _ in range(4))
    executed, taken, calls, generations = [], [], [], []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            generations.append(run_generation)
            running.set()
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    async def hook(text, **kwargs):
        calls.append(text)
        if text == 'B':
            assert runner._peek_session_state(key).turn.agent is _AGENT_PENDING_SENTINEL, 'B saw no agent'
            in_hook.set()
            await asyncio.wait_for(resume_hook.wait(), 30)
            return hook_result
    manager = get_plugin_manager()
    manager._discovered = True
    PluginContext(PluginManifest(name='adm02', key='adm02', source='user'), manager).register_hook(
        'post_gateway_admission', hook)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    sending = asyncio.create_task(adapter.handle_message(MessageEvent(text='B', source=source, message_id='B')))
    await asyncio.wait_for(in_hook.wait(), 30)
    sid = runner.session_store.peek_session_id(key)
    authority.adopt_agent(sid, authority.executing[sid], agent)
    await runner._run_agent_track_agent(SimpleNamespace(
        source=source, session_key=key, run_generation=generations[0], agent_holder=[agent]))
    assert runner._peek_session_state(key).turn.agent is agent
    resume_hook.set()
    await asyncio.wait_for(sending, 30)
    live = authority.sessions[sid]
    await _pass(authority, SessionRef(authority.profile_id, sid), live)  # the adoption's own pass, settled
    if live.steering:
        await live.steering
    boundary.set()
    await async_wait_until(lambda: not runner._is_session_running(key)
                           and all(r['status'] == 'terminal' for r in _rows(authority, runner, key)),
                           timeout=15, what='every admitted row to settle')

    assert calls == ['A', 'B'], 'the hook ran once per message, never again at execution'
    assert executed == ['A'], 'B never ran as its own turn'
    rows = _rows(authority, runner, key)
    if hook_result is not None:
        assert taken == [None] and [r['payload'].get('text') for r in rows] == ['A'], 'the hook consumed B'
        return
    [content] = taken
    assert content is not None and '\n\nB\n' in content and USER_MARKER in content
    assert next(r for r in rows if r['payload'].get('text') == 'B')['outcome'] == 'steered'
