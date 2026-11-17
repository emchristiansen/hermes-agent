"""An internal wake waiting in the adapter's pending slot steers into the running turn once it adopts its
agent (Eric, 2026-11-04: "I want literally every message to always be a steering message").

A wake that reaches a busy session goes to the runner's busy branch (``_hm_handle_running_session_message``):
it steers into the running agent at once, but while the turn is still building its agent it waits in the
adapter's pending slot, and every later wake waits behind it to keep arrival order, so all of them ran
as their own turns after the running one. Now the turn's agent adoption (``_run_agent_track_agent``)
hands the waiting wakes to it in arrival order, exactly as on arrival, and later wakes again steer on
arrival. The first waiting event the turn does not take keeps its place with everything behind it: a
plugin's payload, a command or user text. A draining gateway, or a retiring owner of the routed profile
(reviewer WAKE-01), steers nothing.

Real GatewayRunner, DiscordAdapter, SessionAuthority, admission ledger and drain over SQLite; the wakes
are the loop-tick fallback's own events (``_synthetic_prompt_event``) through ``adapter.handle_message``;
the turn body is a stand-in that adopts its agent through the real ``_run_agent_track_agent``; the
boundary is the canonical ``apply_pending_steer_to_tool_results``.
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_prompt_queued_transfer import BufferedAgent
from tests.gateway.test_remaining_message_paths_steer import AUTOMATION_MARKER, USER_MARKER, _boundary, _gateway
from tests.waiting import async_wait_until


def _waiting(adapter, runner, key):
    """The pending slot's events in drain order: the head, then the overflow."""
    head = adapter._pending_messages.get(key)
    state = runner._peek_session_state(key)
    return ([head.text] if head is not None else []) + [e.text for e in (state.conversation.queued_events if state else [])]


async def _late_adoption(monkeypatch, before, after=(), *, draining=False):
    """Turn A runs and is still building its agent while the ``before`` events arrive; A then adopts its
    agent, the ``after`` events arrive, and A crosses one tool/result boundary. Returns what ran as turns,
    what A's boundary took, and what still waits in the overflow once the slot's head ran: the stand-in
    turn replaces the real one's post-turn drain, which promotes the overflow's next event."""
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.platforms.event import MessageEvent
    from gateway.session_results import execution_result

    runner, adapter, _, source, key = await _gateway(monkeypatch)
    agent = BufferedAgent()
    interrupts = []
    monkeypatch.setattr(agent, 'interrupt', lambda *a, **k: interrupts.append(a), raising=False)
    running, adopt_now, adopted, boundary = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    executed, taken = [], []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            running.set()
            await asyncio.wait_for(adopt_now.wait(), 30)  # the turn is still building its agent
            await runner._run_agent_track_agent(SimpleNamespace(
                source=source, session_key=quick_key, run_generation=run_generation, agent_holder=[agent]))
            adopted.set()
            await asyncio.wait_for(boundary.wait(), 30)
            taken.append(_boundary(agent))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    def wake(item):
        return item(source) if callable(item) else runner._synthetic_prompt_event(source, item, internal=True)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 30)
    assert runner._peek_session_state(key).turn.agent is _AGENT_PENDING_SENTINEL, 'no agent built yet'
    for i, event in enumerate(before):
        await adapter.handle_message(wake(event))
        await async_wait_until(lambda: len(_waiting(adapter, runner, key)) == i + 1, timeout=10,
                               what='the event to wait in the pending slot')
    runner._draining = draining
    adopt_now.set()
    await asyncio.wait_for(adopted.wait(), 10)
    runner._draining = False
    for text in after:
        await adapter.handle_message(wake(text))
        await async_wait_until(lambda: text in (agent._pending_steer or '') or text in _waiting(adapter, runner, key),
                               timeout=10, what='the later wake to steer or wait')
    boundary.set()
    try:
        await async_wait_until(lambda: not runner._is_session_running(key) and key not in adapter._active_sessions
                               and adapter._pending_messages.get(key) is None, timeout=20,
                               what='the turns to end and the slot to drain')
    except AssertionError as exc:
        raise AssertionError(f'{exc}: ran {executed}, waiting {_waiting(adapter, runner, key)}, '
                             f'running {runner._is_session_running(key)}, guard {key in adapter._active_sessions}')
    assert interrupts == [], 'a wake never interrupts'
    return executed, taken, _waiting(adapter, runner, key)


@pytest.mark.asyncio
async def test_wakes_waiting_for_the_agent_steer_into_it_at_adoption_and_later_ones_on_arrival(monkeypatch):
    first, second, third = ('[loop tick] check the queue', '[Kanban task completed] t_1 ready',
                            '[Kanban task completed] t_2 ready')
    executed, [content], waiting = await _late_adoption(monkeypatch, [first, second], [third])

    assert executed == ['A'] and waiting == [], 'no wake ran or waits as its own turn'
    assert content is not None and AUTOMATION_MARKER in content and USER_MARKER not in content
    frame = content[content.index(AUTOMATION_MARKER):]
    assert frame.index(first) < frame.index(second) < frame.index(third), 'arrival order kept'


@pytest.mark.asyncio
async def test_a_plugin_payload_waiting_first_keeps_its_place_and_the_wake_behind_it(monkeypatch):
    """A plugin's untrusted payload (the standalone injection's event) is never taken from the slot; the
    wake that arrived after it stays behind it: the payload runs as its own turn after A, and the wake
    still waits behind it, as before."""
    from gateway.platforms.event import MessageEvent, MessageType
    payload, wake = 'plugin payload', '[Kanban task completed] t_3 ready'

    def plugin(source):
        return MessageEvent(text=payload, message_type=MessageType.TEXT, source=source, internal=True,
                            allow_gateway_control=False,
                            metadata={'hermes_plugin_id': 'fixture', 'hermes_plugin_injection': True})
    executed, taken, waiting = await _late_adoption(monkeypatch, [plugin, wake])

    assert taken == [None], 'nothing reached the running turn'
    assert executed == ['A', payload] and waiting == [wake], 'arrival order kept'


@pytest.mark.asyncio
async def test_a_draining_gateway_steers_no_waiting_wake(monkeypatch):
    wake = '[Kanban task completed] t_4 ready'
    executed, taken, waiting = await _late_adoption(monkeypatch, [wake], draining=True)

    assert taken == [None]
    assert executed == ['A', wake] and waiting == []


@pytest.mark.asyncio
@pytest.mark.parametrize('secondary', [False, True], ids=['launch', 'secondary'])
@pytest.mark.parametrize('retire_owner', [False, True], ids=['serving', 'retiring'])
async def test_a_retiring_owner_keeps_its_waiting_wakes_at_adoption(monkeypatch, secondary, retire_owner):
    """Reviewer WAKE-01 schedule: two wakes wait for a multiplexed profile's internal turn A; that profile's
    authority starts retiring (the launch one keeps serving); then A adopts its agent. A retiring owner
    keeps both wakes waiting in order; a serving one takes both, in order, as Hermes automation's."""
    import threading
    import weakref
    from gateway.run_runtime import _retire_profile_authority
    from gateway.session_authorities import SessionAuthorities, active_authority
    from gateway.session_authority import SessionAuthority
    from gateway.session_identity import RoutingIdentity
    from gateway.session_runtime_workers import track_turn_worker
    from gateway.wake import admit_internal_event
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    runner, adapter, _, source, _ = await _gateway(monkeypatch, authority=False)
    launch_home = get_hermes_home()
    home = launch_home / 'profiles' / 'secondary' if secondary else launch_home
    home.mkdir(parents=True, exist_ok=True)
    dbs = [SessionDB(db_path=home / 'owner.db')]
    owner = SessionAuthority(runner, profile_id=str(home), instance_id='owner', db=dbs[0],
                             epoch=begin_runtime_epoch(dbs[0], instance_id='owner'))
    if secondary:
        dbs.append(SessionDB(db_path=launch_home / 'launch.db'))
        runner.session_authority = SessionAuthority(runner, profile_id=str(launch_home), instance_id='launch',
                                                    db=dbs[1], epoch=begin_runtime_epoch(dbs[1], instance_id='launch'))
    else:
        runner.session_authority = owner
    registry = SessionAuthorities(launch_home)
    registry.add(launch_home, runner.session_authority)
    if secondary:
        registry.add(home, owner, name='secondary')
    runner.session_authorities, runner.config.multiplex_profiles = registry, True
    source.profile = 'secondary' if secondary else None
    source._identity = RoutingIdentity(transport_profile='default', runtime_profile=source.profile or 'default',
                                       authorization_home=launch_home, runtime_home=home,
                                       transport=weakref.ref(adapter), multiplexed=True)
    runner._wire_adapter_handlers(adapter)  # the routing the multiplex registry and config now select
    key = runner.session_store._generate_session_key(source)
    worker, holder, agent = SimpleNamespace(worker_done=threading.Event()), [None], BufferedAgent()
    running, adopt_now, observed, finish = (asyncio.Event() for _ in range(4))
    seen = []

    async def turn(event, routed_source, quick_key, run_generation):
        if event.text != 'A':
            return 'reply'
        assert active_authority(runner) is owner and track_turn_worker(runner, worker, holder) is owner
        running.set()
        try:
            await asyncio.wait_for(adopt_now.wait(), 20)
            holder[0] = agent  # the holder retirement tracks
            await runner._run_agent_track_agent(SimpleNamespace(
                source=routed_source, session_key=quick_key, run_generation=run_generation, agent_holder=holder))
            seen.append((_waiting(adapter, runner, key), _boundary(agent)))
            observed.set()
            await asyncio.wait_for(finish.wait(), 20)
        finally:
            worker.worker_done.set()
        return 'reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    retire = None
    try:
        await admit_internal_event(adapter, runner._synthetic_prompt_event(source, 'A', internal=True))
        await asyncio.wait_for(running.wait(), 10)
        for i, text in enumerate(('WAKE1', 'WAKE2')):
            await admit_internal_event(adapter, runner._synthetic_prompt_event(source, text, internal=True))
            await async_wait_until(lambda: len(_waiting(adapter, runner, key)) == i + 1, timeout=10,
                                   what='the wake to wait in the pending slot')
        if retire_owner:
            retire = asyncio.create_task(_retire_profile_authority(owner, timeout=20))
            await async_wait_until(lambda: owner.retiring, timeout=10, what='retirement to start')
            assert not runner.session_authority.retiring or not secondary, 'only the routed owner retires'
        adopt_now.set()
        await asyncio.wait_for(observed.wait(), 10)
        [(waiting, content)] = seen
        if retire_owner:
            assert waiting == ['WAKE1', 'WAKE2'] and content is None, 'a retiring owner takes no waiting wake'
        else:
            assert waiting == [] and content.index('WAKE1') < content.index('WAKE2')
            assert content.count(AUTOMATION_MARKER) == 2 and USER_MARKER not in content
    finally:
        finish.set()
        worker.worker_done.set()
        if retire is not None:
            await retire
        await async_wait_until(lambda: key not in adapter._active_sessions, timeout=10, what='the turn to end')
        for db in dbs:
            db.close()
