"""D5: a profile stop must be able to interrupt an adopted managed worker through the explicit-stop ABI.

``unserve_profile_runtime`` → ``_retire_profile_authority`` → ``stop_authority_work`` →
``stop_authority_turns(in_process=True)`` hard-interrupts the agent each executing session ADOPTED
when the session is not in ``_managed_workers``. A managed turn's ``finally`` drops its worker from
``_managed_workers`` and closes it while ``adopted`` still names it until settlement, so that agent
can be a ``ManagedWorker``. ``request_hard_interrupt`` calls ``interrupt(message, **kwargs)``; the old
``interrupt(self)`` raised ``TypeError`` out of the profile stop (gateway.log 2026-11-06 ~11:26 PST).
"""
import asyncio
from types import SimpleNamespace

import pytest

from agent.interrupt_compat import request_hard_interrupt
from gateway import run_runtime
from gateway.session_authorities import SessionAuthorities
from gateway.session_managed_worker import ManagedWorker


def _worker(state):
    worker = ManagedWorker(SimpleNamespace(pid=None))  # interrupt() only queues a control frame
    if state == 'closed':
        worker.closed.set()
    elif state == 'backpressure':
        while not worker.commands.full():
            worker.commands.put_nowait({'type': 'filler'})
    return worker


def _queued(worker):
    return [worker.commands.get_nowait() for _ in range(worker.commands.qsize())]


@pytest.mark.parametrize('state', ['open', 'closed', 'backpressure'])
@pytest.mark.asyncio
async def test_request_hard_interrupt_on_managed_worker_does_not_raise(tmp_path, monkeypatch, state):
    """The real unserve path, with the adopted agent a real ManagedWorker already dropped from
    ``_managed_workers``: the profile stop completes, and an open worker still gets its Stop."""
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)  # the turn task itself has ended; its settlement has not run yet
    launch, beta = tmp_path, tmp_path / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    registry = SessionAuthorities(launch)
    registry.add(launch, SimpleNamespace(profile_id=str(launch)))
    worker = _worker(state)
    live = SimpleNamespace(task=done, route='beta-route', event_stream=SimpleNamespace(execution={'execution_generation': 2}))
    authority = SimpleNamespace(sessions={'s': live}, profile_id=str(beta), pending_stops={}, adopted={'s': (2, worker)},
                                waiters={}, native_waiters=set(), agent=lambda ref: None, _turn_workers={},
                                _managed_workers={})
    registry.add(beta, authority, name='beta')
    runner = SimpleNamespace(session_authorities=registry, session_runtime_descriptor={},
                             session_ticket_store=SimpleNamespace(profile_ids=registry.profile_ids()))
    authority.runner = runner
    unbound = []
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda a: unbound.append(a))

    assert await run_runtime.unserve_profile_runtime(runner, beta) is True
    assert registry.for_home(beta) is None and unbound == [authority]
    frames = _queued(worker)
    if state == 'open':
        assert {'type': 'stop'} in frames and worker.stop.is_set()
    elif state == 'closed':
        assert frames == [] and not worker.stop.is_set()
    else:
        assert {'type': 'stop'} not in frames and worker.stop.is_set()  # latched; the read loop escalates


def test_compat_interrupt_keeps_the_stop_control_and_its_errors():
    """Direct ABI: message and tool_reason are accepted, the frame is the same Stop as before, the
    acknowledgment window opens once, and an unexpected control error still propagates."""
    from hermes_state_runtime import RuntimeStoreError

    worker = _worker('open')
    assert request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown') is True
    first = worker.stopped_at
    worker.interrupt()
    assert _queued(worker) == [{'type': 'stop'}, {'type': 'stop'}] and worker.stopped_at == first

    worker.control = lambda frame: (_ for _ in ()).throw(RuntimeStoreError('unexpected_control_failure'))
    with pytest.raises(RuntimeStoreError) as raised:
        worker.interrupt('Profile stopping')
    assert raised.value.reason == 'unexpected_control_failure'
