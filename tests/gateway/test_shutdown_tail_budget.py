"""A stop finishes its whole teardown inside its own watchdog leash, and never skips an agent cleanup.

The waits after the settle (agent finalize/cleanup, adapter teardown, final tool kill, executor
quiesce, status flush) each carried a fixed bound that ignored how much of the shutdown watchdog's
leash was left. When a timed-out drain had already spent most of the leash, a slow agent cleanup or
a slow adapter disconnect let the watchdog os._exit the process before SessionDB handling and the
exit-state writes. A later fixed tail reserve still overran when an adapter was slow, and it skipped
a cleanup that could have finished at once. Production (2026-11-04/05): every SIGTERM stop whose drain
timed out on live work could end in "Shutdown watchdog fired after 60s". The 03:17 dump shows the loop
idle in select (not wedged) with an agent cleanup in flight on a housekeeping worker, a plausible
contributor; which tail await held the stop is not proven. These tests pin the composed contract instead.
"""

import asyncio
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.restart import LAUNCHD_WATCHDOG_DUMP_MARGIN_S
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner


def _slow_disconnect(seconds, done):
    """A disconnect that completes ``seconds`` after it starts (a half-dead transport's close)."""
    async def disconnect():
        finished = asyncio.Event()
        handle = asyncio.get_running_loop().call_later(seconds, finished.set)
        try:
            await finished.wait()
            done.append(seconds)
        finally:
            handle.cancel()
    return disconnect


async def _stop(leash, cleanup_seconds, adapters):
    """Signal-driven stop under a ``leash``-second supervisor budget, one idle cached agent whose
    cleanup takes ``cleanup_seconds``, and ``adapters`` (platform -> adapter). Returns the elapsed
    stop time, whether the cleanup ran to its end, and the runner."""
    runner, _ = make_restart_runner()
    runner.adapters = adapters
    runner._restart_drain_timeout = 0.0
    runner._stop_requested_by_signal = True
    runner._launchd_exit_timeout_s = leash + LAUNCHD_WATCHDOG_DUMP_MARGIN_S
    release, finished = threading.Event(), threading.Event()

    def close():  # e.g. on_session_end ingest or malloc_trim on a large heap
        release.wait(timeout=cleanup_seconds)
        finished.set()

    agent = SimpleNamespace(close=close, shutdown_memory_provider=lambda *a, **k: None, _session_messages=None)
    runner._agent_cache = OrderedDict({"agent:main:local:dm:idle": (agent, "sig")})
    runner._agent_cache_lock = threading.Lock()
    started = time.monotonic()
    try:
        with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            await runner.stop()
        elapsed = time.monotonic() - started
        if cleanup_seconds == 0:
            finished.wait(timeout=2)  # a dispatched, not awaited, cleanup still runs to its end
    finally:
        release.set()
    return elapsed, finished.is_set(), runner


def _final_state(runner):
    args, kwargs = runner._update_runtime_status.call_args
    return kwargs.get("gateway_state", args[0] if args else None)


@pytest.mark.asyncio
async def test_stop_completes_teardown_within_leash_when_agent_cleanup_outlasts_it():
    adapter = RestartTestAdapter()
    adapter.disconnect = AsyncMock()
    elapsed, _, runner = await _stop(7.0, 10.0, {Platform.TELEGRAM: adapter})
    assert elapsed < 7.0, f"stop took {elapsed:.1f}s, past its 7s watchdog leash"
    adapter.disconnect.assert_awaited_once()  # the teardown after the cleanup still ran
    assert runner._shutdown_event.is_set() and _final_state(runner) == "stopped"


@pytest.mark.asyncio
async def test_slow_cleanup_and_slow_adapter_disconnect_both_fit_the_leash():
    """BUDGET-01: the tail reserve must cover a slow disconnect plus the quiesce and the close."""
    adapter, done = RestartTestAdapter(), []
    adapter.disconnect = _slow_disconnect(4.0, done)
    elapsed, _, runner = await _stop(7.0, 10.0, {Platform.TELEGRAM: adapter})
    assert elapsed < 7.0, f"stop took {elapsed:.2f}s, past its 7s watchdog leash"
    assert done == [4.0], "the 4s adapter disconnect was cut short although the leash had room for it"
    assert _final_state(runner) == "stopped"


@pytest.mark.asyncio
async def test_many_slow_adapters_still_leave_the_close_and_exit_state_their_leash():
    adapters, done = {}, []
    for platform in (Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK):
        adapters[platform] = RestartTestAdapter()
        adapters[platform].disconnect = _slow_disconnect(2.5, done)
    elapsed, _, runner = await _stop(7.0, 10.0, adapters)
    assert elapsed < 7.0, f"stop took {elapsed:.2f}s, past its 7s watchdog leash"
    assert done, "no adapter disconnect completed"
    assert runner._shutdown_event.is_set() and _final_state(runner) == "stopped"


@pytest.mark.asyncio
async def test_an_immediate_idle_cleanup_runs_even_when_the_leash_is_nearly_spent():
    """ORD-1: a cleanup that can still finish is dispatched, never skipped for want of reserve."""
    adapter = RestartTestAdapter()
    adapter.disconnect = AsyncMock()
    elapsed, finished, _ = await _stop(4.0, 0.0, {Platform.TELEGRAM: adapter})
    assert finished, "the idle agent's cleanup was skipped with time still on the leash"
    assert elapsed < 4.0
