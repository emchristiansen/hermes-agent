"""A stop finishes its teardown inside its own watchdog leash even when an agent cleanup outlasts it.

The post-settle agent finalize/cleanup waits each carried a fixed bound (10s finalize, 30s cleanup,
per agent) that ignored how much of the shutdown watchdog's leash the drain and settle had already
spent. Under load a timed-out drain left ~10s, a cleanup (memory-provider ``on_session_end``,
``malloc_trim``) took longer, and the watchdog os._exit-ed mid-finalize: adapters, SessionDB close and
the exit-state writes never ran (production 2026-11-04/05: 8 of 10 SIGTERM stops whose drain timed
out ended in "Shutdown watchdog fired after 60s").
"""

import asyncio
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.restart import LAUNCHD_WATCHDOG_DUMP_MARGIN_S
from tests.gateway.restart_test_helpers import make_restart_runner

LEASH_S = 7.0


@pytest.mark.asyncio
async def test_stop_completes_teardown_within_leash_when_agent_cleanup_outlasts_it():
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    # A signal-driven stop under a supervisor exit budget: the leash is that budget minus the dump margin.
    runner._stop_requested_by_signal = True
    runner._launchd_exit_timeout_s = LEASH_S + LAUNCHD_WATCHDOG_DUMP_MARGIN_S
    adapter.disconnect = AsyncMock()

    release = threading.Event()
    close_started = threading.Event()

    def slow_close():  # e.g. on_session_end ingest or malloc_trim on a large heap
        close_started.set()
        release.wait(timeout=LEASH_S + 3)

    idle_agent = SimpleNamespace(close=slow_close, shutdown_memory_provider=lambda *a, **k: None,
                                 _session_messages=None)
    runner._agent_cache = OrderedDict({"agent:main:local:dm:idle": (idle_agent, "sig")})
    runner._agent_cache_lock = threading.Lock()

    started = time.monotonic()
    try:
        with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            await runner.stop()
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert close_started.is_set(), "the idle-cache cleanup never ran"
    assert elapsed < LEASH_S, f"stop took {elapsed:.1f}s, past its {LEASH_S:.0f}s watchdog leash"
    adapter.disconnect.assert_awaited_once()  # the teardown after the cleanup still ran
    assert runner._shutdown_event.is_set()
