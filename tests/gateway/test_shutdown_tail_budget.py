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
Review t_f4ec2b56/run2438 added two holes: a finalize message's adapter send was awaited without any
bound (BUDGET-02), and a final tool sweep the stop stopped waiting for sat on the default executor that
``asyncio.run`` joins after the watchdog is disarmed (KILL-1). The process-exit tests run the stop and
``asyncio.run`` in a child interpreter with the real (non-pytest) watchdogs armed, up to its ``os._exit``.
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.restart import LAUNCHD_WATCHDOG_DUMP_MARGIN_S
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner, make_restart_source


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


def _leashed_runner(leash):
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    runner._stop_requested_by_signal = True
    runner._launchd_exit_timeout_s = leash + LAUNCHD_WATCHDOG_DUMP_MARGIN_S
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("send_delay", [0.0, 9.0])
async def test_finalize_message_delivery_is_bounded_by_the_leash(send_delay, caplog):
    """BUDGET-02: a finalize hook's message is sent while the leash allows, then recorded as lost."""
    runner, adapter = _leashed_runner(8.0)
    key = "agent:main:telegram:dm:123456"
    closed = threading.Event()
    agent = SimpleNamespace(session_id=None, close=closed.set, shutdown_memory_provider=lambda *a, **k: None,
                            _session_messages=None)
    runner._drain_active_agents = AsyncMock(return_value=({key: agent}, False))  # a drained, completed agent
    source = make_restart_source()
    runner._shutdown_notification_target = AsyncMock(return_value=(source, "telegram", source.chat_id, None, None))
    sent = []

    async def send(chat_id, content, **kwargs):
        arrived = asyncio.Event()
        handle = asyncio.get_running_loop().call_later(send_delay, arrived.set)
        try:
            await arrived.wait()
            sent.append(content)
            return SendResult(success=True, message_id="m1")
        finally:
            handle.cancel()

    adapter.send, adapter.disconnect = send, AsyncMock()
    started = time.monotonic()
    with patch("hermes_cli.lifecycle._plugin_hooks", return_value=["finalizer-result"]), \
            patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
        await runner.stop()
    elapsed = time.monotonic() - started
    assert elapsed < 8.0, f"finalize delivery kept the stop {elapsed:.2f}s, past its 8s leash"
    assert closed.is_set(), "the agent cleanup after the delivery did not run"
    adapter.disconnect.assert_awaited_once()
    assert _final_state(runner) == "stopped"
    lost = [r.getMessage() for r in caplog.records if "session-end plugin message(s)" in r.getMessage()]
    if send_delay == 0.0:
        assert sent == ["finalizer-result"] and not lost
    else:
        assert sent == [] and len(lost) == 1 and "'finalizer-result'" in lost[0] and "remaining" in lost[0]


@pytest.mark.asyncio
async def test_a_completion_flush_that_ignores_its_cancel_cannot_hold_the_stop(caplog):
    runner, adapter = _leashed_runner(8.0)
    adapter.disconnect = AsyncMock()
    release = asyncio.Event()

    async def stubborn_flush():
        try:
            await release.wait()
        except asyncio.CancelledError:
            handle = asyncio.get_running_loop().call_later(9.0, release.set)
            try:
                await release.wait()  # ignores the shutdown cancel for 9s
            finally:
                handle.cancel()

    flush = asyncio.ensure_future(stubborn_flush())
    await asyncio.sleep(0)  # health: allow HX013 -- one loop turn so the flush is inside its wait
    runner._completion_notification_batch_flush_tasks = {flush}
    started = time.monotonic()
    try:
        with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            await runner.stop()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        await flush
    assert elapsed < 8.0, f"a cancel-resistant completion flush kept the stop {elapsed:.2f}s"
    adapter.disconnect.assert_awaited_once()
    assert any("completion-batch flush task(s) still running" in r.getMessage() for r in caplog.records)


def _stop_under_asyncio_run(leash, sweep_seconds):
    """KILL-1 boundary: the real stop inside ``asyncio.run``, whose return (not stop()'s) is what lets
    the gateway reach its hard exit; the final tool sweep takes ``sweep_seconds``."""
    observed, release, swept = {}, threading.Event(), threading.Event()

    def sweep(phase):
        assert phase == "final-cleanup"
        release.wait(timeout=sweep_seconds)
        swept.set()
        return []

    async def main():
        runner, _ = _leashed_runner(leash)
        runner._session_db = db = MagicMock()
        await runner.stop()
        observed.update(stop=time.monotonic() - started, watchdog_done=runner._shutdown_watchdog_done.is_set(),
                        db_closed=db._db.close.called, swept_at_stop=swept.is_set())

    started = time.monotonic()
    try:
        with patch.object(GatewayShutdownMixin, "_stop_kill_tool_subprocesses", side_effect=sweep), \
                patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            asyncio.run(main())
        observed["asyncio_run"] = time.monotonic() - started
    finally:
        release.set()
    return observed


def test_a_final_sweep_past_its_wait_cannot_hold_the_exit_after_the_watchdog_is_disarmed():
    observed = _stop_under_asyncio_run(4.0, 6.0)
    assert observed["watchdog_done"] and not observed["swept_at_stop"], observed
    assert observed["asyncio_run"] < 4.0, f"asyncio.run returned past the 4s leash: {observed}"
    assert not observed["db_closed"], f"SessionDB was closed while the final sweep still ran: {observed}"


def test_a_final_sweep_that_finishes_in_time_is_awaited_before_the_close():
    observed = _stop_under_asyncio_run(4.0, 0.0)
    assert observed["swept_at_stop"] and observed["db_closed"], observed
    assert observed["asyncio_run"] < 4.0, observed


def _stop_with_final_sweep_unwind(leash, hold_unwind):
    """The real stop inside ``asyncio.run`` with an immediate final sweep whose daemon thread, after it has
    settled the awaited result, holds its own unwind until released (``hold_unwind``) or exits at once."""
    from agent import memory_provider
    from gateway.run_shutdown import _resolve_gateway_exit_verdict
    real_spawn, unwind, threads, observed = memory_provider.spawn_context_thread, threading.Event(), [], {}

    def spawn_unwinding(target, *, name, **kwargs):
        def run(*args, **kw):
            try:
                return target(*args, **kw)  # settles the awaited result (call_soon_threadsafe)
            finally:
                if name == "hermes-stop-final-cleanup" and hold_unwind:
                    unwind.wait(timeout=5.0)  # the thread's own unwind, after its result is in
        thread = real_spawn(run, name=name, **kwargs)
        if name == "hermes-stop-final-cleanup":
            threads.append(thread)
        return thread

    async def main():
        runner, _ = _leashed_runner(leash)
        runner._exit_with_failure = False  # GatewayRunner.__init__'s default; the bare test runner skips __init__
        runner._session_db = db = MagicMock()
        await runner.stop()
        observed.update(stop=time.monotonic() - started, db_closed=db._db.close.called,
                        sweep_thread_alive_at_return=[t.is_alive() for t in threads],
                        should_exit_with_failure=runner.should_exit_with_failure, exit_code=runner.exit_code,
                        verdict=(_resolve_gateway_exit_verdict(runner, signal_initiated_shutdown=False),
                                 _resolve_gateway_exit_verdict(runner, signal_initiated_shutdown=True)))

    started = time.monotonic()
    try:
        with patch.object(memory_provider, "spawn_context_thread", spawn_unwinding), \
                patch.object(GatewayShutdownMixin, "_stop_kill_tool_subprocesses", return_value=[]), \
                patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            asyncio.run(main())
        observed["asyncio_run"] = time.monotonic() - started
    finally:
        unwind.set()
        for thread in threads:
            thread.join(timeout=10.0)
    observed["sweep_thread_exited_after_release"] = [not t.is_alive() for t in threads]
    return observed


def test_an_awaited_final_sweep_whose_thread_outlives_its_exit_bound_keeps_the_db_open_and_reports_it(caplog):
    """QUIESCE-COMPOSITION-I1, bounded-join contract. The sweep's daemon thread settles its result, then still
    has to unwind; a thread that has not exited can still run its thread-local finalizers, so it stays a writer.
    The stop waits for that unwind only within STOP_SETTLED_SWEEP_EXIT_SECONDS (never past its leash), leaves
    SessionDB open while the thread lives, reports the skipped close as such (not as a clean "done"), and the
    exit verdict stays the one an unobstructed stop reaches. (This replaces A3's
    ``..._has_not_unwound_yet_does_not_hold_the_close``, whose close-anyway expectation contradicts the contract.)"""
    caplog.set_level("INFO", logger="gateway.run")
    held = _stop_with_final_sweep_unwind(4.0, hold_unwind=True)
    messages = [r.getMessage() for r in caplog.records]
    caplog.clear()
    clean = _stop_with_final_sweep_unwind(4.0, hold_unwind=False)

    # Bounded: the stop returned while the thread was still unwinding, inside the leash.
    assert held["sweep_thread_alive_at_return"] == [True], held
    assert held["asyncio_run"] < 4.0, held
    # The DB stays open while the thread lives.
    assert not held["db_closed"], f"SessionDB was closed under a live final-sweep thread: {held}"
    # Truthful close-skip report.
    assert any("final-cleanup tool kill done" in m and "its thread has not exited" in m for m in messages), messages
    assert any("deferred/authority/final-sweep worker(s) still running" in m for m in messages), messages
    assert not any(m.startswith("Shutdown phase: final-cleanup tool kill done") and "has not exited" not in m
                   for m in messages), messages
    # Independent exit verdict: the same as the unobstructed stop, which does close.
    assert clean["db_closed"] and clean["sweep_thread_alive_at_return"] in ([False], [True]), clean
    assert held["should_exit_with_failure"] is False and held["exit_code"] is None, held
    assert held["verdict"] == clean["verdict"] == (True, False), (held, clean)
    assert held["sweep_thread_exited_after_release"] == [True], held



def test_a_final_sweep_still_in_its_final_write_when_its_wait_runs_out_keeps_the_close_open():
    """The other side of the settled-sweep rule (QUIESCE-COMPOSITION-I1): ``start_daemon_call`` posts the
    settlement only after the sweep function has returned, so a sweep that already has its result but is
    still inside its last durable write (its ``finally``) when the stop's wait runs out has published
    nothing. It is still a writer: the stop keeps its thread as a live final sweep and SessionDB is not
    closed under that write."""
    observed, in_final, release, wrote = {}, threading.Event(), threading.Event(), threading.Event()

    def sweep(phase):
        assert phase == "final-cleanup"
        try:
            return []
        finally:
            in_final.set()
            release.wait(timeout=6.0)  # the sweep's final durable write, still running past the wait
            wrote.set()

    async def main():
        runner, _ = _leashed_runner(4.0)
        runner._session_db = db = MagicMock()
        await runner.stop()
        thread = runner._stop_final_sweep_thread
        observed.update(in_final=in_final.is_set(), wrote_at_stop=wrote.is_set(), db_closed=db._db.close.called,
                        sweep_tracked_live=thread is not None and thread.is_alive())

    try:
        with patch.object(GatewayShutdownMixin, "_stop_kill_tool_subprocesses", side_effect=sweep), \
                patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            asyncio.run(main())
    finally:
        release.set()
    assert observed["in_final"] and not observed["wrote_at_stop"], observed
    assert observed["sweep_tracked_live"], observed
    assert not observed["db_closed"], f"SessionDB was closed under the sweep's unfinished final write: {observed}"


REPO = Path(__file__).resolve().parents[2]

# Child: the real stop as asyncio.run's main coroutine, its shutdown watchdog armed (no PYTEST_CURRENT_TEST),
# a 6 s final sweep under a 4 s leash, then the entrypoint's hard exit.
_SWEEP_CHILD = """
import asyncio, os, threading, time
from unittest.mock import patch
from gateway.restart import LAUNCHD_WATCHDOG_DUMP_MARGIN_S
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner

def sweep(phase):
    threading.Event().wait(6.0)
    return []

async def lifecycle():
    runner, _ = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    runner._stop_requested_by_signal = True
    runner._launchd_exit_timeout_s = 4.0 + LAUNCHD_WATCHDOG_DUMP_MARGIN_S
    await runner.stop()
    print("STOPPED", round(time.monotonic() - started, 3), flush=True)
    return True

started = time.monotonic()
with patch.object(GatewayShutdownMixin, "_stop_kill_tool_subprocesses", side_effect=sweep), \\
        patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
    ok = asyncio.run(lifecycle())
print("EXITING", round(time.monotonic() - started, 3), flush=True)
os._exit(0 if ok else 1)
"""

# Child: work the stop gave up on outlives the gateway lifecycle (a default-executor straggler and a task
# that swallows every cancel), so asyncio.run's own teardown never ends; the exit backstop must end it.
_STRAGGLER_CHILD = """
import asyncio, os, sys, time
from gateway.shutdown_watchdog import run_with_exit_backstop

async def deaf():
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

async def lifecycle():
    asyncio.get_running_loop().run_in_executor(None, time.sleep, 60)
    asyncio.ensure_future(deaf())
    await asyncio.sleep(0)
    print("LIFECYCLE_DONE", time.time(), flush=True)
    if sys.argv[1] == "raise":
        raise SystemExit(75)
    return True

try:
    asyncio.run(run_with_exit_backstop(lifecycle, leash_s=1.0))
except SystemExit as exc:
    os._exit(exc.code)
os._exit(0)
"""


def _child(tmp_path, code, *args, timeout=60):
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO))
    proc = subprocess.run([sys.executable, "-c", code, *args], cwd=str(REPO), env=env, capture_output=True,
                          text=True, timeout=timeout, check=False)
    return proc, home, time.time()


def _marks(stdout):
    return {line.split()[0]: float(line.split()[1]) for line in stdout.splitlines()
            if line.split() and line.split()[0] in {"STOPPED", "EXITING", "LIFECYCLE_DONE"}}


def test_the_process_exit_follows_the_stop_while_a_final_sweep_still_runs(tmp_path):
    """KILL-1 at the process boundary: asyncio.run returns and the process exits on the stop's leash."""
    proc, home, _ = _child(tmp_path, _SWEEP_CHILD)
    marks = _marks(proc.stdout)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-4000:])
    assert marks["STOPPED"] < 4.0 and marks["EXITING"] < 4.0, marks
    assert not (home / "logs" / "gateway-shutdown-watchdog.log").exists(), "the stop's watchdog fired"


@pytest.mark.parametrize("ending, exit_code", [("return", 0), ("raise", 75)])
def test_asyncio_run_teardown_cannot_hold_the_exit_past_its_backstop(tmp_path, ending, exit_code):
    proc, home, exited_at = _child(tmp_path, _STRAGGLER_CHILD, ending)
    marks = _marks(proc.stdout)
    assert proc.returncode == exit_code, (proc.returncode, proc.stdout, proc.stderr[-4000:])
    assert exited_at - marks["LIFECYCLE_DONE"] < 10.0, "the exit waited on the stragglers"
    dump = (home / "logs" / "gateway-shutdown-watchdog.log").read_text(encoding="utf-8")
    assert "exit_teardown_watchdog_fired" in dump and '"phase": "asyncio.run teardown"' in dump
