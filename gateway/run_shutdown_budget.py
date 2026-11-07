"""One deadline for a stop: every bounded wait after the settle draws on the shutdown watchdog's leash.

``_stop_impl`` records ``runner._stop_deadline`` (arm time + the effective leash) when it arms the
watchdog. Each later wait is clamped to what is left after the reserve of the phases that still
follow it, in stop order: agent finalize/cleanup → adapter teardown → final tool kill → executor
quiesce → SessionDB close → exit-state writes. A fixed reserve for "the rest of the tail" did not
hold once a later phase was itself slow (a 4s adapter disconnect still overran a 7s leash), so the
reserves are derived from the bounded steps that follow, and those steps are clamped in turn.
Outside a stop (no deadline) every wait keeps its own timeout unchanged.

Work a stop may stop waiting for must not escape the watchdog either: ``asyncio.run`` joins the loop's
default executor after ``stop()`` has disarmed the watchdog, so such work runs on its own daemon thread
(``start_daemon_call``) and, while it lives, counts as a writer that keeps SessionDB open.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import suppress
from typing import Any, Callable, Optional

# SessionDB close and the exit-state writes, after the executor quiesce (the quiesce kept 1s for them).
STOP_CLOSE_RESERVE_SECONDS = 1.0
# How long the stop waits, once the final sweep's future has settled, for its daemon thread to exit: the
# sweep has returned by then and the thread only has its settlement post (``call_soon_threadsafe``) left,
# but the loop can resume on the settled future before the thread is gone (QUIESCE-COMPOSITION-I1).
STOP_SETTLED_SWEEP_EXIT_SECONDS = 0.5
# The share of its 2s bound the executor quiesce is guaranteed after the adapters and the final tool
# kill: it waits for a live writer as long as the deadline allows, and a writer still live when it
# gives up keeps SessionDB open (#101093), so a shorter quiesce is safe where a cut close is not.
STOP_QUIESCE_FLOOR_SECONDS = 0.5


def stop_wait_budget(runner, timeout: float, *, reserve: float) -> float:
    """``timeout``, clamped during a stop to the leash left after ``reserve`` (never negative)."""
    deadline = getattr(runner, "_stop_deadline", None)  # None outside a stop and on bare test doubles
    if deadline is None:
        return timeout
    return max(0.0, min(timeout, deadline - time.monotonic() - reserve))


def stop_wait_timeout(runner, *, reserve: float) -> Optional[float]:
    """The leash left after ``reserve`` during a stop, for a wait with no bound of its own; None
    (unbounded, as before) outside a stop."""
    if getattr(runner, "_stop_deadline", None) is None:
        return None
    return stop_wait_budget(runner, float("inf"), reserve=reserve)


def start_daemon_call(fn: Callable[..., Any], *args: Any, name: str) -> tuple[asyncio.Future, threading.Thread]:
    """``asyncio.to_thread`` on a dedicated daemon thread instead of the loop's default executor: a wait
    the stop gives up on cannot then hold ``asyncio.run``'s executor join (past the disarmed watchdog),
    and the process hard-exit ends it. Returns the awaitable result and the thread (for liveness)."""
    from agent.memory_provider import spawn_context_thread  # the caller's profile scope, like to_thread
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _settle(ok: bool, value: Any) -> None:
        if not future.done():
            future.set_result(value) if ok else future.set_exception(value)

    def _run() -> None:
        try:
            ok, value = True, fn(*args)
        except BaseException as exc:  # health: allow BLE001 -- re-raised in the awaiting coroutine, like to_thread
            ok, value = False, exc
        with suppress(RuntimeError):  # loop already closed: nobody awaits the result any more
            loop.call_soon_threadsafe(_settle, ok, value)

    thread = spawn_context_thread(_run, name=name)
    thread.start()
    return future, thread


def final_sweep_live(runner) -> int:
    """1 while the stop's final tool sweep is still running after its wait, else 0. The sweep does
    blocking durable writes (registry checkpoints, cron job status, delegation interrupts), so the
    quiesce conservatively treats it as a live writer and keeps SessionDB open (#101093)."""
    thread = getattr(runner, "_stop_final_sweep_thread", None)
    return int(thread is not None and thread.is_alive())


def join_settled_final_sweep(runner, thread: threading.Thread, elapsed: Callable[[], float]) -> bool:
    """Wait for the final sweep's thread, whose future has settled, to exit within its exit bound (clamped
    to the stop's leash after the quiesce and the close). Until it has exited ``final_sweep_live`` counts
    it as a live writer and the close is skipped, though the sweep finished in time. Returns whether it
    exited; one still alive past the bound is logged, stays a live writer and SessionDB stays open for it."""
    thread.join(stop_wait_budget(runner, STOP_SETTLED_SWEEP_EXIT_SECONDS, reserve=quiesce_reserve()))
    if not thread.is_alive():
        return True
    logging.getLogger("gateway.run").warning(
        "Shutdown phase: final-cleanup tool kill done at +%.2fs but its thread has not exited; the quiesce "
        "keeps SessionDB open for it", elapsed())
    return False


def quiesce_reserve() -> float:
    """What the adapter teardown and the final tool kill leave: the quiesce floor + the close."""
    return STOP_QUIESCE_FLOOR_SECONDS + STOP_CLOSE_RESERVE_SECONDS


def agent_wait_reserve(runner) -> float:
    """What the agent finalize/cleanup waits leave: one adapter's full teardown bound (each further
    adapter wait is clamped to ``quiesce_reserve`` itself), then the quiesce and the close."""
    from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
    timeout_fn = getattr(runner, "_adapter_disconnect_timeout_secs", None)  # absent on bare test doubles
    adapter = timeout_fn() if callable(timeout_fn) else _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
    return adapter + quiesce_reserve()
