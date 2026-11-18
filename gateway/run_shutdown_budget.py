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
import functools
import logging
import os
import threading
import time
from contextlib import suppress
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

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


def quiesce_writers(runner, budget: float) -> tuple[int, int]:
    """The executor quiesce: join the gateway pools, then, in what is left of the same ``budget``, the
    async-delegation workers the final sweep interrupted (an in-process child shares the registry
    SessionDB and still flushes its failed turn and releases its lease after the interrupt), the
    detached child conversations and deferred closes those pools and workers left behind
    (``tools.detached_writers``) and any state.db maintenance pass on the housekeeping thread
    (``counted_maintenance_pass``), and the auto-title upgrades retired turns handed off
    (``title_upgrades_live``). Returns (live pool workers, live delegation workers + detached writers +
    title upgrades + maintenance passes); either keeps SessionDB open (#101093). The detached read here is not
    the close gate's last one: the starters the quiesce cannot join are sampled after it, so the gate
    re-reads the descendants once they are all at zero (``join_descendants_after_starters``)."""
    from gateway.run import GatewayRunner
    from tools import detached_writers
    from tools.async_delegation import join_workers
    deadline = time.monotonic() + budget
    pool_live = GatewayRunner._shutdown_executor(runner, drain_timeout=budget)
    async_live = join_workers(max(0.0, deadline - time.monotonic()))
    async_live += detached_writers.join(max(0.0, deadline - time.monotonic()))  # after the starters joined above
    async_live += title_upgrades_live(max(0.0, deadline - time.monotonic()))  # a retired turn's title writer
    return pool_live, async_live + fence_and_join_maintenance(max(0.0, deadline - time.monotonic()))


def join_descendants_after_starters(timeout: float) -> int:
    """The close gate's final descendant read (A4-E5-04). ``quiesce_writers`` reads the detached registry
    before the close gate samples the starters it cannot join (API-server workers, cron jobs, deferred and
    managed workers, the final sweep): one of them can hand a writer over (an API worker's timed-out nested
    conversation and its deferred close) after that read and retire before its own sample, so a close
    decided from that read closes under the writer. A starter registers what it hands over before it
    retires, so once every starter has been sampled at zero a read taken now sees every handed-over writer.
    Waits up to ``timeout`` for the async-delegation workers, then the detached writers (a worker, too,
    registers before it retires); returns how many are still live, which keeps SessionDB open (#101093)."""
    from tools import detached_writers
    from tools.async_delegation import join_workers
    deadline = time.monotonic() + max(0.0, timeout)
    live = join_workers(max(0.0, deadline - time.monotonic()))
    live += detached_writers.join(max(0.0, deadline - time.monotonic()))
    return live + title_upgrades_live(max(0.0, deadline - time.monotonic()))  # a turn registers it before it retires


def title_upgrades_live(timeout: float) -> int:
    """The auto-title upgrade threads still running after a wait of up to ``timeout`` (A3-SH-TITLE-01). A
    completed turn's model-title upgrade runs on its own daemon thread with the turn's registry SessionDB
    and writes the title when its model call returns, so while one lives it is a writer that keeps
    SessionDB open and the profile retained (#101093). Read through ``sys.modules``: a process that never
    imported the title generator started none, and the stop never imports it."""
    import sys
    join = getattr(sys.modules.get("agent.title_generator"), "join_title_upgrades", None)
    return join(max(0.0, timeout)) if callable(join) else 0


def title_upgrades_live_under(home, timeout: float, *, fence: bool = False, generation: Optional[int] = None) -> int:
    """Per-home form of ``title_upgrades_live`` (A3-SH-TITLE-02). It returns the auto-title upgrades that write a
    store under *home* and are still running after a wait of up to ``timeout``. Profile retirement reads it before
    the home's stores close and its reservation is released, so another profile's title upgrade never holds this one.
    With ``fence``, a zero read also refuses later upgrades for the home (``join_title_upgrades_under``), but only
    while the home's serve generation is still the one the read is bound to (TITLE-CANCEL-01): ``generation`` when
    the caller bound it before a thread handoff (``title_fence_generation``), else the generation at this call.
    It fails closed: a read that raises counts as one live writer, which keeps the profile retained."""
    import sys
    bound = title_fence_generation(home) if generation is None else generation
    join = getattr(sys.modules.get("agent.title_generator"), "join_title_upgrades_under", None)
    if not callable(join):
        return 0  # a process that never imported the title generator started no upgrade
    try:
        if not fence:
            return join(home, max(0.0, timeout))
        return join(home, max(0.0, timeout), fence=True,
                    still_current=lambda: title_fence_generation(home) == bound)
    except Exception:  # health: allow BLE001 -- fail closed: an unread writer keeps the profile
        logger.error("Title-writer read for %s failed; treated as live", home, exc_info=True)
        return 1


# Serve generation of each home's title fence (TITLE-CANCEL-01), kept here so that it exists whether or not the
# title generator was ever imported (a stop never imports it). ``unfence_title_home`` (every serve) advances it. A
# fencing read publishes its fence only while the generation is still the one it was bound to, so a retirement whose
# read outlives a re-serve (its coroutine was cancelled; its ``to_thread`` read was not) cannot fence the successor,
# however late that read starts, resumes or first finds the generator imported. Retirement binds the generation on
# the event loop BEFORE the thread handoff; a synchronous read binds it at the call.
_title_fence_lock = threading.Lock()
_title_fence_generation: dict = {}


def _title_home_key(home) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(home)))


def title_fence_generation(home) -> int:
    """The current title-fence serve generation of *home* (0 before any serve advanced it)."""
    with _title_fence_lock:
        return _title_fence_generation.get(_title_home_key(home), 0)


def unfence_title_home(home) -> None:
    """The home is served again: begin a new title-fence generation (an obsolete fencing read of an earlier one can no
    longer publish) and lift its title-upgrade fence (``title_upgrades_live_under``). The fence itself is a no-op where
    the generator was never imported (nothing fenced it)."""
    import sys
    key = _title_home_key(home)
    with _title_fence_lock:
        _title_fence_generation[key] = _title_fence_generation.get(key, 0) + 1
    unfence = getattr(sys.modules.get("agent.title_generator"), "unfence_title_home", None)
    if callable(unfence):
        unfence(home)


# state.db maintenance (prune/archive/VACUUM) runs inline on the gateway-housekeeping thread, which the
# bootstrap stops only after ``runner.stop()`` returns: each pass counts as a writer for the close, and
# once the close decision has fenced them no new pass starts for the rest of the process.
_maintenance_lock = threading.Lock()
_maintenance_fenced = False
_maintenance_passes: set[threading.Event] = set()


def counted_maintenance_pass(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``fn`` as one state.db maintenance pass the shutdown close waits for; skipped once fenced."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        done = threading.Event()
        with _maintenance_lock:  # the fence and the registration are one step (no admission window)
            if _maintenance_fenced:
                logger.info("state.db maintenance skipped: the gateway is shutting down")
                return None
            _maintenance_passes.add(done)
        try:
            return fn(*args, **kwargs)
        finally:
            with _maintenance_lock:
                _maintenance_passes.discard(done)
            done.set()
    return run


def fence_and_join_maintenance(timeout: float) -> int:
    """Refuse new maintenance passes, then wait up to ``timeout`` for those in flight; returns how
    many still run."""
    global _maintenance_fenced
    with _maintenance_lock:
        _maintenance_fenced = True
        pending = list(_maintenance_passes)
    deadline = time.monotonic() + max(0.0, timeout)
    for done in pending:
        done.wait(max(0.0, deadline - time.monotonic()))
    return sum(1 for done in pending if not done.is_set())


def _reset_maintenance_for_tests() -> None:
    global _maintenance_fenced
    with _maintenance_lock:
        _maintenance_fenced = False
        _maintenance_passes.clear()


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


def gate_only_live(runner) -> int:
    """Live writers counted only at the stop's SessionDB close gate, never by the drain waits: the
    final tool sweep, plus hosted-room threads that outlived or failed their Stop."""
    from gateway.run_runtime import hosted_room_writers
    return final_sweep_live(runner) + hosted_room_writers(runner)


def retain_open_stores(reason: str) -> None:
    """The close gate left the stores open for a live writer: keep every profile reservation too, so
    the bootstrap's release (``ProfileOwnership.close``) leaves the locks to process exit and no
    independent process can own a store this one can still write."""
    from gateway.runtime_ownership import process_ownership
    process_ownership.retain_until_exit(reason)


def close_step(failed: list) -> Callable[[str, Callable[[], Any]], Any]:
    """``_quiet_step`` for the stop's SessionDB close steps that also records, in *failed*, each step
    that raised: a close that raised did not close its store (SH-01 close-result accounting)."""
    import logging

    def step(label: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception:
            failed.append(label)
            logging.getLogger("gateway.run").warning("%s", label, exc_info=True)
            return None
    return step


def stores_closed(elapsed: float, failed=()) -> None:
    """The stop's close verdict, the only one that lets the bootstrap's release go ahead: no close step
    raised (*failed*) and no SessionDB close was left unfinished (``hermes_state.unsettled_closes``: a
    close that raised in a layer that swallows it, or a token writer still live past its join). Otherwise
    the stores stay retained, as for a live writer. A stop that raises, is cancelled or times out first
    keeps the retention ``stop()`` took before it started."""
    import logging
    from gateway.runtime_ownership import process_ownership
    from hermes_state import unsettled_closes
    unfinished = [*failed, *unsettled_closes()]
    if unfinished:
        retain_open_stores(f"SessionDB close did not finish: {', '.join(unfinished)}")
        logging.getLogger("gateway.run").warning(
            "Shutdown phase: SessionDB close did not finish at +%.2fs (%s): the stores stay owned until "
            "process exit", elapsed, ", ".join(unfinished))
        return
    process_ownership.stores_closed()
    logging.getLogger("gateway.run").info("Shutdown phase: SessionDB close done at +%.2fs", elapsed)
