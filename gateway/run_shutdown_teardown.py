"""Teardown of the runner's per-run state at stop: background tasks, pending-message flush, per-session state and the
final tool-subprocess sweep, for GatewayRunner.

Split out of ``gateway/run_shutdown.py``; bound onto ``GatewayRunner`` via ``GatewayShutdownMixin``.
``gateway.run`` / ``gateway.run_shutdown`` internals are imported lazily inside the method body (import cycle).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from agent.async_utils import consume_detached_task_result
from gateway.run_shutdown_budget import join_settled_final_sweep, quiesce_reserve, stop_wait_budget

if TYPE_CHECKING:
    from gateway.run_shutdown import GatewayShutdownMixin

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTeardownMixin:
    """The stop's release of runner-held runtime state (``_stop_release_runtime_state``)."""

    async def _stop_release_runtime_state(self, ctx: GatewayShutdownMixin._StopContext) -> None:
        """Cancel background tasks, flush pending messages, clear per-session state, final tool kill."""
        from gateway.run import GatewayRunner
        from gateway.run_shutdown import GatewayShutdownMixin
        for _task in list(self._background_tasks):
            # _restart_task awaits _stop_task: cancelling it would tunnel into _stop_impl and skip _shutdown_event.set().
            if _task is self._stop_task or _task is self._restart_task:
                continue
            _task.cancel()
        # The restart orchestration task is awaiting _stop_task right now; cancelling it would propagate
        # CancelledError into this _stop_impl and skip _shutdown_event.set() / _exit_code = 75 (#12875). It
        # self-terminates anyway.
        self._background_tasks.clear()
        ctx.api_live = self._active_api_run_count()
        self.adapters.clear()
        for _session_key in list(self._running_agents):
            self._release_running_agent_state(_session_key)
        # Flush pending messages before clearing: under FTS5 corruption they are the only surviving copy.
        with suppress(Exception):
            from gateway.shutdown_flush import flush_pending_to_file
            flush_pending_to_file(dict(self._pending_messages), reason="shutdown")
        # The overflow FIFO tail lives in SessionState.conversation.queued_events — flush it too.
        with suppress(Exception):
            from gateway.shutdown_flush import flush_overflow_to_file
            flush_overflow_to_file(
                {_k: list(_v) for _k, _v in dict(getattr(self, "_queued_events", None) or {}).items() if _v},
                reason="shutdown",
            )
        # Live SessionState views: clear() resets one field per session (never a wholesale dict swap).
        self._running_agents.clear()
        self._running_agents_ts.clear()
        self._pending_messages.clear()
        self._pending_approvals.clear()
        for _state in list((self.__dict__.get("_sessions") or {}).values()):
            _state.turn.lease = None
        getattr(self, "_busy_ack_ts", {}).clear()  # absent on bare shutdown-path doubles
        self._shutdown_event.set()
        # Global catch-all subprocess kill (safe to repeat) for the graceful path and late respawns, off-loop (#116327).
        # Awaited only within the leash left for the quiesce and the close; past it the live thread keeps SessionDB open
        # (final_sweep_live) until the process exit. Once settled, its thread is joined within a bound (QUIESCE-COMPOSITION-I1).
        _kill, self._stop_final_sweep_thread = GatewayRunner._stop_kill_tool_subprocesses_off_loop("final-cleanup")
        if not (await asyncio.wait({_kill}, timeout=stop_wait_budget(self, 3600.0, reserve=quiesce_reserve())))[0]:
            _kill.add_done_callback(consume_detached_task_result)
            logger.warning("Shutdown phase: final-cleanup tool kill still running at +%.2fs; no longer awaited "
                           "(the process exit ends it)", ctx.elapsed())
        elif join_settled_final_sweep(self, self._stop_final_sweep_thread, ctx.elapsed):  # else it logs why not
            logger.info("Shutdown phase: final-cleanup tool kill done at +%.2fs", ctx.elapsed())
        # Reap the auxiliary-client cache: clients bound to dead worker-thread loops leak httpx transports.
        def _reap_aux_clients() -> None:
            # Reap the process-global auxiliary-client cache once at the very end of teardown. Per-turn
            # cleanup runs in _cleanup_agent_resources for each active agent, but clients bound to
            # worker-thread loops that died with their ThreadPoolExecutor (notably cron ticks) only get
            # swept here. Without this, long-running gateways accumulate async httpx transports until they
            # hit EMFILE on macOS's default RLIMIT_NOFILE=256. See #14210.
            from agent.auxiliary_client import shutdown_cached_clients
            shutdown_cached_clients()

        GatewayShutdownMixin._quiet_step("shutdown_cached_clients error", _reap_aux_clients)
