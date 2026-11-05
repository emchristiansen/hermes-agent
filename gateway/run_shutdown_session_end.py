"""Session finalize dispatch, delivery of plugin ``on_session_finalize`` messages and bounded off-loop agent
cleanup for GatewayRunner.

Split out of ``gateway/run_shutdown.py``; bound onto ``GatewayRunner`` via ``GatewayShutdownMixin``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any, List, Optional

logger = logging.getLogger("gateway.run")

# What a stop keeps of its watchdog leash AFTER the agent finalize/cleanup waits: adapter teardown,
# state release, executor quiesce, SessionDB close and exit-state writes (well under a second when
# healthy). The settle already leaves STOP_TAIL_RESERVE_SECONDS for everything after it, but each
# finalize/cleanup wait has its own fixed bound (10s / 30s, per agent) and used to overrun that tail
# whenever a timed-out drain had spent the leash: the watchdog then os._exit-ed mid-finalize.
STOP_FINAL_RESERVE_SECONDS = 5.0


class GatewaySessionEndMixin:
    """Bounded off-loop ``finalize_session`` plus out-of-band delivery of what plugins return."""

    async def _deliver_session_end_messages(
        self, messages: list[str], *, source: Any = None, session_key: Optional[str] = None,
    ) -> None:
        """Send plugin ``on_session_finalize`` text to the chat that owns the session — a plain adapter send,
        never an agent turn. No resolvable chat (or no adapter) → logged and dropped."""
        if not messages:
            return
        if source is None and session_key:
            target = await self._shutdown_notification_target(session_key)
            source = target[0] if target else None
        adapter = self._delivery_adapter_for(source) if source is not None else None
        if adapter is None:
            logger.info("Session-end plugin message(s) dropped: no chat for session %s", session_key)
            return
        metadata = self._thread_metadata_for_source(source)
        for message in messages:
            await self._send_notice_logged(
                adapter, str(source.chat_id), message, source.platform.value if source.platform else "",
                "Failed to send session-end plugin message to %s:%s: %s", metadata=metadata)

    async def _reset_session_off_loop(self, *, session_key: Optional[str] = None, **kwargs: Any) -> None:
        """Plugin ``on_session_reset`` off-loop under the profile owning ``session_key``, bounded like
        finalize: a viewer's ``/new`` resets its conversation on the owner (gateway/session_viewer_close)."""
        def _call() -> None:
            from hermes_cli.lifecycle import invoke_hook
            invoke_hook("on_session_reset", **kwargs)

        try:
            await asyncio.wait_for(
                self._run_housekeeping_in_executor(self._run_release_in_profile_scope, _call, (), session_key),
                timeout=self._FINALIZE_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning("Session reset hooks (%s) exceeded %ss; proceeding without blocking the event loop.",
                           kwargs.get("old_session_id"), self._FINALIZE_TIMEOUT_S)
        except Exception:
            logger.debug("Session reset hooks (%s) failed", kwargs.get("old_session_id"), exc_info=True)

    async def _finalize_session_off_loop(
        self, *, session_id: Any, platform: str, reason: str, session_key: Optional[str] = None, **extra: Any,
    ) -> list[str]:
        """Run hermes_cli.lifecycle.finalize_session off-loop, bounded; on timeout the worker is left alone.
        ``session_key`` lets an unscoped caller (shutdown) enter the owning profile's scope: plugin
        ``on_session_finalize`` observers and the Relay coordinator (``current_profile_key``) resolve
        profile state at call time. Returns the plugins' user-facing messages (empty on timeout/error)."""
        messages: list[str] = []

        def _call() -> None:
            from hermes_cli.lifecycle import finalize_session, session_end_messages
            messages.extend(session_end_messages(
                finalize_session(session_id=session_id, platform=platform, reason=reason, **extra)))

        timeout = self._stop_wait_budget(self._FINALIZE_TIMEOUT_S)
        if timeout <= 0:
            logger.warning("Session finalize hooks (%s, reason=%s) skipped: the stop's watchdog leash is spent",
                           session_id, reason)
            return []
        try:
            await asyncio.wait_for(
                self._run_housekeeping_in_executor(self._run_release_in_profile_scope, _call, (), session_key),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "Session finalize hooks (%s, reason=%s) exceeded %.1fs; proceeding without blocking the event loop "
                "(the worker thread is left to finish on its own).", session_id, reason, timeout,
            )
        except Exception:
            logger.debug("Session finalize hooks (%s, reason=%s) failed", session_id, reason, exc_info=True)
        return list(messages)

    def _stop_wait_budget(self, timeout: float) -> float:
        """``timeout``, clamped once a stop is in progress to what its watchdog leash has left after
        ``STOP_FINAL_RESERVE_SECONDS``: no single bounded wait may spend the teardown that follows it."""
        deadline = getattr(self, "_stop_deadline", None)  # None outside a stop (and on bare test doubles)
        if deadline is None:
            return timeout
        return max(0.0, min(timeout, deadline - time.monotonic() - STOP_FINAL_RESERVE_SECONDS))

    async def _cleanup_agent_resources_off_loop(
        self, agent: Any, *, context: str = "", session_key: Optional[str] = None,
    ) -> None:
        """Run _cleanup_agent_resources in a worker thread, bounded; on timeout the worker is left alone.

        The teardown fires the memory-provider lifecycle hooks (``flush_pending`` → ``on_session_end`` →
        ``shutdown`` → ``close``), which read credentials/home at call time. In-turn callers carry the
        profile scope through ``_run_housekeeping_in_executor``; shutdown does not (it runs on the main
        loop, outside any adapter handler), so under multiplexing ``on_session_end`` failed closed and the
        session tail was never committed (#110622). ``_run_release_in_profile_scope`` enters the OWNING
        profile's scope from ``session_key`` when the caller has none, exactly like cache eviction.
        During a stop the bound is ``_stop_wait_budget``; with the leash spent the cleanup is not started."""
        if agent is None:
            return
        if context.startswith("shutdown") or context == "session expiry":
            with suppress(Exception):
                agent._end_session_on_close = False
        ctx_label = f" ({context})" if context else ""
        timeout = self._stop_wait_budget(self._CLEANUP_TIMEOUT_S)
        if timeout <= 0:
            logger.warning("Agent resource cleanup%s skipped: the stop's watchdog leash is spent", ctx_label)
            return
        try:
            await asyncio.wait_for(
                self._run_housekeeping_in_executor(
                    self._run_release_in_profile_scope, self._cleanup_agent_resources, (agent,), session_key,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "Agent resource cleanup%s exceeded %.1fs; proceeding without blocking the event loop (the worker "
                "thread is left to finish on its own). (#53175)", ctx_label, timeout,
            )
        except Exception as cleanup_exc:
            logger.warning("Agent resource cleanup%s failed: %s (#53175)", ctx_label, cleanup_exc, exc_info=True)
