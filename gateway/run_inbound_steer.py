"""Every message steers (Eric, 2026-11-04): the inbound steering half of ``GatewayInboundMixin``, moved out of
``gateway/run_inbound.py`` (code-health FILE_LINES). A messaging user's plain text, or an internal wake, that reaches a
running turn steers into that turn's agent instead of waiting for its own next turn."""

from __future__ import annotations

import logging

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")  # as gateway/run_inbound.py: same log records

# Inputs whose producer reads the turn's own result: a whole job, never steered into another turn.
_JOB_PRODUCER_PLATFORMS = frozenset({"a2a", "webhook", "api_server", "msgraph_webhook"})


class GatewayInboundSteerMixin:
    """Steering helpers of ``GatewayRunner``'s inbound path (``run_inbound``, ``run_busy``, ``run_turn``)."""

    def _hm_steer_at_adoption(self, event: MessageEvent, source, _quick_key: str) -> None:
        """Mark (in memory, never on the row) a messaging user's plain text that reached a running turn
        and that ``_hm_busy_preempt`` did not steer, so the authority's busy pass steers it into that
        turn's agent (every message steers, Eric 2026-11-04): at adoption when the turn had none yet, or
        at once when it adopted one while this admission awaited (a post-admission hook). An explicit
        ``/queue`` (``_fifo_only``), a job producer's input, media, another busy mode or a drain keeps
        its own next turn."""
        if (self._is_session_running(_quick_key)
                and not getattr(event, "_fifo_only", False) and not self._draining
                and self._hm_text_only(event) and (event.text or "").strip()
                and source.platform.value not in _JOB_PRODUCER_PLATFORMS
                and self._effective_busy_input_mode(source) == "steer"):
            event._steer_at_adoption = True

    def _hm_steer_delivered(self, event: MessageEvent, _quick_key: str) -> bool:
        """Steer an internal event's text into the running turn as a delivered message (its
        ``turn_author``, or Hermes automation): True when the running agent took it."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        state = self._peek_session_state(_quick_key)
        agent = state.turn.agent if state else None
        text = (event.text or "").strip()
        adapter = self._delivery_adapter_for(event.source)
        if (self._draining or self._hm_owner_retiring()  # the busy pass's gate, for this scope's owner
                or agent is None or agent is _AGENT_PENDING_SENTINEL or not text or not self._hm_text_only(event)
                or not callable(getattr(agent, "withdraw_steer", None))  # only the seam that records a sender
                or (adapter and _quick_key in getattr(adapter, "_pending_messages", {}))):  # keep arrival order
            return False
        try:
            return bool(agent.steer(text, delivered_by=dict((event.metadata or {}).get("turn_author") or {})))
        except Exception as exc:  # health: allow BLE001 -- as _hm_busy_steer: a failed steer queues the event as before
            logger.warning("Internal-event steer failed for session %s: %s", _quick_key, exc)
            return False

    def _hm_owner_retiring(self) -> bool:
        """True when the authority owning the active (routed) scope is retiring, or that scope has no
        serving authority (a removed profile); never under a runner without session authorities."""
        launch = getattr(self, "session_authority", None)
        if launch is None:
            return False
        from gateway.session_authorities import active_authority
        owner = active_authority(self)
        return launch.retiring or owner is None or owner.retiring

    def _hm_steer_pending_at_adoption(self, source: SessionSource, _quick_key: str) -> None:
        """The running turn just adopted its agent: each internal wake that waits in the adapter's
        pending slot steers into it now, as it would have on arrival (``_hm_steer_delivered``), in
        arrival order. The first event the turn does not take keeps its place, and everything behind it
        stays behind it: user text, commands, plugin payloads and job producers' events are never taken."""
        adapter = self._delivery_adapter_for(source)
        slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if not isinstance(slot, dict):
            return
        while (event := slot.get(_quick_key)) is not None:
            if (not getattr(event, "internal", False) or not event.allow_gateway_control or event.get_command()
                    or event.source.platform.value in _JOB_PRODUCER_PLATFORMS):
                return
            del slot[_quick_key]  # _hm_steer_delivered keeps arrival order while an earlier event waits
            if not self._hm_steer_delivered(event, _quick_key):
                slot[_quick_key] = event
                return
            overflow = self._overflow_queue(_quick_key)
            if overflow:
                slot[_quick_key] = overflow.pop(0)
