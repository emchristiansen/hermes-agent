"""Isolated script worlds for the CDP supervisor (``Page.createIsolatedWorld``).

A script evaluated in an isolated world shares the page's DOM but none of its JavaScript: the world has its
own globals and its own built-ins (``performance.now``, the ``HTMLInputElement`` value accessors,
``JSON.stringify``), which page code cannot replace, and its globals are invisible to the page. Page code
still runs synchronously where the DOM calls it (focus and input/change listeners, custom-element
callbacks). A world belongs to one document: after a navigation, evaluating in it fails with "Cannot find
context with specified id" and never runs in the new document.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Names of the worlds the vault creates; only their unique context ids are recorded.
VAULT_WORLD_PREFIX = "hermes-vault:"

# The CDP error message for an evaluation whose execution context no longer exists (the script never started).
CONTEXT_GONE_TEXT = "Cannot find context with specified id"


class WorldTrackingMixin:
    """``create_isolated_world`` and the bookkeeping of the vault worlds' unique context ids (event handlers
    run on the supervisor loop)."""

    async def _create_world(self, session_id: str, name: str, *, timeout: float) -> dict[str, Any]:
        """Create world ``name`` in the main frame of ``session_id``. ``grantUniveralAccess`` is left unset.
        Returns ``{"session_id", "context_id", "unique_id"}``; ``unique_id`` is the context's ``uniqueId``
        from ``Runtime.executionContextCreated`` (never reused across processes) when the event arrived
        before the reply, else None (the caller then refuses the world)."""
        tree = await self._cdp("Page.getFrameTree", session_id=session_id, timeout=timeout)
        frame_id = tree["result"]["frameTree"]["frame"]["id"]
        made = await self._cdp("Page.createIsolatedWorld", {"frameId": frame_id, "worldName": name},
                               session_id=session_id, timeout=timeout)
        context_id = made["result"]["executionContextId"]
        unique_id = self._world_unique_ids.get((session_id, context_id))
        return {"session_id": session_id, "context_id": context_id, "unique_id": unique_id}

    def _on_context_created(self, params: dict[str, Any], session_id: Optional[str]) -> None:
        context = params.get("context") or {}
        if str(context.get("name") or "").startswith(VAULT_WORLD_PREFIX) and context.get("uniqueId"):
            self._world_unique_ids[(session_id, context.get("id"))] = str(context["uniqueId"])

    def _on_context_destroyed(self, params: dict[str, Any], session_id: Optional[str]) -> None:
        self._world_unique_ids.pop((session_id, params.get("executionContextId")), None)

    def _on_contexts_cleared(self, params: dict[str, Any], session_id: Optional[str]) -> None:
        for key in [k for k in self._world_unique_ids if k[0] == session_id]:
            self._world_unique_ids.pop(key, None)

    EVENT_HANDLERS: dict[str, Callable[..., Any]] = {
        "Runtime.executionContextCreated": _on_context_created,
        "Runtime.executionContextDestroyed": _on_context_destroyed,
        "Runtime.executionContextsCleared": _on_contexts_cleared,
    }


# The CDP protocol errors (code, message) of an evaluation addressed by uniqueContextId whose world no longer
# exists. Measured in Chromium 157: the expression never runs, in the old document or the new one. Matched on
# the protocol error object, never on text (an exception description is page-reachable text).
WORLD_GONE_ERRORS = frozenset({(-32000, CONTEXT_GONE_TEXT), (-32602, "uniqueContextId not found")})


def world_gone(result: dict[str, Any]) -> bool:
    """True when ``result`` (a supervisor evaluation result) is one of WORLD_GONE_ERRORS for an evaluation
    addressed by uniqueContextId."""
    error = result.get("protocol_error")
    return (result.get("addressed") == "uniqueContextId" and isinstance(error, dict)
            and (error.get("code"), error.get("message")) in WORLD_GONE_ERRORS)


def world_params(world: dict[str, Any]) -> dict[str, Any]:
    """The ``Runtime.evaluate`` parameter that targets ``world``: its unique context id, never the numeric
    context id (measured: an old number, once reused, runs in the new document)."""
    return {"uniqueContextId": world["unique_id"]}
