"""Durable ``/v1/runs`` admission, status, events, and control handlers."""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]
try:
    from aiohttp.web_request import RequestKey
except ImportError:
    # Separate block: aiohttp < 3.14 lacks RequestKey, and a shared except
    # would reset the already-imported ``web`` to None (500 on POST /v1/runs).
    RequestKey = None  # type: ignore[assignment,misc]

from gateway.platforms.api_server_room_grants import _json_error, _room_grant_error_response
from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES


logger = logging.getLogger("gateway.platforms.api_server")

# Executor-thread lifetime for API-server agent turns (#116535). The handler-side
# ``_inflight_agent_runs`` count in api_server.py drops in the handler's ``finally`` when the
# handler task is cancelled (disconnect/shutdown) while the executor thread behind
# ``run_in_executor`` keeps running -- so the shutdown close gate cannot rely on it. This
# module-level count is taken on the submitting thread before ``run_in_executor`` and released
# in the worker thread's own ``finally``, which also keeps it visible past ``adapters.clear()``.
_API_WORKER_LOCK = threading.Lock()
_API_WORKER_LIVE = 0
# Serializes a run status mutation with the durable snapshot it enqueues: the loop and agent
# worker threads (approval prompts, tool events) both publish statuses, and the store keeps the
# newest ENQUEUED snapshot, so mutation order and enqueue order must be the same order. Held only
# for dict updates and a JSON encode, never across I/O.
_RUN_STATUS_LOCK = threading.RLock()


def api_worker_live_count() -> int:
    """Executor threads still inside an API-server agent turn (``_run_agent`` and ``/v1/runs``)."""
    with _API_WORKER_LOCK:
        return _API_WORKER_LIVE


def _submit_api_worker(loop, fn):
    """``loop.run_in_executor(None, fn)`` with the worker-lifetime count held for the submission.

    Increment on the submitting (handler) thread so the count is live before the worker can
    exit; decrement in the worker thread's own ``finally`` so handler cancellation cannot drop
    it early. When submission itself fails (default executor already shut down during quiesce
    -> RuntimeError) the worker never runs, so the count is released here instead — a leaked
    count would make the shutdown close gate skip the SessionDB close for the process lifetime.
    """
    global _API_WORKER_LIVE
    with _API_WORKER_LOCK:
        _API_WORKER_LIVE += 1

    def _counted():
        try:
            return fn()
        finally:
            global _API_WORKER_LIVE
            with _API_WORKER_LOCK:
                _API_WORKER_LIVE -= 1

    try:
        return loop.run_in_executor(None, _counted)
    except BaseException:
        with _API_WORKER_LOCK:
            _API_WORKER_LIVE -= 1
        raise
_ROOM_RETENTION_REQUEST_KEY = (
    RequestKey("hermes.room_run_retention_until", float) if RequestKey is not None
    else "hermes.room_run_retention_until")
# Forwarded subagent lifecycle fields; free-text ones are secret-redacted.
_SUBAGENT_EVENT_KEYS = (
    "goal", "task_count", "task_index", "subagent_id", "child_session_id", "delegation_id", "parent_id",
    "depth", "model", "tool_count", "status", "summary", "duration_seconds", "input_tokens",
    "output_tokens", "reasoning_tokens", "api_calls", "cost_usd", "files_read", "files_written",
    "output_tail")
_SUBAGENT_TEXT_KEYS = ("goal", "summary", "output_tail")
# Terminal usage payload: (wire key, agent attribute), in wire order. Cache reads ride along so a
# cost poller does not book them as full-price input (#102101).
_USAGE_FIELDS = (
    ("input_tokens", "session_prompt_tokens"), ("output_tokens", "session_completion_tokens"),
    ("total_tokens", "session_total_tokens"), ("cache_read_tokens", "session_cache_read_tokens"),
    ("cache_write_tokens", "session_cache_write_tokens"))
# Tool-progress event -> SSE payload fields (tool_name, preview, kwargs); key order is wire format.
def _call_id(kw: dict[str, Any]) -> dict[str, Any]:
    return {"tool_call_id": kw["tool_call_id"]} if kw.get("tool_call_id") else {}


_FIXED_EVENT_FIELDS = {
    "tool.started": lambda tool, preview, kw: {"tool": tool, "preview": preview, **_call_id(kw)},
    "tool.completed": lambda tool, preview, kw: {
        "tool": tool, "duration": round(kw.get("duration", 0), 3), "error": kw.get("is_error", False),
        **_call_id(kw)},
    "reasoning.available": lambda tool, preview, kw: {"text": preview or ""}}
_TOOL_COMPLETED_PREVIEW_MAX_CHARS = 500


def _tool_completed_preview(result: Any, redact_sensitive_text: Callable[..., str]) -> str:
    """Bounded, secret-redacted result summary for the public run stream — redacted BEFORE
    truncation so a cut never leaves a secret's prefix on the wire."""
    if result is None:
        return ""
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    preview = redact_sensitive_text(text, force=True)
    limit = _TOOL_COMPLETED_PREVIEW_MAX_CHARS
    return preview if len(preview) <= limit else preview[: limit - 3] + "..."


_RUN_STREAM_SUBSCRIBER_OVERFLOW = object()
_RUN_STREAM_WRITE_TIMEOUT = 5.0


class _RunStream:
    """Sequence and retain one run's events while fanning out to SSE clients."""

    BACKLOG_LIMIT = 1000
    SUBSCRIBER_QUEUE_LIMIT = 256

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.backlog: deque[tuple[int, Optional[dict[str, Any]]]] = deque(
            maxlen=self.BACKLOG_LIMIT
        )
        self.next_seq = 0
        self.terminal = False

    def put_nowait(self, event: Optional[dict[str, Any]]) -> None:
        if self.terminal:
            return
        seq = self.next_seq
        self.next_seq += 1
        self.backlog.append((seq, event))
        if event is None:
            self.terminal = True
        for queue in list(self.subscribers):
            try:
                queue.put_nowait((seq, event))
            except asyncio.QueueFull:
                self.detach(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait((seq, _RUN_STREAM_SUBSCRIBER_OVERFLOW))

    def attach(
        self, last_seq: int = -1
    ) -> tuple[asyncio.Queue, list[tuple[int, Optional[dict[str, Any]]]]]:
        replay = [(seq, event) for seq, event in self.backlog if seq > last_seq]
        # Headroom for events produced while the replay is still being written.
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.SUBSCRIBER_QUEUE_LIMIT + len(replay))
        self.subscribers.add(queue)
        return queue, replay

    def detach(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)


def _remember_room_retention(request: web.Request, claims: dict[str, Any]) -> None:
    value = float(claims.get("status_expires_at") or claims.get("expires_at") or 0)
    try:
        request[_ROOM_RETENTION_REQUEST_KEY] = value
    except (AttributeError, TypeError):
        request._hermes_room_run_retention_until = value


def _room_retention_until(request: web.Request) -> float:
    try:
        value = request.get(_ROOM_RETENTION_REQUEST_KEY, 0)
    except AttributeError:
        value = getattr(request, "_hermes_room_run_retention_until", 0)
    return max(0.0, float(value or 0))


def _run_event(run_id: str, name: str, **fields: Any) -> dict[str, Any]:
    """Build one SSE event payload (key order is part of the wire format)."""
    return {"event": name, "run_id": run_id, "timestamp": time.time(), **fields}


def terminal_run_status(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map a ``run_conversation`` result to its terminal run status and the wire fields every
    terminal event/status carries. An interrupted turn is ``cancelled``; a turn that ended
    without finishing (``failed``, ``partial``, or ``completed=False`` such as the iteration
    budget) is ``failed``, so ``completed: true`` never rides next to ``partial: true``."""
    interrupted = bool(result.get("interrupted"))
    finished = (
        not interrupted and not result.get("failed") and not result.get("partial")
        and result.get("completed") is not False
    )
    status = "cancelled" if interrupted else "completed" if finished else "failed"
    fields: dict[str, Any] = {
        "completed": finished, "partial": bool(result.get("partial")), "interrupted": interrupted,
    }
    if not finished and result.get("turn_exit_reason"):
        fields["turn_exit_reason"] = str(result["turn_exit_reason"])
    if result.get("pending_steer"):
        # Undelivered steer text rides on every terminal event/status for client replay.
        fields["pending_steer"] = result["pending_steer"]
    return status, fields


def _run_not_found(_openai_error, run_id: str) -> web.Response:
    return _json_error(_openai_error, f"Run not found: {run_id}", code="run_not_found", status=404)


def _uses_room_run_auth(self, request: web.Request) -> bool:
    return request.path.endswith("/v1/runs") and bool(self._room_grant_token(request))


def _initialize_run_state(self, *, store_factory) -> None:
    """Initialize adapter-owned durable and live ``/v1/runs`` state."""
    self._run_idempotency_store = store_factory()
    self._run_owner_pid = os.getpid()
    try:
        from gateway.status import get_process_start_time
        self._run_owner_started = int(get_process_start_time(self._run_owner_pid) or 0)
    except Exception:
        self._run_owner_started = 0
    # All keyed by run_id: SSE fanout streams (+creation time for the TTL sweep), live
    # agent/task refs for cooperative stop (the executor thread may outlive the request,
    # hence the separate stopping set), pollable statuses, and approval session keys
    # (approval core resolves by session key, clients by run_id).
    self._run_idempotency_ids: set[str] = set()
    self._stopping_run_ids: set[str] = set()
    self._shutdown_interrupted_run_ids: set[str] = set()
    self._run_shutdown_requested_at: Optional[float] = None
    # Streaming run id -> its canonical admission id, held while that request observes the turn.
    self._run_admission_aliases: dict[str, str] = {}
    (
        self._run_owners, self._run_streams, self._run_streams_created, self._active_run_agents,
        self._active_run_tasks, self._run_statuses, self._run_approval_sessions,
    ) = ({} for _ in range(7))


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [
        ("POST", "/v1/runs", self._handle_runs), ("GET", "/v1/runs/{run_id}", self._handle_get_run),
        ("GET", "/v1/runs/{run_id}/events", self._handle_run_events),
        ("POST", "/v1/runs/{run_id}/approval", self._handle_run_approval),
        ("POST", "/v1/runs/{run_id}/clarify", self._handle_run_clarify),
        ("POST", "/v1/runs/{run_id}/steer", self._handle_steer_run),
        ("POST", "/v1/runs/{run_id}/resolve-unknown", self._handle_resolve_unknown_run),
        ("POST", "/v1/runs/{run_id}/stop", self._handle_stop_run)]


def _idempotency_capabilities(self, *, store_type) -> dict[str, Any]:
    return {
        "supported": True,
        "durable": self._run_idempotency_store.durable,
        "retention_seconds": store_type.RETENTION_SECONDS}


def _close_run_state(self) -> None:
    try:
        if getattr(self, "_run_idempotency_store", None) is not None:
            self._run_idempotency_store.close()
    except Exception:
        logger.debug("Failed to close run idempotency store for %s", self.name, exc_info=True)


async def _close_run_state_async(self) -> None:
    """``_close_run_state`` for the event loop: queued status commits drain on the store's own
    worker before its connection closes, and the loop is never the thread that waits for them."""
    try:
        store = getattr(self, "_run_idempotency_store", None)
        aclose = getattr(store, "aclose", None)
        if aclose is not None:
            await aclose()
        elif store is not None:
            store.close()
    except Exception:
        logger.debug("Failed to close run idempotency store for %s", self.name, exc_info=True)


def _set_run_status(self, run_id: str, status: str, **fields: Any) -> dict[str, Any]:
    """Update pollable run status without exposing private agent objects.

    A keyed run's status is persisted write-behind (``enqueue_status``): this may run on the
    event loop or on an agent worker thread, and neither waits for the commit's fsync."""
    with _RUN_STATUS_LOCK:
        return _set_run_status_locked(self, run_id, status, **fields)


def _set_run_status_locked(self, run_id: str, status: str, **fields: Any) -> dict[str, Any]:
    now = time.time()
    current = self._run_statuses.get(run_id, {})
    previous_status = str(current.get("status") or "")
    field_names = set(fields)
    current.update({"object": "hermes.run", "run_id": run_id, "status": status, "updated_at": now})
    current.setdefault("created_at", fields.pop("created_at", now))
    current.update(fields)
    shutdown_requested_at = getattr(self, "_run_shutdown_requested_at", None)
    if shutdown_requested_at is not None and status not in TERMINAL_STATUSES:
        current.setdefault("shutdown_requested_at", shutdown_requested_at)
    if status != "waiting_for_approval":
        current.pop("approval", None)
    self._run_statuses[run_id] = current
    should_persist = (
        status != previous_status
        or status in TERMINAL_STATUSES
        or bool(field_names & {
            "output", "error", "usage", "pending_steer", "session_id", "shutdown_requested_at"}))
    if run_id in self._run_idempotency_ids and should_persist:
        try:
            self._run_idempotency_store.enqueue_status(run_id, current)
        except Exception:
            logger.exception("[api_server] failed to persist idempotent run status %s", run_id)
    return current


def _mark_shutdown_interrupted_runs(self, run_ids) -> None:
    """Publish the shutdown outcome before cooperative interruption can race teardown."""
    for run_id in run_ids:
        self._shutdown_interrupted_run_ids.add(run_id)
        self._set_run_status(
            run_id,
            "interrupted",
            error="Gateway shutdown interrupted the run.",
            last_event="run.interrupted",
        )


def _mark_shutdown_requested(self) -> int:
    """Persist when gateway shutdown began without changing the run state vocabulary."""
    shutdown_requested_at = getattr(self, "_run_shutdown_requested_at", None)
    if shutdown_requested_at is None:
        shutdown_requested_at = time.time()
        self._run_shutdown_requested_at = shutdown_requested_at
    marked = 0
    for run_id, current in list(self._run_statuses.items()):
        if current.get("status") in TERMINAL_STATUSES or "shutdown_requested_at" in current:
            continue
        self._set_run_status(
            run_id, str(current.get("status") or "running"),
            shutdown_requested_at=shutdown_requested_at)
        marked += 1
    return marked


def _make_run_event_callback(self, run_id: str, loop: asyncio.AbstractEventLoop, *, _api_server):
    """Return a callback that pushes structured events to the run SSE queue."""
    redact_sensitive_text = _api_server.redact_sensitive_text

    def _push(event: dict[str, Any]) -> None:
        self._set_run_status(
            run_id, self._run_statuses.get(run_id, {}).get("status", "running"), last_event=event.get("event"))
        q = self._run_streams.get(run_id)
        if q is not None:
            with suppress(Exception):
                loop.call_soon_threadsafe(q.put_nowait, event)

    def _callback(event_type: str, tool_name: str | None = None, preview: str | None = None, args=None, **kwargs):
        # _thinking / subagent.tool / subagent_progress are deliberately dropped (UI noise);
        # lifecycle boundaries must land so clients can observe delegate_task failures.
        fields = _FIXED_EVENT_FIELDS.get(event_type)
        if fields is not None:
            event_fields = fields(tool_name, preview, kwargs)
            if event_type == "tool.completed":
                event_fields["preview"] = _tool_completed_preview(
                    kwargs.get("result"), redact_sensitive_text)
            _push(_run_event(run_id, event_type, **event_fields))
        elif event_type in {"subagent.start", "subagent.complete"}:
            event = _run_event(run_id, event_type)
            if preview is not None:
                event["preview"] = redact_sensitive_text(str(preview), force=True)
            for key in _SUBAGENT_EVENT_KEYS:
                value = kwargs.get(key)
                if value is not None:
                    # Free text may carry child tool output: force secret redaction on this public stream.
                    redact = key in _SUBAGENT_TEXT_KEYS and isinstance(value, str)
                    event[key] = redact_sensitive_text(value, force=True) if redact else value
            _push(event)

    return _callback


# Answering a prompt the run raised is one capability whichever tool asked: approvals
# and clarify share the grant's ``approve`` permission. Acknowledging a lost owner epoch
# ends the run like a stop, so ``/resolve-unknown`` rides the ``stop`` permission.
_ROOM_CONTROL_PERMISSIONS = {
    "/stop": "stop", "/resolve-unknown": "stop", "/approval": "approve", "/clarify": "approve"}


def _room_permission_for(request: web.Request) -> str:
    for suffix, permission in _ROOM_CONTROL_PERMISSIONS.items():
        if request.path.endswith(suffix):
            return permission
    return "status" if request.method == "GET" else "dispatch"


def _run_idempotency_scope(self, request: web.Request, *, _api_server) -> str:
    """Opaque auth/profile namespace; never persist bearer credentials."""
    if self._room_grant_token(request):
        claims = self._room_grant_claims(request, permission=_room_permission_for(request))
        _remember_room_retention(request, claims)
        parts = (claims[k] for k in (
            "room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
            "member_id", "target_install_id", "target_profile"))
    else:
        parts = (_api_server._api_request_profile.get() or "default",
                 self._expected_api_key() or "unauthenticated-test-listener")
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _check_run_auth(self, request: web.Request, *, permission: str, _api_server) -> web.Response | None:
    if not self._room_grant_token(request):
        return self._check_auth(request)
    try:
        self._room_grant_claims(request, permission=permission)
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_api_server._openai_error)
    return None


def _owner_alive(owner_pid: int, owner_started: int) -> bool:
    """True when the recorded owner pid still exists and is the same process incarnation."""
    try:
        from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match
        return owner_pid > 0 and bool(_pid_exists(owner_pid)) and (
            not owner_started
            or start_time_fingerprints_match(owner_started, get_process_start_time(owner_pid) or 0))
    except Exception:
        return False


def _canonical_run_status(self, run_id: str) -> dict[str, Any] | None:
    from gateway.platforms.api_server_authority_runs import run_projection
    canonical = run_projection(self, run_id)
    if canonical is None:
        return None
    # The documented poll shape: the canonical projection decides status/output/usage; the
    # retained run record still names the object, the requested model and the served runtime.
    retained = self._run_statuses.get(run_id) or {}
    return {"object": "hermes.run", **{key: retained[key] for key in ("model", "runtime", "created_at")
                                       if key in retained}, **canonical}


def _stale_owner_status(record: dict[str, Any]) -> dict[str, Any] | None:
    """The ``interrupted`` status a durable non-terminal record whose owner is gone becomes."""
    status = dict(record["status"])
    if status.get("status") in TERMINAL_STATUSES or _owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)):
        return None
    status.update(
        status="interrupted", error="The gateway restarted before this run settled.",
        last_event="run.interrupted", updated_at=time.time())
    return status


def _load_durable_record(store, scope: str, run_id: str, retention_until: float):
    """Store-worker half of a durable status hydration: the scoped read (with its retention
    extension) and the stale-owner check, whose ``interrupted`` status is committed here. A
    provisional reservation (``admitted: False``) whose owner still lives is not a run yet: its
    owner may still release it, so it is not reported (an owner that died undecided is answered
    by the stale-owner rule, as before)."""
    record = store.status_for_run(scope, run_id, retention_until=retention_until)
    if record is None:
        return None
    if record.get("admitted") is False and _owner_alive(
            int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)):
        return None
    stale = _stale_owner_status(record)
    if stale is not None:
        store.update_status(run_id, stale)
    return stale or dict(record["status"])


def _adopt_durable_status(self, run_id: str, scope: str, status: dict[str, Any]) -> dict[str, Any]:
    self._run_statuses[run_id] = status
    self._run_idempotency_ids.add(run_id)
    self._run_owners[run_id] = scope
    return status


async def _durable_run_status_async(self, request: web.Request, run_id: str) -> dict[str, Any] | None:
    """``_durable_run_status`` with every run-store commit and read on the store's worker."""
    canonical = _canonical_run_status(self, run_id)
    if canonical is not None:
        return canonical
    store = self._run_idempotency_store
    status = self._run_statuses.get(run_id)
    if status is not None:
        if run_id in self._run_idempotency_ids:
            if status.get("status") in TERMINAL_STATUSES:
                await _terminal_status_durable(self, run_id)
            scope = self._run_idempotency_scope(request)
            await store.extend_retention_async(scope, run_id, _room_retention_until(request))
        return status
    scope = self._run_idempotency_scope(request)
    hydrated = await asyncio.wrap_future(store.submit(
        _load_durable_record, store, scope, run_id, _room_retention_until(request)))
    # Another request (or this process's own run) may have published the status meanwhile:
    # the live record wins over the durable one, exactly as it does before the read.
    current = self._run_statuses.get(run_id)
    if current is not None:
        return current
    return None if hydrated is None else _adopt_durable_status(self, run_id, scope, hydrated)


def _durable_run_status(self, request: web.Request, run_id: str) -> dict[str, Any] | None:
    """Hydrate a scoped run status and fail stale owners closed (synchronous; the HTTP handlers
    use ``_durable_run_status_async``)."""
    canonical = _canonical_run_status(self, run_id)
    if canonical is not None:
        return canonical
    status = self._run_statuses.get(run_id)
    if status is not None:
        if run_id in self._run_idempotency_ids:
            scope = self._run_idempotency_scope(request)
            self._run_idempotency_store.extend_retention(scope, run_id, _room_retention_until(request))
        return status
    scope = self._run_idempotency_scope(request)
    hydrated = _load_durable_record(
        self._run_idempotency_store, scope, run_id, _room_retention_until(request))
    return None if hydrated is None else _adopt_durable_status(self, run_id, scope, hydrated)


def _resolve_conversation_history(
    self, body: dict, raw_input: Any, *, _openai_error
) -> tuple[list[dict[str, str]], Any, Any, web.Response | None]:
    """Return ``(history, instructions, stored_session_id, error)``; precedence:
    ``conversation_history`` > ``previous_response_id`` chain > all-but-last ``input`` messages."""
    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")
    conversation_history: list[dict[str, str]] = []
    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            return [], instructions, None, _json_error(
                _openai_error, "'conversation_history' must be an array of message objects", status=400)
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or {"role", "content"} - set(entry):
                return [], instructions, None, _json_error(
                    _openai_error, f"conversation_history[{i}] must have 'role' and 'content' fields",
                    status=400)
            conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
        if previous_response_id:
            logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")
    stored_session_id = None
    if not conversation_history and previous_response_id:
        stored = self._current_response_store().get(previous_response_id)
        if stored:
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            if instructions is None:
                instructions = stored.get("instructions")
    if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
        for msg in raw_input[:-1]:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):  # flatten multi-part content blocks to text
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict) and p.get("type") == "text")
                conversation_history.append({"role": msg["role"], "content": str(content)})
    return conversation_history, instructions, stored_session_id, None


async def _resolve_chained_history(self, body: dict, raw_input: Any, *, _openai_error):
    """``_resolve_conversation_history`` with the chain's live session: a deleted chain moves to
    its next generation (an explicit body ``session_id`` still wins in the caller), and a store
    that cannot answer is the retryable store verdict (``continued_session``). The last item is the
    stored session the chain moved past (``walked_from``), else None."""
    from gateway.platforms.api_server_authority_runs import continued_session
    history, instructions, chain_root, error = _resolve_conversation_history(
        self, body, raw_input, _openai_error=_openai_error)
    stored_session_id = chain_root
    if error is None:
        stored_session_id, error = await continued_session(self, chain_root)
    return history, instructions, stored_session_id, error, (
        chain_root if stored_session_id and stored_session_id != chain_root else None)


def _accepted_response(run_id: str, status: str, gateway_session_key, *, replayed: bool) -> web.Response:
    """202 admission response; replays are flagged via ``Idempotency-Replayed``."""
    headers = {"Idempotency-Replayed": "true"} if replayed else {}
    if gateway_session_key:
        headers["X-Hermes-Session-Key"] = gateway_session_key
    return web.json_response(
        {"run_id": run_id, "status": status, "replayed": replayed}, status=202, headers=headers)


# Rounds of join / look up / answer one keyed request may take before it answers "retry": each
# round either waits for an in-process reservation to be decided or reads the key once more.
_KEY_SETTLE_ROUNDS = 8


class _KeyReservation:
    """One in-process ``(scope, key)`` reservation, from the moment its commit is submitted until its
    request is decided (admitted, refused, abandoned) or its row is released. A same-key request
    joins it instead of replaying a row that a cancelled or refused admission may yet delete.

    A release that did not delete the row (``failed``) is still owed: the entry stays registered,
    so no same-key request treats the key as admitted or as free, and the next one retries it."""

    __slots__ = ("failed", "future", "identity", "releasing")

    def __init__(self, identity: tuple[str, str], loop: asyncio.AbstractEventLoop):
        self.identity = identity
        self.future = loop.create_future()
        self.releasing = False  # a release owns settlement: it settles once the row is gone
        self.failed = False


def _key_reservations(self) -> dict:
    registry = getattr(self, "_run_key_reservations", None)
    if registry is None:  # bare ``__new__`` fixtures skip _initialize_run_state
        registry = self._run_key_reservations = {}
    return registry


def _wake_key_joiners(entry: _KeyReservation) -> None:
    if not entry.future.done():
        entry.future.set_result(None)


def _settle_key_reservation(self, entry: _KeyReservation) -> None:
    """Event-loop thread only: the key's decision is final; joiners read the key afresh."""
    registry = _key_reservations(self)
    if registry.get(entry.identity) is entry:
        del registry[entry.identity]
    _wake_key_joiners(entry)


def _release_key_reservation(self, entry: _KeyReservation, loop):
    """Event-loop thread only. Delete a committed reservation whose request was not admitted.
    Joiners resume only once the DELETE has committed (then the key is free) or failed (then the
    release is still owed: ``failed``, retried by the next same-key request). -> the release's
    future (its exception is the failure), or None when the store no longer accepts work."""
    entry.releasing, entry.failed = True, False
    if entry.future.done():
        entry.future = loop.create_future()
    store = self._run_idempotency_store
    try:
        released = store.submit(store.forget, *entry.identity)
    except RuntimeError as exc:  # closed: the row stays provisional; its owner's death answers it
        _key_release_done(self, entry, exc)
        return None

    def _done(done) -> None:
        error = None if not done.cancelled() and done.exception() is None else (
            done.exception() if not done.cancelled() else asyncio.CancelledError())
        with suppress(RuntimeError):  # the loop is closed: nobody is left to join it
            loop.call_soon_threadsafe(_key_release_done, self, entry, error)
    released.add_done_callback(_done)
    return released


def _key_release_done(self, entry: _KeyReservation, error: Optional[BaseException]) -> None:
    entry.releasing = False
    if error is None:
        _settle_key_reservation(self, entry)
        return
    logger.warning("[api_server] an unadmitted run reservation was not released (%s); it stays "
                   "owed and same-key requests are refused as retryable until it is", error)
    entry.failed = True
    _wake_key_joiners(entry)


def _confirm_key_admission(self, entry: Optional[_KeyReservation], run_id: str) -> None:
    """The run exists now: its provisional reservation becomes an admission for every reader, in
    any process (write-behind and owed until it commits; until then another process answers a
    retryable 503, never a replay of a row this one could still release)."""
    if entry is None:
        return
    try:
        self._run_idempotency_store.enqueue_confirmation(*entry.identity, run_id)
    except RuntimeError:
        logger.warning("[api_server] could not confirm an admitted run reservation (store closed)")


async def _admitted_by_authority(self, run_id: str) -> bool:
    """Whether the session authority durably admitted *run_id* (off the loop). An admission it
    committed is the run's truth whatever became of the HTTP request that asked for it -- a
    cancelled caller never confirms its reservation, yet the shielded admission completes and
    runs -- and the refusal path never releases a reservation whose admission exists.

    The state.db read is owned by the gateway runner until its thread has finished
    (``owned_offload``): a caller cancelled mid-read leaves it counted by the shutdown close gate,
    so SessionDB is never closed under it."""
    runner = getattr(self, "gateway_runner", None)
    if getattr(runner, "session_authority", None) is None:
        return False
    from gateway.platforms.api_server_authority_runs import run_admission
    from gateway.run_notifications_offload import owned_offload
    try:
        return await owned_offload(runner, run_admission, self, run_id) is not None
    except Exception as exc:  # health: allow BLE001 -- unknown is answered as still pending
        logger.debug("[api_server] admission lookup for run %s failed: %s", run_id, exc)
        return False


async def _confirm_recognized_admission(self, identity, run_id: str) -> bool:
    """Confirm the provisional reservation of *run_id*, whose durable admission was recognized ->
    committed (a no-op if the key names another run by now: the caller reads the key again);
    a failed commit stays owed (``enqueue_confirmation``) and the caller answers 503 retry."""
    store = self._run_idempotency_store
    try:
        await asyncio.wrap_future(store.submit(store.confirm_admission, *identity, run_id))
        return True
    except Exception as exc:  # health: allow BLE001 -- owed, and the request is answered retryable
        logger.warning("[api_server] failed to confirm a recognized run admission (%s); still owed", exc)
        with suppress(RuntimeError):
            store.enqueue_confirmation(*identity, run_id)
        return False


async def _join_key_reservation(entry: _KeyReservation) -> None:
    # Shielded: a joiner's own cancellation never settles the reservation it waits for.
    await asyncio.shield(entry.future)


@contextmanager
def _key_reservation_held(self, entry: Optional[_KeyReservation]):
    """Joiners of *entry* resume once this request is decided, whichever way; a release in
    progress (or still owed) settles them itself."""
    try:
        yield
    finally:
        if entry is not None and not entry.releasing and not entry.failed:
            _settle_key_reservation(self, entry)


def _lookup_for_replay(store, scope, key, fingerprint, retention_until):
    """Store-worker lookup; a provisional row also says whether its owner still lives."""
    outcome, record = store.lookup(scope, key, fingerprint, retention_until=retention_until)
    if record is not None and record.get("admitted") is False:
        record["owner_alive"] = _owner_alive(record["owner_pid"], record["owner_started"])
    return outcome, record


def _replayable(self, record: dict[str, Any]) -> bool:
    """Whether a reused row names a run that may be acknowledged as admitted. A provisional row
    may still be released by its owner: only that owner (this process, which knows the decision)
    or the stale-owner rule (its owner died undecided -> ``interrupted``) may answer for it."""
    if record.get("admitted") is not False:
        return True
    mine = (record["owner_pid"], record["owner_started"]) == (
        getattr(self, "_run_owner_pid", None), getattr(self, "_run_owner_started", None))
    if mine:
        return str(record["run_id"]) in self._run_statuses
    return not record.get("owner_alive", True)


def _key_in_flight(_openai_error) -> web.Response:
    response = _json_error(
        _openai_error, "Idempotency-Key is still being settled by another request; retry",
        code="idempotency_key_in_flight", status=503)
    response.headers["Retry-After"] = "1"
    return response


async def _keyed_answer(self, request, identity, fingerprint, gateway_session_key, _openai_error, found=None):
    """The answer to a keyed POST whose key is taken (202 replay, 409 conflict, 503 retry), or None
    once the key is free to reserve -- and then no in-process reservation of it is pending.

    A replay is only ever answered for an admitted reservation: one still being decided in this
    process is joined first (an owed release is retried first), a provisional row another live
    process may still release is answered 503 retry, and a row that is gone by the time its run is
    read is looked up again rather than acknowledged from the stale record."""
    registry, store = _key_reservations(self), self._run_idempotency_store
    loop = asyncio.get_running_loop()
    for _ in range(_KEY_SETTLE_ROUNDS):
        pending = registry.get(identity)
        if pending is not None:
            if pending.failed:
                _release_key_reservation(self, pending, loop)
            await _join_key_reservation(pending)
            found = None
            continue
        if found is None:
            found = await asyncio.wrap_future(store.submit(
                _lookup_for_replay, store, *identity, fingerprint, _room_retention_until(request)))
            continue  # look at the registry again: a reservation may have started meanwhile
        outcome, record = found
        if outcome == "missing":
            return None
        if outcome == "conflict":
            return _json_error(
                _openai_error, "Idempotency-Key was already used with a different request payload",
                code="idempotency_key_conflict", status=409)
        if record.get("admitted") is False and "owner_alive" not in record:
            found = None  # a provisional row from ``reserve``: read it with its owner's liveness
            continue
        if not _replayable(self, record):
            recognized = str(record["run_id"])
            if await _admitted_by_authority(self, recognized) and \
                    await _confirm_recognized_admission(self, identity, recognized):
                found = None  # now admitted for every reader: read it again
                continue
            return _key_in_flight(_openai_error)
        original_id = str(record["run_id"])
        status = await _durable_run_status_async(self, request, original_id)
        if status is not None and identity not in registry:
            return _accepted_response(original_id, status.get("status", "queued"), gateway_session_key, replayed=True)
        found = None  # released (or re-reserved) meanwhile: decide again from the store
    return _key_in_flight(_openai_error)


@dataclass(slots=True)
class _RunLaunch:
    """State for an admitted run's background task; contextvars are captured here
    because the task outlives the request (and its middleware profile scope)."""

    owner: Any
    run_id: str
    queue: _RunStream
    session_id: str
    gateway_session_key: Optional[str]
    declared_selected: bool
    user_message: str
    conversation_history: list[dict[str, str]]
    # #98619: only continuation paths that reload session history may grant wake authority —
    # a previous_response_id continuation consumes its ResponseStore snapshot instead, and a
    # caller-supplied conversation_history is authoritative for the turn; neither consumes a
    # SessionDB delivery row, so both stay default-denied.
    session_history_delivery: bool
    agent_kwargs: dict  # ``_create_agent`` keyword arguments (prompt, model overrides, route, room policy)
    request_profile: Any
    browser_control_principal: Any
    browser_control_transport_family: Any
    turn_author: Optional[dict[str, Any]] = None  # memory-attribution label only; grants nothing
    admission: Any = None
    walked_from: Optional[str] = None  # the swept stored session the chain moved past (``bind_api_steps``)

    @property
    def approval_session_key(self) -> str:
        # Isolated per run: session ids are conversation scopes, not authorization namespaces.
        return self.run_id

    def put_event(self, event: Optional[dict]) -> None:
        """Enqueue only while this run still owns live transport state."""
        if self.owner._run_streams.get(self.run_id) is self.queue:
            self.queue.put_nowait(event)


def _forget_run(self, run_id: str, *tables) -> None:
    """Drop *run_id* from the given run-keyed dicts/sets, then release its owner stamp."""
    for table in tables:
        (table.discard if isinstance(table, set) else lambda k: table.pop(k, None))(run_id)
    self._release_run_owner_if_forgotten(run_id)


def _retire_live_run(self, run_id: str) -> None:
    """Retire agent/task/approval control state once the executor-backed task is done."""
    _forget_run(self, run_id, self._active_run_agents, self._active_run_tasks, self._run_approval_sessions,
                self._stopping_run_ids, self._shutdown_interrupted_run_ids)


def _drop_run_transport(self, run_id: str) -> None:
    _forget_run(
        self,
        run_id,
        self._run_streams,
        self._run_streams_created,
    )


async def _resolve_live_session_id(self, session_id: str) -> str:
    """Adopt the live compression-continuation tip for a client-addressed session (#98619):
    a /v1/runs run bound to a pre-rotation id would otherwise load a stale history slice and
    write its turn into the closed parent (CompressionSessionClosedError). Same canonical
    resolution ``/api/sessions/{id}/messages`` reads use; fails open to the original id."""
    db = await self._ensure_session_db_async()
    resolver = getattr(db, "resolve_resume_session_id", None) if db is not None else None
    if not callable(resolver):
        return session_id
    try:
        resolved = await asyncio.to_thread(resolver, session_id)
        return str(resolved) if resolved else session_id
    except Exception:
        logger.debug("/v1/runs live-session resolve failed for %s", session_id, exc_info=True)
        return session_id


async def run_internal_session_turn(self, *, session_id: str, text: str, profile: str,
                                notification_category: str = "result", _api_server) -> None:
    """Run one background wake turn against a raw session id IN-PROCESS (no HTTP, no API key).

    The HTTP wake self-post cannot serve a multiplexed *served* profile: ``/p/<profile>/`` on
    the shared listener authenticates with that profile's own ``API_SERVER_KEY`` — which a
    route-only profile legitimately does not have — while an unprefixed self-post would resume
    the session in the DEFAULT profile's store. ``gateway.wake`` therefore runs the turn here,
    inside the owner profile's runtime scope (the session DB, model resolution and tool policy
    all follow the ambient scope), with ``profile`` naming the profile the caller proved owns
    the session — never derived here, so a missing proof cannot silently become the default.
    Raises on failure so the caller can rewind its cursor: a draining gateway fails at once
    (the HTTP self-post's 503) while a saturated concurrent-run cap is retried with the same
    backoff the HTTP self-post uses for a 429.
    """
    from gateway.wake import _RETRY_DELAYS_SECONDS
    profile = (profile or "").strip()
    if not profile:
        raise ValueError("run_internal_session_turn requires the owning profile")
    token = _api_server._api_request_profile.set(profile)
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    last_err: Optional[BaseException] = None
    try:
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
            if self._draining_response() is not None:
                raise RuntimeError(
                    f"internal wake refused for session {session_id}: the gateway is draining")
            # Transient: the cap clears on its own, exactly as the HTTP self-post's 429 does.
            if self._concurrency_limited_response() is not None:
                last_err = RuntimeError(
                    "internal wake deferred: the API server is at its concurrent-run cap")
                logger.warning("%s; attempt %d/%d", last_err, attempt + 1, attempts)
                continue
            # #98619/#13437: adopt the live continuation tip first, the same canonical
            # resolution the HTTP self-post consumes — a compressed origin must be woken on the
            # transcript that is actually live, never the retired parent slice.
            resolved = await _resolve_live_session_id(self, session_id)
            session, err = await self._get_existing_session_or_404(resolved)
            if err is not None or not session:
                raise RuntimeError(
                    f"internal wake target session {resolved!r} is not in the active profile store")
            history = await self._conversation_history_for_session(resolved)
            # Same route resolution as the HTTP self-post (/v1/chat/completions): a model_routes
            # alias for the virtual model applies to the wake turn too.
            route, overrides, err = self._select_request_route(
                {"model": self._model_name}, session_id=resolved, gateway_session_key=None,
                model_alias=self._model_name)
            if err is not None:
                raise RuntimeError(f"internal wake route conflict for session {resolved!r}")
            await self._run_agent(
                user_message=text, conversation_history=history, session_id=resolved,
                gateway_session_key=None, **overrides, route=route, requested_runtime={},
                route_source="global", session_history_delivery="1",
                notification_category=notification_category,
            )
            return
        raise RuntimeError(
            f"internal wake gave up for session {session_id} after {attempts} attempts: {last_err}")
    finally:
        if token is not None:
            _api_server._api_request_profile.reset(token)


def _run_idempotency_identity(self, request, body, gateway_session_key, idempotency_key) -> tuple:
    """``(principal scope, request fingerprint)`` for a keyed POST /v1/runs; ``("", "")`` unkeyed."""
    if not idempotency_key:
        return "", ""
    return self._run_idempotency_scope(request), hashlib.sha256(json.dumps(
        {"body": body, "gateway_session_key": gateway_session_key or ""},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _run_user_message(raw_input, _openai_error) -> tuple:
    """POST /v1/runs ``input`` -> ``(user_message, error_response_or_None)``."""
    if not raw_input:
        return None, _json_error(_openai_error, "Missing 'input' field", status=400)
    if isinstance(raw_input, str):
        user_message = raw_input
    else:
        user_message = raw_input[-1].get("content", "") if isinstance(raw_input, list) else ""
    if not user_message:
        return None, _json_error(_openai_error, "No user message found in input", status=400)
    return user_message, None


async def _admit_run_to_authority(
    self, launch, run_id, session_history_delivery, _declared_selected,
    idempotency_scope, idempotency_key, _openai_error, reservation: Optional[_KeyReservation] = None,
) -> Optional[web.Response]:
    """Admit a /v1/runs turn to the session authority FIFO -> ``None``, or the refusal."""
    from gateway.platforms.api_server_authority_runs import run_admission, store_error_response
    from gateway.session_api_turn import admit_api_turn_async
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError
    try:
        with self._profile_scope(launch.request_profile):
            launch.admission = await admit_api_turn_async(self, user_message=launch.user_message,
                conversation_history=launch.conversation_history, active_run_id=run_id,
                run_owner_scope=self._run_owners[run_id],
                turn_author=launch.turn_author,
                history_from_session=session_history_delivery,
                session_history_delivery='1' if session_history_delivery else '',
                bind_declared_conversation=_declared_selected, walked_from=launch.walked_from,
                **launch.agent_kwargs)
    except Exception as exc:  # health: allow BLE001 -- answered below; a committed admission is kept
        # A storage failure after the admission committed (a later media-release write) still
        # owns that run: forgetting it would let a keyed retry admit the same work twice.
        owned = None if isinstance(exc, RuntimeStoreError) else await asyncio.to_thread(run_admission, self, run_id)
        if owned is not None:
            launch.admission = (owned[0], SessionRef(owned[0].profile_id, owned[1]['target_session_id']), owned[1])
            return None
        # A refused admission owns no run: drop every reservation so an exact
        # retry is refused again instead of replaying a run nobody executes.
        _forget_run(
            self, run_id, self._run_streams, self._run_streams_created, self._run_approval_sessions,
            self._run_statuses, self._run_owners, self._run_idempotency_ids)
        if reservation is not None:
            released = _release_key_reservation(self, reservation, asyncio.get_running_loop())
            if released is not None:
                with suppress(Exception):  # a failed release stays owed (``_KeyReservation``)
                    await asyncio.shield(asyncio.wrap_future(released))
        elif idempotency_key:
            await self._run_idempotency_store.forget_async(idempotency_scope, idempotency_key)
        return store_error_response(_openai_error, exc)
    return None


async def _reserve_idempotency_key(self, entry: _KeyReservation, fingerprint, run_id, status, request):
    """Commit the ``(scope, key)`` reservation on the store's worker; the caller admits nothing
    until this returns. A handler cancelled meanwhile admits nothing either, so a reservation the
    worker still commits is released, as a refused admission's is (``_admit_run_to_authority``);
    requests that joined *entry* resume only after that release, so none of them replays it."""
    store = self._run_idempotency_store
    loop = asyncio.get_running_loop()
    future = store.submit(
        store.reserve, *entry.identity, fingerprint, run_id, status,
        owner_pid=self._run_owner_pid, owner_started=self._run_owner_started,
        retention_until=_room_retention_until(request), provisional=True)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        entry.releasing = True

        def _decide(created: bool) -> None:  # on the loop
            if created:
                _release_key_reservation(self, entry, loop)
            else:  # nothing was committed
                entry.releasing = False
                _settle_key_reservation(self, entry)

        def _committed(done) -> None:
            created = not done.cancelled() and done.exception() is None and done.result()[0] == "created"
            with suppress(RuntimeError):  # the loop is closed: the row stays provisional, owner-bound
                loop.call_soon_threadsafe(_decide, created)
        future.add_done_callback(_committed)
        raise


async def _reserve_run(self, request, run_id, initial_status, identity, gateway_session_key, _openai_error):
    """Reserve a keyed run's ``(scope, key)`` -> ``(None, reservation)`` once committed, else
    ``(answer, None)`` (replay, conflict or retry). Every non-admission drops the run's live state."""
    scope, key, fingerprint = identity
    reserved = dict(initial_status)  # the snapshot committed, whatever the live dict does meanwhile
    drop = (self._run_streams, self._run_streams_created, self._run_approval_sessions,
            self._run_statuses, self._run_owners)
    registry, loop = _key_reservations(self), asyncio.get_running_loop()
    for _ in range(_KEY_SETTLE_ROUNDS):
        if (scope, key) in registry:  # another request of this key is still being decided
            try:
                answer = await _keyed_answer(self, request, (scope, key), fingerprint, gateway_session_key, _openai_error)
            except BaseException:
                _forget_run(self, run_id, *drop)
                raise
            if answer is not None:
                _forget_run(self, run_id, *drop)
                return answer, None
        entry = registry[(scope, key)] = _KeyReservation((scope, key), loop)
        try:
            outcome, record = await _reserve_idempotency_key(self, entry, fingerprint, run_id, reserved, request)
        except BaseException:
            _forget_run(self, run_id, *drop)
            if not entry.releasing:  # the commit failed: nothing to release
                _settle_key_reservation(self, entry)
            raise
        if outcome == "created":
            break
        _settle_key_reservation(self, entry)  # this request holds nothing
        answer = await _keyed_answer(
            self, request, (scope, key), fingerprint, gateway_session_key, _openai_error, found=(outcome, record))
        if answer is not None:
            _forget_run(self, run_id, *drop)
            return answer, None
    else:
        _forget_run(self, run_id, *drop)
        return _key_in_flight(_openai_error), None
    self._run_idempotency_ids.add(run_id)
    # A status published while the reservation committed (the drain marker) was not yet
    # persistable; it is the run's newest status, so it follows the reserved row now.
    with _RUN_STATUS_LOCK:
        current = self._run_statuses.get(run_id)
        if current is not None and current != reserved:
            try:
                self._run_idempotency_store.enqueue_status(run_id, current)
            except Exception:
                logger.exception("[api_server] failed to persist idempotent run status %s", run_id)
    return None, entry


async def _handle_runs(self, request: web.Request, *, _api_server) -> web.Response:
    """POST /v1/runs — start an agent run, return run_id immediately."""
    _openai_error = _api_server._openai_error
    # Long-term memory scope header (see chat_completions for details).
    gateway_session_key, key_err = self._parse_session_key_header(request)
    if key_err is not None:
        return key_err
    try:
        body = await request.json()
    except Exception:
        return _json_error(_openai_error, "Invalid JSON", status=400)
    body, room_error = await self._normalize_room_dispatch(request, body)
    if room_error is not None:
        return room_error
    room_dispatch, room_execution_policy = (
        v if isinstance(v, dict) else None for v in (
            (body.get("hosted_room_dispatch"), body.get("_room_execution_policy"))
            if isinstance(body, dict) else (None, None)))
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if len(idempotency_key) > 255 or any(ord(ch) < 33 or ord(ch) > 126 for ch in idempotency_key):
        return _json_error(
            _openai_error, "Idempotency-Key must be 1-255 visible ASCII characters",
            code="invalid_idempotency_key", status=400)
    idempotency_scope, idempotency_fingerprint = _run_idempotency_identity(
        self, request, body, gateway_session_key, idempotency_key)
    raw_input = body.get("input")
    user_message, input_err = _run_user_message(raw_input, _openai_error)
    if input_err is not None:
        return input_err
    try:
        turn_author = _api_server._request_turn_author(body)
    except ValueError as exc:
        return _json_error(_openai_error, str(exc), code="invalid_author", status=400)
    conversation_history, instructions, stored_session_id, history_err, walked_from = await _resolve_chained_history(
        self, body, raw_input, _openai_error=_openai_error)
    if history_err is not None:
        return history_err
    previous_response_id = body.get("previous_response_id")
    session_id = body.get("session_id") or stored_session_id
    route = self._resolve_route(body.get("model"))
    agent_overrides = _api_server._request_agent_overrides(body, virtual_model=self._model_name)
    selection_error = self._request_route_conflict_error(
        session_id=session_id, gateway_session_key=gateway_session_key,
        requested_model=agent_overrides.get("requested_model"),
        requested_provider=agent_overrides.get("requested_provider"), route=route)
    if selection_error:
        return _json_error(_openai_error, selection_error, status=400)
    # A lost-acceptance replay must resolve even while the original run holds the last
    # concurrency slot; this read reserves nothing (the atomic reserve below closes the race).
    if idempotency_key:
        answer = await _keyed_answer(
            self, request, (idempotency_scope, idempotency_key), idempotency_fingerprint,
            gateway_session_key, _openai_error)
        if answer is not None:
            return answer
    # Enforce concurrency only for a genuinely new run.
    limited = self._concurrency_limited_response()
    if limited is not None:
        return limited
    run_id = f"run_{uuid.uuid4().hex}"
    self._run_owners[run_id] = self._run_idempotency_scope(request)
    # Same precedence as /v1/responses: body session_id > response chain > X-Hermes-Session-Key
    # conversation > run_id (which would otherwise re-key every affinity surface per run).
    # An explicit or chained session owns its routing key and is never rebound to the header.
    _declared_selected = not session_id and bool(gateway_session_key)
    selected_session_id = session_id or (
        await asyncio.to_thread(self._declared_conversation_session, gateway_session_key)
        if _declared_selected else None)
    # A client-addressed id from before a compression rotation must adopt the live tip (#98619):
    # history loads from it, the turn writes to it, and a detached delivery row persisted to it
    # is what the next same-id run consumes below.
    if selected_session_id:
        selected_session_id = await _resolve_live_session_id(self, str(selected_session_id))
    session_id = selected_session_id or run_id
    # History loads for the session the request actually selected — including one resolved from
    # a declared X-Hermes-Session-Key, whose persisted delivery rows must reach the next
    # same-key run's context (#98619).  previous_response_id continuations keep their
    # ResponseStore snapshot as history (they cannot consume a SessionDB delivery row and are
    # accordingly denied wake capability in _run_agent_sync); the fresh run_id fallback has
    # nothing persisted to load yet.  Wake authority is fixed here, before the load can
    # overwrite ``conversation_history``: a caller-supplied history is authoritative for this
    # turn, never consumes the SessionDB delivery row, and is denied on the same contract.
    session_history_delivery = not previous_response_id and not conversation_history
    if not conversation_history and selected_session_id and not previous_response_id:
        conversation_history = await self._conversation_history_for_session(str(selected_session_id))
    q = self._run_streams[run_id] = _RunStream()
    created_at = self._run_streams_created[run_id] = time.time()
    self._run_approval_sessions[run_id] = run_id  # approval session key (see _RunLaunch)
    initial_status = self._set_run_status(
        run_id, "queued", created_at=created_at, session_id=session_id, model=body.get("model", self._model_name))
    reservation = None
    if idempotency_key:
        replayed, reservation = await _reserve_run(
            self, request, run_id, initial_status, (idempotency_scope, idempotency_key, idempotency_fingerprint),
            gateway_session_key, _openai_error)
        if replayed is not None:
            return replayed
    # Joiners of this key's reservation resume once the request below is decided, whichever way.
    with _key_reservation_held(self, reservation):
        launch = _RunLaunch(
            self, run_id, q, session_id, gateway_session_key, _declared_selected, user_message,
            conversation_history, session_history_delivery,
            agent_kwargs=dict(
                ephemeral_system_prompt=instructions, session_id=session_id, gateway_session_key=gateway_session_key,
                route=route, room_dispatch=room_dispatch, room_execution_policy=room_execution_policy,
                **{k: agent_overrides.get(k) for k in ("requested_model", "requested_provider", "model_options")}),
            request_profile=_api_server._api_request_profile.get(),
            browser_control_principal=_api_server._api_request_browser_control_principal.get(),
            browser_control_transport_family=_api_server._api_request_browser_control_transport_family.get(),
            turn_author=turn_author, walked_from=walked_from if session_id == stored_session_id else None)
        # A canonical Bot Chat that a Desktop holds live is that Desktop's to run: executing here would
        # be a second writer beside its lease (#114959). The owner's mailbox takes the turn and its
        # receipt drives this run's status, so `peer run` keeps its run_id and `peer status` still works.
        # Under session authority the same call admits the turn to the Bot Chat's own FIFO, which is
        # why it is decided BEFORE ``admit_api_turn`` would bind the chat as an API conversation.
        admitted = (await self._admit_to_live_bot_chat(session_id, user_message, turn_author)
                    if selected_session_id else None)
        _retarget_to_delivered_chat(self, launch, admitted)
        if admitted is None and getattr(self.gateway_runner, 'session_authority', None) is not None:
            refused = await _admit_run_to_authority(
                self, launch, run_id, session_history_delivery, _declared_selected,
                idempotency_scope, idempotency_key, _openai_error, reservation=reservation)
            if refused is not None:
                return refused
        self._activate_admitted_request()
        if admitted is not None:
            task = self._active_run_tasks[run_id] = asyncio.create_task(
                _execute_run_via_live_owner(self, launch, *admitted, _api_server=_api_server))
        else:
            task = self._active_run_tasks[run_id] = asyncio.create_task(
                _execute_run(self, launch, _api_server=_api_server))
        with suppress(TypeError):
            self._background_tasks.add(task)  # tracked for shutdown drain
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
        _confirm_key_admission(self, reservation, run_id)
        return _accepted_response(run_id, "started", gateway_session_key, replayed=False)


def _retarget_to_delivered_chat(self, launch: _RunLaunch, admitted) -> None:
    """The owner moved an API-held "Bot Chat" title aside and the turn landed in the local chat it
    minted (MSG-21): the run record (and its idempotency replay) names that chat, not the old row."""
    delivered = admitted[1].get("session_id") if admitted is not None else None
    if delivered and delivered != launch.session_id:
        launch.session_id = delivered
        self._set_run_status(launch.run_id, "queued", session_id=delivered)


def _run_usage(agent) -> dict[str, int]:
    """Terminal ``usage`` payload from the agent's session counters; a missing or non-numeric
    counter (test doubles, agents without cache accounting) reads as ``0``."""
    usage = {}
    for key, attr in _USAGE_FIELDS:
        value = getattr(agent, attr, 0)
        usage[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    return usage


def _served_runtime(agent) -> dict[str, str]:
    """The ``{provider, model}`` pair that actually served the turn. After a ``fallback_providers``
    switch the agent keeps the fallback runtime until the NEXT turn restores the primary, so when
    ``run_conversation()`` returns these attributes name the served pair — the run record's
    ``model`` field only echoes the request (#102101). Non-string attributes read as ``""``."""
    pair = {}
    for key in ("provider", "model"):
        value = getattr(agent, key, "")
        pair[key] = value if isinstance(value, str) else ""
    return pair


def _run_agent_sync(self, run: _RunLaunch, agent, approval_notify, *, _api_server):
    """Executor-thread body of one run; returns ``(result, usage, served_runtime)``."""
    from gateway.session_context import clear_session_vars
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy, bind_room_execution_policy, reset_room_execution_policy)
    # No eager slash-worker pre-warm: slash.exec spawns one on demand (its error path already relies on that
    # respawn to recover from a dead worker). Each worker child runs its own MCP discovery (#61891), so
    # pre-warming one per session forks the full stdio MCP fleet — ~20 OS processes per retained session on
    # a config with a few stdio servers — even for sessions that never run a worker-routed command. Sessions
    # held by a live transport are never reaped, so with the desktop app open for days those fleets
    # accumulate until the OS refuses new process spawns.
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import reset_current_session_key, set_current_session_key
    session_id = run.session_id
    effective_task_id = session_id or run.run_id
    # (token, reset) pairs unwound in the finally block; bound only once each step succeeds.
    resets: list[tuple[Any, Callable]] = []
    with self._profile_scope(run.request_profile):
        try:
            # Contextvars, not process env: concurrent runs must not share identity.
            resets.append((set_current_session_key(run.approval_session_key), reset_current_session_key))
            # chat_id carries the raw session id like _run_agent() does; without it
            # tools.async_delegation sees no HERMES_SESSION_CHAT_ID and forces delegations sync.
            session_tokens = self._bind_api_server_session(
                chat_id=session_id or "", session_key=run.approval_session_key, session_id=session_id or "",
                profile=run.request_profile or "",
                browser_control_principal=run.browser_control_principal,
                browser_control_transport_family=run.browser_control_transport_family,
                # #98619 audited opt-in: the /v1/runs session id is wake-capable only when its
                # own continuation path reloads session history — an explicit body/chained
                # session id or a declared X-Hermes-Session-Key conversation (both load
                # SessionDB in _handle_runs), or the run_id fallback the client can post back
                # as body.session_id.  A previous_response_id continuation consumes its
                # ResponseStore snapshot instead and can never see a SessionDB delivery row,
                # so it stays default-denied until a merge contract exists for that chain;
                # likewise a caller-supplied conversation_history is authoritative for the
                # turn and never reads the delivery row, so it is denied the same way.
                session_history_delivery="1" if run.session_history_delivery else "")
            if session_tokens:
                resets.append((session_tokens, clear_session_vars))
            if run.agent_kwargs["room_dispatch"] is not None:
                policy = RoomExecutionPolicy.from_mapping(run.agent_kwargs["room_execution_policy"] or {})
                resets.append((bind_room_execution_policy(policy), reset_room_execution_policy))
            register_gateway_notify(run.approval_session_key, approval_notify)
            # /v1/runs owns its agent lifecycle (no TurnRunner): record process ownership
            # so stop/cancel reaps only the background processes this run created.
            _api_server._publish_turn_process_ownership(agent, effective_task_id)
            # Passed only when set: a human turn keeps today's call shape.
            author_kwargs = {"turn_author": run.turn_author} if run.turn_author is not None else {}
            r = agent.run_conversation(
                user_message=run.user_message, conversation_history=run.conversation_history,
                task_id=effective_task_id, **author_kwargs)
        finally:
            # Clear ownership now so a later stop can't reap work this run left running.
            _api_server._clear_turn_process_ownership(agent)
            self._memory_sessions.checkin(agent)
            # Declared-conversation binding, same precedence gate as _run_agent.
            if run.declared_selected:
                self._bind_declared_conversation(
                    getattr(agent, "session_id", None) or session_id, run.gateway_session_key)
            try:
                unregister_gateway_notify(run.approval_session_key)
            finally:
                for token, reset in resets:
                    with suppress(Exception):
                        reset(token)
        return r, _run_usage(agent), _served_runtime(agent)


def _make_approval_notify(self, run: _RunLaunch, *, _api_server) -> Callable[[dict[str, Any]], None]:
    """Approval-request bridge: redact, stamp the event envelope, park the run status, enqueue."""
    run_id, q, loop = run.run_id, run.queue, asyncio.get_running_loop()

    def _approval_notify(approval_data: dict[str, Any]) -> None:
        # Clients must never receive the raw flagged command (#48456): the shared builder redacts.
        event = _api_server._approval_request_event(run_id, approval_data)
        self._set_run_status(run_id, "waiting_for_approval", last_event="approval.request", approval=event)
        with suppress(Exception):
            loop.call_soon_threadsafe(q.put_nowait, event)

    return _approval_notify


async def _execute_run_via_live_owner(self, run: _RunLaunch, home, record: dict[str, Any], *, _api_server) -> None:
    """Drive a run whose turn a live Bot Chat owner is executing, from that owner's mailbox receipt.

    The receipt is the only truth about the turn: ``settled`` completes the run with the owner's
    reply, a failed receipt fails it with the owner's classified reason. ``/stop`` cannot reach the
    owner's turn — the mailbox has no recall once a record is claimed — so a stop ends this run
    as ``cancelled`` while the chat finishes on its own; the stop handler already reports that a
    run without an in-process agent is not interruptible here.
    """
    from gateway.platforms.api_server_bot_chat import await_live_delivery

    run_id = run.run_id
    delivery_id = record["delivery_id"]

    def _finish(status: str, **fields: Any) -> None:
        if run_id in self._shutdown_interrupted_run_ids:
            # Shutdown already published "interrupted"; the cancel that follows must not
            # rewrite it as a plain "cancelled" (same guard as the native _execute_run).
            status, fields = "interrupted", {"error": "Gateway shutdown interrupted the run."}
        self._set_run_status(run_id, status, **fields, last_event=f"run.{status}")
        event = _run_event(run_id, f"run.{status}", **fields)
        with suppress(Exception):
            _publish_after_durable(self, run_id, asyncio.get_running_loop(), lambda: run.put_event(event))

    try:
        self._set_run_status(run_id, "running", delivery_id=delivery_id)
        record = await await_live_delivery(
            self, home, record, None, should_stop=lambda: run_id in self._stopping_run_ids)
        if record["status"] in ("queued", "claimed"):
            _finish("cancelled", completed=False, partial=False, interrupted=True)
            return
        if record["status"] == "settled":
            _finish("completed", completed=True, partial=False, interrupted=False,
                    output=record.get("reply") or "", usage={})
        elif record["status"] == "cancelled":
            _finish("cancelled", completed=False, partial=False, interrupted=True)
        else:
            _finish("failed", completed=False, partial=False, interrupted=False,
                    error=record.get("error") or f"Bot Chat delivery {record['status']}",
                    **({"reason": record["reason"]} if record.get("reason") else {}))
    except asyncio.CancelledError:
        _finish("cancelled", completed=False, partial=False, interrupted=True)
        raise
    except Exception as exc:
        logger.exception("[api_server] run %s (live Bot Chat) failed", run_id)
        _finish("failed", completed=False, partial=False, interrupted=False, error=str(exc))
    finally:
        with suppress(Exception):  # sentinel: close the SSE stream, after the terminal event
            _publish_after_durable(self, run_id, asyncio.get_running_loop(), lambda: run.put_event(None))
        _retire_live_run(self, run_id)


async def _execute_run(self, run: _RunLaunch, *, _api_server) -> None:
    """Drive one admitted run, publish its terminal event/status, release live state."""
    _redact_api_error_text = _api_server._redact_api_error_text
    run_id, loop = run.run_id, asyncio.get_running_loop()
    _run_started_at = time.perf_counter()

    def _text_cb(delta: Optional[str]) -> None:
        if delta is None or run_id not in self._run_streams:
            return
        with suppress(Exception):
            loop.call_soon_threadsafe(run.put_event, _run_event(run_id, "message.delta", delta=delta))

    def _interim_cb(text: str, *, already_streamed: bool = False) -> None:
        # Mid-turn assistant commentary (Codex ``phase="commentary"``, text beside tool calls),
        # same ``message.interim`` contract as the TUI gateway; reasoning never reaches this
        # callback and the final answer still arrives via ``run.completed`` (#67580).
        if not isinstance(text, str) or not text.strip() or run_id not in self._run_streams:
            return
        with suppress(Exception):
            loop.call_soon_threadsafe(run.put_event, _run_event(
                run_id, "message.interim", text=text, already_streamed=bool(already_streamed)))

    def _finish(status: str, extra: Optional[dict] = None, **fields: Any) -> None:
        """Terminal status, then best-effort ``run.<status>`` event; key order is wire shape."""
        extra = extra or {}
        if run_id in self._shutdown_interrupted_run_ids:
            status = "interrupted"
            fields = {"error": "Gateway shutdown interrupted the run."}
            extra = {}
        self._set_run_status(run_id, status, **fields, last_event=f"run.{status}", **extra)
        event = _run_event(run_id, f"run.{status}", **fields, **extra)
        with suppress(Exception):
            # A worker can finish before its Future is wrapped, so awaiting it
            # need not yield to already-scheduled commentary/tool callbacks; and the
            # event follows the terminal status's commit (``_publish_after_durable``).
            _publish_after_durable(self, run_id, loop, lambda: run.put_event(event))

    try:
        # Shutdown landed between admission and the task's first tick: nothing to
        # interrupt yet, and starting a turn now would outlive the gateway.
        if run_id in self._shutdown_interrupted_run_ids:
            _finish("interrupted")
            return
        self._set_run_status(run_id, "running")
        if run_id in self._stopping_run_ids:
            _finish("cancelled")
            return
        if run.admission is not None:
            from gateway.session_api_turn import observe_api_controls, observe_api_turn
            tool_cb = self._make_run_event_callback(run_id, loop)

            def _control_cb(event_type, prompt):
                # Same authority prompt the WS viewer and GET pending_controls expose, on the
                # run's own stream so a streaming-only client learns it must respond.
                with suppress(Exception):
                    loop.call_soon_threadsafe(run.put_event, _run_event(run_id, event_type, **prompt))

            def _tool_start(call_id, name, args):
                from agent.display import build_tool_preview
                tool_cb("tool.started", name, build_tool_preview(name, args or {}) or "", args, tool_call_id=call_id)

            def _tool_complete(call_id, name, args, result):
                from agent.display import _detect_tool_failure
                is_error, _ = _detect_tool_failure(name, result)
                tool_cb("tool.completed", name, None, args, tool_call_id=call_id, is_error=bool(is_error))

            with observe_api_controls(run.admission, _control_cb):
                result, usage = await observe_api_turn(
                    run.admission, stream_delta_callback=_text_cb,
                    tool_start_callback=_tool_start, tool_complete_callback=_tool_complete)
            # The canonical turn has no agent object here; the served pair rides on the
            # committed usage/result runtime metadata when the turn recorded one.
            served_runtime = ((usage or {}).get("runtime") if isinstance(usage, dict) else None) or (
                result.get("runtime") if isinstance(result, dict) else None) or {}
        else:
            with self._profile_scope(run.request_profile):
                agent = self._create_agent(
                    stream_delta_callback=_text_cb, tool_progress_callback=self._make_run_event_callback(run_id, loop),
                    interim_assistant_callback=_interim_cb, **run.agent_kwargs)
            self._active_run_agents[run_id] = agent
            approval_notify = _make_approval_notify(self, run, _api_server=_api_server)
            result, usage, served_runtime = await _submit_api_worker(
                loop, lambda: _run_agent_sync(self, run, agent, approval_notify, _api_server=_api_server))
        # Publish request metrics (daily counters + latency) with each completed run (#52323).
        self._record_api_metrics(usage, time.perf_counter() - _run_started_at)
        if not isinstance(result, dict):
            result = {}
        # The committed outcome decides: a stop issued over WS by another viewer interrupts
        # this run just as much as one issued through this adapter's own /stop.
        status, fields = terminal_run_status(result)
        if status == "cancelled":
            _finish("cancelled", fields)
        elif result.get("failed"):
            # Non-retryable client errors (401/400) return failed=True rather than raising.
            _finish("failed", fields, error=_redact_api_error_text(result.get("error") or "agent run failed"))
        else:
            # ``runtime`` rides on both the pollable status and the run.completed event via _finish, in the
            # canonical shape every other api_server surface emits (route_source/requested, cleaned ids).
            requested = {k: run.agent_kwargs.get(f"requested_{k}") for k in ("provider", "model")}
            served_runtime = self._sanitize_runtime_metadata(
                runtime=served_runtime, requested_runtime=requested if any(requested.values()) else None,
                route_source=("model_routes" if run.agent_kwargs.get("route")
                              else "raw_request" if any(requested.values()) else "global"))
            _finish(status, fields, output=result.get("final_response", ""), usage=usage, runtime=served_runtime)
    except asyncio.CancelledError:
        _finish("cancelled")
        raise
    except _api_server._ProviderAuthResolutionError as exc:
        # Same controlled provider-auth message the _run_agent() endpoints give.
        logger.warning("Provider resolution failed for run=%s: %s", run_id, exc)
        _finish("failed", error=exc.user_text())
    except Exception as exc:
        if getattr(exc, "reason", None) == "admission_queued":
            # Still queued at a retiring owner: the next owner runs it once, and GET /v1/runs/{id}
            # reads that row. A run.failed here would invite a poller to resubmit it.
            self._set_run_status(run_id, "queued", last_event="run.queued")
            with suppress(Exception):
                loop.call_soon(run.put_event, _run_event(run_id, "run.queued", message=str(exc)))
            return
        logger.exception("[api_server] run %s failed", run_id)
        _finish("failed", error=_redact_api_error_text(exc))
    finally:
        # On cancellation (/stop) the executor thread may still block on an approval
        # Event; unregistering releases it. Idempotent on normal completion.
        _unregister_approval_notify(run.approval_session_key)
        with suppress(Exception):  # close after the queued events and the terminal commit
            _publish_after_durable(self, run_id, loop, lambda: run.put_event(None))
        _retire_live_run(self, run_id)


def _unregister_approval_notify(approval_session_key: Optional[str]) -> None:
    """Best-effort release of a run's approval waiter (no-op without a key)."""
    with suppress(Exception):
        from tools.approval import unregister_gateway_notify
        if approval_session_key:
            unregister_gateway_notify(approval_session_key)


def _release_run_owner_if_forgotten(self, run_id: str) -> None:
    """Drop the owner stamp only once nothing keyed by *run_id* survives: ownership must
    outlive every surface it protects (retired on different clocks); ownerless = fail-closed."""
    live = (self._run_statuses, self._active_run_agents, self._active_run_tasks, self._run_streams,
            self._run_approval_sessions)
    if not any(run_id in table for table in live):
        self._run_owners.pop(run_id, None)


async def _request_owns_run_async(self, request: web.Request, run_id: str) -> bool:
    """``_request_owns_run`` with the durable-ownership read on the run store's worker."""
    scope = self._run_idempotency_scope(request)
    owner = self._run_owners.get(run_id)
    if owner is not None:
        return owner == scope
    if await self._run_idempotency_store.owns_run_async(scope, run_id):
        return True
    from gateway.session_api_turn import owns_api_run
    return owns_api_run(self, run_id, scope)


def _request_owns_run(self, request: web.Request, run_id: str) -> bool:
    scope = self._run_idempotency_scope(request)
    owner = self._run_owners.get(run_id)
    if owner is not None:
        return owner == scope
    # No in-memory owner: only a durable record under the caller's scope admits it.
    # Under multiplex_profiles every profile holds a valid key, so ownerless = allow-all.
    # Run state that exists without an owner stamp is an unanswered authorization question, not a run anyone
    # may control — under gateway.multiplex_profiles every served profile holds a valid key, so admitting it
    # would make the boundary allow-all (#93689).
    if self._run_idempotency_store.owns_run(scope, run_id):
        return True
    from gateway.session_api_turn import owns_api_run
    return owns_api_run(self, run_id, scope)


async def _load_owned_run(self, request, *, _api_server, permission: Optional[str], active_fallback: bool):
    """Authenticate (*permission* -> room-grant aware; ``None`` -> API key only) and resolve
    ``(run_id, status, agent, task, error)``; *active_fallback* reports a live in-process run
    without pollable status as ``running`` instead of 404."""
    auth_err = self._check_run_auth(request, permission=permission) if permission else self._check_auth(request)
    if auth_err:
        return None, None, None, None, auth_err
    _openai_error = _api_server._openai_error
    run_id = request.match_info["run_id"]
    if not await _request_owns_run_async(self, request, run_id):
        return run_id, None, None, None, _run_not_found(_openai_error, run_id)
    status = await _durable_run_status_async(self, request, run_id)
    # Read after the store awaits: the live refs are whatever the run holds at answer time.
    agent = self._active_run_agents.get(run_id)
    task = self._active_run_tasks.get(run_id)
    if status is None and active_fallback and (agent is not None or task is not None):
        status = self._set_run_status(run_id, "running")
    if status is None:
        return run_id, None, agent, task, _run_not_found(_openai_error, run_id)
    return run_id, status, agent, task, None


async def _handle_get_run(self, request: web.Request, *, _api_server) -> web.Response:
    """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
    _, status, _, _, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission="status", active_fallback=True)
    return err or web.json_response(status)


async def _handle_run_events(self, request: web.Request, *, _api_server) -> web.StreamResponse:
    """GET /v1/runs/{run_id}/events — stream structured agent lifecycle events."""
    # Same door as GET /v1/runs/{id}: a room grant that may poll the run may stream it.
    auth_err = self._check_run_auth(request, permission="status")
    if auth_err:
        return auth_err
    run_id = request.match_info["run_id"]
    if not await _request_owns_run_async(self, request, run_id):
        return _run_not_found(_api_server._openai_error, run_id)
    # Allow subscribing slightly before the run is registered (race window).
    # Confirm the force-kill actually reaped the process before we clear its PID file / scoped locks.
    # SIGKILL can fail to take (e.g. an uninterruptible-sleep or zombie-reaping parent), and if we blindly
    # clear the metadata and start a fresh instance we end up with two live gateways fighting over the same
    # token — the duplicate-gateway failure in #19471.
    for _ in range(20):
        if run_id in self._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        return _run_not_found(_api_server._openai_error, run_id)
    stream = self._run_streams[run_id]
    raw_last_seq = request.headers.get("Last-Event-ID") or request.query.get("last_seq")
    try:
        last_seq = max(-1, int(str(raw_last_seq).strip())) if raw_last_seq is not None else -1
    except (TypeError, ValueError):
        last_seq = -1
    q, replay = stream.attach(last_seq)
    response = web.StreamResponse(status=200, headers=self._sse_headers(request))

    async def _write(data: bytes) -> None:
        try:
            async with asyncio.timeout(_RUN_STREAM_WRITE_TIMEOUT):
                await response.write(data)
        except TimeoutError:
            with suppress(Exception):
                response.force_close()
            raise

    async def _write_event(seq: int, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload["seq"] = seq
        await _write(_api_server._sse_frame(payload, id=seq))

    prepared = False
    try:
        await response.prepare(request)
        prepared = True
        # Flush the response head before waiting on the queue: aiohttp holds the headers
        # until the first body write, so a subscriber that connects before the run's first
        # event (e.g. before `approval.request`) sees no bytes and fetch()/EventSource never
        # resolve. A comment frame is ignored by every conforming SSE consumer.
        await _write(b": open\n\n")
        if replay and replay[0][0] > last_seq + 1:
            truncation = _run_event(
                run_id,
                "replay.truncated",
                oldest_retained_seq=replay[0][0],
            )
            if raw_last_seq is not None:
                truncation["requested_seq"] = last_seq
            await _write(_api_server._sse_frame(truncation))
        for seq, event in replay:
            if event is None:
                await _write(b": stream closed\n\n")
                return response
            await _write_event(seq, event)
        if stream.terminal and q.empty():
            await _write(b": stream closed\n\n")
            return response
        while True:
            try:
                seq, event = await asyncio.wait_for(
                    q.get(), timeout=_api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
            except TimeoutError:
                await _write(b": keepalive\n\n")
                continue
            if event is _RUN_STREAM_SUBSCRIBER_OVERFLOW:
                logger.debug("[api_server] closing slow SSE subscriber for run %s", run_id)
                break
            if event is None:  # run finished
                await _write(b": stream closed\n\n")
                break
            await _write_event(seq, event)
    except Exception as exc:
        if not prepared:
            raise
        logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
    finally:
        stream.detach(q)
        self._release_run_owner_if_forgotten(run_id)
    return response


def _mark_run_event(self, run_id: str, name: str, **fields: Any) -> None:
    """Record a control-plane event on the run status and (best effort) its SSE stream."""
    self._set_run_status(run_id, "running", last_event=name)
    q = self._run_streams.get(run_id)
    if q is not None:
        with suppress(Exception):
            q.put_nowait(_run_event(run_id, name, **fields))


_APPROVAL_CHOICE_ALIASES = {"approve": "once", "approved": "once", "allow": "once"}


async def _handle_run_approval(self, request: web.Request, *, _api_server) -> web.Response:
    """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
    _openai_error = _api_server._openai_error
    run_id, _, _, _, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission="approve", active_fallback=False)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return _json_error(_openai_error, "Invalid JSON", status=400)
    if getattr(self.gateway_runner, 'session_authority', None) is not None:
        if self._room_grant_token(request) and body.get('choice') not in {'once', 'deny'}:
            return _json_error(_openai_error, 'Room approvals require once or deny', status=400)
        return await _respond_authority_run(self, run_id, body, kind='approval', _api_server=_api_server)
    raw_choice = str(body.get("choice", "")).strip().lower()
    choice = _APPROVAL_CHOICE_ALIASES.get(raw_choice, raw_choice)
    room_scoped = bool(self._room_grant_token(request))
    raw_request_id = body.get("request_id")
    request_id = raw_request_id.strip() if isinstance(raw_request_id, str) else ""
    # Room grants may resolve exactly one request and never widen to session/always.
    allowed = {"once", "deny"} if room_scoped else {"once", "session", "always", "deny"}
    resolve_all = any(_api_server._coerce_request_bool(body.get(k), default=False) for k in ("all", "resolve_all"))
    approval_session_key = self._run_approval_sessions.get(run_id)
    for failed, message, code, status in (
        (raw_request_id is not None and (not request_id or len(request_id) > 256),
         "Approval request_id is invalid.", "invalid_approval_request", 400),
        (choice not in allowed,
         "Invalid approval choice; expected one of: " + ", ".join(sorted(allowed)),
         "invalid_approval_choice", 400),
        (room_scoped and resolve_all,
         "Room approvals can resolve only one exact request", "invalid_approval_scope", 400),
        (room_scoped and not request_id,
         "Room approvals require the exact request_id.", "approval_request_required", 400),
        (not approval_session_key,
         f"Run has no active approval session: {run_id}", "approval_not_active", 409)):
        if failed:
            return _json_error(_openai_error, message, code=code, status=status)
    try:
        from tools.approval import resolve_gateway_approval
        resolved = resolve_gateway_approval(
            approval_session_key, choice, resolve_all=resolve_all, request_id=request_id or None)
    except Exception as exc:
        logger.exception("[api_server] approval resolution failed for run %s", run_id)
        return _json_error(_openai_error, str(exc), status=500)
    if resolved <= 0:
        return _json_error(
            _openai_error, f"Run has no pending approval: {run_id}", code="approval_not_pending", status=409)
    request_id_field = {"request_id": request_id} if request_id else {}
    _mark_run_event(self, run_id, "approval.responded", choice=choice, **request_id_field, resolved=resolved)
    return web.json_response({
        "object": "hermes.run.approval_response", "run_id": run_id, "choice": choice, **request_id_field,
        "resolved": resolved})


async def _respond_authority_run(self, run_id, body, *, kind, _api_server):
    from gateway.platforms.api_server_authority_runs import respond_run, store_error_response
    from hermes_state_runtime import RuntimeStoreError
    try:
        result = await respond_run(self, run_id, body, kind=kind)
        return web.json_response({'run_id': run_id, **result})
    except RuntimeStoreError as exc:
        return store_error_response(_api_server._openai_error, exc)


async def _handle_run_clarify(self, request, *, _api_server):
    run_id, _, _, _, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission="approve", active_fallback=False)
    if err is not None:
        return err
    body, err = await self._read_json_body(request)
    if err is not None:
        return err
    return await _respond_authority_run(self, run_id, body, kind='clarify', _api_server=_api_server)


def _steer_refused(_openai_error, run_id):
    return _json_error(_openai_error, f"Run is not currently accepting steer input: {run_id}",
                       code="run_not_accepting_steer", status=409)


async def _handle_steer_run(self, request: web.Request, *, _api_server) -> web.Response:
    """POST /v1/runs/{run_id}/steer — inject guidance into a running agent."""
    _openai_error = _api_server._openai_error
    run_id, status, agent, _, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission=None, active_fallback=False)
    if err is not None:
        return err
    owned = None
    if getattr(self.gateway_runner, 'session_authority', None) is not None:
        from gateway.platforms.api_server_authority_runs import run_admission
        owned = run_admission(self, run_id)
        owned = owned if run_id not in self._stopping_run_ids else None
        agent = None
    # /stop keeps agent refs during cooperative shutdown, so the status gate (not the
    # agent ref) is what rejects stop-then-steer.
    if status.get("status") != "running" or (owned is None and not hasattr(agent, "steer")):
        return _steer_refused(_openai_error, run_id)
    body, err = await self._read_json_body(request)
    if err:
        return err
    raw_text = body.get("input") or body.get("message") or body.get("text") or ""
    steer_text = _api_server._normalize_chat_content(raw_text).strip()
    if not steer_text:
        return _json_error(
            _openai_error, "Missing non-empty steer text; expected 'input', 'message', or 'text'.",
            code="invalid_steer_input", status=400)
    if owned is not None:
        from gateway.session_contract import SessionRef
        from hermes_state_runtime import RuntimeStoreError
        authority, row = owned
        sid, latched = row['target_session_id'], False
        try:
            with authority.sessions[sid].event_stream.lock:
                authority.check_approval_generation(sid, row['generation'])
                # Before the claimed turn adopts its agent the steer is latched for that agent.
                agent, latched = authority.steer_agent(SessionRef(authority.profile_id, sid), row['generation'], steer_text)
        except (KeyError, RuntimeStoreError):
            agent = None
        if not latched and not hasattr(agent, "steer"):
            return _steer_refused(_openai_error, run_id)
        if latched:
            _mark_run_event(self, run_id, "run.steered", accepted=True)
            return web.json_response({"object": "hermes.run.steer", "run_id": run_id, "accepted": True})
    try:
        accepted = bool(agent.steer(steer_text))
    except Exception as exc:
        logger.exception("[api_server] steer failed for run %s", run_id)
        return _json_error(_openai_error, _api_server._redact_api_error_text(exc), code="steer_failed", status=500)
    if not accepted:
        return _json_error(
            _openai_error, f"Run did not accept steer text: {run_id}", code="steer_not_accepted", status=409)
    _mark_run_event(self, run_id, "run.steered", accepted=True)
    return web.json_response({"object": "hermes.run.steer", "run_id": run_id, "accepted": True})


async def _handle_stop_run(self, request: web.Request, *, _api_server) -> web.Response:
    """POST /v1/runs/{run_id}/stop — interrupt a running agent."""
    _openai_error = _api_server._openai_error
    run_id, status, agent, task, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission="stop", active_fallback=True)
    if err is not None:
        return err
    # A run whose turn a live Bot Chat owns (``_execute_run_via_live_owner``) is not an API
    # admission: its stop is the legacy in-process task cancel below, never ``stop_run``.
    if (getattr(self.gateway_runner, 'session_authority', None) is not None
            and not (task is not None and status.get("delivery_id"))):
        from gateway.platforms.api_server_authority_runs import stop_run, store_error_response
        from hermes_state_runtime import RuntimeStoreError
        try:
            return web.json_response(await stop_run(self, run_id))
        except RuntimeStoreError as exc:
            return store_error_response(_openai_error, exc)
    if status.get("status") in TERMINAL_STATUSES:
        return web.json_response(status)
    if agent is None and task is None:
        return _json_error(
            _openai_error, f"Run is not active in this gateway process: {run_id}",
            code="run_not_active", status=409)
    self._set_run_status(run_id, "stopping", last_event="run.stopping")
    self._stopping_run_ids.add(run_id)
    if agent is not None:
        with suppress(Exception):
            _api_server.request_hard_interrupt(agent, "Stop requested via API")
        # Reap only this run's background processes (epoch-gated inside, so a concurrent
        # run on the same session_id keeps its own); no-op if the run already finished.
        _api_server._reap_disconnected_agent_processes(agent, source="api_server_run_stop")
    return web.json_response({"run_id": run_id, "status": "stopping"})


async def _handle_resolve_unknown_run(
    self, request: web.Request, *, _api_server
) -> web.Response:
    """POST /v1/runs/{run_id}/resolve-unknown — acknowledge one lost owner epoch."""
    run_id, _, _, _, err = await _load_owned_run(
        self, request, _api_server=_api_server, permission="stop", active_fallback=False)
    if err is not None:
        return err
    try:
        body = await request.json()
    except (ValueError, LookupError):  # bad JSON/UTF-8 (ValueError) or unknown charset: no body
        body = None
    from gateway.platforms.api_server_authority_runs import resolve_unknown_run, store_error_response
    from hermes_state_runtime import RuntimeStoreError
    try:
        result = await resolve_unknown_run(self, run_id, body)
        return web.json_response({'run_id': run_id, **result})
    except RuntimeStoreError as exc:
        return store_error_response(_api_server._openai_error, exc)


async def _sweep_orphaned_runs(self) -> None:
    """Periodically expire transport buffers and terminal status records."""
    while True:
        await asyncio.sleep(60)
        self._sweep_orphaned_runs_once(time.time())


def _durability_of(self, run_id: str):
    """The run's outstanding status commits as a future (None: nothing outstanding / unkeyed)."""
    store = getattr(self, "_run_idempotency_store", None)
    durable_after = getattr(store, "durable_after", None)
    if durable_after is None or run_id not in getattr(self, "_run_idempotency_ids", ()):
        return None
    return durable_after(run_id)


async def _terminal_status_durable(self, run_id: str) -> None:
    """Wait (off-loop) until the run's published statuses are committed; a failed write keeps its
    established disposition (logged, the live status still answers), so its error is not raised."""
    durable = _durability_of(self, run_id)
    if durable is not None:
        with suppress(Exception):
            await asyncio.shield(asyncio.wrap_future(durable))


def _publish_after_durable(self, run_id: str, loop, publish: Callable[[], None]) -> None:
    """Run *publish* on the loop once every status enqueued for the run is committed, or failed
    and was logged (the established disposition): a terminal event, or the stream close after it,
    is never observable before the terminal status it reports is durable. The commit stays off the
    loop, and the publication does not depend on the producing task surviving cancellation."""
    durable = _durability_of(self, run_id)
    if durable is None:  # queued behind any publication an earlier commit already released
        loop.call_soon(publish)
        return

    def _published(_done) -> None:
        with suppress(RuntimeError):  # the loop is closed: nobody is left to read the stream
            loop.call_soon_threadsafe(publish)
    durable.add_done_callback(_published)


def _sweep_orphaned_runs_once(self, now: Optional[float] = None) -> None:
    """Expire old SSE buffers without treating transport age as run age."""
    if now is None:
        now = time.time()
    for run_id, created_at in list(self._run_streams_created.items()):
        stream = self._run_streams.get(run_id)
        if now - created_at <= self._RUN_STREAM_TTL or (stream is not None and stream.subscribers):
            continue
        logger.debug("[api_server] sweeping expired run transport %s", run_id)
        task = self._active_run_tasks.get(run_id)
        # Transport TTL bounds buffering; live control state survives until the task returns.
        _drop_run_transport(self, run_id)
        if task is None or task.done():
            _unregister_approval_notify(self._run_approval_sessions.get(run_id))
            _retire_live_run(self, run_id)
    for run_id, status in list(self._run_statuses.items()):
        if (status.get("status") in {"completed", "failed", "cancelled"}
                and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL):
            _forget_run(self, run_id, self._run_statuses, self._run_idempotency_ids)
