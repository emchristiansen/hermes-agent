"""Persistent CDP supervisor for browser dialog + frame detection.

One ``CDPSupervisor`` per Hermes ``task_id`` with a reachable CDP endpoint: one
persistent WebSocket, ``Page`` / ``Runtime`` / ``Target`` events on every attached
session (top page + auto-attached OOPIF / worker targets), pending dialogs + frame
tree exposed via a thread-safe snapshot. Not in the tool schema — output reaches the
agent via ``browser_dialog``. Dialog capture and frame tracking
are mixins (``browser_supervisor_dialogs`` / ``browser_supervisor_frames``).
Design spec: ``website/docs/developer-guide/browser-supervisor.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from tools.browser_supervisor_dialogs import (
    DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S, RECENT_DIALOGS_MAX, _VALID_POLICIES, DialogRecord,
    DialogSupervisionMixin, PendingDialog,
)
from tools.browser_supervisor_frames import FrameInfo, FrameTrackingMixin
from tools.browser_supervisor_worlds import WorldTrackingMixin, world_params

# ``websockets`` costs ~22 ms at import and is only needed once a supervisor connects.
if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from tools.browser_supervisor_capture import CapturedCDP

logger = logging.getLogger(__name__)

# Browserbase can transiently drop a CDP socket while a short-lived client
# reconnects.  A locally owned browser endpoint, however, is gone for good
# once its process exits; leave enough room for the former without leaking a
# supervisor thread and warnings forever for the latter.
MAX_POST_ATTACH_RECONNECT_FAILURES = 5


def _redact_cdp_error_text(exc: object) -> str:
    """Redact CDP endpoint credentials from an exception's (or URL's) string form.
    ``websockets`` bakes the raw URL (``?token=`` / ``user:pass@``) into its exception
    messages, so every egress point turning one into log/re-raise text MUST route through
    here; falls back to a fixed sentinel if redaction itself raises (err toward masking)."""
    try:
        from agent.redact import redact_cdp_url

        return redact_cdp_url(str(exc))
    except Exception:
        return "<error redacted>"


class _LoopUnavailable(RuntimeError):
    """The supervisor loop refused new work (closed / shutting down)."""


class _NotConnected(RuntimeError):
    """No supervisor WebSocket: the command was not sent."""


def _schedule(coro, loop, *, timeout: float):
    """Run ``coro`` on the supervisor loop from a sync caller and wait for its result."""
    from agent.async_utils import safe_schedule_threadsafe

    fut = safe_schedule_threadsafe(coro, loop)
    if fut is None:
        raise _LoopUnavailable("Browser supervisor loop unavailable")
    return fut.result(timeout=timeout)


# How long ``evaluate_runtime`` waits for a page's reply by default. A vault fill script carries a page-clock
# budget strictly shorter than this (agent.vault_login_classifier.FILL_PAGE_BUDGET_S).
EVAL_REPLY_WAIT_S = 10.0


def _fail(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


def _unsent(error: str) -> dict[str, Any]:
    """A failure before the command reached the browser."""
    return {**_fail(error), "sent": False}


def _err(exc: BaseException) -> dict[str, Any]:
    out = _fail(f"{type(exc).__name__}: {exc}")
    if isinstance(exc, TimeoutError):  # typed: the page did not answer in time (asyncio/futures timeouts alias it)
        out["timed_out"] = True
    return out


class SupervisorTargetError(RuntimeError):
    """An exact-target supervisor's ``attach_target_id`` is not an open page target: it never falls back
    to another tab or creates one."""


class SupervisorScopeConflict(RuntimeError):
    """``get_or_start`` found a supervisor for the task bound to a different endpoint, target or options
    where either side is target-scoped: it refuses rather than stop or replace that supervisor."""


@dataclass(frozen=True)
class SupervisorSnapshot:
    """Read-only snapshot of supervisor state for tool handlers."""

    pending_dialogs: tuple[PendingDialog, ...]
    recent_dialogs: tuple[DialogRecord, ...]
    frame_tree: dict[str, Any]
    active: bool  # False if supervisor is detached/stopped
    cdp_url: str
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in tool output."""
        out: dict[str, Any] = {"pending_dialogs": [d.to_dict() for d in self.pending_dialogs], "frame_tree": self.frame_tree}
        if self.recent_dialogs:
            out["recent_dialogs"] = [d.to_dict() for d in self.recent_dialogs]
        return out


class CDPSupervisor(DialogSupervisionMixin, FrameTrackingMixin, WorldTrackingMixin):
    """One supervisor per (task_id, cdp_url) pair. ``start()`` spawns a daemon thread
    running its own asyncio loop, connects, attaches to the first page target, enables
    domains and auto-attach. ``snapshot()`` / ``respond_to_dialog()`` / ``evaluate_runtime()``
    are sync, thread-safe bridges onto that loop; all CDP I/O lives on the loop."""

    def __init__(self, task_id: str, cdp_url: str, *, dialog_policy: str = DEFAULT_DIALOG_POLICY,
                 dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S, attach_target_id: Optional[str] = None,
                 dialog_bridge: bool = True) -> None:
        """``attach_target_id`` scopes the supervisor to that one CDP page target (a shared browser whose
        other tabs belong to someone else): it attaches only that target, fails when it is not an open
        page, and focuses nothing else. ``dialog_bridge=False`` installs no dialog bridge (no init script,
        Fetch interception or ``alert``/``confirm``/``prompt`` override) on any tab and makes no automatic
        dialog choice: native dialogs stay with whoever is at the browser. The defaults keep the managed
        behaviour (first page target, bridge on every attached session)."""
        if dialog_policy not in _VALID_POLICIES:
            raise ValueError(f"Invalid dialog_policy {dialog_policy!r}; must be one of {sorted(_VALID_POLICIES)}")
        if attach_target_id is not None and (not isinstance(attach_target_id, str) or not attach_target_id.strip()):
            raise ValueError("attach_target_id must be a non-empty CDP target id")
        self.task_id = task_id
        self.cdp_url = cdp_url
        self.dialog_policy = dialog_policy
        self.dialog_timeout_s = float(dialog_timeout_s)
        self.attach_target_id = attach_target_id
        self.dialog_bridge = bool(dialog_bridge)

        # State protected by ``_state_lock`` for cross-thread reads.
        self._state_lock = threading.Lock()
        self._pending_dialogs: dict[str, PendingDialog] = {}
        # Native dialogs open per page session, recorded passively (``open_dialog``), bridge or not.
        self._open_native_dialogs: dict[str, dict[str, Any]] = {}
        self._recent_dialogs: list[DialogRecord] = []
        # (session id, context id) -> uniqueId of the vault's isolated worlds (``create_isolated_world``).
        self._world_unique_ids: dict[tuple[Optional[str], Any], str] = {}
        self._frames: dict[str, FrameInfo] = {}
        self._active = False
        # Supervisor loop machinery — populated in start().
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._stop_requested = False
        # CDP call tracking (runs on supervisor loop only).
        self._next_call_id = 1
        self._pending_calls: dict[int, asyncio.Future] = {}
        self._ws: Optional[ClientConnection] = None
        self._page_session_id: Optional[str] = None
        # The tab the task's browser_exec daemon last drove (``set_driven_target``); ``focus_page`` prefers it.
        self._driven_target_id: Optional[str] = None
        # Dialog auto-dismiss watchdog handles (per dialog id) + id generator.
        self._dialog_watchdogs: dict[str, asyncio.TimerHandle] = {}
        self._dialog_seq = 0

    # ── Public sync API ──────────────────────────────────────────────────────

    def start(self, timeout: float = 15.0) -> None:
        """Launch the background loop and block until attachment completes; raises what attach failed with (redacted)."""
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._start_error, self._stop_requested = None, False
        self._thread = threading.Thread(target=self._thread_main, name=f"cdp-supervisor-{self.task_id}", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(f"CDP supervisor did not attach within {timeout}s "
                               f"(cdp_url={_redact_cdp_error_text(self.cdp_url)[:80]}...)")
        if (err := self._start_error) is not None:
            self.stop()
            # ``err`` is a raw ``websockets`` exception embedding the full cdp_url (token /
            # userinfo): re-raise redacted, ``from None`` so the traceback chain leaks nothing.
            detail = _redact_cdp_error_text(err) or type(err).__name__  # a bare TimeoutError has no text
            raise RuntimeError(f"CDP supervisor failed to start: {detail}") from None

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the supervisor task and join the thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and loop.is_running():
            # Close the WebSocket from inside the loop so ``async for raw in self._ws``
            # returns cleanly, ``_run`` hits its ``finally``, THEN the thread exits.
            with contextlib.suppress(Exception):  # loop already shutting down / close timed out
                _schedule(self._close_ws(), loop, timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._set_active(False)

    def _set_active(self, value: bool) -> None:
        with self._state_lock:
            self._active = value

    def snapshot(self) -> SupervisorSnapshot:
        """Return an immutable snapshot of current state."""
        with self._state_lock:
            return SupervisorSnapshot(
                pending_dialogs=tuple(self._pending_dialogs.values()),
                recent_dialogs=tuple(self._recent_dialogs[-RECENT_DIALOGS_MAX:]),
                frame_tree=self._build_frame_tree_locked(),
                active=self._active, cdp_url=self.cdp_url, task_id=self.task_id,
            )

    def respond_to_dialog(self, action: str, *, prompt_text: Optional[str] = None,
                          dialog_id: Optional[str] = None, timeout: float = 10.0) -> dict[str, Any]:
        """Accept/dismiss a pending dialog (sync bridge onto the supervisor loop). Returns
        ``{"ok": True, "dialog"}`` or ``{"ok": False, "error"}`` for recoverable errors."""
        if action not in {"accept", "dismiss"}:
            return _fail(f"action must be 'accept' or 'dismiss', got {action!r}")
        with self._state_lock:
            if not self._active:
                return _fail("supervisor is not active")
            pending = list(self._pending_dialogs.values())
            if not pending:
                return _fail("no dialog is currently open")
            if dialog_id:
                dialog = self._pending_dialogs.get(dialog_id)
                if dialog is None:
                    return _fail(f"dialog_id {dialog_id!r} not found (known: {sorted(self._pending_dialogs)})")
            elif len(pending) > 1:
                return _fail(f"{len(pending)} pending dialogs; specify dialog_id. Candidates: {[d.id for d in pending]}")
            else:
                dialog = pending[0]
        loop = self._loop
        if loop is None:
            return _fail("supervisor loop is not running")
        try:
            coro = self._handle_dialog_cdp(dialog, accept=(action == "accept"), prompt_text=prompt_text or "")
            _schedule(coro, loop, timeout=timeout)
        except _LoopUnavailable as e:
            return _fail(str(e))
        except Exception as e:
            return _err(e)
        return {"ok": True, "dialog": dialog.to_dict()}

    def evaluate_runtime(self, expression: str, *, return_by_value: bool = True,
                         await_promise: bool = True, timeout: float = EVAL_REPLY_WAIT_S,
                         world: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Evaluate ``expression`` in the page's Runtime context over the live WS, or with ``world`` (from
        ``create_isolated_world``) in that isolated world, on the session it was created on.
        Returns ``{"ok": True, "result", "result_type"}`` or ``{"ok": False, "error"}``; a failure known to
        have happened before the command was sent carries ``"sent": False``.
        ``return_by_value=True`` JSON-serializes the result (DevTools-console
        semantics); non-serializable objects come back as a description string."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return _unsent("supervisor loop is not running")
        with self._state_lock:
            active, session_id = self._active, self._page_session_id
        if world is not None:
            session_id = world.get("session_id")
        if not active:
            return _unsent("supervisor is not active")
        if not session_id:
            return _unsent("supervisor has no attached page session")

        def _run_eval(by_value: bool) -> dict[str, Any]:
            # userGesture: clipboard / fullscreen APIs need user activation.
            params = {"expression": expression, "returnByValue": by_value,
                      "awaitPromise": await_promise, "userGesture": True,
                      **(world_params(world) if world is not None else {})}
            coro = self._cdp("Runtime.evaluate", params, session_id=session_id, timeout=timeout)
            return _schedule(coro, loop, timeout=timeout + 1)

        try:
            response = _run_eval(return_by_value)
        except (_LoopUnavailable, _NotConnected) as exc:
            return _unsent(str(exc))
        except Exception as exc:
            # Deep-serializing live DOM nodes / NodeLists / Window can blow past
            # CDP's recursion guard (``Object reference chain is too long``).
            # Retry once with returnByValue=False so Chrome returns the description.
            if not (return_by_value and "reference chain is too long" in str(exc).lower()):
                return _err(exc)
            try:
                response = _run_eval(False)
            except Exception as exc2:
                return _err(exc2)

        # Response: {"result": {"result": {"type", "value", ...}, "exceptionDetails"?}}
        result_payload = response.get("result", {}) if isinstance(response, dict) else {}
        exception_details = result_payload.get("exceptionDetails")
        if exception_details:
            exc_text = exception_details.get("text") or "JavaScript exception"
            description = (exception_details.get("exception") or {}).get("description")
            return _fail(f"{exc_text}: {description}" if description else exc_text)

        result_obj = result_payload.get("result", {})
        result_type = result_obj.get("type", "undefined")
        if "value" in result_obj:
            value = result_obj["value"]
        elif result_type == "undefined":
            value = None
        else:
            # Non-serializable (functions, DOM nodes…) — give the model the browser's description.
            value = result_obj.get("description") or result_obj.get("unserializableValue")
        return {"ok": True, "result": value, "result_type": result_type}

    def create_isolated_world(self, name: str, *, timeout: float = 2.0) -> dict[str, Any]:
        """Create isolated world ``name`` in the main frame of the attached page session
        (``tools.browser_supervisor_worlds``). Returns ``{"ok": True, "world"}`` (pass ``world`` to
        ``evaluate_runtime``) or ``{"ok": False, "error"}``."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return _fail("supervisor loop is not running")
        with self._state_lock:
            session_id = self._page_session_id
        if not session_id:
            return _fail("supervisor has no attached page session")
        try:
            world = _schedule(self._create_world(session_id, name, timeout=timeout), loop, timeout=timeout * 2 + 1)
        except (RuntimeError, OSError, TimeoutError, KeyError, TypeError) as exc:  # CDP error, socket, no reply, odd reply
            return _err(exc)
        return {"ok": True, "world": world}

    def set_driven_target(self, target_id: Optional[str]) -> None:
        """Record the CDP target the task's browser_exec session drives (None: unknown)."""
        with self._state_lock:
            self._driven_target_id = target_id

    def focus_page(self, origin: str, *, accept: Optional[str] = None, timeout: float = 10.0) -> dict[str, Any]:
        """Re-attach the supervisor's page session to an open page target on ``origin``
        (``scheme://host[:port]``). The initial attach picks the FIRST page target, but tools
        that open their own tabs (browser_exec) put the login form somewhere else. With
        ``accept`` (a JS expression) the first same-origin tab where it evaluates truthy wins,
        so a login and a checkout tab on one site resolve to the right one. When the tab browser_exec
        drives (``set_driven_target``) is on ``origin`` it is the only candidate, probe or not: on a shared
        CDP browser the first same-origin tab can be another session's or a stale one. Returns
        ``{"ok": True, "url"}`` or ``{"ok": False, "error"}``; on failure the previous session stays."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return _fail("supervisor loop is not running")

        async def _attach(target_id: str) -> str:
            attach = await self._cdp("Target.attachToTarget", {"targetId": target_id, "flatten": True}, timeout=timeout)
            sid = attach["result"]["sessionId"]
            await self._enable_page_domains(sid, timeout=timeout)
            await self._install_dialog_bridge(sid)
            return sid

        async def _focus() -> dict[str, Any]:
            from agent.vault_store import normalize_origin
            targets = (await self._cdp("Target.getTargets", timeout=timeout)).get("result", {}).get("targetInfos", [])
            candidates = []
            for t in targets:
                url = str(t.get("url") or "")
                try:
                    # origin="" = any http(s) page (used to FIND the login tab before its origin is known)
                    if t.get("type") == "page" and url.startswith(("http://", "https://")) \
                            and (not origin or normalize_origin(url) == origin):
                        candidates.append((t["targetId"], url))
                except Exception:
                    continue
            with self._state_lock:
                driven = self._driven_target_id
            probe = accept
            if own := [c for c in candidates if c[0] == driven]:
                # Focus our own tab even before its form renders (the fill then finds no field) rather
                # than let the probe fall through to another session's tab on the same origin.
                candidates, probe = own, None
            for target_id, url in candidates:
                sid = await _attach(target_id)
                if probe:
                    probed = await self._cdp("Runtime.evaluate", {"expression": probe, "returnByValue": True},
                                             session_id=sid, timeout=timeout)
                    if not probed.get("result", {}).get("result", {}).get("value"):
                        await self._cdp("Target.detachFromTarget", {"sessionId": sid}, timeout=timeout)
                        continue
                with self._state_lock:
                    self._page_session_id = sid
                return {"ok": True, "url": url}
            return _fail(f"no open page on {origin or 'any site'}" + (" with the expected form" if accept and candidates else ""))

        # An exact-target supervisor never looks at, or attaches to, another tab.
        focus = _focus() if self.attach_target_id is None else self._focus_exact_target(origin, accept, timeout=timeout)
        try:
            return _schedule(focus, loop, timeout=timeout + 1)
        except Exception as exc:
            return _err(exc)

    async def _focus_exact_target(self, origin: str, accept: Optional[str], *, timeout: float) -> dict[str, Any]:
        """``focus_page`` for an exact-target supervisor: the one attached session, no other tab and no new
        attach. Fails when the target is gone, is not on ``origin`` or ``accept`` is falsy there."""
        from agent.vault_store import VaultError, normalize_origin
        sid = self._page_session_id
        if not sid:
            return _fail(f"target {self.attach_target_id} is not attached")
        info = await self._cdp("Target.getTargetInfo", {"targetId": self.attach_target_id}, timeout=timeout)
        target = info.get("result", {}).get("targetInfo") or {}
        url = str(target.get("url") or "")
        if target.get("type") != "page" or not target.get("attached"):
            return _fail(f"target {self.attach_target_id} is not an attached page")
        try:
            on_origin = not origin or (url.startswith(("http://", "https://")) and normalize_origin(url) == origin)
        except VaultError:  # an http(s) URL without a host
            on_origin = False
        if not on_origin:
            return _fail(f"target {self.attach_target_id} is not on {origin}")
        if accept:
            probe = await self._cdp("Runtime.evaluate", {"expression": accept, "returnByValue": True},
                                    session_id=sid, timeout=timeout)
            if not probe.get("result", {}).get("result", {}).get("value"):
                return _fail(f"target {self.attach_target_id} on {origin or 'its site'} lacks the expected form")
        return {"ok": True, "url": url}

    # ── Supervisor loop internals ────────────────────────────────────────────

    def _thread_main(self) -> None:
        """Entry point for the supervisor's dedicated thread."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except BaseException as e:
            if not self._fail_start(e):
                logger.warning("CDP supervisor %s crashed: %s", self.task_id, e)
        finally:
            # Cancel + flush remaining tasks before closing the loop to avoid
            # "Task was destroyed but it is pending" warnings.
            with contextlib.suppress(Exception):
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with contextlib.suppress(Exception):
                loop.close()
            self._set_active(False)

    def _fail_start(self, e: BaseException) -> bool:
        """Propagate ``e`` to ``start()`` if we never got ready; True if it was consumed."""
        if self._ready_event.is_set():
            return False
        self._start_error = e
        self._ready_event.set()
        return True

    async def _close_ws(self) -> None:
        """Detach and close the current WebSocket, swallowing close errors."""
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    def _reconnect_budget_spent(self, failures: int, e: BaseException) -> bool:
        """True once ``failures`` consecutive post-attach reconnects failed: log ONE final line
        and drop this supervisor from the registry so a dead endpoint (its Chrome exited with
        the task) leaves neither a retrying thread nor a stale registry entry behind. A later
        browser call for the task starts a fresh supervisor via ``get_or_start``."""
        if failures < MAX_POST_ATTACH_RECONNECT_FAILURES:
            return False
        logger.warning("CDP supervisor %s: stopped after %s failed reconnect attempts: %s",
                       self.task_id, failures, _redact_cdp_error_text(e))
        if SUPERVISOR_REGISTRY.get(self.task_id) is self:
            SUPERVISOR_REGISTRY._pop(self.task_id)
        return True

    async def _run(self) -> None:
        """Top-level reconnecting supervisor coroutine. Browserbase tears down the CDP
        socket whenever a short-lived client (a per-call browser_cdp or harness CDP connection)
        disconnects, so on drop we reset per-session ids, re-attach, and keep going.
        A failure before the first successful attach is fatal for ``start()``."""
        reconnect_failures, last_success_at, backoff = 0, 0.0, 0.5
        import websockets  # deferred: only supervisors that connect pay the import
        from agent.proxy_bypass import loopback_connect_kwargs
        connect_kwargs = {"max_size": 50 * 1024 * 1024, **loopback_connect_kwargs(self.cdp_url)}
        while not self._stop_requested:
            try:
                self._ws = await asyncio.wait_for(websockets.connect(self.cdp_url, **connect_kwargs), timeout=10.0)
            except Exception as e:
                if self._fail_start(e):
                    return
                reconnect_failures += 1
                if self._reconnect_budget_spent(reconnect_failures, e):
                    return
                logger.warning("CDP supervisor %s: connect failed (attempt %s/%s): %s",
                               self.task_id, reconnect_failures, MAX_POST_ATTACH_RECONNECT_FAILURES,
                               _redact_cdp_error_text(e))
                await asyncio.sleep(min(backoff, 10.0))
                backoff = min(backoff * 2, 10.0)
                continue

            reader_task = asyncio.create_task(self._read_loop(), name="cdp-reader")
            try:
                # Reset the per-connection page session id; ``_pending_dialogs`` / ``_frames``
                # are deliberately kept — they reconcile as fresh events arrive (worst case a
                # stale dialog entry is rejected with "no dialog is showing", logged only).
                self._page_session_id = None
                await self._attach_initial_page()
                self._set_active(True)
                last_success_at = time.time()
                reconnect_failures = 0
                backoff = 0.5  # reset after a successful attach
                self._ready_event.set()
                await reader_task
            except BaseException as e:
                if self._fail_start(e):
                    raise
                reconnect_failures += 1
                if self._reconnect_budget_spent(reconnect_failures, e):
                    return
                logger.warning("CDP supervisor %s: session dropped after %.1fs (attempt %s/%s): %s",
                               self.task_id, time.time() - last_success_at, reconnect_failures,
                               MAX_POST_ATTACH_RECONNECT_FAILURES, _redact_cdp_error_text(e))
            finally:
                self._set_active(False)
                if not reader_task.done():
                    reader_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await reader_task
                for handle in self._dialog_watchdogs.values():
                    handle.cancel()
                self._dialog_watchdogs.clear()
                await self._close_ws()
                # Replies to calls still in flight died with the socket: fail them now
                # instead of letting each caller sit out its full timeout.
                for fut in self._pending_calls.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("CDP connection closed before reply"))
                self._pending_calls.clear()

            if self._stop_requested:
                return
            logger.debug("CDP supervisor %s: reconnecting in %.1fs...", self.task_id, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _attach_initial_page(self) -> None:
        """Find (or create) a page target, attach flattened, enable domains, install dialog bridge.
        With ``attach_target_id`` only that target qualifies: missing or not a page raises
        ``SupervisorTargetError``, with nothing attached and no target created."""
        targets = (await self._cdp("Target.getTargets")).get("result", {}).get("targetInfos", [])
        if self.attach_target_id is not None:
            page_target = next((t for t in targets if t.get("targetId") == self.attach_target_id), None)
            if page_target is None or page_target.get("type") != "page":
                raise SupervisorTargetError(f"CDP target {self.attach_target_id} is not an open page")
        else:
            page_target = next((t for t in targets if t.get("type") == "page"), None)
        if page_target is None:
            page_target = (await self._cdp("Target.createTarget", {"url": "about:blank"}))["result"]
        attach = await self._cdp("Target.attachToTarget", {"targetId": page_target["targetId"], "flatten": True})
        self._page_session_id = sid = attach["result"]["sessionId"]
        await self._enable_page_domains(sid, timeout=10.0)
        await self._install_dialog_bridge(sid)

    async def _cdp(self, method: str, params: Optional[dict[str, Any]] = None, *,
                   session_id: Optional[str] = None, timeout: Optional[float] = 10.0) -> dict[str, Any]:
        """Send a CDP command and await its response."""
        if self._ws is None:
            raise _NotConnected("supervisor WebSocket is not connected")
        call_id, self._next_call_id = self._next_call_id, self._next_call_id + 1
        payload: dict[str, Any] = {"id": call_id, "method": method}
        payload.update({k: v for k, v in (("params", params), ("sessionId", session_id)) if v})
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def _read_loop(self) -> None:
        """Continuously dispatch incoming CDP frames (responses → futures, events → handlers)."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._stop_requested:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.debug("CDP supervisor: non-JSON frame dropped")
                    continue
                if "id" in msg:
                    fut = self._pending_calls.pop(msg["id"], None)
                    if fut is None or fut.done():
                        continue
                    if "error" in msg:
                        fut.set_exception(RuntimeError(f"CDP error on id={msg['id']}: {msg['error']}"))
                    else:
                        fut.set_result(msg)
                elif handler := self._EVENT_HANDLERS.get(msg.get("method")):
                    result = handler(self, msg.get("params", {}), msg.get("sessionId"))
                    if result is not None:
                        await result
        except Exception as e:
            logger.debug("CDP read loop exited: %s", e)

    # CDP event → handler(self, params, session_id). Async handlers return an
    # awaitable that ``_read_loop`` awaits; sync handlers return None.
    _EVENT_HANDLERS: dict[str, Callable[..., Any]] = {
        **DialogSupervisionMixin.EVENT_HANDLERS, **FrameTrackingMixin.EVENT_HANDLERS,
        **WorldTrackingMixin.EVENT_HANDLERS,
    }


class _SupervisorRegistry:
    """Process-global (task_id → supervisor) map with idempotent start/stop (``SUPERVISOR_REGISTRY``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: dict[str, CDPSupervisor] = {}
        self._home_by_task: dict[str, str] = {}  # profile home that started it (``stop_home``)

    def get(self, task_id: str) -> Optional[CDPSupervisor]:
        with self._lock:
            return self._by_task.get(task_id)

    def _pop(self, task_id: str) -> Optional[CDPSupervisor]:
        with self._lock:
            self._home_by_task.pop(task_id, None)
            return self._by_task.pop(task_id, None)

    def capture(self, task_id: str, *, timeout: float = 10.0) -> CapturedCDP:
        """Public CDP seam for trusted in-process plugins: a handle pinned to this task's
        current connection and default page session (``tools.browser_supervisor_capture``)."""
        from tools.browser_supervisor_capture import capture

        return capture(self, task_id, timeout=timeout)

    def get_or_start(self, task_id: str, cdp_url: str, *, dialog_policy: str = DEFAULT_DIALOG_POLICY,
                     dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S, start_timeout: float = 15.0,
                     attach_target_id: Optional[str] = None, dialog_bridge: bool = True) -> CDPSupervisor:
        """Idempotently ensure a supervisor runs for ``(task_id, cdp_url)``; one bound to a
        different ``cdp_url`` or unhealthy (dead thread / stopped loop) is stopped and replaced.

        Target-scoped (``attach_target_id`` or ``dialog_bridge=False``, see ``CDPSupervisor``): a
        supervisor for the task with the same endpoint, target and options is reused while healthy and a
        dead one replaced; any other supervisor for the task, on either side, raises
        ``SupervisorScopeConflict`` and is left running."""
        from hermes_constants import hermes_home_key
        home = hermes_home_key()
        scope = (cdp_url, attach_target_id, bool(dialog_bridge))
        scoped = attach_target_id is not None or not dialog_bridge

        def _scope_of(sup: CDPSupervisor) -> tuple:
            # getattr: a registered stand-in without the scoped fields is a managed (unscoped) supervisor.
            return (sup.cdp_url, getattr(sup, "attach_target_id", None), bool(getattr(sup, "dialog_bridge", True)))

        def _conflicts(sup: CDPSupervisor) -> bool:
            _, sup_target, sup_bridge = _scope_of(sup)
            return (scoped or sup_target is not None or not sup_bridge) and _scope_of(sup) != scope

        with self._lock:
            existing = self._by_task.get(task_id)
            if existing is not None:
                if _conflicts(existing):
                    raise SupervisorScopeConflict(
                        f"task {task_id!r} already has a supervisor with a different endpoint, target or options")
                thread, loop = existing._thread, existing._loop
                healthy = thread is not None and thread.is_alive() and loop is not None and loop.is_running()
                if existing.cdp_url == cdp_url and healthy:
                    return existing
                self._by_task.pop(task_id, None)
        if existing is not None:
            existing.stop()

        # A managed (unscoped) supervisor is constructed exactly as before the scoped options existed.
        scoped_kwargs = {"attach_target_id": attach_target_id, "dialog_bridge": dialog_bridge} if scoped else {}
        supervisor = CDPSupervisor(task_id=task_id, cdp_url=cdp_url,
                                   dialog_policy=dialog_policy, dialog_timeout_s=dialog_timeout_s, **scoped_kwargs)
        supervisor.start(timeout=start_timeout)
        with self._lock:
            # Guard against a concurrent get_or_start from another thread.
            already = self._by_task.get(task_id)
            if already is not None and (already.cdp_url == cdp_url or _conflicts(already)):
                supervisor.stop()
                if _conflicts(already):
                    raise SupervisorScopeConflict(
                        f"task {task_id!r} already has a supervisor with a different endpoint, target or options")
                return already
            self._by_task[task_id] = supervisor
            self._home_by_task[task_id] = home
        return supervisor

    def stop(self, task_id: str) -> None:
        supervisor = self._pop(task_id)
        if supervisor is not None:
            supervisor.stop()

    def stop_home(self, home_key: str) -> None:
        """Stop the supervisors one profile home started (``/browser connect`` on a multiplexed gateway)."""
        with self._lock:
            tasks = [task for task, home in self._home_by_task.items() if home == home_key]
        for task_id in tasks:
            self.stop(task_id)

    def stop_all(self) -> None:
        """Stop every running supervisor. For shutdown / test teardown."""
        with self._lock:
            items = list(self._by_task.values())
            self._by_task.clear()
            self._home_by_task.clear()
        for supervisor in items:
            supervisor.stop()


SUPERVISOR_REGISTRY = _SupervisorRegistry()


__all__ = ["SUPERVISOR_REGISTRY", "CDPSupervisor", "SupervisorScopeConflict", "SupervisorSnapshot",
           "SupervisorTargetError", "_SupervisorRegistry"]
