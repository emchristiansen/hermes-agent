"""Off-loop notification state.db work stays owned until its thread finishes (W2 vs the shutdown close).

The stop cancels completion-batch flushes and the notification watchers. Cancelling a coroutine that
awaits ``asyncio.to_thread`` does not stop its default-executor thread, and nothing the SessionDB close
gate counts saw that thread. The close then ran under a live completion-admission read, and the read
reopened the closed writer and recreated the WAL the close had just ended (#94736, int-r2 to_thread probe).
``owned_offload`` keeps the executor future registered as a deferred worker (the owner the close gate
already counts) until the thread is done. These tests drive the real admission reader, the real cancel,
and the real close gate. The only synthetic piece is an Event that parks the reader at its real DB
boundary.
"""
import ast
import asyncio
import contextlib
import contextvars
import logging
import threading
import time
from pathlib import Path

import pytest

import hermes_state_runtime
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.test_loop_offload_io import _identity, _native, _owner, _seed


async def _until(pred, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


def _stop_ctx(runner):
    ctx = GatewayShutdownMixin._StopContext(deferred_count=runner._active_deferred_agent_worker_count,
                                            started_at=time.monotonic())
    ctx.api_live = runner._active_api_run_count()
    return ctx


class _Park:
    """Parks a real state.db reader (default: ``list_session_admissions_containing``, the admission
    reader's DB read) before it starts or inside its real ``_read_ctx``; records the closes and any
    reopen of the closed writer."""

    def __init__(self, monkeypatch, db, *, inside, target=(hermes_state_runtime, "list_session_admissions_containing")):
        self.db, self.inside = db, inside
        self.parked, self.release, self.done = threading.Event(), threading.Event(), threading.Event()
        self.events, self.closes = [], []
        module, name = target
        real_list = getattr(module, name)
        real_read_ctx = db._read_ctx
        real_close = type(db).close.__get__(db)
        real_reopen = type(db)._reopen_after_close_locked.__get__(db)
        local = threading.local()

        def hooked_list(*a, **kw):
            try:
                if not inside:
                    self._park()
                local.reader = True
                try:
                    return real_list(*a, **kw)
                finally:
                    local.reader = False
            finally:
                self.done.set()

        @contextlib.contextmanager
        def hooked_read_ctx(*a, **kw):
            with real_read_ctx(*a, **kw) as conn:
                if inside and getattr(local, "reader", False):
                    self._park()
                yield conn

        def close():
            self.closes.append({"reader_live": self.parked.is_set() and not self.done.is_set()})
            return real_close()

        def reopen(context="write"):
            self.events.append(("reopen", context))
            return real_reopen(context)

        monkeypatch.setattr(module, name, hooked_list)
        monkeypatch.setattr(db, "_read_ctx", hooked_read_ctx)
        monkeypatch.setattr(db, "close", close)
        monkeypatch.setattr(db, "_reopen_after_close_locked", reopen)
        self.real_close = real_close

    def _park(self):
        self.parked.set()
        self.events.append(("parked", None))
        assert self.release.wait(60), "reader never released"
        self.events.append(("released", self.db.__dict__.get("_conn") is None))


async def _batch(runner, authority, key, root):
    evt = {"type": "completion", "session_id": "proc_own", "started_at": 1.0, "session_key": key,
           "output": "ok", "exit_code": 0}
    ident = _identity(runner, evt)
    _seed(authority, root, key, ident, _native(ident, [ident]))  # already admitted: nothing is delivered
    runner._completion_notification_batch_window = 0
    return asyncio.ensure_future(runner._enqueue_process_completion_notification("[proc_own done]", evt))


@pytest.mark.asyncio
@pytest.mark.parametrize("inside", [False, True], ids=["before-read", "inside-read"])
async def test_cancelled_flush_reader_keeps_session_db_open_until_it_finishes(tmp_path, monkeypatch, inside):
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    db = authority.db
    park = _Park(monkeypatch, db, inside=inside)
    waiter = await _batch(runner, authority, key, root)
    try:
        assert await _until(park.parked.is_set)
        runner._stop_deadline = time.monotonic() + 30.0
        await runner._cancel_process_completion_batch_tasks()
        assert waiter.done() and waiter.result() is False  # the watcher still gets its retryable result
        assert runner._active_deferred_agent_worker_count() == 1  # the running read is still owned
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
        assert park.closes == [] and db.__dict__.get("_conn") is not None  # close skipped under the live read
        park.release.set()
        assert await _until(park.done.is_set)
        assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
        assert ("reopen", "read") not in park.events
        # Once the read has finished, the very same close proceeds.
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
        assert park.closes and not any(c["reader_live"] for c in park.closes)
        assert db.__dict__.get("_conn") is None
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("inside", [False, True], ids=["before-read", "inside-read"])
async def test_full_stop_never_closes_under_a_cancelled_flush_reader(tmp_path, monkeypatch, caplog, inside):
    from unittest.mock import patch
    import gateway.run as gw
    caplog.set_level(logging.INFO)
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    db = authority.db
    park = _Park(monkeypatch, db, inside=inside)
    waiter = await _batch(runner, authority, key, root)
    monkeypatch.setattr(gw, "_hermes_home", tmp_path)
    runner.adapters = {}
    runner._restart_drain_timeout = 0.5  # the drain now waits for the owned read; keep the test short
    try:
        assert await _until(park.parked.is_set)
        with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            await asyncio.wait_for(runner.stop(), 60)
        assert waiter.done() and waiter.result() is False
        assert park.closes == [], park.closes
        assert "skipping the SessionDB close" in caplog.text
        park.release.set()
        assert await _until(park.done.is_set)
        assert ("reopen", "read") not in park.events
        assert db.__dict__.get("_conn") is not None  # still the original, never-closed writer
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


@pytest.mark.asyncio
async def test_finished_reader_does_not_hold_the_close(tmp_path, monkeypatch):
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    db = authority.db
    park = _Park(monkeypatch, db, inside=False)
    park.release.set()  # the read runs straight through
    waiter = await _batch(runner, authority, key, root)
    assert await asyncio.wait_for(waiter, 30) is True
    assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
    GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
    assert park.closes and db.__dict__.get("_conn") is None
    assert ("reopen", "read") not in park.events


# --- every completion-admission offload: a cancelled caller leaves the read owned ---------------------

def _awaiting(task):
    """Names of the coroutines the task is suspended in, outermost first (its real await chain)."""
    names, coro = [], task.get_coro()
    while coro is not None and hasattr(coro, "cr_code"):
        names.append(coro.cr_code.co_name)
        coro = coro.cr_await
    return names


def _pending(key, name, root):
    from tests.gateway.test_completion_admission import pending
    return dict(pending(key, name), parent_session_id=root)


async def _site_group(runner, key, root):
    return await runner._deliver_async_delegation_group([_pending(key, "own-group", root)])


async def _site_completion(runner, key, root):
    return await runner._deliver_completion_notification("[own-completion]", _pending(key, "own-completion", root))


SITES = {"_deliver_async_delegation_group_scoped": _site_group,
         "_deliver_completion_notification_scoped": _site_completion}


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(SITES))
async def test_completion_admission_site_stays_owned_when_its_caller_is_cancelled(tmp_path, monkeypatch, site):
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    db = authority.db
    park = _Park(monkeypatch, db, inside=False)
    task = asyncio.ensure_future(SITES[site](runner, key, root))
    try:
        assert await _until(park.parked.is_set), "the admission read never ran"
        chain = _awaiting(task)
        assert site in chain and chain[-1] == "owned_offload", chain  # parked at the intended offload site
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert runner._active_deferred_agent_worker_count() == 1
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
        assert park.closes == []
        park.release.set()
        assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
        assert ("reopen", "read") not in park.events
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


async def _cancel_then_close(runner, task, park, site):
    """The shared real-DB contract: *task* is parked in *site*'s owned offload over the gated SessionDB.
    Cancelling it leaves the read owned, the close is skipped, nothing reopens, and once the read has
    finished the very same close proceeds."""
    db = park.db
    assert await _until(park.parked.is_set), "the state.db read never ran"
    chain = _awaiting(task)
    assert site in chain and chain[-1] == "owned_offload", chain  # parked at the intended offload site
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert runner._active_deferred_agent_worker_count() == 1
    GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
    assert park.closes == [] and db.__dict__.get("_conn") is not None
    park.release.set()
    assert await _until(park.done.is_set)
    assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
    assert not [e for e in park.events if e[0] == "reopen"], park.events
    GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
    assert park.closes and not any(c["reader_live"] for c in park.closes)
    assert db.__dict__.get("_conn") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("inside", [False, True], ids=["before-read", "inside-read"])
async def test_process_watcher_admission_recheck_keeps_session_db_open_when_cancelled(tmp_path, monkeypatch, inside):
    """The watcher's lost-ACK recheck: the real ``completion_admission`` over the real authority store."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import tools.process_registry as pr_module
    from gateway import run_notifications as rn
    from tests.gateway.test_background_process_notifications import _FakeRegistry

    runner, authority, key, _root = await _owner(tmp_path, monkeypatch)
    exited = SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="echo", started_at=1.0)
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry([exited] * 20, consumed=False))
    monkeypatch.setattr(rn, "_MAX_COMPLETION_INJECTION_ATTEMPTS", 1)
    runner._enqueue_process_completion_notification = AsyncMock(return_value=False)  # refused: recheck
    park = _Park(monkeypatch, authority.db, inside=inside)
    watcher = {"session_id": "proc_watch_own", "check_interval": 0, "notify_on_complete": True,
               "session_key": key, "platform": "discord", "chat_id": "42"}
    task = asyncio.ensure_future(runner._run_process_watcher(watcher))
    try:
        await _cancel_then_close(runner, task, park, "_run_process_watcher")
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["prepare_heartbeat", "withdraw_process_ticks"])
async def test_heartbeat_ledger_reads_keep_session_db_open_when_cancelled(tmp_path, monkeypatch, site):
    """session_automation: a liveness tick's dedup read and a completion's tick withdrawal, over the real
    admission ledger (the process-notice path the stopped watchers run)."""
    from gateway import session_automation
    runner, authority, key, _root = await _owner(tmp_path, monkeypatch)
    target = ("_process_completed" if site == "prepare_heartbeat" else "list_session_admissions")
    park = _Park(monkeypatch, authority.db, inside=True, target=(session_automation, target))
    event = {"type": "heartbeat" if site == "prepare_heartbeat" else "completion", "session_id": "proc_tick_own",
             "started_at": 1.0, "session_key": key, "event_id": "tick-own", "output": "still running"}
    task = asyncio.ensure_future(getattr(session_automation, site)(runner, event))
    try:
        await _cancel_then_close(runner, task, park, site)
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


@pytest.mark.asyncio
async def test_delegation_delivery_write_keeps_session_db_open_when_cancelled(tmp_path, monkeypatch):
    """wake.persist_delegation_delivery: the delivery-row write into the gateway's own SessionDB (the store
    an api_server adapter selects through its runner) runs after a skipped close, never after a real one."""
    from types import SimpleNamespace
    from gateway import wake
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    db = authority.db
    park = _Park(monkeypatch, db, inside=False, target=(db, "append_delegation_delivery"))
    adapter = SimpleNamespace(gateway_runner=runner, _ensure_session_db=lambda: db)
    evt = {"type": "async_delegation", "delegation_id": "deleg-own", "session_key": key}
    task = asyncio.ensure_future(wake.persist_delegation_delivery(adapter, text="[delegation done]",
                                                                  session_id=root, evt=evt))
    try:
        await _cancel_then_close(runner, task, park, "persist_delegation_delivery")
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


# --- the owned read is visible through the existing active-work interfaces, and only while it runs --

@pytest.mark.asyncio
async def test_in_flight_owned_read_is_reported_as_active_work_only_while_it_runs(tmp_path, monkeypatch):
    """The deferred-worker owner feeds ``_active_work_count`` and so the runtime status ``active_agents``,
    ``_describe_active_work`` (the active-work verb and the drain notice) and the idle-exit probe. An owned
    notification read shows there as one deferred unit while its thread runs (a stop or idle exit waits
    for it, as for any live writer) and is gone the moment it finishes."""
    import gateway.run as gw
    from gateway.run_idle_exit import busy_reason
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    statuses = []
    monkeypatch.setattr(gw, "_write_runtime_status_quiet", lambda **kw: statuses.append(kw.get("active_agents")))
    runner._running, runner._draining, runner._idle_exit_last_activity = True, False, 0.0
    baseline = runner._active_work_count()
    park = _Park(monkeypatch, authority.db, inside=True)
    task = asyncio.ensure_future(_site_completion(runner, key, root))
    try:
        assert await _until(park.parked.is_set)
        assert runner._active_work_count() == baseline + 1
        assert sum(u.get("kind") == "deferred" for u in runner._describe_active_work()) == 1
        assert busy_reason(runner, window=0.0) == "agent work in flight"
        runner._persist_active_agents()
        assert statuses[-1] == baseline + 1  # the runtime status publishes the in-flight read
        park.release.set()
        await asyncio.wait_for(task, 30)
        assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
        assert runner._active_work_count() == baseline
        assert not any(u.get("kind") == "deferred" for u in runner._describe_active_work())
        assert busy_reason(runner, window=0.0) != "agent work in flight"
        runner._persist_active_agents()
        assert statuses[-1] == baseline
    finally:
        park.release.set()
        with contextlib.suppress(Exception):
            park.real_close()


# --- the helper's own contract ---------------------------------------------------------------------

class _Owner:
    _track_deferred_agent_worker = GatewayShutdownMixin._track_deferred_agent_worker
    _active_deferred_agent_worker_count = GatewayShutdownMixin._active_deferred_agent_worker_count


@pytest.mark.asyncio
async def test_owned_offload_runs_off_loop_with_the_callers_context_and_honest_results():
    from gateway.run_notifications_offload import owned_offload
    var = contextvars.ContextVar("profile_scope_probe")
    var.set("served-b")
    loop_thread = threading.get_ident()
    seen = {}

    def read(x):
        seen["thread"], seen["var"] = threading.get_ident(), var.get()
        return x + 1

    owner = _Owner()
    assert await owned_offload(owner, read, 41) == 42
    assert seen == {"thread": seen["thread"], "var": "served-b"} and seen["thread"] != loop_thread

    def boom():
        raise LookupError("not in this profile's store")
    with pytest.raises(LookupError):
        await owned_offload(owner, boom)
    assert await owned_offload(owner, lambda a, *, b: (a, b), 1, b=2) == (1, 2)  # keywords pass through
    assert await owned_offload(None, read, 1) == 2  # no owner: plain to_thread semantics
    assert await _until(lambda: owner._active_deferred_agent_worker_count() == 0)


@pytest.mark.asyncio
async def test_owned_offload_cancelled_before_start_never_touches_the_store():
    from gateway.run_notifications_offload import owned_offload
    owner = _Owner()
    gate, busy, ran = threading.Event(), threading.Event(), []
    loop = asyncio.get_running_loop()
    # Saturate a 1-thread default executor so the call is queued, not running, when its caller is cancelled.
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)

    def occupy():
        busy.set()
        gate.wait(30)
    blocker = loop.run_in_executor(None, occupy)
    assert await _until(busy.is_set)  # the only worker is taken
    task = asyncio.ensure_future(owned_offload(owner, lambda: ran.append(1)))
    await asyncio.sleep(0)  # one step: the call is submitted (queued behind the blocker) and the shield awaited
    assert owner._active_deferred_agent_worker_count() == 1
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert owner._active_deferred_agent_worker_count() == 1  # queued: still owned until it has run
    gate.set()
    await blocker
    assert await _until(lambda: owner._active_deferred_agent_worker_count() == 0)
    assert ran == []
    executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_owned_offload_abandoned_failure_is_logged(caplog):
    from gateway.run_notifications_offload import owned_offload
    owner = _Owner()
    started, release = threading.Event(), threading.Event()

    def fail():
        started.set()
        release.wait(30)
        raise RuntimeError("store went away")
    task = asyncio.ensure_future(owned_offload(owner, fail))
    assert await _until(started.is_set)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    release.set()
    assert await _until(lambda: owner._active_deferred_agent_worker_count() == 0)
    await asyncio.sleep(0)
    assert any("after its caller was cancelled" in r.getMessage() for r in caplog.records)
    # Consumed, not left to the loop's exception handler ("exception in shielded future" at ERROR).
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.records


# --- coverage: every notification offload goes through the owner ------------------------------------

EXPECTED_SITES = {  # module -> enclosing function -> owner expression per owned offload
    "gateway/run_notifications.py": {  # each a state.db reader, opener or writer
        "_self_post_api_server": ["self"],  # served profiles' state.db ownership read
        "_inject_watch_notification": ["self"], "_launching_turn_active": ["self"],  # session store
        "_send_watcher_message": ["self"],  # session store
        "_classify_completion_owner": ["self"],  # session_store._ensure_loaded (SessionDB-backed store)
        "_completion_delivery_ready": ["self", "self"],  # session store + the API adapter's SessionDB open
        "_served_api_server_event_scope": ["self"],  # served profiles' state.db ownership read
        "_deliver_completion_notification_scoped": ["self"], "_flush_process_completion_batch": ["self"],
        "_deliver_async_delegation_group_scoped": ["self"], "_run_process_watcher": ["self"],  # admission reads
        "_async_delegation_watcher": ["self"],  # orphaned-completion ledger sweep (durable writes)
    },
    "gateway/session_automation.py": {  # authority admission-ledger reads/writes
        "prepare_heartbeat": ["runner", "runner", "runner"], "withdraw_process_ticks": ["runner"],
        "move_route_wakes": ["runner"], "_fold_queued_kanban_wakes": ["authority.runner"],
    },
    "gateway/wake.py": {  # API adapter SessionDB open, continuation resolve, delivery-row write
        "persist_delegation_delivery": ["getattr(adapter, 'gateway_runner', None)"] * 3,
    },
    "gateway/run_notifications_ownership.py": {"_requeue_uncertain_completion": ["runner"]},  # ledger requeue
}


@pytest.mark.parametrize("module", sorted(EXPECTED_SITES))
def test_no_completion_path_offload_bypasses_the_owner(module):
    import gateway.run_notifications as rn
    raw, owned = [], {}

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.Call):
                name = ast.unparse(child.func)
                if name.endswith("to_thread") or name.endswith("run_in_executor"):
                    raw.append((fn, child.lineno))
                if name == "owned_offload":
                    owned.setdefault(fn, []).append(ast.unparse(child.args[0]))
            visit(child, inner)
    root = Path(rn.__file__).resolve().parent.parent
    visit(ast.parse((root / module).read_text()), None)
    assert raw == []
    assert owned == EXPECTED_SITES[module]
