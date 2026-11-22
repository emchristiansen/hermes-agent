"""An async delegation whose submit raised never runs as an uncounted writer (A3-E5-02).

``ThreadPoolExecutor.submit`` queues the work item BEFORE it grows the pool, so a worker-thread start
that fails raises out of ``submit`` while the item stays queued for a surviving worker. Treating that
exception as "not admitted" released the unit's writer lifetime, dropped its record and reported a
rejection, and the surviving worker then ran the child anyway: the stop closed the shared SessionDB
under it. The dispatcher and the worker now race on a one-shot claim: a revoked unit never runs, and a
unit a worker already started stays dispatched and counted. Real executor, dispatch, child agent,
SessionDB and stop; only one thread start fails and the model call is held.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_shutdown_async_delegation_writers import _make_agent, _probe_tools_only
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean():
    ad._reset_for_tests()
    yield
    if ad._executor is not None:
        ad._executor.shutdown(wait=True)
    ad._reset_for_tests()


def _dispatch(runner, **kw):
    return ad.dispatch_async_delegation(goal="submit-after-enqueue probe", context=None, toolsets=None,
                                        role="leaf", model=None, session_key="probe", runner=runner, **kw)


def _stop_patches():
    return (patch("tools.process_registry.process_registry.kill_all", return_value=0),
            patch("cron.scheduler.mark_running_jobs_interrupted", return_value=[]),
            patch("tools.terminal_tool_lifecycle.cleanup_all_environments"),
            patch("tools.browser_session_lifecycle.cleanup_all_browsers"),
            patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"))


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_starts_first", [False, True], ids=["revoke-wins", "start-wins"])
async def test_a_unit_whose_submit_raised_after_enqueue_never_writes_uncounted(
        monkeypatch, tmp_path, worker_starts_first):
    from contextlib import ExitStack

    from hermes_state_registry import close_all
    from tools.delegate_tool import _open_child_session_db
    child, db = _make_agent(monkeypatch, tmp_path)
    assert _open_child_session_db(SimpleNamespace(_session_db=db)) is db  # the child writes the shared DB
    child.context_compressor.threshold_tokens = 10**9
    first_entered, first_release = threading.Event(), threading.Event()
    child_entered, child_release = threading.Event(), threading.Event()
    closes, writes = [], []
    real_write, real_close = db._execute_write, db.close

    def observed_write(fn, *a, **kw):
        writes.append({"after_close": bool(closes), "thread": threading.current_thread().name})
        return real_write(fn, *a, **kw)

    def observed_close(*a, **kw):
        closes.append(ad.join_workers(0))
        return real_close(*a, **kw)

    monkeypatch.setattr(db, "_execute_write", observed_write)
    monkeypatch.setattr(db, "close", observed_close)

    def first():
        first_entered.set()
        assert first_release.wait(30)
        return {"status": "completed"}

    def model(_kwargs):
        child_entered.set()
        assert child_release.wait(30), "scheduling bound exceeded"
        raise InterruptedError("held child unwinds after the stop")

    child._interruptible_api_call = model

    def run_child():
        child.run_conversation("queued behind a failed thread start")
        return {"status": "interrupted"}

    real_start, faults = threading.Thread.start, []

    def fail_one_growth(thread):
        if thread.name == "async-delegate_1":
            faults.append(thread.name)
            if worker_starts_first:  # the surviving worker takes the queued item before submit raises
                first_release.set()
                assert child_entered.wait(30)
            raise RuntimeError("injected worker-thread start failure")
        return real_start(thread)

    with ExitStack() as stack:
        stack.enter_context(patch.object(child, "_cleanup_task_resources"))
        stack.enter_context(_probe_tools_only())
        for p in _stop_patches():
            stack.enter_context(p)
        try:
            assert _dispatch(first, max_async_children=1)["status"] == "dispatched"
            assert await asyncio.to_thread(first_entered.wait, 30)
            with patch.object(threading.Thread, "start", fail_one_growth):
                handle = _dispatch(run_child, max_async_children=2)
            assert faults == ["async-delegate_1"]
            first_release.set()
            if worker_starts_first:
                assert handle["status"] == "dispatched", handle
            else:
                assert handle["status"] == "rejected", handle
                # the surviving worker dequeues the revoked item and must not run it
                assert not await asyncio.to_thread(child_entered.wait, 3)
            live_before_stop = ad.join_workers(0)
            runner, _ = make_restart_runner()
            runner._restart_drain_timeout = 0.0
            runner._session_db = db
            await runner.stop()
            closed_at_stop = db._conn is None
        finally:
            first_release.set()
            child_release.set()
            if ad._executor is not None:
                await asyncio.to_thread(ad._executor.shutdown, wait=True)
    observation = {"handle": handle, "child_ran": child_entered.is_set(), "live_before_stop": live_before_stop,
                   "closed_at_stop": closed_at_stop, "closes": closes,
                   "post_close_writes": [w for w in writes if w["after_close"]]}
    db.close()
    close_all()
    if worker_starts_first:  # a live writer: counted, so the close is skipped
        assert live_before_stop == 1 and not closed_at_stop, observation
    else:  # nothing ran: the close may proceed
        assert not observation["child_ran"] and live_before_stop == 0, observation
    assert not observation["post_close_writes"], observation


def test_a_clean_submit_refusal_rejects_and_retires_the_unit():
    """Pre-enqueue refusal (the executor is already shut down): rejected, nothing counted or retained."""
    from hermes_cli.backend_retirement import retirement
    ad._get_executor(1).shutdown(wait=True)
    before = retirement.active_count()
    assert _dispatch(dict)["status"] == "rejected"
    assert not ad._records
    assert ad.join_workers(0) == 0
    assert retirement.active_count() == before


def test_a_unit_queued_behind_a_busy_worker_runs_and_is_counted_until_it_returns(monkeypatch):
    entered, release, ran = threading.Event(), threading.Event(), threading.Event()

    def busy():
        entered.set()
        assert release.wait(30)
        return {"status": "completed"}

    def queued():
        ran.set()
        return {"status": "completed"}

    assert _dispatch(busy, max_async_children=1)["status"] == "dispatched"
    assert entered.wait(30)
    monkeypatch.setattr(ad, "_executor_max_workers", 99)  # no growth: the second unit queues behind the first
    handle = _dispatch(queued, max_async_children=2)
    try:
        assert handle["status"] == "dispatched", handle
        assert ad.join_workers(0) == 2 and not ran.is_set()
    finally:
        release.set()
    assert ad.join_workers(30) == 0 and ran.is_set()
