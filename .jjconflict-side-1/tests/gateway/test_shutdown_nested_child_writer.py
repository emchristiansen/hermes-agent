"""A stop never closes the shared SessionDB under a timed-out nested child conversation (A3-E5-03).

``_ChildRun.await_child`` returns a timeout result while the child's conversation is still running on
its own worker and defers ``child.close()`` to that worker's exit. The delegation unit (or gateway turn)
around it then finishes and stops being counted, while the conversation and the deferred close still
write through the shared SessionDB. Both are now detached writers the stop's quiesce waits for. These
run the real async dispatch → ``_run_single_child`` → await/timeout → AIAgent conversation → deferred
close → gateway stop; only the model call is held and a 5s inactivity cap is configured.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_shutdown_async_delegation_writers import _make_agent, _probe_tools_only
from tools import async_delegation as ad
from tools import detached_writers


@pytest.fixture(autouse=True)
def _clean():
    ad._reset_for_tests()
    detached_writers._reset_for_tests()
    yield
    if ad._executor is not None:
        ad._executor.shutdown(wait=True)
    ad._reset_for_tests()
    detached_writers._reset_for_tests()


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


async def _stop_after_nested_timeout(monkeypatch, tmp_path, *, release_at_quiesce):
    from gateway.run import GatewayRunner
    from hermes_state_registry import close_all
    from tools import delegate_tool as dt
    from tools import delegate_tool_child_run as cr
    child, db = _make_agent(monkeypatch, tmp_path)
    child.context_compressor.threshold_tokens = 10**9
    child._credential_pool = None
    child._subagent_id = None
    entered, release, child_closed = threading.Event(), threading.Event(), threading.Event()
    writes, closes, outer_results, nested = [], [], [], []
    real_write, real_close, real_child_close = db._execute_write, db.close, child.close
    real_defer = cr._defer_close_after_timeout

    def observed_write(fn, *a, **kw):
        writes.append({"after_close": bool(closes), "thread": threading.current_thread().name})
        return real_write(fn, *a, **kw)

    def observed_close(*a, **kw):
        closes.append({"child_closed": child_closed.is_set()})
        return real_close(*a, **kw)

    def observed_child_close(*a, **kw):
        try:
            return real_child_close(*a, **kw)
        finally:
            child_closed.set()

    def observed_defer(ch, future):
        nested.append(future)
        return real_defer(ch, future)

    def model(_kwargs):
        entered.set()
        assert release.wait(30), "scheduling bound exceeded"
        raise InterruptedError("the timed-out child acknowledges its stop")

    monkeypatch.setattr(db, "_execute_write", observed_write)
    monkeypatch.setattr(db, "close", observed_close)
    monkeypatch.setattr(child, "close", observed_child_close)
    monkeypatch.setattr(cr, "_defer_close_after_timeout", observed_defer)
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: 5.0)
    monkeypatch.setattr(cr._ChildRun, "seed_workspace", lambda self: None)
    child._interruptible_api_call = model

    def outer():
        outer_results.append(dt._run_single_child(0, "nested timeout child", child=child,
                                                   parent_agent=SimpleNamespace()))
        return outer_results[-1]

    real_quiesce = GatewayRunner._shutdown_executor

    def quiesce(runner, drain_timeout=0.0):
        if release_at_quiesce:
            release.set()
        return real_quiesce(runner, drain_timeout=drain_timeout)

    with ExitStack() as stack:
        stack.enter_context(patch.object(child, "_cleanup_task_resources"))
        stack.enter_context(_probe_tools_only())
        for target in ("tools.terminal_tool_lifecycle.cleanup_all_environments",
                       "tools.browser_tool_lifecycle.cleanup_all_browsers",
                       "gateway.status.remove_pid_file", "gateway.status.publish_runtime_status"):
            stack.enter_context(patch(target))
        stack.enter_context(patch("tools.process_registry.process_registry.kill_all", return_value=0))
        stack.enter_context(patch("cron.scheduler.mark_running_jobs_interrupted", return_value=[]))
        stack.enter_context(patch.object(GatewayRunner, "_shutdown_executor", quiesce))
        try:
            handle = ad.dispatch_async_delegation(
                goal="outer unit", context=None, toolsets=None, role="leaf", model=None, session_key="probe",
                runner=outer, interrupt_fn=lambda: cr._signal_child_stop(child))
            assert handle["status"] == "dispatched", handle
            assert await asyncio.to_thread(entered.wait, 30)
            # the outer unit times out and finishes; its nested conversation is still held in the model
            assert await asyncio.to_thread(_wait_for, lambda: outer_results and ad.join_workers(0) == 0, 30)
            before = {"outer": outer_results[0].get("status"), "close_deferred": bool(nested),
                      "async_live": ad.join_workers(0), "nested_done": nested[0].done() if nested else None}
            runner, _ = make_restart_runner()
            runner._restart_drain_timeout = 0.0
            runner._session_db = db
            await runner.stop()
            at_stop = {"connection_closed": db._conn is None, "child_closed": child_closed.is_set(),
                       "nested_done": nested[0].done() if nested else None}
        finally:
            release.set()
            if nested:
                await asyncio.to_thread(nested[0].result, 30)
                assert await asyncio.to_thread(child_closed.wait, 30)
            if ad._executor is not None:
                await asyncio.to_thread(ad._executor.shutdown, wait=True)
    observation = {"before": before, "at_stop": at_stop, "closes": closes,
                   "detached_live_after": detached_writers.join(30),
                   "post_close_writes": [w for w in writes if w["after_close"]],
                   "integrity_check": db._read_one("PRAGMA integrity_check")[0]}
    db.close()
    close_all()
    return observation


@pytest.mark.asyncio
async def test_shutdown_keeps_the_shared_db_open_under_a_timed_out_nested_conversation(monkeypatch, tmp_path):
    observation = await _stop_after_nested_timeout(monkeypatch, tmp_path, release_at_quiesce=False)
    before = observation["before"]
    assert before["outer"] == "timeout" and before["close_deferred"], observation
    assert before["async_live"] == 0 and before["nested_done"] is False, observation  # outer unit done, child not
    assert not observation["at_stop"]["connection_closed"], observation
    assert not observation["post_close_writes"], observation
    assert observation["detached_live_after"] == 0 and observation["integrity_check"] == "ok", observation


@pytest.mark.asyncio
async def test_a_nested_conversation_that_ends_within_the_quiesce_lets_the_close_run(monkeypatch, tmp_path):
    observation = await _stop_after_nested_timeout(monkeypatch, tmp_path, release_at_quiesce=True)
    at_stop = observation["at_stop"]
    assert at_stop["nested_done"] and at_stop["child_closed"] and at_stop["connection_closed"], observation
    assert not observation["post_close_writes"], observation
