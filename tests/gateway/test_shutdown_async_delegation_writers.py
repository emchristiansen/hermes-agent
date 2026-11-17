"""A stop never closes the shared SessionDB under an in-process async-delegation child that can still write.

A3-E5-01 (review t_60889c77/run3093): a background delegate's child agent writes through the gateway's
registry-owned SessionDB (``_open_child_session_db`` returns the same object). The final sweep only
*signals* it (``interrupt_all``); the child then still reads the durable tail, flushes its failed turn
and releases its turn lease. The close accounting counted every other writer but not this executor, so
``stop()`` closed (and the registry force-closed) the handle, and the child reopened it and wrote after
the quiescence boundary. These run the real AIAgent turn loop, real registered SessionDB, real async
dispatch/executor/finalizer, real stop lifecycle and close decision; only config, the model call and
external cleanup are synthetic. The first two are the reviewer's witnesses (observations kept in
``tmp_path`` instead of the review's scratch root), unchanged in their assertions.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_run_agent import _mock_response
from tests.gateway.restart_test_helpers import make_restart_runner
from tools.registry import registry

_PROBE_TOOL = "e5_shutdown_probe"
registry.register(
    name=_PROBE_TOOL, toolset="utility", handler=lambda args, **_kw: "probe ok", override=True,
    schema={"name": _PROBE_TOOL, "description": "probe tool for the E5 shutdown witnesses",
            "parameters": {"type": "object", "properties": {}, "required": []}})


def _config() -> dict:
    return {"compression": {"enabled": True, "threshold": 0.50, "target_ratio": 0.20, "protect_first_n": 2,
                            "protect_last_n": 6, "max_attempts": 1},
            "prompt_caching": {"cache_ttl": "5m"}, "sessions": {}, "bedrock": {}}


@contextlib.contextmanager
def _probe_tools_only():
    schema = {"type": "function", "function": {
        "name": _PROBE_TOOL, "description": "probe tool for the E5 shutdown witnesses",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    with patch("model_tools.get_tool_definitions", return_value=[schema]), \
            patch("model_tools.check_toolset_requirements", return_value={}):
        yield


def _make_agent(monkeypatch, tmp_path):
    """A real AIAgent on the registry's shared SessionDB (what a gateway parent hands its children)."""
    from hermes_cli import config as config_mod
    from hermes_state_registry import acquire
    from run_agent import AIAgent

    monkeypatch.setattr(config_mod, "load_config", _config)
    monkeypatch.setattr(config_mod, "load_config_readonly", _config)
    db = acquire(tmp_path / "state.db")
    with contextlib.redirect_stdout(io.StringIO()), _probe_tools_only(), patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(base_url="https://openrouter.ai/api/v1", api_key="test-key", model="test/model",
                        quiet_mode=True, skip_memory=True, skip_context_files=True, session_db=db,
                        session_id="e5-shutdown-child")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    return agent, db


@pytest.fixture(autouse=True)
def _clean_delegation_state():
    from hermes_state_registry import close_all
    from tools import async_delegation as ad
    ad._reset_for_tests()
    yield
    close_all()
    ad._reset_for_tests()


async def _stop_with_live_child(monkeypatch, tmp_path, answer, *, release_on_interrupt=False, terminal_record=False):
    """Dispatch a real async child on the shared DB, let its model call block, stop the gateway, then let
    the child finish and join it. ``answer()`` is the model's reply once released;
    ``release_on_interrupt`` lets the sweep's interrupt release it (a child that retires inside the
    quiesce); ``terminal_record`` force-finalizes the registry record first (the stall monitor's path)
    while the physical worker still runs."""
    from hermes_state_registry import close_all
    from tools.delegate_tool import _open_child_session_db
    from tools.delegate_tool_child_run import _signal_child_stop
    from tools import async_delegation as ad
    from gateway.run_shutdown_budget import final_sweep_live
    from gateway.run_runtime import managed_turn_count

    child, parent_db = _make_agent(monkeypatch, tmp_path)
    child_db = _open_child_session_db(SimpleNamespace(_session_db=parent_db))
    assert child_db is parent_db
    child._session_db = child_db
    child.context_compressor.threshold_tokens = 10**9
    entered, release, finished, interrupted = (threading.Event() for _ in range(4))
    results, writes, closes = [], [], []
    real_write, real_close = child_db._execute_write, child_db.close

    def observed_write(fn, *a, **kw):
        writes.append({'after_close': bool(closes), 'connection_absent': child_db._conn is None,
                       'thread': threading.current_thread().name, 'stack': traceback.format_stack()})
        return real_write(fn, *a, **kw)

    def observed_close(*a, **kw):
        closes.append({'child_finished': finished.is_set(), 'async_live': ad.active_count()})
        return real_close(*a, **kw)

    monkeypatch.setattr(child_db, '_execute_write', observed_write)
    monkeypatch.setattr(child_db, 'close', observed_close)

    def model(_kwargs):
        entered.set()
        assert release.wait(30), 'scheduling bound exceeded'
        return answer()

    def run_child():
        try:
            result = child.run_conversation('E5 child prompt')
            results.append(result)
            return {'status': 'interrupted' if result.get('interrupted') else 'completed',
                    'summary': result.get('final_response')}
        finally:
            finished.set()

    def interrupt():
        _signal_child_stop(child, 'gateway shutdown')
        interrupted.set()
        if release_on_interrupt:
            release.set()

    child._interruptible_api_call = model
    with patch.object(child, '_cleanup_task_resources'), _probe_tools_only(), \
            patch('tools.process_registry.process_registry.kill_all', return_value=0), \
            patch('cron.scheduler.mark_running_jobs_interrupted', return_value=[]), \
            patch('tools.terminal_tool_lifecycle.cleanup_all_environments'), \
            patch('tools.browser_session_lifecycle.cleanup_all_browsers'), \
            patch('gateway.status.remove_pid_file'), patch('gateway.status.publish_runtime_status'):
        handle = ad.dispatch_async_delegation(goal='bounded E5 child', context=None, toolsets=None,
            role='leaf', model='test/model', session_key='agent:main:telegram:dm:e5',
            parent_session_id=None, runner=run_child, interrupt_fn=interrupt)
        assert handle['status'] == 'dispatched', handle
        assert await asyncio.to_thread(entered.wait, 30)
        if terminal_record:
            ad._finalize(handle['delegation_id'], {'status': 'stalled', 'error': 'forced'}, 'stalled')
        runner, _ = make_restart_runner()
        runner._restart_drain_timeout = 0.0
        runner._session_db = child_db
        try:
            await runner.stop()
            at_stop = {'interrupted': interrupted.is_set(), 'child_finished': finished.is_set(),
                       'async_live': ad.active_count(), 'managed_live': managed_turn_count(runner),
                       'deferred_live': runner._active_deferred_agent_worker_count(),
                       'sweep_live': final_sweep_live(runner), 'connection_closed': child_db._conn is None}
            late = ad.dispatch_async_delegation(goal='after the stop', context=None, toolsets=None, role='leaf',
                model='test/model', session_key='agent:main:telegram:dm:e5-late', runner=dict)
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 30)
        await asyncio.to_thread(ad._executor.shutdown, wait=True)  # the real worker and its finalizer
    observation = {'at_stop': at_stop, 'closes': closes,
                   'post_close_writes': [w for w in writes if w['after_close']],
                   'connection_reopened': closes and child_db._conn is not None,
                   'result_interrupted': results[0].get('interrupted'), 'late_dispatch': late,
                   'integrity_check': child_db._read_one('PRAGMA integrity_check')[0],
                   'async_live_after_join': ad.active_count()}
    (tmp_path / 'e5-observation.txt').write_text(repr(observation))
    child_db.close()
    close_all()
    return observation


@pytest.mark.asyncio
async def test_shutdown_does_not_close_shared_db_while_delegate_still_answers(monkeypatch, tmp_path):
    """Reviewer witness test_review_e5.py: the model call returns an answer after the stop."""
    observation = await _stop_with_live_child(
        monkeypatch, tmp_path, lambda: _mock_response(content='synthetic answer after shutdown', finish_reason='stop'))
    assert not observation['at_stop']['connection_closed'], observation
    assert not observation['post_close_writes'], observation


@pytest.mark.asyncio
async def test_shutdown_does_not_close_shared_db_while_interrupted_delegate_finalizes(monkeypatch, tmp_path):
    """Reviewer witnesses test_review_e5_interrupt.py / test_review_e5_parent.py: the API acknowledges the
    interrupt; the child's failed-turn flush and lease release come after the stop's sweep."""
    def acknowledged():
        raise InterruptedError('synthetic API acknowledged shutdown interrupt')

    observation = await _stop_with_live_child(monkeypatch, tmp_path, acknowledged)
    assert observation['result_interrupted'] is True, observation
    assert not observation['at_stop']['connection_closed'], observation
    assert not observation['post_close_writes'], observation
    assert observation['integrity_check'] == 'ok'


@pytest.mark.asyncio
async def test_a_still_running_worker_whose_record_is_terminal_keeps_the_db_open(monkeypatch, tmp_path):
    """The stall monitor can finalize a record while its worker still runs: the logical record says
    nothing is live (and interrupt_all no longer reaches it), the physical worker can still write."""
    def acknowledged():
        raise InterruptedError('late return after a forced stall')

    observation = await _stop_with_live_child(monkeypatch, tmp_path, acknowledged, terminal_record=True)
    assert observation['at_stop']['async_live'] == 0 and not observation['at_stop']['child_finished'], observation
    assert not observation['at_stop']['connection_closed'], observation
    assert not observation['post_close_writes'], observation


@pytest.mark.asyncio
async def test_a_child_that_retires_within_the_quiesce_lets_the_close_run(monkeypatch, tmp_path):
    """After its real retirement (all final writes done) the normal close still happens in the same stop."""
    def acknowledged():
        raise InterruptedError('synthetic API acknowledged shutdown interrupt')

    observation = await _stop_with_live_child(monkeypatch, tmp_path, acknowledged, release_on_interrupt=True)
    assert observation['at_stop']['child_finished'] and observation['at_stop']['connection_closed'], observation
    assert observation['result_interrupted'] is True, observation
    assert not observation['post_close_writes'], observation


@pytest.mark.asyncio
async def test_no_delegation_is_admitted_once_the_stop_has_swept(monkeypatch, tmp_path):
    def answer():
        return _mock_response(content='answer', finish_reason='stop')

    observation = await _stop_with_live_child(monkeypatch, tmp_path, answer, release_on_interrupt=True)
    assert observation['late_dispatch']['status'] == 'rejected', observation
    assert 'shutting down' in observation['late_dispatch']['error'], observation


@pytest.mark.asyncio
async def test_a_dispatch_admitted_before_the_fence_but_not_yet_submitted_keeps_the_db_open(monkeypatch, tmp_path):
    """Admission in flight (A4 owner run3168): a dispatching thread has published its record (so the
    fence no longer refuses it) but has not yet submitted its worker when the stop sweeps and closes.
    Held between publication and submit, it must already count as a writer; released after the stop,
    its child then runs and writes through the shared handle."""
    from hermes_state_registry import close_all
    from tools.delegate_tool import _open_child_session_db
    from tools.delegate_tool_child_run import _signal_child_stop
    from tools import async_delegation as ad

    child, parent_db = _make_agent(monkeypatch, tmp_path)
    child_db = _open_child_session_db(SimpleNamespace(_session_db=parent_db))
    child._session_db = child_db
    child.context_compressor.threshold_tokens = 10**9
    published, hold, finished = threading.Event(), threading.Event(), threading.Event()
    writes, closes, handles = [], [], []
    real_write, real_close, real_persist = child_db._execute_write, child_db.close, ad._persist_dispatch

    def observed_write(fn, *a, **kw):
        writes.append({'after_close': bool(closes), 'thread': threading.current_thread().name})
        return real_write(fn, *a, **kw)

    def observed_close(*a, **kw):
        closes.append({'child_finished': finished.is_set()})
        return real_close(*a, **kw)

    def held_persist(record):
        real_persist(record)
        published.set()
        assert hold.wait(30), 'scheduling bound exceeded'  # between publication and submit

    def run_child():
        try:
            result = child.run_conversation('E5 admission-in-flight child prompt')
            return {'status': 'interrupted' if result.get('interrupted') else 'completed'}
        finally:
            finished.set()

    monkeypatch.setattr(child_db, '_execute_write', observed_write)
    monkeypatch.setattr(child_db, 'close', observed_close)
    monkeypatch.setattr(ad, '_persist_dispatch', held_persist)
    child._interruptible_api_call = lambda _kwargs: _mock_response(content='late child answer', finish_reason='stop')
    dispatcher = threading.Thread(target=lambda: handles.append(ad.dispatch_async_delegation(
        goal='admission in flight', context=None, toolsets=None, role='leaf', model='test/model',
        session_key='agent:main:telegram:dm:e5-admit', runner=run_child,
        interrupt_fn=lambda: _signal_child_stop(child, 'gateway shutdown'))), name='e5-dispatcher')
    with patch.object(child, '_cleanup_task_resources'), _probe_tools_only(), \
            patch('tools.process_registry.process_registry.kill_all', return_value=0), \
            patch('cron.scheduler.mark_running_jobs_interrupted', return_value=[]), \
            patch('tools.terminal_tool_lifecycle.cleanup_all_environments'), \
            patch('tools.browser_session_lifecycle.cleanup_all_browsers'), \
            patch('gateway.status.remove_pid_file'), patch('gateway.status.publish_runtime_status'):
        dispatcher.start()
        assert await asyncio.to_thread(published.wait, 30)
        runner, _ = make_restart_runner()
        runner._restart_drain_timeout = 0.0
        runner._session_db = child_db
        try:
            await runner.stop()
            closed_at_stop = child_db._conn is None
        finally:
            hold.set()
        await asyncio.to_thread(dispatcher.join, 30)
        assert handles and handles[0]['status'] == 'dispatched', handles
        assert await asyncio.to_thread(finished.wait, 30)
        await asyncio.to_thread(ad._executor.shutdown, wait=True)
    post_close_writes = [w for w in writes if w['after_close']]
    observation = {'closed_at_stop': closed_at_stop, 'closes': closes, 'post_close_writes': post_close_writes,
                   'writes': len(writes)}
    (tmp_path / 'e5-admission-observation.txt').write_text(repr(observation))
    child_db.close()
    close_all()
    assert not closed_at_stop, observation
    assert not post_close_writes, observation
