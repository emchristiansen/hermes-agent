"""V3-ASYNC-01 regression (review6658 probe test_review_stale_monitor_writer.py, sha256 01bbf9dc..., adopted as a maintained
test; only the observation path changed from the reviewer's scratch dir to ``tmp_path``): real stale monitor finalizer survives a returned worker.
Scheduling is held at completion persistence; real dispatch, monitor, SQL, close gate.
"""
import asyncio
from contextlib import ExitStack
from unittest.mock import patch
import json
import threading
import time

import pytest

from tools import async_delegation as ad
from tools import detached_writers as dw
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from gateway import run_shutdown_budget as budget
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.parametrize('full_stop', [False, True], ids=['close-gate', 'full-stop'])
@pytest.mark.parametrize('worker_returns', [False, True], ids=['worker-counted-control', 'monitor-only-writer'])
def test_monitor_finalization_is_a_physical_writer(monkeypatch, tmp_path, worker_returns, full_stop):
    from hermes_state_registry import acquire, close_all
    ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    db = acquire(ad._db_path())
    worker_entered, release_worker, persist_entered, release_persist = [threading.Event() for _ in range(4)]
    writes, closes = [], []
    original_persist, original_close = ad._persist_completion, db.close

    def held_persist(event, result):
        persist_entered.set()
        assert release_persist.wait(30)
        writes.append({'after_close': bool(closes), 'thread': threading.current_thread().name})
        return original_persist(event, result)

    def close(*a, **kw):
        closes.append({'physical_worker_count': ad.join_workers(0), 'logical_live': ad.active_count(),
                       'monitor_alive': ad._monitor_thread.is_alive()})
        return original_close(*a, **kw)

    def work():
        worker_entered.set()
        assert release_worker.wait(30)
        return {'status': 'completed'}

    monkeypatch.setattr(ad, '_persist_completion', held_persist)
    monkeypatch.setattr(ad, '_STALE_CHECK_INTERVAL', 0.01)
    monkeypatch.setattr(db, 'close', close)
    h = ad.dispatch_async_delegation(goal='monitor finalizer race', context=None, toolsets=None,
        role='leaf', model=None, session_key='probe', runner=work, progress_fn=lambda: ('unchanged', False))
    assert h['status'] == 'dispatched'
    try:
        assert worker_entered.wait(30)
        with ad._records_lock:
            ad._records[h['delegation_id']].update(status='stalling', _interrupted_at=time.time()-121)
        assert persist_entered.wait(30)
        if worker_returns:
            release_worker.set()
            ad._executor.shutdown(wait=True)
        ad.shutdown_all()
        before = {'workers': ad.join_workers(0), 'logical': ad.active_count(),
                  'monitor_alive': ad._monitor_thread.is_alive()}
        runner, _ = make_restart_runner()
        runner._session_db = db
        ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0, started_at=time.monotonic())
        ctx.api_live = 0
        if full_stop:
            runner._restart_drain_timeout = 0.0
            with ExitStack() as patches:
                for target in ('tools.terminal_tool_lifecycle.cleanup_all_environments',
                               'tools.browser_session_lifecycle.cleanup_all_browsers',
                               'gateway.status.remove_pid_file', 'gateway.status.publish_runtime_status'):
                    patches.enter_context(patch(target))
                patches.enter_context(patch('tools.process_registry.process_registry.kill_all', return_value=0))
                patches.enter_context(patch('cron.scheduler.mark_running_jobs_interrupted', return_value=[]))
                asyncio.run(runner.stop())
        else:
            GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30, ctx)
        closed_at_gate = db._conn is None
    finally:
        release_worker.set(); release_persist.set()
        if ad._executor is not None:
            ad._executor.shutdown(wait=True)
        if ad._monitor_thread is not None:
            ad._monitor_stop.set(); ad._monitor_thread.join(30)
    persisted_state = db._read_one('SELECT state FROM async_delegations WHERE delegation_id=?', (h['delegation_id'],))[0]
    obs = {'worker_returns': worker_returns, 'full_stop': full_stop, 'before': before, 'closed_at_gate': closed_at_gate,
           'closes': closes, 'writes': writes, 'persisted_state_after_release': persisted_state}
    (tmp_path / ('monitor-'+str(worker_returns)+'-full-'+str(full_stop)+'.json')).write_text(json.dumps(obs, indent=2))
    db.close(); close_all(); ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    assert not closed_at_gate, obs
    assert not any(x['after_close'] for x in writes), obs
