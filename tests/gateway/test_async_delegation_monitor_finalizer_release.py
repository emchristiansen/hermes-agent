"""V3-ASYNC-01 companion: once the monitor finalizer that kept the close open has finished its real persistence and
publication, nothing is left counted and the ordinary close happens (no unbounded wait, no leaked admission)."""
import threading
import time

from tools import async_delegation as ad
from tools import detached_writers as dw
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from gateway import run_shutdown_budget as budget
from tests.gateway.restart_test_helpers import make_restart_runner


def _gate(db):
    runner, _ = make_restart_runner()
    runner._session_db = db
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0, started_at=time.monotonic())
    ctx.api_live = 0
    started = time.monotonic()
    GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30, ctx)
    return time.monotonic() - started


def test_a_counted_monitor_finalizer_releases_the_close_once_its_completion_is_persisted(monkeypatch):
    from hermes_state_registry import acquire, close_all
    ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    db = acquire(ad._db_path())
    worker_entered, release_worker, persist_entered, release_persist, persisted = [threading.Event() for _ in range(5)]
    original_persist = ad._persist_completion

    def held_persist(event, result, **kwargs):  # v0.23.2 adds the keyword-only ``delivery``
        persist_entered.set()
        assert release_persist.wait(30)
        try:
            return original_persist(event, result, **kwargs)
        finally:
            persisted.set()

    def work():
        worker_entered.set()
        assert release_worker.wait(30)
        return {'status': 'completed'}

    monkeypatch.setattr(ad, '_persist_completion', held_persist)
    monkeypatch.setattr(ad, '_STALE_CHECK_INTERVAL', 0.01)
    h = ad.dispatch_async_delegation(goal='monitor finalizer release', context=None, toolsets=None,
        role='leaf', model=None, session_key='probe', runner=work, progress_fn=lambda: ('unchanged', False))
    assert h['status'] == 'dispatched'
    try:
        assert worker_entered.wait(30)
        with ad._records_lock:
            ad._records[h['delegation_id']].update(status='stalling', _interrupted_at=time.time() - 121)
        assert persist_entered.wait(30)
        release_worker.set()
        ad._executor.shutdown(wait=True)  # the original worker returned and settled its own admission
        ad.shutdown_all()
        assert ad.join_workers(0) == 1  # the monitor's finalization is the counted physical writer
        held = _gate(db)
        assert db._conn is not None, 'closed under the monitor finalizer'
        assert held < 25, held  # bounded by the quiesce budget, not by the held finalizer
        release_persist.set()
        assert persisted.wait(30)
        assert ad.join_workers(5) == 0  # settled after persistence and publication
        _gate(db)
        assert db._conn is None, 'the ordinary close did not happen once the finalizer finished'
    finally:
        release_worker.set(); release_persist.set()
        if ad._executor is not None:
            ad._executor.shutdown(wait=True)
        if ad._monitor_thread is not None:
            ad._monitor_stop.set(); ad._monitor_thread.join(30)
    state = db._read_one('SELECT state FROM async_delegations WHERE delegation_id=?', (h['delegation_id'],))[0]
    db.close(); close_all(); ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    assert state == 'stalled'
