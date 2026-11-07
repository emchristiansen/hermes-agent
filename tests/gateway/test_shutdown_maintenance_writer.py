"""A stop never closes the shared SessionDB under an in-flight state.db maintenance pass (L7).

``_housekeeping_state_db_maintenance`` (prune/archive/VACUUM) runs inline on the gateway-housekeeping
thread through the registry's shared SessionDB, and the bootstrap stops that thread only after
``runner.stop()`` returns. No close guard counted it, so a stop could close (and ``close_all``
force-close) the handle between two of the pass's writes; the next write then reopened it after the
quiescence boundary (#94736), the same post-close-write class as A3-E5-01. These run the real
maintenance function on a real registered SessionDB against the real stop and close decision; only
the ``sessions:`` config and a pause inside the pass are synthetic.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner

_SESSIONS = {"auto_prune": True, "retention_days": 0, "min_interval_hours": 0,
             "min_vacuum_interval_days": 0, "vacuum_after_prune": True}


@pytest.fixture(autouse=True)
def _clean_state():
    from hermes_state_registry import close_all
    import gateway.run_shutdown_budget as budget
    reset = getattr(budget, "_reset_maintenance_for_tests", lambda: None)  # absent before the fix
    reset()
    yield
    close_all()
    reset()


def _prunable_shared_db():
    from hermes_state_registry import acquire
    db = acquire()  # the profile's registry handle: the gateway's and the maintenance pass's
    for sid in ("old-1", "old-2"):
        db.create_session(sid, "cli")
        db.append_message(sid, "user", "hello")
        db.end_session(sid, "done")
    db.set_meta("last_auto_prune", "0")
    return db


async def _stop_during_maintenance(monkeypatch, *, release_at_quiesce=False, vacuum=True):
    """Pause a real maintenance pass right after its prune (its later writes still to come), stop
    the gateway, then let the pass finish. ``release_at_quiesce`` lets it finish as soon as the
    stop's executor quiesce begins (a pass that ends inside the close's wait); ``vacuum=False`` keeps
    the VACUUM admission's process-table holder scan (seconds on a busy host) out of that pass."""
    from gateway.run import GatewayRunner, _housekeeping_state_db_maintenance
    from hermes_cli import config as config_mod

    config_reads = []  # one per maintenance pass that got past its start (its first step reads config)

    def load_config():
        config_reads.append(threading.current_thread().name)
        return {"sessions": dict(_SESSIONS, vacuum_after_prune=vacuum)}

    monkeypatch.setattr(config_mod, "load_config", load_config)
    db = _prunable_shared_db()
    in_pass, release, done = threading.Event(), threading.Event(), threading.Event()
    writes, closes, pruned = [], [], []
    real_prune, real_write, real_close = db.prune_sessions, db._execute_write, db.close

    def paused_prune(*a, **kw):
        pruned.append(real_prune(*a, **kw))
        in_pass.set()
        assert release.wait(30), "scheduling bound exceeded"
        return pruned[-1]

    def observed_write(fn, *a, **kw):
        writes.append({"after_close": bool(closes), "thread": threading.current_thread().name})
        return real_write(fn, *a, **kw)

    def observed_close(*a, **kw):
        closes.append({"pass_done": done.is_set()})
        return real_close(*a, **kw)

    monkeypatch.setattr(db, "prune_sessions", paused_prune)
    monkeypatch.setattr(db, "_execute_write", observed_write)
    monkeypatch.setattr(db, "close", observed_close)

    def maintenance():
        try:
            _housekeeping_state_db_maintenance()
        finally:
            done.set()

    real_quiesce = GatewayRunner._shutdown_executor

    def quiesce(runner, drain_timeout=0.0):
        if release_at_quiesce:
            release.set()
        return real_quiesce(runner, drain_timeout=drain_timeout)

    housekeeping = threading.Thread(target=maintenance, name="gateway-housekeeping", daemon=True)
    with patch.object(GatewayRunner, "_shutdown_executor", quiesce), \
            patch("tools.process_registry.process_registry.kill_all", return_value=0), \
            patch("cron.scheduler.mark_running_jobs_interrupted", return_value=[]), \
            patch("tools.terminal_tool_lifecycle.cleanup_all_environments"), \
            patch("tools.browser_tool_lifecycle.cleanup_all_browsers"), \
            patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
        housekeeping.start()
        assert await asyncio.to_thread(in_pass.wait, 30)
        runner, _ = make_restart_runner()
        runner._restart_drain_timeout = 0.0
        runner._session_db = db
        try:
            await runner.stop()
            at_stop = {"pass_done": done.is_set(), "connection_closed": db._conn is None}
        finally:
            release.set()
        await asyncio.to_thread(housekeeping.join, 30)
        reads_before = len(config_reads)
        _housekeeping_state_db_maintenance()  # a pass the housekeeping thread would start after the stop
    observation = {"at_stop": at_stop, "closes": closes, "pruned": pruned,
                   "pass_after_stop_ran": len(config_reads) > reads_before,
                   "post_close_writes": [w for w in writes if w["after_close"]],
                   "integrity_check": db._read_one("PRAGMA integrity_check")[0]}
    return observation


@pytest.mark.asyncio
async def test_shutdown_does_not_close_shared_db_under_an_in_flight_maintenance_pass(monkeypatch):
    observation = await _stop_during_maintenance(monkeypatch)
    assert observation["pruned"] and observation["pruned"][0] >= 1, observation
    assert not observation["at_stop"]["connection_closed"], observation
    assert not observation["post_close_writes"], observation
    assert observation["integrity_check"] == "ok", observation


@pytest.mark.asyncio
async def test_a_maintenance_pass_that_ends_within_the_quiesce_lets_the_close_run(monkeypatch):
    observation = await _stop_during_maintenance(monkeypatch, release_at_quiesce=True, vacuum=False)
    assert observation["at_stop"]["pass_done"] and observation["at_stop"]["connection_closed"], observation
    assert not observation["post_close_writes"], observation


@pytest.mark.asyncio
async def test_no_maintenance_pass_starts_once_the_close_has_been_decided(monkeypatch):
    observation = await _stop_during_maintenance(monkeypatch, release_at_quiesce=True, vacuum=False)
    assert not observation["pass_after_stop_ran"], observation
