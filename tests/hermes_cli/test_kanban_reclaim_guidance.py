"""The worker guidance on claim expiry and stale-run recovery states what the reclaim paths commit.

Each case drives a real reclaim (real SQLite, the real sweeps; the process readers are a private map and
the signal hook only records) and checks the committed outcome together with the sentence of
``KANBAN_GUIDANCE`` that describes it, so the text cannot drift from ``reclaim_transaction``.
"""

import time

import pytest

from agent.prompt_builder import KANBAN_GUIDANCE
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

STALE = ("A successful stale-run reclaim, of work past `kanban.dispatch_stale_timeout_seconds` (default four hours; "
         "0 disables) without a heartbeat in the last hour, returns the task to its prior phase without counting a "
         "failure, except that a `max_retries=1` card is held blocked.")
EXPIRED = "A successful reclaim of an expired claim counts as a failure toward the retry limit."
ELIGIBLE = ("Expiry or staleness makes a run eligible, not reclaimed: recovery may instead renew or defer it, for "
            "example while your host-local worker is alive with a missing or at-most-hour-old heartbeat, or survives "
            "termination.")


@pytest.fixture
def world(tmp_path, monkeypatch):
    from gateway import drain_control, status
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd.time, "sleep", lambda _s: None)
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "boot-guidance:1")
    procs = {12345: 100}
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid in procs)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: procs.get(pid))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn, procs
    finally:
        conn.close()


def _running(conn, max_retries, *, heartbeat_age):
    tid = kb.create_task(conn, title="guidance", assignee="worker")
    assert kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, 12345)
    now = int(time.time())
    old, hb = now - 86400, now - heartbeat_age
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_retries=?, started_at=?, claim_expires=?, last_heartbeat_at=? WHERE id=?",
                     (max_retries, old, old, hb, tid))
        conn.execute("UPDATE task_runs SET started_at=?, claim_expires=?, last_heartbeat_at=? WHERE task_id=?",
                     (old, old, hb, tid))
    return tid


def _killing(procs):
    def sig(pid, _signal):
        procs.pop(pid, None)
    return sig


def test_guidance_states_each_recovery_distinction():
    assert "heartbeat at least hourly; heartbeats also renew your claim." in KANBAN_GUIDANCE
    assert ("Claim expiry and stale-run recovery are distinct, and only a successful reclaim ends your ownership of "
            "that run.") in KANBAN_GUIDANCE
    for sentence in (ELIGIBLE, EXPIRED, STALE):
        assert sentence in KANBAN_GUIDANCE
    assert "without a failure penalty" not in KANBAN_GUIDANCE


@pytest.mark.parametrize("max_retries,status", [(None, "ready"), (1, "blocked")],
                         ids=["default-limit", "max_retries=1"])
def test_stale_reclaim_returns_the_prior_phase_uncounted_except_a_first_failure_card(world, max_retries, status):
    conn, procs = world
    tid = _running(conn, max_retries, heartbeat_age=7200)
    assert kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=_killing(procs)) == [tid]
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.consecutive_failures) == (status, None, 0)
    assert STALE in KANBAN_GUIDANCE


def test_expired_claim_reclaim_counts_a_failure(world):
    conn, procs = world
    tid = _running(conn, None, heartbeat_age=7200)
    assert kb.release_stale_claims(conn, signal_fn=_killing(procs)) == 1
    task = kb.get_task(conn, tid)
    assert task.current_run_id is None and task.consecutive_failures == 1
    assert EXPIRED in KANBAN_GUIDANCE


def test_an_expired_claim_of_a_live_worker_with_a_fresh_heartbeat_is_renewed_not_reclaimed(world):
    conn, procs = world
    tid = _running(conn, None, heartbeat_age=60)
    run = kb.get_task(conn, tid).current_run_id
    assert kb.release_stale_claims(conn, signal_fn=_killing(procs)) == 0
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.consecutive_failures) == ("running", run, 0)
    assert task.claim_expires > int(time.time())
    assert ELIGIBLE in KANBAN_GUIDANCE
