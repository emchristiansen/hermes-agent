"""A quota/auth stamp holds the card only while the attempt that recorded it is still the newest.

``tasks.last_failure_error`` outlives the run that wrote it. Card t_39f19a6e: a rate-limited attempt
stamped its quota text, the cooldown elapsed and a new attempt was spawned, then the gateway restart
ended it ``owner_restarted`` (uncounted, stamps nothing). The rate-limit branch no longer saw a
``rate_limited`` latest run, so ``blocker_auth`` matched the old ``quota`` text and parked the card on
every tick before any claim. Through the real dead-worker sweep and the real dispatcher selection
(``dispatch_once(dry_run=True)``); real board SQLite, only the clock and the profile check are faked.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from gateway.session_kanban import record_owner_restart_result
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

COOLDOWN = 300


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """Real time plus a settable offset, so a cooldown can elapse without sleeping."""
    offset = [0]
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + offset[0])
    return offset


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_assignees_spawnable, clock):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", str(COOLDOWN))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _board(conn):
    return conn.execute("PRAGMA database_list").fetchone()[2]


def _attempt(conn, tid):
    """Claim and spawn an attempt whose submitter and bound interpreter are both gone."""
    host = kb._claimer_id().split(":", 1)[0]
    task = kb.claim_task(conn, tid, claimer=f"{host}:owner")
    kbd._set_worker_pid(conn, tid, _dead_pid())  # records ``spawned``
    bound = _dead_pid()
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
    return task, bound


def _rate_limited_attempt(conn, tid):
    """The interpreter's own result: exit 75 after the provider's HTTP 429 (run678)."""
    task, bound = _attempt(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_result", {
            "pid": bound, "claim_lock": task.claim_lock, "exit_code": kb.KANBAN_RATE_LIMIT_EXIT_CODE,
            "last_output": "Subscription rate-limited every one of 3 attempts (HTTP 429)",
        }, run_id=task.current_run_id)
    return task


def _owner_restarted_attempt(conn, tid):
    """The owner's own record of losing the turn to its restart: exit 79 (run720)."""
    task, _bound = _attempt(conn, tid)
    assert record_owner_restart_result({"board": "default", "task_id": tid, "run_id": task.current_run_id},
                                       _board(conn))
    return task


def _tick(conn, tid, clock):
    """One real dispatcher tick (dead-worker sweep, then selection) that spawns nothing, a minute
    after the last: distinct ``ended_at`` seconds, as on a live board."""
    clock[0] += 60
    res = kbd.dispatch_once(conn, dry_run=True)
    guarded = dict(res.respawn_guarded).get(tid)
    selected = tid in [t for t, _, _ in res.spawned]
    assert selected != (guarded is not None)
    return res, guarded


def _history(conn, tid):
    row = conn.execute("SELECT status, last_failure_error, consecutive_failures FROM tasks WHERE id = ?",
                       (tid,)).fetchone()
    return row["status"], row["last_failure_error"], row["consecutive_failures"]


def _outcome(conn, run_id):
    return conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0]


def test_owner_restart_after_a_rate_limit_does_not_revive_the_old_quota_hold(conn, clock):
    tid = kb.create_task(conn, title="core", assignee="w")

    # run655: an ordinary counted crash, so the history the fix must keep is non-trivial.
    crashed, _ = _attempt(conn, tid)
    res, _ = _tick(conn, tid, clock)
    assert tid in res.crashed and _outcome(conn, crashed.current_run_id) == "crashed"
    assert _history(conn, tid)[2] == 1

    # run678: quota wall. Requeued uncounted with the quota text stamped; held for the cooldown.
    limited = _rate_limited_attempt(conn, tid)
    res, guarded = _tick(conn, tid, clock)
    assert tid in res.rate_limited and _outcome(conn, limited.current_run_id) == "rate_limited"
    assert guarded == "rate_limit_cooldown"
    stamped = _history(conn, tid)
    assert stamped[0] == "ready" and "quota" in stamped[1] and stamped[2] == 1

    # Cooldown elapsed: selectable again, exactly as the live dispatcher spawned run720.
    clock[0] += COOLDOWN + 100
    assert _tick(conn, tid, clock)[1] is None

    # run720: spawned, then lost to the gateway restart. Uncounted; stamps nothing.
    restarted = _owner_restarted_attempt(conn, tid)
    res, guarded = _tick(conn, tid, clock)
    assert tid in res.owner_restarted and _outcome(conn, restarted.current_run_id) == "owner_restarted"
    # The old quota text diagnosed run678, not run720: the card is selected, not ``blocker_auth``.
    assert guarded is None
    assert kbd.check_respawn_guard(conn, tid) is None
    assert kbd.check_respawn_guard(conn, tid, lane="review") is None
    # History is kept: the guard decision changed, not the record.
    assert _history(conn, tid) == stamped
    assert not conn.execute("SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'respawn_guarded'"
                            " AND json_extract(payload, '$.reason') = 'blocker_auth'", (tid,)).fetchone()

    # A fresh quota wall on the NEXT attempt still holds for its own cooldown.
    again = _rate_limited_attempt(conn, tid)
    res, guarded = _tick(conn, tid, clock)
    assert tid in res.rate_limited and _outcome(conn, again.current_run_id) == "rate_limited"
    assert guarded == "rate_limit_cooldown"
    assert _history(conn, tid)[2] == 1
    clock[0] += COOLDOWN + 100
    assert _tick(conn, tid, clock)[1] is None


@pytest.mark.parametrize("after_restart", [False, True])
def test_the_latest_attempts_own_auth_failure_still_guards(conn, clock, after_restart):
    tid = kb.create_task(conn, title="bad key", assignee="w")

    if after_restart:
        # An older quota stamp superseded by an owner restart, as above...
        _rate_limited_attempt(conn, tid)
        _tick(conn, tid, clock)
        clock[0] += COOLDOWN + 100
        _owner_restarted_attempt(conn, tid)
        assert _tick(conn, tid, clock)[1] is None

    # ...the next attempt's own genuine auth failure, through the real spawn path.
    def spawn_rejected(task, workspace, board=None):
        raise RuntimeError("provider returned 401 Unauthorized: invalid API key")

    res = kbd.dispatch_once(conn, spawn_fn=spawn_rejected, failure_limit=5)
    assert res.auto_blocked == []
    run = conn.execute("SELECT outcome, error FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                       (tid,)).fetchone()
    assert run["outcome"] == "spawn_failed"
    status, err, failures = _history(conn, tid)
    assert (status, err, failures) == ("ready", run["error"], 1)

    assert _tick(conn, tid, clock)[1] == "blocker_auth"
    assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"
    assert _history(conn, tid) == (status, err, failures)


def test_a_stamp_no_run_recorded_still_guards_after_an_owner_restart(conn, clock):
    """Unknown provenance fails closed: ``record_acceptance`` stamps a PR acceptance refusal on the
    task only, and no run's ``error`` carries it, so a later owner restart cannot release it."""
    tid = kb.create_task(conn, title="acceptance", assignee="w")
    _owner_restarted_attempt(conn, tid)
    assert _tick(conn, tid, clock)[1] is None
    with kb.write_txn(conn):  # the statement ``kanban_pr_acceptance_store.record_acceptance`` runs
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?",
                     ("PR acceptance forbidden: gh api returned 403 for the assignee login", tid))
    _owner_restarted_attempt(conn, tid)
    assert _tick(conn, tid, clock)[1] == "blocker_auth"
    assert json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'owner_restarted' ORDER BY id DESC LIMIT 1",
        (tid,)).fetchone()[0])["exit_code"] == 79
