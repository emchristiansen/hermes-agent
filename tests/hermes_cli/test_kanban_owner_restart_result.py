"""An attempt its owner recorded as lost to the owner's own restart requeues without a failure.

``session_kanban.record_owner_restart_result`` writes the run's ``worker_result`` (exit
``KANBAN_OWNER_RESTART_EXIT_CODE``) under the bound interpreter's pid and claim; the dead-worker
sweep then books ``owner_restarted`` instead of ``crashed``: no ``consecutive_failures``, no
systemic/sticky trip when many attempts end together, and a ``max_retries=1`` card keeps its
reconciliation hold without accounting. Real board SQLite; no dispatcher internals stubbed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gateway.session_kanban import record_owner_restart_result
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _running(conn, title, **kw):
    """A claimed card whose submitter and bound interpreter are both gone."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="w", **kw)
    task = kb.claim_task(conn, tid, claimer=f"{host}:owner")
    kbd._set_worker_pid(conn, tid, _dead_pid())
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": _dead_pid(), "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
    return tid, task


def _card(tid, task):
    return {"board": "default", "task_id": tid, "run_id": task.current_run_id}


def _board(conn):
    return conn.execute("PRAGMA database_list").fetchone()[2]


def test_three_attempts_lost_to_an_owner_restart_requeue_uncounted(conn):
    cards = [_running(conn, f"card {i}") for i in range(3)]
    for tid, task in cards:
        assert record_owner_restart_result(_card(tid, task), _board(conn))
        assert not record_owner_restart_result(_card(tid, task), _board(conn))  # never twice

    crashed = kbd.detect_crashed_workers(conn)

    assert crashed == []
    assert sorted(kbd.detect_crashed_workers._last_owner_restarted) == sorted(t for t, _ in cards)
    for tid, task in cards:
        run = conn.execute("SELECT outcome FROM task_runs WHERE id=?", (task.current_run_id,)).fetchone()[0]
        same = kb.get_task(conn, tid)
        assert (run, same.consecutive_failures, same.status) == ("owner_restarted", 0, "ready")


def test_first_failure_hold_stays_without_accounting(conn):
    tid, task = _running(conn, "held", max_retries=1)
    assert record_owner_restart_result(_card(tid, task), _board(conn))

    kbd.detect_crashed_workers(conn)

    same = kb.get_task(conn, tid)
    assert (same.status, same.consecutive_failures) == ("blocked", 0)
    blocked = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
                           (tid,)).fetchone()
    assert json.loads(blocked[0])["accounting_required"] is False


def test_an_attempt_without_the_owner_record_is_still_a_counted_crash(conn):
    tid, _task = _running(conn, "plain")

    assert kbd.detect_crashed_workers(conn) == [tid]

    assert kb.get_task(conn, tid).consecutive_failures == 1


def test_never_recorded_over_an_interpreter_result_or_for_a_stale_run(conn):
    tid, task = _running(conn, "own result")
    bound = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound'",
                                    (tid,)).fetchone()[0])
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_result", {"pid": bound["pid"], "claim_lock": task.claim_lock,
                         "exit_code": 130, "last_output": ""}, run_id=task.current_run_id)
    assert not record_owner_restart_result(_card(tid, task), _board(conn))
    other, other_task = _running(conn, "stale run")
    assert not record_owner_restart_result(dict(_card(other, other_task), run_id=other_task.current_run_id + 99), _board(conn))
