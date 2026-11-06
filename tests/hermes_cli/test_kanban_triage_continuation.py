"""``kanban promote --expected-terminal-run`` on a TRIAGE card: the satisfied-input continuation.

A worker's run tripped the unblock-loop breaker (``block_loop_detected`` -> ``triage``) and was
then interrupted, so its bound interpreter never wrote a ``worker_result`` (2026-11-05,
t_fc2f7ba0 run 3114). ``settle-triage`` needs that result, ``unblock`` takes blocked/scheduled
cards, plain ``promote`` todo/blocked ones and the exact-run continuation only ready ones: the
card had no fenced way back to its owner once the input it waited for arrived. The same
continuation now takes that exact shape, with positive proof that no writer of the run is left,
and records the interrupted run and the missing result; every other shape is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_parser import build_parser

DEAD_PID = 4_194_305  # above Linux's largest pid_max: never a live process
INPUT = "credential delivered in comment c-4242"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kb.init_db()
    return home


def _cli(*argv: str) -> int:
    """The real ``hermes kanban`` parser and entry point."""
    wrap = argparse.ArgumentParser(add_help=False)
    return kb_cli.kanban_command(build_parser(wrap.add_subparsers(dest="_top")).parse_args(list(argv)))


def _admission(tmp_path: Path, status: str) -> dict:
    db = tmp_path / f"owner-{status}-{len(list(tmp_path.glob('owner-*.db')))}.db"
    with sqlite3.connect(db) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, "
                      "status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('s1', 'r1', ?, 'interrupted')", (status,))
    return {"db": str(db), "session_id": "s1", "request_id": "r1", "pid": DEAD_PID, "birth": 1.0}


def _claim(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    task = kb.get_task(conn, tid)
    return task.current_run_id, task.claim_lock


def _interrupted_loop(conn, tmp_path: Path, *, title="interrupted loop", admission="terminal") -> tuple[str, int]:
    """A triage card whose loop run's owner admitted it, bound a (now dead) interpreter, tripped
    the breaker, and was interrupted before the interpreter recorded a result."""
    tid = kb.create_task(conn, title=title, body="original body", assignee="worker")
    _claim(conn, tid)
    assert kb.block_task(conn, tid, kind="transient", reason="waiting for credential")
    assert kb.unblock_task(conn, tid)
    run_id, claim = _claim(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "owner_admitted", _admission(tmp_path, admission), run_id=run_id)
        kb._append_event(conn, tid, "worker_bound", {
            "pid": DEAD_PID, "claim_lock": claim, "started_at": "boot-1|12345"}, run_id=run_id)
    assert kb.block_task(conn, tid, kind="transient", reason="waiting for credential")
    task = kb.get_task(conn, tid)
    assert task.status == "triage" and task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
    assert kb.latest_run(conn, tid).id == run_id
    return tid, run_id


def _continuations(conn, tid) -> list[dict]:
    rows = conn.execute("SELECT payload, run_id FROM task_events WHERE task_id = ? AND kind = 'promoted_manual'",
                        (tid,)).fetchall()
    return [dict(json.loads(r["payload"]), event_run_id=r["run_id"]) for r in rows]


def _task_row(conn, tid) -> dict:
    return dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())


def test_interrupted_triage_loop_continues_to_ready_with_honest_history(kanban_home: Path, tmp_path: Path) -> None:
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        kb.add_comment(conn, tid, author="operator", body=INPUT)
        before = _task_row(conn, tid)
        runs = [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")]
        events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        comments = len(kb.list_comments(conn, tid))

    assert _cli("promote", tid, INPUT, "--dry-run", "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert _task_row(conn, tid) == before  # the dry run wrote nothing
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0

    with kbc.connect() as conn:
        after = _task_row(conn, tid)
        # Only the status moves: assignee, title, body, block kind/recurrences, failure counters kept.
        assert after.pop("status") == "ready" and before.pop("status") == "triage"
        assert after == before
        # History is appended to, never rewritten: no run is closed, synthesized or edited.
        assert [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")] == runs
        all_events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        assert all_events[:len(events)] == events
        kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,))]
        assert kinds[-1] == "promoted_manual" and len(all_events) == len(events) + 1
        (event,) = _continuations(conn, tid)
        assert event["from"] == "triage" and event["continuation"] is True
        assert event["interrupted_run"] == event["expected_terminal_run"] == run_id == event["event_run_id"]
        assert event["worker_result_recorded"] is False
        assert event["reason"] == INPUT and event["actor"]
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='worker_result'",
                            (tid,)).fetchone()[0] == 0  # nothing fabricated
        assert len(kb.list_comments(conn, tid)) == comments
        # The same owner picks it up; the kept loop memory returns a same-cause re-block to triage.
        spawned = [s[0] for s in kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None).spawned]
        assert tid in spawned
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.assignee == "worker"
        assert kb.block_task(conn, tid, kind="transient", reason="still waiting")
        assert kb.get_task(conn, tid).status == "triage"


def test_recycled_bound_pid_is_not_a_live_writer(kanban_home: Path, tmp_path: Path) -> None:
    """A live process that owns the number but not the bound start identity is not the writer."""
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_set(payload, '$.pid', ?) "
                         "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), tid))
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def _live_fingerprint() -> str:
    fingerprint = kbd._process_fingerprint(os.getpid())
    assert fingerprint
    return fingerprint


REFUSALS = {
    "wrong_run": "is not the latest run",
    "newer_run": "is not the latest run",
    "open_run": "has an open run",
    "claim_lock": "live claim (claim_lock)",
    "claim_expires": "live claim (claim_expires)",
    "worker_pid": "live claim (worker_pid)",
    "current_run": "live claim (current_run_id)",
    "live_bound_worker": "may still be live",
    "live_submitter": "may still be live",
    "unverified_live_worker": "may still be live",
    "blocked_not_triage": "is 'blocked'; a continuation applies only to 'ready' or an interrupted 'triage'",
    "todo_not_triage": "is 'todo'; a continuation applies only to 'ready' or an interrupted 'triage'",
    "worker_result": "has a worker_result; its finished loop is settled with settle-triage",
    "no_loop": "no block_loop_detected",
    "run_not_blocked": "did not end blocked",
    "no_bound": "no bound worker under its claim",
    "foreign_bound": "no bound worker under its claim",
    "owner_running": "owner admission is not terminal yet (running)",
    "no_owner": "has no owner admission",
    "empty_reason": "must name the awaited input",
    "parent_open": "unsatisfied parent dependencies",
}


@pytest.mark.parametrize("drift", sorted(REFUSALS))
def test_triage_continuation_refusals_write_nothing(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture, drift: str,
) -> None:
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path, admission="running" if drift == "owner_running" else "terminal")
        live = _live_fingerprint()
        mutations = {
            "wrong_run": [],
            "newer_run": [("INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
                           "VALUES (?, 'blocked', 1, 2, 'blocked')", (tid,))],
            "open_run": [("UPDATE task_runs SET ended_at=NULL WHERE id=?", (run_id,))],
            "claim_lock": [("UPDATE tasks SET claim_lock='other:1' WHERE id=?", (tid,))],
            "claim_expires": [("UPDATE tasks SET claim_expires=1 WHERE id=?", (tid,))],
            "worker_pid": [("UPDATE tasks SET worker_pid=? WHERE id=?", (DEAD_PID, tid))],
            "current_run": [("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))],
            "live_bound_worker": [("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                                   "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), live, tid))],
            "live_submitter": [("UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
                                (os.getpid(), None, run_id))],
            "unverified_live_worker": [("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                                        "WHERE task_id=? AND kind='worker_bound'",
                                        (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, tid))],
            "blocked_not_triage": [("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))],
            "todo_not_triage": [("UPDATE tasks SET status='todo' WHERE id=?", (tid,))],
            "worker_result": [],
            "no_loop": [("DELETE FROM task_events WHERE task_id=? AND kind='block_loop_detected'", (tid,))],
            "run_not_blocked": [("UPDATE task_runs SET status='crashed', outcome='crashed' WHERE id=?", (run_id,))],
            "no_bound": [("DELETE FROM task_events WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "foreign_bound": [("UPDATE task_events SET payload=json_set(payload, '$.claim_lock', 'foreign') "
                               "WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "owner_running": [],
            "no_owner": [("DELETE FROM task_events WHERE task_id=? AND kind='owner_admitted'", (tid,))],
            "empty_reason": [],
            "parent_open": [],
        }
        with kb.write_txn(conn):
            for sql, params in mutations[drift]:
                conn.execute(sql, params)
            if drift == "worker_result":
                claim = conn.execute("SELECT claim_lock FROM task_runs WHERE id=?", (run_id,)).fetchone()[0]
                kb._append_event(conn, tid, "worker_result", {"pid": DEAD_PID, "claim_lock": claim, "exit_code": 0},
                                 run_id=run_id)
        if drift == "parent_open":
            kb.link_tasks(conn, parent_id=kb.create_task(conn, title="open parent"), child_id=tid)
        snapshot = list(conn.iterdump())

    run = {"wrong_run": run_id - 1}.get(drift, run_id)
    reason = [] if drift == "empty_reason" else [INPUT]
    for dry in ([], ["--dry-run"]):
        capsys.readouterr()
        assert _cli("promote", tid, *reason, *dry, "--expected-terminal-run", str(run)) == 1
        err = capsys.readouterr().err
        assert REFUSALS[drift] in err, (drift, err)
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


def test_triage_continuation_is_orchestrator_only_and_one_card(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        other, _ = _interrupted_loop(conn, tmp_path, title="second")
        snapshot = list(conn.iterdump())
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 1
    assert "orchestrator-only" in capsys.readouterr().err
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _cli("promote", tid, INPUT, "--ids", other, "--expected-terminal-run", str(run_id)) == 2
    # Plain promote still refuses a triage card, and points at the fence.
    assert _cli("promote", tid, INPUT) == 1
    assert "an interrupted 'triage' loop: --expected-terminal-run" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


def test_settle_triage_contract_unchanged_beside_the_continuation(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """settle-triage still needs the run's positive worker result, and still parks the card in a
    sticky external wait; the continuation never takes a card that has one."""
    monkeypatch.setattr(kb_cli, "_profile_author", lambda: "orchestrator")
    settle = ["settle-triage", "--kind", "needs_input", "--owner", "operator",
              "--required-input", "credential", "--event", "operator delivers the credential"]
    with kbc.connect() as conn:
        interrupted, interrupted_run = _interrupted_loop(conn, tmp_path)
        finished, finished_run = _interrupted_loop(conn, tmp_path, title="finished loop")
        claim = conn.execute("SELECT claim_lock FROM task_runs WHERE id=?", (finished_run,)).fetchone()[0]
        with kb.write_txn(conn):
            kb._append_event(conn, finished, "worker_result", {"pid": DEAD_PID, "claim_lock": claim, "exit_code": 0},
                             run_id=finished_run)
        snapshot = list(conn.iterdump())
    # The interrupted loop has no result: settle-triage refuses it exactly as before.
    assert _cli(settle[0], interrupted, "--expected-terminal-run", str(interrupted_run), *settle[1:]) == 1
    assert "exact loop run lacks a matching terminal worker result" in capsys.readouterr().err
    # The finished loop is settle-triage's: the continuation refuses it.
    assert _cli("promote", finished, INPUT, "--expected-terminal-run", str(finished_run)) == 1
    assert "settled with settle-triage" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
    assert _cli(settle[0], finished, "--expected-terminal-run", str(finished_run), *settle[1:]) == 0
    with kbc.connect() as conn:
        task = kb.get_task(conn, finished)
        assert task.status == "blocked" and task.block_kind == "needs_input"
        assert _continuations(conn, finished) == []
        # Its wait is sticky: dispatch does not release it.
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None)
        assert kb.get_task(conn, finished).status == "blocked"
