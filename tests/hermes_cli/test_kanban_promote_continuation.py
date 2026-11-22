"""``kanban promote --expected-terminal-run``: continue a READY card held by ``active_pr``.

A PR worker's run crashed (gateway restart) after its PR comment; the card re-queued itself
to ``ready`` and the dispatcher then held it under ``respawn_guarded(active_pr)`` with no
supported way out (2026-11-05, t_2dfcaea0): ``promote`` only took todo/blocked cards,
``unblock`` only blocked ones, and a same-profile reassign is deliberately no release. The
continuation names the exact latest ended run, keeps the card ``ready`` and records a
``promoted_manual`` (``from: ready``) the guard honours; every other shape is refused.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_parser import build_parser

PR_COMMENT = "Opened https://github.com/example/repo/pull/44 for review."


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


def _crashed_pr_card(conn, title="pr then crash") -> tuple[str, int]:
    """A ready card whose PR worker's latest run crashed after posting its PR."""
    tid = kb.create_task(conn, title=title, assignee="dev")
    run_id = kb.claim_task(conn, tid).current_run_id
    kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
    assert not kbd._record_task_failure(conn, tid, "worker crashed", outcome="crashed",
                                        failure_limit=5, release_claim=True, end_run=True)
    with kb.write_txn(conn):  # second-granularity stamps: the PR comment precedes what follows
        conn.execute("UPDATE task_comments SET created_at = created_at - 60 WHERE task_id = ?", (tid,))
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.latest_run(conn, tid).id == run_id
    return tid, run_id


def _continuations(conn, tid) -> list[dict]:
    rows = conn.execute("SELECT payload, run_id FROM task_events WHERE task_id = ? AND kind = 'promoted_manual'",
                        (tid,)).fetchall()
    return [dict(json.loads(r["payload"]), event_run_id=r["run_id"]) for r in rows]


def _spawned(conn) -> list[str]:
    return [s[0] for s in kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None).spawned]


def test_continuation_releases_crashed_ready_pr_card_and_dispatch_spawns_it(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid, run_id = _crashed_pr_card(conn)
        # Crash, reprioritize and a same-profile reassign keep the PR worker held.
        kb.edit_task(conn, tid, priority=50)
        kb.assign_task(conn, tid, "dev")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        res = kbd.dispatch_once(conn, dry_run=True)
        assert dict(res.respawn_guarded).get(tid) == "active_pr" and tid not in [s[0] for s in res.spawned]
        comments_before = len(kb.list_comments(conn, tid))
        kinds_before = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]

    assert _cli("promote", tid, "--dry-run", "--expected-terminal-run", str(run_id)) == 0
    assert _cli("promote", tid, "gateway restart crashed the PR run",
                "--expected-terminal-run", str(run_id)) == 0

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        (event,) = _continuations(conn, tid)  # the dry run wrote nothing
        assert event["from"] == "ready" and event["continuation"] is True
        assert event["expected_terminal_run"] == run_id == event["event_run_id"]
        assert event["reason"] == "gateway restart crashed the PR run" and event["actor"]
        # Nothing fabricated or removed: no block/unblock, reassign or review; PR comment kept.
        kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]
        assert kinds == kinds_before + ["promoted_manual"]
        assert len(kb.list_comments(conn, tid)) == comments_before
        assert kbd.check_respawn_guard(conn, tid) is None
        assert tid in _spawned(conn)
        assert kb.get_task(conn, tid).status == "running"


def test_newer_pr_comment_after_continuation_guards_again(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid, run_id = _crashed_pr_card(conn)
    assert _cli("promote", tid, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, tid) is None
        kb.add_comment(conn, tid, author="dev", body="Opened https://github.com/example/repo/pull/45")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        assert tid not in _spawned(conn)


def _admission_db(tmp_path: Path, status: str) -> dict:
    db = tmp_path / f"owner-{status}.db"
    with sqlite3.connect(db) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, "
                      "status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('s1', 'r1', ?, NULL)", (status,))
    return {"db": str(db), "session_id": "s1", "request_id": "r1", "pid": 1, "birth": 1.0}


def test_continuation_refusals_leave_the_card_guarded(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    with kbc.connect() as conn:
        tid, run_id = _crashed_pr_card(conn)
        older, older_run = _crashed_pr_card(conn, "two runs")  # a second crashed run supersedes the first
        newer_run = kb.claim_task(conn, older).current_run_id
        kbd._record_task_failure(conn, older, "worker crashed", outcome="crashed",
                                 failure_limit=5, release_claim=True, end_run=True)
        unended, unended_run = _crashed_pr_card(conn, "run not ended")
        claimed, claimed_run = _crashed_pr_card(conn, "live claim")
        executing, executing_run = _crashed_pr_card(conn, "owner executing")
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET ended_at = NULL WHERE id = ?", (unended_run,))
            conn.execute("UPDATE tasks SET claim_lock = 'elsewhere:1' WHERE id = ?", (claimed,))
            kb._append_event(conn, executing, "owner_admitted", _admission_db(tmp_path, "running"),
                             run_id=executing_run)
        parked = kb.create_task(conn, title="todo card", assignee="dev", parents=[kb.create_task(conn, title="p")])
        running = kb.create_task(conn, title="running card", assignee="dev")
        running_run = kb.claim_task(conn, running).current_run_id

    refused = [
        (tid, run_id + 1000, "not the latest run"),
        (older, older_run, "not the latest run"),
        (unended, unended_run, "has not ended"),
        (claimed, claimed_run, "live claim (claim_lock)"),
        (executing, executing_run, "may still be executing"),
        (parked, 0, "'todo'; a continuation applies only to 'ready'"),
        (running, running_run, "'running'; a continuation applies only to 'ready'"),
    ]
    for card, run, why in refused:
        capsys.readouterr()
        assert _cli("promote", card, "--expected-terminal-run", str(run)) == 1, card
        assert why in capsys.readouterr().err, (card, why)
    # Plain promote still refuses a ready card, and points at the fence.
    assert _cli("promote", tid) == 1
    assert "--expected-terminal-run" in capsys.readouterr().err
    # One card per fence.
    assert _cli("promote", tid, "--ids", older, "--expected-terminal-run", str(run_id)) == 2
    # A worker session without board management may not continue cards.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert _cli("promote", tid, "--expected-terminal-run", str(run_id)) == 1
    assert "orchestrator-only" in capsys.readouterr().err

    with kbc.connect() as conn:
        for card in (tid, older, unended, claimed, executing, parked, running):
            assert _continuations(conn, card) == []
        assert kb.get_task(conn, parked).status == "todo"
        assert kb.get_task(conn, running).status == "running"
        for card in (tid, older, claimed, executing):
            assert kbd.check_respawn_guard(conn, card) == "active_pr"
    # The same operands are accepted once the run is the latest and its owner is terminal.
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _cli("promote", older, "--expected-terminal-run", str(newer_run)) == 0
    with kbc.connect() as conn, kb.write_txn(conn):
        kb._append_event(conn, executing, "owner_admitted", _admission_db(tmp_path, "terminal"),
                         run_id=executing_run)
    assert _cli("promote", executing, "--expected-terminal-run", str(executing_run)) == 0
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, older) is None
        assert kbd.check_respawn_guard(conn, executing) is None
