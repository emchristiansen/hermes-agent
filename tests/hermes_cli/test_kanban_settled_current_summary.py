"""Current-summary views of a settled delivered assessment (review7100 SC-02).

A ``settle-delivered`` settlement and any later ``edit --result`` correction are runless events, so a
view that reads only ``task_runs`` offers the ended run's scheduling reason as the card's current
summary. With an empty corrected result, CLI ``show`` printed it as "Latest summary" and the
dashboard's ``result || latest_summary`` promoted it to the final result. The views must read the
settlement's current handoff, and no run row may be rewritten or synthesized for it."""
import argparse
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_parser

SCHEDULED_REASON = "external recovery after terminal attempt"
VERDICT = "CHANGES_REQUIRED_SOURCE_ONLY: PA-05 open; report REVIEW-C7.md sha256 5d4c"


def _load_dashboard():
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_settled_summary", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(name="board")
def isolated_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kb, "_fire_task_hook", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.kanban_db_workspace._cleanup_worker_tmux", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.kanban_db_workspace.release_lsp_clients", lambda *a, **k: None)
    kb.init_db()
    # Outbound connections are forbidden; the TestClient's own event-loop socketpair is local.
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
            patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
        conn = kbc.connect()
        yield conn
        conn.close()


@pytest.fixture(name="dashboard")
def dashboard_client(board):
    app = FastAPI()
    app.include_router(_load_dashboard().router, prefix="/api/plugins/kanban")
    return TestClient(app)


def run(*arguments):
    parser = argparse.ArgumentParser()
    kanban_parser.build_parser(parser.add_subparsers(dest="command"))
    return cli.kanban_command(parser.parse_args(["kanban", *arguments]))


def outcome(capsys, *arguments):
    capsys.readouterr()
    rc = run(*arguments)
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


def claim_with_worker(conn, tid):
    task = kb.claim_task(conn, tid, claimer="dispatcher")
    assert task is not None
    dispatch._set_worker_pid(conn, tid, os.getpid(), run_id=task.current_run_id, claim_lock=task.claim_lock)
    kb._append_event(conn, tid, "worker_bound", {"pid": os.getpid(), "claim_lock": task.claim_lock,
                                                 "started_at": "fixture"}, run_id=task.current_run_id)
    conn.commit()
    return kb.get_task(conn, tid)


def settled_assessment(conn, capsys, *, prior_pass):
    """A reviewer run ends Scheduled with a reason, its worker delivers, and the CLI settles it; with
    ``prior_pass`` the card first completed with an older PASS and was reopened."""
    tid = kb.create_task(conn, title="review c7", assignee="reviewer", project_id="")
    if prior_pass:
        assert kb.complete_task(conn, tid, result="OLD_PASS", summary="OLD_PASS", metadata={"edition": "old"})
        assert _load_dashboard()._set_status_direct(conn, tid, "ready")  # the dashboard's reopen
    reviewer = claim_with_worker(conn, tid)
    assert kb.schedule_task(conn, tid, reason=SCHEDULED_REASON, expected_run_id=reviewer.current_run_id)
    kb._append_event(conn, tid, "worker_result", {"pid": os.getpid(), "claim_lock": reviewer.claim_lock,
                                                  "exit_code": 0, "last_output": VERDICT},
                     run_id=reviewer.current_run_id)
    conn.commit()
    child = kb.create_task(conn, title="consume the review", assignee="implementer", project_id="", parents=[tid])
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer.current_run_id),
                       "--verdict", "changes_required", "--result", VERDICT,
                       "--metadata", json.dumps({"edition": "settled"}))
    assert rc == 0, text
    return tid, child


def runs(conn, tid):
    return [dict(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,))]


def shown_current(text):
    """What CLI ``show`` presents as the card's current result: its Result, else its Latest summary."""
    for header in ("Result:\n", "Latest summary:\n"):
        if header in text:
            return header.rstrip(":\n"), text.split(header, 1)[1].split("\n", 1)[0].strip()
    return None, None


def current_correction(events):
    """The newest settlement correction as a view's raw events carry it."""
    edits = [e for e in events if e["kind"] == "edited"]
    payload = edits[-1]["payload"] if edits else None
    payload = json.loads(payload) if isinstance(payload, str) else payload
    return payload.get("settlement_correction") if payload else None


def dashboard_views(dashboard, tid, child):
    """The shipped dashboard's selections: drawer final result (``result || latest_summary``), the
    parent's child-result row as seen from a parent, and the board card preview."""
    drawer = dashboard.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
    child_view = dashboard.get(f"/api/plugins/kanban/tasks/{child}").json()
    board = dashboard.get("/api/plugins/kanban/board").json()
    card = next(t for col in board["columns"] for t in col["tasks"] if t["id"] == tid)
    return drawer, child_view, card


@pytest.mark.parametrize("prior_pass", [False, True], ids=["no_prior_run", "prior_pass"])
@pytest.mark.parametrize(("result", "summary", "expected"), [
    ("", "LATEST_NEGATIVE_CORRECTION", "LATEST_NEGATIVE_CORRECTION"),
    ("CORRECTED_NEGATIVE changes required", "CORRECTED_SUMMARY", "CORRECTED_SUMMARY"),
], ids=["empty_result", "nonempty_result"])
def test_views_show_the_settlement_correction_not_an_older_run_summary(
        board, dashboard, capsys, prior_pass, result, summary, expected):
    tid, child = settled_assessment(board, capsys, prior_pass=prior_pass)
    before = runs(board, tid)

    rc, text = outcome(capsys, "edit", tid, "--result", result, "--summary", summary,
                       "--metadata", json.dumps({"edition": "corrected"}))
    assert rc == 0, text
    assert runs(board, tid) == before  # no run rewritten or synthesized

    assert kb.latest_summary(board, tid) == expected
    assert kb.latest_summaries(board, [tid, child]) == {tid: expected}

    rc, text = outcome(capsys, "show", tid)
    assert rc == 0, text
    assert shown_current(text) == (("Result", result) if result else ("Latest summary", expected))
    assert SCHEDULED_REASON not in text.split("Events (", 1)[0]
    assert "OLD_PASS" not in text.split("Events (", 1)[0]
    rc, text = outcome(capsys, "show", tid, "--json")
    assert rc == 0, text
    shown = json.loads(text)
    assert shown["latest_summary"] == expected
    assert current_correction(shown["events"]) == {"summary": summary, "metadata": {"edition": "corrected"}}
    assert [r["outcome"] for r in shown["runs"]] == [r["outcome"] for r in before]

    drawer, _child_view, card = dashboard_views(dashboard, tid, child)
    assert drawer["result"] == result
    assert (drawer["result"] or drawer["latest_summary"]) == (result or expected)
    assert card["latest_summary"] == expected
    task_view = dashboard.get(f"/api/plugins/kanban/tasks/{tid}").json()
    assert current_correction(task_view["events"]) == {"summary": summary, "metadata": {"edition": "corrected"}}
    assert [r["id"] for r in task_view["runs"]] == [r["id"] for r in before]
    # The dependent reads the same current handoff and metadata (SI-10), unchanged by this repair.
    context = kb.build_worker_context(board, child)
    assert expected in context and "corrected" in context
    assert runs(board, tid) == before


@pytest.mark.parametrize("prior_pass", [False, True], ids=["no_prior_run", "prior_pass"])
def test_uncorrected_settlement_presents_its_settled_result(board, dashboard, capsys, prior_pass):
    tid, child = settled_assessment(board, capsys, prior_pass=prior_pass)
    before = runs(board, tid)
    assert kb.latest_summary(board, tid) == VERDICT
    assert kb.latest_summaries(board, [tid]) == {tid: VERDICT}
    rc, text = outcome(capsys, "show", tid)
    assert rc == 0 and shown_current(text) == ("Result", VERDICT)
    drawer, _child_view, card = dashboard_views(dashboard, tid, child)
    assert (drawer["result"] or drawer["latest_summary"]) == VERDICT
    assert card["latest_summary"] == VERDICT
    assert runs(board, tid) == before


def test_empty_correction_without_summary_shows_no_stale_run_summary(board, dashboard, capsys):
    tid, child = settled_assessment(board, capsys, prior_pass=False)
    before = runs(board, tid)
    rc, text = outcome(capsys, "edit", tid, "--result", "")
    assert rc == 0, text
    assert kb.latest_summary(board, tid) is None
    assert tid not in kb.latest_summaries(board, [tid])
    rc, text = outcome(capsys, "show", tid)
    assert rc == 0 and shown_current(text) == (None, None)
    drawer, _child_view, card = dashboard_views(dashboard, tid, child)
    assert drawer["result"] == "" and drawer["latest_summary"] is None  # `result || latest_summary` shows nothing stale
    assert card["latest_summary"] is None
    assert runs(board, tid) == before


def test_child_result_rows_read_the_settled_child_current_summary(board, dashboard, capsys):
    """A parent drawer's ``child_results`` uses ``latest_summaries`` for its children."""
    tid, _ = settled_assessment(board, capsys, prior_pass=True)
    parent = kb.create_task(board, title="umbrella", assignee="orchestrator", project_id="")
    kb.link_tasks(board, parent, tid)
    rc, text = outcome(capsys, "edit", tid, "--result", "", "--summary", "LATEST_NEGATIVE_CORRECTION")
    assert rc == 0, text
    rows = dashboard.get(f"/api/plugins/kanban/tasks/{parent}").json()["child_results"]
    row = next(r for r in rows if r["id"] == tid)
    assert (row["result"] or row["latest_summary"]) == "LATEST_NEGATIVE_CORRECTION"


def test_a_newer_run_summary_after_the_settlement_still_wins(board, capsys):
    """Only run summaries no newer than the settlement are superseded."""
    tid, _ = settled_assessment(board, capsys, prior_pass=False)
    settled_at = board.execute("SELECT MAX(created_at) AS t FROM task_events WHERE task_id=? AND kind='completed'",
                               (tid,)).fetchone()["t"]
    board.execute("INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, summary) "
                  "VALUES (?, 'reviewer', 'blocked', ?, ?, 'blocked', 'NEWER_RUN_SUMMARY')",
                  (tid, settled_at + 5, settled_at + 6))
    board.commit()
    assert kb.latest_summary(board, tid) == "NEWER_RUN_SUMMARY"
    assert kb.latest_summaries(board, [tid]) == {tid: "NEWER_RUN_SUMMARY"}


def test_an_ordinary_completion_keeps_reading_runs(board):
    tid = kb.create_task(board, title="ordinary", assignee="worker", project_id="")
    assert kb.complete_task(board, tid, result=None, summary="WORKER_HANDOFF")
    assert kb.latest_summary(board, tid) == "WORKER_HANDOFF"
    assert kb.latest_summaries(board, [tid]) == {tid: "WORKER_HANDOFF"}
