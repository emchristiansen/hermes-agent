"""``settle-delivered``: a Scheduled card whose ended run's own worker already delivered a negative
assessment is settled Done without readying, dispatching or rewriting that run.

2026-11-07, t_234158d0: run 6474 ended rate-limited; reviewer run 6493 was claimed; a stale
orchestrator ``schedule`` ended 6493; its worker kept going, delivered CHANGES_REQUIRED and exited 0,
but ``complete`` refuses ``scheduled`` (and the tool reported 6474's quota error). These tests replay
that history on an isolated board and settle it through the real CLI and kernel."""
import argparse
import hashlib
import json
import os
import socket
from unittest.mock import patch

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_parser

VERDICT = "CHANGES_REQUIRED_SOURCE_ONLY: PA-05 open; report REVIEW-C7.md sha256 5d4c"


@pytest.fixture(name="board")
def isolated_board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    path = tmp_path / "settle.sqlite"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kb, "_fire_task_hook", lambda *a, **k: None)
    # Workspace cleanup is real (rmtree, attachment copies); only its tmux/LSP side effects are stubbed.
    monkeypatch.setattr("hermes_cli.kanban_db_workspace._cleanup_worker_tmux", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.kanban_db_workspace.release_lsp_clients", lambda *a, **k: None)
    with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
        conn = kbc.connect(path)
        yield conn
        conn.close()


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
    # What gateway/session_kanban.py records when it binds the attempt's interpreter.
    kb._append_event(conn, tid, "worker_bound", {"pid": os.getpid(), "claim_lock": task.claim_lock,
                                                 "started_at": "fixture"}, run_id=task.current_run_id)
    conn.commit()
    return kb.get_task(conn, tid)


def worker_result(conn, tid, run_id, claim_lock, exit_code):
    """The bound worker's own result, shaped as gateway/session_kanban.py:_record_worker_result writes it."""
    kb._append_event(conn, tid, "worker_result", {"pid": os.getpid(), "claim_lock": claim_lock,
                                                  "exit_code": exit_code, "last_output": VERDICT}, run_id=run_id)
    conn.commit()


def owner_admission(conn, tid, run_id, tmp_path, status, outcome_):
    import sqlite3
    path = tmp_path / f"owner-{run_id}-{status}.sqlite"
    with sqlite3.connect(path) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('reviewer', 'request', ?, ?)", (status, outcome_))
    kb._append_event(conn, tid, "owner_admitted", {"db": str(path), "session_id": "reviewer", "request_id": "request"},
                     run_id=run_id)
    conn.commit()
    return path


def delivered_after_stale_schedule(conn, exit_code=0, tid=None, **create):
    """Run A ends rate-limited; reviewer run B is claimed; B ends ``scheduled`` (on the installed
    board a stale orchestrator schedule did that; with schedule-CAS c2 only B's own yield can, so the
    fixture yields as B's worker); B's worker then records its own result."""
    tid = tid or kb.create_task(conn, title="review c7", assignee="reviewer", project_id="", **create)
    attempt = claim_with_worker(conn, tid)
    first = attempt.current_run_id
    worker_result(conn, tid, first, attempt.claim_lock, 75)  # run A's own worker: quota wall (6474's exit 75)
    dispatch._record_task_failure(conn, tid, "pid 1767411 exited rate-limited (quota wall)", outcome="rate_limited",
                                  failure_limit=99, release_claim=True, end_run=True)
    reviewer = claim_with_worker(conn, tid)
    assert kb.schedule_task(conn, tid, reason="external recovery after terminal attempt",
                            expected_run_id=reviewer.current_run_id)
    if exit_code is not None:
        worker_result(conn, tid, reviewer.current_run_id, reviewer.claim_lock, exit_code)
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id) == ("scheduled", None)
    return tid, first, reviewer.current_run_id


def image(conn, tid):
    def rows(sql):
        return [dict(r) for r in conn.execute(sql, (tid,))]
    return (rows("SELECT * FROM tasks WHERE id=?"), rows("SELECT * FROM task_runs WHERE task_id=? ORDER BY id"),
            rows("SELECT * FROM task_comments WHERE task_id=? ORDER BY id"),
            rows("SELECT * FROM task_events WHERE task_id=? ORDER BY id"))


def settle(tid, run_id, *extra):
    return ("settle-delivered", tid, "--expected-terminal-run", str(run_id), "--verdict", "changes_required",
            "--result", VERDICT, *extra)


def test_delivered_negative_assessment_is_settled_without_rewriting_its_run(board, capsys, tmp_path):
    tid, _first, reviewer = delivered_after_stale_schedule(board)
    owner_admission(board, tid, reviewer, tmp_path, "terminal", "completed")
    child = kb.create_task(board, title="consume the review", assignee="implementer", project_id="", parents=[tid])
    assert kb.get_task(board, child).status == "todo"
    _, runs, comments, events = image(board, tid)

    rc, text = outcome(capsys, *settle(tid, reviewer, "--metadata", json.dumps({"report_sha256": "5d4c"})))
    assert rc == 0, text
    task = kb.get_task(board, tid)
    assert (task.status, task.result, task.current_run_id) == ("done", VERDICT, None)
    assert task.completed_at is not None and task.last_failure_error is None
    _, after_runs, after_comments, after_events = image(board, tid)
    assert after_runs == runs  # run B keeps status/outcome scheduled; no run synthesized
    assert after_comments == comments
    assert after_events[:len(events)] == events
    added = after_events[len(events):]
    assert [e["kind"] for e in added] == ["completed"]
    assert added[0]["run_id"] is None
    settlement = json.loads(added[0]["payload"])["settlement"]
    assert settlement == {"kind": "delivered_assessment", "actor": cli._audit_author(), "verdict": "changes_required",
                          "expected_terminal_run": reviewer, "from_status": "scheduled",
                          "metadata": {"report_sha256": "5d4c"}}
    assert kb.get_task(board, child).status == "ready"  # dependents release as on any completion
    assert run(*settle(tid, reviewer)) == 1  # settled once: now done


def test_complete_still_refuses_the_scheduled_card_and_names_its_state(board, monkeypatch):
    from tools import kanban_tools as kt

    tid, _first, reviewer = delivered_after_stale_schedule(board)
    before = image(board, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer))
    out = json.loads(kt._handle_complete({"summary": VERDICT}))
    assert f"status is scheduled, current run None, this worker's run {reviewer}" in out["error"]
    assert "quota wall" not in out["error"]
    assert image(board, tid) == before


def _stale(conn, tid, first, reviewer, **_):
    return first, f"run {first} is not the latest attempt (latest is {reviewer})"


def _newer_run(conn, tid, first, reviewer, **_):
    assert kb.unblock_task(conn, tid)
    newer = claim_with_worker(conn, tid).current_run_id
    assert kb.schedule_task(conn, tid, reason="park again", expected_run_id=newer)
    return reviewer, f"run {reviewer} is not the latest attempt (latest is {newer})"


def _active_run(conn, tid, first, reviewer, **_):
    assert kb.unblock_task(conn, tid)
    claim_with_worker(conn, tid)
    return reviewer, "status is running, not scheduled"


def _no_clean_result(conn, tid, first, reviewer, **_):
    return reviewer, f"run {reviewer} has no clean result (exit 0) from its own claimed worker"


def _owner_not_terminal(conn, tid, first, reviewer, tmp_path, **_):
    owner_admission(conn, tid, reviewer, tmp_path, "started", None)
    return reviewer, f"run {reviewer} owner admission is not terminal yet (started)"


def _worker_scope(conn, tid, first, reviewer, monkeypatch, **_):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer))
    return reviewer, "orchestrator-only"


# shape -> (the reviewer worker's recorded exit code, or None for no result; arrange -> (expected run, reason))
REFUSALS = {
    "stale": (0, _stale), "newer_run": (0, _newer_run), "active_run": (0, _active_run),
    "no_clean_result": (None, _no_clean_result), "failed_worker": (1, _no_clean_result),
    "owner_not_terminal": (0, _owner_not_terminal), "worker_scope": (0, _worker_scope),
}


@pytest.mark.parametrize("shape", sorted(REFUSALS))
def test_refusals_change_nothing(board, capsys, monkeypatch, tmp_path, shape):
    exit_code, arrange = REFUSALS[shape]
    tid, first, reviewer = delivered_after_stale_schedule(board, exit_code=exit_code)
    expected, reason = arrange(board, tid, first, reviewer, tmp_path=tmp_path, monkeypatch=monkeypatch)
    before = image(board, tid)
    rc, text = outcome(capsys, *settle(tid, expected))
    assert rc == 1 and reason in text, text
    assert len(text) < 400
    assert image(board, tid) == before


def test_malformed_settlement_is_an_explicit_error(board, capsys):
    from hermes_cli.kanban_db_settle import settle_delivered_assessment

    tid, _first, reviewer = delivered_after_stale_schedule(board)
    before = image(board, tid)
    with pytest.raises(SystemExit) as passing:
        run("settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict", "pass", "--result", "ok")
    assert passing.value.code == 2
    assert outcome(capsys, *settle(tid, 0))[0] == 2
    assert outcome(capsys, *settle(tid, reviewer, "--metadata", "[1]"))[0] != 0
    for bad in ({"expected_terminal_run": True}, {"verdict": "approved"}, {"result": "  "}, {"metadata": [1]}):
        arguments = {"expected_terminal_run": reviewer, "verdict": "changes_required", "result": VERDICT,
                     "author": "orchestrator", **bad}
        with pytest.raises(ValueError):
            settle_delivered_assessment(board, tid, **arguments)
    assert image(board, tid) == before


def dashboard_api():
    import importlib.util
    import sys
    from pathlib import Path

    plugin = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("settle_delivered_dashboard", plugin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scratch_report(home, name):
    workspace = home / "kanban" / "workspaces" / name
    workspace.mkdir(parents=True)
    report = workspace / "REVIEW.md"
    report.write_bytes(b"CHANGES_REQUIRED: reviewer evidence\n")
    return workspace, report, hashlib.sha256(report.read_bytes()).hexdigest()


def test_si01_a_report_named_only_in_the_result_survives_cleanup(board, capsys, tmp_path):
    """SI-01: as complete_task does, a scratch deliverable named only by path in --result becomes a
    byte-identical attachment before the workspace is removed."""
    workspace, report, digest = scratch_report(tmp_path / "home", "negative-assessment")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED {report} sha256 {digest}")
    assert rc == 0, text
    assert not workspace.exists()  # the real scratch cleanup ran
    [attachment] = kb.list_attachments(board, tid)
    assert attachment.filename == "REVIEW.md" and attachment.uploaded_by == "kanban_settle_delivered"
    from pathlib import Path
    assert Path(attachment.stored_path).read_bytes() == content


def test_si02_dependents_read_the_settled_negative_not_an_older_pass(board, capsys):
    """SI-02: an assessment completed PASS, was reopened from the dashboard, re-reviewed and settled
    negative; its released dependent's real worker context hands off the settled verdict."""
    tid = kb.create_task(board, title="prior assessment", assignee="reviewer", project_id="")
    assert kb.complete_task(board, tid, result="OLD_PASS_FROM_PREVIOUS_EDITION")
    assert dashboard_api()._set_status_direct(board, tid, "ready")
    reviewer = claim_with_worker(board, tid)
    assert kb.schedule_task(board, tid, reason="yield", expected_run_id=reviewer.current_run_id)
    worker_result(board, tid, reviewer.current_run_id, reviewer.claim_lock, 0)
    child = kb.create_task(board, title="consumer", assignee="consumer", project_id="", parents=[tid])
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer.current_run_id),
                       "--verdict", "changes_required", "--result", "NEW_CHANGES_REQUIRED_CURRENT_EDITION",
                       "--metadata", json.dumps({"report_sha256": "abc123"}))
    assert rc == 0, text
    assert kb.get_task(board, child).status == "ready"
    context = kb.build_worker_context(board, child)
    assert "NEW_CHANGES_REQUIRED_CURRENT_EDITION" in context
    assert "OLD_PASS_FROM_PREVIOUS_EDITION" not in context
    assert f"from run {reviewer.current_run_id} as a delivered changes_required assessment" in context
    assert "abc123" in context
    # A later ordinary completion supersedes the settlement: the reader returns to completed runs.
    # (Which same-second completed run it then picks is the reader's pre-existing started_at order.)
    assert dashboard_api()._set_status_direct(board, tid, "ready")
    assert kb.complete_task(board, tid, result="LATER_ORDINARY_COMPLETION", summary="LATER_ORDINARY_COMPLETION")
    later = kb.build_worker_context(board, child)
    assert "NEW_CHANGES_REQUIRED_CURRENT_EDITION" not in later and "_settled by" not in later


def test_si03_structured_metadata_survives_a_reopened_database(board, capsys, tmp_path):
    """SI-03: --metadata is kept on the settlement record (run N stays untouched) and reaches consumers."""
    tid, _first, reviewer = delivered_after_stale_schedule(board)
    child = kb.create_task(board, title="consumer", assignee="consumer", project_id="", parents=[tid])
    facts = {"report_sha256": "UNIQUE_REPORT_SHA256_FIELD_VALUE", "question": "settlement evidence"}
    assert run(*settle(tid, reviewer, "--metadata", json.dumps(facts))) == 0
    reopened = kbc.connect(tmp_path / "settle.sqlite")
    try:
        payload = reopened.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='completed'",
                                   (tid,)).fetchone()[0]
        assert json.loads(payload)["settlement"]["metadata"] == facts
        assert "UNIQUE_REPORT_SHA256_FIELD_VALUE" in kb.build_worker_context(reopened, child)
    finally:
        reopened.close()


@pytest.mark.parametrize("contract", ["owner/repository", "https://github.com/owner/repository/pull/1"])
def test_si04_a_pr_contract_keeps_its_acceptance_gate(board, capsys, contract):
    """SI-04: settlement is for local-only assessments; a PR/repository completion contract is refused
    atomically (no done, no released dependent, no acceptance bypass)."""
    tid, _first, reviewer = delivered_after_stale_schedule(board, completion_contract=contract)
    child = kb.create_task(board, title="consumer", assignee="consumer", project_id="", parents=[tid])
    before, child_before = image(board, tid), image(board, child)
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "rejected", "--result", "rejected: implementation not delivered")
    assert rc == 1 and "requires PR acceptance" in text and "nothing changed" in text, text
    assert len(text) < 400
    assert image(board, tid) == before and image(board, child) == child_before
    assert kb.get_task(board, child).status == "todo"


def files_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


@pytest.mark.parametrize("quote", ["`", '"', "unquoted"])
def test_si05_a_report_path_containing_a_space_survives_cleanup(board, capsys, tmp_path, quote):
    """SI-05: a result naming a scratch report whose filename contains a space (backtick- or
    double-quote-delimited, or bare) is attached byte-identically before the workspace is removed."""
    workspace = tmp_path / "home" / "kanban" / "workspaces" / "space-name"
    workspace.mkdir(parents=True)
    report = workspace / "FINAL REVIEW.md"
    report.write_bytes(b"CHANGES_REQUIRED: the final review evidence\n")
    content = report.read_bytes()
    named = str(report) if quote == "unquoted" else f"{quote}{report}{quote}"
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED {named}, see findings")
    assert rc == 0, text
    assert not workspace.exists()  # the real scratch cleanup ran
    [attachment] = kb.list_attachments(board, tid)
    assert attachment.filename == "FINAL REVIEW.md" and attachment.uploaded_by == "kanban_settle_delivered"
    from pathlib import Path
    assert Path(attachment.stored_path).read_bytes() == content


@pytest.mark.parametrize("named", ["`{ws}/MISSING REVIEW.md`", "{ws}/MISSING REVIEW.md", "`{ws}/notes`"])
def test_si05_an_unresolvable_scratch_reference_refuses_with_the_workspace_intact(board, capsys, tmp_path, named):
    """SI-05: a result naming something under the scratch workspace that is not an attachable regular
    file is refused atomically: nothing written, nothing copied, the workspace and its files intact."""
    workspace = tmp_path / "home" / "kanban" / "workspaces" / "unresolved"
    (workspace / "notes").mkdir(parents=True)
    report = workspace / "FINAL REVIEW.md"
    report.write_bytes(b"CHANGES_REQUIRED: the final review evidence\n")
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    before, home_before = image(board, tid), files_under(tmp_path / "home")
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", "CHANGES_REQUIRED " + named.format(ws=workspace))
    assert rc == 1 and "nothing changed" in text and "not a regular file" in text, text
    assert image(board, tid) == before
    assert files_under(tmp_path / "home") == home_before
    assert report.read_bytes() == b"CHANGES_REQUIRED: the final review evidence\n"


def settled_with_older_pass(board, capsys, metadata):
    """An assessment completed PASS (older run), reopened from the dashboard, re-reviewed, settled negative."""
    tid = kb.create_task(board, title="prior assessment", assignee="reviewer", project_id="")
    assert kb.complete_task(board, tid, result="OLD_PASS_FROM_PREVIOUS_EDITION")
    assert dashboard_api()._set_status_direct(board, tid, "ready")
    reviewer = claim_with_worker(board, tid)
    assert kb.schedule_task(board, tid, reason="yield", expected_run_id=reviewer.current_run_id)
    worker_result(board, tid, reviewer.current_run_id, reviewer.claim_lock, 0)
    child = kb.create_task(board, title="consumer", assignee="consumer", project_id="", parents=[tid])
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer.current_run_id),
                       "--verdict", "changes_required", "--result", "NEW_CHANGES_REQUIRED_CURRENT_EDITION",
                       "--metadata", json.dumps(metadata))
    assert rc == 0, text
    return tid, child, reviewer.current_run_id


def test_si06_editing_a_settled_result_keeps_its_handoff_coherent_and_history_intact(board, capsys, tmp_path):
    """SI-06: the supported result edit on a settled assessment hands off the corrected result with the
    corrected metadata, and rewrites no completed run (the older PASS run keeps its own record)."""
    tid, child, settled_run = settled_with_older_pass(
        board, capsys, {"report_sha256": "OLD_SETTLEMENT_DIGEST", "question": "first"})
    _, runs, _, events = image(board, tid)
    rc, text = outcome(capsys, "edit", tid, "--result", "CORRECTED_NEGATIVE_REPORT", "--summary",
                       "CORRECTED_NEGATIVE_SUMMARY", "--metadata",
                       json.dumps({"report_sha256": "NEW_CORRECTED_DIGEST", "question": "corrected"}))
    assert rc == 0, text
    reopened = kbc.connect(tmp_path / "settle.sqlite")
    try:
        context = kb.build_worker_context(reopened, child)
        _, after_runs, _, after_events = image(reopened, tid)
    finally:
        reopened.close()
    # As an ordinary edit's run summary, the explicit corrected handoff summary is what dependents read (SI-10).
    assert "CORRECTED_NEGATIVE_SUMMARY" in context and "NEW_CORRECTED_DIGEST" in context, context
    assert kb.get_task(board, tid).result == "CORRECTED_NEGATIVE_REPORT"
    assert "OLD_SETTLEMENT_DIGEST" not in context and "NEW_CHANGES_REQUIRED_CURRENT_EDITION" not in context
    assert "OLD_PASS_FROM_PREVIOUS_EDITION" not in context
    assert f"from run {settled_run} as a delivered changes_required assessment" in context
    assert after_runs == runs  # no completed run rewritten, none synthesized
    assert after_events[:len(events)] == events  # the settlement record itself is history
    [edited] = after_events[len(events):]
    assert edited["kind"] == "edited" and edited["run_id"] is None
    # A result-only edit keeps the corrected metadata, as an ordinary edit keeps its run's metadata.
    assert run("edit", tid, "--result", "SECOND_CORRECTION") == 0
    context = kb.build_worker_context(board, child)
    assert "SECOND_CORRECTION" in context and "NEW_CORRECTED_DIGEST" in context
    assert image(board, tid)[1] == runs


def test_si06_editing_a_settlement_never_synthesizes_a_reviewer_run(board, capsys):
    """SI-06: with no older completed run, editing the settled result must not invent a completed run."""
    tid, _first, reviewer = delivered_after_stale_schedule(board)
    assert run(*settle(tid, reviewer, "--metadata", json.dumps({"report_sha256": "FIRST"}))) == 0
    runs = image(board, tid)[1]
    assert run("edit", tid, "--result", "CORRECTED", "--metadata", json.dumps({"report_sha256": "SECOND"})) == 0
    assert image(board, tid)[1] == runs
    assert not [r for r in kb.list_runs(board, tid) if r.outcome == "completed"]


def test_si07_a_failure_before_commit_leaves_no_copy_and_a_retry_attaches_once(board, capsys, tmp_path):
    """SI-07: the completion event insert fails after the report was copied; the transaction rolls back,
    the copy is removed, the source survives, and the retry registers exactly one attachment."""
    import sqlite3
    from pathlib import Path
    from hermes_cli.kanban_db_settle import settle_delivered_assessment

    workspace, report, _digest = scratch_report(tmp_path / "home", "db-rollback")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    board.execute("CREATE TEMP TRIGGER fail_settlement BEFORE INSERT ON task_events WHEN NEW.kind='completed' "
                  "BEGIN SELECT RAISE(ABORT, 'injected event write failure'); END")
    board.commit()
    before, home_before = image(board, tid), files_under(tmp_path / "home")
    with pytest.raises(sqlite3.IntegrityError, match="injected event write failure"):
        settle_delivered_assessment(board, tid, expected_terminal_run=reviewer, verdict="changes_required",
                                    result=f"CHANGES_REQUIRED `{report}`", author="orchestrator")
    assert image(board, tid) == before
    assert files_under(tmp_path / "home") == home_before  # no orphan copy
    assert report.read_bytes() == content
    board.execute("DROP TRIGGER fail_settlement")
    board.commit()
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED `{report}`")
    assert rc == 0, text
    [attachment] = kb.list_attachments(board, tid)
    assert attachment.filename == "REVIEW.md"
    stored = Path(attachment.stored_path)
    assert stored.read_bytes() == content
    assert [p for p in stored.parent.iterdir()] == [stored]  # one durable copy, registered
    assert image(board, tid)[1] == before[1]  # the historical runs are untouched


def fail_completed_event(conn):
    conn.execute("CREATE TEMP TRIGGER fail_settlement BEFORE INSERT ON task_events WHEN NEW.kind='completed' "
                 "BEGIN SELECT RAISE(ABORT, 'injected event write failure'); END")
    conn.commit()


def registered_attachment(conn, tid):
    directory = kb.task_attachments_dir(tid)
    directory.mkdir(parents=True, exist_ok=True)
    prior = directory / "PRIOR_REVIEW.md"
    prior.write_bytes(b"EXISTING DURABLE REVIEW\n")
    kb.add_attachment(conn, tid, filename=prior.name, stored_path=str(prior), size=prior.stat().st_size,
                      uploaded_by="previous-review")
    return prior


@pytest.mark.parametrize("extra", [{}, {"_staged_artifacts": "{prior}"}], ids=["mixed", "caller_staged_field"])
def test_si08_a_failed_settlement_never_deletes_an_already_registered_attachment(board, capsys, tmp_path, extra):
    """SI-08: metadata names a registered attachment and a new scratch report; the completion event insert
    fails after staging. Only this attempt's copy is removed: the registered file and its row survive (a
    caller-supplied ``_staged_artifacts`` grants no deletion either), and the retry adds one attachment."""
    import sqlite3
    from pathlib import Path
    from hermes_cli.kanban_db_settle import settle_delivered_assessment

    workspace, report, _digest = scratch_report(tmp_path / "home", "mixed-evidence")
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    prior = registered_attachment(board, tid)
    metadata = {"artifacts": [str(prior), str(report)], "report_sha256": "caller-owned",
                **{key: [value.format(prior=prior)] for key, value in extra.items()}}
    caller = json.loads(json.dumps(metadata))
    fail_completed_event(board)
    before, home_before = image(board, tid), files_under(tmp_path / "home")
    rows_before = [dict(r) for r in board.execute("SELECT * FROM task_attachments WHERE task_id=?", (tid,))]
    with pytest.raises(sqlite3.IntegrityError, match="injected event write failure"):
        settle_delivered_assessment(board, tid, expected_terminal_run=reviewer, verdict="changes_required",
                                    result=f"CHANGES_REQUIRED `{report}`", author="orchestrator", metadata=metadata)
    assert image(board, tid) == before and metadata == caller
    assert [dict(r) for r in board.execute("SELECT * FROM task_attachments WHERE task_id=?", (tid,))] == rows_before
    assert files_under(tmp_path / "home") == home_before  # the registered file kept, no orphan copy
    assert prior.read_bytes() == b"EXISTING DURABLE REVIEW\n"
    board.execute("DROP TRIGGER fail_settlement")
    board.commit()
    rc, text = outcome(capsys, *settle(tid, reviewer, "--metadata", json.dumps(metadata)))
    assert rc == 0, text
    assert [a.filename for a in kb.list_attachments(board, tid)] == ["PRIOR_REVIEW.md", "REVIEW.md"]
    assert sorted(p.name for p in prior.parent.iterdir()) == ["PRIOR_REVIEW.md", "REVIEW.md"]
    assert all(Path(a.stored_path).is_file() for a in kb.list_attachments(board, tid))
    assert prior.read_bytes() == b"EXISTING DURABLE REVIEW\n" and not workspace.exists()


def test_si08_an_exception_after_commit_keeps_the_committed_attachment(board, capsys, tmp_path, monkeypatch):
    """SI-08: commit and cleanup stay consistent: if the transaction helper raises after its COMMIT
    (its post-commit file-length check), the committed attachment row keeps its file."""
    from pathlib import Path
    from hermes_cli.kanban_db_settle import settle_delivered_assessment

    workspace, report, _digest = scratch_report(tmp_path / "home", "postcommit")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))

    def torn(_conn):
        raise RuntimeError("injected post-COMMIT invariant failure")
    monkeypatch.setattr(kbc, "_check_file_length_invariant", torn)
    with pytest.raises(RuntimeError, match="post-COMMIT"):
        settle_delivered_assessment(board, tid, expected_terminal_run=reviewer, verdict="changes_required",
                                    result=f"CHANGES_REQUIRED `{report}`", author="orchestrator")
    assert kb.get_task(board, tid).status == "done"
    [attachment] = kb.list_attachments(board, tid)
    assert Path(attachment.stored_path).read_bytes() == content


def _dot(workspace, home):
    return str(home / "kanban") + "/./workspaces/" + workspace.name + "/REVIEW.md"


def _doubled(workspace, home):
    return str(workspace).replace("/workspaces/", "//workspaces//") + "//REVIEW.md"


def _parent(workspace, home):
    return str(home / "kanban" / "workspaces") + f"/../workspaces/{workspace.name}/REVIEW.md"


def _symlinked(workspace, home):
    link = home.parent / "linked-home"
    link.symlink_to(home, target_is_directory=True)
    return str(link / workspace.relative_to(home) / "REVIEW.md")


def _tilde(workspace, home):
    return "~/" + str((workspace / "REVIEW.md").relative_to(home.parent))


SPELLINGS = {"dot": _dot, "doubled": _doubled, "parent": _parent, "symlinked": _symlinked, "tilde": _tilde}


@pytest.mark.parametrize("quote", ["`", ""])
@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_si09_an_equivalent_absolute_spelling_of_the_report_survives_cleanup(
        board, capsys, tmp_path, monkeypatch, spelling, quote):
    """SI-09: the report named by any absolute spelling that resolves into the scratch workspace
    (``/./``, ``//``, ``..``, a symlinked ancestor, ``~``) is attached byte-identically before cleanup."""
    from pathlib import Path

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace, report, _digest = scratch_report(home, "absolute-" + spelling)
    content = report.read_bytes()
    named = SPELLINGS[spelling](workspace, home)
    assert named != str(report) and Path(named).expanduser().read_bytes() == content
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED {quote}{named}{quote}, see findings")
    assert rc == 0, text
    assert not workspace.exists()  # the real scratch cleanup ran
    [attachment] = kb.list_attachments(board, tid)
    assert attachment.filename == "REVIEW.md" and Path(attachment.stored_path).read_bytes() == content


def test_si09_an_equivalent_spelling_of_a_missing_report_refuses_with_the_workspace_intact(board, capsys, tmp_path):
    """SI-09: fail closed: a differently spelled reference into the workspace that names no regular file
    refuses atomically, as the literal spelling does."""
    workspace, report, _digest = scratch_report(tmp_path / "home", "absolute-missing")
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    before, home_before = image(board, tid), files_under(tmp_path / "home")
    named = str(tmp_path / "home" / "kanban") + "/./workspaces/absolute-missing/MISSING.md"
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED `{named}`")
    assert rc == 1 and "nothing changed" in text and "not a regular file" in text, text
    assert image(board, tid) == before and files_under(tmp_path / "home") == home_before
    assert report.exists()


@pytest.mark.parametrize("result", ["CORRECTED_NEGATIVE_REPORT", ""], ids=["distinct", "empty_result"])
def test_si10_a_corrected_handoff_summary_reaches_the_dependent(board, capsys, tmp_path, result):
    """SI-10: an accepted settlement correction with an explicit handoff summary (beside a distinct or an
    empty result) hands that summary and its metadata to the dependent, after a reopen, with no run touched."""
    tid, child, _settled_run = settled_with_older_pass(board, capsys, {"report_sha256": "OLD_SETTLEMENT_DIGEST"})
    runs = image(board, tid)[1]
    rc, text = outcome(capsys, "edit", tid, "--result", result, "--summary", "CORRECTED_HANDOFF_DO_NOT_DEPLOY",
                       "--metadata", json.dumps({"report_sha256": "CORRECTED_DIGEST"}))
    assert rc == 0, text
    reopened = kbc.connect(tmp_path / "settle.sqlite")
    try:
        context = kb.build_worker_context(reopened, child)
        assert image(reopened, tid)[1] == runs
    finally:
        reopened.close()
    assert "CORRECTED_HANDOFF_DO_NOT_DEPLOY" in context and "CORRECTED_DIGEST" in context, context
    assert "(no result recorded)" not in context and "OLD_SETTLEMENT_DIGEST" not in context
    assert "OLD_PASS_FROM_PREVIOUS_EDITION" not in context and "_settled by" in context


def file_bytes_under(root):
    """``{relative path: sha256}`` of every file under ``root``: a byte inventory, not only names."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def preserved_or_refused(board, tid, home, report, content, before, home_before, rc, text, *, dump_before=None,
                         bytes_before=None):
    """The contract for any reference into the scratch workspace: the full report is attached byte-exact,
    or the settlement is refused with the card, every file and the report itself untouched. When given,
    the whole logical database dump and every file's bytes must also be unchanged on refusal."""
    from pathlib import Path
    if rc == 0:
        assert any(Path(a.stored_path).read_bytes() == content for a in kb.list_attachments(board, tid)), text
        return "preserved"
    assert rc == 1 and "nothing changed" in text, text
    assert image(board, tid) == before and files_under(home) == home_before
    assert report.read_bytes() == content
    if dump_before is not None:
        assert list(board.iterdump()) == dump_before
    if bytes_before is not None:
        assert file_bytes_under(home) == bytes_before
    return "refused"


def settle_naming(board, capsys, tmp_path, name, filename, wrap, extra_files=()):
    home = tmp_path / "home"
    workspace = home / "kanban" / "workspaces" / name
    workspace.mkdir(parents=True)
    report = workspace / filename
    report.write_bytes(b"CHANGES_REQUIRED: the full negative report\n")
    for extra in extra_files:
        (workspace / extra).write_bytes(b"UNRELATED FILE\n")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    before, home_before = image(board, tid), files_under(home)
    dump_before, bytes_before = list(board.iterdump()), file_bytes_under(home)
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", "CHANGES_REQUIRED " + wrap.format(report))
    return preserved_or_refused(board, tid, home, report, content, before, home_before, rc, text,
                                dump_before=dump_before, bytes_before=bytes_before)


# Markdown emphasis/table cells (SI-11) and further wrappers no rule names: each must preserve or refuse.
WRAPPERS = {"bold": "**{}**", "italic": "*{}*", "pipe": "|{}|", "strike": "~~{}~~", "highlight": "=={}==",
            "dollar": "${}$", "curly_quotes": "“{}”", "trailing_stars": "{}**",
            "backtick_then_star": "`{}`*", "bold_backtick": "**`{}`**", "padded_pipe": "| {} |"}


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_si11_a_wrapped_reference_is_preserved_or_refused_never_deleted(board, capsys, tmp_path, wrapper):
    """SI-11: an absolute report path inside any wrapper is attached byte-exact or refused with the
    workspace retained; an unrecognized wrapper never lets cleanup delete an unattached report."""
    settle_naming(board, capsys, tmp_path, "wrapped-" + wrapper, "REVIEW.md", WRAPPERS[wrapper])


# An apostrophe or space inside a bare/quoted name next to an existing shorter prefix file (SI-12).
NAMES = {
    "apostrophe_bare_with_prefix": ("FINAL'S.md", "{}", ["FINAL"]),
    "apostrophe_single_quoted_with_prefix": ("FINAL'S.md", "'{}'", ["FINAL"]),
    "apostrophe_double_quoted_with_prefix": ('FINAL"S.md', '"{}"', ["FINAL"]),
    "space_bare_with_prefix": ("FINAL REVIEW.md", "{}", ["FINAL"]),
    "apostrophe_bare_alone": ("FINAL'S.md", "{}", []),
}


@pytest.mark.parametrize("case", sorted(NAMES))
def test_si12_a_name_with_an_inner_quote_or_space_never_attaches_a_prefix_file_instead(board, capsys, tmp_path, case):
    """SI-12: a bare FINAL'S.md beside an unrelated FINAL (and quoted/space near-variants) attaches the
    full report byte-exact or refuses with everything retained, never the shorter prefix file alone."""
    filename, wrap, extra = NAMES[case]
    settle_naming(board, capsys, tmp_path, case.replace("_", "-"), filename, wrap, extra)


def test_si12_an_apostrophe_name_without_a_prefix_file_is_preserved(board, capsys, tmp_path):
    """SI-12 control: a bare name with an apostrophe and no competing prefix file is attached."""
    assert settle_naming(board, capsys, tmp_path, "apostrophe-control", "FINAL'S.md", "{}") == "preserved"


def test_si11_a_copy_that_does_not_match_its_report_refuses_with_nothing_changed(board, capsys, tmp_path, monkeypatch):
    """Cleanup needs proof: if a staged copy's bytes differ from the referenced report, the settlement
    refuses inside its transaction; no row, no copy, the workspace and report retained."""
    real = kb._copy_capped

    def corrupting(src, dest, artifact):
        real(src, dest, artifact)
        dest.write_bytes(b"NOT THE REPORT\n")
    monkeypatch.setattr(kb, "_copy_capped", corrupting)
    workspace, report, _digest = scratch_report(tmp_path / "home", "miscopied")
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    before, home_before = image(board, tid), files_under(tmp_path / "home")
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED `{report}`")
    assert preserved_or_refused(board, tid, tmp_path / "home", report, report.read_bytes(), before, home_before,
                                rc, text) == "refused"


def test_si11_cleanup_keeps_the_workspace_when_the_committed_copy_is_not_proven(board, capsys, tmp_path, monkeypatch):
    """Cleanup needs proof after commit too: if the committed attachment no longer holds the report's
    bytes when cleanup would run, the workspace and report are kept."""
    from pathlib import Path

    def lose_the_copy(conn):
        for row in conn.execute("SELECT stored_path FROM task_attachments"):
            Path(row[0]).unlink(missing_ok=True)
    workspace, report, _digest = scratch_report(tmp_path / "home", "unproven")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    monkeypatch.setattr(kbc, "_check_file_length_invariant", lose_the_copy)
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", f"CHANGES_REQUIRED `{report}`")
    assert rc == 0 and "workspace kept" in text, text  # settled, but cleanup did not run
    assert workspace.is_dir() and report.read_bytes() == content, text


# Review6909 SI-13/SI-14: a spaced, wrapped or quoted name beside a shorter (or longer) existing file. Each
# reference must attach exactly the named report or refuse with nothing changed; never attach the other file.
IDENTITY = {
    "bold_backtick_space_prefix": ("FINAL REVIEW.md", "**`{}`**", ["FINAL"]),
    "bold_space_prefix": ("FINAL REVIEW.md", "**{}**", ["FINAL"]),
    "italic_space_prefix": ("FINAL REVIEW.md", "*{}*", ["FINAL"]),
    "pipe_space_prefix": ("FINAL REVIEW.md", "|{}|", ["FINAL"]),
    "quoted_newline_prefix": ("FINAL\nREVIEW.md", "`{}`", ["FINAL"]),
    "bare_tab_prefix": ("FINAL\tREVIEW.md", "{}", ["FINAL"]),
    "quoted_tab_prefix": ("FINAL\tREVIEW.md", "`{}`", ["FINAL"]),
    "quoted_trailing_space_prefix": ("FINAL.md ", "`{}`", ["FINAL.md"]),
    "quoted_leading_space_name_prefix": (" FINAL.md", "`{}`", ["FINAL.md"]),
    "space_inside_quote_before_path": ("FINAL.md", "` {}`", []),
    "bare_longer_sibling": ("REVIEW.md", "{} sha256 5d4c", ["REVIEW.md bak"]),
    "quoted_longer_sibling": ("REVIEW.md", "`{}` sha256 5d4c", ["REVIEW.md bak"]),
}


@pytest.mark.parametrize("case", sorted(IDENTITY))
def test_si13_si14_a_reference_attaches_exactly_the_named_report_or_refuses(board, capsys, tmp_path, case):
    """SI-13/SI-14: the named report's exact bytes are attached, or the settlement refuses with the card,
    every file and the report untouched; a shorter/longer existing file is never taken in its place."""
    filename, wrap, extra = IDENTITY[case]
    settle_naming(board, capsys, tmp_path, case.replace("_", "-"), filename, wrap, extra)


@pytest.mark.parametrize("wrap", ["`{}`", '"{}"', "<{}>"])
def test_si13_a_quoted_spaced_name_beside_a_prefix_file_is_preserved(board, capsys, tmp_path, wrap):
    """SI-13 positive control: a closed quotation names the full spaced report, so it is attached even
    though an unrelated prefix file exists."""
    assert settle_naming(board, capsys, tmp_path, "quoted-space", "FINAL REVIEW.md", wrap, ["FINAL"]) == "preserved"


def test_si13_a_bare_spaced_name_beside_a_prefix_file_refuses(board, capsys, tmp_path):
    """SI-13 control: bare words that could name either FINAL or FINAL REVIEW.md are ambiguous: refuse."""
    assert settle_naming(board, capsys, tmp_path, "bare-space", "FINAL REVIEW.md", "{}", ["FINAL"]) == "refused"


def test_si13_an_attachment_under_another_name_is_not_proof(board, capsys, tmp_path, monkeypatch):
    """Cleanup needs reference identity, not just bytes: a copy holding the report's bytes under a name
    other than the referenced one refuses with nothing changed."""
    monkeypatch.setattr(kb, "_unique_attachment_path", lambda directory, filename, used: directory / "OTHER.md")
    assert settle_naming(board, capsys, tmp_path, "renamed-copy", "REVIEW.md", "`{}`") == "refused"


# Review6997 SI-15: a name whose trailing-punctuation or line/space boundary, once normalized away, leaves a
# shorter existing file (FINAL. REVIEW.md beside FINAL). The original reference text decides identity.
SI15 = {
    **{f"period_space_{label}_prefix": ("FINAL. REVIEW.md", wrap, ["FINAL"])
       for label, wrap in {"bold": "**{}**", "italic": "*{}*", "pipe": "|{}|", "plain": "{}"}.items()},
    **{f"punct_{label}_space_{form}_prefix": (f"FINAL{mark} REVIEW.md", wrap, ["FINAL"])
       for label, mark in {"period": ".", "comma": ",", "semicolon": ";", "colon": ":", "bang": "!",
                           "question": "?"}.items()
       for form, wrap in {"bold": "**{}**", "plain": "{} sha256 5d4c"}.items()},
    "paren_space_bold_prefix": ("FINAL) REVIEW.md", "**{}**", ["FINAL"]),
    "multi_period_space_pipe_prefix": ("FINAL... REVIEW.md", "|{}|", ["FINAL"]),
    "period_space_bold_exact_prefix": ("FINAL. REVIEW.md", "**{}**", ["FINAL."]),
    "period_newline_bare_prefix": ("FINAL.\nREVIEW.md", "{}", ["FINAL"]),
    "period_nbsp_bold_prefix": ("FINAL. REVIEW.md", "**{}**", ["FINAL"]),
    "period_tab_bold_prefix": ("FINAL.\tREVIEW.md", "**{}**", ["FINAL"]),
    # quotes8: an embedded closer and then a line break inside a quoted name, beside its prefix file.
    "single_quote_newline_prefix": ("FINAL'S\nREVIEW.md", "'{}'", ["FINAL"]),
    "double_quote_newline_prefix": ('FINAL"S\nREVIEW.md', '"{}"', ["FINAL"]),
    "backtick_newline_prefix": ("FINAL`S\nREVIEW.md", "`{}`", ["FINAL"]),
    "angle_newline_prefix": ("FINAL>S\nREVIEW.md", "<{}>", ["FINAL"]),
    "single_quote_carriage_return_prefix": ("FINAL'S\rREVIEW.md", "'{}'", ["FINAL"]),
    "single_quote_space_prefix": ("FINAL'S REVIEW.md", "'{}'", ["FINAL"]),
}


@pytest.mark.parametrize("case", sorted(SI15))
def test_si15_a_normalized_reference_never_attaches_a_shorter_file_in_place_of_the_named_report(
        board, capsys, tmp_path, case):
    """SI-15: exact named bytes attached, or refused with the logical DB and every file's bytes unchanged;
    a prefix left after stripping punctuation or cutting at a space/line break is never taken instead."""
    filename, wrap, extra = SI15[case]
    settle_naming(board, capsys, tmp_path, "si15-" + case.replace("_", "-"), filename, wrap, extra)


def test_si15_a_quoted_name_with_punctuation_and_space_beside_its_prefix_is_preserved(board, capsys, tmp_path):
    """SI-15 positive control: the closed quotation proves the full name, so it is attached."""
    assert settle_naming(board, capsys, tmp_path, "si15-quoted", "FINAL. REVIEW.md", "`{}`", ["FINAL"]) == "preserved"


# Review6997 SI-16: a quoted name holding its own closer and then a line break; the real closer lies past the
# line end. A nearer embedded closer must never silently win: exact bytes, or refusal with nothing changed.
SI16 = {
    "single_quote_embedded_closer_newline_prefix": ("FINAL'S\nREVIEW.md", "'{}'", ["FINAL"]),
    "backtick_embedded_closer_newline_prefix": ("FINAL`S\nREVIEW.md", "`{}`", ["FINAL"]),
    "single_quote_embedded_closer_newline_no_prefix": ("FINAL'S\nREVIEW.md", "'{}'", []),
    "backtick_embedded_closer_newline_no_prefix": ("FINAL`S\nREVIEW.md", "`{}`", []),
    "backtick_newline_without_embedded_closer_prefix": ("FINAL S\nREVIEW.md", "`{}`", ["FINAL"]),
    "single_quote_embedded_closer_two_lines_later_prefix": ("FINAL'S\n\nREVIEW.md", "'{}'", ["FINAL"]),
}


@pytest.mark.parametrize("case", sorted(SI16))
def test_si16_a_quote_closed_past_its_line_never_lets_an_embedded_closer_select_a_prefix(
        board, capsys, tmp_path, case):
    """SI-16: the named report's exact bytes, or a refusal with the logical DB and every file unchanged."""
    filename, wrap, extra = SI16[case]
    settle_naming(board, capsys, tmp_path, "si16-" + case.replace("_", "-"), filename, wrap, extra)


@pytest.mark.parametrize("wrap", ["'{}'", "`{}`"])
def test_si16_a_quoted_name_without_an_embedded_closer_is_preserved(board, capsys, tmp_path, wrap):
    """SI-16 positive control: single-quoted and backtick names beside a prefix file are attached."""
    assert settle_naming(board, capsys, tmp_path, "si16-control", "FINAL REVIEW.md", wrap, ["FINAL"]) == "preserved"


# Review7055 SI-17: a report whose name begins with whitespace, so a shortened reading is the workspace root
# itself (`/ws/`), while the original text continues into the report. Never treat that as a harmless root.
SI17 = {
    "leading_space_bold": (" REVIEW.md", "**{}**"),
    "leading_space_italic": (" REVIEW.md", "*{}*"),
    "leading_space_pipe": (" REVIEW.md", "|{}|"),
    "leading_tab_bold": ("\tREVIEW.md", "**{}**"),
    "leading_nbsp_bold": ("\u00a0REVIEW.md", "**{}**"),
    "leading_newline_bare": ("\nREVIEW.md", "{}"),
    "leading_newline_quote": ("\nREVIEW.md", "`{}`"),
    "leading_space_trailing_dot_bare": (" REVIEW.md", "{}**."),
}


@pytest.mark.parametrize("case", sorted(SI17))
def test_si17_a_root_reading_never_hides_a_continuation_into_the_named_report(board, capsys, tmp_path, case):
    """SI-17: exact named bytes, or refusal with the logical DB and every file's bytes unchanged."""
    filename, wrap = SI17[case]
    settle_naming(board, capsys, tmp_path, "si17-" + case.replace("_", "-"), filename, wrap)


@pytest.mark.parametrize("wrap", ["{}", "`{}`"])
def test_si17_a_bare_or_quoted_leading_space_name_is_preserved(board, capsys, tmp_path, wrap):
    """SI-17 controls: a bare or closed-backtick leading-space name is the whole report and is attached."""
    assert settle_naming(board, capsys, tmp_path, "si17-control", " REVIEW.md", wrap) == "preserved"


@pytest.mark.parametrize("mention", ["{ws}", "{ws}/", "`{ws}`", "`{ws}/`", "{ws}/ was the scratch workspace"])
def test_si17_a_genuine_workspace_root_mention_still_settles(board, capsys, tmp_path, mention):
    """SI-17 controls: naming the workspace itself (beside an exactly named report) is not a reference to a
    report in it; the settlement succeeds and attaches only the named report."""
    from pathlib import Path
    workspace, report, _digest = scratch_report(tmp_path / "home", "root-mention")
    content = report.read_bytes()
    tid, _first, reviewer = delivered_after_stale_schedule(
        board, workspace_kind="scratch", workspace_path=str(workspace))
    result = "CHANGES_REQUIRED " + mention.format(ws=workspace) + f"\nreport `{report}`"
    rc, text = outcome(capsys, "settle-delivered", tid, "--expected-terminal-run", str(reviewer), "--verdict",
                       "changes_required", "--result", result)
    assert rc == 0, text
    [attachment] = kb.list_attachments(board, tid)
    assert attachment.filename == "REVIEW.md" and Path(attachment.stored_path).read_bytes() == content


# Review7098 SI-18: a report whose whole name is whitespace, named at the very end of --result. The original
# argument (before the whole-result trim) decides; exact bytes or refusal with everything unchanged.
SI18 = {
    "only_space_bare": (" ", "{}"),
    "only_tab_bare": ("\t", "{}"),
    "only_lf_bare": ("\n", "{}"),
    "only_nbsp_bare": (" ", "{}"),
    "two_spaces_bare": ("  ", "{}"),
    "space_lf_bare": (" \n", "{}"),
    "only_space_bold": (" ", "**{}**"),
    "only_space_quote": (" ", "`{}`"),
    "space_then_prose": (" ", "{} full report"),
}


@pytest.mark.parametrize("case", sorted(SI18))
def test_si18_a_whitespace_name_at_the_end_of_the_result_is_judged_before_trimming(board, capsys, tmp_path, case):
    """SI-18: the named report's exact bytes, or a refusal with the logical DB and every file's bytes unchanged."""
    filename, wrap = SI18[case]
    settle_naming(board, capsys, tmp_path, "si18-" + case.replace("_", "-"), filename, wrap)


# Review7098 SI-19: a quote whose first closer is the report name's own first character, so the quoted
# text is the workspace root while the original text continues into the report.
SI19 = {
    "open_tick_root": ("`REVIEW.md", "`{}"),
    "open_apostrophe_root": ("'REVIEW.md", "'{}"),
    "open_angle_root": (">REVIEW.md", "<{}"),
    "closed_tick_root": ("`REVIEW.md", "`{}`"),
    "closed_apostrophe_root": ("'REVIEW.md", "'{}'"),
    "closed_angle_root": (">REVIEW.md", "<{}>"),
    "open_tick_root_then_prose": ("`REVIEW.md", "`{} is the report"),
}


@pytest.mark.parametrize("case", sorted(SI19))
def test_si19_a_quoted_root_never_hides_a_continuation_into_the_named_report(board, capsys, tmp_path, case):
    """SI-19: the named report's exact bytes, or a refusal with the logical DB and every file's bytes unchanged."""
    filename, wrap = SI19[case]
    settle_naming(board, capsys, tmp_path, "si19-" + case.replace("_", "-"), filename, wrap)


@pytest.mark.parametrize("filename", ["`REVIEW.md", "'REVIEW.md", ">REVIEW.md"])
def test_si19_a_bare_name_starting_with_a_quote_character_is_preserved(board, capsys, tmp_path, filename):
    """SI-19 control: without an opening quote before the path, the whole name is attached."""
    assert settle_naming(board, capsys, tmp_path, "si19-bare", filename, "{}") == "preserved"
