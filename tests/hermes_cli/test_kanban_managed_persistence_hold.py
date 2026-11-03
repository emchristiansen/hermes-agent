"""Task-local first-failure containment without erasing historical failures."""
import argparse
import json
import socket
import time
from unittest.mock import patch

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_parser


@pytest.fixture(name="board")
def isolated_board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    path = tmp_path / "policy.sqlite"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(kb, "_fire_task_hook", lambda *a, **k: None)
    with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
        conn = kbc.connect(path)
        yield conn, path
        conn.close()


def stopped(conn, phase="ready", failures=2):
    tid = kb.create_task(conn, title="private policy control", project_id="", max_retries=5)
    for index in range(failures):
        if phase == "review":
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
            conn.commit()
            task = kb.claim_review_task(conn, tid)
        else:
            task = kb.claim_task(conn, tid)
        assert task is not None
        run = task.current_run_id
        dispatch._record_task_failure(conn, tid, "known unrelated fixture failure", outcome="crashed",
                                      failure_limit=99, release_claim=True, end_run=True, force_trip=index == failures - 1)
    assert kb.get_task(conn, tid).status == "blocked"
    return tid, run


def parsed(*arguments):
    parser = argparse.ArgumentParser()
    kanban_parser.build_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["kanban", *arguments])


def image(conn, tid):
    return (dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()),
            [dict(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,))])


def test_supported_cli_tightens_exact_stopped_run_without_other_changes(board):
    conn, _ = board
    tid, run = stopped(conn)
    before, runs = image(conn, tid)
    args = parsed("edit", tid, "--max-retries", "1", "--expected-terminal-run", str(run))
    assert cli._cmd_edit(args) == 0
    after, after_runs = image(conn, tid)
    assert after.pop("max_retries") == 1
    before.pop("max_retries")
    assert after == before and after_runs == runs
    events = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='edited' ORDER BY id DESC", (tid,)).fetchall()
    assert json.loads(events[0][0])["expected_terminal_run"] == run


@pytest.mark.parametrize("phase", ["ready", "review"])
def test_explicit_release_preserves_failures_then_first_failure_holds(board, phase):
    conn, _ = board
    tid, run = stopped(conn, phase=phase)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    before = kb.get_task(conn, tid)
    assert before.consecutive_failures == 2
    assert kb.unblock_task(conn, tid, expected_terminal_run=run)
    assert kb.get_task(conn, tid).consecutive_failures == 2
    assert kb.get_task(conn, tid).status == phase
    assert dispatch.check_respawn_guard(conn, tid, lane=phase) is None
    claimant = kb.claim_task if phase == "ready" else kb.claim_review_task
    task = claimant(conn, tid)
    assert task is not None and task.current_run_id != run
    dispatch._record_task_failure(conn, tid, "known unrelated next failure", outcome="crashed",
                                  release_claim=True, end_run=True)
    after = kb.get_task(conn, tid)
    assert after.status == "blocked" and after.consecutive_failures == 3
    for _ in range(3):
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"
        assert claimant(conn, tid) is None


@pytest.mark.parametrize("value", [0, 2, 6, -1, True])
def test_policy_edit_refuses_other_values_and_preserves_metadata(board, value):
    conn, _ = board
    tid, run = stopped(conn)
    before = image(conn, tid)
    assert not kb.edit_task(conn, tid, max_retries=value, expected_terminal_run=run)
    assert image(conn, tid) == before


@pytest.mark.parametrize("extra", [{"title": "mixed"}, {"body": "mixed"}, {"priority": 4}, {"result": "mixed"}])
def test_policy_edit_rejects_mixed_controls(board, extra):
    conn, _ = board
    tid, run = stopped(conn)
    before = image(conn, tid)
    assert not kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run, **extra)
    assert image(conn, tid) == before


def test_policy_edit_and_release_refuse_stale_or_missing_run(board):
    conn, _ = board
    tid, run = stopped(conn)
    before = image(conn, tid)
    assert not kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run - 1)
    assert not kb.edit_task(conn, tid, max_retries=1)
    assert image(conn, tid) == before
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    enrolled = image(conn, tid)
    assert not kb.unblock_task(conn, tid, expected_terminal_run=run - 1)
    assert not kb.unblock_task(conn, tid)
    assert image(conn, tid) == enrolled


def test_policy_cli_denies_worker_and_unblock_requires_exact_single_row(board, monkeypatch):
    conn, _ = board
    tid, run = stopped(conn)
    args = parsed("edit", tid, "--max-retries", "1", "--expected-terminal-run", str(run))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "not-an-orchestrator")
    before = image(conn, tid)
    assert cli._cmd_edit(args) != 0
    assert image(conn, tid) == before


def test_active_run_and_stale_claim_refuse_policy_edit(board):
    conn, _ = board
    tid, run = stopped(conn)
    assert kb.unblock_task(conn, tid)
    current = kb.claim_task(conn, tid)
    assert current is not None
    before = image(conn, tid)
    assert not kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert image(conn, tid) == before


@pytest.mark.parametrize("phase", ["ready", "review"])
def test_selected_reclaim_never_publishes_claimable_retry_window(board, phase):
    conn, path = board
    tid = kb.create_task(conn, title="private dead claimant", project_id="", max_retries=1)
    if phase == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
        task = kb.claim_review_task(conn, tid)
    else:
        task = kb.claim_task(conn, tid)
    assert task is not None
    # Synthetic metadata only: no process at this impossible Linux pid is spawned or signalled.
    conn.execute("UPDATE tasks SET worker_pid=?,started_at=? WHERE id=?", (999999999, int(time.time()) - 9999, tid))
    conn.commit()
    with patch("hermes_cli.kanban_owner_recovery.owner_reclaim_blocked", return_value=False), \
         patch("hermes_cli.kanban_owner_recovery.managed_run_exit_code", return_value=1):
        sweep = dispatch._reclaim_dead_workers(conn)
    assert tid in sweep.crashed
    # A second normal connection observes the real committed between-phase state.
    other = kbc.connect(path)
    try:
        state = kb.get_task(other, tid).status
        claimant = kb.claim_task if phase == "ready" else kb.claim_review_task
        next_claim = claimant(other, tid)
        assert next_claim is None, f"after-reclaim status={state}; next admitted run={next_claim.current_run_id}"
        assert state == "blocked"
    finally:
        other.close()


def test_default_unselected_retry_is_unchanged(board):
    conn, _ = board
    tid = kb.create_task(conn, title="ordinary default", project_id="")
    assert kb.claim_task(conn, tid)
    dispatch._record_task_failure(conn, tid, "ordinary failure", outcome="crashed",
                                  release_claim=True, end_run=True)
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.get_task(conn, tid).consecutive_failures == 1
    assert kb.claim_task(conn, tid)
