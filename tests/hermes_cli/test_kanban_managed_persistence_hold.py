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


@pytest.mark.parametrize("operation", ["edit", "release"])
def test_unknown_managed_owner_keeps_policy_and_release_unchanged(board, monkeypatch, operation):
    conn, _ = board
    tid, run = stopped(conn)
    if operation == "release":
        assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    before = image(conn, tid)
    monkeypatch.setattr("hermes_cli.kanban_owner_recovery.owner_reclaim_blocked", lambda *a, **k: True)
    if operation == "edit":
        assert not kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    else:
        assert not kb.unblock_task(conn, tid, expected_terminal_run=run)
    assert image(conn, tid) == before


@pytest.mark.parametrize("outcome", ["timed_out", "crashed", "spawn_failed"])
def test_selected_accounted_failure_holds_without_new_counter_budget(board, outcome):
    conn, _ = board
    tid = kb.create_task(conn, title="selected failure class", project_id="", max_retries=1)
    assert kb.claim_task(conn, tid)
    assert dispatch._record_task_failure(conn, tid, "synthetic classification", outcome=outcome,
                                         release_claim=True, end_run=True)
    assert kb.get_task(conn, tid).status == "blocked"
    for _ in range(3):
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, tid) is None


@pytest.mark.parametrize("limit", [None, 1])
def test_infrastructure_refusal_respects_selected_hold_without_counting_it(board, limit):
    conn, _ = board
    tid = kb.create_task(conn, title="isolated host refusal", project_id="", max_retries=limit)
    assert kb.claim_task(conn, tid)
    dispatch._record_task_failure(conn, tid, "host scope unavailable", outcome="spawn_failed",
                                   release_claim=True, end_run=True, infrastructure=True)
    expected = "blocked" if limit == 1 else "ready"
    assert kb.get_task(conn, tid).status == expected
    assert kb.get_task(conn, tid).consecutive_failures == 0
    if limit == 1:
        for _ in range(3):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("owner_state", ["unknown", "queued", "running", "missing", "terminal"])
def test_historical_owner_receipt_is_checked_after_current_run_cleared(board, tmp_path, owner_state):
    import sqlite3

    conn, _ = board
    tid, run = stopped(conn)
    owner_path = tmp_path / "owner.sqlite"
    with sqlite3.connect(owner_path) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT,request_id TEXT,status TEXT)")
        if owner_state != "missing":
            owner.execute("INSERT INTO session_admissions VALUES (?,?,?)", ("exact-session", "exact-request", owner_state))
    kb._append_event(conn, tid, "owner_admitted", {
        "db": str(owner_path), "session_id": "exact-session", "request_id": "exact-request",
    }, run_id=run)
    conn.commit()
    before = image(conn, tid)
    accepted = kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert accepted is (owner_state == "terminal")
    if not accepted:
        assert image(conn, tid) == before


def test_repeating_policy_edit_is_idempotent_and_guarded_cli_release_keeps_counts(board):
    conn, _ = board
    tid, run = stopped(conn)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    before = image(conn, tid)
    events = conn.execute("SELECT count(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert image(conn, tid) == before
    assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0] == events
    assert cli._cmd_unblock(parsed("unblock", tid, "--expected-terminal-run", str(run))) == 0
    assert kb.get_task(conn, tid).consecutive_failures == 2
    assert kb.claim_task(conn, tid) is not None


def test_policy_edit_does_not_raise_an_unsupported_legacy_zero_limit(board):
    conn, _ = board
    tid = kb.create_task(conn, title="legacy zero", project_id="", max_retries=0)
    task = kb.claim_task(conn, tid)
    run = task.current_run_id
    dispatch._record_task_failure(conn, tid, "old failure", outcome="crashed", release_claim=True, end_run=True)
    before = image(conn, tid)
    assert not kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert image(conn, tid) == before


@pytest.mark.parametrize("phase", ["ready", "review"])
def test_timeout_transition_is_not_claimable_before_failure_accounting(board, monkeypatch, phase):
    conn, path = board
    tid = kb.create_task(conn, title="private timeout selector", project_id="", max_retries=1, max_runtime_seconds=1)
    if phase == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    task = (kb.claim_review_task if phase == "review" else kb.claim_task)(conn, tid)
    assert task is not None
    old = int(time.time()) - 9999
    conn.execute("UPDATE tasks SET worker_pid=?,started_at=? WHERE id=?", (999999999, old, tid))
    conn.execute("UPDATE task_runs SET started_at=? WHERE id=?", (old, task.current_run_id))
    conn.commit()
    original = dispatch._record_task_failure
    exact_run = kb.get_task(conn, tid).current_run_id
    observed = []

    def before_accounting(*args, **kwargs):
        other = kbc.connect(path)
        try:
            state = kb.get_task(other, tid).status
            claim = (kb.claim_review_task if phase == "review" else kb.claim_task)(other, tid)
            observed.append((state, claim is None))
            assert not kb.unblock_task(other, tid, expected_terminal_run=exact_run)
        finally:
            other.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch, "_record_task_failure", before_accounting)
    timed = dispatch.enforce_max_runtime(conn, signal_fn=lambda *args: None)
    assert tid in timed
    assert observed == [("blocked", True)]
    assert kb.get_task(conn, tid).consecutive_failures == 1


@pytest.mark.parametrize("phase", ["ready", "review"])
def test_stale_reclaim_is_not_claimable_before_accounting(board, monkeypatch, phase):
    conn, path = board
    tid = kb.create_task(conn, title="private expired claim", project_id="", max_retries=1)
    if phase == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    assert (kb.claim_review_task if phase == "review" else kb.claim_task)(conn, tid)
    conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 9999, tid))
    conn.commit()
    original = kb._record_task_failure
    exact_run = kb.get_task(conn, tid).current_run_id
    observed = []

    def before_accounting(*args, **kwargs):
        other = kbc.connect(path)
        try:
            state = kb.get_task(other, tid).status
            claim = (kb.claim_review_task if phase == "review" else kb.claim_task)(other, tid)
            observed.append((state, claim is None))
            assert not kb.unblock_task(other, tid, expected_terminal_run=exact_run)
        finally:
            other.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(kb, "_record_task_failure", before_accounting)
    assert kb.release_stale_claims(conn, signal_fn=lambda *args: None) == 1
    assert observed == [("blocked", True)]
    assert kb.get_task(conn, tid).consecutive_failures == 1


@pytest.mark.parametrize("key", ["force", "schema", "new_policy"])
def test_policy_wrapper_preserves_closed_callable_signature(board, key):
    conn, _ = board
    tid, run = stopped(conn)
    before = image(conn, tid)
    with pytest.raises(TypeError):
        kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run, **{key: True})
    assert image(conn, tid) == before


def test_policy_edit_notifies_after_commit_once_and_not_on_repeated_selection(board, monkeypatch):
    conn, _ = board
    tid, run = stopped(conn)
    notified = []

    def notify(connection, task_id, fields, *, board=None):
        assert not connection.in_transaction
        notified.append((task_id, fields, board))

    monkeypatch.setattr(kb, "notify_task_updated", notify)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert notified == [(tid, ["max_retries"], None)]


@pytest.mark.parametrize("selector", ["heartbeat", "orphan"])
@pytest.mark.parametrize("phase", ["ready", "review"])
@pytest.mark.parametrize("limit", [None, 1])
def test_other_automatic_reclaim_paths_hold_only_selected_tasks(board, selector, phase, limit):
    conn, _ = board
    tid = kb.create_task(conn, title="private other reclaim", project_id="", max_retries=limit)
    if phase == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    task = (kb.claim_review_task if phase == "review" else kb.claim_task)(conn, tid)
    assert task is not None
    if selector == "heartbeat":
        old = int(time.time()) - 9999
        conn.execute("UPDATE tasks SET started_at=?,last_heartbeat_at=? WHERE id=?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at=? WHERE id=?", (old, task.current_run_id))
        conn.commit()
        assert tid in dispatch.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *args: None)
    else:
        conn.execute("UPDATE tasks SET claim_lock=NULL,claim_expires=NULL WHERE id=?", (tid,))
        conn.commit()
        assert tid in dispatch.reconcile_orphaned_running(conn)
    expected = "blocked" if limit == 1 else ("ready" if selector == "orphan" else phase)
    assert kb.get_task(conn, tid).status == expected
    assert kb.get_task(conn, tid).consecutive_failures == 0
    if limit == 1:
        for _ in range(3):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid, expected_terminal_run=task.current_run_id)
        assert kb.get_task(conn, tid).status == phase
