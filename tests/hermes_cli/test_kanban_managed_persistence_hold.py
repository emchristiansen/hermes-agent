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


@pytest.mark.parametrize('selector', ['ttl', 'heartbeat', 'orphan'])
@pytest.mark.parametrize('phase', ['ready', 'review'])
@pytest.mark.parametrize('limit', [None, 1])
def test_stale_scan_cannot_reblock_an_already_released_attempt(board, monkeypatch, phase, limit, selector):
    conn, path = board
    tid = kb.create_task(conn, title='private stale selector control', project_id='', max_retries=limit)
    if phase == 'review':
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    claim = kb.claim_review_task if phase == 'review' else kb.claim_task
    task = claim(conn, tid)
    rid = task.current_run_id
    old = int(time.time()) - 9999
    if selector == 'ttl':
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (old, tid))
    elif selector == 'heartbeat':
        conn.execute('UPDATE tasks SET started_at=?,last_heartbeat_at=? WHERE id=?', (old, old, tid))
        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (old, rid))
    else:
        conn.execute('UPDATE tasks SET claim_lock=NULL,claim_expires=NULL WHERE id=?', (tid,))
    conn.commit()

    def reclaim(connection):
        if selector == 'ttl':
            return kb.release_stale_claims(connection, signal_fn=lambda *a: None)
        if selector == 'heartbeat':
            return dispatch.detect_stale_running(connection, stale_timeout_seconds=1, signal_fn=lambda *a: None)
        return dispatch.reconcile_orphaned_running(connection)
    from hermes_cli import kanban_owner_recovery as owners
    seam = owners if selector == 'orphan' else kb
    name = 'owner_reclaim_blocked' if selector == 'orphan' else '_terminate_reclaimed_worker'
    real = getattr(seam, name)
    crossed = []
    released_snapshot = []
    expected = 'ready' if selector == 'orphan' and limit is None else phase

    def interleave(*args, **kwargs):
        result = real(*args, **kwargs)
        if not crossed:
            crossed.append(True)
            other = kbc.connect(path)
            try:
                assert reclaim(other)
                if limit == 1:
                    assert kb.unblock_task(other, tid, expected_terminal_run=rid)
                assert kb.get_task(other, tid).status == expected
                released_snapshot.append((image(other, tid), [dict(row) for row in other.execute(
                    "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]))
            finally:
                other.close()
        return result
    monkeypatch.setattr(seam, name, interleave)
    reclaimed = reclaim(conn)
    actual = kb.get_task(conn, tid)
    events = [dict(x) for x in conn.execute('SELECT kind,run_id,payload FROM task_events WHERE task_id=? ORDER BY id', (tid,))]
    print('STALE_SELECTOR=' + json.dumps(dict(selector=selector, limit=limit, phase=phase, reclaimed=reclaimed, status=actual.status, events=events)))
    assert not reclaimed
    assert (image(conn, tid), [dict(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]) == released_snapshot[0]
    assert actual.status == expected, 'stale losing selector re-blocked an explicitly released attempt'
    assert (kb.claim_review_task if expected == 'review' else kb.claim_task)(conn, tid) is not None


@pytest.mark.parametrize('same_claimer', [True, False])
@pytest.mark.parametrize('selector', ['ttl', 'heartbeat', 'orphan'])
@pytest.mark.parametrize('phase', ['ready', 'review'])
@pytest.mark.parametrize('limit', [None, 1])
def test_old_scan_must_not_end_freshly_authorized_run(board, monkeypatch, phase, limit, selector, same_claimer):
    conn, path = board
    tid = kb.create_task(conn, title='private stale selector control', project_id='', max_retries=limit)
    if phase == 'review':
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    claim = kb.claim_review_task if phase == 'review' else kb.claim_task
    task = claim(conn, tid, claimer=None)
    rid = task.current_run_id
    old = int(time.time()) - 9999
    if selector == 'ttl':
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (old, tid))
    elif selector == 'heartbeat':
        conn.execute('UPDATE tasks SET started_at=?,last_heartbeat_at=? WHERE id=?', (old, old, tid))
        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (old, rid))
    else:
        conn.execute('UPDATE tasks SET claim_lock=NULL,claim_expires=NULL WHERE id=?', (tid,))
    conn.commit()

    def reclaim(connection):
        if selector == 'ttl':
            return kb.release_stale_claims(connection, signal_fn=lambda *a: None)
        if selector == 'heartbeat':
            return dispatch.detect_stale_running(connection, stale_timeout_seconds=1, signal_fn=lambda *a: None)
        return dispatch.reconcile_orphaned_running(connection)
    from hermes_cli import kanban_owner_recovery as owners
    seam = owners if selector == 'orphan' else kb
    name = 'owner_reclaim_blocked' if selector == 'orphan' else '_terminate_reclaimed_worker'
    real = getattr(seam, name)
    crossed = []
    next_run = []
    claimed_snapshot = []
    expected = 'ready' if selector == 'orphan' and limit is None else phase

    def interleave(*args, **kwargs):
        result = real(*args, **kwargs)
        if not crossed:
            crossed.append(True)
            other = kbc.connect(path)
            try:
                assert reclaim(other)
                if limit == 1:
                    assert kb.unblock_task(other, tid, expected_terminal_run=rid)
                assert kb.get_task(other, tid).status == expected
                new = (kb.claim_review_task if expected == 'review' else kb.claim_task)(other, tid, claimer=None if same_claimer else 'synthetic-dispatcher:B')
                assert new is not None and new.current_run_id != rid
                next_run.append(new.current_run_id)
                claimed_snapshot.append((image(other, tid), [dict(row) for row in other.execute(
                    "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]))
            finally:
                other.close()
        return result
    monkeypatch.setattr(seam, name, interleave)
    reclaimed = reclaim(conn)
    actual = kb.get_task(conn, tid)
    events = [dict(x) for x in conn.execute('SELECT kind,run_id,payload FROM task_events WHERE task_id=? ORDER BY id', (tid,))]
    print('NEW_RUN_SELECTOR=' + json.dumps(dict(same_claimer=same_claimer, new_run=next_run[0], selector=selector, limit=limit, phase=phase, reclaimed=reclaimed, status=actual.status, events=events)))
    assert (image(conn, tid), [dict(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]) == claimed_snapshot[0]
    assert not reclaimed
    assert actual.status == 'running' and actual.current_run_id == next_run[0]
    assert kb.get_run(conn, next_run[0]).ended_at is None


@pytest.mark.parametrize("phase", ["ready", "review"])
@pytest.mark.parametrize("limit", [None, 1])
def test_scanned_heartbeat_must_still_match_before_retiring_same_run(board, monkeypatch, phase, limit):
    conn, path = board
    tid = kb.create_task(conn, title="private renewed heartbeat", project_id="", max_retries=limit)
    if phase == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    task = (kb.claim_review_task if phase == "review" else kb.claim_task)(conn, tid)
    old = int(time.time()) - 9999
    conn.execute("UPDATE tasks SET started_at=?,last_heartbeat_at=? WHERE id=?", (old, old, tid))
    conn.execute("UPDATE task_runs SET started_at=? WHERE id=?", (old, task.current_run_id))
    conn.commit()
    original = kb._terminate_reclaimed_worker
    updated = []

    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        if not updated:
            other = kbc.connect(path)
            try:
                assert dispatch.heartbeat_worker(other, tid, expected_run_id=task.current_run_id)
                updated.append((image(other, tid), [dict(row) for row in other.execute(
                    "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]))
            finally:
                other.close()
        return result

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", renew)
    assert dispatch.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *args: None) == []
    assert (image(conn, tid), [dict(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]) == updated[0]
    assert kb.get_task(conn, tid).current_run_id == task.current_run_id
    assert kb.get_run(conn, task.current_run_id).ended_at is None


def owner_store(tmp_path, status, outcome):
    """An owner state DB holding the attempt's admission, as ``owner_admitted`` names it."""
    import sqlite3

    path = tmp_path / f"owner-{status}-{outcome}.sqlite"
    with sqlite3.connect(path) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('owner-session', 'request', ?, ?)", (status, outcome))
    return path, {"db": str(path), "session_id": "owner-session", "request_id": "request"}


def selected_and_scheduled(conn, *, owner_receipt=None, exit_code=0):
    """A first-failure card whose next claimed attempt yields by scheduling itself; its bound worker
    then records ``exit_code`` (None: no worker result yet)."""
    tid, run = stopped(conn)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert kb.unblock_task(conn, tid, expected_terminal_run=run)
    task = kb.claim_task(conn, tid)
    assert task is not None
    if owner_receipt is not None:
        kb._append_event(conn, tid, "owner_admitted", owner_receipt, run_id=task.current_run_id)
        conn.commit()
    assert kb.schedule_task(conn, tid, reason="await the author's callback")
    if exit_code is not None:
        kb._append_event(conn, tid, "worker_result", {"pid": 1, "claim_lock": task.claim_lock, "exit_code": exit_code,
                                                       "last_output": ""}, run_id=task.current_run_id)
        conn.commit()
    assert kb.get_task(conn, tid).status == "scheduled"
    return tid, task.current_run_id


def test_ordinary_callback_wakes_a_selected_card_that_cleanly_scheduled_itself(board, tmp_path):
    """2026-11-04, t_674db54b run549: the card scheduled itself for a callback, its worker exited
    0 and its owner admission completed, and the author's plain ``unblock`` was refused because
    max_retries=1 demanded an exact run; only ``--expected-terminal-run 549`` woke it."""
    conn, _ = board
    _, receipt = owner_store(tmp_path, "terminal", "completed")
    tid, _own_run = selected_and_scheduled(conn, owner_receipt=receipt)
    failures = kb.get_task(conn, tid).consecutive_failures
    assert cli._cmd_unblock(parsed("unblock", tid)) == 0
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.consecutive_failures == failures  # a wake is not a failure-history reset
    assert kb.unblock_task(conn, tid) is False  # already woken: nothing to release


@pytest.mark.parametrize("admission, exit_code", [
    (("terminal", "interrupted"), None),  # owner lost the attempt: no worker result
    (("terminal", "interrupted"), 0),
    (("terminal", "failed"), 1),
    (("terminal", "completed"), 1),  # the worker failed after scheduling
    (("unknown", None), 0),
    (("started", None), None),  # the worker is still running
    (None, None),  # unmanaged attempt whose worker has not exited
    (None, 1),
])
def test_only_a_clean_yield_wakes_without_the_exact_run(board, tmp_path, admission, exit_code):
    """t_93eb808d: a scheduled attempt that was interrupted, failed, lost or is still running is an
    uncertain effect, not a deliberate yield; its release still names the exact run (and still
    passes every terminal/owner check when it does)."""
    conn, _ = board
    receipt = owner_store(tmp_path, *admission)[1] if admission else None
    tid, own_run = selected_and_scheduled(conn, owner_receipt=receipt, exit_code=exit_code)
    assert not kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "scheduled"
    exact = kb.unblock_task(conn, tid, expected_terminal_run=own_run)
    assert exact == (admission is None or admission[0] == "terminal")


def test_callback_wake_waits_for_the_clean_attempts_owner_to_settle(board, tmp_path):
    """Scheduling ends the run before its worker exits; until the owner admission completes,
    neither an ordinary nor an exact release may start a competing attempt."""
    import sqlite3

    owner_db, receipt = owner_store(tmp_path, "started", None)
    tid, own_run = selected_and_scheduled(board[0], owner_receipt=receipt)
    conn = board[0]
    assert not kb.unblock_task(conn, tid)
    assert not kb.unblock_task(conn, tid, expected_terminal_run=own_run)
    with sqlite3.connect(owner_db) as owner:
        owner.execute("UPDATE session_admissions SET status='terminal', outcome='completed'")
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "ready"


def test_scheduling_a_held_card_keeps_the_exact_run_fence(board):
    """A first-failure hold that an orchestrator parks in ``scheduled`` ends no claimed attempt
    of its own, so its release still names the exact run."""
    conn, _ = board
    tid, run = stopped(conn)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)
    assert kb.schedule_task(conn, tid, reason="parked by an operator")
    assert kb.get_task(conn, tid).status == "scheduled"
    latest = conn.execute("SELECT max(id) FROM task_runs WHERE task_id=?", (tid,)).fetchone()[0]
    assert cli._cmd_unblock(parsed("unblock", tid)) != 0
    assert kb.get_task(conn, tid).status == "scheduled"
    assert kb.unblock_task(conn, tid, expected_terminal_run=latest)


def test_refused_unblock_names_its_actual_reason(board, tmp_path, capsys):
    """The refusal used to read ``not blocked/scheduled?`` for every cause, including a scheduled
    card refused only for a missing exact-run operand."""
    conn, _ = board
    tid, run = stopped(conn)
    assert kb.edit_task(conn, tid, max_retries=1, expected_terminal_run=run)

    def refusal(*arguments):
        capsys.readouterr()
        assert cli._cmd_unblock(parsed("unblock", tid, *arguments)) != 0
        captured = capsys.readouterr()
        return captured.out + captured.err

    assert "exact terminal run is named (--expected-terminal-run)" in refusal()
    assert f"run {run - 1} is not the latest attempt (latest is {run})" in refusal("--expected-terminal-run", str(run - 1))
    assert kb.unblock_task(conn, tid, expected_terminal_run=run)
    assert "status is ready, not blocked/scheduled" in refusal()

    _, receipt = owner_store(tmp_path, "started", None)
    task = kb.claim_task(conn, tid)
    kb._append_event(conn, tid, "owner_admitted", receipt, run_id=task.current_run_id)
    conn.commit()
    assert kb.schedule_task(conn, tid, reason="await a callback")
    assert f"run {task.current_run_id} owner admission is not terminal yet (started)" in refusal(
        "--expected-terminal-run", str(task.current_run_id))
    assert "only a clean self-scheduled yield" in refusal()
