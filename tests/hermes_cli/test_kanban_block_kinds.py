"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking —
  unless no parent is open, in which case the wait can never be satisfied
  and the block is recorded as ``needs_input`` (sticky, loop-counted).
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", parents=()):
    """Create a task (linked under ``parents`` first) and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    for parent in parents:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _settled_loop(conn):
    tid = _running_task(conn)
    assert kb.block_task(conn, tid, kind="capability", reason="missing producer")
    assert kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)
    task = kb.get_task(conn, tid)
    run_id, claim = task.current_run_id, task.claim_lock
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {
            "pid": 12345, "claim_lock": claim, "started_at": "test-birth",
        }, run_id=run_id)
    assert kb.block_task(conn, tid, kind="capability", reason="missing producer")
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_result", {
            "pid": 12345, "claim_lock": claim, "exit_code": 0,
        }, run_id=run_id)
    return tid, run_id


def _settle(conn, tid, run_id):
    from hermes_cli.kanban_db_graph import settle_triage_task
    return settle_triage_task(
        conn, tid, expected_terminal_run_id=run_id, kind="capability",
        owner="source-owner", required_input="qualified source result",
        event="source-owner supplies the result", author="orchestrator",
    )


def test_settled_loop_wait_preserves_history_and_resumes_once_after_owner_event(kanban_home):
    with kbc.connect_closing() as conn:
        tid, run_id = _settled_loop(conn)
        before = kb.get_task(conn, tid)
        runs = [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")]
        events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        ok, reason = _settle(conn, tid, run_id)
        assert ok, reason
        after = kb.get_task(conn, tid)
        for field in ("body", "title", "assignee", "workspace_path", "workspace_kind", "block_recurrences"):
            assert getattr(after, field) == getattr(before, field)
        assert [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")] == runs
        assert [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")][:len(events)] == events
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.claim_task(conn, tid, claimer="no-event") is None
        parent = kb.create_task(conn, title="real prerequisite", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
        kb.add_comment(conn, tid, author="source-owner", body="Qualified source result delivered: artifact-v2")
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "todo"
        assert kb.claim_task(conn, tid, claimer="still-parent-gated") is None
        assert kb.claim_task(conn, parent, claimer="worker")
        assert kb.complete_task(conn, parent, result="prerequisite supplied")
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, tid, claimer="worker")
        assert kb.claim_task(conn, tid, claimer="second-worker") is None


def test_settlement_cli_requires_explicit_fields_and_orchestrator(kanban_home, monkeypatch):
    from hermes_cli.kanban_parser import build_parser
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    with kbc.connect_closing() as conn:
        tid, run_id = _settled_loop(conn)
        before = list(conn.iterdump())
    argv = ["kanban", "settle-triage", tid, "--expected-terminal-run", str(run_id),
            "--kind", "capability", "--owner", "producer", "--required-input", "source result",
            "--event", "source result delivered"]
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(argv[:-2])
    assert missing.value.code == 2
    args = parser.parse_args(argv)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert kanban_cli.kanban_command(args) != 0
    with kbc.connect_closing() as conn:
        assert not _settle(conn, tid, run_id)[0]
        assert list(conn.iterdump()) == before
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setattr(kanban_cli, "_profile_author", lambda: "orchestrator")
    assert kanban_cli.kanban_command(args) == 0
    with kbc.connect_closing() as conn:
        after = list(conn.iterdump())
        assert kb.get_task(conn, tid).status == "blocked"
    assert kanban_cli.kanban_command(args) != 0
    with kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == after


@pytest.mark.parametrize("drift", [
    "run", "state", "claim", "expiry", "pid", "current_run", "open_run",
    "missing_result", "foreign_result", "missing_bound", "unknown_exit", "foreign_pid",
    "result_before_loop", "missing_loop", "unverified_birth",
])
def test_triage_settlement_refuses_drift_and_unknown_without_mutation(kanban_home, drift):
    with kbc.connect_closing() as conn:
        tid, run_id = _settled_loop(conn)
        mutations = {
            "run": ("SELECT 1", ()),
            "state": ("UPDATE tasks SET status='ready' WHERE id=?", (tid,)),
            "claim": ("UPDATE tasks SET claim_lock='other-owner' WHERE id=?", (tid,)),
            "expiry": ("UPDATE tasks SET claim_expires=1 WHERE id=?", (tid,)),
            "pid": ("UPDATE tasks SET worker_pid=99999 WHERE id=?", (tid,)),
            "current_run": ("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid)),
            "open_run": ("UPDATE task_runs SET ended_at=NULL WHERE id=?", (run_id,)),
            "missing_result": ("DELETE FROM task_events WHERE task_id=? AND kind='worker_result'", (tid,)),
            "missing_bound": ("DELETE FROM task_events WHERE task_id=? AND kind='worker_bound'", (tid,)),
            "missing_loop": ("DELETE FROM task_events WHERE task_id=? AND kind='block_loop_detected'", (tid,)),
            "result_before_loop": (
                "UPDATE task_events SET id=(SELECT MAX(id)+1 FROM task_events) "
                "WHERE task_id=? AND kind='block_loop_detected'", (tid,)),
            "unverified_birth": (
                "UPDATE task_events SET payload=json_set(payload, '$.started_at', 'unverified') "
                "WHERE task_id=? AND kind='worker_bound'", (tid,)),
            "foreign_result": (
                "UPDATE task_events SET payload=json_set(payload, '$.claim_lock', 'foreign') "
                "WHERE task_id=? AND kind='worker_result'", (tid,)),
            "unknown_exit": (
                "UPDATE task_events SET payload=json_set(payload, '$.exit_code', NULL) "
                "WHERE task_id=? AND kind='worker_result'", (tid,)),
            "foreign_pid": (
                "UPDATE task_events SET payload=json_set(payload, '$.pid', 666) "
                "WHERE task_id=? AND kind='worker_result'", (tid,)),
        }
        with kb.write_txn(conn):
            conn.execute(*mutations[drift])
        snapshot = list(conn.iterdump())
        ok, reason = _settle(conn, tid, run_id + 1 if drift == "run" else run_id)
        assert not ok and reason
        assert list(conn.iterdump()) == snapshot


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn,
            parent_id=parent,
            child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_with_terminal_parents_parks_then_escalates(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``dependency`` block whose parents are all terminal can never be
    satisfied by ``recompute_ready``: it must park in ``blocked`` as
    ``needs_input`` (no ``dependency_wait``, no re-promotion), say so on the
    CLI, and count toward the loop breaker so a re-block after an unblock
    reaches ``triage``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        # The edge exists before the run: link_tasks refuses to gate a running child retroactively.
        child = _running_task(conn, title="child-of-done", parents=(parent,))

        # `hermes kanban block <child> --kind dependency waiting on upstream`
        args = argparse.Namespace(task_id=child, ids=None, reason=["waiting", "on", "upstream"], kind="dependency")
        assert kanban_cli._cmd_block(args) == 0
        assert "needs_input" in capsys.readouterr().out
        parked = kb.get_task(conn, child)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
        events = kb.list_events(conn, child)
        assert not [e for e in events if e.kind == "dependency_wait"]
        blocked = [e for e in events if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"

        # A cron/human unblocks; the worker re-declares the same impossible wait.
        assert kb.unblock_task(conn, child)
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.block_task(conn, child, reason="still waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "triage"
        loop = [e for e in kb.list_events(conn, child) if e.kind == "block_loop_detected"][-1].payload
        assert loop["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Control: a genuine wait on an incomplete parent parks in ``todo``
    without counting a recurrence, survives a dispatch tick unspawned, and
    resumes once the parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent then stays
        # open while the child runs — the reopened-parent shape, forced the same way the loop below does.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            assert kb.block_task(conn, child, reason="wait", kind="dependency")
            parked = kb.get_task(conn, child)
            assert (parked.status, parked.block_kind, parked.block_recurrences) == ("todo", "dependency", 0)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, child).status == "todo"
        assert child not in spawns
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert child in [row[0] for row in res2.spawned]


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


