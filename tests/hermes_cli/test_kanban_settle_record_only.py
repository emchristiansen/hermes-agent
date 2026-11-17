"""``settle-record-only``: record an accepted, stopped card done with no cleanup, and keep its workspace.

t_11fe6077, 2026-11-14: the card's outcome was accepted, its run had crashed and a separate
housekeeping admission for its scratch workspace was of unknown outcome. Every route to Done
deleted that workspace after the commit, released its LSP clients and could kill a dead assignee
tmux session. These tests replay that shape on isolated boards and go through the real CLI: the
settlement, then each later cleanup route (archive, ``gc``, the parent sweep, the gateway
worker's end-of-turn retry, another card's cleanup of an overlapping path, board removal, a
restarted process). They also check every refusal leaves the card as it was, that the settlement
and a removal cannot interleave, and that ordinary completion still cleans up."""
import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_retention as kbr
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_parser

RESULT = "Accepted: both outcomes verified by the original executor run; no replay."
ACCEPTED = "comment 98459 (canonical acceptance)"
CUSTODY = "housekeeping admission c419 unknown; workspace retained unchanged"


@pytest.fixture(name="env")
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    calls = {"lsp": [], "tmux": [], "hook": []}
    # The real cleanup runs (rmtree, containment, in-use checks); only the effects that reach
    # outside the fixture (language servers, tmux, plugin hooks) are recorded instead.
    monkeypatch.setattr(kbw, "release_lsp_clients", lambda path: calls["lsp"].append(path))
    monkeypatch.setattr(kbw, "_cleanup_worker_tmux", lambda conn, tid: calls["tmux"].append(tid))
    monkeypatch.setattr(kb, "_fire_task_hook", lambda *a, **k: calls["hook"].append(a[0]))
    with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
        conn = kbc.connect()
        yield conn, calls, home
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


def scratch(conn, title="accepted work", path=None, **create):
    """A card with a real managed scratch directory holding content."""
    tid = kb.create_task(conn, title=title, assignee="implementer", workspace_kind="scratch",
                         workspace_path=str(path) if path else None, **create)
    ws = Path(kb.get_task(conn, tid).workspace_path or kb.workspaces_root() / tid)
    if not kb.get_task(conn, tid).workspace_path:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (str(ws), tid))
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "evidence.txt").write_text("original bytes")
    return tid, ws


def crashed_and_blocked(conn, tid):
    """The original shape: the executor's run binds a worker, the worker records its own exit 1,
    the run ends crashed and the card is blocked. Returns the ended run id."""
    task = kb.claim_task(conn, tid, claimer="dispatcher")
    run_id, lock = task.current_run_id, task.claim_lock
    dispatch._set_worker_pid(conn, tid, os.getpid(), run_id=run_id, claim_lock=lock)
    kb._append_event(conn, tid, "worker_bound", {"pid": os.getpid(), "claim_lock": lock,
                                                 "started_at": "fixture"}, run_id=run_id)
    kb._append_event(conn, tid, "worker_result", {"pid": os.getpid(), "claim_lock": lock, "exit_code": 1,
                                                  "last_output": "crashed"}, run_id=run_id)
    conn.commit()
    # The breaker gives up at once, as the dispatcher did: blocked with the crashed run still latest.
    dispatch._record_task_failure(conn, tid, "worker exited 1", outcome="crashed", failure_limit=1,
                                  release_claim=True, end_run=True)
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.claim_lock) == ("blocked", None, None)
    return run_id


def settle_args(tid, run_id, *extra):
    return ("settle-record-only", tid, "--expected-terminal-run", str(run_id), "--result", RESULT,
            "--accepted-by", ACCEPTED, "--custody", CUSTODY, *extra)


def image(conn, tid):
    def rows(sql):
        return [dict(r) for r in conn.execute(sql, (tid,))]
    return (rows("SELECT * FROM tasks WHERE id=?"), rows("SELECT * FROM task_runs WHERE task_id=? ORDER BY id"),
            rows("SELECT * FROM task_events WHERE task_id=? ORDER BY id"),
            rows("SELECT * FROM task_workspace_retentions WHERE task_id=? ORDER BY id"))


def tree(ws):
    return sorted((str(p.relative_to(ws)), p.read_bytes()) for p in ws.rglob("*") if p.is_file())


# --- the settlement itself -------------------------------------------------------------------


def test_settlement_records_done_retains_workspace_and_attributes_nothing_to_the_executor(env, capsys):
    conn, calls, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    before = tree(ws)
    executor_run = dict(conn.execute("SELECT * FROM task_runs WHERE id=?", (original,)).fetchone())
    calls["hook"].clear()  # the fixture's own claim fired kanban_task_claimed

    rc, out = outcome(capsys, *settle_args(tid, original, "--metadata", json.dumps({"bead": "ob-x"})))

    assert rc == 0, out
    assert f"workspace retained at {ws}" in out
    task = kb.get_task(conn, tid)
    assert (task.status, task.result, task.assignee) == ("done", RESULT, "implementer")
    assert ws.is_dir() and tree(ws) == before
    assert calls == {"lsp": [], "tmux": [], "hook": []}
    runs = conn.execute("SELECT id, profile, outcome, metadata, started_at, ended_at FROM task_runs "
                        "WHERE task_id=? ORDER BY id", (tid,)).fetchall()
    assert runs[0]["id"] == original and runs[0]["profile"] == "implementer"
    # The executor's run is untouched, byte for byte.
    assert dict(conn.execute("SELECT * FROM task_runs WHERE id=?", (original,)).fetchone()) == executor_run
    book = runs[-1]
    assert book["profile"] is None and book["outcome"] == "completed"
    assert book["started_at"] == book["ended_at"]
    record = json.loads(book["metadata"])
    settlement = record[kbr.SETTLEMENT_KEY]
    assert record["bead"] == "ob-x"
    assert settlement["performed_work"] is False and settlement["kind"] == "record_only"
    assert settlement["expected_terminal_run"] == original and settlement["original_profile"] == "implementer"
    assert settlement["accepted_by"] == ACCEPTED and settlement["custody"] == CUSTODY
    assert settlement["actor"] and settlement["cleanup"] == "none"
    assert settlement["workspace_retained"] == {"kind": "scratch", "path": str(ws)}
    event = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='completed'",
                                    (tid,)).fetchone()[0])
    assert event["settlement"] == settlement
    (row,) = kbr.retentions_for_task(conn, tid)
    assert row["workspace_path"] == str(ws) and row["settlement"] == settlement

    rc, out = outcome(capsys, "show", tid, "--json")
    shown = json.loads(out)
    assert shown["workspace_retentions"][0]["workspace_path"] == str(ws)
    rc, out = outcome(capsys, "show", tid)
    assert f"retained:  {ws}" in out and "no expiry" in out


def test_ordinary_completion_still_cleans_an_unretained_workspace(env, capsys):
    conn, calls, _ = env
    tid, ws = scratch(conn)
    crashed_and_blocked(conn, tid)
    rc, out = outcome(capsys, "complete", tid, "--result", "done normally")
    assert rc == 0, out
    assert not ws.exists()
    assert calls["lsp"] == [str(ws)] and calls["tmux"] == [tid]


# --- later cleanup routes --------------------------------------------------------------------


def settled(conn, capsys, **kw):
    tid, ws = scratch(conn, **kw)
    rc, out = outcome(capsys, *settle_args(tid, crashed_and_blocked(conn, tid)))
    assert rc == 0, out
    return tid, ws, tree(ws)


def test_archive_and_gc_keep_the_retained_workspace(env, capsys):
    conn, calls, _ = env
    tid, ws, before = settled(conn, capsys)
    rc, out = outcome(capsys, "archive", tid)
    assert rc == 0, out
    assert kb.get_task(conn, tid).status == "archived"
    rc, out = outcome(capsys, "gc")
    assert rc == 0 and "0 workspace(s)" in out, out
    assert tree(ws) == before
    assert calls["lsp"] == [] and calls["tmux"] == []
    # The retention is the record: no per-sweep deferral events pile up.
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind="
                        "'workspace_cleanup_deferred_shared'", (tid,)).fetchone()[0] == 0


def test_gc_still_collects_unretained_archived_workspaces(env, capsys):
    conn, _, _ = env
    tid, ws, before = settled(conn, capsys)
    other, other_ws = scratch(conn, title="ordinary")
    assert kb.archive_task(conn, other)
    other_ws.mkdir(exist_ok=True)  # archive's cleanup removed it; gc is the backstop for a survivor
    run("archive", tid)
    rc, out = outcome(capsys, "gc")
    assert rc == 0 and "1 workspace(s)" in out, out
    assert not other_ws.exists() and tree(ws) == before


def test_gateway_end_of_turn_retry_keeps_the_retained_workspace(env, capsys):
    from gateway.session_kanban import _remove_finished_workspace
    conn, calls, _ = env
    tid, ws, before = settled(conn, capsys)
    cwd = os.getcwd()
    try:
        _remove_finished_workspace(conn, {"task_id": tid, "db": str(kb.kanban_db_path())})
    finally:
        os.chdir(cwd)
    assert tree(ws) == before and calls["tmux"] == []


def test_parent_sweep_keeps_a_retained_parent(env, capsys):
    conn, calls, _ = env
    parent, ws = scratch(conn, title="parent")
    original = crashed_and_blocked(conn, parent)
    child = kb.create_task(conn, title="child", assignee="implementer", workspace_kind="dir",
                           workspace_path=str(ws.parent / "child-dir"))
    kb.link_tasks(conn, parent, child)
    rc, out = outcome(capsys, *settle_args(parent, original))
    assert rc == 0, out
    before = tree(ws)
    rc, out = outcome(capsys, "complete", child, "--result", "child done")
    assert rc == 0, out
    assert tree(ws) == before and calls["lsp"] == []


def test_another_cards_cleanup_cannot_remove_an_ancestor_or_the_same_directory(env, capsys):
    conn, calls, _ = env
    group = kb.workspaces_root() / "group"
    _, ws, before = settled(conn, capsys, path=group / "inner")
    ancestor, _ = scratch(conn, title="ancestor", path=group)
    same, _ = scratch(conn, title="same dir", path=ws)
    for other in (ancestor, same):
        rc, out = outcome(capsys, "complete", other, "--result", "other card done")
        assert rc == 0, out
    assert tree(ws) == before and group.is_dir()
    assert calls["lsp"] == []
    # Their own (unrelated) assignee tmux cleanup is ordinary completion behaviour.
    assert calls["tmux"] == [ancestor, same]


def test_retention_survives_a_restarted_process(env, capsys):
    conn, _, _ = env
    tid, ws, before = settled(conn, capsys)
    run("archive", tid)
    conn.close()
    kbc._INITIALIZED_PATHS.clear()
    # A fresh interpreter, the normal CLI entry point and no fixture stubs: gc must keep it.
    env_vars = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_KANBAN_TASK")}
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "gc"], env=env_vars,
                          capture_output=True, text=True, timeout=120, check=False,
                          cwd=Path(__file__).resolve().parents[2])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 workspace(s)" in proc.stdout
    assert tree(ws) == before


def test_hard_deleting_the_task_keeps_its_retention(env, capsys):
    conn, _, _ = env
    tid, ws, before = settled(conn, capsys)
    assert kb.archive_task(conn, tid) and kb.delete_archived_task(conn, tid)
    assert kb.get_task(conn, tid) is None
    assert kbw._workspace_in_use_by_other(conn, "t_other", ws) == kbr.RETAINED
    other, _ = scratch(conn, title="reuses the path", path=ws)
    run("complete", other, "--result", "done")
    assert tree(ws) == before


def test_a_sibling_boards_retention_protects_the_path(env, capsys):
    conn, _, _ = env
    from hermes_cli import kanban_db_boards as boards
    boards.create_board("other")
    with kbc.connect_closing(board="other") as other_conn:
        tid, ws = scratch(other_conn, title="retained elsewhere")
        original = crashed_and_blocked(other_conn, tid)
        ok, reason = kbr.settle_record_only(other_conn, tid, expected_terminal_run=original, result=RESULT,
                                            accepted_by=ACCEPTED, custody=CUSTODY, author="operator")
        assert ok, reason
    assert kbw._workspace_in_use_by_other(conn, "t_default", ws) == kbr.RETAINED


def test_board_removal_is_refused_while_it_retains_a_workspace(env, capsys):
    _ = env
    from hermes_cli import kanban_db_boards as boards
    boards.create_board("held")
    with kbc.connect_closing(board="held") as held:
        tid, ws = scratch(held)
        original = crashed_and_blocked(held, tid)
        assert kbr.settle_record_only(held, tid, expected_terminal_run=original, result=RESULT,
                                      accepted_by=ACCEPTED, custody=CUSTODY, author="operator")[0]
    for archive in (True, False):
        with pytest.raises(ValueError, match="records 1 retained workspace"):
            boards.remove_board("held", archive=archive)
    assert ws.is_dir() and kb.board_dir("held").is_dir()
    boards.create_board("empty")
    assert boards.remove_board("empty", archive=False)["action"] == "deleted"


# --- refusals leave everything as it was -------------------------------------------------------


def assert_refused(capsys, conn, tid, ws, arguments, needle, rc_expected=1):
    before, files = image(conn, tid), tree(ws) if ws.is_dir() else None
    rc, out = outcome(capsys, *arguments)
    assert rc == rc_expected and needle in out, out
    assert image(conn, tid) == before
    assert (tree(ws) if ws.is_dir() else None) == files


def test_refuses_a_running_card_with_a_live_claim(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    task = kb.claim_task(conn, tid, claimer="dispatcher")
    assert_refused(capsys, conn, tid, ws, settle_args(tid, task.current_run_id), "status is running")


def test_refuses_review_scheduled_and_todo(env, capsys):
    conn, _, _ = env
    for status in ("review", "scheduled", "todo", "done"):
        tid, ws = scratch(conn, title=status)
        original = crashed_and_blocked(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        assert_refused(capsys, conn, tid, ws, settle_args(tid, original), f"status is {status}")


def test_refuses_a_pr_completion_contract(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET completion_contract='pr:github.com/o/r' WHERE id=?", (tid,))
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original), "requires PR acceptance")


def test_refuses_an_unfinished_parent(env, capsys):
    conn, _, _ = env
    parent = kb.create_task(conn, title="open parent", assignee="implementer")
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    kb.link_tasks(conn, parent, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original), "a parent is not done")


def test_refuses_a_run_that_is_not_the_latest_ended_attempt(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original + 1), "is not the latest attempt")


def test_refuses_while_a_bound_worker_may_be_live(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    # A later binding of the same pid with no result of its own: that worker may still be in its turn.
    kb._append_event(conn, tid, "worker_bound", {"pid": os.getpid(), "claim_lock": "x", "started_at": "fixture"},
                     run_id=original)
    conn.commit()
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original), "may still be running")


def test_refuses_when_the_workspace_is_already_gone(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    for path in sorted(ws.rglob("*"), reverse=True):
        path.unlink()
    ws.rmdir()
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original), "does not exist, so it cannot be retained")


def test_refuses_empty_evidence_reserved_metadata_and_a_denied_worker(env, capsys, monkeypatch):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    base = list(settle_args(tid, original))
    blank = base.copy()
    blank[blank.index("--custody") + 1] = "  "
    assert_refused(capsys, conn, tid, ws, blank, "custody must be nonempty", 2)
    assert_refused(capsys, conn, tid, ws, [*base, "--metadata", json.dumps({kbr.SETTLEMENT_KEY: {}})],
                   "must not supply", 2)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setattr("agent.swarm_capabilities.worker_board_management_enabled", lambda: False)
    assert_refused(capsys, conn, tid, ws, base, "orchestrator-only")


# --- races and unreadable state fail closed ----------------------------------------------------


def test_a_removal_holding_the_fence_makes_the_settlement_refuse(env, capsys, monkeypatch):
    conn, _, _ = env
    monkeypatch.setattr(kbr, "FENCE_TIMEOUT_SECONDS", 0.2)
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    with kbr.removal_fence() as held:
        assert held
        assert_refused(capsys, conn, tid, ws, settle_args(tid, original), "holds the removal fence")
    rc, out = outcome(capsys, *settle_args(tid, original))
    assert rc == 0, out


def test_a_held_fence_makes_removal_keep_the_directory(env, monkeypatch):
    conn, calls, _ = env
    monkeypatch.setattr(kbr, "FENCE_TIMEOUT_SECONDS", 0.2)
    tid, ws = scratch(conn)
    with kbr.removal_fence() as held:
        assert held
        assert kbw._remove_scratch_dir(conn, tid, ws) is False
    assert ws.is_dir() and calls["lsp"] == []


def test_a_settlement_racing_an_overlapping_removal_never_claims_a_deleted_workspace(env):
    """Another card's cleanup has passed its in-use check and is about to delete its directory,
    which contains this card's workspace (its LSP release runs between the two) when the settlement starts in another
    thread. The settlement must wait for the removal and then refuse, not record a retention of a
    directory that is gone."""
    conn, calls, _ = env
    group = kb.workspaces_root() / "group"
    tid, _ = scratch(conn, path=group / "inner")
    original = crashed_and_blocked(conn, tid)
    other, _ = scratch(conn, title="ancestor directory", path=group)
    outcome_box, started = {}, threading.Event()

    def settle_in_another_process_like_connection():
        with kbc.connect_closing() as own:
            started.set()
            outcome_box["result"] = kbr.settle_record_only(
                own, tid, expected_terminal_run=original, result=RESULT, accepted_by=ACCEPTED,
                custody=CUSTODY, author="operator")

    worker = threading.Thread(target=settle_in_another_process_like_connection)

    def lsp_release_then_race(path):
        calls["lsp"].append(path)
        worker.start()
        assert started.wait(5)
        # Give the settlement every chance to commit before the delete. Under the fence it
        # cannot, so this wait expires and the removal proceeds; without the fence it would
        # record a retention here and the delete would then remove the retained directory.
        worker.join(2)

    kbw.release_lsp_clients = lsp_release_then_race  # restored by the fixture's monkeypatch
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (other,))
    kbw._cleanup_workspace(conn, other)
    worker.join(30)
    ok, reason = outcome_box["result"]
    assert not group.exists()
    assert not ok and "does not exist" in reason
    assert kb.get_task(conn, tid).status == "blocked"
    assert kbr.retentions_for_task(conn, tid) == []


def test_unreadable_retention_state_keeps_every_directory(env, capsys):
    conn, calls, _ = env
    tid, ws = scratch(conn, title="ordinary archived")
    with kb.write_txn(conn):
        conn.execute("DROP TABLE task_workspace_retentions")
        conn.execute("CREATE TABLE task_workspace_retentions (garbage TEXT)")
        conn.execute("INSERT INTO task_workspace_retentions VALUES ('x')")
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))
    assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
    # The CLI re-runs the schema pass first; a malformed table stops gc before any sweep.
    rc, out = outcome(capsys, "gc")
    assert rc == 1 and "could not initialize database" in out, out
    assert kbw._remove_scratch_dir(conn, tid, ws) is False
    kbw._cleanup_workspace(conn, tid)
    assert ws.is_dir() and calls["lsp"] == [] and calls["tmux"] == []


def test_event_gc_cannot_erase_the_retention(env, capsys):
    """``gc`` prunes old events of done/archived cards, the settlement's ``completed`` event
    included; the retention lives in its own table and is untouched."""
    conn, _, _ = env
    tid, ws, before = settled(conn, capsys)
    run("archive", tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (tid,))
    rc, out = outcome(capsys, "gc", "--event-retention-days", "1")
    assert rc == 0, out
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
                        (tid,)).fetchone()[0] == 0
    (row,) = kbr.retentions_for_task(conn, tid)
    assert row["settlement"]["kind"] == "record_only"
    assert tree(ws) == before


def retained_inside_board(conn, board):
    """A default-board card whose retained workspace lies inside *board*'s directory."""
    from hermes_cli import kanban_db_boards as boards
    boards.create_board(board)
    tid, ws = scratch(conn, title="lives in another board", path=kb.board_dir(board) / "workspaces" / "kept")
    return tid, ws


def test_board_removal_is_refused_when_another_boards_retention_lies_inside_it(env):
    conn, _, _ = env
    from hermes_cli import kanban_db_boards as boards
    tid, ws = retained_inside_board(conn, "victim")
    original = crashed_and_blocked(conn, tid)
    assert kbr.settle_record_only(conn, tid, expected_terminal_run=original, result=RESULT,
                                  accepted_by=ACCEPTED, custody=CUSTODY, author="operator")[0]
    before = tree(ws)
    for archive in (True, False):
        with pytest.raises(ValueError, match=f"holds the workspace retained for task {tid}"):
            boards.remove_board("victim", archive=archive)
    assert tree(ws) == before and kb.board_dir("victim").is_dir()


def test_board_removal_fails_closed_when_a_board_database_is_unreadable(env):
    _ = env
    from hermes_cli import kanban_db_boards as boards
    boards.create_board("victim")
    with kbc.connect_closing(board="victim"):
        pass
    boards.create_board("broken")
    (kb.board_dir("broken") / "kanban.db").write_bytes(b"not a database at all, just bytes" * 8)
    with pytest.raises(ValueError, match="cannot be read"):
        boards.remove_board("victim", archive=False)
    assert kb.board_dir("victim").is_dir()


def test_board_removal_holding_the_fence_beats_a_racing_settlement(env, monkeypatch):
    """The board's check has passed (nothing retained yet) and it is about to delete the directory
    when a settlement of a workspace inside it starts. The settlement must wait for the fence and
    then refuse; without the fence it would record a retention that the delete then destroys."""
    conn, _, _ = env
    from hermes_cli import kanban_db_boards as boards
    tid, ws = retained_inside_board(conn, "victim")
    original = crashed_and_blocked(conn, tid)
    box, started = {}, threading.Event()

    def settle():
        with kbc.connect_closing() as own:
            started.set()
            box["result"] = kbr.settle_record_only(own, tid, expected_terminal_run=original, result=RESULT,
                                                   accepted_by=ACCEPTED, custody=CUSTODY, author="operator")

    worker = threading.Thread(target=settle)
    real_current = kb.get_current_board

    def current_board_then_race():  # runs inside remove_board after its retention check
        # The settlement's own connect() asks for the current board too; only the board
        # removal's call (on this thread) starts the race.
        if threading.current_thread() is threading.main_thread() and not worker.is_alive() \
                and "result" not in box:
            worker.start()
            assert started.wait(5)
            worker.join(2)
        return real_current()

    monkeypatch.setattr(kb, "get_current_board", current_board_then_race)
    assert boards.remove_board("victim", archive=False)["action"] == "deleted"
    worker.join(30)
    ok, reason = box["result"]
    assert not ws.exists()
    assert not ok and "does not exist" in reason
    assert kbr.retentions_for_task(conn, tid) == [] and kb.get_task(conn, tid).status == "blocked"


def test_board_removal_refuses_while_the_fence_is_held(env, monkeypatch):
    from hermes_cli import kanban_db_boards as boards
    monkeypatch.setattr(kbr, "FENCE_TIMEOUT_SECONDS", 0.2)
    boards.create_board("busy")
    with kbr.removal_fence() as held:
        assert held
        with pytest.raises(ValueError, match="holds the removal fence"):
            boards.remove_board("busy", archive=False)
    assert kb.board_dir("busy").is_dir()


# --- review RET-R2-01: only databases the cleanup routes read can hold a retention ------------


def fresh_cli(*arguments, **env_extra):
    """The normal CLI entry point in a new interpreter, with no fixture stubs."""
    env_vars = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_KANBAN_TASK")}
    env_vars.update(env_extra)
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *arguments], env=env_vars,
                          capture_output=True, text=True, timeout=180, check=False,
                          cwd=Path(__file__).resolve().parents[2])


def test_settlement_refuses_an_archived_boards_database_through_the_real_cli(env):
    """Review RET-R2-01: settling through an archived board's still-writable database used to
    succeed and record a retention no cleanup route reads; an equal-path card's archive then
    deleted the workspace. The settlement now refuses there with nothing changed."""
    _ = env
    from hermes_cli import kanban_db_boards as boards
    boards.create_board("gone")
    outside = kb.workspaces_root() / "outside-the-board"
    with kbc.connect_closing(board="gone") as gone:
        tid, ws = scratch(gone, title="settled from an archived board", path=outside)
        original = crashed_and_blocked(gone, tid)
    archived = Path(boards.remove_board("gone", archive=True)["new_path"]) / "kanban.db"
    assert archived.is_file()
    kbc._INITIALIZED_PATHS.clear()
    before = tree(ws)

    proc = fresh_cli(*settle_args(tid, original), HERMES_KANBAN_DB=str(archived))

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not one the workspace cleanup routes read" in proc.stderr + proc.stdout
    with sqlite3.connect(archived) as db:
        assert db.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] == "blocked"
        assert db.execute("SELECT COUNT(*) FROM task_workspace_retentions").fetchone()[0] == 0
    assert tree(ws) == before


def test_settlement_refuses_a_database_pinned_outside_the_kanban_home(env, tmp_path):
    _ = env
    pinned = tmp_path / "elsewhere" / "pinned.sqlite"
    pinned.parent.mkdir()
    other = kbc.connect(pinned)
    try:
        tid, _ = scratch(other, title="pinned")
        original = crashed_and_blocked(other, tid)
        ok, reason = kbr.settle_record_only(other, tid, expected_terminal_run=original, result=RESULT,
                                            accepted_by=ACCEPTED, custody=CUSTODY, author="operator")
        assert not ok and "not one the workspace cleanup routes read" in reason
        assert kb.get_task(other, tid).status == "blocked" and kbr.retentions_for_task(other, tid) == []
    finally:
        other.close()


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_another_boards_retention_survives_alias_archive_and_gc_in_fresh_processes(env, relation):
    """A retention on a named board protects its path from a default-board alias card's archive
    and from gc, both run as fresh normal CLI processes."""
    conn, _, _ = env
    from hermes_cli import kanban_db_boards as boards
    boards.create_board("holder")
    group = kb.workspaces_root() / "shared-group"
    retained = group / "inner"
    with kbc.connect_closing(board="holder") as holder:
        tid, ws = scratch(holder, title="retained on holder", path=retained)
        original = crashed_and_blocked(holder, tid)
        ok, reason = kbr.settle_record_only(holder, tid, expected_terminal_run=original, result=RESULT,
                                            accepted_by=ACCEPTED, custody=CUSTODY, author="operator")
        assert ok, reason
    alias_path = {"equal": retained, "ancestor": group, "descendant": retained / "sub"}[relation]
    alias, _ = scratch(conn, title=f"{relation} alias", path=alias_path)
    before = tree(ws)
    conn.close()
    kbc._INITIALIZED_PATHS.clear()
    for arguments in (("archive", alias), ("gc",)):
        proc = fresh_cli(*arguments)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ws.is_dir() and tree(ws) == before and alias_path.is_dir()


# --- review RET-R2-03: no artifact declarations on a route with no file effects -----------------


@pytest.mark.parametrize("metadata", [
    {"artifacts": ["/absolute/outside-report.txt"]},
    {"artifacts": ["WORKSPACE/evidence.txt"]},
    {"unattached_artifacts": [{"path": "/x", "reason": "forged"}]},
    {"_staged_artifacts": ["/x"]},
])
def test_artifact_declarations_are_refused_with_nothing_changed(env, capsys, metadata):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    text = json.dumps(metadata).replace("WORKSPACE", str(ws))
    assert_refused(capsys, conn, tid, ws, settle_args(tid, original, "--metadata", text),
                   "a record-only settlement has no file effects", 2)
    assert conn.execute("SELECT COUNT(*) FROM task_attachments WHERE task_id=?", (tid,)).fetchone()[0] == 0


def test_opaque_metadata_is_kept_and_nothing_is_promoted_for_delivery(env, capsys):
    conn, _, _ = env
    tid, ws = scratch(conn)
    original = crashed_and_blocked(conn, tid)
    opaque = {"bead": "ob-x", "notes": {"paths_mentioned": ["/absolute/outside-report.txt"]}}
    rc, out = outcome(capsys, *settle_args(tid, original, "--metadata", json.dumps(opaque)))
    assert rc == 0, out
    metadata = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                                       (tid,)).fetchone()[0])
    assert {k: metadata[k] for k in opaque} == opaque
    event = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='completed'",
                                    (tid,)).fetchone()[0])
    assert "artifacts" not in event and "unattached_artifacts" not in event
    assert conn.execute("SELECT COUNT(*) FROM task_attachments WHERE task_id=?", (tid,)).fetchone()[0] == 0
    assert ws.is_dir()
