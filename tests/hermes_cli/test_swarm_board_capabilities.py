"""Opted-in workers coordinate through the existing CLI under their own identities."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import pytest
from agent.delegation_context import delegated_child_context
from gateway.session_context import scoped_current_session_id
from hermes_cli import kanban as kc, kanban_db as kb, kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "theseus-implementer")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("kanban:\n  worker_board_management: true\n")
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="existing task", initial_status="blocked")
    return home, tid


def test_delegated_child_can_coordinate_without_borrowing_parent_identity(board, monkeypatch):
    _, tid = board
    monkeypatch.setenv("HERMES_KANBAN_TASK", "parent-owned")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "19")
    with delegated_child_context("child-session"):
        result = kc.run_slash(f"comment {tid} child-result --author forged-parent")
        assert "Comment added" in result
        with kbc.connect() as conn:
            row = conn.execute("SELECT author,body FROM task_comments WHERE task_id=?", (tid,)).fetchone()
            assert row["author"] == "theseus-implementer/session:child-session"
            assert row["body"] == "child-result"
        from tools.environments.local import build_subprocess_env
        child_env = build_subprocess_env()
        assert child_env["HERMES_SESSION_ID"] == "child-session"
        assert "HERMES_KANBAN_TASK" not in child_env
        assert child_env["HERMES_DELEGATED_CHILD_CONTEXT"]
        script = "from hermes_cli.kanban import run_slash; print(run_slash(" + repr(f"comment {tid} subprocess-result") + "))"
        result = subprocess.run([sys.executable, "-c", script], env=child_env,
                                capture_output=True, text=True, check=True)
        assert "Comment added" in result.stdout
        with kbc.connect() as conn:
            row = conn.execute("SELECT author FROM task_comments WHERE task_id=? AND body=?",
                               (tid, "subprocess-result")).fetchone()
            assert row["author"] == "theseus-implementer/session:child-session"


def test_worker_can_unblock_with_explicit_grant_without_changing_run_identity(board, monkeypatch):
    _, tid = board
    monkeypatch.setenv("HERMES_KANBAN_TASK", "owned-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "19")
    with scoped_current_session_id("finite-session"):
        result = kc.run_slash(f"unblock {tid}")
        assert "Unblocked" in result
        with pytest.raises(ValueError, match="scoped to task"):
            kc._worker_run_id_for(tid)
        assert kc._worker_run_id_for("owned-task") == 19
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status != "blocked"
        payload = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='unblocked'",
                               (tid,)).fetchone()[0]
        assert json.loads(payload)["actor"] == "theseus-implementer/session:finite-session"


def test_ungranted_child_remains_read_only(board):
    home, tid = board
    (home / "config.yaml").write_text("kanban:\n  worker_board_management: false\n")
    with delegated_child_context("child-session"):
        result = kc.run_slash(f"comment {tid} forbidden")
        assert "cannot mutate" in result
        with kbc.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,)).fetchone()[0] == 0


def test_peer_message_uses_actual_finite_session_author(board, monkeypatch):
    from hermes_cli.subcommands.peer import _turn_body
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    with scoped_current_session_id("finite-session"):
        body = _turn_body("question", message_key="input")
    assert body["author"]["is_bot"] is True
    assert body["author"]["id"].endswith("/theseus-implementer/session:finite-session")
    assert body["author"]["name"] == "theseus-implementer/session:finite-session"


def test_grant_without_session_identity_does_not_promote_child(board, monkeypatch):
    _, tid = board
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    with delegated_child_context(""):
        assert "cannot mutate" in kc.run_slash(f"comment {tid} anonymous")


def test_peer_sender_does_not_borrow_inbound_author(board, monkeypatch):
    from hermes_cli.subcommands.peer import _turn_body
    monkeypatch.setenv("HERMES_TURN_AUTHOR", '{"id":"bot:parent","name":"parent","is_bot":true}')
    with delegated_child_context("child-session"):
        author = _turn_body("question", message_key="input")["author"]
    assert author["name"] == "theseus-implementer/session:child-session"
    assert "parent" not in author["id"]


def test_opted_in_worker_release_still_requires_exact_terminal_run(board, monkeypatch):
    from hermes_cli import kanban_db_dispatch as dispatch
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="terminal attempt")
        task = kb.claim_task(conn, tid)
        rid = task.current_run_id
        dispatch._record_task_failure(conn, tid, "fixture failure", outcome="crashed",
                                      release_claim=True, end_run=True, force_trip=True)
        assert kb.get_task(conn, tid).status == "blocked"
    monkeypatch.setenv("HERMES_KANBAN_TASK", "owned-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "19")
    with scoped_current_session_id("finite-session"):
        assert "Edited" in kc.run_slash(f"edit {tid} --max-retries 1 --expected-terminal-run {rid}")
        assert "cannot unblock" in kc.run_slash(f"unblock {tid} --expected-terminal-run {rid + 1}")
        with kbc.connect() as conn:
            assert kb.get_task(conn, tid).status == "blocked"
            assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'",
                                (tid,)).fetchone()[0] == 0
        assert "Unblocked" in kc.run_slash(f"unblock {tid} --expected-terminal-run {rid}")


@pytest.mark.parametrize("section", ["null", "[]", "wrong", "{worker_board_management: wrong}"])
def test_malformed_grant_denies_child_without_breaking_regular_events(board, section):
    home, tid = board
    (home / "config.yaml").write_text(f"kanban: {section}\n")
    with scoped_current_session_id("ordinary-session"):
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, "ordinary", "still usable")
    with delegated_child_context("child-session"):
        assert "cannot mutate" in kc.run_slash(f"comment {tid} denied")


def test_grant_follows_bound_profile_and_cleared_child_identity(board, tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.swarm_capabilities import worker_board_management_enabled
    other = tmp_path / "other-profile"
    other.mkdir()
    (other / "config.yaml").write_text("kanban:\n  worker_board_management: false\n")
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-parent")
    with scoped_current_session_id("child-session"):
        assert worker_board_management_enabled()
        token = set_hermes_home_override(other)
        try:
            assert not worker_board_management_enabled()
        finally:
            reset_hermes_home_override(token)
        assert worker_board_management_enabled()
    with scoped_current_session_id(""):
        assert not worker_board_management_enabled()
