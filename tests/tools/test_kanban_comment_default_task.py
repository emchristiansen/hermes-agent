"""A dispatcher-owned worker's bare ``kanban_comment`` goes to its own task.

Cryo t_7752959d run 25388: the worker called ``kanban_comment(body=...)`` and was told "this
session has no dispatcher-assigned task to default to", which was false: the same worker's
``kanban_block`` resolved its task and run. Only the comment handler skipped the default. A
caller that merely inherited ``HERMES_KANBAN_TASK`` (a delegate child, an in-process cron run)
must still name its task, and a chat session with no task still gets the kanban_list pointer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _comments(tid):
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return [r[0] for r in conn.execute("SELECT body FROM task_comments WHERE task_id = ?", (tid,))]
    finally:
        conn.close()


def test_a_workers_bare_comment_goes_to_its_own_task(worker_task):
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_comment({"body": "progress note"}))
    assert out.get("ok") and out["task_id"] == worker_task, out
    assert _comments(worker_task) == ["progress note"]


def test_an_inherited_task_is_not_a_default(monkeypatch, worker_task):
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_is_dispatcher_owned_worker", lambda: False)
    out = json.loads(kt._handle_comment({"body": "from a cron run"}))
    assert out.get("error") and "does not own" in out["error"], out
    assert _comments(worker_task) == []


def test_a_chat_session_still_names_kanban_list(monkeypatch, worker_task):
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = json.loads(kt._handle_comment({"body": "hello"}))
    assert out.get("error") and "kanban_list" in out["error"], out
    assert _comments(worker_task) == []
