"""A worker that completes its card from inside its scratch workspace keeps working there.

2026-11-04, t_b34a3c8b run 574: the worker's ``kanban_complete`` removed the scratch workspace
that was its own cwd, while the turn went on (the model reads the tool result; a steered comment
arrived). The next system-prompt rebuild during compression raised FileNotFoundError from
``os.getcwd()``, the worker exited 1 after its card was already done, and its admission settled
interrupted. Removal (and any parent cleanup it triggers) now waits until the bound worker's
turns are over; the worker then steps out to the board directory and runs it.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_workspace
from hermes_cli.kanban_db_connect import connect_closing


@pytest.fixture
def bound_scratch_worker(tmp_path, monkeypatch):
    """A claimed scratch card bound to this process, as ``bind_worker_context`` leaves a worker:
    its environment names the card and its cwd is the workspace."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="private fixture", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        workspace = kanban_db_workspace.resolve_workspace(task)
        kanban_db_workspace.set_workspace_path(conn, task_id, workspace)
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "worker_bound",
                             {"pid": os.getpid(), "claim_lock": task.claim_lock}, run_id=task.current_run_id)
    context = {"db": str(path), "task_id": task_id, "run_id": task.current_run_id,
               "claim_lock": task.claim_lock, "workspace": str(workspace)}
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.chdir(workspace)
    return path, context, Path(workspace)


def test_turn_after_completion_keeps_its_workspace_until_the_worker_ends(bound_scratch_worker, monkeypatch):
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools

    path, context, workspace = bound_scratch_worker
    (workspace / "AGENTS.md").write_text("project rules for the worker\n")
    seen = {}

    def turns(*_args):
        completed = json.loads(kanban_tools._handle_complete({"summary": "assessment delivered"}))
        assert completed["ok"] is True
        # The same turn continues: tools and the prompt rebuilt for compression still resolve
        # the worker's cwd and read its project context.
        seen["workspace_after_complete"] = workspace.is_dir()
        seen["prompt"] = build_context_files_prompt(cwd=None)
        (Path.cwd() / "late-note.txt").write_text("written after completion\n")
        return {"completed": True}

    monkeypatch.setattr(session_kanban, "_run_task_turns", turns)
    frame = {"policy": {"kanban_json": json.dumps(context)}}
    assert session_kanban.run_worker_turns(SimpleNamespace(), frame, []) == {"completed": True}

    assert seen["workspace_after_complete"] is True
    assert "project rules for the worker" in seen["prompt"]
    assert session_kanban.worker_exit_code(path, context) == 0  # not exit 1 after the card was done
    with connect_closing(path) as conn:
        assert kb.get_task(conn, context["task_id"]).status == "done"
    assert not workspace.exists()  # removed once the worker's turns ended, late writes with it
    assert Path.cwd() == Path(path).resolve().parent  # stepped out to the board directory


def test_completion_by_another_process_still_removes_it_at_once(bound_scratch_worker, monkeypatch, tmp_path):
    """An orchestrator (any process that is not the card's own worker) keeps today's behavior."""
    path, context, workspace = bound_scratch_worker
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, context["task_id"], summary="done elsewhere")
    assert not workspace.exists()


def test_an_unfinished_card_keeps_its_workspace_when_the_worker_ends(bound_scratch_worker, monkeypatch):
    """Only a done card's deferred removal runs at the worker's end; a failed attempt's workspace
    stays for the retry, as before."""
    path, context, workspace = bound_scratch_worker
    monkeypatch.setattr(session_kanban, "_run_task_turns", lambda *_args: {"failed": True})
    frame = {"policy": {"kanban_json": json.dumps(context)}}
    session_kanban.run_worker_turns(SimpleNamespace(), frame, [])
    assert session_kanban.worker_exit_code(path, context) == 1
    assert workspace.is_dir()
    assert Path.cwd() == workspace.resolve()


def test_real_task_turn_keeps_artifacts_and_context_after_completion(bound_scratch_worker):
    """The real ``_run_task_turns`` path (only the model call is scripted): the completion's
    declared artifact is attached, the prompt still reads the workspace's AGENTS.md, a late
    relative write lands in the workspace, and the workspace goes once the worker ends.
    (Derived from the t_19d9e76d reviewer control.)"""
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools

    path, context, workspace = bound_scratch_worker
    context.update(skills=[], context="review", goal_mode=False)
    frame = {"text": "review", "policy": {"kanban_json": json.dumps(context)}}
    (workspace / "AGENTS.md").write_text("Independent context sentinel.\n")
    (workspace / "report.txt").write_text("durable bytes\n")

    def conversation(*_args, **_kwargs):
        response = json.loads(kanban_tools._handle_complete({
            "summary": "review delivered", "artifacts": [str(workspace / "report.txt")]}))
        assert response["ok"], response
        assert Path.cwd() == workspace
        assert "Independent context sentinel." in build_context_files_prompt(cwd=None)
        Path("late.txt").write_text("same directory\n")
        assert (workspace / "late.txt").read_text() == "same directory\n"
        return {"completed": True, "final_response": "done"}

    agent = SimpleNamespace(session_id="probe", run_conversation=conversation)
    assert session_kanban.run_worker_turns(agent, frame, [])["completed"]
    assert session_kanban.worker_exit_code(path, context) == 0
    assert not workspace.exists()
    with connect_closing(path) as conn:
        attachments = kb.list_attachments(conn, context["task_id"])
        assert [Path(a.stored_path).read_text() for a in attachments] == ["durable bytes\n"]


@pytest.mark.parametrize("kind", ["dir", "scratch"])
def test_last_child_sharing_its_parents_workspace_keeps_it_until_its_worker_ends(
        bound_scratch_worker, monkeypatch, kind):
    """t_19d9e76d CWD-01: a completed parent retains its workspace for a child working in it;
    the child's own completion used to run the parent's deferred removal and delete the child's
    live cwd. Now the child's whole cleanup, the parent sweep included, waits for its worker."""
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools

    path, parent_context, workspace = bound_scratch_worker
    (workspace / "AGENTS.md").write_text("handoff rules\n")
    with connect_closing(path) as conn:
        child = kb.create_task(conn, title="consume parent files", assignee="default",
                               parents=[parent_context["task_id"]], workspace_kind=kind,
                               workspace_path=str(workspace))
        assert kb.complete_task(conn, parent_context["task_id"], summary="handoff",
                                expected_run_id=parent_context["run_id"])
    session_kanban._record_worker_result(parent_context, 0, "done")  # the parent's worker ends
    assert workspace.is_dir()  # retained for the active child
    monkeypatch.chdir(workspace)
    with connect_closing(path) as conn:
        child_task = kb.claim_task(conn, child, claimer="private-host:child")
        with kb.write_txn(conn):
            kb._append_event(conn, child, "worker_bound",
                             {"pid": os.getpid(), "claim_lock": child_task.claim_lock}, run_id=child_task.current_run_id)
    child_context = {"db": str(path), "task_id": child, "run_id": child_task.current_run_id,
                     "claim_lock": child_task.claim_lock, "workspace": str(workspace)}
    monkeypatch.setenv("HERMES_KANBAN_TASK", child)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(child_task.current_run_id))

    assert json.loads(kanban_tools._handle_complete({"summary": "child delivered"}))["ok"]
    assert "handoff rules" in build_context_files_prompt(cwd=None)
    Path("late.txt").write_text("written after completion\n")
    assert (workspace / "late.txt").is_file()

    session_kanban._record_worker_result(child_context, 0, "done")
    assert not workspace.exists()  # nothing needs it any more: removed, not leaked


@pytest.mark.parametrize("where", ["inside", "subdirectory", "outside"])
def test_worker_end_runs_the_deferred_cleanup_wherever_its_cwd_is(bound_scratch_worker, monkeypatch, tmp_path, where):
    """t_19d9e76d CWD-02: the deferred cleanup ran only if the worker's cwd was still inside the
    workspace at its end; a worker that had moved elsewhere leaked it."""
    path, context, workspace = bound_scratch_worker
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, context["task_id"], summary="finished", expected_run_id=context["run_id"])
    assert workspace.is_dir()  # deferred: this process is the card's worker
    if where == "subdirectory":
        (workspace / "sub").mkdir()
        monkeypatch.chdir(workspace / "sub")
    elif where == "outside":
        monkeypatch.chdir(tmp_path)
    session_kanban._record_worker_result(context, 0, "done")
    assert not workspace.exists()


def test_a_waiting_childs_handoff_survives_the_parents_worker_end(bound_scratch_worker, monkeypatch):
    path, context, workspace = bound_scratch_worker
    with connect_closing(path) as conn:
        child = kb.create_task(conn, title="waiting consumer", assignee="default", parents=[context["task_id"]])
        (workspace / "handoff.txt").write_text("handoff bytes")
        assert kb.complete_task(conn, context["task_id"], summary="handoff", expected_run_id=context["run_id"])
    session_kanban._record_worker_result(context, 0, "done")
    assert (workspace / "handoff.txt").read_text() == "handoff bytes"
    monkeypatch.delenv("HERMES_KANBAN_TASK")  # the child is completed elsewhere, by another process
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, child, summary="consumed")
    assert not workspace.exists()
