"""A worker that completes its card from inside its scratch workspace keeps working there.

2026-11-04, t_b34a3c8b run 574: the worker's ``kanban_complete`` removed the scratch workspace
that was its own cwd, while the turn went on (the model reads the tool result; a steered comment
arrived). The next system-prompt rebuild during compression raised FileNotFoundError from
``os.getcwd()``, the worker exited 1 after its card was already done, and its admission settled
interrupted. No route removes a directory while a live bound worker (of this card or any other)
works at or under it; the last such worker's end steps out to the board directory and runs the
cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
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


def _ended_process_identity():
    """``(pid, started_at)`` of a process that has exited, fingerprinted as ``bind_worker_context``
    records a bound worker: what a hard-killed worker leaves behind in ``worker_bound``."""
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    fingerprint = _process_fingerprint(process.pid)
    process.communicate(b"", timeout=30)
    assert fingerprint
    return process.pid, fingerprint


def _rebind(path, context, pid, started_at):
    """Replace the fixture's live binding (this process, now finished) with a worker ``pid``."""
    with connect_closing(path) as conn, kb.write_txn(conn):
        kb._append_event(conn, context["task_id"], "worker_result",
                         {"pid": os.getpid(), "claim_lock": context["claim_lock"], "exit_code": 0},
                         run_id=context["run_id"])
        kb._append_event(conn, context["task_id"], "worker_bound",
                         {"pid": pid, "claim_lock": context["claim_lock"], "started_at": started_at},
                         run_id=context["run_id"])


def test_completion_by_another_process_removes_it_once_no_worker_is_live(bound_scratch_worker, monkeypatch, tmp_path):
    """An orchestrator completing a card whose worker died hard (no ``worker_result``) removes
    the workspace at once: a dead bound pid no longer holds it."""
    path, context, workspace = bound_scratch_worker
    _rebind(path, context, *_ended_process_identity())
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, context["task_id"], summary="done elsewhere")
    assert not workspace.exists()


def test_completion_by_another_process_waits_for_the_live_worker(bound_scratch_worker, monkeypatch, tmp_path):
    """An orchestrator completing a card while its worker's turn still runs there keeps the
    workspace; the worker's end removes it."""
    path, context, workspace = bound_scratch_worker
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    with connect_closing(path) as conn:
        assert kb.complete_task(conn, context["task_id"], summary="done elsewhere")
    assert workspace.is_dir()
    session_kanban._record_worker_result(context, 0, "done")
    assert not workspace.exists()


def test_gc_keeps_an_archived_cards_workspace_while_its_worker_is_live(bound_scratch_worker, monkeypatch, tmp_path):
    """``kanban gc`` asks the same question: an archived card's scratch stays while its bound
    worker is alive, and goes once that worker has died hard."""
    from hermes_cli import kanban_ops
    path, context, workspace = bound_scratch_worker
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(event_retention_days=0, log_retention_days=0)
    with connect_closing(path) as conn:
        assert kb.archive_task(conn, context["task_id"], signal_fn=lambda *_: None)
    assert workspace.is_dir()
    assert kanban_ops._cmd_gc(args) == 0
    assert workspace.is_dir()
    _rebind(path, context, *_ended_process_identity())
    assert kanban_ops._cmd_gc(args) == 0
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


# A second, real worker process: it claims its card, binds as the owner binds a managed worker
# (environment, ``worker_bound`` with its pid fingerprint, cwd = its workspace) and runs a real
# task turn whose scripted model completes the card, then waits for "continue" on stdin before
# the rest of its turn (a prompt rebuild and a relative write in its cwd). Its turn's end runs
# the real finalizer. (Derived from the t_99ab13a0 reviewer probe.)
_OVERLAP_WORKER = r'''
import json, sys
repo, db, task_id, *lane = sys.argv[1:]
sys.path.insert(0, repo)
from pathlib import Path
from types import SimpleNamespace
from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
with connect_closing(Path(db)) as conn:
    if lane == ["review"]:
        task = kb.claim_review_task(conn, task_id, claimer="overlap-worker")
    else:
        task = kb.claim_task(conn, task_id, claimer="overlap-worker")
context = dict(task_id=task_id, board="default", db=db, run_id=task.current_run_id, claim_lock=task.claim_lock,
               workspace=task.workspace_path, workspaces_root=str(kb.workspaces_root()), profile="default",
               branch=None, tenant=None, goal_mode=False, max_runtime_seconds=None, skills=[], context="")
frame = {"text": "work", "policy": {"source": "kanban", "kanban_json": json.dumps(context)}}
session_kanban.bind_worker_context(frame)


def conversation(*_args, **_kwargs):
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools
    assert json.loads(kanban_tools._handle_complete({"summary": "child delivered"}))["ok"]
    print("COMPLETED", flush=True)
    assert sys.stdin.readline().strip() == "continue"
    seen = {}
    try:
        seen["prompt"] = build_context_files_prompt(cwd=None)
        Path("late-child.txt").write_text("child late bytes")
        seen["late"] = Path(context["workspace"], "late-child.txt").read_text()
    except Exception as exc:
        seen["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(seen), flush=True)
    return {"completed": True, "final_response": "done"}


session_kanban.run_worker_turns(SimpleNamespace(session_id="overlap", run_conversation=conversation), frame, [])
print(json.dumps({"exit_code": session_kanban.worker_exit_code(db, context)}), flush=True)
'''


def _start_overlap_worker(db, task_id, lane="work"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
    repo = Path(__file__).resolve().parents[2]
    worker = subprocess.Popen([sys.executable, "-u", "-c", _OVERLAP_WORKER, str(repo), str(db), task_id, lane],
                              cwd=repo, env=env, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    line = worker.stdout.readline()
    if line.strip() != "COMPLETED":
        worker.kill()
        raise AssertionError((line, worker.communicate(timeout=30)))
    return worker


def _finish_overlap_worker(worker):
    """Let the worker's turn go on, then end; returns what its late turn saw."""
    out, err = worker.communicate("continue\n", timeout=120)
    assert worker.returncode == 0, (out, err)
    seen, ended = (json.loads(line) for line in out.strip().splitlines()[-2:])
    assert ended == {"exit_code": 0}, (out, err)
    return seen


@pytest.mark.parametrize("order,kind", [("child-first", "separate"), ("parent-first", "dir"),
                                        ("parent-first", "scratch")])
def test_overlapping_worker_turns_keep_each_live_workers_workspace(bound_scratch_worker, order, kind):
    """t_99ab13a0 CWD-03, two real worker processes whose post-completion turns overlap.
    child-first: a child with its own workspace ends while the parent's turn still runs; the
    child's finalizer swept the parent's workspace (the parent's cwd). parent-first: the parent
    ends while a child sharing its workspace (dir or scratch) is still in its turn; the parent's
    finalizer removed the child's cwd. Card status cannot tell; the bound workers' liveness does.
    The last worker to end removes what nothing needs any more."""
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools

    path, parent, workspace = bound_scratch_worker
    parent.update(skills=[], context="", goal_mode=False)
    (workspace / "AGENTS.md").write_text("overlap sentinel\n")
    child_workspace = workspace.parent / "overlap-child" if kind == "separate" else workspace
    child_workspace.mkdir(exist_ok=True)
    with connect_closing(path) as conn:
        child = kb.create_task(conn, title="overlap child", assignee="default", parents=[parent["task_id"]],
                               workspace_kind="scratch" if kind == "separate" else kind,
                               workspace_path=str(child_workspace))
    observed = {}

    def parent_turn(*_args, **_kwargs):
        assert json.loads(kanban_tools._handle_complete({"summary": "parent delivered"}))["ok"]
        worker = _start_overlap_worker(path, child)
        if order == "parent-first":
            observed["worker"] = worker
        else:
            observed["child"] = _finish_overlap_worker(worker)  # child's turn and finalizer end first
            assert not child_workspace.exists()  # its own scratch: nothing else uses it
            # this parent's turn goes on in its workspace
            assert "overlap sentinel" in build_context_files_prompt(cwd=None)
            Path("late-parent.txt").write_text("parent late bytes")
            assert (workspace / "late-parent.txt").read_text() == "parent late bytes"
        return {"completed": True, "final_response": "done"}

    agent = SimpleNamespace(session_id="overlap-parent", run_conversation=parent_turn)
    assert session_kanban.run_worker_turns(agent, {"text": "work", "policy": {"kanban_json": json.dumps(parent)}}, [])
    assert session_kanban.worker_exit_code(path, parent) == 0
    if order == "parent-first":
        assert workspace.is_dir()  # the child's worker is still in its turn there
        observed["child"] = _finish_overlap_worker(observed["worker"])
        assert "overlap sentinel" in observed["child"].get("prompt", ""), observed["child"]
        assert observed["child"].get("late") == "child late bytes", observed["child"]
    assert "error" not in observed["child"], observed["child"]
    assert not workspace.exists()  # the last worker's end removed it
    with connect_closing(path) as conn:
        assert {kb.get_task(conn, t).status for t in (parent["task_id"], child)} == {"done"}


def test_a_review_handoff_keeps_the_implementers_workspace_until_it_ends(bound_scratch_worker):
    """t_3d35658c CWD-04: the implementer hands its card to review (``kanban_request_review``)
    and its turn goes on; a reviewer process claims the same card as a new run
    (``claim_review_task``), completes it and ends first. Only the card's latest binding used to
    count, so the reviewer's finalizer removed the implementer's cwd. Every unfinished binding
    counts now; the implementer's own end removes the workspace."""
    from agent.prompt_builder import build_context_files_prompt
    from tools import kanban_tools

    path, implementer, workspace = bound_scratch_worker
    implementer.update(skills=[], context="", goal_mode=False)
    (workspace / "AGENTS.md").write_text("implementer sentinel\n")
    seen = {}

    def implementer_turn(*_args, **_kwargs):
        assert json.loads(kanban_tools._handle_request_review({"summary": "candidate ready"}))["ok"]
        seen["reviewer"] = _finish_overlap_worker(_start_overlap_worker(path, implementer["task_id"], "review"))
        with connect_closing(path) as conn:
            seen["status"] = kb.get_task(conn, implementer["task_id"]).status
        seen["prompt"] = build_context_files_prompt(cwd=None)
        Path("late-implementer.txt").write_text("late bytes")
        seen["late"] = (workspace / "late-implementer.txt").read_text()
        return {"completed": True, "final_response": "handed off"}

    agent = SimpleNamespace(session_id="implementer", run_conversation=implementer_turn)
    session_kanban.run_worker_turns(agent, {"text": "work", "policy": {"kanban_json": json.dumps(implementer)}}, [])
    assert "error" not in seen["reviewer"], seen["reviewer"]
    assert seen["status"] == "done"  # the reviewer completed the card and its worker ended
    assert "implementer sentinel" in seen["prompt"]
    assert seen["late"] == "late bytes"
    assert not workspace.exists()  # the implementer, the last worker, removed it at its end


def _bindings(path, task_id):
    with connect_closing(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='worker_bound'",
                            (task_id,)).fetchone()[0]


def test_event_retention_keeps_a_live_workers_binding(bound_scratch_worker, monkeypatch, tmp_path):
    """t_3d35658c CWD-05: ``kanban gc`` with positive event retention pruned an old unfinished
    ``worker_bound`` of a still-live worker; the next ``gc`` found no live worker and removed its
    archived card's workspace. (The binding is aged artificially, not a 31-day run.) The binding
    is kept while its worker lives and pruned, with the workspace, once it is gone."""
    from hermes_cli import kanban_ops
    path, context, workspace = bound_scratch_worker
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    with connect_closing(path) as conn:
        assert kb.archive_task(conn, context["task_id"], signal_fn=lambda *_: None)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET created_at = created_at - 31 * 86400 WHERE task_id = ?",
                         (context["task_id"],))
    args = argparse.Namespace(event_retention_days=30, log_retention_days=0)
    for _ in range(2):
        assert kanban_ops._cmd_gc(args) == 0
        assert workspace.is_dir()
        assert _bindings(path, context["task_id"]) == 1
    _rebind(path, context, *_ended_process_identity())  # the worker ends without a result
    with connect_closing(path) as conn, kb.write_txn(conn):
        conn.execute("UPDATE task_events SET created_at = created_at - 31 * 86400 WHERE task_id = ?",
                     (context["task_id"],))
    assert kanban_ops._cmd_gc(args) == 0
    assert _bindings(path, context["task_id"]) == 0  # a dead worker's binding is pruned
    assert kanban_ops._cmd_gc(args) == 0
    assert not workspace.exists()


def test_only_a_recycled_pid_releases_a_binding(bound_scratch_worker, monkeypatch, tmp_path):
    """A live pid whose identity reads differently is another process: it does not hold the
    workspace. A live pid whose identity cannot be read proves nothing: it does."""
    from gateway import status
    from hermes_cli import kanban_db_dispatch
    path, context, workspace = bound_scratch_worker
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.chdir(tmp_path)
    _rebind(path, context, os.getpid(), kanban_db_dispatch._process_fingerprint(os.getpid()))
    with monkeypatch.context() as unreadable, connect_closing(path) as conn:
        unreadable.setattr(status, "get_process_start_time", lambda _pid: None)
        assert kb.complete_task(conn, context["task_id"], summary="done elsewhere")
    assert workspace.is_dir()  # unreadable identity: kept
    _rebind(path, context, os.getpid(), "0|0")  # this pid, someone else's birth
    with connect_closing(path) as conn:
        kanban_db_workspace._cleanup_workspace(conn, context["task_id"])
    assert not workspace.exists()


@pytest.mark.parametrize("route", ["complete", "gc"])
@pytest.mark.parametrize("reading", ["matching", "unknown", "recovered", "epoch-unknown", "foreign", "dead"])
def test_one_identity_reading_decides_whether_a_worker_is_live(bound_scratch_worker, monkeypatch, route, reading):
    """t_bc10a050 CWD-06: the live worker's identity was read twice; an unreadable first read
    and a matching second one released the binding, so completion removed the live cwd and
    positive-retention event GC pruned its binding (the next cleanup removed the cwd). Only the
    identity reads are faulted; the pid, database, completion, ``gc_events``, cleanup and prompt
    builder are real. A binding is released only by a reading that succeeds and differs, or a
    dead pid; otherwise both the binding and the workspace survive until the worker ends."""
    from agent.prompt_builder import build_context_files_prompt
    from gateway import drain_control, status
    from hermes_cli import kanban_db_dispatch
    path, context, workspace = bound_scratch_worker
    worker = _ended_process_identity() if reading == "dead" else (
        os.getpid(), kanban_db_dispatch._process_fingerprint(os.getpid()))
    _rebind(path, context, *worker)
    (workspace / "AGENTS.md").write_text("identity sentinel\n")
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    actual = status.get_process_start_time(os.getpid())
    reads = []

    def start_time(_pid):
        reads.append(_pid)
        if reading == "unknown" or (reading == "recovered" and len(reads) == 1):
            return None
        return actual + 100000 if reading == "foreign" else actual

    with connect_closing(path) as conn:
        if route == "gc":
            assert kb.archive_task(conn, context["task_id"], signal_fn=lambda *_: None)
            with kb.write_txn(conn):
                conn.execute("UPDATE task_events SET created_at = created_at - 31 * 86400 "
                             "WHERE task_id = ? AND kind = 'worker_bound'", (context["task_id"],))
        with monkeypatch.context() as fault:
            fault.setattr(status, "get_process_start_time", start_time)
            if reading == "epoch-unknown":
                fault.setattr(drain_control, "current_instantiation_epoch", lambda: "")
            if route == "complete":
                assert kb.complete_task(conn, context["task_id"], summary="completed by another actor")
            else:
                kb.gc_events(conn, older_than_seconds=30 * 86400)
        if route == "gc":
            kanban_db_workspace._cleanup_workspace(conn, context["task_id"])  # a healthy later sweep
    released = reading in ("foreign", "dead")
    assert bool(reads) is (reading != "dead")  # a dead pid needs no identity reading
    # The binding is the live worker's only protection: kept unless the worker is provably gone
    # (the gc route prunes a released one; completion never touches events).
    with connect_closing(path) as conn:
        kept = [json.loads(payload) for (payload,) in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'worker_bound'", (context["task_id"],))]
    assert sum((b["pid"], b.get("started_at")) == worker for b in kept) == (0 if released and route == "gc" else 1)
    if released:
        assert not workspace.exists()
        return
    try:
        assert "identity sentinel" in build_context_files_prompt(cwd=None)
        Path("late.txt").write_text("still running")
    finally:
        monkeypatch.chdir(Path(path).parent)
    session_kanban._record_worker_result(context, 0, "finished")
    assert not workspace.exists()
