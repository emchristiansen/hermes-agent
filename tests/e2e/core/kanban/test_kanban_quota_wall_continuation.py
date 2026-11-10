"""An owner that yielded for its author's result consumes that result after a long quota wall,
without relaunching the author (t_d4ba87d4).

Real processes end to end: every tick is a real ``hermes kanban dispatch`` process, every worker a
real dispatcher-spawned worker, talking to the recording fake provider (the only fake: it stands in
for the vendor HTTP API and plays the owner's model). The owner's rule, given in its card body: read
the task (``kanban_show``); if the context names the author already launched for it, consume that
author's returns and complete; otherwise launch the author (a ``LAUNCH-AUTHOR`` comment, the effect
that must never repeat) and yield (``kanban_block``, needs_input) naming it.

Flow (``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS=0``):

1. attempt 1 launches the author and yields (``blocked``) naming it;
2. the author returns RESULT-R1, resends it after a lost acknowledgement (a duplicate), and its
   callback releases the card;
3. the provider answers HTTP 429 to the next ``WALLS`` attempts (more than the context's attempt
   window); a later RESULT-R2 arrives in the middle;
4. capacity returns: the next attempt must read both returns and complete, launching nothing.

Verdicts come from ``kanban.db`` rows and the requests the fake provider recorded.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.kanban._helpers import Board
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="worker liveness reads /proc"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs this board's reparented workers
]

OWNER_MARK = "E2E_OWNER_RULE_c41d"
_TASK_RE = re.compile(r"work kanban task (t_[0-9a-f]+)")
PENDING = "PENDING-CALLBACK author cmwAUTHOR1: consume its RESULT comments, never relaunch it"
WALLS = 11  # > _CTX_MAX_PRIOR_ATTEMPTS: the yield falls outside a newest-10 attempt window
BODY = (f"{OWNER_MARK}: call kanban_show first. If your prior attempts name the author you already "
        "launched, complete with that author's RESULT comments; otherwise launch the author (comment) and "
        "yield (kanban_block) naming it.")


def _section(context: str, heading: str) -> str:
    """One ``## heading`` section of the worker context, up to the next section."""
    start = context.find(f"## {heading}")
    if start < 0:
        return ""
    end = context.find("\n## ", start + 3)
    return context[start:] if end < 0 else context[start:end]


class OwnerModel:
    """The owner's scripted model; ``capacity`` False answers every owner request with HTTP 429."""

    def __init__(self) -> None:
        self.capacity = True
        self.tools: list[str] = []
        self.shown: list[str] = []  # each kanban_show result the model read
        self._lock = threading.Lock()

    def __call__(self, rec: dict[str, Any]) -> Any:
        msgs = rec["body"].get("messages", [])
        if OWNER_MARK not in json.dumps(msgs):
            return Text("ok")  # auxiliary requests (titles, judges) are not the owner
        with self._lock:
            if not self.capacity:
                return Error(status=429, message="The usage limit has been reached", retry_after=0)
            return self._next(msgs)

    def _call(self, name: str, args: dict) -> ToolCall:
        self.tools.append(name)
        return ToolCall(name, args)

    def _next(self, msgs: list) -> Any:
        """Decide from THIS attempt's conversation: the tool this model last called, and its result."""
        called = [c["function"]["name"] for m in msgs if m.get("role") == "assistant"
                  for c in (m.get("tool_calls") or [])]
        previous = called[-1] if called else None
        last = msgs[-1] if msgs else {}
        task = _TASK_RE.search(json.dumps(msgs)).group(1)
        if previous is None or last.get("role") != "tool":
            return self._call("kanban_show", {"task_id": task})
        content = str(last.get("content") or "")
        if previous == "kanban_show":
            self.shown.append(content)
            if PENDING in _section(content, "Prior attempts on this task"):
                thread = _section(content, "Comment thread")
                results = sorted({r for r in ("RESULT-R1", "RESULT-R2") if r in thread})
                return self._call("kanban_complete", {"task_id": task,
                                                      "summary": f"consumed {' '.join(results)} from cmwAUTHOR1"})
            return self._call("kanban_comment", {"task_id": task, "body": "LAUNCH-AUTHOR cmwAUTHOR1"})
        if previous == "kanban_comment":
            return self._call("kanban_block", {"task_id": task, "reason": PENDING, "kind": "needs_input"})
        return Text("done")


def _attempt(board: Board, tid: str) -> dict:
    """One real tick; wait for the spawned worker of ``tid`` to exit."""
    res = board.dispatch()
    if tid in [s["task_id"] for s in res.get("spawned", [])]:
        pid = board.events(tid, "spawned")[-1]["payload"]["pid"]
        board.wait_worker_exit(tid, int(pid))
    return board.task(tid)


def _comment(board: Board, tid: str, body: str) -> None:
    board.cli("comment", tid, body, "--author", "cmwAUTHOR1")


@pytest.fixture(scope="module")
def flow(tmp_path_factory: pytest.TempPathFactory):
    model = OwnerModel()
    root = tmp_path_factory.mktemp("kanban-quota-wall")
    with FakeLLMServer(model) as srv:
        board = Board(root, srv.base_url)
        tid = board.create("owner awaiting its author", "--body", BODY)
        try:
            assert _attempt(board, tid)["status"] == "blocked", f"tools={model.tools}\n{board.diag(tid)}"
            _comment(board, tid, "RESULT-R1 sealed at /scratch/r1")
            _comment(board, tid, "RESULT-R1 sealed at /scratch/r1")  # resent: its ACK was lost
            board.cli("unblock", tid)  # the author's callback
            model.capacity = False
            walls = 0
            for _ in range(WALLS * 3):
                before = len(board.runs(tid))
                _attempt(board, tid)
                walls += len(board.runs(tid)) > before
                if walls == 5:
                    _comment(board, tid, "RESULT-R2 sealed at /scratch/r2")
                if walls == WALLS:
                    break
            model.capacity = True  # the next tick reaps the last wall and respawns on capacity
            for _ in range(4):
                if _attempt(board, tid)["status"] == "done":
                    break
        finally:
            board.kill_workers()
        yield board, tid, model


def test_the_owner_consumes_its_authors_returns_after_the_wall_and_launches_nothing(flow) -> None:
    board, tid, model = flow
    diag = f"tools={model.tools}\n{board.diag(tid)}"
    outcomes = [r["outcome"] for r in board.runs(tid)]
    assert outcomes == ["blocked", *["rate_limited"] * WALLS, "completed"], diag
    assert board.task(tid)["consecutive_failures"] == 0, diag
    launches = [c for c in board._q("SELECT body FROM task_comments WHERE task_id=?", (tid,))
                if "LAUNCH-AUTHOR" in c["body"]]
    assert len(launches) == 1 and model.tools.count("kanban_comment") == 1, diag  # the author ran once
    final = board.runs(tid)[-1]
    assert "RESULT-R1" in final["summary"] and "RESULT-R2" in final["summary"], diag
    # The consuming attempt read the yield naming its author and every return, the duplicate included.
    assert PENDING in _section(model.shown[-1], "Prior attempts on this task"), diag
    assert _section(model.shown[-1], "Comment thread").count("RESULT-R1") == 2, diag
