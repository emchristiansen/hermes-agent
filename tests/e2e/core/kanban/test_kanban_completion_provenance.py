"""A worker's ``created_cards`` survive the real dispatcher -> owner -> worker -> CLI -> completion path.

A swarm-granted worker (``kanban.worker_board_management``) creates a card through the real
``hermes kanban create`` CLI from a shell that holds no ``HERMES_KANBAN_*`` task pins (the shape of
a native executor's deliberately unowned shell), then names that card in ``kanban_complete``. The
owner daemon admits the attempt to one session and the CLI stamps that session on the card, so the
completion must verify it rather than reject it as invented; a card another session created must
still be rejected. Verdicts come from kanban.db rows only.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

from tests.e2e.core.kanban._helpers import Board
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc worker liveness"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs this board's reparented workers
]

GRANT = "kanban:\n  dispatch_in_gateway: false\n  worker_board_management: true\n"
_TASK_RE = re.compile(r"work kanban task (t_[0-9a-f]+)")
_CARD_RE = re.compile(r'\\?"id\\?":\s*\\?"(t_[0-9a-f]+)')  # the tool output JSON-escapes it


def _task_id(rec: dict) -> str:
    for m in rec["body"]["messages"]:
        hit = _TASK_RE.search(str(m.get("content")))
        if hit:
            return hit.group(1)
    raise AssertionError("worker prompt did not name its task")


def _tool_results(rec: dict) -> list[str]:
    return [str(m.get("content")) for m in rec["body"]["messages"] if m.get("role") == "tool"]


@pytest.mark.parametrize("creator", ["own_session", "other_session"])
def test_cli_created_card_verifies_only_for_the_admitted_session(tmp_path, creator: str) -> None:
    board_ref: dict[str, Board] = {}
    state: dict[str, str] = {}

    def responder(rec: dict):
        results = _tool_results(rec)
        tid = _task_id(rec)
        if not results:
            # Unowned shell: strip every task pin. ``other_session`` also replaces the session.
            session = ' HERMES_SESSION_ID=local-someone-else' if creator == "other_session" else ""
            command = (f"env -u HERMES_KANBAN_TASK -u HERMES_KANBAN_RUN_ID -u HERMES_KANBAN_CLAIM_LOCK"
                       f"{session} \"$HERMES_BIN\" kanban create 'assessment for {tid}' "
                       f"--assignee default --json")
            return ToolCall("terminal", {"command": command})
        if "card" not in state:
            hit = _CARD_RE.search(results[-1])
            assert hit, f"CLI create output carried no card id: {results[-1][:400]}"
            state["card"] = hit.group(1)
            return ToolCall("kanban_complete", {"summary": "assessment created",
                                                "created_cards": [state["card"]]})
        state["completion"] = results[-1]
        return Text("done")

    with FakeLLMServer(responder) as srv:
        board = board_ref["b"] = Board(tmp_path, srv.base_url, extra_config=GRANT)
        try:
            tid = board.create("provenance card")
            board.dispatch()
            pid = board.task(tid)["worker_pid"]
            assert pid, board.diag(tid)
            board.wait_worker_exit(tid, int(pid))
            card = state.get("card")
            assert card, board.diag(tid)
            created = board.task(card)
            admitted = board.events(tid, "owner_admitted")
            assert admitted, board.diag(tid)
            session = admitted[-1]["payload"]["session_id"]
            if creator == "own_session":
                assert created["created_by"] == f"default/session:{session}", (created, admitted)
                assert board.task(tid)["status"] == "done", board.diag(tid)
                completed = board.events(tid, "completed")[-1]["payload"]
                assert completed["verified_cards"] == [card], completed
                assert not board.events(tid, "completion_blocked_hallucination")
            else:
                assert created["created_by"] == "default/session:local-someone-else", created
                blocked = board.events(tid, "completion_blocked_hallucination")
                assert blocked and blocked[-1]["payload"]["phantom_cards"] == [card], board.diag(tid)
                assert card in state.get("completion", ""), state
        finally:
            board.kill_workers()
