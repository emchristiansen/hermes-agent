"""A worker's ``created_cards`` verify when its own admitted session created them.

Composed through the real producers and consumers: cards are created by the real ``hermes kanban
create`` CLI in a subprocess shaped like a native worker's unowned shell (session bound, no
``HERMES_KANBAN_TASK``), and handoffs go through the native ``kanban_complete`` tool handler against a
real board database. The only fixture standing in for a runtime component is the owner's
``owner_admitted`` event, which the gateway writes when it admits a dispatcher attempt to a session
(``gateway/session_kanban.py``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw

ASSIGNEE = "theseus-implementer"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", ASSIGNEE)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_SESSION_ID",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    (home / "config.yaml").write_text("kanban:\n  worker_board_management: true\n")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _start_run(title: str = "work") -> tuple[str, int]:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title=title, assignee=ASSIGNEE, workspace_kind="scratch")
        kbw.set_workspace_path(conn, tid, kbw.resolve_workspace(kb.get_task(conn, tid)))
        assert kb.claim_task(conn, tid) is not None
        return tid, int(kb.get_task(conn, tid).current_run_id)


def _admit(tid: str, run_id: int, session: str) -> None:
    """The owner's admission record: this session executes exactly this (task, run) attempt."""
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        kb._append_event(conn, tid, "owner_admitted",
                         {"session_id": session, "request_id": f"kanban:[null,{tid!r},{run_id}]"},
                         run_id=run_id)


def _worker(monkeypatch, tid: str, run_id: int, session: str | None) -> None:
    """The native worker's tool process: the dispatcher pins its task/run; its session is bound."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    if session is None:
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    else:
        monkeypatch.setenv("HERMES_SESSION_ID", session)


def _cli_create(title: str, *, session: str | None, created_by: str | None = None) -> str:
    """The real ``hermes kanban create`` from an unowned shell: no ``HERMES_KANBAN_*`` task pins."""
    argv = ["create", title, "--assignee", "theseus-adversarial-reviewer", "--json"]
    if created_by:
        argv += ["--created-by", created_by]
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
    env.pop("HERMES_SESSION_ID", None)
    if session is not None:
        env["HERMES_SESSION_ID"] = session
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *argv],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["id"]


def _complete(**args) -> dict:
    from tools import kanban_tools as kt
    return json.loads(kt._handle_complete({"summary": "done with evidence", **args}))


def _task(tid: str):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid)


def _events(tid: str, kind: str) -> list[dict]:
    with kbc.connect_closing() as conn:
        return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_cards_the_admitted_session_created_through_the_cli_verify(board, monkeypatch):
    session = "local-aaaa"
    tid, run_id = _start_run()
    _admit(tid, run_id, session)
    cards = [_cli_create(f"assessment {i}", session=session) for i in range(2)]
    created = _task(cards[0])
    assert created.created_by == f"{ASSIGNEE}/session:{session}"
    assert _events(cards[0], "created")[0]["creator_task_id"] is None  # the unowned shell's shape
    _worker(monkeypatch, tid, run_id, session)

    out = _complete(created_cards=cards)

    assert out.get("ok"), out
    assert _task(tid).status == "done"
    assert _events(tid, "completed")[-1]["verified_cards"] == cards
    assert not _events(tid, "completion_blocked_hallucination")


@pytest.mark.parametrize("case", ["other_session", "admitted_elsewhere", "no_grant", "forged_label",
                                  "other_task"])
def test_cards_that_are_not_this_attempts_stay_phantom(board, monkeypatch, case):
    session = "local-aaaa"
    tid, run_id = _start_run()
    _admit(tid, run_id, "local-other" if case == "admitted_elsewhere" else session)
    if case == "other_session":
        card = _cli_create("same profile, another session", session="local-bbbb")
    elif case == "forged_label":
        # Without a bound session the CLI honours --created-by: the label matches exactly, but the
        # created event carries no session actor, so it is not this session's card.
        card = _cli_create("forged", session=None, created_by=f"{ASSIGNEE}/session:{session}")
        assert _task(card).created_by == f"{ASSIGNEE}/session:{session}"
    elif case == "other_task":
        other, other_run = _start_run("another card's attempt")
        _admit(other, other_run, "local-cccc")
        card = _cli_create("created during the other attempt", session="local-cccc")
    else:
        card = _cli_create("this session's card", session=session)
    if case == "no_grant":
        (board / "config.yaml").write_text("kanban:\n  worker_board_management: false\n")
    _worker(monkeypatch, tid, run_id, session)

    out = _complete(created_cards=[card])

    assert "error" in out and card in out["error"], out
    assert _task(tid).status == "running"
    assert _events(tid, "completion_blocked_hallucination")[-1]["phantom_cards"] == [card]


def test_session_admitted_for_an_earlier_run_does_not_vouch_for_the_current_run(board, monkeypatch):
    tid, first_run = _start_run()
    _admit(tid, first_run, "local-first")
    card = _cli_create("made in the first attempt", session="local-first")
    with kbc.connect_closing() as conn:
        kb.reclaim_task(conn, tid, reason="next attempt")
        assert kb.claim_task(conn, tid) is not None
        second_run = int(kb.get_task(conn, tid).current_run_id)
    assert second_run != first_run
    _admit(tid, second_run, "local-second")
    # Neither the first session (not admitted for this run) nor the second (did not create it).
    for session in ("local-first", "local-second"):
        _worker(monkeypatch, tid, second_run, session)
        out = _complete(created_cards=[card])
        assert "error" in out and card in out["error"], out
    assert _task(tid).status == "running"


def test_wrong_run_cannot_complete_even_with_a_verified_card(board, monkeypatch):
    session = "local-aaaa"
    tid, run_id = _start_run()
    _admit(tid, run_id, session)
    card = _cli_create("assessment", session=session)
    _worker(monkeypatch, tid, run_id + 1000, session)

    out = _complete(created_cards=[card])

    assert "error" in out, out
    assert _task(tid).status == "running" and _task(tid).current_run_id == run_id
