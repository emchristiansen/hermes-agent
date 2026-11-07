"""Editing an existing task's skill binding (``hermes kanban set-skills``).

A card created with a skill its assignee lacks fails every attempt before the conversation starts
(``kanban_skills_unavailable``); the binding must be changeable on the same card, between attempts,
with nothing else about the card rewritten.
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_skills import set_task_skills


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _skills_column(conn, tid):
    return conn.execute("SELECT skills FROM tasks WHERE id = ?", (tid,)).fetchone()["skills"]


def _dump(conn):
    return list(conn.iterdump())


def _skills_events(conn, tid):
    return [json.loads(r["payload"]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'skills_set' ORDER BY id", (tid,))]


def _ended_blocked_card(conn, **create):
    """The incident shape: a blocked card with two ended attempts and no claim left."""
    tid = kb.create_task(conn, title="review", assignee="reviewer", skills=["github-code-review"], **create)
    for _ in range(2):
        assert kb.claim_task(conn, tid) is not None
        assert kb.reclaim_task(conn, tid, reason="launch failed")
    assert kb.block_task(conn, tid, reason="breaker")
    task = kb.get_task(conn, tid)
    assert task.status == "blocked" and task.claim_lock is None and task.current_run_id is None
    return tid


# --- API --------------------------------------------------------------------

def test_replace_records_an_auditable_event(conn):
    tid = _ended_blocked_card(conn)
    ok, refusal = set_task_skills(conn, tid, ["translation", "translation", " humanizer "],
                                  actor="operator", reason="skill not installed")
    assert (ok, refusal) == (True, None)
    assert kb.get_task(conn, tid).skills == ["translation", "humanizer"]
    assert _skills_events(conn, tid) == [{
        "skills": ["translation", "humanizer"], "from": ["github-code-review"],
        "actor": "operator", "reason": "skill not installed",
    }]


def test_clear_is_null_and_none_is_an_explicit_empty_list(conn):
    tid = _ended_blocked_card(conn)
    assert set_task_skills(conn, tid, None) == (True, None)
    assert _skills_column(conn, tid) is None and kb.get_task(conn, tid).skills is None
    assert set_task_skills(conn, tid, []) == (True, None)
    assert _skills_column(conn, tid) == "[]" and kb.get_task(conn, tid).skills == []
    assert [(e["from"], e["skills"]) for e in _skills_events(conn, tid)] == [
        (["github-code-review"], None), (None, []),
    ]


def test_unchanged_binding_writes_nothing(conn):
    tid = _ended_blocked_card(conn)
    before = _dump(conn)
    assert set_task_skills(conn, tid, ["github-code-review"]) == (True, None)
    assert _dump(conn) == before


@pytest.mark.parametrize("skills, match", [
    (["github-code-review,translation"], "cannot contain comma"),
    (["web"], "toolset name"),
    (["", "  "], "no skill name given"),
])
def test_invalid_names_are_refused_without_writes(conn, skills, match):
    tid = _ended_blocked_card(conn)
    before = _dump(conn)
    with pytest.raises(ValueError, match=match):
        set_task_skills(conn, tid, skills)
    assert _dump(conn) == before


def test_unknown_task(conn):
    assert set_task_skills(conn, "t_missing", None) == (False, "no such task: t_missing")


@pytest.mark.parametrize("live", ["claimed", "claim_lock", "current_run_id", "worker_pid", "status", "open_run"])
def test_refused_while_an_attempt_holds_the_card(conn, live):
    tid = kb.create_task(conn, title="t", assignee="reviewer", skills=["github-code-review"])
    claimed = kb.claim_task(conn, tid)
    if live != "claimed":
        # Leave exactly one trace of the live attempt (the rest as if released).
        keep = {"claim_lock": "claim_lock", "current_run_id": "current_run_id", "worker_pid": "worker_pid",
                "status": "status"}.get(live)
        cols = {"claim_lock": None, "current_run_id": None, "worker_pid": None, "status": "'ready'"}
        if keep == "worker_pid":
            cols.pop("worker_pid")
            conn.execute("UPDATE tasks SET worker_pid = 4242 WHERE id = ?", (tid,))
        elif keep:
            cols.pop(keep)
        sets = ", ".join(f"{c} = {'NULL' if v is None else v}" for c, v in cols.items())
        conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (tid,))
        conn.commit()
    before = _dump(conn)
    ok, refusal = set_task_skills(conn, tid, None)
    assert not ok
    assert ("open run" if live == "open_run" else "live claim") in refusal
    assert _dump(conn) == before
    assert claimed.current_run_id is not None


def test_archived_task_is_refused(conn):
    tid = kb.create_task(conn, title="t", assignee="reviewer", skills=["github-code-review"])
    assert kb.archive_task(conn, tid)
    before = _dump(conn)
    assert set_task_skills(conn, tid, None) == (False, f"task {tid} is archived")
    assert _dump(conn) == before


def test_expected_is_compare_and_set(conn):
    tid = _ended_blocked_card(conn)
    before = _dump(conn)
    for wrong in (None, [], ["translation"]):
        ok, refusal = set_task_skills(conn, tid, None, expected=wrong)
        assert not ok and "not the expected" in refusal
    assert _dump(conn) == before
    assert set_task_skills(conn, tid, None, expected=["github-code-review"]) == (True, None)
    assert _skills_events(conn, tid)[-1]["expected"] == ["github-code-review"]
    assert set_task_skills(conn, tid, [], expected=None) == (True, None)
    assert set_task_skills(conn, tid, ["translation"], expected=[]) == (True, None)


def test_workers_cannot_rebind_skills(conn, monkeypatch):
    tid = _ended_blocked_card(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    before = _dump(conn)
    ok, refusal = set_task_skills(conn, tid, None)
    assert not ok and "orchestrator-only" in refusal
    assert _dump(conn) == before


def test_every_other_field_and_row_is_preserved(conn, tmp_path):
    parent = kb.create_task(conn, title="parent", assignee="implementer")
    tid = _ended_blocked_card(
        conn, body="the body", priority=8, tenant="acme", max_runtime_seconds=1800,
        model_override="m", provider_override="custom", reasoning_effort="high", goal_mode=True,
        goal_max_turns=5, session_id="sess-1", completion_contract="local-only", created_by="implementer",
    )
    kb.link_tasks(conn, parent, tid)
    kb.add_comment(conn, tid, "implementer", "evidence: see run 2")
    blob = tmp_path / "evidence.txt"
    blob.write_text("x")
    kb.add_attachment(conn, tid, filename="evidence.txt", stored_path=str(blob), uploaded_by="implementer")
    conn.execute("UPDATE tasks SET consecutive_failures = 2, last_failure_error = 'kanban_skills_unavailable' "
                 "WHERE id = ?", (tid,))
    conn.commit()

    def snapshot():
        row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
        tables = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                  for t in ("task_runs", "task_comments", "task_links", "task_attachments", "kanban_notify_subs")}
        events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        others = [tuple(r) for r in conn.execute("SELECT * FROM tasks WHERE id != ? ORDER BY id", (tid,))]
        return row, tables, events, others

    row0, tables0, events0, others0 = snapshot()
    assert set_task_skills(conn, tid, None, actor="operator") == (True, None)
    row1, tables1, events1, others1 = snapshot()
    assert {k for k in row0 if row0[k] != row1[k]} == {"skills"}
    assert row1["skills"] is None and row1["status"] == "blocked"
    assert tables1 == tables0 and others1 == others0
    assert events1[:len(events0)] == events0
    assert [e[3] for e in events1[len(events0):]] == ["skills_set"]


# --- the next attempt reads the new binding ----------------------------------

def test_cleared_binding_reaches_the_next_attempts_owner_and_dispatcher(tmp_path, monkeypatch):
    """The skill is read from the claimed row when the owner admits each attempt
    (``build_kanban_policy``), with nothing cached across attempts: after the clear, the
    next attempt's frozen context carries no skill and its turn passes the skills gate that
    raised ``kanban_skills_unavailable`` before."""
    from gateway import session_kanban
    from gateway.session_contract import Principal
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env",
                        lambda name: str(tmp_path) if name == "default" else str(tmp_path / name))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("owned")
    actor = Principal("owner", str(tmp_path), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    config = {"model": {"default": "m", "provider": "custom"}, "platform_toolsets": {"cli": ["file"]}}

    class Reached(Exception):
        pass

    class Agent:
        session_id = "s"

        def run_conversation(self, *_args, **_kwargs):
            raise Reached

    def attempt(conn, tid):
        task = kb.claim_task(conn, tid)
        params = dict(board="owned", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
        policy, _ = session_kanban.build_kanban_policy(connection, params, config)
        return task, json.loads(policy.kanban_json)

    with closing(kbc.connect(board="owned")) as conn:
        tid = kb.create_task(conn, title="review", assignee="default", workspace_kind="dir",
                             workspace_path=str(workspace), skills=["not-installed-skill"])
        first, context = attempt(conn, tid)
        assert context["skills"] == ["not-installed-skill"]
        with pytest.raises(ValueError, match="kanban_skills_unavailable"):
            session_kanban._run_task_turns(Agent(), {"text": "go"}, [], context)
        assert set_task_skills(conn, tid, None)[0] is False  # the attempt still holds the card
        assert kb.reclaim_task(conn, tid, reason="skills unavailable")

        assert set_task_skills(conn, tid, None, expected=["not-installed-skill"]) == (True, None)
        second, context = attempt(conn, tid)
        assert second.current_run_id != first.current_run_id
        assert context["skills"] == []
        with pytest.raises(Reached):  # past the skills gate, into the conversation
            session_kanban._run_task_turns(Agent(), {"text": "go"}, [], context)
        assert kb.reclaim_task(conn, tid, reason="next")

        spawned = []
        monkeypatch.setattr(kbd, "_set_worker_pid", lambda *a, **k: None)
        kbd.dispatch_once(conn, board="owned", spawn_fn=lambda task, ws: spawned.append(task) or None)
        assert [t.id for t in spawned] == [tid] and not spawned[0].skills


# --- CLI --------------------------------------------------------------------

def _cli(*argv):
    root = argparse.ArgumentParser(prog="hermes")
    kb_cli.build_parser(root.add_subparsers(dest="command"))
    return kb_cli.kanban_command(root.parse_args(["kanban", *argv]))


def test_cli_replace_clear_and_none(kanban_home, capsys):
    with kbc.connect() as c:
        tid = _ended_blocked_card(c)
    assert _cli("set-skills", tid, "translation", "humanizer", "--reason", "swap") == 0
    assert "translation, humanizer" in capsys.readouterr().out
    assert _cli("set-skills", tid, "--none") == 0
    assert "(none, explicit)" in capsys.readouterr().out
    assert _cli("set-skills", tid, "--clear", "--expected", "[]", "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"task_id": tid, "ok": True, "skills": None, "error": None}
    with kbc.connect() as c:
        events = _skills_events(c, tid)
    assert [e["skills"] for e in events] == [["translation", "humanizer"], [], None]
    assert events[0]["reason"] == "swap" and events[0]["actor"]


@pytest.mark.parametrize("argv, rc, needle", [
    (("--clear", "--none"), 2, "exactly one"),
    ((), 2, "exactly one"),
    (("a", "--clear"), 2, "exactly one"),
    (("a,b",), 2, "cannot contain comma"),
    (("--clear", "--expected", "not json"), 2, "--expected"),
    (("--clear", "--expected", '"github-code-review"'), 2, "--expected must be"),
    (("--clear", "--expected", "null"), 1, "not the expected"),
])
def test_cli_refusals_write_nothing(kanban_home, capsys, argv, rc, needle):
    with kbc.connect() as c:
        tid = _ended_blocked_card(c)
        before = _dump(c)
    assert _cli("set-skills", tid, *argv) == rc
    assert needle in capsys.readouterr().err
    with kbc.connect() as c:
        assert _dump(c) == before


def test_cli_unknown_task_and_live_claim(kanban_home, capsys):
    assert _cli("set-skills", "t_missing", "--clear") == 1
    assert "no such task" in capsys.readouterr().err
    with kbc.connect() as c:
        tid = kb.create_task(c, title="t", assignee="reviewer", skills=["github-code-review"])
        kb.claim_task(c, tid)
    assert _cli("set-skills", tid, "--clear", "--json") == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "live claim" in out["error"]
    assert out["skills"] == ["github-code-review"]  # the binding still in force
    with kbc.connect() as c:
        assert kb.get_task(c, tid).skills == ["github-code-review"]


# --- who may rebind ------------------------------------------------------------

@pytest.fixture
def implementer_board(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "theseus-implementer")
    with kbc.connect() as c:
        return _ended_blocked_card(c)


def test_plain_operator_cli_can_rebind(implementer_board, capsys):
    """No dispatcher-owned task in the environment: the operator's CLI, as for `unblock`."""
    assert _cli("set-skills", implementer_board, "--clear") == 0
    with kbc.connect() as c:
        assert kb.get_task(c, implementer_board).skills is None
        assert _skills_events(c, implementer_board)[0]["actor"] == "theseus-implementer"


def test_granted_worker_session_can_rebind_under_its_own_identity(implementer_board, kanban_home, monkeypatch):
    """An orchestrating profile's own worker session with ``kanban.worker_board_management``
    (the theseus-implementer shape) rebinds another card, attributed to its bound session."""
    from gateway.session_context import scoped_current_session_id
    (kanban_home / "config.yaml").write_text("kanban:\n  worker_board_management: true\n")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_orchestrator_own")
    with scoped_current_session_id("implementer-session"):
        assert "default: none" in kb_cli.run_slash(f"set-skills {implementer_board} --clear")
    with kbc.connect() as c:
        assert kb.get_task(c, implementer_board).skills is None
        actor = "theseus-implementer/session:implementer-session"
        assert _skills_events(c, implementer_board)[0]["actor"] == actor


def test_ungranted_worker_session_is_refused(implementer_board, kanban_home, monkeypatch, capsys):
    from gateway.session_context import scoped_current_session_id
    (kanban_home / "config.yaml").write_text("kanban:\n  worker_board_management: false\n")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker_own")
    with scoped_current_session_id("worker-session"):
        assert _cli("set-skills", implementer_board, "--clear") == 1
    assert "orchestrator-only" in capsys.readouterr().err
    with kbc.connect() as c:
        assert kb.get_task(c, implementer_board).skills == ["github-code-review"]
