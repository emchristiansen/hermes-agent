"""Closing the app before onboarding finishes does not count: the next launch's ``onboarding.start``
(``fresh_launch``) ends the unfinished run as ``replaced`` and starts over at the name beat in a new
chat, while the old chat stays in the history with every message. A second window or a reconnect in
the same launch (no flag) keeps the run."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from agent import onboarding_record
from agent.onboarding_flow import PENDING, Answered, Facts, Shown, is_active
from agent.onboarding_text import CHAT_TITLE
from hermes_cli.config import read_user_config_raw
from tui_gateway import server, server_requests

FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=())


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: FACTS)
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")  # a model to run on
    return home


@pytest.fixture
def client(root, monkeypatch):
    """session.resume and prompt.submit as a recorder; a submit stores its user turn, as the real turn does."""
    calls = []

    def resume(rid, params):
        calls.append(("resume", params))
        return {"result": {"session_id": f"rt-{params['session_id']}"}}

    def submit(rid, params):
        calls.append(("submit", params))
        server._get_db().append_message(params["session_id"].removeprefix("rt-"), "user", params["text"])
        return {"result": {"status": "streaming"}}

    monkeypatch.setitem(server._methods, "session.resume", resume)
    monkeypatch.setitem(server._methods, "prompt.submit", submit)
    from hermes_cli.process_identity import serving_local_desktop
    with serving_local_desktop(True):  # the local Desktop's requests: only it starts a first run
        yield calls


@pytest.fixture
def ended(monkeypatch) -> list[dict]:
    told: list[dict] = []
    monkeypatch.setattr(onboarding_record, "_ended_listeners", [told.append])
    return told


def _start(params: dict | None = None) -> dict:
    response = server.handle_request({"jsonrpc": "2.0", "id": 7, "method": "onboarding.start", "params": params or {}})
    assert "result" in response, response
    return response["result"]


def _run_flag(root: Path) -> bool:
    return read_user_config_raw(root / "config.yaml")["onboarding"]["run"]


def _quit_mid_run(db, old: str) -> list[dict]:
    """The user got past the name card and talked in the chat, then quit the app; returns its messages."""
    onboarding_record.advance(db, old, Shown("name"))
    onboarding_record.advance(db, old, Answered("name", {"name": "Sid"}))
    db.append_message(old, "assistant", "Hi! What should I call you?")
    db.append_message(old, "user", "Call me Sid.")
    db.append_message(old, "assistant", "Nice to meet you, Sid. Want to pick a theme next?")
    messages = db.get_messages(old)
    assert len(messages) == 4  # the hidden opening the start stored, then the dialogue
    return messages


def test_a_new_launch_keeps_the_unfinished_chat_and_starts_over_at_the_name_beat(root, client, ended):
    first = _start({"fresh_launch": True})
    old = first["stored_session_id"]
    db = server._get_db()
    old_title = db.get_session(old)["title"]
    messages = _quit_mid_run(db, old)
    assert onboarding_record.load(db, old)["beats"]["name"] != PENDING

    again = _start({"fresh_launch": True})
    new = again["stored_session_id"]

    assert new != old and again["resumed"] is False and again["run_id"] != first["run_id"]
    # The old chat and every message in it stay; its run ended as replaced, and every window was told.
    assert db.get_session(old) is not None and db.get_messages(old) == messages
    old_record = onboarding_record.load(db, old)
    assert old_record["ended_by"] == "replaced" and not is_active(old_record)
    assert [(e["run_id"], e["stored_session_id"], e["outcome"]) for e in ended] == [
        (first["run_id"], old, "replaced")]
    record = onboarding_record.load(db, new)
    assert onboarding_record.open_session_ids(db) == [new] and record["beats"]["name"] == PENDING
    # The fresh chat plays the opening again; the kept chat keeps its title, the new one is next in the lineage.
    assert [kind for kind, _ in client] == ["resume", "submit", "resume", "submit"]
    assert old_title == CHAT_TITLE and db.get_session(old)["title"] == old_title
    assert db.get_session(new)["title"] == f"{CHAT_TITLE} #2"
    assert _run_flag(root) is True
    # The old run is never resumed: a second window or a reconnect gets the new one.
    resumed = _start()
    assert resumed["stored_session_id"] == new and resumed["resumed"] is True
    assert server.handle_request({"jsonrpc": "2.0", "id": 8, "method": "onboarding.state",
                                  "params": {}})["result"]["session_id"] == new


def test_a_second_window_or_a_reconnect_keeps_the_run(root, client, ended):
    first = _start({"fresh_launch": True})

    again = _start()

    assert again["stored_session_id"] == first["stored_session_id"] and again["resumed"] is True
    assert again["run_id"] == first["run_id"] and ended == []


def test_a_new_launch_closes_the_unfinished_chat_still_live_here_and_withdraws_its_card(root, client, ended,
                                                                                       monkeypatch):
    old = _start({"fresh_launch": True})["stored_session_id"]
    db = server._get_db()
    messages = _quit_mid_run(db, old)
    closed, events = [], []
    monkeypatch.setitem(server._sessions, "rt-live", {"session_key": old})
    monkeypatch.setattr(server, "_close_session_by_id",
                        lambda sid, **kw: closed.append((sid, kw, server._get_db().get_session(old) is not None)))
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", lambda frame: None)
    monkeypatch.setattr(server_requests, "_emit", lambda kind, sid, payload: events.append((kind, sid, payload)))
    reply = []
    waiter = threading.Thread(target=lambda: reply.append(server._setup_beat_request(
        "rt-live", {"beat": "theme", "card": {"picker": "accent"}})), daemon=True)
    waiter.start()
    from tests.waiting import wait_until
    wait_until(lambda: server_requests.pending_kind("rt-live") == "setup_beat")

    _start({"fresh_launch": True})
    waiter.join(timeout=5)

    assert closed == [("rt-live", {"end_reason": "onboarding_restart"}, True)]
    assert reply == [None] and [(kind, sid) for kind, sid, _ in events] == [("request.cancel", "rt-live")]
    assert db.get_messages(old) == messages
    assert onboarding_record.load(db, old)["ended_by"] == "replaced"
    assert [e["outcome"] for e in ended] == ["replaced"]


def test_a_new_launch_keeps_a_chat_a_turn_still_writes_and_ends_its_run_as_replaced(root, client, ended):
    old = _start({"fresh_launch": True})["stored_session_id"]
    db = server._get_db()
    messages = _quit_mid_run(db, old)
    holder = f"pid={os.getpid()}:turn=onboarding-test"
    assert db.try_acquire_session_turn_lease(old, holder, ttl_seconds=300.0)
    try:
        new = _start({"fresh_launch": True})["stored_session_id"]
    finally:
        db.release_session_turn_lease(old, holder)

    assert new != old and db.get_messages(old) == messages
    assert onboarding_record.load(db, old)["ended_by"] == "replaced"
    assert onboarding_record.open_session_ids(db) == [new]
    assert [(e["stored_session_id"], e["outcome"]) for e in ended] == [(old, "replaced")]


def test_deleting_a_kept_chat_or_the_new_run_by_hand_still_deletes(root, client, ended):
    old = _start({"fresh_launch": True})["stored_session_id"]
    db = server._get_db()
    _quit_mid_run(db, old)
    new = _start({"fresh_launch": True})["stored_session_id"]

    # The kept chat is an ordinary chat now: the user can still delete it; its run already ended.
    assert server.handle_request({"jsonrpc": "2.0", "id": 9, "method": "session.delete",
                                  "params": {"session_id": old}})["result"] == {"deleted": old}
    assert db.get_session(old) is None and db.get_messages(old) == []
    assert [e["outcome"] for e in ended] == ["replaced"]
    # Deleting the chat of the run in progress ends it (deleted) and tells every window, as before.
    assert server.handle_request({"jsonrpc": "2.0", "id": 10, "method": "session.delete",
                                  "params": {"session_id": new}})["result"] == {"deleted": new}
    assert db.get_session(new) is None
    assert [(e["stored_session_id"], e["outcome"]) for e in ended] == [(old, "replaced"), (new, "deleted")]
    assert onboarding_record.active_session_id(db) is None and _run_flag(root) is False
