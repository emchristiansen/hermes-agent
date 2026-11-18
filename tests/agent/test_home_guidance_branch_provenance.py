"""A branch keeps the provenance its rows carry (R8-BRANCH-01, R8-NIT-03).

The CLI and gateway ``/branch`` commands copy the parent's transcript into a child session. A failed-turn notice that
stands for a real reply (``display_metadata.stands_for_turn_row``), a delivery mirror (``delivery_mirror``) and the
notice's own type (``display_kind``) are what home guidance reads to tell a conversation under way from one with no
reply yet, so the copy keeps them: a child whose prompt survives carries the parent's decision, one whose prompt is lost
says so explicitly, and a child of a history with no reply takes up the configured file on its first actual turn.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent import home_guidance as hg
from tests.agent.test_home_guidance import CORE_B, _agent, _expected_block, _turn
from tests.agent.test_home_guidance_boundaries import MIRRORS, _mirror, _mirror_store, _next_turn, _unavailable_only
from tests.agent.test_home_guidance_failed_provenance import _compact_live_tail, _gateway_close, _produce

pytest_plugins = ["tests.agent.test_home_guidance"]


@pytest.fixture(autouse=True)
def _own_prompt_toolkit_session():
    """The CLI ``/branch`` prints through prompt_toolkit, whose default session keeps the first output it
    builds (this test's captured stdout) for the whole process, so a later test's capsys would see
    nothing: print through a session of the test's own."""
    from prompt_toolkit.application import create_app_session
    with create_app_session():
        yield


def _parent(store, carrier):
    """The parent session id: the gateway's own session for its chat, or a CLI session."""
    if carrier == "gateway":
        from tests.gateway.test_branch_routing_columns import _make_source
        return store.get_or_create_session(_make_source()).session_id
    return "cli-parent"


def _branch(store, carrier, parent):
    """Run the real ``/branch`` command of *carrier* over the parent's live transcript; returns the child id."""
    db = store._db
    if carrier == "gateway":
        from tests.gateway.test_branch_routing_columns import _make_branch_runner, _make_event, _make_source
        asyncio.run(_make_branch_runner(store)._handle_branch_command(_make_event("/branch --here probe")))
        child = store.get_or_create_session(_make_source()).session_id
    else:
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        cli = SimpleNamespace(_agent_running=False, _session_db=db, session_id=parent, agent=None,
                              conversation_history=store.load_transcript(parent), model="anthropic/claude-sonnet-4.6",
                              max_turns=20, reasoning_config={}, _new_session_id=lambda *args, **kwargs: "cli-child")
        CLICommandsMixin._handle_branch_command(cli, "/branch probe")
        child = cli.session_id
    assert child != parent
    return child


@pytest.mark.parametrize("lose_prompt", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("shape", ["verification-notice", "late-partial"])
@pytest.mark.parametrize("carrier", ["cli", "gateway"])
def test_a_branch_of_a_reply_behind_a_notice_keeps_the_conversation_under_way(home, no_network, monkeypatch, carrier,
                                                                             shape, compacted, lose_prompt):
    """Real producers write a reply with a failed-turn notice after or before it; the parent is optionally compacted;
    the real ``/branch`` copies it with the parent's prompt. The child carries the notice row that stands for the
    reply. With its prompt it carries A even after the file changes to B; without it, an explicit notice, never B.
    The parent's prompt is unchanged."""
    from agent.message_metadata import STANDS_FOR_TURN_ROW
    store = _mirror_store(home, monkeypatch)
    db = store._db
    try:
        parent = _parent(store, carrier)
        _produce(home, store, parent, shape)
        if compacted:
            _compact_live_tail(db, parent, 1)
        parent_prompt = db.get_session(parent)["system_prompt"]
        child = _branch(store, carrier, parent)
        assert db.get_session(child)["system_prompt"] == parent_prompt
        if lose_prompt:
            db.update_system_prompt(child, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, child, "construction")
        assert "Fixture rule B" not in sent and db.get_session(parent)["system_prompt"] == parent_prompt
        if lose_prompt:
            assert _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
        else:
            assert "Fixture rule A" in sent and isinstance(decided, hg.Carried)
        assert [m.get("display_metadata") for m in db.get_messages_as_conversation(child)
                if m.get("display_kind") == "failed_turn"] == [{STANDS_FOR_TURN_ROW: True}]
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("history", ["mirror-only", "notice-only"])
@pytest.mark.parametrize("carrier", ["cli", "gateway"])
def test_a_branch_of_a_history_with_no_reply_takes_up_the_file_first(home, no_network, monkeypatch, carrier, history):
    """A parent holding only a delivered brief and outgoing mirror (no turn), or a request whose turn failed before any
    agent was built (the gateway closer's notice): the branch keeps those rows' provenance, so the child's first actual
    turn takes up the configured file instead of reporting a lost decision (R8-NIT-03 for mirrors)."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    store = _mirror_store(home, monkeypatch)
    db = store._db
    try:
        parent = _parent(store, carrier)
        if carrier == "cli":
            db.create_session(parent, source="cli")
        if history == "mirror-only":
            _mirror(parent, MIRRORS["brief-then-outgoing"])
        else:
            db.append_message(parent, "user", "first request")
            _gateway_close(store, parent, FAILED_TURN_NOTICE)
        assert db.get_session(parent)["system_prompt"] is None
        child = _branch(store, carrier, parent)
        kinds = [(m.get("display_kind"), m.get("display_metadata")) for m in db.get_messages_as_conversation(child)
                 if m["role"] == "assistant"]
        db.append_message(child, "user", "First actual request to the branch")
        sent, decided = _next_turn(db, child, "construction")
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
        assert kinds and all(k == "failed_turn" or (meta or {}).get("delivery_mirror") for k, meta in kinds)
    finally:
        store.close_all_db_handles()


def test_a_cli_branch_copies_rows_without_sharing_or_dropping_their_fields(home, no_network, monkeypatch):
    """The CLI copy writes the child's rows from the parent's live list without mutating it: the parent's dicts keep
    their metadata; the child's rows carry the uid, data-use requirement and provenance; and the request the child's
    next turn builds (``build_api_messages``) carries the text but none of that persistence-only metadata."""
    from agent.message_metadata import STANDS_FOR_TURN_ROW
    from agent.turn_context import build_api_messages
    store = _mirror_store(home, monkeypatch)
    db = store._db
    try:
        _produce(home, store, "cli-parent", "late-partial")
        history = store.load_transcript("cli-parent")
        history[-1]["display_metadata"] = {**history[-1]["display_metadata"], "data_use": ["no_training"]}
        history[-1]["data_use"] = ["no_training"]
        before = json.dumps(history, sort_keys=True, default=str)
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        cli = SimpleNamespace(_agent_running=False, _session_db=db, session_id="cli-parent", agent=None,
                              conversation_history=history, model="anthropic/claude-sonnet-4.6", max_turns=20,
                              reasoning_config={}, _new_session_id=lambda *args, **kwargs: "cli-child")
        CLICommandsMixin._handle_branch_command(cli, "/branch probe")
        assert json.dumps(history, sort_keys=True, default=str) == before
        copied = db.get_messages_as_conversation("cli-child")[-1]
        assert copied.get("display_kind") == "failed_turn" and copied["message_uid"] == history[-1]["message_uid"]
        assert copied["display_metadata"] == {STANDS_FOR_TURN_ROW: True, "data_use": ["no_training"]}
        assert copied["data_use"] == ["no_training"]
        child_history = db.get_messages_as_conversation("cli-child") + [{"role": "user", "content": "next question"}]
        agent = _agent(session_id="cli-child", session_db=db)
        try:
            prompt = _turn(agent, child_history)
            api, _system = build_api_messages(agent, child_history, current_turn_user_idx=len(child_history) - 1,
                                              ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
                                              active_system_prompt=prompt)
        finally:
            agent.close()
        wire = json.dumps(api, default=str)
        assert "Here is the first half of the answer you asked" in wire
        assert "display_metadata" not in wire and STANDS_FOR_TURN_ROW not in wire and "display_kind" not in wire
    finally:
        store.close_all_db_handles()
