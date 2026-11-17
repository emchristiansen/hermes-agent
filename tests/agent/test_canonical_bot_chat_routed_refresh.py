"""A canonical Bot Chat whose gateway route was fresh-reset onto a successor session must still
adopt capability changes: the successor is where every message to the canonical chat lands, but
its title came from its first message, so the title-only "Bot Chat" gates skipped it and its
pinned tools[] (with tools later deferred) stayed frozen.

Real SessionDB, real gateway routing write, real prompt builder, real stale check and real
tool-definition builder; only the unrelated host-environment probe is suppressed."""

from __future__ import annotations

import contextlib
import io
import json
import textwrap
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _responses_tools
from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import build_system_prompt
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools import bot_mode_probe
from tools.bot_mode_probe import stored_prompt_capability_stale
from tools.mcp_tool_agent import tool_pin_version

PROFILE = "editor"
LOGICAL = "local-logical-canonical-bot-chat"
SUCCESSOR = "20261108_141119_successor"
TASK_TITLE = "Write final outage and containment notice"
# Deferred below and present in the pre-deferral tools. manage_connections is deferred too, but this
# fixture has no Nous sign-in, so the connector gate (tools/managed_tool_gateway.py) never offers it.
DEFERRED = ["clarify", "memory", "web_extract"]
PRE_CONFIG = "agent:\n  disabled_toolsets: []\n"
POST_CONFIG = textwrap.dedent(
    """\
    agent:
      disabled_toolsets: [tts]
      disabled_tools: [skill_manage]
    tools:
      tool_search:
        defer: [clarify, memory, web_extract, manage_connections]
    """
)


def _names(tools):
    return [t.get("function", t).get("name") for t in tools]


def _build_tools(home, disabled_toolsets):
    import model_tools
    from hermes_cli import config as hermes_config
    hermes_config._LOAD_CONFIG_CACHE.clear()
    for name in ("_TOOL_DEFS_CACHE", "_tool_defs_cache"):
        cache = getattr(model_tools, name, None)
        if hasattr(cache, "clear"):
            cache.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        return model_tools.get_tool_definitions(
            enabled_toolsets=None, disabled_toolsets=disabled_toolsets, quiet_mode=True) or []


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    for name in (PROFILE, "researcher"):
        d = root / "profiles" / name
        d.mkdir(parents=True)
        (d / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    shape: cloud\n", encoding="utf-8")
    home = root / "profiles" / PROFILE
    (home / "SOUL.md").write_text("# Fixture editor identity\n", encoding="utf-8")
    (home / "config.yaml").write_text(PRE_CONFIG, encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("agent.prompt_builder.build_environment_hints", lambda: "")
    bot_mode_probe._reset_cache_for_tests()
    token = set_hermes_home_override(str(home))
    db = SessionDB(db_path=home / "state.db")
    try:
        yield SimpleNamespace(root=root, home=home, db=db)
    finally:
        db.close()
        reset_hermes_home_override(token)
        bot_mode_probe._reset_cache_for_tests()


def _agent(db, sid, tools, builds):
    agent = SimpleNamespace(
        session_id=sid, _session_db=db, _owns_session_db=True, _session_title_hint=None,
        load_soul_identity=True, skip_context_files=True, valid_tool_names=set(_names(tools)),
        tools=deepcopy(tools), enabled_toolsets=None, disabled_toolsets=None,
        _environment_probe=False, _memory_store=None, _memory_manager=None, model="fixture-model",
        provider="openai", platform="local", pass_session_id=True, _use_prompt_caching=False,
        _cached_system_prompt=None, _emit_diagnostic_status=lambda *_: None,
        _memory_enabled=False, _user_profile_enabled=False, _tool_use_enforcement="auto",
    )

    def build(system_message):
        builds.append(system_message)
        return build_system_prompt(agent, system_message)

    agent._build_system_prompt = build
    return agent


def _route(db, sid, chat_id, *, platform="local", chat_type="dm", raw=None):
    entry = raw if raw is not None else json.dumps({
        "session_id": sid, "session_key": f"agent:{PROFILE}:{platform}:{chat_type}:{chat_id}",
        "platform": platform, "chat_type": chat_type,
        "origin": {"chat_id": chat_id, "chat_type": chat_type, "platform": platform, "profile": PROFILE},
    })
    db.save_gateway_routing_entry(f"agent:{PROFILE}:{platform}:{chat_type}:{chat_id}", entry)


def _legacy_successor(fleet, *, logical_title="Bot Chat"):
    """The live shape: a "Bot Chat" logical row, and a task-titled successor holding an unstamped
    prompt and a pin built before the deferral config existed. Returns (pre_tools, history)."""
    db = fleet.db
    db.create_session(LOGICAL, "local", model="fixture-model", user_id="uid:1000",
                      session_key=f"agent:{PROFILE}:local:dm:{LOGICAL}", profile_name=PROFILE)
    db.set_session_title(LOGICAL, logical_title)
    db.create_session(SUCCESSOR, "local", model="fixture-model", user_id="uid:1000",
                      session_key=f"agent:{PROFILE}:local:dm:{LOGICAL}", profile_name=PROFILE)
    db.set_session_title(SUCCESSOR, TASK_TITLE)
    db.append_message(SUCCESSOR, "user", "Draft the notice.")
    db.append_message(SUCCESSOR, "assistant", "Drafted.")
    pre_tools = _build_tools(fleet.home, [])
    builds = []
    first = _agent(db, SUCCESSOR, pre_tools, builds)
    _restore_or_build_system_prompt(first, None, [])  # first turn, before the route moved here
    db.update_session_tool_names(SUCCESSOR, {"version": tool_pin_version(), "tools": pre_tools})
    return pre_tools, db.get_messages(SUCCESSOR)


def test_routed_successor_adopts_capability_change_once_and_keeps_session(fleet):
    db, home = fleet.db, fleet.home
    pre_tools, history = _legacy_successor(fleet)
    stored = db.get_session(SUCCESSOR)["system_prompt"]
    assert "Capability epoch:" not in stored  # the live unstamped state
    assert set(DEFERRED) <= set(_names(pre_tools))
    _route(db, SUCCESSOR, LOGICAL)  # the canonical route's fresh reset now targets the successor
    before = db.get_session(SUCCESSOR)
    logical_before = db.get_session(LOGICAL)
    saved_history = deepcopy(history)

    (home / "config.yaml").write_text(POST_CONFIG, encoding="utf-8")
    fresh = _build_tools(home, ["tts"])
    assert not set(DEFERRED) & set(_names(fresh)) and "skill_manage" not in _names(fresh)
    builds = []
    agent = _agent(db, SUCCESSOR, fresh, builds)
    _restore_or_build_system_prompt(agent, None, history)

    # The config-only builder does not assemble this agent-specific dynamic tool.
    from tools.bot_mode_dm import message_agent_tool_schema
    expected = fresh + [message_agent_tool_schema()]

    rebuilt = db.get_session(SUCCESSOR)["system_prompt"]
    assert len(builds) == 1 and rebuilt == agent._cached_system_prompt
    assert "Capability epoch:" in rebuilt and not stored_prompt_capability_stale(rebuilt, home)
    assert "`@researcher`" in rebuilt  # the Bot Chat teammate protocol the successor had lost
    pin = json.loads(db.get_session(SUCCESSOR)["tool_names"])
    assert pin["tools"] == expected
    assert not set(DEFERRED) & set(_names(pin["tools"])) and "skill_manage" not in _names(pin["tools"])
    # The completed pin includes both config-built and canonical dynamic schemas.
    assert _responses_tools(pin["tools"]) == _responses_tools(expected)

    # Next turn, config unchanged: the persisted bytes and pin are reused, no second rebuild.
    builds2 = []
    again = _agent(db, SUCCESSOR, fresh, builds2)
    _restore_or_build_system_prompt(again, None, history)
    assert builds2 == [] and again._cached_system_prompt == rebuilt
    assert db.get_session(SUCCESSOR)["system_prompt"] == rebuilt
    assert json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"] == expected

    # Same physical session, history, title and logical row.
    after = db.get_session(SUCCESSOR)
    for key in ("id", "source", "model", "user_id", "session_key", "profile_name",
                "parent_session_id", "started_at", "ended_at", "message_count", "title"):
        assert after[key] == before[key], key
    assert db.get_messages(SUCCESSOR) == saved_history
    assert db.get_session(LOGICAL) == logical_before


@pytest.mark.parametrize("case", ["unrouted", "non_bot_chat_target", "telegram_origin", "entry_without_origin"])
def test_noncanonical_session_keeps_its_pin_and_prompt(fleet, case):
    db, home = fleet.db, fleet.home
    _pre_tools, history = _legacy_successor(fleet, logical_title=("Triage" if case == "non_bot_chat_target" else "Bot Chat"))
    if case in ("unrouted",):
        pass
    elif case == "telegram_origin":
        _route(db, SUCCESSOR, LOGICAL, platform="telegram")
    elif case == "entry_without_origin":  # the DB refuses malformed JSON; an origin-less entry is writable
        _route(db, SUCCESSOR, LOGICAL, raw=json.dumps({"session_id": SUCCESSOR}))
    else:
        _route(db, SUCCESSOR, LOGICAL)
    stored = db.get_session(SUCCESSOR)["system_prompt"]
    predicate = getattr(bot_mode_probe, "session_is_canonical_bot_chat", None)  # absent on the unpatched control tree
    assert predicate is None or not predicate(db, SUCCESSOR, TASK_TITLE)

    (home / "config.yaml").write_text(POST_CONFIG, encoding="utf-8")
    fresh = _build_tools(home, ["tts"])
    builds = []
    agent = _agent(db, SUCCESSOR, fresh, builds)
    _restore_or_build_system_prompt(agent, None, history)

    assert builds == [] and agent._cached_system_prompt == stored
    assert db.get_session(SUCCESSOR)["system_prompt"] == stored
    assert "Capability epoch:" not in stored
    # Ordinary pin-freeze stability: the pinned, still-permitted tools stay in the request.
    assert set(DEFERRED) <= set(_names(agent.tools))


def test_predicate_fails_closed(fleet):
    from tools.bot_mode_probe import session_is_canonical_bot_chat
    db = fleet.db
    assert session_is_canonical_bot_chat(None, SUCCESSOR, TASK_TITLE) is False
    assert session_is_canonical_bot_chat(db, None, None) is False
    assert session_is_canonical_bot_chat(db, "absent-session", "") is False
    assert session_is_canonical_bot_chat(None, None, "Bot Chat") is True  # today's title rule

    class Broken:
        def gateway_routing_entry_for_session(self, _sid):
            raise RuntimeError("database is locked")

    assert session_is_canonical_bot_chat(Broken(), SUCCESSOR, TASK_TITLE) is False

    # A route whose chat id is the session itself is not a successor.
    db.create_session(LOGICAL, "local", model="m", user_id="u", profile_name=PROFILE)
    db.set_session_title(LOGICAL, "Notes")
    _route(db, LOGICAL, LOGICAL)
    assert session_is_canonical_bot_chat(db, LOGICAL, "Notes") is False


def _moved_route_post_config(fleet):
    """The live shape after HOME5: an unstamped routed successor, post-deferral config on disk."""
    db, home = fleet.db, fleet.home
    _pre, history = _legacy_successor(fleet)
    _route(db, SUCCESSOR, LOGICAL)
    (home / "config.yaml").write_text(POST_CONFIG, encoding="utf-8")
    return history, _build_tools(home, ["tts"])


def test_refreshed_pin_is_the_full_config_built_wire_array(fleet):
    """Not just names: the persisted pin is the fresh definitions byte for byte, and what the
    transport sends from it (Responses conversion plus the reserved-name alias) is identical."""
    from agent.transports.codex import _XAI_RESERVED_TOOL_NAMES, _alias_reserved_tools
    db = fleet.db
    history, fresh = _moved_route_post_config(fleet)
    agent = _agent(db, SUCCESSOR, fresh, [])
    _restore_or_build_system_prompt(agent, None, history)
    pin = json.loads(db.get_session(SUCCESSOR)["tool_names"])
    assert pin["version"] == tool_pin_version()
    from tools.bot_mode_dm import message_agent_tool_schema
    expected = fresh + [message_agent_tool_schema()]
    assert json.dumps(pin["tools"], sort_keys=True) == json.dumps(expected, sort_keys=True)
    wire_pin, alias_pin = _alias_reserved_tools(_responses_tools(pin["tools"]) or [], _XAI_RESERVED_TOOL_NAMES)
    wire_fresh, alias_fresh = _alias_reserved_tools(_responses_tools(expected) or [], _XAI_RESERVED_TOOL_NAMES)
    assert json.dumps(wire_pin, sort_keys=True) == json.dumps(wire_fresh, sort_keys=True)
    assert alias_pin == alias_fresh


def test_failed_persist_keeps_this_turn_and_refires_once_next_turn(fleet, monkeypatch, caplog):
    """A failed prompt write must not strand the session: this turn already runs the rebuilt
    prompt, the stored prompt is left unstamped, and the next turn re-fires and persists once.
    The pin is written first, so it may already be current: a fresh pin under a stale prompt
    re-fires harmlessly; only the reverse (a fresh stamp over an old pin) would never retry."""
    db = fleet.db
    history, fresh = _moved_route_post_config(fleet)
    stored = db.get_session(SUCCESSOR)["system_prompt"]
    stored_pin = db.get_session(SUCCESSOR)["tool_names"]
    real = SessionDB.update_system_prompt
    calls = {"n": 0}

    def flaky(self, sid, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return real(self, sid, prompt)

    monkeypatch.setattr(SessionDB, "update_system_prompt", flaky)
    builds1 = []
    first = _agent(db, SUCCESSOR, fresh, builds1)
    with caplog.at_level("WARNING"):
        _restore_or_build_system_prompt(first, None, history)
    assert len(builds1) == 1 and "Capability epoch:" in first._cached_system_prompt
    assert db.get_session(SUCCESSOR)["system_prompt"] == stored
    assert db.get_session(SUCCESSOR)["tool_names"] != stored_pin
    assert _names(json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"]) == _names(first.tools)
    assert any("will re-fire next turn" in r.getMessage() for r in caplog.records)

    builds2 = []
    second = _agent(db, SUCCESSOR, fresh, builds2)
    _restore_or_build_system_prompt(second, None, history)
    assert len(builds2) == 1
    persisted = db.get_session(SUCCESSOR)["system_prompt"]
    assert persisted == second._cached_system_prompt and "Capability epoch:" in persisted
    from tools.bot_mode_dm import message_agent_tool_schema
    assert json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"] == fresh + [message_agent_tool_schema()]

    builds3 = []
    third = _agent(db, SUCCESSOR, fresh, builds3)
    _restore_or_build_system_prompt(third, None, history)
    assert builds3 == [] and third._cached_system_prompt == persisted


def test_gateway_policy_turn_refreshes_through_the_real_rebuild(fleet):
    """Live local DM turns run under a frozen session policy, so the stale branch rebuilds tools
    with ``refresh_agent_mcp_tools(enabled_override=policy.toolsets)``. The real rebuild reads the
    profile's live ``disabled_tools`` and deferral config, publishes, and the turn re-pins it."""
    from gateway.session_policy import _ACTIVE_POLICY
    from tools.registry import registry
    db = fleet.db
    history, _fresh = _moved_route_post_config(fleet)
    toolsets = sorted({getattr(e, "toolset", None) for e in registry.get_all_entries()} - {None})
    assert toolsets
    builds = []
    # The agent arrives holding the pre-HOME5 pin (a cached agent's frozen surface).
    pinned = json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"]
    agent = _agent(db, SUCCESSOR, pinned, builds)
    # A policy stub carrying only what this path reads (toolsets; source for the skills prompt variant).
    token = _ACTIVE_POLICY.set(SimpleNamespace(toolsets=tuple(toolsets), source="local"))
    try:
        _restore_or_build_system_prompt(agent, None, history)
    finally:
        _ACTIVE_POLICY.reset(token)
    assert len(builds) == 1
    live = _names(agent.tools)
    assert not set(DEFERRED) & set(live) and "skill_manage" not in live
    persisted = _names(json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"])
    assert persisted == live
    assert "Capability epoch:" in db.get_session(SUCCESSOR)["system_prompt"]


@pytest.mark.parametrize("under_policy", [False, True])
def test_failed_pin_write_never_stamps_the_prompt_and_refires_next_fresh_agent(fleet, monkeypatch, caplog, under_policy):
    """FRESH STAMP IMPLIES FRESH PIN. If the refreshed tools[] pin cannot be written, a current
    epoch must not reach the row: it would read as fresh over the pre-refresh pin and never
    refresh again. The row stays as it was, the warning says so, and the next fresh agent (a
    rebuilt cache entry) re-fires the refresh and persists pin and stamp."""
    from gateway.session_policy import _ACTIVE_POLICY
    from tools.registry import registry
    db = fleet.db
    history, fresh = _moved_route_post_config(fleet)
    stored = db.get_session(SUCCESSOR)["system_prompt"]
    stored_pin = db.get_session(SUCCESSOR)["tool_names"]
    pinned = json.loads(stored_pin)["tools"]
    real = SessionDB.update_session_tool_names
    broken = {"on": True}

    def flaky(self, sid, pin):
        if broken["on"]:
            raise RuntimeError("database is locked")
        return real(self, sid, pin)

    monkeypatch.setattr(SessionDB, "update_session_tool_names", flaky)
    toolsets = sorted({getattr(e, "toolset", None) for e in registry.get_all_entries()} - {None})
    policy = SimpleNamespace(toolsets=tuple(toolsets), source="local") if under_policy else None

    def turn(tools):
        builds = []
        agent = _agent(db, SUCCESSOR, tools, builds)
        token = _ACTIVE_POLICY.set(policy)
        try:
            _restore_or_build_system_prompt(agent, None, history)
        finally:
            _ACTIVE_POLICY.reset(token)
        return agent, builds

    with caplog.at_level("WARNING"):
        first, builds1 = turn(pinned if under_policy else fresh)
    assert len(builds1) == 1
    # This (possibly cached) agent runs the refreshed prompt and tools in memory...
    assert "Capability epoch:" in first._cached_system_prompt
    assert not set(DEFERRED) & set(_names(first.tools))
    # ...but the row keeps the stale prompt and the old pin together, and says why.
    assert db.get_session(SUCCESSOR)["system_prompt"] == stored
    assert db.get_session(SUCCESSOR)["tool_names"] == stored_pin
    assert any("the tools[] pin could not be persisted" in r.getMessage() for r in caplog.records)

    broken["on"] = False
    second, builds2 = turn(pinned if under_policy else fresh)  # the next fresh agent
    assert len(builds2) == 1
    persisted_prompt = db.get_session(SUCCESSOR)["system_prompt"]
    persisted_pin = _names(json.loads(db.get_session(SUCCESSOR)["tool_names"])["tools"])
    assert "Capability epoch:" in persisted_prompt and persisted_prompt == second._cached_system_prompt
    assert persisted_pin == _names(second.tools)
    assert not set(DEFERRED) & set(persisted_pin) and "skill_manage" not in persisted_pin

    third, builds3 = turn(fresh)
    assert builds3 == [] and third._cached_system_prompt == persisted_prompt


def _real_local_policy(home):
    from gateway.session_policy import LocalSessionPolicy
    from tools.registry import registry
    return LocalSessionPolicy(
        source="cli", platform="local", cwd=str(home), model="fixture-model",
        toolsets=tuple(sorted({entry.toolset for entry in registry.get_all_entries()})),
        config_json="{}", request_json="{}", terminal_json="{}",
    )


def test_tool_builder_failure_prevents_capability_completion_and_retries(fleet, monkeypatch):
    """A failed prerequisite must not commit an epoch over the previous tool surface."""
    import model_tools
    from gateway.session_policy import _ACTIVE_POLICY
    history, _fresh = _moved_route_post_config(fleet)
    before = fleet.db.get_session(SUCCESSOR)
    previous_tools = json.loads(before["tool_names"])["tools"]
    real = model_tools.get_tool_definitions

    def unavailable(**kwargs):
        raise OSError("injected tool definition build failure")

    token = _ACTIVE_POLICY.set(_real_local_policy(fleet.home))
    try:
        monkeypatch.setattr(model_tools, "get_tool_definitions", unavailable)
        first = _agent(fleet.db, SUCCESSOR, previous_tools, [])
        with pytest.raises(OSError, match="injected tool definition build failure"):
            _restore_or_build_system_prompt(first, None, history)
        after = fleet.db.get_session(SUCCESSOR)
        assert after["system_prompt"] == before["system_prompt"]
        assert after["tool_names"] == before["tool_names"]
        assert first._cached_system_prompt is None

        monkeypatch.setattr(model_tools, "get_tool_definitions", real)
        builds = []
        again = _agent(fleet.db, SUCCESSOR, previous_tools, builds)
        _restore_or_build_system_prompt(again, None, history)
        assert len(builds) == 1
        assert not set(DEFERRED) & set(_names(again.tools))
        assert not stored_prompt_capability_stale(again._cached_system_prompt, fleet.home)
    finally:
        _ACTIVE_POLICY.reset(token)


def test_canonical_message_authority_and_wire_survive_fresh_agent(fleet):
    from agent.transports.codex import _XAI_RESERVED_TOOL_NAMES, _alias_reserved_tools
    from gateway.session_policy import _ACTIVE_POLICY
    from tools.bot_mode_dm import ensure_message_agent_tool, message_agent_authorized, message_agent_tool
    history, fresh = _moved_route_post_config(fleet)
    saved_title = fleet.db.get_session_title(SUCCESSOR)
    saved_history = deepcopy(history)
    token = _ACTIVE_POLICY.set(_real_local_policy(fleet.home))
    try:
        first = _agent(fleet.db, SUCCESSOR, fresh, [])
        _restore_or_build_system_prompt(first, None, history)
        assert ensure_message_agent_tool(first) and message_agent_authorized(first)
        original_wire = _alias_reserved_tools(_responses_tools(first.tools), _XAI_RESERVED_TOOL_NAMES)
        builds = []
        again = _agent(fleet.db, SUCCESSOR, fresh, builds)
        _restore_or_build_system_prompt(again, None, history)
        assert ensure_message_agent_tool(again) and message_agent_authorized(again)
        assert builds == [] and again._cached_system_prompt == first._cached_system_prompt
        assert _alias_reserved_tools(_responses_tools(again.tools), _XAI_RESERVED_TOOL_NAMES) == original_wire
        # Empty body reaches input validation only after the real invocation authorization gate.
        # It cannot dispatch a message.
        result = json.loads(message_agent_tool(agent=again))
        assert result["error"] == "message is required — compose what you want to say to that agent."
        assert fleet.db.get_session_title(SUCCESSOR) == saved_title
        assert fleet.db.get_messages(SUCCESSOR) == saved_history
    finally:
        _ACTIVE_POLICY.reset(token)


def test_noncanonical_message_invocation_is_refused(fleet):
    from tools.bot_mode_dm import ensure_message_agent_tool, message_agent_authorized, message_agent_tool
    _pre, _history = _legacy_successor(fleet, logical_title="Triage")
    _route(fleet.db, SUCCESSOR, LOGICAL)
    agent = _agent(fleet.db, SUCCESSOR, [], [])
    assert not ensure_message_agent_tool(agent)
    assert not message_agent_authorized(agent)
    result = json.loads(message_agent_tool(agent=agent))
    assert result["error"].startswith("message_agent is only available in a Bot Mode 'Bot Chat' session.")

