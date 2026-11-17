"""Cap-only uptake through the real prompt builder, stale check and SessionDB."""

import json
from copy import deepcopy
from types import SimpleNamespace

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import build_system_prompt
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.bot_mode_dm import message_agent_tool_schema
from tools.bot_mode_probe import stored_prompt_capability_stale


def test_cap_only_next_turn_persists_whole_soul_without_replacing_session(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "editor"
    home.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (home / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    title: Editor\n", encoding="utf-8")
    config = home / "config.yaml"
    config.write_text("context_file_max_chars: 138728\n", encoding="utf-8")
    # No leading/trailing whitespace: exact whole-file equality survives loader.strip().
    soul = "# Fixture identity\n" + "x" * (138976 - len("# Fixture identity\n"))
    (home / "SOUL.md").write_text(soul, encoding="utf-8")
    # Only suppress the unrelated host probe; loader/build/epoch/persistence are real.
    monkeypatch.setattr("agent.prompt_builder.build_environment_hints", lambda: "")
    token = set_hermes_home_override(str(home))
    db = SessionDB(db_path=home / "state.db")
    sid = "local-cap-refresh-fixture"
    try:
        db.create_session(sid, "cli", model="fixture-model", user_id="fixture-owner",
                          session_key="fixture-route", profile_name="editor")
        db.set_session_title(sid, "Bot Chat")
        db.append_message(sid, "user", "Keep the existing conversation.")
        db.append_message(sid, "assistant", "The existing history stays.")
        history = db.get_messages(sid)
        saved_history = deepcopy(history)
        agent = SimpleNamespace(
            session_id=sid, _session_db=db, _owns_session_db=True,
            _session_title_hint="Bot Chat", load_soul_identity=True,
            skip_context_files=True, valid_tool_names=[], tools=[],
            enabled_toolsets=None, disabled_toolsets=None, _environment_probe=False,
            _memory_store=None, _memory_manager=None, model="fixture-model",
            provider="openai", platform="cli", pass_session_id=True,
            _use_prompt_caching=False, _cached_system_prompt=None,
            _emit_diagnostic_status=lambda *_: None,
        )
        builds = []

        def build(system_message):
            builds.append(system_message)
            return build_system_prompt(agent, system_message)

        agent._build_system_prompt = build
        _restore_or_build_system_prompt(agent, None, [])
        clipped = db.get_session(sid)["system_prompt"]
        assert clipped == agent._cached_system_prompt
        assert soul not in clipped
        assert "truncated" in clipped
        assert not stored_prompt_capability_stale(clipped, home)
        before = db.get_session(sid)
        assert db.try_acquire_session_turn_lease(sid, "fixture-turn-owner")
        lease_before = db.session_turn_lease_expires_at(sid, "fixture-turn-owner")

        config.write_text("context_file_max_chars: 138976\n", encoding="utf-8")
        _restore_or_build_system_prompt(agent, None, history)
        whole = db.get_session(sid)["system_prompt"]
        assert whole.count(soul) == 1
        assert "truncated" not in whole
        assert whole == agent._cached_system_prompt
        assert not stored_prompt_capability_stale(whole, home)
        assert len(builds) == 2
        # An equivalent numeric spelling must reuse the persisted bytes too.
        config.write_text("context_file_max_chars: 138976.9\n", encoding="utf-8")
        agent._cached_system_prompt = None
        _restore_or_build_system_prompt(agent, None, history)
        assert agent._cached_system_prompt == whole
        assert db.get_session(sid)["system_prompt"] == whole
        assert len(builds) == 2
        assert history == saved_history == db.get_messages(sid)
        after = db.get_session(sid)
        for key in ("id", "source", "model", "user_id", "session_key", "profile_name",
                    "parent_session_id", "started_at", "ended_at", "message_count"):
            assert after[key] == before[key], key
        # The refresh re-pins the canonical Bot Chat's tools with its message_agent schema
        # (_refresh_bot_chat_tools) and changes nothing else in the pin.
        pinned_before, pinned_after = json.loads(before["tool_names"]), json.loads(after["tool_names"])
        assert pinned_after == {**pinned_before, "tools": [*pinned_before["tools"], message_agent_tool_schema()]}
        assert agent._session_db is db and agent._owns_session_db is True
        assert agent.session_id == sid
        assert db.session_turn_lease_expires_at(sid, "fixture-turn-owner") == lease_before
        assert not (root / "state.db").exists()
    finally:
        db.release_session_turn_lease(sid, "fixture-turn-owner")
        db.close()
        reset_hermes_home_override(token)
