"""Where a home-guidance edition is taken up, and where it is not (``agent.home_guidance_file``).

A recorded conversation keeps its decision; a conversation from before decisions were recorded keeps the edition its
identity slot shows (or its absence); a damaged record is unusable, never legacy; a continuing conversation whose stored
prompt is missing or unreadable carries an explicit notice; a bypass session's frozen config carries the administrator's
pinned setting from the moment it is created, and a later edit of the managed layer reaches only sessions created after
it.
"""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import hermes_yaml as yaml
import pytest

from agent import home_guidance as hg
from tests.agent.test_home_guidance import CORE_B, SOURCE, _agent, _expected_block, _prompt, _restore

pytest_plugins = ["tests.agent.test_home_guidance"]  # its ``home`` fixture

STATE_PREFIX = "<!-- Hermes home guidance: "


def _without_record(prompt):
    """The prompt as a build from before decisions were recorded wrote it: the same bytes without the trailer's record."""
    head, sep, tail = prompt.partition("\n" + STATE_PREFIX)
    assert sep and tail.count(" -->") >= 1
    return head + tail.split(" -->", 1)[1]


# ── A conversation from before decisions were recorded ──────────────────────────────────────────────────────────────
# Such a prompt carried whatever its SOUL.md held and added nothing. Restoring it recovers that edition, or its
# absence, from the prompt's identity slot, and the conversation keeps it through every rebuild like a recorded
# decision (which the next build records); only a new conversation takes up the current inputs.

LEGACY_A = '<workstation_guidance source="fixture">\nLEGACY SOUL EDITION A\n</workstation_guidance>'
SOUL_B = '<workstation_guidance source="fixture">\nNEW SOUL EDITION B\n</workstation_guidance>'


def _legacy(home, session_id, soul):
    """A stored prompt as a build from before decisions were recorded wrote it, with *soul* as its SOUL.md."""
    (home.hermes / "SOUL.md").write_text(soul, encoding="utf-8")
    first = _agent(session_id=session_id)
    try:
        legacy = _without_record(_prompt(first))
    finally:
        first.close()
    assert STATE_PREFIX not in legacy and hg.persisted_state(legacy) is hg.LEGACY
    return legacy


def _resume_and_rebuild(session_id, stored):
    """A new agent resumes *stored* through the real restore path, then rebuilds at the compaction boundary."""
    resumed = _agent(session_id=session_id)
    _restore(resumed, stored)
    resumed._invalidate_system_prompt()
    return resumed, _prompt(resumed)


def test_a_legacy_conversation_takes_up_nothing_on_a_rebuild(home):
    _write_config(home)
    legacy = _legacy(home, "legacy", "Legacy persona.\n")
    assert "workstation_guidance" not in legacy
    _write_config(home, home_guidance_file=SOURCE)  # configured while the old conversation continues
    resumed, rebuilt = _resume_and_rebuild("legacy", legacy)
    try:
        assert "workstation_guidance" not in rebuilt and "Legacy persona." in rebuilt
        assert hg.frozen(resumed) is None and hg.persisted_state(rebuilt) is None  # its absence is now recorded
        # A delegate child is a new conversation: it decides from the inputs configured now.
        from tests.tools.test_delegate_home_guidance import _child, _child_prompt
        child = _child(resumed)
        try:
            assert child._home_guidance_inherited is None
            assert _child_prompt(child).count(_expected_block(home.core)) == 1
        finally:
            child.close()
        # /new ends the legacy conversation; its first build decides and records.
        resumed.reset_session_state()
        fresh = _prompt(resumed)
        assert fresh.count(_expected_block(home.core)) == 1 and hg.persisted_state(fresh) == hg.frozen(resumed)
    finally:
        resumed.close()


def test_a_legacy_soul_block_renders_as_before(home):
    """Before decisions were recorded, a SOUL.md block was SOUL.md's own text: a legacy rebuild keeps it as read."""
    _write_config(home)
    legacy = _legacy(home, "legacy-soul", f"Role.\n\n{LEGACY_A}\n")
    resumed, rebuilt = _resume_and_rebuild("legacy-soul", legacy)
    try:
        assert rebuilt.count(LEGACY_A) == 1 and rebuilt.count("<workstation_guidance") == 1
        assert hg.frozen(resumed) == hg.Carried(LEGACY_A, "soul") == hg.persisted_state(rebuilt)
    finally:
        resumed.close()


SOUL_LATER = {"gains B": f"Role, revised.\n\n{SOUL_B}\n", "drops its block": "Role, revised.\n"}


@pytest.mark.parametrize("later", sorted(SOUL_LATER))
@pytest.mark.parametrize("original", ["absent", "carried"])
def test_a_legacy_conversation_keeps_its_edition_when_soul_changes(home, original, later):
    """R3-LEGACY-01: SOUL.md changes after a legacy conversation began (absent -> B, A -> B, A -> none). Its rebuilds
    keep exactly the edition it carried — never B, never losing A — while the rest of SOUL.md follows its re-read,
    and a second resume of the rebuilt prompt keeps it from the record; a new conversation takes up the current SOUL."""
    _write_config(home)
    legacy = _legacy(home, "legacy-drift", "Role.\n\n" + (LEGACY_A if original == "carried" else "") + "\n")
    assert legacy.count(LEGACY_A) == (original == "carried") and legacy.count("<workstation_guidance") == legacy.count(LEGACY_A)
    _write_config(home, home_guidance_file=SOURCE)  # a configured file must not stand in for the legacy edition either
    (home.hermes / "SOUL.md").write_text(SOUL_LATER[later], encoding="utf-8")
    home.core.write_text(CORE_B, encoding="utf-8")
    kept = [LEGACY_A] if original == "carried" else []
    resumed, rebuilt = _resume_and_rebuild("legacy-drift", legacy)
    try:
        assert "NEW SOUL EDITION B" not in rebuilt and "Fixture rule" not in rebuilt and "Role, revised." in rebuilt
        assert re.findall(r"<workstation_guidance .*?</workstation_guidance>", rebuilt, re.DOTALL) == kept
        expected = hg.Carried(LEGACY_A, "soul") if kept else None
        assert hg.frozen(resumed) == expected == hg.persisted_state(rebuilt)
    finally:
        resumed.close()
    again, rebuilt_again = _resume_and_rebuild("legacy-drift", rebuilt)
    try:
        assert hg.frozen(again) == expected and "NEW SOUL EDITION B" not in rebuilt_again
        assert re.findall(r"<workstation_guidance .*?</workstation_guidance>", rebuilt_again, re.DOTALL) == kept
        again.reset_session_state()  # /new: the next conversation decides from the current SOUL.md
        fresh = _prompt(again)
        if later == "gains B":
            assert fresh.count(SOUL_B) == 1 and LEGACY_A not in fresh and hg.frozen(again) == hg.Carried(SOUL_B, "soul")
        else:
            assert LEGACY_A not in fresh and "Fixture rule B" in fresh  # no SOUL block: the configured file decides
    finally:
        again.close()


def test_a_block_quoted_outside_a_legacy_identity_is_not_its_edition(home, tmp_path, monkeypatch):
    """Only the identity slot shows a legacy edition: a block quoted in project context is never taken for one, so
    a SOUL.md that gains a block later is still not taken up."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text(f"Project rule. Example:\n{LEGACY_A}\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    _write_config(home)
    legacy = _legacy(home, "legacy-quoted", "Role.\n")
    assert legacy.count(LEGACY_A) == 1 and hg.legacy_decision(legacy) is None
    (home.hermes / "SOUL.md").write_text(f"Role.\n\n{SOUL_B}\n", encoding="utf-8")
    resumed, rebuilt = _resume_and_rebuild("legacy-quoted", legacy)
    try:
        assert hg.frozen(resumed) is None and "NEW SOUL EDITION B" not in rebuilt
        assert rebuilt.count(LEGACY_A) == 1  # the project's quotation, still where the project put it
    finally:
        resumed.close()


@pytest.mark.parametrize("configured", [True, False])
def test_a_legacy_prompt_without_one_identity_boundary_takes_up_nothing(home, configured):
    """A legacy SOUL.md that quotes the Hermes help paragraph leaves two candidate identity boundaries: the edition
    cannot be read from the prompt, so the conversation takes up neither SOUL.md's later block nor the file. With a
    file configured it carries an explicit notice; where nothing is configured, none (a notice would name a setting
    that does not exist)."""
    from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS
    _write_config(home)
    legacy = _legacy(home, "legacy-ambiguous", f"Role.\n\n{HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS}\n\n{LEGACY_A}\n")
    if configured:
        _write_config(home, home_guidance_file=SOURCE)
        home.core.write_text(CORE_B, encoding="utf-8")
    (home.hermes / "SOUL.md").write_text(f"Role.\n\n{SOUL_B}\n", encoding="utf-8")
    resumed, rebuilt = _resume_and_rebuild("legacy-ambiguous", legacy)
    try:
        assert "NEW SOUL EDITION B" not in rebuilt and "LEGACY SOUL EDITION A" not in rebuilt
        assert "Fixture rule" not in rebuilt
        if configured:
            assert isinstance(hg.frozen(resumed), hg.Unavailable) and "<workstation_guidance_unavailable " in rebuilt
        else:
            assert hg.frozen(resumed) is None and "workstation_guidance" not in rebuilt
    finally:
        resumed.close()


# ── A damaged record is unusable, never legacy (R3-RECORD-02) ───────────────────────────────────────────────────────


def _record_line(prompt):
    lines = [line for line in prompt.splitlines() if line.startswith(STATE_PREFIX)]
    assert len(lines) == 1
    return lines[0]


DAMAGE = {"closing cut short": lambda line: line[:-1], "closing missing": lambda line: line[:-len(" -->")],
          "half the record": lambda line: line[:len(line) // 2], "opening only": lambda line: STATE_PREFIX,
          "opening without its space": lambda line: STATE_PREFIX.rstrip()}


@pytest.mark.parametrize("decision", ["carried", "none"])
@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_a_damaged_record_is_unavailable_and_never_legacy(home, damage, decision):
    """A trailer line that starts like the renderer's record but is cut off is an unusable record: the continuing
    conversation carries an explicit notice, takes up neither the configured file nor a SOUL block, and its child
    inherits that failure. The intact record of the same prompt restores its decision."""
    _write_config(home, **({"home_guidance_file": SOURCE} if decision == "carried" else {}))
    first = _agent(session_id="damaged")
    try:
        stored = _prompt(first)
        recorded = hg.frozen(first)
    finally:
        first.close()
    assert hg.persisted_state(stored) == recorded and (recorded is None) == (decision == "none")
    line = _record_line(stored)
    damaged = stored.replace(line, DAMAGE[damage](line), 1)
    assert isinstance(hg.persisted_state(damaged), hg.Unavailable)
    _write_config(home, home_guidance_file=SOURCE)
    home.core.write_text(CORE_B, encoding="utf-8")
    (home.hermes / "SOUL.md").write_text(f"Role.\n\n{SOUL_B}\n", encoding="utf-8")
    resumed, rebuilt = _resume_and_rebuild("damaged", damaged)
    try:
        assert isinstance(hg.frozen(resumed), hg.Unavailable)
        assert "Fixture rule B" not in rebuilt and "NEW SOUL EDITION B" not in rebuilt
        assert "<workstation_guidance_unavailable " in rebuilt
        assert isinstance(hg.for_child(resumed), hg.Unavailable)
    finally:
        resumed.close()


def test_a_prompt_whose_trailer_ends_in_other_text_is_legacy(home):
    """Only a trailer line that starts like the renderer's record is an attempted record: a prompt without one is a
    legacy prompt (its identity slot then decides), and a record quoted anywhere else is no record at all."""
    _write_config(home)
    first = _agent(session_id="no-record")
    try:
        stored = _prompt(first)
    finally:
        first.close()
    legacy = _without_record(stored)
    assert hg.persisted_state(legacy) is hg.LEGACY
    quoted = legacy.replace("\n\n", f"\n\nQuoted: {STATE_PREFIX}" + '{"state":"none","v":1} -->\n\n', 1)
    assert hg.persisted_state(quoted) is hg.LEGACY


# ── Managed pin versus the profile's own value, at the real config boundary ─────────────────────────────────────────


def _write_config(home, **agent):
    cfg = {"model": {"default": "anthropic/claude-sonnet-4.6"}, "agent": {"environment_probe": False, **agent}}
    (home.hermes / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture
def managed(tmp_path, monkeypatch):
    directory = tmp_path / "etc-hermes"
    directory.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(directory))

    def pin(value):
        (directory / "config.yaml").write_text(yaml.safe_dump({"agent": {"home_guidance_file": value}}),
                                               encoding="utf-8")
    return pin


@pytest.mark.parametrize("profile", [{}, {"home_guidance_file": ""}, {"home_guidance_file": None},
                                     {"home_guidance_file": "~/.codex/OTHER.md"}],
                         ids=["unset", "blank", "null", "another file"])
def test_the_administrators_pin_wins_over_the_profiles_value(home, managed, profile):
    """Managed scope pins a key a standard user cannot override (website/docs/user-guide/managed-scope.md); a blank,
    null or different profile value is overridden, exactly as for every other pinned key."""
    managed(SOURCE)
    _write_config(home, **profile)
    agent = _agent()
    try:
        assert _prompt(agent).count(_expected_block(home.core)) == 1
    finally:
        agent.close()


@pytest.mark.parametrize("profile, expected", [({}, "none"), ({"home_guidance_file": ""}, "none"),
                                               ({"home_guidance_file": None}, "none"),
                                               ({"home_guidance_file": False}, "unavailable")],
                         ids=["unset", "blank", "null", "false"])
def test_without_a_pin_the_profile_decides_and_nothing_is_guessed(home, profile, expected):
    """Without an administrator pin, an unset, blank or null value means "not configured"; any other non-string is
    an explicit error, never a guessed default."""
    _write_config(home, **profile)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        if expected == "none":
            assert "workstation_guidance" not in prompt and hg.frozen(agent) is None
        else:
            assert "the setting must be a path string, not bool" in prompt and "Fixture rule A" not in prompt
    finally:
        agent.close()


# ── The bypass producer: capture at creation, frozen afterwards ─────────────────────────────────────────────────────


def _bypass(tmp_path, **flags):
    from gateway.session_local import _bypass_policy
    cwd = tmp_path / "work"
    cwd.mkdir(exist_ok=True)
    params = {"cwd": str(cwd), "model": "safe-fixture", "provider": "custom", "base_url": "http://127.0.0.1:9/v1",
              "toolsets": ["file"], **flags}
    return _bypass_policy(params, private_secrets={})


@pytest.mark.parametrize("flags", [{"safe_mode": True}, {"ignore_user_config": True}], ids=["safe", "config-only"])
def test_a_bypass_session_freezes_the_administrators_setting_and_nothing_of_the_profile(home, managed, tmp_path,
                                                                                       flags):
    _write_config(home, home_guidance_file="~/.codex/PROFILE-ONLY.md", max_turns=7)
    unpinned = json.loads(_bypass(tmp_path, **flags).config_json)
    assert "home_guidance_file" not in (unpinned.get("agent") or {})  # no pin yet: nothing captured, nothing of the profile
    managed(SOURCE)
    first = _bypass(tmp_path, **flags)
    frozen = json.loads(first.config_json)
    assert frozen["agent"]["home_guidance_file"] == SOURCE and frozen["agent"].get("max_turns") != 7
    managed("~/.codex/LATER.md")  # the administrator edits the pin after the first session was created
    assert json.loads(first.config_json)["agent"]["home_guidance_file"] == SOURCE  # the frozen policy is unchanged
    assert json.loads(_bypass(tmp_path, **flags).config_json)["agent"]["home_guidance_file"] == "~/.codex/LATER.md"


def _run_bound_worker(tmp_path, policy, home, *, label):
    """A fresh exec that binds *policy* exactly as the managed worker bootstrap does (agent/managed_worker.py
    ``bind_worker_policy``), then builds a real agent's prompt; returns the guidance it renders."""
    frame = {"policy": asdict(policy), "safe_mode": policy.safe_mode, "ignore_user_config": policy.ignore_user_config}
    script = tmp_path / f"bound-{label}.py"
    root = Path(__file__).resolve().parents[2]
    script.write_text(f"import sys; sys.path.insert(0, {str(root)!r})\n" + textwrap.dedent('''
        import json, re, sys
        frame = json.loads(sys.stdin.read())
        from agent.managed_worker import bind_worker_policy
        bind_worker_policy(frame)
        from run_agent import AIAgent
        agent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                        api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli",
                        quiet_mode=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
        prompt = agent._build_system_prompt()
        print(json.dumps({"guidance": re.findall(r"<workstation_guidance .*?</workstation_guidance>|"
                                                 r"<workstation_guidance_unavailable [^>]*/>", prompt, re.DOTALL),
                          "skip_context_files": agent.skip_context_files}))
    '''), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG")}
    env.update(HERMES_HOME=str(home.hermes), HOME=str(home.os_home))
    result = subprocess.run([sys.executable, str(script)], input=json.dumps(frame), capture_output=True, text=True,
                            env=env, cwd=root, timeout=120, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_bound_bypass_worker_renders_its_captured_setting_not_a_later_pin(home, managed, tmp_path):
    """Producer → bootstrap → constructor → prompt: the worker reads only its frozen config, so the pin captured at
    creation is what it renders even after the administrator repoints the pin; a session created afterwards renders
    the new one."""
    _write_config(home)
    managed(SOURCE)
    old = _bypass(tmp_path, safe_mode=True)
    later = home.os_home / ".codex" / "LATER.md"
    later.write_text(CORE_B, encoding="utf-8")
    managed(str(later))
    result = _run_bound_worker(tmp_path, old, home, label="old")
    assert result["skip_context_files"] is True
    assert result["guidance"] == [_expected_block(home.core)]
    new = _bypass(tmp_path, safe_mode=True)
    result = _run_bound_worker(tmp_path, new, home, label="new")
    assert len(result["guidance"]) == 1 and "Fixture rule B" in result["guidance"][0]
    assert f'source="{later}"' in result["guidance"][0]


# ── A continuing conversation whose stored prompt is missing or unreadable (R3-OPUS-02) ─────────────────────────────
# Persisted messages without the stored prompt that recorded their conversation's decision (a failed write, a
# ``hermes sessions repair``) are a continuing conversation, not a new one: it carries an explicit notice and takes
# up nothing. Only a session without messages (a new one, or one reset in place) decides from the current inputs.


def _real_db(tmp_path):
    from hermes_state import SessionDB
    return SessionDB(tmp_path / "state.db")


def _unavailable_only(prompt):
    from tests.agent.test_home_guidance import _guidance
    found = _guidance(prompt)
    return len(found) == 1 and found[0].startswith("<workstation_guidance_unavailable ")


@pytest.mark.parametrize("compress_first", [False, True])
@pytest.mark.parametrize("missing", [None, ""], ids=["null", "empty"])
@pytest.mark.parametrize("decision", ["carried", "none", "unavailable"])
def test_a_continuing_acp_conversation_without_its_stored_prompt_takes_up_nothing(home, tmp_path, no_network, decision,
                                                                                missing, compress_first):
    """Real SessionDB -> ACP cold restore -> optional pre-turn ``/compress`` -> normal turn: two persisted messages
    and a NULL or empty stored prompt (the database API's own write), then the inputs change to file B. The turn
    carries an explicit notice, not B, records it, keeps it across a rebuild, and its unprofiled child inherits it."""
    from acp_adapter.commands import SlashCommandsMixin
    from tests.agent.test_home_guidance import _acp_restore, _stored_session, _turn, _child_guidance
    db = _real_db(tmp_path)
    try:
        _stored_session(home, db, "acp-missing", decision, platform="acp")
        db.update_system_prompt("acp-missing", missing)
        assert db.get_session("acp-missing")["system_prompt"] == missing
        host, state = _acp_restore(db, "acp-missing")
        agent = state.agent
        try:
            assert isinstance(hg.frozen(agent), hg.Unavailable)  # decided when the agent was built for the session
            if compress_first:
                SlashCommandsMixin._cmd_compress(host, "", state)
                assert len(state.history) == 2 and isinstance(hg.frozen(agent), hg.Unavailable)
            sent = _turn(agent, state.history)
            assert "Fixture rule" not in sent and _unavailable_only(sent)
            assert "stored prompt" in hg.frozen(agent).reason
            assert isinstance(hg.persisted_state(sent), hg.Unavailable)
            assert len(_child_guidance(agent)) == 1 and "_unavailable" in _child_guidance(agent)[0]
            agent._invalidate_system_prompt()
            assert _unavailable_only(_prompt(agent))
        finally:
            agent.close()
    finally:
        db.close()


def _decided_then_prompt_lost(home, db, session_id):
    """A conversation a real first build decided for (prompt and tools[] pin stored, a reply persisted), then its
    stored prompt is cleared as ``hermes sessions repair`` clears it; returns the transcript."""
    from tests.agent.test_home_guidance import _turn
    agent = _agent(session_id=session_id, session_db=db)
    try:
        _turn(agent, [])
        agent._persist_session([{"role": "user", "content": "hello"},
                                {"role": "assistant", "content": "old reply"}], [])
    finally:
        agent.close()
    assert db.get_session(session_id)["system_prompt"]
    db.update_system_prompt(session_id, None)
    return db.get_messages_as_conversation(session_id)


def test_a_turn_time_read_of_a_decided_row_without_its_prompt_is_unavailable(home, tmp_path, no_network):
    """The normal turn entry on its own (no construction-time adoption): a row that lost its prompt but still shows a
    decided conversation, with history to continue, is unavailable; an agent that already holds its own decision
    keeps it."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        history = _decided_then_prompt_lost(home, db, "turn-null")
        home.core.write_text(CORE_B, encoding="utf-8")
        fresh = _agent(session_id="turn-null")  # built without the database: nothing adopted yet
        try:
            fresh._session_db = db
            _restore_or_build_system_prompt(fresh, None, history)
            assert isinstance(hg.frozen(fresh), hg.Unavailable) and "Fixture rule" not in fresh._cached_system_prompt
            assert hg.persisted_state(db.get_session("turn-null")["system_prompt"]) == hg.frozen(fresh)  # now recorded
        finally:
            fresh.close()
        db.update_system_prompt("turn-null", None)  # missing again for the next agent
        own = _agent(session_id="turn-null")
        try:
            decided = hg.frozen(own) if _prompt(own) else None
            assert isinstance(decided, hg.Carried) and "Fixture rule B" in decided.block
            own._session_db = db
            own._invalidate_system_prompt()  # a compaction rebuild while the stored prompt is missing
            _restore_or_build_system_prompt(own, None, history)
            assert hg.frozen(own) == decided and own._cached_system_prompt.count(decided.block) == 1
        finally:
            own.close()
    finally:
        db.close()


SEEDED = {"first message stored before its agent": ([("user", "hello")], None),
          "outgoing message mirrored into a fresh session": ([("assistant", "Delivered from another chat.")], None),
          "cron or webhook brief": ([("user", "[cron] brief")], None),
          "transcript imported from another tool": ([("user", "hello"), ("assistant", "imported reply")],
                                                    {"imported_from": {"tool": "codex", "path": "/x.jsonl",
                                                                       "through_message_id": 2}}),
          "import recorded before its boundary was": ([("user", "hello"), ("assistant", "imported reply")],
                                                      {"imported_from": {"tool": "codex", "path": "/x.jsonl"}}),
          "request whose turn failed before a reply": ([("user", "hello"),
                                                        ("assistant", "Your request was not processed. Send it again "
                                                                      "if you still want me to carry it out.")], None)}


@pytest.mark.parametrize("seeded", sorted(SEEDED))
def test_a_row_no_build_decided_for_is_a_new_conversation(home, tmp_path, no_network, seeded):
    """Messages that are not a reply to a user message (a TUI first message stored before its agent is built, a cron
    or webhook brief, an outgoing message mirrored into a fresh session, a failed-turn notice stored before the typed
    ``display_kind`` existed), and a transcript imported from another tool with no reply here yet, do not show a
    decided conversation: the agent built for the session, and its first turn, decide from the current inputs."""
    from tests.agent.test_home_guidance import _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        messages, origin = SEEDED[seeded]
        db.create_session(session_id="seeded", source="cli", **({"origin_json": json.dumps(origin)} if origin else {}))
        for role, content in messages:
            db.append_message("seeded", role, content)
        agent = _agent(session_id="seeded", session_db=db)
        try:
            assert hg.frozen(agent) is None and "_home_guidance_persisted" not in agent.__dict__
            history = db.get_messages_as_conversation("seeded")
            sent = _turn(agent, history[:-1] if seeded == "first message stored before its agent" else history)
            assert sent.count(_expected_block(home.core)) == 1 and isinstance(hg.frozen(agent), hg.Carried)
        finally:
            agent.close()
    finally:
        db.close()


def test_an_acp_fork_keeps_its_parents_decision(home, tmp_path, no_network, monkeypatch):
    """An ACP fork continues its parent's conversation: it starts from the parent's stored prompt (as every other
    branch does), so its first turn, a pre-turn ``/compress`` and a later cold restore of the fork keep the parent's
    edition A after the file changed to B — never B, and never a notice for a missing prompt."""
    from acp_adapter.commands import SlashCommandsMixin
    from acp_adapter.session import SessionManager
    from tests.agent.test_home_guidance import _acp_restore, _stored_session, _turn
    db = _real_db(tmp_path)
    try:
        recorded = _stored_session(home, db, "acp-parent", "carried", platform="acp")
        _host, parent = _acp_restore(db, "acp-parent")
        import uuid
        from acp_adapter import session as acp_session
        fork_id = "00000000-0000-4000-8000-000000000f0c"
        monkeypatch.setattr(acp_session.uuid, "uuid4", lambda: uuid.UUID(fork_id))

        def factory():  # the production factory builds the agent for the session id it is given
            agent = _agent(platform="acp", session_id=fork_id, session_db=db)
            agent.compression_enabled = False
            return agent
        manager = SessionManager(db=db, agent_factory=factory)
        manager._sessions[parent.session_id] = parent
        fork = manager.fork_session(parent.session_id)
        try:
            assert fork is not None and fork.session_id == fork_id and db.get_session(fork_id)["system_prompt"]
            restored = SessionManager(db=db, agent_factory=factory).get_session(fork.session_id)  # a cold restore
            try:
                assert restored is not None and len(restored.history) == len(fork.history)
                assert isinstance(hg.frozen(restored.agent), hg.Carried) and recorded[0] in hg.frozen(restored.agent).block
            finally:
                restored.agent.close()
            assert SlashCommandsMixin._cmd_compress(SimpleNamespace(session_manager=manager), "", fork)
            fork.cwd = ""
            sent = _turn(fork.agent, fork.history)
            assert recorded[0] in sent and "Fixture rule B" not in sent and "_unavailable" not in sent
        finally:
            fork.agent.close()
            parent.agent.close()
    finally:
        db.close()


class _FailingReads:
    """A session database whose reads fail (``sqlite3.OperationalError``); writes are ignored."""

    def __init__(self, path):
        self.db_path = path

    def get_session(self, session_id):
        import sqlite3
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_an_unreadable_session_row_is_unavailable_never_a_fresh_decision(home, no_network):
    """A failed read of the session's row cannot tell a new conversation from a continuing one: the agent built for
    the session, and a turn whose read fails with history to continue, carry an explicit notice instead of file B."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    _write_config(home, home_guidance_file=SOURCE)
    failing = _FailingReads(home.hermes / "state.db")
    built = _agent(session_id="unreadable", session_db=failing)
    try:
        assert isinstance(hg.frozen(built), hg.Unavailable) and "OperationalError" in hg.frozen(built).reason
        assert "Fixture rule" not in _prompt(built)
    finally:
        built.close()
    turn = _agent(session_id="unreadable")
    try:
        turn._session_db = failing
        _restore_or_build_system_prompt(turn, None, [{"role": "user", "content": "hello"}])
        assert isinstance(hg.frozen(turn), hg.Unavailable) and "Fixture rule" not in turn._cached_system_prompt
    finally:
        turn.close()


def test_where_nothing_is_configured_a_conversation_without_its_stored_prompt_carries_none(home, tmp_path,
                                                                                           no_network):
    """Nothing configured: a continuing conversation whose stored prompt is missing has nothing it could take up and
    no setting a notice could name. It carries none — no notice, and a block SOUL.md gains is kept out — and records
    it, so configuring a file later does not reach it either."""
    from tests.agent.test_home_guidance import _turn
    _write_config(home)
    soul = tmp_path / "SOUL.md"  # an agent with a database reads SOUL.md from the database's directory
    soul.write_text("Role.\n", encoding="utf-8")
    db = _real_db(tmp_path)
    try:
        agent = _agent(session_id="unconfigured", session_db=db)
        try:
            _turn(agent, [])
            agent._persist_session([{"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "old reply"}], [])
        finally:
            agent.close()
        db.update_system_prompt("unconfigured", None)
        soul.write_text(f"Role.\n\n{SOUL_B}\n", encoding="utf-8")
        resumed = _agent(session_id="unconfigured", session_db=db)
        try:
            history = db.get_messages_as_conversation("unconfigured")
            assert history and hg.frozen(resumed) is None
            sent = _turn(resumed, history)
            assert "workstation_guidance" not in sent and "Role." in sent
            assert hg.persisted_state(sent) is None
            _write_config(home, home_guidance_file=SOURCE)
            resumed._invalidate_system_prompt()
            assert "workstation_guidance" not in _prompt(resumed)
        finally:
            resumed.close()
    finally:
        db.close()


def test_a_new_session_and_one_without_messages_decide_from_the_current_inputs(home, tmp_path, no_network):
    """Controls: a session with no row, and an existing row with neither prompt nor messages (a reset one), are new
    conversations: the first turn takes up the configured file and records it."""
    from tests.agent.test_home_guidance import _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        db.create_session(session_id="empty-row", source="cli")
        for session_id in ("genuinely-new", "empty-row"):
            agent = _agent(session_id=session_id, session_db=db)
            try:
                sent = _turn(agent, [])
                assert sent.count(_expected_block(home.core)) == 1 and isinstance(hg.frozen(agent), hg.Carried)
                assert hg.persisted_state(sent) == hg.frozen(agent)
            finally:
                agent.close()
    finally:
        db.close()


# ── An imported transcript continued here (R4-IMPORT-01) ────────────────────────────────────────────────────────────
# An import is decided by no build here, so its first turn takes up the current inputs. Both producers record the
# import's last message row; a reply in a later row is a Hermes turn, after which losing the stored prompt never
# renews the import's eligibility. Rewrites and compaction insert new rows, so they never hide a Hermes reply.
# v0.23.1's CLI import is an ordinary ``cli`` row written by the gateway's ``session.mutate import`` with no
# ``origin_json`` (resume refuses a row with one); its record is kept beside the row and read through
# ``foreign_import_view``.

FOREIGN = [("user", "foreign question"), ("assistant", "foreign reply")]


def _import(db, producer, tmp_path):
    """Import FOREIGN through the real CLI (``import_foreign_session``, whose ``session.mutate import`` reaches a real
    gateway owner over *db*) or desktop (``import_foreign_history``) producer; returns the new session id."""
    if producer == "cli":
        import asyncio
        from types import SimpleNamespace
        from gateway.session_authority import SessionAuthority
        from gateway.session_contract import SessionActor, SessionRef
        from gateway.session_mutations import mutate_session
        from hermes_cli.foreign_sessions import import_foreign_session
        from hermes_state_runtime import begin_runtime_epoch
        foreign = tmp_path / "claude.jsonl"
        foreign.write_text("".join(json.dumps({"type": role, "message": {
            "role": role, "content": text if role == "user" else [{"type": "text", "text": text}]}}) + "\n"
            for role, text in FOREIGN), encoding="utf-8")
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="p", instance_id="owner",
                                     db=db, epoch=begin_runtime_epoch(db, instance_id="owner"))
        actor = SessionActor("owner", "p", frozenset({"session:read", "session:control", "session:create"}), "cli")

        def call(method, **params):
            from hermes_cli.gateway_client import GatewayClientError
            from hermes_state_runtime import RuntimeStoreError
            try:
                return asyncio.run(mutate_session(authority, actor, SessionRef("p", params["session_id"]), params))
            except (RuntimeStoreError, ValueError) as exc:  # a refusal crosses the wire as a bounded code
                raise GatewayClientError(getattr(exc, "reason", "invalid_params")) from exc
        store = SimpleNamespace(call=call)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("hermes_cli.gateway_session_store.session_store", lambda: store)
            return import_foreign_session("claude", foreign)
    return db.import_foreign_history({"tool": "codex", "path": str(tmp_path / "x.jsonl"), "foreign_session_id": "f-1"},
                                     [{"role": role, "content": text} for role, text in FOREIGN],
                                     title="imported", cwd=None, profile=None)["session_id"]


def _next_turn(db, session_id, entry):
    """The next turn's prompt for *session_id* through *entry*: ``construction`` (an agent built for the session, as
    a cold restore or gateway turn builds it) or ``turn`` (the turn-time read alone, on an agent built without it,
    given the history as a resumed client holds it: without row ids)."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tests.agent.test_home_guidance import _turn
    history = db.get_messages_as_conversation(session_id)
    if entry == "construction":
        agent = _agent(session_id=session_id, session_db=db)
        try:
            return _turn(agent, history), hg.frozen(agent)
        finally:
            agent.close()
    agent = _agent(session_id=session_id)
    try:
        agent._session_db = db
        _restore_or_build_system_prompt(agent, None, history)
        return agent._cached_system_prompt, hg.frozen(agent)
    finally:
        agent.close()


def _continued_in_hermes(home, db, session_id):
    """A first Hermes turn on an import: it takes up the configured file A and persists its reply. Returns A's block."""
    from tests.agent.test_home_guidance import _turn
    block_a = _expected_block(home.core)
    first = _agent(session_id=session_id, session_db=db)
    try:
        history = db.get_messages_as_conversation(session_id)
        assert _turn(first, history).count(block_a) == 1
        first._persist_session(history + [{"role": "user", "content": "Hermes turn"},
                                          {"role": "assistant", "content": "Hermes reply"}], [])
    finally:
        first.close()
    assert db.get_session(session_id)["system_prompt"]
    return block_a


def _imported_from(db, session_id):
    """The session's ``imported_from`` record as home guidance reads it (a CLI import keeps it beside the row)."""
    return json.loads(hg.foreign_import_view(db, db.get_session(session_id))["origin_json"])["imported_from"]


def _as_recorded_before_the_boundary(db, session_id):
    """The session's import provenance as an import from before the boundary was recorded wrote it."""
    from hermes_state_portability import FOREIGN_IMPORT_META_PREFIX, FOREIGN_IMPORT_THROUGH
    origin = {"imported_from": _imported_from(db, session_id)}
    del origin["imported_from"][FOREIGN_IMPORT_THROUGH]
    if db.get_session(session_id)["origin_json"]:
        db._execute_write(lambda conn: conn.execute("UPDATE sessions SET origin_json = ? WHERE id = ?",
                                                    (json.dumps(origin), session_id)))
    else:
        db.set_meta(FOREIGN_IMPORT_META_PREFIX + session_id, json.dumps(origin["imported_from"]))


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("producer", ["cli", "desktop"])
def test_an_imports_first_turn_takes_up_the_current_file(home, tmp_path, no_network, producer, entry):
    """Real import -> SessionDB -> first Hermes turn: the producer records its last message row as the boundary, and
    the first turn, with no reply after it, takes up the current file (B, configured after the import)."""
    from hermes_state_portability import FOREIGN_IMPORT_THROUGH
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id = _import(db, producer, tmp_path)
        row, rows = db.get_session(session_id), db.get_messages(session_id)
        assert _imported_from(db, session_id)[FOREIGN_IMPORT_THROUGH] == max(r["id"] for r in rows)
        assert not row["system_prompt"] and len(rows) == len(FOREIGN)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, entry)
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        db.close()


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("prompt", ["intact", None, ""], ids=["intact", "null", "empty"])
@pytest.mark.parametrize("producer", ["cli", "desktop"])
def test_an_import_continued_in_hermes_keeps_its_decision(home, tmp_path, no_network, producer, prompt, entry):
    """Real import -> first Hermes turn takes up A and persists a reply -> the file becomes B -> the stored prompt is
    kept, or cleared to NULL or empty through the database API. The next turn keeps A, or carries an explicit notice
    without its prompt — never B. A new conversation takes up B."""
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id = _import(db, producer, tmp_path)
        block_a = _continued_in_hermes(home, db, session_id)
        home.core.write_text(CORE_B, encoding="utf-8")
        if prompt != "intact":
            db.update_system_prompt(session_id, prompt)
        sent, decided = _next_turn(db, session_id, entry)
        assert "Fixture rule B" not in sent
        if prompt == "intact":
            assert sent.count(block_a) == 1 and decided == hg.Carried(block_a, "file")
        else:
            assert _unavailable_only(sent) and isinstance(decided, hg.Unavailable) and "stored prompt" in decided.reason
        new, _decided = _next_turn(db, "after-import-new", "construction")
        assert new.count(_expected_block(home.core)) == 1
    finally:
        db.close()


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("change", ["rewrite", "compaction"])
def test_a_rewritten_or_compacted_import_still_shows_its_hermes_turn(home, tmp_path, no_network, change, entry):
    """After a Hermes turn on an import, the database's own transcript writes — a /retry or /undo style rewrite
    (``replace_messages``, which re-inserts every row) or in-place compaction (``archive_and_compact``: a summary and
    the carried Hermes turn as fresh rows) — then prompt loss and file B: the Hermes reply is still after the boundary,
    so the turn carries an explicit notice, never B."""
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id = _import(db, "cli", tmp_path)
        _continued_in_hermes(home, db, session_id)
        kept = [{"role": m["role"], "content": m["content"]} for m in db.get_messages_as_conversation(session_id)]
        if change == "rewrite":
            db.replace_messages(session_id, kept)
        else:
            db.archive_and_compact(session_id, [{"role": "user", "content": "[summary of the imported conversation]"},
                                                *kept[-2:]], tail_count=2)
        assert [m["content"] for m in db.get_messages_as_conversation(session_id)][-1] == "Hermes reply"
        home.core.write_text(CORE_B, encoding="utf-8")
        db.update_system_prompt(session_id, None)
        sent, decided = _next_turn(db, session_id, entry)
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        db.close()


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("history", ["used", "unused", "reply-without-usage"])
def test_an_import_recorded_before_its_boundary(home, tmp_path, no_network, history, entry):
    """An import from before the boundary was recorded cannot split its transcript, so its session row decides: a
    model call made here (``update_token_counts``, the real per-call write) with a reply means a conversation under
    way, which after prompt loss carries an explicit notice, never B; an import never used here takes up the current
    file on its first turn. Residual: a reply persisted here without any recorded model call (its usage write lost
    too) cannot be told from an imported one and decides afresh."""
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id = _import(db, "cli", tmp_path)
        _as_recorded_before_the_boundary(db, session_id)
        if history != "unused":
            _continued_in_hermes(home, db, session_id)
            db.update_system_prompt(session_id, None)
        if history == "used":
            db.update_token_counts(session_id, input_tokens=12, output_tokens=3, api_call_count=1, source="claude-code")
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, entry)
        if history == "used":
            assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
        else:
            assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        db.close()


def _filler(db, count):
    """Other conversations' rows in *db*, so its next message ids are well above 1."""
    db.create_session(session_id="filler", source="cli")
    for i in range(count):
        db.append_message("filler", "user" if i % 2 == 0 else "assistant", f"filler {i}")


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("case", ["used-prompt-lost", "used-prompt-intact", "unused"])
def test_an_import_moved_to_another_store_keeps_its_boundary(home, tmp_path, no_network, case, entry):
    """The supported cross-store move (``export_session_for_move`` -> ``import_moved_session`` ->
    ``delete_moved_session``, as ``hermes sessions repair-profiles`` runs it) renumbers message rows, and the
    boundary with them. A used import moved from a busy store into an empty one, then prompt loss and file B: its
    Hermes reply is still after the boundary, so the turn carries a notice. An unused import moved into a busy store
    still takes up the current file on its first turn. An intact prompt keeps A."""
    from hermes_state import SessionDB
    from hermes_state_portability import FOREIGN_IMPORT_THROUGH
    _write_config(home, home_guidance_file=SOURCE)
    source = _real_db(tmp_path)
    (tmp_path / "target").mkdir()
    target = SessionDB(tmp_path / "target" / "state.db")
    try:
        _filler(source if case != "unused" else target, 40)
        session_id = _import(source, "desktop", tmp_path)
        block_a = _continued_in_hermes(home, source, session_id) if case != "unused" else None
        if case == "used-prompt-lost":
            source.update_system_prompt(session_id, None)
        payload = source.export_session_for_move(session_id)
        assert target.import_moved_session(payload, profile_name="default") == "imported"
        assert source.delete_moved_session(session_id)
        moved_rows = target.get_messages(session_id)
        through = json.loads(target.get_session(session_id)["origin_json"])["imported_from"][FOREIGN_IMPORT_THROUGH]
        assert [r["content"] for r in moved_rows if r["id"] <= through] == [text for _role, text in FOREIGN]
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(target, session_id, entry)
        if case == "used-prompt-lost":
            assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
        elif case == "used-prompt-intact":
            assert sent.count(block_a) == 1 and decided == hg.Carried(block_a, "file")
        else:
            assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        source.close()
        target.close()


def test_import_replies_count_only_after_the_boundary():
    """The classifier itself: an import row counts only replies whose row id is after its boundary (a row without an
    id cannot be placed and does not count); a compression child, which copies its parent's origin, counts all of its
    transcript; a failed-turn notice and a delivery mirror are never a reply."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND
    origin = json.dumps({"imported_from": {"tool": "codex", "path": "/x.jsonl", "through_message_id": 2}})
    imported = [{"role": "user", "content": "q", "_row_id": 1}, {"role": "assistant", "content": "a", "_row_id": 2}]
    later = [{"role": "user", "content": "q2", "_row_id": 7}, {"role": "assistant", "content": "a2", "_row_id": 8}]
    row = {"origin_json": origin}
    assert not hg.continues_without_prompt(row, imported)
    assert hg.continues_without_prompt(row, imported + later)
    assert hg.continues_without_prompt(row, imported + later[1:])  # the user it answers may be an imported one
    assert not hg.continues_without_prompt(row, [{k: v for k, v in m.items() if k != "_row_id"} for m in imported + later])
    assert not hg.continues_without_prompt(row, imported + [later[0], {**later[1], "display_kind": FAILED_TURN_DISPLAY_KIND}])
    assert hg.continues_without_prompt(row, imported + [later[0], {"role": "tool", "content": "ran", "_row_id": 9}])
    assert hg.continues_without_prompt({**row, "parent_session_id": "import-row"}, imported)
    mirrored = {"role": "assistant", "content": "sent", "display_metadata": {"delivery_mirror": "cli"}}
    assert not hg.continues_without_prompt({}, [{"role": "user", "content": "[cron] brief"}, mirrored])
    assert hg.continues_without_prompt({}, [{"role": "user", "content": "q"}, mirrored, {"role": "assistant", "content": "a"}])


# ── A failed turn's notice is not a reply (R4-OPUS-05) ──────────────────────────────────────────────────────────────
# The gateway closes a turn that failed before any agent was built (model or credential resolution) with its own
# display-only notice. No prompt was built for it, so it shows no decision; a real reply, and a tool row a failed
# turn leaves after a call ran, still do.


def _gateway_transcript(home, kind):
    """A session the real gateway store holds after its first request: the user row, then what *kind* leaves —
    ``failed`` (the gateway's real closer, ``_hmwa_close_failed_turn``, with no tool run), ``partial`` (a tool call
    ran, then the same closer), or ``reply`` (a model reply). Returns ``(store, session id)``."""
    import asyncio
    from gateway.config import GatewayConfig
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session import AsyncSessionStore, SessionStore
    from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE
    store = SessionStore(home.hermes / "sessions", GatewayConfig())
    db, session_id = store._db, "gateway-first"
    db.create_session(session_id, source="telegram")
    db.append_message(session_id, "user", "first request")
    if kind == "reply":
        db.append_message(session_id, "assistant", "a real model reply")
    else:
        if kind == "partial":
            db.append_message(session_id, "assistant", None, tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
            db.append_message(session_id, "tool", "ran", tool_call_id="c1", tool_name="terminal")
        notice = PARTIAL_FAILED_TURN_NOTICE if kind == "partial" else FAILED_TURN_NOTICE
        runner = SimpleNamespace(async_session_store=AsyncSessionStore(store))
        asyncio.run(GatewayTurnPersistenceMixin._hmwa_close_failed_turn(runner, session_id, notice))
    return store, session_id


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("kind", ["failed", "partial", "reply"])
def test_a_failed_first_turns_notice_does_not_consume_the_decision(home, no_network, kind, entry):
    """Real gateway closer -> SessionDB -> constructor or turn-time read: after a turn that failed before any agent
    was built, the next real turn takes up the configured file. After a tool ran, or a real reply, the conversation is
    under way and its missing prompt carries an explicit notice."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND
    _write_config(home, home_guidance_file=SOURCE)
    store, session_id = _gateway_transcript(home, kind)
    db = store._db
    try:
        rows = db.get_messages(session_id)
        assert db.get_session(session_id)["system_prompt"] is None
        if kind != "reply":
            assert rows[-1]["role"] == "assistant" and rows[-1]["display_kind"] == FAILED_TURN_DISPLAY_KIND
        sent, decided = _next_turn(db, session_id, entry)
        if kind == "failed":
            assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
        else:
            assert _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        store.close_all_db_handles()


# ── A delivery mirror is not a reply (R5-OPUS-03) ───────────────────────────────────────────────────────────────────
# ``gateway/mirror.py`` writes cron and webhook briefs (user role) and outgoing ``send_message`` messages (assistant role)
# into a destination session that no agent may ever have built for. It records that provenance on the row, so a
# mirrored message never reads as this conversation's reply; real replies still do.

MIRRORS = {"brief": [("user", "[cron job daily] brief", "cron")],
           "outgoing": [("assistant", "Outgoing result from another session", "cli")],
           "brief-then-outgoing": [("user", "[cron job daily] brief", "cron"),
                                   ("assistant", "Outgoing result from another session", "cli")],
           "webhook-then-outgoing": [("user", "[Webhook delivery: alerts]\nbuild failed", "webhook"),
                                     ("assistant", "Outgoing result from another session", "telegram")]}


def _mirror_store(home, monkeypatch):
    """The gateway's real session store in a profile outside ``HOME/.hermes`` (``mirror_to_session`` opens the active
    profile's database through the registry, whose live-system guard stays enabled)."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    home.hermes = home.hermes.parent / "mirror-profile"
    home.hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home.hermes))
    _write_config(home, home_guidance_file=SOURCE)
    return SessionStore(home.hermes / "sessions", GatewayConfig())


def _mirror(session_id, rows):
    """Deliver *rows* into *session_id* through the real producer, as cron, webhook and ``send_message`` call it."""
    from gateway.mirror import mirror_to_session
    for role, text, label in rows:
        assert mirror_to_session("telegram", "123", text, source_label=label, role=role, session_id=session_id)


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("rows", sorted(MIRRORS))
def test_delivery_mirrors_into_a_fresh_session_leave_its_first_turn_to_decide(home, no_network, monkeypatch, rows,
                                                                            entry):
    """Real ``mirror_to_session`` -> SQLite -> the first actual request -> constructor or turn-time read: mirrored
    briefs and outgoing messages, alone or together, show no decision, so the first turn takes up the configured file
    A and records it. The rows replay to the model unchanged; only their provenance is recorded (display metadata)."""
    store = _mirror_store(home, monkeypatch)
    db, session_id = store._db, "mirror-first-turn"
    try:
        db.create_session(session_id, source="telegram")
        _mirror(session_id, MIRRORS[rows])
        stored = db.get_messages_as_conversation(session_id)
        assert [(m["role"], m["content"]) for m in stored] == [(role, text) for role, text, _label in MIRRORS[rows]]
        row = db.get_session(session_id)
        assert row["system_prompt"] is None and not row["api_call_count"]
        db.append_message(session_id, "user", "First actual request to this agent")
        sent, decided = _next_turn(db, session_id, entry)
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
        assert hg.persisted_state(db.get_session(session_id)["system_prompt"]) == decided
        from agent.message_metadata import DELIVERY_MIRROR, PERSISTENCE_ONLY_MESSAGE_FIELDS
        assert [m["display_metadata"] for m in stored] == [{DELIVERY_MIRROR: label} for *_rt, label in MIRRORS[rows]]
        assert "display_metadata" in PERSISTENCE_ONLY_MESSAGE_FIELDS  # stripped from every request
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("reply", ["assistant", "tool"])
def test_delivery_mirrors_never_hide_a_real_reply(home, no_network, monkeypatch, reply, entry):
    """Control: a destination conversation a real turn decided for (A, with an assistant reply or a tool result),
    then a mirrored brief and outgoing message, then prompt loss and file B: its own reply still shows the conversation
    under way, so the next turn carries an explicit notice, never B."""
    from tests.agent.test_home_guidance import _turn
    store = _mirror_store(home, monkeypatch)
    db, session_id = store._db, "mirror-after-reply"
    try:
        db.create_session(session_id, source="telegram")
        first = _agent(session_id=session_id, session_db=db)
        try:
            assert _turn(first, []).count(_expected_block(home.core)) == 1
            turn = [{"role": "user", "content": "question"}]
            if reply == "tool":
                turn += [{"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                         {"role": "tool", "content": "ran", "tool_call_id": "c1", "tool_name": "terminal"}]
            else:
                turn += [{"role": "assistant", "content": "a real reply"}]
            first._persist_session(turn, [])
        finally:
            first.close()
        _mirror(session_id, MIRRORS["brief-then-outgoing"])
        db.update_system_prompt(session_id, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, entry)
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        store.close_all_db_handles()


# ── A legacy edition from a SOUL.md the reader newline-translated (R4-CRLF-01) ──────────────────────────────────────


@pytest.mark.parametrize("later", ["same", "changed"])
def test_a_legacy_crlf_soul_edition_renders_once_on_rebuild(home, tmp_path, no_network, later):
    """A SOUL.md with a CRLF block: the reader delivers it newline-translated, so it is never delivered home guidance
    (a new conversation records none), yet a build from before decisions were recorded carried exactly that text.
    Restoring such a prompt keeps it, and a rebuild with the same SOUL.md shows it once, without a note that it is
    not delivered. If SOUL.md later carries another CRLF block, the rebuild keeps the frozen edition (with the note
    that the new block is not delivered) and never takes the new one up as the edition."""
    from tests.agent.test_home_guidance import _turn
    crlf_a = LEGACY_A.replace("\n", "\r\n")
    soul = [home.hermes / "SOUL.md", tmp_path / "SOUL.md"]  # the database's directory is where the agent reads it
    for path in soul:
        path.write_bytes(("Role.\r\n\r\n" + crlf_a).encode("utf-8"))
    _write_config(home)
    db = _real_db(tmp_path)
    try:
        first = _agent(session_id="crlf", session_db=db)
        try:
            sent = _turn(first, [])
            first._persist_session([{"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "reply"}], [])
        finally:
            first.close()
        assert sent.count("LEGACY SOUL EDITION A") == 1 and hg.persisted_state(sent) is None  # fidelity rule kept
        db.update_system_prompt("crlf", _without_record(sent))
        if later == "changed":
            for path in soul:
                path.write_bytes(("Role.\r\n\r\n" + SOUL_B.replace("\n", "\r\n")).encode("utf-8"))
        resumed = _agent(session_id="crlf", session_db=db)
        try:
            assert hg.frozen(resumed) == hg.Carried(LEGACY_A, "soul")
            _turn(resumed, db.get_messages_as_conversation("crlf"))
            resumed._invalidate_system_prompt()
            rebuilt = _prompt(resumed)
            assert rebuilt.count("LEGACY SOUL EDITION A") == 1 and hg.persisted_state(rebuilt) == hg.frozen(resumed)
            if later == "same":
                assert "is not delivered home guidance" not in rebuilt and "NEW SOUL EDITION B" not in rebuilt
            else:
                assert rebuilt.count("is not delivered home guidance") == 1
        finally:
            resumed.close()
    finally:
        db.close()


# ── A gateway compression rotation continues the conversation (R4-OPUS-06) ──────────────────────────────────────────


@pytest.mark.parametrize("in_place", [False, True], ids=["rotation", "in-place"])
def test_a_gateway_compression_keeps_the_conversations_decision(home, tmp_path, no_network, monkeypatch, in_place):
    """Real gateway session -> a first build records A -> the canonical compression transaction
    (``compress_in_transaction``; a prepared summary, no summarizer) -> the file becomes B -> the next turn on the
    published target. A rotation child keeps the parent's stored prompt in the same transaction, which advances the
    generation once and leaves the parent's prompt and transcript alone, so the turn keeps A. A new conversation
    takes up B."""
    import asyncio
    from hermes_state_mutation_compress import compress_in_transaction
    from hermes_state_mutation_prepared import local_snapshot
    from tests.agent.test_home_guidance import _turn
    from tests.gateway.test_resume_latest_local_cli import _call, _owner
    store, authority, owner = _owner(home.hermes, monkeypatch)
    _write_config(home, home_guidance_file=SOURCE)  # _owner wrote its own config
    db = authority.db
    try:
        created = asyncio.run(_call(owner, "session.create", request_id="rotation", source="cli", cwd=str(tmp_path),
                                    toolsets=[], model="m"))
        session_id = created["session_id"]
        target = db._execute_write(lambda conn: local_snapshot(db, conn, session_id)["target"])
        db.create_session(target, source="cli")
        block_a = _expected_block(home.core)
        decided = _agent(session_id=target, session_db=db)
        try:
            assert _turn(decided, []).count(block_a) == 1
            decided._persist_session([{"role": "user", "content": "question"},
                                      {"role": "assistant", "content": "reply"}], [])
        finally:
            decided.close()
        recorded = db.get_session(target)["system_prompt"]
        assert hg.persisted_state(recorded) == hg.Carried(block_a, "file")
        parent_messages = db.get_messages_as_conversation(target)
        generation = db._execute_write(lambda conn: conn.execute(
            "SELECT runtime_generation FROM sessions WHERE id=?", (session_id,)).fetchone()[0])

        def write(conn):
            prepared = {"snapshot": local_snapshot(db, conn, session_id), "in_place": in_place,
                        "messages": [{"role": "user", "content": "[summary of the conversation]"},
                                     {"role": "assistant", "content": "reply"}]}
            return compress_in_transaction(db, conn, session_id, {}, prepared)
        _affected, receipt = db._execute_write(write)
        child = receipt["target_session_id"]
        assert (child == target) is in_place and receipt["previous_target_session_id"] == target
        assert receipt["execution_generation"] == generation + 1
        assert db.get_session(child)["system_prompt"] == recorded
        if not in_place:
            assert db.get_session(target)["system_prompt"] == recorded
            assert db.get_messages_as_conversation(target) == parent_messages
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, frozen = _next_turn(db, child, "construction")
        assert sent.count(block_a) == 1 and "Fixture rule B" not in sent and frozen == hg.Carried(block_a, "file")
        new, _frozen = _next_turn(db, "after-compression-new", "construction")
        assert new.count(_expected_block(home.core)) == 1
    finally:
        asyncio.run(owner.close())
        store.close_all_db_handles()


# ── Compaction keeps the conversation's replies; a reset in place ends them (R5-COMPACT-01) ─────────────────────────
# A context engine may return any valid compacted list, and the commit guard lets it fold an oversized last reply, so
# the live transcript can hold no reply. The rows compaction summarized away (``active = 0, compacted = 1``) still
# belong to the conversation, so after prompt loss it carries a notice, never a new edition. ACP /reset is the one
# operation that ends a conversation in place: it takes every row out of it, so its next turn still decides afresh.


def _decided_with_a_long_reply(home, db, imported, tmp_path, session_id):
    """A conversation that took up A here and persisted replies, the last one long; returns ``(session id, A)``."""
    if imported:
        session_id = _import(db, "cli" if imported == "cli" else "desktop", tmp_path)
    else:
        db.create_session(session_id, source="cli")
    block_a = _continued_in_hermes(home, db, session_id)
    db.append_message(session_id, "user", "give the long result")
    db.append_message(session_id, "assistant", "oversized answer " * 1000)
    return session_id, block_a


def _replies(db, session_id, **read):
    return [m for m in db.get_messages_as_conversation(session_id, **read) if m["role"] == "assistant"]


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("prompt", ["intact", "lost"])
@pytest.mark.parametrize("imported", [None, "cli"], ids=["native", "imported"])
def test_a_compaction_that_keeps_no_reply_live_still_continues_the_conversation(home, tmp_path, no_network, imported,
                                                                             prompt, entry):
    """Real commit guard (``_ensure_compressed_keeps_last_assistant_reply`` accepts folding the dominating last reply)
    -> ``archive_and_compact`` of a reply-less compacted list -> prompt kept or lost -> file B -> constructor or
    turn-time read: the summarized-away replies still show the conversation, so it keeps A once, or carries an explicit
    notice without its prompt — never B."""
    from agent.conversation_compression_reply_anchor import _ensure_compressed_keeps_last_assistant_reply
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id, block_a = _decided_with_a_long_reply(home, db, imported, tmp_path, "compacted")
        compacted = [{"role": "user", "content": "[summary] " + "retained earlier facts " * 60},
                     {"role": "user", "content": "continue the same task"}]
        assert _ensure_compressed_keeps_last_assistant_reply(
            db.get_messages_as_conversation(session_id), compacted, session_id=session_id) is None
        db.archive_and_compact(session_id, compacted, tail_count=1)
        assert not _replies(db, session_id) and _replies(db, session_id, include_compacted=True)
        if prompt == "lost":
            db.update_system_prompt(session_id, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, entry)
        assert "Fixture rule B" not in sent
        if prompt == "lost":
            assert _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
        else:
            assert sent.count(block_a) == 1 and decided == hg.Carried(block_a, "file")
    finally:
        db.close()


@pytest.mark.parametrize("prompt", ["intact", "lost"])
@pytest.mark.parametrize("imported", [None, "desktop"], ids=["native", "imported"])
def test_a_committed_reply_less_compression_then_a_cold_restore_continues(home, tmp_path, no_network, imported, prompt):
    """The real compression facade (``_compress_context``: commit fence, user/reply and anti-growth guards, durable
    transaction, prompt rebuild), with only the engine's returned list substituted by a valid reply-less one -> prompt
    kept or lost -> file B -> cold restore: A once, or an explicit notice — never B."""
    from unittest.mock import MagicMock
    from agent.conversation_compression import CompressionCommitFence
    from tests.agent.test_home_guidance import _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        session_id, block_a = _decided_with_a_long_reply(home, db, imported, tmp_path, "full-compress")
        live = db.get_messages_as_conversation(session_id, include_row_ids=True)
        live.append({"role": "user", "content": "continue the same task"})
        agent = _agent(session_id=session_id, session_db=db)
        try:
            assert _turn(agent, live).count(block_a) == 1
            engine = agent.context_compressor
            engine.compress = MagicMock(return_value=[
                {"role": "user", "content": "[Recent Summary (d0, node 490)]\n\n" + "retained earlier facts " * 60},
                {"role": "user", "content": "continue the same task"}])
            engine.compression_count, engine.last_prompt_tokens, engine.last_completion_tokens = 1, 0, 0
            engine._last_summary_error, engine._last_compress_aborted, engine._last_summary_auth_failure = None, False, False
            engine._last_aux_model_failure_model = engine._last_aux_model_failure_error = None
            agent._persist_user_message_idx = len(live) - 1
            agent._compress_context(live, "", approx_tokens=120_000, commit_fence=CompressionCommitFence())
            assert engine.compress.called
        finally:
            agent.close()
        assert not _replies(db, session_id) and _replies(db, session_id, include_compacted=True)
        if prompt == "lost":
            db.update_system_prompt(session_id, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, "construction")
        assert "Fixture rule B" not in sent
        if prompt == "lost":
            assert _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
        else:
            assert sent.count(block_a) == 1
    finally:
        db.close()


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("owner", ["live", "restored"])
def test_acp_reset_ends_the_conversation_rows_but_not_the_session(home, tmp_path, no_network, owner, compacted,
                                                                   reconnect):
    """The real ACP ``/reset`` on the live agent (it ran and persisted the turn, so it owns persistence and the
    manager's save leaves the rows alone) or on a cold-restored one, optionally after an in-place compaction: no row
    remains in the conversation — none live, none summarized away — though all stay on disk; the prompt is gone and the
    usage counters are kept. The next turn, on the same agent or after a reconnect, decides from the current file B."""
    from acp_adapter.commands import SlashCommandsMixin
    from acp_adapter.session import SessionManager
    from tests.agent.test_home_guidance import _guidance, _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    session_id = "acp-live"

    def factory():
        created = _agent(platform="acp", session_id=session_id, session_db=db)
        created.compression_enabled = False
        return created
    try:
        agent = factory()
        assert _turn(agent, []).count(_expected_block(home.core)) == 1
        agent._persist_session([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}], [])
        db.update_token_counts(session_id, input_tokens=12, output_tokens=3, api_call_count=1, source="claude-code")
        if compacted:
            db.archive_and_compact(session_id, [{"role": "user", "content": "[summary] earlier"}])
        manager = SessionManager(db=db, agent_factory=factory)
        if owner == "live":
            assert agent._session_db_created
            state = SimpleNamespace(agent=agent, session_id=session_id, cwd="", model=None,
                                    history=db.get_messages_as_conversation(session_id))
            manager._sessions[session_id] = state
        else:
            agent.close()
            state = manager.get_session(session_id)
        on_disk = len(db.get_messages(session_id, include_inactive=True))
        usage = {k: db.get_session(session_id)[k] for k in ("api_call_count", "input_tokens", "output_tokens")}
        assert SlashCommandsMixin._cmd_reset(SimpleNamespace(session_manager=manager), "", state) \
            == "Conversation history cleared."
        assert db.get_messages_as_conversation(session_id, include_compacted=True) == [] and state.history == []
        assert len(db.get_messages(session_id, include_inactive=True)) == on_disk
        row = db.get_session(session_id)
        assert row["system_prompt"] is None and row["message_count"] == 0
        assert {k: row[k] for k in usage} == usage == {"api_call_count": 1, "input_tokens": 12, "output_tokens": 3}
        home.core.write_text(CORE_B, encoding="utf-8")
        if reconnect:
            state.agent.close()
            state = SessionManager(db=db, agent_factory=factory).get_session(session_id)
            assert state.history == [] and hg.frozen(state.agent) is None
        sent = _guidance(_turn(state.agent, state.history))
        assert len(sent) == 1 and "Fixture rule B" in sent[0]
        state.agent.close()
    finally:
        db.close()


# ── An import's boundary binds to its own rows (R5-OPUS-02) ─────────────────────────────────────────────────────────
# v0.23.1's CLI import is one gateway ``session.mutate import`` write: the session, every imported row and the boundary
# commit together, so no client can resume the session (and write a Hermes turn) before its boundary is recorded. The
# desktop producer adopts or mints its snapshot in one transaction too (``import_foreign_history``).


@pytest.mark.parametrize("entry", ["construction", "turn"])
def test_a_hermes_turn_after_a_cli_import_is_never_inside_its_boundary(home, tmp_path, no_network, monkeypatch, entry):
    """Real CLI import through a real gateway owner: while the boundary is recorded, the session is not yet visible to
    another connection (one transaction), and the boundary is the import's last row. A Hermes turn after it therefore
    always counts: after prompt loss and file B the next turn carries a notice."""
    import sqlite3
    from contextlib import closing
    import hermes_state_portability as portability
    from hermes_state_portability import FOREIGN_IMPORT_THROUGH
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        seen = []
        record = portability.record_foreign_import_meta

        def record_foreign_import_meta(conn, session_id, origin):
            with closing(sqlite3.connect(db.db_path)) as other:  # another client's view mid-import
                seen.append((conn.in_transaction,
                             other.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)).fetchone()[0]))
            return record(conn, session_id, origin)
        monkeypatch.setattr(portability, "record_foreign_import_meta", record_foreign_import_meta)
        session_id = _import(db, "cli", tmp_path)
        assert seen == [(True, 0)]  # recorded inside the import's own uncommitted transaction
        _continued_in_hermes(home, db, session_id)
        through = _imported_from(db, session_id)[FOREIGN_IMPORT_THROUGH]
        rows = db.get_messages(session_id)
        imported = [r["id"] for r in rows if (r["role"], r["content"]) in FOREIGN]
        hermes = [r["id"] for r in rows if r["id"] not in imported]  # the turn's own user row too
        assert len(imported) == len(FOREIGN) and "Hermes reply" in [r["content"] for r in rows]
        assert through == imported[-1] and all(h > through for h in hermes)
        db.update_system_prompt(session_id, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, session_id, entry)
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        db.close()


@pytest.mark.parametrize("producer", ["cli", "desktop"])
def test_a_reimport_keeps_each_imports_own_boundary(home, tmp_path, no_network, producer):
    """Control: importing the same foreign session again. The CLI imports a new copy whose boundary is its own last
    row, and its first turn takes up the current file; the desktop finds the earlier import and leaves it alone, so the
    Hermes turn written on it in between stays after its boundary."""
    from hermes_state_portability import FOREIGN_IMPORT_THROUGH

    def through(session_id):
        return _imported_from(db, session_id)[FOREIGN_IMPORT_THROUGH]
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    try:
        first = _import(db, producer, tmp_path)
        first_through = through(first)
        _continued_in_hermes(home, db, first)
        home.core.write_text(CORE_B, encoding="utf-8")
        again = _import(db, producer, tmp_path)
        assert through(first) == first_through
        if producer == "desktop":
            assert again == first
            db.update_system_prompt(first, None)
            sent, decided = _next_turn(db, first, "construction")
            assert "Fixture rule B" not in sent and isinstance(decided, hg.Unavailable)
        else:
            assert again != first and through(again) == max(r["id"] for r in db.get_messages(again))
            sent, decided = _next_turn(db, again, "construction")
            assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        db.close()


# ── A fold keeps the delivery mark only while it stands for mirrors alone (R6-MIRROR-01) ────────────────────────────
# Live replay merges consecutive assistant rows (``repair_message_sequence``): a real reply that lands right after an
# outgoing mirror folds into the mirror's row. The survivor then stands for the real reply too, so it must not keep a
# mark that says its whole row was delivered from elsewhere; otherwise compaction, which carries the merged survivor
# and retires the originals, would leave no reply behind.

USER_Q = {"role": "user", "content": "question"}
MIRROR_ROW = {"role": "assistant", "content": "Outgoing result from another session",
              "display_metadata": {"delivery_mirror": "cli"}}
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
FOLDS = {"mirror+mirror": ([MIRROR_ROW, {**MIRROR_ROW, "content": "Second outgoing message"}], True),
         "mirror+text": ([MIRROR_ROW, {"role": "assistant", "content": "the real reply"}], False),
         "mirror+tool-call": ([MIRROR_ROW, {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
                               {"role": "tool", "content": "ran", "tool_call_id": "c1"}], False),
         "mirror+empty-reply": ([MIRROR_ROW, {"role": "assistant", "content": ""}], False),
         "text+mirror": ([{"role": "assistant", "content": "the real reply"}, MIRROR_ROW], False),
         "mirror+failed-notice": ([MIRROR_ROW, {"role": "assistant", "content": "Your request was not processed. Send "
                                                "it again if you still want me to carry it out.",
                                                "display_kind": "failed_turn"}], True),
         "mirror+legacy-failed-notice": ([MIRROR_ROW, {"role": "assistant", "content": "Your request was not "
                                                       "processed. Send it again if you still want me to carry it "
                                                       "out."}], True)}


@pytest.mark.parametrize("fold", sorted(FOLDS))
def test_a_fold_keeps_the_delivery_mark_only_while_every_row_is_a_mirror(fold):
    """The real alternation repair over consecutive assistant rows: the survivor keeps the delivery mark only while
    everything it absorbed was written without a turn too (another mirror, a typed or legacy failed-turn notice); with
    a real reply (text, tool calls, even an empty one) it is a reply. A copy made before the fold keeps its own
    metadata."""
    from agent.agent_runtime_helpers import repair_message_sequence
    rows, mirror_only = FOLDS[fold]
    messages = [dict(USER_Q)] + [{**row} for row in rows]
    earlier_copy = {**messages[1]}
    assert repair_message_sequence(None, messages) >= 1
    survivor = messages[1]
    assert survivor["role"] == "assistant" and [m["role"] for m in messages[:2]] == ["user", "assistant"]
    assert ("delivery_mirror" in (survivor.get("display_metadata") or {})) is mirror_only
    assert hg._replies_to_a_user(messages) is not mirror_only
    assert earlier_copy.get("display_metadata") == rows[0].get("display_metadata")


@pytest.mark.parametrize("later", ["mirror", "real-reply"])
def test_a_failed_notice_survivor_keeps_its_notice_type(later):
    """A typed failed-turn notice that absorbs a later assistant row keeps its ``display_kind`` (the fold keeps the
    survivor's display fields). With a mirror after it, nothing in it is a reply. With a real reply after it (a closer
    can write the notice while the abandoned turn still persists its partial: R7-FAILED-01,
    ``tests/agent/test_home_guidance_failed_provenance.py``) it records that it stands for that reply and is one."""
    from agent.agent_runtime_helpers import repair_message_sequence
    from agent.message_metadata import STANDS_FOR_TURN_ROW
    notice = {"role": "assistant", "content": "Your request was not processed.", "display_kind": "failed_turn"}
    row = MIRROR_ROW if later == "mirror" else {"role": "assistant", "content": "a later assistant row"}
    messages = [dict(USER_Q), dict(notice), {**row}]
    repair_message_sequence(None, messages)
    assert len(messages) == 2 and messages[1]["display_kind"] == "failed_turn"
    assert hg._replies_to_a_user(messages) is (later == "real-reply")
    assert (STANDS_FOR_TURN_ROW in (messages[1].get("display_metadata") or {})) is (later == "real-reply")


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("carried", ["raw", "compacted"])
@pytest.mark.parametrize("closer", ["gateway-closer", "unattended-notice"])
def test_a_mirror_then_a_failed_first_turn_leaves_the_decision_open(home, no_network, monkeypatch, closer, carried,
                                                                  entry):
    """The first request is stored, an outgoing mirror lands (real producer), then the turn fails before any agent is
    built. The real gateway closer (``_hmwa_close_failed_turn``) writes no notice: the mirror already closed the
    durable tail. The unattended owner's closer (``close_discarded_turn`` with its notice, written even on a closed
    tail) appends one after the mirror, and live replay folds it into the mirror. Raw, or after a compaction carrying
    the live-replay tail, nothing here is a reply: the gateway's turn history shows no conversation under way and the
    next real turn takes up the configured file."""
    import asyncio
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session import AsyncSessionStore
    from gateway.session_results import close_discarded_turn
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE, FAILED_TURN_NOTICE
    store = _mirror_store(home, monkeypatch)
    db = store._db
    session_id = "mirror-then-failed"
    try:
        db.create_session(session_id, source="telegram")
        db.append_message(session_id, "user", "first request")
        _mirror(session_id, MIRRORS["outgoing"])
        if closer == "gateway-closer":
            runner = SimpleNamespace(async_session_store=AsyncSessionStore(store))
            asyncio.run(GatewayTurnPersistenceMixin._hmwa_close_failed_turn(runner, session_id, FAILED_TURN_NOTICE))
            assert db.get_messages(session_id)[-1]["display_metadata"] == {"delivery_mirror": "cli"}  # no notice
        else:
            assert db._execute_write(lambda conn: close_discarded_turn(
                db, conn, {"target_session_id": session_id}, AUTOMATION_INTERRUPTED_NOTICE))
            assert db.get_messages(session_id)[-1]["display_kind"] == "failed_turn"
        live = store.load_transcript(session_id)
        assert [m["role"] for m in live] == ["user", "assistant"]
        assert not hg.continues_without_prompt(db.get_session(session_id), live)  # the gateway's turn history
        if carried == "compacted":
            db.archive_and_compact(session_id, [{"role": "user", "content": "[summary]"}, live[-1]], tail_count=1)
        assert db.get_session(session_id)["system_prompt"] is None
        sent, decided = _next_turn(db, session_id, entry)
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        store.close_all_db_handles()


def _decided_with_a_mirror_before_its_reply(home, monkeypatch, session_id, *, big=False):
    """A real first turn decides A and persists its user row; an outgoing ``send_message`` mirror lands (real
    producer); then the turn's real reply is persisted after it. On disk: Q, M (mirror), R."""
    from tests.agent.test_home_guidance import _turn
    store = _mirror_store(home, monkeypatch)
    db = store._db
    db.create_session(session_id, source="telegram")
    first = _agent(session_id=session_id, session_db=db)
    try:
        assert _turn(first, []).count(_expected_block(home.core)) == 1
        turn = [{"role": "user", "content": "question " + ("long pasted log line " * 3000 if big else "")}]
        first._persist_session(turn, [])
        _mirror(session_id, MIRRORS["outgoing"])
        turn.append({"role": "assistant", "content": "the real reply"})
        first._persist_session(turn, [])
    finally:
        first.close()
    return store, db


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("carried", ["raw", "tail-1", "tail-2"])
def test_a_real_reply_folded_into_a_mirror_survives_compaction(home, no_network, monkeypatch, carried, entry):
    """Q, M, R (real mirror producer) -> the live-replay read (``repair_alternation``, row ids) folds R into M ->
    in-place compaction carrying that merged tail (``archive_and_compact``, which retires the carried originals) ->
    prompt loss -> file B: the carried survivor is the conversation's reply, so the next turn carries an explicit
    notice, never B. ``raw`` (no compaction) is the control."""
    store, db = _decided_with_a_mirror_before_its_reply(home, monkeypatch, "mirror-then-reply")
    try:
        if carried != "raw":
            tail = int(carried[-1])
            live = db.get_messages_as_conversation("mirror-then-reply", repair_alternation=True, include_row_ids=True)
            assert [m["role"] for m in live] == ["user", "assistant"]
            compacted = [{"role": "user", "content": "[summary] earlier facts"}, *live[-tail:]]
            if tail == 1:
                compacted.insert(1, {"role": "user", "content": "continue"})
            db.archive_and_compact("mirror-then-reply", compacted, tail_count=tail)
            assert not [m for m in db.get_messages("mirror-then-reply") if m["content"] == "the real reply"]
        db.update_system_prompt("mirror-then-reply", None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, "mirror-then-reply", entry)
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("loader", ["turn-lease", "gateway"])
def test_a_committed_compression_of_a_folded_mirror_keeps_the_conversation(home, no_network, monkeypatch, loader):
    """The real compression facade (``_compress_context``: commit fence, guards, durable commit) over the live-replay
    list in which R folded into M — read as the turn lease reads it (row ids) or as the gateway reads it
    (``load_transcript``) — with only the engine's returned list substituted, tagging its tail as the compressor does.
    Then prompt loss and file B: an explicit notice, never B."""
    from unittest.mock import MagicMock
    from agent.conversation_compression import CompressionCommitFence
    from tests.agent.test_home_guidance import _turn
    store, db = _decided_with_a_mirror_before_its_reply(home, monkeypatch, "mirror-facade", big=True)
    try:
        db.append_message("mirror-facade", "user", "continue the same task")
        live = (store.load_transcript("mirror-facade") if loader == "gateway" else
                db.get_messages_as_conversation("mirror-facade", repair_alternation=True, include_row_ids=True))
        agent = _agent(session_id="mirror-facade", session_db=db)
        try:
            _turn(agent, live)
            engine = agent.context_compressor
            engine.compress = MagicMock(return_value=[
                {"role": "user", "content": "[Recent Summary (d0, node 490)]\n\n" + "retained earlier facts " * 60},
                {**live[-2], "_compaction_tail": True}, {**live[-1], "_compaction_tail": True}])
            engine.compression_count, engine.last_prompt_tokens, engine.last_completion_tokens = 1, 0, 0
            engine._last_summary_error, engine._last_compress_aborted, engine._last_summary_auth_failure = None, False, False
            engine._last_aux_model_failure_model = engine._last_aux_model_failure_error = None
            agent._persist_user_message_idx = len(live) - 1
            agent._compress_context(live, "", approx_tokens=120_000, commit_fence=CompressionCommitFence())
            assert engine.compress.called
        finally:
            agent.close()
        tip = db.get_compression_tip("mirror-facade") or "mirror-facade"
        db.update_system_prompt(tip, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, tip, "construction")
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("entry", ["construction", "turn"])
def test_mirrors_alone_stay_mirrors_through_a_compaction(home, no_network, monkeypatch, entry):
    """Control: a brief and two outgoing mirrors (no turn) -> the live-replay read folds the two mirrors -> compaction
    carries that survivor -> the first actual request takes up the configured file A."""
    store = _mirror_store(home, monkeypatch)
    db = store._db
    try:
        db.create_session("mirrors-only", source="telegram")
        _mirror("mirrors-only", MIRRORS["brief-then-outgoing"] + [("assistant", "Second outgoing message", "cli")])
        live = db.get_messages_as_conversation("mirrors-only", repair_alternation=True, include_row_ids=True)
        assert [m["role"] for m in live] == ["user", "assistant"]
        db.archive_and_compact("mirrors-only", [{"role": "user", "content": "[summary]"}, live[-1]], tail_count=1)
        db.append_message("mirrors-only", "user", "First actual request to this agent")
        sent, decided = _next_turn(db, "mirrors-only", entry)
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        store.close_all_db_handles()


def test_a_folded_reply_reaches_no_provider_with_its_metadata(home, no_network, monkeypatch):
    """The request the next turn builds from the folded live-replay list (``build_api_messages``) carries the merged
    text but no ``display_metadata`` (persistence-only) and so no delivery mark."""
    from agent.turn_context import build_api_messages
    from tests.agent.test_home_guidance import _turn
    store, db = _decided_with_a_mirror_before_its_reply(home, monkeypatch, "mirror-wire")
    try:
        db.append_message("mirror-wire", "user", "next question")
        history = store.load_transcript("mirror-wire")
        agent = _agent(session_id="mirror-wire", session_db=db)
        try:
            prompt = _turn(agent, history)
            api, _system = build_api_messages(agent, history, current_turn_user_idx=len(history) - 1,
                                              ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
                                              active_system_prompt=prompt)
            wire = json.dumps(api, default=str)
            assert "the real reply" in wire and "display_metadata" not in wire and "delivery_mirror" not in wire
        finally:
            agent.close()
    finally:
        store.close_all_db_handles()


# ── A reset in place ends every compression segment of the conversation (R6-RESET-02) ───────────────────────────────


def _rotate_by_publication(db, parent, child, agent):
    """A compression rotation through the database's real publication primitive; the child keeps the parent's
    prompt and the live agent moves to it."""
    db.publish_compression_child(parent_session_id=parent, child_session_id=child, source="acp",
                                 messages=[{"role": "user", "content": f"[summary of {parent}]"},
                                           {"role": "assistant", "content": f"tail reply {parent}"}],
                                 system_prompt=db.get_session(parent)["system_prompt"], require_compression_lease=False)
    agent.session_id = child


@pytest.mark.parametrize("rotations", [0, 1, 2, 3])
def test_acp_reset_after_rotations_ends_the_whole_lineage(home, tmp_path, no_network, rotations):
    """A live ACP agent decides A, persists a turn, rotates 0-3 times (``compression.in_place: false``), then the real
    ``/reset``: no segment keeps its prompt or a row in the conversation; the display projection
    (``get_resume_conversations``) and the model projection are empty; every row stays on disk and every segment's
    usage is kept; a neighboring session is untouched; the next turn takes up B."""
    from acp_adapter.commands import SlashCommandsMixin
    from acp_adapter.session import SessionManager
    from tests.agent.test_home_guidance import _guidance, _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    root = "acp-rotated"

    def factory():
        created = _agent(platform="acp", session_id=root, session_db=db)
        created.compression_enabled = False
        return created
    try:
        neighbor = _agent(session_id="neighbor", session_db=db)
        try:
            _turn(neighbor, [])
            neighbor._persist_session([{"role": "user", "content": "n"}, {"role": "assistant", "content": "nr"}], [])
        finally:
            neighbor.close()
        neighbor_before = (db.get_session("neighbor")["system_prompt"], db.get_messages_as_conversation("neighbor"))
        agent = factory()
        assert _turn(agent, []).count(_expected_block(home.core)) == 1
        agent._persist_session([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}], [])
        segments = [root]
        for i in range(rotations):
            _rotate_by_publication(db, segments[-1], f"{root}-{i + 1}", agent)
            segments.append(f"{root}-{i + 1}")
        for sid in segments:
            db.update_token_counts(sid, input_tokens=5, output_tokens=2, api_call_count=1, source="claude-code")
        assert db.get_compression_lineage(agent.session_id) == segments
        on_disk = {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments}
        manager = SessionManager(db=db, agent_factory=factory)
        state = SimpleNamespace(agent=agent, session_id=root, cwd="", model=None,
                                history=db.get_messages_as_conversation(root))
        manager._sessions[root] = state
        assert SlashCommandsMixin._cmd_reset(SimpleNamespace(session_manager=manager), "", state) \
            == "Conversation history cleared."
        for sid in segments:
            row = db.get_session(sid)
            assert row["system_prompt"] is None and db.get_messages_as_conversation(sid, include_compacted=True) == []
            assert len(db.get_messages(sid, include_inactive=True)) == on_disk[sid]
            assert (row["api_call_count"], row["input_tokens"], row["output_tokens"]) == (1, 5, 2)
        model, display = db.get_resume_conversations(db.resolve_resume_session_id(root))
        assert model == [] and display == []
        assert (db.get_session("neighbor")["system_prompt"], db.get_messages_as_conversation("neighbor")) \
            == neighbor_before
        home.core.write_text(CORE_B, encoding="utf-8")
        sent = _guidance(_turn(agent, state.history))
        assert len(sent) == 1 and "Fixture rule B" in sent[0]
        agent.close()
    finally:
        db.close()


def test_acp_reset_reports_a_segment_that_could_not_be_ended(home, tmp_path, no_network, monkeypatch, caplog):
    """After two rotations, ending the root segment fails: the real ``/reset`` reports its reset-failed diagnostic
    (never success) and logs it; the agent's own segment, ended first, holds no conversation."""
    import logging
    from acp_adapter.commands import SlashCommandsMixin
    from acp_adapter.session import SessionManager
    from tests.agent.test_home_guidance import _turn
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)
    root = "acp-partial"

    def factory():
        return _agent(platform="acp", session_id=root, session_db=db)
    try:
        agent = factory()
        _turn(agent, [])
        agent._persist_session([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}], [])
        _rotate_by_publication(db, root, f"{root}-1", agent)
        _rotate_by_publication(db, f"{root}-1", f"{root}-2", agent)
        end = db.end_conversation_in_place

        def end_or_refuse(session_id):
            if session_id == root:
                raise RuntimeError("database is locked")
            end(session_id)
        monkeypatch.setattr(db, "end_conversation_in_place", end_or_refuse)
        manager = SessionManager(db=db, agent_factory=factory)
        state = SimpleNamespace(agent=agent, session_id=root, cwd="", model=None, history=[])
        manager._sessions[root] = state
        with caplog.at_level(logging.WARNING):
            result = SlashCommandsMixin._cmd_reset(SimpleNamespace(session_manager=manager), "", state)
        assert result == "Conversation history was not cleared: agent session state reset failed; see logs."
        assert "database is locked" in caplog.text
        assert db.get_messages_as_conversation(f"{root}-2", include_compacted=True) == []
        assert db.get_session(root)["system_prompt"]
        agent.close()
    finally:
        db.close()
