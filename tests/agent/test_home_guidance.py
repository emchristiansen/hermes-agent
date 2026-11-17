"""``agent.home_guidance_file``: the user's own home instruction file in the stable prompt tier.

Real ``AIAgent`` prompt builds against a temporary ``HERMES_HOME``/``HOME`` with owned fixture cores. The contract: at most one
complete edition per prompt, byte-exact with source/resolved/sha256 metadata, an explicit notice instead of partial text, frozen
per agent across compression and resume, and no change to safe-mode, ignore-rules, ``skip_context_files`` or ``skip_memory``."""

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from agent import home_guidance as hg

CORE_A = "# Home core A\n\nFixture rule A: cite the original artifact.\n"
CORE_B = "# Home core B\n\nFixture rule B: a later edition.\n"
SOURCE = "~/.codex/AGENTS.md"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """``~/.codex/AGENTS.md`` is a link into a store directory, as Home Manager installs it."""
    os_home = tmp_path / "os-home"
    hermes = os_home / ".hermes"
    hermes.mkdir(parents=True)
    store = tmp_path / "store"
    store.mkdir()
    core = store / "CORE.md"
    core.write_bytes(CORE_A.encode("utf-8"))
    (os_home / ".codex").mkdir()
    (os_home / ".codex" / "AGENTS.md").symlink_to(core)
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    _no_config_backup_copies(monkeypatch)
    return SimpleNamespace(hermes=hermes, core=core, os_home=os_home)


def _no_config_backup_copies(monkeypatch):
    """Every distinct config.yaml these tests write is copied to ``backups/config`` on its first parse
    (``hermes_cli/config.py::_load_config_impl``); on the runner's ZFS scratch root each copy takes seconds. The copy is
    a write-side safety net, not part of reading the config, so it is skipped; the config readers stay real."""
    from hermes_cli import config_backups
    monkeypatch.setattr(config_backups, "backup_config", lambda *a, **k: None)


def _config(hermes, **settings):
    cfg = {"model": {"default": "anthropic/claude-sonnet-4.6"}, "agent": {"environment_probe": False}}
    for key, value in settings.items():  # ``section__name`` for a nested setting, a bare name for a top-level one
        section, nested, name = key.partition("__")
        if nested:
            cfg.setdefault(section, {})[name] = value
        else:
            cfg[key] = value
    (hermes / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg


def _agent(**kw):
    from run_agent import AIAgent

    base = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter", api_mode="chat_completions",
                model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True, skip_memory=True,
                save_trajectories=False, enabled_toolsets=["file"])
    base.update(kw)
    return AIAgent(**base)


def _expected_block(core, source=SOURCE):
    data = core.read_bytes()
    return (f'<workstation_guidance source="{source}" resolved="{os.path.realpath(core)}" '
            f'sha256="{hashlib.sha256(data).hexdigest()}">\n{data.decode("utf-8")}</workstation_guidance>')


def _prompt(agent):
    agent._cached_system_prompt = agent._build_system_prompt()
    return agent._cached_system_prompt


def _identity(agent):
    from agent.system_prompt import _identity_parts
    return _identity_parts(agent, None)[0][0]


def test_default_home_gets_the_exact_core_once_right_after_identity(home):
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        block = _expected_block(home.core)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(block) == 1
        from agent.system_prompt import build_system_prompt_parts
        assert build_system_prompt_parts(agent)["stable"].startswith(_identity(agent).strip() + "\n\n" + block)
        assert f'resolved="{home.core}"' in block  # the real path on this host, not the configured link
    finally:
        agent.close()


def test_a_soul_without_a_block_keeps_its_text_and_gets_only_the_core(home):
    (home.hermes / "SOUL.md").write_text("I am the default profile.\n", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        from agent.system_prompt import build_system_prompt_parts
        stable = build_system_prompt_parts(agent)["stable"]
        assert stable.startswith("I am the default profile.\n\n" + _expected_block(home.core) + "\n\n")
    finally:
        agent.close()


def test_disabled_by_default(home):
    _config(home.hermes)
    agent = _agent()
    try:
        assert "workstation_guidance" not in _prompt(agent)
    finally:
        agent.close()


SOUL_BLOCKS = {
    "source only": '<workstation_guidance source="~/.codex/AGENTS.md">\nSOUL edition\n</workstation_guidance>',
    "no attributes": "<workstation_guidance>\nSOUL edition\n</workstation_guidance>",
    "full S4 metadata": ('<workstation_guidance source="~/.codex/AGENTS.md" resolved="/nix/store/x-core/AGENTS.md" '
                         'sha256="' + "0" * 64 + '">\nSOUL edition\n</workstation_guidance>'),
}


@pytest.mark.parametrize("case", sorted(SOUL_BLOCKS))
def test_a_soul_with_one_complete_block_owns_delivery(home, case):
    soul_block = SOUL_BLOCKS[case]
    (home.hermes / "SOUL.md").write_text(f"I am a swarm role.\n\n{soul_block}\n", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert prompt.count("<workstation_guidance") == 1 and soul_block in prompt
        assert "Fixture rule A" not in prompt  # no second edition
    finally:
        agent.close()


MALFORMED = {
    "opening only": '<workstation_guidance source="x">\nlegacy text',
    "closing only": "legacy text\n</workstation_guidance>",
    "two blocks": '<workstation_guidance a="1">\none\n</workstation_guidance>\n<workstation_guidance a="2">\ntwo\n'
                  "</workstation_guidance>",
    "nested": '<workstation_guidance a="1">\n<workstation_guidance a="2">\ninner\n</workstation_guidance>\n'
              "</workstation_guidance>",
    "unterminated attribute": '<workstation_guidance source="unterminated>\nOLD\n</workstation_guidance>',
    "unquoted attribute": "<workstation_guidance source=x>\nOLD\n</workstation_guidance>",
    "empty": "<workstation_guidance>\n</workstation_guidance>",
    "whitespace only": '<workstation_guidance source="s">\n  \n\t\n</workstation_guidance>',
    "closing before opening": '</workstation_guidance>\nOLD\n<workstation_guidance source="s">',
    "incomplete tag at the end": "<workstation_guidance",
    "self-closing tag": "<workstation_guidance/>",
    "closing tag with a space": "<workstation_guidance>\nOLD\n</workstation_guidance >",
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_soul_markers_never_count_as_delivered(home, case):
    (home.hermes / "SOUL.md").write_text(f"I am a role.\n\n{MALFORMED[case]}\n", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        block = _expected_block(home.core)
        assert prompt.count(block) == 1
        note = prompt.index("which is not delivered home guidance")
        assert note < prompt.index(block) and hg._SUPERSEDES in prompt
    finally:
        agent.close()


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_soul_markers_add_nothing_when_disabled(home, case):
    (home.hermes / "SOUL.md").write_text(f"I am a role.\n\n{MALFORMED[case]}\n", encoding="utf-8")
    _config(home.hermes)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert "not delivered home guidance" not in prompt and "workstation_guidance_unavailable" not in prompt
    finally:
        agent.close()


def _missing(home):
    home.core.unlink()
    return SOURCE, "could not be read"


def _not_utf8(home):
    home.core.write_bytes(b"# core\n\xff\xfe broken\n")
    return SOURCE, "is not valid UTF-8"


def _empty(home):
    home.core.write_bytes(b"")
    return SOURCE, "is empty"


def _whitespace_only(home):
    home.core.write_bytes(b" \n\t\n")
    return SOURCE, "is empty or holds only whitespace"


def _markers_inside(home):
    home.core.write_text("text </workstation_guidance> more\n", encoding="utf-8")
    return SOURCE, "itself contains workstation_guidance markers"


def _relative(home):
    return "relative/AGENTS.md", "must be absolute or start with ~"


def _not_a_string(home):
    return ["/x"], "must be a path string, not list"


FAILURES = {"missing": _missing, "not utf-8": _not_utf8, "empty": _empty, "whitespace only": _whitespace_only,
            "markers inside": _markers_inside, "relative": _relative, "not a string": _not_a_string}


@pytest.mark.parametrize("case", sorted(FAILURES))
def test_configured_failures_are_explicit_and_never_partial(home, case):
    value, reason = FAILURES[case](home)
    _config(home.hermes, agent__home_guidance_file=value)
    agent = _agent()
    statuses = []
    agent._emit_diagnostic_status = statuses.append
    try:
        prompt = _prompt(agent)
        assert "<workstation_guidance_unavailable" in prompt and reason in prompt
        assert "<workstation_guidance " not in prompt and "Fixture rule A" not in prompt
        assert any("Home guidance" in s and reason in s for s in statuses), statuses
    finally:
        agent.close()


def _fill(core, chars):
    """Rewrite the core as exactly *chars* characters (ASCII, so characters are bytes)."""
    head = "# Home core A\n\nFixture rule A: cite the original artifact.\n"
    core.write_text(head + "a" * (chars - len(head)), encoding="utf-8")


@pytest.mark.parametrize("over", [False, True], ids=["at the limit", "one character over"])
def test_the_whole_file_limit_is_a_quarter_of_the_window_and_never_truncates(home, over):
    """The trusted file is bounded by the model's window, not by the project context-file cap: up to a quarter of the
    window (the runtime's 4 characters per token) it loads whole even under a 20-character context-file cap; one
    character more and it is Unavailable, never truncated."""
    from agent.model_metadata import CHARS_PER_TOKEN
    _config(home.hermes, agent__home_guidance_file=SOURCE, context_file_max_chars=20, model__context_length=80_000)
    agent = _agent()
    try:
        window = agent.context_compressor.context_length
        assert window == 80_000 and CHARS_PER_TOKEN == 4
        limit = window * CHARS_PER_TOKEN // 4
        _fill(home.core, limit + over)
        prompt = _prompt(agent)
        if over:
            assert (f"has {limit + 1} characters, over the home-guidance limit of {limit} (the larger of the "
                    f"context-file limit 20 and a quarter of the model's {window * 4}-character window); it is never "
                    "truncated") in prompt
            assert "Fixture rule A" not in prompt and "TRUNCATED" not in prompt
        else:
            assert prompt.count(_expected_block(home.core)) == 1
    finally:
        agent.close()


def test_a_larger_context_file_limit_still_admits_a_larger_file(home):
    """A profile's explicit context-file limit (the role floors) keeps working when it exceeds a quarter window."""
    _config(home.hermes, agent__home_guidance_file=SOURCE, context_file_max_chars=100_000, model__context_length=64_000)
    agent = _agent()
    try:
        assert agent.context_compressor.context_length == 64_000  # Hermes's minimum window; a quarter is 64000 chars
        _fill(home.core, 100_000)
        assert _prompt(agent).count(_expected_block(home.core)) == 1
        _fill(home.core, 100_001)
        assert hg.load_file(SOURCE, 64_000).reason.endswith(
            "has 100001 characters, over the home-guidance limit of 100000 (the larger of the context-file limit "
            "100000 and a quarter of the model's 256000-character window); it is never truncated")
    finally:
        agent.close()


def test_an_unknown_window_falls_back_to_the_context_file_limit(home):
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    _fill(home.core, 20_001)
    result = hg.load_file(SOURCE, None)
    assert isinstance(result, hg.Unavailable) and result.reason.endswith(
        "has 20001 characters, over the home-guidance limit of 20000 (the context-file limit; the model's window is "
        "unknown); it is never truncated")
    _fill(home.core, 20_000)
    assert isinstance(hg.load_file(SOURCE, None), hg.Carried)


@pytest.mark.parametrize("window, loads", [(272_000, True), (256_000, True), (128_000, True), (64_000, False)])
def test_a_plain_profile_loads_a_106083_character_edition_without_a_cap(home, window, loads):
    """A new plain profile writes no context_file_max_chars (hermes_cli/profiles.py create_profile seeds only the
    model): an edition of approved A's size loads on 272K (the default route), 256K (the unknown-model fallback) and
    128K windows, and is refused whole, with the reason, on a 64K one."""
    _config(home.hermes, agent__home_guidance_file=SOURCE, model__context_length=window)
    agent = _agent()
    try:
        assert agent.context_compressor.context_length == window
        _fill(home.core, 106_083)
        prompt = _prompt(agent)
        assert (prompt.count(_expected_block(home.core)) == 1) is loads
        assert ("over the home-guidance limit of 64000" in prompt) is not loads
    finally:
        agent.close()


def test_crlf_bytes_are_delivered_and_hashed_exactly(home):
    raw = b"# CRLF core\r\n\r\nFixture rule A.\r\n"
    home.core.write_bytes(raw)
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert f'sha256="{hashlib.sha256(raw).hexdigest()}">\n' in prompt
        body = prompt.split('">\n', 1)[1].split("</workstation_guidance>", 1)[0]
        assert body.encode("utf-8") == raw
    finally:
        agent.close()


def _read_text_before_3_13(self, encoding=None, errors=None):
    """``Path.read_text`` as Python 3.11 and 3.12 define it (no *newline*); the project supports both."""
    with self.open(encoding=encoding, errors=errors) as handle:
        return handle.read()


def test_the_timed_reader_keeps_or_translates_line_endings_on_every_supported_python(tmp_path, monkeypatch):
    """``_read_text_with_timeout`` takes *newline* as ``open()`` does, with only APIs every supported Python has: the
    default translates CRLF for SOUL.md and context files as before; ``newline=""`` keeps the exact bytes."""
    from agent import prompt_builder as pb
    monkeypatch.setattr(Path, "read_text", _read_text_before_3_13)
    path = tmp_path / "crlf.md"
    raw = b"\xef\xbb\xbf# Core\r\n\r\nRule.\r\n"
    path.write_bytes(raw)
    assert pb._read_text_with_timeout(path, timeout=5) == "# Core\n\nRule.\n"
    assert pb._read_text_with_timeout(path, timeout=5, newline="").encode("utf-8") == raw[3:]
    assert pb._read_context_file(path) == "# Core\n\nRule."


def test_an_ignore_rules_constructor_still_gets_the_file_but_not_soul_or_project_rules(home, tmp_path, monkeypatch):
    """--ignore-rules suppresses AGENTS.md, SOUL.md, .cursorrules, memory and preloaded skills; the configured home
    guidance is none of those, so the ignore-rules and cron-like constructors carry it once — and still no persona
    or project rules."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("PROJECT RULE SENTINEL\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    (home.hermes / "SOUL.md").write_text("PERSONA SENTINEL\n", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    ignore = _agent(skip_context_files=True, load_soul_identity=False)
    cron_like = _agent(skip_context_files=True, load_soul_identity=True)
    try:
        ignored = _prompt(ignore)
        assert ignored.count(_expected_block(home.core)) == 1
        assert "PERSONA SENTINEL" not in ignored and "PROJECT RULE SENTINEL" not in ignored
        cron = _prompt(cron_like)
        assert cron.count(_expected_block(home.core)) == 1 and "PERSONA SENTINEL" in cron
        assert "PROJECT RULE SENTINEL" not in cron
    finally:
        ignore.close()
        cron_like.close()


def _bind(monkeypatch, *, safe, cfg):
    from agent import safe_worker_policy
    monkeypatch.setattr(safe_worker_policy, "_policy", safe_worker_policy._SafeWorkerPolicy(
        os.getpid(), safe, safe, json.dumps(cfg)))


def test_safe_mode_renders_the_file_its_frozen_config_names_and_nothing_of_the_profile(home, monkeypatch):
    """A safe worker reads only its frozen config (the bypass producer puts nothing of the profile there, only the
    administrator's pin): the file it names loads once, while the SOUL persona and every other customization stay off."""
    (home.hermes / "SOUL.md").write_text("PERSONA SENTINEL\n", encoding="utf-8")
    _config(home.hermes)  # the live profile names nothing
    _bind(monkeypatch, safe=True, cfg={"agent": {"home_guidance_file": SOURCE}})
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert prompt.count(_expected_block(home.core)) == 1 and "PERSONA SENTINEL" not in prompt
        assert agent.skip_context_files and not agent.load_soul_identity  # safe mode's own restrictions are unchanged
    finally:
        agent.close()


def test_safe_mode_without_a_captured_setting_renders_none(home, monkeypatch):
    _config(home.hermes, agent__home_guidance_file=SOURCE)  # the live profile names it; the worker never reads it
    _bind(monkeypatch, safe=True, cfg={})
    agent = _agent()
    try:
        assert "workstation_guidance" not in _prompt(agent)
    finally:
        agent.close()


def test_managed_worker_reads_its_frozen_config_not_the_live_profile(home, monkeypatch):
    """An ordinary managed worker binds the owner's frozen config before imports (agent/managed_worker.py
    ``bind_worker_policy``); every config reader returns it, so the setting rides the frozen copy."""
    frozen_cfg = {"model": {"default": "anthropic/claude-sonnet-4.6"},
                  "agent": {"environment_probe": False, "home_guidance_file": SOURCE}}
    _config(home.hermes)  # the live profile has no setting
    _bind(monkeypatch, safe=False, cfg=frozen_cfg)
    from agent.safe_worker_policy import safe_worker_enabled
    assert not safe_worker_enabled()
    agent = _agent()
    try:
        assert _prompt(agent).count(_expected_block(home.core)) == 1
    finally:
        agent.close()


def test_the_edition_is_frozen_across_compression_rebuilds(home):
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        first = _expected_block(home.core)
        assert _prompt(agent).count(first) == 1
        home.core.write_text(CORE_B, encoding="utf-8")
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        assert rebuilt.count(first) == 1 and "Fixture rule B" not in rebuilt
    finally:
        agent.close()
    fresh = _agent()
    try:
        assert _prompt(fresh).count(_expected_block(home.core)) == 1  # a new session reads the new edition
    finally:
        fresh.close()


@pytest.mark.parametrize("configured", [False, True], ids=["nothing configured", "file configured"])
def test_a_block_added_to_soul_mid_session_is_not_adopted(home, configured):
    """The frozen result covers absence too: a block SOUL.md gains after the first build is replaced by the frozen
    content (removed when that was "none"), while the rest of the re-read SOUL.md still refreshes as before."""
    soul = home.hermes / "SOUL.md"
    soul.write_text("I am a role.\n", encoding="utf-8")
    _config(home.hermes, **({"agent__home_guidance_file": SOURCE} if configured else {}))
    agent = _agent()
    try:
        first = _prompt(agent)
        block_a = _expected_block(home.core)
        assert first.count(block_a) == (1 if configured else 0)
        soul.write_text('I am a role, revised.\n\n<workstation_guidance source="s">\nLate SOUL edition\n'
                        "</workstation_guidance>\n", encoding="utf-8")
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        assert "I am a role, revised." in rebuilt and "Late SOUL edition" not in rebuilt
        assert rebuilt.count("<workstation_guidance") == (1 if configured else 0)
        assert rebuilt.count(block_a) == (1 if configured else 0)
    finally:
        agent.close()


def test_a_soul_edition_change_does_not_reach_a_running_agent(home):
    soul_a = '<workstation_guidance source="s">\nSOUL edition A\n</workstation_guidance>'
    soul_b = '<workstation_guidance source="s">\nSOUL edition B\n</workstation_guidance>'
    soul = home.hermes / "SOUL.md"
    soul.write_text(f"I am a role.\n\n{soul_a}\n", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        assert _prompt(agent).count(soul_a) == 1
        soul.write_text(f"I am a role, revised.\n\n{soul_b}\n", encoding="utf-8")
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        assert rebuilt.count("<workstation_guidance") == 1 and soul_a in rebuilt and "edition B" not in rebuilt
        assert "I am a role, revised." in rebuilt  # the rest of SOUL.md still follows its own re-read
    finally:
        agent.close()


def _restore(agent, stored):
    """Resume through the real restore path; only the session row is stubbed. Its ``db_path`` stays in the agent's own
    home, which is where the prompt builder reads SOUL.md from (``agent/system_prompt.py::_agent_home``)."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    db = MagicMock()
    db.db_path = Path(os.environ["HERMES_HOME"]) / "state.db"
    db.get_session.return_value = {"system_prompt": stored}
    agent._session_db = db
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])


def test_a_resumed_session_keeps_its_stored_edition_through_compression(home):
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="s1")
    try:
        stored = _prompt(first)
        block_a = _expected_block(home.core)
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    resumed = _agent(session_id="s1")
    try:
        _restore(resumed, stored)
        assert resumed._cached_system_prompt == stored
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert rebuilt.count(block_a) == 1 and "Fixture rule B" not in rebuilt
    finally:
        resumed.close()


def test_a_resumed_soul_edition_wins_over_a_changed_soul(home):
    soul = home.hermes / "SOUL.md"
    soul_a = '<workstation_guidance source="s">\nSOUL edition A\n</workstation_guidance>'
    soul.write_text(f"I am a role.\n\n{soul_a}\n", encoding="utf-8")
    _config(home.hermes)
    first = _agent(session_id="s2")
    try:
        stored = _prompt(first)
    finally:
        first.close()
    soul.write_text('I am a role, revised.\n\n<workstation_guidance source="s">\nSOUL edition B\n'
                    "</workstation_guidance>\n", encoding="utf-8")
    resumed = _agent(session_id="s2")
    try:
        _restore(resumed, stored)
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert rebuilt.count("<workstation_guidance") == 1 and soul_a in rebuilt
        assert rebuilt.startswith("I am a role, revised.")  # the rest of SOUL.md still follows its own re-read
    finally:
        resumed.close()


def test_a_resumed_session_without_guidance_does_not_adopt_a_late_soul_block(home):
    soul = home.hermes / "SOUL.md"
    soul.write_text("I am a role.\n", encoding="utf-8")
    _config(home.hermes)
    first = _agent(session_id="s5")
    try:
        stored = _prompt(first)
    finally:
        first.close()
    soul.write_text('I am a role, revised.\n\n<workstation_guidance source="s">\nLate SOUL edition\n'
                    "</workstation_guidance>\n", encoding="utf-8")
    resumed = _agent(session_id="s5")
    try:
        _restore(resumed, stored)
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert "workstation_guidance" not in rebuilt and rebuilt.startswith("I am a role, revised.")
    finally:
        resumed.close()


def test_a_model_switch_rebuild_keeps_the_stored_edition(home):
    """A stored prompt for another model is rebuilt rather than reused (``_stored_prompt_matches_runtime``); the
    conversation still keeps the edition it started with instead of reading the file again."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="s6")
    try:
        stored = _prompt(first)
        block_a = _expected_block(home.core)
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    resumed = _agent(session_id="s6", model="anthropic/claude-opus-4.6")
    try:
        _restore(resumed, stored)
        rebuilt = resumed._cached_system_prompt
        assert rebuilt != stored and "anthropic/claude-opus-4.6" in rebuilt
        assert rebuilt.count(block_a) == 1 and "Fixture rule B" not in rebuilt
    finally:
        resumed.close()


def test_markers_quoted_in_project_context_do_not_change_the_restored_edition(home, tmp_path, monkeypatch):
    """Only the identity head of a stored prompt holds the session's edition; a project file quoting the markers
    neither hides the SOUL edition nor reads as one."""
    project = tmp_path / "project"
    project.mkdir()
    quoted = '<workstation_guidance source="quoted">\nQuoted project example\n</workstation_guidance>'
    (project / "AGENTS.md").write_text(f"Project rule: the guidance block looks like this:\n\n{quoted}\n",
                                       encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    soul = home.hermes / "SOUL.md"
    soul_a = '<workstation_guidance source="s">\nSOUL edition A\n</workstation_guidance>'
    soul.write_text(f"I am a role.\n\n{soul_a}\n", encoding="utf-8")
    _config(home.hermes)
    first = _agent(session_id="s7")
    try:
        stored = _prompt(first)
        assert soul_a in stored and quoted in stored
    finally:
        first.close()
    soul.write_text('I am a role.\n\n<workstation_guidance source="s">\nSOUL edition B\n</workstation_guidance>\n',
                    encoding="utf-8")
    resumed = _agent(session_id="s7")
    try:
        _restore(resumed, stored)
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert soul_a in rebuilt and "SOUL edition B" not in rebuilt and quoted in rebuilt
    finally:
        resumed.close()


def test_a_bot_chat_capability_rebuild_keeps_the_stored_edition(home):
    """The capability-refresh branch rebuilds a resumed prompt without a prior cached prompt; the stored edition
    still holds because the restore seeds the frozen result first."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="s3")
    try:
        stored = _prompt(first)
        block_a = _expected_block(home.core)
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    resumed = _agent(session_id="s3")
    try:
        with patch("agent.conversation_loop._bot_chat_prompt_stale", return_value=True), \
                patch("agent.conversation_loop._refresh_bot_chat_tools"):
            _restore(resumed, stored)
        assert resumed._cached_system_prompt.count(block_a) == 1
        assert "Fixture rule B" not in resumed._cached_system_prompt
    finally:
        resumed.close()


@pytest.mark.parametrize("case", ["missing", "whitespace only"])
def test_an_unavailable_result_survives_a_resume(home, case):
    _value, reason = FAILURES[case](home)
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="s4")
    try:
        stored = _prompt(first)
        assert reason in stored
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    resumed = _agent(session_id="s4")
    try:
        _restore(resumed, stored)
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert "<workstation_guidance_unavailable" in rebuilt and reason in rebuilt
        assert "<workstation_guidance " not in rebuilt and "Fixture rule B" not in rebuilt
    finally:
        resumed.close()


def test_project_context_and_memory_behavior_are_unchanged(home, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("Project rule: run the project tests.\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    (home.hermes / "memories").mkdir(exist_ok=True)
    (home.hermes / "memories" / "MEMORY.md").write_text("Memory sentinel: the fixture remembers.", encoding="utf-8")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent(skip_memory=False)
    skipped = _agent(skip_context_files=True, load_soul_identity=True)  # skip_memory=True
    try:
        prompt = _prompt(agent)
        block = _expected_block(home.core)
        assert prompt.index(block) < prompt.index("Project rule: run the project tests.")
        assert "Memory sentinel: the fixture remembers." in prompt
        skipped_prompt = _prompt(skipped)
        assert skipped_prompt.count(block) == 1
        assert "Project rule" not in skipped_prompt  # skip_context_files still skips project files
        assert "Memory sentinel" not in skipped_prompt and skipped._memory_store is None  # skip_memory still skips
    finally:
        agent.close()
        skipped.close()


# ── Review findings: restricted rebuilds, SOUL completeness, recorded state, marker forms ──────────────────────────────


def _soul_carrying_home(home):
    block = '<workstation_guidance source="fixture">\nUSER CORE SENTINEL\n</workstation_guidance>'
    (home.hermes / "SOUL.md").write_text(f"Fixture identity\n{block}\n", encoding="utf-8")
    return block


@pytest.mark.parametrize("source", ["file", "soul"])
@pytest.mark.parametrize("restriction", ["safe", "ignore"])
@pytest.mark.parametrize("rebuild", ["model mismatch", "compression"])
def test_a_restricted_rebuild_keeps_the_recorded_edition(home, monkeypatch, source, restriction, rebuild):
    """A resumed session keeps its recorded edition through every rebuild: safe mode and ignore-rules constructors drop
    the SOUL persona (as on baseline) but not the conversation's home guidance, which they render exactly once."""
    cfg = _config(home.hermes, **({"agent__home_guidance_file": SOURCE} if source == "file" else {}))
    block = _expected_block(home.core) if source == "file" else _soul_carrying_home(home)
    first = _agent(session_id="policy")
    try:
        stored = _prompt(first)
        assert stored.count(block) == 1
    finally:
        first.close()
    if restriction == "safe":
        _bind(monkeypatch, safe=True, cfg=cfg)
    kw = {"skip_context_files": True, "load_soul_identity": False} if restriction == "ignore" else {}
    model = {"model": "anthropic/claude-opus-4.6"} if rebuild == "model mismatch" else {}
    resumed = _agent(session_id="policy", **model, **kw)
    try:
        _restore(resumed, stored)
        if rebuild == "compression":
            assert resumed._cached_system_prompt == stored  # same-model reuse sends the stored bytes, as on baseline
            resumed._invalidate_system_prompt()
            _prompt(resumed)
        rebuilt = resumed._cached_system_prompt
        assert rebuilt != stored and rebuilt.count(block) == 1 and rebuilt.count("<workstation_guidance") == 1
        assert "Fixture identity" not in rebuilt  # the persona stays suppressed
        assert hg.frozen(resumed) == hg.Carried(block, source)
    finally:
        resumed.close()


def _over_cap_soul(home):
    """A SOUL.md whose one complete block is longer than the cap; its tags survive the reader's head/tail truncation."""
    payload = "START\n" + "fixture instruction.\n" * 200 + "END\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (home.hermes / "SOUL.md").write_text(f'<workstation_guidance source="s" sha256="{digest}">\n{payload}'
                                         "</workstation_guidance>", encoding="utf-8")


@pytest.mark.parametrize("configured", [False, True], ids=["nothing configured", "file configured"])
def test_a_truncated_soul_block_is_never_taken_as_delivered(home, configured):
    _over_cap_soul(home)
    settings = {"context_file_max_chars": 1000, **({"agent__home_guidance_file": SOURCE} if configured else {})}
    _config(home.hermes, **settings)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert "[...truncated SOUL.md" in prompt  # the existing SOUL reader still caps SOUL.md
        assert "not delivered home guidance" in prompt if configured else "not delivered" not in prompt
        snapshot = hg.frozen(agent)
        if configured:
            assert snapshot == hg.Carried(_expected_block(home.core), "file") and prompt.count(snapshot.block) == 1
        else:
            assert snapshot is None
    finally:
        agent.close()


def test_a_crlf_soul_block_the_reader_normalizes_is_not_delivered_guidance(home):
    """SOUL.md is read with newline translation, so a CRLF block reaches the prompt altered; it is not exact guidance and
    the configured file supplies the edition. The rest of SOUL.md is loaded as before."""
    raw = b'I am a CRLF role.\r\n\r\n<workstation_guidance source="s">\r\nCRLF SOUL edition\r\n</workstation_guidance>\r\n'
    (home.hermes / "SOUL.md").write_bytes(raw)
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        prompt = _prompt(agent)
        assert prompt.startswith("I am a CRLF role.\n\n")
        assert hg.frozen(agent) == hg.Carried(_expected_block(home.core), "file")
        assert "did not deliver whole" in prompt and prompt.count(_expected_block(home.core)) == 1
    finally:
        agent.close()


def test_an_lf_soul_block_identical_to_the_file_is_delivered(home):
    block = '<workstation_guidance source="s">\nLF SOUL edition\n</workstation_guidance>'
    (home.hermes / "SOUL.md").write_bytes(f"I am a role.\n\n{block}\n".encode())
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent()
    try:
        _prompt(agent)
        assert hg.frozen(agent) == hg.Carried(block, "soul")
    finally:
        agent.close()


_EXAMPLE_UNAVAILABLE = '<workstation_guidance_unavailable source="example" reason="example-only"/>'
_FAKE_STATE = '<!-- Hermes home guidance: {"state":"none","v":1} -->'


def _help_in_core(home):
    home.core.write_text(f"# Home\nDocumentation for Hermes:\n{pb_help()}\nKeep this final rule.\n", encoding="utf-8")


def _unavailable_in_core(home):
    home.core.write_text(f"# Home\nThe diagnostic format is:\n{_EXAMPLE_UNAVAILABLE}\nKeep this final rule.\n",
                         encoding="utf-8")


def _state_line_in_core(home):
    home.core.write_text(f"# Home\nA quoted record:\n{_FAKE_STATE}\n", encoding="utf-8")


def _help_before_soul(home):
    (home.hermes / "SOUL.md").write_text(f"{pb_help()}\n<workstation_guidance>SOUL EDITION A</workstation_guidance>",
                                         encoding="utf-8")


def _unavailable_in_soul(home):
    (home.hermes / "SOUL.md").write_text(f"<workstation_guidance>Diagnostic format: {_EXAMPLE_UNAVAILABLE}\n"
                                         "SOUL EDITION A</workstation_guidance>", encoding="utf-8")


def _state_line_in_soul(home):
    (home.hermes / "SOUL.md").write_text(f"I am a role.\n{_FAKE_STATE}\n", encoding="utf-8")


def _supersedes_in_repeated_soul(home):
    (home.hermes / "SOUL.md").write_text(f"Legacy note: {hg._SUPERSEDES}\n<workstation_guidance>OLD ONE"
                                         "</workstation_guidance>\n<workstation_guidance>OLD TWO</workstation_guidance>",
                                         encoding="utf-8")


def _repeated_soul(home):
    (home.hermes / "SOUL.md").write_text("<workstation_guidance>OLD ONE</workstation_guidance>\n"
                                         "<workstation_guidance>OLD TWO</workstation_guidance>", encoding="utf-8")


def pb_help():
    from agent import prompt_builder
    return prompt_builder.HERMES_AGENT_HELP_GUIDANCE


ROUNDTRIP = {"help in core": _help_in_core, "unavailable example in core": _unavailable_in_core,
             "state line in core": _state_line_in_core, "help before the SOUL block": _help_before_soul,
             "unavailable example in SOUL": _unavailable_in_soul, "state line in SOUL": _state_line_in_soul,
             "supersedes note in a repeated SOUL": _supersedes_in_repeated_soul, "repeated SOUL": _repeated_soul}


@pytest.mark.parametrize("case", sorted(ROUNDTRIP))
def test_text_inside_guidance_or_identity_never_reads_as_recorded_state(home, case):
    """Accepted guidance and SOUL.md may quote the help text, the unavailable notice, the fallback note or the state
    record itself; a resumed session still restores exactly the edition the first build chose."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    ROUNDTRIP[case](home)
    first = _agent(session_id="roundtrip")
    try:
        stored = _prompt(first)
        original = hg.frozen(first)
        assert isinstance(original, hg.Carried) and stored.count(original.block) == 1
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    (home.hermes / "SOUL.md").write_text("I am a role, revised.\n", encoding="utf-8")
    resumed = _agent(session_id="roundtrip")
    try:
        _restore(resumed, stored)
        assert hg.frozen(resumed) == original
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert rebuilt.count(original.block) == 1 and "Fixture rule B" not in rebuilt
        assert hg.frozen(resumed) == original
    finally:
        resumed.close()


def test_a_state_record_quoted_in_project_context_is_ignored(home, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text(f"Project rule.\n{_FAKE_STATE}\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="project-record")
    try:
        stored = _prompt(first)
        block_a = _expected_block(home.core)
        assert _FAKE_STATE in stored
    finally:
        first.close()
    home.core.write_text(CORE_B, encoding="utf-8")
    resumed = _agent(session_id="project-record")
    try:
        _restore(resumed, stored)
        resumed._invalidate_system_prompt()
        assert _prompt(resumed).count(block_a) == 1
    finally:
        resumed.close()


def test_a_record_that_does_not_match_its_prompt_is_not_trusted(home, caplog):
    """A stored prompt whose recorded block bytes were altered yields no edition (logged): the continuing conversation
    carries an explicit unavailable notice rather than deciding afresh mid-way."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    first = _agent(session_id="tampered")
    try:
        stored = _prompt(first)
    finally:
        first.close()
    tampered = stored.replace("Fixture rule A", "Fixture rule X", 1)
    assert hg.persisted_state(stored) == hg.Carried(_expected_block(home.core), "file")
    with caplog.at_level("WARNING"):
        state = hg.persisted_state(tampered)
    assert isinstance(state, hg.Unavailable) and state.source == "agent.home_guidance_file"
    assert state.reason.startswith("this conversation's stored home-guidance record is unusable (")
    assert "home-guidance record is unusable" in caplog.text
    resumed = _agent(session_id="tampered")
    try:
        _restore(resumed, tampered)
        resumed._invalidate_system_prompt()
        rebuilt = _prompt(resumed)
        assert "<workstation_guidance_unavailable " in rebuilt and "Fixture rule A" not in rebuilt.split("-->")[-1]
        assert _guidance(rebuilt) == [_guidance(state.notice())[0]]
    finally:
        resumed.close()


def test_a_live_model_switch_keeps_the_frozen_edition(home):
    """``switch_model`` clears the cached prompt without the compression invalidator; the next turn's restore/rebuild
    still renders the edition this agent froze."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent(session_id="live-switch")
    try:
        stored = _prompt(agent)
        block_a = _expected_block(home.core)
        home.core.write_text(CORE_B, encoding="utf-8")
        agent.switch_model("anthropic/claude-opus-4.6", "openrouter", api_key="k",
                           base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
        assert agent._cached_system_prompt is None
        _restore(agent, stored)
        rebuilt = agent._cached_system_prompt
        assert "anthropic/claude-opus-4.6" in rebuilt and rebuilt.count(block_a) == 1 and "Fixture rule B" not in rebuilt
    finally:
        agent.close()


@pytest.mark.parametrize("text", ["<workstation_guidance", "<workstation_guidance/>", "<workstation_guidance / >",
                                  "x </workstation_guidance"])
def test_incomplete_marker_forms_are_malformed(text):
    block, problem = hg.scan_blocks(text)
    assert block is None and problem is not None


def test_longer_tag_names_are_not_markers():
    assert hg.scan_blocks(f"{_EXAMPLE_UNAVAILABLE} <workstation_guidance_extra>") == (None, None)


# ── Session boundaries on a reused agent (HG-SESSION-1): /resume binds the target's edition, /new decides afresh ────────


def _target_file_b(home):
    home.core.write_text(CORE_B, encoding="utf-8")
    block_b = _expected_block(home.core)
    return lambda prompt: prompt.count(block_b) == 1


def _target_soul_b(home):
    soul_b = '<workstation_guidance source="s">\nSOUL edition B\n</workstation_guidance>'
    (home.hermes / "SOUL.md").write_text(f"I am a role.\n\n{soul_b}\n", encoding="utf-8")
    return lambda prompt: prompt.count(soul_b) == 1


def _target_none(home):
    _config(home.hermes)
    return lambda prompt: "workstation_guidance" not in prompt


def _target_unavailable(home):
    home.core.unlink()
    return lambda prompt: "<workstation_guidance_unavailable" in prompt and "<workstation_guidance " not in prompt


TARGETS = {"file edition B": _target_file_b, "SOUL edition B": _target_soul_b, "none": _target_none,
           "unavailable": _target_unavailable}


def _after_target_session(home):
    """Undo the target's inputs so a fresh decision would differ from both editions: core C, no SOUL, configured."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    (home.hermes / "SOUL.md").unlink(missing_ok=True)
    home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")


@pytest.mark.parametrize("rebuild", ["compression", "model mismatch"])
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_resuming_another_session_on_the_same_agent_binds_that_sessions_edition(home, target, rebuild):
    """The CLI's real /resume helper repoints a built agent at another session; that session's recorded decision —
    carried, unavailable or none — governs, never the edition of the conversation the agent left."""
    from hermes_cli.cli_session_switch import _sync_agent_to_session
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent(session_id="session-a")
    try:
        block_a = _expected_block(home.core)
        assert _prompt(agent).count(block_a) == 1
        holds_target = TARGETS[target](home)
        other = _agent(session_id="session-b")
        try:
            target_prompt = _prompt(other)
            assert holds_target(target_prompt) and block_a not in target_prompt
        finally:
            other.close()
        _after_target_session(home)
        cli = SimpleNamespace(agent=agent, conversation_history=[{"role": "user", "content": "session b history"}])
        _sync_agent_to_session(cli, "session-b", parent_session_id="session-a", reason="resume")
        if rebuild == "model mismatch":
            agent.switch_model("anthropic/claude-opus-4.6", "openrouter", api_key="k",
                               base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
        _restore(agent, target_prompt)
        if rebuild == "compression":
            assert agent._cached_system_prompt == target_prompt
            agent._invalidate_system_prompt()
            _prompt(agent)
        rebuilt = agent._cached_system_prompt
        assert holds_target(rebuilt)
        assert "Fixture rule A" not in rebuilt and "Fixture rule C" not in rebuilt
    finally:
        agent.close()


NEW_SESSION = {"file edition B": _target_file_b, "none": _target_none, "unavailable": _target_unavailable}


@pytest.mark.parametrize("current", sorted(NEW_SESSION))
def test_a_new_session_on_the_same_agent_decides_from_current_inputs(home, current):
    """The CLI's /new transition on a built agent (``cli_session_mixin``: new id, ``reset_session_state``, invalidate)
    starts a conversation that reads the current inputs; the previous conversation's edition is not carried over."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent(session_id="session-a")
    try:
        block_a = _expected_block(home.core)
        assert _prompt(agent).count(block_a) == 1
        holds_current = NEW_SESSION[current](home)
        agent.session_id = "session-new"
        agent.reset_session_state()
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        assert holds_current(rebuilt) and "Fixture rule A" not in rebuilt
    finally:
        agent.close()


def test_a_session_boundary_keeps_freezing_within_the_new_session(home):
    """After /new the new conversation freezes its own edition across compression, as any session does."""
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _agent(session_id="session-a")
    try:
        _prompt(agent)
        home.core.write_text(CORE_B, encoding="utf-8")
        agent.session_id = "session-new"
        agent.reset_session_state()
        agent._invalidate_system_prompt()
        block_b = _expected_block(home.core)
        assert _prompt(agent).count(block_b) == 1
        home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        assert rebuilt.count(block_b) == 1 and "Fixture rule C" not in rebuilt
    finally:
        agent.close()


# ── ACP /reset (HG-ACP-1) and an agent that holds a prompt it never built (cached-only reconstruction) ────────────────


def _guidance(prompt):
    """Every home-guidance block or unavailable notice tag a prompt carries, in order."""
    return re.findall(r"<workstation_guidance .*?</workstation_guidance>|<workstation_guidance_unavailable [^>]*/>",
                      prompt, re.DOTALL)


def _acp_reset_then_next_turn(agent):
    """ACP ``/reset`` through the real ``SlashCommandsMixin._cmd_reset``, then the next turn's real preparation
    (``build_turn_context`` with the production restore-or-build); returns the prompt that turn sends."""
    from acp_adapter.commands import SlashCommandsMixin
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tests.agent.test_turn_context import _build
    state = SimpleNamespace(agent=agent, session_id=agent.session_id, history=[{"role": "user", "content": "earlier"}])
    host = SimpleNamespace(session_manager=SimpleNamespace(save_session=MagicMock()))
    assert SlashCommandsMixin._cmd_reset(host, "", state) == "Conversation history cleared."
    assert state.history == []
    return _build(agent, conversation_history=state.history,
                  restore_or_build_system_prompt=_restore_or_build_system_prompt).active_system_prompt


def _acp_agent():
    agent = _agent(platform="acp", session_id="acp-session")
    agent.compression_enabled = False
    return agent


@pytest.mark.parametrize("current", sorted(NEW_SESSION))
def test_acp_reset_starts_a_conversation_whose_child_gets_the_same_edition(home, current):
    """ACP ``/reset`` ends the conversation on the same agent: the next turn decides from the current inputs, and an
    unprofiled delegate child of that turn carries exactly what its parent now sends — never the previous edition."""
    from tests.tools.test_delegate_home_guidance import _child, _child_prompt
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _acp_agent()
    try:
        assert _prompt(agent).count(_expected_block(home.core)) == 1
        holds_current = NEW_SESSION[current](home)
        sent = _acp_reset_then_next_turn(agent)
        assert holds_current(sent) and "Fixture rule A" not in sent
        child = _child(agent)
        try:
            child_prompt = _child_prompt(child)
        finally:
            child.close()
        assert holds_current(child_prompt) and "Fixture rule A" not in child_prompt
        assert _guidance(child_prompt) == _guidance(sent)
    finally:
        agent.close()


def test_without_acp_reset_the_conversation_and_its_child_keep_their_edition(home):
    """Control for the reset test: the same next-turn path without ``/reset`` reuses the conversation's prompt, so a
    changed file reaches neither the turn nor its child."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tests.agent.test_turn_context import _build
    from tests.tools.test_delegate_home_guidance import _child, _child_prompt
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _acp_agent()
    try:
        block_a = _expected_block(home.core)
        original = _prompt(agent)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent = _build(agent, conversation_history=[{"role": "user", "content": "earlier"}],
                      restore_or_build_system_prompt=_restore_or_build_system_prompt).active_system_prompt
        assert sent == original
        child = _child(agent)
        try:
            child_prompt = _child_prompt(child)
        finally:
            child.close()
        assert _guidance(child_prompt) == _guidance(sent) == [block_a]
    finally:
        agent.close()


def test_the_conversation_after_acp_reset_keeps_freezing(home):
    """After ``/reset`` the new conversation freezes its own edition across compression, for itself and its child."""
    from tests.tools.test_delegate_home_guidance import _child, _child_prompt
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    agent = _acp_agent()
    try:
        _prompt(agent)
        home.core.write_text(CORE_B, encoding="utf-8")
        block_b = _expected_block(home.core)
        assert _guidance(_acp_reset_then_next_turn(agent)) == [block_b]
        home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")
        agent._invalidate_system_prompt()
        rebuilt = _prompt(agent)
        child = _child(agent)
        try:
            child_prompt = _child_prompt(child)
        finally:
            child.close()
        assert _guidance(rebuilt) == _guidance(child_prompt) == [block_b]
    finally:
        agent.close()


CACHED_DECISIONS = {"carried": {"agent__home_guidance_file": SOURCE}, "none": {},
                    "unavailable": {"agent__home_guidance_file": "~/.codex/MISSING.md"}}


@pytest.mark.parametrize("decision", sorted(CACHED_DECISIONS))
def test_a_cache_parity_fork_keeps_the_decision_of_the_prompt_it_shares(home, decision):
    """A same-model review or /btw fork (the real ``build_cache_parity_fork``) holds its parent's cached prompt without
    having built it. Rebuilding the static prefix for that prompt (the real failover redecoration) and a later
    compression rebuild keep the shared prompt's decision — carried, none or unavailable — not a fresh read."""
    from agent.background_review import build_cache_parity_fork
    from agent.conversation_loop import _ensure_cached_system_prompt_static
    _config(home.hermes, **CACHED_DECISIONS[decision])
    parent = _agent()
    try:
        shared = _prompt(parent)
        decided = _guidance(shared)
        assert (decided == [_expected_block(home.core)] if decision == "carried"
                else decided == [] if decision == "none" else len(decided) == 1 and "_unavailable" in decided[0])
        home.core.write_text(CORE_B, encoding="utf-8")
        _config(home.hermes, agent__home_guidance_file=SOURCE)
        fork, _rt, routed = build_cache_parity_fork(parent, max_iterations=4)
        try:
            assert routed is False and fork._cached_system_prompt == shared
            assert "_home_guidance" not in fork.__dict__
            fork._use_prompt_caching = True  # a failover to a cache-on provider
            _ensure_cached_system_prompt_static(fork)
            assert fork._cached_system_prompt == shared
            assert hg.frozen(fork) == hg.frozen(parent)
            fork._invalidate_system_prompt()
            rebuilt = _prompt(fork)
            assert _guidance(rebuilt) == decided and "Fixture rule B" not in rebuilt
        finally:
            fork.close()
    finally:
        parent.close()


# ── A stored conversation's decision is adopted before any pre-turn build (HG-LIFECYCLE-1) ─────────────────────────────
# Real SessionDB, real ACP SessionManager restore and slash commands, real turn preparation and child construction. The
# pre-turn /compress has two messages, so the compressor aborts without progress and without any model call; sockets are
# refused to prove it.


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("no network in these tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def _turn(agent, history):
    """The next ordinary turn's real preparation (auto-titling, a model call, is skipped); returns its prompt."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tests.agent.test_turn_context import _build
    with patch("agent.turn_context._maybe_title_session_at_turn_start"):
        return _build(agent, conversation_history=history,
                      restore_or_build_system_prompt=_restore_or_build_system_prompt).active_system_prompt


def _child_expectation(home, decision, recorded):
    """A delegate child inherits a carried or unavailable parent decision exactly; a parent that carries none hands
    nothing over, and the child, a new conversation, decides from the current inputs (here the readable file B)."""
    return [_expected_block(home.core)] if decision == "none" else recorded


def _child_guidance(agent):
    from tests.tools.test_delegate_home_guidance import _child, _child_prompt
    child = _child(agent)
    try:
        return _guidance(_child_prompt(child))
    finally:
        child.close()


def _configure_core_b(home):
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    home.core.write_text(CORE_B, encoding="utf-8")


# What each recorded decision is set up with, then how the inputs change so a fresh read would give a readable file B.
STORED = {"carried": ({"agent__home_guidance_file": SOURCE}, lambda home: home.core.write_text(CORE_B, encoding="utf-8")),
          "none": ({}, _configure_core_b),
          "unavailable": ({"agent__home_guidance_file": "~/.codex/LATER.md"},
                          lambda home: (home.os_home / ".codex" / "LATER.md").write_text(CORE_B, encoding="utf-8"))}


def _stored_session(home, db, session_id, decision, **kw):
    """A persisted conversation (prompt + two messages) recorded with *decision*; then the inputs change to file B."""
    settings, change = STORED[decision]
    _config(home.hermes, **settings)
    agent = _agent(session_id=session_id, session_db=db, **kw)
    try:
        sent = _turn(agent, [])
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}]
        agent._persist_session(history, [])
    finally:
        agent.close()
    recorded = _guidance(sent)
    assert (recorded == []) == (decision == "none")
    assert decision != "carried" or "Fixture rule A" in recorded[0]
    assert decision != "unavailable" or "_unavailable" in recorded[0]
    change(home)
    return recorded


def _stored_decision(db, session_id):
    return hg.persisted_state(db.get_session(session_id)["system_prompt"])


def _acp_restore(db, session_id, **kw):
    """The real ACP cold restore (``SessionManager.get_session`` -> ``_restore``), building its agent at that point."""
    from acp_adapter.session import SessionManager

    def factory():
        agent = _agent(platform="acp", session_id=session_id, session_db=db, **kw)
        agent.compression_enabled = False
        return agent
    manager = SessionManager(db=db, agent_factory=factory)
    state = manager.get_session(session_id)
    assert state is not None and len(state.history) == 2 and state.agent._cached_system_prompt is None
    state.cwd = ""
    return SimpleNamespace(session_manager=manager), state


@pytest.mark.parametrize("compress_first", [False, True])
@pytest.mark.parametrize("decision", sorted(STORED))
def test_a_cold_acp_restore_keeps_its_recorded_decision_through_a_pre_turn_compress(home, tmp_path, no_network,
                                                                                    decision, compress_first):
    """An ACP conversation restored from the database keeps its recorded decision — carried, none or unavailable — even
    when ``/compress`` runs before its first turn; the parent's prompt and its unprofiled child agree."""
    from acp_adapter.commands import SlashCommandsMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        recorded = _stored_session(home, db, "acp-stored", decision, platform="acp")
        host, state = _acp_restore(db, "acp-stored")
        try:
            if compress_first:
                SlashCommandsMixin._cmd_compress(host, "", state)
                assert len(state.history) == 2  # no progress
                assert hg.frozen(state.agent) == _stored_decision(db, "acp-stored")
            sent = _turn(state.agent, state.history)
            assert _guidance(sent) == recorded and _child_guidance(state.agent) == _child_expectation(home, decision, recorded)
        finally:
            state.agent.close()
    finally:
        db.close()


@pytest.mark.parametrize("decision", sorted(STORED))
def test_a_runtime_mismatch_rebuild_after_a_pre_turn_compress_keeps_the_recorded_decision(home, tmp_path, no_network,
                                                                                          decision):
    """The same restore with another model: the first turn rebuilds instead of reusing the stored bytes, and the rebuild
    and the child still carry the recorded decision."""
    from acp_adapter.commands import SlashCommandsMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        recorded = _stored_session(home, db, "acp-stored", decision, platform="acp")
        stored = db.get_session("acp-stored")["system_prompt"]
        host, state = _acp_restore(db, "acp-stored", model="anthropic/claude-opus-4.6")
        try:
            SlashCommandsMixin._cmd_compress(host, "", state)
            sent = _turn(state.agent, state.history)
            assert sent != stored and "Model: anthropic/claude-opus-4.6" in sent
            assert _guidance(sent) == recorded and _child_guidance(state.agent) == _child_expectation(home, decision, recorded)
        finally:
            state.agent.close()
    finally:
        db.close()


def test_acp_reset_after_a_cold_restore_decides_afresh_and_keeps_freezing(home, tmp_path, no_network):
    """Adoption belongs to the restored conversation only: ``/reset`` then a pre-turn ``/compress`` (nothing to compress)
    and a turn decide from the current file B, never the session row's A; the new edition then stays frozen."""
    from acp_adapter.commands import SlashCommandsMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        recorded = _stored_session(home, db, "acp-stored", "carried", platform="acp")
        host, state = _acp_restore(db, "acp-stored")
        agent = state.agent
        try:
            assert hg.frozen(agent) == _stored_decision(db, "acp-stored")
            assert SlashCommandsMixin._cmd_reset(host, "", state) == "Conversation history cleared."
            assert SlashCommandsMixin._cmd_compress(host, "", state) == "Nothing to compress — conversation is empty."
            block_b = _expected_block(home.core)
            sent = _turn(agent, state.history)
            assert _guidance(sent) == [block_b] != recorded and _child_guidance(agent) == [block_b]
            home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")
            agent._invalidate_system_prompt()
            assert _guidance(_prompt(agent)) == [block_b] and _child_guidance(agent) == [block_b]
        finally:
            agent.close()
    finally:
        db.close()


@pytest.mark.parametrize("command", ["resume", "branch"])
@pytest.mark.parametrize("decision", sorted(STORED))
def test_a_cli_switch_adopts_the_target_decision_before_a_pre_turn_compress(home, tmp_path, no_network, decision,
                                                                           command):
    """The retained in-process CLI seam: a built agent switched by the shared /resume and /branch helper
    (``_sync_agent_to_session``; a /branch target row holds the parent's prompt, written as ``/branch`` writes it), then
    the manual-compression core before the first turn. The target's decision holds before, during and after."""
    from agent.conversation_compression_manual import compress_now, parse_compress_args
    from hermes_cli.cli_session_switch import _sync_agent_to_session
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        recorded = _stored_session(home, db, "stored", decision)
        target = "stored"
        if command == "branch":
            target = "stored-branch"
            db.create_session(session_id=target, source="cli", parent_session_id="stored",
                              system_prompt=db.get_session("stored")["system_prompt"])
        agent = _agent(session_id="live", session_db=db)
        try:
            _turn(agent, [])
            history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}]
            if command == "branch":
                agent._session_db.append_message(target, "user", "hello")
                agent._session_db.append_message(target, "assistant", "old reply")
            cli = SimpleNamespace(agent=agent, conversation_history=history)
            _sync_agent_to_session(cli, target, parent_session_id="live", reason=command)
            assert hg.frozen(agent) == _stored_decision(db, target)
            compress_now(agent, history, parse_compress_args(""), task_id=target)
            assert hg.frozen(agent) == _stored_decision(db, target)
            sent = _turn(agent, history)
            assert _guidance(sent) == recorded and _child_guidance(agent) == _child_expectation(home, decision, recorded)
        finally:
            agent.close()
    finally:
        db.close()


@pytest.mark.parametrize("decision", sorted(STORED))
def test_adopting_the_stored_prompt_replaces_a_decision_made_before_it(home, tmp_path, no_network, decision):
    """The late-adoption half of the invariant: an agent built without its session handle (so nothing was adopted at
    construction) decides from file B in a pre-turn ``/compress``; the turn's restore then adopts the stored prompt,
    whose record replaces B for the parent, the recorded state and the child — never A sent with B delegated."""
    from acp_adapter.commands import SlashCommandsMixin
    from acp_adapter.session import SessionManager
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        recorded = _stored_session(home, db, "acp-stored", decision, platform="acp")
        agent = _agent(platform="acp", session_id="acp-stored")
        agent.compression_enabled = False
        manager = SessionManager(db=db, agent_factory=lambda: agent)
        state = manager.get_session("acp-stored")
        state.cwd = ""
        try:
            agent._session_db = db
            SlashCommandsMixin._cmd_compress(SimpleNamespace(session_manager=manager), "", state)
            assert decision == "none" or "Fixture rule B" in repr(hg.frozen(agent))  # decided before adoption
            sent = _turn(agent, state.history)
            assert hg.frozen(agent) == _stored_decision(db, "acp-stored")
            assert _guidance(sent) == recorded and _child_guidance(agent) == _child_expectation(home, decision, recorded)
        finally:
            agent.close()
    finally:
        db.close()


# ── A reset conversation stays reset across a reconnect (HG-RESET-RECONNECT-1) ────────────────────────────────────────
# The session's stored prompt records its current conversation's decision; a reset in place (ACP /reset) is the one
# operation that ends the conversation and keeps the session, and it drops that stored prompt, so nothing adopts it.


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("decision", sorted(STORED))
def test_acp_reset_stays_a_new_conversation_across_a_reconnect(home, tmp_path, no_network, decision, reconnect):
    """Cold restore -> ``/reset`` (saved) -> optionally a reconnect (a fresh ACP manager restores the session again)
    -> pre-turn ``/compress`` (nothing to compress) -> turn: the reset dropped the ended conversation's stored prompt,
    so the new conversation decides from the current file B, for the parent and its unprofiled child; it then stays
    frozen."""
    from acp_adapter.commands import SlashCommandsMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        _stored_session(home, db, "acp-stored", decision, platform="acp")
        stored = db.get_session("acp-stored")["system_prompt"]
        host, state = _acp_restore(db, "acp-stored")
        agent = state.agent
        try:
            assert SlashCommandsMixin._cmd_reset(host, "", state) == "Conversation history cleared."
            assert state.history == [] and db.get_messages_as_conversation("acp-stored") == []
            assert stored and db.get_session("acp-stored")["system_prompt"] is None  # the ended prompt is dropped
            if reconnect:
                agent.close()
                from acp_adapter.session import SessionManager

                def factory():
                    created = _agent(platform="acp", session_id="acp-stored", session_db=db)
                    created.compression_enabled = False
                    return created
                host = SimpleNamespace(session_manager=SessionManager(db=db, agent_factory=factory))
                state = host.session_manager.get_session("acp-stored")
                agent = state.agent
                assert state.history == [] and hg.frozen(agent) is None  # nothing adopted from the ended prompt
            assert SlashCommandsMixin._cmd_compress(host, "", state) == "Nothing to compress — conversation is empty."
            sent = _guidance(_turn(agent, state.history))
            assert len(sent) == 1 and "Fixture rule B" in sent[0] and _child_guidance(agent) == sent
            home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")
            (home.os_home / ".codex" / "LATER.md").unlink(missing_ok=True)
            agent._invalidate_system_prompt()
            assert _guidance(_prompt(agent)) == sent and _child_guidance(agent) == sent
        finally:
            agent.close()
    finally:
        db.close()


# ── Review forks: same-model forks share the parent's prompt; routed forks build their own (unchanged contract) ──────


@pytest.mark.parametrize("routed", [False, True])
def test_a_review_fork_shares_its_parent_decision_only_on_the_same_model(home, tmp_path, monkeypatch, no_network,
                                                                       routed):
    """``build_cache_parity_fork`` copies the parent's cached prompt only on the same model, so that fork renders the
    parent's decision. A different-model fork is built without the parent's session handle or id and deliberately
    builds a fresh model-specific prompt, so it reads the current file (B). Only credential/runtime resolution is
    substituted."""
    from agent.background_review import build_cache_parity_fork
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    _config(home.hermes, agent__home_guidance_file=SOURCE)
    parent = _agent(session_id="parent", session_db=db)
    try:
        parent_decision = _guidance(_turn(parent, []))
        home.core.write_text(CORE_B, encoding="utf-8")
        cfg = {}
        if routed:
            cfg = {"provider": "openrouter", "model": "anthropic/claude-opus-4.6",
                   "base_url": "https://openrouter.ai/api/v1", "api_key": "k"}
            monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
                "provider": "openrouter", "model": kw["target_model"], "api_key": "k",
                "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"})
        fork, _rt, actual_routed = build_cache_parity_fork(parent, cfg, max_iterations=4)
        try:
            assert actual_routed is routed and fork._session_db is None
            sent = _guidance(_turn(fork, [{"role": "user", "content": "review history"}]))
            expected = [_expected_block(home.core)] if routed else parent_decision
            assert sent == expected and _child_guidance(fork) == expected
        finally:
            fork.close()
    finally:
        parent.close()
        db.close()


# ── Undo/rewind keeps the conversation and its decision, also across a reconnect (HG-UNDO-RECONNECT-1) ──────────────
# The stored prompt is the durable record of the session's conversation: only a reset in place drops it (above). Undoing
# every turn empties the active transcript but neither ends the session nor drops the stored prompt.


def _undo_all(surface, agent, db, history, monkeypatch):
    """The real CLI ``undo_last`` or TUI rewind-to-zero over the whole two-message history."""
    if surface == "cli undo":
        from hermes_cli.cli_session_mixin import CLISessionMixin
        cli = CLISessionMixin.__new__(CLISessionMixin)
        cli.agent, cli._session_db, cli.session_id, cli.conversation_history = agent, db, agent.session_id, history
        assert cli.undo_last(1, prefill=False) == 1 and cli.conversation_history == []
    else:
        from tui_gateway import server
        monkeypatch.setattr(server, "_db", db)
        session = {"agent": agent, "history": history, "history_version": 0, "session_key": agent.session_id}
        installed, _view, count = server._rewind_active_session_history(session, 0)
        assert installed == [] and count == 2


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("surface", ["cli undo", "tui rewind"])
@pytest.mark.parametrize("decision", sorted(STORED))
def test_undoing_every_turn_keeps_the_conversation_and_its_decision(home, tmp_path, no_network, monkeypatch, decision,
                                                                   surface, reconnect):
    """After undoing every turn — and after a reconnect that builds a new agent for the same session, as the TUI factory
    does — the next turn and its unprofiled child carry the conversation's recorded decision (carried, none or
    unavailable), not the current file B; the decision then stays frozen across a rebuild."""
    from hermes_state import SessionDB
    settings, change = STORED[decision]
    _config(home.hermes, **settings)
    db = SessionDB(tmp_path / "state.db")
    agent = _agent(session_id="undo-session", session_db=db)
    agent.compression_enabled = False
    try:
        recorded = _guidance(_turn(agent, []))
        assert (recorded == []) == (decision == "none")
        db.append_message(agent.session_id, "assistant", "old reply")  # the turn already stored the user message
        history = db.get_messages_as_conversation(agent.session_id, repair_alternation=True)
        assert [m["role"] for m in history] == ["user", "assistant"]
        change(home)
        _undo_all(surface, agent, db, history, monkeypatch)
        row = db.get_session(agent.session_id)
        assert db.get_resume_message_count(agent.session_id, tip_only=True) == 0
        assert row["system_prompt"] and row["ended_at"] is None
        if reconnect:
            agent.close()
            agent = _agent(session_id="undo-session", session_db=db)
            agent.compression_enabled = False
        model_history, _display = db.get_resume_conversations(agent.session_id)
        assert model_history == []
        sent = _guidance(_turn(agent, model_history))
        assert sent == recorded and _child_guidance(agent) == _child_expectation(home, decision, recorded)
        home.core.write_text("# Home core C\n\nFixture rule C.\n", encoding="utf-8")
        agent._invalidate_system_prompt()
        assert _guidance(_prompt(agent)) == recorded and _child_guidance(agent) == _child_expectation(home, decision, recorded)
    finally:
        agent.close()
        db.close()


def test_acp_reset_reports_a_failed_clear_of_the_ended_prompt(home, tmp_path, no_network, monkeypatch, caplog):
    """If dropping the ended conversation's stored prompt fails, the real ``/reset`` reports its existing reset-failed
    diagnostic (and logs the error) instead of success; the stored prompt is still there, so nothing claims the
    session was durably reset."""
    import logging
    from acp_adapter.commands import SlashCommandsMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        _stored_session(home, db, "acp-stored", "carried", platform="acp")
        stored = db.get_session("acp-stored")["system_prompt"]
        host, state = _acp_restore(db, "acp-stored")
        try:
            def refuse(session_id):
                raise RuntimeError("database is locked")
            monkeypatch.setattr(db, "end_conversation_in_place", refuse)
            with caplog.at_level(logging.WARNING, logger="acp_adapter.server"):
                result = SlashCommandsMixin._cmd_reset(host, "", state)
            assert result == "Conversation history was not cleared: agent session state reset failed; see logs."
            assert "ACP session state reset failed for acp-stored" in caplog.text and "database is locked" in caplog.text
            assert db.get_session("acp-stored")["system_prompt"] == stored
        finally:
            state.agent.close()
    finally:
        db.close()
