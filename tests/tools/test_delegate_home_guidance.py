"""Delegate children carry their parent's exact home-guidance edition (``agent.home_guidance_file``).

Real ``_build_child_agent`` -> ``AIAgent`` construction and real prompt builds: an unprofiled child loads no SOUL, so it gets
its parent's block (never a fresh read of a file that may have changed); a profile-routed child's own SOUL block still wins;
a parent that has none (or ignore-rules) passes none on; ``skip_context_files``/``skip_memory`` keep their effect."""

import hashlib
import json
import os

import hermes_yaml as yaml
import pytest

CORE_A = "# Home core A\n\nFixture rule A.\n"
CORE_B = "# Home core B\n\nFixture rule B.\n"
SOURCE = "~/.codex/AGENTS.md"
MODEL = "anthropic/claude-sonnet-4.6"


def _config(home, top=None, **agent):
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": MODEL}, "agent": {"environment_probe": False, **agent}, **(top or {})}), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    launch = os_home / ".hermes"
    launch.mkdir(parents=True)
    core = tmp_path / "store" / "CORE.md"
    core.parent.mkdir()
    core.write_text(CORE_A, encoding="utf-8")
    (os_home / ".codex").mkdir()
    (os_home / ".codex" / "AGENTS.md").symlink_to(core)
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dtc, "_load_config", dict)
    # Skip only the write-side copy of each new config.yaml to backups/config (seconds per copy on the runner's ZFS
    # scratch root); the config readers stay real.
    from hermes_cli import config_backups
    monkeypatch.setattr(config_backups, "backup_config", lambda *a, **k: None)
    return launch, core


def _block(core):
    data = core.read_bytes()
    return (f'<workstation_guidance source="{SOURCE}" resolved="{os.path.realpath(core)}" '
            f'sha256="{hashlib.sha256(data).hexdigest()}">\n{data.decode("utf-8")}</workstation_guidance>')


def _parent(**kw):
    from run_agent import AIAgent
    base = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                api_mode="chat_completions", model=MODEL, platform="cli", quiet_mode=True, skip_memory=True,
                save_trajectories=False, enabled_toolsets=["file"], session_id="parent")
    base.update(kw)
    parent = AIAgent(**base)
    parent._cached_system_prompt = parent._build_system_prompt()
    return parent


def _child(parent, **route):
    from tools import delegate_tool as dt
    return dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                 max_iterations=4, task_count=1, parent_agent=parent, **route)


def _child_prompt(child):
    from tools.delegate_tool_routes import delegated_profile_scope
    with delegated_profile_scope(getattr(child, "_delegate_profile_home", None)):
        return child._build_system_prompt()


MEMORY = "Memory sentinel: the parent remembers."


def test_unprofiled_child_carries_the_parents_exact_edition_once(env):
    launch, core = env
    (launch / "memories").mkdir(exist_ok=True)  # the home may already hold an empty memories dir
    (launch / "memories" / "MEMORY.md").write_text(MEMORY, encoding="utf-8")
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent(skip_memory=False)
    block_a = _block(core)
    assert parent._cached_system_prompt.count(block_a) == 1 and MEMORY in parent._cached_system_prompt
    core.write_text(CORE_B, encoding="utf-8")  # a newer edition lands while the parent runs
    child = _child(parent)
    try:
        assert child.skip_context_files and not child.load_soul_identity
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(block_a) == 1
        assert "Fixture rule B" not in prompt
        assert MEMORY not in prompt and child._memory_store is None  # skip_memory still skips memory
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
        assert prompt.startswith(DEFAULT_AGENT_IDENTITY.strip() + "\n\n" + block_a)  # no SOUL, no roster
    finally:
        child.close()
        parent.close()


def test_unprofiled_child_of_a_soul_carrying_parent_gets_that_soul_block_only(env):
    """The parent's SOUL.md owns its edition; the unprofiled child gets that block alone — not the rest of the parent's
    SOUL, not the configured file, and not a newer SOUL edition written after the parent built its prompt."""
    launch, _core = env
    soul_a = (f'<workstation_guidance source="{SOURCE}" resolved="/nix/store/x-core/AGENTS.md" sha256="{"a" * 64}">\n'
              "SOUL-carried edition A\n</workstation_guidance>")
    soul = launch / "SOUL.md"
    soul.write_text(f"I am the launch role; roster and duties follow.\n\n{soul_a}\n", encoding="utf-8")
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent()
    assert parent._cached_system_prompt.count(soul_a) == 1
    assert "Fixture rule A" not in parent._cached_system_prompt
    soul.write_text("I am the launch role.\n\n<workstation_guidance>\nSOUL edition B\n</workstation_guidance>\n",
                    encoding="utf-8")
    child = _child(parent)
    try:
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(soul_a) == 1
        assert "I am the launch role" not in prompt and "edition B" not in prompt and "Fixture rule A" not in prompt
    finally:
        child.close()
        parent.close()


def test_a_parent_carrying_none_hands_nothing_over_and_its_child_decides_afresh(env):
    """A parent whose conversation began before the file was configured carries none and keeps none; its child is a
    new conversation, so it takes up the file configured now instead of inheriting the stale none."""
    launch, core = env
    _config(launch)
    parent = _parent()
    _config(launch, home_guidance_file=SOURCE)  # configured only after the parent started
    child = _child(parent)
    try:
        assert child._home_guidance_inherited is None
        assert "workstation_guidance" not in parent._build_system_prompt()  # the parent's decision stays frozen
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(_block(core)) == 1
    finally:
        child.close()
        parent.close()


def test_an_ignore_rules_parent_carries_the_file_and_hands_its_exact_edition_on(env):
    launch, core = env
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent(skip_context_files=True, load_soul_identity=False)
    block_a = _block(core)
    assert parent._cached_system_prompt.count(block_a) == 1
    core.write_text(CORE_B, encoding="utf-8")
    child = _child(parent)
    try:
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(block_a) == 1 and "Fixture rule B" not in prompt
    finally:
        child.close()
        parent.close()


def test_a_parents_unavailable_result_is_inherited_not_retried(env):
    launch, core = env
    core.unlink()
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent()
    assert "<workstation_guidance_unavailable" in parent._cached_system_prompt
    core.write_text(CORE_B, encoding="utf-8")
    child = _child(parent)
    try:
        prompt = _child_prompt(child)
        assert "<workstation_guidance_unavailable" in prompt and "Fixture rule B" not in prompt
    finally:
        child.close()
        parent.close()


def test_a_safe_mode_child_gets_its_parents_exact_block(env, monkeypatch):
    """Safe mode suppresses the persona, project rules, memory and plugins, not home guidance: a child built under a
    bound safe policy carries its parent's exact edition, not a fresh read of the file its frozen config names."""
    from agent import safe_worker_policy
    launch, _core = env
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent()
    assert parent._cached_system_prompt.count("<workstation_guidance ") == 1
    frozen_cfg = {"model": {"default": MODEL}, "agent": {"environment_probe": False, "home_guidance_file": SOURCE}}
    monkeypatch.setattr(safe_worker_policy, "_policy", safe_worker_policy._SafeWorkerPolicy(
        os.getpid(), True, True, json.dumps(frozen_cfg)))
    assert safe_worker_policy.safe_worker_enabled()
    block_a = _block(_core)
    _core.write_text(CORE_B, encoding="utf-8")
    child = _child(parent)
    try:
        assert child._home_guidance_inherited is not None
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(block_a) == 1 and "Fixture rule B" not in prompt
    finally:
        child.close()
        parent.close()


def test_grandchild_keeps_the_original_edition(env):
    launch, core = env
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent()
    block_a = _block(core)
    core.write_text(CORE_B, encoding="utf-8")
    child = _child(parent)
    try:
        child._cached_system_prompt = _child_prompt(child)
        grandchild = _child(child)
        try:
            assert _child_prompt(grandchild).count(block_a) == 1
        finally:
            grandchild.close()
    finally:
        child.close()
        parent.close()


def _profile(launch, soul):
    coder = launch / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": MODEL},
        "agent": {"environment_probe": False, "home_guidance_file": SOURCE}}), encoding="utf-8")
    (coder / "SOUL.md").write_text(soul, encoding="utf-8")
    return coder


def test_profile_routed_child_with_its_own_block_keeps_only_its_soul_edition(env):
    launch, _core = env
    _config(launch, home_guidance_file=SOURCE)
    soul_block = '<workstation_guidance source="~/.codex/AGENTS.md">\nCoder SOUL edition\n</workstation_guidance>'
    coder = _profile(launch, f"I am the coder profile.\n\n{soul_block}\n")
    parent = _parent()
    child = _child(parent, profile_home=coder)
    try:
        assert child.load_soul_identity
        prompt = _child_prompt(child)
        assert "I am the coder profile." in prompt
        assert prompt.count("<workstation_guidance") == 1 and soul_block in prompt
    finally:
        child.close()
        parent.close()


def test_profile_routed_child_without_a_block_gets_the_parents_edition(env):
    launch, core = env
    _config(launch, home_guidance_file=SOURCE)
    coder = _profile(launch, "I am the coder profile.\n")
    parent = _parent()
    block_a = _block(core)
    core.write_text(CORE_B, encoding="utf-8")
    child = _child(parent, profile_home=coder)
    try:
        prompt = _child_prompt(child)
        assert prompt.startswith("I am the coder profile.\n\n" + block_a)
        assert "Fixture rule B" not in prompt
    finally:
        child.close()
        parent.close()


def _over_cap_soul(launch):
    payload = "START\n" + "fixture instruction.\n" * 200 + "END\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (launch / "SOUL.md").write_text(f'<workstation_guidance source="s" sha256="{digest}">\n{payload}'
                                    "</workstation_guidance>", encoding="utf-8")


def test_an_over_cap_soul_parent_hands_its_child_the_complete_file_edition(env):
    """The parent's SOUL reader truncates its block, so that block is not delivered guidance; the configured file is,
    and the unprofiled child carries that complete edition — never the truncated SOUL text."""
    launch, core = env
    _over_cap_soul(launch)
    _config(launch, top={"context_file_max_chars": 1000}, home_guidance_file=SOURCE)
    parent = _parent()
    block_a = _block(core)
    assert "[...truncated SOUL.md" in parent._cached_system_prompt and parent._cached_system_prompt.count(block_a) == 1
    child = _child(parent)
    try:
        prompt = _child_prompt(child)
        assert prompt.count("<workstation_guidance") == 1 and prompt.count(block_a) == 1
        assert "fixture instruction." not in prompt and "[...truncated" not in prompt
    finally:
        child.close()
        parent.close()


def test_an_over_cap_soul_parent_without_a_file_passes_none_on(env):
    launch, _core = env
    _over_cap_soul(launch)
    _config(launch, top={"context_file_max_chars": 1000})
    parent = _parent()
    child = _child(parent)
    try:
        assert child._home_guidance_inherited is None
        assert "workstation_guidance" not in _child_prompt(child)
    finally:
        child.close()
        parent.close()


@pytest.mark.parametrize("parent_kind", ["ordinary", "ignore-rules"])
def test_a_resumed_parents_child_gets_what_the_parent_may_render(env, parent_kind):
    """A resumed parent that reuses its stored prompt hands children the recorded edition, whether or not it is an
    ignore-rules constructor."""
    from unittest.mock import MagicMock
    from agent.conversation_loop import _restore_or_build_system_prompt
    launch, core = env
    _config(launch, home_guidance_file=SOURCE)
    first = _parent(session_id="resumed-parent")
    stored, block_a = first._cached_system_prompt, _block(core)
    first.close()
    core.write_text(CORE_B, encoding="utf-8")
    from run_agent import AIAgent
    kw = {"skip_context_files": True, "load_soul_identity": False} if parent_kind == "ignore-rules" else {}
    resumed = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                      api_mode="chat_completions", model=MODEL, platform="cli", quiet_mode=True, skip_memory=True,
                      save_trajectories=False, enabled_toolsets=["file"], session_id="resumed-parent", **kw)
    db = MagicMock()
    db.db_path = launch / "state.db"
    db.get_session.return_value = {"system_prompt": stored}
    resumed._session_db = db
    _restore_or_build_system_prompt(resumed, None, [{"role": "user", "content": "hi"}])
    assert resumed._cached_system_prompt == stored
    child = _child(resumed)
    try:
        prompt = _child_prompt(child)
        assert "Fixture rule B" not in prompt
        assert prompt.count(block_a) == 1
    finally:
        child.close()
        resumed.close()


def test_a_child_reset_to_a_new_session_does_not_keep_its_parents_edition(env):
    """Inheritance belongs to the delegated conversation: after a session reset the agent decides from its own inputs
    (an unprofiled child reads the configured file afresh), so the parent's edition is not resurrected from the
    inherited, recorded or still-cached state."""
    launch, core = env
    _config(launch, home_guidance_file=SOURCE)
    parent = _parent()
    block_a = _block(core)
    child = _child(parent)
    try:
        child._cached_system_prompt = _child_prompt(child)
        assert child._cached_system_prompt.count(block_a) == 1
        core.write_text(CORE_B, encoding="utf-8")  # the new conversation's own read must see this edition
        child.session_id = "child-new"
        child.reset_session_state()
        child._invalidate_system_prompt()
        prompt = _child_prompt(child)
        assert block_a not in prompt and prompt.count(_block(core)) == 1
    finally:
        child.close()
        parent.close()
