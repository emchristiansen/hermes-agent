"""The internal consumers that build an ``AIAgent`` with ``skip_context_files=True`` and no SOUL carry the operating
home guidance (``agent.home_guidance_file``) and keep everything else they chose.

Each test drives the real production entry point — ``batch_runner._process_single_prompt`` / ``_process_batch_worker``,
``agent.curator._run_llm_review``, ``feishu_comment._run_comment_agent`` and the CI ``scripts/ci/prompt_surface.py``
renderer — with its own constructor arguments and the real home-guidance reader. Only the model request is replaced
(``_interruptible_api_call`` / ``_interruptible_streaming_api_call``, the provider boundary), sockets are refused and
the OpenRouter model-metadata and models.dev fetches are answered from an empty cache. The world holds a persona
SOUL.md, a project AGENTS.md in the working directory and a MEMORY.md/USER.md; a control agent shows each sentinel does
reach an ordinary prompt, so its absence from a consumer's prompt is meaningful."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.agent.test_home_guidance import SOURCE, _agent, _config, _expected_block, _guidance, _prompt

pytest_plugins = ["tests.agent.test_home_guidance"]  # its ``home`` fixture

MODEL = "anthropic/claude-sonnet-4.6"
PERSONA = "Persona sentinel: I am the configured SOUL."
PROJECT = "Project sentinel: AGENTS.md rule from the working directory."
MEMORY = "Memory sentinel: the user prefers fixtures."
USER = "User-profile sentinel: the user is a fixture."
SENTINELS = (PERSONA, PROJECT, MEMORY, USER)
REPLY = "Fixture reply."


@pytest.fixture
def world(home, tmp_path, monkeypatch):
    """The ``home`` fixture plus every input the consumers suppress, and a configured home-guidance file."""
    (home.hermes / "SOUL.md").write_text(PERSONA + "\n", encoding="utf-8")
    memories = home.hermes / "memories"
    memories.mkdir(exist_ok=True)
    (memories / "MEMORY.md").write_text(MEMORY + "\n", encoding="utf-8")
    (memories / "USER.md").write_text(USER + "\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text(PROJECT + "\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")  # provider resolution reads it; nothing is sent
    _config(home.hermes, agent__home_guidance_file=SOURCE,
            model={"default": MODEL, "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"})
    home.project = project
    return home


@pytest.fixture
def no_network(monkeypatch):
    """Sockets are refused and recorded; the OpenRouter model-metadata and models.dev registry fetches (context window
    and capability lookups) are answered from an empty cache, as the suite's other offline agent tests do."""
    import socket
    from agent import agent_init, model_metadata, models_dev, usage_pricing
    for module in (model_metadata, agent_init, usage_pricing):
        monkeypatch.setattr(module, "fetch_model_metadata", lambda *a, **k: {})
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **k: {})
    attempts = []

    def refuse(sock, address, *args, **kwargs):
        import traceback
        callers = [f"{f.filename.rsplit('/', 2)[-2:]}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[:-1]
                   if "site-packages" not in f.filename and "/lib/python" not in f.filename]
        attempts.append((address, callers[-6:]))  # who tried, for the failure message
        raise ConnectionRefusedError("no network in these tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    return attempts


@pytest.fixture
def model(monkeypatch, no_network):
    """Replaces only the provider request: records the agent and the wire kwargs of each call, answers with text."""
    from run_agent import AIAgent
    calls = []

    def answer(agent, api_kwargs, *args, **kwargs):
        calls.append(SimpleNamespace(
            agent=agent, kwargs=api_kwargs, cached=agent._cached_system_prompt,
            skip_context_files=agent.skip_context_files,
            load_soul_identity=agent.load_soul_identity, memory_store=agent._memory_store,
            enabled_toolsets=list(agent.enabled_toolsets or []), tools=set(agent.valid_tool_names or ()),
            platform=agent.platform, save_trajectories=agent.save_trajectories))
        message = SimpleNamespace(content=REPLY, tool_calls=None, reasoning="Fixture reasoning.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model=MODEL,
                               usage=None)
    monkeypatch.setattr(AIAgent, "_interruptible_api_call", answer)
    monkeypatch.setattr(AIAgent, "_interruptible_streaming_api_call", answer)
    return calls


def _wire_system(call):
    """The system message text the provider request carries."""
    messages = call.kwargs.get("messages") or []
    assert messages and messages[0]["role"] == "system", list(call.kwargs)
    content = messages[0]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def _assert_carries_guidance_only(home, call):
    """Exactly one home-guidance block, the configured file's, in the stable tier the request sends; no persona,
    project or memory text."""
    block = _expected_block(home.core)
    for prompt in (call.cached, _wire_system(call)):
        assert _guidance(prompt) == [block] and prompt.count("<workstation_guidance") == 1
        for sentinel in SENTINELS:
            assert sentinel not in prompt, sentinel
    from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
    assert call.cached.startswith(DEFAULT_AGENT_IDENTITY.strip() + "\n\n" + block)  # default identity, no SOUL
    assert call.skip_context_files is True and not call.load_soul_identity
    assert call.memory_store is None


def _toolset_tools(*names):
    """The tools of *names*, plus the tool-search bridge every agent may expose for deferred tools."""
    from toolsets import resolve_toolset
    from tools.tool_search import BRIDGE_TOOL_NAMES
    return {tool for name in names for tool in resolve_toolset(name)} | set(BRIDGE_TOOL_NAMES)


# ── The fixture is potent: an ordinary agent in the same world sees every sentinel ──────────────────────────────────


def test_control_an_ordinary_agent_sees_persona_project_and_memory(world, no_network):
    agent = _agent(skip_context_files=False, skip_memory=False)
    try:
        prompt = _prompt(agent)
        assert PERSONA in prompt and PROJECT in prompt and MEMORY in prompt and USER in prompt
        assert _guidance(prompt) == [_expected_block(world.core)]
    finally:
        agent.close()
    assert no_network == []


# ── batch_runner.py: _process_batch_worker -> _process_single_prompt -> AIAgent(skip_context_files=True, ...) ─────────


def _batch_config():
    import batch_runner
    config = {key: None for key in batch_runner._AGENT_PASSTHROUGH}
    config.update(model=MODEL, max_iterations=3, distribution="terminal_only", verbose=False, log_prefix_chars=100,
                  base_url="https://openrouter.ai/api/v1", api_key="fixture-key")
    return config


def test_batch_prompt_carries_guidance_and_its_trajectory_does_not(world, model, no_network, tmp_path):
    import batch_runner
    out = tmp_path / "batch-out"
    out.mkdir()
    summary = batch_runner._process_batch_worker((0, [(0, {"prompt": "hello"})], str(out), set(), _batch_config()))
    assert summary["completed_prompts"] == [0] and summary["discarded_no_reasoning"] == 0
    assert len(model) == 1
    call = model[0]
    _assert_carries_guidance_only(world, call)
    # Its own semantics: the sampled toolsets (terminal_only rolls terminal and file at 100%), batch platform,
    # trajectories saved by the runner rather than the agent.
    assert call.enabled_toolsets == ["terminal", "file"] and call.platform == "batch"
    assert "terminal" in call.tools and "read_file" in call.tools and call.tools <= _toolset_tools("terminal", "file")
    assert call.save_trajectories is False
    # The exported row: ``convert_to_trajectory_format`` writes the fixed function-calling system prompt with the tool
    # list, never the agent's system prompt, so no guidance, persona, project or memory text reaches the dataset.
    rows = [json.loads(line) for line in (out / "batch_0.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["toolsets_used"] == ["terminal", "file"]
    exported = json.dumps(rows[0], ensure_ascii=False)
    assert "workstation_guidance" not in exported and "Home core A" not in exported
    for sentinel in SENTINELS:
        assert sentinel not in exported
    conversations = rows[0]["conversations"]
    assert conversations[0]["from"] == "system"
    assert conversations[0]["value"].startswith("You are a function calling AI model.")
    assert [turn["from"] for turn in conversations[1:]] == ["human", "gpt"] and conversations[1]["value"] == "hello"
    assert REPLY in conversations[2]["value"]
    assert not list(Path(world.project).glob("trajectory*.jsonl"))  # the agent itself saved nothing
    assert no_network == []


# ── agent/curator.py: _run_llm_review -> AIAgent(enabled_toolsets=["skills"], platform="curator", ...) ────────────────


def test_curator_review_carries_guidance_and_keeps_its_skills_only_fork(world, model, no_network):
    from agent import curator
    result = curator._run_llm_review("Review the skills.")
    assert result.get("error") in (None, ""), result
    assert result["final"] == REPLY and result["model"] == MODEL
    assert len(model) == 1
    call = model[0]
    _assert_carries_guidance_only(world, call)
    assert call.enabled_toolsets == ["skills"] and call.platform == "curator"
    assert call.tools <= _toolset_tools("skills") and "terminal" not in call.tools
    agent = call.agent
    assert agent._memory_nudge_interval == 0 and agent._skill_nudge_interval == 0
    assert agent._memory_write_origin == "background_review"
    assert no_network == []


# ── plugins/platforms/feishu/feishu_comment.py: _run_comment_agent -> AIAgent(enabled_toolsets=[feishu_doc, ...]) ─────


def test_feishu_comment_agent_carries_guidance_and_its_saved_history_does_not(world, model, no_network, monkeypatch):
    from plugins.platforms.feishu import feishu_comment
    monkeypatch.setattr(feishu_comment, "_session_cache", {})
    key = feishu_comment._session_key("docx", "fixture-token")
    response = feishu_comment._run_comment_agent("Please summarise.", client=None, session_key=key)
    assert response == REPLY
    assert len(model) == 1
    call = model[0]
    _assert_carries_guidance_only(world, call)
    assert call.enabled_toolsets == ["feishu_doc", "feishu_drive"]
    assert call.tools <= _toolset_tools("feishu_doc", "feishu_drive")
    # Cross-card history keeps user/assistant turns only: the system prompt (and its guidance) is never stored.
    saved = feishu_comment._session_cache[key]["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert "workstation_guidance" not in json.dumps(saved, ensure_ascii=False)
    from tools import feishu_doc_tool
    assert feishu_doc_tool.get_client() is None  # the lark client is released after the run
    assert no_network == []


# ── scripts/ci/prompt_surface.py: _render_surface -> AIAgent(skip_context_files=True, ...) on a stock install ─────────


def _prompt_surface_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "ci" / "prompt_surface.py"
    spec = importlib.util.spec_from_file_location("prompt_surface_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_prompt_surface_renders_a_configured_file_once_and_the_stock_snapshot_has_none(world, no_network):
    surface = _prompt_surface_module()
    (world.os_home / "plain").mkdir()
    tiers, _tools = surface._render_surface(world.os_home, "cli", surface.MODELS["claude"], False)
    block = _expected_block(world.core)
    assert _guidance(tiers["stable"]) == [block] and _guidance(tiers["context"] + tiers["volatile"]) == []
    rendered = "".join(tiers.values())
    assert PERSONA not in rendered and PROJECT not in rendered
    # The renderer does not pass skip_memory: its agent keeps the memory store, exactly as before this feature
    # (a stock install's memory is empty, so the committed snapshot holds none).
    assert MEMORY in rendered and USER in rendered
    # The renderer's stock install sets no ``agent.home_guidance_file`` (no default exists), so the committed
    # snapshot every PR is diffed against carries no block.
    from hermes_cli.config import DEFAULT_CONFIG
    assert "home_guidance_file" not in (DEFAULT_CONFIG.get("agent") or {})
    snapshot = sorted((surface.SNAPSHOT / "prompts").glob("*.txt"))
    assert snapshot and not any("workstation_guidance" in p.read_text(encoding="utf-8-sig") for p in snapshot)
    assert no_network == []


def test_ci_prompt_surface_render_child_reads_an_empty_managed_directory(monkeypatch, tmp_path):
    """``render()`` itself, not ``_render_surface``: the child it starts gets a fresh environment, where nothing marks
    a test run, so it would read the host's ``/etc/hermes``. It names an existing empty managed directory instead, and
    a real interpreter given that exact environment resolves no managed scope and no managed setting. (The child's
    full render is stopped after that check; the committed-snapshot test runs it whole.)"""
    import os
    import subprocess
    import sys
    surface = _prompt_surface_module()
    seen, real_run = {}, subprocess.run

    def run(args, *, env, cwd, **kwargs):
        managed = Path(env["HERMES_MANAGED_DIR"])
        seen["managed"], seen["listing"] = managed, sorted(os.listdir(managed))
        probe = real_run(
            [sys.executable, "-c", "import json; from hermes_cli.managed_scope import get_managed_dir, "
             "apply_managed_overlay; print(json.dumps([str(get_managed_dir()), apply_managed_overlay({})]))"],
            env=env, cwd=cwd, capture_output=True, text=True, timeout=120, check=False)
        seen["probe"] = (probe.returncode, probe.stdout, probe.stderr[-2000:])
        return subprocess.CompletedProcess(args, 3, "", "stopped by the test after the managed-scope check")
    monkeypatch.setattr(surface.subprocess, "run", run)
    assert surface.render(tmp_path / "out") == 3
    assert seen["managed"] != Path("/etc/hermes") and seen["listing"] == []
    code, out, err = seen["probe"]
    assert code == 0, err
    assert json.loads(out.strip().splitlines()[-1]) == [str(seen["managed"]), {}]
