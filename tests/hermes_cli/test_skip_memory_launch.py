"""`--skip-memory`: a per-launch switch (typed only; no environment equivalent) that runs THIS session's agents
with AIAgent ``skip_memory=True`` (no memory provider, no MEMORY.md/USER.md) while SOUL, AGENTS.md, config and
skills load as without it. It travels like a typed ``--ignore-rules``: parser -> HermesCLI -> ``session.create``
policy (classic CLI, thin client) and is checked on resume. The Ink TUI and ACP, which carry launch flags only
through the environment, refuse it rather than silently running with memory."""
from __future__ import annotations

from contextlib import asynccontextmanager
import sys
import types
from unittest.mock import patch

import pytest

from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_gateway_session_store import FakeClient, _frozen_launch, cli


@pytest.fixture
def store():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


# -- parser ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [["--skip-memory"], ["--skip-memory", "chat"], ["chat", "--skip-memory"]])
def test_parser_accepts_skip_memory_on_hermes_and_hermes_chat(argv):
    from hermes_cli._parser import build_top_level_parser
    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(argv)
    assert args.skip_memory is True
    assert getattr(args, "ignore_rules", False) is False


def test_parser_default_is_off_and_chat_does_not_reset_a_top_level_flag():
    from hermes_cli._parser import build_top_level_parser
    parser, _subparsers, _chat = build_top_level_parser()
    assert parser.parse_args([]).skip_memory is False
    assert getattr(parser.parse_args(["chat"]), "skip_memory", False) is False
    # chat's SUPPRESS default never resets the parent's parsed value.
    assert parser.parse_args(["--skip-memory", "chat"]).skip_memory is True


def _run_cmd_chat(monkeypatch, argv):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser
    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(argv)
    captured = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: captured.update(kwargs)
    fake_banner = types.ModuleType("hermes_cli.banner")
    fake_banner.prefetch_update_check = lambda: None
    fake_skills_sync = types.ModuleType("tools.skills_sync")
    fake_skills_sync.sync_skills = lambda quiet=True: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
    monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    main_mod.cmd_chat(args)
    return captured


def test_cmd_chat_passes_skip_memory_to_the_classic_cli(monkeypatch):
    captured = _run_cmd_chat(monkeypatch, ["chat", "--skip-memory"])
    assert captured["skip_memory"] is True
    assert captured.get("ignore_rules") is False  # rules, SOUL and skills still load


def test_cmd_chat_without_the_flag_is_unchanged(monkeypatch):
    captured = _run_cmd_chat(monkeypatch, ["chat"])
    assert "skip_memory" not in captured


def test_acp_and_the_tui_refuse_skip_memory_rather_than_run_with_memory():
    from hermes_cli.main_agent_cmds import cmd_acp
    from hermes_cli.main_tui_launch import _refuse_unforwarded_tui_flags
    with pytest.raises(SystemExit, match="--skip-memory"):
        cmd_acp(types.SimpleNamespace(accept_hooks=False, ignore_rules=False, skip_memory=True))
    with pytest.raises(SystemExit):
        _refuse_unforwarded_tui_flags(types.SimpleNamespace(skip_memory=True))
    _refuse_unforwarded_tui_flags(types.SimpleNamespace(skip_memory=False))


# -- thin client -------------------------------------------------------------------------------------

def test_thin_client_carries_a_typed_skip_memory_into_session_create():
    from hermes_cli.gateway_chat import _requested_policy, env_launch_flags
    assert "skip_memory" not in env_launch_flags()  # no environment equivalent
    assert _requested_policy(types.SimpleNamespace(skip_memory=True)) == {"skip_memory": True}
    assert _requested_policy(types.SimpleNamespace(skip_memory=False)) == {}


# -- classic CLI (HermesCLI -> gateway session) -----------------------------------------------------

def _host(typed=False, resumed=False):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    h = cli("local-old", resumed=resumed)
    CLIInitMixin._init_checkpoints_and_rules(h, False, False, False, typed)
    return h


def test_cli_attribute_reads_the_typed_flag():
    assert _host().skip_memory is False and _host()._explicit_skip_memory is False
    typed = _host(typed=True)
    assert typed.skip_memory is True and typed._explicit_skip_memory is True and typed.ignore_rules is False


def test_cli_launch_policy_carries_skip_memory_into_session_create(store):
    from hermes_cli.cli_init_mixin import _gateway_launch
    db, client = store
    launch = _gateway_launch(_host(typed=True))
    assert launch["skip_memory"] is True and launch["ignore_rules"] is None
    db.open_session(cli(), launch)
    params = [p for m, p in client.calls if m == "session.create"][-1]
    assert params.get("skip_memory") is True and "ignore_rules" not in params
    # Without the flag the create request is exactly as before.
    db.open_session(cli(), _gateway_launch(_host()))
    assert "skip_memory" not in [p for m, p in client.calls if m == "session.create"][-1]


def test_a_typed_skip_memory_on_a_session_created_without_it_is_refused(store):
    """The frozen policy is kept: resuming an ordinary session with --skip-memory is refused, never applied."""
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_client import GatewayClientError
    db, client = store
    _frozen_launch(client, {"model": "m-frozen"})
    bare = _host(resumed=True)
    assert GatewaySessionStore(db.connection).open_session(bare, _gateway_launch(bare)) == "local-old"
    typed = _host(typed=True, resumed=True)
    with pytest.raises(GatewayClientError, match="--skip-memory"):
        GatewaySessionStore(db.connection).open_session(typed, _gateway_launch(typed))


def test_typed_skip_memory_repeated_on_a_skip_memory_session_resumes(store):
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_session_store import GatewaySessionStore
    db, client = store
    _frozen_launch(client, {"model": "m-frozen", "skip_memory": True})
    typed = _host(typed=True, resumed=True)
    assert GatewaySessionStore(db.connection).open_session(typed, _gateway_launch(typed)) == "local-old"


def test_thin_client_resume_checks_a_typed_skip_memory():
    from hermes_cli.gateway_chat import check_resume_policy
    from hermes_cli.gateway_client import GatewayClientError
    frozen = {"info": {"launch_request": {"model": "m"}}}
    with pytest.raises(GatewayClientError, match="--skip-memory"):
        check_resume_policy(types.SimpleNamespace(skip_memory=True), frozen)
    check_resume_policy(types.SimpleNamespace(skip_memory=True),
                        {"info": {"launch_request": {"model": "m", "skip_memory": True}}})
    check_resume_policy(types.SimpleNamespace(skip_memory=False), frozen)


# -- TUI background agents of a gateway session ----------------------------------------------------

def test_tui_background_agent_of_a_skip_memory_session_skips_memory():
    # server.py rebinds the split helpers onto its own namespace (tui_gateway/method_ctx.py).
    from tui_gateway.server import _background_agent_kwargs
    with patch("tui_gateway.server._load_cfg", return_value={}), \
            patch("tui_gateway.server._get_db", return_value=None), \
            patch("tui_gateway.server._load_reasoning_config", return_value=None):
        parent = types.SimpleNamespace(model="m", _fallback_chain=[], enabled_toolsets=["file"],
                                       disabled_toolsets=["x"], reasoning_config={"effort": "low"},
                                       service_tier="normal")
        assert "skip_memory" not in _background_agent_kwargs(parent, "bg_1")
        parent._session_skip_memory = True
        assert _background_agent_kwargs(parent, "bg_1")["skip_memory"] is True
