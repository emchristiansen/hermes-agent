"""SKM-07 (review t_f73a4309/run19612 on 082001b9): the classic CLI's ``--skip-memory`` is the gateway session's
policy. ``_init_agent`` builds a ``GatewayAgent`` (no memory store; read-only ``tools``/``valid_tool_names``
mirroring the session's surface) and must not apply the opt-out to it: that raised ``AttributeError`` for any
nonempty tool surface. The launch carries ``skip_memory`` and the gateway applies it to the agent that runs the
turn (``session_execution``). Real ``HermesCLI._init_agent``, real ``GatewayAgent``, a scripted connection."""
from types import SimpleNamespace

import pytest

from tests.hermes_cli import test_gateway_agent as gateway_agent_tests

make_agent = gateway_agent_tests.make_agent  # the fixture: a GatewayAgent over a scripted connection


def _first_use(make_agent, monkeypatch, groups, skip):
    import cli as cli_module
    from hermes_cli import cli_agent_setup_mixin as setup
    adapter, _client = make_agent([])
    adapter.connection.state(adapter.session_id).info = {"tools": groups}
    store = SimpleNamespace(connection=adapter.connection)
    monkeypatch.setattr(cli_module.HermesCLI, "_init_session_store", lambda self: setattr(self, "_session_db", store))
    monkeypatch.setattr(setup, "_current_runtime",
                        lambda self: {"provider": "openrouter", "api_key": "fixture", "base_url": "https://example.invalid"})
    shell = cli_module.HermesCLI(model="fixture", provider="openrouter", api_key="fixture",
                                 base_url="https://example.invalid", compact=True, skip_memory=skip)
    shell.session_id = adapter.session_id
    shell._resumed, shell.conversation_history = False, []
    shell._install_tool_callbacks = lambda: None
    shell._ensure_runtime_credentials = lambda: True
    shell._ensure_gateway_session = lambda: None
    assert shell._init_agent() is True
    return shell


@pytest.mark.parametrize("skip", [False, True], ids=["ordinary", "skip-memory"])
@pytest.mark.parametrize("groups", [{"terminal": ["terminal"]}, {"terminal": ["terminal"], "memory": ["memory"]}, {}],
                         ids=["terminal-only", "with-memory-toolset", "empty"])
def test_classic_cli_first_use_builds_the_gateway_agent_under_skip_memory(make_agent, monkeypatch, groups, skip):
    shell = _first_use(make_agent, monkeypatch, groups, skip)
    expected = {name for members in groups.values() for name in members}
    assert shell.agent.valid_tool_names == expected  # the session's own surface, reported as the gateway sent it
    assert getattr(shell.agent, "_session_skip_memory", False) is False  # the client agent is never the owner


@pytest.mark.parametrize("skip", [False, True])
def test_the_launch_carries_skip_memory_to_the_gateway_session(skip):
    from hermes_cli.cli_init_mixin import _gateway_launch
    assert _gateway_launch(SimpleNamespace(skip_memory=skip))["skip_memory"] is (True if skip else None)
