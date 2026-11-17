"""`--skip-memory` review t_3c43d141 (SKM-02/03/04): the session opt-out survives the legacy background agents into
their delegates, and no automatic path loads the configured memory provider for a suppressed session. Sessions without
the flag behave as before (the provider identity still keys the cached agent; namespaced skills still load).

The real handlers run: classic-CLI ``/bg`` (``_handle_background_command``), TUI ``prompt.background`` and
``preview.restart``, the gateway's ``_resolve_turn_agent`` cache key, and ``_auto_load_parts`` through the real skill
reader. Only the model boundary (AIAgent), the provider loader and plugin inventory are replaced: no provider runs."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.gateway.test_skip_memory_session_policy import _delegate_child, _parent, _policy, _tool


def _side_agent_class(built):
    """The model boundary: records its kwargs and carries a parent-shaped surface so the real delegate builder can
    run on the constructed background agent itself."""

    class SideAgent:
        def __init__(self, **kwargs):
            built.append(self)
            self.kwargs = kwargs
            vars(self).update(vars(_parent()))
            vars(self).update({k: v for k, v in kwargs.items() if v is not None})
            self.tools, self.valid_tool_names = [_tool('memory'), _tool('terminal')], {'memory', 'terminal'}
            self.compression_checkpoint_required = True  # what _build_context_engine installs under the config
            self._print_fn = self.thinking_callback = None

        def run_conversation(self, **kwargs):
            return {'final_response': 'ok', 'messages': []}

        def close(self):
            pass
    return SideAgent


def _assert_side_agent(bg, skip, checkpoint_required):
    """The background agent and the child it would delegate to."""
    captured, child = _delegate_child(bg, checkpoint_required)
    if skip:
        assert bg._session_skip_memory is True and bg.compression_checkpoint_required is False
        assert bg.valid_tool_names == {'terminal'}
        assert captured['skip_memory'] is True and child._session_skip_memory is True
    else:  # an ordinary session: unchanged; its child inherits the provider only under an armed gate
        assert getattr(bg, '_session_skip_memory', False) is False and 'memory' in bg.valid_tool_names
        # (tests/agent/test_fork_memory_under_checkpoint_gate.py); the opt-out branch above stays True.
        assert captured['skip_memory'] is (not checkpoint_required)


@pytest.mark.parametrize('checkpoint_required', [True, False])
@pytest.mark.parametrize('skip', [True, False], ids=['skip-memory-launch', 'ordinary-launch'])
def test_classic_cli_background_agent_carries_the_opt_out_to_its_delegate(monkeypatch, skip, checkpoint_required):
    import cli as cli_module
    from tests.hermes_cli.test_cli_approval_ui import _make_background_cli_stub
    built = []
    cli = _make_background_cli_stub()
    cli.skip_memory = skip
    with patch('run_agent.AIAgent', _side_agent_class(built)), patch.object(cli_module, '_cprint'), \
            patch.object(cli_module, 'ChatConsole'):
        cli._handle_background_command('/bg inspect workspace')
        for thread in list(cli._background_tasks.values()):
            thread.join(timeout=10)
    [bg] = built
    assert bg.kwargs.get('skip_memory', False) is skip
    _assert_side_agent(bg, skip, checkpoint_required)


def _tui(monkeypatch, skip):
    from tui_gateway import server
    parent = SimpleNamespace(model='m', _fallback_chain=[], enabled_toolsets=['file'], disabled_toolsets=['x'],
                             reasoning_config={}, service_tier='normal',
                             **({'_session_skip_memory': True} if skip else {}))
    session = {'agent': parent, 'history': []}
    for name, value in (('_load_cfg', dict), ('_get_db', lambda: None), ('_emit', lambda *a, **k: None),
                        ('_load_reasoning_config', lambda *a: None), ('_sess', lambda params, rid: (session, None)),
                        ('_preview_restart_callbacks', lambda *a: {}),
                        ('_spawn_side_agent', lambda rid, session, task_id, parent, event, body, **kw: {'text': body()})):
        monkeypatch.setattr(server, name, value)
    return server


@pytest.mark.parametrize('checkpoint_required', [True, False])
@pytest.mark.parametrize('skip', [True, False], ids=['skip-memory-session', 'ordinary-session'])
@pytest.mark.parametrize('method, params', [('prompt.background', {'text': 'go'}),
                                            ('preview.restart', {'url': 'http://127.0.0.1:1/'})])
def test_tui_side_agents_carry_the_opt_out_to_their_delegates(monkeypatch, skip, checkpoint_required, method, params):
    server = _tui(monkeypatch, skip)
    built = []
    monkeypatch.setattr('run_agent.AIAgent', _side_agent_class(built))
    reply = server._methods[method]('rid', {'session_id': 's', **params})
    assert reply == {'text': 'ok'}, reply
    [bg] = built
    if method == 'prompt.background':
        assert bg.kwargs.get('skip_memory', False) is skip
    _assert_side_agent(bg, skip, checkpoint_required)


# -- SKM-03: the in-process cache key -----------------------------------------------------------------------------

def _resolve_key(tmp_path, monkeypatch, memo, **params):
    """Real ``_resolve_turn_agent`` with a real frozen policy, up to its cache key: the provider loads it caused,
    the identity hooks it called, and the key's memory entries."""
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.session_execution import GatewaySessionAgentMixin
    import plugins.memory

    class Stop(BaseException):
        pass

    loads, hooks, keys = [], [], []

    class Provider:
        def identity_signature(self):
            hooks.append('hindsight')
            return {'bank': 'b1'}

    class Cache(GatewayAgentCacheMixin):
        _CACHE_BUSTING_CONFIG_KEYS = ()
        _MEMORY_IDENTITY_PROVIDER_MEMO = {'hindsight': Provider()} if memo == 'warm' else {}

    def load(name, **kwargs):
        loads.append(name)
        return Provider()

    def stop(*args, cache_keys, **kwargs):
        keys.append({k: v for k, v in cache_keys.items() if k.startswith('memory.')})
        raise Stop()
    policy = _policy(tmp_path, **params)
    monkeypatch.setattr(plugins.memory, 'load_memory_provider', load)
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda *a: policy)
    host = object.__new__(type('Host', (GatewaySessionAgentMixin,), {}))
    host._ctx = SimpleNamespace(user_config={'memory': {'provider': 'hindsight'}}, enabled_toolsets=[],
                                source=SimpleNamespace())
    host._runner = SimpleNamespace(_agent_config_signature=stop, _extract_cache_busting_config=Cache._extract_cache_busting_config)
    with pytest.raises(Stop):
        host._resolve_turn_agent({'model': 'm', 'runtime': {}}, 'cli', None, 3, None, {})
    return loads, hooks, keys[0]


@pytest.mark.parametrize('memo', ['cold', 'warm'])
@pytest.mark.parametrize('params', [{'skip_memory': True}, {'ignore_rules': True}])
def test_a_suppressed_turn_neither_loads_nor_asks_the_provider_for_its_cache_key(tmp_path, monkeypatch, memo, params):
    loads, hooks, key = _resolve_key(tmp_path, monkeypatch, memo, **params)
    assert loads == [] and hooks == [] and key == {}


@pytest.mark.parametrize('memo', ['cold', 'warm'])
def test_an_ordinary_turn_still_keys_its_cached_agent_on_the_provider_identity(tmp_path, monkeypatch, memo):
    loads, hooks, key = _resolve_key(tmp_path, monkeypatch, memo)
    assert loads == ([] if memo == 'warm' else ['hindsight']) and hooks == ['hindsight']
    assert key == {'memory.bank': 'b1'}


# -- SKM-04: skills.auto_load ------------------------------------------------------------------------------------

def _auto_load(tmp_path, monkeypatch, skill, session_skip_memory, registered=None):
    """Real ``_auto_load_parts`` -> ``build_auto_load_prompt`` -> skill reader, with ``hindsight`` the active
    provider. ``registered``: a plugin skill file the registry already holds. Returns (provider loads, prompt)."""
    from agent.system_prompt import _auto_load_parts
    import hermes_cli.plugins
    import plugins.memory
    import tools.skills_tool
    monkeypatch.setattr(tools.skills_tool, '_memory_suppressed_sessions', set())  # per-test process state
    cfg = {'memory': {'provider': 'hindsight'}, 'skills': {'auto_load': [skill]}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: cfg)
    monkeypatch.setattr(hermes_cli.plugins, 'discover_plugins', lambda: None)
    namespace, _, bare = skill.partition(':')
    entries = {skill: {'path': registered, 'plugin': namespace, 'plugin_key': namespace, 'plugin_name': namespace,
                       'portable': False, 'bare_name': bare, 'description': 'd', 'frontmatter': {}}} if registered else {}
    pm = SimpleNamespace(_plugins={}, _plugin_skills=entries,
                         find_plugin_skill=lambda name: (entries.get(name) or {}).get('path'),
                         list_plugin_skills=lambda ns: [e['bare_name'] for q, e in entries.items() if q.startswith(ns + ':')],
                         remove_plugin_skill=lambda name: None)
    monkeypatch.setattr(hermes_cli.plugins, 'get_plugin_manager', lambda: pm)
    # receiver _resolve_plugin_skill has no legacy portable-name alias step to neutralize (r3 F6, ADAPTATIONS.md B1)
    monkeypatch.setattr(plugins.memory, '_get_active_memory_provider', lambda: 'hindsight')
    monkeypatch.setattr(plugins.memory, '_prune_inactive_memory_provider_skills', lambda *a: None)
    loads = []
    monkeypatch.setattr(plugins.memory, 'load_memory_provider', lambda name, **kw: loads.append(name))
    agent = SimpleNamespace(skip_context_files=False, valid_tool_names={'skill_view'}, session_id='s',
                            _context_home=str(tmp_path), **({'_session_skip_memory': True} if session_skip_memory else {}))
    return loads, _auto_load_parts(agent)


@pytest.mark.parametrize('session_skip_memory, expected', [(True, []), (False, ['hindsight'])],
                         ids=['skip-memory-session', 'ordinary-session'])
def test_auto_load_never_loads_a_suppressed_provider_for_a_namespaced_skill(tmp_path, monkeypatch,
                                                                             session_skip_memory, expected):
    loads, _ = _auto_load(tmp_path, monkeypatch, 'hindsight:review-example', session_skip_memory)
    assert loads == expected
    from tools.skills_tool import memory_provider_loads_suppressed
    assert memory_provider_loads_suppressed.get() is False, 'the suppression ends with the auto-load'


def test_auto_load_of_a_registered_namespaced_skill_still_works_under_suppression(tmp_path, monkeypatch):
    """Skills stay: a provider-namespaced skill already in the registry is served without loading the provider."""
    skill = tmp_path / 'review-example' / 'SKILL.md'
    skill.parent.mkdir()
    skill.write_text('---\nname: review-example\ndescription: d\n---\nREVIEW-EXAMPLE-BODY\n')
    loads, parts = _auto_load(tmp_path, monkeypatch, 'hindsight:review-example', True, registered=skill)
    assert loads == [] and parts and 'REVIEW-EXAMPLE-BODY' in parts[0], parts
