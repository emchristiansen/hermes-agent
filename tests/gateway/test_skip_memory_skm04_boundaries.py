"""SKM-04 remainder: a --skip-memory session's typed launch preload and its model-facing (registered) ``skill_view``
never load the configured memory provider to find a namespaced skill. Real ``render_launch_skills`` ->
``build_preloaded_skills_prompt`` -> skill reader, real ``_auto_load_parts`` and the real registered handler via
``tools.registry.registry.dispatch``; only the provider loader and the plugin inventory are fixtures.
Registered provider skills, ordinary sessions (default loading) and context cleanup on success and error stay."""
from types import SimpleNamespace

import pytest

SKILL = 'hindsight:review-example'


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """``hindsight`` is the active provider, not loaded; ``entries`` is the plugin skill registry it would fill."""
    import hermes_cli.plugins
    import plugins.memory
    import tools.skills_tool as st
    monkeypatch.setattr(st, '_memory_suppressed_sessions', set())  # per-test process state
    cfg = {'memory': {'provider': 'hindsight'}, 'skills': {'auto_load': [SKILL]}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: cfg)
    monkeypatch.setattr(hermes_cli.plugins, 'discover_plugins', lambda: None)
    entries = {}
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
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    def register(qualified, body):
        namespace, _, bare = qualified.partition(':')
        path = tmp_path / 'registered' / bare / 'SKILL.md'
        path.parent.mkdir(parents=True)
        path.write_text(f'---\nname: {bare}\ndescription: d\n---\n{body}\n')
        entries[qualified] = {'path': path, 'plugin': namespace, 'plugin_key': namespace, 'plugin_name': namespace,
                              'portable': False, 'bare_name': bare, 'description': 'd', 'frontmatter': {}}
    return SimpleNamespace(loads=loads, register=register, st=st)


def _agent(session_id, skip, tmp_path):
    return SimpleNamespace(skip_context_files=False, valid_tool_names={'skill_view'}, session_id=session_id,
                           _context_home=str(tmp_path), **({'_session_skip_memory': True} if skip else {}))


def _view(name, **kw):
    import json
    from tools.registry import registry
    return json.loads(registry.dispatch('skill_view', {'name': name}, **kw))


@pytest.mark.parametrize('skip, loads', [(True, []), (False, ['hindsight', 'hindsight'])], ids=['skip-memory', 'ordinary'])
def test_the_typed_launch_preload_loads_the_provider_only_for_an_ordinary_session(provider, skip, loads):
    from gateway.session_policy import render_launch_skills
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='unknown_skill'):  # the missing skill is still refused
        render_launch_skills({'skip_memory': skip, 'skills': [SKILL]}, 'launch-1')
    assert provider.loads == loads and provider.st.memory_provider_loads_suppressed.get() is False
    # The launched session's later model-facing read is bound by the same opt-out.
    _view(SKILL, task_id='launch-1')
    assert provider.loads == loads + ([] if skip else ['hindsight'])


def test_the_typed_launch_preload_serves_a_registered_provider_skill_under_suppression(provider):
    from gateway.session_policy import render_launch_skills
    provider.register(SKILL, 'REGISTERED-BODY')
    assert 'REGISTERED-BODY' in render_launch_skills({'skip_memory': True, 'skills': [SKILL]}, 'launch-2')
    assert provider.loads == []


@pytest.mark.parametrize('identity', [{'task_id': 'sess-x'}, {'task_id': 'turn-7', 'session_id': 'sess-x'}],
                         ids=['task-id', 'session-id'])
def test_the_registered_skill_view_after_a_suppressed_prompt_never_loads_the_provider(provider, tmp_path, identity):
    from agent.system_prompt import _auto_load_parts
    _auto_load_parts(_agent('sess-x', True, tmp_path))
    result = _view(SKILL, **identity)
    assert provider.loads == [] and not result.get('success'), result  # an honest missing-skill answer
    assert provider.st.memory_provider_loads_suppressed.get() is False
    provider.register(SKILL, 'REGISTERED-BODY')  # a registered provider skill is still served
    assert 'REGISTERED-BODY' in _view(SKILL, **identity)['content'] and provider.loads == []


def test_an_ordinary_session_still_loads_the_provider_even_after_a_suppressed_one(provider, tmp_path):
    """Distinct ordinary sessions keep loading. For the SAME id, an unmarked prompt producer (e.g. an internal fork
    sharing a live parent's id) declares nothing (SKM-05); only a launch declares an id ordinary again."""
    from agent.system_prompt import _auto_load_parts
    from gateway.session_policy import render_launch_skills
    from hermes_state_runtime import RuntimeStoreError
    _auto_load_parts(_agent('sess-x', True, tmp_path))
    _view(SKILL, task_id='other-session')
    assert provider.loads == ['hindsight']  # default behaviour for a session that did not opt out
    _auto_load_parts(_agent('sess-x', False, tmp_path))  # an unmarked producer for the same id
    _view(SKILL, task_id='sess-x')
    assert provider.loads == ['hindsight']
    with pytest.raises(RuntimeStoreError, match='unknown_skill'):
        render_launch_skills({'skip_memory': False, 'skills': [SKILL]}, 'sess-x')  # a new ordinary launch
    assert provider.loads == ['hindsight'] * 3  # the ordinary launch preload looks the provider up
    _view(SKILL, task_id='sess-x')
    assert provider.loads == ['hindsight'] * 4


def test_a_general_plugin_skill_is_served_under_suppression(provider, tmp_path):
    from agent.system_prompt import _auto_load_parts
    _auto_load_parts(_agent('sess-x', True, tmp_path))
    provider.register('other-plugin:tool-guide', 'GENERAL-BODY')
    assert 'GENERAL-BODY' in _view('other-plugin:tool-guide', task_id='sess-x')['content'] and provider.loads == []


@pytest.mark.parametrize('initial', [False, True])
def test_the_suppression_is_restored_when_either_reader_fails(provider, tmp_path, monkeypatch, initial):
    import agent.skill_commands
    from gateway.session_policy import render_launch_skills
    st = provider.st

    def fail(*a, **kw):
        assert st.memory_provider_loads_suppressed.get() is True
        raise RuntimeError('fixture-reader-failure')
    token = st.memory_provider_loads_suppressed.set(initial)
    try:
        monkeypatch.setattr(agent.skill_commands, 'build_preloaded_skills_prompt', fail)
        with pytest.raises(RuntimeError, match='fixture-reader-failure'):
            render_launch_skills({'skip_memory': True, 'skills': [SKILL]}, 'launch-3')
        assert st.memory_provider_loads_suppressed.get() is initial
        monkeypatch.setattr(st, 'skill_view', fail)
        with pytest.raises(RuntimeError, match='fixture-reader-failure'):
            st._skill_view_with_bump({'name': SKILL}, task_id='launch-3')
        assert st.memory_provider_loads_suppressed.get() is initial
    finally:
        st.memory_provider_loads_suppressed.reset(token)


def test_the_explicit_session_helper_binds_its_session(provider):
    from agent.agent_init_skip_memory import apply_session_skip_memory
    agent = SimpleNamespace(session_id='helper-sess', task_id='helper-task', tools=[], valid_tool_names=set(),
                            _cached_system_prompt=None, _cached_system_prompt_static=None)
    apply_session_skip_memory(agent)
    _view(SKILL, task_id='helper-task')
    _view(SKILL, task_id='t', session_id='helper-sess')
    assert provider.loads == []
