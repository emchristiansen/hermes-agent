"""SKM-05/06: an explicit --skip-memory policy belongs to the live agent that declared it, not to whichever prompt
producer last ran for its session id, and it follows that agent's CURRENT session id.

* SKM-05: a routed internal fork (``agent.background_review.build_cache_parity_fork``) takes its parent's session id
  without the explicit marker; its autoload (empty or not) must not revoke the live parent's suppression.
* SKM-06: a rotating compaction (``agent.conversation_compression._publish_rotated_compaction``) or a live-child
  adoption assigns the agent a new session id; the cached autoload never re-registers, and the next dispatch names
  only the new id. It stays suppressed.

Real fork builder, real rotation publisher + carry helper, real ``_auto_load_parts`` (cached), real registered
``skill_view`` via ``tools.registry.registry.dispatch``; only the provider loader, plugin inventory, model
construction and the session DB are inert fixtures. Ordinary and distinct sessions keep loading the provider.
"""
from types import SimpleNamespace as NS

import pytest

from tests.gateway import test_skip_memory_skm04_boundaries as skm04

SKILL, _view = skm04.SKILL, skm04._view
provider = skm04.provider  # the SKM-04 provider fixture (inert loader, plugin registry), shared


def _agent(session_id):
    return NS(session_id=session_id, tools=[], valid_tool_names={'skill_view'}, skip_context_files=False,
              _cached_system_prompt=None, _cached_system_prompt_static=None, _memory_manager=None,
              context_compressor=NS(), model='fixture-model', provider='fixture-provider', platform='cli',
              _session_db=None, _memory_store=None, _memory_enabled=False, _user_profile_enabled=False)


def _explicit(session_id):
    from agent.agent_init_skip_memory import apply_session_skip_memory
    from agent.system_prompt import _auto_load_parts
    agent = _agent(session_id)
    apply_session_skip_memory(agent)
    _auto_load_parts(agent)
    return agent


@pytest.mark.parametrize('autoload', [[], [SKILL]], ids=['empty-autoload', 'nonempty-autoload'])
def test_a_routed_internal_fork_never_revokes_its_live_parents_policy(provider, monkeypatch, autoload):
    import hermes_cli.config
    import run_agent
    import agent.background_review as br
    from agent.system_prompt import _auto_load_parts
    cfg = hermes_cli.config.load_config_readonly()
    monkeypatch.setattr(hermes_cli.config, 'load_config_readonly', lambda: {**cfg, 'skills': {'auto_load': autoload}})
    parent = _explicit('live-parent')
    _view(SKILL, task_id='fresh-turn', session_id=parent.session_id)
    assert provider.loads == []
    monkeypatch.setattr(run_agent, 'AIAgent', lambda **kw: _agent('generated-fork-id'))
    monkeypatch.setattr(br, '_resolve_review_runtime', lambda *a: {'routed': True})
    monkeypatch.setattr(br, '_routed_reasoning_config', lambda *a: None)
    monkeypatch.setattr(br, '_detach_fork_compression', lambda *a: None)
    monkeypatch.setattr(br, '_review_input_token_budget', lambda *a: 100)
    fork, _, routed = br.build_cache_parity_fork(parent, {'model': 'different'}, max_iterations=4)
    assert routed and fork.session_id == parent.session_id
    assert fork._session_skip_memory is True  # the fork carries its parent's explicit policy
    del fork._session_skip_memory  # and even an UNMARKED producer for the shared id revokes nothing
    _auto_load_parts(fork)
    _view(SKILL, task_id='fresh-turn', session_id=parent.session_id)
    _view(SKILL, task_id=parent.session_id)
    assert parent._session_skip_memory is True and provider.loads == []


def test_a_rotating_compaction_keeps_the_policy_on_the_new_session_id(provider, monkeypatch):
    import agent.conversation_compression as cc
    from agent.system_prompt import _auto_load_parts
    agent = _explicit('archived-parent')
    published = []
    agent._session_db = NS(get_active_message_watermark=lambda _: 0, get_session_title=lambda _: None,
                           publish_compression_child=lambda **kw: published.append(kw))
    agent._flush_messages_to_session_db = lambda *a, **kw: None
    agent._session_init_model_config = {}
    monkeypatch.setattr(cc, '_parent_deliberately_ended', lambda *a: False)
    monkeypatch.setattr(cc, 'mint_session_id', lambda: 'continuation-child')
    monkeypatch.setattr(cc, '_compression_child_source', lambda *a: 'cli')
    monkeypatch.setattr(cc, '_rebind_session_context', lambda *a: None)
    monkeypatch.setattr('agent.compression_automation.carry_compression_automation', lambda *a: None)
    monkeypatch.setattr('agent.anthropic_thinking_replay.carry_rejected_thinking_to_session', lambda *a: None)
    cc._publish_rotated_compaction(agent, [], [], new_system_prompt='fixture',
                                   lease=NS(holder=None, ttl=10, watermark=None), old_session_id='archived-parent',
                                   compressed_user_turn_outcome='absent')
    assert agent.session_id == 'continuation-child' and published[0]['child_session_id'] == 'continuation-child'
    _auto_load_parts(agent)  # resolved cache: registers nothing new
    _view(SKILL, task_id=agent.session_id, session_id=agent.session_id)
    _view(SKILL, task_id=agent.session_id)
    assert provider.loads == []


def test_a_live_child_adoption_keeps_the_policy_on_the_adopted_id(provider, monkeypatch):
    """The real ``_adopt_live_compression_child`` moves a stale contender onto the live continuation tip."""
    from agent import conversation_compression as cc, conversation_loop
    memoryless = _explicit('pre-adoption')
    memoryless.context_compressor = NS()

    class DB:  # the adoption reads its DB through the type: an inert lineage with one live tip
        def get_compression_tip(self, session_id):
            return 'adopted-child'

        def get_session(self, session_id):
            return {'id': session_id, 'ended_at': None, 'system_prompt': None}

        def get_messages_as_conversation(self, session_id):
            return [{'role': 'user', 'content': 'handoff'}]
    monkeypatch.setattr(cc, '_rebind_session_context', lambda *a: None)
    monkeypatch.setattr(cc, '_hand_off_metrics_segment', lambda *a: None)
    monkeypatch.setattr(conversation_loop, '_stored_prompt_matches_runtime', lambda *a: False)
    assert cc._adopt_live_compression_child(memoryless, DB(), 'pre-adoption')
    assert memoryless.session_id == 'adopted-child'
    _view(SKILL, task_id='adopted-child', session_id='adopted-child')
    assert provider.loads == []


def test_ordinary_and_distinct_sessions_still_load_and_a_launch_can_declare_an_id_ordinary(provider):
    from agent.system_prompt import _auto_load_parts
    from gateway.session_policy import render_launch_skills
    explicit = _explicit('explicit')
    ordinary = _agent('ordinary')
    _auto_load_parts(ordinary)
    assert provider.loads == ['hindsight']
    _view(SKILL, task_id='distinct', session_id='distinct')
    assert provider.loads == ['hindsight', 'hindsight']
    _view(SKILL, task_id='fresh', session_id=explicit.session_id)
    assert provider.loads == ['hindsight', 'hindsight']
    # A launch is an authoritative declaration: a new ordinary launch of a previously suppressed plain id loads.
    from hermes_state_runtime import RuntimeStoreError
    from tools.skills_tool import suppress_memory_provider_loads_for
    suppress_memory_provider_loads_for('relaunched')
    with pytest.raises(RuntimeStoreError, match='unknown_skill'):
        render_launch_skills({'skip_memory': False, 'skills': [SKILL]}, 'relaunched')
    _view(SKILL, task_id='relaunched')
    assert provider.loads == ['hindsight'] * 5
