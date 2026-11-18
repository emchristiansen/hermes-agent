"""`--skip-memory` on the gateway: the frozen session policy carries it, every agent of that session is
built with AIAgent ``skip_memory=True`` (while ``skip_context_files`` stays off and SOUL identity loads),
and no fork of that session loads the memory provider, even under ``compression.checkpoint_required``.
Sessions without the flag, and --ignore-rules / --safe-mode sessions, behave exactly as before."""
from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_init_skip_memory import apply_session_skip_memory


class _ConstructorStop(BaseException):
    """Abort at the agent constructor (BaseException: the sites catch ``except Exception``)."""


def _policy(tmp_path, **params):
    from gateway.session_policy import build_policy
    return build_policy({'cwd': str(tmp_path), 'model': 'm', **params}, {'platform_toolsets': {'cli': ['terminal']}})


# -- frozen policy ---------------------------------------------------------------------------------

def test_skip_memory_is_a_frozen_create_field(tmp_path):
    from gateway.session_policy import CREATE_FIELDS, restore_policy
    assert 'skip_memory' in CREATE_FIELDS
    policy = _policy(tmp_path, skip_memory=True)
    assert policy.skip_memory is True and policy.skip_memory_requested is True
    assert policy.ignore_rules is False  # rules, SOUL, skills untouched
    assert json.loads(policy.request_json)['skip_memory'] is True  # echoed in launch_request for resume checks
    restored = restore_policy(asdict(policy))
    assert restored.skip_memory is True and restored.ignore_rules is False


def test_skip_memory_defaults_off(tmp_path):
    policy = _policy(tmp_path)
    assert policy.skip_memory is False and policy.skip_memory_requested is False
    assert _policy(tmp_path, skip_memory=False).skip_memory is False


@pytest.mark.parametrize('params', [{'ignore_rules': True}, {'safe_mode': True}])
def test_ignore_rules_and_safe_mode_imply_skip_memory(tmp_path, params):
    policy = _policy(tmp_path, **params)
    assert policy.ignore_rules is True and policy.skip_memory is True
    assert policy.skip_memory_requested is False


@pytest.mark.parametrize('bad', ['true', 1, None])
def test_skip_memory_must_be_a_boolean(tmp_path, bad):
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        _policy(tmp_path, skip_memory=bad)


# -- agent constructor sites -----------------------------------------------------------------------

def _execution_agent(policy, monkeypatch):
    """``GatewaySessionAgentMixin._build_fresh_agent`` (the in-process turn) with a capturing AIAgent."""
    from gateway.session_execution import GatewaySessionAgentMixin
    import gateway.run
    captured = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
    monkeypatch.setattr('gateway.session_authorities.active_authority', lambda runner: None)
    monkeypatch.setattr('gateway.session_worker_construct.session_prefill', lambda *a: None)
    monkeypatch.setattr(gateway.run, '_checkpoint_agent_kwargs', lambda cfg: {})
    monkeypatch.setattr('gateway.session_execution.api_runtime_locked', lambda: False)
    host = object.__new__(type('Host', (GatewaySessionAgentMixin,), {}))
    src = SimpleNamespace(user_id='u', user_id_alt=None, user_name=None, chat_id='c', chat_name=None,
                          chat_type='dm', thread_id=None)
    host._ctx = SimpleNamespace(source=src, user_config={}, enabled_toolsets=['terminal'], disabled_toolsets=None,
                                session_id='sid', session_key='key', AIAgent=recorder)
    host._runner = SimpleNamespace(_prefill_messages=None, _service_tier=None, _session_db=None,
                                   _refresh_fallback_model=lambda: None)
    skip_context_files = host._skip_context_files('local')
    agent = host._build_fresh_agent({'model': 'm', 'runtime': {}}, 'local', None, 5, None, {}, skip_context_files)
    return captured, agent


def test_in_process_turn_agent_skips_only_memory(tmp_path, monkeypatch):
    captured, agent = _execution_agent(_policy(tmp_path, skip_memory=True), monkeypatch)
    assert captured['skip_memory'] is True
    assert captured['skip_context_files'] is False and captured['load_soul_identity'] is True
    assert agent._session_skip_memory is True
    # t53: it compacts with no memory checkpoint, even with compression.checkpoint_required on
    assert agent.compression_checkpoint_required is False


@pytest.mark.parametrize('params, skip', [({}, False), ({'ignore_rules': True}, True)])
def test_in_process_turn_agent_unchanged_without_the_flag(tmp_path, monkeypatch, params, skip):
    captured, agent = _execution_agent(_policy(tmp_path, **params), monkeypatch)
    assert captured['skip_memory'] is skip
    assert captured['skip_context_files'] is skip and captured['load_soul_identity'] is (not skip)
    assert not hasattr(agent, '_session_skip_memory')


class _SidePolicy:
    """The frozen policy's memory/rules answers with the route lookups stubbed (``_side_agent``)."""

    def __init__(self, real):
        self.ignore_rules, self.skip_memory = real.ignore_rules, real.skip_memory
        self.skip_memory_requested = real.skip_memory_requested
        self.max_turns, self.platform = 3, 'cli'

    def config(self, authority=None, **_):
        return {}

    def current_toolsets(self, config):
        return ['terminal']

    def current_reasoning_config(self, runner, route):
        return None

    def current_service_tier(self, runner, route):
        return None


@pytest.mark.parametrize('params, skip, marked', [({'skip_memory': True}, True, True), ({}, False, False),
                                                  ({'ignore_rules': True}, True, False)])
def test_side_agent_follows_the_session(tmp_path, monkeypatch, params, skip, marked):
    from gateway import session_tui_side
    import gateway.run
    captured = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(session_tui_side, '_resolve_runtime', lambda authority, policy: ('m', {}))
    monkeypatch.setattr('gateway.session_worker_construct.session_routing', lambda authority, policy: {})
    monkeypatch.setattr(gateway.run, '_checkpoint_agent_kwargs', lambda cfg: {})
    monkeypatch.setattr('run_agent.AIAgent', recorder)
    real = _policy(tmp_path, **params)
    agent = session_tui_side._side_agent(SimpleNamespace(runner=SimpleNamespace(), db=None), _SidePolicy(real),
                                         SimpleNamespace(route='r'), 'bg_1')
    assert captured['skip_memory'] is skip
    assert captured['load_soul_identity'] is (not real.ignore_rules)
    assert getattr(agent, '_session_skip_memory', False) is marked


@pytest.mark.parametrize('skip_memory', [True, False])
def test_managed_worker_agent_follows_the_session(tmp_path, monkeypatch, skip_memory):
    import threading
    from agent import managed_worker as worker
    from tests.gateway.test_managed_turn_contract import assignment
    frame, _ = assignment(tmp_path, monkeypatch)
    request = {**json.loads(frame['policy']['request_json']), **({'skip_memory': True} if skip_memory else {})}
    frame = {**frame, 'policy': {**frame['policy'], 'request_json': json.dumps(request, sort_keys=True)}}
    built, adopted, marked_at_publish = {}, [], []

    class Agent:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.session_prompt_tokens = self.session_completion_tokens = 0
            self.steered = []

        def steer(self, text):
            self.steered.append(text)

        def run_conversation(self, text, **kwargs):
            return {'final_response': 'ok', 'completed': True}

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    controls_made = []

    class Controls:
        # Production publishes the constructed agent through ``WorkerControls.adopt``
        # (agent/managed_worker.py) right before stop-interrupt and ``ready``: under the controls' lock it
        # binds ``controls.agent`` and steers the corrections latched while the agent was being built.
        # This double keeps that contract and records each adopted agent and its marker at that moment.
        def __init__(self, *args):
            self.agent = None
            self.lock = threading.Lock()
            self.latched = ['correction latched during construction']
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
            controls_made.append(self)
        def adopt(self, agent):
            with self.lock:
                adopted.append(agent)
                marked_at_publish.append(getattr(agent, '_session_skip_memory', False))
                self.agent = agent
                for text in self.latched:
                    agent.steer(text)
                self.latched.clear()
        def approval(self, data): pass
        def clarify(self, questions): pass
        def settle(self, result, **kwargs): return None
        def consumed(self, result): return []

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    # execute() owns its worker process: it sets HERMES_GATEWAY_SESSION and the approval session key
    # (a ContextVar) for good. Scope both to this test so later tests keep their own surface.
    import contextvars
    monkeypatch.setenv('HERMES_GATEWAY_SESSION', '1')
    contextvars.copy_context().run(worker.execute, frame, SimpleNamespace(send=lambda kind, **payload: None))
    assert built['skip_memory'] is skip_memory
    assert built['skip_context_files'] is False and built['load_soul_identity'] is True
    assert getattr(adopted[0], '_session_skip_memory', False) is skip_memory
    assert isinstance(adopted[0], Agent) and marked_at_publish[0] is skip_memory  # applied before publish
    # Adopted exactly once, bound to the controls, and the latched correction steered into that agent.
    [controls] = controls_made
    assert len(adopted) == 1 and controls.agent is adopted[0]
    assert adopted[0].steered == ['correction latched during construction'] and controls.latched == []


# -- AIAgent memory init under skip_memory -----------------------------------------------------------

# r4 (B2): every protected seam of receiver ``agent.agent_init._init_memory`` (agent/agent_init.py:1299-1383), in
# call order. The built-in store block (:1326-1341) runs under ``suppress(Exception)`` and the provider block
# (:1350-1380) under ``except Exception`` (:1378-1380 restores ``_memory_manager = None``); both swallow an
# AssertionError, so a forbidding stub alone proves nothing. Each stub records a distinct attempt BEFORE raising.
# Downstream calls (``is_available``, ``add_provider``, ``initialize_for_agent``, ``unavailable_reason``,
# ``_warn_memory_provider_unavailable``) are reachable only through objects these seams return.
_INIT_MEMORY_SEAMS = (
    ('builtin_memory_config', 'tools.memory_tool.get_builtin_memory_config'),        # :1329
    ('builtin_memory_flags', 'tools.memory_tool.get_builtin_memory_store_flags'),    # :1330
    ('builtin_memory_store', 'tools.memory_tool.MemoryStore'),                       # :1335 (+ load_from_disk :1341)
    ('core_provider_check', 'agent.agent_init.is_core_memory_provider'),             # :1353
    ('memory_manager', 'agent.memory_manager.MemoryManager'),                        # :1354/:1356
    ('load_memory_provider', 'plugins.memory.load_memory_provider'),                 # :1355/:1357 (+ :1362)
    ('recover_at_startup', 'hermes_cli.memory_provider_migration.recover_at_startup'),  # :1360/:1361
)
_SEAM_NAMES = [name for name, _ in _INIT_MEMORY_SEAMS]


class _ScaffoldManager:
    """A benign manager (mutation/positive controls only): lets control reach the seams after it."""
    providers = ()


class _ScaffoldStore:
    """A benign built-in store (scaffolding only; with the scaffold flags False the receiver never builds it)."""
    def __init__(self, **kwargs): pass
    def load_from_disk(self): pass


# What a seam returns when it is only scaffolding that lets control reach the seam under test (not recorded).
_SCAFFOLD = {
    'builtin_memory_config': lambda target: (lambda cfg=None: {'provider': 'hindsight'}),
    # True flags only when the store itself is under test: the receiver constructs it only for a true flag.
    'builtin_memory_flags': lambda target: (lambda cfg=None: (True, True) if target == 'builtin_memory_store'
                                            else (False, False)),
    'builtin_memory_store': lambda target: _ScaffoldStore,
    'core_provider_check': lambda target: (lambda name: False),
    'memory_manager': lambda target: _ScaffoldManager,
    'load_memory_provider': lambda target: (lambda name, **kw: None),  # None: the receiver then tries recovery
}


def _install_seams(monkeypatch, attempts, target=None):
    """Every seam records ``(name, args, kwargs)`` into ``attempts`` and raises. With ``target``, the seams
    BEFORE it are unrecorded scaffolding so control reaches ``target`` at its real call site; it and every
    later seam still record and raise."""
    reached_target = target is None
    for name, path in _INIT_MEMORY_SEAMS:
        reached_target = reached_target or name == target
        if not reached_target:
            monkeypatch.setattr(path, _SCAFFOLD[name](target))
            continue

        def forbidden(*a, _name=name, **k):
            attempts.append((_name, a, k))
            raise AssertionError(f'skip_memory=True must never reach the memory seam {_name}')
        monkeypatch.setattr(path, forbidden)


def _memory_init_agent():
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.session_id, agent.valid_tool_names, agent.tools = 'sid', set(), []
    agent.enabled_toolsets = agent.disabled_toolsets = None
    return agent


_MEMORY_INIT_CFG = {'memory': {'provider': 'hindsight', 'memory_enabled': True, 'user_profile_enabled': True}}


def test_init_memory_under_skip_memory_builds_no_provider_and_no_builtin_store(monkeypatch):
    from agent import agent_init
    from agent.safe_worker_policy import safe_worker_enabled
    assert safe_worker_enabled() is False  # r4: else :1309-1311 returns before the skip guards and proves nothing
    attempts = []
    _install_seams(monkeypatch, attempts)  # r4: recording seams (were: raising-only stubs + is_core -> False)
    agent = _memory_init_agent()
    cfg = _MEMORY_INIT_CFG
    agent_init._init_memory(agent, cfg, True, 'cli')
    assert attempts == []  # r4 oracle: no provider/manager/recovery/built-in-store ATTEMPT, even a swallowed one
    assert agent._memory_manager is None and agent._memory_store is None
    assert agent._memory_enabled is False and agent._user_profile_enabled is False
    assert agent.tools == [] and agent.valid_tool_names == set()


def _end_state_holds(agent):
    """The pre-r4 end-state assertions of the test above, as one predicate."""
    return (agent._memory_manager is None and agent._memory_store is None
            and agent._memory_enabled is False and agent._user_profile_enabled is False
            and agent.tools == [] and agent.valid_tool_names == set())


@pytest.mark.parametrize('target', _SEAM_NAMES)
def test_ordinary_init_memory_reaches_each_recorded_seam(monkeypatch, target):
    """Positive control: the REAL receiver ``_init_memory`` with ``skip_memory=False`` reaches each seam at its
    real call site, so every recorder above is live (an empty attempt list is not a dead patch)."""
    from agent import agent_init
    attempts = []
    _install_seams(monkeypatch, attempts, target)
    agent_init._init_memory(_memory_init_agent(), _MEMORY_INIT_CFG, False, 'cli')
    # A swallowed built-in-store failure after the config read leaves ``mem_config`` bound, so the ordinary
    # provider block then runs and records its first seam too.
    after = ['core_provider_check'] if target in ('builtin_memory_flags', 'builtin_memory_store') else []
    assert [a[0] for a in attempts] == [target, *after]


def _skip_guard_removed_init_memory(*guards):
    """MUTANT of the receiver's own ``_init_memory`` source with the named ``skip_memory`` guards replaced by
    ``True`` (``builtin``: :1323, ``provider``: :1350), compiled against ``agent.agent_init``'s real globals.
    Only these guard lines differ from the receiver; the module itself is not changed."""
    import inspect
    import textwrap
    from agent import agent_init
    source = textwrap.dedent(inspect.getsource(agent_init._init_memory))
    edits = {'builtin': ('    if not skip_memory or _memory_toolset_requested:\n', '    if True:\n'),
             'provider': ('    elif not skip_memory:\n', '    elif True:\n')}
    for guard in guards:
        old, new = edits[guard]
        assert source.count(old) == 1, f'receiver guard {guard!r} not found exactly once'
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, f'{agent_init.__file__}[r4 mutant: {"+".join(guards)} guard removed]', 'exec'),
         agent_init.__dict__, namespace)
    return namespace['_init_memory']


# Expected pre-r4 end-state verdict under each mutant: the receiver sets ``_memory_enabled``/``_user_profile_enabled``
# from the flags (:1330) before it constructs the store (:1335), so a store attempt leaves the flags True and the old
# assertions catch it; every other seam's attempt leaves the old end state intact.
_OLD_ASSERTIONS_STILL_PASS = {name: name != 'builtin_memory_store' for name in _SEAM_NAMES}


@pytest.mark.parametrize('target', _SEAM_NAMES)
def test_mutation_a_skip_memory_attempt_at_each_seam_fails_only_the_new_oracle(monkeypatch, target):
    """Mutation check of the r4 oracle: a skip-memory ``_init_memory`` (``skip_memory=True``) whose guard is removed
    attempts ``target`` through the receiver's real call site. The new ``attempts == []`` oracle FAILS for every seam;
    the pre-r4 end-state assertions alone still pass for all but the built-in store (see above)."""
    from agent.safe_worker_policy import safe_worker_enabled
    assert safe_worker_enabled() is False
    builtin = target.startswith('builtin_')
    mutant = _skip_guard_removed_init_memory(*(('builtin',) if builtin else ('builtin', 'provider')))
    attempts = []
    _install_seams(monkeypatch, attempts, target)
    agent = _memory_init_agent()
    mutant(agent, _MEMORY_INIT_CFG, True, 'cli')  # swallowed: the receiver's own suppress / except Exception
    assert [a[0] for a in attempts] == [target]  # i.e. the oracle ``attempts == []`` fails on exactly this seam
    assert _end_state_holds(agent) is _OLD_ASSERTIONS_STILL_PASS[target]


# -- forks under compression.checkpoint_required ------------------------------------------------------

def _cfg(checkpoint_required):
    return {'compression': {'checkpoint_required': checkpoint_required}, 'agent': {}, 'memory': {}, 'delegation': {}}


def _delegate_child(parent, checkpoint_required, toolsets=None, **child_attrs):
    import run_agent
    from tools import delegate_tool
    captured, children = {}, []

    def recorder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**child_attrs)

    def stop_after_marker(child):
        children.append(child)
        raise _ConstructorStop()

    cfg = _cfg(checkpoint_required)
    with ExitStack() as stack:
        stack.enter_context(patch('hermes_cli.config.load_config', return_value=cfg))
        stack.enter_context(patch('hermes_cli.config.load_config_readonly', return_value=cfg))
        stack.enter_context(patch.object(run_agent, 'AIAgent', recorder))
        stack.enter_context(patch.object(delegate_tool, '_resolve_child_credential_pool', lambda *a, **k: None))
        stack.enter_context(patch.object(delegate_tool, '_apply_child_cache_ttl', stop_after_marker))
        with pytest.raises(_ConstructorStop):
            delegate_tool._build_child_agent(task_index=0, goal='g', context=None, toolsets=toolsets, model=None,
                                             max_iterations=3, task_count=1, parent_agent=parent)
    return captured, children[0]


def _parent(**extra):
    return SimpleNamespace(model='m', provider='openrouter', api_key='k', base_url='https://openrouter.ai/api/v1',
                           api_mode='chat_completions', session_id='parent', enabled_toolsets=None,
                           disabled_toolsets=None, request_overrides={}, prefill_messages=None, _delegate_depth=0,
                           **extra)


@pytest.mark.parametrize('checkpoint_required', [True, False])
def test_delegate_child_of_a_skip_memory_session_skips_memory(checkpoint_required):
    captured, child = _delegate_child(_parent(_session_skip_memory=True), checkpoint_required)
    assert captured['skip_memory'] is True
    assert child._session_skip_memory is True  # grandchildren inherit it
    assert child.compression_checkpoint_required is False


def _tool(name):
    return {'type': 'function', 'function': {'name': name}}


@pytest.mark.parametrize('skip', [True, False], ids=['skip-memory-session', 'flush-agent-control'])
def test_an_explicit_memory_toolset_never_reaches_a_skip_memory_session(tmp_path, monkeypatch, skip):
    """``skip_memory=True`` at construction skips only the provider: an agent that names the ``memory`` toolset
    still gets the built-in store with MEMORY.md (#65429; a flush agent keeps it). A ``--skip-memory``
    SESSION's agent drops that store, its flags and the memory tool before any prompt is built, and every later
    tool refresh keeps the tool out (``toolsets.agent_tool_drops``)."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'memories').mkdir()
    (tmp_path / 'memories' / 'MEMORY.md').write_text('NOTE-FROM-MEMORY-MD\n')
    (tmp_path / 'memories' / 'USER.md').write_text('NOTE-FROM-USER-MD\n')
    from agent import agent_init
    from run_agent import AIAgent
    from toolsets import agent_tool_drops
    agent = object.__new__(AIAgent)
    agent.session_id, agent.platform = 'sid', 'cli'
    agent.tools, agent.valid_tool_names = [_tool('memory'), _tool('terminal')], {'memory', 'terminal'}
    agent.enabled_toolsets, agent.disabled_toolsets = ['memory', 'terminal'], None
    agent_init._init_memory(agent, {'memory': {'memory_enabled': True, 'user_profile_enabled': True}}, True, 'cli')
    assert agent._memory_manager is None
    assert 'NOTE-FROM-MEMORY-MD' in agent._memory_store.format_for_system_prompt('memory')  # #65429, as before
    if skip:
        apply_session_skip_memory(agent)
        assert agent._memory_store is None
        assert agent._memory_enabled is False and agent._user_profile_enabled is False
        assert [t['function']['name'] for t in agent.tools] == ['terminal'] and agent.valid_tool_names == {'terminal'}
        assert 'memory' in agent_tool_drops(agent)
    else:
        assert agent._memory_store is not None and 'memory' in agent.valid_tool_names
        assert 'memory' not in agent_tool_drops(agent)


@pytest.mark.parametrize('checkpoint_required', [True, False])
def test_an_inherited_memory_toolset_never_reaches_a_skip_memory_sessions_child(checkpoint_required):
    """A delegate child asked for the ``memory`` toolset under a ``--skip-memory`` parent: built with
    ``skip_memory=True`` (no provider), then its built-in store and memory tool are dropped as for the parent."""
    captured, child = _delegate_child(
        _parent(_session_skip_memory=True), checkpoint_required, toolsets=['memory', 'terminal'],
        tools=[_tool('memory'), _tool('terminal')], valid_tool_names={'memory', 'terminal'},
        _memory_store=object(), _memory_enabled=True, _user_profile_enabled=True)
    assert captured['skip_memory'] is True
    assert child._session_skip_memory is True and child.compression_checkpoint_required is False
    assert child._memory_store is None and child._memory_enabled is False and child._user_profile_enabled is False
    assert [t['function']['name'] for t in child.tools] == ['terminal'] and child.valid_tool_names == {'terminal'}


def test_a_tool_refresh_never_brings_memory_back_to_a_skip_memory_session(monkeypatch):
    """The MCP refresh / prefix restore path re-filters fresh definitions through ``_drop_session_tools``
    (receiver ``tools.mcp_tool_agent``; the ancestor's ``toolsets.agent_tool_surface`` is absent there)."""
    from tools.mcp_tool_agent import _drop_session_tools
    agent = SimpleNamespace(platform='cli', tools=[_tool('memory')], valid_tool_names={'memory'})
    apply_session_skip_memory(agent)
    defs, names = _drop_session_tools(agent, [_tool('memory'), _tool('web')], {'memory', 'web'})
    assert [t['function']['name'] for t in defs] == ['web']
    assert names == {'web'}  # r3 addition: the published valid_tool_names drop memory too


@pytest.mark.parametrize('checkpoint_required', [True, False])
def test_delegate_child_of_an_ordinary_session_is_unchanged(checkpoint_required):
    captured, child = _delegate_child(_parent(), checkpoint_required)
    # No opt-out: built provider-less unless checkpoint_required is armed, when it inherits the provider
    # (tests/agent/test_fork_memory_under_checkpoint_gate.py). The opt-out cases above stay skip_memory=True.
    assert captured['skip_memory'] is (not checkpoint_required)
    assert not hasattr(child, '_session_skip_memory')


async def _get_session(session_id):
    return None


def _fork_kwargs(site, policy, checkpoint_required, monkeypatch):
    captured = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        raise _ConstructorStop()

    monkeypatch.setattr('hermes_cli.config.load_config', lambda: _cfg(checkpoint_required))
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: policy)
    monkeypatch.setattr('run_agent.AIAgent', recorder)
    host = SimpleNamespace(_session_db=SimpleNamespace(get_session=_get_session))
    source = SimpleNamespace(platform='local', chat_id='c')
    if site == 'hygiene':
        from gateway.run_turn_hygiene import GatewayTurnHygieneMixin
        call = GatewayTurnHygieneMixin._hmwa_hygiene_build_agent(host, 'm', {}, SimpleNamespace(session_id='s'),
                                                                  source)
    else:
        from gateway.slash_commands_session import GatewaySessionCommandsMixin
        call = GatewaySessionCommandsMixin._build_manual_compression_agent(host, 's', 'm', {}, source)
    with pytest.raises(_ConstructorStop):
        asyncio.run(call)
    return captured


@pytest.mark.parametrize('site', ['hygiene', 'compress'])
@pytest.mark.parametrize('checkpoint_required', [True, False])
def test_compaction_fork_of_a_skip_memory_session_never_loads_the_provider(tmp_path, monkeypatch, site,
                                                                           checkpoint_required):
    policy = _policy(tmp_path, skip_memory=True)
    assert _fork_kwargs(site, policy, checkpoint_required, monkeypatch)['skip_memory'] is True


@pytest.mark.parametrize('site', ['hygiene', 'compress'])
@pytest.mark.parametrize('params', [None, {}, {'ignore_rules': True}])
def test_compaction_fork_of_other_sessions_is_unchanged(tmp_path, monkeypatch, site, params):
    """Messaging (no policy), ordinary and --ignore-rules sessions keep today's rule: the provider loads
    for the checkpoint only when checkpoint_required is armed."""
    policy = None if params is None else _policy(tmp_path, **params)
    assert _fork_kwargs(site, policy, True, monkeypatch)['skip_memory'] is False
    assert _fork_kwargs(site, policy, False, monkeypatch)['skip_memory'] is True

