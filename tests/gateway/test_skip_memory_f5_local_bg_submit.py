"""F5 (SKM7 owner t53/run20975): raw ``/bg <prompt>`` text submitted to a LOCAL session through ``prompt.submit``.

The real local ingress runs: ``SessionAuthority.submit`` admits the typed row, the real FIFO drain
(``gateway.session_ingress.execute_admission``) builds the ``MessageEvent``, and the REAL ``runner._handle_message``
authorizes the LOCAL source and dispatches ``/bg`` to ``_handle_background_command`` ->
``_run_background_task`` -> ``_run_background_task_inner``, which constructs the background agent. A messaging-origin
``/bg`` (Telegram) goes through the same real ``_handle_message`` dispatch and constructor.

Only inert boundaries are replaced: the model boundary (``run_agent.AIAgent``, a recorder that runs no model), the
provider route (``route_runtime`` / ``_resolve_runtime_agent_kwargs`` return an inert key and an unroutable local
endpoint; the run is network-free), the gateway config, and the messaging adapter. The session store is a private
``state.db`` under ``tmp_path``; no live session is touched."""
from __future__ import annotations

import asyncio

import pytest

from tests.waiting import async_wait_until

_INERT_RUNTIME = {'api_key': 'inert-test-key', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1',
                  'api_mode': 'chat_completions'}


def _tool(name):
    return {'type': 'function', 'function': {'name': name, 'description': name, 'parameters': {}}}


def _agent_recorder(built):
    """The model boundary: records the constructor kwargs and carries the surface ``apply_session_skip_memory``
    acts on (a memory tool, a built-in store, a compaction checkpoint) so its effect is observable."""

    class BackgroundAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = kwargs.get('session_id')
            self.tools, self.valid_tool_names = [_tool('memory'), _tool('terminal')], {'memory', 'terminal'}
            self._memory_store = object()
            self._memory_enabled = self._user_profile_enabled = True
            self.compression_checkpoint_required = True
            self._cached_system_prompt = self._cached_system_prompt_static = 'frozen'
            built.append(self)

        def run_conversation(self, user_message=None, task_id=None, **kwargs):
            self.ran = (user_message, task_id)
            return {'final_response': 'background done', 'messages': []}

        def shutdown_memory_provider(self, *a, **k):
            pass

        def close(self):
            pass
    return BackgroundAgent


async def _local_runner(tmp_path, monkeypatch, built):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'},
                                                              'platform_toolsets': {'cli': [], 'telegram': []}})
    monkeypatch.setattr('run_agent.AIAgent', _agent_recorder(built))
    monkeypatch.setattr('gateway.session_worker_construct.route_runtime',
                        lambda policy, frozen, model, key, fallback=True: (dict(_INERT_RUNTIME), None))
    monkeypatch.setattr(run, '_resolve_runtime_agent_kwargs', lambda *a, **k: dict(_INERT_RUNTIME))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner', db=db)
    return runner, authority, db


async def _drain(runner):
    for task in list(getattr(runner, '_background_tasks', ())):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize('skip_memory', [True, False], ids=['skip-memory-session', 'ordinary-session'])
async def test_raw_bg_text_submitted_to_a_local_session_builds_its_agent_under_the_frozen_policy(
        tmp_path, monkeypatch, skip_memory):
    from gateway.session_contract import SessionActor, Submission
    from gateway.session_local import create_local_session
    from gateway.session_policy import policy_for_source

    built = []
    runner, authority, db = await _local_runner(tmp_path, monkeypatch, built)
    try:
        viewer = SessionActor('test-owner', 'default', frozenset(
            {'session:create', 'session:submit', 'session:read'}), 'viewer')
        ref = create_local_session(authority, viewer, dict(
            request_id='cli', source='cli', model='fixture', toolsets=[], **({'skip_memory': True} if skip_memory else {})))
        live = authority.sessions[ref.session_id]
        await authority.attach(viewer, ref)
        # The session's frozen policy, as the owner restored it: the control's premise.
        assert policy_for_source(runner, live.source).skip_memory_requested is skip_memory

        await authority.submit(viewer, Submission('typed-bg', ref, {'text': '/bg inspect the workspace'}, 'queue'))
        await async_wait_until(lambda: built and hasattr(built[0], 'ran'), timeout=20, what='the /bg agent run')
        await _drain(runner)
    finally:
        db.close()

    [bg] = built
    assert bg.ran[0] == 'inspect the workspace' and bg.session_id.startswith('bg_')
    if skip_memory:
        # Built memoryless (no provider) AND marked: no built-in store, no memory tool, no compaction checkpoint.
        assert bg.kwargs.get('skip_memory') is True
        assert bg._session_skip_memory is True and bg._memory_store is None
        assert bg._memory_enabled is False and bg._user_profile_enabled is False
        assert bg.valid_tool_names == {'terminal'} and bg.compression_checkpoint_required is False
    else:  # an ordinary local session: unchanged
        assert 'skip_memory' not in bg.kwargs
        assert getattr(bg, '_session_skip_memory', False) is False
        assert 'memory' in bg.valid_tool_names and bg._memory_store is not None


@pytest.mark.asyncio
async def test_a_messaging_origin_bg_is_unchanged(tmp_path, monkeypatch):
    """A messenger's ``/bg`` (no local launch policy) keeps its receiver behaviour: memory on, no session marker."""
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_policy import policy_for_source

    built = []
    monkeypatch.setenv('GATEWAY_ALLOW_ALL_USERS', 'true')
    runner, _authority, db = await _local_runner(tmp_path, monkeypatch, built)
    adapter = AsyncMock()
    adapter.extract_media = MagicMock(side_effect=lambda text: ([], text))
    adapter.extract_images = MagicMock(side_effect=lambda text: ([], text))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id='12345', chat_id='67890', user_name='tester')
    try:
        assert policy_for_source(runner, source) is None
        reply = await runner._handle_message(MessageEvent(text='/bg inspect the workspace', source=source))
        await async_wait_until(lambda: built and hasattr(built[0], 'ran'), timeout=20, what='the /bg agent run')
        await _drain(runner)
    finally:
        db.close()
    assert reply  # the started notice
    [bg] = built
    assert bg.ran[0] == 'inspect the workspace'
    assert 'skip_memory' not in bg.kwargs and getattr(bg, '_session_skip_memory', False) is False
    assert 'memory' in bg.valid_tool_names and bg._memory_store is not None
