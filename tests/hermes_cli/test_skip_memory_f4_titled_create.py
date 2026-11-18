"""F4 (SKM7 owner t53/run20975): ``hermes chat -c NAME --create-if-missing --skip-memory`` on the classic CLI.

The whole classic flow runs for real: the resolver (``hermes_cli.main._resolve_continue_arg``, title lookup through
``_resolve_session_by_name_or_id`` on the owner's ``state.db``) creates the titled session with
``gateway_session_store.create_titled_session``, and the startup resume is ``GatewaySessionStore.open_session`` with
the CLI's own ``_gateway_launch`` and resume check. The owner is real too: a private in-test session authority
(``tests.gateway.test_local_source_labels._authority``) behind ``AuthorityConnection.dispatch``, which freezes the
creation request into the session's policy and echoes it as ``info.launch_request`` on resume.

Only the CLI's RPC boundary (``GatewaySessionStore.call``) is replaced, by a recorder that sends each call to that
in-test owner and keeps the request and the reply. Nothing dials a gateway socket, no live session or provider is
touched, and the run is network-free."""
from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

import pytest


class _Recorder:
    """The classic CLI's gateway RPC (``GatewaySessionStore.call``), bridged from the CLI's own thread to the real
    owner loop, as tests/hermes_cli/test_gateway_chat_retry_validate.py bridges ``rpc``."""

    def __init__(self, connection, loop):
        self.connection, self.loop, self.calls = connection, loop, []

    def __call__(self, store, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        sent = json.loads(json.dumps(params))  # the request as sent; the owner's handlers pop fields off theirs
        frame = {'id': len(self.calls), 'method': method, 'params': json.loads(json.dumps(params))}
        reply = asyncio.run_coroutine_threadsafe(self.connection.dispatch(frame), self.loop).result(60)
        self.calls.append((method, sent, reply))
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        return reply['result']

    def params(self, method):
        return [params for name, params, _ in self.calls if name == method]


def _launch_args(title, skip_memory):
    """``hermes chat -c TITLE --create-if-missing [--skip-memory]`` as the chat parser leaves it."""
    return argparse.Namespace(continue_last=title, resume=None, create_if_missing=True, skip_memory=skip_memory)


def _classic_cli(session_id, skip_memory):
    """The attributes HermesCLI's init sets for this launch and ``_gateway_launch`` / ``open_session`` read:
    ``_resumed = bool(resume)`` (cli_init_mixin :413), ``skip_memory = _explicit_skip_memory = bool(skip_memory)``
    (:344), and no other typed flag."""
    return SimpleNamespace(
        session_id=session_id, _resumed=True, model='fixture', _explicit_model_override=False, _explicit_provider=None,
        _explicit_api_key=None, _typed_api_key=None, _explicit_base_url=None, enabled_toolsets=None,
        _explicit_toolsets=False, _typed_toolsets=None, _explicit_reasoning=None, _explicit_max_turns=None,
        _explicit_checkpoints=False, pass_session_id=False, ignore_rules=False, _explicit_ignore_rules=False,
        skip_memory=bool(skip_memory), _explicit_skip_memory=bool(skip_memory), _explicit_run_budget=None,
        verbose=False, _preload_skills_requested=None)


async def _owner(tmp_path, monkeypatch):
    from pathlib import Path

    from gateway.session_controls import AuthorityConnection
    from hermes_cli import gateway_connection, gateway_session_store
    from tests.gateway.test_local_source_labels import _authority

    authority, store = await _authority(tmp_path, monkeypatch)
    home = Path(authority.profile_id)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.chdir(tmp_path)
    for name in ('TERMINAL_CWD', 'HERMES_YOLO_MODE', 'HERMES_IGNORE_RULES', 'HERMES_SAFE_MODE',
                 'HERMES_IGNORE_USER_CONFIG', 'HERMES_ACCEPT_HOOKS'):
        monkeypatch.delenv(name, raising=False)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    recorder = _Recorder(conn, asyncio.get_running_loop())
    monkeypatch.setattr(gateway_session_store.GatewaySessionStore, 'call',
                        lambda self, method, **params: recorder(self, method, **params))
    # A private viewer connection per test: it never dials (every call goes through the recorder).
    viewer = gateway_connection.GatewayConnection()
    monkeypatch.setattr(gateway_connection.GatewayConnection, 'shared', classmethod(lambda cls: viewer))
    return authority, store, conn, recorder, viewer


def _frozen_policy(authority, session_id):
    """The policy the owner froze for the session (its committed receipt), not any client-side view of it."""
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    return restore_policy(local_receipt(authority.db, session_id)['policy'])


def _resolve_create_resume(title, skip_memory, viewer):
    """The classic CLI's startup, in its own thread: resolve ``-c`` (creating on a miss), then open the session."""
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from hermes_cli.main import _resolve_continue_arg
    args = _launch_args(title, skip_memory)
    _resolve_continue_arg(args, use_tui=False)
    cli = _classic_cli(args.resume, skip_memory)
    store = GatewaySessionStore(viewer)
    return args.resume, store.open_session(cli, _gateway_launch(cli))


@pytest.mark.asyncio
@pytest.mark.parametrize('skip_memory', [True, False], ids=['skip-memory-launch', 'ordinary-launch'])
async def test_a_titled_create_if_missing_launch_freezes_its_typed_skip_memory_and_resumes(
        tmp_path, monkeypatch, skip_memory):
    authority, store, conn, recorder, viewer = await _owner(tmp_path, monkeypatch)
    try:
        created_id, opened = await asyncio.to_thread(_resolve_create_resume, 'probe-title', skip_memory, viewer)
        [create] = recorder.params('session.create')
        assert create['title'] == 'probe-title'
        # The typed flag rides the title-creation request itself, so the owner freezes it at creation.
        assert create.get('skip_memory') is (True if skip_memory else None)
        assert _frozen_policy(authority, created_id).skip_memory_requested is skip_memory
        # The startup resume of the same launch is admitted: the frozen launch matches the typed one.
        assert opened == created_id
        [resume_reply] = [reply for name, _, reply in recorder.calls if name == 'session.resume']
        frozen = resume_reply['result']['info']['launch_request']
        assert frozen.get('skip_memory', False) is skip_memory

        # Running the same command line again finds the title and resumes the same, still-frozen session.
        again_id, again = await asyncio.to_thread(_resolve_create_resume, 'probe-title', skip_memory, viewer)
        assert again_id == again == created_id
        assert len(recorder.params('session.create')) == 1
        assert _frozen_policy(authority, created_id).skip_memory_requested is skip_memory
    finally:
        await conn.close()
        authority.db.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_ordinary_titled_session_still_refuses_a_skip_memory_resume(tmp_path, monkeypatch):
    """The resume guard is unchanged: a session created WITHOUT the flag is not turned memoryless by a later
    ``-c NAME --skip-memory``; the owner's frozen policy stays ordinary and the resume is refused."""
    from hermes_cli.gateway_client import GatewayClientError
    authority, store, conn, recorder, viewer = await _owner(tmp_path, monkeypatch)
    try:
        created_id, _ = await asyncio.to_thread(_resolve_create_resume, 'plain-title', False, viewer)
        with pytest.raises(GatewayClientError, match='skip-memory'):
            await asyncio.to_thread(_resolve_create_resume, 'plain-title', True, viewer)
        assert len(recorder.params('session.create')) == 1
        assert _frozen_policy(authority, created_id).skip_memory_requested is False
    finally:
        await conn.close()
        authority.db.close()
        store.close_all_db_handles()
