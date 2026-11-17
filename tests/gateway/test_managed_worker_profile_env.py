"""A served secondary profile's managed worker never inherits launch-profile residue."""
import os
from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active


def test_secondary_worker_env_drops_launch_terminal_policy_and_env_settings(tmp_path, monkeypatch):
    """Same contract as the cron and kanban dispatchers: a worker for served profile X gets the
    env a standalone ``hermes -p X`` gateway would build — X's HERMES_HOME and X's ``.env`` over a
    base without the launch profile's ``TERMINAL_*`` policy or launch-only ``.env`` settings."""
    from gateway.session_managed_worker import _worker_env
    root = tmp_path / '.hermes'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (root / '.env').write_text('HERMES_MODEL=launch-model\nTERMINAL_ENV=docker\n')
    (beta / '.env').write_text('BETA_ONLY_TOKEN=beta-secret\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_MODEL', 'launch-model')
    monkeypatch.setenv('TERMINAL_ENV', 'docker')
    monkeypatch.setenv('TERMINAL_CWD', str(root))
    set_multiplex_active(True)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert env['HERMES_HOME'] == str(beta)
    assert env['BETA_ONLY_TOKEN'] == 'beta-secret'
    assert not {k for k in env if k.startswith('TERMINAL_')}, env
    assert 'HERMES_MODEL' not in env


def test_launch_scrub_runs_before_the_turn_context_is_injected(tmp_path, monkeypatch):
    """The launch residue strip runs on the RAW environ, before ``build_subprocess_env`` bridges this
    turn's own session context: a launch ``.env`` that names ``HERMES_SESSION_ID`` must not erase the
    id derived for the secondary's turn (#109338 ordering)."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import scoped_current_session_id
    from gateway.session_managed_worker import _worker_env
    root = tmp_path / '.hermes'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (root / '.env').write_text('HERMES_SESSION_ID=launch-session\nLAUNCH_ONLY=launch-value\n')
    (beta / '.env').write_text('BETA_ONLY_TOKEN=beta-secret\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_SESSION_ID', 'launch-session')
    monkeypatch.setenv('LAUNCH_ONLY', 'launch-value')
    before = dict(os.environ)
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(beta), scoped_current_session_id('owned-session'):
            env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert env['HERMES_SESSION_ID'] == 'owned-session'
    assert 'LAUNCH_ONLY' not in env
    assert env['BETA_ONLY_TOKEN'] == 'beta-secret'
    assert dict(os.environ) == before


@pytest.fixture
def restored_environ():
    """The real bridge writes ``os.environ``; put the process env back exactly afterwards."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _profiles(tmp_path, monkeypatch, *, launch_pinned, secondary_pinned):
    root = tmp_path / '.hermes'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (root / 'config.yaml').write_text('terminal:\n  docker_image: launch/image:1\n' if launch_pinned else '{}\n')
    (beta / 'config.yaml').write_text('terminal:\n  docker_image: beta/image:2\n' if secondary_pinned else '{}\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    for key in [k for k in os.environ if k.startswith('TERMINAL_')]:
        monkeypatch.delenv(key)
    return root, beta


def _bridge(env=None):
    from hermes_cli.config import apply_terminal_config_to_env
    return apply_terminal_config_to_env(env=env)


@pytest.mark.parametrize('secondary_pinned', [False, True], ids=['secondary-default', 'secondary-pinned'])
@pytest.mark.parametrize('launch_pinned', [False, True], ids=['launch-pin-0', 'launch-pin-1'])
def test_the_launch_bridge_pin_never_reaches_a_secondary_worker(
        tmp_path, monkeypatch, restored_environ, launch_pinned, secondary_pinned):
    """FR-ENV-01: the real launch bridge sets TERMINAL_DOCKER_IMAGE_PINNED (0 or 1) in the process
    env. The secondary's worker env drops it with the image, and the secondary's own bridge then
    derives the pin from the secondary's config, not the launch profile's."""
    from gateway.session_managed_worker import _worker_env
    _root, beta = _profiles(tmp_path, monkeypatch, launch_pinned=launch_pinned, secondary_pinned=secondary_pinned)
    _bridge()
    assert os.environ['TERMINAL_DOCKER_IMAGE_PINNED'] == ('1' if launch_pinned else '0')
    set_multiplex_active(True)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert 'TERMINAL_DOCKER_IMAGE_PINNED' not in env, env
    assert not {k for k in env if k.startswith('TERMINAL_')}, env
    # The worker child starts under the secondary's home and bridges its own config.
    monkeypatch.setenv('HERMES_HOME', str(beta))
    _bridge(env)
    assert env['TERMINAL_DOCKER_IMAGE_PINNED'] == ('1' if secondary_pinned else '0')
    if secondary_pinned:
        assert env['TERMINAL_DOCKER_IMAGE'] == 'beta/image:2'


@pytest.mark.parametrize('launch_pinned', [False, True], ids=['launch-pin-0', 'launch-pin-1'])
def test_the_launch_profiles_own_worker_keeps_its_bridged_pin(tmp_path, monkeypatch, restored_environ, launch_pinned):
    """Same-profile policy is unchanged: the launch profile's own worker keeps its frozen launch env."""
    from gateway.session_managed_worker import _worker_env
    root, _beta = _profiles(tmp_path, monkeypatch, launch_pinned=launch_pinned, secondary_pinned=False)
    _bridge()
    set_multiplex_active(True)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(root)))
    finally:
        set_multiplex_active(False)
    assert env['TERMINAL_DOCKER_IMAGE_PINNED'] == ('1' if launch_pinned else '0')


def test_an_administrator_managed_pin_stays_policy_for_a_secondary(tmp_path, monkeypatch, restored_environ):
    """A pin the administrator-managed ``.env`` supplies is policy for every profile: never stripped."""
    from gateway.session_managed_worker import _worker_env
    from hermes_cli import env_loader
    _root, beta = _profiles(tmp_path, monkeypatch, launch_pinned=False, secondary_pinned=False)
    monkeypatch.setattr(env_loader, '_MANAGED_DOTENV_KEYS', {'TERMINAL_DOCKER_IMAGE_PINNED'})
    monkeypatch.setenv('TERMINAL_DOCKER_IMAGE_PINNED', '1')
    _bridge()
    set_multiplex_active(True)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert env['TERMINAL_DOCKER_IMAGE_PINNED'] == '1'
