"""Launch-only authority is bound to the home whose files a scope actually reads.

``owner_secret_scope`` adds the launch env's credentials, and ``build_profile_secret_scope`` adds
the bridged ``allow_all_users`` grant, only when the home "is the launch home". Under
multiplexing a scope that reads one profile's ``.env`` while carrying the launch profile's
env-only credentials or allow-all grant is a cross-profile authority leak.

The regression (review t_e75cbcaf run 2442, SCOPE-IDENTITY-01): a symlinked home alias was warmed
while it named the launch home, then retargeted to another profile's home without a restart. The
``.env`` followed the link to the new home, but a remembered identity still answered "launch home",
so the scope held the other profile's files plus the launch credential and grant.

These drive the real producers (``owner_secret_scope``, ``build_profile_secret_scope``) and the real
gate reader (``platform_gate_env``) under the real ``allow_all_users`` bridge. Every value is
synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import agent.secret_scope as ss
import hermes_constants

LAUNCH_ONLY_KEY = "OPENAI_API_KEY"  # injected into the launch process env only (systemd / op run shape)
LAUNCH_ONLY_VALUE = "fake-launch-env-only"


@pytest.fixture
def world(tmp_path, monkeypatch):
    from gateway import config_loader
    import tui_gateway.launch_profile_policy as launch_policy

    launch = tmp_path / "launch" / ".hermes"
    served = launch / "profiles" / "served"
    for home in (launch, served):
        home.mkdir(parents=True)
        (home / ".env").write_bytes(b"PROFILE_MARKER=" + home.name.encode() + b"\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "no-managed"))
    monkeypatch.setenv(LAUNCH_ONLY_KEY, LAUNCH_ONLY_VALUE)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    # Restored on teardown: the real bridge below writes both.
    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)
    monkeypatch.setattr(launch_policy, "_snapshot", None)
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(ss, "_AUTO_PINNED_HOME", None)
    hermes_constants.reset_hermes_home_key_cache()
    ss.invalidate_env_file_cache()
    try:
        yield launch, served
    finally:
        ss.set_multiplex_active(False)  # releases the launch-home pin activation created
        hermes_constants.reset_hermes_home_key_cache()
        ss.invalidate_env_file_cache()


def _activate_with_bridged_allow_all():
    """The launch profile's config.yaml says ``allow_all_users: true``; then a second profile
    turns multiplexing on (which pins the launch home and freezes the launch env)."""
    from gateway.config_loader import bridge_core_env_settings
    bridge_core_env_settings({"allow_all_users": True}, {})
    assert os.environ.get("GATEWAY_ALLOW_ALL_USERS") == "true"
    ss.set_multiplex_active(True)


def _bound_gate(home: Path) -> tuple[str | None, str]:
    """(marker, allow-all gate) as a turn bound to *home*'s scope sees them."""
    from gateway.platforms._shared import platform_gate_env
    token = ss.set_secret_scope(ss.build_profile_secret_scope(home), profile_home=str(home))
    try:
        return ss.get_secret("PROFILE_MARKER"), platform_gate_env("GATEWAY_ALLOW_ALL_USERS")
    finally:
        ss.reset_secret_scope(token)


def _retarget(alias: Path, target: Path) -> None:
    alias.unlink()
    alias.symlink_to(target, target_is_directory=True)


def test_retargeted_alias_never_pairs_another_homes_files_with_launch_authority(world, tmp_path):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "alias"
    alias.symlink_to(launch, target_is_directory=True)

    # Warm every identity path while the alias names the launch home: it IS the launch profile.
    assert ss._is_process_home(alias) is True
    warm = owner_secret_scope(alias)
    assert (warm["PROFILE_MARKER"], warm.get(LAUNCH_ONLY_KEY)) == (".hermes", LAUNCH_ONLY_VALUE)
    assert _bound_gate(alias) == (".hermes", "true")

    _retarget(alias, served)  # no restart, no cache reset
    after = owner_secret_scope(alias)
    leaked = {k: after.get(k) for k in (LAUNCH_ONLY_KEY, "GATEWAY_ALLOW_ALL_USERS") if k in after}
    assert (after["PROFILE_MARKER"], leaked) == ("served", {}), "served files + launch authority"
    assert _bound_gate(alias) == ("served", ""), "served profile's files paired with the launch grant"
    assert ss._is_process_home(alias) is False

    _retarget(alias, launch)  # and back: no remembered negative either
    assert ss._is_process_home(alias) is True
    assert owner_secret_scope(alias).get(LAUNCH_ONLY_KEY) == LAUNCH_ONLY_VALUE
    assert _bound_gate(alias) == (".hermes", "true")


def test_launch_home_identity_follows_the_pin_not_the_mirrored_env(world):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    # Unpinned (single-profile): the launch home is whatever HERMES_HOME names now.
    os.environ["HERMES_HOME"] = str(served)
    assert [ss._is_process_home(launch), ss._is_process_home(served)] == [False, True]
    os.environ["HERMES_HOME"] = str(launch)

    _activate_with_bridged_allow_all()
    # Pinned at activation: a host mirroring the served profile into HERMES_HOME changes nothing.
    os.environ["HERMES_HOME"] = str(served)
    assert [ss._is_process_home(launch), ss._is_process_home(served)] == [True, False]
    launch_scope, served_scope = owner_secret_scope(launch), owner_secret_scope(served)
    assert (launch_scope["PROFILE_MARKER"], launch_scope.get(LAUNCH_ONLY_KEY)) == (".hermes", LAUNCH_ONLY_VALUE)
    assert served_scope["PROFILE_MARKER"] == "served"
    assert LAUNCH_ONLY_KEY not in served_scope
    assert _bound_gate(launch) == (".hermes", "true")
    assert _bound_gate(served) == ("served", "")


def test_stable_alias_of_a_served_home_is_that_home(world, tmp_path):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    _launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "served-alias"
    alias.symlink_to(served, target_is_directory=True)
    for _ in range(2):
        scope = owner_secret_scope(alias)
        assert scope["PROFILE_MARKER"] == "served"
        assert LAUNCH_ONLY_KEY not in scope
        assert _bound_gate(alias) == ("served", "")


def test_retarget_between_the_verdict_and_the_read_cannot_split_files_from_authority(
        world, tmp_path, monkeypatch):
    """Not just a stale cache: a FRESH verdict taken before the ``.env`` open (two path
    resolutions, as at 92f9d132) still lets a retarget land in between. The verdict and the bytes must come from
    one opened directory, so whichever home the scope ends up reading, its authority matches."""
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "alias"
    alias.symlink_to(launch, target_is_directory=True)
    # The .env reader the producer calls (``read_env_file``; before it existed, ``load_env_file``).
    reader = "read_env_file" if hasattr(ss, "read_env_file") else "load_env_file"
    real_read = getattr(ss, reader)
    swapped = []

    def read_after_retarget(*args, **kwargs):
        if not swapped:
            _retarget(alias, served)
            swapped.append(True)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ss, reader, read_after_retarget)
    scope = owner_secret_scope(alias)
    assert swapped
    authority = {k: scope[k] for k in (LAUNCH_ONLY_KEY, "GATEWAY_ALLOW_ALL_USERS") if k in scope}
    assert (scope["PROFILE_MARKER"], authority) in [
        (".hermes", {LAUNCH_ONLY_KEY: LAUNCH_ONLY_VALUE, "GATEWAY_ALLOW_ALL_USERS": "true"}),
        ("served", {}),
    ], (scope["PROFILE_MARKER"], authority)
