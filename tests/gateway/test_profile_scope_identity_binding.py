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


# --- Without a directory descriptor (review t_983c1ee8 run 2525, SCOPE-FALLBACK-01) -------------
#
# Where no directory descriptor is available (no ``dir_fd`` support, e.g. Windows; or a home that
# refuses one on a platform without O_PATH) ``.env`` is read by path. 7bcce995 then took the
# launch-home verdict from a path resolution made AFTER the read: read the served home's ``.env``
# through S, retarget S to the launch home, and the served files came back with the launch env's
# credential and the allow-all grant. Each case forces that branch explicitly.


@pytest.fixture(params=["no-dirfd", "dirfd-refused"])
def path_read(request, monkeypatch):
    """Force the no-descriptor branch: no ``dir_fd`` support, or the home refusing the descriptor."""
    if request.param == "no-dirfd":
        monkeypatch.setattr(ss, "_HOME_DIR_FD", False)
    else:
        real_open = os.open

        def refusing_open(path, flags, *args, **kwargs):
            if flags == ss._HOME_DIR_FLAGS and "dir_fd" not in kwargs:
                raise PermissionError(13, "Permission denied", os.fspath(path))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", refusing_open)
    return request.param


def _on_env_open(monkeypatch, before=None, after=None):
    """Run *before*/*after* around the FIRST open of a ``.env`` by the scope builder, whichever
    reader it goes through; returns the list that records that it happened."""
    import builtins
    real_open = builtins.open
    fired = []

    def hooked(file, *args, **kwargs):
        if fired or not os.fspath(file).endswith(".env"):
            return real_open(file, *args, **kwargs)
        fired.append(os.fspath(file))
        if before:
            before()
        handle = real_open(file, *args, **kwargs)
        if after:
            after()
        return handle

    monkeypatch.setattr(ss, "open", hooked, raising=False)
    return fired


def _authority(scope) -> tuple[str | None, dict[str, str], str]:
    """(marker, launch-only keys in the mapping, allow-all gate as a turn bound to it reads it)."""
    from gateway.platforms._shared import platform_gate_env
    held = {k: scope[k] for k in (LAUNCH_ONLY_KEY, "GATEWAY_ALLOW_ALL_USERS") if k in scope}
    token = ss.set_secret_scope(scope, profile_home="/bound-by-test")
    try:
        return scope.get("PROFILE_MARKER"), held, platform_gate_env("GATEWAY_ALLOW_ALL_USERS")
    finally:
        ss.reset_secret_scope(token)


_LAUNCH_AUTHORITY = {LAUNCH_ONLY_KEY: LAUNCH_ONLY_VALUE, "GATEWAY_ALLOW_ALL_USERS": "true"}


def test_path_read_of_a_served_home_then_retarget_to_launch_gains_no_launch_authority(
        world, tmp_path, monkeypatch, path_read):
    """The reviewer's witness: S names the served home while its ``.env`` is read, then S is
    retargeted to the launch home before any later identity check."""
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "alias"
    alias.symlink_to(served, target_is_directory=True)
    fired = _on_env_open(monkeypatch, after=lambda: _retarget(alias, launch))
    scope = owner_secret_scope(alias)
    assert fired == [str(alias / ".env")]
    assert _authority(scope) == ("served", {}, ""), "served files + launch authority"


def test_path_read_of_another_homes_env_between_two_launch_resolutions_gains_none(
        world, tmp_path, monkeypatch, path_read):
    """S names the launch home before and after the read, but the ``.env`` opened was the served
    home's: agreement of two resolutions is not enough, the file read must be the launch one."""
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "alias"
    alias.symlink_to(launch, target_is_directory=True)
    fired = _on_env_open(monkeypatch, before=lambda: _retarget(alias, served),
                         after=lambda: _retarget(alias, launch))
    scope = owner_secret_scope(alias)
    assert fired
    assert _authority(scope) == ("served", {}, ""), "served files + launch authority"


def test_path_read_of_the_launch_home_then_retarget_away_is_never_mixed(
        world, tmp_path, monkeypatch, path_read):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "alias"
    alias.symlink_to(launch, target_is_directory=True)
    fired = _on_env_open(monkeypatch, after=lambda: _retarget(alias, served))
    scope = owner_secret_scope(alias)
    assert fired
    # Launch files with launch authority, or (failing closed) without it; never another mix.
    assert _authority(scope) in [(".hermes", _LAUNCH_AUTHORITY, "true"), (".hermes", {}, "")]


def test_path_read_keeps_launch_authority_for_the_launch_home_and_only_it(world, tmp_path, path_read):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    launch, served = world
    _activate_with_bridged_allow_all()
    alias = tmp_path / "launch-alias"
    alias.symlink_to(launch, target_is_directory=True)
    for home in (launch, alias):
        assert _authority(owner_secret_scope(home)) == (".hermes", _LAUNCH_AUTHORITY, "true")
    assert _authority(owner_secret_scope(served)) == ("served", {}, "")
    (launch / ".env").unlink()  # a launch home with no .env keeps its env-only credentials
    assert _authority(owner_secret_scope(launch)) == (None, _LAUNCH_AUTHORITY, "true")


def test_launch_home_not_created_yet_keeps_its_launch_credentials(tmp_path, monkeypatch, world):
    from tui_gateway.launch_profile_policy import owner_secret_scope

    _launch, served = world
    missing = tmp_path / "not-created" / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(missing))
    hermes_constants.reset_hermes_home_key_cache()
    assert owner_secret_scope(missing).get(LAUNCH_ONLY_KEY) == LAUNCH_ONLY_VALUE
    assert LAUNCH_ONLY_KEY not in owner_secret_scope(served)
