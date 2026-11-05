"""Filesystem work a multiplexed profile scope entry does on the calling (event-loop) thread.

``owner_scope`` / ``_profile_runtime_scope(..., hydrate_secrets=False)`` is entered synchronously on
the gateway's event loop by ~30 session/RPC paths, once per entry. Under a CPU-busy gateway every
blocking syscall there gives the GIL away and waits to get it back, so loop-lag samples landed on
exactly these calls: ``secret_scope._is_process_home`` -> ``Path.resolve`` (1.75 s) and
``build_profile_terminal_scope`` -> ``load_env_file`` (1.38 s).

The contract under test, counted rather than timed:
- a warm entry walks no home path (no ``lstat``): home identity comes from ``hermes_home_key``;
- the profile ``.env`` is opened once for the secret scope and once for the terminal policy, not a
  third time for a separate exists()/read probe;
- neither shortcut weakens freshness or scoping: an edited ``.env`` (even same-size inside one
  mtime tick) is seen by the very next entry, the served profile never sees the launch profile's
  values, and an unreadable ``.env`` still fails the terminal policy closed.

Files are written as bytes so the same-size rewrite stays same-size on every platform.
"""

from __future__ import annotations

import builtins
import io
import os
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

import agent.secret_scope as ss
import hermes_constants

# Past the coarsest mtime resolution in common use (see test_secret_scope_env_cache.py).
_MTIME_STEP_NS = 10_000_000_000


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch" / ".hermes"
    profile = launch / "profiles" / "p1"
    for home, key, timeout in ((launch, b"launch-key", b"11"), (profile, b"p1-key-aa", b"22")):
        home.mkdir(parents=True)
        (home / ".env").write_bytes(b"OPENAI_API_KEY=" + key + b"\nTERMINAL_TIMEOUT=" + timeout + b"\n")
        (home / "config.yaml").write_bytes(b"model:\n  provider: openrouter\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    hermes_constants.reset_hermes_home_key_cache()
    ss.invalidate_env_file_cache()
    yield launch, profile
    hermes_constants.reset_hermes_home_key_cache()
    ss.invalidate_env_file_cache()


def _enter(home: Path, probe=None):
    """Enter *home*'s scope; return what it binds (or what *probe* returns inside it)."""
    from gateway.run import _profile_runtime_scope
    from tools.terminal_scope import terminal_env

    with _profile_runtime_scope(home, hydrate_secrets=False):
        if probe is not None:
            return probe()
        return ss.get_secret("OPENAI_API_KEY"), terminal_env("TERMINAL_TIMEOUT")


_REPO = Path(__file__).resolve().parents[2]
_TESTS = _REPO / "tests"


def _hermes_caller() -> bool:
    """Is the nearest repo frame above the spy Hermes code (not the test harness's own I/O guard)?"""
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(str(_REPO)):
            return not filename.startswith(str(_TESTS))
        frame = frame.f_back
    return False


def _count_fs_ops(monkeypatch, env_path: Path) -> Counter:
    """Count the path walks (``lstat``), stats and opens of *env_path* Hermes makes on this thread."""
    counts: Counter = Counter()
    me = threading.get_ident()
    target = os.fspath(env_path)

    def spy(name, real):
        def wrapper(*args, **kwargs):
            if threading.get_ident() == me and _hermes_caller():
                counts[name] += 1
                if name == "open" and args and os.fspath(args[0]) == target:
                    counts["open .env"] += 1
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(os, "lstat", spy("lstat", os.lstat))
    monkeypatch.setattr(os, "stat", spy("stat", os.stat))
    monkeypatch.setattr(builtins, "open", spy("open", builtins.open))
    monkeypatch.setattr(io, "open", spy("open", io.open))
    return counts


def test_warm_scope_entry_walks_no_home_path_and_opens_the_env_twice(homes, monkeypatch):
    _launch, profile = homes
    assert _enter(profile) == ("p1-key-aa", "22")  # warm: home keys and the .env memo populated

    counts = _count_fs_ops(monkeypatch, profile / ".env")
    entries = 5
    for _ in range(entries):
        assert _enter(profile) == ("p1-key-aa", "22")
    monkeypatch.undo()

    per_entry = {name: n / entries for name, n in counts.items()}
    # Before: 16 lstat (two resolve() walks in _is_process_home, one in get_secret_source_values)
    # and 3 opens of the .env (secret scope, terminal exists()/read_bytes() probe, terminal parse).
    assert per_entry.get("lstat", 0) == 0, per_entry
    assert per_entry.get("open .env", 0) == 2, per_entry


def test_edited_env_is_seen_by_the_next_entry_even_same_size_in_one_tick(homes):
    _launch, profile = homes
    env = profile / ".env"
    assert _enter(profile) == ("p1-key-aa", "22")

    before = env.stat()
    env.write_bytes(b"OPENAI_API_KEY=p1-key-bb\nTERMINAL_TIMEOUT=33\n")  # same size
    assert env.stat().st_size == before.st_size
    os.utime(env, ns=(before.st_mtime_ns, before.st_mtime_ns))  # and the same mtime
    assert _enter(profile) == ("p1-key-bb", "33")

    env.write_bytes(b"OPENAI_API_KEY=p1-rotated-longer-key\n")  # TERMINAL_TIMEOUT removed
    bumped = before.st_mtime_ns + _MTIME_STEP_NS
    os.utime(env, ns=(bumped, bumped))
    key, timeout = _enter(profile)
    assert key == "p1-rotated-longer-key"
    assert timeout != "33"  # back to the policy default, never the stale value

    env.unlink()
    assert _enter(profile)[0] is None  # gone: no stale secret survives, launch value not borrowed


def test_served_profile_never_resolves_as_the_launch_home(homes, tmp_path):
    launch, profile = homes
    assert ss._is_process_home(launch) is True
    assert ss._is_process_home(profile) is False
    alias = tmp_path / "alias-to-launch"
    alias.symlink_to(launch, target_is_directory=True)
    assert ss._is_process_home(alias) is True  # identity is the resolved home, not the spelling
    # Each tenant reads only its own .env, whichever entered first.
    assert _enter(profile) == ("p1-key-aa", "22")
    assert _enter(launch) == ("launch-key", "11")
    assert _enter(profile) == ("p1-key-aa", "22")


def test_unreadable_env_still_fails_the_terminal_policy_closed(homes):
    from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope

    _launch, profile = homes
    env = profile / ".env"
    env.unlink()
    assert build_profile_terminal_scope(profile)["TERMINAL_DOCKER_IMAGE_PINNED"] == "0"  # absent: fine
    env.mkdir()  # present but unreadable as a file
    with pytest.raises(TerminalPolicyUnavailable, match="cannot read"):
        build_profile_terminal_scope(profile)

    def run_terminal():
        from tools.terminal_scope import enforce_no_refusal
        with pytest.raises(TerminalPolicyUnavailable):
            enforce_no_refusal()

    _enter(profile, probe=run_terminal)  # the scope installs the refusal, it does not fall back
