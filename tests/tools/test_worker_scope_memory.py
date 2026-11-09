"""A background command's systemd scope gets a ``MemoryMax`` only from a selected limit.

Production 2026-11-09: ordinary ``nix eval``/``nix flake check`` runs and Happier workers were
OOM-killed in their own ``hermes-worker-*.scope`` at a fixed 4 GiB while every enclosing cgroup was
unlimited; the cap came from host RAM, not from any configured limit. The selected limits stay: the
``TERMINAL_LOCAL_MEMORY_MAX_MB`` override and every finite ``memory.max`` on the spawner's cgroup and its
ancestors (its unit's ``MemoryMax``, a custom slice's), which the scope would otherwise leave behind.
"""
from pathlib import Path

import pytest

import tools.process_registry as pr

GIB = 1024 * 1024 * 1024


@pytest.fixture
def host(monkeypatch):
    """The spawner sits in ``app.slice/gw.service``; ``memory.max`` of that cgroup is settable.
    Physical RAM is irrelevant to the result, so it is made tiny and huge in turn by the tests."""
    files = {"/proc/self/cgroup": "0::/app.slice/gw.service\n",
             "/sys/fs/cgroup/app.slice/gw.service/memory.max": "max\n"}
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) in files:
            if isinstance(files[str(path)], BaseException):
                raise files[str(path)]
            return files[str(path)]
        if str(path).startswith(("/proc/self/cgroup", "/sys/fs/cgroup")):
            raise FileNotFoundError(str(path))
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pr.Path, "read_text", read_text)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)

    def set_limit(value: str) -> None:
        files["/sys/fs/cgroup/app.slice/gw.service/memory.max"] = value + "\n"
    set_limit.files = files
    return set_limit


def _ram(monkeypatch, total_bytes: int) -> None:
    monkeypatch.setattr(pr.os, "sysconf", lambda name: {"SC_PHYS_PAGES": total_bytes // 4096,
                                                        "SC_PAGE_SIZE": 4096}[name])


def _scope_properties() -> list[str]:
    argv = pr._build_systemd_scope_argv(["/bin/bash", "-lc", "true"], unit_suffix="proc_test")
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--property"]


@pytest.mark.parametrize("ram", [2 * GIB, 1024 * GIB])
def test_an_unlimited_spawner_without_an_override_gets_no_memory_max(host, monkeypatch, ram):
    _ram(monkeypatch, ram)
    assert pr._worker_memory_max_bytes() is None
    assert _scope_properties() == ["MemoryAccounting=yes"]


def test_the_spawners_finite_limit_is_carried_into_its_scope(host, monkeypatch):
    _ram(monkeypatch, 1024 * GIB)
    host(str(3 * GIB))
    assert _scope_properties() == ["MemoryAccounting=yes", f"MemoryMax={3 * GIB}"]


def test_a_finite_limit_on_any_ancestor_of_the_spawner_is_carried(host, monkeypatch):
    """The scope may land in another slice (``app.slice``), so a custom ``Slice=``'s limit is carried too."""
    _ram(monkeypatch, 1024 * GIB)
    host.files["/sys/fs/cgroup/app.slice/memory.max"] = f"{2 * GIB}\n"
    assert _scope_properties() == ["MemoryAccounting=yes", f"MemoryMax={2 * GIB}"]
    host(str(GIB))
    assert pr._worker_memory_max_bytes() == GIB


def test_a_small_spawner_limit_is_carried_as_it_is(host, monkeypatch):
    """The 64 MiB floor screens the override only; a configured cgroup limit is never dropped."""
    _ram(monkeypatch, 1024 * GIB)
    host(str(32 * 1024 * 1024))
    assert pr._worker_memory_max_bytes() == 32 * 1024 * 1024


@pytest.mark.parametrize(("override_mb", "spawner_limit", "expected"), [
    ("512", "max", 512 * 1024 * 1024),          # the override alone is the selected limit
    ("16384", "max", 16 * GIB),                 # not clipped to any host-derived ceiling
    ("16384", str(GIB), GIB),                   # never above the spawner's own limit
    ("512", str(GIB), 512 * 1024 * 1024),
    ("32", "max", None),                        # below the 64 MiB floor: ignored with a warning
    ("not-a-number", "max", None),
])
def test_the_override_is_a_selected_limit_bounded_by_the_spawners_own(
        host, monkeypatch, override_mb, spawner_limit, expected):
    _ram(monkeypatch, 8 * GIB)
    host(spawner_limit)
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", override_mb)
    assert pr._worker_memory_max_bytes() == expected


_DENIED = PermissionError(13, "Permission denied")


@pytest.mark.parametrize(("levels", "override_mb", "expected"), [
    # own level unreadable, readable selected parent: carried, with no override and under a looser one
    ({"limited.slice/worker.service": _DENIED, "limited.slice": "33554432"}, None, 33554432),
    ({"limited.slice/worker.service": _DENIED, "limited.slice": "33554432"}, "16384", 33554432),
    # a closer readable bound, an unreadable middle level, a stricter ancestor: the minimum of the readable ones
    ({"limited.slice/mid.slice/worker.service": str(GIB), "limited.slice/mid.slice": _DENIED,
      "limited.slice": "33554432"}, None, 33554432),
    ({"limited.slice/mid.slice/worker.service": "33554432", "limited.slice/mid.slice": _DENIED,
      "limited.slice": str(GIB)}, None, 33554432),
    # nothing readable and nothing selected: no invented cap; an override alone still applies
    ({"limited.slice/worker.service": _DENIED, "limited.slice": _DENIED}, None, None),
    ({"limited.slice/worker.service": _DENIED, "limited.slice": _DENIED}, "512", 512 * 1024 * 1024),
    # an unreadable level beside an unlimited one, and a missing level: nothing selected
    ({"limited.slice/worker.service": _DENIED, "limited.slice": "max"}, None, None),
    ({"limited.slice/worker.service": "max"}, None, None),
], ids=["denied-own-parent", "denied-own-parent-looser-override", "closer-denied-stricter",
        "stricter-denied-looser", "all-denied", "all-denied-override", "denied-and-unlimited", "missing-level"])
def test_an_unreadable_cgroup_level_never_hides_a_readable_selected_bound(
        host, monkeypatch, levels, override_mb, expected):
    """MEMCG-R5-08: a PermissionError (or any read failure) at one level skips that level only; every other
    readable finite ``memory.max`` and a valid override are still compared, through the real scope argv."""
    _ram(monkeypatch, 1024 * GIB)
    host.files.clear()
    own = max(levels, key=len)
    host.files["/proc/self/cgroup"] = f"0::/{own}\n"
    host.files.update({f"/sys/fs/cgroup/{level}/memory.max": value if isinstance(value, BaseException)
                       else value + "\n" for level, value in levels.items()})
    if override_mb is not None:
        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", override_mb)
    assert pr._worker_memory_max_bytes() == expected
    assert _scope_properties() == ["MemoryAccounting=yes", *([f"MemoryMax={expected}"] if expected else [])]
