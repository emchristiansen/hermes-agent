"""Auto-decompose fails closed on a failed config read (SKL-LKG-1).

``_resolve_auto_decompose_settings`` promises ``(False, 3)`` on a config read error: auto-decompose is
the live safety toggle a user flips to stop unwanted fan-out. The real ``load_config`` never raises on a
broken config.yaml; it serves defaults (cold) or the last good config (hot) as a ``FailedConfigRead``.
The resolver must treat that fallback as the read error it is, while the loader keeps serving the
last good config to every other consumer. Everything below goes through the real loader, managed
overlay and resolver; the dispatch-loop case drives the real ``_kanban_dispatcher_watcher`` with a
recording dispatcher in place of the board work.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_common import _resolve_auto_decompose_settings
from tests.hermes_cli.test_managed_scope_lkg_retention import _reset_caches

BROKEN = "kanban: [\n"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_caches()
    yield home, managed
    _reset_caches()


def _load():
    from hermes_cli.config import load_config
    return load_config()


@pytest.mark.parametrize("warm", [False, True], ids=["cold-defaults", "hot-last-good-true"])
def test_failed_read_disables_auto_decompose(homes, warm):
    from hermes_cli.config_read_errors import FailedConfigRead
    home, _ = homes
    if warm:
        (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n  auto_decompose_per_tick: 5\n")
        assert _resolve_auto_decompose_settings(_load) == (True, 5)
    (home / "config.yaml").write_text(BROKEN)
    served = _load()
    assert isinstance(served, FailedConfigRead)
    observed = [_resolve_auto_decompose_settings(_load) for _ in range(2)]  # second: cached fallback
    print("REACHED-RESOLVER", warm, type(served).__name__, observed, flush=True)
    if warm:  # the loader still serves the last good config to the consumers that rely on it
        assert served["kanban"]["auto_decompose"] is True
    assert observed == [(False, 3), (False, 3)]


@pytest.mark.parametrize("value,expected", [("true", (True, 3)), ("false", (False, 3))])
def test_valid_config_controls(homes, value, expected):
    home, _ = homes
    (home / "config.yaml").write_text(f"kanban:\n  auto_decompose: {value}\n")
    assert _resolve_auto_decompose_settings(_load) == expected


def test_absent_config_keeps_the_default_on(homes):
    assert _resolve_auto_decompose_settings(_load) == (True, 3)  # no file is not a failed read


def test_repaired_config_reenables_on_next_read(homes):
    home, _ = homes
    (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n")
    assert _resolve_auto_decompose_settings(_load) == (True, 3)
    (home / "config.yaml").write_text(BROKEN)
    assert _resolve_auto_decompose_settings(_load) == (False, 3)
    (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n  auto_decompose_per_tick: 2\n")
    assert _resolve_auto_decompose_settings(_load) == (True, 2)


def test_managed_disable_over_broken_hot_yaml(homes):
    home, managed = homes
    (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n")
    assert _resolve_auto_decompose_settings(_load) == (True, 3)
    (home / "config.yaml").write_text(BROKEN)
    (managed / "config.yaml").write_text("kanban:\n  auto_decompose: false\n")
    assert _resolve_auto_decompose_settings(_load) == (False, 3)
    assert _resolve_auto_decompose_settings(_load) == (False, 3)


def test_raising_loader_still_fails_closed():
    def boom():
        raise OSError("EMFILE")
    assert _resolve_auto_decompose_settings(boom) == (False, 3)


class _Recorder:
    def __init__(self, runner, ticks):
        self.runner, self.ticks, self.calls = runner, ticks, []

    def auto_decompose_tick(self, per_tick):
        self.calls.append(("auto_decompose_tick", per_tick))
        return 0

    def tick_once(self):
        self.calls.append(("tick_once",))
        if sum(1 for c in self.calls if c[0] == "tick_once") >= self.ticks:
            self.runner._running = False
        return []

    def ready_nonempty(self):
        return False


def _drive_dispatch_loop(monkeypatch, ticks=2):
    """Run the real embedded dispatcher loop with the real config loader for *ticks* ticks."""
    import gateway.kanban_watchers as kw
    from hermes_cli import kanban_db_dispatch
    from hermes_cli.config import load_config
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    recorder = _Recorder(runner, ticks)

    async def direct(fn, *args):
        return fn(*args)

    async def no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (load_config, SimpleNamespace(), {}), raising=False)
    monkeypatch.setattr(runner, "_sleep_between_ticks", no_sleep, raising=False)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings", lambda cfg, kb: SimpleNamespace(interval=1.0))
    monkeypatch.setattr(kw, "_KanbanDispatcher", lambda kb, settings: recorder)
    monkeypatch.setattr(kw, "_to_thread_process_service", direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kanban_db_dispatch, "reap_worker_zombies", list)
    monkeypatch.setattr(kw.asyncio, "sleep", no_sleep)  # the loop's initial 5s adapter delay
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=30))
    return recorder.calls


@pytest.mark.parametrize("warm", [False, True], ids=["cold-defaults", "hot-last-good-true"])
def test_dispatch_loop_never_enters_auto_decompose_on_failed_read(homes, monkeypatch, warm):
    from hermes_cli.config_read_errors import FailedConfigRead
    home, _ = homes
    if warm:
        (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n")
        assert not isinstance(_load(), FailedConfigRead)
    (home / "config.yaml").write_text(BROKEN)
    calls = _drive_dispatch_loop(monkeypatch)
    print("REACHED-DISPATCH-DECISION", warm, calls, flush=True)  # two real loop ticks ran
    assert calls == [("tick_once",), ("tick_once",)]  # dispatch continues; auto-decompose never entered


def test_dispatch_loop_control_enters_auto_decompose_on_valid_true(homes, monkeypatch):
    home, _ = homes
    (home / "config.yaml").write_text("kanban:\n  auto_decompose: true\n  auto_decompose_per_tick: 4\n")
    calls = _drive_dispatch_loop(monkeypatch)
    print("REACHED-DISPATCH-DECISION", "valid-true", calls, flush=True)
    assert calls == [("auto_decompose_tick", 4), ("tick_once",), ("auto_decompose_tick", 4), ("tick_once",)]
