"""Provider-selected synchronous budgets must reach the host without removing its guard."""
import threading

import pytest

import agent.memory_manager as mm
from agent.memory_manager import MemoryManager
from tests.agent.test_memory_provider import FakeMemoryProvider


class BudgetProvider(FakeMemoryProvider):
    prefetch_timeout = 120.0

    def prefetch(self, query, *, session_id="", deadline=None):
        self.deadline = deadline
        return "current recall"


def test_provider_budget_reaches_wait_and_deadline(monkeypatch):
    waits = []
    original = mm.spawn_context_thread

    def spawn(target, **kwargs):
        thread = original(target, **kwargs)
        join = thread.join

        def observed_join(timeout):
            waits.append(timeout)
            join(5)

        thread.join = observed_join
        return thread

    monkeypatch.setattr(mm, "spawn_context_thread", spawn)
    p = BudgetProvider("budget")
    mgr = MemoryManager()
    mgr.add_provider(p)
    before = mm.time.monotonic()
    assert mgr.prefetch_all("current work") == "current recall"
    assert 120 <= waits[0] <= 122
    assert before + 120 <= p.deadline <= mm.time.monotonic() + 120
    assert not mgr._PREFETCH_GATE.calls


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan"), 1e20, "120", True])
def test_invalid_budget_never_starts_provider(budget):
    p = BudgetProvider("bad-budget")
    p.prefetch_timeout = budget
    mgr = MemoryManager()
    mgr.add_provider(p)
    assert mgr.prefetch_all("current work") == ""
    assert not hasattr(p, "deadline")
    assert not mgr._PREFETCH_GATE.calls


def test_legacy_signature_keeps_legacy_wait(monkeypatch):
    waits = []
    original = mm.spawn_context_thread

    def spawn(target, **kwargs):
        thread = original(target, **kwargs)
        join = thread.join

        def observed_join(timeout):
            waits.append(timeout)
            join(5)

        thread.join = observed_join
        return thread

    monkeypatch.setattr(mm, "spawn_context_thread", spawn)
    p = FakeMemoryProvider("legacy")
    p._prefetch_result = "cached recall"
    mgr = MemoryManager(external_prefetch_timeout=3)
    mgr.add_provider(p)
    assert mgr.prefetch_all("current work") == "cached recall"
    assert waits == [3]


def test_advertised_budget_does_not_allow_unbounded_stuck_work(monkeypatch):
    started, release = threading.Event(), threading.Event()
    workers = []
    original = mm.spawn_context_thread

    class Hanging(BudgetProvider):
        prefetch_timeout = 0.01

        def prefetch(self, query, *, session_id="", deadline=None):
            started.set()
            release.wait(5)
            return "late result"

    def spawn(target, **kwargs):
        thread = original(target, **kwargs)
        workers.append(thread)
        return thread

    monkeypatch.setattr(mm, "spawn_context_thread", spawn)
    monkeypatch.setattr(mm, "_MAX_STUCK_CALLS", 1)
    monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 1e9)
    p = Hanging("stuck-budget")
    mgr = MemoryManager()
    mgr.add_provider(p)
    try:
        assert mgr.prefetch_all("first") == ""
        assert started.is_set()
        assert mgr.prefetch_all("second") == ""
        assert len(workers) == 2  # second thread is allocated but never started
        assert workers[1].ident is None
    finally:
        release.set()
        workers[0].join(5)
        assert not workers[0].is_alive()
    assert not mgr._PREFETCH_GATE.calls


@pytest.mark.parametrize("finished_at, injected", [(119.999, True), (120.0, True), (120.001, False)])
def test_completion_boundary_not_later_caller_wakeup(monkeypatch, finished_at, injected):
    from types import SimpleNamespace
    from agent.memory_provider import RecallStatus

    main = threading.current_thread()
    ready = threading.Event()

    def monotonic():
        if threading.current_thread() is not main:
            return finished_at
        return 130.0 if ready.is_set() else 0.0

    class Timed(BudgetProvider):
        def prefetch(self, query, *, session_id="", deadline=None):
            ready.set()
            return "current recall"

        def recall_status(self):
            return RecallStatus("Timed", 1)

    monkeypatch.setattr(mm, "time", SimpleNamespace(monotonic=monotonic))
    p = Timed("boundary")
    mgr = MemoryManager()
    mgr.add_provider(p)
    assert bool(mgr.prefetch_all("current work")) is injected
    assert bool(mgr.describe_recall()) is injected
    assert monotonic() == 130
    assert not mgr._PREFETCH_GATE.calls
