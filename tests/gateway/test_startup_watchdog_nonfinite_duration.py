"""LEASE-6157-1 (review run6157): a non-finite lease duration must not disable the watchdog.

Before scoped leases, ``max(existing, now + nan)`` kept the existing deadline, so a NaN duration
claimed nothing and the ordinary expiry still fired. With per-claim leases a NaN deadline never
compares as expired, could never be pruned and was honored on every pass. These tests use only the
public API that predates the change, so they also run (and pass) on the baseline as controls.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import StartupWatchdogHandle, arm_startup_watchdog, report_startup_progress


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_cpu_progress(monkeypatch):
    monkeypatch.setattr(StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: 0.0))


class _ExitCapture:
    def __init__(self):
        self.codes: list[int] = []
        self.fired = threading.Event()

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.fired.set()


@pytest.fixture
def exit_capture(monkeypatch):
    capture = _ExitCapture()
    monkeypatch.setattr(StartupWatchdogHandle, "_exit", staticmethod(capture))
    return capture


def test_nan_progress_does_not_suppress_ordinary_expiry(monkeypatch):
    """Drives the real ``_run`` synchronously on this thread with a fixed clock (no watchdog thread,
    no sleeps), as in review run6157's bounded probe. The CPU seam disarms on its fifth read so a
    regressed build that keeps honoring the NaN claim returns (and fails) instead of spinning."""
    clock = [100.0]
    monkeypatch.setattr(sw, "time", SimpleNamespace(monotonic=lambda: clock[0], process_time=lambda: 0.0))
    monkeypatch.setattr(sw, "_log_quietly", lambda *args: None)
    handle = StartupWatchdogHandle(10.0, sw.SERVICE_RESTART_EXIT_CODE)
    monkeypatch.setattr(sw, "_handle", handle)
    report_startup_progress(float("nan"), phase="bad-duration")
    clock[0] = 111.0  # past the 110.0 deadline
    reads = [0]
    fired: list[int] = []

    def cpu_seconds():
        reads[0] += 1
        if reads[0] >= 5:
            handle.disarm()  # bound the loop on a regressed build
        return 0.0

    handle._process_cpu_seconds = cpu_seconds
    handle._fire = lambda: fired.append(handle.exit_code)
    handle._run()
    assert fired == [sw.SERVICE_RESTART_EXIT_CODE], (
        f"a NaN duration kept the watchdog from firing (cpu reads={reads[0]}, deadline={handle._deadline})")
    assert handle._deadline == handle._deadline, "the deadline must never become NaN"


def test_nan_progress_leaves_an_existing_lease_intact(exit_capture):
    handle = arm_startup_watchdog(timeout_s=30.0)
    report_startup_progress(60.0, phase="real")
    before = handle._lease_until
    report_startup_progress(float("nan"), phase="real")
    report_startup_progress(float("nan"), phase="other")
    assert handle._lease_until == before
    assert before == before and before > time.monotonic() + 50


def test_infinite_progress_is_clamped_to_the_maximum_lease():
    handle = arm_startup_watchdog(timeout_s=30.0)
    report_startup_progress(float("inf"), phase="unbounded")
    remaining = handle._lease_until - time.monotonic()
    assert sw._MAX_LEASE_S - 5 < remaining <= sw._MAX_LEASE_S


@pytest.mark.parametrize("value", [float("-inf"), -1.0, 0.0, "garbage", None])
def test_nonpositive_or_garbage_progress_claims_nothing(value, exit_capture):
    handle = arm_startup_watchdog(timeout_s=0.3)
    report_startup_progress(value, phase="nothing")  # type: ignore[arg-type]
    assert handle._lease_until == 0.0
    assert exit_capture.fired.wait(timeout=10)
