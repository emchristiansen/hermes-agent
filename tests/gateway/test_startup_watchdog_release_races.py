"""Deterministic interleavings at the watchdog's fire decision (curator run6093 race questions 1 and 2).

``_run`` decides to fire after (a) reading an expired deadline, (b) finding no live lease and (c)
finding no CPU progress. A concurrent ``end_phase`` (which withdraws a lease-granted deadline and
opens a fresh window) or ``begin_phase`` (a new live lease) can land between (b) and the fire
transition. Each test injects that operation at exactly that point, on the watchdog thread itself,
through the existing ``_process_cpu_seconds`` seam (step (c) is its last call before the fire), so
the interleaving is reached without sleeps. The fire transition must re-check under the state lock.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import (
    StartupWatchdogHandle,
    arm_startup_watchdog,
    begin_startup_phase,
    end_startup_phase,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


class _ExitCapture:
    def __init__(self):
        self.codes: list[int] = []
        self.at: list[float] = []
        self.fired = threading.Event()

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.at.append(time.monotonic())
        self.fired.set()


@pytest.fixture
def exit_capture(monkeypatch):
    capture = _ExitCapture()
    monkeypatch.setattr(StartupWatchdogHandle, "_exit", staticmethod(capture))
    return capture


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "hermes_startup_watchdog"]


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_release_landing_after_the_no_lease_check_is_not_fired_through(exit_capture, monkeypatch):
    """Race 1: the honored lease lapses by time while its phase is still open; the watchdog finds no
    live lease, and the phase then ends (withdrawing the lease-granted deadline: one fresh window)
    before the fire transition. Firing on the stale expired reading would ignore that window."""
    timeout = 1.0
    state = {"handle": None, "token": None, "released_at": None}

    def cpu_seam():
        handle, token = state["handle"], state["token"]
        if (handle is not None and token is not None and state["released_at"] is None
                and handle._deadline_lease_key is token.key and not handle._leases):
            # Step (c) of a pass that already found no live lease: the phase ends right here.
            end_startup_phase(token)
            state["released_at"] = time.monotonic()
        return 0.0

    monkeypatch.setattr(StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(cpu_seam))
    handle = arm_startup_watchdog(timeout_s=timeout)
    state["handle"] = handle
    state["token"] = begin_startup_phase("slow_phase", expected_s=timeout * 1.5)
    assert exit_capture.fired.wait(timeout=15), "the watchdog must still fire once the fresh window lapses"
    assert state["released_at"] is not None, "the interleaving was not reached"
    waited = exit_capture.at[0] - state["released_at"]
    assert waited >= timeout * 0.9, (
        f"fired {waited:.3f}s after the release withdrew the deadline; the fresh {timeout}s window was ignored")


def test_lease_claimed_after_the_no_lease_check_is_not_fired_through(exit_capture, monkeypatch, caplog):
    """Race 2: no lease is held when the deadline expires; a phase claims one before the fire
    transition. The claim is live when the watchdog would fire, so it must be honored."""
    caplog.set_level(logging.WARNING, logger="hermes_startup_watchdog")
    state = {"handle": None, "token": None}

    def cpu_seam():
        handle = state["handle"]
        if (handle is not None and state["token"] is None and not handle._leases
                and time.monotonic() >= handle._deadline):
            state["token"] = begin_startup_phase("late_claim", expected_s=60.0)
        return 0.0

    monkeypatch.setattr(StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(cpu_seam))
    state["handle"] = arm_startup_watchdog(timeout_s=0.5)
    assert _wait_for(lambda: state["token"] is not None or exit_capture.fired.is_set())
    assert state["token"] is not None, "the interleaving was not reached"
    honored = _wait_for(lambda: exit_capture.fired.is_set()
                        or any("late_claim" in m and "honoring it" in m for m in _messages(caplog)))
    assert honored and not exit_capture.fired.is_set(), "a live lease claimed before the fire transition was fired through"
    end_startup_phase(state["token"])
    assert exit_capture.fired.wait(timeout=15), "after release the ordinary fallback still fires"


def test_plain_expiry_without_progress_still_fires(exit_capture, monkeypatch):
    """Control: with no lease, no release and no CPU, the re-check does not prevent the ordinary fire."""
    monkeypatch.setattr(StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: 0.0))
    arm_startup_watchdog(timeout_s=0.3)
    assert exit_capture.fired.wait(timeout=10)
    assert exit_capture.codes == [sw.SERVICE_RESTART_EXIT_CODE]
