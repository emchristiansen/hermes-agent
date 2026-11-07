"""Regression for the observed stale state.db lease (D1b, run6005); uses only APIs present before the fix.

2026-11-07 gateway start: the last profile's ``SessionDB`` init claimed ``state_db_data_migrations``
(600 s) at ~11:08:41Z and finished seconds later; at 11:13:26Z the watchdog logged "phase
'state_db_data_migrations' holds a progress lease for another 315s — honoring it" with no state.db
work in flight. Against X d27b53a5 this test fails the same way (the finished init's lease keeps the
watchdog muted and the warning names ``state_db_*``); with scoped phases it passes.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import StartupWatchdogHandle, arm_startup_watchdog


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


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "hermes_startup_watchdog"]


class TestStateDbInitScoping:
    def test_finished_state_db_init_lease_is_not_honored_afterwards(self, exit_capture, caplog, tmp_path):
        """The observed shape: a profile's SessionDB init (init_schema + data_migrations) finishes, then a
        later, unrelated startup step is still running when the deadline expires. Before the fix that step
        was covered by the finished DB's lease (up to 600 s) and the warning named ``state_db_*``; now no
        lease survives the init and the expiry is judged on CPU progress (frozen at zero here: it fires).

        Deterministic: the deadline is long while the store opens (opening scans /proc and can take more
        than a short timeout), then forced to expire once the open has returned."""
        from hermes_state import SessionDB

        db_path = tmp_path / "profiles" / "theseus-probe" / "state.db"
        db_path.parent.mkdir(parents=True)
        SessionDB(db_path=db_path).close()  # unarmed: creates the store (the second open runs data migrations)

        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        handle = arm_startup_watchdog(timeout_s=60.0)
        SessionDB(db_path=db_path).close()
        assert not exit_capture.fired.is_set(), "precondition: the deadline did not expire during the open"
        stale = handle._lease_until - time.monotonic()

        with handle._state_lock:
            handle._deadline = time.monotonic()  # the later startup step overruns now
        fired = exit_capture.fired.wait(timeout=15)
        honoring = [m for m in _messages(caplog) if "honoring it" in m]
        assert fired, f"a finished state.db init must not keep the watchdog muted; warnings seen: {honoring}"
        assert not [m for m in honoring if "state_db_" in m]
        assert stale <= 0, f"a finished state.db init still held a lease for {stale:.0f}s (last claim: {handle._lease_phase})"

    def test_state_db_init_steps_log_begin_end_naming_db_and_profile(self, caplog, tmp_path):
        from hermes_state import SessionDB

        db_path = tmp_path / "profiles" / "theseus-probe" / "state.db"
        db_path.parent.mkdir(parents=True)
        SessionDB(db_path=db_path).close()
        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        handle = arm_startup_watchdog(timeout_s=60.0)
        SessionDB(db_path=db_path).close()
        messages = _messages(caplog)
        assert getattr(handle, "_lease_release_count", 0) == 2
        for phase in ("state_db_init_schema", "state_db_data_migrations"):
            assert any(m.startswith(f"Startup phase {phase} begin") and f"db={db_path}" in m
                       and "profile=theseus-probe" in m for m in messages), phase
            assert any(m.startswith(f"Startup phase {phase} end after") and "profile=theseus-probe" in m
                       for m in messages), phase
        assert any("from_version=" in m and "to_version=" in m for m in messages)
