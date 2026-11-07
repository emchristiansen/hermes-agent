"""Scoped startup phases: a finished phase's lease is never honored afterwards (D1b, run6005).

Observed on the 2026-11-07 gateway start: the last ``state_db_data_migrations`` lease was
claimed at ~11:08:41Z (600 s) by the last profile's ``SessionDB`` init, which finished
seconds later; at 11:13:26Z the watchdog still logged "phase 'state_db_data_migrations'
holds a progress lease for another 315s — honoring it", although no state.db work was in
flight. Leases were extend-only (``max()``) and never released, so the label named the
last phase that *claimed*, and the claim muted the watchdog for unrelated later work.

These tests pin the replacement semantics: scoped claims are released at phase end, a
deadline granted by a released claim is withdrawn, warnings name the holder and the open
phases, and startup steps log begin/end lines naming the DB/profile.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import (
    StartupWatchdogHandle,
    arm_startup_watchdog,
    begin_startup_phase,
    end_startup_phase,
    get_startup_watchdog_dump_path,
    report_startup_progress,
    startup_phase,
)


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


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _dump_records(tmp_path) -> list[dict]:
    path = get_startup_watchdog_dump_path(tmp_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


class TestScopedLeaseRelease:
    def test_released_phase_lease_does_not_protect(self, exit_capture):
        arm_startup_watchdog(timeout_s=0.3)
        token = begin_startup_phase("state_db_data_migrations", expected_s=60.0)
        end_startup_phase(token)
        assert exit_capture.fired.wait(timeout=5), "a released lease must not keep muting the watchdog"

    def test_open_scoped_lease_protects_then_release_withdraws_the_granted_deadline(self, exit_capture, caplog):
        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        handle = arm_startup_watchdog(timeout_s=0.3)
        with startup_phase("state_db_init_schema", expected_s=60.0, detail="db=/x/profiles/p/state.db profile=p"):
            assert _wait_for(lambda: any("honoring it" in m for m in _messages(caplog)), timeout=5)
            assert not exit_capture.fired.wait(timeout=0.6), "an open phase's lease must still be honored"
            assert handle._deadline - time.monotonic() > 30, "honoring extends to the lease deadline"
        # The deadline the lease granted is withdrawn at phase end: one ordinary window remains.
        assert handle._deadline - time.monotonic() <= 0.3 + 0.05
        assert exit_capture.fired.wait(timeout=5)
        assert handle._lease_release_count == 1

    def test_honoring_warning_names_holder_detail_and_open_phases(self, exit_capture, caplog):
        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        arm_startup_watchdog(timeout_s=0.3)
        with startup_phase("bot_mailbox_recovery", detail="profile=theseus-x db=/d"):
            with startup_phase("state_db_data_migrations", expected_s=60.0, detail="db=/d profile=theseus-x"):
                assert _wait_for(lambda: any("honoring it" in m for m in _messages(caplog)), timeout=5)
        warning = next(m for m in _messages(caplog) if "honoring it" in m)
        assert "'state_db_data_migrations' (db=/d profile=theseus-x)" in warning
        assert "open startup phases: bot_mailbox_recovery (profile=theseus-x db=/d) open" in warning
        assert "state_db_data_migrations (db=/d profile=theseus-x) open" in warning

    def test_logging_only_phase_claims_no_lease_and_is_named_in_fire_record(self, exit_capture, tmp_path):
        handle = arm_startup_watchdog(timeout_s=0.3)
        with startup_phase("gateway_runtime_api_start"):
            assert handle._lease_until == 0.0
            assert exit_capture.fired.wait(timeout=5), "attribution-only phases must not mute the watchdog"
        fired = [r for r in _dump_records(tmp_path) if r.get("tag") == "startup_watchdog.fired"]
        assert fired and any(p.startswith("gateway_runtime_api_start open") for p in fired[-1]["open_phases"])

    def test_failed_phase_logs_failed_and_releases(self, caplog):
        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        handle = arm_startup_watchdog(timeout_s=30.0)
        with pytest.raises(RuntimeError):
            with startup_phase("state_db_init_schema", expected_s=60.0, detail="db=/f"):
                raise RuntimeError("boom")
        assert handle._lease_until == 0.0
        assert handle._open_phases == {}
        assert any(m.startswith("Startup phase state_db_init_schema failed after") and "(db=/f)" in m
                   for m in _messages(caplog))

    def test_released_lease_cannot_be_honored_by_a_racing_check(self):
        handle = arm_startup_watchdog(timeout_s=30.0)
        token = begin_startup_phase("p", expected_s=60.0)
        end_startup_phase(token)
        assert handle._extend_for_lease_if_armed(time.monotonic()) is None

    def test_unscoped_report_keeps_time_lapse_semantics(self):
        handle = arm_startup_watchdog(timeout_s=30.0)
        report_startup_progress(60.0, phase="state_db_auto_vacuum")
        assert handle._lease_until > time.monotonic() + 50
        assert handle._lease_phase == "state_db_auto_vacuum"

    def test_startup_phase_is_silent_and_free_when_unarmed(self, caplog):
        caplog.set_level(logging.DEBUG, logger="hermes_startup_watchdog")
        with startup_phase("state_db_init_schema", expected_s=60.0, detail="db=/u"):
            pass
        assert not [m for m in _messages(caplog) if m.startswith("Startup phase")]


class TestRuntimeApiAttribution:
    def test_runtime_api_steps_log_begin_end_per_profile(self, monkeypatch, caplog):
        import gateway.run_api as run_api
        import gateway.run_runtime as run_runtime

        async def fake_start_gateway_api(runner):
            return SimpleNamespace(api_origin="http://127.0.0.1:0")

        recovered = []

        async def fake_recover(runner, authority):
            recovered.append(authority.profile_id)

        authorities = [
            SimpleNamespace(profile_id="/h", db=SimpleNamespace(db_path="/h/state.db")),
            SimpleNamespace(profile_id="/h/profiles/b", db=SimpleNamespace(db_path="/h/profiles/b/state.db")),
        ]
        registry = SimpleNamespace(profile_name=lambda a: None if a.profile_id == "/h" else "b")
        runner = SimpleNamespace(session_authorities=registry, session_runtime_descriptor={})
        monkeypatch.setattr(run_api, "start_gateway_api", fake_start_gateway_api)
        monkeypatch.setattr(run_runtime, "_authorities", lambda r: authorities)
        monkeypatch.setattr(run_runtime, "_recover_profile_bot_mailbox", fake_recover)

        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        arm_startup_watchdog(timeout_s=30.0)
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))

        assert recovered == ["/h", "/h/profiles/b"]
        messages = _messages(caplog)
        assert any(m.startswith("Startup phase gateway_runtime_api_start begin") for m in messages)
        assert any(m.startswith("Startup phase gateway_runtime_api_start end after") for m in messages)
        for detail in ("profile=launch db=/h/state.db", "profile=b db=/h/profiles/b/state.db"):
            assert any(m.startswith("Startup phase bot_mailbox_recovery begin") and detail in m for m in messages)
            assert any(m.startswith("Startup phase bot_mailbox_recovery end after") and detail in m for m in messages)
