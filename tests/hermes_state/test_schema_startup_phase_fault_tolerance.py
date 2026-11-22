"""SCHEMA-6220-1 (review run6220): startup-phase diagnostics must never fail a real SessionDB open.

Composition through the real ``SessionDB`` constructor (real SQLite, real schema init, a real
threadless watchdog handle, DEBUG enabled), injecting only diagnostic faults: a ``_startup_phase_detail``
that raises and a watchdog log handler whose ``emit`` raises. With either fault, both, or neither the
store must open at the current schema with no lease or open phase left behind. The step's own errors
must still propagate unchanged, with phases released. On the baseline (no attribution) the patched
detail helper is simply unused, so the matrix runs there as a control.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from unittest import mock

import pytest

import hermes_startup_watchdog as sw
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION
from hermes_state_schema import SessionSchemaMixin


class _BrokenHandler(logging.Handler):
    def emit(self, record):
        raise RuntimeError("diagnostic sink failed")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    previous = sw.logger.level
    sw.logger.setLevel(logging.DEBUG)
    handle = sw.StartupWatchdogHandle(300.0, sw.SERVICE_RESTART_EXIT_CODE)  # armed state, no thread
    monkeypatch.setattr(sw, "_handle", handle)
    yield home, handle
    sw.logger.setLevel(previous)
    sw._reset_for_tests()


@contextlib.contextmanager
def _faults(bad_log: bool, bad_detail: bool):
    handler = _BrokenHandler()
    if bad_log:
        sw.logger.addHandler(handler)
    detail = (mock.patch.object(SessionSchemaMixin, "_startup_phase_detail",
                                side_effect=ValueError("detail failed"), create=True)
              if bad_detail else contextlib.nullcontext())
    try:
        with detail:
            yield
    finally:
        sw.logger.removeHandler(handler)


def _claims(handle):
    return len(getattr(handle, "_leases", {})), len(getattr(handle, "_open_phases", {}))


@pytest.mark.parametrize("bad_log", [False, True], ids=["log_ok", "log_raises"])
@pytest.mark.parametrize("bad_detail", [False, True], ids=["detail_ok", "detail_raises"])
def test_session_db_opens_whatever_the_diagnostics_do(bad_log, bad_detail, _isolate):
    home, handle = _isolate
    path = home / f"log{int(bad_log)}-detail{int(bad_detail)}.db"
    with _faults(bad_log, bad_detail):
        db = SessionDB(path)
        try:
            assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        finally:
            db.close()
    assert _claims(handle) == (0, 0)


@pytest.mark.parametrize("bad_log", [False, True], ids=["log_ok", "log_raises"])
@pytest.mark.parametrize("bad_detail", [False, True], ids=["detail_ok", "detail_raises"])
def test_step_errors_still_propagate_and_release_phases(bad_log, bad_detail, _isolate, monkeypatch):
    home, handle = _isolate

    def failing_reconcile(self, cursor):
        raise sqlite3.OperationalError("schema step failed for real")

    monkeypatch.setattr(SessionSchemaMixin, "_reconcile_columns", failing_reconcile)
    with _faults(bad_log, bad_detail), pytest.raises(sqlite3.OperationalError, match="schema step failed for real"):
        SessionDB(home / "failing.db")
    assert _claims(handle) == (0, 0)
