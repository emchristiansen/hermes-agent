"""SCHEMA-6157-1 and SCHEMA-6157-2 (review run6157): the startup-phase wrappers on SessionDB schema steps.

1. Attribution must never change a step's call contract: ``_run_data_migrations`` accepts positional,
   keyword and mixed calls, armed or not, exactly as on the baseline (whose methods are unwrapped; the
   contract tests run there as controls).
2. The ``profile=`` label is the canonical owner of the DB's home (``profile_name_for_home``), including a
   profile legally named ``profiles``, and none for a custom home under an incidental ``profiles`` ancestor.
"""

from __future__ import annotations

import logging

import pytest

import hermes_startup_watchdog as sw
from hermes_constants import profile_name_for_home
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield home
    sw._reset_for_tests()


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "hermes_startup_watchdog"]


def _call_forms(db):
    return {
        "positional": lambda: db._run_data_migrations(db._conn.cursor(), SCHEMA_VERSION, True),
        "all_keyword": lambda: db._run_data_migrations(
            cursor=db._conn.cursor(), current_version=SCHEMA_VERSION, fts5_available=True),
        "mixed": lambda: db._run_data_migrations(db._conn.cursor(), SCHEMA_VERSION, fts5_available=True),
        "keyword_version": lambda: db._run_data_migrations(
            db._conn.cursor(), current_version=SCHEMA_VERSION, fts5_available=True),
    }


@pytest.mark.parametrize("armed", [False, True], ids=["unarmed", "armed"])
def test_data_migrations_accept_every_valid_call_form(armed, _isolate, caplog):
    db = SessionDB(db_path=_isolate / "state.db")
    try:
        caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
        if armed:
            sw.arm_startup_watchdog(timeout_s=600.0)
        for name, call in _call_forms(db).items():
            call()  # must not raise
        if armed and hasattr(sw, "startup_phase"):
            begins = [m for m in _messages(caplog) if m.startswith("Startup phase state_db_data_migrations begin")]
            assert len(begins) == 4 and all(f"from_version={SCHEMA_VERSION}" in m for m in begins), begins
    finally:
        db.close()


def test_data_migrations_still_reject_an_invalid_call(_isolate):
    db = SessionDB(db_path=_isolate / "state.db")
    try:
        with pytest.raises(TypeError):
            db._run_data_migrations(db._conn.cursor(), SCHEMA_VERSION, True, unexpected=1)
    finally:
        db.close()


_CASES = {
    "default_home": (lambda home, tmp: home / "state.db", "default"),
    "named_profile": (lambda home, tmp: home / "profiles" / "alpha" / "state.db", "alpha"),
    "profile_named_profiles": (lambda home, tmp: home / "profiles" / "profiles" / "state.db", "profiles"),
    "custom_home_under_incidental_profiles": (
        lambda home, tmp: tmp / "profiles" / "customer" / ".hermes" / "state.db", None),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_canonical_profile_owner_expectations(case, _isolate, tmp_path):
    """Control (runs on the baseline too): the resolver the label must follow."""
    make, expected = _CASES[case]
    db_path = make(_isolate, tmp_path)
    assert profile_name_for_home(db_path.parent) == expected, db_path


@pytest.mark.skipif(not hasattr(sw, "startup_phase"), reason="startup-phase attribution does not exist on the baseline")
@pytest.mark.parametrize("case", list(_CASES))
def test_startup_phase_profile_label_is_the_canonical_owner(case, _isolate, tmp_path, caplog):
    make, expected = _CASES[case]
    db_path = make(_isolate, tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    caplog.set_level(logging.INFO, logger="hermes_startup_watchdog")
    sw.arm_startup_watchdog(timeout_s=600.0)
    SessionDB(db_path=db_path).close()
    begins = [m for m in _messages(caplog) if m.startswith("Startup phase state_db_init_schema begin")]
    assert len(begins) == 1, (db_path, _messages(caplog))
    line = begins[0]
    assert f"db={db_path}" in line
    if expected is None:
        assert "profile=" not in line, line
    else:
        assert f"profile={expected})" in line or f"profile={expected} " in line, line
