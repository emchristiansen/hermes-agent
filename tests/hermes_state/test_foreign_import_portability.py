"""A CLI foreign import's boundary survives native backup/restore and the cross-store move (review t_42a8d5e2 AGENT-02).

A CLI import of a foreign transcript is an ordinary ``cli`` row with no ``origin_json``; its ``imported_from`` record and
boundary (``through_message_id``) are kept in ``state_meta`` (``hermes_state_portability.FOREIGN_IMPORT_META_PREFIX``).
Home guidance reads that boundary to tell imported replies from replies written here
(``agent/home_guidance.py::foreign_import_view`` / ``continues_without_prompt``). An export carries the record beside the
session and the import remaps its boundary to the destination's own row ids: the restored history keeps its
first-turn decision, an import already continued here stays continued, and the row stays a writable CLI row (no
``origin_json``). Real SessionDB stores over SQLite; real export, import, move and guidance readers.
"""
import json

import pytest

from agent.home_guidance import continues_without_prompt, foreign_import_view, stored_messages
from hermes_state import SessionDB
from hermes_state_portability import FOREIGN_IMPORT_META_PREFIX, FOREIGN_IMPORT_THROUGH, record_foreign_import_meta

SID = "cli-foreign-import"
ORIGIN = {"tool": "claude", "path": "transcript.json", "foreign_session_id": "foreign-1"}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    source, target = SessionDB(tmp_path / "source.db"), SessionDB(tmp_path / "target.db")
    yield source, target
    source.close()
    target.close()


def _cli_import(db, *, continued=False):
    """A CLI import of a foreign user/assistant exchange (``hermes_state_mutations``' record), optionally continued
    by a real Hermes exchange after it."""
    db.create_session(SID, source="cli")
    db.append_message(SID, "user", "imported question")
    db.append_message(SID, "assistant", "imported answer, not a Hermes turn")
    db._execute_write(lambda conn: record_foreign_import_meta(conn, SID, ORIGIN))
    if continued:
        db.append_message(SID, "user", "a question asked here")
        db.append_message(SID, "assistant", "a Hermes reply")


def _continuing(db):
    return continues_without_prompt(foreign_import_view(db, db.get_session(SID)), stored_messages(db, SID))


def _through(db):
    record = db.get_meta(FOREIGN_IMPORT_META_PREFIX + SID)
    return json.loads(record)[FOREIGN_IMPORT_THROUGH] if record else None


def _occupy(db, rows=5):
    """A nonempty destination: its own rows take the low message ids."""
    db.create_session("resident", source="cli")
    for index in range(rows):
        db.append_message("resident", "user", f"resident {index}")


def _imported_rows(db):
    return [m["_row_id"] for m in stored_messages(db, SID)][:2]


@pytest.mark.parametrize("continued", [False, True], ids=["fresh", "continued-here"])
@pytest.mark.parametrize("occupied", [False, True], ids=["empty-target", "nonempty-target"])
@pytest.mark.parametrize("export", ["session", "all"])
def test_native_backup_restore_keeps_the_import_boundary(stores, continued, occupied, export):
    source, target = stores
    _cli_import(source, continued=continued)
    before = _continuing(source)
    assert before is continued
    if occupied:
        _occupy(target)
    payload = ([source.export_session(SID, include_inactive=True)] if export == "session"
               else [s for s in source.export_all(include_inactive=True) if s["id"] == SID])
    result = target.import_sessions(payload)
    assert result["ok"] and result["imported_ids"] == [SID], result
    assert _continuing(target) is before
    # The boundary names the destination's own row: the last imported row, never the source's number.
    assert _through(target) == _imported_rows(target)[-1]
    if occupied:
        assert _through(target) != _through(source)
    row = target.get_session(SID)
    assert row["source"] == "cli" and not row.get("origin_json")  # still a writable CLI row


def test_a_restored_import_keeps_its_persisted_prompt_and_boundary(stores):
    source, target = stores
    _cli_import(source)
    source.update_system_prompt(SID, "the prompt a first Hermes build stored")
    payload = source.export_session(SID, include_inactive=True)
    assert target.import_sessions([payload])["ok"]
    assert target.get_session(SID)["system_prompt"] == "the prompt a first Hermes build stored"
    assert _through(target) == _imported_rows(target)[-1]


@pytest.mark.parametrize("continued", [False, True], ids=["fresh", "continued-here"])
def test_the_cross_store_move_remaps_the_import_boundary(stores, continued):
    source, target = stores
    _cli_import(source, continued=continued)
    before = _continuing(source)
    _occupy(target)
    payload = source.export_session_for_move(SID)
    assert target.import_moved_session(payload, profile_name="moved") == "imported"
    assert _continuing(target) is before
    assert _through(target) == _imported_rows(target)[-1] != _through(source)
    assert source.delete_moved_session(SID)
    assert source.get_meta(FOREIGN_IMPORT_META_PREFIX + SID) is None  # no record left for a session not here


def test_an_ordinary_session_exports_no_import_record(stores):
    source, target = stores
    source.create_session("plain", source="cli")
    source.append_message("plain", "user", "hello")
    payload = source.export_session("plain", include_inactive=True)
    assert "foreign_import" not in payload
    assert target.import_sessions([payload])["ok"]
    assert target.get_meta(FOREIGN_IMPORT_META_PREFIX + "plain") is None
