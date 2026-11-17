"""The dashboard/runtime restore keeps a CLI foreign import's boundary (review22815 AGENT-02).

The documented restore endpoint (``hermes_cli/web_routers/sessions.py``) imports an exported session through
``mutate_runtime_session(operation='import')``, not ``SessionDB.import_sessions``. That runtime import must
keep the exported ``foreign_import`` record with its boundary remapped to the destination's own rows, inside
the receipt's own transaction: a fresh import keeps its first-Hermes-turn decision, an import continued
here stays continued, an identical-request retry returns the same receipt and writes nothing new, and a
failed import leaves neither the rows nor the record. Real SessionDB stores, real runtime mutation, export
and home-guidance readers; the original import is the real CLI import mutation (``imported_from``).
"""
import json

import pytest

from agent.home_guidance import continues_without_prompt, foreign_import_view, stored_messages
from hermes_state import SessionDB
from hermes_state_portability import FOREIGN_IMPORT_META_PREFIX, FOREIGN_IMPORT_THROUGH
from hermes_state_runtime import begin_runtime_epoch, mutate_runtime_session

SID = "runtime-cli-import"
ORIGIN = {"tool": "claude", "path": "transcript.json", "foreign_session_id": "foreign-1"}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    source, target = SessionDB(tmp_path / "source.db"), SessionDB(tmp_path / "target.db")
    yield source, target
    source.close()
    target.close()


def _continuing(db):
    return continues_without_prompt(foreign_import_view(db, db.get_session(SID)), stored_messages(db, SID))


def _record(db):
    value = db.get_meta(FOREIGN_IMPORT_META_PREFIX + SID)
    return json.loads(value) if value else None


def _import(db, sessions, request_id):
    epoch = begin_runtime_epoch(db, instance_id=f"test-{request_id}")
    return dict(epoch=epoch, principal_id="tester", session_id=SID, request_id=request_id, expected_revision=0,
                operation="import", payload={"sessions": sessions})


def _cli_import(source, *, continued):
    """The real CLI import of a foreign transcript, through the runtime mutation (``imported_from``)."""
    raw = {"id": SID, "source": "cli", "imported_from": ORIGIN, "messages": [
        {"role": "user", "content": "foreign question"}, {"role": "assistant", "content": "foreign reply"}]}
    assert mutate_runtime_session(source, **_import(source, [raw], "cli-import"))["imported_ids"] == [SID]
    if continued:
        source.append_message(SID, "user", "a question asked here")
        source.append_message(SID, "assistant", "a Hermes reply")
    return source.export_session(SID, include_inactive=True)


def _occupy(db, rows=5):
    db.create_session("resident", source="cli")
    for index in range(rows):
        db.append_message("resident", "user", f"resident {index}")


@pytest.mark.parametrize("continued", [False, True], ids=["fresh", "continued-here"])
@pytest.mark.parametrize("occupied", [False, True], ids=["empty-target", "nonempty-target"])
def test_the_runtime_restore_keeps_the_remapped_boundary(stores, continued, occupied):
    source, target = stores
    exported = _cli_import(source, continued=continued)
    before = _continuing(source)
    assert before is continued
    assert "foreign_import" in exported
    if occupied:
        _occupy(target)
    args = _import(target, [exported], "dashboard-restore")
    receipt = mutate_runtime_session(target, **args)
    assert receipt["imported_ids"] == [SID]
    rows = [m["_row_id"] for m in stored_messages(target, SID)]
    record = _record(target)
    assert record is not None and record[FOREIGN_IMPORT_THROUGH] == rows[1]  # the destination's own row
    assert {k: record[k] for k in ORIGIN} == ORIGIN
    assert _continuing(target) is before
    # An identical-request retry returns the same receipt and leaves the record as it was.
    assert mutate_runtime_session(target, **args) == receipt
    assert _record(target) == record
    assert target.get_session(SID)["source"] == "cli" and not target.get_session(SID).get("origin_json")


def test_a_failed_runtime_restore_leaves_neither_rows_nor_record(stores, monkeypatch):
    source, target = stores
    exported = _cli_import(source, continued=False)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic failure after the rows and record were written")

    monkeypatch.setattr(target, "_attach_import_parents", fail)
    with pytest.raises(Exception):
        mutate_runtime_session(target, **_import(target, [exported], "dashboard-restore"))
    assert target.get_session(SID) is None
    assert target.get_meta(FOREIGN_IMPORT_META_PREFIX + SID) is None
    monkeypatch.undo()
    assert mutate_runtime_session(target, **_import(target, [exported], "dashboard-restore-again"))["imported_ids"] == [SID]
    assert _record(target) is not None and _continuing(target) is False


def test_an_ordinary_restored_session_gets_no_record(stores):
    source, target = stores
    source.create_session(SID, source="cli")
    source.append_message(SID, "user", "hello")
    exported = source.export_session(SID, include_inactive=True)
    assert mutate_runtime_session(target, **_import(target, [exported], "plain"))["imported_ids"] == [SID]
    assert target.get_meta(FOREIGN_IMPORT_META_PREFIX + SID) is None
