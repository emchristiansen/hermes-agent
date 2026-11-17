"""A restored gateway turn keeps every row's durable ``timestamp``, as the live list does.

The gateway rebuilds the agent's history from state.db on every message. Live rows carry ``timestamp`` on every
role (``append_message``), and ``timestamp`` is persistence-only (popped from every outgoing copy). The restore
used to keep it on user rows alone, so a context engine that recognises stored rows by their stamp saw every
restored assistant and tool row as new and stored the whole list again on each turn.
"""
from __future__ import annotations

from agent.message_metadata import MESSAGE_UID, PERSISTENCE_ONLY_MESSAGE_FIELDS, without_persistence_fields

_CALL = [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]


def _restore(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.run import _build_gateway_agent_history
    from gateway.session import SessionStore
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    try:
        db.create_session("room", source="local")
        db.append_message("room", "user", "run it", timestamp=100.0)
        db.append_message("room", "assistant", None, tool_calls=_CALL, timestamp=101.0)
        db.append_message("room", "tool", "out", tool_call_id="c1", tool_name="terminal", timestamp=102.0)
        db.append_message("room", "assistant", "done", timestamp=103.0)
        history, _observed = _build_gateway_agent_history(store.load_transcript("room"))
    finally:
        db.close()
        store.close_all_db_handles()
    return history


def test_restored_rows_keep_their_durable_timestamp_on_every_role(tmp_path):
    history = _restore(tmp_path)
    assert [(m["role"], m.get("timestamp")) for m in history] == [
        ("user", 100.0), ("assistant", 101.0), ("tool", 102.0), ("assistant", 103.0)]
    assert all(m.get(MESSAGE_UID) for m in history)  # the uid already rides every role


def test_restored_timestamps_never_reach_the_provider_copy(tmp_path):
    assert "timestamp" in PERSISTENCE_ONLY_MESSAGE_FIELDS
    for message in _restore(tmp_path):
        assert "timestamp" not in without_persistence_fields(message)
