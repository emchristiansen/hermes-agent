"""Ours' message-identity read and in-place conversation end for ``SessionMessagesMixin``, moved out of
``hermes_state_messages.py`` (code-health FILE_LINES after the v0.23.1 merge). Both run on SessionDB's own
connection helpers (``_read_one``, ``_execute_write``, ``_delete_unreferenced_system_prompts``)."""

from __future__ import annotations

from typing import Optional


class SessionMessageConversationMixin:
    """Message-identity read and in-place conversation end (see the module docstring)."""

    def get_message_identity(self, session_id: str, row_id: int) -> Optional[tuple[str, Optional[str]]]:
        """``(role, message_uid)`` of the active message at *row_id* in *session_id*, or ``None``."""
        if not session_id:
            return None
        row = self._read_one(
            "SELECT role, message_uid FROM messages WHERE id = ? AND session_id = ? AND active = 1",
            (int(row_id), session_id))
        return (row[0], row[1]) if row else None

    def end_conversation_in_place(self, session_id: str) -> None:
        """A reset in place (ACP ``/reset``): the session keeps its row, usage counters and messages on disk, but its
        conversation ends. In one transaction, drop the stored prompt that recorded the ended conversation and take
        its rows out of the session's conversation — the live ones and those compaction summarized away
        (``active = 1 OR compacted = 1``) get the marking rewind gives rows the user took back (``active = 0,
        compacted = 0``): kept on disk for audit (``include_inactive=True`` reads and searches find them; default
        search does not), never replayed or shown as the conversation's history again."""
        self._execute_write(lambda conn: self._end_conversation_in_transaction(conn, session_id))

    def _end_conversation_in_transaction(self, conn, session_id: str) -> None:
        """``end_conversation_in_place`` on the caller's own write transaction (the managed worker's
        receipt transaction shares it, so a worker's reset commits or rolls back with its receipt)."""
        from hermes_state_messages import _DISPLAY_ACTIVE_CLAUSE, _RESET_COUNTERS_SQL

        conn.execute("UPDATE sessions SET system_prompt_hash = NULL, system_prompt = NULL WHERE id = ?",
                     (session_id,))
        self._delete_unreferenced_system_prompts(conn)
        conn.execute(f"UPDATE messages SET active = 0, compacted = 0 WHERE session_id = ?{_DISPLAY_ACTIVE_CLAUSE}",
                     (session_id,))
        conn.execute(_RESET_COUNTERS_SQL, (session_id,))
