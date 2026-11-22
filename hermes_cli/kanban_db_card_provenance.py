"""Created-card provenance: which claimed ``created_cards`` the completing worker's own admitted
session created, for ``kanban_db._verify_created_cards``.

Split out of ``hermes_cli.kanban_db``; facade-resident names are reached late-bound via ``_kb``.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def created_by_completing_session(
    conn: sqlite3.Connection, completing_task_id: str, assignee: Optional[str], run_id: Optional[int],
    card_ids: list[str],
) -> set[str]:
    """Cards created by the session the owner admitted for the completing task's current run.

    A session holding the swarm board grant stamps ``<profile>/session:<id>`` on the cards it
    creates through any route, including the CLI from a shell the dispatcher deliberately leaves
    unowned (so no ``creator_task_id`` or parent link exists). The owner's ``owner_admitted`` event
    binds exactly one session to one (task, run) attempt, and that session's id derives from the
    attempt, so a card is this worker's only when all of these hold: the completing call is bound
    to a session of the task's assignee; the owner admitted that same session for the task's
    current run; and the card's ``created_by`` and its ``created`` event's stamped ``actor`` both
    equal that exact session actor. Another session of the same profile, a session admitted for
    an earlier run or another task, or a bare profile string is not this worker's."""
    from agent.swarm_capabilities import bound_session_actor

    actor = bound_session_actor()
    prefix = f"{assignee}/session:" if assignee else ""
    if not actor or not prefix or not actor.startswith(prefix) or run_id is None:
        return set()
    session_id = actor[len(prefix):]
    admitted = [
        _kb._json_or(_kb._lossy_text(row["payload"])) for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'owner_admitted'",
            (completing_task_id, int(run_id)))
    ]
    if not any(isinstance(p, dict) and p.get("session_id") == session_id for p in admitted):
        return set()
    placeholders = ",".join(["?"] * len(card_ids))
    rows = conn.execute(
        "SELECT t.id, e.payload FROM tasks t JOIN task_events e "
        "ON e.task_id = t.id AND e.kind = 'created' "
        f"WHERE t.id IN ({placeholders}) AND t.created_by = ?",
        (*card_ids, actor),
    ).fetchall()
    return {
        r["id"] for r in rows
        if (payload := _kb._json_or(_kb._lossy_text(r["payload"]))) and isinstance(payload, dict)
        and payload.get("actor") == actor
    }


from hermes_cli import kanban_db as _kb  # tail import: the facade imports this module
