"""Created-card provenance: which claimed ``created_cards`` the completing worker's own admitted
session created.

A session holding the swarm board grant (``kanban.worker_board_management``) stamps
``<profile>/session:<id>`` on every card it creates through any route, including the CLI from a shell
the dispatcher deliberately leaves unowned, so no ``creator_task_id`` or parent link exists. The
owner's ``owner_admitted`` event binds one session to one (task, run) attempt
(``gateway/session_kanban.py``), which is what lets such a card count as the worker's own.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from agent.swarm_capabilities import bound_session_actor


def _payload(raw: Any) -> Optional[dict]:
    """A ``task_events.payload`` cell as a dict; anything else (NULL, bad JSON, a non-object) is None."""
    try:
        parsed = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def created_by_completing_session(
    conn: sqlite3.Connection, completing_task_id: str, assignee: Optional[str], run_id: Optional[int],
    card_ids: list[str],
) -> set[str]:
    """The cards in ``card_ids`` created by the session the owner admitted for the completing task's
    current run.

    A card is this worker's only when all of these hold: the completing call is bound to a session of
    the task's assignee under the grant; the owner's admission of the task's current run names that
    same session; and the card's ``created_by`` and its ``created`` event's stamped ``actor`` both
    equal that exact session actor. Another session of the same profile, a session admitted for an
    earlier run or another task, or a bare ``--created-by`` label (its event carries no session actor)
    is not this worker's. Never mutates."""
    actor = bound_session_actor()
    prefix = f"{assignee}/session:" if assignee else ""
    if not card_ids or not actor or not prefix or not actor.startswith(prefix) or run_id is None:
        return set()
    marker = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'owner_admitted' "
        "ORDER BY id DESC LIMIT 1", (completing_task_id, int(run_id)),
    ).fetchone()
    admitted = _payload(marker[0]) if marker else None
    if admitted is None or admitted.get("session_id") != actor[len(prefix):]:
        return set()
    placeholders = ",".join(["?"] * len(card_ids))
    rows = conn.execute(
        "SELECT t.id, e.payload FROM tasks t JOIN task_events e ON e.task_id = t.id AND e.kind = 'created' "
        f"WHERE t.id IN ({placeholders}) AND t.created_by = ?",
        (*card_ids, actor),
    ).fetchall()
    return {row[0] for row in rows if (_payload(row[1]) or {}).get("actor") == actor}
