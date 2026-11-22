"""Edit an existing task's force-loaded skill binding (``tasks.skills``) between attempts.

The binding is read once per attempt: the assignee's owner resolves it from the claimed row when it
admits the claim (``gateway.session_kanban.build_kanban_policy``) and freezes it into that attempt's
session policy. So an edit is refused while any attempt holds the card, and the next claimed attempt
reads the new value from the row.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from typing import Any, Optional

# ``expected`` omitted: no compare-and-set (``None`` is a real binding, so it cannot mean "unset").
UNSET: Any = object()


def set_task_skills(
    conn: sqlite3.Connection, task_id: str, skills: Optional[Iterable[str]], *,
    expected: Any = UNSET, actor: Optional[str] = None, reason: Optional[str] = None,
    board: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Replace or clear the skills the next claimed attempt of ``task_id`` force-loads.

    ``skills``: names replace the binding (validated by ``_normalize_task_skills``, as at create
    time); ``None`` clears it to the create-time default (NULL, no forced skills); ``[]`` stores an
    explicit empty list. Both clears load nothing at dispatch. ``expected`` (``None``, ``[]`` or
    names) is compare-and-set against the current binding. Refused for an archived task and while
    the task holds a live claim or an open run. Orchestrator-only, as unblock. Every other column,
    run, comment and event is left as it was; one ``skills_set`` event records the change.
    Returns ``(ok, refusal)``; an unchanged binding is ``(True, None)`` with nothing written.
    """
    from agent.swarm_capabilities import worker_board_management_enabled
    from hermes_cli import kanban_db as kb

    if os.environ.get("HERMES_KANBAN_TASK") and not worker_board_management_enabled():
        return False, "a skill binding edit is orchestrator-only; workers must hand off their assigned task"
    raw = None if skills is None else list(skills)
    new = kb._normalize_task_skills(raw)
    if raw and not new:
        raise ValueError("no skill name given; clear the binding instead of passing empty names")
    want = UNSET if expected is UNSET else kb._normalize_task_skills(expected)
    with kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return False, f"no such task: {task_id}"
        task = kb.Task.from_row(row)
        refusal = _refusal(conn, task, want)
        if refusal is not None:
            return False, refusal
        if task.skills == new:
            return True, None
        conn.execute(
            "UPDATE tasks SET skills = ? WHERE id = ?",
            (json.dumps(new) if new is not None else None, task_id),
        )
        kb._append_event(conn, task_id, "skills_set", {
            "skills": new, "from": task.skills, "actor": actor, "reason": reason,
            **({} if want is UNSET else {"expected": want}),
        })
    kb.notify_task_updated(conn, task_id, ("skills",), board=board)
    return True, None


def _refusal(conn: sqlite3.Connection, task, want) -> Optional[str]:
    """Why this binding edit must not land now, or None."""
    if task.status == "archived":
        return f"task {task.id} is archived"
    live = [name for name, value in (
        ("claim_lock", task.claim_lock), ("current_run_id", task.current_run_id),
        ("worker_pid", task.worker_pid),
    ) if value is not None]
    if task.status == "running" or live:
        held = ", ".join(live) or "status running"
        return (f"task {task.id} holds a live claim ({held}); its attempt froze its skills at "
                "admission. Wait for the attempt to end or reclaim it, then edit")
    open_run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
        (task.id,),
    ).fetchone()
    if open_run is not None:
        return f"task {task.id} has an open run {open_run['id']}; wait for it to end, then edit"
    if want is not UNSET and task.skills != want:
        return (f"task {task.id} skills are {json.dumps(task.skills)}, not the expected "
                f"{json.dumps(want)}")
    return None
