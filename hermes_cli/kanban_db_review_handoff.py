"""Review-handoff provenance: who reviews a re-review, and which earlier review conversation a
reviewer continues when the handoff asks for it (``metadata.reviewer_context == "continue"``).

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via ``_kb``.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

import hermes_cli.kanban_db as _kb


def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _kb._latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _kb._json_dict(_kb._row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False


def review_continuity_error(conn: sqlite3.Connection, task_id: str, metadata: Any) -> Optional[str]:
    """Why a review handoff's continuity keys are invalid, or None. Other metadata is not judged."""
    if not isinstance(metadata, dict) or not {"reviewer_context", "review_continues_task"} & metadata.keys():
        return None
    if metadata.get("reviewer_context") != "continue":
        return 'metadata.reviewer_context must be "continue" (review_continues_task requires it)'
    if "review_continues_task" not in metadata:
        return None
    other = metadata["review_continues_task"]
    if not isinstance(other, str) or not other.strip():
        return "metadata.review_continues_task must be a task id"
    if other == task_id:
        return "metadata.review_continues_task must name a different task"
    if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (other,)).fetchone() is None:
        return f"metadata.review_continues_task {other!r} is not a task on this board"
    return None


def _latest_admitted_review(conn: sqlite3.Connection, task_id: str, before: int) -> Optional[tuple[sqlite3.Row, dict]]:
    """``(run row, owner_admitted payload)`` of *task_id*'s newest review-lane run below
    *before* that an owner admitted; a run that never reached a session has nothing to continue."""
    runs = conn.execute("SELECT id, profile, ended_at FROM task_runs WHERE task_id = ? AND id < ? ORDER BY id DESC",
                        (task_id, before)).fetchall()
    for run in runs:
        if _kb._retry_status_for_run(conn, task_id, run["id"]) != "review":
            continue
        marker = _kb._json_dict(_kb._row_get(_kb._latest_event(conn, task_id, "owner_admitted", run["id"]), "payload"))
        if isinstance(marker.get("session_id"), str) and isinstance(marker.get("db"), str):
            return run, marker
    return None


def review_continuity_source(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[dict]:
    """The earlier review session review run *run_id* continues, read-only, frozen into its policy.

    None unless the latest review handoff opted in. The source is this task's newest admitted
    review run (a correction round, a block then resume), else the newest one on
    ``review_continues_task`` (a spec review continuing into its implementation's review)."""
    handoff = conn.execute("SELECT metadata FROM task_runs WHERE task_id = ? AND outcome = 'review_requested' "
                           "AND id < ? ORDER BY id DESC LIMIT 1", (task_id, run_id)).fetchone()
    metadata = _kb._json_dict(_kb._row_get(handoff, "metadata"))
    if metadata.get("reviewer_context") != "continue":
        return None
    found, source_task = _latest_admitted_review(conn, task_id, run_id), task_id
    if found is None and isinstance(metadata.get("review_continues_task"), str):
        source_task = metadata["review_continues_task"]
        found = _latest_admitted_review(conn, source_task, run_id)
    if found is None:
        return {"unavailable": "no_predecessor"}
    source_run, marker = found
    current = conn.execute("SELECT profile FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    if current is None or current["profile"] != source_run["profile"]:
        return {"unavailable": "changed_profile"}
    # A cross-card source may still be running; the first-turn note must not call it closed.
    return {"task_id": source_task, "run_id": source_run["id"], "session_id": marker["session_id"],
            "db": marker["db"], "run_ended": source_run["ended_at"] is not None}

