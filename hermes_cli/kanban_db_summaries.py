"""Current handoff summaries for task views (CLI ``show``, dashboard drawer, child results and cards).

Ordinarily the newest non-empty run summary. A delivered-assessment settlement and its result
corrections are runless events (``kanban_db_settle``), so for a task whose newest completion is one,
its current handoff supersedes run summaries no newer than it (review7100 SC-02)."""
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Current handoff summary, or None. Workers hand off via ``summary`` and
    leave ``tasks.result`` NULL, so views need this or a done task looks empty.
    See :func:`latest_summaries` for a settled delivered assessment."""
    return latest_summaries(conn, [task_id]).get(task_id)


def latest_summaries(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, str]:
    """``{task_id: current handoff summary}``; tasks without one are omitted.

    Ordinarily that is the newest non-empty run summary (one window-function query,
    SQLite >= 3.25). A delivered-assessment settlement and its result corrections are
    runless events, so for a task whose newest completion is one, its current handoff
    (:func:`_settled_current_summaries`) supersedes every run summary no newer than the
    settlement; the ended run's scheduling reason or an older PASS is never offered as current."""
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary, at FROM (
            SELECT task_id, summary, COALESCE(ended_at, started_at) AS at,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    summaries = {r["task_id"]: r["summary"] for r in rows}
    run_at = {r["task_id"]: r["at"] for r in rows}
    for task_id, (current, settled_at) in _settled_current_summaries(conn, ids).items():
        if task_id in run_at and (run_at[task_id] or 0) > settled_at:
            continue  # a run newer than the settlement speaks for itself
        if current:
            summaries[task_id] = current
        else:
            summaries.pop(task_id, None)
    return summaries


def _settled_current_summaries(
    conn: sqlite3.Connection, task_ids: list[str],
) -> dict[str, tuple[Optional[str], int]]:
    """``{task_id: (current handoff summary or None, settled_at)}`` for each task whose newest
    completion is a delivered-assessment settlement (``kanban_db_settle.settled_handoff``): the
    newest correction's summary, else the settled result, as dependents read it
    (``kanban_db_context._ctx_parent_results``). Never a run's summary, and nothing is written."""
    placeholders = ",".join("?" for _ in task_ids)
    # Only tasks with a settlement event at all need the per-task reader.
    candidates = [r["task_id"] for r in conn.execute(
        f"SELECT DISTINCT task_id FROM task_events WHERE kind = 'completed' "
        f"AND task_id IN ({placeholders}) AND payload LIKE '%delivered_assessment%'",
        task_ids,
    )]
    if not candidates:
        return {}
    from hermes_cli.kanban_db_settle import settled_handoff
    current: dict[str, tuple[Optional[str], int]] = {}
    for task_id in candidates:
        settled = settled_handoff(conn, task_id)
        if settled is None:
            continue
        result, settlement, settled_at = settled
        summary = settlement.get("summary")
        handoff = summary if isinstance(summary, str) and summary.strip() else result
        current[task_id] = (handoff or None, int(settled_at or 0))
    return current
