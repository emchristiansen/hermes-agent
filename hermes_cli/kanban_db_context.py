"""Worker-context rendering for ``build_worker_context``: the capped markdown sections a spawned
worker reads about its task (header, attachments, prior attempts, parent handoffs, role
history, comments).

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via
``_kb`` so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Optional

import hermes_cli.kanban_db as _kb

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Attachment, Comment, Task

# build_worker_context() caps, sized for a ~100k-char prompt with headroom.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_FOLDED_TURN_REPORTS = 2      # earliest and latest N turn reports shown in one folded wall run
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # per comment
# Event marking a comment that ``session_kanban.keep_on_board`` kept as a correction to a running
# attempt: the next attempt's context is where it is delivered, so it is shown whole (no caps).
KEPT_CORRECTION_EVENT = "correction_kept"


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """``just now`` / ``18h ago`` / ``3d ago``; "" for a missing/invalid ts. An LLM
    reads a bare absolute timestamp as current fact — the relative age is what
    prompts a worker to re-verify stale sibling work."""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 60:  # includes negative = clock skew across machines; never claim "in the future"
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _ctx_cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
    """Truncate to ``limit`` chars with a visible ellipsis."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"


def _ctx_stamp(ts: int, now: int) -> str:
    """``YYYY-MM-DD HH:MM`` plus a relative age when one is available."""
    disp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    age = _relative_age(ts, now)
    return f"{disp}, {age}" if age else disp


def _ctx_metadata_line(metadata: Any) -> Optional[str]:
    if not metadata:
        return None
    try:
        return f"_metadata_: `{_ctx_cap(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}`"
    except Exception:
        return None


def _ctx_tail(items: list, cap: int, noun: str) -> tuple[list, Optional[str]]:
    """Keep the newest ``cap`` items; describe the omitted head, if any."""
    omitted = max(0, len(items) - cap)
    if not omitted:
        return items, None
    return items[-cap:], (
        f"_({omitted} earlier {noun}{'s' if omitted != 1 else ''} "
        f"omitted; showing most recent {cap})_"
    )


def _ctx_header(lines: list[str], task: Task) -> None:
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _kb._worker_terminal_timeout_env(
            task.max_runtime_seconds, os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")
    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_ctx_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")


def _ctx_attachments(lines: list[str], attachments: list[Attachment]) -> None:
    """Absolute on-disk paths so the worker's file tools read them directly
    (remote terminal backends need the attachments dir mounted)."""
    if not attachments:
        return
    lines.append("## Attachments")
    lines.append(
        "Files attached to this task. Read them with the file/terminal "
        "tools at the absolute paths below:"
    )
    for att in attachments:
        size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
        size_str = f", {size_kb} KB" if size_kb else ""
        ctype = f", {att.content_type}" if att.content_type else ""
        lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
    lines.append("")


def _ctx_prior_attempts(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Closed runs on this task (the active run is this worker), newest
    ``_CTX_MAX_PRIOR_ATTEMPTS`` entries in full, older ones as a one-line marker.

    A run of consecutive rate-limited attempts with no summary is ONE entry: a quota wall can requeue
    a card for hours, and one row per walled attempt would push the last attempt that did work (its
    summary names what it was waiting for) out of the window the retry reads."""
    all_prior = [r for r in _kb.list_runs(conn, task_id) if r.ended_at is not None]
    entries = _ctx_attempt_entries(all_prior)
    shown = entries[-_CTX_MAX_PRIOR_ATTEMPTS:]
    if not shown:
        return
    lines.append("## Prior attempts on this task")
    omitted = shown[0][0] - 1
    if omitted:
        lines.append(f"_({omitted} earlier attempt{'s' if omitted != 1 else ''} omitted; "
                     f"showing the most recent {len(all_prior) - omitted})_")
    for first, runs in shown:
        if len(runs) == 1:
            _ctx_attempt(lines, first, runs[0], now)
        else:
            _ctx_wall_run(lines, first, runs, now)
        lines.append("")


def _ctx_attempt_entries(runs: list) -> list[tuple[int, list]]:
    """``(first attempt number, runs)`` entries; consecutive summary-less walls share one."""
    entries: list[tuple[int, list]] = []
    for number, run in enumerate(runs, start=1):
        if _ctx_is_wall(run) and entries and _ctx_is_wall(entries[-1][1][-1]):
            entries[-1][1].append(run)
        else:
            entries.append((number, [run]))
    return entries


# What the crash sweep books on a rate-limited run; any other key is something the attempt or an
# operator recorded, so that run keeps its own entry. ``attempt_note`` is the sweep's own fold of the
# owner's ledger verdict (``kanban_owner_recovery.apply_attempt_note``), written on every booked run
# whose owner settled the admission; the run's error carries the same clause.
_WALL_BOOKING_KEYS = frozenset({"pid", "claimer", "exit_code", "retry_status", "provider_wall", "attempt_note"})


def _ctx_is_wall(run) -> bool:
    """A rate-limited attempt that left nothing but the sweep's own booking."""
    metadata = run.metadata if isinstance(run.metadata, dict) else ({} if run.metadata is None else None)
    return ((run.outcome or run.status) == "rate_limited" and not (run.summary or "").strip()
            and metadata is not None and set(metadata) <= _WALL_BOOKING_KEYS)


def _ctx_attempt(lines: list[str], number: int, run, now: int) -> None:
    profile = run.profile or "(unknown)"
    outcome = run.outcome or run.status
    lines.append(f"### Attempt {number} — {outcome} ({profile}, {_ctx_stamp(run.started_at, now)})")
    if run.summary and run.summary.strip():
        lines.append(_ctx_cap(run.summary))
    if run.error and run.error.strip():
        lines.append(f"_error_: {_ctx_cap(run.error)}")
    meta_line = _ctx_metadata_line(run.metadata)
    if meta_line:
        lines.append(meta_line)


def _ctx_wall_run(lines: list[str], first: int, runs: list, now: int) -> None:
    """Consecutive walled attempts: their span, then the newest one's own error and evidence."""
    newest = runs[-1]
    profiles = ", ".join(dict.fromkeys(r.profile or "(unknown)" for r in runs))
    lines.append(
        f"### Attempts {first}–{first + len(runs) - 1} — rate_limited ×{len(runs)} ({profiles}, "
        f"{_ctx_stamp(runs[0].started_at, now)} to {_ctx_stamp(newest.ended_at, now)})"
    )
    lines.append("Each requeued on a provider wall without counting a failure.")
    # An attempt whose turn ran before its wall may have acted: keep what it reported. The earliest
    # and latest reports are shown; the rest stay in their runs, named here so they can be read.
    turns = [(number, run, wall) for number, run in enumerate(runs[:-1], start=first)
             if (wall := _ctx_turn_wall(run)) is not None]
    edge = _CTX_FOLDED_TURN_REPORTS
    hidden = turns[edge:-edge]
    for number, _run, wall in turns[:edge]:
        lines.append(f"- attempt {number} ran a turn: {_ctx_cap(str(wall.get('evidence') or ''), 300)}")
    if hidden:
        lines.append(f"- {len(hidden)} more attempt{'s' if len(hidden) != 1 else ''} ran a turn "
                     f"(attempts {hidden[0][0]}–{hidden[-1][0]}, runs #{hidden[0][1].id}–#{hidden[-1][1].id}); "
                     f"each report is its run's `metadata.provider_wall`: `hermes kanban runs {newest.task_id} --json`")
    for number, _run, wall in turns[max(edge, len(turns) - edge):]:
        lines.append(f"- attempt {number} ran a turn: {_ctx_cap(str(wall.get('evidence') or ''), 300)}")
    lines.append("The newest:")
    if newest.error and newest.error.strip():
        lines.append(f"_error_: {_ctx_cap(newest.error)}")
    meta_line = _ctx_metadata_line(newest.metadata)
    if meta_line:
        lines.append(meta_line)


def _ctx_turn_wall(run) -> Optional[dict]:
    """The provider wall a run's turn ended on, when its turn ran before the wall."""
    wall = (run.metadata or {}).get("provider_wall") if isinstance(run.metadata, dict) else None
    return wall if isinstance(wall, dict) and wall.get("phase") == "turn" else None


def _ctx_parent_results(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Done-parent handoffs: newest ``completed`` run's summary+metadata,
    falling back to ``task.result`` for pre-runs-table data. Stamped with a
    relative age so the worker re-verifies stale upstream results. A parent whose newest completion
    is a delivered-assessment settlement hands off that settled result (or its corrected handoff
    summary) and its provenance instead."""
    from hermes_cli.kanban_db_settle import settled_handoff
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
    ).fetchall()
    wrote_header = False
    for pid in (r["parent_id"] for r in parent_rows):
        pt = _kb.get_task(conn, pid)
        if not pt or pt.status != "done":
            continue
        settled = settled_handoff(conn, pid)
        runs = [] if settled else [r for r in _kb.list_runs(conn, pid) if r.outcome == "completed"]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        run = runs[0] if runs else None
        if not wrote_header:
            lines.append("## Parent task results")
            lines.append(
                "_Handoffs from upstream tasks, captured when each parent "
                "completed (see age below). These are point-in-time "
                "snapshots, not live state — if a result drives your "
                "current work and it's not recent, re-verify against the "
                "source before acting on it as current._"
            )
            wrote_header = True
        done_ts = run.ended_at if run is not None and run.ended_at else (
            settled[2] if settled else pt.completed_at or None)
        age = _relative_age(done_ts, now)
        lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))
        # A settlement correction's handoff summary stands where an ordinary edit's run summary does (SI-10).
        handoff = run.summary if run is not None else settled[1].get("summary") if settled else None
        if handoff and handoff.strip():
            lines.append(_ctx_cap(handoff))
        elif pt.result:
            lines.append(_ctx_cap(pt.result))
        else:
            lines.append("(no result recorded)")
        if settled:
            lines.append(f"_settled by {settled[1].get('actor')} from run {settled[1].get('expected_terminal_run')} "
                         f"as a delivered {settled[1].get('verdict')} assessment_")
        meta_line = _ctx_metadata_line(run.metadata if run is not None else settled[1].get("metadata") if settled else None)
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_role_history(lines: list[str], conn: sqlite3.Connection, task: Task, now: int) -> None:
    """The assignee's 5 most recent completed runs on OTHER tasks — implicit
    role continuity without wiring anything into SOUL.md / MEMORY.md."""
    if not task.assignee:
        return
    role_rows = conn.execute(
        "SELECT t.id, t.title, r.summary, r.ended_at "
        "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
        "WHERE r.profile = ? AND r.task_id != ? "
        "  AND r.outcome = 'completed' "
        "ORDER BY r.ended_at DESC LIMIT 5", (task.assignee, task.id),
    ).fetchall()
    if not role_rows:
        return
    lines.append(f"## Recent work by @{task.assignee}")
    for row in role_rows:
        first = _kb._first_line(row["summary"], 200) or "(no summary)"
        lines.append(
            f"- {row['id']} — {row['title']} ({_ctx_stamp(_kb._to_epoch(row['ended_at']) or 0, now)}): {first}"
        )
    lines.append("")


def kept_correction_ids(conn: Any, task_id: str) -> set[int]:
    """Ids of the comments kept as corrections to a running attempt (``KEPT_CORRECTION_EVENT``)."""
    ids = set()
    for (payload,) in conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind=?",
                                   (task_id, KEPT_CORRECTION_EVENT)):
        try:
            comment_id = json.loads(payload or "{}").get("comment_id")
        except (ValueError, AttributeError):
            continue
        if type(comment_id) is int:
            ids.add(comment_id)
    return ids


def _ctx_comments(lines: list[str], comments: list[Comment], now: int, whole: set[int] = frozenset()) -> None:
    """Newest ``_CTX_MAX_COMMENTS`` comments, plus every kept correction (``whole``: comment ids)
    wherever it sits and uncapped: this is how its text reaches the task's next attempt. The
    explicit "comment from worker" framing stops an operator-controlled HERMES_PROFILE like
    "hermes-system" being read as a system directive above an
    attacker-influenceable body (defense-in-depth)."""
    tail, omitted_note = _ctx_tail([c for c in comments if c.id not in whole], _CTX_MAX_COMMENTS, "comment")
    shown = sorted([*tail, *(c for c in comments if c.id in whole)], key=lambda c: c.id)
    if not shown:
        return
    lines.append("## Comment thread")
    if omitted_note:
        lines.append(omitted_note)
    for c in shown:
        # Render author with explicit "comment from worker" framing so operator-controlled HERMES_PROFILE
        # values like "hermes-system" or "operator" can't be misread by the next worker as a system
        # directive above the (attacker-influenceable) comment body. Defense-in-depth — the LLM-controlled
        # author-forgery surface was already closed in #22435. See #22452.
        safe_author = (c.author or "").replace("`", "")
        lines.append(f"comment from worker `{safe_author}` at {_ctx_stamp(c.created_at, now)}:")
        lines.append((c.body or "").strip() if c.id in whole else _ctx_cap(c.body, _CTX_MAX_COMMENT_BYTES))
        lines.append("")
