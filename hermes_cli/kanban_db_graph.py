"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import sqlite3
import time
import os
from typing import Any, Optional


def automatic_triage_eligible(conn: sqlite3.Connection, task_id: str) -> bool:
    """Loop-breaker triage needs an explicit owner disposition, not another spec.

    Call again inside the applying write transaction: a provider response may
    arrive after a worker has tripped the breaker or an operator has parked it.
    Completion resets the recurrence counter; explicit specify/decompose stays
    separate and can resolve a loop without erasing its history.
    """
    from hermes_cli.kanban_db import BLOCK_RECURRENCE_LIMIT

    return conn.execute(
        "SELECT 1 FROM tasks WHERE id=? AND status='triage' "
        "AND current_run_id IS NULL AND claim_lock IS NULL AND claim_expires IS NULL "
        "AND worker_pid IS NULL AND COALESCE(block_recurrences, 0) < ?",
        (task_id, BLOCK_RECURRENCE_LIMIT),
    ).fetchone() is not None


def _settlement_terminal_run(conn: sqlite3.Connection, task_id: str, expected: int):
    """Positive managed-worker completion, not PID absence or lease expiry.

    A terminal run retains the submitter PID. The interpreter may differ; its
    immutable worker_bound/result pair identifies the completed attempt. This
    attests the managed turn's completion, not settlement of external effects.
    """
    run = conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if (run is None or run["id"] != expected or run["ended_at"] is None
            or run["status"] != "blocked" or run["outcome"] != "blocked"
            or not run["claim_lock"] or run["claim_expires"] is not None):
        return None
    if conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id=? AND ended_at IS NULL", (task_id,),
    ).fetchone():
        return None
    return run if _settlement_worker_finished(conn, task_id, run) else None


def _settlement_worker_finished(conn: sqlite3.Connection, task_id: str, run: sqlite3.Row) -> bool:
    """The bound interpreter's result must follow this run's loop transition."""
    from hermes_cli.kanban_db import _json_dict
    from hermes_cli.kanban_db_dispatch import UNVERIFIED_WORKER_FINGERPRINT

    events = {}
    for kind in ("worker_bound", "block_loop_detected", "worker_result"):
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id=? AND run_id=? AND kind=? "
            "ORDER BY id DESC LIMIT 1", (task_id, run["id"], kind),
        ).fetchone()
        if row is None:
            return False
        events[kind] = row
    if not (events["worker_bound"]["id"] < events["block_loop_detected"]["id"]
            < events["worker_result"]["id"]):
        return False
    bound = _json_dict(events["worker_bound"]["payload"])
    result = _json_dict(events["worker_result"]["payload"])
    if not isinstance(bound.get("started_at"), str) or bound["started_at"] == UNVERIFIED_WORKER_FINGERPRINT:
        return False
    if (type(bound.get("pid")) is not int or bound["pid"] <= 0 or not bound.get("started_at")
            or bound.get("claim_lock") != run["claim_lock"]
            or result.get("claim_lock") != run["claim_lock"]
            or result.get("pid") != bound["pid"] or type(result.get("exit_code")) is not int):
        return False
    return True


def settle_triage_task(
    conn: sqlite3.Connection, task_id: str, *, expected_terminal_run_id: int,
    kind: str, owner: str, required_input: str, event: str, author: str,
) -> tuple[bool, str]:
    """Orchestrator settlement of an exact finished loop into an external wait.

    No run is closed or synthesized, no worker is signalled, and no workspace
    is touched. The existing guarded unblock is used only when the named owner
    supplies the substantive event; ordinary dispatch cannot release this wait.
    """
    from hermes_cli.kanban_db import _append_event, _json_dict, _latest_event, write_txn

    if os.environ.get("HERMES_KANBAN_TASK"):
        return False, "triage settlement is orchestrator-only"
    if type(expected_terminal_run_id) is not int or expected_terminal_run_id <= 0:
        raise ValueError("expected terminal run must be a positive run id")
    if kind not in {"capability", "needs_input"}:
        raise ValueError("external wait kind must be capability or needs_input")
    if any(not isinstance(value, str) or not value.strip()
           for value in (owner, required_input, event, author)):
        raise ValueError("external owner, required input, event and author must be nonempty")
    with write_txn(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None or task["status"] != "triage":
            return False, "task is not in triage"
        if any(task[field] is not None for field in
               ("current_run_id", "claim_lock", "claim_expires", "worker_pid")):
            return False, "task still has a run or claim; execution is not settled"
        run = _settlement_terminal_run(conn, task_id, expected_terminal_run_id)
        loop = _latest_event(conn, task_id, "block_loop_detected", expected_terminal_run_id)
        if run is None or loop is None:
            return False, "exact loop run lacks a matching terminal worker result"
        source_status = _json_dict(loop["payload"]).get("source_status")
        conn.execute("UPDATE tasks SET status='blocked', block_kind=? WHERE id=?", (kind, task_id))
        _append_event(conn, task_id, "blocked", {
            "kind": kind, "settled_from": "triage", "owner": owner,
            "required_input": required_input, "event": event, "author": author,
            "source_status": source_status, "expected_terminal_run_id": expected_terminal_run_id,
        }, run_id=expected_terminal_run_id)
    return True, "waiting for the named external owner event"


def inherit_creator_origin(
    conn: sqlite3.Connection, task_id: str, creator_task_id: Optional[str], *,
    created_at: int,
) -> None:
    """Copy durable origin inside creation's transaction, never adding dependencies."""
    if not creator_task_id:
        return
    from hermes_cli.kanban_db import _inherit_notify_subs

    conn.execute(
        "UPDATE tasks SET session_id = COALESCE(session_id, "
        "(SELECT session_id FROM tasks WHERE id = ?)) WHERE id = ?",
        (creator_task_id, task_id),
    )
    _inherit_notify_subs(conn, task_id, (creator_task_id,), created_at=created_at)


def initial_task_state(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str,
    triage: bool, tenant: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve state and tenant under the creator's write transaction.

    Parent order breaks ties in this soft namespace; explicit tenant wins.
    Validate parents even for parked tasks so links never dangle.
    """
    rows = {}
    if parents:
        rows = {row["id"]: row for row in conn.execute(
            "SELECT id, status, tenant FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        )}
        missing = [pid for pid in parents if pid not in rows]
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
        if tenant is None:
            tenant = next((rows[pid]["tenant"] for pid in parents if rows[pid]["tenant"]), None)
    if initial_status == "blocked":
        return "blocked", tenant
    if triage:
        return "triage", tenant
    if any(row["status"] != "done" for row in rows.values()):
        return "todo", tenant
    return "ready", tenant


def _validate_children_graph(children: list) -> None:
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True, automatic: bool = False,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    Returns child ids in input order, or None when the root is missing / not
    in triage, or has already decomposed. Atomic: malformed entries abort fan-out.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        write_txn, recompute_ready,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        if automatic and not automatic_triage_eligible(conn, task_id):
            return None
        # Dependency links alone do not imply lineage. The completion event is
        # committed with the graph, and survives re-triage or unlinking.
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        child_ids = [
            _insert_decomposed_child(conn, task_id, root_row, child, author, now)
            for child in children
        ]
        # Sibling edges within the decomposed graph.
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Root waits for the whole graph: link it under EVERY child (simpler
        # than computing leaves; cycle-free since the root is only ever a child).
        for cid in child_ids:
            _link(conn, cid, task_id)
        # Flip the root triage -> todo, assignee -> orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed", {"child_ids": child_ids, "root_assignee": root_assignee},
        )
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int,
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.
    """
    from hermes_cli.kanban_db import (
        _new_task_id, _canonical_assignee, _append_event,
    )

    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    new_id = _new_task_id()
    body = child.get("body")
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"),
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id
