"""Record-only settlement of an accepted, stopped card, and the workspace retention it records.

t_11fe6077, 2026-11-14: a card's outcome was accepted on the board, but its run had crashed and a
separate housekeeping admission for its scratch workspace was still of unknown outcome. Every route
to Done ran ``_cleanup_workspace`` after the commit: it released the directory's LSP clients,
deleted it and killed a dead assignee tmux session. ``fire_lifecycle_hook=False`` and the dashboard
changed nothing, and ``kanban edit`` cannot change a workspace. Recording the accepted outcome
would have destroyed the workspace whose custody was unknown.

:func:`settle_record_only` records that outcome and performs no cleanup. In one write transaction
it moves a ``blocked`` or ``ready`` card to ``done``, fenced on its exact latest ended run. It also
writes a zero-duration bookkeeping run and a ``task_workspace_retentions`` row for the card's
workspace. The bookkeeping run has no profile and says ``performed_work: false``, so the original
executor's runs are not reattributed and no review is implied. It releases no LSP client, kills no
tmux session, removes no directory and fires no lifecycle hook. The existing guards stay in force:
the caller must be an orchestrator, a PR completion contract must pass its own acceptance, every
parent must be done, the card must hold no claim, run or live bound worker, and the named run must
be the latest ended attempt with no unsettled accounting hold.

A retention row has no expiry and no release. Every removal route asks
``kanban_db_workspace._workspace_in_use_by_other`` before deleting: the card's own completion or
archive cleanup, the parent sweep, the gateway worker's end-of-turn retry and ``kanban gc``. That
check returns ``"retained"`` for a path at, inside or containing a retained workspace, on this
board or any sibling board. Own-card cleanup also leaves the tmux session and LSP clients alone.
Removing a board, by delete or by archive (which moves its workspaces), is refused while its own
database records a retention or any board's retention lies inside its directory. Hard-deleting the task keeps the row, which still protects the path.
Anything that cannot be read counts as retained.

A check followed by a removal could still lose a race with a settlement that commits between the
two. Every removal, board removal included, therefore runs its in-use check and its removal
inside :func:`removal_fence`, a lock file shared by every board under the kanban home, and the
settlement commits inside the same fence. Either the removal sees the retention, or the settlement sees the directory gone and
refuses to claim a retention it cannot provide. The fence is bounded: when a removal cannot get it
in time, it keeps the directory; when a settlement cannot, it refuses and nothing changes.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Optional

RETAINED = "retained"
# Stopped, accepted cards only. ``review`` keeps its own approval, ``scheduled`` keeps
# ``settle-delivered``, and ``running``/``todo`` are not stopped accepted work.
SETTLEABLE_STATUSES = ("blocked", "ready")
SETTLEMENT_KEY = "record_only_settlement"
# Completion metadata that ordinary completion turns into file effects or delivery: ``artifacts``
# is staged and attached (or listed in ``unattached_artifacts``) and promoted to the ``completed``
# event, where the gateway notifier uploads it. This route performs none of that, so it refuses them.
DELIVERY_METADATA_KEYS = frozenset({"artifacts", "unattached_artifacts", "_staged_artifacts"})

# The retention table, created on every board by the connect pass right after the facade's
# SCHEMA_SQL. A row has no expiry and no release: every workspace removal route treats a path at,
# inside or containing ``workspace_path`` (or its resolved ``path_key``) as in use. ``settlement``
# is the JSON audit record: actor, the exact ended run, acceptance evidence and the custody
# retained. Not deleted with its task, so a hard-deleted card's directory stays protected.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS task_workspace_retentions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        TEXT NOT NULL,
    workspace_kind TEXT,
    workspace_path TEXT NOT NULL,
    path_key       TEXT NOT NULL,
    settlement     TEXT NOT NULL,
    created_by     TEXT NOT NULL,
    created_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retentions_task ON task_workspace_retentions(task_id);
"""

FENCE_TIMEOUT_SECONDS = 10.0
_FENCE_POLL_SECONDS = 0.05

_TABLE_SQL = "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_workspace_retentions'"


def _has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(_TABLE_SQL).fetchone() is not None


@contextlib.contextmanager
def removal_fence(timeout: Optional[float] = None) -> Iterator[bool]:
    """Hold the kanban home's workspace-removal lock; yields whether it was acquired within *timeout*
    (default :data:`FENCE_TIMEOUT_SECONDS`).

    The caller must not remove anything, or settle anything, when this yields False. Each call
    opens its own handle, so two holders exclude each other within one process as well as across
    processes; never take it while already holding it."""
    from hermes_cli.kanban_db_connect import _try_lock_nb, _unlock
    lock_dir = _kb.kanban_home() / "kanban"
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / "workspace-removal.lock").open("a+b")
    acquired = False
    try:
        timeout = FENCE_TIMEOUT_SECONDS if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            try:
                acquired = _try_lock_nb(handle)
            except OSError:
                acquired = False
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(_FENCE_POLL_SECONDS)
        if not acquired:
            _kb._log.warning("workspace removal fence %s not acquired within %.0fs", handle.name, timeout)
        yield acquired
    finally:
        try:
            if acquired:
                _unlock(handle)
        finally:
            handle.close()


def _key(path: str) -> str:
    from hermes_cli.kanban_db_workspace import _path_key
    return _path_key(Path(path).expanduser().resolve(strict=False))


def _overlaps(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    return pa == pb or pa.is_relative_to(pb) or pb.is_relative_to(pa)


def retention_covering(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """The task whose retained workspace is at, inside or contains the resolved path *key*, else None.

    A board whose schema predates retention holds none. The path is checked both as resolved when
    it was retained and as it resolves now, so changing a symlink does not release it. A read
    error propagates (``sqlite3.Error``/``OSError``); callers treat it as reason to keep."""
    if not key or not _has_table(conn):
        return None
    for row in conn.execute("SELECT task_id, workspace_path, path_key FROM task_workspace_retentions"):
        if _overlaps(row[2], key) or _overlaps(_key(row[1]), key):
            return row[0]
    return None


def task_is_retained(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether this board retains *task_id*'s own workspace. A read error propagates."""
    return _has_table(conn) and conn.execute(
        "SELECT 1 FROM task_workspace_retentions WHERE task_id=? LIMIT 1", (task_id,)).fetchone() is not None


def cleanup_board_db_files() -> list[Path]:
    """The board databases every workspace removal route reads, resolved and deduplicated:
    ``<home>/kanban.db`` and ``<home>/kanban/boards/<slug>/kanban.db``. Archived boards
    (``boards/_archived/<slug>-<ts>/``) and a ``HERMES_KANBAN_DB`` pinned elsewhere are not in it,
    so a settlement refuses there. Raises ``OSError`` when the set cannot be listed.

    ``kanban_db_path`` follows ``HERMES_KANBAN_DB`` and would collapse every slug onto the pinned
    file, so this uses the on-disk layout."""
    candidates = [_kb.kanban_home() / "kanban.db"]
    root = _kb.boards_root()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                candidates.append(child / "kanban.db")
    found: list[Path] = []
    for path in candidates:
        if path.is_file():
            resolved = path.resolve()
            if resolved not in found:
                found.append(resolved)
    return found


def _settlement_db_refusal(conn: sqlite3.Connection) -> Optional[str]:
    """Why a retention recorded through *conn* would not be read by the removal routes, or None."""
    from hermes_cli.kanban_db_connect import _main_db_file
    current = _main_db_file(conn)
    try:
        readers = cleanup_board_db_files()
    except OSError as exc:
        return f"the board databases that cleanup reads cannot be listed ({exc})"
    if not current or Path(current).resolve() not in readers:
        return (f"its board database {current or '(in memory)'} is not one the workspace cleanup routes "
                "read (an archived board or a database pinned outside the kanban home), so a retention "
                "recorded there would not protect the workspace")
    return None


def _all_board_db_files() -> list[Path]:
    """:func:`cleanup_board_db_files` plus archived boards' databases. Settlement never writes a
    retention to an archived board, so these are read only to fail closed on older state."""
    found = cleanup_board_db_files()
    root = _kb.boards_root()
    if root.is_dir():
        found += [path.resolve() for path in sorted(root.glob("_archived/*/kanban.db"))
                  if path.is_file() and path.resolve() not in found]
    return found


def board_removal_refusal(board_dir: Path) -> Optional[str]:
    """Why removing (deleting or archiving) the board directory *board_dir* would lose a retention,
    or None. Refuses when its own database records any retention, which would go with it, and when
    any board's retention lies at, inside or above the directory. Unreadable means refuse. Call it
    inside :func:`removal_fence` and remove the board in the same fence."""
    try:
        key = _key(str(board_dir))
        own = (board_dir / "kanban.db").resolve(strict=False)
        for db_file in _all_board_db_files():
            conn = sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True, timeout=5.0)
            try:
                if not _has_table(conn):
                    continue
                if db_file.resolve(strict=False) == own:
                    count = int(conn.execute("SELECT COUNT(*) FROM task_workspace_retentions").fetchone()[0])
                    if count:
                        return f"it records {count} retained workspace(s) of record-only settlements"
                task = retention_covering(conn, key)
                if task:
                    return f"it holds the workspace retained for task {task} (board database {db_file})"
            finally:
                conn.close()
    except (OSError, sqlite3.Error) as exc:
        return f"its workspace retentions cannot be read ({exc}); repair the board databases first"
    return None


def retentions_for_task(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    """The card's retention records, oldest first, with each settlement record decoded."""
    if not _has_table(conn):
        return []
    rows = conn.execute(
        "SELECT id, task_id, workspace_kind, workspace_path, path_key, settlement, created_by, created_at "
        "FROM task_workspace_retentions WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
    out = []
    for row in rows:
        record = dict(zip(("id", "task_id", "workspace_kind", "workspace_path", "path_key", "settlement",
                           "created_by", "created_at"), tuple(row)))
        record["settlement"] = _kb._json_dict(record["settlement"])
        out.append(record)
    return out


def _live_binding(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether a worker bound to one of this card's attempts may still be running (fails closed)."""
    from hermes_cli.kanban_db_workspace import _UNFINISHED_BINDING_SQL, _binding_may_be_live
    rows = conn.execute(
        f"SELECT e.payload FROM task_events e WHERE e.task_id=? AND {_UNFINISHED_BINDING_SQL}", (task_id,),
    ).fetchall()
    return any(_binding_may_be_live(row[0]) for row in rows)


def record_only_refusal(conn: sqlite3.Connection, task_id: str, expected_terminal_run: int) -> Optional[str]:
    """Why *task_id* cannot be settled record-only from run *expected_terminal_run*, or None."""
    task = conn.execute(
        "SELECT status, completion_contract, current_run_id, claim_lock, claim_expires, worker_pid "
        "FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if task is None:
        return "no such task"
    refusal = _settlement_db_refusal(conn)
    if refusal is not None:
        return refusal
    if task["status"] not in SETTLEABLE_STATUSES:
        return (f"status is {task['status']}; record-only settlement takes a stopped "
                f"{' or '.join(SETTLEABLE_STATUSES)} card")
    contract = task["completion_contract"]
    if contract and contract != "local-only":
        return (f"its completion contract {contract[:120]!r} requires PR acceptance; "
                "record-only settlement settles only local-only cards")
    if any(task[key] is not None for key in ("current_run_id", "claim_lock", "claim_expires", "worker_pid")):
        return f"the task still holds a run or claim (current run {task['current_run_id']})"
    if conn.execute("SELECT 1 FROM task_runs WHERE task_id=? AND ended_at IS NULL", (task_id,)).fetchone():
        return "an attempt is still open"
    from hermes_cli.kanban_db_retry_policy import terminal_run_refusal
    refusal = terminal_run_refusal(conn, task_id, expected_terminal_run)
    if refusal is not None:
        return refusal
    if _live_binding(conn, task_id):
        return "a worker bound to one of its attempts may still be running"
    if not _kb._parents_satisfied(conn, task_id):
        return "a parent is not done"
    return None


def settle_record_only(conn: sqlite3.Connection, task_id: str, *, expected_terminal_run: int, result: str,
                       accepted_by: str, custody: str, author: str,
                       metadata: Optional[dict] = None) -> tuple[bool, str]:
    """Orchestrator-only ``blocked|ready`` -> ``done`` that records an accepted outcome and retains the
    workspace, with no cleanup effect.

    *accepted_by* names the acceptance evidence and *custody* the resources whose custody is
    retained, both in the caller's words. Returns ``(ok, reason)``. Malformed arguments raise
    ``ValueError``. Nothing is written on refusal."""
    from agent.swarm_capabilities import worker_board_management_enabled
    if os.environ.get("HERMES_KANBAN_TASK") and not worker_board_management_enabled():
        return False, "record-only settlement is orchestrator-only"
    if type(expected_terminal_run) is not int or expected_terminal_run < 1:
        raise ValueError("expected terminal run must be a positive run id")
    for name, value in (("result", result), ("accepted_by", accepted_by), ("custody", custody),
                        ("author", author)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata is not None and SETTLEMENT_KEY in metadata:
        raise ValueError(f"metadata must not supply {SETTLEMENT_KEY!r}; the settlement writes it")
    delivery = sorted(DELIVERY_METADATA_KEYS & set(metadata or {}))
    if delivery:
        raise ValueError(
            f"metadata must not declare {', '.join(delivery)}: a record-only settlement has no file effects, "
            "so it cannot attach or classify artifacts; attach files with `kanban attach` first, or "
            "complete the card normally")
    result = result.strip()
    with removal_fence() as fenced:
        if not fenced:
            return False, "a workspace removal holds the removal fence; retry"
        ok, reason, run_id, path = _settle_fenced(
            conn, task_id, expected_terminal_run, result, accepted_by, custody, author, metadata)
    if not ok:
        return False, reason
    # Bookkeeping only: dependents may now be ready and the failure counter is spent. No
    # _cleanup_workspace, parent sweep or lifecycle hook.
    _kb._clear_failure_counter(conn, task_id)
    _kb.recompute_ready(conn)
    retained = f"; workspace retained at {path}" if path else "; no workspace to retain"
    return True, f"recorded done from run {expected_terminal_run} with bookkeeping run {run_id}{retained}"


def _settle_fenced(conn, task_id, expected_terminal_run, result, accepted_by, custody, author, metadata):
    """The settlement's one write transaction; the caller holds :func:`removal_fence`.
    Returns ``(ok, reason, run_id, workspace_path)``."""
    now = int(time.time())
    with _kb.write_txn(conn):
        refusal = record_only_refusal(conn, task_id, expected_terminal_run)
        if refusal is not None:
            return False, refusal, None, None
        row = conn.execute(
            "SELECT t.status, t.workspace_kind, t.workspace_path, r.profile FROM tasks t "
            "JOIN task_runs r ON r.id=? WHERE t.id=?", (expected_terminal_run, task_id),
        ).fetchone()
        path = row["workspace_path"]
        if path and not Path(path).expanduser().is_dir():
            return False, (f"its workspace {path} does not exist, so it cannot be retained; "
                           "nothing was recorded"), None, None
        settlement = {
            "kind": "record_only", "performed_work": False, "actor": author.strip(),
            "from_status": row["status"], "expected_terminal_run": expected_terminal_run,
            "original_profile": row["profile"], "accepted_by": accepted_by.strip(),
            "custody": custody.strip(), "cleanup": "none",
            "workspace_retained": ({"kind": row["workspace_kind"], "path": path} if path else None),
        }
        cur = conn.execute(
            "UPDATE tasks SET status='done', result=?, completed_at=?, block_kind=NULL, block_recurrences=0 "
            "WHERE id=? AND status IN ('blocked', 'ready') AND current_run_id IS NULL AND claim_lock IS NULL "
            "AND claim_expires IS NULL AND worker_pid IS NULL", (result, now, task_id),
        )
        if cur.rowcount != 1:
            return False, "it changed while being settled; re-read the card", None, None
        if path:
            conn.execute(
                "INSERT INTO task_workspace_retentions (task_id, workspace_kind, workspace_path, path_key, "
                "settlement, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, row["workspace_kind"], path, _key(path), json.dumps(settlement, sort_keys=True),
                 author.strip(), now),
            )
        run_metadata = {**(metadata or {}), SETTLEMENT_KEY: settlement}
        # No profile: nobody executed this run. The original attempt keeps its profile.
        run_id = _kb._synthesize_ended_run(
            conn, task_id, outcome="completed", summary=result, metadata=run_metadata, profile=None)
        _kb._append_event(conn, task_id, "completed", {
            **_kb._completed_event_payload(result, result, [], metadata),
            "settlement": settlement,
        }, run_id=run_id)
    return True, "", run_id, path


# Late-bound origin namespace, as the other split modules use it.
from hermes_cli import kanban_db as _kb
