"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A cron delivery is
a MESSAGE (a chat/notification send): at least once, a duplicate is acceptable and
a lost delivery is not.  A send whose outcome is uncertain (its gateway died
mid-send, or it returned but its outcome could not be recorded) is queued again,
never given up: each retry waits a capped backoff (``retry_after``), and a row
waiting out its backoff never holds back any other delivery.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.deadline import poll_until
from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
# Claimed rows whose send thread never started: nothing was attempted (MR-DISPATCH-07).
_UNSTARTED_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
MAX_TERMINAL_DELIVERIES = 1000
# Backoff before an uncertain send is claimed again: base * 2**(claims-1), capped; never a give-up.
RETRY_BACKOFF_BASE_SECONDS = 30.0
RETRY_BACKOFF_MAX_SECONDS = 3600.0
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata."""
    conn.execute(
        """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND (job_json != '{}' OR content != '')"""
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM deliveries "
            "WHERE status IN ('delivered','failed','unknown','suppressed')"
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at)
               SELECT execution_id, status, finished_at FROM deliveries
               WHERE status IN ('delivered','failed','unknown','suppressed')
               ORDER BY julianday(finished_at), finished_at,
                        julianday(created_at), created_at, execution_id
               LIMIT ?""",
            (excess,),
        )
        conn.execute(
            """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown','suppressed')
                 ORDER BY julianday(finished_at), finished_at,
                          julianday(created_at), created_at, execution_id
                 LIMIT ?
               )""",
            (excess,),
        )


def queue_path(home: Optional[Path] = None) -> Path:
    """The queue file of ``home`` (the active home when None); a test override wins."""
    if DELIVERY_DB is not None:
        return DELIVERY_DB
    root = Path(home) if home is not None else get_hermes_home()
    return root.resolve() / "cron" / "deliveries.db"


def _path() -> Path:
    return queue_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    # SQLite cannot widen a CHECK in place. Preserve all old rows atomically,
    # including claimed sends, while admitting a distinct never-sent disposition.
    for table in ("deliveries", "delivery_tombstones"):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if row and "'suppressed'" not in row[0]:
            conn.execute("SAVEPOINT notification_disposition")
            try:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                conn.execute(row[0].replace("'unknown'", "'unknown','suppressed'"))
                conn.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")
                conn.execute(f"DROP TABLE {table}_old")
                conn.execute("RELEASE notification_disposition")
            except BaseException:
                conn.execute("ROLLBACK TO notification_disposition")
                conn.execute("RELEASE notification_disposition")
                raise

    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             execution_id TEXT PRIMARY KEY,
             job_json TEXT NOT NULL,
             content TEXT NOT NULL,
             for_failure INTEGER NOT NULL DEFAULT 0,
             status TEXT NOT NULL CHECK(status IN
               ('pending','delivering','delivered','failed','unknown','suppressed')),
             owner_process_id TEXT,
             owner_pid INTEGER,
             owner_started_at INTEGER,
             created_at TEXT NOT NULL,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_tombstones (
             execution_id TEXT PRIMARY KEY,
             terminal_status TEXT NOT NULL CHECK(terminal_status IN
               ('delivered','failed','unknown','suppressed')),
             finished_at TEXT
           )"""
    )
    add_column_if_missing(
        conn, "deliveries", "for_failure",
        "for_failure INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(conn, "deliveries", "attempts", "attempts INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing(conn, "deliveries", "retry_after", "retry_after REAL")


def _now_seconds() -> float:
    return time.time()


def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from hermes_cli.sqlite_util import open_db

    path = _path()
    conn = open_db(path, db_label="cron/deliveries.db", synchronous_full=True, initialize=_initialize_schema)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    # Pruning is done explicitly by the paths that create terminal
    # rows (_finish / recover_abandoned / _terminalize_wait_timeout);
    # read-only polls must not pay for a full-table UPDATE + COUNT.
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    with _transaction() as conn:
        # Serialize the tombstone check and insert with retention in other
        # processes, which can move a terminal delivery into the tombstone table.
        conn.execute("BEGIN IMMEDIATE")
        tombstone = conn.execute(
            "SELECT terminal_status, finished_at FROM delivery_tombstones "
            "WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
            }
        conn.execute(
            """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                str(content),
                int(bool(for_failure)),
                _hermes_now().isoformat(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
    return dict(row)


def enqueue_result(job: dict, content: str, *, for_failure: bool = False,
                   failure_incident_id: Optional[str] = None) -> dict:
    """Accept a composed message under the caller's fire fence, never its business work.

    Pin the accepted destinations, including failure-lane expansion; retrying must
    not reinterpret an old message using a later job definition or home target.
    Transport credentials remain the draining owner's responsibility.
    """
    from cron.scheduler_delivery import _resolve_delivery_targets

    execution_id = job["execution_id"]
    queued_job = dict(job, _queued_delivery_targets=_resolve_delivery_targets(job, for_failure=for_failure))
    if failure_incident_id:
        queued_job["_failure_incident_id"] = failure_incident_id
    row = enqueue(execution_id, queued_job, content, for_failure=for_failure)
    job["last_delivery_queued"] = {"canonical": {"status": row["status"], "execution_id": execution_id}}
    return row


def get_status(execution_id: str) -> Optional[dict]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
        if row is not None:
            return dict(row)
        tombstone = conn.execute(
            "SELECT execution_id, terminal_status, finished_at "
            "FROM delivery_tombstones WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "error": None,
    }


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        # created_at carries a DST-varying offset: order by instant, not text.
        # A row waiting out its retry backoff is skipped, never a barrier to later deliveries.
        row = conn.execute(
            "SELECT execution_id FROM deliveries WHERE status='pending' "
            "AND (retry_after IS NULL OR retry_after<=?) "
            "ORDER BY julianday(created_at), created_at, execution_id LIMIT 1",
            (_now_seconds(),),
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?, attempts=attempts+1
               WHERE execution_id=? AND status='pending'""",
            (_PROCESS_ID, pid, started, row["execution_id"]),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)
        ).fetchone()
        _ACTIVE_DELIVERIES.add(row["execution_id"])
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _finish(execution_id: str, *, error: Optional[str], suppressed: bool = False) -> bool:
    status = "failed" if error else "suppressed" if suppressed else "delivered"
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                safe_error,
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def _requeue_uncertain(conn: sqlite3.Connection, execution_id: str) -> bool:
    """Queue an uncertain send again (at least once: a duplicate, never a loss), after its backoff."""
    row = conn.execute("SELECT attempts FROM deliveries WHERE execution_id=? AND status='delivering'",
                       (str(execution_id),)).fetchone()
    if row is None:
        return False
    with _lock:
        unstarted = str(execution_id) in _UNSTARTED_DELIVERIES
        _UNSTARTED_DELIVERIES.discard(str(execution_id))
    if unstarted:  # never attempted: claimable again at once, and the claim's attempt is not counted
        return conn.execute(
            """UPDATE deliveries SET status='pending', owner_process_id=NULL, owner_pid=NULL,
               owner_started_at=NULL, retry_after=NULL, attempts=MAX(0, attempts-1)
               WHERE execution_id=? AND status='delivering'""",
            (str(execution_id),),
        ).rowcount == 1
    return conn.execute(
        """UPDATE deliveries SET status='pending', owner_process_id=NULL, owner_pid=NULL,
           owner_started_at=NULL, retry_after=? WHERE execution_id=? AND status='delivering'""",
        (_now_seconds() + _retry_delay(int(row["attempts"])), str(execution_id)),
    ).rowcount == 1


def _retry_delay(attempts: int) -> float:
    """The capped backoff after ``attempts`` claims. The exponent saturates BEFORE the power, so a row
    retried for as long as it takes never overflows the float conversion (MR-RETRY-14): 2**32 base
    periods already exceed any cap."""
    return min(RETRY_BACKOFF_BASE_SECONDS * 2.0 ** min(max(0, attempts - 1), 32), RETRY_BACKOFF_MAX_SECONDS)


def _retry_failed_send(row: dict, error: str) -> bool:
    """A send that returned an error (negative result, timeout, no acknowledgement) is never terminal: its
    payload stays and it is queued again after its backoff, independently of every other row (at least once;
    a duplicate is accepted, a loss never). The returned error is kept as the row's last error, and the run
    that queued it is told: ``delivery_retrying`` with that error (msg-retry-on-13f-6)."""
    execution_id = row["execution_id"]
    with _transaction() as conn:
        owned = conn.execute(
            """UPDATE deliveries SET error=? WHERE execution_id=? AND status='delivering'
               AND owner_process_id=? AND owner_pid=?""",
            (str(error), execution_id, _PROCESS_ID, os.getpid()),
        ).rowcount == 1
        requeued = owned and _requeue_uncertain(conn, execution_id)
    if requeued:
        logger.warning("Cron delivery %s: send returned an error; queued again (at least once)", execution_id)
        from cron.delivery_outcome import settle_quietly

        settle_quietly(row["job"].get("id"), execution_id)
    return requeued


def recover_abandoned() -> int:
    """Queue again every send whose outcome is uncertain; never fence a message as given up.

    A dead owner's send may or may not have reached its target; one this gateway returned from but
    could not record may have failed. Either way it is queued again after its backoff and settles
    (job outcome written back) when the resend finishes. Returns how many were queued again."""
    requeued = 0
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT execution_id, owner_process_id, owner_pid, owner_started_at "
            "FROM deliveries WHERE status='delivering'"
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if row["execution_id"] in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            if _requeue_uncertain(conn, row["execution_id"]):
                requeued += 1
                logger.warning("Cron delivery %s: send outcome uncertain (%s); queued again (at least once)",
                               row["execution_id"], "not recorded" if same_process else "its gateway exited")
    return requeued


def _send_one(send: Callable[[dict, str, bool], Optional[str]], row: dict,
              errors: Optional[list] = None) -> None:
    """One claimed row's own send: its outcome is recorded (and its job settled) by this row alone."""
    try:
        try:
            error = send(
                row["job"], row["content"], bool(row["for_failure"])
            )
        except BaseException as exc:  # health: allow BLE001 -- moved verbatim from drain: a raising send is this row's recorded failure
            error = f"{type(exc).__name__}: {exc}"
        try:
            if error:  # MR-RETRY-08: retried with its payload, never finished as failed
                _retry_failed_send(row, error)
            elif _finish(row["execution_id"], error=None,
                         suppressed=bool(row["job"].get("_notification_all_targets_suppressed"))):
                from cron.delivery_outcome import settle_quietly

                settle_quietly(row["job"].get("id"), row["execution_id"])
        except BaseException as exc:  # health: allow BLE001 -- handed to the waiting drain, which re-raises it; else logged with its traceback
            # Outcome not recorded: the row stays claimed and the next drain's recover_abandoned queues it
            # again (at least once). A waiting drain re-raises it to its caller, as the serial drain did.
            if errors is None:
                logger.warning("Cron delivery %s: send outcome not recorded (%s); the next drain queues it "
                               "again (at least once)", row["execution_id"], exc, exc_info=True)
            else:
                errors.append(exc)
    finally:
        with _lock:
            _ACTIVE_DELIVERIES.discard(row["execution_id"])


def drain(
    send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20, wait: bool = True
) -> int:
    """Deliver pending rows through *send*, terminalizing every claimed row.

    Every claimed row sends on its own thread, so a slow or hung send never holds a later ready row
    (DISPATCH-PRIORITY-1: each eligible row gets its own in-flight send; order is best effort only). ``wait``
    joins those sends before returning; a periodic caller passes ``wait=False`` so the next drain (new rows)
    is never held behind an in-flight one. A send this process abandons (exit mid-send) stays ``delivering``
    under its owner and is queued again by the next ``recover_abandoned`` (at least once)."""
    recover_abandoned()
    sends, errors = [], ([] if wait else None)
    for _ in range(max(0, limit)):
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(row["execution_id"])
        # Under the caller's contextvars: the profile scope selects this queue and its adapters' home.
        from agent.memory_provider import spawn_context_thread
        try:
            thread = spawn_context_thread(_send_one, args=(send, row, errors), daemon=True,
                                          name=f"cron-delivery-{row['execution_id']}")
            thread.start()
        except BaseException:
            # MR-DISPATCH-07: nothing was sent. Release this claim so the next drain's recover_abandoned (in
            # this live gateway) queues it again at once, uncounted; then surface the failure.
            with _lock:
                _ACTIVE_DELIVERIES.discard(row["execution_id"])
                _UNSTARTED_DELIVERIES.add(row["execution_id"])
            raise
        sends.append(thread)
    if wait:
        for thread in sends:
            thread.join()
        if errors:
            raise errors[0]
    return len(sends)


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  A row caught mid-send is a deferral
    too: a live gateway still finishes and settles it; a dead one's send is
    queued again (at least once), never fenced as given up.
    """
    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        owner = conn.execute(
            "SELECT owner_pid, owner_started_at FROM deliveries WHERE execution_id=? AND status='delivering'",
            (str(execution_id),),
        ).fetchone()
        if owner is not None:
            if not _owner_is_live(int(owner["owner_pid"]), owner["owner_started_at"]):
                _requeue_uncertain(conn, str(execution_id))
            logger.warning("Cron delivery %s: still in flight past the wait budget; a gateway settles it",
                           execution_id)
            return ""
        row = conn.execute(
            "SELECT status, error FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] in {"delivered", "suppressed"}:
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return None if queued["status"] in {"delivered", "suppressed"} else str(
            queued.get("error") or f"delivery {queued['status']}"
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    def _terminal_row() -> Optional[dict]:
        row = get_status(execution_id)
        return row if row and row["status"] in _TERMINAL else None

    row = poll_until(_terminal_row, wait_timeout, 1.0)
    if row is None:
        return _terminalize_wait_timeout(execution_id) or None
    return None if row["status"] in {"delivered", "suppressed"} else str(
        row.get("error") or f"delivery {row['status']}"
    )
