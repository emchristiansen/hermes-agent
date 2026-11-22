"""Durable idempotency reservations for API server runs."""

import asyncio
import concurrent.futures
import hmac
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict

from gateway.hosted_room_peer import MAX_STATUS_GRANT_TTL_SECONDS
from hermes_cli.sqlite_util import add_column_if_missing


# Keep the extracted store's log records on the API server logger.
logger = logging.getLogger("gateway.platforms.api_server")

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
# A key closed by ``find_or_close``: no request body hashes to it, so it never replays as a run.
_CLOSED_FINGERPRINT = "closed"

_SELECT_BY_KEY = (
    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at, admitted "
    "FROM run_idempotency WHERE scope=? AND idempotency_key=?")
# ``retention_until < ?`` keeps ``MAX(retention_until, ?)`` exactly while matching no row when the
# horizon would not move: SQLite rewrites a matched row even when no value changes, and under
# ``synchronous=FULL`` that is a WAL frame and an fsync for every unchanged room-grant poll.
_EXTEND_RETENTION_BY_KEY = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND idempotency_key=? AND fingerprint=? AND retention_until < ?")
_EXTEND_RETENTION_BY_RUN = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND run_id=? AND retention_until < ?")
_UPDATE_STATUS = "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?"
# Columns added after the first schema shipped; applied when missing.
_MIGRATIONS = {
    "owner_pid": "INTEGER NOT NULL DEFAULT 0",
    "owner_started": "INTEGER NOT NULL DEFAULT 0",
    "retention_until": "REAL NOT NULL DEFAULT 0",
    "acknowledged_at": "REAL",
    # 0 while the reserving request may still abandon the row (its admission is undecided);
    # rows written before this column existed were reserved by code that admitted synchronously.
    "admitted": "INTEGER NOT NULL DEFAULT 1"}


# A confirmation names the run it vouches for: a key that terminal retention released and a later
# request reserved again belongs to another run, which this write must never admit (a no-op).
_CONFIRM = ("UPDATE run_idempotency SET admitted=1 "
            "WHERE scope=? AND idempotency_key=? AND run_id=? AND admitted=0")


def _encode_status(status: dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True, separators=(",", ":"))


def _record(run_id, status_json, owner_pid, owner_started, updated_at, admitted=None) -> dict[str, Any]:
    record = {
        "run_id": run_id, "status": json.loads(status_json), "owner_pid": int(owner_pid or 0),
        "owner_started": int(owner_started or 0), "updated_at": float(updated_at or 0)}
    if admitted is not None:
        record["admitted"] = bool(admitted)
    return record


class _PendingStatus:
    """One run's newest unwritten status snapshot and everyone waiting for it to be durable."""

    __slots__ = ("encoded", "waiters")

    def __init__(self, encoded: str, waiters: list):
        self.encoded = encoded
        self.waiters = waiters


def _outcome(row, fingerprint):
    """Classify a stored ``(scope, key)`` row against the caller's fingerprint."""
    return ("reused" if hmac.compare_digest(row[0], fingerprint) else "conflict"), _record(*row[1:])


class RunIdempotencyStore:
    """Durable, tenant-scoped reservations for ``POST /v1/runs``: a unique ``(scope, key)`` row
    inserted inside ``BEGIN IMMEDIATE`` so separate workers cannot both admit one request. Only
    fingerprints and public run status are stored — never request bodies or credentials.

    Every commit is an fsync (``synchronous=FULL``), so the event loop never runs one: it awaits
    the ``*_async`` operations, which run on this store's own single worker thread, and status
    writes are write-behind (``enqueue_status``). Correctness does not rest on that thread's FIFO
    order: a status write takes the newest pending snapshot of its run under ``_lock`` (so an
    older snapshot can never land after a newer one), and every other operation applies the
    pending snapshots first under the same lock (read-your-writes for any thread). ``close`` drains
    the worker, so a status published before shutdown is durable before the connection closes."""

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60

    @property
    def durable(self) -> bool:
        """Whether reservations survive this process."""
        return self._db_path is not None
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "runs_idempotency.db")
            except Exception:
                db_path = ":memory:"
        self._db_path = None if db_path == ":memory:" else db_path
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        except Exception as exc:
            # Docker may create the container object before `docker run` fails to start it (e.g. exit code
            # 125 when the daemon isn't ready, or a timeout mid-pull). That orphan is left in "Created"
            # state — which the exited-only orphan reaper (reap_orphan_containers, status=exited) never
            # catches, so it leaks permanently. Remove it by its known name before re-raising. See #7439.
            logger.warning(
                "Run idempotency storage is unavailable; falling back to "
                "process memory, so replay will not survive a restart: %s", exc)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0,
                owner_started INTEGER NOT NULL DEFAULT 0,
                retention_until REAL NOT NULL DEFAULT 0,
                acknowledged_at REAL,
                admitted INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )"""
        )
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(run_idempotency)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in columns:
                add_column_if_missing(self._conn, "run_idempotency", column, f"{column} {ddl}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS run_idempotency_run_id ON run_idempotency(run_id)")
        self._conn.commit()
        self._lock = threading.Lock()
        # run_id -> newest status not yet written / being written; guarded by _pending_lock,
        # which is never held across I/O (a loop-thread enqueue waits only for dict updates).
        self._pending_status: dict[str, _PendingStatus] = {}
        self._writing_status: dict[str, _PendingStatus] = {}
        self._pending_lock = threading.Lock()
        # Admissions whose confirmation has not committed yet: owed until one does (each store
        # operation retries them first, as it writes pending statuses first).
        self._owed_confirmations: set[tuple[str, str, str]] = set()  # (scope, key, run_id)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hermes-run-idempotency")
        # The one close operation (see ``_begin_close``), owned by the store, not by any caller.
        self._closing: concurrent.futures.Future | None = None
        self._close_lock = threading.Lock()
        self._tighten_permissions()

    def _tighten_permissions(self) -> None:
        for suffix in ("", "-wal", "-shm") if self._db_path else ():
            candidate = Path(self._db_path + suffix)
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict run idempotency store permissions", exc_info=True)

    @contextmanager
    def _immediate_txn(self):
        """Hold the lock inside ``BEGIN IMMEDIATE``; the body commits, errors roll back."""
        with self._lock:
            self._apply_pending_locked()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._conn.rollback()
                raise

    # -- off-loop execution ---------------------------------------------------------------

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        """Run one store operation on the store's worker thread (``RuntimeError`` once closed)."""
        return self._executor.submit(fn, *args, **kwargs)

    async def _run(self, fn, /, *args, **kwargs):
        return await asyncio.wrap_future(self.submit(partial(fn, *args, **kwargs)))

    async def lookup_async(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        return await self._run(self.lookup, scope, key, fingerprint, retention_until=retention_until)

    async def status_for_run_async(self, scope: str, run_id: str, *, retention_until: float = 0):
        return await self._run(self.status_for_run, scope, run_id, retention_until=retention_until)

    async def extend_retention_async(self, scope: str, run_id: str, until: float) -> bool:
        return await self._run(self.extend_retention, scope, run_id, until)

    async def owns_run_async(self, scope: str, run_id: str) -> bool:
        return await self._run(self.owns_run, scope, run_id)

    async def forget_async(self, scope: str, key: str) -> None:
        return await self._run(self.forget, scope, key)

    # -- write-behind status --------------------------------------------------------------

    def enqueue_status(self, run_id: str, status: dict[str, Any]) -> None:
        """Persist *status* as the run's newest durable status without waiting for the commit.

        The snapshot is encoded now, on the caller's thread, so later in-memory mutation cannot
        leak into it. Raises (``RuntimeError`` once closed, an encoding error) like a synchronous
        write would; a failed commit is logged on the worker thread. ``durable_after`` waits."""
        encoded = _encode_status(status)
        with self._pending_lock:
            previous = self._pending_status.get(run_id)
            entry = self._pending_status[run_id] = _PendingStatus(
                encoded, previous.waiters if previous is not None else [])
        try:
            self.submit(self._flush_status, run_id)
        except BaseException:
            with self._pending_lock:
                if self._pending_status.get(run_id) is entry:
                    if previous is not None:
                        self._pending_status[run_id] = previous
                    else:
                        del self._pending_status[run_id]
            raise

    def durable_after(self, run_id: str) -> concurrent.futures.Future | None:
        """A future that settles once every status enqueued for *run_id* so far has been written
        (its result) or has failed to be (its exception, already logged); None when nothing is
        outstanding. Publishing a terminal result from its callback keeps the original order:
        the successful commit first, then the publication. Callbacks run on the store's worker
        under an internal lock: they must only hand work to another thread (``call_soon_threadsafe``)."""
        with self._pending_lock:
            entry = self._pending_status.get(run_id) or self._writing_status.get(run_id)
            if entry is None:
                return None
            waiter: concurrent.futures.Future = concurrent.futures.Future()
            entry.waiters.append(waiter)
            return waiter

    def enqueue_confirmation(self, scope: str, key: str, run_id: str) -> None:
        """Owe the confirmation of *run_id*'s admitted reservation (write-behind, like
        ``enqueue_status``). It stays owed until its commit succeeds: a failed attempt is logged
        and retried before every later store operation (the run's own status writes included) and
        at close. The write matches the run as well as the key, so once the key names another run
        (expired and reserved again) the owed confirmation lands as a no-op.
        Raises ``RuntimeError`` once closed."""
        with self._pending_lock:
            self._owed_confirmations.add((scope, key, run_id))
        self.submit(self._flush_status, None)

    def _confirm_owed_locked(self) -> None:
        with self._pending_lock:
            owed = sorted(self._owed_confirmations)
        for identity in owed:
            try:
                self._conn.execute(_CONFIRM, identity)
                self._conn.commit()
            except Exception:
                logger.exception("[api_server] failed to confirm an admitted run reservation; still owed")
                try:
                    self._conn.rollback()
                except Exception:
                    logger.debug("Run idempotency rollback failed", exc_info=True)
                continue
            with self._pending_lock:
                self._owed_confirmations.discard(identity)

    def _flush_status(self, run_id: str | None) -> None:
        with self._lock:  # reported before any later store operation runs
            self._flush_pending_locked(run_id)

    def _apply_pending_locked(self) -> None:
        """Read-your-writes for every other operation. A failed status write stays what it always
        was (logged, never the caller's error), so each operation keeps its own error contract."""
        self._flush_pending_locked()

    def _flush_pending_locked(self, run_id: str | None = None) -> None:
        """Write pending status snapshots (one run's, or all); the caller holds ``_lock``.

        Popping under ``_pending_lock`` while holding ``_lock`` is what orders the writes: a
        snapshot is written only by the holder that popped it, and whatever is pending at a later
        pop is newer, so the stored status only ever moves forward in enqueue order.

        Each run's snapshot is its own statement and commit, as each synchronous publication was
        before: one run's failed write (logged, then rolled back) never takes another run's
        accepted publication with it, and a failed commit leaves the next run's write attempted.
        Owed admission confirmations are retried first (``enqueue_confirmation``)."""
        self._confirm_owed_locked()
        with self._pending_lock:
            if run_id is None:
                items = list(self._pending_status.items())
                self._pending_status.clear()
            else:
                entry = self._pending_status.pop(run_id, None)
                items = [] if entry is None else [(run_id, entry)]
            for pending_run_id, entry in items:
                self._writing_status[pending_run_id] = entry
        for pending_run_id, entry in items:
            failure = None
            try:
                self._conn.execute(_UPDATE_STATUS, (entry.encoded, time.time(), pending_run_id))
                self._conn.commit()
            except Exception as exc:
                failure = exc
                logger.exception("[api_server] failed to persist idempotent run status %s", pending_run_id)
                try:
                    self._conn.rollback()
                except Exception:
                    logger.debug("Run idempotency rollback failed", exc_info=True)
            # Settled before the entry stops being findable: a ``durable_after`` that finds
            # nothing outstanding can rely on every earlier waiter's callbacks having run.
            with self._pending_lock:
                if self._writing_status.get(pending_run_id) is entry:
                    del self._writing_status[pending_run_id]
                for waiter in entry.waiters:
                    if failure is None:
                        waiter.set_result(None)
                    else:
                        waiter.set_exception(failure)
                entry.waiters = []

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str, status: dict[str, Any], *,
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0,
                provisional: bool = False):
        """Atomically reserve a key; return ``(outcome, stored_record)``. A *provisional* row is not
        yet an admission (``confirm_admission`` or ``forget`` decides it): readers in any process
        see ``admitted: False`` and must not acknowledge it as admitted work."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is not None:
                if retention_until:
                    self._conn.execute(
                        _EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint, retention_until))
                self._conn.commit()
                return _outcome(row, fingerprint)
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,created_at,updated_at,admitted"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, now, now, 0 if provisional else 1))
            self._conn.commit()
            return "created", _record(run_id, encoded, owner_pid, owner_started, now, not provisional) | {
                "status": status}

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._immediate_txn():
            if retention_until:
                self._conn.execute(
                    _EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint, retention_until))
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            self._conn.commit()
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def find_or_close(self, scope: str, key: str):
        """``("found", record)`` for the run *key* admitted, else ``("closed", None)`` after
        closing it in the same write transaction as ``reserve``: a closed key never admits, since
        a later POST under it answers a fingerprint conflict. A key whose run ended and aged out
        (only terminal rows are pruned) is closed too, which still means nothing runs under it."""
        now = time.time()
        # Another grant alive at this close may outlast the one carrying the request.
        retention_until = now + MAX_STATUS_GRANT_TTL_SECONDS
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO run_idempotency(scope,idempotency_key,fingerprint,run_id,status_json,"
                    "retention_until,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (scope, key, _CLOSED_FINGERPRINT, f"closed_{uuid.uuid4().hex}",
                     _encode_status({"status": "cancelled"}), retention_until, now, now))
            elif row[0] == _CLOSED_FINGERPRINT:
                self._conn.execute(_EXTEND_RETENTION_BY_KEY,
                                   (retention_until, scope, key, _CLOSED_FINGERPRINT))
            self._conn.commit()
        if row is None or row[0] == _CLOSED_FINGERPRINT:
            return "closed", None
        return "found", _record(*row[1:])

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        for stale_scope, stale_key, stale_status in stale:
            try:
                terminal = json.loads(stale_status).get("status") in TERMINAL_STATUSES
            except Exception:
                terminal = False
            if terminal:
                self._conn.execute(
                    "DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (stale_scope, stale_key))

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0) -> dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            self._apply_pending_locked()
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_RUN, (retention_until, scope, run_id, retention_until))
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at, admitted "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id)).fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            self._apply_pending_locked()
            changed = self._conn.execute(
                _EXTEND_RETENTION_BY_RUN, (checked_until, scope, run_id, checked_until)).rowcount
            self._conn.commit()
            if changed == 1:
                return True
            # Unchanged horizon: the row still answers True exactly as the unconditional
            # ``MAX`` update did, without writing (and fsyncing) an identical row.
            row = self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone()
        return row is not None

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._lock:
            self._apply_pending_locked()
            row = self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone()
        return row is not None

    def update_status(self, run_id: str, status: dict[str, Any]) -> None:
        """Synchronously persist *status*, ordered after every status enqueued before it; raises
        this write's own error."""
        encoded = _encode_status(status)
        waiter: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            with self._pending_lock:
                previous = self._pending_status.get(run_id)
                self._pending_status[run_id] = _PendingStatus(
                    encoded, [*(previous.waiters if previous is not None else ()), waiter])
            self._flush_pending_locked()
        waiter.result()

    def forget(self, scope: str, key: str) -> None:
        """Release a reservation whose run was refused before it existed (raises when the row
        could not be deleted: the caller still owes the release)."""
        with self._lock:
            self._apply_pending_locked()
            try:
                self._conn.execute("DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (scope, key))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def confirm_admission(self, scope: str, key: str, run_id: str) -> None:
        """Mark *run_id*'s provisional reservation admitted: the run exists and may be replayed
        anywhere. A no-op when the key no longer names that run."""
        with self._lock:
            self._apply_pending_locked()
            try:
                self._conn.execute(_CONFIRM, (scope, key, run_id))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _close_connection(self) -> None:
        with self._lock:
            self._apply_pending_locked()
            self._conn.close()

    def _begin_close(self) -> concurrent.futures.Future:
        """The store's single close operation, started on first use: queued behind every pending
        commit on the worker, which then stops accepting work. Every later ``close``/``aclose``
        joins this same operation, so its outcome (done, or its error) is what each one reports;
        no caller's cancellation can withdraw it (the executor never cancels a queued job unless
        asked to with ``cancel_futures``, and nobody asks)."""
        with self._close_lock:
            if self._closing is None:
                self._closing = self.submit(self._close_connection)
                self._executor.shutdown(wait=False)
            return self._closing

    def close(self) -> None:
        """Drain the worker (queued commits land first), then close the connection; raises the
        close's error, as every later call does too."""
        closing = self._begin_close()
        self._executor.shutdown(wait=True)
        closing.result()

    async def aclose(self) -> None:
        """``close`` without blocking the event loop: the drain and the final commit run on the
        store's own worker. A cancelled caller stops waiting, never the close itself; a later call
        joins the same operation and reports its real outcome."""
        await asyncio.shield(asyncio.wrap_future(self._begin_close()))
