"""A dead attempt behind an unresolved owner receipt must not hold its claim forever.

Regression for #140858. Card t_5ea03658 sat ``running`` for 16 hours: claimed 07:48, worker
killed by a restart at 08:59, claim expired 08:03, no heartbeat, and no ``worker_bound``
event. ``owner_reclaim_paused`` was true — a killed owner leaves a ``terminal/interrupted``
admission, or a pid that is gone — and every reclaim path read that as "leave it alone", so
``release_stale_claims``, ``detect_crashed_workers`` and ``reconcile_orphaned_running`` all
declined. ``bound_interpreter_gone``, the one override, needs the ``worker_bound`` event that
run never wrote. The card held its profile's ``max_in_progress_per_profile`` slot while every
card queued behind it was deferred, and no diagnostics rule could see a ``running`` card at
all, so the alarms named the victims and never the cause.

The pause is about effects that may still be in flight. These tests pin both sides of that
line: a live worker, a fresh heartbeat, a live claim, a foreign host's lock or an identity
that cannot be read all keep the card; a provably dead attempt is reclaimed by the platform's
own path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd
from hermes_cli import kanban_owner_recovery
from hermes_cli.process_identity import process_birth, worker_birth
from hermes_cli.kanban_owner_recovery import (
    bound_interpreter_gone,
    owner_attempt_provably_over,
    owner_reclaim_blocked,
    owner_reclaim_paused,
)

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"


def _absent_pid(base: int = 3_500_000) -> int:
    """A pid that provably does not exist on this host, so a fixture can never hit a process."""
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


class _Board:
    """A scratch board plus the scratch admission ledger an owner receipt points at."""

    def __init__(self, root: Path):
        self.path = root / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.ledger = root / "state.db"
        with sqlite3.connect(self.ledger) as store:
            store.execute(
                "CREATE TABLE session_admissions "
                "(target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)"
            )

    def close(self) -> None:
        self.conn.close()

    # --- seeding --------------------------------------------------------------------------
    def dead_attempt(
        self, tid: str = "t_ghost", *, worker_pid=None, worker_fingerprint=FINGERPRINT,
        worker_bound: bool = False, record_worker_pid: bool = True, owner_pid=None,
        owner_birth=None, admission=("terminal", "interrupted"), dead_hours: int = 15,
        expires_age_seconds=None, lock_host=None, heartbeat_age=None,
    ) -> int:
        """A ``running`` card whose worker died the way t_5ea03658's did."""
        worker_pid = _absent_pid() if worker_pid is None else worker_pid
        owner_pid = _absent_pid(base=3_600_000) if owner_pid is None else owner_pid
        lock = f"{lock_host or HOST}:{owner_pid}"
        expires = NOW - (expires_age_seconds if expires_age_seconds is not None
                         else dead_hours * 3600)
        started = NOW - dead_hours * 3600
        run_id = self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'default', 'running', ?, ?, ?, ?)",
            (tid, lock, expires, worker_pid if record_worker_pid else None, started),
        ).lastrowid
        assert run_id is not None
        run = int(run_id)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'default', 'running', ?, ?, ?, ?, ?, ?, ?, ?, 5)",
            (tid, tid, NOW - 86400, lock, expires,
             worker_pid if record_worker_pid else None,
             worker_fingerprint if record_worker_pid else None,
             (NOW - heartbeat_age) if heartbeat_age is not None else None, run, started),
        )
        self.event(tid, run, "claimed", {"source_status": "ready"}, started)
        if record_worker_pid:
            spawned = {"pid": worker_pid, "started_at": worker_fingerprint}
        else:
            spawned = {}
        self.event(tid, run, "spawned", spawned, started)
        payload = {"db": str(self.ledger), "session_id": "sess-1", "pid": owner_pid,
                   "request_id": f'kanban:["default","{tid}",1]'}
        if owner_birth is not None:
            payload["birth"] = owner_birth
        self.event(tid, run, "owner_admitted", payload, started + 30)
        if worker_bound:
            self.event(tid, run, "worker_bound",
                       {"pid": worker_pid, "claim_lock": lock, "started_at": worker_fingerprint},
                       started + 60)
        if admission:
            with sqlite3.connect(self.ledger) as store:
                store.execute("INSERT INTO session_admissions VALUES ('sess-1', ?, ?, ?)",
                              (payload["request_id"], admission[0], admission[1]))
        self.conn.commit()
        return run

    def event(self, tid: str, run: int, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (tid, run, kind, json.dumps(payload), ts))

    def ready(self, tid: str) -> None:
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, priority) "
            "VALUES (?, ?, 'default', 'ready', ?, 5)", (tid, tid, NOW - 86400))
        self.conn.commit()

    # --- reading --------------------------------------------------------------------------
    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def events(self, tid: str) -> list:
        return self.conn.execute(
            "SELECT kind, payload, created_at FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,)).fetchall()

    def diagnostics(self, tid: str, *, config: dict | None = None) -> list:
        return kd.compute_task_diagnostics(
            self.state(tid), self.events(tid), [], now=NOW, config=config or {})


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- the incident shape is named by diagnostics and reclaimed by the platform's own paths --

def test_dead_attempt_behind_an_unresolved_receipt_is_named_and_reclaimed(board):
    """Every reclaim path declined this card and no diagnostic could see it; both must now act."""
    board.dead_attempt("t_ghost")
    board.dead_attempt("t_ghost2")
    board.dead_attempt("t_live", worker_pid=os.getpid(), worker_fingerprint=kbd._process_fingerprint(os.getpid()))
    assert owner_reclaim_paused(board.conn, "t_ghost") is True      # the pause still applies
    assert bound_interpreter_gone(board.conn, "t_ghost") is False   # nothing bound to read
    assert owner_reclaim_blocked(board.conn, "t_ghost") is False

    hits = [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]
    assert [d.severity for d in hits] == ["critical"]                  # 15x the 1h grace
    assert [a.kind for a in hits[0].actions] == ["reclaim", "cli_hint"]
    assert not [d for d in board.diagnostics("t_live") if d.kind == "stale_running"]

    assert kb.release_stale_claims(board.conn) == 2
    row = board.state("t_ghost")
    assert (row["status"], row["claim_lock"], row["worker_pid"], row["current_run_id"]) == ("ready", None, None, None)
    run = board.conn.execute("SELECT status, outcome FROM task_runs WHERE task_id = 't_ghost'").fetchone()
    assert tuple(run) == ("reclaimed", "reclaimed")
    assert board.state("t_live")["status"] == "running"


@pytest.mark.parametrize("shape", [
    dict(worker_pid=os.getpid(), worker_fingerprint=kbd._process_fingerprint(os.getpid())),  # live worker
    dict(worker_bound=True, worker_pid=os.getpid(),
         worker_fingerprint=kbd._process_fingerprint(os.getpid())),                          # live bound interpreter
    # A settled admission is proof on its own (see the next test), so these model an attempt
    # that may still be executing: its admission is still ``started``.
    dict(heartbeat_age=300, admission=("started", None)),                                    # fresh heartbeat
    dict(expires_age_seconds=-3600, admission=("started", None)),                            # live claim
    dict(lock_host="some-other-host"),                                                       # foreign host
    dict(owner_pid=os.getpid(), owner_birth=None, admission=("started", None)),               # live owner in flight
    dict(owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()),
         admission=("unknown", None)),                                                       # live owner, unsettled
    dict(owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()), admission=None),  # live owner, no row
    dict(record_worker_pid=False, admission=("started", None)),                              # no pid recorded
    dict(record_worker_pid=False, expires_age_seconds=-3600),                                # no pid yet, settled
], ids=["live-worker", "live-bound", "fresh-heartbeat", "live-claim", "foreign-host", "live-owner",
        "live-owner-unsettled", "live-owner-no-row", "no-pid", "no-pid-live-claim"])
def test_the_pause_still_holds_every_attempt_that_may_be_alive(board, shape):
    """Missing evidence is never proof: reclaiming beside a possibly-live attempt repeats its effects."""
    board.dead_attempt("t_ghost", **shape)
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert kbd.detect_crashed_workers(board.conn) == []
    assert board.state("t_ghost")["status"] == "running"


def test_a_settled_admission_is_proof_even_while_the_owner_gateway_is_alive(board):
    """The owner is the long-lived gateway (alive by design): its settled admission, not its
    death, proves a lost managed worker's attempt over. A bound run stays the crash sweep's."""
    board.dead_attempt("t_ghost2")  # admission=("terminal", "interrupted"), dead owner pid
    board.dead_attempt("t_gateway", owner_pid=os.getpid(), owner_birth=None,
                       admission=("terminal", "interrupted"))
    board.dead_attempt("t_bound", worker_bound=True, owner_pid=os.getpid(), owner_birth=None,
                       admission=("terminal", "interrupted"))
    assert owner_reclaim_paused(board.conn, "t_gateway") is True   # effects still uncertain
    assert owner_attempt_provably_over(board.conn, "t_gateway") is True
    assert owner_attempt_provably_over(board.conn, "t_bound") is False
    assert kb.release_stale_claims(board.conn) == 2
    row = board.state("t_gateway")
    assert (row["status"], row["claim_lock"], row["worker_pid"]) == ("ready", None, None)
    run = board.conn.execute(
        "SELECT outcome, error FROM task_runs WHERE task_id = 't_gateway'").fetchone()
    assert run["outcome"] == "reclaimed"
    assert "owner admission settled interrupted" in run["error"]
    # Booked against the breaker like any other non-completing attempt (A3).
    assert board.state("t_gateway")["consecutive_failures"] == 1
    assert board.state("t_bound")["status"] == "running"


# --- a live owner that settled the attempt no longer holds it -------------------------------

_RELEASE_PATHS = {
    "ttl": lambda conn: kb.release_stale_claims(conn),
    "crash": lambda conn: kbd.detect_crashed_workers(conn),
}


@pytest.mark.parametrize("max_retries, landing", [(None, "ready"), (1, "blocked")])
@pytest.mark.parametrize("path", sorted(_RELEASE_PATHS))
def test_an_attempt_a_live_owner_settled_terminal_is_released(board, path, max_retries, landing):
    """2026-11-04, runs 440-446: the owner's managed child died in ``worker.adopt`` before
    ``worker_bound``, and the owner (still running, same birth) settled the admission
    ``terminal``/``interrupted``. Its pid stayed alive, so the claim was held with every worker
    pid dead and no heartbeat for as long as that owner lived. Release books the run as a failure
    under the card's ordinary retry policy; the owner's interrupted row is left as it was."""
    run = board.dead_attempt("t_fenced", owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()),
                             admission=("terminal", "interrupted"))
    board.conn.execute("UPDATE tasks SET max_retries = ? WHERE id = 't_fenced'", (max_retries,))
    board.conn.commit()
    assert owner_reclaim_paused(board.conn, "t_fenced") is True
    assert owner_attempt_provably_over(board.conn, "t_fenced") is True

    assert _RELEASE_PATHS[path](board.conn)
    row = board.state("t_fenced")
    assert (row["status"], row["claim_lock"], row["worker_pid"], row["current_run_id"]) == (landing, None, None, None)
    ended = board.conn.execute("SELECT status, outcome FROM task_runs WHERE id = ?", (run,)).fetchone()
    assert ended["outcome"] in {"reclaimed", "crashed"}
    with sqlite3.connect(board.ledger) as store:
        assert store.execute("SELECT status, outcome FROM session_admissions").fetchall() == [("terminal", "interrupted")]


def test_a_live_owners_settled_attempt_waits_for_its_worker(board):
    """Settlement proves the owner stopped, not that the dispatcher's own worker did. (A live claim
    no longer holds a settled attempt whose recorded pids are dead: upstream's
    ``test_a_settled_admission_is_proof_even_while_the_owner_gateway_is_alive``.)"""
    owner = dict(owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()),
                 admission=("terminal", "interrupted"))
    board.dead_attempt("t_live_worker", worker_pid=os.getpid(),
                       worker_fingerprint=kbd._process_fingerprint(os.getpid()), **owner)
    assert owner_attempt_provably_over(board.conn, "t_live_worker") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert kbd.detect_crashed_workers(board.conn) == []


# --- the gate is decided under the same write lock as the release ---------------------------

_RECLAIM_PATHS = {
    "ttl": lambda conn: kb.release_stale_claims(conn),
    "stale": lambda conn: kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *_: None),
    "runtime": lambda conn: kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None),
}


@pytest.mark.parametrize("path", sorted(_RECLAIM_PATHS))
def test_an_interpreter_bound_after_the_gate_keeps_its_claim(board, monkeypatch, path):
    """Binding changes neither claim_lock nor worker_pid, so only a re-check inside the
    reclaim transaction can see an owner that recovered after the first look."""
    run = board.dead_attempt("t_ghost")
    board.conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = 't_ghost'")
    board.conn.commit()
    real_gate, calls = owner_reclaim_blocked, []

    def gate_then_bind(conn, task_id, **kwargs):
        verdict = real_gate(conn, task_id, **kwargs)
        if not calls:  # the recovered owner binds a live interpreter right after the first look
            board.event(task_id, run, "worker_bound",
                        {"pid": os.getpid(), "started_at": kbd._process_fingerprint(os.getpid())}, NOW)
            board.conn.commit()
        calls.append(verdict)
        return verdict

    monkeypatch.setattr(kanban_owner_recovery, "owner_reclaim_blocked", gate_then_bind)
    assert not _RECLAIM_PATHS[path](board.conn)
    assert calls[0] is False and calls[-1] is True
    row = board.state("t_ghost")
    assert (row["status"], row["current_run_id"]) == ("running", run)



@pytest.mark.parametrize("path", ["ttl", "stale"])
def test_a_foreign_host_interpreter_bound_after_the_gate_keeps_its_claim(board, monkeypatch, path):
    """No local signal was sent for a foreign-host claim, so its bound interpreter's pid says
    nothing here: an absent local pid is no proof that the remote attempt ended."""
    run = board.dead_attempt("t_remote", lock_host="some-other-host", admission=("terminal", "completed"))
    real_gate, calls = owner_reclaim_blocked, []

    def gate_then_remote_bind(conn, task_id, **kwargs):
        verdict = real_gate(conn, task_id, **kwargs)
        if not calls:  # the remote owner admits and binds an interpreter right after the first look
            board.event(task_id, run, "worker_bound", {"pid": _absent_pid(base=3_700_000)}, NOW)
            board.conn.commit()
            with sqlite3.connect(board.ledger) as store:
                store.execute("UPDATE session_admissions SET status = 'unknown', outcome = NULL")
        calls.append(verdict)
        return verdict

    monkeypatch.setattr(kanban_owner_recovery, "owner_reclaim_blocked", gate_then_remote_bind)
    assert not _RECLAIM_PATHS[path](board.conn)
    row = board.state("t_remote")
    assert (row["status"], row["current_run_id"]) == ("running", run)

@pytest.mark.platforms("linux", "macos")
def test_a_timed_out_managed_run_is_booked_timed_out_not_left_for_the_crash_sweep(board):
    """Killing a managed run's worker takes its bound interpreter down with it, and the owner
    settles the admission ``unknown``. The under-lock re-check must not read that pause, which
    the timeout itself caused, as an owner still in flight: the run is ``timed_out``, never left
    running for the crash sweep to book ``crashed`` on the next tick."""
    worker, interpreter = (subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
                           for _ in range(2))
    try:
        run = board.dead_attempt(
            "t_slow", worker_pid=worker.pid, worker_fingerprint=kbd._process_fingerprint(worker.pid),
            owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()), admission=("started", None))
        board.event("t_slow", run, "worker_bound", {"pid": interpreter.pid,
                    "started_at": kbd._process_fingerprint(interpreter.pid)}, NOW)
        board.conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = 't_slow'")
        board.conn.commit()

        def sigterm_ends_the_attempt(pid, sig):
            os.kill(pid, sig)
            interpreter.kill()  # _watch_submitter: the interpreter exits with its worker
            interpreter.wait()
            with sqlite3.connect(board.ledger) as store:  # the owner settles the lost turn
                store.execute("UPDATE session_admissions SET status = 'unknown'")

        assert kbd.enforce_max_runtime(board.conn, signal_fn=sigterm_ends_the_attempt) == ["t_slow"]
        outcome = board.conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (run,)).fetchone()[0]
        assert (board.state("t_slow")["status"], outcome) == ("ready", "timed_out")
        assert kbd.detect_crashed_workers(board.conn) == []
    finally:
        for child in (worker, interpreter):
            child.kill()
            child.wait()


@pytest.mark.platforms("linux", "macos")
def test_an_exited_unreaped_owner_no_longer_holds_the_slot(board):
    """A zombie keeps its birth until reaped, but it has exited: it cannot be executing anything."""
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        birth = worker_birth(psutil.Process(child.pid))
        child.stdin.close()
        deadline = time.time() + 10
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.time() < deadline, "child never exited"
            time.sleep(0.02)
        board.dead_attempt("t_ghost", owner_pid=child.pid, owner_birth=birth)
        assert kb.release_stale_claims(board.conn) == 1
        assert board.state("t_ghost")["status"] == "ready"
    finally:
        child.wait()


def test_an_earlier_runs_live_pid_does_not_hide_the_current_dead_run(board):
    """Only the current attempt's pids speak for it in the stale_running diagnostic."""
    run = board.dead_attempt("t_ghost")
    board.event("t_ghost", run - 1, "spawned", {"pid": os.getpid()}, NOW - 100_000)
    board.conn.commit()
    assert [d.kind for d in board.diagnostics("t_ghost") if d.kind == "stale_running"] == ["stale_running"]


# --- receipts written by an owner from before the boot-tick birth (old base, int-1) ------------

def _epoch_birth(pid=None) -> float:
    """The birth an owner from before ``worker_birth`` wrote: the epoch ``create_time``."""
    return process_birth(psutil.Process(os.getpid() if pid is None else pid))


def _lapsed_unsettled(board, tid, admission, **owner) -> int:
    """An attempt only its owner can still be running: the claim lapsed with no heartbeat and
    every recorded worker pid is dead, but the owner's own admission is not settled."""
    return board.dead_attempt(tid, admission=admission, **owner)


@pytest.mark.platforms("linux")
def test_the_two_owner_clocks_differ_for_a_live_process():
    """The precondition of every case below: the old and current receipt births of one process differ."""
    me = psutil.Process()
    assert process_birth(me) != worker_birth(me)


_OWNER_GATES = {
    "ttl": lambda conn: kb.release_stale_claims(conn),
    "crash": lambda conn: kbd.detect_crashed_workers(conn),
    "stale": lambda conn: kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *_: None),
}


@pytest.mark.parametrize("admission", [("unknown", None), ("queued", None)], ids=["unknown", "queued"])
@pytest.mark.parametrize("path", sorted(_OWNER_GATES))
@pytest.mark.parametrize("clock", ["epoch-receipt", "boot-receipt"])
def test_a_live_owner_keeps_its_attempt_whichever_clock_its_receipt_used(board, path, admission, clock):
    """I1: an owner that admitted the attempt before an update wrote its epoch ``process_birth``.
    While that owner still runs it may still execute the admission (``unknown``) or start it
    (``queued``), so no reclaim path may take the card. Read with ``worker_birth`` alone, the receipt
    made the live owner look like a recycled pid: ``_owner_process_gone`` said gone and every path
    reclaimed beside it."""
    birth = _epoch_birth() if clock == "epoch-receipt" else worker_birth(psutil.Process())
    _lapsed_unsettled(board, "t_old_owner", admission, owner_pid=os.getpid(), owner_birth=birth)
    assert kanban_owner_recovery._owner_process_gone({"pid": os.getpid(), "birth": birth}) is False
    assert owner_attempt_provably_over(board.conn, "t_old_owner") is False
    assert owner_reclaim_blocked(board.conn, "t_old_owner") is True
    assert not _OWNER_GATES[path](board.conn)
    row = board.state("t_old_owner")
    assert (row["status"], row["claim_lock"]) == ("running", f"{HOST}:{os.getpid()}")


@pytest.mark.parametrize("clock", ["epoch-receipt", "boot-receipt"])
def test_a_dead_owners_attempt_is_reclaimed_whichever_clock_its_receipt_used(board, clock):
    """The owner exited after writing its receipt: nothing can run the attempt, so it is released."""
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        proc = psutil.Process(child.pid)
        birth = process_birth(proc) if clock == "epoch-receipt" else worker_birth(proc)
    finally:
        child.stdin.close()
        child.wait()
    assert not psutil.pid_exists(child.pid) or psutil.Process(child.pid).create_time() != proc.create_time()
    _lapsed_unsettled(board, "t_dead_owner", ("unknown", None), owner_pid=child.pid, owner_birth=birth)
    assert owner_attempt_provably_over(board.conn, "t_dead_owner") is True
    assert kb.release_stale_claims(board.conn) == 1
    assert board.state("t_dead_owner")["status"] == "ready"


def test_a_recycled_owner_pid_is_still_detected_from_a_boot_receipt(board):
    """The receipt's number now belongs to another live process: its boot-relative birth (which no
    clock step moves) matches on neither clock, so the recorded owner is gone and its unsettled
    attempt is released."""
    other = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        # This test process's birth stands in for the earlier incarnation that held other.pid.
        birth = worker_birth(psutil.Process())
        _lapsed_unsettled(board, "t_recycled", ("unknown", None), owner_pid=other.pid, owner_birth=birth)
        assert kanban_owner_recovery._owner_process_gone({"pid": other.pid, "birth": birth}) is True
        assert kb.release_stale_claims(board.conn) == 1
        assert board.state("t_recycled")["status"] == "ready"
    finally:
        other.stdin.close()
        other.wait()


@pytest.mark.platforms("linux")
def test_a_legacy_epoch_receipt_mismatch_holds_until_the_pid_is_gone(board):
    """I1-HB-01's availability cost. A legacy epoch receipt whose birth matches neither clock of the
    live process holding its pid is UNKNOWN, not death: a recycled number and the same owner after a
    btime re-anchor read alike, so the unsettled attempt is held. It is released as soon as the pid
    is absent (or the owner settles the admission)."""
    from hermes_cli.process_identity import Identity

    other = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        birth = _epoch_birth()  # this process's epoch birth: other.pid's, if it had been the owner
        _lapsed_unsettled(board, "t_recycled", ("unknown", None), owner_pid=other.pid, owner_birth=birth)
        receipt = {"pid": other.pid, "birth": birth}
        assert kanban_owner_recovery._owner_identity(receipt) is Identity.UNKNOWN
        assert kanban_owner_recovery._owner_process_gone(receipt) is False
        assert kb.release_stale_claims(board.conn) == 0
        assert kbd.detect_crashed_workers(board.conn) == []
        assert board.state("t_recycled")["status"] == "running"
    finally:
        other.stdin.close()
        other.wait()
    assert kanban_owner_recovery._owner_identity(receipt) is Identity.GONE
    assert kb.release_stale_claims(board.conn) == 1
    assert board.state("t_recycled")["status"] == "ready"


@pytest.mark.parametrize("clock", ["epoch-receipt", "boot-receipt"])
def test_an_owner_whose_birth_cannot_be_read_is_never_proof_of_death(board, monkeypatch, clock):
    """Permission or other psutil errors on either clock leave the attempt held, as before."""
    from hermes_cli import process_identity

    birth = _epoch_birth() if clock == "epoch-receipt" else worker_birth(psutil.Process())

    def denied(proc):
        raise psutil.AccessDenied(proc.pid)

    monkeypatch.setattr(process_identity, "worker_birth", denied)
    monkeypatch.setattr(process_identity, "process_birth", denied)
    _lapsed_unsettled(board, "t_unreadable", ("unknown", None), owner_pid=os.getpid(), owner_birth=birth)
    assert kb.release_stale_claims(board.conn) == 0
    assert kbd.detect_crashed_workers(board.conn) == []
    assert board.state("t_unreadable")["status"] == "running"


def test_a_live_epoch_receipt_owner_reads_as_the_same_owner_to_the_pause(board):
    """``owner_reclaim_paused`` applies the same identity rule: an old-format receipt of the live owner
    with a ``started`` admission reads as that owner, exactly as a current-format one does."""
    board.dead_attempt("t_epoch", owner_pid=os.getpid(), owner_birth=_epoch_birth(), admission=("started", None))
    board.dead_attempt("t_boot", owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()),
                       admission=("started", None))
    board.dead_attempt("t_other", owner_pid=os.getpid(), owner_birth=_epoch_birth() - 1.0,
                       admission=("started", None))
    assert owner_reclaim_paused(board.conn, "t_epoch") is False
    assert owner_reclaim_paused(board.conn, "t_boot") is False
    assert owner_reclaim_paused(board.conn, "t_other") is True


# --- identity that cannot be proven is never proof of death (I1-HB-01, I1-HB-02) --------------

def _reclaim(board, path):
    def no_live_signal(pid, sig):
        assert not psutil.pid_exists(pid), ("signal to a live pid", pid, sig)
        raise ProcessLookupError(pid)
    if path == "ttl":
        return kb.release_stale_claims(board.conn, signal_fn=no_live_signal)
    if path == "crash":
        return kbd.detect_crashed_workers(board.conn)
    return kbd.detect_stale_running(board.conn, stale_timeout_seconds=1, signal_fn=no_live_signal)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("path", ["ttl", "crash", "stale"])
@pytest.mark.parametrize("clock", ["epoch-receipt", "boot-receipt"])
def test_a_btime_reanchor_never_makes_a_live_owner_look_dead(board, monkeypatch, path, clock):
    """I1-HB-01: only psutil's epoch boot anchor moves (WSL2 resume, a clock step). A legacy epoch
    receipt of the live owner then matches neither clock; that is UNKNOWN, and the unsettled
    attempt is held on every reclaim path. A boot-relative receipt still reads SAME."""
    from hermes_cli.process_identity import Identity

    pid = os.getpid()
    birth = _epoch_birth() if clock == "epoch-receipt" else worker_birth(psutil.Process())
    _lapsed_unsettled(board, "t_clock", ("unknown", None), owner_pid=pid, owner_birth=birth)
    real_boot_time = psutil._pslinux.boot_time
    monkeypatch.setattr(psutil._pslinux, "boot_time", lambda: real_boot_time() + 1)
    expected = Identity.UNKNOWN if clock == "epoch-receipt" else Identity.SAME
    assert kanban_owner_recovery._owner_identity({"pid": pid, "birth": birth}) is expected
    assert kanban_owner_recovery._owner_process_gone({"pid": pid, "birth": birth}) is False
    assert owner_attempt_provably_over(board.conn, "t_clock") is False
    assert not _reclaim(board, path)
    assert board.state("t_clock")["status"] == "running"


@pytest.mark.parametrize("path", ["ttl", "crash"])
@pytest.mark.parametrize("unreadable", [False, True], ids=["readable", "unreadable"])
def test_an_unreadable_worker_identity_is_not_proof_of_death(board, monkeypatch, path, unreadable):
    """I1-HB-02: a live recorded worker whose start time cannot be read now is UNKNOWN. The settled
    admission's death proof needs every recorded worker GONE, so the attempt is held, and nothing
    is signalled."""
    from gateway import status
    from hermes_cli.process_identity import Identity

    pid = os.getpid()
    fingerprint = kbd._process_fingerprint(pid)
    run = board.dead_attempt("t_read", worker_pid=pid, worker_fingerprint=fingerprint, owner_pid=pid,
                             owner_birth=worker_birth(psutil.Process()), admission=("terminal", "interrupted"))
    assert kbd.heartbeat_worker(board.conn, "t_read", expected_run_id=run)
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 1, "t_read"))
    if unreadable:
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: None)
    assert kbd._worker_identity(pid, fingerprint) is (Identity.UNKNOWN if unreadable else Identity.SAME)
    assert kbd._worker_alive(pid, fingerprint) is True
    assert kbd._pid_recycled(pid, fingerprint) is unreadable  # never signalled while unproven
    assert owner_attempt_provably_over(board.conn, "t_read") is False
    assert not _reclaim(board, path)
    assert board.state("t_read")["status"] == "running"


def test_worker_identity_verdicts(monkeypatch):
    """GONE only on positive evidence; an unreadable reading or an unusable record is UNKNOWN."""
    from gateway import status
    from hermes_cli.process_identity import Identity

    me = os.getpid()
    fingerprint = kbd._process_fingerprint(me)
    start = status.get_process_start_time(me)
    epoch = fingerprint.rsplit("|", 1)[0]
    assert kbd._worker_identity(_absent_pid(), fingerprint) is Identity.GONE
    assert kbd._worker_identity(me, fingerprint) is Identity.SAME
    assert kbd._worker_identity(me, None) is Identity.UNKNOWN  # no record: existence proves nothing
    assert kbd._worker_identity(_absent_pid(), None) is Identity.GONE
    assert kbd._worker_identity(me, start) is Identity.SAME  # legacy integer start time
    assert kbd._worker_identity(me, f"{epoch}|{start + 1}") is Identity.GONE  # recycled number
    assert kbd._worker_identity(me, start + 1000) is Identity.GONE
    assert kbd._worker_identity(me, kbd.UNVERIFIED_WORKER_FINGERPRINT) is Identity.UNKNOWN
    assert kbd._worker_identity(me, "junk") is Identity.UNKNOWN
    monkeypatch.setattr(status, "get_process_start_time", lambda _pid: None)
    assert kbd._worker_identity(me, fingerprint) is Identity.UNKNOWN
    assert kbd._worker_identity(me, start) is Identity.UNKNOWN
    assert kbd._worker_identity(_absent_pid(), fingerprint) is Identity.GONE


def test_an_unreadable_worker_identity_is_held_and_never_signalled_by_termination(board, monkeypatch):
    """The signal paths: an UNKNOWN live identity is refused a signal and reported as surviving, so
    the reclaim defers instead of releasing beside it; enforce_max_runtime neither signals nor
    releases it."""
    from gateway import status

    pid = os.getpid()
    fingerprint = kbd._process_fingerprint(pid)
    lock = f"{HOST}:{pid}"
    signalled = []

    def sig(p, s):
        signalled.append((p, s))

    tid = kb.create_task(board.conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(board.conn, tid)
    kbd._set_worker_pid(board.conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
        board.conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid))
    monkeypatch.setattr(status, "get_process_start_time", lambda _pid: None)
    info = kbd._terminate_reclaimed_worker(pid, lock, signal_fn=sig, started_at=fingerprint)
    assert info["signal_refused"] is True and info["terminated"] is False
    assert kbd._worker_survived_termination(info) is True
    assert kbd.enforce_max_runtime(board.conn, signal_fn=sig) == []
    assert signalled == []
    assert kb.get_task(board.conn, tid).status == "running"
