"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    from gateway import status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


# --- a missing record proves nothing about the live pid (tree-13c) ----------------------------
# Behaviour only: real entry points (enforce_max_runtime, release_stale_claims, reclaim_task), the
# signal hook (records, never signals) and the board row. No identity internals are imported.

def _absent_pid() -> int:
    pid = 3_500_000
    while kb._pid_alive(pid):
        pid += 1
    return pid


def test_a_live_pid_with_no_recorded_fingerprint_is_held_and_never_signalled(board):
    """A legacy NULL-fingerprint row: that SOME process holds the number does not prove it is the
    worker. The max-runtime and TTL paths hold its claim and send nothing; an operator reclaim
    releases it (human override) and still sends nothing. This test's own pid stands in for the
    unproven process; the hook only records."""
    conn = board
    killed = []

    def sig(pid, s):
        killed.append((pid, s))

    tid = _claimed_running(conn, pid=os.getpid(), started_at=None, max_runtime=1)
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    with kb.write_txn(conn):  # a stale heartbeat sends the TTL path to its termination step
        conn.execute("UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (int(time.time()) - 86400, tid))
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []
    assert kb.get_task(conn, tid).status == "ready"


def test_an_absent_pid_with_no_recorded_fingerprint_is_reclaimed(board):
    """A legacy row whose pid is gone is positively dead: the claim is released."""
    conn = board
    pid = _absent_pid()
    tid = _claimed_running(conn, pid=pid, started_at=None)

    def sig(p, s):
        assert not kb._pid_alive(p), ("signal to a live pid", p, s)
        raise ProcessLookupError(p)

    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert kb.get_task(conn, tid).status == "ready"


def test_a_matching_composed_fingerprint_still_signals_the_worker(board):
    """The positive control: a live pid whose composed fingerprint equals its record is signalled by
    the timeout path (through the recording hook only)."""
    conn = board
    killed = []
    fingerprint = kbd._process_fingerprint(os.getpid())
    _claimed_running(conn, pid=os.getpid(), started_at=fingerprint, max_runtime=1)
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, s: killed.append((pid, s)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


# --- only SAME is signalled, only GONE ends the run (tree-13d: ID13C-01/02/03) -------------------
# Real entry points and SQLite rows; the signal hook only records (no OS signal is ever sent) and
# the grace poll's sleep is a no-op. Our own pid stands in for the worker.

_COMPLETE_EPOCH = "boot-13d:42"


def _complete_epoch_worker(conn, monkeypatch, *, max_runtime=1):
    """A worker whose fingerprint carries a complete epoch, so the live pid reads SAME on entry."""
    from gateway import drain_control
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: _COMPLETE_EPOCH)
    monkeypatch.setattr(kbd.time, "sleep", lambda _s: None)
    fingerprint = kbd._process_fingerprint(os.getpid())
    tid = _claimed_running(conn, pid=os.getpid(), started_at=fingerprint, max_runtime=max_runtime)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET worker_started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (fingerprint, tid))
    return tid, fingerprint


def _custody(conn, tid):
    row = conn.execute("SELECT status, claim_lock, current_run_id, worker_pid, worker_started_at FROM tasks WHERE id = ?",
                       (tid,)).fetchone()
    run = conn.execute("SELECT status, ended_at, outcome FROM task_runs WHERE id = ?", (row["current_run_id"],)).fetchone()
    return dict(row), (dict(run) if run else None)


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


def test_identity_unreadable_after_sigterm_holds_the_timeout(board, monkeypatch):
    """ID13C-01: SIGTERM is sent to a SAME worker, then its start time can no longer be read (the
    process is still alive). UNKNOWN is no proof of death: no SIGKILL, and the claim, current run
    and worker fields are kept, the run stays open (no ``timed_out``) and ``reclaim_deferred`` is
    recorded. The next tick keeps holding while UNKNOWN; once the pid is gone the run is booked
    ``timed_out`` without any further signal."""
    from gateway import status

    conn = board
    tid, _fingerprint = _complete_epoch_worker(conn, monkeypatch)
    before = _custody(conn, tid)
    signals = []

    def sig(pid, s):
        signals.append((pid, int(s)))
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: None)

    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert signals == [(os.getpid(), int(signal.SIGTERM))]
    assert _custody(conn, tid) == before
    assert before[0]["status"] == "running" and before[1]["status"] == "running" and before[1]["ended_at"] is None
    kinds = _kinds(conn, tid)
    assert "timed_out" not in kinds and "reclaim_deferred" in kinds
    deferred = [e for e in kb.list_events(conn, tid) if e.kind == "reclaim_deferred"][-1]
    assert deferred.payload["reason"] == "max_runtime_identity_unknown"
    assert deferred.payload["signal_refused"] is True and deferred.payload["terminated"] is False

    # Next tick, still UNKNOWN: held, nothing signalled.
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert len(signals) == 1 and _custody(conn, tid) == before

    # The worker is gone: GONE ends the run as timed_out, still without another signal.
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == [tid]
    assert len(signals) == 1
    task, _run = _custody(conn, tid)
    assert task["status"] == "ready" and task["claim_lock"] is None and task["worker_pid"] is None
    last_run = conn.execute("SELECT status, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                            (tid,)).fetchone()
    assert (last_run["status"], last_run["outcome"]) == ("timed_out", "timed_out")


def test_a_worker_gone_after_sigterm_is_still_booked_timed_out(board, monkeypatch):
    """Control: SIGTERM ends the SAME worker (the pid is then absent): GONE, booked ``timed_out``
    at once with no SIGKILL, as before."""
    conn = board
    tid, _ = _complete_epoch_worker(conn, monkeypatch)
    signals = []

    def sig(pid, s):
        signals.append((pid, int(s)))
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == [tid]
    assert signals == [(os.getpid(), int(signal.SIGTERM))]
    task, _ = _custody(conn, tid)
    assert task["status"] == "ready" and task["claim_lock"] is None
    ev = [e for e in kb.list_events(conn, tid) if e.kind == "timed_out"][-1]
    assert ev.payload["sigkill"] is False


def test_a_same_worker_that_survives_sigterm_gets_sigkill_and_is_booked_once_gone(board, monkeypatch):
    """Control: the normal kill path is unchanged. A worker still SAME after the grace poll is
    SIGKILLed (recorded only); once that ends it (GONE) the run is booked ``timed_out`` with
    ``sigkill``."""
    conn = board
    tid, _ = _complete_epoch_worker(conn, monkeypatch)
    signals = []
    sigkill = int(getattr(signal, "SIGKILL", signal.SIGTERM))

    def sig(pid, s):
        signals.append((pid, int(s)))
        if int(s) == sigkill:
            monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == [tid]
    assert signals == [(os.getpid(), int(signal.SIGTERM)), (os.getpid(), sigkill)]
    ev = [e for e in kb.list_events(conn, tid) if e.kind == "timed_out"][-1]
    assert ev.payload["sigkill"] is True


def test_a_same_worker_still_alive_after_sigkill_is_held(board, monkeypatch):
    """ID13C-01 (still-live after attempted KILL): only GONE proves death. A worker that still reads
    SAME after SIGTERM and SIGKILL keeps its claim, current run and worker fields
    (``reclaim_deferred``, reason ``max_runtime_worker_alive``); the next tick signals it again."""
    conn = board
    tid, _ = _complete_epoch_worker(conn, monkeypatch)
    before = _custody(conn, tid)
    signals = []
    sigkill = int(getattr(signal, "SIGKILL", signal.SIGTERM))
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, s: signals.append((pid, int(s)))) == []
    assert signals == [(os.getpid(), int(signal.SIGTERM)), (os.getpid(), sigkill)]
    assert _custody(conn, tid) == before and before[1]["ended_at"] is None
    kinds = _kinds(conn, tid)
    assert "timed_out" not in kinds
    deferred = [e for e in kb.list_events(conn, tid) if e.kind == "reclaim_deferred"][-1]
    assert deferred.payload["reason"] == "max_runtime_worker_alive" and deferred.payload["sigkill"] is True
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, s: signals.append((pid, int(s)))) == []
    assert len(signals) == 4  # re-signalled next tick, still held


@pytest.mark.parametrize("entry", ["timeout", "ttl", "operator", "termination"])
def test_an_absent_pid_is_released_without_any_signal(board, entry):
    """ID13C-03: an absent pid is GONE, which permits release but not a signal (only SAME is
    signalled; the number could be reused between the reading and the syscall)."""
    conn = board
    pid = _absent_pid()
    tid = _claimed_running(conn, pid=pid, started_at="old-boot:1|19", max_runtime=1)
    signals = []

    def sig(p, s):
        signals.append((p, int(s)))

    if entry == "timeout":
        assert kbd.enforce_max_runtime(conn, signal_fn=sig) == [tid]
    elif entry == "ttl":
        assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    elif entry == "operator":
        assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    else:
        info = kbd._terminate_reclaimed_worker(pid, kb.get_task(conn, tid).claim_lock,
                                               started_at="old-boot:1|19", signal_fn=sig)
        assert info["terminated"] is True and not info["termination_attempted"]
        assert not kbd._worker_survived_termination(info)
    assert signals == []
    if entry != "termination":
        assert kb.get_task(conn, tid).status == "ready"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("epoch", ["", ":42", "boot-only:"], ids=["empty", "pid1-only", "boot-only"])
def test_an_identical_partial_epoch_is_never_signal_authority(board, monkeypatch, epoch):
    """ID13C-02: the real producer (``_set_worker_pid``) persists a partial epoch when a component
    cannot be read. The same partial reading later cannot tell this boot or container from another
    (Linux start ticks are boot-relative), so it is UNKNOWN: held on the timeout, TTL and
    termination paths, never signalled."""
    from gateway import drain_control
    from hermes_cli.process_identity import Identity

    conn = board
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: epoch)
    monkeypatch.setattr(kbd.time, "sleep", lambda _s: None)
    tid = kb.create_task(conn, title="partial", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    fp = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    assert fp.rsplit("|", 1)[0] == epoch
    assert kbd._worker_identity(os.getpid(), fp) is Identity.UNKNOWN
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                     (old, old, old - 86400, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    signals = []

    def sig(p, s):
        signals.append((p, int(s)))

    info = kbd._terminate_reclaimed_worker(os.getpid(), kb.get_task(conn, tid).claim_lock, started_at=fp, signal_fn=sig)
    assert info.get("signal_refused") and not info["terminated"]
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert signals == []
    assert kb.get_task(conn, tid).status == "running"


def test_a_complete_epoch_still_reads_same_and_a_differing_one_gone(monkeypatch):
    """Controls for ID13C-02: a complete epoch read alike is SAME; a complete epoch that differs is
    GONE (the start time alone recurs across boots)."""
    from gateway import drain_control
    from hermes_cli.process_identity import Identity

    me = os.getpid()
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: _COMPLETE_EPOCH)
    fp = kbd._process_fingerprint(me)
    assert kbd._worker_identity(me, fp) is Identity.SAME
    assert kbd._worker_identity(me, "other-boot:7|" + fp.rsplit("|", 1)[1]) is Identity.GONE


@pytest.mark.platforms("macos")
def test_off_linux_the_designed_empty_epoch_still_reads_same():
    """Off Linux the epoch is ``""`` by design (``current_instantiation_epoch``) and the start time is
    the clock-step-stable ``process_birth``, so the real equal reading stays SAME there."""
    from hermes_cli.process_identity import Identity

    me = os.getpid()
    fp = kbd._process_fingerprint(me)
    assert fp.rsplit("|", 1)[0] == ""
    assert kbd._worker_identity(me, fp) is Identity.SAME


# --- a stale operation never writes a successor attempt (tree-13e: ID13D-CONC-01/02) -----------
# The claim lock is a reusable host:pid: while an old timeout/reclaim waits on its kill, an operator
# release plus a fresh claim can open run2 under the same lock (here also with the old pid recycled to
# a new start time). Process readers are a private map; the hook only records; real SQLite rows.

def _snapshot(conn, tid):
    return (dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()),
            [dict(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,))],
            [dict(r) for r in conn.execute("SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,))])


@pytest.mark.parametrize("recycle", [False, True], ids=["new-pid", "recycled-pid"])
@pytest.mark.parametrize("entry", ["timeout", "ttl", "stale"])
def test_a_stale_reclaim_leaves_a_successor_attempt_untouched(board, tmp_path, monkeypatch, entry, recycle):
    """ID13D-CONC-01 (defer) and -02 (timeout finalisation): the old operation's UNKNOWN or GONE
    evidence is about run1 only. Run2's lease, worker fields, open run, events and failure count
    are left exactly as the successor wrote them."""
    from gateway import drain_control, status
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: _COMPLETE_EPOCH)
    monkeypatch.setattr(kbd.time, "sleep", lambda _s: None)
    procs = {12345: 100, 12346: 200}
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid in procs)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: procs.get(pid))
    tid = _claimed_running(board, pid=12345, started_at=f"{_COMPLETE_EPOCH}|100", max_runtime=1)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (int(time.time()) - 86400, tid))
    other = kbc.connect(tmp_path / "kanban.db")
    fresh = []

    def sig(pid, s):
        procs[12345] = None  # live but unreadable (UNKNOWN), so the operator release sends nothing
        assert kb.reclaim_task(other, tid, reason="replacement", signal_fn=lambda *_: pytest.fail("signalled"))
        assert kb.claim_task(other, tid, ttl_seconds=3600)
        if recycle:
            procs[12345] = 300  # the number recycled to a new start time
        kbd._set_worker_pid(other, tid, 12345 if recycle else 12346)
        fresh.append(_snapshot(other, tid))

    try:
        if entry == "timeout":
            assert kbd.enforce_max_runtime(board, signal_fn=sig) == []
        elif entry == "ttl":
            assert kb.release_stale_claims(board, signal_fn=sig) == 0
        else:
            assert kbd.detect_stale_running(board, stale_timeout_seconds=1, signal_fn=sig) == []
        assert fresh and _snapshot(board, tid) == fresh[0]
        assert fresh[0][0]["consecutive_failures"] == 0
    finally:
        other.close()
