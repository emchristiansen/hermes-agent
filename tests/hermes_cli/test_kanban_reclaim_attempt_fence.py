"""A reclaim acts only for the attempt and eligibility it selected (tree-13f: ID13E-CONC-03/04/05).

Between a sweep's SELECT and its effects another connection can renew the lease or heartbeat, adopt
the worker of the same run, or claim a successor. Real SQLite, real entry points and producers on a
second connection; the process readers are a private map and the signal hook only records.
"""

import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_owner_recovery as recovery


@pytest.fixture
def world(tmp_path, monkeypatch):
    from gateway import drain_control, status
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd.time, "sleep", lambda _s: None)
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "boot-13f:42")
    procs = {12345: 100, 12346: 200}
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid in procs)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: procs.get(pid))
    conn, other = kbc.connect(tmp_path / "kanban.db"), kbc.connect(tmp_path / "kanban.db")
    tid = kb.create_task(conn, title="fence", assignee="worker", max_runtime_seconds=1)
    assert kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, 12345)
    _age(conn, tid)
    try:
        yield conn, other, tid, procs
    finally:
        conn.close()
        other.close()


def _age(conn, tid):
    old = int(time.time()) - 86400
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at=?, claim_expires=?, last_heartbeat_at=? WHERE id=?", (old, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at=?, claim_expires=?, last_heartbeat_at=? WHERE task_id=?",
                     (old, old, old, tid))


def _at_first_owner_look(monkeypatch, action):
    """Run ``action`` once, at the sweep's first owner-gate read: after its SELECT, before any signal."""
    real, done = recovery.owner_reclaim_blocked, []

    def gate(conn, task_id, **kwargs):
        if not done:
            done.append(action())
        return real(conn, task_id, **kwargs)
    monkeypatch.setattr(recovery, "owner_reclaim_blocked", gate)
    return done


def _sweep(path, conn, sig):
    if path == "ttl":
        return kb.release_stale_claims(conn, signal_fn=sig)
    return kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=sig)


@pytest.mark.parametrize("path,renewal", [("ttl", "heartbeat"), ("ttl", "lease"), ("stale", "heartbeat")])
def test_a_renewal_after_selection_cancels_the_signal(world, monkeypatch, path, renewal):
    """ID13E-CONC-03: the worker renewed before the effect, so the obsolete eligibility signals nothing
    and books nothing; the renewed custody is left exactly as the worker wrote it."""
    conn, other, tid, _procs = world
    run = kb.get_task(conn, tid).current_run_id

    def renew():
        if renewal == "heartbeat":
            return kbd.heartbeat_worker(other, tid, expected_run_id=run)
        return kb.heartbeat_claim(other, tid, ttl_seconds=3600)
    assert _at_first_owner_look(monkeypatch, renew) is not None
    signals = []
    assert not _sweep(path, conn, lambda pid, s: signals.append((pid, s)))
    task = kb.get_task(conn, tid)
    assert signals == [] and task.status == "running" and task.current_run_id == run and task.worker_pid == 12345
    assert task.consecutive_failures == 0


@pytest.mark.parametrize("path", ["ttl", "stale"])
def test_an_unrenewed_eligible_worker_is_still_terminated(world, path):
    """Control: with nothing renewed the ordinary eligible termination and reclaim still happen."""
    conn, _other, tid, procs = world
    signals = []

    def sig(pid, s):
        signals.append((pid, s))
        procs.pop(pid, None)
    assert _sweep(path, conn, sig)
    assert signals and kb.get_task(conn, tid).status == "ready"


def test_a_worker_adopted_after_stale_selection_keeps_its_run(world, monkeypatch):
    """ID13E-CONC-04: the stale sweep selected a run with no worker; the run's worker is adopted
    before the effect. The workerless snapshot must not end that run or clear the adopted worker."""
    conn, other, tid, procs = world
    procs[12345] = None  # live but unreadable: the operator release sends nothing
    assert kb.reclaim_task(other, tid, signal_fn=lambda *_: pytest.fail("signalled"))
    assert kb.claim_task(other, tid)
    _age(other, tid)
    run = kb.get_task(other, tid).current_run_id
    _at_first_owner_look(monkeypatch, lambda: kbd.adopt_worker_pid(other, tid, run, 12346))
    assert kbd.detect_stale_running(conn, stale_timeout_seconds=1,
                                    signal_fn=lambda *_: pytest.fail("signalled")) == []
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.worker_pid) == ("running", run, 12346)


@pytest.mark.parametrize("max_retries", [None, 1], ids=["default-limit", "first-failure"])
@pytest.mark.parametrize("path", ["timeout", "ttl"])
def test_old_failure_accounting_never_blocks_an_admitted_successor(world, monkeypatch, path, max_retries):
    """ID13E-CONC-05: the old attempt's failure is counted in its own later commit. A successor
    claimed in between is never flipped to ``blocked`` by it, and the old count is kept (1 -> 2).
    With ``max_retries=1`` the ending commit already parks the card, so no successor can be claimed
    and the first-failure hold and ``gave_up`` stand as before."""
    conn, other, tid, procs = world
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET consecutive_failures=1, max_retries=? WHERE id=?", (max_retries, tid))
    module = kbd if path == "timeout" else kb
    real, successor = module._record_task_failure, []

    def between_commits(*args, **kwargs):
        if kb.get_task(other, tid).status == "ready":
            assert kb.claim_task(other, tid, ttl_seconds=3600)
            kbd._set_worker_pid(other, tid, 12346)
            successor.append(kb.get_task(other, tid).current_run_id)
        return real(*args, **kwargs)
    monkeypatch.setattr(module, "_record_task_failure", between_commits)

    def sig(pid, s):
        procs.pop(pid, None)
    assert (kbd.enforce_max_runtime(conn, signal_fn=sig) if path == "timeout"
            else kb.release_stale_claims(conn, signal_fn=sig))
    task = kb.get_task(conn, tid)
    assert task.consecutive_failures == 2
    if max_retries is None:
        assert successor and (task.status, task.current_run_id, task.worker_pid) == ("running", successor[0], 12346)
        assert "gave_up" not in [e.kind for e in kb.list_events(conn, tid)]
    else:
        assert not successor and task.status == "blocked" and task.current_run_id is None
        assert "gave_up" in [e.kind for e in kb.list_events(conn, tid)]


@pytest.mark.parametrize("path", ["timeout", "ttl"])
def test_old_failure_accounting_still_trips_with_no_successor(world, path):
    """Control: with no successor the ordinary breaker trip at the default limit is unchanged."""
    conn, _other, tid, procs = world
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET consecutive_failures=1 WHERE id=?", (tid,))

    def sig(pid, s):
        procs.pop(pid, None)
    assert (kbd.enforce_max_runtime(conn, signal_fn=sig) if path == "timeout"
            else kb.release_stale_claims(conn, signal_fn=sig))
    task = kb.get_task(conn, tid)
    assert (task.status, task.consecutive_failures) == ("blocked", 2)
    assert "gave_up" in [e.kind for e in kb.list_events(conn, tid)]
