"""The dispatcher continues a READY card held by ``active_pr`` once its owner's run was interrupted.

A whole-outcome owner posts its PR URL on the card, then its run ends under it: the worker
crashes after a gateway receipt loss (2026-11-09, t_db6a2298 run 9342) or the owner's gateway
restarts (t_d2af4304 run 9302, t_02e9f8ff run 8962). The crash sweep re-queues the card to
``ready``, but ``check_respawn_guard`` then held it as ``active_pr`` on every tick until an
operator ran ``promote --expected-terminal-run``: an open PR read as a live owner. The dispatcher
now records that continuation itself (``continued``), but only on positive proof that nothing can
still write for the exact interrupted run; every other shape stays held, as before.

Real board SQLite and the real producers: ``claim_task``, ``_set_worker_pid``, the owner's
``owner_admitted``/``worker_bound`` events, ``record_owner_restart_result`` and the dead-worker
sweep ``detect_crashed_workers``; the selection path is ``dispatch_once`` with a ``spawn_fn``.

c2 (review t_878eb1fc SAFETY-C1-01): a counted failure is ended in one commit and counted in a
later one; the run carries ``failure_accounting: pending`` between them and ``settled`` after.
c3 (review t_af7f160f C2-MIGRATION-01, C2-ACCOUNT-01): only ``settled`` proves a counted end was
counted (no record, e.g. an older runtime's run, stays held), and a ``continued`` releases only
while its own interrupted run is the latest, so a later attempt is proved afresh. Within one
interrupted run, a same-second second dispatcher writes no second ``continued`` (C1-Q01).
c4 (review t_eccfd63f C3-RETRY-01): a later run the host refused before launch, proved by the
argv producer that runs before ``Popen``, does not end that release: the owner is retried after the
infrastructure cooldown; anything that may have run still ends it.
c5 (review t_cf892b2a C4-TIE-01, C4-FENCE-01): a same-second ``continued`` (held by the strict tie)
followed only by refused-before-launch runs is honoured as when it was written; and a dispatcher whose
claim was reclaimed and re-claimed before its spawn failure returns writes nothing over the newer claim.
c6 (review t_cf892b2a C5-WORKSPACE-01): the same stale dispatcher whose workspace resolves successfully
publishes no workspace path, branch or scratch tip over the newer claim and spawns nothing.
c7 (review t_cf892b2a, c5 final precision): the tie is also witnessed as it arises, in real chronological
order on one frozen second before the first claim and refusal, in both orders of PR comment and ``continued``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from gateway.session_kanban import record_owner_restart_result
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_active_pr as kap
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "DRAFT PR OPEN: https://github.com/example/repo/pull/597 (head t-1/ops)."


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _admission(tmp_path: Path, status: str = "terminal", outcome: str = "interrupted") -> dict:
    """The owner's ``session_admissions`` row an ``owner_admitted`` payload names."""
    db = tmp_path / f"owner-{len(list(tmp_path.glob('owner-*.db')))}.db"
    with sqlite3.connect(db) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, "
                      "status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('s1', 'r1', ?, ?)", (status, outcome))
    return {"db": str(db), "session_id": "s1", "request_id": "r1", "pid": _dead_pid(), "birth": 1.0}


def _set_admission(binding: dict, status: str, outcome) -> None:
    with sqlite3.connect(binding["db"]) as owner:
        owner.execute("UPDATE session_admissions SET status=?, outcome=?", (status, outcome))


def _owner_run(conn, tmp_path: Path, tid: str, *, admitted=True, comment=True) -> tuple[int, str, dict | None]:
    """Claim ``tid`` on this host (or take the dispatcher's claim), admit it to an owner (later
    settled ``terminal/interrupted``), bind a since-dead interpreter and, during the run, post the
    owner's PR comment."""
    task = kb.get_task(conn, tid) if kb.get_task(conn, tid).status == "running" else kb.claim_task(conn, tid)
    run_id, claim = task.current_run_id, task.claim_lock
    kbd._set_worker_pid(conn, tid, _dead_pid())
    binding = _admission(tmp_path, status="running", outcome=None) if admitted else None
    with kb.write_txn(conn):
        if binding is not None:
            kb._append_event(conn, tid, "owner_admitted", binding, run_id=run_id)
        kb._append_event(conn, tid, "worker_bound", {"pid": _dead_pid(), "claim_lock": claim, "started_at": "x"},
                         run_id=run_id)
    if comment:
        kb.add_comment(conn, tid, author="theseus-implementer", body=PR_COMMENT)
        with kb.write_txn(conn):  # second-granularity stamps: the PR comment precedes the crash
            conn.execute("UPDATE task_comments SET created_at = created_at - 60 WHERE task_id = ?", (tid,))
    if binding is not None:
        _set_admission(binding, "terminal", "interrupted")
    return run_id, claim, binding


def _crashed_owner(conn, tmp_path: Path, title="crashed PR owner", max_retries=None, **kw) -> tuple[str, int, dict | None]:
    tid = kb.create_task(conn, title=title, assignee="theseus-implementer", max_retries=max_retries)
    run_id, _claim, binding = _owner_run(conn, tmp_path, tid, **kw)
    assert kbd.detect_crashed_workers(conn) == [tid]
    return tid, run_id, binding


def _restarted_owner(conn, tmp_path: Path, title="restart casualty") -> tuple[str, int]:
    tid = kb.create_task(conn, title=title, assignee="theseus-implementer")
    run_id, _claim, _binding = _owner_run(conn, tmp_path, tid)
    board = conn.execute("PRAGMA database_list").fetchone()[2]
    assert record_owner_restart_result({"board": "default", "task_id": tid, "run_id": run_id}, board)
    assert kbd.detect_crashed_workers(conn) == []
    return tid, run_id


def _continued(conn, tid) -> list[dict]:
    rows = conn.execute("SELECT payload, run_id FROM task_events WHERE task_id=? AND kind='continued' ORDER BY id",
                        (tid,)).fetchall()
    return [dict(json.loads(r["payload"]), event_run_id=r["run_id"]) for r in rows]


def _spawned(conn, *, dry_run=False) -> list[str]:
    res = kbd.dispatch_once(conn, dry_run=dry_run, spawn_fn=lambda task, workspace: None)
    return [s[0] for s in res.spawned]


def _held(conn, tid) -> bool:
    """A real dispatch tick leaves ``tid`` unclaimed in ``ready`` (and writes no continuation)."""
    res = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None)
    return tid not in [s[0] for s in res.spawned] and kb.get_task(conn, tid).status == "ready"


@pytest.mark.parametrize("interruption", ["crashed", "owner_restarted"])
def test_interrupted_pr_owner_is_continued_by_dispatch(conn, tmp_path: Path, interruption: str) -> None:
    if interruption == "crashed":
        tid, run_id, _binding = _crashed_owner(conn, tmp_path)
    else:
        tid, run_id = _restarted_owner(conn, tmp_path)
    run = kb.latest_run(conn, tid)
    assert (run.id, run.outcome, kb.get_task(conn, tid).status) == (run_id, interruption, "ready")
    # The guard still names the hold; only the dispatcher's proven continuation lifts it.
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    comments = len(kb.list_comments(conn, tid))
    events = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]

    assert tid in _spawned(conn, dry_run=True)
    assert _continued(conn, tid) == []  # the dry run wrote nothing
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0] == events

    assert tid in _spawned(conn)

    (event,) = _continued(conn, tid)
    assert event == {"from": "ready", "continuation": True, "interrupted_run": run_id, "outcome": interruption,
                     "event_run_id": run_id}
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee) == ("running", "theseus-implementer")  # the same owner continues
    assert kb.latest_run(conn, tid).id > run_id  # a new attempt; the interrupted one is untouched
    assert conn.execute("SELECT outcome FROM task_runs WHERE id=?", (run_id,)).fetchone()[0] == interruption
    assert len(kb.list_comments(conn, tid)) == comments  # the PR and every return stay on the card


def _live_claim(conn, tid, binding, tmp_path):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock='lealwit-elsewhere:1' WHERE id=?", (tid,))


def _owner_still_running(conn, tid, binding, tmp_path):
    _set_admission(binding, "running", None)


def _admission_unreadable(conn, tid, binding, tmp_path):
    Path(binding["db"]).unlink()


def _second_live_writer(conn, tid, binding, tmp_path):
    """Another binding of the same run whose interpreter is still alive (this test process)."""
    run = kb.latest_run(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {
            "pid": os.getpid(), "claim_lock": run.claim_lock, "started_at": kbd._process_fingerprint(os.getpid())},
            run_id=run.id)


def _foreign_host_binding(conn, tid, binding, tmp_path):
    run = kb.latest_run(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": _dead_pid(), "claim_lock": "other-host:42",
                                                     "started_at": "x"}, run_id=run.id)


def _open_older_run(conn, tid, binding, tmp_path):
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at=NULL WHERE task_id=? AND id<(SELECT MAX(id) FROM task_runs "
                     "WHERE task_id=?)", (tid, tid))


@pytest.mark.parametrize("hazard", [_live_claim, _owner_still_running, _admission_unreadable, _second_live_writer,
                                    _foreign_host_binding])
def test_unproven_interruption_stays_held_and_writes_nothing(conn, tmp_path: Path, hazard) -> None:
    tid, _run_id, binding = _crashed_owner(conn, tmp_path, title=hazard.__name__)
    hazard(conn, tid, binding, tmp_path)

    assert _held(conn, tid)
    assert _held(conn, tid)  # every later tick too
    assert _continued(conn, tid) == []
    assert kb.latest_run(conn, tid).id == _run_id  # no new attempt
    assert kap.interrupted_owner_refusal(conn, tid) is not None


def test_an_older_open_run_stays_held(conn, tmp_path: Path) -> None:
    tid, _first, _binding = _crashed_owner(conn, tmp_path, title="two runs", max_retries=5)
    _owner_run(conn, tmp_path, tid, comment=True)  # a newer interrupted attempt; the older one reopened below
    assert kbd.detect_crashed_workers(conn) == [tid]
    _open_older_run(conn, tid, None, tmp_path)
    assert _held(conn, tid) and _continued(conn, tid) == []
    assert kap.interrupted_owner_refusal(conn, tid) == f"{tid} has an open run"


def test_unmanaged_run_without_owner_admission_stays_held(conn, tmp_path: Path) -> None:
    tid, _run_id, _binding = _crashed_owner(conn, tmp_path, admitted=False)
    assert _held(conn, tid) and _continued(conn, tid) == []
    assert "no owner admission" in kap.interrupted_owner_refusal(conn, tid)


def test_owner_verdict_is_not_an_interruption(conn, tmp_path: Path) -> None:
    """A run the owner ended itself (here ``blocked``, then the hold auto-recovered by the breaker
    shape a ``from: blocked`` promotion records) is no interruption: it stays held."""
    tid = kb.create_task(conn, title="owner blocked itself", assignee="theseus-implementer")
    _owner_run(conn, tmp_path, tid)
    assert kb.block_task(conn, tid, kind="transient", reason="waiting")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb._append_event(conn, tid, "promoted", {"from": "blocked"})
    assert kb.latest_run(conn, tid).outcome == "blocked"
    assert _held(conn, tid) and _continued(conn, tid) == []
    assert "not interrupted" in kap.interrupted_owner_refusal(conn, tid)


def test_continuation_is_one_release_and_a_newer_pr_guards_again(conn, tmp_path: Path) -> None:
    tid, run_id, _binding = _crashed_owner(conn, tmp_path, max_retries=5)
    assert kap.continue_interrupted(conn, tid, dry_run=False)
    assert kap.continue_interrupted(conn, tid, dry_run=False)  # a second dispatcher honours the first release
    assert len(_continued(conn, tid)) == 1
    assert kbd.check_respawn_guard(conn, tid) is None

    # The continued owner runs, posts a newer PR comment and is interrupted again: that comment
    # guards again, and only a fresh proof on the NEW run continues it once more.
    with kb.write_txn(conn):  # second-granularity stamps: release < newer PR comment < next crash
        conn.execute("UPDATE task_events SET created_at = created_at - 120 WHERE task_id = ?", (tid,))
        conn.execute("UPDATE task_comments SET created_at = created_at - 120 WHERE task_id = ?", (tid,))
    _owner_run(conn, tmp_path, tid, comment=False)
    kb.add_comment(conn, tid, author="theseus-implementer", body="Rebased: https://github.com/example/repo/pull/598")
    with kb.write_txn(conn):
        conn.execute("UPDATE task_comments SET created_at = created_at - 30 WHERE task_id = ? AND body LIKE '%598%'",
                     (tid,))
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert tid in _spawned(conn)
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [run_id, run_id + 1]


def test_repeated_crashes_stay_bounded_by_the_breaker(conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, _run_id, _binding = _crashed_owner(conn, tmp_path, max_retries=2)
    assert tid in _spawned(conn)
    _owner_run(conn, tmp_path, tid, comment=False)
    assert kbd.detect_crashed_workers(conn) == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "blocked" and task.consecutive_failures == 2  # the breaker parks it
    assert tid not in _spawned(conn)


def _reopened(conn):
    """A fresh connection to the same board: nothing in memory survives a dispatcher's loss."""
    fresh = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2], isolation_level=None)
    fresh.row_factory = sqlite3.Row
    return fresh


def _accounting(conn, run_id):
    return json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()[0] or "{}"
                      ).get("failure_accounting")


def _second_attempt_selected_by_dispatch(conn, tmp_path: Path, max_retries) -> str:
    """The first crash (before any PR) is counted in full and the card respawned by real dispatch."""
    tid = kb.create_task(conn, title="lost count", assignee="theseus-implementer", max_retries=max_retries)
    first, _claim, _binding = _owner_run(conn, tmp_path, tid, comment=False)
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kb.get_task(conn, tid).consecutive_failures == 1 and _accounting(conn, first) == "settled"
    assert _spawned(conn) == [tid]
    return tid


@pytest.mark.parametrize("max_retries, prior", [(2, 1), (None, 1), (None, 0), (3, 1)])
def test_dispatcher_lost_before_counting_a_crash_never_starts_another_attempt(
        conn, tmp_path: Path, max_retries, prior: int) -> None:
    """SAFETY-C1-01: ``_reclaim_dead_workers`` commits (run ended, card ``ready``) and the
    dispatcher dies before ``_account_crashes``; the next dispatcher, on a fresh connection, must
    not start another attempt while that failure is uncounted, on any tick."""
    if prior:
        tid = _second_attempt_selected_by_dispatch(conn, tmp_path, max_retries)
    else:
        tid = kb.create_task(conn, title="lost first count", assignee="theseus-implementer", max_retries=max_retries)
    run_id, _claim, _binding = _owner_run(conn, tmp_path, tid)  # posts the PR comment during this attempt
    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.crashed == [tid] and kb.latest_run(conn, tid).outcome == "crashed"
    fresh = _reopened(conn)
    try:
        for _tick in range(2):
            assert _held(fresh, tid)
            assert kbd.detect_crashed_workers(fresh) == []  # the claim is gone: nothing re-finds the run
        assert _continued(fresh, tid) == [] and kb.latest_run(fresh, tid).id == run_id
        assert kb.get_task(fresh, tid).consecutive_failures == prior
        assert "accounting is pending" in kap.interrupted_owner_refusal(fresh, tid)
        assert _accounting(fresh, run_id) == "pending"
    finally:
        fresh.close()

    # Had the accounting commit landed (the sweep's own record): counted once, the mark settled.
    tripped = kbd._account_crashes(conn, sweep.crash_details)
    task = kb.get_task(conn, tid)
    assert task.consecutive_failures == prior + 1 and _accounting(conn, run_id) == "settled"
    limit = max_retries or kbd.DEFAULT_FAILURE_LIMIT
    if prior + 1 >= limit:  # that failure exhausts the budget: parked, never another attempt
        assert tripped == [tid] and task.status == "blocked"
        assert tid not in _spawned(conn) and tid not in _spawned(conn)
    else:
        assert tripped == [] and _spawned(conn) == [tid]
        assert [e["interrupted_run"] for e in _continued(conn, tid)] == [run_id]
    assert kb.get_task(conn, tid).consecutive_failures == prior + 1


def test_selected_first_failure_lost_before_counting_stays_parked(conn, tmp_path: Path) -> None:
    tid = kb.create_task(conn, title="first failure", assignee="theseus-implementer", max_retries=1)
    run_id, _claim, _binding = _owner_run(conn, tmp_path, tid)
    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.crashed == [tid]
    fresh = _reopened(conn)
    try:
        for _tick in range(2):
            assert tid not in _spawned(fresh) and kb.get_task(fresh, tid).status == "blocked"
        assert _continued(fresh, tid) == [] and kb.get_task(fresh, tid).consecutive_failures == 0
        assert _accounting(fresh, run_id) == "pending"
    finally:
        fresh.close()
    assert kbd._account_crashes(conn, sweep.crash_details) == [tid]
    assert kb.get_task(conn, tid).consecutive_failures == 1 and _accounting(conn, run_id) == "settled"
    assert tid not in _spawned(conn) and kb.get_task(conn, tid).status == "blocked"


def test_uncounted_ends_owe_no_accounting(conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``owner_restarted`` and a rate-limited exit never count a failure, so nothing is pending:
    the restart casualty is continued; the quota wall keeps its own cooldown."""
    tid, run_id = _restarted_owner(conn, tmp_path)
    assert _accounting(conn, run_id) is None and kb.get_task(conn, tid).consecutive_failures == 0
    assert tid in _spawned(conn)

    limited = kb.create_task(conn, title="quota wall", assignee="theseus-implementer")
    limited_run, _claim, _binding = _owner_run(conn, tmp_path, limited)
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.rate_limited == [limited] and sweep.crash_details == []
    assert kb.latest_run(conn, limited).outcome == "rate_limited" and _accounting(conn, limited_run) is None
    assert kb.get_task(conn, limited).consecutive_failures == 0
    assert _held(conn, limited) and _continued(conn, limited) == []


def test_below_budget_protocol_violation_is_settled_uncounted(conn, tmp_path: Path, monkeypatch) -> None:
    tid = kb.create_task(conn, title="clean exit while running", assignee="theseus-implementer")
    run_id, _claim, _binding = _owner_run(conn, tmp_path, tid)
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("clean_exit", 0))
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kb.get_task(conn, tid).consecutive_failures == 0 and _accounting(conn, run_id) == "settled"
    assert tid in _spawned(conn)  # its accounting is complete, so the owner is continued


@pytest.mark.parametrize("order", ["comment_first", "continued_first"])
def test_same_second_release_is_written_once(conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             order: str) -> None:
    """C1-Q01: a ``continued`` in the same second as the newest PR comment, in either order, does
    not lift the guard (strict-after tie, fail closed, never ``>=``); a second dispatcher in that
    second must not append another ``continued`` for the run."""
    tid, run_id, _binding = _crashed_owner(conn, tmp_path)
    stamp = int(kbd.time.time())
    monkeypatch.setattr(kbd.time, "time", lambda: stamp)
    monkeypatch.setattr(kb.time, "time", lambda: stamp)
    if order == "comment_first":
        with kb.write_txn(conn):
            conn.execute("UPDATE task_comments SET created_at=? WHERE task_id=?", (stamp, tid))
        assert kap.continue_interrupted(conn, tid, dry_run=False)
    else:
        assert kap.continue_interrupted(conn, tid, dry_run=False)  # the older comment is released...
        assert kbd.check_respawn_guard(conn, tid) is None
        kb.add_comment(conn, tid, author="operator", body="Same PR: https://github.com/example/repo/pull/597")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"  # the tie stays reported
    assert kap.continue_interrupted(conn, tid, dry_run=False)
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [run_id]


def test_strictly_newer_comment_gets_a_fresh_continuation_record(conn, tmp_path: Path) -> None:
    tid, run_id, _binding = _crashed_owner(conn, tmp_path)
    assert kap.continue_interrupted(conn, tid, dry_run=False)
    kb.add_comment(conn, tid, author="operator", body="Same PR: https://github.com/example/repo/pull/597")
    with kb.write_txn(conn):  # second-granularity stamps: that release < the newer comment < now
        conn.execute("UPDATE task_events SET created_at = created_at - 120 WHERE task_id = ?", (tid,))
        conn.execute("UPDATE task_comments SET created_at = created_at - 60 WHERE task_id = ? AND author = 'operator'",
                     (tid,))
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert kap.continue_interrupted(conn, tid, dry_run=False)
    assert kbd.check_respawn_guard(conn, tid) is None
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [run_id, run_id]


@pytest.mark.parametrize("outcome", ["timed_out", "reclaimed"])
def test_lost_timeout_or_stale_reclaim_count_stays_held(conn, tmp_path: Path, monkeypatch, outcome: str) -> None:
    """The timeout and stale-claim reclaim commit their end first and count it in their own
    later ``_record_task_failure`` commit, exactly like the crash sweep. (They reclaim an owner
    run only once its admission is settled other than ``interrupted``; that pause is theirs.)"""
    tid = kb.create_task(conn, title=outcome, assignee="theseus-implementer", max_runtime_seconds=1)
    run_id, _claim, binding = _owner_run(conn, tmp_path, tid)
    _set_admission(binding, "terminal", "completed")
    def lost(*_args, **_kwargs) -> bool:  # the dispatcher dies before the counting commit
        return False

    with monkeypatch.context() as patch:
        if outcome == "timed_out":
            with kb.write_txn(conn):
                conn.execute("UPDATE task_runs SET started_at = started_at - 30 WHERE id = ?", (run_id,))
            patch.setattr(kbd, "_record_task_failure", lost)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None) == [tid]
        else:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(kbd.time.time()) - 1, tid))
            patch.setattr(kb, "_record_task_failure", lost)
            assert kb.release_stale_claims(conn) == 1
    assert kb.latest_run(conn, tid).outcome == outcome
    fresh = _reopened(conn)
    try:
        assert _held(fresh, tid) and _held(fresh, tid) and _continued(fresh, tid) == []
        assert kb.get_task(fresh, tid).consecutive_failures == 0
        assert "accounting is pending" in kap.interrupted_owner_refusal(fresh, tid)
        assert _accounting(fresh, run_id) == "pending"
    finally:
        fresh.close()


def _strip_accounting(conn, run_id) -> None:
    """The run as a runtime that never marked accounting left it: a counted end with no record,
    whether or not its failure was counted (the two cannot be told apart)."""
    data = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()[0] or "{}")
    data.pop("failure_accounting", None)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(data, ensure_ascii=False), run_id))


def _dispatch_from(monkeypatch, tmp_path: Path, conn, other_home: bool):
    """A fresh dispatcher connection to the same board, in this home or another home sharing it."""
    if other_home:
        home = tmp_path / "other-home" / ".hermes"
        home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
        monkeypatch.setenv("HERMES_KANBAN_DB", conn.execute("PRAGMA database_list").fetchone()[2])
        monkeypatch.setattr(Path, "home", lambda: home.parent)
    return _reopened(conn)


@pytest.mark.parametrize("other_home", [False, True])
def test_unmarked_counted_crash_stays_held_for_the_operator_continuation(
        conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_home: bool) -> None:
    """C2-MIGRATION-01: a second crash ended by a runtime that never marked accounting, with no
    record whether it was counted, is not continued automatically; the explicit operator
    continuation still resumes it. (The reviewer's probe replays the real M9 producer.)"""
    tid = _second_attempt_selected_by_dispatch(conn, tmp_path, max_retries=2)
    run_id, _claim, _binding = _owner_run(conn, tmp_path, tid)
    assert kbd._reclaim_dead_workers(conn).crashed == [tid]  # ...and its count never lands
    _strip_accounting(conn, run_id)
    fresh = _dispatch_from(monkeypatch, tmp_path, conn, other_home)
    try:
        for _tick in range(2):
            assert _held(fresh, tid)
        assert _continued(fresh, tid) == [] and kb.latest_run(fresh, tid).id == run_id
        assert kb.get_task(fresh, tid).consecutive_failures == 1
        assert "no settled accounting record (none)" in kap.interrupted_owner_refusal(fresh, tid)
    finally:
        fresh.close()
    ok, reason = kb.promote_task(conn, tid, actor="operator", reason="count reconciled", expected_terminal_run=run_id)
    assert ok, reason
    assert _spawned(conn) == [tid] and _continued(conn, tid) == []


def test_unmarked_owner_restart_is_still_continued(conn, tmp_path: Path) -> None:
    """``owner_restarted`` is uncounted by every producer, old or new: no record is needed."""
    tid, run_id = _restarted_owner(conn, tmp_path)
    _strip_accounting(conn, run_id)
    assert kap.interrupted_owner_refusal(conn, tid) is None
    assert _spawned(conn) == [tid]
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [run_id]


def test_unreadable_accounting_record_stays_held(conn, tmp_path: Path) -> None:
    tid, run_id = _restarted_owner(conn, tmp_path)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET metadata='{not json' WHERE id=?", (run_id,))
    assert _held(conn, tid) and _continued(conn, tid) == []
    assert "(unreadable)" in kap.interrupted_owner_refusal(conn, tid)


@pytest.mark.parametrize("max_retries", [2, 3])
@pytest.mark.parametrize("other_home", [False, True])
def test_an_earlier_automatic_continuation_does_not_cover_a_later_attempt(
        conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_home: bool, max_retries: int) -> None:
    """C2-ACCOUNT-01: run 1 is legitimately continued; run 2 posts no newer PR comment and its
    dispatcher is lost between ending and counting it. A fresh dispatcher (same home, or another
    home on the shared board) must not let run 1's proof launch run 3 past the budget."""
    tid, first, _binding = _crashed_owner(conn, tmp_path, max_retries=max_retries)
    assert _spawned(conn) == [tid] and [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    second, _claim, _binding = _owner_run(conn, tmp_path, tid, comment=False)
    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.crashed == [tid] and kb.latest_run(conn, tid).id == second
    fresh = _dispatch_from(monkeypatch, tmp_path, conn, other_home)
    try:
        assert kbd.check_respawn_guard(fresh, tid) == "active_pr"  # run 1's release no longer covers
        for _tick in range(2):
            assert _held(fresh, tid)
        assert kb.latest_run(fresh, tid).id == second and kb.get_task(fresh, tid).consecutive_failures == 1
        assert "accounting is pending" in kap.interrupted_owner_refusal(fresh, tid)
    finally:
        fresh.close()

    tripped = kbd._account_crashes(conn, sweep.crash_details)  # the count, had it landed
    assert kb.get_task(conn, tid).consecutive_failures == 2 and _accounting(conn, second) == "settled"
    if max_retries == 2:
        assert tripped == [tid] and kb.get_task(conn, tid).status == "blocked"
        assert tid not in _spawned(conn)
    else:  # proved afresh on run 2 itself
        assert tripped == [] and _spawned(conn) == [tid]
        assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first, second]


def test_an_explicit_operator_release_still_covers_later_retries(conn, tmp_path: Path) -> None:
    """The operator continuation keeps its persistent meaning (APR-02): later breaker-bounded
    retries need no new release until a newer PR comment."""
    tid, first, _binding = _crashed_owner(conn, tmp_path, max_retries=3)
    ok, reason = kb.promote_task(conn, tid, actor="operator", reason="resume", expected_terminal_run=first)
    assert ok, reason
    assert _spawned(conn) == [tid]
    _owner_run(conn, tmp_path, tid, comment=False)
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kbd.check_respawn_guard(conn, tid) is None
    assert _spawned(conn) == [tid] and _continued(conn, tid) == []


def _host_cannot_place_workers(monkeypatch) -> None:
    """A managed gateway with no user systemd bus: the real restart-safe scope helper refuses."""
    from tools import process_registry
    monkeypatch.setenv("INVOCATION_ID", "synthetic-managed-gateway")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)

    def never(*_args, **_kwargs):
        raise AssertionError("a host refusal must come before any worker launch")
    monkeypatch.setattr(kbd.subprocess, "Popen", never)


def _after_cooldown(monkeypatch, ticks: int) -> None:
    stamp = int(time.time()) + ticks * (kb._resolve_rate_limit_cooldown_seconds() + 1)
    monkeypatch.setattr(kbd.time, "time", lambda: stamp)


def _refused_by_the_host(conn, monkeypatch, tid, tick: int) -> int:
    """One real dispatch through ``_default_spawn``, refused by the host before launch."""
    with monkeypatch.context() as patch:
        _host_cannot_place_workers(patch)
        if tick:
            _after_cooldown(patch, tick)
        result = kbd.dispatch_once(conn)
        assert not result.spawned
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"
    run = kb.latest_run(conn, tid)
    assert run.outcome == "spawn_failed" and run.metadata["launch"] == "refused_before_launch"
    assert kb.get_task(conn, tid).consecutive_failures == 0 and kb.get_task(conn, tid).status == "ready"
    return run.id


def test_a_host_refusal_before_launch_keeps_the_continuation(conn, tmp_path: Path, monkeypatch) -> None:
    """C3-RETRY-01: run 1 is continued; the host refuses runs 2 and 3 before any worker starts
    (real ``_default_spawn`` → ``_restart_safe_worker_argv``). Nothing ran, so run 1's proof still
    stands: once the infrastructure cooldown passes and the host recovers, the owner is retried."""
    tid, first = _restarted_owner(conn, tmp_path)
    second = _refused_by_the_host(conn, monkeypatch, tid, tick=0)
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    assert kap.never_launched(conn, tid, conn.execute("SELECT * FROM task_runs WHERE id=?", (second,)).fetchone())
    third = _refused_by_the_host(conn, monkeypatch, tid, tick=1)  # still refused: spaced, uncharged
    assert third == second + 1
    _after_cooldown(monkeypatch, 2)
    assert kbd.check_respawn_guard(conn, tid) is None
    assert _spawned(conn) == [tid]
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    assert kb.get_task(conn, tid).consecutive_failures == 0


def _admitted(conn, tid, run_id):
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "owner_admitted", {"db": "x", "session_id": "s", "request_id": "r"}, run_id=run_id)


def _bound(conn, tid, run_id):
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": 1, "claim_lock": "x", "started_at": "x"}, run_id=run_id)


def _with_pid(conn, tid, run_id):
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET worker_pid=12345 WHERE id=?", (run_id,))


def _bare_infrastructure_label(conn, tid, run_id):
    data = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()[0])
    del data["launch"]
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(data), run_id))


@pytest.mark.parametrize("evidence", [_admitted, _bound, _with_pid, _bare_infrastructure_label])
def test_a_refused_attempt_that_may_have_run_ends_the_continuation(conn, tmp_path: Path, monkeypatch, evidence) -> None:
    """Any sign the later attempt began (an owner admission, a bound worker, a pid) or a failure the
    launch boundary did not prove pre-launch leaves the card held for an operator, as in c3."""
    tid, _first = _restarted_owner(conn, tmp_path)
    second = _refused_by_the_host(conn, monkeypatch, tid, tick=0)
    evidence(conn, tid, second)
    _after_cooldown(monkeypatch, 1)
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert _held(conn, tid) and _held(conn, tid)
    assert "ended 'spawn_failed', not interrupted" in kap.interrupted_owner_refusal(conn, tid)


def test_an_untagged_host_refusal_is_not_proof_of_no_launch(conn, tmp_path: Path, monkeypatch) -> None:
    """The exception type alone is not the proof: only the argv producer, which runs before
    ``Popen``, tags the refused run. A spawn path raising it from anywhere else stays held."""
    from tools.process_registry import RestartSafeScopeUnavailable
    tid, _first = _restarted_owner(conn, tmp_path)

    def refused_elsewhere(task, workspace):
        raise RestartSafeScopeUnavailable("raised outside the argv producer")
    assert not kbd.dispatch_once(conn, spawn_fn=refused_elsewhere).spawned
    run = kb.latest_run(conn, tid)
    assert run.metadata.get("infrastructure") is True and "launch" not in run.metadata
    _after_cooldown(monkeypatch, 1)
    assert _held(conn, tid) and kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_an_ordinary_spawn_failure_ends_the_continuation(conn, tmp_path: Path, monkeypatch) -> None:
    tid, _first = _restarted_owner(conn, tmp_path)

    def broken(task, workspace):
        raise RuntimeError("spawn failed for an unknown reason")
    assert not kbd.dispatch_once(conn, spawn_fn=broken).spawned
    assert kb.get_task(conn, tid).consecutive_failures == 1
    _after_cooldown(monkeypatch, 1)
    assert _held(conn, tid) and kbd.check_respawn_guard(conn, tid) == "active_pr"


def _tie_with_the_pr_comment(conn, tid) -> int:
    """Stamp the card's PR comments in the very second of its ``continued`` (the strict tie)."""
    with kb.write_txn(conn):
        stamp = conn.execute("SELECT created_at FROM task_events WHERE task_id=? AND kind='continued'", (tid,)).fetchone()[0]
        conn.execute("UPDATE task_comments SET created_at=? WHERE task_id=?", (stamp, tid))
    return stamp


def test_a_same_second_continuation_survives_a_host_refusal(conn, tmp_path: Path, monkeypatch) -> None:
    """C4-TIE-01: run 1's ``continued`` ties the newest PR comment, so ``released_after`` keeps reporting
    ``active_pr`` and only the tie handling let run 2 be claimed. The host refuses run 2 before launch;
    after the cooldown run 2 (``spawn_failed``) is no interruption to prove, yet nothing has run since
    run 1's proof: the owner is retried now (c4 held it until another release or until the PR comment
    left the guard window), and no second ``continued`` is written."""
    tid, first = _restarted_owner(conn, tmp_path)
    second = _refused_by_the_host(conn, monkeypatch, tid, tick=0)
    _tie_with_the_pr_comment(conn, tid)
    _after_cooldown(monkeypatch, 1)
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"  # the tie stays reported
    assert kap.interrupted_owner_refusal(conn, tid) is not None  # run 2 is no interruption
    assert _spawned(conn) == [tid]
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    assert kb.latest_run(conn, tid).id == second + 1


@pytest.mark.parametrize("order", ["comment_then_continued", "continued_then_comment"])
def test_a_tie_arising_before_the_first_refusal_is_retried_after_the_cooldown(conn, tmp_path: Path, monkeypatch,
                                                                                 order: str) -> None:
    """C4-TIE-01 in real order, nothing restamped afterwards: on one frozen second the owner posts a newer PR
    comment and the dispatcher writes run 1's ``continued`` (either first: the comment before the tick, or
    between the continuation and the claim of run 2), then claims run 2 and the host refuses it before launch.
    After the cooldown the tie still reads ``active_pr``, yet the owner is retried, with no second release."""
    tid, first = _restarted_owner(conn, tmp_path)
    stamp = int(time.time()) + 5  # one second for the whole first tick, later than the owner's run
    with monkeypatch.context() as patch:
        patch.setattr(time, "time", lambda: stamp)
        _host_cannot_place_workers(patch)
        if order == "comment_then_continued":
            kb.add_comment(conn, tid, author="theseus-implementer", body=PR_COMMENT)
        else:
            claim_task = kb.claim_task

            def comment_then_claim(*args, **kwargs):
                if _continued(conn, tid) and conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id=?",
                                                          (tid,)).fetchone()[0] == 1:
                    kb.add_comment(conn, tid, author="theseus-implementer", body=PR_COMMENT)
                return claim_task(*args, **kwargs)
            patch.setattr(kb, "claim_task", comment_then_claim)
        assert not kbd.dispatch_once(conn).spawned
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"
    second = kb.latest_run(conn, tid)
    assert second.outcome == "spawn_failed" and second.metadata["launch"] == "refused_before_launch"
    released = conn.execute("SELECT id, created_at FROM task_events WHERE task_id=? AND kind='continued'",
                            (tid,)).fetchall()
    comments = conn.execute("SELECT id, created_at FROM task_comments WHERE task_id=? ORDER BY id", (tid,)).fetchall()
    assert len(released) == 1 and len(comments) == 2
    assert released[0]["created_at"] == comments[1]["created_at"] == stamp  # the tie, as written
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    # The cooldown runs from the refusal's own (frozen) second, not from the restored clock.
    after = stamp + kb._resolve_rate_limit_cooldown_seconds() + 1
    monkeypatch.setattr(kbd.time, "time", lambda: after)
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"  # the tie stays reported
    assert _spawned(conn) == [tid]
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]
    assert kb.latest_run(conn, tid).id == second.id + 1 and kb.get_task(conn, tid).consecutive_failures == 0


@pytest.mark.parametrize("later", ["newer_pr_comment", "run_admitted", "run_bound"])
def test_a_same_second_continuation_is_not_stretched(conn, tmp_path: Path, monkeypatch, later: str) -> None:
    """The tie handling covers only runs refused before launch, and only while no comment is newer."""
    tid, first = _restarted_owner(conn, tmp_path)
    second = _refused_by_the_host(conn, monkeypatch, tid, tick=0)
    stamp = _tie_with_the_pr_comment(conn, tid)
    if later == "run_admitted":
        _admitted(conn, tid, second)
    elif later == "run_bound":
        _bound(conn, tid, second)
    else:
        kb.add_comment(conn, tid, author="operator", body="Same PR: https://github.com/example/repo/pull/597")
        with kb.write_txn(conn):
            conn.execute("UPDATE task_comments SET created_at=? WHERE task_id=? AND author='operator'", (stamp + 1, tid))
    _after_cooldown(monkeypatch, 1)
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert _spawned(conn) == []
    assert [e["interrupted_run"] for e in _continued(conn, tid)] == [first]


def _card_state(conn, tid) -> tuple:
    task = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    runs = [dict(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,))]
    events = [dict(r) for r in conn.execute("SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]
    return task, runs, events


def _taken_over_by_b(conn, tid, seen: dict) -> None:
    """While dispatcher A stalls, the independent supported routes run (not a second dispatch tick, which
    the board lock serializes): the stale-claim sweep ``release_stale_claims`` reclaims A's run and
    claimer B takes the next with ``claim_task``. ``seen`` keeps A's claim, B's and B's card state."""
    a = kb.get_task(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 1, tid))
    assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1
    b = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:dispatcher-b")
    assert b is not None and b.current_run_id != a.current_run_id
    seen.update(a=a, a_run=a.current_run_id, b=b, before=_card_state(conn, tid))


@pytest.mark.parametrize("failure", ["host_refusal", "workspace"])
def test_a_stale_spawn_failure_never_touches_the_newer_claim(conn, tmp_path: Path, monkeypatch, failure: str) -> None:
    """C4-FENCE-01, normal path: dispatcher A claims run 1 and stalls past its claim TTL before its spawn
    fails; meanwhile the independent supported routes run (not a second dispatch tick, which the board
    lock serializes): the stale-claim sweep ``release_stale_claims`` reclaims run 1 and claimer B takes
    run 2 with ``claim_task``. A's failure (a real host refusal from the argv producer, tagged run 1; or
    a workspace failure) must change nothing: B's task row, runs, events and accounting stay exactly as
    B left them (c4 ended run 2, cleared B's claim and restored the phase), and B's spawn still lands."""
    from tools import process_registry
    tid = kb.create_task(conn, title="re-claimed while A stalled", assignee="theseus-implementer")
    seen: dict = {}

    def b_takes_over() -> None:
        _taken_over_by_b(conn, tid, seen)

    if failure == "host_refusal":
        _host_cannot_place_workers(monkeypatch)
        real = process_registry.restart_safe_gateway_child_argv

        def stalled(*args, **kwargs):
            b_takes_over()
            return real(*args, **kwargs)  # the host refuses: RestartSafeScopeUnavailable, tagged run 1
        monkeypatch.setattr(process_registry, "restart_safe_gateway_child_argv", stalled)
        result = kbd.dispatch_once(conn)
    else:
        def stalled_workspace(claimed, board=None):
            b_takes_over()
            raise OSError("workspace unavailable")
        monkeypatch.setattr(kbd._kbw, "resolve_workspace", stalled_workspace)
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None)
    assert tid not in [s[0] for s in result.spawned] and tid not in result.auto_blocked
    assert _card_state(conn, tid) == seen["before"]
    b = seen["b"]
    assert kb.get_task(conn, tid).status == "running" and kb.get_task(conn, tid).claim_lock == b.claim_lock
    kbd._set_worker_pid(conn, tid, os.getpid(), run_id=b.current_run_id, claim_lock=b.claim_lock)
    assert kb.get_task(conn, tid).worker_pid == os.getpid()
    assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (b.current_run_id,)).fetchone()[0] == os.getpid()


@pytest.mark.parametrize("kind", ["scratch", "worktree"])
def test_a_stale_resolved_workspace_is_never_published_over_the_newer_claim(conn, tmp_path: Path, monkeypatch, kind: str) -> None:
    """C5-WORKSPACE-01: A's claim is reclaimed and re-claimed by B while A's workspace resolves, and the
    resolution SUCCEEDS. A must publish nothing (c5 wrote ``workspace_path``, a worktree's ``branch_name``
    and the first-use scratch tip by task id over B's card) and spawn nothing; B's task row, runs and
    events stay exactly as B left them, and B's own publication and spawn bookkeeping still land."""
    from hermes_cli import kanban_db_workspace as kbw
    from hermes_cli.kanban_db_retry_policy import SpawnClaim
    if kind == "worktree":
        tid = kb.create_task(conn, title="re-claimed while A resolved", assignee="theseus-implementer",
                             workspace_kind="worktree", workspace_path=str(tmp_path / "b-tree"), branch_name="wt/b-branch")
    else:
        tid = kb.create_task(conn, title="re-claimed while A resolved", assignee="theseus-implementer")
    assert not kbw._scratch_tip_shown()
    seen: dict = {}
    a_workspace = tmp_path / "a-resolved"

    def resolved_after_takeover(claimed, board=None):
        _taken_over_by_b(conn, tid, seen)
        a_workspace.mkdir()
        return (a_workspace, "wt/a-resolved") if kind == "worktree" else a_workspace
    resolver = "_resolve_worktree_workspace" if kind == "worktree" else "resolve_workspace"
    monkeypatch.setattr(kbd._kbw, resolver, resolved_after_takeover)
    spawns: list = []
    result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawns.append((task.id, workspace)) or 4242)
    assert spawns == [] and tid not in [s[0] for s in result.spawned] and tid not in result.auto_blocked
    assert _card_state(conn, tid) == seen["before"]
    assert not kbw._scratch_tip_shown()
    a, b = seen["a"], seen["b"]
    assert not kbw.publish_claimed_workspace(conn, SpawnClaim.of(a), tid, a_workspace, workspace_kind=kind,
                                             branch_name="wt/a-resolved" if kind == "worktree" else None)
    assert _card_state(conn, tid) == seen["before"]
    b_workspace = tmp_path / "b-resolved"
    assert kbw.publish_claimed_workspace(conn, SpawnClaim.of(b), tid, b_workspace, workspace_kind=kind,
                                         branch_name="wt/b-branch" if kind == "worktree" else None)
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.claim_lock) == ("running", b.current_run_id, b.claim_lock)
    assert task.workspace_path == str(b_workspace)
    assert task.branch_name == ("wt/b-branch" if kind == "worktree" else b.branch_name)
    tips = [e for e in _card_state(conn, tid)[2] if e["kind"] == "tip_scratch_workspace"]
    assert len(tips) == (1 if kind == "scratch" else 0) and kbw._scratch_tip_shown() == (kind == "scratch")
    kbd._set_worker_pid(conn, tid, os.getpid(), run_id=b.current_run_id, claim_lock=b.claim_lock)
    assert kb.get_task(conn, tid).worker_pid == os.getpid()


def test_a_current_host_refusal_is_still_recorded_and_retried(conn, tmp_path: Path, monkeypatch) -> None:
    """The fence keeps the ordinary path: a refusal for the claim still current ends that run uncharged,
    tagged, back to ``ready`` (``_refused_by_the_host`` asserts each), and a healthy host then spawns."""
    tid = kb.create_task(conn, title="plain refusal", assignee="theseus-implementer")
    first = _refused_by_the_host(conn, monkeypatch, tid, tick=0)
    assert kap.never_launched(conn, tid, conn.execute("SELECT * FROM task_runs WHERE id=?", (first,)).fetchone())
    _after_cooldown(monkeypatch, 1)
    assert _spawned(conn) == [tid]


def test_a_spawn_claim_tag_for_another_run_marks_nothing() -> None:
    """``SpawnClaim`` carries the producer's proof only for the exact claimed run."""
    from hermes_cli.kanban_db_retry_policy import SpawnClaim
    from tools.process_registry import RestartSafeScopeUnavailable

    class Claimed:
        current_run_id, claim_lock = 7, "host:1"
    tagged = RestartSafeScopeUnavailable("no restart-safe scope")
    tagged.kanban_refused_run = 6
    assert SpawnClaim.of(Claimed, tagged).launch_detail() == {}
    tagged.kanban_refused_run = 7
    assert SpawnClaim.of(Claimed, tagged).launch_detail() == {"launch": "refused_before_launch"}
    assert SpawnClaim.of(Claimed, RuntimeError("untagged")).launch_detail() == {}
    claim = SpawnClaim.of(Claimed)
    assert claim.holds({"status": "running", "current_run_id": 7, "claim_lock": "host:1"})
    for row in ({"status": "ready", "current_run_id": 7, "claim_lock": "host:1"},
                {"status": "running", "current_run_id": 8, "claim_lock": "host:1"},
                {"status": "running", "current_run_id": 7, "claim_lock": "host:2"}):
        assert not claim.holds(row)
