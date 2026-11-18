"""``kanban promote --expected-terminal-run`` on a TRIAGE card: the satisfied-input continuation.

A worker's run tripped the unblock-loop breaker (``block_loop_detected`` -> ``triage``) and was
then interrupted, so its bound interpreter never wrote a ``worker_result`` (2026-11-05,
t_fc2f7ba0 run 3114). ``settle-triage`` needs that result, ``unblock`` takes blocked/scheduled
cards, plain ``promote`` todo/blocked ones and the exact-run continuation only ready ones: the
card had no fenced way back to its owner once the input it waited for arrived. The same
continuation now takes that exact shape, with positive proof that no writer of the run is left,
and records the interrupted run and the missing result; every other shape is refused.

r4: the loop run may instead have FINISHED with its own bound worker's terminal result (2026-11-06,
t_61a203e3 run 3870: blocked solely on a STOP order that was later lifted by the original
resumption). ``settle-triage`` would record a sticky wait for a MISSING input, but none is missing,
so the same continuation takes that shape too when settle-triage's own result proof
(``_settlement_terminal_run``) holds, and records ``finished_run``, the result's event id and exit
code. Every guard of the interrupted shape still applies; any other result is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_parser import build_parser

DEAD_PID = 4_194_305  # above Linux's largest pid_max: never a live process
INPUT = "credential delivered in comment c-4242"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kb.init_db()
    return home


def _cli(*argv: str) -> int:
    """The real ``hermes kanban`` parser and entry point."""
    wrap = argparse.ArgumentParser(add_help=False)
    return kb_cli.kanban_command(build_parser(wrap.add_subparsers(dest="_top")).parse_args(list(argv)))


def _admission(tmp_path: Path, status: str) -> dict:
    db = tmp_path / f"owner-{status}-{len(list(tmp_path.glob('owner-*.db')))}.db"
    with sqlite3.connect(db) as owner:
        owner.execute("CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, "
                      "status TEXT, outcome TEXT)")
        owner.execute("INSERT INTO session_admissions VALUES ('s1', 'r1', ?, 'interrupted')", (status,))
    return {"db": str(db), "session_id": "s1", "request_id": "r1", "pid": DEAD_PID, "birth": 1.0}


def _claim(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid) is not None  # this host's ``host:pid`` claim
    task = kb.get_task(conn, tid)
    return task.current_run_id, task.claim_lock


def _interrupted_loop(conn, tmp_path: Path, *, title="interrupted loop", admission="terminal",
                      kind="transient", reason="waiting for credential", spawned_pid=None,
                      before_block=None) -> tuple[str, int]:
    """A triage card whose loop run's owner admitted it, bound a (now dead) interpreter, tripped
    the breaker, and was interrupted before the interpreter recorded a result."""
    tid = kb.create_task(conn, title=title, body="original body", assignee="worker")
    _claim(conn, tid)
    assert kb.block_task(conn, tid, kind=kind, reason=reason)
    assert kb.unblock_task(conn, tid)
    run_id, claim = _claim(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "owner_admitted", _admission(tmp_path, admission), run_id=run_id)
        if spawned_pid is not None:  # the dispatcher's submitter, distinct from the bound interpreter
            kb._append_event(conn, tid, "spawned", {"pid": spawned_pid, "started_at": "boot-1|12300"}, run_id=run_id)
        kb._append_event(conn, tid, "worker_bound", {
            "pid": DEAD_PID, "claim_lock": claim, "started_at": "boot-1|12345"}, run_id=run_id)
    if before_block is not None:
        before_block(conn, tid, run_id, claim)
    assert kb.block_task(conn, tid, kind=kind, reason=reason)
    task = kb.get_task(conn, tid)
    assert task.status == "triage" and task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
    assert kb.latest_run(conn, tid).id == run_id
    return tid, run_id


def _result(conn, tid, run_id, claim, *, pid=DEAD_PID, exit_code=0) -> int:
    """The worker_result ``session_kanban._record_worker_result`` appends (bound pid, claim, exit)."""
    with kb.write_txn(conn):
        return kb._append_event(conn, tid, "worker_result", {
            "pid": pid, "claim_lock": claim, "exit_code": exit_code, "last_output": "turn finished"}, run_id=run_id)


def _result_id(conn, tid) -> int:
    return conn.execute("SELECT MAX(id) FROM task_events WHERE task_id=? AND kind='worker_result'", (tid,)).fetchone()[0]


def _finished_loop(conn, tmp_path: Path, *, title="finished loop", exit_code=0, **kw) -> tuple[str, int]:
    """The interrupted loop whose bound interpreter then recorded its own terminal result."""
    tid, run_id = _interrupted_loop(conn, tmp_path, title=title, **kw)
    _result(conn, tid, run_id, kb.latest_run(conn, tid).claim_lock, exit_code=exit_code)
    return tid, run_id


def _continuations(conn, tid) -> list[dict]:
    rows = conn.execute("SELECT payload, run_id FROM task_events WHERE task_id = ? AND kind = 'promoted_manual'",
                        (tid,)).fetchall()
    return [dict(json.loads(r["payload"]), event_run_id=r["run_id"]) for r in rows]


def _task_row(conn, tid) -> dict:
    return dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())


def test_interrupted_triage_loop_continues_to_ready_with_honest_history(kanban_home: Path, tmp_path: Path) -> None:
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        loop = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='block_loop_detected'",
                            (tid,)).fetchone()[0]
        assert json.loads(loop)["source_status"] == "ready"  # an implementation-lane loop
        # Local positive: the run's claim and binding were issued by this host.
        assert kb.latest_run(conn, tid).claim_lock.startswith(kb._host_prefix())
        kb.add_comment(conn, tid, author="operator", body=INPUT)
        before = _task_row(conn, tid)
        runs = [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")]
        events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        comments = len(kb.list_comments(conn, tid))

    assert _cli("promote", tid, INPUT, "--dry-run", "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert _task_row(conn, tid) == before  # the dry run wrote nothing
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0

    with kbc.connect() as conn:
        after = _task_row(conn, tid)
        # Only the status moves: assignee, title, body, block kind/recurrences, failure counters kept.
        assert after.pop("status") == "ready" and before.pop("status") == "triage"
        assert after == before
        # History is appended to, never rewritten: no run is closed, synthesized or edited.
        assert [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")] == runs
        all_events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        assert all_events[:len(events)] == events
        kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,))]
        assert kinds[-1] == "promoted_manual" and len(all_events) == len(events) + 1
        (event,) = _continuations(conn, tid)
        assert event["from"] == "triage" and event["continuation"] is True
        assert event["interrupted_run"] == event["expected_terminal_run"] == run_id == event["event_run_id"]
        assert event["worker_result_recorded"] is False
        assert event["reason"] == INPUT and event["actor"]
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='worker_result'",
                            (tid,)).fetchone()[0] == 0  # nothing fabricated
        assert len(kb.list_comments(conn, tid)) == comments
        # The same owner picks it up; the kept loop memory returns a same-cause re-block to triage.
        spawned = [s[0] for s in kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None).spawned]
        assert tid in spawned
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.assignee == "worker"
        assert kb.block_task(conn, tid, kind="transient", reason="still waiting")
        assert kb.get_task(conn, tid).status == "triage"


def test_recycled_bound_pid_is_not_a_live_writer(kanban_home: Path, tmp_path: Path) -> None:
    """A live process that owns the number but not the bound start identity is not the writer."""
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_set(payload, '$.pid', ?) "
                         "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), tid))
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def _live_fingerprint() -> str:
    fingerprint = kbd._process_fingerprint(os.getpid())
    assert fingerprint
    return fingerprint


REFUSALS = {
    "wrong_run": "is not the latest run",
    "newer_run": "is not the latest run",
    "open_run": "has an open run",
    "claim_lock": "live claim (claim_lock)",
    "claim_expires": "live claim (claim_expires)",
    "worker_pid": "live claim (worker_pid)",
    "current_run": "live claim (current_run_id)",
    "live_bound_worker": "may still be live",
    "live_submitter": "may still be live",
    "unverified_live_worker": "may still be live",
    "blocked_not_triage": "is 'blocked'; a continuation applies only to 'ready' or a 'triage' loop",
    "todo_not_triage": "is 'todo'; a continuation applies only to 'ready' or a 'triage' loop",
    "no_loop": "no block_loop_detected",
    "run_not_blocked": "did not end blocked",
    "no_bound": "no bound worker under its claim",
    "foreign_bound": "no bound worker under its claim",
    "owner_running": "owner admission is not terminal yet (running)",
    "owner_unknown": "owner admission is not terminal yet (unknown)",
    "no_owner": "has no owner admission",
    "empty_reason": "must name the awaited input",
    "parent_open": "unsatisfied parent dependencies",
    "earlier_live_binding": "may still be live",
    "earlier_live_spawned": "may still be live",
    "live_registered": "may still be live",
    "unreadable_record": "unreadable worker record (spawned)",
    "foreign_host": "was claimed on another host (elsewhere.example)",
    "foreign_binding": "was claimed on another host (elsewhere.example)",
    "hostless_claim": "the host of run",
    "missing_claim": "the host of run",
    "loop_says_review": "is a review-origin loop (claimed from review)",
    "loop_lane_unprovable": "cannot be proven (source_status=None)",
}


RESULT_REFUSED = "worker_result is not proven its own bound worker's terminal result"
# The finished shape: the result check runs before the writer proof, so drifts that break the
# bound record (or append a newer binding after the loop) fail settle-triage's result proof first.
FINISHED_REFUSALS = dict(REFUSALS, **{drift: RESULT_REFUSED for drift in (
    "no_bound", "foreign_bound", "unverified_live_worker", "earlier_live_binding", "foreign_binding",
    "missing_claim")})


@pytest.mark.parametrize("shape", ["interrupted", "finished"])
@pytest.mark.parametrize("drift", sorted(REFUSALS))
def test_triage_continuation_refusals_write_nothing(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture, drift: str, shape: str,
) -> None:
    with kbc.connect() as conn:
        admission = {"owner_running": "running", "owner_unknown": "unknown"}.get(drift, "terminal")
        tid, run_id = _interrupted_loop(conn, tmp_path, admission=admission)
        live = _live_fingerprint()
        mutations = {
            "wrong_run": [],
            "newer_run": [("INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
                           "VALUES (?, 'blocked', 1, 2, 'blocked')", (tid,))],
            "open_run": [("UPDATE task_runs SET ended_at=NULL WHERE id=?", (run_id,))],
            "claim_lock": [("UPDATE tasks SET claim_lock='other:1' WHERE id=?", (tid,))],
            "claim_expires": [("UPDATE tasks SET claim_expires=1 WHERE id=?", (tid,))],
            "worker_pid": [("UPDATE tasks SET worker_pid=? WHERE id=?", (DEAD_PID, tid))],
            "current_run": [("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))],
            "live_bound_worker": [("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                                   "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), live, tid))],
            "live_submitter": [("UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
                                (os.getpid(), None, run_id))],
            "unverified_live_worker": [("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                                        "WHERE task_id=? AND kind='worker_bound'",
                                        (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, tid))],
            "blocked_not_triage": [("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))],
            "todo_not_triage": [("UPDATE tasks SET status='todo' WHERE id=?", (tid,))],
            "no_loop": [("DELETE FROM task_events WHERE task_id=? AND kind='block_loop_detected'", (tid,))],
            "run_not_blocked": [("UPDATE task_runs SET status='crashed', outcome='crashed' WHERE id=?", (run_id,))],
            "no_bound": [("DELETE FROM task_events WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "foreign_bound": [("UPDATE task_events SET payload=json_set(payload, '$.claim_lock', 'foreign') "
                               "WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "owner_running": [],
            "owner_unknown": [],
            "no_owner": [("DELETE FROM task_events WHERE task_id=? AND kind='owner_admitted'", (tid,))],
            "empty_reason": [],
            "parent_open": [],
            # Newest-only selection would see only the dead record appended last.
            "earlier_live_binding": [("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                                      "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), live, tid))],
            "earlier_live_spawned": [],
            "live_registered": [],
            "unreadable_record": [],
            # Pid liveness is host-local: a foreign or unprovable claim's absent pid proves nothing.
            "foreign_host": [("UPDATE task_runs SET claim_lock='elsewhere.example:4242' WHERE id=?", (run_id,)),
                             ("UPDATE task_events SET payload=json_set(payload, '$.claim_lock', "
                              "'elsewhere.example:4242') WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "foreign_binding": [],
            "hostless_claim": [("UPDATE task_runs SET claim_lock='worker' WHERE id=?", (run_id,)),
                               ("UPDATE task_events SET payload=json_set(payload, '$.claim_lock', 'worker') "
                                "WHERE task_id=? AND kind='worker_bound'", (tid,))],
            "loop_says_review": [("UPDATE task_events SET payload=json_set(payload, '$.source_status', 'review') "
                                  "WHERE task_id=? AND kind='block_loop_detected'", (tid,))],
            "loop_lane_unprovable": [("UPDATE task_events SET payload=json_remove(payload, '$.source_status') "
                                      "WHERE task_id=? AND kind='block_loop_detected'", (tid,))],
            "missing_claim": [("UPDATE task_runs SET claim_lock=NULL WHERE id=?", (run_id,)),
                              ("UPDATE task_events SET payload=json_remove(payload, '$.claim_lock') "
                               "WHERE task_id=? AND kind='worker_bound'", (tid,))],
        }
        later = {
            "earlier_live_binding": [("worker_bound", {"pid": DEAD_PID, "started_at": "boot-1|1"})],
            "earlier_live_spawned": [("spawned", {"pid": os.getpid(), "started_at": live}),
                                     ("spawned", {"pid": DEAD_PID, "started_at": "boot-1|1"})],
            "live_registered": [("worker_registered", {"pid": os.getpid(), "started_at": live})],
            "unreadable_record": [("spawned", {"pid": None, "started_at": "boot-1|1"})],
        }
        foreign_binding = {"pid": DEAD_PID, "started_at": "boot-1|1", "claim_lock": "elsewhere.example:4242"}
        with kb.write_txn(conn):
            for sql, params in mutations[drift]:
                conn.execute(sql, params)
            claim = conn.execute("SELECT claim_lock FROM task_runs WHERE id=?", (run_id,)).fetchone()[0]
            for kind, payload in later.get(drift, []):
                kb._append_event(conn, tid, kind, dict(payload, claim_lock=claim), run_id=run_id)
            if drift == "foreign_binding":  # the run's own claim and binding stay local
                kb._append_event(conn, tid, "worker_bound", foreign_binding, run_id=run_id)
        if shape == "finished":  # as the real writer: the newest binding's pid under the run's claim
            bound = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound' "
                                 "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
            _result(conn, tid, run_id, claim, pid=json.loads(bound[0]).get("pid") if bound else DEAD_PID)
        if drift == "parent_open":
            kb.link_tasks(conn, parent_id=kb.create_task(conn, title="open parent"), child_id=tid)
        snapshot = list(conn.iterdump())

    run = {"wrong_run": run_id - 1}.get(drift, run_id)
    reason = [] if drift == "empty_reason" else [INPUT]
    for dry in ([], ["--dry-run"]):
        capsys.readouterr()
        assert _cli("promote", tid, *reason, *dry, "--expected-terminal-run", str(run)) == 1
        err = capsys.readouterr().err
        assert (REFUSALS if shape == "interrupted" else FINISHED_REFUSALS)[drift] in err, (drift, err)
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


def test_live_writer_with_unreadable_current_identity_is_refused(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """A live pid whose start identity cannot be read now is not proven gone (``_worker_identity``
    is UNKNOWN, never signalled)."""
    from gateway import status
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        live = _live_fingerprint()
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_set(payload, '$.pid', ?, '$.started_at', ?) "
                         "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), live, tid))
        snapshot = list(conn.iterdump())
    real = status.get_process_start_time
    monkeypatch.setattr(status, "get_process_start_time",
                        lambda pid: None if pid == os.getpid() else real(pid))
    from hermes_cli.process_identity import Identity
    assert kbd._worker_identity(os.getpid(), live) is Identity.UNKNOWN  # unproven: held, never signalled
    assert kbd._worker_alive(os.getpid(), live) and kbd._pid_recycled(os.getpid(), live)
    for dry in ([], ["--dry-run"]):
        capsys.readouterr()
        assert _cli("promote", tid, INPUT, *dry, "--expected-terminal-run", str(run_id)) == 1
        err = capsys.readouterr().err
        assert f"the identity of live pid {os.getpid()} of run {run_id} cannot be read" in err, err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
    # Control: the same live writer with a readable identity is refused as live, not unreadable.
    monkeypatch.setattr(status, "get_process_start_time", real)
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 1
    assert "may still be live" in capsys.readouterr().err


@pytest.mark.parametrize("shape", ["interrupted", "finished"])
def test_review_origin_loop_is_refused_without_writes(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture, shape: str,
) -> None:
    """A reviewer run's interrupted loop (review run 3747's sequence: request_review,
    claim_review_task, block, unblock, claim_review_task, block) is refused: landing it in ready
    would make dispatch claim it as an implementation run, dropping source_status=review and
    sdlc-review, and request_changes would then fail ("active run was not claimed from review")."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review loop", body="original body", assignee="worker")
        assert kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=kb.get_task(conn, tid).current_run_id)
        assert kb.claim_review_task(conn, tid)
        assert kb.block_task(conn, tid, kind="transient", reason="waiting for credential")
        assert kb.unblock_task(conn, tid)
        assert kb.claim_review_task(conn, tid)
        task = kb.get_task(conn, tid)
        run_id = task.current_run_id
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "owner_admitted", _admission(tmp_path, "terminal"), run_id=run_id)
            kb._append_event(conn, tid, "worker_bound", {
                "pid": DEAD_PID, "claim_lock": task.claim_lock, "started_at": "boot-1|12345"}, run_id=run_id)
        assert kb.block_task(conn, tid, kind="transient", reason="waiting for credential")
        task = kb.get_task(conn, tid)
        assert task.status == "triage" and task.assignee == "reviewer"
        loop = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='block_loop_detected'",
                            (tid,)).fetchone()[0]
        assert json.loads(loop)["source_status"] == "review"
        if shape == "finished":  # the reviewer's own bound worker finished after its loop
            _result(conn, tid, run_id, kb.latest_run(conn, tid).claim_lock)
        snapshot = list(conn.iterdump())

    for dry in ([], ["--dry-run"]):
        capsys.readouterr()
        rc = _cli("promote", tid, INPUT, *dry, "--expected-terminal-run", str(run_id))
        err = capsys.readouterr().err
        with kbc.connect() as conn:  # r2 wrote Ready here: (0, 'ready', ['triage'])
            outcome = (rc, kb.get_task(conn, tid).status, [e["from"] for e in _continuations(conn, tid)])
        assert outcome == (1, "triage", []), (dry, outcome, err)
        assert f"run {run_id} of {tid} is a review-origin loop (claimed from review)" in err, err
        assert "would drop the review lane" in err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot  # zero writes


def test_triage_continuation_is_orchestrator_only_and_one_card(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        other, _ = _interrupted_loop(conn, tmp_path, title="second")
        snapshot = list(conn.iterdump())
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 1
    assert "orchestrator-only" in capsys.readouterr().err
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _cli("promote", tid, INPUT, "--ids", other, "--expected-terminal-run", str(run_id)) == 2
    # Plain promote still refuses a triage card, and points at the fence.
    assert _cli("promote", tid, INPUT) == 1
    assert "a 'triage' loop: --expected-terminal-run" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


def test_settle_triage_contract_unchanged_beside_the_continuation(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """settle-triage still needs the run's positive worker result and still parks the card in a
    sticky external wait for a MISSING input; a settled wait is not a triage card, so the
    continuation refuses it. A finished loop whose input has ARRIVED continues instead, and is then
    no longer settle-triage's."""
    monkeypatch.setattr(kb_cli, "_profile_author", lambda: "orchestrator")
    settle = ["settle-triage", "--kind", "needs_input", "--owner", "operator",
              "--required-input", "credential", "--event", "operator delivers the credential"]
    with kbc.connect() as conn:
        interrupted, interrupted_run = _interrupted_loop(conn, tmp_path)
        missing, missing_run = _finished_loop(conn, tmp_path, title="finished loop, input missing")
        arrived, arrived_run = _finished_loop(conn, tmp_path, title="finished loop, input arrived")
        snapshot = list(conn.iterdump())
    # The interrupted loop has no result: settle-triage refuses it exactly as before.
    assert _cli(settle[0], interrupted, "--expected-terminal-run", str(interrupted_run), *settle[1:]) == 1
    assert "exact loop run lacks a matching terminal worker result" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
    # Input still missing: the operator settles the finished loop into a sticky external wait.
    assert _cli(settle[0], missing, "--expected-terminal-run", str(missing_run), *settle[1:]) == 0
    with kbc.connect() as conn:
        task = kb.get_task(conn, missing)
        assert task.status == "blocked" and task.block_kind == "needs_input"
        snapshot = list(conn.iterdump())
    for dry in ([], ["--dry-run"]):  # a settled wait is not continued by the triage fence
        assert _cli("promote", missing, INPUT, *dry, "--expected-terminal-run", str(missing_run)) == 1
        assert "is 'blocked'; a continuation applies only to 'ready' or a 'triage' loop" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
        assert _continuations(conn, missing) == []
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None)  # its wait is sticky
        assert kb.get_task(conn, missing).status == "blocked"
    # Input arrived: the finished loop continues, and settle-triage no longer applies to it.
    assert _cli("promote", arrived, INPUT, "--expected-terminal-run", str(arrived_run)) == 0
    assert _cli(settle[0], arrived, "--expected-terminal-run", str(arrived_run), *settle[1:]) == 1
    assert "task is not in triage" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert kb.get_task(conn, arrived).status == "ready"
        assert [e["from"] for e in _continuations(conn, arrived)] == ["triage"]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_finished_triage_loop_continues_to_ready_with_accurate_event(
    kanban_home: Path, tmp_path: Path, exit_code: int,
) -> None:
    """The run's own bound worker finished after its loop (any int exit, recorded as is): the
    continuation moves the card to ready keeping everything else, and records the real result."""
    with kbc.connect() as conn:
        tid, run_id = _finished_loop(conn, tmp_path, exit_code=exit_code)
        result_event = _result_id(conn, tid)
        kb.add_comment(conn, tid, author="operator", body=INPUT)
        before = _task_row(conn, tid)
        runs = [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")]
        events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        snapshot = list(conn.iterdump())

    assert _cli("promote", tid, INPUT, "--dry-run", "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot  # the dry run wrote nothing
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0

    with kbc.connect() as conn:
        after = _task_row(conn, tid)
        assert after.pop("status") == "ready" and before.pop("status") == "triage"
        assert after == before  # assignee, title, body, block kind/recurrences, failure counters kept
        assert [tuple(r) for r in conn.execute("SELECT * FROM task_runs ORDER BY id")] == runs
        all_events = [tuple(r) for r in conn.execute("SELECT * FROM task_events ORDER BY id")]
        assert all_events[:len(events)] == events and len(all_events) == len(events) + 1
        (event,) = _continuations(conn, tid)
        assert event == {
            "actor": event["actor"], "reason": INPUT, "from": "triage", "continuation": True,
            "expected_terminal_run": run_id, "finished_run": run_id, "worker_result_recorded": True,
            "worker_result_event": result_event, "worker_exit_code": exit_code, "event_run_id": run_id,
        }
        assert event["actor"] and "interrupted_run" not in event
        # The one real result is referenced, never duplicated or synthesized.
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='worker_result'",
                            (tid,)).fetchone()[0] == 1
        # The same owner picks it up; the kept loop memory returns a same-cause re-block to triage.
        assert tid in [s[0] for s in kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None).spawned]
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.assignee == "worker"
        assert kb.block_task(conn, tid, kind="transient", reason="still waiting")
        assert kb.get_task(conn, tid).status == "triage"


def test_real_worker_result_writer_is_accepted(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result written by the real bound-interpreter writer (``session_kanban._record_worker_result``)
    after the loop passes the same proof. This interpreter's pid stands in for the bound one under a
    different recorded start identity, so it is a recycled number, not a live writer."""
    import threading

    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "_TURNS_DONE", threading.Event())
    with kbc.connect() as conn:
        tid, run_id = _interrupted_loop(conn, tmp_path)
        claim = kb.latest_run(conn, tid).claim_lock
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload=json_set(payload, '$.pid', ?) "
                         "WHERE task_id=? AND kind='worker_bound'", (os.getpid(), tid))
    session_kanban._record_worker_result(
        {"task_id": tid, "run_id": run_id, "claim_lock": claim, "db": str(kb.kanban_db_path())}, 0, "done")
    with kbc.connect() as conn:
        result_event = _result_id(conn, tid)
        assert result_event is not None
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        (event,) = _continuations(conn, tid)
        assert (event["finished_run"], event["worker_result_event"], event["worker_exit_code"]) == (run_id, result_event, 0)
        assert kb.get_task(conn, tid).status == "ready"


def test_finished_loop_with_done_parent_continues(kanban_home: Path, tmp_path: Path) -> None:
    """Existing parents are allowed once done (the open-parent refusal is in the drift table)."""
    with kbc.connect() as conn:
        tid, run_id = _finished_loop(conn, tmp_path)
        parent = kb.create_task(conn, title="done parent", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
        assert kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="supplied")
        assert kb.get_task(conn, tid).status == "triage"
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def _result_before_loop(conn, tid, run_id, claim):
    _result(conn, tid, run_id, claim)


RESULT_DRIFTS = {
    # (sql mutations applied to the finished loop, extra events appended after it)
    "result_pid_mismatch": ([("UPDATE task_events SET payload=json_set(payload, '$.pid', ?) "
                              "WHERE task_id=? AND kind='worker_result'", (DEAD_PID + 1, "{tid}"))], []),
    "result_claim_mismatch": ([("UPDATE task_events SET payload=json_set(payload, '$.claim_lock', 'other:1') "
                                "WHERE task_id=? AND kind='worker_result'", ("{tid}",))], []),
    "result_exit_not_int": ([("UPDATE task_events SET payload=json_set(payload, '$.exit_code', '0') "
                              "WHERE task_id=? AND kind='worker_result'", ("{tid}",))], []),
    "result_exit_missing": ([("UPDATE task_events SET payload=json_remove(payload, '$.exit_code') "
                              "WHERE task_id=? AND kind='worker_result'", ("{tid}",))], []),
    "bound_identity_unverified": ([("UPDATE task_events SET payload=json_set(payload, '$.started_at', ?) "
                                    "WHERE task_id=? AND kind='worker_bound'",
                                    (kbd.UNVERIFIED_WORKER_FINGERPRINT, "{tid}"))], []),
    "run_claim_unreleased": ([("UPDATE task_runs SET claim_expires=1 WHERE id=?", ("{run}",))], []),
    # A newer result under another pid: the newest result is the one proven, so it is refused.
    "newer_foreign_result": ([], [("worker_result", {"pid": DEAD_PID + 1, "exit_code": 0})]),
    # A newer binding after the result: the result no longer follows the newest binding.
    "rebound_after_result": ([], [("worker_bound", {"pid": DEAD_PID + 1, "started_at": "boot-1|2"})]),
    "result_before_loop": ([], []),
}


@pytest.mark.parametrize("drift", sorted(RESULT_DRIFTS))
def test_unproven_worker_result_is_refused_without_writes(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture, drift: str,
) -> None:
    """Only settle-triage's own proof (``_settlement_terminal_run``) admits a finished loop; every
    other result (wrong pid/claim, unordered, non-int exit, unverified binding, unreleased run
    claim, newer foreign result or rebinding) is refused, real and dry-run, with zero writes."""
    with kbc.connect() as conn:
        if drift == "result_before_loop":
            tid, run_id = _interrupted_loop(conn, tmp_path, before_block=_result_before_loop)
        else:
            tid, run_id = _finished_loop(conn, tmp_path)
        claim = kb.latest_run(conn, tid).claim_lock
        mutations, later = RESULT_DRIFTS[drift]
        with kb.write_txn(conn):
            for sql, params in mutations:
                conn.execute(sql, tuple({"{tid}": tid, "{run}": run_id}.get(p, p) for p in params))
            for kind, payload in later:
                kb._append_event(conn, tid, kind, dict(payload, claim_lock=claim), run_id=run_id)
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='worker_result'",
                            (tid,)).fetchone()[0] >= 1
        snapshot = list(conn.iterdump())
    for dry in ([], ["--dry-run"]):
        capsys.readouterr()
        assert _cli("promote", tid, INPUT, *dry, "--expected-terminal-run", str(run_id)) == 1
        err = capsys.readouterr().err
        assert f"run {run_id}'s {RESULT_REFUSED}" in err, (drift, err)
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


@pytest.mark.parametrize("shape", ["interrupted", "finished"])
def test_repeated_continuation_never_reruns_the_triage_move(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture, shape: str,
) -> None:
    """A repeat of the same command finds a READY card: no second ``from: triage`` move, no state
    change, the one real result untouched. Before any new run the ready path's existing rule
    applies (one ``from: ready`` release on the same exact run, as r3); once the owner has claimed a
    new run, or that run looped back to triage, the old run is refused with zero writes."""
    with kbc.connect() as conn:
        tid, run_id = (_finished_loop if shape == "finished" else _interrupted_loop)(conn, tmp_path)
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        row = _task_row(conn, tid)
        results = _result_id(conn, tid)
    assert _cli("promote", tid, INPUT, "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert _task_row(conn, tid) == row and _result_id(conn, tid) == results
        assert [e["from"] for e in _continuations(conn, tid)] == ["triage", "ready"]
        assert tid in [s[0] for s in kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: None).spawned]
        snapshot = list(conn.iterdump())
    for dry in ([], ["--dry-run"]):
        assert _cli("promote", tid, INPUT, *dry, "--expected-terminal-run", str(run_id)) == 1
        assert "is 'running'; a continuation applies only to" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
        assert kb.block_task(conn, tid, kind="transient", reason="still waiting")
        assert kb.get_task(conn, tid).status == "triage"
        snapshot = list(conn.iterdump())
    for dry in ([], ["--dry-run"]):
        assert _cli("promote", tid, INPUT, *dry, "--expected-terminal-run", str(run_id)) == 1
        assert f"run {run_id} is not the latest run of {tid}" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot


STOP = "STOP: Eric's cleanup order; do not continue until he lifts it"
RESUMPTION = "Resumption: Eric lifted the cleanup STOP after settlement; resume under existing permissions"


def test_lifted_stop_finished_loop_corporate_shape(
    kanban_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """Fixture of t_61a203e3 run 3870 (no live board): needs_input loop on a STOP order, a prior
    run, owner admitted, a dispatcher submitter pid distinct from the bound interpreter, the bound
    worker's own result (exit 0) after the loop, no claim, open run or parents; the STOP is then
    lifted by the original resumption comment. While the run's owner admission is still ``unknown``
    it is refused with zero writes; once terminal, the continuation records the finished run."""
    with kbc.connect() as conn:
        tid, run_id = _finished_loop(conn, tmp_path, title="corporate", admission="unknown",
                                     kind="needs_input", reason=STOP, spawned_pid=DEAD_PID - 1)
        assert kb.latest_run(conn, tid).id == run_id and run_id - 1 in [r.id for r in kb.list_runs(conn, tid)]
        task = _task_row(conn, tid)
        assert (task["status"], task["block_kind"], task["current_run_id"], task["claim_lock"]) == (
            "triage", "needs_input", None, None)
        assert conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id=?", (tid,)).fetchone()[0] == 0
        kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=? AND run_id=? ORDER BY id",
                                            (tid, run_id))]
        assert kinds.index("worker_bound") < kinds.index("block_loop_detected") < kinds.index("worker_result")
        result_event = _result_id(conn, tid)
        comment = kb.add_comment(conn, tid, author="interlocutor", body=RESUMPTION)
        admission = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='owner_admitted'",
                                            (tid,)).fetchone()[0])
        snapshot = list(conn.iterdump())
    reason = f"Satisfied input: STOP lifted by resumption comment {comment}"
    for dry in ([], ["--dry-run"]):
        assert _cli("promote", tid, reason, *dry, "--expected-terminal-run", str(run_id)) == 1
        assert "owner admission is not terminal yet (unknown)" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
    with sqlite3.connect(admission["db"]) as owner:  # the owner settles its admission
        owner.execute("UPDATE session_admissions SET status='terminal'")
    assert _cli("promote", tid, reason, "--dry-run", "--expected-terminal-run", str(run_id)) == 0
    with kbc.connect() as conn:
        assert list(conn.iterdump()) == snapshot
    assert _cli("promote", tid, reason, "--expected-terminal-run", str(run_id)) == 0
    assert f"Continued {tid} -> ready after terminal run {run_id}" in capsys.readouterr().out
    with kbc.connect() as conn:
        after = _task_row(conn, tid)
        assert after.pop("status") == "ready" and task.pop("status") == "triage" and after == task
        (event,) = _continuations(conn, tid)
        assert (event["finished_run"], event["worker_result_recorded"], event["worker_result_event"],
                event["worker_exit_code"], event["reason"]) == (run_id, True, result_event, 0, reason)
        # No missing-input wait was fabricated: no settlement ``blocked`` event after the loop.
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='blocked' "
                            "AND json_extract(payload, '$.settled_from') IS NOT NULL", (tid,)).fetchone()[0] == 0


def test_promote_help_names_both_loop_shapes(capsys: pytest.CaptureFixture) -> None:
    wrap = argparse.ArgumentParser(add_help=False)
    with pytest.raises(SystemExit) as done:
        build_parser(wrap.add_subparsers(dest="_top")).parse_args(["promote", "--help"])
    assert done.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "interrupted with no worker result or finished with its own bound worker's result" in text
    assert "A result whose input is still missing is settle-triage's" in text
