"""Quota-walled attempts never push the owner's pending callback out of its retry's context.

Regression for t_d4ba87d4: an owner card yielded (``scheduled``) naming the author result it waited
for, was released, then hit a provider quota wall on ~70 consecutive attempts. The retry context shows
the newest ``_CTX_MAX_PRIOR_ATTEMPTS`` attempts, so after ten walls the yield that named the pending
author, run and callback was no longer in the context of the first attempt that could act. Real
board SQLite, real attempt bookings through the dead-worker sweep, real ``build_worker_context``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.auth_codex import _codex_quota_exhausted_error

PENDING = "PENDING-CALLBACK author cmwAUTHOR1 run 41: consume its result, do not relaunch the author"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    offset = [0]
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + offset[0])
    return offset


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_assignees_spawnable, clock):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _claim(conn, tid):
    host = kb._claimer_id().split(":", 1)[0]
    task = kb.claim_task(conn, tid, claimer=f"{host}:owner")
    kbd._set_worker_pid(conn, tid, _dead_pid())
    return task


def _walled_attempt(conn, tid, clock, monkeypatch):
    """One attempt that dies at credential resolution on the exhausted pool, reaped by a real tick."""
    task = _claim(conn, tid)
    bound = _dead_pid()
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
    context = {"db": conn.execute("PRAGMA database_list").fetchone()[2], "task_id": tid,
               "run_id": task.current_run_id, "claim_lock": task.claim_lock, "route_inputs": None}
    frame = {"home": "/nonexistent", "policy": {"model": "gpt-test", "kanban_json": json.dumps(context)}}
    with monkeypatch.context() as m:
        m.setattr(os, "getpid", lambda: bound)
        session_kanban.record_start_failure(frame, _codex_quota_exhausted_error(109934))
    clock[0] += 60
    res = kbd.dispatch_once(conn, dry_run=True)
    assert tid in res.rate_limited


def _unknown_attempt(conn, tid, clock):
    """An attempt whose result never arrived: its interpreter vanished without recording one."""
    _claim(conn, tid)
    clock[0] += 60
    res = kbd.dispatch_once(conn, dry_run=True)
    assert tid in res.crashed


def _yield_waiting_for_author(conn, tid):
    task = kb.claim_task(conn, tid, claimer="host:owner")
    assert kb.schedule_task(conn, tid, reason=PENDING, expected_run_id=task.current_run_id)
    return task.current_run_id


def test_quota_walls_keep_the_pending_callback_and_every_return_in_the_retry_context(conn, clock, monkeypatch):
    tid = kb.create_task(conn, title="owner", assignee="w", body="Consume the author's result.")
    yielded = _yield_waiting_for_author(conn, tid)
    # The author's callback releases the exact yielded attempt; its return comment is resent after a
    # lost acknowledgement (a duplicate), and a later return follows.
    kb.add_comment(conn, tid, "cmwAUTHOR1", "RESULT-R1 sealed at /scratch/r1")
    assert kb.unblock_task(conn, tid, expected_terminal_run=yielded)
    for n in range(12):
        _walled_attempt(conn, tid, clock, monkeypatch)
        if n == 3:
            kb.add_comment(conn, tid, "cmwAUTHOR1", "RESULT-R1 sealed at /scratch/r1")  # resend: ACK was lost
        if n == 6:
            _unknown_attempt(conn, tid, clock)  # an unknown result stays retryable
            assert kb.get_task(conn, tid).status == "ready"
        if n == 9:
            kb.add_comment(conn, tid, "cmwAUTHOR1", "RESULT-R2 sealed at /scratch/r2")
    assert kb.get_task(conn, tid).status == "ready"

    context = kb.build_worker_context(conn, tid)
    assert PENDING in context  # the yield that names what the owner waits for
    assert context.count("RESULT-R1") == 2 and "RESULT-R2" in context
    # The walls are still all accounted for, as two entries around the unknown attempt.
    assert "### Attempts 2–8 — rate_limited ×7" in context and "### Attempts 10–14 — rate_limited ×5" in context
    assert "### Attempt 9 — crashed" in context
    assert "Codex provider quota exhausted (429); retry after 109934s" in context


def test_a_wall_that_recorded_more_than_the_sweeps_booking_keeps_its_own_entry(conn, clock, monkeypatch):
    tid = kb.create_task(conn, title="owner", assignee="w")
    for _ in range(3):
        _walled_attempt(conn, tid, clock, monkeypatch)
    middle = conn.execute("SELECT id, metadata FROM task_runs WHERE task_id=? ORDER BY id", (tid,)).fetchall()[1]
    meta = json.loads(middle["metadata"])
    meta["callback"] = {"author": "cmwAUTHOR2", "job": "build-7"}
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(meta), middle["id"]))
    context = kb.build_worker_context(conn, tid)
    assert "### Attempt 1 — rate_limited" in context and "### Attempt 3 — rate_limited" in context
    assert "### Attempt 2 — rate_limited" in context and "cmwAUTHOR2" in context and "build-7" in context


def test_the_newest_attempts_cap_still_bounds_the_context(conn, clock, monkeypatch):
    tid = kb.create_task(conn, title="owner", assignee="w")
    for n in range(14):
        task = kb.claim_task(conn, tid, claimer="host:owner")
        kb.schedule_task(conn, tid, reason=f"YIELD-{n}", expected_run_id=task.current_run_id)
        kb.unblock_task(conn, tid)
    context = kb.build_worker_context(conn, tid)
    assert "4 earlier attempts omitted" in context
    assert "YIELD-3" not in context and "YIELD-4" in context and "YIELD-13" in context


def test_a_folded_wall_whose_turn_ran_keeps_what_it_reported(conn, clock, monkeypatch):
    """A wall met mid-turn may follow real work: folding keeps its own report line."""
    from types import SimpleNamespace
    tid = kb.create_task(conn, title="owner", assignee="w")
    _walled_attempt(conn, tid, clock, monkeypatch)
    task = _claim(conn, tid)
    bound = _dead_pid()
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
    context = {"db": conn.execute("PRAGMA database_list").fetchone()[2], "task_id": tid,
               "run_id": task.current_run_id, "claim_lock": task.claim_lock}
    result = {"failed": True, "failure_reason": "rate_limit", "final_response": "MIDTURN-ACTED launched job j-9; then HTTP 429"}
    with monkeypatch.context() as m:
        m.setattr(os, "getpid", lambda: bound)
        m.setattr(session_kanban, "_run_task_turns", lambda *_a: result)
        session_kanban.run_worker_turns(SimpleNamespace(provider="openai-codex", model="gpt-test"),
                                        {"policy": {"kanban_json": json.dumps(context)}}, [])
    clock[0] += 60
    assert tid in kbd.dispatch_once(conn, dry_run=True).rate_limited
    _walled_attempt(conn, tid, clock, monkeypatch)
    rendered = kb.build_worker_context(conn, tid)
    assert "### Attempts 1–3 — rate_limited ×3" in rendered
    assert "- attempt 2 ran a turn: MIDTURN-ACTED launched job j-9" in rendered


def _turn_wall(conn, tid, clock, monkeypatch, evidence):
    """One attempt whose turn ran, then ended on a provider wall, booked by the real producer and sweep."""
    from types import SimpleNamespace
    task = _claim(conn, tid)
    bound = _dead_pid()
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
    context = {"db": conn.execute("PRAGMA database_list").fetchone()[2], "task_id": tid,
               "run_id": task.current_run_id, "claim_lock": task.claim_lock}
    result = {"failed": True, "failure_reason": "rate_limit", "final_response": evidence}
    with monkeypatch.context() as m:
        m.setattr(os, "getpid", lambda: bound)
        m.setattr(session_kanban, "_run_task_turns", lambda *_a: result)
        session_kanban.run_worker_turns(SimpleNamespace(provider="openai-codex", model="gpt-test"),
                                        {"policy": {"kanban_json": json.dumps(context)}}, [])
    clock[0] += 60
    assert tid in kbd.dispatch_once(conn, dry_run=True).rate_limited


def _repeat_turn_walls(conn, tid, count, evidence_of):
    """``count`` more closed turn walls copied from the card's newest booked one (the durable row the
    real producer and sweep wrote), each with its own report: a long wall without launching workers."""
    row = dict(conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone())
    columns = [key for key in row if key != "id"]
    sql = f"INSERT INTO task_runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
    with kb.write_txn(conn):
        for n in range(count):
            meta = json.loads(row["metadata"])
            meta["provider_wall"]["evidence"] = evidence_of(n)
            conn.execute(sql, [json.dumps(meta) if key == "metadata" else row[key] for key in columns])


@pytest.mark.parametrize("walls", [20, 400])
def test_a_long_run_of_turn_walls_keeps_the_context_bounded_and_the_handoff(conn, clock, monkeypatch, walls):
    """Hundreds of retries whose turns ran before a quota wall fold into one entry whose size does not
    grow with their number: the earliest and latest reports stay, the rest are counted and named by
    run, and every durable row keeps its report. The yield naming the pending author still leads."""
    tid = kb.create_task(conn, title="owner", assignee="w")
    yielded = _yield_waiting_for_author(conn, tid)
    assert kb.unblock_task(conn, tid, expected_terminal_run=yielded)
    _turn_wall(conn, tid, clock, monkeypatch, "EARLY-ACTED launched job j-1; then HTTP 429")

    def filler(n):
        return f"turn report {n}: HTTP 429 " + "x" * 280

    _repeat_turn_walls(conn, tid, walls - 2, filler)
    _repeat_turn_walls(conn, tid, 1, lambda n: "LATE-ACTED launched job j-2; then HTTP 429")
    _repeat_turn_walls(conn, tid, 1, filler)  # the newest wall, shown in full as before
    context = kb.build_worker_context(conn, tid)
    prior = context[context.index("## Prior attempts on this task"):]
    folded = prior.split("The newest:")[0]
    assert PENDING in context
    assert "EARLY-ACTED launched job j-1" in folded and "LATE-ACTED launched job j-2" in folded
    # Attempt 1 yielded; attempts 2..walls+1 ran a turn before the newest wall: the earliest two and
    # the latest two reports are shown, the rest counted and named by run.
    runs = [r.id for r in kb.list_runs(conn, tid)]
    hidden = walls - 4
    assert folded.count(" ran a turn: ") == 4 and len(prior) < 4000, len(prior)
    assert (f"- {hidden} more attempts ran a turn (attempts 4–{walls - 1}, runs #{runs[3]}–#{runs[walls - 2]}); "
            f"each report is its run's `metadata.provider_wall`: `hermes kanban runs {tid} --json`") in folded
    durable = [json.loads(r["metadata"])["provider_wall"]["evidence"] for r in conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id=? AND outcome='rate_limited' ORDER BY id", (tid,))]
    assert len(durable) == walls + 1 and durable[0].startswith("EARLY-ACTED") and durable[-2].startswith("LATE-ACTED")
