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
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
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
