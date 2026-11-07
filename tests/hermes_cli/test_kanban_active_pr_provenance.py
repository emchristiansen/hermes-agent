"""Which ``promoted`` events release the ``active_pr`` respawn guard (review t_a2b1cc53, APR-01).

Before ``promoted`` carried ``from``, ``recompute_ready`` wrote the same NULL payload (or only the
session ``actor`` stamp) for a todo dependency release and for a breaker park recovering on its
own. Those persisted rows stay on boards, so a missing ``from`` is never read as a deliberate
release: only the worker's own ``dependency_wait`` as the last parking event corroborates one.
Checked at the dispatcher's selection boundary with rows exactly as the old producer wrote them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "Opened https://github.com/example/repo/pull/44 for review."


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _backdate_comments(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE task_comments SET created_at = created_at - 60 WHERE task_id = ?", (tid,))


def _old_producer_promotion(conn, tid, payload=None):
    """What a09e3eb6's ``recompute_ready`` persisted for todo and blocked alike."""
    with kb.write_txn(conn):
        assert conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ? AND status IN ('todo', 'blocked')",
                            (tid,)).rowcount == 1
        kb._append_event(conn, tid, "promoted", payload)


def _breaker_parked(conn, title):
    """The PR worker crashes into its breaker: ``blocked`` + ``gave_up`` after its PR comment."""
    tid = kb.create_task(conn, title=title, assignee="dev")
    kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
    assert kbd._record_task_failure(conn, tid, "worker crashed", outcome="crashed",
                                    failure_limit=1, release_claim=True, end_run=True)
    assert kb.get_task(conn, tid).status == "blocked"
    _backdate_comments(conn, tid)
    return tid


def _dependency_parked(conn, title):
    """The PR worker parks its card on a parent: ``dependency_wait`` lands it in ``todo``."""
    tid = kb.create_task(conn, title=title, assignee="dev")
    parent = kb.create_task(conn, title=f"parent of {title}", assignee="reviewer")
    run = kb.claim_task(conn, tid)
    kb.link_tasks(conn, parent, tid, expected_child_run_id=run.current_run_id)
    kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
    assert kb.block_task(conn, tid, reason="await review", kind="dependency", expected_run_id=run.current_run_id)
    assert kb.get_task(conn, tid).status == "todo"
    _backdate_comments(conn, tid)
    return tid, parent


def _promotion_payloads(conn, tid):
    return [json.loads(r[0]) if r[0] else None for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'promoted' ORDER BY id", (tid,))]


def test_legacy_breaker_promotion_stays_guarded_and_evidenced_dependency_release_resumes(conn) -> None:
    legacy_breaker = _breaker_parked(conn, "legacy breaker, NULL payload")
    _old_producer_promotion(conn, legacy_breaker)
    stamped_breaker = _breaker_parked(conn, "legacy breaker, actor stamp only")
    _old_producer_promotion(conn, stamped_breaker, {"actor": "reviewer/session:s1"})

    legacy_dependency, parent = _dependency_parked(conn, "legacy dependency release")
    with kb.write_txn(conn):  # the parent finished; the old producer re-readied the card
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
    _old_producer_promotion(conn, legacy_dependency, {"actor": "reviewer/session:s2"})

    # Parked on a dependency once, but the breaker park came later: still the crash loop.
    waited_then_tripped, parent = _dependency_parked(conn, "dependency wait, then breaker")
    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_links WHERE child_id = ?", (waited_then_tripped,))
    _old_producer_promotion(conn, waited_then_tripped)
    kb.claim_task(conn, waited_then_tripped)
    assert kbd._record_task_failure(conn, waited_then_tripped, "worker crashed", outcome="crashed",
                                    failure_limit=1, release_claim=True, end_run=True)
    with kb.write_txn(conn):  # all of that happened before the PR worker's newer PR comment
        for table in ("task_events", "task_comments"):
            conn.execute(f"UPDATE {table} SET created_at = created_at - 120 WHERE task_id = ?",
                         (waited_then_tripped,))
    kb.add_comment(conn, waited_then_tripped, author="dev", body=PR_COMMENT)
    _backdate_comments(conn, waited_then_tripped)
    _old_producer_promotion(conn, waited_then_tripped)

    # Created under an open parent: creation itself records the dependency_wait it sat in.
    created_waiting = kb.create_task(conn, title="created under a parent", assignee="dev",
                                     parents=[kb.create_task(conn, title="p")])
    assert kb._latest_event(conn, created_waiting, "dependency_wait") is not None
    kb.add_comment(conn, created_waiting, author="human", body=PR_COMMENT)
    _backdate_comments(conn, created_waiting)
    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_links WHERE child_id = ?", (created_waiting,))
    _old_producer_promotion(conn, created_waiting)

    # The current producer annotates both shapes.
    annotated_breaker = _breaker_parked(conn, "annotated breaker")
    annotated_dependency, parent = _dependency_parked(conn, "annotated dependency release")
    assert kb.complete_task(conn, parent, result="ok", force=True) is True
    kb.recompute_ready(conn)

    res = kbd.dispatch_once(conn, dry_run=True)
    spawned, guarded = [s[0] for s in res.spawned], dict(res.respawn_guarded)
    for tid in (legacy_breaker, stamped_breaker, waited_then_tripped, annotated_breaker):
        assert guarded.get(tid) == "active_pr", tid
        assert tid not in spawned
    for tid in (legacy_dependency, created_waiting, annotated_dependency):
        assert tid in spawned, tid
    assert _promotion_payloads(conn, annotated_breaker) == [{"from": "blocked"}]
    assert _promotion_payloads(conn, annotated_dependency) == [{"from": "todo"}]


def test_release_persists_across_retries_until_a_newer_pr_comment(conn) -> None:
    """APR-02, the stated semantics: a release is not consumed by the claim it allowed. The
    retry's crash neither lifts the guard nor cancels the release; a newer PR comment does."""
    tid, parent = _dependency_parked(conn, "released then crashed")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
    _old_producer_promotion(conn, tid)
    assert tid in [s[0] for s in kbd.dispatch_once(conn, dry_run=True).spawned]
    kb.claim_task(conn, tid)
    assert not kbd._record_task_failure(conn, tid, "worker crashed", outcome="crashed",
                                        failure_limit=5, release_claim=True, end_run=True)
    assert kb.get_task(conn, tid).status == "ready"
    assert kbd.check_respawn_guard(conn, tid) is None
    kb.add_comment(conn, tid, author="dev", body="Opened https://github.com/example/repo/pull/45")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert dict(kbd.dispatch_once(conn, dry_run=True).respawn_guarded).get(tid) == "active_pr"
