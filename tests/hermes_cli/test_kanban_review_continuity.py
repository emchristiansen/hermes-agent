"""Reviewer continuation on the board: the handoff opt-in is validated where CLI and tool share
it, and a review run resolves the earlier review session it continues, read-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_review_handoff import review_continuity_source


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _running(conn, title="card", assignee="impl"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.recompute_ready(conn)
    return kb.claim_task(conn, tid)


def _hand_off(conn, task, metadata=None, reviewer="rev"):
    ok, reason = kb.request_review(conn, task.id, summary="done", metadata=metadata, reviewer=reviewer,
                                   expected_run_id=task.current_run_id, with_reason=True)
    assert ok, reason


def _review(conn, tid, session=None):
    """Claim the review lane; *session* stamps the owner admission the run's session got."""
    task = kb.claim_review_task(conn, tid)
    if session is not None:
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "owner_admitted", {"db": "/owner/state.db", "session_id": session,
                             "request_id": "kanban:x"}, run_id=task.current_run_id)
    return task


@pytest.mark.parametrize("metadata", [
    {"reviewer_context": "yes"},
    {"reviewer_context": 1},
    {"review_continues_task": "OTHER"},
    {"reviewer_context": "continue", "review_continues_task": "SELF"},
    {"reviewer_context": "continue", "review_continues_task": "t_missing"},
    {"reviewer_context": "continue", "review_continues_task": "  "},
    {"reviewer_context": "continue", "review_continues_task": ["OTHER"]},
])
def test_invalid_continuity_metadata_is_refused_before_any_change(conn, metadata):
    other = _running(conn, title="spec")
    task = _running(conn)
    metadata = {k: {"OTHER": other.id, "SELF": task.id}.get(v, v) if isinstance(v, str) else v
                for k, v in metadata.items()}
    events = kb.list_events(conn, task.id)
    ok, reason = kb.request_review(conn, task.id, summary="done", metadata=metadata, reviewer="rev",
                                   expected_run_id=task.current_run_id, with_reason=True)
    assert not ok and "metadata." in reason
    after = kb.get_task(conn, task.id)
    assert (after.status, after.assignee, after.current_run_id) == ("running", "impl", task.current_run_id)
    assert kb.list_events(conn, task.id) == events


def test_valid_continuity_metadata_is_stored_untouched(conn):
    spec = _running(conn, title="spec")
    task = _running(conn)
    metadata = {"reviewer_context": "continue", "review_continues_task": spec.id, "tests_run": ["a", "b"]}
    _hand_off(conn, task, metadata)
    run = conn.execute("SELECT outcome, metadata FROM task_runs WHERE id=?", (task.current_run_id,)).fetchone()
    assert run["outcome"] == "review_requested" and json.loads(run["metadata"]) == metadata


def test_no_opt_in_resolves_nothing(conn):
    task = _running(conn)
    _hand_off(conn, task)
    first = _review(conn, task.id, session="s-first")
    kb.request_changes(conn, task.id, reason="fix it", expected_run_id=first.current_run_id)
    _hand_off(conn, kb.claim_task(conn, task.id), {"tests_run": ["x"]})
    second = _review(conn, task.id)
    assert review_continuity_source(conn, task.id, second.current_run_id) is None


def test_same_card_correction_continues_previous_review(conn):
    task = _running(conn)
    _hand_off(conn, task)
    first = _review(conn, task.id, session="s-first")
    kb.request_changes(conn, task.id, reason="fix it", expected_run_id=first.current_run_id)
    _hand_off(conn, kb.claim_task(conn, task.id), {"reviewer_context": "continue"})
    second = _review(conn, task.id)
    assert review_continuity_source(conn, task.id, second.current_run_id) == {
        "task_id": task.id, "run_id": first.current_run_id, "session_id": "s-first", "db": "/owner/state.db",
        "run_ended": True}


def test_blocked_review_resumes_from_last_admitted_review_run(conn):
    task = _running(conn)
    _hand_off(conn, task, {"reviewer_context": "continue", "free": 1})
    admitted = _review(conn, task.id, session="s-admitted")
    kb.block_task(conn, task.id, reason="wait", expected_run_id=admitted.current_run_id)
    kb.unblock_task(conn, task.id)
    _review(conn, task.id)  # reclaimed before its owner admitted it: nothing to continue
    assert kb.reclaim_task(conn, task.id, reason="lost launcher")
    resumed = _review(conn, task.id)
    source = review_continuity_source(conn, task.id, resumed.current_run_id)
    assert source["run_id"] == admitted.current_run_id and source["session_id"] == "s-admitted"


def test_cross_card_continues_named_spec_review(conn):
    spec = _running(conn, title="spec")
    _hand_off(conn, spec)
    spec_review = _review(conn, spec.id, session="s-spec")
    kb.complete_task(conn, spec.id, summary="spec ok", expected_run_id=spec_review.current_run_id)
    impl = _running(conn, title="impl")
    _hand_off(conn, impl, {"reviewer_context": "continue", "review_continues_task": spec.id})
    review = _review(conn, impl.id)
    assert review_continuity_source(conn, impl.id, review.current_run_id) == {
        "task_id": spec.id, "run_id": spec_review.current_run_id, "session_id": "s-spec", "db": "/owner/state.db",
        "run_ended": True}


def test_changed_profile_and_missing_predecessor_are_unavailable(conn):
    spec = _running(conn, title="spec")
    _hand_off(conn, spec)
    spec_review = _review(conn, spec.id, session="s-spec")
    kb.complete_task(conn, spec.id, summary="spec ok", expected_run_id=spec_review.current_run_id)
    impl = _running(conn, title="impl")
    _hand_off(conn, impl, {"reviewer_context": "continue", "review_continues_task": spec.id}, reviewer="other-rev")
    review = _review(conn, impl.id)
    assert review_continuity_source(conn, impl.id, review.current_run_id) == {"unavailable": "changed_profile"}
    lone = _running(conn, title="lone")
    _hand_off(conn, lone, {"reviewer_context": "continue"})
    review = _review(conn, lone.id)
    assert review_continuity_source(conn, lone.id, review.current_run_id) == {"unavailable": "no_predecessor"}


def test_cross_card_source_still_running_is_marked_not_ended(conn):
    spec = _running(conn, title="spec")
    _hand_off(conn, spec)
    spec_review = _review(conn, spec.id, session="s-spec")
    impl = _running(conn, title="impl")
    _hand_off(conn, impl, {"reviewer_context": "continue", "review_continues_task": spec.id})
    review = _review(conn, impl.id)
    source = review_continuity_source(conn, impl.id, review.current_run_id)
    assert source["run_id"] == spec_review.current_run_id and source["run_ended"] is False
