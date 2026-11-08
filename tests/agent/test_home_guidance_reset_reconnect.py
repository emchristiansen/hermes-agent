"""A reset in place after a reconnect ends the conversation every resume reads (R7-RESET-03).

With ``compression.in_place: false`` a rotation moves the live agent to a compression child. A reconnect restores the
ACP-facing root and reopens it (clearing its compression end marker), so the root's compression lineage no longer
reaches the old tip, while the resume selector (``SessionDB.resolve_resume_session_id``) still resolves the root to
that tip. ``/reset`` must end that tip and its lineage too: no resume may bring back the ended conversation or its A
decision, and unrelated sessions and branch children stay untouched.
"""
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agent import home_guidance as hg
from tests.agent.test_home_guidance import CORE_B, SOURCE, _agent, _expected_block, _turn
from tests.agent.test_home_guidance_boundaries import _next_turn, _real_db, _rotate_by_publication, _write_config

pytest_plugins = ["tests.agent.test_home_guidance"]

ROOT = "acp-root"


def _conversation(home, db, rotations, factory):
    """A live ACP agent decides A, persists a turn and rotates *rotations* times by real publication; a branch child
    of the root and an unrelated session each hold their own conversation. Returns (agent, segments)."""
    agent = factory()
    assert _turn(agent, []).count(_expected_block(home.core)) == 1
    agent._persist_session([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "old reply"}], [])
    for other, parent in (("unrelated", None), ("branch", ROOT)):
        db.create_session(other, source="acp", parent_session_id=parent,
                          model_config={"_branched_from": ROOT} if parent else None)
        db.append_message(other, "user", f"{other} question")
        db.append_message(other, "assistant", f"{other} reply")
        db.update_system_prompt(other, f"{other} prompt")
    segments = [ROOT]
    for i in range(rotations):
        _rotate_by_publication(db, segments[-1], f"{ROOT}-{i + 1}", agent)
        segments.append(f"{ROOT}-{i + 1}")
    return agent, segments


def _untouched(db):
    return {sid: (db.get_session(sid)["system_prompt"], db.get_messages_as_conversation(sid))
            for sid in ("unrelated", "branch")}


def _manager_with(db, factory, agent, reconnect):
    """The ACP session manager holding the live agent, or — after a reconnect — a new manager that restored the root
    from the database (the real ``SessionManager.get_session`` restore, which reopens the root)."""
    from acp_adapter.session import SessionManager
    manager = SessionManager(db=db, agent_factory=factory)
    state = SimpleNamespace(agent=agent, session_id=ROOT, cwd="", model=None,
                            history=db.get_messages_as_conversation(agent.session_id))
    manager._sessions[ROOT] = state
    if not reconnect:
        return manager, state
    manager.end_all_sessions()
    agent.close()
    manager = SessionManager(db=db, agent_factory=factory)
    return manager, manager.get_session(ROOT)


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("rotations", [0, 1, 2])
def test_a_reset_after_a_reconnect_ends_the_conversation_every_resume_reads(home, tmp_path, no_network, rotations,
                                                                            reconnect):
    """Live or reconnected, after 0-2 rotations: the real ``/reset`` reports success; the session the resume selector
    resolves for the root has no model or display history and no stored prompt; every segment keeps its rows on disk
    and its usage; the branch child and the unrelated session are untouched; the next build there takes up B."""
    from acp_adapter.commands import SlashCommandsMixin
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)

    def factory():
        created = _agent(platform="acp", session_id=ROOT, session_db=db)
        created.compression_enabled = False
        return created
    agent = None
    try:
        agent, segments = _conversation(home, db, rotations, factory)
        for sid in segments:
            db.update_token_counts(sid, input_tokens=5, output_tokens=2, api_call_count=1, source="claude-code")
        on_disk = {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments}
        others = _untouched(db)
        manager, state = _manager_with(db, factory, agent, reconnect)
        agent = state.agent
        if reconnect and rotations:  # the shape the reset must cover: root cut from its lineage, resume at the tip
            assert db.get_compression_lineage(ROOT) == [ROOT] and db.resolve_resume_session_id(ROOT) == segments[-1]
        assert SlashCommandsMixin._cmd_reset(SimpleNamespace(session_manager=manager), "", state) \
            == "Conversation history cleared."
        resumed = db.resolve_resume_session_id(ROOT)
        assert db.get_resume_conversations(resumed) == ([], [])
        for sid in segments:
            row = db.get_session(sid)
            assert row["system_prompt"] is None and db.get_messages_as_conversation(sid, include_compacted=True) == []
            assert len(db.get_messages(sid, include_inactive=True)) == on_disk[sid]
            assert (row["api_call_count"], row["input_tokens"], row["output_tokens"]) == (1, 5, 2)
        assert _untouched(db) == others
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, resumed, "construction")
        assert "Fixture rule B" in sent and isinstance(decided, hg.Carried)
    finally:
        if agent is not None:
            agent.close()
        db.close()


def test_a_reconnected_reset_that_cannot_end_the_tip_reports_failure_and_a_retry_finishes(home, tmp_path, no_network,
                                                                                         monkeypatch):
    """After two rotations and a reconnect, ending the resumed tip fails: ``/reset`` reports its reset-failed
    diagnostic, never success, and the tip keeps its prompt; a retry once the write succeeds ends it."""
    from acp_adapter.commands import SlashCommandsMixin
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)

    def factory():
        return _agent(platform="acp", session_id=ROOT, session_db=db)
    agent = None
    try:
        agent, segments = _conversation(home, db, 2, factory)
        manager, state = _manager_with(db, factory, agent, True)
        agent = state.agent
        end = db.end_conversation_in_place
        refuse = {segments[-1]}

        def end_or_refuse(session_id):
            if session_id in refuse:
                raise RuntimeError("database is locked")
            end(session_id)
        monkeypatch.setattr(db, "end_conversation_in_place", end_or_refuse)
        reset = SimpleNamespace(session_manager=manager)
        assert SlashCommandsMixin._cmd_reset(reset, "", state) \
            == "Conversation history was not cleared: agent session state reset failed; see logs."
        assert db.get_session(segments[-1])["system_prompt"]
        refuse.clear()
        assert SlashCommandsMixin._cmd_reset(reset, "", state) == "Conversation history cleared."
        assert db.get_resume_conversations(db.resolve_resume_session_id(ROOT)) == ([], [])
        assert all(db.get_session(sid)["system_prompt"] is None for sid in segments)
    finally:
        if agent is not None:
            agent.close()
        db.close()


def _reconnected_after_two_rotations(home, db, factory):
    """Two rotations, then a reconnect: returns (manager, state, segments, rows on disk per segment)."""
    agent, segments = _conversation(home, db, 2, factory)
    manager, state = _manager_with(db, factory, agent, True)
    return manager, state, segments, {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments}


def test_a_reset_that_cannot_read_what_a_resume_selects_reports_failure(home, tmp_path, no_network, monkeypatch):
    """R8-RESET-03: the resume selector's child query fails during ``/reset`` (a real SQLite error raised by the
    database read): the reset reports failure, never success, while the tip a resume reads still holds A; every row
    stays on disk and the session keeps its history. Ordinary readers stay best effort (the root). With the read
    working again a retry clears it and the next build takes up B."""
    from acp_adapter.commands import SlashCommandsMixin
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)

    def factory():
        return _agent(platform="acp", session_id=ROOT, session_db=db)
    state = None
    try:
        manager, state, segments, on_disk = _reconnected_after_two_rotations(home, db, factory)
        history = list(state.history)
        read = db._read_ctx

        class FailingChildQuery:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, *args):
                if "FROM sessions AS child" in sql:
                    raise sqlite3.OperationalError("disk I/O error")
                return self.conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self.conn, name)

        @contextmanager
        def failing_read(*args, **kwargs):
            with read(*args, **kwargs) as conn:
                yield FailingChildQuery(conn)
        monkeypatch.setattr(db, "_read_ctx", failing_read)
        assert db.resolve_resume_session_id(ROOT) == ROOT
        reset = SimpleNamespace(session_manager=manager)
        assert SlashCommandsMixin._cmd_reset(reset, "", state) \
            == "Conversation history was not cleared: agent session state reset failed; see logs."
        monkeypatch.setattr(db, "_read_ctx", read)
        assert db.get_session(segments[-1])["system_prompt"] and state.history == history
        assert {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments} == on_disk
        assert SlashCommandsMixin._cmd_reset(reset, "", state) == "Conversation history cleared."
        resumed = db.resolve_resume_session_id(ROOT)
        assert db.get_resume_conversations(resumed) == ([], [])
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, resumed, "construction")
        assert "Fixture rule B" in sent and isinstance(decided, hg.Carried)
    finally:
        if state is not None:
            state.agent.close()
        db.close()


def test_a_reset_the_database_refuses_keeps_every_row_and_a_retry_finishes(home, tmp_path, no_network):
    """R8-RESET-04: a real SQLite trigger refuses the root's prompt-clearing write during ``/reset`` after a
    reconnect (the restored agent does not own persistence): the reset reports failure, the session keeps its
    history, and no save replaces the root's rows with the cleared history. Once the write is allowed, a retry
    reports success; every row of every segment is still on disk, and no resume holds the ended conversation."""
    from acp_adapter.commands import SlashCommandsMixin
    _write_config(home, home_guidance_file=SOURCE)
    db = _real_db(tmp_path)

    def factory():
        return _agent(platform="acp", session_id=ROOT, session_db=db)
    state = None
    try:
        manager, state, segments, on_disk = _reconnected_after_two_rotations(home, db, factory)
        assert on_disk[ROOT] > 0
        history = list(state.history)
        db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER refuse_reset BEFORE UPDATE OF system_prompt ON sessions "
            f"WHEN OLD.id = '{ROOT}' AND NEW.system_prompt IS NULL BEGIN SELECT RAISE(ABORT, 'refused'); END"))
        reset = SimpleNamespace(session_manager=manager)
        reply = SlashCommandsMixin._cmd_reset(reset, "", state)
        assert {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments} == on_disk
        assert reply == "Conversation history was not cleared: agent session state reset failed; see logs."
        assert state.history == history
        db._execute_write(lambda conn: conn.execute("DROP TRIGGER refuse_reset"))
        assert SlashCommandsMixin._cmd_reset(reset, "", state) == "Conversation history cleared."
        assert {sid: len(db.get_messages(sid, include_inactive=True)) for sid in segments} == on_disk
        assert db.get_resume_conversations(db.resolve_resume_session_id(ROOT)) == ([], [])
        assert all(db.get_session(sid)["system_prompt"] is None for sid in segments)
    finally:
        if state is not None:
            state.agent.close()
        db.close()
