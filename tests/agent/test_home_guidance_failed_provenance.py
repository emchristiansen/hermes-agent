"""A failed-turn notice that stands for a real reply (R7-FAILED-01, R7-FAILED-02).

A failed-turn notice is written without a turn of its conversation producing it, so alone it is no reply. But a closer
can write it on a durable tail while the turn's abandoned executor is still running (a gateway inactivity timeout:
the interrupt handler then persists its streamed partial reply), and the unattended owner stop writes it even on a
closed tail (after a real provisional verification answer). Live replay then folds the late reply into the notice,
or supersedes the verification answer with it, and compaction retires the original reply rows. The surviving notice
must still say it stands for a reply; a history of notices and mirrors alone stays eligible for its first decision.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from agent import home_guidance as hg
from tests.agent.test_home_guidance import CORE_B, _agent, _expected_block, _turn
from tests.agent.test_home_guidance_boundaries import (MIRROR_ROW, MIRRORS, USER_Q, _mirror, _mirror_store, _next_turn,
                                                       _unavailable_only)

pytest_plugins = ["tests.agent.test_home_guidance"]

NOTICE = {"role": "assistant", "content": "Your request was not processed. Send it again if you still want me to "
                                          "carry it out.", "display_kind": "failed_turn"}
LEGACY_NOTICE = {"role": "assistant", "content": NOTICE["content"]}
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}


# ── The fold owner, on its own ────────────────────────────────────────────────────────────────────────────────────────

# rows after the user question -> whether the result still holds nothing but rows written without a turn
NOTICE_FOLDS = {"notice+text": ([NOTICE, {"role": "assistant", "content": "the late partial reply"}], False),
                "notice+empty": ([NOTICE, {"role": "assistant", "content": ""}], False),
                "notice+tool-call": ([NOTICE, {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
                                      {"role": "tool", "content": "ran", "tool_call_id": "c1"}], False),
                "legacy-notice+text": ([LEGACY_NOTICE, {"role": "assistant", "content": "the late reply"}], False),
                "text+notice": ([{"role": "assistant", "content": "the reply"}, NOTICE], False),
                "verification+notice": ([{"role": "assistant", "content": "the interim answer",
                                          "finish_reason": "verification_required"}, NOTICE], False),
                "verification+legacy-notice": ([{"role": "assistant", "content": "the interim answer",
                                                 "finish_reason": "verification_required"}, LEGACY_NOTICE], False),
                "notice+notice": ([NOTICE, {**NOTICE, "content": "Interrupted."}], True),
                "notice+mirror": ([NOTICE, MIRROR_ROW], True),
                "mirror+notice": ([MIRROR_ROW, NOTICE], True),
                "notice+mirror+text": ([NOTICE, MIRROR_ROW, {"role": "assistant", "content": "the late reply"}],
                                       False)}


@pytest.mark.parametrize("fold", sorted(NOTICE_FOLDS))
def test_a_notice_survivor_says_whether_it_stands_for_a_reply(fold):
    """The real alternation repair (fold and verification supersede): a notice survivor that absorbed a row a turn
    wrote is a reply, and keeps the notice's text and type for display; one that absorbed only notices or mirrors is
    not. A copy made before the repair keeps its own metadata, and no reply row's uid is claimed by a supersede."""
    from agent.agent_runtime_helpers import repair_message_sequence
    rows, turnless = NOTICE_FOLDS[fold]
    messages = [dict(USER_Q)] + [{**row, "message_uid": f"uid-{i}"} for i, row in enumerate(rows)]
    copies = [{**m} for m in messages]
    assert repair_message_sequence(None, messages) >= 1
    survivor = messages[1]
    assert [m["role"] for m in messages[:2]] == ["user", "assistant"]
    assert hg._replies_to_a_user(messages) is not turnless
    assert [m.get("display_metadata") for m in copies] == [None] + [r.get("display_metadata") for r in rows]
    if fold.startswith("verification"):
        assert survivor["content"] == rows[1]["content"] and "_absorbed_message_uids" not in survivor
    if fold.startswith(("notice+", "verification+notice")):
        assert survivor["display_kind"] == "failed_turn"


# ── Real producers, live replay, compaction and the next decision ─────────────────────────────────────────────────────

def _gateway_close(store, session_id, notice):
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session import AsyncSessionStore
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store))
    asyncio.run(GatewayTurnPersistenceMixin._hmwa_close_failed_turn(runner, session_id, notice))


def _decided_with_q(home, db, session_id, *, big=False):
    """A real first build decides A and stores its prompt; the turn's own flush persists Q."""
    agent = _agent(session_id=session_id, session_db=db)
    assert _turn(agent, []).count(_expected_block(home.core)) == 1
    messages = [{"role": "user", "content": "question " + ("long pasted log line " * 3000 if big else "")}]
    agent._persist_session(messages, [])
    return agent, messages


def _late_partial(agent, messages, partial="Here is the first half of the answer you asked"):
    """The provider call's real interrupt handler, as the abandoned turn thread runs it once the gateway's hard
    interrupt lands: it appends the streamed partial and persists it through the agent's own flush."""
    from agent.turn_api_call import handle_api_interrupt
    agent._current_streamed_assistant_text = partial
    verdict = handle_api_interrupt(agent, _retry=SimpleNamespace(), thinking_spinner=None, messages=messages,
                                   conversation_history=[], api_start_time=time.time() - 600, interrupted=False,
                                   final_response=None)
    assert verdict.action == "break" and verdict.final_response == partial


def _unattended_notice(db, session_id):
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_results import close_discarded_turn
    assert db._execute_write(lambda conn: close_discarded_turn(
        db, conn, {"target_session_id": session_id}, AUTOMATION_INTERRUPTED_NOTICE))


def _verification_candidate(agent, messages):
    """A real provisional verification answer, flushed by the stop gate's interim path."""
    from agent.turn_stop_gates import _append_interim_answer
    _append_interim_answer(agent, {"role": "assistant", "content": "the interim answer",
                                   "finish_reason": "verification_required"}, messages, [], "interim flush failed")


def _produce(home, store, session_id, shape, *, big=False):
    """On disk after Q: ``late-partial`` (gateway closer notice F on the open Q tail, then the abandoned turn's
    partial P), ``partial-then-notice`` (P, then the unattended notice on the closed tail), ``verification-notice``
    (V, then the unattended notice), ``notice-only`` (the gateway closer's notice, no agent output)."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    db = store._db
    db.create_session(session_id, source="telegram")
    agent, messages = _decided_with_q(home, db, session_id, big=big)
    try:
        if shape == "late-partial":
            _gateway_close(store, session_id, FAILED_TURN_NOTICE)
            _late_partial(agent, messages)
        elif shape == "partial-then-notice":
            _late_partial(agent, messages)
            _unattended_notice(db, session_id)
        elif shape == "verification-notice":
            _verification_candidate(agent, messages)
            _unattended_notice(db, session_id)
        else:
            _gateway_close(store, session_id, FAILED_TURN_NOTICE)
    finally:
        agent.close()
    roles = [(m["role"], m.get("display_kind")) for m in db.get_messages(session_id)]
    # after the turn harness's own opening user row
    expected = {"late-partial": [("user", None), ("assistant", "failed_turn"), ("assistant", None)],
                "partial-then-notice": [("user", None), ("assistant", None), ("assistant", "failed_turn")],
                "verification-notice": [("user", None), ("assistant", None), ("assistant", "failed_turn")],
                "notice-only": [("user", None), ("assistant", "failed_turn")]}[shape]
    assert roles[-len(expected):] == expected and len(roles) == len(expected) + 1


def _compact_live_tail(db, session_id, tail):
    """In-place compaction carrying the live-replay tail (``archive_and_compact`` retires the carried originals)."""
    live = db.get_messages_as_conversation(session_id, repair_alternation=True, include_row_ids=True)
    assert [m["role"] for m in live] == ["user", "assistant"]
    compacted = [{"role": "user", "content": "[summary] earlier facts"}, *live[-tail:]]
    if tail == 1:
        compacted.insert(1, {"role": "user", "content": "continue"})
    db.archive_and_compact(session_id, compacted, tail_count=tail)


def _compress_through_the_facade(db, session_id, live):
    """The real compression facade (``_compress_context``: commit fence, guards, durable commit) over *live*, with
    only the engine's returned list substituted, tagging its tail as the compressor does."""
    from unittest.mock import MagicMock
    from agent.conversation_compression import CompressionCommitFence
    agent = _agent(session_id=session_id, session_db=db)
    try:
        _turn(agent, live)
        engine = agent.context_compressor
        engine.compress = MagicMock(return_value=[
            {"role": "user", "content": "[Recent Summary (d0, node 490)]\n\n" + "retained earlier facts " * 60},
            {**live[-2], "_compaction_tail": True}, {**live[-1], "_compaction_tail": True}])
        engine.compression_count, engine.last_prompt_tokens, engine.last_completion_tokens = 1, 0, 0
        engine._last_summary_error, engine._last_compress_aborted, engine._last_summary_auth_failure = None, False, False
        engine._last_aux_model_failure_model = engine._last_aux_model_failure_error = None
        agent._persist_user_message_idx = len(live) - 1
        agent._compress_context(live, "", approx_tokens=120_000, commit_fence=CompressionCommitFence())
        assert engine.compress.called
    finally:
        agent.close()
    return db.get_compression_tip(session_id) or session_id


def _carry(store, session_id, carried):
    """Leave the conversation raw, compact its live-replay tail in place, or compress it through the facade over
    the turn lease's read (row ids) or the gateway's read (``load_transcript``). Returns the session read next."""
    db = store._db
    if carried in ("tail-1", "tail-2"):
        _compact_live_tail(db, session_id, int(carried[-1]))
    elif carried.startswith("facade"):
        db.append_message(session_id, "user", "continue the same task")
        live = (store.load_transcript(session_id) if carried == "facade-gateway" else
                db.get_messages_as_conversation(session_id, repair_alternation=True, include_row_ids=True))
        return _compress_through_the_facade(db, session_id, live)
    return session_id


REPLY_SHAPES = ["late-partial", "partial-then-notice", "verification-notice"]


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("carried", ["raw", "tail-1", "tail-2", "facade-lease", "facade-gateway"])
@pytest.mark.parametrize("shape", REPLY_SHAPES)
def test_a_reply_behind_a_notice_keeps_the_conversation_under_way(home, no_network, monkeypatch, shape, carried,
                                                                  entry):
    """Real producers write Q and a real reply with a failed-turn notice before or after it; live replay folds or
    supersedes them; the conversation is carried raw, by in-place compaction or by the compression facade; the stored
    prompt is lost and file B is configured: the next turn carries an explicit notice, never B."""
    store = _mirror_store(home, monkeypatch)
    session_id = f"{shape}-{carried}"
    try:
        _produce(home, store, session_id, shape, big=carried.startswith("facade"))
        assert hg.continues_without_prompt(store._db.get_session(session_id), store.load_transcript(session_id))
        target = _carry(store, session_id, carried)
        db = store._db
        db.update_system_prompt(target, None)
        home.core.write_text(CORE_B, encoding="utf-8")
        sent, decided = _next_turn(db, target, entry)
        assert "Fixture rule B" not in sent and _unavailable_only(sent) and isinstance(decided, hg.Unavailable)
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize("entry", ["construction", "turn"])
@pytest.mark.parametrize("carried", ["raw", "tail-1"])
@pytest.mark.parametrize("shape", ["notice-only", "mirror-then-unattended-notice"])
def test_notices_and_mirrors_alone_leave_the_first_decision_open(home, no_network, monkeypatch, shape, carried,
                                                               entry):
    """Controls: a stored first request then only the gateway closer's notice (no agent was built: no prompt stored),
    or a request, an outgoing mirror and the unattended notice; raw or after compaction, the first actual turn takes
    up the configured file."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    store = _mirror_store(home, monkeypatch)
    db = store._db
    session_id = f"{shape}-{carried}"
    try:
        db.create_session(session_id, source="telegram")
        db.append_message(session_id, "user", "first request")
        if shape == "notice-only":
            _gateway_close(store, session_id, FAILED_TURN_NOTICE)
        else:
            _mirror(session_id, MIRRORS["outgoing"])
            _unattended_notice(db, session_id)
        assert [m["role"] for m in store.load_transcript(session_id)] == ["user", "assistant"]
        if carried == "tail-1":
            _compact_live_tail(db, session_id, 1)
        db.append_message(session_id, "user", "First actual request to this agent")
        sent, decided = _next_turn(db, session_id, entry)
        assert sent.count(_expected_block(home.core)) == 1 and isinstance(decided, hg.Carried)
    finally:
        store.close_all_db_handles()


def test_the_mark_reaches_no_provider_request(home, no_network, monkeypatch):
    """Q, F, P from the real producers, then a new question; the request the next turn builds from the gateway's
    folded read (``build_api_messages``) carries the notice and the partial reply but no ``display_metadata`` and so
    no mark, and the caller's marked row keeps its mark."""
    from agent.message_metadata import STANDS_FOR_TURN_ROW
    from agent.turn_context import build_api_messages
    store = _mirror_store(home, monkeypatch)
    db = store._db
    try:
        _produce(home, store, "wire", "late-partial")
        db.append_message("wire", "user", "next question")
        history = store.load_transcript("wire")
        survivor = history[-2]
        assert survivor["display_kind"] == "failed_turn" and survivor["display_metadata"] == {STANDS_FOR_TURN_ROW: True}
        agent = _agent(session_id="wire", session_db=db)
        try:
            prompt = _turn(agent, history)
            api, _system = build_api_messages(agent, history, current_turn_user_idx=len(history) - 1,
                                              ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
                                              active_system_prompt=prompt)
        finally:
            agent.close()
        wire = json.dumps(api, default=str)
        assert "Here is the first half of the answer you asked" in wire and survivor["content"].split("\n")[0] in wire
        assert "display_metadata" not in wire and STANDS_FOR_TURN_ROW not in wire and "display_kind" not in wire
        assert any(m is survivor for m in history) and survivor["display_metadata"] == {STANDS_FOR_TURN_ROW: True}
    finally:
        store.close_all_db_handles()
