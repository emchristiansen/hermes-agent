"""A steer accepted while turn-start compression runs, and the steer after it, through the real loop.

Live shape (Agonist, 2026-11-05 09:17-09:21): the history ends with an unanswered user row (a
background-process notice after an interrupted reply), so this turn's inbound message follows a user
row. A scheduled wake was handed to the running turn as a ticketed steer while the preflight summary
call was in flight; the gateway booked it ``steered``.

- A4: the first iteration's pre-API drain spliced the wake into history, after the newest tool row
  of an EARLIER turn (before that turn's assistant reply), where the steer contract tells the model it
  is an already-handled replay. The turn never acted on it.
- A5: the alternation repair then folded this turn's user row into the unanswered notice, re-anchoring
  found no verbatim row and fell back to the steer row, and every flush wrote steer rows through the
  current-turn persist override: each steer row in state.db holds the inbound message's text and
  timestamp, and the steer texts are in no row.

These tests run the real AIAgent turn loop, the built-in compressor (only its summary LLM call is
stubbed, and the steer lands inside it), the real steer buffer and drains, sequence repair and SessionDB.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.prompt_builder import STEER_DISPLAY_KIND, steer_user_row
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.registry import registry

_PROBE_TOOL = "steer_compress_probe"


def _probe_tool(args, **_kwargs):
    return "probe ok"


registry.register(
    name=_PROBE_TOOL,
    toolset="utility",
    schema={
        "name": _PROBE_TOOL,
        "description": "probe tool for the steer-during-compression regression",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=_probe_tool,
    override=True,
)

SESSION = "steer-preflight-compress"
INBOUND = "FromHeartbeat: remaining blocker 7941 on t_2dfcaea0 is READY/noRun after the restart."
INBOUND_TS = 1793899065.285853
WAKE = "theseus-agonist-wake: scheduled pass 778e3320, run the Agonist pass now."
FOLLOW = "Family2394 recovery collision now observed: promote succeeded, readback shows the same run."
BG_NOTICE = (
    "[IMPORTANT: Background process proc_8e42c774efd5 completed normally (exit code 0). "
    "Command: probe]\nOutput: done"
)


def _config() -> dict:
    return {
        "compression": {
            "enabled": True, "threshold": 0.50, "target_ratio": 0.20,
            "protect_first_n": 2, "protect_last_n": 6, "max_attempts": 1,
        },
        "prompt_caching": {"cache_ttl": "5m"}, "sessions": {}, "bedrock": {},
    }


@contextlib.contextmanager
def _probe_tools_only():
    tool_schema = {"type": "function", "function": {
        "name": _PROBE_TOOL, "description": "probe tool for the steer-during-compression regression",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_schema]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        yield


def _make_agent(monkeypatch, tmp_path: Path) -> tuple[AIAgent, SessionDB]:
    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", _config)
    monkeypatch.setattr(config_mod, "load_config_readonly", _config)
    db = SessionDB(db_path=tmp_path / "state.db")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        _probe_tools_only(),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            base_url="https://openrouter.ai/api/v1", api_key="test-key", model="test/model",
            quiet_mode=True, skip_memory=True, skip_context_files=True,
            session_db=db, session_id=SESSION,
        )
    assert _PROBE_TOOL in agent.valid_tool_names
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    return agent, db


def _history(db: SessionDB) -> list[dict]:
    """Older turns, then the live tail: tool call, its result, an interrupted reply, an unanswered notice."""
    rows: list[dict] = []
    for i in range(12):
        rows.append({"role": "user", "content": f"older question {i} " + "x" * 400})
        rows.append({"role": "assistant", "content": f"older answer {i} " + "y" * 400})
    rows += [
        {"role": "user", "content": "check the background job"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_old", "type": "function",
            "function": {"name": _PROBE_TOOL, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_old", "content": "Background process started"},
        {"role": "assistant", "content": "An automated delivery was interrupted and was not run again."},
        {"role": "user", "content": BG_NOTICE},
    ]
    db.create_session(session_id=SESSION, source="local")
    for row in rows:
        db.append_message(SESSION, role=row["role"], content=row["content"],
                          tool_calls=row.get("tool_calls"), tool_call_id=row.get("tool_call_id"))
    return db.get_messages_as_conversation(SESSION)


def _tool_call(call_id: str):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=_PROBE_TOOL, arguments="{}"))


def _run_turn(monkeypatch, tmp_path, *, tool_rounds: int = 2, compress: bool = True, loop_compress: bool = False):
    from tests.agent.test_run_agent import _mock_response

    agent, db = _make_agent(monkeypatch, tmp_path)
    history = _history(db)
    payloads: list[list[dict]] = []

    def summary_call(_self, _prompt, _started):
        # The gateway's transfer of the queued wake admission to the running claim (ticketed by the
        # admission id, delivered by Hermes automation) lands while the summary call is in flight.
        assert agent.steer(WAKE, ticket="admission-481", delivered_by={}) is True
        compressor.threshold_tokens = 10**9  # one pass, as live: no further compaction this turn
        return "## Summary\nOlder questions were answered."

    def model_call(api_kwargs):
        payloads.append([dict(m) for m in api_kwargs["messages"]])
        if len(payloads) > tool_rounds:
            return _mock_response(content="Handled.", finish_reason="stop")
        if len(payloads) == 1:
            if not compress:  # no summary call to land in: the wake is handed over during the first request
                assert agent.steer(WAKE, ticket="admission-481", delivered_by={}) is True
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[_tool_call("call_1")])
        if len(payloads) == 2:
            # The next admission is transferred while this request is in flight; the post-batch drain
            # delivers it ("Delivered /steer ... as new user message").
            assert agent.steer(FOLLOW, ticket="admission-482", delivered_by={}) is True
        return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[_tool_call("call_2")])

    agent._interruptible_api_call = model_call
    compressor = agent.context_compressor
    # loop_compress (with compress=False): no turn-start pass; the loop's pre-API gate compacts after the merge.
    compressor.threshold_tokens = 2_000 if compress or loop_compress else 10**9
    compressor.note_usage_less_response()
    estimates = iter([50_000] if compress else [])

    def estimate(*_a, **_k):
        return next(estimates, 500)

    with (
        patch("agent.turn_context.estimate_request_tokens_rough", side_effect=estimate),
        patch.object(type(compressor), "_call_summary_llm", summary_call),
        patch.object(agent, "_cleanup_task_resources"),
        _probe_tools_only(),
    ):
        result = agent.run_conversation(
            "[gateway context]\n" + INBOUND, conversation_history=history,
            persist_user_message=INBOUND, persist_user_timestamp=INBOUND_TS,
            persist_user_display_kind="internal_notification",
            persist_user_platform_id="bot:63e1988df0cc4c4b9e82e84ddd646f62",
        )
    rows = db.get_messages(SESSION)
    return agent, db, result, payloads, rows


def _turn_start(payload: list[dict]) -> int:
    """Index of this turn's user row on the wire (it carries the inbound text)."""
    return max(i for i, m in enumerate(payload) if m.get("role") == "user" and INBOUND in str(m.get("content")))


def test_steer_accepted_during_preflight_compression_reaches_this_turn(monkeypatch, tmp_path):
    """A4: the wake booked ``steered`` must reach the model inside this turn, after this turn's
    user row and after one of this turn's tool results, never spliced into earlier history."""
    agent, db, result, payloads, _rows = _run_turn(monkeypatch, tmp_path)
    try:
        assert result["completed"] is True
        assert len(payloads) == 3
        delivered = []
        for payload in payloads:
            start = _turn_start(payload)
            for i, m in enumerate(payload):
                if m.get("role") == "user" and WAKE in str(m.get("content")):
                    delivered.append((i, start, payload[i - 1].get("role")))
        assert delivered, "the steered wake never reached a request"
        for i, start, before in delivered:
            assert i > start, f"wake spliced into history at {i}, before this turn's user row at {start}"
            assert before == "tool"
        # The gateway's settlement reads it consumed, so the admission's ``steered`` outcome is true.
        assert agent.withdraw_steer("admission-481") is False
    finally:
        db.close()


def test_steer_with_no_tool_batch_left_in_the_turn_goes_back_unconsumed(monkeypatch, tmp_path):
    """A4, other branch: the turn answers without a tool call, so no tool batch of this turn can carry
    the wake. It must reach no request (not even as history) and read unconsumed at settlement, so
    the gateway returns the admission to its own queue place instead of booking a false ``steered``."""
    agent, db, result, payloads, rows = _run_turn(monkeypatch, tmp_path, tool_rounds=0)
    try:
        assert result["completed"] is True and len(payloads) == 1
        assert not any(WAKE in str(m.get("content")) for m in payloads[0])
        assert not any(WAKE in str(r.get("content")) for r in rows)
        assert agent.withdraw_steer("admission-481") is True
    finally:
        db.close()


def test_each_persisted_steer_row_carries_its_own_text(monkeypatch, tmp_path):
    """A5: every steer row in state.db holds its own steer text; the inbound message is persisted once."""
    _agent, db, result, _payloads, rows = _run_turn(monkeypatch, tmp_path)
    try:
        assert result["completed"] is True
        steer_rows = [r for r in rows if r.get("display_kind") == STEER_DISPLAY_KIND]
        assert len(steer_rows) == 2, [(r["role"], str(r["content"])[:60]) for r in rows[-12:]]
        assert all(INBOUND not in r["content"] for r in steer_rows), [r["content"] for r in steer_rows]
        assert WAKE in steer_rows[0]["content"] and FOLLOW in steer_rows[1]["content"]
        assert all(r["timestamp"] != INBOUND_TS for r in steer_rows)
        inbound_rows = [r for r in rows if INBOUND in str(r.get("content"))]
        assert len(inbound_rows) == 1, [(r["id"], r.get("display_kind")) for r in inbound_rows]
    finally:
        db.close()


def test_persist_override_never_rewrites_a_steer_row():
    """The current-turn persist override (clean inbound text, platform timestamp) is the turn's own
    row's; a steer row addressed by a stale index keeps its own bytes in the flush and in memory."""
    from agent.session_persistence import _db_flush_row

    agent = object.__new__(AIAgent)
    agent._persist_user_message_override = INBOUND
    agent._persist_user_message_timestamp = INBOUND_TS
    agent._persist_user_message_platform_id = "bot:x"
    row = steer_user_row(WAKE)
    row["timestamp"] = INBOUND_TS + 100
    written = _db_flush_row(agent, row, True)
    assert WAKE in written["content"] and written["timestamp"] == INBOUND_TS + 100
    messages = [{"role": "user", "content": INBOUND}, row]
    agent._persist_user_message_idx = 1
    agent._apply_persist_user_message_override(messages)
    assert WAKE in row["content"] and row["timestamp"] == INBOUND_TS + 100
    assert "platform_message_id" not in row
