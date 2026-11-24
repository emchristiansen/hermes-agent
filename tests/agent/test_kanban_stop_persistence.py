"""A Kanban stop nudge must not erase the assistant's real response."""

from unittest.mock import patch

from agent.turn_stop_gates import apply_stop_gates
from hermes_state import SessionDB
from run_agent import AIAgent


def test_stop_nudge_persists_attempted_answer_before_continuing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    with patch("agent.process_bootstrap.OpenAI"), patch(
        "agent.model_metadata.fetch_model_metadata", return_value={}
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            provider="openai-compat",
            model="test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    db = SessionDB(db_path=tmp_path / "session.db")
    db.create_session(session_id="test", source="tui", model="test")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "test"
    agent._persist_disabled = False
    messages = [{"role": "user", "content": "do the work"}]
    answer = {"role": "assistant", "content": "I will run the remaining tests."}
    with (
        patch("agent.turn_stop_gates._late_lsp_verdicts", return_value=None),
        patch("agent.turn_stop_gates._verify_on_stop_nudge", return_value=None),
        patch("agent.turn_stop_gates._pre_verify_nudge", return_value=None),
    ):
        verdict = apply_stop_gates(
            agent,
            answer,
            final_response=answer["content"],
            messages=messages,
            conversation_history=[],
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )
    assert verdict.continue_turn
    durable = db.get_messages_as_conversation("test")
    assert [(m["role"], m["content"]) for m in durable] == [
        ("user", "do the work"),
        ("assistant", "I will run the remaining tests."),
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    assert db.get_messages_as_conversation("test") == durable
    db.close()
