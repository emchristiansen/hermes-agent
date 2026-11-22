"""The ``kanban_request_review`` tool refuses an invalid reviewer-continuation opt-in with an
error the model sees, leaving the card running."""
from __future__ import annotations

import json

pytest_plugins = ["tests.tools.test_kanban_tools"]  # its ``worker_env`` fixture


def test_tool_refuses_continuation_of_unknown_task(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_request_review({"summary": "Ready for review.", "metadata": {
        "reviewer_context": "continue", "review_continues_task": "t_nowhere"}}))

    assert "review_continues_task" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"


def test_tool_accepts_same_card_continuation(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_request_review({"summary": "Ready for review.",
                                                "metadata": {"reviewer_context": "continue"}}))

    assert out["ok"] is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "review"
