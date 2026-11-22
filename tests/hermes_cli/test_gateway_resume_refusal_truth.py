"""A resume the gateway refuses ends the CLI run; it never becomes a turn the gateway must refuse.

t_b42e0875 / t_c556a615: ``hermes chat --resume <id> -q ...`` on a stored session the gateway
cannot open. The owner answered ``session.resume`` with ``not_found``, the CLI loaded the stored
transcript through its read-only reader, printed "Resumed session", submitted the query anyway and
showed the owner's ``permission_denied`` as "The model request failed. Run /model to switch".
"""
from __future__ import annotations

import pytest

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_cli.gateway_session_store import RESUME_REFUSED, GatewaySessionStore


class _Connection:
    final = False
    launch_api_key = None
    launch_api_key_typed = False

    def __init__(self, reason):
        self.reason = reason
        self.calls = []

    def resume(self, call, session_id, *, typed=False):
        self.calls.append(("session.resume", session_id))
        raise GatewayRPCError(self.reason)

    def call(self, method, **params):
        self.calls.append((method, params.get("session_id")))
        raise AssertionError(f"{method} must not be sent after a refused resume")


class _Reader:
    def __init__(self, rows):
        self.rows = rows

    def get_session(self, session_id):
        return self.rows.get(session_id)


class _Cli:
    _resumed = True
    session_id = "20261120_082151_da8fbd"


def _store(reason, rows):
    store = GatewaySessionStore(_Connection(reason))
    store._reader = _Reader(rows)
    return store


@pytest.mark.parametrize("reason", ["not_found", "permission_denied"])
def test_a_refused_resume_of_a_stored_session_is_an_error(reason):
    store = _store(reason, {_Cli.session_id: {"id": _Cli.session_id, "source": "subagent"}})
    with pytest.raises(GatewayClientError) as raised:
        store.open_session(_Cli())
    assert str(raised.value) == f"{RESUME_REFUSED} {_Cli.session_id} ({reason}); nothing was sent."
    assert store.connection.calls == [("session.resume", _Cli.session_id)]


def test_a_session_nobody_knows_keeps_mains_not_found_path():
    store = _store("not_found", {})
    assert store.open_session(_Cli()) is None


def test_the_refusal_is_the_usage_error_exit():
    from hermes_cli.gateway_chat import exit_on_resume_refusal
    with pytest.raises(SystemExit) as raised:
        exit_on_resume_refusal(GatewayClientError(f"{RESUME_REFUSED} x (not_found); nothing was sent."))
    assert raised.value.code == 2


def test_a_refused_turn_is_shown_as_the_gateways_refusal_not_the_models():
    from hermes_cli.cli_chat_error_copy import chat_error_response
    text = "The gateway refused this turn: permission_denied."
    shown = chat_error_response(text, provider="openai-codex", model="gpt-6.1-sol", failure_reason="gateway_refusal")
    assert shown == text
    assert "/model" not in shown


class _HistoryReader(_Reader):
    def get_messages_as_conversation(self, session_id, repair_alternation=False):
        return [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]


def _late_loader(open_):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class _Viewer(CLIAgentSetupMixin):
        tool_progress_mode = "off"  # plain status lines on stderr
        session_id = _Cli.session_id
        conversation_history = []

        def _follow_compression_chain(self, meta, announce):
            self._resume_read_id = self.session_id
            return meta

        def _resume_history_limit_error(self, tip_only=False):
            return None

        def _restore_session_state(self, meta, *, quiet=False):
            pass

    viewer = _Viewer()
    viewer._session_db = GatewaySessionStore(_Connection("not_found"))
    viewer._session_db._reader = _HistoryReader({_Cli.session_id: {"id": _Cli.session_id, "title": "t"}})
    viewer._gateway_session_open = open_
    return viewer


@pytest.mark.parametrize("attached", [True, False])
def test_resumed_is_announced_only_after_the_gateway_attached(capsys, attached):
    viewer = _late_loader(attached)
    assert viewer._load_resumed_history_late() is True
    assert len(viewer.conversation_history) == 2  # the reader's history still loads
    assert ("Resumed" in capsys.readouterr().err) is attached
