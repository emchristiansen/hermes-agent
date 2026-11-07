"""A Kanban wake reaches a local (Bot Chat) session through its authority FIFO.

The notifier resolves a LOCAL subscription to the session authority's LocalSessionAdapter.
That adapter is created by the authority, never wired by the runner: it has no message handler
and no ``gateway_runner``. ``deliver_wake`` used to read the runner only from the adapter, so it
skipped the FIFO admission and fell back to ``handle_message``, which discards every inbound
event on a handler-less adapter. The notifier saw ``WakeNotAccepted``, rewound its cursor and
retried forever, so LOCAL subscribers never received their task wakes (2026-11-04: 147 stuck
subscriptions, the oldest 65 hours behind).
"""
import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_kanban_wake_is_admitted_into_a_local_session(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import LocalSessionAdapter, create_local_session
    from gateway.wake import deliver_wake
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    turns = []

    async def answer(event):
        turns.append(event.text)
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", answer)
    try:
        viewer = Principal("test-owner", "default", frozenset(
            {"session:create", "session:submit", "session:read"}), "viewer")
        ref = create_local_session(authority, viewer, dict(request_id="cli", source="cli", model="fixture",
                                                          toolsets=[]))
        live = authority.sessions[ref.session_id]
        adapter = runner._adapter_for_source(live.source)
        assert isinstance(adapter, LocalSessionAdapter)
        assert adapter._message_handler is None  # the production shape: authority-created, unwired

        # What KanbanNotifier.wake() builds from a LOCAL subscription row.
        source = SessionSource(platform=Platform.LOCAL, chat_id=live.source.chat_id, chat_type="dm",
                               user_id=live.source.user_id)
        await deliver_wake(adapter, text="kanban: t_1 completed", source=source)
        await async_wait_until(lambda: "kanban: t_1 completed" in turns, timeout=10, what="the wake turn")
    finally:
        db.close()
