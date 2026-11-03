"""Explicit profile grant for shared-board coordination through existing CLI routes.

This grants board operations, never dispatcher ownership or a different sender identity.
The board location and ordinary run compare-and-swap checks remain authoritative.
"""
from __future__ import annotations


def worker_board_management_enabled() -> bool:
    from hermes_cli.config import load_config
    from gateway.session_context import get_session_env

    if not get_session_env("HERMES_SESSION_ID", "").strip():
        return False
    section = load_config().get("kanban")
    return isinstance(section, dict) and section.get("worker_board_management") is True


def board_actor(fallback: str, requested: str | None = None) -> str:
    """Stamp the actual bound session; opted-in agents cannot supply another author."""
    from hermes_cli.profiles import current_profile_name
    from gateway.session_context import get_session_env

    profile = current_profile_name(fallback) or fallback
    if worker_board_management_enabled():
        return f"{profile}/session:{get_session_env('HERMES_SESSION_ID', '').strip()}"
    return requested or profile


def peer_session_author() -> dict | None:
    """An explicit swarm sender is a session, not its profile's standing Bot Chat."""
    if not worker_board_management_enabled():
        return None
    from agent.turn_author import local_origin

    actor = board_actor("worker")
    return {"id": f"session:{local_origin()}/{actor}", "name": actor, "is_bot": True}


def attributed_board_payload(payload: dict | None) -> dict | None:
    """Attribute every granted board event, including commands without a reason."""
    if not worker_board_management_enabled():
        return payload
    return {**(payload or {}), "actor": board_actor("worker")}
