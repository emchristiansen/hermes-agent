"""Task-event persistence with session-bound attribution for an explicitly trusted swarm."""
import time


def append_event(conn, task_id: str, kind: str, payload: dict | None = None, *, run_id: int | None = None) -> None:
    """Stay in the caller's transaction; preserve attempt grouping and exact payload encoding."""
    from agent.swarm_capabilities import attributed_board_payload
    from hermes_cli import kanban_db

    payload = attributed_board_payload(payload)
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, kanban_db._json_or_null(payload), int(time.time())),
    )
