"""What a task's stamped ``last_failure_error`` still says about its next attempt.

The dispatcher's ``blocker_auth`` respawn guard matches the stamp against
``RESPAWN_BLOCKER_RE``. The column outlives the attempt that wrote it: a later
attempt spawned past it that ends without a diagnosis of its own (an owner
restart requeues uncounted and stamps nothing) leaves the old text behind, and
it must not re-park the card as though it diagnosed that attempt (t_39f19a6e).
History is never rewritten here; only the guard's reading of it.
"""
import re

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ation)|authz|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# The guard's one total order over a card's ended runs, newest attempt first. A card's
# runs open one at a time (``current_run_id``) under an ascending AUTOINCREMENT id, so
# the id is the attempt order. ``ended_at`` is not: it is whole seconds (a sweep and a
# spawn failure in one dispatch tick tie) and a wall clock (a clock correction, or a
# board imported onto a machine whose clock is behind, ends a newer attempt at an
# earlier second). Every "latest run" / "later run" decision of the guard uses this
# order; ``ended_at`` is read only for elapsed-time cooldowns.
LATEST_RUN_FIRST = "id DESC"


def stamp_superseded(conn, task_id, err) -> bool:
    """True when the run that recorded ``err`` is no longer the card's newest attempt.

    Every writer that stamps ``tasks.last_failure_error`` for a run (the
    dead-worker sweep, ``_record_task_failure``) copies the same text, capped at
    500 chars, into that run's own ``task_runs.error``, so the newest ended run
    whose error matches is the stamp's provenance (newest in ``LATEST_RUN_FIRST``
    order). It is superseded when a run later in that order was actually spawned
    (``worker_pid``, ``spawned`` or ``worker_registered``) and ended: had that
    attempt diagnosed a failure it would have re-stamped the column and be the
    match itself. A stamp no run recorded (PR acceptance, a
    legacy ``last_spawn_error``) is never superseded — the guard fails closed.
    """
    from hermes_cli.kanban_db import _lossy_text
    origin = None
    for run in conn.execute(
        "SELECT id, error FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        f"AND error IS NOT NULL ORDER BY {LATEST_RUN_FIRST}",
        (task_id,),
    ):
        if (_lossy_text(run["error"]) or "")[:500] == err:
            origin = run
            break
    if origin is None:
        return False
    # "Later" in the same order: a higher id, whatever either run's wall-clock ended_at.
    return conn.execute(
        "SELECT 1 FROM task_runs r WHERE r.task_id = ? AND r.id > ? AND r.ended_at IS NOT NULL "
        "AND (r.worker_pid IS NOT NULL OR EXISTS (SELECT 1 FROM task_events e WHERE e.task_id = r.task_id "
        "AND e.run_id = r.id AND e.kind IN ('spawned', 'worker_registered'))) LIMIT 1",
        (task_id, origin["id"]),
    ).fetchone() is not None
