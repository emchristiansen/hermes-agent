"""Release provenance for the dispatcher's ``active_pr`` respawn guard, and the operator's
exact-run continuation of a READY card that guard holds.

A PR URL in a recent comment holds the ready lane (a re-spawned opener risks a duplicate PR)
until a *release* is recorded strictly after the newest such comment. A release persists:
later crash/reclaim retries of the same card (breaker-bounded) do not consume it, and only a
newer PR comment guards again. Nothing here infers intent from a missing field.
"""
import os

from hermes_cli.kanban_db_retry_policy import terminal_run_matches
from hermes_cli.kanban_owner_recovery import owner_reclaim_blocked

RELEASE_KINDS = ('assigned', 'changes_requested', 'review_reopened', 'unblocked', 'promoted', 'promoted_manual')
# Every event that moves a card into or out of a parked phase. The newest one before a legacy
# ``promoted`` says where that promotion came from.
_LANDINGS = ('dependency_wait', 'blocked', 'gave_up', 'block_loop_detected', 'status', 'unblocked',
             'review_reopened', 'changes_requested', 'claimed', 'reclaimed', 'crashed', 'timed_out',
             'stale', 'spawn_failed', 'rate_limited', 'promoted', 'promoted_manual')


def released_after(conn, task_id, since) -> bool:
    """True when a release event was recorded strictly after ``since`` (a same-second tie
    stays guarded: fail closed)."""
    from hermes_cli.kanban_db_dispatch import _rows_since
    return any(is_release(conn, task_id, e['id'], e['kind'], e['payload']) for _, e in _rows_since(
        conn, 'task_events', task_id, since, f'AND kind IN {RELEASE_KINDS!r}', cols=', id, kind, payload', strict=True))


def is_release(conn, task_id, event_id, kind, payload) -> bool:
    """Whether one event after the newest PR comment names who must now work on THAT PR.

    * ``assigned`` only when it moves the card to a DIFFERENT profile: a no-op re-assign, an
      unassign, the dispatcher's ``kanban.default_assignee`` fill-in, or an event without
      ``from`` (written before it was recorded) is no handoff.
    * ``promoted`` (``recompute_ready``) when it left ``todo``, i.e. a dependency release; a
      breaker park auto-recovering (``from: blocked``) is the PR worker's own crash loop. An
      older ``promoted`` has no ``from``: it counts only when the newest parking event before
      it is the worker's ``dependency_wait``; a ``gave_up``/``blocked`` or anything else in
      between cannot be told apart from breaker recovery, so it stays guarded.
    * ``changes_requested``, ``review_reopened``, ``unblocked`` and ``promoted_manual`` are
      explicit verdicts or operator acts (``promoted_manual`` includes the ready-card
      continuation below).
    """
    from hermes_cli import kanban_db as kb
    data = kb._json_dict(payload)
    if kind == 'assigned':
        to = data.get('assignee')
        return (data.get('source') != 'kanban.default_assignee' and bool(to)
                and 'from' in data and data['from'] != to)
    if kind != 'promoted':
        return True
    if 'from' in data:
        return data['from'] == 'todo'
    placeholders = ', '.join('?' for _ in _LANDINGS)
    before = conn.execute(
        f"SELECT kind FROM task_events WHERE task_id = ? AND id < ? AND kind IN ({placeholders}) "
        "ORDER BY id DESC LIMIT 1", (task_id, event_id, *_LANDINGS)).fetchone()
    return before is not None and before['kind'] == 'dependency_wait'


def continuation_refusal(conn, task_id, expected_run):
    """Why the exact-run continuation of a READY card is refused, else None."""
    task = conn.execute('SELECT status,claim_lock,worker_pid,current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None:
        return f'task {task_id} not found'
    if task['status'] != 'ready':
        return (f"task {task_id} is {task['status']!r}; a continuation applies only to 'ready' "
                "(promote todo/blocked without --expected-terminal-run)")
    latest = conn.execute('SELECT id,ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1',
                          (task_id,)).fetchone()
    if latest is None or type(expected_run) is not int or latest['id'] != expected_run:
        return f"run {expected_run} is not the latest run of {task_id} (latest: {latest['id'] if latest else 'none'})"
    if latest['ended_at'] is None:
        return f'run {expected_run} of {task_id} has not ended'
    live = [key for key in ('claim_lock', 'worker_pid', 'current_run_id') if task[key] is not None]
    if live:
        return f"{task_id} still has a live claim ({', '.join(live)})"
    if owner_reclaim_blocked(conn, task_id) or not terminal_run_matches(conn, task_id, expected_run):
        return (f'the owner of run {expected_run} may still be executing it '
                '(admission not terminal, or accounting pending)')
    return None


def continue_ready(conn, task_id, expected_run, *, actor, reason, dry_run):
    """Explicit operator continuation of a READY card held by ``active_pr``.

    The card stays ``ready``; a ``promoted_manual`` event marked ``from: ready`` on run
    ``expected_run`` is the release the guard honours after the newest PR comment. Fenced on the
    exact latest ended run with no live claim and no owner still executing it, re-checked under
    the write lock; a refused or dry-run attempt writes nothing. Orchestrator-only, as unblock.
    Returns ``(ok, refusal)``."""
    from agent.swarm_capabilities import worker_board_management_enabled
    from hermes_cli import kanban_db as kb
    if os.environ.get('HERMES_KANBAN_TASK') and not worker_board_management_enabled():
        return False, 'a ready-card continuation is orchestrator-only; workers must hand off their assigned task'
    refusal = continuation_refusal(conn, task_id, expected_run)
    if refusal or dry_run:
        return refusal is None, refusal
    with kb.write_txn(conn):
        refusal = continuation_refusal(conn, task_id, expected_run)
        if refusal is None:
            kb._append_event(conn, task_id, 'promoted_manual', {
                'actor': actor, 'reason': reason, 'from': 'ready', 'continuation': True,
                'expected_terminal_run': expected_run,
            }, run_id=expected_run)
    return refusal is None, refusal
