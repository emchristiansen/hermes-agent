"""Release provenance for the dispatcher's ``active_pr`` respawn guard, and the operator's
exact-run continuation of a READY card that guard holds, or of a TRIAGE card whose loop run was
interrupted before its worker recorded a result (the input it awaited has since arrived).

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


def continuation_refusal(conn, task_id, expected_run, reason=None):
    """Why the exact-run continuation of a READY (or interrupted TRIAGE) card is refused, else None."""
    task = conn.execute('SELECT status,claim_lock,worker_pid,current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None:
        return f'task {task_id} not found'
    if task['status'] == 'triage':
        return triage_continuation_refusal(conn, task_id, expected_run, reason)
    if task['status'] != 'ready':
        return (f"task {task_id} is {task['status']!r}; a continuation applies only to 'ready' or an "
                "interrupted 'triage' loop (promote todo/blocked without --expected-terminal-run)")
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


def _run_event(conn, task_id, run_id, kind):
    """The newest ``kind`` event of run ``run_id`` as ``(id, payload dict)``, else None."""
    from hermes_cli import kanban_db as kb
    row = conn.execute('SELECT id,payload FROM task_events WHERE task_id=? AND run_id=? AND kind=? '
                       'ORDER BY id DESC LIMIT 1', (task_id, run_id, kind)).fetchone()
    return None if row is None else (row['id'], kb._json_dict(row['payload']))


def triage_continuation_refusal(conn, task_id, expected_run, reason):
    """Why a TRIAGE card may not continue to ``ready`` after its exact interrupted loop run, else None.

    The shape ``settle-triage`` cannot take: the run tripped the loop breaker
    (``block_loop_detected``) and was then interrupted, so its bound interpreter never wrote a
    ``worker_result``. Continuing needs positive proof that nothing can still write for it: the
    latest run, ended ``blocked``, no open run or claim on the card, every worker pid recorded
    for the run (``worker_bound`` under the run's claim, the spawned submitter) dead or recycled
    by its start identity, and the run's owner admission recorded and terminal. ``reason`` names
    the awaited input that has now arrived (e.g. a comment id) and is required."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _worker_alive
    from hermes_cli.kanban_db_retry_policy import terminal_run_refusal
    if type(expected_run) is not int:
        return 'no exact terminal run was named (--expected-terminal-run)'
    if not isinstance(reason, str) or not reason.strip():
        return ('a triage continuation must name the awaited input that has arrived '
                '(reason, e.g. the comment id carrying it)')
    task = conn.execute('SELECT claim_lock,claim_expires,worker_pid,current_run_id FROM tasks WHERE id=?',
                        (task_id,)).fetchone()
    latest = conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1', (task_id,)).fetchone()
    if latest is None or latest['id'] != expected_run:
        return f"run {expected_run} is not the latest run of {task_id} (latest: {latest['id'] if latest else 'none'})"
    if conn.execute('SELECT 1 FROM task_runs WHERE task_id=? AND ended_at IS NULL', (task_id,)).fetchone():
        return f'{task_id} has an open run'
    live = [key for key in ('claim_lock', 'claim_expires', 'worker_pid', 'current_run_id') if task[key] is not None]
    if live:
        return f"{task_id} still has a live claim ({', '.join(live)})"
    if latest['status'] != 'blocked' or latest['outcome'] != 'blocked':
        return f"run {expected_run} did not end blocked ({latest['status']}/{latest['outcome']})"
    if _run_event(conn, task_id, expected_run, 'block_loop_detected') is None:
        return f'run {expected_run} did not route {task_id} to triage (no block_loop_detected on it)'
    if _run_event(conn, task_id, expected_run, 'worker_result') is not None:
        return (f'run {expected_run} has a worker_result; its finished loop is settled with '
                'settle-triage, not continued')
    bound = _run_event(conn, task_id, expected_run, 'worker_bound')
    bound = bound[1] if bound else {}
    if type(bound.get('pid')) is not int or bound['pid'] <= 0 or bound.get('claim_lock') != latest['claim_lock']:
        return f'run {expected_run} has no bound worker under its claim; its writer cannot be proven gone'
    spawned = _run_event(conn, task_id, expected_run, 'spawned')
    spawned = spawned[1] if spawned else {}
    pids = [(bound['pid'], bound.get('started_at')), (latest['worker_pid'], latest['worker_started_at']),
            (spawned.get('pid'), spawned.get('started_at'))]
    alive = sorted({int(pid) for pid, started in pids if pid and _worker_alive(int(pid), started)})
    if alive:
        return f"a worker of run {expected_run} may still be live (pid {', '.join(map(str, alive))})"
    if _run_event(conn, task_id, expected_run, 'owner_admitted') is None:
        return f'run {expected_run} has no owner admission; the owner cannot be proven terminal'
    owner = terminal_run_refusal(conn, task_id, expected_run)
    if owner is not None:
        return f'the owner of run {expected_run} may still be executing it ({owner})'
    if not kb._parents_satisfied(conn, task_id):
        return (f'{task_id} has unsatisfied parent dependencies; a continuation lands only in ready')
    return None


def continue_ready(conn, task_id, expected_run, *, actor, reason, dry_run):
    """Explicit operator continuation of a READY card held by ``active_pr``, or of a TRIAGE card
    whose exact loop run was interrupted without a worker result (``triage_continuation_refusal``).

    A ready card stays ``ready``; a ``promoted_manual`` event marked ``from: ready`` on run
    ``expected_run`` is the release the guard honours after the newest PR comment. A triage card
    moves to ``ready`` with the same assignee, title, body and counters (``block_kind``,
    ``block_recurrences`` and the failure counters are kept, so a same-cause re-block returns it
    to triage); its ``promoted_manual`` (``from: triage``) records the interrupted run, that no
    worker result was recorded, and the input that arrived. Fenced on the exact latest ended run,
    re-checked under the write lock; a refused or dry-run attempt writes nothing.
    Orchestrator-only, as unblock. Returns ``(ok, refusal)``."""
    from agent.swarm_capabilities import worker_board_management_enabled
    from hermes_cli import kanban_db as kb
    if os.environ.get('HERMES_KANBAN_TASK') and not worker_board_management_enabled():
        return False, 'a ready-card continuation is orchestrator-only; workers must hand off their assigned task'
    refusal = continuation_refusal(conn, task_id, expected_run, reason)
    if refusal or dry_run:
        return refusal is None, refusal
    with kb.write_txn(conn):
        refusal = continuation_refusal(conn, task_id, expected_run, reason)
        if refusal is None and kb._task_status(conn, task_id) == 'triage':
            conn.execute("UPDATE tasks SET status='ready' WHERE id=? AND status='triage'", (task_id,))
            kb._append_event(conn, task_id, 'promoted_manual', {
                'actor': actor, 'reason': reason, 'from': 'triage', 'continuation': True,
                'expected_terminal_run': expected_run, 'interrupted_run': expected_run,
                'worker_result_recorded': False,
            }, run_id=expected_run)
        elif refusal is None:
            kb._append_event(conn, task_id, 'promoted_manual', {
                'actor': actor, 'reason': reason, 'from': 'ready', 'continuation': True,
                'expected_terminal_run': expected_run,
            }, run_id=expected_run)
    return refusal is None, refusal
