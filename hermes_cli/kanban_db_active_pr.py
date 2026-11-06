"""Release provenance for the dispatcher's ``active_pr`` respawn guard, and the operator's
exact-run continuation of a READY card that guard holds, or of a TRIAGE card whose loop run was
interrupted before its worker recorded a result or finished with its own bound worker's result
(either way the input it awaited has since arrived).

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
    """Why the exact-run continuation of a READY (or loop-run TRIAGE) card is refused, else None."""
    task = conn.execute('SELECT status,claim_lock,worker_pid,current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None:
        return f'task {task_id} not found'
    if task['status'] == 'triage':
        return triage_continuation_refusal(conn, task_id, expected_run, reason)
    if task['status'] != 'ready':
        return (f"task {task_id} is {task['status']!r}; a continuation applies only to 'ready' or a "
                "'triage' loop (promote todo/blocked without --expected-terminal-run)")
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
    """The newest ``kind`` event of run ``run_id`` as ``(id, payload dict)``, else None (an
    existence test here; the writer proof reads every event, ``_run_writers``)."""
    from hermes_cli import kanban_db as kb
    row = conn.execute('SELECT id,payload FROM task_events WHERE task_id=? AND run_id=? AND kind=? '
                       'ORDER BY id DESC LIMIT 1', (task_id, run_id, kind)).fetchone()
    return None if row is None else (row['id'], kb._json_dict(row['payload']))


def _run_writers(conn, task_id, run):
    """Every ``(source, pid, recorded identity, payload)`` that may write for ``run``: ALL its
    ``worker_bound``, ``spawned`` and ``worker_registered`` events (no append is unique per run:
    ``session_kanban.bind_worker_context`` and ``kanban_db_dispatch._set_worker_pid`` append
    unconditionally) and the run row's retained submitter."""
    from hermes_cli import kanban_db as kb
    writers = [(kind, kb._json_dict(payload)) for kind, payload in conn.execute(
        "SELECT kind,payload FROM task_events WHERE task_id=? AND run_id=? AND kind IN ('worker_bound','spawned','worker_registered') "
        'ORDER BY id', (task_id, run['id']))]
    rows = [(kind, data.get('pid'), data.get('started_at'), data) for kind, data in writers]
    if run['worker_pid'] is not None:
        rows.append(('task_runs.worker_pid', run['worker_pid'], run['worker_started_at'], {}))
    return rows


def _writer_state(pid, recorded):
    """``gone`` when ``pid`` is dead or its CURRENT identity was read and differs from ``recorded``
    (recycled); ``unreadable`` when it is live and its current identity cannot be read; else
    ``live``. Fails closed like ``kanban_db_workspace._binding_may_be_live``: an unreadable,
    legacy (None) or ``unverified`` identity never proves the writer gone. (``_worker_alive``
    reads an unreadable current identity as recycled, right for signalling, not for this proof.)"""
    from gateway.status import get_process_start_time
    from hermes_cli.kanban_db_dispatch import _pid_alive
    from hermes_cli.kanban_db_workspace import _identity_differs
    if not _pid_alive(pid):
        return 'gone'
    if get_process_start_time(pid) is None:
        return 'unreadable'
    return 'gone' if _identity_differs(pid, recorded) else 'live'


def _host_refusal(run, writers):
    """Why ``run``'s pids cannot be judged on this host, else None. Pid liveness is host-local
    (``kanban_db_dispatch.adopt_worker_pid`` and ``_reap_terminal_worker_row`` gate on
    ``_host_prefix`` too): a foreign or unprovable claim's absent pid proves nothing."""
    from hermes_cli import kanban_db as kb
    claims = [run['claim_lock']] + [data.get('claim_lock') for kind, _p, _r, data in writers if kind == 'worker_bound']
    if any(not isinstance(claim, str) or ':' not in claim or not claim.split(':', 1)[0] for claim in claims):
        return f"the host of run {run['id']} cannot be proven from its claim; its writers cannot be probed here"
    foreign = sorted({claim.split(':', 1)[0] for claim in claims if not claim.startswith(kb._host_prefix())})
    if foreign:
        return (f"run {run['id']} was claimed on another host ({', '.join(foreign)}); "
                'its writers cannot be proven gone from this host')
    return None


def _writers_refusal(conn, task_id, run):
    """Why the writers of ``run`` are not all provably gone, else None."""
    writers = _run_writers(conn, task_id, run)
    if not any(kind == 'worker_bound' and data.get('claim_lock') == run['claim_lock']
               for kind, _pid, _rec, data in writers):
        return f"run {run['id']} has no bound worker under its claim; its writer cannot be proven gone"
    host = _host_refusal(run, writers)
    if host is not None:
        return host
    bad = [kind for kind, pid, _rec, _data in writers if type(pid) is not int or pid <= 0]
    if bad:
        return f"run {run['id']} has an unreadable worker record ({', '.join(bad)}); its writer cannot be proven gone"
    states = [(pid, _writer_state(pid, rec)) for _kind, pid, rec, _data in writers]  # per record, not per pid
    unreadable = sorted({pid for pid, state in states if state == 'unreadable'})
    if unreadable:
        return (f"the identity of live pid {', '.join(map(str, unreadable))} of run {run['id']} cannot be read; "
                'it may still be the writer')
    alive = sorted({pid for pid, state in states if state == 'live'})
    if alive:
        return f"a worker of run {run['id']} may still be live (pid {', '.join(map(str, alive))})"
    return None


def _origin_refusal(conn, task_id, run_id, loop):
    """Why run ``run_id``'s loop is not an implementation-lane loop, else None. The continuation
    lands only in ``ready``, where dispatch makes an ordinary ``claim_task``: a reviewer run's loop
    would lose its review claim (``source_status=review``, sdlc-review) and its verdict path
    (``request_changes`` needs a review claim). Review resumption is not offered here; the run's
    own ``claimed`` event (``kanban_db._retry_status_for_run``) and the loop's recorded
    ``source_status`` must both say the implementation lane, else refuse."""
    from hermes_cli import kanban_db as kb
    if kb._retry_status_for_run(conn, task_id, run_id) == 'review' or loop.get('source_status') == 'review':
        return (f'run {run_id} of {task_id} is a review-origin loop (claimed from review); a triage '
                'continuation lands only in ready and would drop the review lane, so it is refused')
    if loop.get('source_status') != 'ready':
        return (f"the lane of run {run_id}'s loop cannot be proven (source_status="
                f"{loop.get('source_status')!r}); a triage continuation needs an implementation-lane loop")
    return None


def _finished_result(conn, task_id, run_id):
    """``(event id, payload)`` of run ``run_id``'s worker_result when it is the run's own bound
    worker's terminal result after its loop, else None. The check is settle-triage's own
    (``kanban_db_graph._settlement_terminal_run``: bound < loop < result, same pid and claim, int
    exit, verified start identity, run ended blocked with its claim released and no open run),
    reused, not restated."""
    from hermes_cli.kanban_db_graph import _settlement_terminal_run
    if _settlement_terminal_run(conn, task_id, run_id) is None:
        return None
    return _run_event(conn, task_id, run_id, 'worker_result')


def triage_continuation_refusal(conn, task_id, expected_run, reason):
    """Why a TRIAGE card may not continue to ``ready`` after its exact loop run, else None.

    The run tripped the loop breaker (``block_loop_detected``) and either was interrupted before
    its bound interpreter wrote a ``worker_result`` (a shape ``settle-triage`` cannot take), or
    finished with its own bound worker's terminal result (``_finished_result``); any other result
    is refused. Either way the awaited input has arrived, so no missing-input wait is recorded.
    Continuing needs positive proof that nothing can still write for it: the
    latest run, ended ``blocked``, from the implementation lane (``_origin_refusal``: a review-origin
    loop is refused), no open run or claim on the card, a ``worker_bound`` under the
    run's claim, that claim and every binding's issued by THIS host, EVERY worker pid recorded for the run dead or recycled by a successful reading of
    its start identity (``_writers_refusal``), and the run's owner admission recorded and
    terminal. ``reason`` names the awaited input that has now arrived (e.g. a comment id)."""
    from hermes_cli import kanban_db as kb
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
    loop = _run_event(conn, task_id, expected_run, 'block_loop_detected')
    if loop is None:
        return f'run {expected_run} did not route {task_id} to triage (no block_loop_detected on it)'
    origin = _origin_refusal(conn, task_id, expected_run, loop[1])
    if origin is not None:
        return origin
    if _run_event(conn, task_id, expected_run, 'worker_result') is not None and _finished_result(conn, task_id, expected_run) is None:
        return (f"run {expected_run}'s worker_result is not proven its own bound worker's terminal result "
                "(settle-triage's check: run ended blocked with its claim released, bound < loop < result "
                'order, matching pid and claim, int exit, verified bound identity)')
    writers = _writers_refusal(conn, task_id, latest)
    if writers is not None:
        return writers
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
    whose exact loop run was interrupted without a worker result or finished with its own bound
    worker's result (``triage_continuation_refusal``).

    A ready card stays ``ready``; a ``promoted_manual`` event marked ``from: ready`` on run
    ``expected_run`` is the release the guard honours after the newest PR comment. A triage card
    moves to ``ready`` with the same assignee, title, body and counters (``block_kind``,
    ``block_recurrences`` and the failure counters are kept, so a same-cause re-block returns it
    to triage); its ``promoted_manual`` (``from: triage``) records the input that arrived and the
    loop run: ``interrupted_run`` with ``worker_result_recorded: false``, or ``finished_run`` with
    ``worker_result_recorded: true``, the result's event id and exit code. Fenced on the exact latest ended run,
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
            finished = _finished_result(conn, task_id, expected_run)
            shape = ({'interrupted_run': expected_run, 'worker_result_recorded': False} if finished is None else
                     {'finished_run': expected_run, 'worker_result_recorded': True,
                      'worker_result_event': finished[0], 'worker_exit_code': finished[1]['exit_code']})
            kb._append_event(conn, task_id, 'promoted_manual', {
                'actor': actor, 'reason': reason, 'from': 'triage', 'continuation': True,
                'expected_terminal_run': expected_run, **shape,
            }, run_id=expected_run)
        elif refusal is None:
            kb._append_event(conn, task_id, 'promoted_manual', {
                'actor': actor, 'reason': reason, 'from': 'ready', 'continuation': True,
                'expected_terminal_run': expected_run,
            }, run_id=expected_run)
    return refusal is None, refusal
