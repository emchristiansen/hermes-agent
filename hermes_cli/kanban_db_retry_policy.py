"""Guarded first-failure policy and atomic hold records for existing tasks."""
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
import json
import os
from pathlib import Path
import sqlite3
from typing import Optional

from hermes_cli import kanban_owner_recovery


def guarded_edit(operations):
    """Guard new policy edits; resolve the facade's transaction hook only after its import shims."""
    def decorate(editor):
        call_signature = signature(editor)
        @wraps(editor)
        def edit(conn, task_id, **values):
            call_signature.bind(conn, task_id, **values)
            limit = values.get('max_retries')
            expected_run = values.get('expected_terminal_run')
            if limit is None and expected_run is None:
                return editor(conn, task_id, **values)
            mixed = tuple(values.get(key) for key in ('title', 'body', 'priority', 'result', 'summary', 'metadata'))
            write_txn, append_event, notify_updated = operations()
            return edit_retry_policy(conn, task_id, limit, expected_run, mixed,
                                     write_txn=write_txn, append_event=append_event,
                                     owner_blocked=kanban_owner_recovery.owner_reclaim_blocked,
                                     notify_updated=notify_updated, board=values.get('board'))
        return edit
    return decorate


@dataclass
class CrashSweep:
    """Dead attempts collected under one reclaim transaction; accounting follows."""

    crashed: list[str] = field(default_factory=list)
    rate_limited: list[str] = field(default_factory=list)
    owner_restarted: list[str] = field(default_factory=list)
    crash_details: list[tuple[str, int, str, object, int]] = field(default_factory=list)
    """``(task_id, pid, claim_lock, DeadWorker, ended run id)`` per counted crash."""
    exited_hook_payloads: list[dict] = field(default_factory=list)



@dataclass(frozen=True)
class SpawnClaim:
    """The exact claim a dispatcher's spawn-failure transition belongs to (``_record_task_failure``).

    The dispatcher can stall between its claim and that commit (claim TTL expires); the card may then
    have been reclaimed and claimed again by another dispatcher, whose worker may be running. The commit
    therefore changes nothing unless the card is still exactly this claim (``holds``): it never ends
    another run, clears another claim or pid, charges another attempt or moves another phase.
    ``refused_before_launch`` comes only from the launch-boundary producer's tag for this very run
    (``_restart_safe_worker_argv`` raises before ``Popen``); under ``holds`` it marks the run it ends."""

    run_id: int
    claim_lock: Optional[str]
    refused_before_launch: bool = False

    @classmethod
    def of(cls, claimed, exc: Optional[BaseException] = None) -> "SpawnClaim":
        refused = exc is not None and getattr(exc, "kanban_refused_run", None) == claimed.current_run_id
        return cls(claimed.current_run_id, claimed.claim_lock, refused)

    def holds(self, task_row) -> bool:
        """Whether the card (``tasks`` row read in the failure commit) is still this running claim."""
        return (task_row["status"] == "running" and task_row["current_run_id"] == self.run_id
                and task_row["claim_lock"] == self.claim_lock)

    def launch_detail(self) -> dict:
        """The run detail a host refusal records for the run it ends (only under ``holds``)."""
        return {"launch": "refused_before_launch"} if self.refused_before_launch else {}


# Worker result "the attempt's owner ended under it" (``session_kanban.record_owner_restart_result``):
# requeued WITHOUT counting a failure, as the host's restart is not the card's.
KANBAN_OWNER_RESTART_EXIT_CODE = 79


@dataclass
class DeadWorker:
    """How ``detect_crashed_workers`` should book one dead worker."""

    kind: str
    code: Optional[int]
    error_text: str
    event_kind: str
    event_payload: dict
    protocol_violation: bool = False
    rate_limited: bool = False
    owner_restarted: bool = False
    """``KANBAN_OWNER_RESTART_EXIT_CODE``: the owner, not the card, ended the attempt."""
    terminal_provider: bool = False
    """``KANBAN_TERMINAL_PROVIDER_EXIT_CODE``: the provider rejected the worker's
    credential/model — trips the breaker on this first occurrence."""

    @property
    def run_outcome(self) -> str:
        # A rate-limited requeue is recorded as ``rate_limited`` so board history
        # doesn't show a phantom crash for a quota wall.
        if self.rate_limited:
            return "rate_limited"
        return "owner_restarted" if self.owner_restarted else "crashed"

    @property
    def counts_failure(self) -> bool:
        """Whether the death is the card's: a quota wall or its owner's restart is the host's."""
        return not (self.rate_limited or self.owner_restarted)


def terminal_run_matches(conn, task_id, expected_run):
    """A latest ended attempt with no live board claim and a known owner terminal."""
    return terminal_run_refusal(conn, task_id, expected_run) is None


def terminal_run_refusal(conn, task_id, expected_run):
    """Why ``expected_run`` is not a releasable terminal attempt, or None when it is."""
    if type(expected_run) is not int:
        return 'no exact terminal run was named (--expected-terminal-run)'
    task = conn.execute(
        "SELECT claim_lock,worker_pid,current_run_id FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    latest = conn.execute(
        "SELECT id,ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if task is None or latest is None:
        return 'the task has no attempt to release'
    if latest['id'] != expected_run:
        return f'run {expected_run} is not the latest attempt (latest is {latest["id"]})'
    if latest['ended_at'] is None:
        return f'run {expected_run} has not ended'
    if any(task[key] is not None for key in ('claim_lock', 'worker_pid', 'current_run_id')):
        return 'the task still holds a live claim'
    hold = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
                        (task_id, expected_run)).fetchone()
    try:
        pending_account = json.loads(hold['payload']).get('accounting_required', False) if hold else False
    except (ValueError, TypeError, AttributeError):
        return f'run {expected_run} has an unreadable hold record'
    if pending_account and conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='gave_up'",
                                        (task_id, expected_run)).fetchone() is None:
        return f'run {expected_run} failure accounting is not recorded yet'
    # Unlike current-run reclaim, a stopped card requires the exact historical run's owner event.
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
        (task_id, expected_run),
    ).fetchone()
    if event is None:
        return None  # A genuinely unmanaged/legacy attempt has no native admission to reconcile.
    try:
        binding = json.loads(event['payload'])
    except (ValueError, TypeError):
        binding = None
    state = _owner_admission_state(binding) if isinstance(binding, dict) else None
    if state is None:
        return f'run {expected_run} owner admission is unreadable'
    if state[0] != 'terminal':
        return f'run {expected_run} owner admission is not terminal yet ({state[0]})'
    return None


def edit_retry_policy(conn, task_id, limit, expected_run, mixed_values, *,
                      write_txn, append_event, owner_blocked, notify_updated, board):
    """Tighten only the selected field; leave lifecycle, counts, and original attempts untouched."""
    from agent.swarm_capabilities import worker_board_management_enabled
    denied_worker = os.environ.get('HERMES_KANBAN_TASK') and not worker_board_management_enabled()
    if (denied_worker or type(limit) is not int or limit != 1
            or any(value is not None for value in mixed_values)):
        return False
    with write_txn(conn):
        task = conn.execute('SELECT status,max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
        if (task is None or task['status'] not in {'blocked', 'triage'}
                or (task['max_retries'] is not None and task['max_retries'] < 1)
                or owner_blocked(conn, task_id) or not terminal_run_matches(conn, task_id, expected_run)):
            return False
        if task['max_retries'] == 1:
            return True
        conn.execute('UPDATE tasks SET max_retries=1 WHERE id=?', (task_id,))
        append_event(conn, task_id, 'edited', {
            'fields': ['max_retries'], 'max_retries': 1, 'expected_terminal_run': expected_run,
        }, run_id=expected_run)
    notify_updated(conn, task_id, ['max_retries'], board=board)
    return True


def _run_metadata(conn, run_id):
    """The run's decoded metadata; None when unreadable (missing run or malformed JSON)."""
    row = conn.execute('SELECT metadata FROM task_runs WHERE id=?', (run_id,)).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row['metadata']) if row['metadata'] else {}
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _set_run_metadata(conn, run_id, data):
    """Write ``data`` back exactly as ``kanban_db._end_run`` serializes metadata (empty -> NULL)."""
    conn.execute('UPDATE task_runs SET metadata=? WHERE id=?',
                 (json.dumps(data, ensure_ascii=False) if data else None, run_id))


def owe_failure_accounting(conn, run_id):
    """In the commit that ends a counted run: mark it ``failure_accounting: pending``.

    The run is ended and its card re-queued in one commit and the failure counted in a later one
    (``_reclaim_dead_workers`` / ``_account_crashes``, timeout, stale-claim reclaim); a dispatcher
    lost between them leaves a retryable card whose failure no counter holds, and this mark."""
    data = _run_metadata(conn, run_id)
    if data is not None:
        _set_run_metadata(conn, run_id, {**data, 'failure_accounting': 'pending'})


def settle_failure_accounting(conn, run_id):
    """In the commit that accounts ``run_id``'s failure (counted or deliberately not): record it
    ``failure_accounting: settled``, the only positive proof that the accounting happened."""
    data = _run_metadata(conn, run_id) if run_id is not None else None
    if data is not None:
        _set_run_metadata(conn, run_id, {**data, 'failure_accounting': 'settled'})


def failure_accounting_state(conn, run_id):
    """``'pending'``, ``'settled'``, None (never marked: an uncounted end, or a run ended by a
    runtime that did not mark) or ``'unreadable'``; absence is never proof of settlement."""
    data = _run_metadata(conn, run_id)
    if data is None:
        return 'unreadable'
    state = data.get('failure_accounting')
    return state if state in (None, 'pending', 'settled') else 'unreadable'


def own_scheduled_attempt(conn, task_id):
    """The run a scheduled card's ordinary callback wake resumes, if the card cleanly yielded.

    Only a claimed attempt that ended by scheduling its own card, whose bound worker then
    recorded exit 0 and whose owner admission (if managed) settled ``completed``, qualifies:
    that is a deliberate yield, not a failure whose effects need reconciling. Anything else
    (no worker result yet, a nonzero exit, an interrupted or unknown admission, a run
    synthesized by scheduling a held card) keeps the first-failure fence's exact-run release."""
    task = conn.execute('SELECT status FROM tasks WHERE id=?', (task_id,)).fetchone()
    latest = conn.execute(
        "SELECT id,outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if task is None or task['status'] != 'scheduled' or latest is None or latest['outcome'] != 'scheduled':
        return None
    run = latest['id']
    if conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' LIMIT 1",
                    (task_id, run)).fetchone() is None:
        return None
    result = _last_event_payload(conn, task_id, run, 'worker_result')
    if not isinstance(result, dict) or result.get('exit_code') != 0:
        return None
    admitted = _last_event_payload(conn, task_id, run, 'owner_admitted')
    if admitted is not None and _owner_admission_state(admitted, ('status', 'outcome')) != ('terminal', 'completed'):
        return None
    return run


def _last_event_payload(conn, task_id, run_id, kind):
    """The latest ``kind`` event's decoded payload for the run, None when absent; unreadable is {}."""
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                       (task_id, run_id, kind)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row['payload']) if row['payload'] else {}
    except (ValueError, TypeError):
        return {}


def _owner_admission_state(binding, columns=('status',)):
    """The ``columns`` of the owner admission an ``owner_admitted`` payload names: a tuple,
    ('missing',) * n when absent, None when unreadable."""
    try:
        with closing(sqlite3.connect(Path(binding['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            row = owner.execute(
                f"SELECT {','.join(columns)} FROM session_admissions WHERE target_session_id=? AND request_id=?",
                (binding['session_id'], binding['request_id']),
            ).fetchone()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        return None
    return ('missing',) * len(columns) if row is None else tuple(row)


def release_is_fenced(conn, task_id, expected_run):
    """Ordinary releases keep their old path; selected releases must match the exact attempt.

    A selected card that scheduled itself names its own attempt, so an ordinary callback wake
    passes the same terminal-attempt and owner checks as an explicit exact-run release."""
    return release_refusal(conn, task_id, expected_run) is None


def release_refusal(conn, task_id, expected_run):
    """Why a release of ``task_id`` is refused by this policy, or None when it may proceed."""
    row = conn.execute('SELECT max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
    if row is None:
        return 'no such task'
    if row['max_retries'] != 1 and expected_run is None:
        return None
    if expected_run is None:
        expected_run = own_scheduled_attempt(conn, task_id)
        if expected_run is None:
            return ('max_retries=1 holds release until an exact terminal run is named (--expected-terminal-run); '
                    'only a clean self-scheduled yield (worker exit 0, owner admission completed) wakes without it')
    refusal = terminal_run_refusal(conn, task_id, expected_run)
    if refusal is None and kanban_owner_recovery.owner_reclaim_blocked(conn, task_id):
        refusal = 'its owner may still be executing the attempt'
    return refusal


def first_failure_landing(conn, task_id, phase):
    """Select before the reclaim transaction publishes a potentially claimable source phase."""
    row = conn.execute('SELECT max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
    return 'blocked' if row is not None and row['max_retries'] == 1 else phase


@contextmanager
def reclaim_transaction(conn, task_id, write_txn, append_event, *, accounting_required=False):
    """Hold only the exact successful reclaim explicitly witnessed by the caller inside this transaction."""
    with write_txn(conn):
        before = conn.execute('SELECT status,current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
        transition = {}
        yield transition
        after = conn.execute('SELECT status,max_retries,current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
        if accounting_required and transition.get('run_id') is not None:
            owe_failure_accounting(conn, transition['run_id'])
        if ('run_id' in transition and before is not None and before['status'] == 'running'
                and before['current_run_id'] == transition['run_id'] and after is not None
                and after['max_retries'] == 1 and after['status'] in {'ready', 'review'}
                and after['current_run_id'] is None):
            claimed = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' ORDER BY id DESC LIMIT 1",
                                   (task_id, before['current_run_id'])).fetchone()
            try:
                source = json.loads(claimed['payload']) if claimed else {}
            except (ValueError, TypeError):
                source = {}
            phase = 'review' if isinstance(source, dict) and source.get('source_status') == 'review' else after['status']
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
            record_first_failure_hold(conn, task_id, before['current_run_id'], phase, append_event, accounting_required=accounting_required)


def record_first_failure_hold(conn, task_id, run_id, phase, append_event, *, accounting_required=False):
    """Called inside the reclaim transaction, before another connection can claim a retry.

    ``accounting_required``: a later commit counts this failure, so the run is marked pending for
    every policy (``owe_failure_accounting``); ``max_retries=1`` also parks the card ``blocked``."""
    if accounting_required:
        owe_failure_accounting(conn, run_id)
    row = conn.execute('SELECT max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
    if row is None or row['max_retries'] != 1:
        return
    conn.execute("UPDATE tasks SET block_kind='capability' WHERE id=?", (task_id,))
    append_event(conn, task_id, 'blocked', {
        'kind': 'capability', 'retry_status': phase, 'accounting_required': accounting_required,
        'reason': 'selected first-failure hold; reconcile the exact attempt before explicit release',
    }, run_id=run_id)


def unblock_refusal(conn, task_id, expected_run=None):
    """A truthful reason ``unblock_task`` refused ``task_id``, read after the refusal."""
    task = conn.execute('SELECT status FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None:
        return 'no such task'
    if task['status'] not in ('blocked', 'scheduled'):
        return f'status is {task["status"]}, not blocked/scheduled'
    return release_refusal(conn, task_id, expected_run) or 'it changed while being released; retry'
