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
    crash_details: list[tuple[str, int, str, object]] = field(default_factory=list)
    exited_hook_payloads: list[dict] = field(default_factory=list)



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
    if type(expected_run) is not int:
        return False
    task = conn.execute(
        "SELECT claim_lock,worker_pid,current_run_id FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    latest = conn.execute(
        "SELECT id,ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if (task is None or latest is None or latest['id'] != expected_run or latest['ended_at'] is None
            or any(task[key] is not None for key in ('claim_lock', 'worker_pid', 'current_run_id'))):
        return False
    hold = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
                        (task_id, expected_run)).fetchone()
    try:
        pending_account = json.loads(hold['payload']).get('accounting_required', False) if hold else False
    except (ValueError, TypeError, AttributeError):
        return False
    if pending_account and conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='gave_up'",
                                        (task_id, expected_run)).fetchone() is None:
        return False
    # Unlike current-run reclaim, a stopped card requires the exact historical run's owner event.
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
        (task_id, expected_run),
    ).fetchone()
    if event is None:
        return True  # A genuinely unmanaged/legacy attempt has no native admission to reconcile.
    try:
        binding = json.loads(event['payload'])
        with closing(sqlite3.connect(Path(binding['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            state = owner.execute(
                "SELECT status FROM session_admissions WHERE target_session_id=? AND request_id=?",
                (binding['session_id'], binding['request_id']),
            ).fetchone()
        return state is not None and state[0] == 'terminal'
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        return False


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


def release_is_fenced(conn, task_id, expected_run):
    """Ordinary releases keep their old path; selected releases must match the exact attempt."""
    row = conn.execute('SELECT max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
    if row is None:
        return False
    if row['max_retries'] != 1 and expected_run is None:
        return True
    return terminal_run_matches(conn, task_id, expected_run) and not kanban_owner_recovery.owner_reclaim_blocked(conn, task_id)


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
    """Called inside the reclaim transaction, before another connection can claim a retry."""
    row = conn.execute('SELECT max_retries FROM tasks WHERE id=?', (task_id,)).fetchone()
    if row is None or row['max_retries'] != 1:
        return
    conn.execute("UPDATE tasks SET block_kind='capability' WHERE id=?", (task_id,))
    append_event(conn, task_id, 'blocked', {
        'kind': 'capability', 'retry_status': phase, 'accounting_required': accounting_required,
        'reason': 'selected first-failure hold; reconcile the exact attempt before explicit release',
    }, run_id=run_id)
