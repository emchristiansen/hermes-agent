"""Reviewer continuation: a review session starts with a copy of the review conversation it
continues, recorded once in its creation receipt.

The copy is independent (no ``parent_session_id``: that link means a compression step) and its
rows are the source's stored rows, so the worker's first-turn history read sees them and the
send-path replay repair, not a re-execution, handles any unanswered tool call at the tail.
"""
import json

from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import RuntimeStoreError, _epoch, _json

_ACTIVE_MAX = 'SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=? AND active=1'


def _source_target(conn, intent, owner_db):
    """``(physical transcript, None)`` of the source review session, or ``(None, reason)``."""
    if 'unavailable' in intent:
        return None, intent['unavailable']
    if intent['db'] != owner_db:
        return None, 'different_owner_store'
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + intent['session_id'],)).fetchone()
    if row is None:
        return None, 'source_session_missing'
    from hermes_state_local_lineage import validate_local_lineage
    try:
        receipt = json.loads(row[0])
        physical = validate_local_lineage(conn, receipt)
        bound = json.loads(receipt['policy'].get('kanban_json') or 'null')
    except (RuntimeStoreError, KeyError, TypeError, ValueError):
        return None, 'source_lineage_invalid'
    if not isinstance(bound, dict) or (bound.get('task_id'), bound.get('run_id')) != (intent['task_id'], intent['run_id']):
        return None, 'source_binding_mismatch'
    return physical, None


def _seed(db, conn, session_id, intent, owner_db):
    physical, reason = _source_target(conn, intent, owner_db)
    if reason is None and conn.execute(_ACTIVE_MAX, (session_id,)).fetchone()[0]:
        reason = 'destination_not_empty'
    if reason is not None:
        return {'outcome': 'unavailable', 'reason': reason}
    record = {'source_task': intent['task_id'], 'source_run': intent['run_id'], 'source_session': intent['session_id'],
              'source_physical': physical, 'source_run_ended': intent['run_ended']}
    # The write transaction is the snapshot: rows the source appends later have larger ids.
    boundary = conn.execute(_ACTIVE_MAX, (physical,)).fetchone()[0]
    if not boundary:
        return dict(record, outcome='source_empty', boundary=0, count=0)
    ids, tools = db._tail_rows_after_watermark(conn,
        'SELECT id,tool_calls FROM messages WHERE session_id=? AND active=1 AND id<=? ORDER BY id', [physical, boundary])
    db._clone_message_rows(conn, ids, session_id=session_id)
    conn.execute('UPDATE sessions SET message_count=?,tool_call_count=? WHERE id=?', (len(ids), tools, session_id))
    from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS, require_idle
    try:
        require_idle(db, conn, list(dict.fromkeys([intent['session_id'], physical])))
        busy = False
    except (RuntimeStoreError, *MUTATION_GUARD_REFUSALS):
        # Recorded, not refused: the copy is the snapshot up to ``boundary``. A live turn lease
        # or compression lock on the source is the guard's busy verdict, like a running admission.
        busy = True
    return dict(record, outcome='inherited', boundary=boundary, count=len(ids),
                dest_last_row=conn.execute(_ACTIVE_MAX, (session_id,)).fetchone()[0], source_busy=busy)


def seed_review_continuity(db, *, epoch, session_id, intent, owner_db):
    """Seed review session *session_id* from frozen *intent* once; returns the stored record.

    One owner write transaction: the copy and ``review_seed`` in the creation receipt commit
    together, so a retry (crash, duplicate launch) returns the record and copies nothing."""
    key = POLICY_PREFIX + session_id

    def write(conn):
        _epoch(conn, epoch)
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if row is None:
            raise RuntimeStoreError('storage_unavailable')
        receipt = json.loads(row[0])
        if 'review_seed' not in receipt:
            receipt['review_seed'] = _seed(db, conn, session_id, intent, owner_db)
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(receipt), key))
        return receipt['review_seed']
    return db._execute_write(write)


def first_turn_note(record, context):
    """What the reviewer is told about the history above, derived only from the stored record
    (and the frozen policy), so a retry reproduces the admitted payload byte for byte."""
    current = f"Your current assignment is task {context['task_id']} run {context['run_id']}; it needs its own fresh verdict."
    if record['outcome'] == 'unavailable':
        return (f"The previous review conversation was not attached ({record['reason']}). "
                f"Review cold from the board history. {current}")
    origin = f"task {record['source_task']} review run {record['source_run']}"
    if record['outcome'] == 'source_empty':
        return (f"The previous review conversation ({origin}) was not attached (source_empty: it had no messages). "
                f"Review cold from the board history. {current}")
    span = f"{record['count']} messages, through source message {record['boundary']}"
    if record['source_busy'] or not record['source_run_ended']:
        return (f"The conversation above is a snapshot of the review conversation of {origin} ({span}). "
                f"That review may still be active and may have continued past this snapshot. {current}")
    return f"The conversation above is the review conversation of {origin} ({span}), which ended before this run. {current}"
