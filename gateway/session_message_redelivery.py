"""A person's message whose turn an owner's end cut short runs again: at least once.

Eric's rule: "A replay is completely fine. I don't give a fuck whether the same message is sent twice.
I just want all messages to be delivered at least once." A turn in flight when its owner stopped (or
died) was settled ``interrupted`` or fenced ``unknown`` and never answered: the person had to resend it.

The person's message is a MESSAGE effect: it is queued again in its own FIFO place (its ``seq``, the
same admission, a new claim and generation), so the next owner runs it as an ordinary turn. The turn's
own work is not replayed: only a turn that reached no tool is queued again. A turn that ran (or called)
any tool may have had side effects (a BUSINESS effect: files, commands, external calls) and is never
re-run silently; it keeps the owner's notice and waits for the person, as before.

Unattended rows (cron, automation, webhook, Kanban card) keep their own recovery
(``session_unattended_unknown``): a card is re-dispatched by its board, and nothing here touches them.

Durable inputs, both written in the transaction that decides them:
- the claim's transcript mark (``CLAIM_MARK_PREFIX``, ``claim_session_input``): rows above it on the
  conversation were written by this turn, so a tool row there means the turn reached a tool;
- the owed redelivery (``REDELIVER_PREFIX``), naming the claim and the status it is owed from: a stopping
  owner commits it with an interrupted turn's settlement (``finish_result``, ``terminal``), and the next
  owner's boot commits it with the fence of a lost turn (``recover_session_inputs``, ``unknown``). A failed
  owner-stop settlement owes the same redelivery with its fallback fence (``_mark_unknown``). The boot
  queues every one it finds again, so a boot cut short after the fence still runs the turn at the next boot
  (MR-RECOVERY-17), and until then the marker keeps the turn's accepted media held (MR-MEDIA-18).
"""
from __future__ import annotations

import json
import logging

from hermes_state_runtime import CLAIM_MARK_PREFIX

logger = logging.getLogger(__name__)

REDELIVER_PREFIX = 'gateway.admission.redeliver.v1.'
# Branch D: the next owner's boot queued the lost turn again (the transcript boundary it writes there).
REDELIVERY_NOTICE = ('Hermes restarted before this turn finished. It had not run any tools, so your message '
                     'will run again (you may see it answered twice).')
# Branch S1: a stopping owner owes the redelivery; the next owner's boot runs it. The live warning and the
# transcript carry the same copy, by cause.
RESTART_REDELIVERY_NOTICE = ('Gateway restarting: this turn was interrupted before it ran any tools, so your message '
                             'will run again once Hermes reconnects (you may see it answered twice).')
SHUTDOWN_REDELIVERY_NOTICE = ('Gateway shutting down: this turn was interrupted before it ran any tools, so your '
                              'message will run again once Hermes is back (you may see it answered twice).')


def _claim_mark(conn, row):
    """The saved mark of ``row``'s own claim (``mark``: the transcript high-water id; ``physical``: the
    transcript it wrote to), or None (no mark, or one of another claim)."""
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (CLAIM_MARK_PREFIX + row['admission_id'],)).fetchone()
    if saved is None:
        return None
    from hermes_state_runtime import RuntimeStoreError
    try:
        mark = json.loads(saved[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable', field='claim_mark', detail='invalid claim-mark proof') from exc
    if not isinstance(mark, dict):
        raise RuntimeStoreError('storage_unavailable', field='claim_mark', detail='invalid claim-mark proof')
    if mark.get('epoch') != row['owner_epoch'] or mark.get('generation') != row['generation']:
        return None
    return mark if type(mark.get('mark')) is int and mark['mark'] >= 0 else None


def _reached_a_tool(conn, row, saved):
    """True when the turn wrote any tool call or tool result above its claim's mark, on its logical
    session, its lineage, the transcript it ran on (``saved['physical']``) or the compression chain of
    that transcript and of the current one: a compression or rotation since never hides a tool step."""
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_runtime import _canonical_chain
    mark = saved['mark']
    sessions = set(json.loads(row['lineage_json'] or '[]')) | {row['target_session_id']}
    sessions.update(_canonical_chain(conn, local_physical_target(conn, row['target_session_id'])))
    if isinstance(saved.get('physical'), str) and conn.execute(
            'SELECT 1 FROM sessions WHERE id=?', (saved['physical'],)).fetchone() is not None:
        sessions.update(_canonical_chain(conn, saved['physical']))
    marks = ','.join('?' * len(sessions))
    return conn.execute(
        f"SELECT 1 FROM messages WHERE session_id IN ({marks}) AND id>? AND (role='tool' OR "
        f"(role='assistant' AND tool_calls IS NOT NULL AND tool_calls NOT IN ('', '[]'))) LIMIT 1",
        (*sessions, mark)).fetchone() is not None


def redeliverable(conn, row) -> bool:
    """A person's queued message whose lost turn reached no tool (its mark proves where it began)."""
    from gateway.session_unattended_unknown import is_unattended
    payload = row['payload'] if 'payload' in row.keys() else json.loads(row['payload_json'])
    if row['intent'] != 'queue' or is_unattended({'principal_id': row['principal_id'], 'payload': payload}):
        return False
    saved = _claim_mark(conn, row)
    return saved is not None and not _reached_a_tool(conn, row, saved)


def owner_stop_redelivery(conn, row, notice):
    """In a stopping owner's settlement of an interrupted turn (``finish_result``): owe a redelivery
    of a redeliverable person's message and return the notice the turn gets: the transcript boundary AND
    the live ``message.complete.warning`` (one decision, made in this write)."""
    current = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (row['admission_id'],)).fetchone()
    if current is None or not redeliverable(conn, current):
        return notice
    _owe(conn, current, 'terminal')
    return redelivery_notice(notice)


def owe_failed_owner_stop(conn, row, captured):
    """In the failed settlement's unknown fence: keep only a qualified owner-stop redelivery owed.

    The captured cause belongs to this claim, not the owner's current shutdown flags. Completed
    answers, user Stops, absent proof and tool-bearing/automation work keep explicit uncertainty.
    The marker and the unknown status commit together; a failed proof read/write rolls both back.
    """
    from gateway.session_settlement_recovery import (
        RESTART_INTERRUPT_NOTICE, SHUTDOWN_INTERRUPT_NOTICE, owner_stop_notice)
    value = (captured or {}).get('result') or {}
    if not value.get('interrupted'):
        return
    notice = owner_stop_notice(captured, 'interrupted')
    if notice in (RESTART_INTERRUPT_NOTICE, SHUTDOWN_INTERRUPT_NOTICE) and redeliverable(conn, row):
        _owe(conn, row, 'unknown')


def owe_lost_turns(conn, epoch):
    """In the boot's fence (``recover_session_inputs``, before it sets them ``unknown``): owe a redelivery
    of each redeliverable person's turn a previous owner left started, in that same write."""
    for row in conn.execute("SELECT * FROM session_admissions WHERE status='started' AND owner_epoch!=?",
                            (epoch,)).fetchall():
        if redeliverable(conn, row):
            _owe(conn, row, 'unknown')


def _owe(conn, row, status):
    """Owe a redelivery of ``row``'s claim once it is ``status`` (its settlement or its fence)."""
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                 (REDELIVER_PREFIX + row['admission_id'],
                  json.dumps({'epoch': row['owner_epoch'], 'generation': row['generation'], 'status': status})))


def _owed_row(conn, admission_id, owed):
    """The admission an owed redelivery still names: the same claim, still in the status it is owed from
    (a marker written before the status was recorded is a settlement's), else None."""
    status = owed.get('status', 'terminal')
    if status not in ('terminal', 'unknown'):
        return None
    return conn.execute("SELECT * FROM session_admissions WHERE admission_id=? AND status=? AND owner_epoch=? "
                        "AND generation=?" + (" AND outcome='interrupted'" if status == 'terminal' else ''),
                        (admission_id, status, owed.get('epoch'), owed.get('generation'))).fetchone()


def owed_admissions(conn) -> set[str]:
    """The admissions whose owed redelivery the next owner has not yet run: they still own their inputs."""
    return {key[len(REDELIVER_PREFIX):] for key, value in
            conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ?', (REDELIVER_PREFIX + '%',)).fetchall()
            if _owed_row(conn, key[len(REDELIVER_PREFIX):], json.loads(value)) is not None}


def redelivery_notice(owner_stop_notice):
    """The owed-redelivery copy for the owner-stop cause that ``owner_stop_notice`` names."""
    from gateway.session_settlement_recovery import RESTART_INTERRUPT_NOTICE
    return RESTART_REDELIVERY_NOTICE if owner_stop_notice == RESTART_INTERRUPT_NOTICE else SHUTDOWN_REDELIVERY_NOTICE


def owed_redelivery(conn, row) -> bool:
    """True when this claim's settlement owed a redelivery the next owner has not yet run."""
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (REDELIVER_PREFIX + row['admission_id'],)).fetchone()
    if saved is None:
        return False
    owed = json.loads(saved[0])
    return (owed.get('epoch') == row['owner_epoch'] and owed.get('generation') == row['generation']
            and owed.get('status', 'terminal') == row['status'])


def _requeue(conn, row, status):
    """Back to its FIFO place: a new claim runs it. The lost claim's worker rows retire with it."""
    from hermes_state_runtime import _retire_admission_workers
    from hermes_state_terminal import RESULT_PREFIX
    if row['generation'] is not None:
        _retire_admission_workers(conn, row, row['owner_epoch'])
    changed = conn.execute("UPDATE session_admissions SET status='queued',outcome=NULL,generation=NULL "
                           "WHERE admission_id=? AND status=?", (row['admission_id'], status)).rowcount == 1
    for prefix in (CLAIM_MARK_PREFIX, REDELIVER_PREFIX, RESULT_PREFIX):
        conn.execute('DELETE FROM state_meta WHERE key=?', (prefix + row['admission_id'],))
    if changed:
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
    return changed


def redeliver_recovered_messages(db, epoch) -> int:
    """Boot, after ``recover_session_inputs`` fenced the previous owner's turns: queue again every owed
    redelivery: each person's message whose lost turn reached no tool (owed by the fence, now ``unknown``;
    its transcript gets ``REDELIVERY_NOTICE`` as the lost turn's boundary), including one a boot cut short
    after its fence left owed, and each redelivery a stopping owner owed. Returns how many were queued
    again. Claim marks of settled turns are dropped."""
    from gateway.session_results import close_discarded_turn
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, epoch)
        queued = 0
        for key, value in conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ? ORDER BY key',
                                       (REDELIVER_PREFIX + '%',)).fetchall():
            row = _owed_row(conn, key[len(REDELIVER_PREFIX):], json.loads(value))
            if row is None or (row['status'] == 'unknown' and not redeliverable(conn, row)):
                conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
                continue
            if row['status'] == 'unknown':
                close_discarded_turn(db, conn, row, REDELIVERY_NOTICE)
            queued += _requeue(conn, row, row['status'])
        conn.execute("DELETE FROM state_meta WHERE key LIKE ? AND substr(key, ?) IN "
                     "(SELECT admission_id FROM session_admissions WHERE status='terminal')",
                     (CLAIM_MARK_PREFIX + '%', len(CLAIM_MARK_PREFIX) + 1))
        return queued
    queued = db._execute_write(write)
    if queued:
        logger.info('Queued %d interrupted message(s) again for the next turn (at least once)', queued)
    return queued
