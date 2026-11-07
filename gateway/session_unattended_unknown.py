"""An ``unknown`` turn nobody is waiting on settles as interrupted (#134889 GW-10, #139532).

A turn lost across an owner restart is fenced ``unknown`` and pauses its session's FIFO, because
only the person who sent it knows whether to resend it. An automation delivery has no such person:
a cron run (``CRON_PRINCIPAL_ID``), a process completion or heartbeat (``automation:<route>``,
``local_automation_v1``, a native envelope's ``automation`` block) or a webhook delivery
(``webhook_delivery``) or a Kanban card attempt (``kanban_card_v1``, set by the dispatcher). Pausing on it stalls every message queued behind it until an operator
notices (a Bot Chat sat ten hours behind one).

Such a row settles through the same fenced resolution a person's Discard uses
(``resolve_unknown_session_input``): terminal ``interrupted``, never requeued or replayed, with a
Hermes transcript notice committed in the same transaction. No second recovery store.

A Kanban card attempt lost because its owner itself ended (this owner already stopping or restarting
when it fenced the lost worker, or a predecessor that left the turn in flight) carries that cause
durably: ``OWNER_RESTART``, committed in the same transaction as its ``unknown`` fence. Its terminal
settlement first publishes the cause as the run's result (``publish_owed_restart``), so the
dispatcher requeues the card without counting a failure. The states are:

* ``unknown``/``OWNER_RESTART``: owed. The receipt reads as in flight. Boot tries once; the drain
  retries a storage failure in place while its owner serves. An ending owner leaves it to the next
  boot.
* the result is recorded, or provably moot (a stale run or claim, a result the interpreter already
  recorded, no board for the attempt): it settles ``terminal``/``interrupted``. That write replaces
  the cause, and a repeat publication finds the result and records nothing.

A turn lost under a live, continuing owner never carries the cause, so it keeps its crash booking.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3

from hermes_state_keys import CRON_PRINCIPAL_ID

logger = logging.getLogger(__name__)


def is_unattended(row) -> bool:
    """True for an admission no person submitted (so nobody can be asked to resend it)."""
    principal = row.get('principal_id') or ''
    if principal == CRON_PRINCIPAL_ID or principal.startswith('automation:'):
        return True
    payload = row.get('payload') or {}
    native = payload.get('native_text_v1') or {}
    # A webhook delivery: the provider already got its 202 and nobody can be asked to resend it.
    # A Kanban card attempt (#134889 KB-8): the dispatcher's marker, committed only by
    # ``SessionAuthority.submit(_kanban_card=...)``; a card row main minted without it still pauses.
    return ('local_automation_v1' in payload or 'kanban_card_v1' in payload
            or 'automation' in native or 'webhook_delivery' in native)


def _unattended_unknown(rows):
    return [row for row in rows if row['status'] == 'unknown' and is_unattended(row)]


def owner_ending(authority) -> bool:
    """This owner is stopping or restarting: a drain, a profile retirement, or a stop signal."""
    runner = getattr(authority, 'runner', None)
    return bool(getattr(authority, 'retiring', False) or getattr(runner, '_draining', False)
                or getattr(runner, '_stop_requested_by_signal', False)
                or getattr(runner, '_signal_initiated_shutdown', False)
                or getattr(runner, '_restart_requested', False))


# Transient board/store failures: the owed result stays owed and its terminal receipt waits.
_STORAGE_ERRORS = (OSError, sqlite3.OperationalError)
# Capped backoff for the drain's in-place retry of an owed publication.
_OWED_RETRY_MIN_S, _OWED_RETRY_MAX_S = 0.5, 5.0


def publish_owed_restart(db, row) -> bool:
    """Publish the restart result ``row`` owes (``OWNER_RESTART``) before its terminal settlement, so
    a submitter or dispatcher that reads the terminal receipt also reads the cause. True once the row
    may settle: recorded now, nothing owed, or moot (``record_owner_restart_result`` refuses a stale
    run or claim and never writes over an existing result; an attempt whose board policy is not this
    one's has nowhere to record it). The row's session fences an attempt whose interpreter never
    bound. A store that cannot take the write now raises (``_STORAGE_ERRORS``): the row stays owed.
    Any other failure is logged and settles without it, leaving the ordinary crash booking."""
    from hermes_state_runtime import OWNER_RESTART
    card = (row.get('payload') or {}).get('kanban_card_v1')
    if row.get('outcome') != OWNER_RESTART or card is None:
        return True
    from gateway.session_kanban import card_board_db, record_owner_restart_result
    try:
        with db._read_ctx() as conn:
            board_db = card_board_db(conn, row)
        if board_db is not None and record_owner_restart_result(card, board_db, row.get('target_session_id')):
            logger.info('Kanban attempt %s/%s run %s: owner restart recorded (not replayed)',
                        card.get('board'), card.get('task_id'), card.get('run_id'))
    except _STORAGE_ERRORS:
        raise
    except Exception:
        logger.warning('Kanban attempt %r: owner restart not recordable; settling without it', card, exc_info=True)
    return True


def settle_recovered_unattended(db, epoch) -> int:
    """Boot recovery, right after ``recover_session_inputs`` fenced lost turns ``unknown`` and before
    any FIFO is scheduled: settle every unattended one. Synchronous; runs off the owner loop. A row
    whose owed restart result cannot be published now stays ``unknown`` for its drain to retry
    (``recover_local_sessions`` schedules it)."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_results import close_discarded_turn
    from hermes_state_runtime import RuntimeStoreError, _row, resolve_unknown_session_input
    with db._read_ctx() as conn:
        rows = [_row(r) for r in conn.execute(
            "SELECT * FROM session_admissions WHERE status='unknown' ORDER BY seq").fetchall()]
    settled = 0
    for row in _unattended_unknown(rows):
        try:
            publish_owed_restart(db, row)
        except _STORAGE_ERRORS as exc:
            logger.warning('Lost Kanban admission %s left owed: owner restart not published (%r)',
                           row['admission_id'], exc)
            continue
        try:
            resolve_unknown_session_input(
                db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'],
                _terminal_write=lambda conn, lost: close_discarded_turn(db, conn, lost, AUTOMATION_INTERRUPTED_NOTICE))
        except RuntimeStoreError as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        release_admission_media(db, row['admission_id'])
        settled += 1
        logger.info('Lost automation admission %s on session %s settled interrupted (not replayed)',
                    row['admission_id'], row['target_session_id'])
    return settled


async def _published_in_drain(authority, row) -> bool:
    """``publish_owed_restart`` from the owner's drain: while this owner serves, a storage failure is
    retried in place (the session's FIFO is held behind the row anyway); an ending owner leaves the
    durable cause to its successor's boot. False when the row must stay ``unknown``."""
    delay = _OWED_RETRY_MIN_S
    while True:
        try:
            return await asyncio.to_thread(publish_owed_restart, authority.db, row)
        except _STORAGE_ERRORS as exc:
            if owner_ending(authority):
                logger.warning('Lost Kanban admission %s left owed for the next owner: owner restart not '
                               'published (%r)', row['admission_id'], exc)
                return False
            logger.warning('Lost Kanban admission %s: owner restart not published (%r); retrying in %.1fs',
                           row['admission_id'], exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, _OWED_RETRY_MAX_S)


async def settle_unattended_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head (fenced in this owner, e.g. a lost managed worker):
    settle the unattended ones through the owner's resolution. True when any settled, so the
    drain re-reads its FIFO instead of pausing; a person's ``unknown`` row still pauses it."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
    settled = False
    for row in _unattended_unknown(rows):
        if not await _published_in_drain(authority, row):
            continue
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=AUTOMATION_INTERRUPTED_NOTICE)
        except RuntimeStoreError as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
    return settled
