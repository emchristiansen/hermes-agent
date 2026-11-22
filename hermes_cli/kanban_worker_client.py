"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap" and not "hermes_bootstrap".startswith(f"{exc.name}."):
        raise

import asyncio
import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Optional

# A worker that loses its owner (a dropped socket, a restart, `hermes update` draining it or a new
# owner still starting) waits for it instead of exiting: the condition is the host's, not the card's,
# and a failed exit here spends the card's failure budget and lets the owner end the turn.
# ``OWNER_WAIT_S`` is the longest the owner may stay silent before it has admitted the claim (the
# first ``kanban.run`` answer has its own wait, ``admission_answer_budget``); once admitted, the
# bound is ``RECEIPT_OUTAGE_S``.
OWNER_WAIT_S = 180.0
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0
# One ``prompt.receipt`` poll's answer budget. The poll is a read, but the owner answers it on its
# event loop, which in-process work (compaction, LCM, schema reconciles) can starve of the GIL for
# tens of seconds. 180 s is this client's own finite policy: past those stalls and past the owner
# loop watchdog's nominal default (three missed 30 s probes, about 120 s), but that watchdog is
# configurable and runs in the same process, so the owner is not guaranteed to exit first.
# A shorter poll deadline ends healthy turns: under the 30 s RPC default, every submitter polling
# through one such stall exits at once and each bound worker's submitter watch then ends its turn
# (the signature, not a directly observed chain, of the 2026-11-04 mass losses).
RECEIPT_ANSWER_BUDGET_S = 180.0
# Once admitted, the turn runs in its bound worker whatever happens to this socket, and that worker
# lives only while this process does (its submitter watch). So a lost or unanswered receipt read is
# re-read on a new connection, never resubmitted, until the owner has said nothing for this long
# (reconnects and pauses included): one unanswered poll's budget plus an owner restart's wait.
RECEIPT_OUTAGE_S = RECEIPT_ANSWER_BUDGET_S + OWNER_WAIT_S


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


@dataclass
class _Claim:
    """What the owner has said about this attempt, carried across connections.

    ``in_doubt``: a ``kanban.run`` was sent and no answer came back, so whether the owner took the
    claim is unknown and it is never sent again. ``silence``: once admitted, the timeout bounding
    the owner's silence (reconnects and pauses included); each answer pushes it back."""
    answered: float  # monotonic time of the owner's last answer
    session_id: Optional[str] = None
    admission_id: Optional[str] = None
    receipt: Optional[dict] = None
    in_doubt: bool = False
    silence: Optional[asyncio.Timeout] = None

    def silent_for(self):
        return time.monotonic() - self.answered

    def answer(self, receipt):
        self.receipt = receipt
        self.answered = time.monotonic()
        if self.silence is not None:
            self.silence.reschedule(asyncio.get_running_loop().time() + RECEIPT_OUTAGE_S)


def _Admission(session_id, receipt):
    """The owner's answer to this claim's one ``kanban.run``: what every later receipt read names."""
    return _Claim(answered=time.monotonic(), session_id=session_id, admission_id=receipt.get('admission_id'),
                  receipt=receipt)


def _quoted(exc):
    """WebSocket errors can embed remote bodies; only our own messages (and local OS errors) are quoted."""
    from hermes_cli.gateway_client import GatewayClientError
    if isinstance(exc, (GatewayClientError, OSError)):
        return exc
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return type(exc).__name__ + (f' (HTTP {status})' if status else '')


async def run(params, board_db):
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_owner_restart import restartable
    from websockets.exceptions import WebSocketException
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    # A lost socket is not a lost attempt, and exiting here would let ``_watch_submitter`` end the
    # owner's in-flight turn. Before admission the wait is bounded from the owner's last answer
    # (``OWNER_WAIT_S``); once admitted, ``_follow`` re-reads the same admission (``RECEIPT_OUTAGE_S``).
    claim = _Claim(answered=time.monotonic())
    # The owner incarnation live when this attempt started, for ``_owner_restart_grace``.
    from hermes_cli.kanban_presubmit_grace import baseline
    owner = await asyncio.to_thread(baseline)
    # Set once ``connect_gateway`` yields a connection, the only way ``kanban.run`` can be sent. From
    # then on the card's one pre-submission grace is never spent: only a wait that never connected is.
    connected = False
    failures = 0
    while True:
        seen = claim.answered
        try:
            async with _receipts() as client:
                connected = True
                await _submit_claim(client, params, board_db, claim)
                return await _follow(claim, client, params, board_db)
        # A restart also surfaces as a refused or reset dial, a failed handshake, or an answer
        # that never came.
        except (GatewayClientError, WebSocketException, OSError, TimeoutError) as exc:
            # Past admission, ``_follow`` has already decided; a claim that may have reached the
            # owner without an answer is never resubmitted.
            if claim.admission_id is not None or claim.in_doubt or not restartable(exc):
                raise
            if claim.silent_for() >= OWNER_WAIT_S:
                if not connected and await asyncio.to_thread(_owner_restart_grace, owner, params, board_db):
                    from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
                    return KANBAN_OWNER_RESTART_EXIT_CODE
                raise GatewayClientError(f'Kanban owner silent for {claim.silent_for():.0f}s: {_quoted(exc)}') from exc
        failures = 1 if claim.answered != seen else failures + 1
        # Jittered (1-2 s first, up to 16 s): workers of one dead owner do not all redial, and so
        # start a replacement, in the same instant.
        await asyncio.sleep(jittered_backoff(failures, base_delay=1.0, max_delay=8.0, jitter_ratio=1.0))


def _owner_restart_grace(owner, params, board_db):
    """The owner never became ready within the wait and nothing was submitted: True when the owner
    incarnation live at this attempt's start was replaced by another, and the card's one grace before
    an actual admission was still unspent and is now spent on this exact run
    (``kanban_presubmit_grace``). Otherwise False, and the wait's ordinary counted exit follows."""
    from hermes_cli.kanban_presubmit_grace import record_grace, replacement
    home, old = owner
    new = replacement(old, home)
    if new is None or not record_grace(board_db, params, old, new, home):
        return False
    print(f'Kanban attempt not submitted: its owner {old[0]} was replaced by {new[0]} and never became ready '
          f'within {OWNER_WAIT_S:g} s; owner restart recorded (the card\'s one grace before an admission)',
          file=sys.stderr)
    return True


@contextlib.asynccontextmanager
async def _receipts(ready_timeout=None):
    """A connection to the profile owner whose turn events are discarded: kanban.run subscribes
    the socket to the card's session, but only receipts matter here, and undrained events fill
    the client's bounded queue until its reader stops with the next receipt behind them.

    Never into a restart's after-turn wait: it cut this socket so the turn's submitter waits for
    the NEXT owner, and a worker polling its receipt inside the wait keeps the turn it waits on
    alive, holding `hermes update` until restart_after_turn_timeout."""
    from hermes_cli.gateway_client import connect_gateway, outside_restart_wait
    with outside_restart_wait():
        async with (connect_gateway() if ready_timeout is None else connect_gateway(ready_timeout=ready_timeout)) as client:
            drain = asyncio.create_task(_discard(client.events))
            try:
                yield client
            finally:
                drain.cancel()


async def _discard(events):
    while True:
        await events.get()


async def _submit_claim(client, params, board_db, claim=None):
    """Send the claim's one ``kanban.run`` and record the owner's admission on *claim* (a new one
    when None); return it."""
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, NOT_SENT
    claim = claim if claim is not None else _Claim(answered=time.monotonic())
    db = str(await asyncio.to_thread(Path(board_db).resolve))
    # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
    # in one board write transaction. The board (the cross-profile dispatch bus) can make that
    # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
    # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
    # admission was left ``unknown``). The wait is still finite: once the marker commits,
    # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
    # that stalls after admitting is bounded by this wait and ``RECEIPT_OUTAGE_S``.
    claim.in_doubt = True
    try:
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
    except GatewayClientError as exc:
        # An owner's answer (a refusal, e.g. ``runtime_draining``) or a request that never left
        # took nothing: only those may be asked again.
        if isinstance(exc, GatewayRPCError) or str(exc) == NOT_SENT:
            claim.in_doubt = False
        raise
    claim.in_doubt = False
    claim.session_id, claim.admission_id = accepted['session_id'], accepted['receipt']['admission_id']
    claim.answer(accepted['receipt'])
    print(f'Session: {claim.session_id}', file=sys.stderr)
    return claim


async def _follow(admission, client, params, board_db):
    """Read the admission's receipt until it is terminal or unknown, first on ``client`` and, after
    a lost or unanswered read, on new connections, until the owner has been silent for
    ``RECEIPT_OUTAGE_S``. An owner refusal ends it at once."""
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_owner_restart import restartable
    from websockets.exceptions import InvalidStatus, WebSocketException
    failures = 0
    try:
        async with asyncio.timeout(RECEIPT_OUTAGE_S) as admission.silence:
            while True:
                seen = admission.answered
                try:
                    if client is None:
                        # The owner ensure runs in a thread that a cancel cannot stop, and the
                        # process cannot exit before it ends: give it only the silence left.
                        left = admission.silence.when() - asyncio.get_running_loop().time()
                        async with _receipts(ready_timeout=max(left, 0.1)) as fresh:
                            await _observe(fresh, admission)
                    else:
                        await _observe(client, admission)
                    break
                except (GatewayClientError, WebSocketException, OSError, TimeoutError) as exc:
                    # An owner refusing the upgrade (InvalidStatus) is an answer, as is any other
                    # final refusal: observation ends at once.
                    if isinstance(exc, InvalidStatus) or not restartable(exc):
                        raise
                    if admission.silent_for() >= RECEIPT_OUTAGE_S:
                        raise _no_receipt(admission, exc) from exc
                    print(f'Receipt read lost ({_quoted(exc)}); re-reading admission {admission.admission_id}',
                          file=sys.stderr)
                client = None
                failures = 1 if admission.answered != seen else failures + 1
                await asyncio.sleep(jittered_backoff(failures, base_delay=1.0, max_delay=8.0, jitter_ratio=1.0))
    except TimeoutError:
        if admission.silence is None or not admission.silence.expired():
            raise
        raise _no_receipt(admission) from None
    return _settled(admission, params, board_db)


def _no_receipt(admission, exc=None):
    from hermes_cli.gateway_client import GatewayClientError
    detail = f' (owner silent for {admission.silent_for():.0f}s: {_quoted(exc)})' if exc is not None else ''
    return GatewayClientError(f'Gateway gave no receipt for admission {admission.admission_id} '
                              f'for {RECEIPT_OUTAGE_S:g} s; Kanban execution unknown{detail}')


async def _observe(client, admission):
    while admission.receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        admission.answer(await _receipt(client, admission))


async def _await_receipt(client, params, board_db, claim):
    """Submit the claim (or, when it was already admitted, re-read its admission) on *client* and
    return its settled receipt, recording each owner answer on *claim*."""
    if claim.admission_id is None:
        await _submit_claim(client, params, board_db, claim)
    await _observe(client, claim)
    return claim.receipt


def _settled(admission, params, board_db):
    return _exit_code(admission.receipt, params, board_db, admission.admission_id)


def _exit_code(receipt, params, board_db, admission_id=None):
    from hermes_cli.gateway_client import GatewayClientError
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
    if result.get('exit_code') == KANBAN_OWNER_RESTART_EXIT_CODE:
        admission_id = admission_id or receipt.get('admission_id')
        print(f'Kanban attempt interrupted by its owner\'s restart (admission {admission_id} '
              'settled interrupted; not replayed)', file=sys.stderr)
    return result.get('exit_code', 1)


async def _receipt(client, claim):
    return await client.rpc('prompt.receipt', _timeout=RECEIPT_ANSWER_BUDGET_S,
                            session_id=claim.session_id, admission_id=claim.admission_id)


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
