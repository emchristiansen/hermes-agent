"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import contextlib
import os
from pathlib import Path
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
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


class _Admission:
    """The owner's answer to this claim's one ``kanban.run``: what every later receipt read names.
    ``silence`` is the timeout bounding the owner's silence; each answer pushes it back."""

    def __init__(self, session_id, receipt):
        self.session_id, self.receipt, self.silence = session_id, receipt, None

    def answered(self, receipt):
        self.receipt = receipt
        self.silence.reschedule(asyncio.get_running_loop().time() + RECEIPT_OUTAGE_S)


async def run(params, board_db):
    from hermes_cli.gateway_client import GatewayClientError, GatewayUnavailableError
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    deadline = time.monotonic() + OWNER_WAIT_S
    admission = None
    while True:
        try:
            async with _receipts() as client:
                admission = await _submit_claim(client, params, board_db)
                return await _follow(admission, client, params, board_db)
        except GatewayUnavailableError as exc:
            # Only before admission: an admitted claim is never submitted again.
            if admission is not None or exc.state not in _TRANSIENT_OWNER_STATES or time.monotonic() >= deadline:
                raise
        await asyncio.sleep(1)


@contextlib.asynccontextmanager
async def _receipts(ready_timeout=None):
    """A connection to the profile owner whose turn events are discarded: kanban.run subscribes
    the socket to the card's session, but only receipts matter here, and undrained events fill
    the client's bounded queue until its reader stops with the next receipt behind them."""
    from hermes_cli.gateway_client import connect_gateway
    async with (connect_gateway() if ready_timeout is None else connect_gateway(ready_timeout=ready_timeout)) as client:
        drain = asyncio.create_task(_discard(client.events))
        try:
            yield client
        finally:
            drain.cancel()


async def _discard(events):
    while True:
        await events.get()


async def _submit_claim(client, params, board_db):
    db = str(await asyncio.to_thread(Path(board_db).resolve))
    # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
    # in one board write transaction. The board (the cross-profile dispatch bus) can make that
    # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
    # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
    # admission was left ``unknown``). The wait is still finite: once the marker commits,
    # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
    # that stalls after admitting is bounded only here. A hang-up fails the wait sooner.
    accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
    print(f'Session: {accepted["session_id"]}', file=sys.stderr)
    return _Admission(accepted['session_id'], accepted['receipt'])


async def _follow(admission, client, params, board_db):
    """Read the admission's receipt until it is terminal or unknown, first on ``client`` and, after
    a lost or unanswered read, on new connections, until the owner has been silent for
    ``RECEIPT_OUTAGE_S``. An owner refusal ends it at once."""
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
    try:
        async with asyncio.timeout(RECEIPT_OUTAGE_S) as admission.silence:
            while True:
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
                except GatewayRPCError:
                    raise
                except (GatewayClientError, OSError) as exc:
                    print(f'Receipt read lost ({exc}); re-reading admission {admission.receipt["admission_id"]}',
                          file=sys.stderr)
                client = None
                await asyncio.sleep(1)
    except TimeoutError:
        raise GatewayClientError(f'Gateway gave no receipt for admission {admission.receipt["admission_id"]} '
                                 f'for {RECEIPT_OUTAGE_S:g} s; Kanban execution unknown') from None
    return _settled(admission, params, board_db)


async def _observe(client, admission):
    while admission.receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        admission.answered(await client.rpc('prompt.receipt', _timeout=RECEIPT_ANSWER_BUDGET_S,
                                            session_id=admission.session_id,
                                            admission_id=admission.receipt['admission_id']))


def _settled(admission, params, board_db):
    from hermes_cli.gateway_client import GatewayClientError
    if admission.receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


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
