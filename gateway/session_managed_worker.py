"""Admission-owned production exec; observers never own the worker lifetime."""
import asyncio
from contextlib import contextmanager, suppress
from collections import deque
from dataclasses import asdict, replace
from functools import partial
import json
import os
import queue
import threading
from types import SimpleNamespace
from typing import NamedTuple
from pathlib import Path
import subprocess
import sys
import time

from agent.managed_worker import accept_result, encode_frame, read_frame
from gateway.session_worker_reservation import owner_worker_subject, reserve_admission_worker
from hermes_state_runtime import RuntimeStoreError

# The interpreter only imports psutil before it introduces itself; a child silent this long
# is wedged (loader/stdio stall), not slow.
HELLO_SECONDS = 60
# After Stop the child interrupts its agent and emits its result; the owner terminates a
# child that has not acknowledged within this window instead of waiting on the pipe forever.
STOP_ACK_SECONDS = 30
# How long a lost attempt's board note may hold up settling its admission (the note is
# advisory; a busy board's write finishes in the background).
LOSS_RECORD_WAIT_S = 10
# A correction's verdict comes back over the frame loop from the child's control reader, which
# answers as it reads. Controls on one client socket dispatch in arrival order, so this bounds how
# long a later message on that socket can wait behind an earlier unanswered correction; past it the
# outcome is unknown and its sender resends (a duplicate correction is acceptable, a lost one is not).
CORRECTION_ACK_SECONDS = 5
# Own session (POSIX): a worker whose owner died kills its whole group (managed_worker
# _die_with_owner); on Windows it exits itself.
# detached-spawn: ok — covered: the worker kills its own group when its owner dies (_die_with_owner)
WORKER_POPEN = {'start_new_session': True, 'close_fds': True}


def _describe_exit(returncode):
    """``returncode`` for the loss log: the signal's name when one ended the child, ``running`` when
    it has not exited yet."""
    if returncode is None:
        return 'running'
    if returncode < 0:
        import signal
        try:
            return f'{returncode} ({signal.Signals(-returncode).name})'
        except ValueError:
            return str(returncode)
    return str(returncode)


def _kanban_attempt(request_id):
    """The dispatcher attempt a Kanban admission carries in its request identity
    (``session_kanban``: ``kanban:[board, task_id, run_id]``), else None. Read, never reconstructed."""
    if not isinstance(request_id, str) or not request_id.startswith('kanban:'):
        return None
    try:
        attempt = json.loads(request_id[len('kanban:'):])
    except ValueError:
        return None
    if not isinstance(attempt, list) or len(attempt) != 3:
        return None
    board, task_id, run_id = attempt
    return {'board': board, 'task_id': task_id, 'run_id': run_id}


def _managed_exit_record(row, scope, process, lifecycle, returncode, exit_observed_at):
    """The worker's lifecycle as the owner observed it on the ordinary finish path, stored with the
    admission's settled result (``managed_exit``) so the finish, the ACK and the actual OS exit are
    correlated durably by card/run (the Kanban request identity), admission, generation and execution.

    ``exit`` is only what ``process.wait`` returned for the reserved interpreter itself. A wait that
    timed out, or a handle that is not the reserved interpreter (a launcher), leaves the status
    ``unknown`` with its reason: an exit is never inferred from the turn's or the board's result.
    Times are owner wall-clock seconds: ``result_received_at`` (the worker's result frame accepted),
    ``ack_sent_at`` (the owner's ``finish`` ACK written to the worker), ``finish_observed_at`` (the
    worker's ``finished``, which it sends only after its durable execution.finish receipt)."""
    handle_is_worker = process.pid == scope['pid']
    if returncode is None:
        exit_ = {'status': 'unknown', 'returncode': None, 'observed_at': None, 'reason': 'wait_timeout'}
    elif not handle_is_worker:
        exit_ = {'status': 'unknown', 'returncode': None, 'observed_at': None, 'reason': 'handle_is_not_the_worker',
                 'handle_returncode': returncode, 'handle_observed_at': exit_observed_at}
    elif returncode < 0:
        import signal
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = None
        exit_ = {'status': 'signalled', 'returncode': returncode, 'signal': name, 'observed_at': exit_observed_at}
    else:
        exit_ = {'status': 'exited', 'returncode': returncode, 'observed_at': exit_observed_at}
    return {'v': 1, 'request_id': row.get('request_id'), 'kanban': _kanban_attempt(row.get('request_id')),
            'admission_id': row['admission_id'], 'generation': row['generation'], 'owner_epoch': scope.get('epoch'),
            'execution_id': scope['execution_id'], 'worker_pid': scope['pid'], 'worker_birth': scope['birth'],
            'handle_pid': process.pid, **lifecycle, 'exit': exit_}


async def _fence_lost_worker(authority, row, scope, owner_restart=None):
    """Fence a lost worker's admission ``unknown``. The loss's cause is bound where the loss is found:
    an owner already stopping lost it to its own end, and that cause commits with the fence itself,
    so the settlement that follows (this drain's, or a later owner's boot) owes the card its restart
    result. A kill under a live owner stays a crash even if a stop begins afterwards. A caller that
    waits before fencing (``_settle_lost``) passes the ``owner_restart`` it read at the loss; None
    reads it now."""
    from gateway.session_runtime_workers import tracked_write
    from gateway.session_unattended_unknown import owner_ending
    from gateway.session_worker_reservation import lose_admission_worker
    if owner_restart is None:
        owner_restart = owner_ending(authority)
    await tracked_write(authority, partial(lose_admission_worker, authority, row, scope, owner_restart))


async def _settle_transferred(authority, ref, row, worker):
    """The fenced claim's transfers: the corrections its child confirmed consumed ran, so they settle
    ``interrupted`` and never run again; one whose hand-over was not recorded (its write failed, and
    may have committed) is queued again (``settle_steer_transfers``, as for an unknown in-process
    claim, ``_mark_unknown``): a duplicate, never left in transfer. A failed write leaves them
    registered to the claim, and the supported resolution of its ``unknown`` admission settles them
    the same way (``SessionAuthority._close_resolved``). Nothing for a worker that transferred none."""
    if not worker.transferred:
        return
    from gateway.session_queued_controls import publish_transfers, steer_transfers
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import _epoch, settle_steer_transfers
    with authority.sessions[ref.session_id].event_stream.lock:
        transfers = steer_transfers(authority, ref.session_id, row['generation'])

    def write(conn):
        _epoch(conn, authority.epoch)
        settle_steer_transfers(conn, epoch=authority.epoch, generation=row['generation'], transfers=transfers,
                               outcome='unknown')
    try:
        await tracked_write(authority, partial(authority.db._execute_write, write))
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Managed claim %s: transfers left for its resolution to settle',
                                              row['admission_id'])
        return
    publish_transfers(authority, ref, row['generation'])


class ManagedExecutionUnknown(RuntimeError):
    """The committed unknown verdict stops the FIFO without forging failed settlement."""


def managed_policy(authority, ref):
    """Bypass (safe / config-only) sessions always execute out of process; other local
    sessions only under the explicit custom-provider opt-in.

    The frozen creation snapshot, not current profile config, chooses execution.
    """
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        return None
    if policy.ignore_user_config or policy.kanban_json is not None:
        return policy
    if policy.config().get('gateway', {}).get('managed_workers') is not True:
        return None
    request = json.loads(policy.request_json)
    if policy.platform != 'cli' or request.get('provider') != 'custom' or not request.get('base_url'):
        raise RuntimeStoreError('unsupported_managed_policy')
    return policy


def _bootstrap(authority, ref, row, policy, scope, text=None):
    from gateway.session_ingress import row_turn_author
    from gateway.session_policy import route_launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    terminal = json.loads(policy.terminal_json)
    if policy.config_secret_ref:
        for path, value in recover_config_secrets(authority, policy).items():
            if path[0] is None:
                terminal[path[1]] = value
    live = authority.sessions[ref.session_id]
    # This turn's facts ride the per-turn hydrated request (the bootstrap field set is closed):
    # the admission's one-shot flags, the route's YOLO as of now (never the frozen launch flag) and
    # the committed surface (HUD / live voice / voice turn) when admission recorded one.
    request = dict(json.loads(policy.request_json), turn_v1={
        'finite': row['payload'].get('finite', False), 'unattended': row['payload'].get('unattended') is True,
        'yolo': _session_yolo(authority, live.route, policy),
        **({'surface_v1': row['payload']['surface_v1']} if row['payload'].get('surface_v1') else {}),
        **({'display_v1': row['payload']['display_v1']} if row['payload'].get('display_v1') else {})})
    from gateway.session_worker_construct import construct_inputs
    request['construct_v1'] = construct_inputs(authority, policy, live.source.chat_id)
    config = _worker_config(authority, policy)
    hydrated = replace(policy, config_json=json.dumps(config), request_json=json.dumps(request),
                       terminal_json=json.dumps(terminal), credential_ref=None, config_secret_ref=None)
    return {'version': 1, 'home': authority.profile_id, 'scope': scope,
            'policy': asdict(hydrated), 'api_key': route_launch_key(authority, policy, config),
            'text': row['payload']['text'] if text is None else text, 'route': live.route,
            **({'attachments_v1': row['payload']['attachments_v1']} if 'attachments_v1' in row['payload'] else {}),
            'user_id': live.source.user_id, 'chat_id': live.source.chat_id,
            'turn_author': row_turn_author(policy, row),
            'safe_mode': policy.safe_mode, 'ignore_user_config': policy.ignore_user_config}


def _worker_config(authority, policy):
    """The frozen session config the child binds as its only config (``bind_worker_policy``).
    ``command_allowlist`` is approval state rather than launch policy: an ``always`` grant an
    earlier worker of an ordinary session persisted, or one the operator removed by hand, comes
    from the live profile file, as it did before the child's config was frozen. Bypass sessions
    never read the profile."""
    config = policy.config(authority)
    home = Path(str(authority.profile_id))
    if policy.ignore_user_config or not home.is_absolute():
        return config
    from gateway.run import _load_gateway_config
    live = _load_gateway_config(home / 'config.yaml')
    config.pop('command_allowlist', None)
    if live.get('command_allowlist') is not None:
        config['command_allowlist'] = live['command_allowlist']
    return config


def _session_yolo(authority, route, policy):
    """The route's bypass as the in-process turn arms it on the owner, the one place a revocation
    is recorded: a ``--yolo`` launch seeded once per boundary, then the persisted ``/yolo`` copy."""
    from tools.approval import is_session_yolo_enabled
    from tools.approval_yolo import restore_gateway_yolo
    store = getattr(authority.runner, 'session_store', None)
    entry = store.lookup_by_session_key(route) if store is not None else None
    restore_gateway_yolo(route, getattr(entry, 'yolo', None), launch=bool(policy.yolo))
    return is_session_yolo_enabled(route)


@contextmanager
def worker_turn_scope(frame):
    """Child side of ``turn_v1``: bind what the in-process turn binds on the owner
    (execute_finite_admission, the route's YOLO, the committed surface), so ``chat -q``/``-z``
    never park a prompt, the session's current YOLO governs this child and a HUD / live-voice /
    voice turn reaches the model as it does in process. Frames without it bind nothing."""
    from gateway.session_finite import finite_turn_scope
    from gateway.session_display import display_turn_scope, restore_display
    from gateway.session_surface import restore_surface, surface_turn_scope
    turn = json.loads(frame['policy'].get('request_json') or '{}').get('turn_v1')
    if turn is None:
        yield
        return
    flags = {k: v for k, v in turn.items() if k not in ('surface_v1', 'display_v1')} if isinstance(turn, dict) else None
    if (flags is None or set(flags) != {'finite', 'unattended', 'yolo'}
            or any(type(v) is not bool for v in flags.values()) or (turn['unattended'] and not turn['finite'])):
        raise ValueError('invalid_managed_worker_bootstrap')
    try:
        surface = restore_surface(turn['surface_v1']) if 'surface_v1' in turn else None
        display = restore_display(turn['display_v1']) if 'display_v1' in turn else None
    except RuntimeStoreError as exc:
        raise ValueError('invalid_managed_worker_bootstrap') from exc
    if turn['yolo']:
        from tools.approval import enable_session_yolo
        enable_session_yolo(frame['route'])
    with finite_turn_scope(turn['finite'], turn['unattended']), surface_turn_scope(surface), \
            display_turn_scope(frame['scope']['execution_id'], display):
        yield


class ManagedWorker:
    def __init__(self, process):
        self.process = process
        # Verified interpreter behind the handle (a launcher trampoline may sit between).
        self.worker = None
        self.write_lock = threading.Lock()
        self.commands = queue.Queue(maxsize=16)
        self.closed = threading.Event()
        # Latched the moment Stop is admitted, before any pipe write: the owner's read loop
        # supervises it even while the child has not yet said hello or read its bootstrap.
        self.stop = asyncio.Event()
        # Monotonic instant of the FIRST Stop: one acknowledgment budget runs from here, never
        # renewed by the child's later output or by a repeated Stop.
        self.stopped_at = None
        # The gateway restart/shutdown reason when that stop (not a user's) signalled this worker.
        self.stop_cause = None
        self.writer = threading.Thread(target=self._write_controls, name='managed-control-writer', daemon=True)
        # One dedicated reader thread per worker (never the loop's shared default executor, which
        # parked pipe reads would exhaust). It reads one frame per demand, so the pipe keeps its
        # backpressure, and posts it here; a cancelled next_frame leaves the frame for the next.
        self.reader = None
        self._demand = threading.Semaphore(0)
        self._requested = False
        self._frames = deque()
        self._arrived = None
        self.stderr_drain = None
        # Exact secrets this worker was handed; the stderr sink scrubs them besides the patterns.
        self.secrets = []
        # correction id -> owner-loop future the child's ``correction`` frame answers (``correct``).
        self.corrections = {}
        # correction id -> its durable keeping (``KeptCorrection``), until ``consumed`` confirms it.
        self.kept = {}
        # Kept admissions registered as transfers to this claim (``settle_consumed``), whether or not
        # their hand-over write was confirmed: a loss settles every one (``_settle_transferred``).
        self.transferred = []
        # The launch policy this worker executes (``execute_managed``); its ``kanban_json`` names the
        # running card attempt whose task keeps this worker's corrections.
        self.policy = None
        # The child's ``ready``: its agent exists and its control reader answers as it reads.
        self.ready = False
        # The owner session and reserved bootstrap scope this worker executes (``execute_managed``):
        # the destination a private peer message must name exactly.
        self.session_id = None
        self.scope = None
        # (delivery id, attempt n) -> (owner-loop future of the child's verdict, settle callback) for
        # each forwarded private peer message attempt (``peer_message``).
        self.peer_messages = {}

    def _write_controls(self):
        try:
            while not self.closed.is_set():
                try:
                    frame = self.commands.get(timeout=.5)
                except queue.Empty:
                    continue
                self.send(frame)
        except (OSError, ValueError):
            self.closed.set()

    def _read_frames(self, loop):
        """Reader thread body. It owns ``stdout``: no other thread closes it under a blocked read."""
        try:
            while self._demand.acquire() and not self.closed.is_set():
                try:
                    item = read_frame(self.process.stdout)
                except (EOFError, ValueError, OSError, RecursionError) as exc:
                    item = exc  # EOF / invalid or oversized frame / closed pipe: the owner raises it
                try:
                    loop.call_soon_threadsafe(self._deliver, item)
                except RuntimeError:
                    return  # the owner loop has closed; nobody is left to read for
                if isinstance(item, Exception):
                    return
        finally:
            self.process.stdout.close()

    def _deliver(self, item):
        self._frames.append(item)
        self._arrived.set()

    def _take(self):
        item = self._frames[0]
        if isinstance(item, Exception):
            raise item  # stays queued: the reader has exited, every later read fails the same way
        self._frames.popleft()
        self._requested = False
        if not self._frames:
            self._arrived.clear()
        return item

    def drain_stderr(self, path):
        """Route the child's stderr (its tracebacks and redirected prints) into the profile's
        bounded, redacted, private log instead of discarding it."""
        from agent.memory_provider import spawn_context_thread
        from gateway.session_managed_worker_log import drain_worker_stderr
        self.stderr_drain = spawn_context_thread(drain_worker_stderr, name='managed-stderr-drain',
                                                 args=(self.process.stderr, path, self.process.pid, lambda: self.secrets))
        self.stderr_drain.start()

    def control(self, frame):
        if self.closed.is_set():
            raise RuntimeStoreError('managed_worker_lost')
        if frame == {'type': 'stop'}:
            if self.stopped_at is None:
                self.stopped_at = time.monotonic()
            self.stop.set()
        try:
            self.commands.put_nowait(frame)
        except queue.Full as exc:
            raise RuntimeStoreError('worker_control_backpressure') from exc

    def _stop_budget(self, ack):
        """Seconds left of the ``ack`` window that opened at the first Stop. Pure: only
        ``control`` stamps ``stopped_at``, so a read before any Stop (the hello) never starts
        the window a later Stop must receive in full."""
        return self.stopped_at + ack - time.monotonic()

    async def next_frame(self, timeout, ack):
        """Read one frame. A requested Stop bounds supervision to ``ack`` seconds TOTAL from the
        first Stop: a child that keeps emitting valid frames is escalated on the same deadline as a
        silent one, instead of leaving the turn started behind it."""
        if self.reader is None:
            self._arrived = asyncio.Event()
            from agent.memory_provider import spawn_context_thread
            self.reader = spawn_context_thread(self._read_frames, name='managed-frame-reader',
                                               args=(asyncio.get_running_loop(),))
            self.reader.start()
        if not self._frames and not self._requested:
            self._requested = True
            self._demand.release()
        arrived = asyncio.ensure_future(self._arrived.wait())
        stopper = asyncio.ensure_future(self.stop.wait())
        try:
            if not self._frames and not self.stop.is_set():
                await asyncio.wait({arrived, stopper}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not self._frames and not self.stop.is_set():
                    raise RuntimeStoreError('managed_worker_hello_timeout')
            if not self._frames:
                # Only a latched Stop gets here; its window runs from the first Stop.
                remaining = self._stop_budget(ack)
                if remaining > 0:
                    await asyncio.wait({arrived}, timeout=remaining)
            if not self._frames:
                raise RuntimeStoreError('managed_worker_stopped')
            return self._take()
        finally:
            stopper.cancel()
            arrived.cancel()

    def interrupt(self, message=None, **kwargs):
        """The Stop control, under the explicit-stop ABI (``agent.interrupt_compat``): the profile
        stop and ``adopt_agent`` reach an adopted worker through ``request_hard_interrupt``, which
        passes a message and may pass ``tool_reason``. The stop frame carries neither; a gateway
        stop's cause is recorded by ``stop_authority_turns``. The execution's ``finally`` drops the
        worker from ``_managed_workers`` and closes it while ``adopted`` still names it, so a closed
        worker has nothing left to stop. A full control queue is not an error either: ``control``
        latched ``stop`` first and the owner's read loop escalates on that latch."""
        try:
            self.control({'type': 'stop'})
        except RuntimeStoreError as exc:
            if exc.reason not in {'managed_worker_lost', 'worker_control_backpressure'}:
                raise
            import logging
            logging.getLogger(__name__).debug('Managed Stop not queued (%s)', exc.reason)

    def correct(self, verb, text, kept):
        """Hand ``session.steer`` / ``session.redirect`` text, already ``kept`` durably, to the
        child's agent over the control pipe: the future of the child's verdict
        (``WorkerControls.correct``), read by the owner loop's ``answer_correction``. Before
        ``ready`` the child can only latch the text for the agent it is building (it rejects
        nothing then but a Stop), so the verdict is ``queued`` at once instead of waiting out the
        child's startup; the child's own verdict is still read."""
        correction_id = kept.id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.corrections[correction_id] = future
        self.kept[correction_id] = kept
        try:
            self.control({'type': 'correct', 'id': correction_id, 'verb': verb, 'text': text})
        except BaseException:
            self.corrections.pop(correction_id, None)
            raise
        if self.ready:
            return future
        answered = loop.create_future()
        answered.set_result('rejected' if self.stop.is_set() else 'queued')
        return answered

    def peer_message(self, message, n, settle=None):
        """Forward attempt ``n`` of a private peer message to the child's agent: the future of the
        child's verdict (``WorkerControls.peer_message``): ``steered`` (buffered under the attempt's own
        ticket; never consumption), ``rejected`` (nothing was buffered) or ``unknown`` (the agent raised
        while taking it). A child lost before answering fails the future (outcome unknown); one that
        finished without reading the frame answers ``rejected``.

        Called under the session's stream lock, so it neither blocks nor writes. Before ``ready``, for
        a worker executing no card attempt, or for a message naming anything but this worker's exact
        profile, owner session, generation, board, task and run, it answers ``rejected`` at once:
        nothing is queued (never latched). ``RuntimeStoreError``: nothing was queued either.

        ``settle(outcome)`` runs off the loop, once, after the verdict, with ``consumed`` (a completed,
        flushed turn that ran it and left nothing unrun) or ``unconsumed`` (withdrawn unread when intake
        closed). An attempt whose consumption is unknown is never settled. The child answers before it
        settles; a settlement that still found its attempt unanswered resolves it ``steered`` (only a
        buffered attempt is settled), and one whose waiter gave up (cancelled) still runs ``settle``."""
        loop = asyncio.get_running_loop()
        key = (message.delivery_id, n)
        refused = loop.create_future()
        refused.set_result('rejected')
        if not self.ready or self.stop.is_set() or not self._names_this_attempt(message) or key in self.peer_messages:
            return refused
        future = loop.create_future()
        self.peer_messages[key] = (future, settle)
        try:
            self.control({'type': 'peer_message', 'id': message.delivery_id, 'n': n, 'text': message.text,
                          'author': dict(message.sender.author), 'profile': message.profile,
                          'session_id': self.scope['session_id'], 'generation': message.generation,
                          'board_db': message.board_db, 'task_id': message.task_id, 'run_id': message.run_id})
        except BaseException:
            self.peer_messages.pop(key, None)
            raise
        return future

    def _names_this_attempt(self, message):
        kanban = json.loads(getattr(self.policy, 'kanban_json', None) or 'null')
        return (kanban is not None and self.scope is not None
                and type(generation := getattr(message, 'generation', None)) is int
                and (getattr(message, 'profile', None), message.session_id, generation, message.board_db, message.task_id,
                     message.run_id) == (self.scope['profile_id'], self.session_id, self.scope['generation'],
                                         kanban['db'], kanban['task_id'], kanban['run_id']))

    def answer_peer_message(self, frame):
        """The child's ``peer_message.verdict`` frame resolves its forwarded attempt; one arriving after
        its waiter gave up is dropped. False for any other frame."""
        if (frame.get('type') != 'peer_message.verdict' or set(frame) != {'type', 'id', 'n', 'status'}
                or frame['status'] not in {'steered', 'rejected', 'unknown'}):
            return False
        entry = self.peer_messages.get((frame['id'], frame['n']))
        if entry is not None and not entry[0].done():
            entry[0].set_result(frame['status'])
        return True

    async def settle_peer_message(self, frame):
        """The child's ``peer_settled`` frame: run the attempt's ``settle`` once. False for any other frame."""
        if (frame.get('type') != 'peer_settled' or set(frame) != {'type', 'id', 'n', 'outcome'}
                or frame['outcome'] not in {'consumed', 'unconsumed'}):
            return False
        entry = self.peer_messages.pop((frame['id'], frame['n']), None)
        if entry is None:
            return True
        future, settle = entry
        if not future.done():
            # Only a buffered attempt is settled, so it was steered, even if its verdict has not arrived
            # (NC15-PEER-01): the attempt resolves here instead of being lost with its entry.
            future.set_result('steered')
        elif not future.cancelled() and (future.exception() is not None or future.result() != 'steered'):
            return True
        # A waiter that gave up (a cancelled future) does not end the attempt: its settlement still runs.
        if settle is not None:
            try:
                await asyncio.to_thread(settle, frame['outcome'])
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Peer message settlement failed')
        return True

    def answer_correction(self, frame):
        """The child's ``correction`` verdict frame resolves its waiting ``correct``; a verdict that
        arrives after its waiter gave up is dropped. False for any other frame."""
        if (frame.get('type') != 'correction' or set(frame) != {'type', 'id', 'status'}
                or frame['status'] not in {'redirected', 'queued', 'rejected', 'unknown'}):
            return False
        future = self.corrections.pop(frame['id'], None)
        if future is not None and not future.done():
            future.set_result(frame['status'])
        return True

    def abandon_corrections(self, *, unread):
        """The execution has ended with corrections still unanswered. ``unread``: the child finished
        normally, so it never read them (it answers every correction it reads before the owner's
        ``finish``) and no agent saw them. Otherwise it was lost and their outcome is unknown."""
        for future in self.corrections.values():
            if not future.done():
                if unread:
                    future.set_result('rejected')
                else:
                    future.set_exception(RuntimeStoreError('correction_outcome_unknown'))
        self.corrections.clear()
        # Private peer attempts the same way; a buffered attempt never settled stays unknown.
        for future, _ in self.peer_messages.values():
            if not future.done():
                if unread:
                    future.set_result('rejected')
                else:
                    future.set_exception(RuntimeStoreError('peer_message_outcome_unknown'))
        self.peer_messages.clear()

    def respond(self, kind, prompt_id, value):
        self.control({'type': kind, 'prompt_id': prompt_id, 'value': value})

    def send(self, frame):
        with self.write_lock:
            self.process.stdin.write(encode_frame(frame))
            self.process.stdin.flush()

    def _signal_worker(self, kill):
        """Signal the verified interpreter, not only the handle: a launcher that exec-chained
        or exited leaves the real worker outside the Popen's reach."""
        import psutil
        if self.worker is None or self.worker[0] == self.process.pid:
            return
        try:
            proc = psutil.Process(self.worker[0])
            from hermes_cli.process_identity import worker_birth
            if worker_birth(proc) == self.worker[1]:
                (proc.kill if kill else proc.terminate)()
        except psutil.Error:
            pass

    def _signal_group(self, kill):
        """POSIX: signal the worker's whole process group (it leads one, WORKER_POPEN), so the tool
        children a killed worker never reaps with _die_with_owner go too. Only while the leader is
        unreaped and still leads its group; otherwise (Windows, gone) the handle alone."""
        if os.name != 'nt' and self.process.poll() is None:
            import signal
            try:
                if os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGKILL if kill else signal.SIGTERM)  # windows-footgun: ok — POSIX only
                    return
            except (ProcessLookupError, PermissionError):
                pass
        (self.process.kill if kill else self.process.terminate)()

    def close(self):
        self.closed.set()
        self._demand.release()  # a reader parked between frames exits; one mid-read ends at EOF
        if self.process.poll() is None:
            self._signal_worker(kill=False)
            self._signal_group(kill=False)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._signal_worker(kill=True)
            self._signal_group(kill=True)
            # Still unreaped after SIGKILL (a launcher's handle): the teardown below must still run.
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
        else:
            self._signal_worker(kill=True)
        if self.writer.ident is not None:
            self.writer.join(timeout=5)
        self.process.stdin.close()
        if self.reader is None:
            self.process.stdout.close()
        else:
            self.reader.join(timeout=5)
        if self.stderr_drain is not None:
            self.stderr_drain.join(timeout=5)  # EOF once the child is gone; the thread closes it
        elif self.process.stderr is not None:
            self.process.stderr.close()


class KeptCorrection(NamedTuple):
    """Where a managed correction is durable before its sender hears anything: ``admission_id``, a
    queued row of the session's FIFO that runs as its own turn unless the child confirms it ran the
    text in this turn (``settle_consumed``); or, for a running Kanban card attempt, a comment on its
    task (``admission_id`` None: a queued row there would start a turn against an ended claim)."""
    id: str
    admission_id: str | None

    def unreached(self):
        """The ACK when the running turn never took the text: it still reaches the session as its own
        queued turn, or the card's next attempt through the comment."""
        if self.admission_id is None:
            return {'status': 'rejected', 'delivered': False, 'kept': 'task_comment'}
        return {'status': 'queued', 'admission_id': self.admission_id}


async def keep_correction(authority, actor, ref, worker, text):
    """Make the correction durable before the child sees it: nothing is acknowledged that only a
    process's memory holds. A failure here is an error, never an ACK, and its sender resends."""
    import uuid
    correction_id = uuid.uuid4().hex
    try:
        if worker.policy.kanban_json:
            from gateway.session_kanban import keep_on_board
            await asyncio.to_thread(keep_on_board, json.loads(worker.policy.kanban_json), text)
            return KeptCorrection(correction_id, None)
        from gateway.session_contract import Submission
        # The sender's own next prompt, exactly as ``prompt.submit`` admits one (route checks,
        # FIFO cap, scheduling): run as its own turn whenever this turn did not consume it.
        receipt = await authority.submit(actor, Submission('correction:' + correction_id, ref, {'text': text}, 'queue'))
    except RuntimeStoreError:
        raise
    except Exception as exc:
        raise RuntimeStoreError('correction_not_kept') from exc
    return KeptCorrection(correction_id, receipt.admission_id)


async def managed_correction(authority, actor, ref, worker, verb, text, generation):
    """``session.steer`` / ``session.redirect`` against a managed worker. The text is kept durably
    first (``keep_correction``), then the child's verdict is the canonical correction ACK. A
    ``rejected``, or a turn that ended before the text could be sent, answers where the kept text
    will still be delivered (``KeptCorrection.unreached``). A lost child, a failed application or a
    verdict past ``CORRECTION_ACK_SECONDS`` is ``correction_outcome_unknown``: the kept text is
    still delivered, and a resend is a harmless duplicate."""
    kept = await keep_correction(authority, actor, ref, worker, text)
    ack = {'text': text, 'execution_generation': generation, 'authority_epoch': authority.epoch}
    with authority.sessions[ref.session_id].event_stream.lock:
        reachable = (getattr(authority, '_managed_workers', {}).get(ref.session_id) is worker
                     and authority.executing.get(ref.session_id) == generation)
        verdict = worker.correct(verb, text, kept) if reachable else None
    if verdict is None:
        return {**ack, **kept.unreached()}
    try:
        status = await asyncio.wait_for(verdict, CORRECTION_ACK_SECONDS)
    except TimeoutError as exc:
        raise RuntimeStoreError('correction_outcome_unknown') from exc
    if status == 'unknown':
        raise RuntimeStoreError('correction_outcome_unknown')
    if status == 'rejected':
        return {**ack, **kept.unreached()}
    return {**ack, 'status': status, **({'admission_id': kept.admission_id} if kept.admission_id else {})}


def card_context(authority, session_id, generation):
    """The card attempt (its ``kanban_json``) a Kanban session was launched for, when ``generation``
    is one of that session's own claims; None for every other session or generation. A Kanban
    session's corrections belong to its card's task, after its claim ends as while it runs."""
    from hermes_state_local import POLICY_PREFIX
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
        claimed = conn.execute('SELECT 1 FROM session_admissions WHERE target_session_id=? AND generation=? LIMIT 1',
                               (session_id, generation)).fetchone()
    try:
        context = json.loads(json.loads(saved[0])['policy'].get('kanban_json') or 'null') if saved else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if (claimed is None or not isinstance(context, dict) or not isinstance(context.get('db'), str)
            or not isinstance(context.get('task_id'), str) or type(context.get('run_id')) is not int):
        return None
    return context


async def keep_card_correction(authority, context, text, generation, *, ended):
    """``session.steer`` / ``session.redirect`` for a Kanban session that no live agent can take:
    its claim has ended (``ended``: the ACK was lost, the claim replaced), or the claim runs but its
    worker is not yet attached. Kept on the card's task exactly as a running attempt's correction is
    (``keep_on_board``): the running worker's first poll reads a comment above its claim's floor, and
    the task's next attempt and its operator read it too. Answered as the kept ACK, so no client
    re-admits it as an unbound prompt; a resend keeps a duplicate, never a loss."""
    from gateway.session_kanban import keep_on_board
    try:
        await asyncio.to_thread(partial(keep_on_board, context, text, ended=ended))
    except Exception as exc:
        raise RuntimeStoreError('correction_not_kept') from exc
    return {'text': text, 'execution_generation': generation, 'authority_epoch': authority.epoch,
            **KeptCorrection(None, None).unreached()}


async def settle_consumed(authority, ref, row, worker, ids):
    """The child ran these corrections in its completed turn (its ``consumed`` frame, sent before
    the result). Each one's kept row is handed to this claim as a delivered transfer, so the claim's
    settlement records it ``steered`` (``settle_steer_transfers``) instead of running it again. A
    hand-over that fails leaves the row queued: it runs again, a duplicate and never a loss."""
    from gateway.session_queued_controls import _register, _unregister
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import confirm_steer_transfer, steer_session_input
    live, generation = authority.sessions[ref.session_id], row['generation']
    for correction_id in ids:
        kept = worker.kept.pop(correction_id, None) if isinstance(correction_id, str) else None
        if kept is None or kept.admission_id is None:
            continue
        with live.event_stream.lock:
            _register(authority, ref.session_id, generation, kept.admission_id)
        worker.transferred.append(kept.admission_id)
        try:
            await tracked_write(authority, partial(steer_session_input, authority.db, epoch=authority.epoch,
                                                   admission_id=kept.admission_id, generation=generation))
        except RuntimeStoreError:
            # Definitely not handed over (e.g. its sender cancelled it): it stays as it is.
            with live.event_stream.lock:
                _unregister(authority, ref.session_id, generation, kept.admission_id)
            continue
        except Exception:
            # May have committed: the registration stays undelivered, so settlement queues it again.
            import logging
            logging.getLogger(__name__).exception('Managed correction %s could not be recorded consumed',
                                                  kept.admission_id)
            continue
        with live.event_stream.lock:
            authority.steer_transfers[ref.session_id][1][kept.admission_id] = True
        try:
            # Durable beside the row, so a restarted owner too settles it as consumed, never runs it
            # again; unrecorded, a restart queues it again: a duplicate, never a loss.
            await tracked_write(authority, partial(confirm_steer_transfer, authority.db, epoch=authority.epoch,
                                                   admission_id=kept.admission_id, generation=generation))
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Managed correction %s consumption could not be recorded durably',
                                                  kept.admission_id)
    authority._publish_pending(ref)


def interrupt_managed(authority, actor, ref, generation):
    worker = getattr(authority, '_managed_workers', {}).get(ref.session_id)
    if worker is None:
        return False
    authority.authorize(actor, ref, 'session:control')
    authority.check_approval_generation(ref.session_id, generation)
    worker.control({'type': 'stop'})
    return True


async def _control_frame(authority, ref, row, worker, frame):
    """A frame answering or raising an owner control: the child's correction verdict or consumption,
    or one of its approval / clarify prompts (registered under the session's stream lock)."""
    if worker.answer_correction(frame) or worker.answer_peer_message(frame):
        return True
    if await worker.settle_peer_message(frame):
        return True
    if (frame.get('type') == 'consumed' and set(frame) == {'type', 'ids'} and isinstance(frame['ids'], list)):
        await settle_consumed(authority, ref, row, worker, frame['ids'])
        return True
    with authority.sessions[ref.session_id].event_stream.lock:
        return _prompt_frame(authority, ref, row, worker, frame)


def _prompt_frame(authority, ref, row, worker, frame):
    live = authority.sessions[ref.session_id]
    controls = live.controls
    kind = frame.get('type')
    if kind == 'prompt_settled' and set(frame) == {'type', 'prompt_id'}:
        prompt_id = frame['prompt_id']
        controls.remote_responders.pop(prompt_id, None)
        saved = controls.pending.pop(prompt_id, None)
        if saved:
            live.event_stream.publish(ref.session_id, {'prompt_id': prompt_id,
                'execution_generation': row['generation']}, event_type=saved[1]['kind'] + '.settled')
        return True
    if kind not in {'approval', 'clarify'}:
        return False
    if len(controls.remote_responders) >= 16:
        raise RuntimeStoreError('worker_control_backpressure')
    if kind == 'approval':
        fields = {'request_id', 'command', 'description', 'allow_session', 'allow_permanent', 'smart_denied', 'edit'}
        data = frame.get('data')
        if set(frame) != {'type', 'data'} or not isinstance(data, dict) or set(data) - fields:
            raise RuntimeStoreError('invalid_worker_frame')
        prompt_id = data.get('request_id')
    else:
        if (set(frame) != {'type', 'prompt_id', 'question', 'choices', 'multi_select'}
                or not isinstance(frame['question'], str) or not isinstance(frame['choices'], list)
                or any(not isinstance(c, str) for c in frame['choices']) or type(frame['multi_select']) is not bool):
            raise RuntimeStoreError('invalid_worker_frame')
        prompt_id = frame['prompt_id']
    if not isinstance(prompt_id, str) or not prompt_id or prompt_id in controls.pending:
        raise RuntimeStoreError('invalid_worker_frame')
    controls.remote_responders[prompt_id] = worker.respond
    if kind == 'approval':
        authority.register_approval(ref.session_id, row['generation'], live.route, data)
    else:
        entry = SimpleNamespace(clarify_id=prompt_id, question=frame['question'], choices=frame['choices'],
                                multi_select=frame['multi_select'], event=threading.Event())
        authority.register_clarify(ref.session_id, row['generation'], entry)
    return True


def _worker_env(authority):
    """Child env for the OWNING profile under multiplex: its HERMES_HOME plus its ``.env``
    secrets over a scrubbed base, never the launch profile's process environment (the same
    rule MCP stdio children and shell hooks follow) — except for the launch profile's own
    worker, whose scope legitimately includes its frozen launch env. Single-profile gateways
    inherit the process env byte-for-byte, exactly as before."""
    from pathlib import Path
    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import _is_routed_home
    home = Path(str(authority.profile_id))
    if not home.is_absolute():
        return None
    # Keyed on the worker's OWNING profile, not only the process-wide multiplex flag: a worker for
    # another profile must never inherit the launch environ even when that flag reads False.
    routed = _is_routed_home(home)
    if not routed and not is_multiplex_active():
        return None
    from agent.secret_scope import build_profile_secret_scope
    from tools.environments.local import _scrub_credentials, build_subprocess_env, strip_launch_profile_env
    # The scrub removes credentials, not settings: the launch profile's TERMINAL_* policy and
    # its ``.env`` settings would otherwise reach the secondary's worker (cron/kanban rule).
    # Strip the RAW environ before the constructor injects this turn's own session/bridge context:
    # a launch ``.env`` name (HERMES_SESSION_ID...) must not erase a value derived for this turn.
    env = build_subprocess_env(base=strip_launch_profile_env(os.environ.copy(), home), scrub_secrets=True)
    secrets = build_profile_secret_scope(home)
    if routed:
        # Same rule as served_profile_child_env: env_passthrough / first-party carve-outs must not
        # forward launch-process provider credentials that no .env or source snapshot recorded.
        _scrub_credentials(env, inherit_credentials=False)
    else:
        secrets = {**_launch_env_only_credentials(home, env), **secrets}
    env.update({k: v for k, v in secrets.items() if v is not None})
    env['HERMES_HOME'] = str(home)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _launch_env_only_credentials(home, env):
    """Credentials the LAUNCH profile's owner resolves from its frozen launch env
    (``launch_secret_scope``: systemd ``Environment=`` / ``op run`` keys with no ``.env`` line) that
    the scrub removed from *env* — the same mapping the owner's own ``get_secret`` reads for this
    profile. Credential names only: non-credential names the constructor removed on purpose
    (unbound session context, venv markers) stay removed, and Hermes-internal secrets
    (``AUXILIARY_*_API_KEY``, relay auth) never reach a child. Only the launch home's worker calls
    this; a routed home's worker sees its own files only."""
    from tools.environments.local import _scrub_credentials
    from tools.environments.local_env_policy import _is_hermes_internal_secret
    from tui_gateway.launch_profile_policy import launch_secret_scope
    missing = {k: v for k, v in launch_secret_scope(home).items()
               if k not in env and not _is_hermes_internal_secret(k)}
    plain = _scrub_credentials(dict(missing), inherit_credentials=False)
    return {k: v for k, v in missing.items() if k not in plain}


def _interrupted_before_bootstrap(authority, row, cause=None):
    authority.pending_results[row['admission_id']] = {
        'result': {'final_response': '', 'interrupted': True, **({'interrupt_message': cause} if cause else {})},
        'usage': {}}
    return ''


def _latched_stop_cause(authority, session_id, generation):
    """The gateway restart/shutdown cause latched for this generation before its worker existed
    (``run_runtime._stop_spawning_managed``), consumed once; None for a user's Stop."""
    latched = getattr(authority, 'pending_stop_causes', {}).pop(session_id, None)
    return latched[1] if latched and latched[0] == generation else None


def _with_stop_cause(accepted, worker):
    """The wire result carries no interruption cause; the owner recorded which stop it sent."""
    if accepted.get('interrupted') and worker.stop_cause:
        return {**accepted, 'interrupt_message': worker.stop_cause}
    return accepted


def _settled_session_usage(authority, session_id, accepted, execution_id=None):
    """A managed turn's ``message.complete.usage``: the worker exits with its turn, so the session-cumulative counters come from the rows its usage writes committed. A worker that compresses mid-turn ends on a child of this reservation's session; the owner's execution row records that tip (compression moves it in the same transaction), so the path from the tip back to the reservation plus the reservation's own compression ancestors is summed, and stale siblings or the conversation a branch came from never count."""
    import sqlite3
    db = authority.db
    try:
        tip = _execution_tip(db, execution_id)
        path = _continuation_path(db, session_id, tip) if tip and tip != session_id else None
        ids = list(dict.fromkeys([*_compression_ancestors(db, session_id), *(path or [session_id])]))
        rows = [db.get_session(sid) or {} for sid in ids]
    except sqlite3.Error:  # usage is display-only; settlement never waits on it
        return {}
    counts = {key: sum(int(row.get(key) or 0) for row in rows) for key in (
        'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens')}
    prompt = counts['input_tokens'] + counts['cache_read_tokens'] + counts['cache_write_tokens']
    return {'session_usage': {**counts, 'prompt_tokens': prompt, 'completion_tokens': counts['output_tokens'],
                              'total_tokens': prompt + counts['output_tokens'],
                              'api_calls': sum(int(row.get('api_call_count') or 0) for row in rows),
                              'model': str(accepted.get('model') or rows[-1].get('model') or '')}}


def _execution_tip(db, execution_id):
    """The session this worker execution last wrote to, as the owner's ledger records it."""
    if not execution_id:
        return None
    with db._read_ctx() as conn:
        row = conn.execute('SELECT session_id FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    return row[0] if row else None


def _continuation_path(db, start, tip):
    """Ids from *start* to *tip* when *tip* reaches *start* through compression continuations only, else None."""
    path, current = [tip], tip
    while current != start and len(path) <= 200:
        row = db.get_session(current)
        if not row or not db._is_compression_child_row(row):
            return None
        current = row['parent_session_id']
        path.append(current)
    return path[::-1] if current == start else None


def _compression_ancestors(db, session_id):
    """Root-first ids of the sessions *session_id* continues by compression, excluding itself."""
    ancestors, current = [], db.get_session(session_id)
    while current and db._is_compression_child_row(current) and len(ancestors) < 200:
        parent = current['parent_session_id']
        if parent in ancestors or parent == session_id:
            break
        ancestors.append(parent)
        current = db.get_session(parent)
    return ancestors[::-1]


def _mark_spawning(authority, ref, row):
    """Mark the generation pre-worker: until the worker registers, a gateway stop has nothing to
    signal, so it latches for this generation instead (#134889 GW-22). Set before ``@`` expansion,
    which may await the network, so a Stop there is never lost."""
    spawning = getattr(authority, '_managed_spawning', None)
    if spawning is None:
        spawning = authority._managed_spawning = {}
    spawning[ref.session_id] = row.get('generation')
    return spawning


def _stopped_before_spawn(authority, ref, row):
    """A Stop latched for this generation while it had no worker (a gateway stop during ``@``
    expansion, or a user's Stop): consume it and record the interrupted result, never spawning."""
    generation = row.get('generation')
    stops = getattr(authority, 'pending_stops', {})
    if generation is None or stops.get(ref.session_id) != generation:
        return False
    del stops[ref.session_id]
    authority._managed_spawning.pop(ref.session_id, None)
    _interrupted_before_bootstrap(authority, row, _latched_stop_cause(authority, ref.session_id, generation))
    return True


def _record_lost_attempt(authority, policy, exc, reason, scope):
    """Start the board disposition (``worker_lost``) for a lost Kanban card attempt; the task, or None.

    A worker that dies in ``worker.adopt`` never writes ``worker_bound``/``worker_result``, so
    without it the card's run learns nothing from its own loss. A deliberate Stop or a
    superseded generation is not a loss. Tracked, so retirement joins the write.
    """
    if (not getattr(policy, 'kanban_json', None) or isinstance(exc, asyncio.CancelledError)
            or reason in ('managed_worker_stopped', 'stale_generation')):
        return None
    from gateway.session_kanban import record_worker_loss
    from gateway.session_runtime_workers import track_mutation
    return track_mutation(authority, asyncio.to_thread(partial(record_worker_loss, policy, reason,
                                                               pid=(scope or {}).get('pid'), bound=scope is not None)))


async def _settle_lost(authority, row, scope, recording, *, owner_restart=None):
    """Settle a lost worker's admission ``unknown`` once its board note (``recording``) landed.

    The note comes first: the submitter exits on the settled 'unknown' and the crash sweep then
    books the bound interpreter, so a later write is dropped and an exit-0 worker_result reads
    as a protocol violation. A busy board holds this up ``LOSS_RECORD_WAIT_S`` at most (the
    write carries on, tracked), and a cancellation during the wait still settles.
    ``owner_restart``: the cause the caller read when it found the loss (``_fence_lost_worker``).
    """
    try:
        if recording is not None:
            await asyncio.wait({recording}, timeout=LOSS_RECORD_WAIT_S)
    finally:
        await _fence_lost_worker(authority, row, scope, owner_restart)


async def _spawn_registered_worker(authority, ref, row):
    """Spawn the turn's child and register it for Stops: ``(worker, process)``, or None when the spawn was cancelled (the
    interrupted result is already recorded). The generation stays marked spawning until registration."""
    spawning = _mark_spawning(authority, ref, row)
    generation = row.get('generation')
    try:
        env = await asyncio.to_thread(_worker_env, authority)
        cwd = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
        from gateway.session_worker_spawn import acquire_process
        from gateway.session_managed_worker_log import worker_log_path
        log_path = worker_log_path(authority.profile_id)
        process = await acquire_process(subprocess.Popen, [sys.executable, '-m', 'agent.managed_worker'],
            cwd=cwd, stdin=subprocess.PIPE, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if log_path is None else subprocess.PIPE, **WORKER_POPEN)
    except asyncio.CancelledError:
        # acquire_process owns late-child cleanup; no bootstrap or reservation was sent.
        spawning.pop(ref.session_id, None)
        _interrupted_before_bootstrap(authority, row, _latched_stop_cause(authority, ref.session_id, generation))
        return None
    except BaseException:
        spawning.pop(ref.session_id, None)
        raise
    worker = ManagedWorker(process)
    if log_path is not None:
        worker.drain_stderr(log_path)
    workers = getattr(authority, '_managed_workers', None)
    if workers is None:
        workers = authority._managed_workers = {}
    workers[ref.session_id] = worker
    spawning.pop(ref.session_id, None)
    worker.stop_cause = _latched_stop_cause(authority, ref.session_id, generation)
    return worker, process


async def admit_managed(authority, ref, row, policy):
    """The ingress route into a managed worker. A removed folder is refused before a worker exists,
    like the in-process runner: a worker failing in its policy scope reads as a LOST execution
    (``unknown``) and parks every follower behind it."""
    from gateway.session_policy import workspace_ready, workspace_unavailable_result
    if not workspace_ready(policy):
        result = workspace_unavailable_result(policy)
        authority.pending_results[row['admission_id']] = {'result': result, 'usage': {}}
        return result['final_response']
    return await execute_managed(authority, ref, row, policy)


async def _finalize_managed_refusal(authority, session_id, accepted):
    """Owner-side drop of a managed worker's safety-refused turn (Bot Chat / safe-mode).

    The worker flushed the refused turn to the SHARED transcript and has now exited (its lease
    released), but a managed worker cannot rewind its own transcript — so the refused turn must be
    dropped HERE or the next worker reloads it and the deterministic safety filter re-refuses every
    later message (the "stuck session" report; #149708 review). No cached owner agent to evict (the
    subprocess is gone), so session_key is None. Best-effort: a refusal must never crash settlement.
    """
    if accepted.get('failure_reason') != 'content_policy_blocked':
        return
    with suppress(Exception):
        await authority.runner._hmwa_retain_content_policy_refused_turn(
            session_id, None, accepted.get('refused_turn_boundary_row_id'))


async def execute_managed(authority, ref, row, policy):
    from gateway.run_turn_progress import publish_worker_tool_event
    # This route never passes the inbound prepare step, so the worker would read ``@`` tokens as typed.
    from gateway.session_context_refs import managed_turn_text, refused_turn_result
    _mark_spawning(authority, ref, row)
    try:
        text, refusal = await managed_turn_text(authority, row, policy)
    except BaseException:
        authority._managed_spawning.pop(ref.session_id, None)
        raise
    if refusal is not None:
        authority._managed_spawning.pop(ref.session_id, None)
        authority.pending_results[row['admission_id']] = {'result': refused_turn_result(row, refusal), 'usage': {}}
        return refusal
    if _stopped_before_spawn(authority, ref, row):
        return ''
    spawned = await _spawn_registered_worker(authority, ref, row)
    if spawned is None:
        return ''
    worker, process = spawned
    # Set before this coroutine first yields, so no correction reaches the worker without it.
    worker.policy = policy
    worker.session_id = ref.session_id
    workers = authority._managed_workers
    # A Stop acknowledged during the env/spawn awaits found no worker and was latched for this
    # generation; consume it in the same step that makes the worker reachable to interrupt_managed.
    authority.adopt_agent(ref.session_id, row['generation'], worker)
    accepted = usage = None
    scope = None
    lifecycle = {}
    try:
        # The interpreter behind the handle introduces itself first; the owner verifies that
        # identity (alive, same birth, descends from the handle) before reserving for it.
        hello = await worker.next_frame(HELLO_SECONDS, ack=0)
        if worker.stop.is_set():
            # Stopped before the child could receive controls: hello raced the latch; never bootstrap.
            raise RuntimeStoreError('managed_worker_stopped')
        # psutil + BEGIN IMMEDIATE: off-loop and retirement-tracked, like admission and claim.
        from gateway.session_runtime_workers import tracked_write
        scope = await tracked_write(authority, partial(reserve_admission_worker, authority,
                    admission_id=row['admission_id'], process=process, principal_id=row['principal_id'], hello=hello,
                    worker_subject=owner_worker_subject()))
        worker.worker = (scope['pid'], scope['birth'])
        worker.scope = scope
        # The child reads nothing else until the exact reservation has committed.
        frame = await asyncio.to_thread(_bootstrap, authority, ref, row, policy, scope, text)
        worker.secrets.extend(v for v in (frame['api_key'], scope['secret']) if isinstance(v, str) and v)
        await asyncio.to_thread(worker.send, frame)
        worker.writer.start()
        while True:
            frame = await worker.next_frame(None, ack=STOP_ACK_SECONDS)
            # The claim's execution stamp, not a ledger read per frame (publish_execution fences the same way).
            if authority.executing.get(ref.session_id) != row['generation']:
                raise RuntimeStoreError('stale_generation')
            if await _control_frame(authority, ref, row, worker, frame):
                continue
            if frame == {'type': 'error', 'reason': 'managed_worker_failed'}:
                raise RuntimeStoreError('managed_worker_failed')
            kind = frame.get('type')
            if kind == 'ready' and set(frame) == {'type', 'pid'} and frame['pid'] == scope['pid']:
                worker.ready = True
                continue
            if kind == 'delta' and set(frame) == {'type', 'text'} and isinstance(frame['text'], str):
                authority.publish_execution(ref.session_id, row['generation'], 'message.delta', {'text': frame['text']})
                continue
            if publish_worker_tool_event(authority, ref.session_id, row['generation'], frame):
                continue
            if kind == 'result' and set(frame) == {'type', 'result'} and accepted is None:
                try:
                    accepted, usage = accept_result(frame['result'])
                except ValueError as exc:
                    raise RuntimeStoreError('invalid_worker_result') from exc
                lifecycle['result_received_at'] = time.time()
                authority.sessions[ref.session_id].controls.snapshot(ref.session_id, None)
                await asyncio.to_thread(worker.send, {'type': 'finish'})
                lifecycle['ack_sent_at'] = time.time()
                continue
            if frame == {'type': 'finished'} and accepted is not None:
                # 'finished' follows the worker's durable execution.finish: the turn is complete.
                # A slow or non-zero interpreter exit after it is close()'s cleanup, not a loss.
                lifecycle['finish_observed_at'] = time.time()
                returncode = exit_observed_at = None
                with suppress(subprocess.TimeoutExpired):
                    returncode = await asyncio.to_thread(process.wait, 10)
                    exit_observed_at = time.time()
                managed_exit = _managed_exit_record(row, scope, process, lifecycle, returncode, exit_observed_at)
                import logging
                logging.getLogger(__name__).info(
                    'Managed worker finished: admission %s generation %s request %s execution %s pid %s exit %s (%s)',
                    row['admission_id'], row['generation'], row.get('request_id'), scope['execution_id'], scope['pid'],
                    managed_exit['exit']['status'], managed_exit['exit']['returncode']
                    if managed_exit['exit']['returncode'] is not None else managed_exit['exit']['reason'])
                accepted = _with_stop_cause(accepted, worker)
                await _finalize_managed_refusal(authority, scope['session_id'], accepted)
                # Like in-process execution, settlement belongs to the drain's stream lock.
                authority.pending_results[row['admission_id']] = {
                    'result': {**accepted, 'local_viewer': True}, 'usage': usage, 'managed_exit': managed_exit,
                    **await asyncio.to_thread(_settled_session_usage, authority, scope['session_id'], accepted, scope['execution_id'])}
                return accepted['final_response']
            raise RuntimeStoreError('invalid_worker_frame')
    except (Exception, asyncio.CancelledError) as exc:
        import logging
        reason = exc.reason if isinstance(exc, RuntimeStoreError) else type(exc).__name__
        if scope is None:
            logging.getLogger(__name__).warning('Managed worker lost: %s', reason)
            if isinstance(exc, asyncio.CancelledError) or (
                    isinstance(exc, RuntimeStoreError) and exc.reason == 'managed_worker_stopped'):
                # Stopped before the child ever received its bootstrap: nothing executed, so
                # this settles like an ordinary interrupted turn (the finally kills the child).
                return _interrupted_before_bootstrap(authority, row, worker.stop_cause)
            _record_lost_attempt(authority, policy, exc, reason, scope)  # nothing bound: a late note is dropped, never misbooked
            raise
        # The reserved child's exit status tells its own exit from a kill (a negative code names
        # the signal). Read without waiting: only the board note's bounded wait precedes the loss write.
        logging.getLogger(__name__).warning('Managed worker lost: %s (pid %s, exit %s)', reason,
            process.pid, _describe_exit(process.poll()))
        # The loss's cause is bound now, where it is found, before the board note's bounded wait.
        from gateway.session_unattended_unknown import owner_ending
        await _settle_lost(authority, row, scope, _record_lost_attempt(authority, policy, exc, reason, scope),
                           owner_restart=owner_ending(authority))
        await _settle_transferred(authority, ref, row, worker)
        live = authority.sessions[ref.session_id]
        with live.event_stream.lock:
            live.controls.snapshot(ref.session_id, None)
            authority._publish_pending(ref)
            live.event_stream.publish(ref.session_id, {'text': 'Worker execution is unknown.',
                'content': 'Worker execution is unknown.', 'admission_id': row['admission_id'], 'outcome': 'unknown',
                'status': 'error'})  # the message.complete TurnStatus: never painted as a normal finish
        waiter = authority.waiters.pop(row['admission_id'], None)
        if waiter is not None and not waiter.done():
            waiter.set_result('Worker execution is unknown.')
        # Stop this drain without its ordinary Exception→failed settlement. The
        # committed unknown row deliberately pauses every accepted follower.
        raise ManagedExecutionUnknown('managed_worker_unknown') from exc
    finally:
        workers.pop(ref.session_id, None)
        worker.abandon_corrections(unread=lifecycle.get('finish_observed_at') is not None)
        await asyncio.to_thread(worker.close)
