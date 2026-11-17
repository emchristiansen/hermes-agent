"""Private exec entry. No agent/config imports before adopted process authority.

stdin is the owner's bounded bootstrap/control pipe; a duplicate of stdout is
reserved for framed events before runtime imports redirect ordinary output.
Neither assignment secrets nor launch credentials appear in argv or logs.
"""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap" and not "hermes_bootstrap".startswith(f"{exc.name}."):
        raise

import contextlib
import json
import logging
import os
import re
from pathlib import Path
import sys
import threading

logger = logging.getLogger(__name__)

MAX_FRAME = 4 * 1024 * 1024
BOOTSTRAP_FIELDS = {'version', 'home', 'scope', 'policy', 'api_key', 'text', 'route', 'user_id', 'chat_id',
                    'turn_author', 'safe_mode', 'ignore_user_config'}
# The bounded run_conversation fields a turn receipt keeps (outcome, exit reason, tokens, cost,
# model): what the in-process projection commits and ``-z --usage-file`` reads.
RESULT_FIELDS = ('final_response', 'failed', 'interrupted', 'completed', 'partial', 'error', 'turn_exit_reason',
                 'api_calls', 'model', 'provider', 'prompt_tokens', 'completion_tokens', 'total_tokens',
                 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens',
                 'estimated_cost_usd', 'cost_status', 'cost_source', 'service_tier', 'failure_reason',
                 'session_id', 'requested_model', 'served_model', '_turn_input_tokens', '_turn_output_tokens',
                 'failure_retryable', 'failure_resets_at', 'billing_block', 'free_tier', 'usage_window_message',
                 'refused_turn_boundary_row_id')
# The structured fields and their allowed keys: ``billing_block`` (``agent/billing_links.BillingBlock``
# plus ``unverified``) and ``free_tier`` (``agent/turn_recovery._stamp_free_tier``). With
# ``failure_reason``, ``failure_retryable`` and ``failure_resets_at`` they are what
# ``agent.error_surface.build_error_surface_from_result`` reads for the owner's error card.
BILLING_FIELDS = ('provider', 'provider_label', 'model', 'billing_url', 'is_nous', 'message', 'unverified')
STRUCTURED_FIELDS = {'billing_block': BILLING_FIELDS, 'free_tier': ('kind', 'message')}


def _scalar(value):
    return value is None or type(value) in (bool, int, float) or (isinstance(value, str) and len(value) <= 4096)


def _structured(key, value):
    return isinstance(value, dict) and not set(value) - set(STRUCTURED_FIELDS[key]) and all(map(_scalar, value.values()))


def result_frame(result):
    """Worker side: the receipt fields of ``run_conversation``'s result, scalars only, text bounded."""
    frame = {k: (v[:4096] if isinstance(v, str) and k != 'final_response' else v) for k, v in result.items()
             if k in RESULT_FIELDS and k not in STRUCTURED_FIELDS and (v is None or type(v) in (bool, int, float, str))}
    for key, fields in STRUCTURED_FIELDS.items():
        if isinstance(block := result.get(key), dict):
            frame[key] = {k: v[:4096] if isinstance(v, str) else v for k, v in block.items()
                          if k in fields and (v is None or type(v) in (bool, int, float, str))}
    return frame


def accept_result(result):
    """Owner side of the worker's result frame: ``(result, usage)`` in the in-process turn's shape
    (gateway projection: prompt/completion tokens as input/output), or ValueError."""
    if (not isinstance(result, dict) or set(result) - set(RESULT_FIELDS) or not isinstance(result.get('final_response'), str)
            or any(result.get(key) is not None and not _structured(key, result[key]) for key in STRUCTURED_FIELDS)
            or any(not _scalar(value) for key, value in result.items()
                   if key != 'final_response' and key not in STRUCTURED_FIELDS)):
        raise ValueError('invalid_worker_result')
    accepted = dict(result)
    for key in STRUCTURED_FIELDS:
        if accepted.get(key) is None:
            # A failure other than a billing wall or a free-tier refusal carries no block.
            accepted.pop(key, None)
    turn_in, turn_out = accepted.pop('_turn_input_tokens', None), accepted.pop('_turn_output_tokens', None)
    if any(value is not None and (type(value) is not int or value < 0) for value in (turn_in, turn_out)):
        raise ValueError('invalid_worker_result')
    incoming, outgoing = accepted.pop('prompt_tokens', None), accepted.pop('completion_tokens', None)
    if incoming is None and outgoing is None:
        return accepted, {}
    usage = {'input_tokens': incoming or 0 if turn_in is None else turn_in,
             'output_tokens': outgoing or 0 if turn_out is None else turn_out}
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    accepted.update(input_tokens=incoming or 0, output_tokens=outgoing or 0)
    return accepted, usage


def read_frame(stream):
    line = stream.readline(MAX_FRAME + 1)
    if not line:
        raise EOFError('managed_worker_pipe_closed')
    if len(line) > MAX_FRAME or not line.endswith(b'\n'):
        raise ValueError('managed_worker_frame_too_large')
    frame = json.loads(line)
    if not isinstance(frame, dict):
        raise ValueError('invalid_managed_worker_frame')
    return frame


def encode_frame(frame):
    data = json.dumps(frame, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(data) > MAX_FRAME:
        raise ValueError('managed_worker_frame_too_large')
    return data


def validate_bootstrap(frame):
    if set(frame) - {'attachments_v1'} != BOOTSTRAP_FIELDS or frame['version'] != 1:
        raise ValueError('invalid_managed_worker_bootstrap')
    scope = frame['scope']
    fields = {'profile_id', 'session_id', 'execution_id', 'generation', 'pid', 'birth', 'secret', 'epoch'}
    if (not isinstance(scope, dict) or set(scope) != fields or scope['pid'] != os.getpid()
            or scope['profile_id'] != frame['home'] or not Path(frame['home']).is_absolute()
            or not isinstance(frame['policy'], dict)
            or any(not isinstance(frame[k], str) for k in ('text', 'route', 'user_id', 'chat_id'))
            or type(frame['safe_mode']) is not bool or type(frame['ignore_user_config']) is not bool
            or (frame['safe_mode'] and not frame['ignore_user_config'])
            or (frame['api_key'] is not None and not isinstance(frame['api_key'], str))):
        raise ValueError('invalid_managed_worker_bootstrap')
    from agent.turn_author import parse_turn_author
    if frame['turn_author'] is not None and (not isinstance(frame['turn_author'], dict)
                                            or parse_turn_author(frame['turn_author']) is None):
        raise ValueError('invalid_managed_worker_bootstrap')
    return frame


def bind_worker_policy(frame):
    """Freeze the owner's assignment process-wide before any config/provider/agent import.

    The session's frozen config (never the live profile file) is what every config reader in
    this process returns afterwards, so a profile edit after creation cannot reshape it (MCP
    server definitions included). Ordinary sessions keep plugins, MCP and hooks; bypass sessions
    also bind the safe / ignore-user-config policy.
    """
    config = json.loads(frame['policy']['config_json'])
    if not frame['ignore_user_config']:
        from agent.safe_worker_policy import _bind_worker_config
        _bind_worker_config(config)
        return
    from agent.safe_worker_policy import _bind_safe_worker_policy
    _bind_safe_worker_policy(safe_mode=frame['safe_mode'], ignore_user_config=True, config=config)


class WorkerChannel:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def send(self, kind, **payload):
        data = encode_frame({'type': kind, **payload})
        with self.lock:
            self.stream.write(data)
            self.stream.flush()


_INTERRUPTED = object()
# The owner's control pipe closing under a live worker means the owner died (a closing owner
# signals the worker itself). The agent is interrupted at once; cooperative tools get this long.
ORPHAN_GRACE_SECONDS = 10


def _die_with_owner():
    """Then end this worker and, on POSIX, every descendant still in its process group: the owner
    spawns it as a group leader (``start_new_session``). Tools that open their own session (the
    terminal tool, background processes meant to outlive the turn) are not in it. Never a group
    this process does not lead."""
    def kill():
        if os.name != 'nt' and os.getpgrp() == os.getpid():
            import signal
            os.killpg(os.getpid(), signal.SIGKILL)  # windows-footgun: ok — POSIX only
        os._exit(1)
    threading.Timer(ORPHAN_GRACE_SECONDS, kill).start()


def _control_input_stream():
    """A private buffer cannot strand the interpreter-owned stdin lock on exit."""
    descriptor = os.dup(sys.stdin.fileno())
    return os.fdopen(descriptor, 'rb')


# The ``peer_message`` control: one attempt of a private peer message for this exact card attempt.
_PEER_FRAME = {'type', 'id', 'n', 'text', 'author', 'profile', 'session_id', 'generation', 'board_db', 'task_id', 'run_id'}
# The destination a peer message must name: this interpreter's bootstrap scope and its bound card attempt.
_PEER_DESTINATION = ('profile', 'session_id', 'generation', 'board_db', 'task_id', 'run_id')


def peer_ticket(delivery_id, n):
    """The steer ticket of attempt *n* of a peer message. Per attempt, never per id: a resent id is
    buffered again as its own part, and ``withdraw_steer`` settles every part carrying one ticket at
    once (and answers False for a ticket it no longer knows), so a shared ticket would let one
    attempt's settlement decide, or misreport, another's."""
    return f'peer:{delivery_id}:{n}'


def peer_destination(frame):
    """The private-peer destination of this bootstrap (``frame``): its scope's profile, physical
    session and generation, and the card attempt ``bind_worker_context`` bound (board, task, run and
    claim lock). None when it executes no card attempt: such a worker takes no peer message."""
    kanban = json.loads(frame['policy'].get('kanban_json') or 'null')
    if kanban is None:
        return None
    scope = frame['scope']
    return {'profile': scope['profile_id'], 'session_id': scope['session_id'], 'generation': scope['generation'],
            'board_db': kanban['db'], 'db': kanban['db'], 'task_id': kanban['task_id'], 'run_id': kanban['run_id'],
            'claim_lock': kanban['claim_lock']}


class WorkerControls:
    # The private-peer destination (``peer_destination``) once ``accept_peer_messages`` opts this card
    # worker in; None takes no peer message.
    peer = None

    def __init__(self, channel, route, stream=None):
        self.channel, self.route = channel, route
        # This daemon may still be reading when a failed turn exits. Never hold
        # the interpreter-owned stdin buffer's lock during Py_Finalize: that
        # turns an ordinary persistence error into _enter_buffered_busy/ABRT.
        # main passes the SAME private stream used for bootstrap, preserving
        # any control frames already read ahead with that bootstrap.
        self.stream = stream if stream is not None else _control_input_stream()
        self.agent = None
        self.stopped = threading.Event()
        self.finish = threading.Event()
        self.clarifications = {}
        self.lock = threading.Lock()
        # Owner corrections (``session.steer`` / ``session.redirect``): accepted until ``settle``
        # closes intake; text that arrives before the agent exists is latched for it (``adopt``).
        # The owner keeps each one durably before sending it; ``consumed`` names the ones this
        # execution can confirm it ran, so the owner stops keeping them.
        self.corrections_open = True
        self.latched = []
        self.accepted = []
        self.unconsumed = False
        # Private peer messages (``peer_message``): each buffered attempt's ticket -> (delivery id, n),
        # and, once ``settle`` closes intake, each ticket's fate: True withdrawn unread, False drained
        # into a turn. They share ``corrections_open`` and ``self.lock`` with the corrections: intake
        # check, buffering and close are one order.
        self.peer_tickets = {}
        self.peer_fates = {}
        self.reader = threading.Thread(target=self.read, name='managed-controls', daemon=True)
        self.reader.start()

    def stop(self):
        self.stopped.set()
        if self.agent is not None:
            from agent.interrupt_compat import request_hard_interrupt
            request_hard_interrupt(self.agent)  # parity with the canonical Stop (SessionAuthority.interrupt)
        with self.lock:
            for state in self.clarifications.values():
                state['answer'] = _INTERRUPTED
                state['event'].set()

    def read(self):
        from tools.approval import resolve_gateway_approval
        try:
            while True:
                frame = read_frame(self.stream)
                if frame == {'type': 'stop'}:
                    self.stop()
                    continue
                if frame == {'type': 'finish'}:
                    self.finish.set()
                    return
                if frame.get('type') == 'correct':
                    self.correct(frame)
                    continue
                if frame.get('type') == 'peer_message':
                    self.peer_message(frame)
                    continue
                if (set(frame) != {'type', 'prompt_id', 'value'} or frame['type'] not in {'approval', 'clarify'}
                        or not isinstance(frame['prompt_id'], str) or not isinstance(frame['value'], str)
                        or len(frame['value']) > 16384):
                    raise ValueError('invalid_managed_control')
                if frame['type'] == 'approval':
                    if frame['value'] not in {'once', 'deny', 'session', 'always'}:
                        raise ValueError('invalid_managed_control')
                    resolve_gateway_approval(self.route, frame['value'], request_id=frame['prompt_id'])
                else:
                    with self.lock:
                        state = self.clarifications.get(frame['prompt_id'])
                        if state is not None:
                            state['answer'] = frame['value']
                            state['event'].set()
        except (EOFError, OSError, ValueError) as exc:
            self.stop()
            self.finish.set()
            if not isinstance(exc, ValueError):
                _die_with_owner()
        finally:
            self.stream.close()

    def adopt(self, agent):
        """The constructed agent takes the controls. Corrections latched while it was being built
        steer into it now, before its first model call (the owner's ``adopt_agent`` parity)."""
        with self.lock:
            self.agent = agent
            for text in self.latched:
                agent.steer(text)
            self.latched.clear()

    def correct(self, frame):
        """An owner correction, answered with this child's verdict in a ``correction`` frame.

        ``redirected``: only when a redirect cancelled the in-flight model request
        (``redirect_outcome`` path ``model``); that response is fenced and the same turn rebuilds
        around the text. ``queued``: the text was taken for the next boundary: a steer, or a
        redirect that met a tool batch (steered there, tool workers asked to yield), Codex's
        native turn steer, or no live request. ``rejected``: no agent saw the text (intake closed
        or the turn was stopped). ``unknown``: the agent raised while taking it. None of these is
        consumption: the owner keeps the text until ``consumed`` confirms it."""
        if (set(frame) != {'type', 'id', 'verb', 'text'} or frame['verb'] not in {'steer', 'redirect'}
                or not isinstance(frame['id'], str) or not frame['id']
                or not isinstance(frame['text'], str) or not frame['text'].strip()):
            raise ValueError('invalid_managed_control')
        try:
            with self.lock:
                status = self._apply_correction(frame['verb'], frame['text'])
                if status in ('queued', 'redirected'):
                    self.accepted.append(frame['id'])
        except Exception:
            # The agent may or may not hold the text: the owner reports the outcome unknown (its
            # sender resends; a duplicate is acceptable) and this reader keeps serving Stop.
            logger.exception('managed correction failed')
            status = 'unknown'
        self.channel.send('correction', id=frame['id'], status=status)

    def accept_peer_messages(self, destination):
        """Take private peer messages naming exactly ``destination`` (``peer_destination``), from now on."""
        with self.lock:
            self.peer = destination

    def peer_message(self, frame):
        """One attempt of a private peer message, answered with a ``peer_message.verdict`` frame:
        ``steered`` once its text is in the agent's steer buffer under its own attempt ticket, else
        ``rejected`` (nothing was buffered), or ``unknown`` when the agent raised while taking it.
        ``steered`` is never consumption: ``peer_settlements`` reports that at the turn's end.

        The text reaches the agent only while intake is open, the agent exists, the frame names exactly
        this interpreter's destination, and the bound claim is the card's live run (the board fence,
        ``steer_peer_if_bound``); all of it, and the verdict, under ``self.lock``, the lock ``settle``
        closes intake under, so no attempt is buffered after its settlement was decided, nor answered
        after it was settled. It is never latched, kept on the board or turned into a prompt."""
        if (set(frame) != _PEER_FRAME or not isinstance(frame['id'], str) or not frame['id']
                or type(frame['n']) is not int or frame['n'] < 1
                or not isinstance(frame['text'], str) or not frame['text'].strip()
                or not isinstance(frame['author'], dict)):
            raise ValueError('invalid_managed_control')
        ticket = peer_ticket(frame['id'], frame['n'])
        # The verdict is written before ``self.lock`` is released (NC15-PEER-01): ``settle`` closes
        # intake under it, and the turn thread sends ``peer_settled`` only after that, so on the wire an
        # attempt's verdict always precedes its settlement. (``self.channel`` takes only its own write
        # lock, never this one.)
        with self.lock:
            try:
                status = self._take_peer_message(frame, ticket)
            except Exception:
                logger.exception('managed peer message failed')
                status = 'unknown'
            self.channel.send('peer_message.verdict', id=frame['id'], n=frame['n'], status=status)

    def _take_peer_message(self, frame, ticket):
        if (self.peer is None or not self.corrections_open or self.stopped.is_set() or self.agent is None
                or ticket in self.peer_tickets
                or any(frame[key] != self.peer[key] for key in _PEER_DESTINATION)):
            return 'rejected'
        from gateway.session_kanban import steer_peer_if_bound
        agent = self.agent
        if not steer_peer_if_bound(self.peer, lambda: agent.steer(frame['text'], ticket=ticket,
                                                                   delivered_by=dict(frame['author']))):
            return 'rejected'
        self.peer_tickets[ticket] = (frame['id'], frame['n'])
        return 'steered'

    def peer_settlements(self, result):
        """``(delivery id, n, outcome)`` for each buffered peer attempt, for the ``peer_settled``
        frames sent before the result. ``unconsumed``: withdrawn unread when intake closed (never run;
        its sender may resend it). ``consumed``: a turn drained it, and that turn completed, flushed and
        closed intake with nothing unrun, the same proof ``consumed`` requires. Any other drained
        attempt gets no settlement: its consumption is unknown."""
        with self.lock:
            proven = not (self.unconsumed or self.stopped.is_set() or self.corrections_open
                          or result.get('completed') is not True or result.get('failed') or result.get('interrupted'))
            settled = []
            for ticket, (delivery_id, n) in self.peer_tickets.items():
                unread = self.peer_fates.get(ticket)
                if unread is True:
                    settled.append((delivery_id, n, 'unconsumed'))
                elif unread is False and proven:
                    settled.append((delivery_id, n, 'consumed'))
            return settled

    def _apply_correction(self, verb, text):
        if not self.corrections_open or self.stopped.is_set():
            return 'rejected'
        if self.agent is None:
            self.latched.append(text)
            return 'queued'
        if verb == 'redirect':
            path = self.agent.redirect_outcome(text)
            if path == 'model':
                return 'redirected'
            if path is not None:
                return 'queued'
        return 'queued' if self.agent.steer(text) else 'rejected'

    def settle(self, result, *, may_continue=True):
        """After each turn: the steer text that turn could not drain (its ``pending_steer``, plus
        any steer that landed since) for a follow-up turn of this admission, or None.

        A None closes intake: the check and the close are one step under the lock, so no
        correction lands between the last drain and the result frame. Nothing follows a stopped
        turn, or one whose caller may not continue (a Kanban attempt whose claim ended). That text
        is not run here, and this execution then confirms none of its corrections (``consumed``):
        each stays kept by the owner, durably, for a later turn or attempt."""
        with self.lock:
            parts = [result.pop('pending_steer', None) if isinstance(result, dict) else None]
            if self.agent is not None:
                drain = getattr(self.agent, '_drain_leftover_steer', None) or self.agent._drain_pending_steer
                parts.append(drain())
            text = '\n'.join(part for part in parts if part)
            if text and may_continue and not self.stopped.is_set():
                return text
            self.corrections_open = False
            self.unconsumed = self.unconsumed or bool(text)
            # Intake is closed: each peer attempt is withdrawn now if no turn drained it, so none can
            # be read after its settlement (``withdraw_steer``: True unread, False drained).
            if self.peer is not None:
                for ticket in self.peer_tickets:
                    self.peer_fates[ticket] = self.agent.withdraw_steer(ticket) if self.agent is not None else False
        if text:
            logger.warning('Managed worker ended its admission with %d characters of steer text it did not run; '
                           'the owner keeps its corrections', len(text))
        return None

    def consumed(self, result):
        """The corrections this execution ran, for the ``consumed`` frame sent before its result:
        every one it accepted, but only after a completed turn whose final flush succeeded and
        whose intake closed with nothing left unrun. Any doubt confirms nothing, and the owner's
        kept copies then run again (a duplicate, never a loss)."""
        with self.lock:
            if (self.unconsumed or self.stopped.is_set() or self.corrections_open
                    or result.get('completed') is not True or result.get('failed') or result.get('interrupted')):
                return []
            return list(self.accepted)

    def approval(self, data):
        from tools.approval import ack_gateway_approval
        fields = {'request_id', 'command', 'description', 'allow_session', 'allow_permanent', 'smart_denied', 'edit'}
        self.channel.send('approval', data={k: v for k, v in data.items() if k in fields})
        ack_gateway_approval(self.route, data['request_id'])

    def clarify(self, questions):
        """clarify_tool's batch contract: ``callback(questions) -> {answers, outcome, notice?}``.
        One owner prompt per question, stopping at the first that is interrupted or unanswered;
        an empty answer is a skip (``None``), as on the other surfaces. A finite turn (``chat -q`` /
        ``-z``, bound by worker_turn_scope) answers headless at once: its viewer cannot answer."""
        from gateway.session_finite import finite_clarify_reply, finite_turn_required
        if finite_turn_required() is True:
            return finite_clarify_reply()
        answers = {}
        reply = {'answers': answers, 'outcome': 'submitted'}
        for entry in questions:
            answer = self._ask(entry['question'], entry['choices'], entry['multi_select'])
            if answer is _INTERRUPTED:
                reply['outcome'] = 'cancelled'
                break
            if answer is None:
                reply['outcome'] = 'timed_out'
                break
            answers[entry['qid']] = answer.strip() or None
        return reply

    def _ask(self, question, choices, multi_select):
        import uuid
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()  # the frozen worker config; <= 0 is unlimited, as on every surface
        prompt_id = uuid.uuid4().hex
        state = {'event': threading.Event(), 'answer': None}
        with self.lock:
            if self.stopped.is_set() or len(self.clarifications) >= 16:
                return _INTERRUPTED
            self.clarifications[prompt_id] = state
        try:
            self.channel.send('clarify', prompt_id=prompt_id, question=question,
                              choices=list(choices or []), multi_select=bool(multi_select))
            state['event'].wait(timeout if timeout > 0 else None)
            return state['answer']
        finally:
            with self.lock:
                self.clarifications.pop(prompt_id, None)
            self.channel.send('prompt_settled', prompt_id=prompt_id)


_OUTBOX_NAME = re.compile(r'[A-Za-z0-9_-]{1,200}')


def outbox_name(execution_id):
    """Validate the legacy portable execution-key domain, without interpreting ownership."""
    name = execution_id.replace(':', '-') if isinstance(execution_id, str) else ''
    if not _OUTBOX_NAME.fullmatch(name):
        raise ValueError('invalid_execution_id')
    return name


def outbox_dir(home, execution_id):
    """Portable execution identity, preserving exact matching pre-v2 journals.

    ':' -> '-' was not injective for the supported legacy admission-ID domain.
    Keep ordinary first-ID paths, but encode other strings without collisions or
    path syntax. Never open/reconcile another execution's retained retry journal.
    """
    name = outbox_name(execution_id)
    root = Path(home) / 'worker-outboxes'
    encoded = execution_id.encode('utf-8', 'surrogatepass').hex()
    current = root / 'v2'
    # Reservation maps arbitrary imported IDs to short portable execution keys.
    # Hex is injective over that execution domain, including ':' versus '-'.
    for start in range(0, len(encoded), 120):
        current /= encoded[start:start + 120]
    current /= 'outbox'
    path = root
    if path.is_symlink():
        raise ValueError('unsafe_outbox')
    for part in current.relative_to(root).parts:
        path /= part
        if path.is_symlink():
            raise ValueError('unsafe_outbox')
    if (current / 'pending.json').is_symlink():
        raise ValueError('unsafe_outbox')
    legacy = root / name
    journal = legacy / 'pending.json'
    if legacy.is_symlink() or journal.is_symlink():
        raise ValueError('unsafe_outbox')
    if journal.exists():
        try:
            old_id = json.loads(journal.read_text(encoding='utf-8-sig'))['scope']['execution_id']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid_outbox_identity') from exc
        if old_id == execution_id:
            if (current / 'pending.json').exists():
                raise ValueError('outbox_identity_conflict')
            return legacy
        return current
    # Lowercase portable first-ID spellings are injective even on a case-folding
    # filesystem. All others use hex unless an existing journal proves identity.
    if (current / 'pending.json').exists():
        return current
    if (execution_id.startswith('admission-worker:')
            and execution_id == execution_id.lower()
            and _OUTBOX_NAME.fullmatch(execution_id[len('admission-worker:'):])):
        return legacy
    return current


def discover_profile_mcp(policy):
    """The owner's MCP registry never crosses into this fresh interpreter; connect the profile's
    configured servers here, filtered to the frozen toolset selection, before tool discovery snapshots
    agent.tools. Safe mode keeps its deliberate no-MCP policy (discover_mcp_tools returns [])."""
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools.mcp_tool_discovery import discover_mcp_tools
    with suppress_interactive_oauth():
        discover_mcp_tools(allowed_mcp_names=list(policy.toolsets))


def retire_agent(agent):
    """A settled admission is a turn boundary, not the end of the session: the owner's
    in-process agent keeps its background processes, sandbox and browser between turns
    (release_clients), so the worker must too. Only memory extraction is turn-final work."""
    # A boundary task queued this turn (LLM-bound extraction) would be cancelled by the
    # provider shutdown's short drain; give it the same bounded head start cli_shutdown does.
    manager = getattr(agent, '_memory_manager', None)
    if manager is not None:
        manager.flush_pending(timeout=10)
    messages = getattr(agent, '_session_messages', None)
    agent.shutdown_memory_provider(messages if isinstance(messages, list) else None)
    agent.release_clients()


def tool_frame(call_id, name, args, *result):
    """The callback arguments an in-process turn publishes from (executor-redacted display args,
    the result and its failure verdict); the owner builds the shared tool event from them."""
    frame = {'tool_call_id': str(call_id or ''), 'name': str(name or 'tool'), 'args': args if isinstance(args, dict) else {}}
    if result:
        from agent.display import _detect_tool_failure
        frame['result'] = result[0] if isinstance(result[0], str) else str(result[0])
        frame['is_error'] = bool(_detect_tool_failure(frame['name'], result[0])[0])
    from gateway.session_tool_events import bounded_args, bounded_result
    frame['args'] = bounded_args(frame['args'])
    if result:
        frame['result'] = bounded_result(result[0], frame['is_error'])
    return json.loads(json.dumps(frame, default=str))


def execute(frame, channel, *, control_stream=None):
    # The owner RPC below imports gateway/config modules (hermes_cli.config, providers,
    # hermes_cli.plugins) transitively; the policy must already be frozen when they load.
    bind_worker_policy(frame)
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
    scope = dict(frame['scope'])
    rpc = WorkerRPC(frame['home'])
    adopted = rpc('worker.adopt', **{k: v for k, v in scope.items() if k != 'epoch'})
    if adopted['owner_epoch'] != scope['epoch']:
        raise RuntimeError('stale_epoch')
    store = RuntimeSessionStore(rpc, scope, outbox_dir(frame['home'], scope['execution_id']), adopted)
    from gateway.session_kanban import bind_worker_context
    bind_worker_context(frame)
    # Background processes a previous admission of this session started outlive that worker
    # (F24); adopt them so process_manage in this turn can poll and kill them.
    from tools.process_registry import process_registry
    process_registry.recover_from_checkpoint()
    # Store construction binds the delegation ledger before tool discovery.
    from gateway.session_policy import restore_policy, policy_scope, register_worker_hooks
    policy = restore_policy(frame['policy'])
    register_worker_hooks(policy)
    discover_profile_mcp(policy)
    from run_agent import AIAgent
    from agent.skill_utils import parse_config_string_list
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import set_current_session_key
    set_current_session_key(frame['route'])
    os.environ['HERMES_GATEWAY_SESSION'] = '1'
    from gateway.session_worker_construct import checkpoint_agent_kwargs, construct_kwargs, routing_agent_kwargs
    construct = {'ephemeral_system_prompt': policy.skills_prompt, **construct_kwargs(frame)}
    controls = WorkerControls(channel, frame['route'], control_stream)
    peer = peer_destination(frame)
    if peer is not None:
        controls.accept_peer_messages(peer)
    register_gateway_notify(frame['route'], controls.approval)
    agent = None
    try:
        with policy_scope(policy):
            agent = _construct_agent(frame, AIAgent, policy, model=policy.model, provider=policy.provider, base_url=policy.base_url,
                api_key=frame['api_key'], session_db=store, session_id=scope['session_id'],
                # delegate_task stays offered: each child is reserved by the owner under this
                # execution (agent/runtime_session_child.py) and retires with it.
                enabled_toolsets=list(policy.toolsets), max_iterations=policy.max_turns,
                disabled_toolsets=parse_config_string_list(
                    (policy.config().get('agent') or {}).get('disabled_toolsets')) or None,
                reasoning_config=policy.reasoning_config,
                # ``tool`` / ``oneshot`` are storage labels for a CLI-surface agent (session_policy.SURFACES).
                platform=policy.platform if policy.source in ('tool', 'oneshot') else policy.source,
                gateway_session_key=frame['route'], user_id=frame['user_id'], chat_id=frame['chat_id'],
                skip_context_files=policy.ignore_rules, load_soul_identity=not policy.ignore_rules,
                skip_memory=policy.skip_memory, skip_background_review=True, quiet_mode=True,
                pass_session_id=policy.pass_session_id,
                **checkpoint_agent_kwargs(policy.config()), **construct,
                **routing_agent_kwargs(policy.config(), policy.model, policy.provider, policy.base_url),
                stream_delta_callback=lambda text: channel.send('delta', text=text) if text else None,
                clarify_callback=controls.clarify,
                tool_start_callback=lambda call_id, name, args: channel.send('tool.start', **tool_frame(call_id, name, args)),
                tool_complete_callback=lambda call_id, name, args, result: channel.send('tool.complete',
                                                                                         **tool_frame(call_id, name, args, result)))
            if policy.skip_memory_requested:
                from agent.agent_init_skip_memory import apply_session_skip_memory
                apply_session_skip_memory(agent)
            controls.adopt(agent)
            if controls.stopped.is_set():
                from agent.interrupt_compat import request_hard_interrupt
                request_hard_interrupt(agent)
            channel.send('ready', pid=os.getpid())
            history = store.get_messages_as_conversation(scope['session_id'])
            from gateway.session_kanban import run_worker_turns
            if 'attachments_v1' in frame:
                from gateway.session_ingress_media import restore_attachments
                from agent.image_routing import build_native_content_parts
                media = restore_attachments(frame)
                content, skipped = build_native_content_parts(frame['text'], media['media_urls'])
                if skipped:
                    raise ValueError('managed_attachment_unavailable')
                frame = {**frame, 'text': content}
            before = (getattr(agent, 'session_prompt_tokens', 0) or 0,
                      getattr(agent, 'session_completion_tokens', 0) or 0)

            def finalize(turns):
                """The turns' result as the owner will hear it, after the final usage flush; a card
                attempt books its exit from this (``run_worker_turns``), never from ``turns`` alone."""
                nonlocal agent
                # Most failed results name no provider or model; the owner's error card needs the agent's.
                result = {'provider': getattr(agent, 'provider', None), 'model': getattr(agent, 'model', None), **turns}
                result['_turn_input_tokens'] = max(0, (getattr(agent, 'session_prompt_tokens', 0) or 0) - before[0])
                result['_turn_output_tokens'] = max(0, (getattr(agent, 'session_completion_tokens', 0) or 0) - before[1])
                retire_agent(agent)
                agent = None
                # The auto-title thread bills through this store from a daemon thread; a title landing
                # after execution.finish is a stale_generation write that fails the close-time flush.
                from agent.title_generator import wait_for_title_upgrades
                wait_for_title_upgrades()
                return _flushed(store, result)
            result = run_worker_turns(agent, frame, history, settle=controls.settle, finalize=finalize)
            if result.get('final_response') is None and (result.get('interrupted') or result.get('failed')):
                result['final_response'] = ''
            consumed = controls.consumed(result)
            if consumed:
                channel.send('consumed', ids=consumed)
            if peer is not None:
                for delivery_id, n, outcome in controls.peer_settlements(result):
                    channel.send('peer_settled', id=delivery_id, n=n, outcome=outcome)
            channel.send('result', result=result_frame(result))
            if not controls.finish.wait(30):
                raise ValueError('managed_finish_timeout')
            store.finish()
            channel.send('finished')
    finally:
        unregister_gateway_notify(frame['route'])
        if agent is not None:
            retire_agent(agent)
        try:
            store.close()
        finally:
            with contextlib.suppress(Exception):  # a failing socket close never replaces the turn's outcome
                rpc.close()


def _flushed(store, result):
    """``result`` after the execution's final usage flush. A failed flush is this turn's
    persistence failure, reported as the failed result it is (the owner settles it failed, never
    completed) instead of a lost worker; the owner still decides the execution's final state from
    ``finished``, which a store that cannot persist never sends."""
    try:
        store.flush_token_counts()
    except Exception as exc:
        logger.exception('managed worker usage flush failed')
        from hermes_state_errors import classify_persistence_error
        return {**result, 'completed': False, 'failed': True,
                'error': 'session storage could not be written: ' + (str(exc) or type(exc).__name__),
                'turn_exit_reason': 'session_persistence_failed',
                'failure_reason': 'session_persistence_failed:' + classify_persistence_error(exc)}
    return result


def _construct_agent(frame, factory, policy, **kwargs):
    """Agent construction resolves the provider credentials; a Kanban attempt that fails here
    still books the one-shot exit mapping (quota wall 75, re-login 78) before the crash."""
    try:
        from gateway.session_worker_construct import worker_route
        return factory(**{**kwargs, **worker_route(policy, kwargs)})
    except Exception as exc:
        from gateway.session_kanban import record_start_failure
        record_start_failure(frame, exc)
        raise


def hello():
    """Identity the owner verifies before it reserves: this interpreter's pid and birth plus
    the ancestor chain it observes. Launchers (uv's venv python.exe) put the owner's Popen
    handle one or two hops above; the owner, never this process, decides whether they match."""
    import psutil
    from hermes_cli.process_identity import worker_birth
    me = psutil.Process()
    return {'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]}


def main():
    channel = WorkerChannel(os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0))  # windows-footgun: ok — binary frames
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # This process's own stderr only, never the channel. The owner drains it into the profile's
    # private managed-worker.log, redacting each line (patterns plus this worker's launch key and
    # assignment secret) before it reaches disk; this process never opens a profile log itself.
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.propagate = False
    try:
        channel.send('hello', **hello())
        # Bootstrap and controls share a private buffer, never sys.stdin's
        # interpreter-finalized buffer. The control reader owns it once started.
        control_stream = _control_input_stream()
        frame = validate_bootstrap(read_frame(control_stream))
        execute(frame, channel, control_stream=control_stream)
    except Exception:
        # Runtime exceptions can contain credentials; the owner gets no raw traceback.
        logger.exception('managed worker failed')
        channel.send('error', reason='managed_worker_failed')
        return 1
    finally:
        channel.stream.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
