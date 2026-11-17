"""The managed-worker correction transport at its own entry points, plus a redirect that meets a
running tool on the real composed path (owner JSON-RPC -> worker control pipe -> child ``AIAgent``).

- The child answers each correction with what happened to it; ``redirected`` only for a cancelled
  model request, never for a tool-boundary steer.
- Late steer text runs as a follow-up of the same admission; intake closes atomically.
- The owner keeps every correction durably before any ACK (a queued admission, or a task comment for
  a card attempt); a keep that fails is an error, never an ACK, and holds no later correction.
- The child confirms consumption (``consumed``) only after a completed, flushed turn that ran all of
  it; a failed usage flush is a failed result, never a lost worker and never a confirmation.
- The owner never leaves a waiter unanswered and never holds a later correction behind an earlier one
  past ``CORRECTION_ACK_SECONDS``.
- A redirect during a foreground command yields the live process (same PID), whose natural result
  stays consumable by the same turn.
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_managed_worker_corrections import (
    MANAGED, _generation, _kept, _managed_exit, _managed_session, _turns, _user_text,
)
from tests.gateway.test_session_busy_controls import admissions, owner, settled
from tests.waiting import async_wait_until


class YieldModel(BaseHTTPRequestHandler):
    """Turn: a foreground terminal command (it records its PID, then runs ~3 s to a natural exit 7
    that writes an artifact); once the correction reached the model, ``process_manage wait`` on the
    yielded handle; then a final answer."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        import re
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        seen = json.dumps(messages)
        name = args = None
        if messages and messages[-1]['role'] == 'user' and _user_text(messages[-1]) == 'YIELD_START':
            name, args = 'terminal', {'command': f'echo $$ > {self.server.pidfile}; echo STARTED; sleep 3; '
                                                 f'echo NATURAL > {self.server.artifact}; echo BG_DONE; exit 7',
                                      'timeout': 90}
        elif 'CHECK_BG' in seen and '"exit_code": 7' not in seen.replace('\\"', '"'):
            proc = re.findall(r'proc_[0-9a-f]{12}', seen)
            if proc:
                name, args = 'tool_call', {'calls': [{'name': 'process_manage', 'arguments': {
                    'action': 'wait', 'session_id': proc[0], 'timeout': 60}}]}
        message = ({'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'c{len(messages)}', 'index': 0,
                    'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
                   if name else {'role': 'assistant', 'content': 'CONTROL_ACK'})
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if name else 'stop'}
        frame = {'id': 'yield', 'model': 'm', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_redirect_during_a_tool_yields_the_live_process_and_its_natural_result_stays_consumable(tmp_path):
    """A redirect that meets a running foreground command is NOT a model abort: the ACK says
    ``queued``, the command is handed to the background registry alive (same PID), the correction
    reaches the model at the tool boundary, and the same turn later waits on that SAME handle for
    its natural completion: real output, true exit code 7 and the artifact it wrote."""
    config = {**MANAGED, 'approvals': {'mode': 'off'}}
    with owner(tmp_path, config, handler=YieldModel) as (home, peer, desc):
        peer.pidfile, peer.artifact = tmp_path / 'command.pid', tmp_path / 'artifact.txt'
        (tmp_path / 'user' / '.hushlogin').touch()  # keep login-shell banners out of the command output

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='YIELD_START')
                await async_wait_until(lambda: peer.pidfile.exists() and peer.pidfile.read_text().strip(),
                                       timeout=60, interval=.05, what='the foreground command to start')
                corrected = await rpc(ws, 'session.redirect', session_id=sid,
                                      execution_generation=await _generation(ws, sid), text='CHECK_BG')
                assert corrected.get('result', {}).get('status') == 'queued', corrected  # never 'redirected'
                await settled(home)
                return (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
        messages = asyncio.run(probe())
        tools = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        yielded, waited = tools[0], tools[-1]
        assert yielded['status'] == 'yielded_to_background' and yielded['exit_code'] is None, yielded
        assert int(peer.pidfile.read_text()) == yielded['pid'], (peer.pidfile.read_text(), yielded)
        assert waited['status'] == 'exited' and waited['exit_code'] == 7, waited
        assert 'BG_DONE' in waited['output'], waited
        assert peer.artifact.read_text() == 'NATURAL\n'
        assert any('CHECK_BG' in json.dumps(r['messages']) for r in peer.requests)
        assert _turns(home) == [('turn', 'terminal')], _kept(home)
        # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
        assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
        assert _managed_exit(home) == ['terminal']


# ── child and owner halves of the transport, at their own entry points ──────────────────────────

class _Agent:
    """Records what reached it. ``redirect_outcome`` follows ``AIAgent``'s paths: ``model`` while a model
    request is in flight, ``tools`` (a steer plus a yield request) while a tool batch runs."""

    def __init__(self, in_flight=True, fail=False):
        self.in_flight, self.fail, self.executing_tools = in_flight, fail, False
        self.steered, self.redirected, self.parts = [], [], []

    def redirect_outcome(self, text):
        if self.fail:
            raise RuntimeError('abort failed')
        if self.executing_tools:
            self.steer(text)
            return 'tools'
        if self.in_flight:
            self.redirected.append(text)
            return 'model'
        return None

    def steer(self, text):
        self.steered.append(text)
        self.parts.append(text)
        return True

    def _drain_leftover_steer(self):
        text, self.parts = '\n'.join(self.parts) or None, []
        return text


def _controls():
    from agent import managed_worker as worker
    sent = []
    controls = worker.WorkerControls.__new__(worker.WorkerControls)  # no stdin reader thread
    controls.channel = SimpleNamespace(send=lambda kind, **payload: sent.append({'type': kind, **payload}))
    controls.agent, controls.stopped, controls.lock, controls.clarifications = None, threading.Event(), threading.Lock(), {}
    controls.corrections_open, controls.latched, controls.accepted, controls.unconsumed = True, [], [], False
    return controls, sent


def _correct(controls, verb, text, ident):
    controls.correct({'type': 'correct', 'id': ident, 'verb': verb, 'text': text})


def test_child_answers_each_correction_with_what_actually_happened_to_it():
    controls, sent = _controls()
    _correct(controls, 'redirect', 'EARLY', 'e')  # the agent is still being constructed
    agent = _Agent()
    controls.adopt(agent)
    assert agent.steered == ['EARLY'] and controls.latched == []
    _correct(controls, 'redirect', 'LIVE', 'r')
    agent.executing_tools = True  # a tool batch: steered at its boundary, never reported as an abort
    _correct(controls, 'redirect', 'AT_TOOLS', 't')
    agent.executing_tools = agent.in_flight = False  # no model request: the redirect degrades to a steer
    _correct(controls, 'redirect', 'BETWEEN', 'b')
    _correct(controls, 'steer', 'STEER', 's')
    assert agent.redirected == ['LIVE'] and agent.steered == ['EARLY', 'AT_TOOLS', 'BETWEEN', 'STEER']
    assert [(f['id'], f['status']) for f in sent] == [('e', 'queued'), ('r', 'redirected'), ('t', 'queued'),
                                                     ('b', 'queued'), ('s', 'queued')]
    with pytest.raises(ValueError, match='invalid_managed_control'):
        controls.correct({'type': 'correct', 'id': 'x', 'verb': 'stop', 'text': 'nope'})


def test_settle_hands_back_undrained_text_then_closes_intake_atomically():
    controls, sent = _controls()
    agent = _Agent()
    controls.adopt(agent)
    _correct(controls, 'steer', 'AFTER_FINAL', 'a')
    assert controls.settle({'pending_steer': 'FROM_RESULT'}) == 'FROM_RESULT\nAFTER_FINAL'
    _correct(controls, 'steer', 'DURING_FOLLOW_UP', 'f')  # intake is still open for the follow-up turn
    assert controls.settle({}) == 'DURING_FOLLOW_UP'
    assert controls.settle({}) is None  # nothing left: intake closes in the same step
    _correct(controls, 'redirect', 'TOO_LATE', 'late')
    assert sent[-1] == {'type': 'correction', 'id': 'late', 'status': 'rejected'}
    assert 'TOO_LATE' not in agent.steered
    # Everything accepted ran: a completed turn confirms exactly those, never the rejected one.
    assert controls.consumed({'completed': True}) == ['a', 'f']
    for unfinished in ({'completed': False}, {'completed': True, 'failed': True}, {'completed': True, 'interrupted': True}):
        assert controls.consumed(unfinished) == []  # the owner's kept copies run again instead


def test_stopped_or_failing_child_never_claims_delivery():
    controls, sent = _controls()
    controls.adopt(_Agent(fail=True))
    _correct(controls, 'redirect', 'BROKEN', 'u')
    assert sent[-1]['status'] == 'unknown'  # the owner answers correction_outcome_unknown: resend
    controls.stopped.set()
    _correct(controls, 'steer', 'AFTER_STOP', 'x')
    assert sent[-1]['status'] == 'rejected'
    controls.agent.parts.append('UNDRAINED')
    assert controls.settle({}) is None  # a stopped turn has no follow-up
    assert controls.consumed({'completed': True}) == []  # nor confirms anything: the owner still keeps it


@pytest.mark.parametrize('flush_fails', [False, True], ids=['flushed', 'flush-failed'])
def test_leftover_steer_runs_as_follow_up_turn_inside_the_managed_execution(tmp_path, monkeypatch, flush_fails):
    """Through ``execute``: the first turn ends with ``pending_steer``; the same agent runs it as a
    follow-up turn before the single result frame, instead of the result frame dropping it. The
    correction accepted during the turn is confirmed ``consumed`` before that result, unless the
    final usage flush fails: then the result is the failed persistence it is (never a completed
    turn, never a lost worker) and nothing is confirmed, so the owner's kept copy runs again."""
    import os
    from agent import managed_worker as worker
    from tests.gateway.test_managed_turn_contract import assignment
    frame, _ = assignment(tmp_path, monkeypatch, finite=False)
    turns, frames = [], []

    class Agent:
        def __init__(self, **kwargs):
            self.session_prompt_tokens = self.session_completion_tokens = 0
            self.session_id = 'sid'
            self._session_db = Store()

        def run_conversation(self, text, **kwargs):
            turns.append(text)
            if len(turns) == 1:
                return {'final_response': 'FIRST_FINAL', 'pending_steer': 'LATE_STEER', 'completed': True}
            return {'final_response': 'FOLLOW_UP_FINAL', 'completed': True}

        def steer(self, text):
            return True

        def _drain_leftover_steer(self):
            return None

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self):
            if flush_fails:
                raise TimeoutError('worker_receipt_timeout')
        def finish(self): pass
        def close(self): pass

    read, write = os.pipe()
    os.write(write, b'{"type":"correct","id":"c1","verb":"steer","text":"OWNER_STEER"}\n{"type":"finish"}\n')
    controls = worker.WorkerControls
    monkeypatch.setattr(worker, 'WorkerControls', lambda channel, route, stream=None: controls(
        channel, route, os.fdopen(read, 'rb')))
    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    try:
        worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: frames.append({'type': kind, **payload})))
    finally:
        os.close(write)
    assert turns == ['hello', 'LATE_STEER']
    results = [f for f in frames if f['type'] == 'result']
    assert len(results) == 1 and results[0]['result']['final_response'] == 'FOLLOW_UP_FINAL'
    assert {'type': 'correction', 'id': 'c1', 'status': 'queued'} in frames
    kinds = [f['type'] for f in frames if f['type'] in ('consumed', 'result', 'finished')]
    if flush_fails:
        failed = results[0]['result']
        assert failed['failed'] is True and failed['completed'] is False, failed
        assert failed['turn_exit_reason'] == 'session_persistence_failed', failed
        assert failed['failure_reason'].startswith('session_persistence_failed:'), failed
        assert kinds == ['result', 'finished']  # reported, not lost; and no consumption confirmed
    else:
        assert kinds == ['consumed', 'result', 'finished']
        assert next(f for f in frames if f['type'] == 'consumed')['ids'] == ['c1']


ACTOR = SimpleNamespace(subject='uid:test')
REF = SimpleNamespace(session_id='sid')


def _authority(worker, fail_first=0):
    """The owner surface ``managed_correction`` uses: ``submit`` is ``prompt.submit``'s admission (a
    kept row per correction here, recorded), failing its first ``fail_first`` calls."""
    from gateway.session_contract import Submission
    kept, failures = [], [fail_first]

    async def submit(actor, request):
        assert isinstance(request, Submission) and request.intent == 'queue' and set(request.payload) == {'text'}
        if failures[0]:
            failures[0] -= 1
            raise OSError('database is locked')
        kept.append((actor.subject, request.request_id, request.payload['text']))
        return SimpleNamespace(admission_id='adm-' + request.request_id)
    if worker.policy is None:  # ``execute_managed`` hands the worker its launch policy: not a card attempt
        worker.policy = SimpleNamespace(kanban_json=None)
    live = SimpleNamespace(event_stream=SimpleNamespace(lock=threading.Lock()))
    return SimpleNamespace(sessions={'sid': live}, _managed_workers={'sid': worker}, executing={'sid': 7},
                           epoch=1, submit=submit, kept=kept)


async def _send(authority, worker, verb, text):
    from gateway.session_managed_worker import managed_correction
    return asyncio.ensure_future(managed_correction(authority, ACTOR, REF, worker, verb, text, 7))


@pytest.mark.asyncio
async def test_owner_keeps_each_correction_before_any_ack_and_never_strands_a_waiter():
    from gateway.session_managed_worker import ManagedWorker
    from hermes_state_runtime import RuntimeStoreError
    worker = ManagedWorker(SimpleNamespace(pid=0))
    authority = _authority(worker)
    # Before the child is ready it can only latch the text: answered at once, after the durable keep.
    early = await (await _send(authority, worker, 'redirect', 'EARLY'))
    assert early['status'] == 'queued' and early['admission_id'] == 'adm-' + authority.kept[0][1], early
    assert worker.commands.get_nowait()['text'] == 'EARLY'  # still handed to the child
    worker.corrections.clear()
    worker.ready = True
    first, second, third = [await _send(authority, worker, 'redirect', t) for t in ('A', 'B', 'A')]
    await async_wait_until(lambda: worker.commands.qsize() == 3, what='three corrections sent')
    written = [worker.commands.get_nowait() for _ in range(3)]
    assert [(f['type'], f['verb'], f['text']) for f in written] == [('correct', 'redirect', t) for t in 'ABA']
    assert len({f['id'] for f in written}) == 3  # a resend is a new correction, never deduplicated away
    # Every one is durable before its ACK, the resend included: duplicates are accepted.
    assert [text for _, _, text in authority.kept] == ['EARLY', 'A', 'B', 'A']
    assert all(subject == 'uid:test' and rid.startswith('correction:') for subject, rid, _ in authority.kept)
    # A's verdict is lost; B's arrives on its own and is not held behind A.
    assert worker.answer_correction({'type': 'correction', 'id': written[1]['id'], 'status': 'redirected'})
    assert (await second)['status'] == 'redirected'
    # The resent A is refused by the child: its kept row still runs, so the ACK says queued, with it.
    assert worker.answer_correction({'type': 'correction', 'id': written[2]['id'], 'status': 'rejected'})
    assert await third == {'status': 'queued', 'admission_id': worker.kept[written[2]['id']].admission_id,
                           'text': 'A', 'execution_generation': 7, 'authority_epoch': 1}
    worker.abandon_corrections(unread=False)  # the child was lost with A unanswered
    with pytest.raises(RuntimeStoreError, match='correction_outcome_unknown'):
        await first  # unknown here, yet A's kept row delivers it anyway
    assert not worker.corrections


@pytest.mark.asyncio
async def test_a_correction_that_cannot_be_kept_is_an_error_and_holds_no_later_one():
    """A failed durable keep is never acknowledged (nothing reaches the child). The next correction
    proceeds on its own, and the resend of the failed one is kept and delivered."""
    from gateway.session_managed_worker import ManagedWorker
    from hermes_state_runtime import RuntimeStoreError
    worker = ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    authority = _authority(worker, fail_first=1)
    with pytest.raises(RuntimeStoreError, match='correction_not_kept'):
        await (await _send(authority, worker, 'steer', 'A'))
    assert worker.commands.empty() and not worker.corrections and authority.kept == []
    later, resent = await _send(authority, worker, 'steer', 'B'), await _send(authority, worker, 'steer', 'A')
    await async_wait_until(lambda: worker.commands.qsize() == 2, what='B and the resent A sent')
    for frame in (worker.commands.get_nowait(), worker.commands.get_nowait()):
        assert worker.answer_correction({'type': 'correction', 'id': frame['id'], 'status': 'queued'})
    assert (await later)['status'] == (await resent)['status'] == 'queued'
    assert [text for _, _, text in authority.kept] == ['B', 'A']


@pytest.mark.asyncio
async def test_a_card_attempt_keeps_corrections_on_its_task_never_as_a_queued_turn(monkeypatch):
    from gateway import session_kanban
    from gateway.session_managed_worker import ManagedWorker
    worker = ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    worker.policy = SimpleNamespace(kanban_json=json.dumps({'db': 'board.db', 'task_id': 't1', 'run_id': 5}))
    comments = []
    monkeypatch.setattr(session_kanban, 'keep_on_board', lambda context, text: comments.append((context['task_id'], text)))
    authority = _authority(worker)
    pending = await _send(authority, worker, 'redirect', 'CARD_FIX')
    await async_wait_until(lambda: not worker.commands.empty(), what='sent')
    assert comments == [('t1', 'CARD_FIX')] and authority.kept == []  # kept on the task before it is sent
    assert worker.answer_correction({'type': 'correction', 'id': worker.commands.get_nowait()['id'], 'status': 'rejected'})
    assert await pending == {'status': 'rejected', 'delivered': False, 'kept': 'task_comment', 'text': 'CARD_FIX',
                             'execution_generation': 7, 'authority_epoch': 1}


@pytest.mark.asyncio
async def test_a_turn_that_ended_before_the_send_still_delivers_the_kept_text():
    from gateway.session_managed_worker import ManagedWorker
    worker = ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    authority = _authority(worker)
    authority.executing['sid'] = None  # the claim settled while the keep was being written
    reply = await (await _send(authority, worker, 'steer', 'LATE'))
    assert reply['status'] == 'queued' and reply['admission_id'] == 'adm-' + authority.kept[0][1]
    assert worker.commands.empty()  # nothing sent to a turn that is over; the kept row runs as its own turn


@pytest.mark.asyncio
async def test_unanswered_correction_is_unknown_within_the_bound_and_the_next_one_proceeds(monkeypatch):
    """Controls on one socket dispatch in order: an earlier correction whose verdict never comes holds
    the next one at most ``CORRECTION_ACK_SECONDS``, then is reported unknown (resend it; its kept
    copy is delivered anyway)."""
    from gateway import session_managed_worker as managed
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setattr(managed, 'CORRECTION_ACK_SECONDS', .2)
    worker = managed.ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    authority = _authority(worker)
    with pytest.raises(RuntimeStoreError, match='correction_outcome_unknown'):
        await (await _send(authority, worker, 'redirect', 'A'))
    later = await _send(authority, worker, 'redirect', 'B')
    await async_wait_until(lambda: worker.commands.qsize() == 2, what='B sent')
    frames = [worker.commands.get_nowait() for _ in range(2)]
    assert worker.answer_correction({'type': 'correction', 'id': frames[1]['id'], 'status': 'redirected'})
    assert (await later)['status'] == 'redirected'
    # A's verdict arriving late is consumed, never an invalid frame.
    assert worker.answer_correction({'type': 'correction', 'id': frames[0]['id'], 'status': 'redirected'})
    assert [text for _, _, text in authority.kept] == ['A', 'B']


def test_kanban_follow_up_runs_only_under_a_live_claim_and_confirms_nothing_it_did_not_run():
    """``_follow_up`` as the Kanban branch calls it: a follow-up turn runs while the attempt's claim
    is live. Once it is not, no model turn runs and intake closes (a later correction is
    ``rejected``), and the execution confirms none of its corrections: the owner kept each one on
    the task before sending it, for the next attempt."""
    from gateway.session_kanban import _follow_up
    controls, _sent = _controls()
    agent = _Agent()
    agent.session_id, turns = 'sid', []
    agent._session_db = SimpleNamespace(get_messages_as_conversation=lambda sid: [])
    agent.run_conversation = lambda text, **kwargs: turns.append(text) or {'final_response': 'FOLLOW_UP', 'completed': True}
    controls.adopt(agent)
    live = [True]
    _correct(controls, 'steer', 'WHILE_LIVE', 'w')
    result = _follow_up(agent, {'final_response': 'FIRST', 'completed': True}, controls.settle, may_continue=lambda: live[0])
    assert result['final_response'] == 'FOLLOW_UP' and turns == ['WHILE_LIVE']
    assert controls.consumed(result) == ['w']
    controls2, sent2 = _controls()
    controls2.adopt(agent)
    _correct(controls2, 'steer', 'CLAIM_GONE', 'g')
    live[0] = False
    result = _follow_up(agent, {'final_response': 'LAST', 'completed': True}, controls2.settle,
                        may_continue=lambda: live[0])
    assert result['final_response'] == 'LAST' and turns == ['WHILE_LIVE']
    assert controls2.consumed(result) == []
    _correct(controls2, 'steer', 'AFTER_CLAIM', 'late')
    assert sent2[-1] == {'type': 'correction', 'id': 'late', 'status': 'rejected'}


def test_a_follow_up_marks_the_admission_intervened_even_when_its_own_turn_ran_untouched():
    """The follow-up turn takes the steered text as its own user message, so the agent reports that
    turn untouched (``user_intervened`` False, its own ``no_model_output``). The admission's result
    must still say other input shaped it, or a reply-bearing ticket's settlement would take the
    follow-up's final as the original prompt's answer. An admission no follow-up ran keeps the
    turn's own explicit verdict."""
    from gateway.session_kanban import _follow_up
    controls, _sent = _controls()
    agent = _Agent()
    agent.session_id = 'sid'
    agent._session_db = SimpleNamespace(get_messages_as_conversation=lambda sid: [])
    agent.run_conversation = lambda text, **kwargs: {
        'final_response': '', 'user_intervened': False, 'no_model_output': True}
    controls.adopt(agent)
    result = _follow_up(agent, {'final_response': 'FIRST', 'user_intervened': False, 'no_model_output': False,
                                'pending_steer': 'LATE'}, controls.settle)
    assert result == {'final_response': '', 'user_intervened': True, 'no_model_output': True}
    untouched, _ = _controls()
    untouched.adopt(agent)
    assert _follow_up(agent, {'final_response': 'ONLY', 'user_intervened': False}, untouched.settle) == {
        'final_response': 'ONLY', 'user_intervened': False}


def test_a_card_correction_is_kept_as_a_comment_the_next_attempt_reads(tmp_path, monkeypatch):
    """``keep_on_board`` on a real disposable board: the whole correction lands on the task's thread,
    captioned as accepted (not as applied or unapplied), which ``build_worker_context`` puts in front
    of the task's next attempt whole: longer than an ordinary comment's cap, and behind more than the
    ordinary comments shown. No claim is touched."""
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_kanban import keep_on_board
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_PROFILE', 'worker-bot')
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned', assignee='default')
        claimed = kb.claim_task(conn, tid)
    text = 'LATE_CORRECTION_HEAD ' + 'x' * 3000 + ' LATE_CORRECTION_TAIL'
    keep_on_board({'db': str(kb.kanban_db_path(board='owned')), 'task_id': tid, 'run_id': claimed.current_run_id},
                  text)
    with closing(connect(board='owned')) as conn:
        for i in range(31):
            kb.add_comment(conn, tid, author='operator', body=f'ORDINARY_{i:02}')
        task = kb.get_task(conn, tid)
        assert (task.current_run_id, task.claim_lock) == (claimed.current_run_id, claimed.claim_lock)
        context = kb.build_worker_context(conn, tid)
        assert text in context, context[-500:]
        # Ordinary comments keep their caps: the oldest is omitted, and said to be.
        assert 'ORDINARY_00' not in context and 'ORDINARY_30' in context and '1 earlier comment omitted' in context
        assert f'Correction accepted for run {claimed.current_run_id} while it was running' in context, context
        assert 'whether that run applied it is not recorded here' in context, context
