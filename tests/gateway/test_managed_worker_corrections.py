"""A running managed worker takes ``session.steer`` / ``session.redirect`` over its private control
channel, on the real composed path: owner JSON-RPC -> ``correct`` -> worker control pipe -> the
child's ``AIAgent`` -> the same admission's provider loop.

- A redirect lands while the turn's provider request is still running. The stale response is
  fenced: its tool call never runs and its text is never published, even when it is answered the
  moment the ``redirected`` ACK arrives.
- A steer that lands after the final model call is not dropped with the result frame. It runs as a
  follow-up turn of the same admission.
- A correction whose reply was lost never strands a later one, on another socket or the same one.
  Resending it is delivery, and the duplicate is accepted (at-least-once).
- Generation, profile and parameter fences hold on the managed target.
"""
import asyncio
import json
import sqlite3
from http.server import BaseHTTPRequestHandler

import pytest

from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import admissions, owner, settled
from tests.waiting import async_wait_until


def _user_text(message):
    content = message.get('content')
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    return content or ''


class ManagedCorrectionModel(BaseHTTPRequestHandler):
    """``BLOCK_*`` / ``HOLD_*`` requests park until ``release``. Released, ``BLOCK_STALE`` answers with
    a terminal tool call that creates the marker file, and ``BLOCK_FINAL`` with a final text. Every
    other request answers ``CONTROL_ACK``. A fenced (aborted) request still gets its answer written,
    into a socket the worker has shut down."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        text = next((_user_text(m) for m in reversed(messages) if m['role'] == 'user'), '')
        message, finish = {'role': 'assistant', 'content': 'CONTROL_ACK'}, 'stop'
        parked = 'BLOCK_' in text or 'HOLD_' in text
        if parked:
            self.server.blocked.append(text)
            self.server.release.wait(60)
            if text.endswith('BLOCK_STALE'):
                message = {'role': 'assistant', 'content': 'STALE_TEXT', 'tool_calls': [{
                    'id': 'stale-tool', 'type': 'function', 'function': {
                        'name': 'terminal',
                        'arguments': json.dumps({'command': f'touch {self.server.marker}', 'timeout': 5})}}]}
                finish = 'tool_calls'
            elif text.endswith('BLOCK_FINAL'):
                message = {'role': 'assistant', 'content': 'FIRST_FINAL'}
        payload = json.dumps({'id': 'managed', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            payload = ('data: ' + json.dumps({'id': 'managed', 'choices': [{'index': 0, 'delta': delta,
                       'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        if parked:
            self.server.answered.append(text)


MANAGED = {'gateway': {'managed_workers': True, 'multiplex_profiles': False},
           'platform_toolsets': {'cli': ['terminal']}}


def _peer(peer, marker):
    peer.blocked, peer.answered, peer.marker = [], [], marker
    return peer


async def _stale_answered(peer, home):
    """Every parked request has been answered and the admission settled: a falsely fenced stale
    response would have been consumed, and its tool call run, inside the turn before it settled."""
    await async_wait_until(lambda: len(peer.answered) >= len(peer.blocked), timeout=40, interval=.05,
                           what='every parked request answered')
    await settled(home)


async def _managed_session(ws, home, peer):
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    created = await rpc(ws, 'session.create', request_id='managed', source='cli', cwd=str(home),
                        model='controls-fixture', provider='custom', base_url=url, api_key='loopback-only',
                        toolsets=['terminal'], ignore_rules=True)
    assert 'result' in created, created
    return created['result']['session_id']


async def _blocked(peer, count):
    await async_wait_until(lambda: len(peer.blocked) >= count, timeout=40, interval=.05,
                           what=f'{count} blocked provider request(s)')


async def _generation(ws, sid):
    return (await rpc(ws, 'session.resume', session_id=sid))['result']['execution_generation']


def _transcript(home, sid):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return [(role, content or '') for role, content in
                db.execute('SELECT role, content FROM messages WHERE session_id=? ORDER BY id', (sid,))]


def _turns(home):
    """The session's own admissions, without the corrections the owner kept as queued rows."""
    return [r[:2] for r in admissions(home) if not r[0].startswith('correction:')]


def _kept(home):
    """Each kept correction's ``(status, outcome, text)``, in FIFO order: ``steered`` once the turn
    that ran it confirmed it; ``completed`` when it ran as its own turn instead."""
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return [(status, outcome, json.loads(payload)['text']) for status, outcome, payload in db.execute(
            "SELECT status,outcome,payload_json FROM session_admissions WHERE request_id LIKE 'correction:%' ORDER BY seq")]


def _managed_exit(home):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return [row[0] for row in db.execute('SELECT status FROM worker_executions')]


@pytest.mark.platforms("linux")
def test_redirect_reaches_the_running_managed_call_and_fences_its_stale_response(tmp_path):
    marker = tmp_path / 'stale-tool-ran'
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, marker)

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_STALE')
                assert 'result' in submitted, submitted
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                corrected = await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation,
                                      text='CORRECT_REDIRECT')
                assert corrected.get('result', {}).get('status') == 'redirected', corrected
                assert corrected['result']['execution_generation'] == generation
                # Consumed while the original provider call is still parked in the peer: the SAME
                # admission's loop rebuilt its request around the correction.
                await async_wait_until(lambda: any('CORRECT_REDIRECT' in json.dumps(r['messages'])
                                                   for r in peer.requests), timeout=40, interval=.05,
                                       what='the corrected request')
                assert not peer.release.is_set()
                peer.release.set()  # the stale response now arrives in a socket the worker shut down
                await _stale_answered(peer, home)
            assert not marker.exists(), 'a fenced response dispatched its tool call'
            assert _turns(home) == [('turn', 'terminal')], _kept(home)
            # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
            assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
            rows = _transcript(home, sid)
            assert not any('STALE_TEXT' in content for _, content in rows), rows
            assert any(role == 'user' and 'CORRECT_REDIRECT' in content for role, content in rows), rows
            assert rows[-1] == ('assistant', 'CONTROL_ACK'), rows
            assert _managed_exit(home) == ['terminal']
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_lost_correction_reply_never_strands_a_later_correction(tmp_path):
    """A's reply is lost (its connection closes unread), so its sender resends it after B. B is
    delivered whatever became of A, and A's duplicate is accepted rather than refused or deduplicated."""
    marker = tmp_path / 'stale-tool-ran'
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, marker)

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_STALE')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                async with websocket(home, desc) as lost:
                    await lost.send(json.dumps({'jsonrpc': '2.0', 'id': 'lost', 'method': 'session.redirect',
                        'params': {'session_id': sid, 'execution_generation': generation, 'text': 'HOLD_A'}}))
                # Nothing about A is known to its sender; B goes now.
                b = await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation, text='HOLD_B')
                assert b.get('result', {}).get('status') in {'redirected', 'queued'}, b
                again = await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation,
                                  text='HOLD_A')
                assert again.get('result', {}).get('status') in {'redirected', 'queued'}, again
                if b['result']['status'] == 'redirected':  # B cancelled the parked call: it is in the next one
                    await async_wait_until(lambda: any('HOLD_B' in json.dumps(r['messages']) for r in peer.requests),
                                           timeout=40, interval=.05, what=f'B in a provider request ({b}, {again})')
                # Otherwise B landed between model calls (A's cancelled request had ended, the next one
                # had not begun) and was taken for the next boundary: the parked call must return first.
                peer.release.set()
                await _stale_answered(peer, home)
            assert not marker.exists(), 'a fenced response dispatched its tool call'
            assert _turns(home) == [('turn', 'terminal')], _kept(home)
            # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
            assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
            delivered = ' '.join(content for role, content in _transcript(home, sid) if role == 'user')
            assert 'HOLD_B' in delivered
            assert delivered.count('HOLD_A') >= 1  # once or twice: a duplicate is delivery, not failure
            final = peer.requests[-1]['messages']
            assert 'HOLD_A' in json.dumps(final) and 'HOLD_B' in json.dumps(final)
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_steer_after_the_final_model_call_runs_as_a_follow_up_of_the_same_admission(tmp_path):
    """The steer lands while the turn's last provider call (a final answer, no tool batch) is running,
    so no boundary is left to drain it. In process it becomes the next user turn; a managed worker
    must not drop it with its result frame."""
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                steered = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation,
                                    text='LATE_STEER')
                assert steered.get('result', {}).get('status') == 'queued', steered
                peer.release.set()
                await settled(home)
            assert _turns(home) == [('turn', 'terminal')], _kept(home)
            # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
            assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
            assert any('LATE_STEER' in json.dumps(r['messages']) for r in peer.requests), peer.requests
            rows = _transcript(home, sid)
            assert ('assistant', 'FIRST_FINAL') in rows and rows[-1] == ('assistant', 'CONTROL_ACK'), rows
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_corrections_to_a_settled_managed_turn_are_generation_fenced(tmp_path):
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                for extra, reason in [({'execution_generation': generation - 1}, 'stale_generation'),
                                      ({'profile': 'foreign'}, 'profile_mismatch'),
                                      ({'execution_generation': True}, 'invalid_params'),
                                      ({'text': '   '}, 'invalid_params'),
                                      ({'request_id': 'smuggled'}, 'invalid_params')]:
                    for verb in ('steer', 'redirect'):
                        params = {'session_id': sid, 'text': 'DENIED_MARKER', 'execution_generation': generation,
                                  **extra}
                        denied = await rpc(ws, 'session.' + verb, **params)
                        assert denied.get('error', {}).get('message') == reason, (verb, extra, denied)
                other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=other-controls-actor'
                async with connect(other_url) as other:
                    # Operator sharing does not bypass the execution-generation fence.
                    denied = await rpc(other, 'session.redirect', session_id=sid,
                                       execution_generation=generation - 1, text='DENIED_MARKER')
                    assert denied.get('error', {}).get('message') == 'stale_generation', denied
                peer.release.set()
                await settled(home)
                late = await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation,
                                 text='TOO_LATE')
                assert late.get('error', {}).get('message') == 'stale_generation', late
            assert not any('DENIED_MARKER' in json.dumps(r) or 'TOO_LATE' in json.dumps(r) for r in peer.requests)
        asyncio.run(probe())



@pytest.mark.platforms("linux")
def test_stale_response_released_right_after_the_redirect_ack_is_never_used(tmp_path):
    """False-fence catcher: the parked request is answered (with a tool call) the moment the
    ``redirected`` ACK arrives, so a child that claimed the fence without cancelling the request
    would consume that stale answer and run its tool."""
    marker = tmp_path / 'stale-tool-ran'
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, marker)

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_STALE')
                await _blocked(peer, 1)
                corrected = await rpc(ws, 'session.redirect', session_id=sid,
                                      execution_generation=await _generation(ws, sid), text='CORRECT_FAST')
                peer.release.set()
                assert corrected.get('result', {}).get('status') == 'redirected', corrected
                await _stale_answered(peer, home)
            assert not marker.exists(), 'a fenced response dispatched its tool call'
            rows = _transcript(home, sid)
            assert not any('STALE_TEXT' in content for _, content in rows), rows
            assert any(role == 'user' and 'CORRECT_FAST' in content for role, content in rows), rows
            assert _turns(home) == [('turn', 'terminal')], _kept(home)
            # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
            assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
        asyncio.run(probe())


async def _replies(ws, ids):
    """The replies to *ids* in whatever order they arrive (events in between are skipped)."""
    replies = {}
    async with asyncio.timeout(30):
        while not set(ids) <= set(replies):
            reply = json.loads(await ws.recv())
            if reply.get('id') in ids:
                replies[reply['id']] = reply
    return replies


@pytest.mark.platforms("linux")
def test_corrections_pipelined_on_one_socket_need_no_earlier_ack(tmp_path):
    """A client sends A, B and A again on ONE socket without waiting for any reply, and treats A's
    reply as lost. Controls on a socket dispatch in order, yet each is answered from the child as it
    is read, so B is answered and delivered whatever happened to A, and A's resend is accepted."""
    marker = tmp_path / 'stale-tool-ran'
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, marker)

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_STALE')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                for ident, text in (('a', 'HOLD_A'), ('b', 'HOLD_B'), ('a-again', 'HOLD_A')):
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': ident, 'method': 'session.redirect', 'params': {
                        'session_id': sid, 'execution_generation': generation, 'text': text}}))
                replies = await _replies(ws, ('b', 'a-again'))  # A's own reply is never read
                for ident in ('b', 'a-again'):
                    assert replies[ident].get('result', {}).get('status') in {'redirected', 'queued'}, replies
                await async_wait_until(lambda: any('HOLD_B' in json.dumps(r['messages']) for r in peer.requests),
                                       timeout=40, interval=.05, what='B in a provider request')
                peer.release.set()
                await _stale_answered(peer, home)
            assert not marker.exists(), 'a fenced response dispatched its tool call'
            delivered = ' '.join(content for role, content in _transcript(home, sid) if role == 'user')
            assert 'HOLD_B' in delivered and delivered.count('HOLD_A') >= 1
            assert _turns(home) == [('turn', 'terminal')], _kept(home)
            # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
            assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
        asyncio.run(probe())
