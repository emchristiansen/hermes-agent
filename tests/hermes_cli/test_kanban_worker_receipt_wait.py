"""The Kanban submitter outlasts an owner event-loop stall instead of ending the turn.

2026-11-04: the gateway's event loop stalled (``ws write slow (loop stalled >10s)``; cause not
measured). A submitter polling ``prompt.receipt`` through such a stall hits the 30 s RPC default and
exits "Gateway connection failed", and its bound worker's submitter watch then ``os._exit``-s the
managed turn: the signature of 7 runs at 17:19:53 and of 315-318 at 16:30:18.
Merged with v0.23.1's ``_Claim`` loop: an unanswered poll fails at the RPC default and the worker
redials; the owner may stay silent for ``OWNER_WAIT_S`` (180 s, from its last answer), longer than
those stalls. Past admission the receipt is only read, so a lost or unanswered read is re-read on a
new connection for the same admission. A ``kanban.run`` whose answer was lost is sent again for the
same attempt: the owner deduplicates it (tests/gateway/test_kanban_run_resend_dedup.py), so at-least-
once delivery never runs a card twice. Time is scaled 1000x: 30 s -> 30 ms.
"""
import asyncio
import contextlib
import json
import time

import pytest

from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import DEFAULT_RPC_TIMEOUT, GatewayClient
from websockets.protocol import State


@pytest.mark.asyncio
@pytest.mark.parametrize('url, keepalive', [
    ('ws://127.0.0.1:8377/api/ws', {'ping_interval': None}),
    ('ws://localhost:8377/api/ws', {'ping_interval': None}),
    ('wss://gateway.example.com/api/ws', {}),
])
async def test_no_keepalive_ping_toward_a_loopback_owner(monkeypatch, url, keepalive):
    """The loopback listener sends no keepalive (its pong would wait on a starved loop); the
    client must not impose one either. A non-loopback owner keeps the library keepalive."""
    calls = []

    class _Ws:
        subprotocol = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

    @contextlib.asynccontextmanager
    async def fake_connect(target, **kwargs):
        calls.append((target, kwargs))
        yield _Ws()

    import websockets.asyncio.client as ws_client
    monkeypatch.setattr(ws_client, 'connect', fake_connect)
    monkeypatch.setenv('HERMES_TUI_GATEWAY_URL', url)
    from hermes_cli.gateway_client import connect_gateway
    async with connect_gateway():
        pass
    [(target, kwargs)] = calls
    assert target == url
    assert {k: v for k, v in kwargs.items() if k.startswith('ping')} == keepalive


class _Owner:
    """One owner life on one socket: ``answers[method]`` builds each reply; ``None`` leaves the
    request unanswered and ``'hang-up'`` closes the socket instead."""

    state = State.OPEN  # GatewayClient.rpc refuses a socket that is not open

    def __init__(self, **answers):
        self.answers, self.inbox, self.sent = answers, asyncio.Queue(), []

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        answer = self.answers[req['method']]
        if answer == 'hang-up':
            await self.inbox.put(None)
        elif answer is not None:
            await self.inbox.put(json.dumps({'id': req['id'], 'result': answer}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self.inbox.get()
        if raw is None:
            raise StopAsyncIteration
        return raw


_ADMITTED = {'session_id': 's1', 'receipt': {'status': 'running', 'admission_id': 'a1'}}
_DONE = {'status': 'terminal', 'admission_id': 'a1'}


def _owner_lives(monkeypatch, *owners):
    """``connect_gateway`` dials the next owner life each time (an ``asyncio.Event``: a connect that
    sets it and never completes); one past the last is unreachable."""
    from hermes_cli import gateway_client
    lives, dialled = iter(owners), []

    @contextlib.asynccontextmanager
    async def dial(ready_timeout=None):
        dialled.append(ready_timeout)
        owner = next(lives, None)
        if owner is None:
            raise gateway_client.GatewayClientError('Gateway connection failed; no local fallback')
        if isinstance(owner, asyncio.Event):
            owner.set()
            await asyncio.Event().wait()
        async with GatewayClient(owner) as client:
            yield client
    monkeypatch.setattr(gateway_client, 'connect_gateway', dial)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0, 'last_output': 'done'})
    return dialled


_PARAMS = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}


@pytest.mark.asyncio
async def test_a_receipt_poll_stalled_past_its_budget_is_re_read_and_completes_the_run(monkeypatch, tmp_path):
    """The owner's loop stalls through the first poll (unanswered past the 30 s RPC default): the
    worker does not exit and end the turn; it re-reads the same admission on a new connection."""
    real_wait_for, budgets = asyncio.wait_for, []

    def scaled(fut, timeout):
        budgets.append(timeout)
        return real_wait_for(fut, None if timeout is None else timeout / 1000)
    monkeypatch.setattr(asyncio, 'wait_for', scaled)
    stalled = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': None})
    recovered = _Owner(**{'prompt.receipt': _DONE})
    _owner_lives(monkeypatch, stalled, recovered)
    code = await real_wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 10)
    assert code == 0
    assert [r['method'] for r in stalled.sent] == ['kanban.run', 'prompt.receipt']
    assert [(r['method'], r['params']) for r in recovered.sent] == [
        ('prompt.receipt', {'session_id': 's1', 'admission_id': 'a1'})]
    assert budgets[1] == DEFAULT_RPC_TIMEOUT
    assert all('_timeout' not in r['params'] for r in stalled.sent + recovered.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize('lost', ['hang-up', None])
async def test_a_lost_receipt_read_is_re_read_for_the_same_admission(monkeypatch, tmp_path, lost):
    """A hang-up or an unanswered poll after admission re-reads the same admission on a new
    connection; the claim is never submitted twice and the bound turn completes."""
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, 'wait_for', lambda fut, timeout: real_wait_for(fut, None if timeout is None else timeout / 1000))
    first = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': lost})
    second = _Owner(**{'prompt.receipt': _DONE})
    dialled = _owner_lives(monkeypatch, first, second)
    code = await real_wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 5)
    assert code == 0
    # Every connection ensures the owner with the ensure's own deadline (v0.23.1).
    assert dialled == [None, None]
    assert [r['method'] for r in first.sent] == ['kanban.run', 'prompt.receipt']
    assert [(r['method'], r['params']) for r in second.sent] == [('prompt.receipt', {'session_id': 's1', 'admission_id': 'a1'})]


@pytest.mark.asyncio
async def test_a_claim_whose_admission_answer_was_lost_is_resubmitted_for_the_same_attempt(monkeypatch, tmp_path):
    """At-least-once: the answer was lost, so the worker cannot know whether the owner admitted the
    attempt and asks again with the identical attempt identity. The owner returns the original
    admission for a copy (tests/gateway/test_kanban_run_resend_dedup.py), so the card runs once."""
    first = _Owner(**{'kanban.run': 'hang-up'})
    second = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': _DONE})
    _owner_lives(monkeypatch, first, second)
    code = await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 5)
    assert code == 0
    assert [r['method'] for r in first.sent] == ['kanban.run']
    assert [r['method'] for r in second.sent] == ['kanban.run', 'prompt.receipt']
    assert second.sent[0]['params'] == first.sent[0]['params']  # the same attempt, never a new one


@pytest.mark.asyncio
async def test_receipt_reads_stop_once_the_owner_has_been_silent_past_the_wait(monkeypatch, tmp_path):
    """An admitted attempt whose owner never answers again fails the worker honestly once the owner
    has been silent for ``OWNER_WAIT_S`` from its last answer; the claim is not resubmitted."""
    from hermes_cli.gateway_client import GatewayClientError
    first = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': 'hang-up'})
    _owner_lives(monkeypatch, first)
    monkeypatch.setattr(kanban_worker_client, 'OWNER_WAIT_S', 0.5)
    began = time.monotonic()
    with pytest.raises(GatewayClientError, match=r'^Kanban owner silent for \d+s'):
        await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 10)
    assert time.monotonic() - began < 0.5 + 2.5  # the bound plus one jittered pause (at most 2 s)
    assert [r['method'] for r in first.sent] == ['kanban.run', 'prompt.receipt']


@pytest.mark.asyncio
async def test_a_real_owner_closing_between_polls_is_re_read_on_a_new_connection(monkeypatch, tmp_path):
    """Real sockets through the real ``connect_gateway``: the owner closes cleanly right after
    admitting, so the next poll's send meets a closed socket; the same admission is re-read on a
    new connection and ``kanban.run`` is not sent again."""
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from gateway import session_kanban
    from hermes_cli import gateway_client, gateway_runtime
    connections = []

    async def owner(ws):
        sent = []
        connections.append(sent)
        async for raw in ws:
            req = json.loads(raw)
            sent.append((req['method'], req['params']))
            await ws.send(json.dumps({'id': req['id'], 'result': _ADMITTED if req['method'] == 'kanban.run' else _DONE}))
            if len(connections) == 1:
                await ws.close(code=1000, reason='owner rotates the socket')
                return

    async with serve(owner, '127.0.0.1', 0, ping_interval=None, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
        port = server.sockets[0].getsockname()[1]
        ready = SimpleNamespace(state='ready', reason_code=None,
                                endpoint=SimpleNamespace(api_origin=f'http://127.0.0.1:{port}'))
        monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
        monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', lambda *a, **k: ready)
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda *a, **k: 'ticket')
        monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0})
        code = await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 10)
    assert code == 0
    assert [m for m, _ in connections[0]] == ['kanban.run']
    assert connections[1:] == [[('prompt.receipt', {'session_id': 's1', 'admission_id': 'a1'})]]


@pytest.mark.asyncio
async def test_an_unanswered_request_says_so_instead_of_a_failed_connection(monkeypatch):
    from hermes_cli.gateway_client import GatewayClientError
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, 'wait_for', lambda fut, timeout: real_wait_for(fut, timeout / 1000))
    async with GatewayClient(_Owner(**{'prompt.receipt': None})) as client:
        with pytest.raises(GatewayClientError, match=r'^Gateway did not answer prompt.receipt within 180 s$'):
            await client.rpc('prompt.receipt', _timeout=180.0, session_id='s1', admission_id='a1')


async def _handshake_edge(monkeypatch, tmp_path, second_answer):
    """The owner admits and closes; the re-read's first connection meets a raw TCP peer that
    answers the WebSocket upgrade with ``second_answer`` (``b''``: drops it); the next connection
    reaches the owner again. Real sockets, real ``connect_gateway``."""
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from gateway import session_kanban
    from hermes_cli import gateway_client, gateway_runtime
    requests, upgrades, ensures = [], [], []

    async def owner(ws):
        async for raw in ws:
            req = json.loads(raw)
            requests.append(req['method'])
            await ws.send(json.dumps({'id': req['id'], 'result': _ADMITTED if req['method'] == 'kanban.run' else _DONE}))
            if req['method'] == 'kanban.run':
                await ws.close()
                return

    async def edge(reader, writer):
        upgrades.append(await reader.read(4096))
        writer.write(second_answer)
        await writer.drain()
        writer.close()

    async with serve(owner, '127.0.0.1', 0, ping_interval=None, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as good:
        async with await asyncio.start_server(edge, '127.0.0.1', 0) as bad:
            def ensure(home, **kwargs):
                ensures.append(kwargs)
                port = (bad if len(ensures) == 2 else good).sockets[0].getsockname()[1]
                return SimpleNamespace(state='ready', reason_code=None,
                                       endpoint=SimpleNamespace(api_origin=f'http://127.0.0.1:{port}'))
            monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
            monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
            monkeypatch.setattr(gateway_client, '_session_ticket', lambda *a, **k: 'ticket')
            monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0})
            try:
                outcome = await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 10)
            except Exception as exc:  # the outcome under test
                outcome = exc
    return outcome, requests, len(upgrades), len(ensures)


@pytest.mark.asyncio
async def test_a_reconnect_whose_upgrade_is_cut_off_is_retried_for_the_same_admission(monkeypatch, tmp_path):
    outcome, requests, upgrades, ensures = await _handshake_edge(monkeypatch, tmp_path, b'')
    assert outcome == 0
    assert requests == ['kanban.run', 'prompt.receipt'] and upgrades == 1 and ensures == 3


@pytest.mark.asyncio
async def test_an_owner_refusing_the_upgrade_is_retried_for_the_same_admission(monkeypatch, tmp_path):
    """v0.23.1: a refused handshake is a restarting owner until the silence wait runs out (its
    status is named then); the admitted attempt is only re-read, never resubmitted."""
    outcome, requests, upgrades, ensures = await _handshake_edge(
        monkeypatch, tmp_path, b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
    assert outcome == 0
    assert requests == ['kanban.run', 'prompt.receipt'] and upgrades == 1 and ensures == 3


@pytest.mark.asyncio
async def test_a_send_cut_off_by_a_close_reports_an_unknown_delivery():
    """A send can fail after the frame left (while it waits to drain): never claim nothing was sent."""
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close
    from hermes_cli.gateway_client import GatewayClientError

    class _Closing(_Owner):
        async def send(self, raw):
            raise ConnectionClosedOK(Close(1000, ''), Close(1000, ''), True)

    async with GatewayClient(_Closing()) as client:
        with pytest.raises(GatewayClientError, match='turn outcome is unknown'):
            await client.rpc('prompt.receipt', session_id='s1', admission_id='a1')
