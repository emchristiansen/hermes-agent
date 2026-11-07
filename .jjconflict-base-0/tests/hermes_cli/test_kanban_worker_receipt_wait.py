"""The Kanban submitter outlasts an owner event-loop stall instead of ending the turn.

2026-11-04: the gateway's event loop stalled (``ws write slow (loop stalled >10s)``; cause not
measured). A submitter polling ``prompt.receipt`` through such a stall hits the 30 s RPC default and
exits "Gateway connection failed", and its bound worker's submitter watch then ``os._exit``-s the
managed turn: the signature of 7 runs at 17:19:53 and of 315-318 at 16:30:18.
The receipt budget (180 s) is a finite client policy longer than those stalls.
Past admission the receipt is only read, so a lost or unanswered read is re-read on a new
connection for the same admission; ``kanban.run`` is never sent twice. Time is scaled 1000x:
30 s -> 30 ms.
"""
import asyncio
import contextlib
import json
import time

import pytest

from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClient


class _SlowReceiptOwner:
    """Admits at once with a running receipt; answers the first poll only after a 120 s stall."""

    def __init__(self):
        self.inbox, self.sent = asyncio.Queue(), []

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        if req['method'] == 'kanban.run':
            receipt = {'status': 'running', 'admission_id': 'a1'}
            await self.inbox.put(json.dumps({'id': req['id'], 'result': {'session_id': 's1', 'receipt': receipt}}))
        elif req['method'] == 'prompt.receipt':
            async def answer():
                await asyncio.sleep(0.12)  # health: allow HX013 -- a scaled 120 s owner stall, past the 30 ms default
                await self.inbox.put(json.dumps({'id': req['id'], 'result': {
                    'status': 'terminal', 'admission_id': 'a1'}}))
            asyncio.ensure_future(answer())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


@pytest.mark.asyncio
async def test_a_receipt_poll_answered_after_a_loop_stall_completes_the_run(monkeypatch, tmp_path):
    real_wait_for, budgets = asyncio.wait_for, []

    def scaled(fut, timeout):
        budgets.append(timeout)
        return real_wait_for(fut, None if timeout is None else timeout / 1000)
    monkeypatch.setattr(asyncio, 'wait_for', scaled)
    owner = _SlowReceiptOwner()
    _owner_lives(monkeypatch, owner)
    code = await real_wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 5)
    assert code == 0
    assert [r['method'] for r in owner.sent] == ['kanban.run', 'prompt.receipt']
    assert budgets[1] == kanban_worker_client.RECEIPT_ANSWER_BUDGET_S
    assert all('_timeout' not in r['params'] for r in owner.sent)


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
    # The first connection ensures with its own deadline; the re-read only with the silence left.
    assert dialled[0] is None and 0 < dialled[1] < kanban_worker_client.RECEIPT_OUTAGE_S
    assert [r['method'] for r in first.sent] == ['kanban.run', 'prompt.receipt']
    assert [(r['method'], r['params']) for r in second.sent] == [('prompt.receipt', {'session_id': 's1', 'admission_id': 'a1'})]


@pytest.mark.asyncio
async def test_a_claim_whose_admission_answer_was_lost_is_never_resubmitted(monkeypatch, tmp_path):
    from hermes_cli.gateway_client import GatewayClientError
    first = _Owner(**{'kanban.run': 'hang-up'})
    second = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': _DONE})
    _owner_lives(monkeypatch, first, second)
    with pytest.raises(GatewayClientError, match='Gateway disconnected'):
        await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 5)
    assert [r['method'] for r in first.sent] == ['kanban.run'] and not second.sent


@pytest.mark.asyncio
@pytest.mark.parametrize('next_life', ['unreachable', 'answers-late', 'connect-hangs'])
async def test_receipt_reads_stop_once_the_owner_has_been_silent_past_the_outage_budget(monkeypatch, tmp_path, next_life):
    """The silence bound is absolute: it covers reconnects and the pause between them, so an
    owner reached only after it expired is not read, and a hanging connect does not outlast it."""
    from hermes_cli.gateway_client import GatewayClientError
    first = _Owner(**{'kanban.run': _ADMITTED, 'prompt.receipt': 'hang-up'})
    second, connecting = _Owner(**{'prompt.receipt': _DONE}), asyncio.Event()
    # The 1 s pause between reads: 0.5 s expires within it; 1.5 s expires inside the next connect.
    lives, outage = {'unreachable': ([first], 0.5), 'answers-late': ([first, second], 0.5),
                     'connect-hangs': ([first, connecting], 1.5)}[next_life]
    _owner_lives(monkeypatch, *lives)
    monkeypatch.setattr(kanban_worker_client, 'RECEIPT_OUTAGE_S', outage)
    began = time.monotonic()
    with pytest.raises(GatewayClientError, match=rf'^Gateway gave no receipt for admission a1 for {outage:g} s'):
        await asyncio.wait_for(kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db')), 5)
    assert time.monotonic() - began < outage + 0.5
    assert [r['method'] for r in first.sent] == ['kanban.run', 'prompt.receipt'] and not second.sent
    assert connecting.is_set() == (next_life == 'connect-hangs')


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


def test_the_process_ends_within_the_silence_bound_even_while_an_owner_ensure_runs(monkeypatch, tmp_path):
    """Real sockets and the real ``connect_gateway``; the owner ensure is a thread honouring its
    ``timeout`` (as ``ensure_gateway_runtime`` does). After the owner closes, the re-read's ensure
    gets only the silence left, so ``asyncio.run`` (which waits for that thread) returns within
    the bound instead of the ensure's own 60 s default."""
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from hermes_cli import gateway_client, gateway_runtime

    async def owner(ws):
        async for raw in ws:
            req = json.loads(raw)
            await ws.send(json.dumps({'id': req['id'], 'result': _ADMITTED}))
            await ws.close(code=1000, reason='owner going away')
            return

    async def scenario():
        async with serve(owner, '127.0.0.1', 0, ping_interval=None,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            ready = SimpleNamespace(state='ready', reason_code=None, endpoint=SimpleNamespace(
                api_origin=f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'))
            starting = SimpleNamespace(state='starting', reason_code='not_ready', endpoint=None)
            calls = []

            def ensure(home, *, timeout=60.0, idle_exit=False):
                calls.append(timeout)
                if len(calls) == 1:
                    return ready
                time.sleep(min(timeout, 20))  # health: allow HX013 -- the ensure deliberately consumes the deadline it is given, as an owner that never becomes ready does
                return starting
            monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
            with pytest.raises(GatewayClientError, match='^Gateway gave no receipt'):
                await kanban_worker_client.run(_PARAMS, str(tmp_path / 'kanban.db'))
            return calls

    from hermes_cli.gateway_client import GatewayClientError
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    monkeypatch.setattr(gateway_client, '_session_ticket', lambda *a, **k: 'ticket')
    monkeypatch.setattr(kanban_worker_client, 'RECEIPT_OUTAGE_S', 1.5)
    began = time.monotonic()
    calls = asyncio.run(scenario())
    assert time.monotonic() - began < 3.0
    assert calls[0] == 60.0 and all(t <= 1.5 for t in calls[1:])


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
async def test_an_owner_refusing_the_upgrade_ends_observation_at_once(monkeypatch, tmp_path):
    from websockets.exceptions import InvalidStatus
    outcome, requests, upgrades, ensures = await _handshake_edge(
        monkeypatch, tmp_path, b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
    assert isinstance(outcome, InvalidStatus)
    assert requests == ['kanban.run'] and upgrades == 1 and ensures == 2


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
        with pytest.raises(GatewayClientError, match='whether the gateway received it is unknown'):
            await client.rpc('prompt.receipt', session_id='s1', admission_id='a1')
