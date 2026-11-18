"""``hermes kanban redirect|steer``: the exit verdict for each owner answer, through the real owner code.

The reply fed to the CLI mapping is produced by the owner's own code on a real temporary board: a
correction kept on the card (ended claim, or a live child that rejected it) is exit 0 "kept on card, not
delivered"; a correction whose outcome the owner cannot state (a real verdict timeout, an ``unknown``
verdict) is exit 4 "ambiguous; text retained; resend allowed" (a message resend, never a business replay);
authorization, stale and validation refusals stay exit 3, a transport failure stays exit 4."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_client as gc
from hermes_cli.kanban_redirect import _send


def _wire(monkeypatch, result=None, error=None):
    calls = []

    class Client:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'session.kanban_run':
                return {'session_id': 'the-original-session', 'execution_generation': 7}
            if error:
                raise error
            return result

    @asynccontextmanager
    async def connect():
        yield Client()
    monkeypatch.setattr(gc, 'connect_gateway', connect)
    return calls


def _card(title='Original task'):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    with connect_closing() as conn:
        task_id = kb.create_task(conn, title=title, body='Do work', workspace_kind='scratch')
    return task_id, {'db': str(kb.kanban_db_path()), 'task_id': task_id, 'run_id': 12}


def _comments(task_id):
    from hermes_cli.kanban_db_connect import connect_closing
    with connect_closing() as conn:
        return [row[0] for row in conn.execute('SELECT body FROM task_comments WHERE task_id=?', (task_id,))]


def _verdict(monkeypatch, task_id, reply, verb='redirect', text='RETAIN_THIS_TEXT'):
    _wire(monkeypatch, result=reply)
    return asyncio.run(_send(task_id, 'default', 12, verb, text))


def test_kept_real_board_ack_is_success(monkeypatch):
    from gateway.session_managed_worker import keep_card_correction
    task_id, context = _card()
    reply = asyncio.run(keep_card_correction(SimpleNamespace(epoch=13), context, 'RETAIN_THIS_TEXT', 7, ended=True))
    assert any('RETAIN_THIS_TEXT' in body for body in _comments(task_id))
    assert reply['kept'] == 'task_comment' and reply['status'] == 'rejected' and reply['delivered'] is False
    code, payload = _verdict(monkeypatch, task_id, reply)
    assert code == 0, (code, payload)
    assert payload['status'] == 'kept' and payload['note'] == 'kept on card, not delivered'
    assert payload['result'] == reply  # the owner's raw answer is preserved


@pytest.mark.asyncio
async def test_child_rejected_keep_is_success_and_resend_keeps_a_duplicate(monkeypatch):
    """A live child answers ``rejected``: the owner has already kept the text on the card's task."""
    from gateway.session_managed_worker import ManagedWorker, managed_correction
    from tests.gateway.test_managed_worker_correction_controls import ACTOR, REF, _authority
    from tests.waiting import async_wait_until
    task_id, context = _card()
    worker = ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    worker.policy = SimpleNamespace(kanban_json=json.dumps(context))
    authority = _authority(worker)
    pending = asyncio.ensure_future(managed_correction(authority, ACTOR, REF, worker, 'steer', 'CHILD_REJECTED', 7))
    await async_wait_until(lambda: worker.commands.qsize() == 1, what='correction sent')
    frame = worker.commands.get_nowait()
    assert worker.answer_correction({'type': 'correction', 'id': frame['id'], 'status': 'rejected'})
    reply = await pending
    assert reply['kept'] == 'task_comment' and reply['delivered'] is False
    assert any('CHILD_REJECTED' in body for body in _comments(task_id))
    code, payload = await asyncio.to_thread(_verdict, monkeypatch, task_id, reply, 'steer', 'CHILD_REJECTED')
    assert code == 0 and payload['note'] == 'kept on card, not delivered'


@pytest.mark.parametrize('reply', [{'status': 'rejected', 'delivered': False},
                                   {'status': 'rejected', 'delivered': False, 'kept': None}])
def test_rejected_and_not_kept_stays_refused(monkeypatch, reply):
    code, payload = _verdict(monkeypatch, 'card', reply)
    assert code == 3 and payload['status'] == 'rejected', (code, payload)


@pytest.mark.parametrize('reply', [{'status': 'redirected'}, {'status': 'queued'}, {'status': 'steered'}])
def test_delivered_and_queued_are_success(monkeypatch, reply):
    assert _verdict(monkeypatch, 'card', reply)[0] == 0


@pytest.mark.asyncio
async def test_real_owner_verdict_timeout_is_ambiguous_exit_4(monkeypatch):
    """The owner's own timeout (a child verdict past ``CORRECTION_ACK_SECONDS``) as the RPC error it becomes."""
    from gateway import session_managed_worker as managed
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_managed_worker_correction_controls import ACTOR, REF, _authority
    monkeypatch.setattr(managed, 'CORRECTION_ACK_SECONDS', .2)
    worker = managed.ManagedWorker(SimpleNamespace(pid=0))
    worker.ready = True
    authority = _authority(worker)
    with pytest.raises(RuntimeStoreError) as raised:
        await managed.managed_correction(authority, ACTOR, REF, worker, 'redirect', 'A', 7)
    assert raised.value.reason == 'correction_outcome_unknown'
    _wire(monkeypatch, error=gc.GatewayRPCError(raised.value.reason))
    code, payload = await _send('card', 'default', 12, 'redirect', 'A')
    assert code == 4 and payload['status'] == 'ambiguous', (code, payload)
    assert payload['note'] == 'ambiguous; text retained; resend allowed'
    assert payload['session_id'] == 'the-original-session' and payload['execution_generation'] == 7


def test_unanswered_outcome_is_ambiguous_exit_4(monkeypatch):
    _wire(monkeypatch, error=gc.GatewayRPCError('correction_outcome_unknown'))
    code, payload = asyncio.run(_send('card', 'default', 12, 'steer', 'A'))
    assert code == 4 and payload['status'] == 'ambiguous' and payload['reason'] == 'correction_outcome_unknown'
    assert payload['note'] == 'ambiguous; text retained; resend allowed'


@pytest.mark.parametrize('error', [gc.GatewayRPCError('unknown_method', -32601),
                                   gc.GatewayRPCError('request_failed', -32601, {'reason': 'unknown_method'}),
                                   gc.GatewayRPCError('invalid_timeout'),
                                   gc.GatewayRPCError('correction_timeout'),
                                   gc.GatewayRPCError('outcome_unknown')])
def test_definitive_refusals_that_merely_resemble_ambiguity_are_exit_3_and_not_retained(monkeypatch, error):
    _wire(monkeypatch, error=error)
    code, payload = asyncio.run(_send('card', 'default', 12, 'redirect', 'A'))
    assert code == 3 and payload['status'] == 'refused' and 'note' not in payload, (code, payload)
    assert payload['reason'] == str(error)


@pytest.mark.parametrize('message', ['Gateway did not answer session.redirect within 30 s',
                                     'Gateway disconnected; turn outcome is unknown.'])
def test_transport_timeout_and_connection_loss_are_exit_4(monkeypatch, message):
    _wire(monkeypatch, error=gc.GatewayClientError(message))
    code, payload = asyncio.run(_send('card', 'default', 12, 'redirect', 'A'))
    assert code == 4 and payload['status'] == 'unreachable', (code, payload)


@pytest.mark.parametrize('reason', ['permission_denied', 'stale_kanban_claim', 'stale_execution_generation',
                                    'invalid_params', 'correction_not_kept', 'profile_mismatch'])
def test_authorization_stale_and_validation_refusals_stay_exit_3(monkeypatch, reason):
    _wire(monkeypatch, error=gc.GatewayRPCError(reason))
    code, payload = asyncio.run(_send('card', 'default', 12, 'redirect', 'A'))
    assert code == 3 and payload['status'] == 'refused' and payload['reason'] == reason


def test_discovery_refusal_stays_exit_3_and_transport_failure_is_exit_4(monkeypatch):
    class Client:
        async def rpc(self, method, **params):
            raise gc.GatewayRPCError('stale_kanban_claim')

    @asynccontextmanager
    async def connect():
        yield Client()
    monkeypatch.setattr(gc, 'connect_gateway', connect)
    code, payload = asyncio.run(_send('card', 'default', 12, 'redirect', 'A'))
    assert code == 3 and payload['stage'] == 'discovery'
    _wire(monkeypatch, error=gc.GatewayClientError('gateway rpc timed out'))
    code, payload = asyncio.run(_send('card', 'default', 12, 'redirect', 'A'))
    assert code == 4 and payload['status'] == 'unreachable'
