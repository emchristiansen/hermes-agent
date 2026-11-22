"""Managed persistence and receipt polling must not take SQLite locks on the owner loop."""
import asyncio
from contextlib import contextmanager
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_worker_reservation import reserve_admission_worker
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tests.gateway.test_worker_admission_reservation import hello


@pytest.fixture
def runtime(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='test-owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='test-owner', db=db, epoch=epoch)
        authority.sessions['owned'] = LiveSession(None, 'route')
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned',
                            request_id='input', payload={})
        row = claim_session_input(db, epoch=epoch, session_id='owned')
        yield authority, row


def forbid_loop_reads(db, monkeypatch):
    loop_thread = threading.get_ident()
    original = db._read_ctx

    @contextmanager
    def read_context(**kwargs):
        assert threading.get_ident() != loop_thread, 'SQLite read can block the owner event loop'
        with original(**kwargs) as conn:
            yield conn

    monkeypatch.setattr(db, '_read_ctx', read_context)


@pytest.mark.asyncio
async def test_worker_authentication_and_persistence_read_off_loop(runtime, monkeypatch):
    authority, row = runtime
    process = await asyncio.to_thread(subprocess.Popen, [sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=process,
                                         principal_id='human', hello=hello(process))
        connection = AuthorityConnection(authority, object(), {
            'user_id': 'human', 'provider': 'local', 'profile_id': authority.profile_id,
            'instance_id': authority.instance_id, 'capabilities': ['worker:adopt']})
        forbid_loop_reads(authority.db, monkeypatch)
        identity = {key: value for key, value in scope.items() if key != 'epoch'}
        adopted = await connection.dispatch({'id': 1, 'method': 'worker.adopt', 'params': identity})
        assert 'error' not in adopted, adopted
        reply = await connection.dispatch({'id': 2, 'method': 'worker.persist', 'params': scope | {
            'sequence': 1, 'operation': 'session.sidecars', 'payload': {'patch': {'_usage_anchor': {'value': 1}}}}})
        assert 'error' not in reply, reply
    finally:
        process.stdin.close()
        process.wait(timeout=5)


@pytest.mark.asyncio
async def test_submitter_receipt_poll_reads_off_loop(runtime, monkeypatch):
    authority, row = runtime
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    forbid_loop_reads(authority.db, monkeypatch)
    reply = await connection.dispatch({'id': 1, 'method': 'prompt.receipt', 'params': {
        'session_id': 'owned', 'admission_id': row['admission_id'], 'include_result': True}})
    assert 'error' not in reply, reply
    assert reply['result']['status'] == 'started'


@pytest.mark.asyncio
@pytest.mark.parametrize('retire', [False, True])
async def test_terminal_result_read_off_loop_rechecks_retirement(runtime, monkeypatch, retire):
    from gateway import session_results
    authority, row = runtime
    session_results.retain_result(authority.db, epoch=authority.epoch, row=row,
        result={'result': {'final_response': 'completed once', 'messages': []}, 'usage': {}})
    original = session_results.admission_result
    loop = asyncio.get_running_loop()

    def read_result(*args):
        result = original(*args)
        if retire:
            loop.call_soon_threadsafe(setattr, authority, 'retired', True)
        return result

    monkeypatch.setattr(session_results, 'admission_result', read_result)
    forbid_loop_reads(authority.db, monkeypatch)
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    reply = await connection.dispatch({'id': 1, 'method': 'prompt.receipt', 'params': {
        'session_id': 'owned', 'admission_id': row['admission_id'], 'include_result': True}})
    if retire:
        assert reply['error']['message'] == 'runtime_retired'
    else:
        assert reply['result']['status'] == 'terminal'
        assert reply['result']['result']['final_response'] == 'completed once'


@pytest.mark.asyncio
async def test_blocked_receipt_read_does_not_block_another_owner_connection(runtime, tmp_path, monkeypatch):
    """The documented reader-pool fallback may wait on a real writer lock, never the loop."""
    authority, row = runtime
    db = authority.db
    loop = asyncio.get_running_loop()
    reading = asyncio.Event()
    locked, release, timed_out = threading.Event(), threading.Event(), threading.Event()

    def hold_writer():
        with db._lock:
            locked.set()
            if not release.wait(5):
                timed_out.set()

    def unavailable_reader():
        loop.call_soon_threadsafe(reading.set)

    holder = threading.Thread(target=hold_writer)
    holder.start()
    await asyncio.to_thread(locked.wait, 5)
    monkeypatch.setattr(db, '_checkout_read_conn', unavailable_reader)
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    request = {'id': 1, 'method': 'prompt.receipt', 'params': {
        'session_id': 'owned', 'admission_id': row['admission_id']}}
    pending = asyncio.create_task(connection.dispatch(request))
    try:
        await asyncio.wait_for(reading.wait(), 5)
        # A different profile's receipt on the same loop must remain deliverable.
        with SessionDB(tmp_path / 'other.db') as other_db:
            other_db.create_session('other', 'cli')
            other_epoch = begin_runtime_epoch(other_db, instance_id='other-owner')
            other = SessionAuthority(SimpleNamespace(_draining=False), profile_id='other',
                                     instance_id='other-owner', db=other_db, epoch=other_epoch)
            other.sessions['other'] = LiveSession(None, 'other-route')
            other_row = admit_session_input(other_db, epoch=other_epoch, principal_id='human',
                session_id='other', request_id='other-input', payload={})
            peer = AuthorityConnection(other, object(), {'user_id': 'human'})
            reply = await peer.dispatch({'id': 2, 'method': 'prompt.receipt', 'params': {
                'session_id': 'other', 'admission_id': other_row['admission_id']}})
            assert reply['result']['status'] == 'queued'
            assert not timed_out.is_set(), 'receipt read held the event loop until writer watchdog released'
            assert not pending.done()
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 5)
        reply = await asyncio.wait_for(pending, 5)
    assert reply['result']['status'] == 'started'


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['worker.adopt', 'worker.persist'])
async def test_retirement_during_worker_verification_prevents_operation(runtime, monkeypatch, method):
    from gateway import session_worker
    authority, row = runtime
    process = await asyncio.to_thread(subprocess.Popen, [sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    verified, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = session_worker._verify

    def gated_verify(*args):
        original(*args)
        loop.call_soon_threadsafe(verified.set)
        assert release.wait(5), 'verification must yield to retirement'

    try:
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=process,
                                         principal_id='human', hello=hello(process))
        connection = AuthorityConnection(authority, object(), {
            'user_id': 'human', 'provider': 'local', 'profile_id': authority.profile_id,
            'instance_id': authority.instance_id, 'capabilities': ['worker:adopt']})
        identity = {key: value for key, value in scope.items() if key != 'epoch'}
        if method == 'worker.persist':
            assert 'result' in await connection.dispatch({'id': 1, 'method': 'worker.adopt', 'params': identity})
        monkeypatch.setattr(session_worker, '_verify', gated_verify)
        params = identity if method == 'worker.adopt' else scope | {
            'sequence': 1, 'operation': 'session.sidecars', 'payload': {'patch': {'_usage_anchor': {'value': 1}}}}
        pending = asyncio.create_task(connection.dispatch({'id': 2, 'method': method, 'params': params}))
        try:
            await asyncio.wait_for(verified.wait(), 5)
            authority.retired = True
        finally:
            release.set()
        reply = await asyncio.wait_for(pending, 5)
        assert reply['error']['message'] == 'runtime_retired'
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 0
            status = conn.execute('SELECT status FROM worker_executions').fetchone()[0]
        assert status == ('registered' if method == 'worker.adopt' else 'running')
    finally:
        release.set()
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)

