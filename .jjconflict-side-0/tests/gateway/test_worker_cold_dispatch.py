"""Real reserved-child and cold-worker receipts must leave the owner loop available."""
import asyncio
from contextlib import contextmanager
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from gateway.config import Platform
from gateway.session_controls import AuthorityConnection
from gateway.session_worker_reservation import reserve_admission_worker
from tests.gateway import test_worker_receipt_loop_isolation as receipt_tests
from tests.gateway.test_worker_admission_reservation import hello

runtime = receipt_tests.runtime


async def prepare(authority, row):
    process = await asyncio.to_thread(subprocess.Popen,
        [sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=process,
                                     principal_id='human', hello=hello(process))
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'human', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': ['worker:adopt']})
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    reply = await connection.dispatch({'id': 1, 'method': 'worker.adopt', 'params': identity})
    assert 'result' in reply, reply
    return process, connection, scope


async def child_scope(authority, connection, scope):
    authority.sessions['owned'].source = SimpleNamespace(platform=Platform.TELEGRAM)
    reply = await connection.dispatch({'id': 2, 'method': 'worker.persist', 'params': scope | {
        'sequence': 1, 'operation': 'child.reserve', 'payload': {'secret': 'child-review-secret'}}})
    assert 'result' in reply, reply
    child = reply['result']['child']
    assert child['session_id'] not in authority.sessions
    return scope | child | {'secret': 'child-review-secret'}


@pytest.mark.asyncio
@pytest.mark.parametrize('session_state', ['warm', 'child', 'cold'])
async def test_worker_dispatch_reads_off_loop(runtime, monkeypatch, session_state):
    authority, row = runtime
    process, connection, scope = await prepare(authority, row)
    try:
        if session_state == 'child':
            scope = await child_scope(authority, connection, scope)
        elif session_state == 'cold':
            authority.sessions.pop(scope['session_id'])
        loop_thread = threading.get_ident()
        reads = []
        original = authority.db._read_ctx
        @contextmanager
        def record(**kwargs):
            reads.append(threading.get_ident())
            with original(**kwargs) as conn:
                yield conn
        monkeypatch.setattr(authority.db, '_read_ctx', record)
        reply = await connection.dispatch({'id': 3, 'method': 'worker.persist', 'params': scope | {
            'sequence': 1, 'operation': 'session.sidecars', 'payload': {'patch': {'_usage_anchor': {'value': 1}}}}})
        assert 'result' in reply, reply
        assert reads
        assert loop_thread not in reads, 'Actual child worker dispatch performed a DB read on owner loop'
    finally:
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('session_state', ['warm', 'child', 'cold'])
async def test_blocked_worker_dispatch_keeps_loop_available(runtime, monkeypatch, session_state):
    authority, row = runtime
    process, connection, scope = await prepare(authority, row)
    release, locked, timed_out = threading.Event(), threading.Event(), threading.Event()
    holder = None
    pending = None
    try:
        if session_state == 'child':
            scope = await child_scope(authority, connection, scope)
        elif session_state == 'cold':
            authority.sessions.pop(scope['session_id'])
        loop = asyncio.get_running_loop()
        reading = asyncio.Event()
        def unavailable():
            loop.call_soon_threadsafe(reading.set)
        monkeypatch.setattr(authority.db, '_checkout_read_conn', unavailable)
        def hold():
            with authority.db._lock:
                locked.set()
                if not release.wait(5):
                    timed_out.set()
        holder = threading.Thread(target=hold)
        holder.start()
        assert await asyncio.to_thread(locked.wait, 5)
        pending = asyncio.create_task(connection.dispatch({'id': 4, 'method': 'worker.persist', 'params': scope | {
            'sequence': 1, 'operation': 'session.sidecars', 'payload': {'patch': {'_usage_anchor': {'value': 2}}}}}))
        await asyncio.wait_for(reading.wait(), 8)
        assert not timed_out.is_set(), 'Child preflight stalled owner loop until writer watchdog release'
        assert not pending.done()
    finally:
        release.set()
        if holder is not None:
            await asyncio.to_thread(holder.join, 5)
        if pending is not None:
            reply = await asyncio.wait_for(pending, 5)
            assert 'result' in reply, reply
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)


@pytest.mark.asyncio
async def test_receipt_rechecks_retirement_after_admission_read(runtime, monkeypatch):
    from gateway import session_authority
    authority, row = runtime
    loop = asyncio.get_running_loop()
    original = session_authority.get_session_admission
    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        loop.call_soon_threadsafe(setattr, authority, 'retired', True)
        return result
    monkeypatch.setattr(session_authority, 'get_session_admission', read)
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    reply = await connection.dispatch({'id': 5, 'method': 'prompt.receipt', 'params': {
        'session_id': 'owned', 'admission_id': row['admission_id']}})
    assert reply['error']['message'] == 'runtime_retired'


@pytest.mark.asyncio
@pytest.mark.parametrize('session_state', ['child', 'cold'])
@pytest.mark.parametrize('fault,reason', [
    ('profile_id', 'profile_mismatch'), ('secret', 'permission_denied'),
    ('generation', 'stale_generation'), ('retired', 'runtime_retired'),
])
async def test_cold_worker_refusals_preserve_receipts(runtime, monkeypatch, session_state, fault, reason):
    from gateway import session_worker
    authority, row = runtime
    process, connection, scope = await prepare(authority, row)
    try:
        if session_state == 'child':
            scope = await child_scope(authority, connection, scope)
        else:
            authority.sessions.pop(scope['session_id'])
        with authority.db._read_ctx() as conn:
            before = conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0]
        if fault == 'retired':
            original = session_worker._verify
            loop = asyncio.get_running_loop()

            def retire_after_verify(*args):
                original(*args)
                loop.call_soon_threadsafe(setattr, authority, 'retired', True)

            monkeypatch.setattr(session_worker, '_verify', retire_after_verify)
        else:
            value = scope['generation'] + 1 if fault == 'generation' else 'different-profile-or-secret'
            scope = scope | {fault: value}
        reply = await connection.dispatch({'id': 6, 'method': 'worker.persist', 'params': scope | {
            'sequence': 1, 'operation': 'session.sidecars', 'payload': {'patch': {'_usage_anchor': {'value': 3}}}}})
        assert reply['error']['message'] == reason, reply
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == before
    finally:
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)

