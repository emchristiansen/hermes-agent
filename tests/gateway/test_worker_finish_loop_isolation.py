"""A committed worker finish must yield through lineage lookup before waking its FIFO."""
import asyncio
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest

from gateway import session_worker
from gateway.config import Platform
from tests.gateway import test_worker_receipt_loop_isolation as receipt_tests
from tests.gateway.test_worker_cold_dispatch import child_scope, prepare

runtime = receipt_tests.runtime


async def prepare_finish(authority, row, state):
    process, connection, scope = await prepare(authority, row)
    authority.sessions['owned'].source = SimpleNamespace(platform=Platform.TELEGRAM, user_id='human')
    if state == 'child':
        scope = await child_scope(authority, connection, scope)
    elif state == 'cold':
        authority.sessions.pop(scope['session_id'])
    return process, connection, scope


def finish(scope):
    return {'id': 9, 'method': 'worker.persist', 'params': scope | {
        'sequence': 1, 'operation': 'execution.finish', 'payload': {}}}


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['warm', 'child', 'cold'])
async def test_finish_and_retry_resolve_off_loop_but_schedule_on_loop(runtime, monkeypatch, state):
    authority, row = runtime
    process, connection, scope = await prepare_finish(authority, row, state)
    owner_thread = threading.get_ident()
    reads, scheduled = [], []
    original_read = authority.db._read_ctx

    @contextmanager
    def read(**kwargs):
        reads.append(threading.get_ident())
        with original_read(**kwargs) as conn:
            yield conn

    def schedule(ref):
        scheduled.append((ref.session_id, threading.get_ident()))

    # Observe the scheduling boundary; real FIFO execution is covered separately.
    monkeypatch.setattr(authority, '_schedule', schedule)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(authority.db, '_read_ctx', read)
            first = await connection.dispatch(finish(scope))
            retry = await connection.dispatch(finish(scope))
        assert first == retry
        assert first['result']['status'] == 'terminal'
        assert reads and owner_thread not in reads
        assert scheduled == ([('owned', owner_thread)] * 2 if state == 'warm' else [])
        with authority.db._read_ctx() as conn:
            stored = conn.execute('SELECT status,last_sequence FROM worker_executions WHERE execution_id=?',
                                  (scope['execution_id'],)).fetchone()
            count = conn.execute('SELECT COUNT(*) FROM worker_receipts WHERE execution_id=?',
                                 (scope['execution_id'],)).fetchone()[0]
        assert tuple(stored) == ('terminal', 1)
        assert count == 1
    finally:
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['warm', 'child', 'cold'])
async def test_postcommit_writer_lock_leaves_finish_and_loop_serviceable(runtime, monkeypatch, state):
    authority, row = runtime
    process, connection, scope = await prepare_finish(authority, row, state)
    release, locked, timed_out = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    reading = asyncio.Event()
    holder = pending = None
    original = session_worker.mutate_worker_execution

    def hold():
        with authority.db._lock:
            locked.set()
            if not release.wait(5):
                timed_out.set()

    def commit_then_contend(*args, **kwargs):
        nonlocal holder
        result = original(*args, **kwargs)
        holder = threading.Thread(target=hold)
        holder.start()
        assert locked.wait(5)
        return result

    def fallback():
        if locked.is_set():
            loop.call_soon_threadsafe(reading.set)

    try:
        monkeypatch.setattr(session_worker, 'mutate_worker_execution', commit_then_contend)
        monkeypatch.setattr(authority.db, '_checkout_read_conn', fallback)
        pending = asyncio.create_task(connection.dispatch(finish(scope)))
        await asyncio.wait_for(reading.wait(), 8)
        # An unrelated control refusal must answer while the finish waits on storage.
        reply = await connection.dispatch({'id': 10, 'method': 'worker.persist', 'params': []})
        assert reply['error']['message'] == 'invalid_params'
        assert not timed_out.is_set(), 'Finish blocked the owner loop until the writer watchdog fired'
        assert not pending.done()
    finally:
        release.set()
        if holder is not None:
            await asyncio.to_thread(holder.join, 5)
        if pending is not None:
            reply = await asyncio.wait_for(pending, 5)
            assert reply['result']['status'] == 'terminal', reply
        live = authority.sessions.get(scope['session_id'])
        if live is not None and live.task is not None:
            await asyncio.wait_for(live.task, 5)
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('keep_physical', [False, True])
async def test_compressed_finish_wakes_live_logical_owner_on_loop(runtime, monkeypatch, keep_physical):
    from gateway.session_authority import LiveSession
    authority, _row = runtime
    authority.db.end_session('owned', 'compression')
    authority.db.create_session('tip', source='cli', parent_session_id='owned')
    if keep_physical:
        authority.sessions['tip'] = LiveSession(None, 'tip-route')
    scheduled = []
    owner_thread = threading.get_ident()
    receipt_tests.forbid_loop_reads(authority.db, monkeypatch)

    def schedule(ref):
        scheduled.append((ref.session_id, threading.get_ident()))

    monkeypatch.setattr(authority, '_schedule', schedule)
    await authority.wake_after_worker('tip')
    expected = ['tip', 'owned'] if keep_physical else ['owned']
    assert scheduled == [(sid, owner_thread) for sid in expected]


@pytest.mark.asyncio
async def test_retirement_during_finish_lineage_read_preserves_commit_without_scheduling(runtime, monkeypatch):
    authority, row = runtime
    process, connection, scope = await prepare(authority, row)
    original = authority.logical_owner
    loop = asyncio.get_running_loop()
    scheduled = []

    def retire_after_read(session_id):
        result = original(session_id)
        loop.call_soon_threadsafe(setattr, authority, 'retired', True)
        return result

    monkeypatch.setattr(authority, 'logical_owner', retire_after_read)
    monkeypatch.setattr(authority, '_schedule', scheduled.append)
    try:
        reply = await connection.dispatch(finish(scope))
        assert reply['error']['message'] == 'runtime_retired', reply
        assert scheduled == []
        with authority.db._read_ctx() as conn:
            stored = conn.execute('SELECT status,last_sequence FROM worker_executions WHERE execution_id=?',
                                  (scope['execution_id'],)).fetchone()
        assert tuple(stored) == ('terminal', 1), 'Refusing the stale owner must not erase a committed finish'
    finally:
        process.stdin.close()
        await asyncio.to_thread(process.wait, 5)
