"""An owner-restart cause survives the loss of the recovery that found it (RST-01, RST-02).

A card attempt lost because its owner ended carries that cause in the owner's state DB, committed
with its ``unknown`` fence (``OWNER_RESTART``). Its terminal receipt is published only after the run's
restart result is on the board, or is provably moot. So a cancelled drain, an owner that dies between
its boot fence and its settlement, and a locked board each still end with exactly one exit-79
result: the real submitter reader returns 79 and the real dispatcher counts no failure. Real
SessionStore and board SQLite, real boot (``initialize_session_authority``) and drain; a board lock
is a real competing writer, and only its retry budget is shortened.
"""
import asyncio
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session_unattended_unknown import owner_ending
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionActor, SessionRef
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_state_keys import local_identity
from hermes_state_runtime import OWNER_RESTART, admit_session_input, claim_session_input, get_session_admission
from tests.hermes_cli.test_kanban_owner_restart_result import _running
from tests.waiting import async_wait_until

# The attempt's local session: its identity is derived from (profile, principal, request).
SID = local_identity('p', 'kanban', 'attempt')


class _OwnerDied(BaseException):
    """The owner process ends here: nothing after this point of its boot runs."""


@pytest.fixture
def card(tmp_path, monkeypatch):
    """A claimed card whose submitter and bound interpreter are gone, admitted to a profile owner
    whose turn is running: ``(store, runner, board_conn, params, board_db)``."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    kbd._recent_worker_exits.clear()
    kb.init_db()
    conn = kbc.connect()
    tid, task = _running(conn, 'card')
    board_db = conn.execute('PRAGMA database_list').fetchone()[2]
    params = {'board': 'default', 'task_id': tid, 'run_id': task.current_run_id, 'claim_lock': task.claim_lock}
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _adapter_for_source=lambda source: None)
    store._db.create_session(SID, source='test')
    from hermes_state_local import POLICY_PREFIX
    context = {'board': 'default', 'task_id': tid, 'run_id': task.current_run_id, 'db': board_db}
    store._db._execute_write(lambda c: c.execute('INSERT INTO state_meta(key,value) VALUES (?,?)', (
        POLICY_PREFIX + SID, json.dumps({'session_id': SID, 'profile_id': 'p', 'principal_id': 'kanban', 'request_id': 'attempt',
                                         'entry': {'session_id': SID}, 'policy': {'kanban_json': json.dumps(context)}}))))
    yield store, runner, conn, params, board_db
    conn.close()
    store.close_all_db_handles()


async def _owner(runner, name):
    authority = await initialize_session_authority(runner, profile_id='p', instance_id=name)
    authority.sessions[SID] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='card'), SID)
    return authority


def _start_turn(authority, params):
    card = {key: params[key] for key in ('board', 'task_id', 'run_id')}
    admit_session_input(authority.db, epoch=authority.epoch, principal_id='kanban', session_id=SID,
                        request_id='attempt', payload={'text': 'work', 'kanban_card_v1': card})
    return claim_session_input(authority.db, epoch=authority.epoch, session_id=SID)


def _bind_worker(authority, row):
    authority.db._execute_write(lambda c: c.execute(
        "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
        "VALUES('execution',?,'compute',?,?,'registered','test')", (SID, authority.epoch, row['generation'])))
    return {'epoch': authority.epoch, 'generation': row['generation'], 'execution_id': 'execution'}


class _BoardLock:
    """A real competing board writer holding ``BEGIN IMMEDIATE``; the owner's writer gets no retries."""

    def __init__(self, board_db, monkeypatch):
        monkeypatch.setenv('HERMES_KANBAN_BUSY_TIMEOUT_MS', '1')
        monkeypatch.setattr(kbc, '_BUSY_MAX_RETRIES', 0)
        self.conn = sqlite3.connect(board_db, check_same_thread=False)
        self.conn.execute('BEGIN IMMEDIATE')

    def release(self):
        if self.conn.in_transaction:
            self.conn.rollback()


def _results(conn, params):
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result'",
        (params['task_id'], params['run_id']))]


def _consumed(db, conn, params, board_db, admission_id):
    """What the attempt's real submitter returns and how the real dispatcher books it."""
    from hermes_cli.kanban_worker_client import _Admission, _settled
    receipt = get_session_admission(db, admission_id=admission_id)
    code = _settled(_Admission(SID, receipt), params, board_db)
    kbd.detect_crashed_workers(conn)
    run = conn.execute('SELECT outcome FROM task_runs WHERE id=?', (params['run_id'],)).fetchone()[0]
    return (receipt['status'], receipt['outcome']), code, run, kb.get_task(conn, params['task_id']).consecutive_failures


def _assert_restart_booked(db, conn, params, board_db, admission_id):
    from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
    assert [r['exit_code'] for r in _results(conn, params)] == [KANBAN_OWNER_RESTART_EXIT_CODE]
    assert _consumed(db, conn, params, board_db, admission_id) == (
        ('terminal', 'interrupted'), KANBAN_OWNER_RESTART_EXIT_CODE, 'owner_restarted', 0)


@pytest.mark.asyncio
async def test_second_boot_after_the_recovery_fence_still_records_the_restart(card, monkeypatch):
    store, runner, conn, params, board_db = card
    first = await _owner(runner, 'owner-1')
    row = _start_turn(first, params)
    from gateway import session_unattended_unknown as unattended
    settle = unattended.settle_recovered_unattended

    def dies(*args, **kwargs):
        raise _OwnerDied
    # The next owner fences the row its predecessor left running, then dies before settling it.
    monkeypatch.setattr(unattended, 'settle_recovered_unattended', dies)
    with pytest.raises(_OwnerDied):
        await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')
    assert get_session_admission(store._db, admission_id=row['admission_id'])['status'] == 'unknown'
    monkeypatch.setattr(unattended, 'settle_recovered_unattended', settle)

    await initialize_session_authority(runner, profile_id='p', instance_id='owner-3')

    _assert_restart_booked(store._db, conn, params, board_db, row['admission_id'])


@pytest.mark.asyncio
async def test_drain_cancelled_after_the_loss_fence_still_records_the_restart(card, monkeypatch):
    store, runner, conn, params, board_db = card
    owner = await _owner(runner, 'owner-1')
    row = _start_turn(owner, params)
    scope = _bind_worker(owner, row)
    runner._draining = True  # the owner is stopping when it finds its worker lost
    from gateway import session_managed_worker as managed
    from gateway import session_worker_reservation as reservation
    fence, fenced, release = reservation.lose_admission_worker, asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def fence_then_wait(*args):
        fence(*args)  # the real fence commits
        loop.call_soon_threadsafe(fenced.set)
        assert release.wait(30)
    monkeypatch.setattr(reservation, 'lose_admission_worker', fence_then_wait)
    lost = asyncio.create_task(managed._fence_lost_worker(owner, row, scope, owner_ending(owner)))
    await asyncio.wait_for(fenced.wait(), 30)
    lost.cancel()  # the drain is cancelled as its fence returns: nothing after it runs
    with pytest.raises(asyncio.CancelledError):
        await lost
    release.set()
    await asyncio.gather(*owner._mutation_tasks)
    assert get_session_admission(store._db, admission_id=row['admission_id'])['status'] == 'unknown'
    assert not _results(conn, params)

    await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')

    _assert_restart_booked(store._db, conn, params, board_db, row['admission_id'])


@pytest.mark.asyncio
async def test_board_locked_at_boot_holds_the_receipt_until_the_drain_records_the_restart(card, monkeypatch):
    store, runner, conn, params, board_db = card
    first = await _owner(runner, 'owner-1')
    row = _start_turn(first, params)
    lock = _BoardLock(board_db, monkeypatch)
    try:
        owner = await _owner(runner, 'owner-2')
        # Not published, so not terminal: the row still owes its cause and reads as in flight, so
        # the submitter polling it keeps waiting instead of exiting on a cause-less receipt.
        held = get_session_admission(store._db, admission_id=row['admission_id'])
        assert (held['status'], held['outcome']) == ('unknown', OWNER_RESTART)
        assert not _results(conn, params)
        actor = SessionActor('kanban', 'p', frozenset({'session:submit', 'session:control'}), 'cli')
        assert (await owner.receipt(actor, SessionRef('p', SID), row['admission_id'])).status == 'started'

        from gateway import session_unattended_unknown as unattended
        publish, attempts = unattended.publish_owed_restart, []

        def lock_released_after_first_failure(db, owed):
            attempts.append(owed['admission_id'])
            try:
                return publish(db, owed)
            except sqlite3.OperationalError:
                lock.release()  # the competing writer finishes while the drain waits to retry
                raise
        monkeypatch.setattr(unattended, 'publish_owed_restart', lock_released_after_first_failure)
        live = owner.sessions[SID]
        owner._schedule(SessionRef('p', SID))  # what boot's ``recover_local_sessions`` schedules
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        live.task.result()
    finally:
        lock.release()
    assert attempts == [row['admission_id']] * 2  # retried in place, not dropped

    _assert_restart_booked(store._db, conn, params, board_db, row['admission_id'])
    # No replay: a later boot finds nothing owed and records nothing more.
    await initialize_session_authority(runner, profile_id='p', instance_id='owner-3')
    assert len(_results(conn, params)) == 1


@pytest.mark.asyncio
async def test_board_locked_on_the_stopping_drain_leaves_the_cause_for_the_next_boot(card, monkeypatch):
    store, runner, conn, params, board_db = card
    owner = await _owner(runner, 'owner-1')
    row = _start_turn(owner, params)
    scope = _bind_worker(owner, row)
    runner._draining = True
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    lock = _BoardLock(board_db, monkeypatch)
    try:
        await _fence_lost_worker(owner, row, scope, owner_ending(owner))
        assert not await settle_unattended_unknown(owner, SessionRef('p', SID))
    finally:
        lock.release()
    held = get_session_admission(store._db, admission_id=row['admission_id'])
    assert (held['status'], held['outcome']) == ('unknown', OWNER_RESTART)
    assert not _results(conn, params)

    await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')

    _assert_restart_booked(store._db, conn, params, board_db, row['admission_id'])


@pytest.mark.asyncio
async def test_an_owed_restart_never_replaces_the_interpreters_own_result(card):
    store, runner, conn, params, board_db = card
    first = await _owner(runner, 'owner-1')
    row = _start_turn(first, params)
    bound = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound'",
                                    (params['task_id'],)).fetchone()[0])
    with kb.write_txn(conn):
        kb._append_event(conn, params['task_id'], 'worker_result', {
            'pid': bound['pid'], 'claim_lock': params['claim_lock'], 'exit_code': 130, 'last_output': ''},
            run_id=params['run_id'])

    await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')

    assert [r['exit_code'] for r in _results(conn, params)] == [130]
    assert _consumed(store._db, conn, params, board_db, row['admission_id']) == (
        ('terminal', 'interrupted'), 130, 'crashed', 1)


@pytest.mark.asyncio
async def test_an_owed_restart_for_a_stale_claim_settles_without_a_result(card):
    store, runner, conn, params, _board_db = card
    first = await _owner(runner, 'owner-1')
    row = _start_turn(first, params)
    with kb.write_txn(conn):  # the board moved on: this attempt's claim is no longer the task's
        conn.execute("UPDATE tasks SET claim_lock='elsewhere:other' WHERE id=?", (params['task_id'],))

    await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')

    settled = get_session_admission(store._db, admission_id=row['admission_id'])
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    assert not _results(conn, params)


def test_boot_schedules_a_session_whose_only_unknown_row_owes_a_restart(monkeypatch):
    from gateway import session_local_recovery as recovery
    rows = {'owed': [{'status': 'unknown', 'outcome': OWNER_RESTART}],
            'person': [{'status': 'unknown', 'outcome': None}],
            'mixed': [{'status': 'unknown', 'outcome': OWNER_RESTART}, {'status': 'unknown', 'outcome': None}]}
    scheduled = []
    authority = SimpleNamespace(profile_id='p', db=None, _schedule=lambda ref: scheduled.append(ref.session_id))
    monkeypatch.setattr(recovery, 'local_receipt_ids', lambda authority: list(rows))
    monkeypatch.setattr(recovery, 'restore_local_session', lambda authority, sid: SessionRef('p', sid))
    monkeypatch.setattr(recovery, 'list_session_admissions', lambda db, session_id: rows[session_id])
    monkeypatch.setattr('gateway.session_viewer_close.rearm_persisted_closes', lambda authority: [])
    # main's boot also arms startup orphan closes (a store read); this fake authority has no store.
    monkeypatch.setattr('gateway.session_viewer_close.arm_startup_orphan_closes', lambda authority, orphans: None)

    recovery.recover_local_sessions(authority, schedule=True)

    assert scheduled == ['owed']
