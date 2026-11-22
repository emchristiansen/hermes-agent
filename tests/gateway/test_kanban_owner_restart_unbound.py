"""An attempt its owner admitted but never bound is booked like a bound one when that owner restarts.

Run 2642 of t_08749b6f (2026-11-05, 12:44): the old gateway claimed and admitted it (``owner_admitted``)
and was stopped before the attempt's interpreter wrote ``worker_bound``. The owner fenced the lost
worker with its own end as the cause, but ``record_owner_restart_result`` fenced only by the bound
claim, so it recorded nothing, and the admission settled ``terminal``/``interrupted``. That reads as
"paused" (``owner_reclaim_paused``), and with no ``worker_bound`` only ``owner_attempt_provably_over``
could lift it, which waited for the claim to expire and for the TASK row's heartbeat (stamped by the
previous run) to age past an hour. The 19 bound runs of that restart were booked ``owner_restarted`` at
the first tick; 2642 stayed ``running`` on the dead claim, headed for a counted crash.

Real SessionStore and board SQLite, the owner's real drain (``_fence_lost_worker`` then
``settle_unattended_unknown``) or the successor's real boot (``initialize_session_authority``), the
submitter's result reader and the real dispatcher tick (``dispatch_once``, whose crash sweep books the
run and whose claim starts the successor).
"""
import json
import os
import time

import psutil
import pytest

from gateway.session_contract import SessionRef
from gateway.session_unattended_unknown import owner_ending
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
from hermes_cli.kanban_owner_recovery import owner_attempt_provably_over, owner_reclaim_blocked
from hermes_cli.process_identity import worker_birth
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
from hermes_state_runtime import get_session_admission
from tests.gateway.test_kanban_owner_restart_durable import SID, _bind_worker, _owner, _start_turn
from tests.hermes_cli.test_kanban_owner_restart_result import _dead_pid

OWNER_RESTART = KANBAN_OWNER_RESTART_EXIT_CODE


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A card claimed by an owner that is now gone; nothing is admitted yet. ``(store, runner, conn,
    params, board_db)`` as in ``test_kanban_owner_restart_durable``'s ``card``, without ``worker_bound``."""
    yield from _board(tmp_path, monkeypatch, register_worker=True)


@pytest.fixture
def unregistered(tmp_path, monkeypatch):
    """``board`` as the dispatcher leaves it between spawning the submitter and ``_set_worker_pid``:
    the run has no worker pid yet."""
    yield from _board(tmp_path, monkeypatch, register_worker=False)


def _board(tmp_path, monkeypatch, *, register_worker):
    from types import SimpleNamespace
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state_local import POLICY_PREFIX
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, 'profile_exists', lambda name: True)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    conn = kbc.connect()
    host = kb._claimer_id().split(':', 1)[0]
    tid = kb.create_task(conn, title='card', assignee='w')
    task = kb.claim_task(conn, tid, claimer=f'{host}:{_dead_pid()}')  # the old gateway's dispatcher
    if register_worker:
        kbd._set_worker_pid(conn, tid, _dead_pid())  # its spawned submitter, gone by the sweep
    board_db = conn.execute('PRAGMA database_list').fetchone()[2]
    params = {'board': 'default', 'task_id': tid, 'run_id': task.current_run_id, 'claim_lock': task.claim_lock}
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _adapter_for_source=lambda source: None)
    store._db.create_session(SID, source='test')
    context = dict(params, db=board_db)
    store._db._execute_write(lambda c: c.execute('INSERT INTO state_meta(key,value) VALUES (?,?)', (
        POLICY_PREFIX + SID, json.dumps({'session_id': SID, 'profile_id': 'p', 'principal_id': 'kanban', 'request_id': 'attempt',
                                         'entry': {'session_id': SID}, 'policy': {'kanban_json': json.dumps(context)}}))))
    yield store, runner, conn, params, board_db
    conn.close()
    store.close_all_db_handles()


def _admitted(conn, store, params, *, owner_pid, owner_birth):
    """The owner's ``owner_admitted`` marker on the run, as ``session_kanban.run_task`` writes it."""
    with kb.write_txn(conn):
        kb._append_event(conn, params['task_id'], 'owner_admitted', {
            'db': str(store._db.db_path), 'session_id': SID, 'request_id': 'attempt',
            'pid': owner_pid, 'birth': owner_birth}, run_id=params['run_id'])


def _dead_owner(conn, store, params):
    _admitted(conn, store, params, owner_pid=_dead_pid(), owner_birth=1.0)


def _live_owner(conn, store, params):
    _admitted(conn, store, params, owner_pid=os.getpid(), owner_birth=worker_birth(psutil.Process()))


def _bound(conn, params):
    with kb.write_txn(conn):
        kb._append_event(conn, params['task_id'], 'worker_bound', {
            'pid': _dead_pid(), 'claim_lock': params['claim_lock'], 'started_at': 'x'}, run_id=params['run_id'])


def _submitter_exited(params, rc):
    """The submitter's own log trailer, as ``kanban_worker_client.main`` leaves it (run 2642's said 1)."""
    path = kb.worker_log_path(params['task_id'], board='default')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'Gateway disconnected; turn outcome is unknown.\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n')


def _results(conn, params):
    return [json.loads(p) for (p,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result' ORDER BY id",
        (params['task_id'], params['run_id']))]


async def _lost_on_the_stopping_drain(store, runner, params):
    """The owner is stopping when it finds the attempt's managed worker lost, before it bound."""
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    owner = await _owner(runner, 'old')
    row = _start_turn(owner, params)
    scope = _bind_worker(owner, row)  # the owner's worker registration; the board has no worker_bound
    runner._draining = True
    await _fence_lost_worker(owner, row, scope, owner_ending(owner))
    assert await settle_unattended_unknown(owner, SessionRef('p', SID))
    return row


async def _left_in_flight_for_the_next_boot(store, runner, params):
    """The owner dies with the attempt started and unbound; its successor boots."""
    from gateway.session_authority import initialize_session_authority
    row = _start_turn(await _owner(runner, 'old'), params)
    await initialize_session_authority(runner, profile_id='p', instance_id='successor')
    return row


def _tick(conn, spawned):
    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return _dead_pid()
    return kbd.dispatch_once(conn, spawn_fn=spawn)


def _run(conn, run_id):
    return tuple(conn.execute('SELECT status, outcome FROM task_runs WHERE id=?', (run_id,)).fetchone())


# --- the residual: admitted, never bound ------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('claim_lapsed', [False, True], ids=['claim-live', 'claim-lapsed'])
@pytest.mark.parametrize('lost', [_lost_on_the_stopping_drain, _left_in_flight_for_the_next_boot],
                         ids=['stopping-drain', 'successor-boot'])
async def test_an_admitted_attempt_that_never_bound_is_booked_owner_restarted(board, lost, claim_lapsed):
    """``claim-lapsed``: the submitter outlived the claim. The tick's stale-claim reclaim runs before
    its crash sweep and must leave the accounted run to it, as it leaves a bound one."""
    store, runner, conn, params, board_db = board
    _dead_owner(conn, store, params)
    row = await lost(store, runner, params)
    if claim_lapsed:
        with kb.write_txn(conn):
            conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (int(time.time()) - 1, params['task_id']))

    settled = get_session_admission(store._db, admission_id=row['admission_id'])
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    submitter_pid, recorded = kb.get_task(conn, params['task_id']).worker_pid, _results(conn, params)
    from hermes_cli.kanban_worker_client import _Admission, _settled
    submitter_read = _settled(_Admission(SID, settled), params, board_db)

    # The submitter itself lost the owner and exited 1, as run 2642's did; the claim is still live.
    _submitter_exited(params, 1)
    spawned = []
    res = _tick(conn, spawned)

    # Booked at the first tick, uncounted, and its successor dispatched, as the bound runs were.
    assert res.owner_restarted == [params['task_id']] and not res.crashed and not res.reclaimed
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    # Control for COMPOSE-ACCOUNT-N1: this ADMITTED turn was settled interrupted (asserted above), and
    # its persisted error, from the owner's recorded 79, still says exactly that.
    (error,) = conn.execute('SELECT error FROM task_runs WHERE id=?', (params['run_id'],)).fetchone()
    assert "its turn was settled interrupted (not replayed)" in error and 'nothing was sent' not in error
    successor = kb.get_task(conn, params['task_id'])
    assert spawned == [params['task_id']]
    assert (successor.status, successor.consecutive_failures) == ('running', 0)
    assert successor.current_run_id != params['run_id']
    # From the owner's record, under the run's own claim and its spawned worker's pid; the
    # submitter's own reader agrees.
    assert [(r['exit_code'], r['claim_lock'], r['pid']) for r in recorded] == [
        (OWNER_RESTART, params['claim_lock'], submitter_pid)]
    assert submitter_read == OWNER_RESTART


@pytest.mark.asyncio
async def test_the_unbound_record_is_fenced_to_the_admitted_session_and_the_exact_run(board):
    from gateway.session_kanban import record_owner_restart_result
    store, _runner, conn, params, board_db = board
    card = {key: params[key] for key in ('board', 'task_id', 'run_id')}
    # Never admitted (no marker): nothing to account for, whoever asks.
    assert not record_owner_restart_result(card, board_db, SID)
    _dead_owner(conn, store, params)
    assert not record_owner_restart_result(card, board_db)  # no session to fence by
    assert not record_owner_restart_result(card, board_db, 'local-another-session')
    assert not record_owner_restart_result(dict(card, run_id=params['run_id'] + 1), board_db, SID)
    assert not _results(conn, params)
    assert record_owner_restart_result(card, board_db, SID)
    assert not record_owner_restart_result(card, board_db, SID)  # never twice
    assert len(_results(conn, params)) == 1


# --- before admission ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_owner_restart_after_claim_before_admission_leaves_the_attempt_to_its_submitter(board):
    """No admission: the successor's boot has nothing of this attempt to settle or record, and no owner
    pause holds it. A live submitter keeps it (it submits to the successor once that is ready); a
    dead one is booked from its own exit at the next tick, never parked on a dead claim."""
    from gateway.session_authority import initialize_session_authority
    _store, runner, conn, params, _board_db = board
    await _owner(runner, 'old')
    await initialize_session_authority(runner, profile_id='p', instance_id='successor')
    assert not _results(conn, params)
    assert not owner_reclaim_blocked(conn, params['task_id'], crash_sweep=True)

    with kb.write_txn(conn):  # the submitter still waits for an owner
        conn.execute('UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?',
                     (os.getpid(), kbd._process_fingerprint(os.getpid()), params['task_id']))
    assert kbd.detect_crashed_workers(conn) == []
    assert _run(conn, params['run_id']) == ('running', None)

    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET worker_pid=?, worker_started_at=NULL WHERE id=?',
                     (_dead_pid(), params['task_id']))
    _submitter_exited(params, 1)
    assert kbd.detect_crashed_workers(conn) == [params['task_id']]  # its own exit decides, at once
    assert _run(conn, params['run_id']) == ('crashed', 'crashed')


# --- per-run liveness ----------------------------------------------------------------------------

def _previous_run_heartbeat(conn, params, age):
    """The task row's heartbeat, stamped by the previous run ``age`` seconds ago; the current run's
    own column stays NULL (it never heartbeated). Its claim has expired."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (now - 60, params['run_id']))
        conn.execute('UPDATE tasks SET last_heartbeat_at=?, claim_expires=? WHERE id=?',
                     (now - age, now - 1, params['task_id']))


async def _interrupted_without_result(runner, params):
    """The owner's ledger as the old code left 2642's: the admission settled ``terminal``/``interrupted``
    and no result on the board (here a loss with no restart cause, so none is owed)."""
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    owner = await _owner(runner, 'old')
    row = _start_turn(owner, params)
    await _fence_lost_worker(owner, row, _bind_worker(owner, row), owner_ending(owner))
    assert await settle_unattended_unknown(owner, SessionRef('p', SID))


@pytest.mark.asyncio
async def test_a_previous_runs_heartbeat_is_no_evidence_the_current_attempt_lives(board):
    """2642's shape at 13:19 under the old code: admitted, unbound, terminal/interrupted, no result,
    claim expired, owner and submitter gone, and the task row's heartbeat (run 2605's, 41 minutes old)
    inside the hour. The run never heartbeated, so nothing attests it is alive: the stale-claim path
    releases it (here with no owner record, so as an ordinary reclaim)."""
    store, runner, conn, params, _board_db = board
    _dead_owner(conn, store, params)
    await _interrupted_without_result(runner, params)
    _previous_run_heartbeat(conn, params, age=41 * 60)

    assert owner_attempt_provably_over(conn, params['task_id'])
    assert kb.release_stale_claims(conn) == 1
    assert _run(conn, params['run_id'])[0] == 'reclaimed'


@pytest.mark.asyncio
async def test_the_current_runs_own_heartbeat_still_holds_the_attempt(board):
    store, runner, conn, params, _board_db = board
    _dead_owner(conn, store, params)
    await _interrupted_without_result(runner, params)
    _previous_run_heartbeat(conn, params, age=41 * 60)
    assert kbd.heartbeat_worker(conn, params['task_id'], expected_run_id=params['run_id'])

    assert not owner_attempt_provably_over(conn, params['task_id'])
    assert kb.release_stale_claims(conn) == 0
    assert kbd.detect_crashed_workers(conn) == []
    assert _run(conn, params['run_id']) == ('running', None)


# --- controls -------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_control_a_bound_attempt_lost_to_the_restart_is_booked_as_before(board):
    store, runner, conn, params, _board_db = board
    _dead_owner(conn, store, params)
    _bound(conn, params)
    row = await _lost_on_the_stopping_drain(store, runner, params)
    assert get_session_admission(store._db, admission_id=row['admission_id'])['status'] == 'terminal'
    bound = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound'",
                                    (params['task_id'],)).fetchone()[0])
    assert [(r['exit_code'], r['pid']) for r in _results(conn, params)] == [(OWNER_RESTART, bound['pid'])]
    _submitter_exited(params, OWNER_RESTART)
    spawned = []
    res = _tick(conn, spawned)

    assert res.owner_restarted == [params['task_id']] and spawned == [params['task_id']]
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', [False, True], ids=['before-bind', 'bound'])
async def test_control_a_worker_lost_under_a_live_owner_still_counts(board, bound):
    """The owner is serving, not ending, when its managed worker dies: no restart cause, no record,
    and the run is a counted crash once nothing can still be executing it."""
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    store, runner, conn, params, _board_db = board
    _live_owner(conn, store, params)
    if bound:
        _bound(conn, params)
    owner = await _owner(runner, 'live')
    row = _start_turn(owner, params)
    await _fence_lost_worker(owner, row, _bind_worker(owner, row), owner_ending(owner))
    assert await settle_unattended_unknown(owner, SessionRef('p', SID))
    settled = get_session_admission(store._db, admission_id=row['admission_id'])
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    assert not _results(conn, params)
    with kb.write_txn(conn):  # the claim lapsed; no heartbeat
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (int(time.time()) - 1, params['task_id']))
    _submitter_exited(params, 1)

    assert kbd.detect_crashed_workers(conn) == [params['task_id']]
    assert _run(conn, params['run_id']) == ('crashed', 'crashed')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 1


# --- the owner records before the dispatcher registers the run's worker (review2881 CF-1) ----------

# No process can hold this pid (it is above every Linux ``pid_max``): the registered worker is dead,
# and a signal a reclaim path sends it can reach no other process.
NEVER_PID = 2 ** 31 - 1


def _register(conn, params, *, verified=False):
    """The dispatcher's ``_set_worker_pid`` for the run, which lands only after the spawn.
    ``verified``: the spawn's fingerprint was captured, so a reclaim path signals the pid."""
    assert not psutil.pid_exists(NEVER_PID)
    kbd._set_worker_pid(conn, params['task_id'], NEVER_PID, run_id=params['run_id'], claim_lock=params['claim_lock'])
    if verified:
        with kb.write_txn(conn):
            conn.execute('UPDATE tasks SET worker_started_at=? WHERE id=?',
                         (kbd._process_fingerprint(os.getpid()), params['task_id']))
    assert kb.get_task(conn, params['task_id']).worker_pid == NEVER_PID


def _lapse(conn, params):
    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (int(time.time()) - 1, params['task_id']))


@pytest.mark.asyncio
@pytest.mark.parametrize('claim_lapsed', [False, True], ids=['claim-live', 'claim-lapsed'])
@pytest.mark.parametrize('lost', [_lost_on_the_stopping_drain, _left_in_flight_for_the_next_boot],
                         ids=['stopping-drain', 'successor-boot'])
async def test_a_restart_recorded_before_the_worker_pid_is_registered_survives_the_registration(
        unregistered, lost, claim_lapsed):
    """The dispatcher spawns, then registers the pid; the owner admitted the attempt and ended in
    between, so its record has no pid. The registration that lands next is the run's own worker's:
    the record speaks for it, and the tick books the run ``owner_restarted``, uncounted."""
    store, runner, conn, params, _board_db = unregistered
    _dead_owner(conn, store, params)
    await lost(store, runner, params)
    assert kb.get_task(conn, params['task_id']).worker_pid is None
    assert [(r['exit_code'], r['claim_lock'], r['pid']) for r in _results(conn, params)] == [
        (OWNER_RESTART, params['claim_lock'], None)]
    _register(conn, params)
    if claim_lapsed:
        _lapse(conn, params)
    _submitter_exited(params, 1)
    spawned = []
    res = _tick(conn, spawned)

    assert res.owner_restarted == [params['task_id']] and not res.crashed and not res.reclaimed
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    successor = kb.get_task(conn, params['task_id'])
    assert spawned == [params['task_id']]
    assert (successor.status, successor.consecutive_failures) == ('running', 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('then', ['late-registration', 'operator-reclaim'])
async def test_a_restart_recorded_for_a_worker_never_registered_is_held_uncounted(unregistered, then):
    """Registration never arrives (the dispatcher died between spawn and ``_set_worker_pid``): no
    sweep can prove the run's worker dead, so the run keeps its claim, uncounted and with the
    owner's record intact, tick after tick; nothing is spawned beside it. A registration that
    arrives at last is booked ``owner_restarted``; otherwise an operator ``reclaim`` releases it,
    uncounted."""
    store, runner, conn, params, _board_db = unregistered
    _dead_owner(conn, store, params)
    await _lost_on_the_stopping_drain(store, runner, params)
    _lapse(conn, params)
    _submitter_exited(params, 1)
    for _ in range(2):
        spawned = []
        res = _tick(conn, spawned)
        assert not (res.owner_restarted or res.crashed or res.reclaimed or res.stale or res.timed_out or spawned)
        task = kb.get_task(conn, params['task_id'])
        assert (task.status, task.claim_lock, task.worker_pid, task.consecutive_failures) == (
            'running', params['claim_lock'], None, 0)
        assert _run(conn, params['run_id']) == ('running', None)
    assert [(r['exit_code'], r['pid']) for r in _results(conn, params)] == [(OWNER_RESTART, None)]

    if then == 'late-registration':
        _register(conn, params)
        res = _tick(conn, [])
        assert res.owner_restarted == [params['task_id']] and not res.crashed and not res.reclaimed
        assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    else:
        assert kb.reclaim_task(conn, params['task_id'])
        assert _run(conn, params['run_id']) == ('reclaimed', 'reclaimed')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0


# --- a restart committed while a reclaim path terminates the worker (review2881 CF-2) -------------

_ENTRIES = {'max-runtime': kbd.enforce_max_runtime, 'ttl': kb.release_stale_claims}


async def _reclaim_with_owner_step_in_kill(entry, owner_step):
    """Run the real reclaimer ``entry`` on its own board connection. Its termination step (the
    ``signal_fn`` hook, standing in for the signal) runs ``owner_step`` on the owner's event loop and
    returns once that committed: between the reclaimer's first look and its under-lock re-check."""
    import asyncio
    from contextlib import closing
    loop = asyncio.get_running_loop()
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        if len(signals) == 1:
            asyncio.run_coroutine_threadsafe(owner_step(), loop).result(timeout=120)

    def reclaim():
        with closing(kbc.connect()) as reclaimer_conn:
            return _ENTRIES[entry](reclaimer_conn, signal_fn=signal_fn)
    return await asyncio.to_thread(reclaim), signals


async def _serving_owner_past_its_limits(store, runner, conn, params, bound):
    """A live owner runs the admitted attempt (its admission ``running``, so the reclaimer's first
    look passes); the run is past its max runtime and its claim has expired."""
    _live_owner(conn, store, params)
    if bound:
        _bound(conn, params)
    _register(conn, params, verified=True)
    _lapse(conn, params)
    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET max_runtime_seconds=1 WHERE id=?', (params['task_id'],))
        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time()) - 10, params['run_id']))
    owner = await _owner(runner, 'old')
    row = _start_turn(owner, params)
    scope = _bind_worker(owner, row)
    assert not owner_reclaim_blocked(conn, params['task_id'])
    assert not _results(conn, params)
    return owner, row, scope


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', [False, True], ids=['unbound', 'bound'])
@pytest.mark.parametrize('owner_step', ['drain-settled', 'recorded-only'])
@pytest.mark.parametrize('entry', ['max-runtime', 'ttl'])
async def test_a_restart_committed_during_a_reclaim_kill_is_left_to_the_crash_sweep(unregistered, entry, owner_step, bound):
    """``drain-settled``: the owner's real stopping drain records ``owner_restarted`` and settles the
    admission ``terminal``/``interrupted`` while the reclaimer waits on its kill. ``recorded-only``:
    the drain has committed the record but not yet settled the admission. Either way the committed
    cause outranks the reclaimer: it books nothing and counts nothing, and the crash sweep books the
    run ``owner_restarted``, uncounted."""
    from gateway.session_kanban import record_owner_restart_result
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    import signal
    store, runner, conn, params, board_db = unregistered
    owner, row, scope = await _serving_owner_past_its_limits(store, runner, conn, params, bound)

    async def restart():
        if owner_step == 'recorded-only':
            assert record_owner_restart_result({key: params[key] for key in ('board', 'task_id', 'run_id')}, board_db, SID)
            return
        runner._draining = True
        await _fence_lost_worker(owner, row, scope, owner_ending(owner))
        assert await settle_unattended_unknown(owner, SessionRef('p', SID))

    booked, signals = await _reclaim_with_owner_step_in_kill(entry, restart)

    assert signals == [(NEVER_PID, signal.SIGTERM)]
    assert [r['exit_code'] for r in _results(conn, params)] == [OWNER_RESTART]
    assert not booked
    assert _run(conn, params['run_id']) == ('running', None)
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0

    assert kbd.detect_crashed_workers(conn) == []
    assert kbd.detect_crashed_workers._last_owner_restarted == [params['task_id']]
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', [False, True], ids=['unbound', 'bound'])
@pytest.mark.parametrize('entry', ['max-runtime', 'ttl'])
async def test_control_a_reclaim_kill_with_no_restart_cause_still_counts(unregistered, entry, bound):
    """The same interleaving with a serving owner: it loses the worker the reclaimer just killed and
    settles the admission ``terminal``/``interrupted`` with no restart cause, so no record. The
    reclaimer's own kill caused that pause; it books and counts the run as before."""
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import settle_unattended_unknown
    import signal
    store, runner, conn, params, _board_db = unregistered
    owner, row, scope = await _serving_owner_past_its_limits(store, runner, conn, params, bound)

    async def lost_under_a_serving_owner():
        await _fence_lost_worker(owner, row, scope, owner_ending(owner))
        assert await settle_unattended_unknown(owner, SessionRef('p', SID))

    booked, signals = await _reclaim_with_owner_step_in_kill(entry, lost_under_a_serving_owner)

    assert signals == [(NEVER_PID, signal.SIGTERM)]
    settled = get_session_admission(store._db, admission_id=row['admission_id'])
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    assert not _results(conn, params)
    assert booked == ([params['task_id']] if entry == 'max-runtime' else 1)
    outcome = 'timed_out' if entry == 'max-runtime' else 'reclaimed'
    assert _run(conn, params['run_id']) == (outcome, outcome)
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 1


# --- the interpreter binds after the owner's pid-less record (review2986 CF-3) --------------------
# The interpreter adopted the attempt before its owner ended and binds only after the owner recorded
# the restart (with no pid: none was registered yet) and the dispatcher registered its submitter.
# The binder checks the run and claim, not the owner's record, so the late bind succeeds. The
# binder (and the interpreter's own result) run in a disposable child: ``bind_worker_context``
# starts a watch that hard-exits its process once the submitter is dead.

_INTERPRETER = '''
import json, os, sys
from gateway import session_kanban as sk
frame, result, hold = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3] == "hold"
sk.bind_worker_context(frame)
if result != "none":  # the interpreter's own result, through its real recorder
    sk._record_worker_result(json.loads(frame["policy"]["kanban_json"]), int(result), "interpreter ended")
print("bound", os.getpid(), flush=True)
if hold:
    sys.stdin.read()  # alive until the test closes stdin
os._exit(0)
'''


def _interpreter(conn, params, board_db, workspace, *, result=None, hold=False):
    """A managed interpreter that binds the run now (real ``bind_worker_context``), optionally records
    its own result (real ``_record_worker_result``), and then exits, or stays alive while ``hold``."""
    import subprocess
    import sys
    from pathlib import Path
    context = dict(params, db=board_db, workspace=str(workspace), workspaces_root=str(workspace), profile='default',
                   branch=None, tenant=None, goal_mode=False, max_runtime_seconds=None)
    frame = {'policy': {'source': 'kanban', 'kanban_json': json.dumps(context)}}
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    child = subprocess.Popen([sys.executable, '-c', _INTERPRETER, json.dumps(frame), str(result or 'none'),
                              'hold' if hold else 'exit'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    line = child.stdout.readline()
    assert line.startswith('bound '), line + (child.communicate()[0] or '')
    if not hold:
        assert child.wait(timeout=120) == 0
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' "
                       "ORDER BY id DESC LIMIT 1", (params['task_id'], params['run_id'])).fetchone()
    assert (json.loads(row[0])['pid'], json.loads(row[0])['claim_lock']) == (child.pid, params['claim_lock'])
    return child


async def _restart_recorded_before_registration(store, runner, conn, params):
    """CF-1's shape: the owner ended before the dispatcher registered the submitter; its record has
    no pid. Then the registration lands."""
    _dead_owner(conn, store, params)
    await _lost_on_the_stopping_drain(store, runner, params)
    assert [(r['exit_code'], r['claim_lock'], r['pid']) for r in _results(conn, params)] == [
        (OWNER_RESTART, params['claim_lock'], None)]
    _register(conn, params)


def _events(conn, params):
    return [(kind, json.loads(payload) if payload else None) for kind, payload in conn.execute(
        'SELECT kind, payload FROM task_events WHERE task_id=? AND run_id=? ORDER BY id', (params['task_id'], params['run_id']))]


@pytest.mark.asyncio
@pytest.mark.parametrize('claim_lapsed', [False, True], ids=['claim-live', 'claim-lapsed'])
@pytest.mark.parametrize('interpreter_result', [None, 1], ids=['died-without-result', 'later-result-1'])
async def test_an_interpreter_that_binds_after_the_pid_less_record_does_not_override_it(
        unregistered, tmp_path, interpreter_result, claim_lapsed):
    """Owner record (no pid) -> submitter registered -> interpreter binds (and, ``later-result-1``,
    records exit 1 itself) -> both dead -> the real tick books ``owner_restarted``, uncounted. The
    owner's record was first; the late bind and the interpreter's later result do not displace it."""
    store, runner, conn, params, board_db = unregistered
    await _restart_recorded_before_registration(store, runner, conn, params)
    _interpreter(conn, params, board_db, tmp_path, result=interpreter_result)
    results = _results(conn, params)
    assert [r['exit_code'] for r in results] == [OWNER_RESTART] + ([1] if interpreter_result else [])
    if claim_lapsed:
        _lapse(conn, params)
    _submitter_exited(params, 1)
    spawned = []
    res = _tick(conn, spawned)

    assert res.owner_restarted == [params['task_id']] and not res.crashed and not res.reclaimed
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    successor = kb.get_task(conn, params['task_id'])
    assert spawned == [params['task_id']]
    assert (successor.status, successor.consecutive_failures) == ('running', 0)


@pytest.mark.asyncio
async def test_a_live_late_bound_interpreter_holds_the_run_until_it_is_dead(unregistered, tmp_path):
    """The submitter is dead but the late-bound interpreter still lives (it has recorded its own exit 1
    and not yet exited): nothing books the run, whatever the owner's record says. Once it is dead, the
    owner's record decides: ``owner_restarted``, uncounted."""
    store, runner, conn, params, board_db = unregistered
    await _restart_recorded_before_registration(store, runner, conn, params)
    child = _interpreter(conn, params, board_db, tmp_path, result=1, hold=True)
    try:
        _lapse(conn, params)
        _submitter_exited(params, 1)
        for _ in range(2):
            spawned = []
            res = _tick(conn, spawned)
            assert not (res.owner_restarted or res.crashed or res.reclaimed or res.timed_out or spawned)
            assert _run(conn, params['run_id']) == ('running', None)
    finally:
        child.stdin.close()
        assert child.wait(timeout=120) == 0
    res = _tick(conn, [])
    assert res.owner_restarted == [params['task_id']] and not res.crashed and not res.reclaimed
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0


@pytest.mark.asyncio
async def test_a_late_bound_restart_keeps_the_uncounted_first_failure_hold_and_exact_release(unregistered, tmp_path):
    """``max_retries=1``: the late-bound restart is held ``blocked`` with ``accounting_required=False``,
    no ``gave_up``, no failure counted; only the exact terminal run releases it."""
    store, runner, conn, params, board_db = unregistered
    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET max_retries=1 WHERE id=?', (params['task_id'],))
    await _restart_recorded_before_registration(store, runner, conn, params)
    _interpreter(conn, params, board_db, tmp_path)
    _submitter_exited(params, 1)
    spawned = []
    res = _tick(conn, spawned)

    assert res.owner_restarted == [params['task_id']] and not res.crashed and not spawned
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    task = kb.get_task(conn, params['task_id'])
    assert (task.status, task.consecutive_failures) == ('blocked', 0)
    kinds = [kind for kind, _ in _events(conn, params)]
    assert 'gave_up' not in kinds and 'crashed' not in kinds
    assert [p['accounting_required'] for kind, p in _events(conn, params) if kind == 'blocked'] == [False]
    assert not kb.unblock_task(conn, params['task_id'])
    assert not kb.unblock_task(conn, params['task_id'], expected_terminal_run=params['run_id'] - 1)
    assert kb.unblock_task(conn, params['task_id'], expected_terminal_run=params['run_id'])
    task = kb.get_task(conn, params['task_id'])
    assert (task.status, task.consecutive_failures) == ('ready', 0)


@pytest.mark.asyncio
async def test_a_restart_record_then_late_bind_is_held_by_a_timeout_kill_and_booked_uncounted(unregistered, tmp_path):
    """CF-2 composed with CF-3: the owner has recorded the restart (no pid yet) but not yet settled
    its admission, so the attempt still reads as running under a live owner. The submitter is
    registered, the interpreter binds late, and the run is past its max runtime. ``enforce_max_runtime``
    passes its first look, kills the submitter, then finds the owner's record under the lock and
    books nothing; the crash sweep books it ``owner_restarted``, uncounted."""
    import signal
    from gateway.session_kanban import record_owner_restart_result
    store, runner, conn, params, board_db = unregistered
    _live_owner(conn, store, params)
    _start_turn(await _owner(runner, 'old'), params)
    assert record_owner_restart_result({key: params[key] for key in ('board', 'task_id', 'run_id')}, board_db, SID)
    assert [r['pid'] for r in _results(conn, params)] == [None]
    _register(conn, params, verified=True)
    _interpreter(conn, params, board_db, tmp_path)
    assert not owner_reclaim_blocked(conn, params['task_id'])
    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET max_runtime_seconds=1 WHERE id=?', (params['task_id'],))
        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time()) - 10, params['run_id']))
    _submitter_exited(params, 1)
    signals = []
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: signals.append((pid, sig))) == []
    assert signals == [(NEVER_PID, signal.SIGTERM)]
    assert _run(conn, params['run_id']) == ('running', None)
    assert kbd.detect_crashed_workers(conn) == []
    assert _run(conn, params['run_id']) == ('owner_restarted', 'owner_restarted')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 0


# Negatives and counted controls: the same late-bind path without a qualifying owner record counts.

def _forged_result(conn, params, **payload):
    with kb.write_txn(conn):
        kb._append_event(conn, params['task_id'], 'worker_result',
                         dict({'pid': None, 'claim_lock': params['claim_lock'], 'exit_code': OWNER_RESTART,
                               'last_output': ''}, **payload), run_id=params['run_id'])


@pytest.mark.asyncio
@pytest.mark.parametrize('record', ['none', 'non-owner-null-pid', 'owner-null-pid-other-claim'])
async def test_control_a_late_bound_crash_without_the_owners_record_still_counts(unregistered, tmp_path, record):
    """A serving owner loses the worker (no restart cause, so no record), the run is registered,
    bound late and dies: a counted crash. ``non-owner-null-pid``: a pid-less exit-79 result that is
    not the owner's; ``owner-null-pid-other-claim``: the owner's pid-less record under another
    claim. Neither is the run's restart cause, so both still count."""
    from hermes_cli.kanban_db_workspace import OWNER_RECORDED_RESULT_KEY
    store, runner, conn, params, board_db = unregistered
    _live_owner(conn, store, params)
    await _interrupted_without_result(runner, params)
    assert not _results(conn, params)
    if record == 'non-owner-null-pid':
        _forged_result(conn, params)
    elif record == 'owner-null-pid-other-claim':
        _forged_result(conn, params, claim_lock=params['claim_lock'] + '-stale', **{OWNER_RECORDED_RESULT_KEY: 1})
    _register(conn, params)
    _interpreter(conn, params, board_db, tmp_path)
    _lapse(conn, params)
    _submitter_exited(params, 1)
    res = _tick(conn, [])

    assert res.crashed == [params['task_id']] and not res.owner_restarted
    assert _run(conn, params['run_id']) == ('crashed', 'crashed')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 1


@pytest.mark.asyncio
async def test_control_a_stale_runs_record_does_not_speak_for_the_next_run(unregistered, tmp_path):
    """The owner's pid-less record belongs to run N. An operator releases N; run N+1 is claimed,
    admitted, registered, bound late and lost under a serving owner: a counted crash."""
    store, runner, conn, params, board_db = unregistered
    await _restart_recorded_before_registration(store, runner, conn, params)
    assert kb.reclaim_task(conn, params['task_id'])
    task = kb.claim_task(conn, params['task_id'], claimer=params['claim_lock'])
    assert task.current_run_id != params['run_id']
    stale_run = params['run_id']
    params = dict(params, run_id=task.current_run_id, claim_lock=task.claim_lock)
    _live_owner(conn, store, params)  # the same session's admission, already terminal/interrupted
    _register(conn, params)
    _interpreter(conn, params, board_db, tmp_path)
    assert not _results(conn, params)
    _submitter_exited(params, 1)
    res = _tick(conn, [])

    assert res.crashed == [params['task_id']] and not res.owner_restarted
    assert _run(conn, params['run_id']) == ('crashed', 'crashed')
    assert _run(conn, stale_run) == ('reclaimed', 'reclaimed')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 1


@pytest.mark.asyncio
async def test_control_a_late_bound_timeout_without_the_owners_record_still_counts(unregistered, tmp_path):
    """The same late bind under a serving owner with no restart record: a genuine timeout, counted."""
    import signal
    store, runner, conn, params, board_db = unregistered
    await _serving_owner_past_its_limits(store, runner, conn, params, bound=False)
    _interpreter(conn, params, board_db, tmp_path)
    signals = []
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: signals.append((pid, sig))) == [params['task_id']]
    assert signals == [(NEVER_PID, signal.SIGTERM)]
    assert _run(conn, params['run_id']) == ('timed_out', 'timed_out')
    assert kb.get_task(conn, params['task_id']).consecutive_failures == 1
