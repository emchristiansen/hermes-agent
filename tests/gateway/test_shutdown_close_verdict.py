"""SH-01 successor 3: the bootstrap's release follows a truthful close verdict on every exit.

Invariant (as in ``test_shutdown_hosted_room_writers``): while a thread that can still write a
profile's store is alive, or may be, the profile's ``gateway.lock`` is not released, so no
independent process can own that store; only a stop that actually closed the stores releases.

- STARTUP-01: the real ``start_gateway`` outer finally, cancelled in its pre-stop drain while a
  writer on the runner's executor is live, must not release the launch profile.
- A stop cancelled before its first step (the same bypass one step later) keeps the reservations.
- TOKEN-01: a SessionDB token writer still live past its close join keeps its store open and owned.
- CLOSE-QUAL-01: a store close that raises (here SessionDB's own retired-generation capture
  failure) does not attest a closed store, even where a quiet layer swallows the exception.
- Controls: a clean startup failure and a close that finishes still release.

Failures are injected (a held writer, a slow room Stop, a held token batch, a failed capture);
cancellation is the test's own. The contender is an independent process (``_contender``)."""
import asyncio
from contextlib import ExitStack, closing
import json
import logging
import os
import sqlite3
import threading
from unittest.mock import patch

import pytest

from tests.gateway.test_hosted_group_members import _gateway
from tests.gateway.test_shutdown_hosted_room_writers import _alpha, _contender, _probe, _stuck_room


def _answer(contender):
    return json.loads(contender.stdout.readline())['contender']


def _let_go(contender):
    if contender is not None:
        if contender.stdin:
            contender.stdin.close()
        contender.wait(10)
        contender.stdout.close()


def _bootstrap_patches(run_runner):
    """The external boot chores and listeners of ``start_gateway`` (no live services, no provider);
    runtime initialization, the drain, the stop, ownership and SessionDB stay real."""
    from gateway import run, run_bootstrap as boot, run_runtime as runtime

    async def nothing(*args, **kwargs):
        return None

    class _Control:
        async def stop(self):
            return None

    async def control(runner):
        return _Control()
    return [
        patch.object(boot, '_start_gateway_resolve_terminal_cwd'),
        patch.object(boot, '_start_gateway_early_environment_probe'),
        patch.object(boot, '_start_gateway_configure_logging'),
        patch.object(boot, '_start_gateway_claim_pid_file', return_value=True),
        patch.object(boot, '_start_gateway_install_signal_handlers'),
        patch.object(boot, '_start_gateway_start_control_socket', control),
        patch.object(boot, '_start_gateway_run_runner', run_runner),
        patch.object(run, '_host_attach_or_none', nothing),
        patch.object(run, '_enable_multiplex_log_routing'),
        patch.object(run, '_refresh_host_gateway_record'),
        patch.object(run, '_log_standalone_profiles_at_boot'),
        patch.object(run, '_ensure_windows_gateway_venv_imports'),
        patch.object(run, '_run_planned_stop_watcher'),
        patch('gateway.run_startup.recover_left_core_at_gateway_start'),
        patch.object(runtime, 'start_gateway_runtime_api', nothing),
        patch('hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive'),
        patch('gateway.lifecycle_ledger.record_startup', return_value=None),
        patch('gateway.code_skew.record_boot_fingerprint'),
        patch('hermes_cli.resource_limits.apply_nofile_soft_limit'),
    ]


def _launch_home():
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    (home / 'config.yaml').write_text('{}')
    return home


def _config():
    from gateway.config import GatewayConfig
    config = GatewayConfig()
    config.multiplex_profiles = False
    return config


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_a_cancelled_startup_cleanup_keeps_the_launch_profile_owned(tmp_path):
    """STARTUP-01. Startup fails after a writer started on the runner's executor; the bootstrap's
    outer finally is cancelled while its drain awaits a (slow) room Stop, before any stop ran. The
    release in that finally must keep the launch profile: the writer can still write its store.
    Then (control) a stop that closes the stores lets the same release go ahead."""
    from gateway.run_bootstrap import start_gateway
    from gateway.runtime_ownership import process_ownership
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.status import release_gateway_runtime_lock
    home = _launch_home()
    release_writer, release_stop, stop_entered = threading.Event(), threading.Event(), threading.Event()
    box, probes = {}, {}

    async def failed_start(runner, signals):
        box['runner'] = runner
        authority = runner.session_authority
        box['db_path'] = authority.db.db_path
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        authority.hosted_room_service = service
        original = service.stop

        def slow_stop(*, timeout=5):
            stop_entered.set()
            release_stop.wait(60)
            return original(timeout=timeout)
        service.stop = slow_stop

        def writer():
            release_writer.wait(60)
            probes['at_write'] = _probe(home)  # who holds the launch profile's flock as it writes
            authority.db.create_session('startup-late-write', source='test')
            return 'wrote'
        box['future'] = runner._get_executor().submit(writer)
        raise RuntimeError('injected startup failure after a writer started')

    contender = control_contender = None
    with ExitStack() as stack:
        for p in _bootstrap_patches(failed_start):
            stack.enter_context(p)
        task = asyncio.create_task(start_gateway(_config(), verbosity=None))
        try:
            assert await asyncio.to_thread(stop_entered.wait, 120), 'the outer finally drain was not reached'
            runner = box['runner']
            observed = {'before': {'owned': process_ownership.owns(home), 'stop_task': repr(runner._stop_task),
                                   'writer_live': not box['future'].done()}}
            task.cancel()  # a bounded caller gives up on the cleanup
            try:
                await asyncio.wait_for(task, 60)
                outcome = 'returned'
            except asyncio.CancelledError:
                outcome = 'CancelledError'
            except Exception as exc:  # the test reports it
                outcome = repr(exc)
            observed['after_cancel'] = {
                'outcome': outcome, 'owned': process_ownership.owns(home), 'stop_task': repr(runner._stop_task),
                'writer_live': not box['future'].done(), 'retention': process_ownership._retained_for}
            contender = _contender(home)
            observed['after_cancel']['contender'] = _answer(contender)
            release_writer.set()
            observed['late_write'] = await asyncio.wait_for(asyncio.wrap_future(box['future']), 60)
            observed['probe_at_write'] = probes.get('at_write')
            observed['old_pid'] = os.getpid()
            _let_go(contender)
            contender = None
            # Control: a stop that closes the stores, then the bootstrap's release, frees the profile.
            release_stop.set()
            await asyncio.wait_for(runner.stop(), 60)
            release_gateway_runtime_lock()
            control_contender = _contender(home)
            observed['control'] = {'owned': process_ownership.owns(home), 'contender': _answer(control_contender)}
            with closing(sqlite3.connect(f"file:{box['db_path']}?mode=ro", uri=True)) as db:
                observed['row_written'] = db.execute(
                    "SELECT count(*) FROM sessions WHERE id='startup-late-write'").fetchone()[0] == 1
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['before']['owned'] and observed['before']['writer_live'], observed
            assert observed['after_cancel']['outcome'] == 'CancelledError', observed
            assert observed['after_cancel']['stop_task'] == 'None', observed  # no stop ran: the bypass
            assert observed['after_cancel']['writer_live'], observed
            assert observed['after_cancel']['owned'], ('the cancelled startup cleanup released the launch profile '
                                                       'under a live writer', observed)
            assert observed['after_cancel']['contender'] == 'REFUSED', observed
            assert observed['late_write'] == 'wrote' and observed['row_written'], observed
            assert observed['probe_at_write'] == {'held': True, 'record_pid': os.getpid()}, observed
            assert observed['control'] == {'owned': False, 'contender': 'ACQUIRED'}, observed
        finally:
            release_writer.set()
            release_stop.set()
            if not task.done():
                task.cancel()
            _let_go(contender)
            _let_go(control_contender)
            if box.get('runner') is not None and box['runner']._stop_task is None:
                await asyncio.wait_for(box['runner'].stop(), 60)
            if process_ownership.owns(home):
                process_ownership.release(home)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_a_clean_startup_failure_still_closes_and_releases(tmp_path, caplog):
    """Control for STARTUP-01: startup fails with no writer live; the outer finally's stop closes the
    stores and its release frees the launch profile, as before."""
    from gateway.run_bootstrap import start_gateway
    from gateway.runtime_ownership import process_ownership
    caplog.set_level(logging.INFO)
    home = _launch_home()

    async def failed_start(runner, signals):
        raise RuntimeError('injected clean startup failure')
    contender = None
    with ExitStack() as stack:
        for p in _bootstrap_patches(failed_start):
            stack.enter_context(p)
        try:
            with pytest.raises(RuntimeError, match='injected clean startup failure'):
                await asyncio.wait_for(start_gateway(_config(), verbosity=None), 120)
            contender = _contender(home)
            observed = {'owned': process_ownership.owns(home), 'contender': _answer(contender),
                        'store_closed': 'SessionDB close done' in caplog.text}
            assert observed == {'owned': False, 'contender': 'ACQUIRED', 'store_closed': True}, observed
        finally:
            _let_go(contender)
            if process_ownership.owns(home):
                process_ownership.release(home)


@pytest.mark.platforms('linux')
def test_a_stop_cancelled_before_its_first_step_keeps_the_reservations(tmp_path, monkeypatch):
    """The bootstrap's inner ``await runner.stop()`` cancelled in the loop turn the stop task was
    created: the task is cancelled before ``_stop_impl`` runs a line. The outer finally's stop then
    re-raises the cancellation and its release must keep alpha (a room thread can still write)."""
    from gateway.run import GatewayRunner
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    began = []
    real_begin = GatewayRunner._stop_begin_teardown

    async def begin(self, ctx):
        began.append(True)
        return await real_begin(self, ctx)
    monkeypatch.setattr(GatewayRunner, '_stop_begin_teardown', begin)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        _, service = _alpha(runner, homes)
        release, thread = _stuck_room(service)
        contender = None
        try:
            async def cancelled_before_first_step():
                outer = asyncio.ensure_future(runner.stop())
                await asyncio.sleep(0)  # outer runs stop(): it creates the stop task and awaits it
                outer.cancel()  # ... which cancels the stop task before its first step
                try:
                    await outer
                except asyncio.CancelledError:
                    return 'CancelledError'
                return 'returned'
            observed = {'first': call(cancelled_before_first_step())}
            observed['stop_task_cancelled'] = runner._stop_task.cancelled()
            observed['stop_impl_began'] = bool(began)
            try:
                call(runner.stop())  # the outer finally's stop
                observed['second'] = None
            except BaseException as exc:  # the test reports it
                observed['second'] = type(exc).__name__
            release_gateway_runtime_lock()  # the outer finally's release
            observed.update(room_thread_alive=thread.is_alive(), still_owned=process_ownership.owns(homes['alpha']))
            contender = _contender(homes['alpha'])
            observed['contender'] = _answer(contender)
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['first'] == 'CancelledError' and observed['stop_task_cancelled'], observed
            assert not observed['stop_impl_began'], observed
            assert observed['second'] == 'CancelledError' and observed['room_thread_alive'], observed
            assert observed['still_owned'], ('a stop cancelled before it began released alpha', observed)
            assert observed['contender'] == 'REFUSED', observed
        finally:
            release.set()
            thread.join(5)
            _let_go(contender)
            with service.runtime._status_lock:
                service.runtime._room_threads.pop('stuck-room', None)


@pytest.mark.platforms('linux')
def test_a_token_writer_that_outlives_its_close_keeps_the_store_open_and_owned(tmp_path, monkeypatch, caplog):
    """TOKEN-01. The store's own token writer is mid-batch when the whole runtime stops and stays so
    past SessionDB's native 10 s join. The stop must neither close that connection (the writer would
    reopen it) nor let the release free the profile; the writer's late write lands while still owned."""
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    with _gateway(tmp_path, monkeypatch, served=()) as (runner, homes, call):
        home, db = homes['default'], runner.session_authority.db
        db.create_session('token-late-write', source='test')
        entered, release, wrote = threading.Event(), threading.Event(), threading.Event()
        probes = {}
        original = db._apply_token_batch

        def held(batch):
            entered.set()
            if not release.wait(180):
                raise RuntimeError('the test never released the token batch')
            probes['at_write'] = _probe(home)
            original(batch)
            wrote.set()
        monkeypatch.setattr(db, '_apply_token_batch', held)
        db.queue_token_counts('token-late-write', input_tokens=7)
        assert entered.wait(10)
        writer = db._token_writer_thread
        contender = None
        try:
            async def running_loop():
                return asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(runner.stop(), call(running_loop())).result(150)
            release_gateway_runtime_lock()
            observed = {'writer_live_after_stop': writer.is_alive(), 'connection_closed': db._conn is None,
                        'store_closed': 'SessionDB close done' in caplog.text,
                        'still_owned': process_ownership.owns(home), 'retention': process_ownership._retained_for}
            contender = _contender(home)
            observed['contender'] = _answer(contender)
            release.set()
            writer.join(60)
            with closing(sqlite3.connect(f'file:{db.db_path}?mode=ro', uri=True)) as ro:
                observed['persisted_input_tokens'] = ro.execute(
                    "SELECT input_tokens FROM sessions WHERE id='token-late-write'").fetchone()[0]
            observed.update(writer_completed=wrote.is_set(), probe_at_write=probes.get('at_write'),
                            reopened='reopening (teardown/worker race' in caplog.text, old_pid=os.getpid())
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['writer_live_after_stop'], observed
            assert not observed['connection_closed'], ('the stop closed the store under its live token writer', observed)
            assert not observed['store_closed'], observed
            assert observed['still_owned'], ('the release freed the profile under a live token writer', observed)
            assert observed['contender'] == 'REFUSED', observed
            assert observed['writer_completed'] and observed['persisted_input_tokens'] == 7, observed
            assert not observed['reopened'], observed
            assert observed['probe_at_write'] == {'held': True, 'record_pid': os.getpid()}, observed
        finally:
            release.set()
            writer.join(15)
            _let_go(contender)


@pytest.mark.platforms('linux')
def test_a_token_writer_that_settles_lets_the_stop_close_and_release(tmp_path, monkeypatch, caplog):
    """Control for TOKEN-01: a queued delta the writer applies in time; the stop closes and releases."""
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    with _gateway(tmp_path, monkeypatch, served=()) as (runner, homes, call):
        home, db = homes['default'], runner.session_authority.db
        db.create_session('token-settled', source='test')
        db.queue_token_counts('token-settled', input_tokens=5)
        call(runner.stop())
        release_gateway_runtime_lock()
        with closing(sqlite3.connect(f'file:{db.db_path}?mode=ro', uri=True)) as ro:
            persisted = ro.execute("SELECT input_tokens FROM sessions WHERE id='token-settled'").fetchone()[0]
        assert 'SessionDB close done' in caplog.text and db._conn is None
        assert not process_ownership.owns(home)
        assert persisted == 5


def _failed_capture(db):
    """SessionDB's own close failure: a lost WAL generation whose capture fails leaves the handle open
    (``hermes_state._settle_lost_generation_locked``); every production caller but the stop's direct
    close reaches it through a layer that swallows the exception (``hermes_state_registry``)."""
    from hermes_state_dbfile import RetiredGenerationCaptureError

    def capture_fails():
        raise RetiredGenerationCaptureError('injected: retired WAL generation not captured')
    db._db_wal_generation_lost = True
    db._settle_lost_generation_locked = capture_fails

    def undo():
        db.__dict__.pop('_settle_lost_generation_locked', None)
        db._db_wal_generation_lost = False
    return undo


@pytest.mark.platforms('linux')
def test_a_store_close_that_raises_keeps_the_profile_owned(tmp_path, monkeypatch, caplog):
    """CLOSE-QUAL-01 at the whole-runtime stop: the store's close raises, so the stop has not closed
    it; the release must keep the profile (no 'close done' attestation)."""
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    with _gateway(tmp_path, monkeypatch, served=()) as (runner, homes, call):
        home, db = homes['default'], runner.session_authority.db
        undo = _failed_capture(db)
        contender = None
        try:
            call(runner.stop())
            release_gateway_runtime_lock()
            observed = {'store_closed': 'SessionDB close done' in caplog.text, 'connection_open': db._conn is not None,
                        'still_owned': process_ownership.owns(home), 'retention': process_ownership._retained_for}
            contender = _contender(home)
            observed['contender'] = _answer(contender)
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['connection_open'], observed
            assert not observed['store_closed'], ('a close that raised was attested as closed', observed)
            assert observed['still_owned'], ('the release freed the profile after a failed close', observed)
            assert observed['contender'] == 'REFUSED', observed
        finally:
            undo()
            _let_go(contender)


@pytest.mark.platforms('linux')
def test_a_close_swallowed_by_a_quiet_layer_is_not_a_closed_store(tmp_path):
    """CLOSE-QUAL-01 at the verdict: a SessionDB close that raised inside ``release_or_close`` (which
    swallows it) keeps the reservation; once a later close of the same handle finishes, the verdict
    releases (control)."""
    from gateway.run_shutdown_budget import stores_closed
    from gateway.runtime_ownership import process_ownership
    from hermes_state import SessionDB
    from hermes_state_registry import release_or_close
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    process_ownership.reserve([home])
    db = SessionDB(db_path=home / 'state.db')
    undo = _failed_capture(db)
    try:
        release_or_close(db)  # swallowed: no exception reaches the stop
        stores_closed(0.0)
        process_ownership.close()  # the bootstrap's release
        failed = {'owned': process_ownership.owns(home), 'retention': process_ownership._retained_for}
        undo()
        db.close()  # this close finishes
        stores_closed(0.0)
        process_ownership.close()
        settled = {'owned': process_ownership.owns(home), 'connection_closed': db._conn is None}
        print('OBSERVATION', {'failed': failed, 'settled': settled})
        assert failed['owned'], ('a swallowed close failure was attested as closed', failed)
        assert settled == {'owned': False, 'connection_closed': True}, settled
    finally:
        undo()
        db.close()
        if process_ownership.owns(home):
            process_ownership.release(home)
