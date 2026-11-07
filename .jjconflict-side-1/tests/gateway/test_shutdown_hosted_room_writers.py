"""SH-01: the whole-runtime stop (``runner.stop()`` → ``_stop_impl``, then the bootstrap's
``release_gateway_runtime_lock``) against a profile's hosted-room writer.

Invariant: while a hosted-room thread that can still write a profile's store is alive, shutdown
neither closes that store's SessionDB handles nor releases the profile's ``gateway.lock``; and a
hosted-room Stop that raises does not abort the stop sequence (it is a failed Stop, not a reason to
skip settle, the DB gate and the exit state). Real gateway, real canonical room services, real
``HostedRoomRuntime.stop``; the only injection is a room thread that does not finish within its Stop
(a member turn still running) or a room Stop that raises."""
import asyncio
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.test_hosted_group_members import _gateway


def _stuck_room(service, write=None, wrote=None):
    """A room thread of *service*'s real runtime that outlives any Stop until released, then (with
    *write*) performs one real write to its profile's store, recording the outcome in *wrote*."""
    release = threading.Event()

    def room():
        release.wait(30)
        if write is not None:
            try:
                write()
                wrote['ok'] = True
            except Exception as exc:  # the test reports it
                wrote['ok'] = repr(exc)
    thread = threading.Thread(target=room, name='hosted-room-stuck', daemon=True)
    thread.start()
    runtime = service.runtime
    with runtime._status_lock:
        runtime._room_threads['stuck-room'] = thread
    return release, thread


def _alpha(runner, homes):
    authority = runner.session_authorities.for_home(homes['alpha'])
    assert authority is not None and authority.hosted_room_service is not None
    return authority, authority.hosted_room_service


_CONTENDER = (
    'import json, os, sys, pathlib\n'
    'from gateway.runtime_ownership import ProfileOwnership, OwnershipConflict\n'
    'own = ProfileOwnership()\n'  # kept: its flock lives as long as this object and this process
    'try:\n'
    '    own.reserve([pathlib.Path(sys.argv[1])])\n'
    '    print(json.dumps({"contender": "ACQUIRED", "pid": os.getpid()}), flush=True)\n'
    '    sys.stdin.read()\n'  # hold the profile until the test lets go
    'except OwnershipConflict:\n'
    '    print(json.dumps({"contender": "REFUSED", "pid": os.getpid()}), flush=True)\n')

# A third process, run while the old room thread writes: is alpha's gateway.lock held right now, and
# by whom (the pid the current holder wrote into the lock record when it reserved)?
_PROBE = (
    'import json, sys, pathlib\n'
    'from gateway.runtime_ownership import ProfileOwnership, OwnershipConflict\n'
    'from gateway.status import _read_gateway_lock_record\n'
    'home = pathlib.Path(sys.argv[1])\n'
    'record = _read_gateway_lock_record(home / "gateway.lock") or {}\n'
    'own = ProfileOwnership()\n'
    'try:\n'
    '    own.reserve([home])\n'
    '    held = False\n'
    '    own.close()\n'
    'except OwnershipConflict:\n'
    '    held = True\n'
    'print(json.dumps({"held": held, "record_pid": record.get("pid")}), flush=True)\n')


def _python(code, *args, stdin=None):
    root = Path(__file__).resolve().parents[2]
    return subprocess.Popen([sys.executable, '-c', code, *map(str, args)], cwd=root, text=True,
                            env={**os.environ, 'PYTHONPATH': str(root)},
                            stdin=stdin, stdout=subprocess.PIPE)


def _contender(home):
    """An independent process (a successor gateway) trying to own *home*; holds it if it can."""
    return _python(_CONTENDER, home, stdin=subprocess.PIPE)


def _probe(home):
    proc = _python(_PROBE, home)
    out, _ = proc.communicate(timeout=60)
    return json.loads(out)


def _row_written(db_path, sid):
    with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as db:
        return db.execute('SELECT count(*) FROM sessions WHERE id=?', (sid,)).fetchone()[0] == 1


@pytest.mark.platforms('linux')
def test_a_room_thread_that_outlives_its_stop_keeps_the_store_open_and_owned(tmp_path, monkeypatch, caplog):
    """A member turn still running in a room thread when the whole runtime stops: the stop and the
    bootstrap release must leave alpha's store open and its gateway.lock held, so no independent
    process can own alpha while that thread can still write it."""
    import json as _json
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        authority, service = _alpha(runner, homes)
        home, db_path = homes['alpha'], authority.db.db_path
        wrote, probes = {}, {}

        def late_write():
            probes['before'] = _probe(home)  # who holds alpha's flock as the old thread writes
            authority.db.create_session('late-room-write', source='test')
            probes['after'] = _probe(home)
        release, thread = _stuck_room(service, late_write, wrote)
        contender = None
        try:
            call(runner.stop())
            release_gateway_runtime_lock()  # the bootstrap's own release, after the stop
            observed = {'room_thread_alive': thread.is_alive(),
                        'store_closed': 'SessionDB close done' in caplog.text,
                        'still_owned': process_ownership.owns(home), 'old_pid': os.getpid()}
            contender = _contender(home)
            answer = _json.loads(contender.stdout.readline())
            observed.update(contender=answer['contender'], contender_pid=answer['pid'])
            release.set()  # the old room thread's late write runs now
            thread.join(60)
            observed.update(late_write=wrote.get('ok'), probes=dict(probes),
                            row_written=_row_written(db_path, 'late-room-write'),
                            contender_alive=contender.poll() is None)
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            # Assertions (the invariant), separate from the observation above.
            assert observed['room_thread_alive'], observed
            assert not observed['store_closed'], ('the store was closed under a live room writer', observed)
            assert observed['still_owned'], ("alpha's gateway.lock was released under a live room writer", observed)
            assert observed['contender'] == 'REFUSED', ('an independent process owned alpha under a live room writer', observed)
            assert probes['before']['held'] and probes['before']['record_pid'] == observed['old_pid'], observed
        finally:
            release.set()
            thread.join(5)
            if contender is not None:
                if contender.stdin:
                    contender.stdin.close()
                contender.wait(10)
                contender.stdout.close()
            with service.runtime._status_lock:
                service.runtime._room_threads.pop('stuck-room', None)


def _spy_phases(monkeypatch):
    """Record which stop phases run, calling the real ones."""
    from gateway import run_runtime
    from gateway.run import GatewayRunner
    ran = []
    settle = run_runtime.settle_gateway_runtime

    async def settled(*args, **kwargs):
        ran.append('settle')
        return await settle(*args, **kwargs)
    monkeypatch.setattr(run_runtime, 'settle_gateway_runtime', settled)
    for name in ('_stop_finalize_agents_and_adapters', '_stop_quiesce_and_close_session_dbs',
                 '_stop_persist_exit_state'):
        real = getattr(GatewayRunner, name)
        if asyncio.iscoroutinefunction(real):
            async def phase(self, *args, _real=real, _name=name, **kwargs):
                ran.append(_name)
                return await _real(self, *args, **kwargs)
        else:
            def phase(self, *args, _real=real, _name=name, **kwargs):
                ran.append(_name)
                return _real(self, *args, **kwargs)
        monkeypatch.setattr(GatewayRunner, name, phase)
    return ran


@pytest.mark.platforms('linux')
def test_a_room_stop_that_raises_does_not_abort_the_stop_sequence(tmp_path, monkeypatch, caplog):
    """alpha's room Stop raises: the stop still runs settle, finalize, the DB gate and the exit
    state, beta's rooms are still stopped, and alpha's (unconfirmed) store stays open and owned."""
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    ran = _spy_phases(monkeypatch)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        _, alpha = _alpha(runner, homes)
        beta = runner.session_authorities.for_home(homes['beta']).hosted_room_service
        real_stop = alpha.stop

        def failing_stop(*, timeout=5.0):
            raise RuntimeError('hosted room stop failed')
        alpha.stop = failing_stop
        contender = None
        try:
            outcome = None
            try:
                call(runner.stop())
            except Exception as exc:  # the test reports it
                outcome = repr(exc)
            release_gateway_runtime_lock()
            observed = {'stop_raised': outcome, 'phases': list(ran),
                        'beta_stopped': beta.runtime._stop.is_set() and not beta.live_threads(),
                        'store_closed': 'SessionDB close done' in caplog.text,
                        'still_owned': process_ownership.owns(homes['alpha'])}
            contender = _contender(homes['alpha'])
            observed['contender'] = json.loads(contender.stdout.readline())['contender']
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['stop_raised'] is None, observed
            assert observed['phases'] == ['settle', '_stop_finalize_agents_and_adapters',
                                          '_stop_quiesce_and_close_session_dbs', '_stop_persist_exit_state'], observed
            assert observed['beta_stopped'], observed
            assert not observed['store_closed'] and observed['still_owned'], observed
            assert observed['contender'] == 'REFUSED', observed
        finally:
            alpha.stop = real_stop
            if contender is not None:
                contender.stdin.close()
                contender.wait(10)
                contender.stdout.close()


@pytest.mark.platforms('linux')
def test_a_cancelled_drain_propagates_from_a_room_stop(tmp_path, monkeypatch):
    """Cancellation is not a failed Stop: it propagates out of the hosted-room stop unchanged."""
    from gateway.run_runtime import drain_gateway_runtime
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        _, alpha = _alpha(runner, homes)
        real_stop, entered, release = alpha.stop, threading.Event(), threading.Event()

        def slow_stop(*, timeout=5.0):
            entered.set()
            release.wait(10)
            return real_stop(timeout=timeout)
        alpha.stop = slow_stop

        async def cancelled():
            task = asyncio.ensure_future(drain_gateway_runtime(runner))
            await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return 'cancelled'
            return 'completed'
        try:
            assert call(cancelled()) == 'cancelled'
            assert not getattr(runner, '_hosted_rooms_unconfirmed', False)  # never taken for a failed Stop
        finally:
            release.set()
            alpha.stop = real_stop


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('writer', ['executor', 'final_sweep'])
def test_any_live_writer_that_skips_the_close_also_keeps_the_reservations(tmp_path, monkeypatch, writer):
    """A4r6 C2: the DB gate's skip-close verdict reaches the release for the other counted writer
    kinds too (tested: a live executor worker, a live final tool sweep; both are counted only at the
    gate, so the drain does not wait on them)."""
    from gateway.run import GatewayRunner
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    if writer == 'executor':
        monkeypatch.setattr(GatewayRunner, '_shutdown_executor', lambda self, drain_timeout: 1)
    else:
        # A live final tool sweep, wherever the stop's close gate reads it from.
        from gateway import run_shutdown, run_shutdown_budget
        monkeypatch.setattr(run_shutdown_budget, 'final_sweep_live', lambda runner: 1)
        if hasattr(run_shutdown, 'final_sweep_live'):
            monkeypatch.setattr(run_shutdown, 'final_sweep_live', lambda runner: 1)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        contender = None
        try:
            call(runner.stop())
            release_gateway_runtime_lock()
            assert process_ownership.owns(homes['alpha'])
            contender = _contender(homes['alpha'])
            assert json.loads(contender.stdout.readline())['contender'] == 'REFUSED'
        finally:
            if contender is not None:
                contender.stdin.close()
                contender.wait(10)
                contender.stdout.close()



@pytest.mark.platforms('linux')
@pytest.mark.parametrize('failure', ['raises', 'times_out'])
def test_a_stop_that_does_not_finish_keeps_the_reservations(tmp_path, monkeypatch, caplog, failure):
    """The exception and outer-finally paths: a stop phase raises, or the caller's bounded wait on the
    stop expires and cancels it, before the DB gate. The bootstrap's outer finally then stops again
    (the failed stop task re-raises) and releases. With a room thread still able to write alpha's open
    store, that release must keep alpha's gateway.lock: only a stop that closed the stores releases."""
    from gateway.run import GatewayRunner
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)

    async def finalize(self, ctx):
        if failure == 'raises':
            raise RuntimeError('finalize failed')
        await asyncio.Event().wait()  # never set: hangs until the caller's bounded wait cancels the stop
    monkeypatch.setattr(GatewayRunner, '_stop_finalize_agents_and_adapters', finalize)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        _, service = _alpha(runner, homes)
        release, thread = _stuck_room(service)
        contender = None
        try:
            outcomes = []
            first = runner.stop if failure == 'raises' else (lambda: asyncio.wait_for(runner.stop(), 8))
            for stop in (first, runner.stop):  # the stop, then the outer finally's stop
                try:
                    call(stop())
                    outcomes.append(None)
                except BaseException as exc:  # the test reports it
                    outcomes.append(type(exc).__name__)
            release_gateway_runtime_lock()  # the bootstrap's outer finally
            observed = {'failure': failure, 'stop_outcomes': outcomes, 'room_thread_alive': thread.is_alive(),
                        'store_closed': 'SessionDB close done' in caplog.text,
                        'still_owned': process_ownership.owns(homes['alpha'])}
            contender = _contender(homes['alpha'])
            observed['contender'] = json.loads(contender.stdout.readline())['contender']
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert None not in observed['stop_outcomes'], observed  # neither stop finished
            assert observed['room_thread_alive'] and not observed['store_closed'], observed
            assert observed['still_owned'], ("alpha's gateway.lock was released after a stop that did not finish", observed)
            assert observed['contender'] == 'REFUSED', observed
        finally:
            release.set()
            thread.join(5)
            if contender is not None:
                contender.stdin.close()
                contender.wait(10)
                contender.stdout.close()
            with service.runtime._status_lock:
                service.runtime._room_threads.pop('stuck-room', None)

_OLD_OWNER = (
    'import sys, pathlib\n'
    'from gateway.runtime_ownership import ProfileOwnership\n'
    'own = ProfileOwnership()\n'
    'own.reserve([pathlib.Path(sys.argv[1])])\n'
    'own.retain_until_exit("live writer at the SessionDB close")\n'
    'own.close()\n'  # the bootstrap's release: retained
    'print("RETAINED" if own.owns(pathlib.Path(sys.argv[1])) else "RELEASED", flush=True)\n'
    'sys.stdin.read()\n')


@pytest.mark.platforms('linux')
def test_a_retained_reservation_is_the_kernel_lock_and_ends_with_the_process(tmp_path):
    """What is retained is the flock on gateway.lock: refused while the old owner lives, free once
    that process has actually exited."""
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    old = subprocess.Popen([sys.executable, '-c', _OLD_OWNER, str(home)], cwd=root, text=True,
                           env={**os.environ, 'PYTHONPATH': str(root)}, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert old.stdout.readline().strip() == 'RETAINED'
        refused = _contender(home)
        assert json.loads(refused.stdout.readline())['contender'] == 'REFUSED'
        refused.wait(10)
        old.stdin.close()
        assert old.wait(10) == 0
        acquired = _contender(home)
        assert json.loads(acquired.stdout.readline())['contender'] == 'ACQUIRED'
        acquired.stdin.close()
        acquired.wait(10)
        for proc in (refused, acquired):
            proc.stdout.close()
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(10)
        old.stdout.close()


@pytest.mark.platforms('linux')
def test_rooms_that_stop_still_let_shutdown_close_and_release(tmp_path, monkeypatch, caplog):
    """Control: nothing outlives its Stop, so the close and the release happen as before."""
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    caplog.set_level(logging.INFO)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        call(runner.stop())
        release_gateway_runtime_lock()
        assert 'SessionDB close done' in caplog.text
        assert not process_ownership.owns(homes['alpha'])
