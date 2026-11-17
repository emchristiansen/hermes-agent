"""SH-01 TOKEN-RELEASE-02: a secondary profile's release follows its stores' close verdict.

Invariant (as for the whole stop in ``test_shutdown_close_verdict``): while a thread that can still write
a profile's store is alive, that profile's ``gateway.lock`` is not released, so no independent process
can own the store. Hot unserve retires the profile's authority and the registry closes its shared
SessionDB; a token writer still live past that close's native 10 s join leaves the connection open, and
the per-profile release must then keep the profile reserved (until a later close of that store
finishes, or process exit) instead of releasing it.

Real: the initialized multi-profile gateway (``_gateway``), authority retirement and removal, the
handle caches and registry teardown, the SessionDB token queue and writer thread with its native join,
``release_profile_home`` and an independent contender process. Substituted: the memory provider's
handle release, scoped MCP shutdown and the profile log-handler release (external effects), and the
token batch is held on an event before the original apply runs.

Also: a bare runner (``object.__new__``, no handle cache) whose stop closed its only store still
releases: an absent cache means this runner opened no handle, not a failed close.
"""
import asyncio
from contextlib import ExitStack
import json
import os
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.gateway.test_hosted_group_members import _gateway
from tests.gateway.test_shutdown_hosted_room_writers import _contender, _probe


def _answer(contender):
    return json.loads(contender.stdout.readline())


def _let_go(contender):
    if contender is not None:
        if contender.stdin:
            contender.stdin.close()
        contender.wait(15)
        contender.stdout.close()


def _external_teardown():
    """External effects of unserve (memory provider, scoped MCP servers, log handlers) are excluded."""
    provider = SimpleNamespace(MemoryStore=SimpleNamespace(release_all_under=lambda home: None))
    stack = ExitStack()
    stack.enter_context(patch('plugins.memory.import_provider_module', return_value=provider))
    stack.enter_context(patch('tools.mcp_tool_lifecycle.shutdown_mcp_servers'))
    stack.enter_context(patch('hermes_logging.release_profile_log_handlers'))
    return stack


def _unfinished(home):
    from hermes_state import unsettled_closes
    return [p for p in unsettled_closes() if p.startswith(str(home) + os.sep)]


@pytest.mark.platforms('linux')
def test_a_token_writer_that_outlives_unserve_keeps_the_profile_owned(tmp_path, monkeypatch):
    """alpha's token writer is mid-apply when alpha is unserved. The registry's close waits its native
    join and leaves the connection open; unserve must answer False and keep alpha's gateway.lock, so an
    independent contender is refused and the late write commits under this process's lock. beta, with
    no live writer, is unserved and released meanwhile. Once the writer settled, a close of alpha's store
    that finishes lets the same release go ahead (contender ACQUIRED)."""
    from gateway.run_runtime import release_profile_home
    from gateway.runtime_ownership import process_ownership
    import hermes_state_registry as registry
    observed, contenders = {}, []
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home, beta = homes['alpha'], homes['beta']
        authority = runner.session_authorities.for_home(home)
        db = authority.db
        observed['registry_shared'] = db._shared_registry_owned
        db.create_session('late-tokens', source='test')
        entered, release = threading.Event(), threading.Event()
        original, probes = db._apply_token_batch, {}

        def held(batch):
            entered.set()
            release.wait(120)
            probes['at_write'] = _probe(home)  # who holds alpha's flock as the old writer commits
            original(batch)
        db._apply_token_batch = held
        db.queue_token_counts('late-tokens', input_tokens=7)
        assert entered.wait(15)
        writer = db._token_writer_thread

        async def loop():
            return asyncio.get_running_loop()
        loop = call(loop())

        def on_loop(coroutine, timeout=120):
            return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout)
        try:
            with _external_teardown():
                observed['unserve'] = on_loop(runner._unserve_profile('alpha', home))
            observed.update(
                authority_removed=runner.session_authorities.for_home(home) is None,
                served=sorted(n for n, _h in runner.config._runtime_profile_homes), writer_alive=writer.is_alive(),
                conn_open=db._conn is not None, owned=process_ownership.owns(home), unfinished=_unfinished(home))
            contenders.append(_contender(home))
            observed['contender'] = _answer(contenders[-1])['contender']
            # Registry reacquire of alpha's path in this process while the old writer is live (alpha still
            # owned by this process): recorded, not changed by this repair.
            replacement = registry.acquire(db.db_path)
            observed['reacquire_new_generation'] = replacement is not db and replacement._conn is not None
            registry.release(replacement)
            # Unrelated-profile progress: beta (no live writer) is unserved and released meanwhile.
            with _external_teardown():
                observed['beta_unserve'] = on_loop(runner._unserve_profile('beta', beta))
            contenders.append(_contender(beta))
            observed['beta'] = {'owned': process_ownership.owns(beta), 'contender': _answer(contenders[-1])['contender']}
            # The old writer settles: its late write commits under this process's lock.
            release.set()
            writer.join(30)
            observed.update(writer_finished=not writer.is_alive(), probe_at_write=probes.get('at_write'))
            with sqlite3.connect(f'file:{db.db_path}?mode=ro', uri=True) as ro:
                observed['tokens'] = ro.execute("SELECT input_tokens FROM sessions WHERE id='late-tokens'").fetchone()[0]
            observed['still_owned_after_write'] = process_ownership.owns(home)
            _let_go(contenders.pop(0))
            # Eventual release: a later close of alpha's store finishes, then the same release goes ahead.
            db._apply_token_batch = original
            db.close()
            observed['unfinished_after_close'] = _unfinished(home)
            observed['late_release'] = release_profile_home(runner, home)
            contenders.append(_contender(home))
            observed['late'] = {'owned': process_ownership.owns(home), 'contender': _answer(contenders[-1])['contender']}
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))  # retained basetemp
            assert observed['registry_shared'] and observed['authority_removed'], observed
            assert 'alpha' not in observed['served'] and 'beta' in observed['served'], observed  # withdrawn
            assert observed['writer_alive'] and observed['conn_open'], observed
            assert observed['unserve'] is False, ('unserve reported alpha released under its live token writer', observed)
            assert observed['owned'] and observed['contender'] == 'REFUSED', observed
            assert observed['unfinished'] == [str(db.db_path)], observed
            assert observed['writer_finished'] and observed['tokens'] == 7, observed
            assert observed['probe_at_write'] == {'held': True, 'record_pid': os.getpid()}, observed
            assert observed['still_owned_after_write'], observed
            assert observed['beta_unserve'] is True and observed['beta'] == {'owned': False, 'contender': 'ACQUIRED'}, observed
            assert observed['unfinished_after_close'] == [], observed
            assert observed['late_release'] is True, observed
            assert observed['late'] == {'owned': False, 'contender': 'ACQUIRED'}, observed
        finally:
            release.set()
            writer.join(30)
            db._apply_token_batch = original
            db.close()
            for contender in contenders:
                _let_go(contender)
            for profile in (home, beta):
                if process_ownership.owns(profile):
                    process_ownership.release(profile)


@pytest.mark.platforms('linux')
def test_unserve_without_a_live_writer_still_releases(tmp_path, monkeypatch):
    """Control: the same unserve with alpha's token queue idle answers True and frees alpha."""
    from gateway.runtime_ownership import process_ownership
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        home = homes['alpha']
        db = runner.session_authorities.for_home(home).db
        db.create_session('settled-tokens', source='test')
        db.queue_token_counts('settled-tokens', input_tokens=5)
        db.flush_token_counts()
        contender = None
        try:
            with _external_teardown():
                unserved = call(runner._unserve_profile('alpha', home))
            contender = _contender(home)
            observed = {'unserve': unserved, 'owned': process_ownership.owns(home),
                        'contender': _answer(contender)['contender'], 'unfinished': _unfinished(home),
                        'conn_open': db._conn is not None}
            print('OBSERVATION', observed)
            assert observed == {'unserve': True, 'owned': False, 'contender': 'ACQUIRED', 'unfinished': [],
                                'conn_open': False}, observed
        finally:
            _let_go(contender)
            if process_ownership.owns(home):
                process_ownership.release(home)


@pytest.mark.asyncio
async def test_a_bare_runner_without_a_handle_cache_still_closes_and_releases(tmp_path, monkeypatch):
    """A runner built with ``object.__new__`` never opened a per-profile handle (the cache is created on
    first open): its stop's verdict must not count the missing cache as a failed close."""
    from gateway import runtime_ownership as ro
    from gateway.status import release_gateway_runtime_lock
    from hermes_state import SessionDB
    from tests.gateway.restart_test_helpers import make_restart_runner
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    own = ro.ProfileOwnership()
    monkeypatch.setattr(ro, 'process_ownership', own)
    home = tmp_path / 'home'
    own.reserve([home])
    db = SessionDB(home / 'state.db')
    runner, _ = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    runner._session_db = db
    observed = {'has_cache': hasattr(runner, '_session_db_handle_cache')}
    with patch('tools.process_registry.process_registry.kill_all', return_value=0), \
            patch('cron.scheduler.mark_running_jobs_interrupted', return_value=[]), \
            patch('tools.terminal_tool_lifecycle.cleanup_all_environments'), \
            patch('tools.browser_session_lifecycle.cleanup_all_browsers'), \
            patch('gateway.status.remove_pid_file'), patch('gateway.status.publish_runtime_status'):
        await runner.stop()
    observed.update(db_closed=db._conn is None, retention=own._retained_for)
    release_gateway_runtime_lock()
    contender = await asyncio.to_thread(_contender, home)
    try:
        observed.update(owned=own.owns(home), contender=_answer(contender)['contender'])
    finally:
        _let_go(contender)
        own.close()
    print('OBSERVATION', observed)
    assert observed == {'has_cache': False, 'db_closed': True, 'retention': None, 'owned': False,
                        'contender': 'ACQUIRED'}, observed
