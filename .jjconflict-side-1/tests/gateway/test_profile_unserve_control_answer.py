"""A hot ``unserve-profile`` whose ownership release was refused is never answered as unserved.

``_unserve_profile`` answers False when it kept the profile's ``gateway.lock`` (SH-01 TOKEN-RELEASE-02 and the
older retained-retirement verdict): a store close under the home did not finish because a writer of that store
is still live, or retirement could not prove the profile's turns stopped. The control-socket verb
(``_profile_lifecycle_verb``) ignored that answer and replied ``{"unserved": name}``; the CLI accepts exactly
that as "stopped" (``ack_confirmed``), prints that the profile's bots and cron are stopped and, for
``restart``, goes on to hot-serve the profile again while this process still holds a live writer into it.

Real here: the initialized multi-profile gateway (``initialize_gateway_runtime``: one authority and registry
SessionDB per home), the PRODUCTION control listener with its verb table
(``run_bootstrap._start_gateway_start_control_socket``), the client over the real Unix socket
(``request_unserve_profile`` / ``request_serve_profile_hot``), the CLI consumer ``profile_lifecycle``,
``_unserve_profile`` with authority retirement, the registry close with the native token-writer join,
``release_profile_home``, ``process_ownership`` and an independent contender process.

Substituted: the external effects of unserve (memory-provider handle release, scoped MCP shutdown, log handlers;
as in ``test_profile_store_release``), the CLI's host discovery (which gateway serves the profile, live gateway
pids, the live-work policy check), the token batch apply is held on an event before the original apply runs,
and, in the store-close case, the token writer's join is shortened from its native 10 s to 1 s so the verb's
5 s answer wait sees the completed unserve. The native 10 s join is kept in the never-confirmed control. The turn
case runs an in-process agent that ignores its Stop through the drain's execution seam with a 1 s settle.
"""
import asyncio
from contextlib import contextmanager
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_profile_store_release import _answer, _external_teardown, _let_go, _unfinished
from tests.gateway.test_session_authorities_multiplex import _reserve_homes
from tests.gateway.test_shutdown_hosted_room_writers import _contender

SHORT_JOIN = 1.0


@contextmanager
def _host(tmp_path, monkeypatch, *, served):
    """A running multiplexer over ``default`` + ``served`` as ``start_gateway`` leaves it: reserved homes,
    initialized runtime, served set recorded, ``_running``, and the production control listener."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_bootstrap import _start_gateway_start_control_socket
    from gateway.run_runtime import initialize_gateway_runtime, recover_gateway_native_sessions
    from gateway.runtime_ownership import process_ownership

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=served)
    homes = dict(homes)
    for home in homes.values():
        (home / 'config.yaml').write_text(json.dumps({'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}}))
    reserved = [(name, homes[name]) for name in ('default', *served)]
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(reserved)
    process_ownership.reserve([home for _, home in reserved])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    runner = GatewayRunner(config)
    server = None

    def call(coroutine, timeout=60):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout)
    try:
        call(initialize_gateway_runtime(runner))
        runner.session_runtime_descriptor.update(state='ready', capabilities=['session-authority-v1'],
                                                 api_origin='http://127.0.0.1:1', supervisor='none')
        server = call(_start_gateway_start_control_socket(runner))
        assert server is not None and runner.session_control_server is server
        call(recover_gateway_native_sessions(runner))
        runner._running = True
        runner._record_served_profiles('default', reserved)
        yield runner, root, homes, call
    finally:
        runner._running = False
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        if server is not None:
            call(server.stop())
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        assert not thread.is_alive()
        loop.close()
        for _, home in reserved:
            if process_ownership.owns(home):
                process_ownership.release(home)


def _cli_as(monkeypatch, name, home, host_root):
    """``hermes -p <name> gateway ...`` with the host multiplexer at *host_root* serving it (discovery only)."""
    from hermes_cli import gateway as gw, gateway_lifecycle_guards as guards, gateway_profile_lifecycle as lifecycle
    monkeypatch.setattr(gw, '_current_profile_name', lambda: name)
    monkeypatch.setattr(lifecycle, 'get_hermes_home', lambda: home)
    monkeypatch.setattr(gw, 'find_gateway_pids', lambda *a, **k: [])
    monkeypatch.setattr(gw, '_served_by_another_host_gateway', lambda *a, **k: SimpleNamespace(home=host_root))
    monkeypatch.setattr(guards, 'refuse_while_sessions_live', lambda *a, **k: None)


def _record_requests(monkeypatch):
    """Every lifecycle request the CLI sends, with the host's answer (the real client and socket)."""
    from gateway import control_socket
    real_unserve, real_serve, requests = control_socket.request_unserve_profile, control_socket.request_serve_profile_hot, []

    def unserve(home, name):
        answer = real_unserve(home, name)
        requests.append(('unserve-profile', name, answer))
        return answer

    def serve(home, name):
        answer = real_serve(home, name)
        requests.append(('serve-profile', name, answer))
        return answer
    monkeypatch.setattr(control_socket, 'request_unserve_profile', unserve)
    monkeypatch.setattr(control_socket, 'request_serve_profile_hot', serve)
    return requests


def _hold_token_writer(db, sid, tokens):
    """alpha's token writer thread enters its batch apply and waits there (a live writer of alpha's store)."""
    db.create_session(sid, source='test')
    entered, release, original = threading.Event(), threading.Event(), db._apply_token_batch

    def held(batch):
        entered.set()
        release.wait(120)
        original(batch)
    db._apply_token_batch = held
    db.queue_token_counts(sid, input_tokens=tokens)
    assert entered.wait(15)
    return release, original, db._token_writer_thread


def _settled(runner, call):
    """Wait for any in-flight lifecycle verb: it holds the reconcile lock across unserve + bookkeeping."""
    async def locked():
        async with runner._reconcile_lock():
            return True
    return call(locked(), timeout=120)


@pytest.mark.platforms('linux')
def test_a_store_close_that_did_not_finish_is_answered_retained_and_restart_stops(tmp_path, monkeypatch, capsys):
    """alpha's store has a live token writer when ``hermes -p alpha gateway restart`` asks the host to unserve
    it. The answer must be an explicit retained error (cause: the store close), the CLI must not report a
    stop or send serve-profile, and alpha's lock must stay held (contender REFUSED). Unrelated beta, with no
    live writer, is stopped through the same CLI and socket meanwhile (unserved, ACQUIRED). A repeat of the
    verb finds alpha no longer served. Once the writer settled and a close of alpha's store finished, the same
    release goes ahead (ACQUIRED)."""
    from gateway import control_socket
    from gateway.run_runtime import release_profile_home
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import gateway_profile_lifecycle as lifecycle
    from hermes_state import SessionDB
    native_stop = SessionDB._stop_token_writer
    monkeypatch.setattr(SessionDB, '_stop_token_writer',
                        lambda self, join_timeout=10.0: native_stop(self, join_timeout=min(join_timeout, SHORT_JOIN)))
    observed, contenders = {}, []
    with _host(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, root, homes, _call):
        home, beta = homes['alpha'], homes['beta']
        db = runner.session_authorities.for_home(home).db
        release, original, writer = _hold_token_writer(db, 'late-tokens', 7)
        requests = _record_requests(monkeypatch)
        try:
            _cli_as(monkeypatch, 'alpha', home, root)
            with _external_teardown():
                observed['alpha_handled'] = lifecycle.profile_lifecycle('restart', SimpleNamespace(all=False, force=False))
            observed['alpha_cli'] = capsys.readouterr().out
            observed['alpha_requests'] = list(requests)
            answer = requests[0][2] if requests else None
            observed['alpha_confirmed'] = lifecycle.ack_confirmed(answer, 'unserved', 'alpha')
            observed.update(
                authority_removed=runner.session_authorities.for_home(home) is None,
                served=runner.served_profile_names(), writer_alive=writer.is_alive(), conn_open=db._conn is not None,
                owned=process_ownership.owns(home), unfinished=_unfinished(home))
            contenders.append(_contender(home))
            observed['contender'] = _answer(contenders[-1])['contender']

            del requests[:]
            _cli_as(monkeypatch, 'beta', beta, root)
            with _external_teardown():
                observed['beta_handled'] = lifecycle.profile_lifecycle('stop', SimpleNamespace(all=False, force=False))
            observed['beta_cli'] = capsys.readouterr().out
            observed['beta_requests'] = list(requests)
            contenders.append(_contender(beta))
            observed['beta'] = {'owned': process_ownership.owns(beta), 'contender': _answer(contenders[-1])['contender']}

            observed['repeat'] = control_socket.query_gateway_control(root, 'unserve-profile', params={'name': 'alpha'})

            release.set()
            writer.join(30)
            with sqlite3.connect(f'file:{db.db_path}?mode=ro', uri=True) as ro:
                observed['tokens'] = ro.execute("SELECT input_tokens FROM sessions WHERE id='late-tokens'").fetchone()[0]
            observed['still_owned_after_write'] = process_ownership.owns(home)
            _let_go(contenders.pop(0))
            db._apply_token_batch = original
            db.close()
            observed['unfinished_after_close'] = _unfinished(home)
            observed['late_release'] = release_profile_home(runner, home)
            contenders.append(_contender(home))
            observed['late'] = {'owned': process_ownership.owns(home), 'contender': _answer(contenders[-1])['contender']}
            print('OBSERVATION', observed)
            (tmp_path / 'observation.json').write_text(json.dumps(observed, default=str))

            assert observed['alpha_handled'] is True, observed
            assert [r[:2] for r in observed['alpha_requests']] == [('unserve-profile', 'alpha')], \
                ('the CLI went on after an unconfirmed unserve', observed)
            assert isinstance(answer, dict) and 'unserved' not in answer, ('alpha answered unserved', observed)
            assert 'store close' in answer.get('error', '') and 'ownership is retained' in answer['error'], observed
            assert 'turns are not proven stopped' not in answer['error'], observed
            assert answer['served_profiles'] == ['default', 'beta'] == observed['served'], observed
            assert observed['alpha_confirmed'] is False, observed
            assert 'restart was not confirmed' in observed['alpha_cli'], observed
            assert 'restarted' not in observed['alpha_cli'], observed
            assert observed['authority_removed'] and observed['writer_alive'] and observed['conn_open'], observed
            assert observed['owned'] and observed['contender'] == 'REFUSED', observed
            assert observed['unfinished'] == [str(db.db_path)], observed
            assert observed['beta_handled'] is True, observed
            assert observed['beta_requests'] == [('unserve-profile', 'beta',
                                                  {'unserved': 'beta', 'served_profiles': ['default']})], observed
            assert 'its bots and cron are stopped' in observed['beta_cli'], observed
            assert observed['beta'] == {'owned': False, 'contender': 'ACQUIRED'}, observed
            assert observed['repeat'] == {'error': "profile 'alpha' is not served"}, observed
            assert observed['tokens'] == 7 and observed['still_owned_after_write'], observed
            assert observed['unfinished_after_close'] == [] and observed['late_release'] is True, observed
            assert observed['late'] == {'owned': False, 'contender': 'ACQUIRED'}, observed
        finally:
            release.set()
            writer.join(30)
            db._apply_token_batch = original
            for contender in contenders:
                _let_go(contender)


@pytest.mark.platforms('linux')
def test_an_unserve_held_at_the_native_join_is_never_confirmed_then_retained(tmp_path, monkeypatch):
    """Control at the native 10 s token-writer join: the request is never confirmed. Observed: that join runs
    in the registry close inside ``_unserve_profile`` ON the event loop, so the listener cannot even write the
    verb's 5 s ``pending`` answer before the client's 8 s timeout (the client gets None, "did not answer");
    either non-confirmation is accepted and recorded. The unserve then finishes with alpha withdrawn but still
    owned (contender REFUSED), and a repeat finds alpha no longer served."""
    from gateway.control_socket import request_unserve_profile
    from gateway.runtime_ownership import process_ownership
    from hermes_cli.gateway_profile_lifecycle import ack_confirmed, ack_failure
    observed, contenders = {}, []
    with _host(tmp_path, monkeypatch, served=('alpha',)) as (runner, root, homes, call):
        home = homes['alpha']
        db = runner.session_authorities.for_home(home).db
        release, original, writer = _hold_token_writer(db, 'late-tokens', 3)
        try:
            with _external_teardown():
                started = time.monotonic()
                answer = request_unserve_profile(root, 'alpha')
                observed.update(answer=answer, answered_after=round(time.monotonic() - started, 2),
                                confirmed=ack_confirmed(answer, 'unserved', 'alpha'), failure=ack_failure(answer))
                observed['settled'] = _settled(runner, call)
            observed.update(served=runner.served_profile_names(), owned=process_ownership.owns(home),
                            unfinished=_unfinished(home), authority_removed=runner.session_authorities.for_home(home) is None)
            contenders.append(_contender(home))
            observed['contender'] = _answer(contenders[-1])['contender']
            observed['repeat'] = request_unserve_profile(root, 'alpha')
            print('OBSERVATION', observed)
            assert observed['answer'] is None or observed['answer'] == {
                'pending': True, 'served_profiles': ['default', 'alpha']}, observed
            assert observed['confirmed'] is False, observed
            assert observed['failure'] in ('host operation is still pending', 'host control socket did not answer'), observed
            assert observed['settled'] and observed['served'] == ['default'], observed
            assert observed['authority_removed'] and observed['unfinished'] == [str(db.db_path)], observed
            assert observed['owned'] and observed['contender'] == 'REFUSED', observed
            assert observed['repeat'] == {'error': "profile 'alpha' is not served"}, observed
        finally:
            release.set()
            writer.join(30)
            db._apply_token_batch = original
            for contender in contenders:
                _let_go(contender)


class _DeafAgent:
    """An in-process agent that ignores its Stop: its turn outlives the retirement's settle deadline."""

    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()

    def hard_interrupt(self, message=None, *, tool_reason=None):
        return None

    def run(self):
        self.started.set()
        assert self.release.wait(60)
        return {'final_response': '', 'interrupted': True, 'messages': []}


@pytest.mark.platforms('linux')
def test_a_turn_that_outlived_its_stop_is_answered_retained_not_unserved(tmp_path, monkeypatch):
    """The retirement branch: alpha's running turn ignores the Stop past the (1 s) settle deadline, so
    ``_unserve_profile`` keeps alpha's authority, store and lock. The verb must answer that retention (cause:
    turns), never unserved; alpha stays owned (contender REFUSED) and its retiring authority registered."""
    from gateway import run_runtime, session_finite
    from gateway.control_socket import request_unserve_profile
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authority import LiveSession
    from gateway.session_contract import Principal, SessionRef, Submission
    from hermes_cli.gateway_profile_lifecycle import ack_confirmed
    agent = _DeafAgent()

    async def execute(authority, ref, row):
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        result = await asyncio.get_running_loop().run_in_executor(None, agent.run)
        authority.pending_results[row['admission_id']] = {'result': result, 'usage': {}}
        return ''
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 1.0)
    observed, contender, authority = {}, None, None
    with _host(tmp_path, monkeypatch, served=('alpha',)) as (runner, root, homes, call):
        home = homes['alpha']
        authority = runner.session_authorities.for_home(home)
        authority.db.create_session('deaf', source='test')
        authority.sessions['deaf'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route-deaf')
        actor = Principal('human', authority.profile_id, frozenset({'session:submit', 'session:control'}), 'cli')
        try:
            call(authority.submit(actor, Submission(request_id='deaf-1', ref=SessionRef(authority.profile_id, 'deaf'),
                                                    payload={'text': 'go'}, intent='queue')))
            assert agent.started.wait(15)
            with _external_teardown():
                answer = request_unserve_profile(root, 'alpha')
                observed['settled'] = _settled(runner, call)
            observed.update(answer=answer, confirmed=ack_confirmed(answer, 'unserved', 'alpha'),
                            served=runner.served_profile_names(), owned=process_ownership.owns(home),
                            still_registered=runner.session_authorities.for_home(home) is authority,
                            retiring=bool(getattr(authority, 'retiring', False)))
            contender = _contender(home)
            observed['contender'] = _answer(contender)['contender']
            print('OBSERVATION', observed)
            assert isinstance(answer, dict) and 'unserved' not in answer, ('alpha answered unserved', observed)
            assert 'turns are not proven stopped' in answer.get('error', ''), observed
            assert 'ownership is retained' in answer['error'] and 'store close' not in answer['error'], observed
            assert answer['served_profiles'] == ['default'] == observed['served'], observed
            assert observed['confirmed'] is False, observed
            assert observed['still_registered'] and observed['retiring'], observed
            assert observed['owned'] and observed['contender'] == 'REFUSED', observed
        finally:
            agent.release.set()
            _let_go(contender)
            live = authority.sessions.get('deaf') if authority is not None else None
            if live is not None and live.task is not None:
                async def drained():
                    await asyncio.gather(live.task, return_exceptions=True)
                call(drained())
