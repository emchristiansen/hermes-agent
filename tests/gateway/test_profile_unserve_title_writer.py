"""A3-SH-TITLE-02: a completed turn's auto-title upgrade is a writer of its profile's store for that profile's unserve.

review6224 (t_60889c77) found that ``_unserve_profile`` closed alpha's store and released its reservation beneath a live
title thread. The thread later reopened the store and wrote it under an independent contender's lock. The repair counts,
before the home's stores close or its reservation goes, only the title upgrades writing a store under THAT home.

The real-producer pair uses the shape of the review's ``test_review_q06_title_unserve.py``: an initialized runner serving
alpha and beta, alpha's registry SessionDB, a real ``run_conversation`` on the gateway executor, the real title
spawn/registration/persistence, the production unserve and stop, and separate contender processes. Only the model answers
(the title one held on an event) and the unrelated MCP/memory/log teardown are substituted. Here the held title is
released after the unserves, before the whole stop. The stop then closes and releases normally (the review's probe holds
it through the stop instead).

The control-consumer case drives the same held writer through R4's production consumer: ``hermes -p alpha gateway stop`` over
the real control socket. It requires the explicit retained answer, contender refusal and unrelated beta's stop. It then
requires the eventual safe release through the same consumer once the title retired: ``gateway start`` (serve-profile:
its retirement finishes), then ``gateway stop`` (unserve-profile; contender ACQUIRED).

The deferred case builds a real deferred upgrade (``maybe_auto_title`` on alpha's store) and hands it over only after alpha
retired and released. This is a turn the retirement did not join, and the fence must refuse it.

The remaining cases pin the per-home scope, the registration-before-read ordering inside the retirement, the fence and its
lifting, the failed-start and unstarted bounds, and the fail-closed reader (A4 review r6264 A4-SH-TITLE-UNSERVE-01 is the
same defect).

The A3-SH-TITLE-03 cases at the end (review6390) cover a re-serve whose initialization fails: the home stays fenced through
the release's zero read and the release itself, an upgrade started while the failed attempt still owned the home keeps the
reservation, beta releases independently, and a later serve titles the new generation and finally releases."""
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tests.agent.test_run_agent import _mock_response
from tests.gateway.test_hosted_group_members import _gateway
from tests.gateway.test_profile_store_release import _external_teardown, _let_go
from tests.gateway.test_shutdown_async_delegation_writers import _make_agent
from tests.gateway.test_shutdown_hosted_room_writers import _contender, _probe


def _read(contender):
    return json.loads(contender.stdout.readline())['contender']


@pytest.mark.parametrize('retire_before_unserve', [False, True], ids=['title-held', 'title-retired'])
def test_unserve_keeps_only_the_profile_whose_title_writer_lives(tmp_path, monkeypatch, retire_before_unserve):
    from agent import title_generator as tg
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    release = threading.Event(); entered = threading.Event(); writes = []; closes = []; contenders = []; obs = {}
    upgrade = None

    def model(*args, **kwargs):
        entered.set(); assert release.wait(100)
        return 'Profile local title witness'
    monkeypatch.setattr(tg, 'generate_title', model)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home = homes['alpha']; beta = homes['beta']; db = runner.session_authorities.for_home(home).db
        child, child_db = _make_agent(monkeypatch, home)
        assert child_db is db
        child.platform = 'api_server'
        child.context_compressor.threshold_tokens = 10**9
        child._interruptible_api_call = lambda _kwargs: _mock_response(content='Completed alpha turn', finish_reason='stop')
        real_write = db._execute_write; real_close = db.close

        def write(fn, *a, **kw):
            row = {'thread': threading.current_thread().name, 'owned': process_ownership.owns(home),
                   'connection_absent': db._conn is None, 'after_close': bool(closes), 'probe': _probe(home)}
            writes.append(row)
            result = real_write(fn, *a, **kw); row['returned'] = True
            return result

        def close(*a, **kw):
            closes.append({'thread_live': upgrade.is_alive() if upgrade else None})
            return real_close(*a, **kw)
        monkeypatch.setattr(db, '_execute_write', write)
        turn = call(runner._run_in_executor_with_context(child.run_conversation, 'Investigate profile local title ownership'))
        assert turn['final_response'] == 'Completed alpha turn', turn
        assert entered.wait(15)
        upgrades = [t for t in tg._UPGRADE_THREADS if t.is_alive()]
        assert len(upgrades) == 1, upgrades
        upgrade = upgrades[0]
        monkeypatch.setattr(db, 'close', close)
        try:
            if retire_before_unserve:
                release.set(); upgrade.join(20); assert not upgrade.is_alive()
            with _external_teardown():
                obs['alpha_unserve'] = call(runner._unserve_profile('alpha', home))
            contender = _contender(home); contenders.append(contender)
            obs['alpha'] = {'writer_alive': upgrade.is_alive(), 'db_closed': db._conn is None,
                            'owned': process_ownership.owns(home), 'contender': _read(contender)}
            with _external_teardown():
                obs['beta_unserve'] = call(runner._unserve_profile('beta', beta))
            contender = _contender(beta); contenders.append(contender)
            obs['beta'] = {'owned': process_ownership.owns(beta), 'contender': _read(contender)}
            release.set(); upgrade.join(20)        # the held title finishes before the whole stop
            obs['writer_alive_before_stop'] = upgrade.is_alive()
            obs['title_before_stop'] = db.get_session_title(child.session_id) if db._conn is not None else None
            call(runner.stop()); release_gateway_runtime_lock()
            contender = _contender(home); contenders.append(contender)
            obs['after_stop'] = {'owned': process_ownership.owns(home), 'db_closed': db._conn is None,
                                 'contender': _read(contender)}
            obs.update(writes=writes, closes=closes)
            (tmp_path / 'observation.json').write_text(json.dumps(obs, indent=2, default=str))
            assert not obs['writer_alive_before_stop'], obs
            assert obs['beta_unserve'] is True and obs['beta'] == {'owned': False, 'contender': 'ACQUIRED'}, obs
            assert writes and all(w['owned'] and not w['after_close'] and not w['connection_absent'] for w in writes), obs
            if retire_before_unserve:
                assert obs['alpha_unserve'] is True and obs['alpha']['db_closed'] and not obs['alpha']['owned'], obs
                assert obs['alpha']['contender'] == 'ACQUIRED', obs
            else:
                assert obs['alpha_unserve'] is False and not obs['alpha']['db_closed'] and obs['alpha']['owned'], obs
                assert obs['alpha']['writer_alive'] and obs['alpha']['contender'] == 'REFUSED', obs
                assert all(not c['thread_live'] for c in closes), obs   # the stop closed only after the title retired
                assert obs['after_stop']['db_closed'] and not obs['after_stop']['owned'], obs
                assert obs['after_stop']['contender'] == 'ACQUIRED', obs
                assert obs['title_before_stop'] == 'Profile local title witness', obs
        finally:
            release.set(); upgrade.join(20)
            for contender in contenders:
                _let_go(contender)
            db.close()


def test_retirement_reads_titles_after_the_turns_that_register_them_retired(tmp_path, monkeypatch):
    """The read follows the turn join. A turn still retiring when the unserve begins registers its upgrade
    before it retires (here: during the join), and the retirement must still see that upgrade."""
    from agent import title_generator as tg
    from gateway import session_runtime_workers as workers
    from gateway.runtime_ownership import process_ownership
    gate = threading.Event(); registered = []
    real_join = workers.join_authority_work
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home = homes['alpha']; db = runner.session_authorities.for_home(home).db

        async def join(authority, timeout):
            if authority.db is db and not registered:   # alpha's last turn hands its title over as it retires
                thread = threading.Thread(target=gate.wait, args=(30,), name='auto-title', daemon=True)
                tg._UPGRADE_STORES[thread] = str(db.db_path)
                tg.start_title_upgrade(thread); registered.append(thread)
            return await real_join(authority, timeout)
        monkeypatch.setattr(workers, 'join_authority_work', join)
        try:
            with _external_teardown():
                alpha = call(runner._unserve_profile('alpha', home))
            assert registered and alpha is False and process_ownership.owns(home) and db._conn is not None
            with _external_teardown():
                assert call(runner._unserve_profile('beta', homes['beta'])) is True
            assert not process_ownership.owns(homes['beta'])
        finally:
            gate.set()
            for thread in registered:
                thread.join(10)


def _held(tg, store=None):
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, args=(30,), name='auto-title', daemon=True)
    if store is not None:
        tg._UPGRADE_STORES[thread] = str(store)
    tg.start_title_upgrade(thread)
    return gate, thread


def test_a_title_upgrade_holds_only_the_home_its_store_lives_under(tmp_path):
    from agent import title_generator as tg
    alpha = tmp_path / 'profiles' / 'alpha'; beta = tmp_path / 'profiles' / 'beta'
    alpha.mkdir(parents=True); beta.mkdir(parents=True)
    gate, thread = _held(tg, beta / 'state.db')
    try:
        t0 = time.monotonic(); assert tg.join_title_upgrades_under(alpha, 2.0) == 0
        assert time.monotonic() - t0 < 1.0          # another home's writer is not waited for
        t0 = time.monotonic(); assert tg.join_title_upgrades_under(beta, 0.3) == 1
        assert 0.25 <= time.monotonic() - t0 < 5.0  # its own home waits within the bound
        assert tg.join_title_upgrades(0.0) == 1     # the whole-stop count still sees it
    finally:
        gate.set(); thread.join(10)
    assert tg.join_title_upgrades_under(beta, 5.0) == 0


def test_an_upgrade_without_a_recorded_store_holds_every_home(tmp_path):
    from agent import title_generator as tg
    gate, thread = _held(tg)
    try:
        assert tg.join_title_upgrades_under(tmp_path / 'alpha', 0.05) == 1
        assert tg.join_title_upgrades_under(tmp_path / 'beta', 0.05) == 1
    finally:
        gate.set(); thread.join(10)


def test_a_registered_upgrade_is_waited_for_until_it_starts_then_joined(tmp_path):
    from agent import title_generator as tg
    thread = threading.Thread(target=threading.Event().wait, args=(0.1,), name='auto-title', daemon=True)
    tg._UPGRADE_STORES[thread] = str(tmp_path / 'state.db')
    tg._UPGRADE_THREADS.add(thread)          # registered; start_title_upgrade starts it right after
    starter = threading.Timer(0.2, thread.start); starter.start()
    try:
        assert tg.join_title_upgrades_under(tmp_path, 5.0) == 0
        assert thread.ident is not None and not thread.is_alive()
    finally:
        starter.join(5); tg._UPGRADE_THREADS.discard(thread)


def test_a_registered_upgrade_that_never_starts_counts_as_live(tmp_path):
    from agent import title_generator as tg
    thread = threading.Thread(target=lambda: None, name='auto-title', daemon=True)
    tg._UPGRADE_STORES[thread] = str(tmp_path / 'state.db')
    tg._UPGRADE_THREADS.add(thread)
    try:
        t0 = time.monotonic(); assert tg.join_title_upgrades_under(tmp_path, 0.2) == 1
        assert time.monotonic() - t0 < 5.0
    finally:
        tg._UPGRADE_THREADS.discard(thread)


def test_an_upgrade_whose_start_failed_is_unregistered(tmp_path, monkeypatch):
    from agent import title_generator as tg
    thread = threading.Thread(target=lambda: None, name='auto-title', daemon=True)
    tg._UPGRADE_STORES[thread] = str(tmp_path / 'state.db')

    def refuse():
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(thread, 'start', refuse)
    with pytest.raises(RuntimeError):
        tg.start_title_upgrade(thread)
    assert thread not in tg._UPGRADE_THREADS
    assert tg.join_title_upgrades_under(tmp_path, 0.0) == 0


def test_the_per_home_reader_fails_closed_and_ignores_an_unloaded_generator(tmp_path, monkeypatch):
    from agent import title_generator as tg
    from gateway import run_shutdown_budget as budget

    def broken(home, timeout, *, fence=False):
        raise RuntimeError('unreadable registry')
    monkeypatch.setattr(tg, 'join_title_upgrades_under', broken)
    assert budget.title_upgrades_live_under(tmp_path, 0.05) == 1
    monkeypatch.delitem(sys.modules, 'agent.title_generator')   # a process that never titled started none
    assert budget.title_upgrades_live_under(tmp_path, 0.05) == 0


def _root(home):
    return os.path.normcase(os.path.realpath(os.fspath(home)))


@pytest.mark.platforms('linux')
def test_the_control_consumer_answers_a_held_title_retained_then_releases_after_it(tmp_path, monkeypatch, capsys):
    from agent import title_generator as tg
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import gateway as gw, gateway_profile_lifecycle as lifecycle
    from tests.gateway.test_profile_unserve_control_answer import _cli_as, _host, _record_requests
    release = threading.Event(); entered = threading.Event(); writes = []; closes = []; contenders = []; obs = {}
    upgrade = None

    def model(*args, **kwargs):
        entered.set(); assert release.wait(100)
        return 'Control consumer title witness'
    monkeypatch.setattr(tg, 'generate_title', model)
    with _host(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, root, homes, call):
        home = homes['alpha']; beta = homes['beta']; db = runner.session_authorities.for_home(home).db
        child, child_db = _make_agent(monkeypatch, home)
        assert child_db is db
        child.platform = 'api_server'
        child.context_compressor.threshold_tokens = 10**9
        child._interruptible_api_call = lambda _kwargs: _mock_response(content='Completed consumer turn', finish_reason='stop')
        real_write = db._execute_write; real_close = db.close

        def write(fn, *a, **kw):
            row = {'thread': threading.current_thread().name, 'owned': process_ownership.owns(home),
                   'connection_absent': db._conn is None, 'after_close': bool(closes), 'probe': _probe(home)}
            writes.append(row)
            result = real_write(fn, *a, **kw); row['returned'] = True
            return result

        def close(*a, **kw):
            closes.append({'thread_live': upgrade.is_alive() if upgrade else None})
            return real_close(*a, **kw)
        monkeypatch.setattr(db, '_execute_write', write)
        turn = call(runner._run_in_executor_with_context(child.run_conversation, 'Investigate control consumer title ownership'))
        assert turn['final_response'] == 'Completed consumer turn', turn
        assert entered.wait(15)
        upgrades = [t for t in tg._UPGRADE_THREADS if t.is_alive()]
        assert len(upgrades) == 1, upgrades
        upgrade = upgrades[0]
        monkeypatch.setattr(db, 'close', close)
        requests = _record_requests(monkeypatch)
        try:
            _cli_as(monkeypatch, 'alpha', home, root)
            with _external_teardown():
                obs['alpha_handled'] = lifecycle.profile_lifecycle('stop', SimpleNamespace(all=False, force=False))
            obs['alpha_cli'] = capsys.readouterr().out
            answer = requests[0][2] if requests else None
            obs['alpha_answer'] = answer
            obs['alpha_confirmed'] = lifecycle.ack_confirmed(answer, 'unserved', 'alpha')
            obs['alpha'] = {'writer_alive': upgrade.is_alive(), 'db_closed': db._conn is None, 'owned': process_ownership.owns(home),
                            'retiring_registered': runner.session_authorities.for_home(home) is not None,
                            'fenced': _root(home) in tg._FENCED_HOMES}
            contenders.append(_contender(home)); obs['alpha']['contender'] = _read(contenders[-1])
            del requests[:]
            _cli_as(monkeypatch, 'beta', beta, root)
            with _external_teardown():
                obs['beta_handled'] = lifecycle.profile_lifecycle('stop', SimpleNamespace(all=False, force=False))
            obs['beta_cli'] = capsys.readouterr().out
            obs['beta_requests'] = list(requests)
            contenders.append(_contender(beta)); obs['beta'] = {'owned': process_ownership.owns(beta), 'contender': _read(contenders[-1])}
            release.set(); upgrade.join(20)
            obs['title'] = db.get_session_title(child.session_id)
            obs['owned_after_title'] = process_ownership.owns(home)
            _let_go(contenders.pop(0))                     # alpha's refused contender gives up
            del requests[:]
            _cli_as(monkeypatch, 'alpha', home, root)
            monkeypatch.setattr(gw, '_host_multiplexer_for_all_verb', lambda: SimpleNamespace(home=root))
            obs['start_handled'] = lifecycle.profile_lifecycle('start', SimpleNamespace(all=False, force=False))
            obs['fenced_after_serve'] = _root(home) in tg._FENCED_HOMES
            with _external_teardown():
                obs['restop_handled'] = lifecycle.profile_lifecycle('stop', SimpleNamespace(all=False, force=False))
            obs['again_cli'] = capsys.readouterr().out
            obs['again_requests'] = list(requests)
            contenders.append(_contender(home)); obs['after_release'] = {
                'owned': process_ownership.owns(home), 'db_closed': db._conn is None, 'contender': _read(contenders[-1]),
                'fenced': _root(home) in tg._FENCED_HOMES}
            obs.update(writes=writes, closes=closes)
            (tmp_path / 'observation.json').write_text(json.dumps(obs, indent=2, default=str))
            assert obs['alpha_handled'] is True and obs['alpha_confirmed'] is False, obs
            assert isinstance(answer, dict) and 'unserved' not in answer, ('alpha answered unserved', obs)
            assert "title writer" in answer.get('error', '') and 'ownership is retained' in answer['error'], obs
            assert obs['alpha'] == {'writer_alive': True, 'db_closed': False, 'owned': True, 'retiring_registered': True,
                                    'fenced': False, 'contender': 'REFUSED'}, obs
            assert [r[:2] for r in obs['beta_requests']] == [('unserve-profile', 'beta')], obs
            assert obs['beta_requests'][0][2].get('unserved') == 'beta', obs
            assert obs['beta'] == {'owned': False, 'contender': 'ACQUIRED'}, obs
            assert obs['title'] == 'Control consumer title witness' and obs['owned_after_title'], obs
            assert obs['start_handled'] is True and obs['restop_handled'] is True, obs
            assert [r[:2] for r in obs['again_requests']] == [('serve-profile', 'alpha'), ('unserve-profile', 'alpha')], obs
            assert obs['again_requests'][0][2].get('served') == 'alpha', obs        # the retirement finished on serve
            assert obs['again_requests'][1][2].get('unserved') == 'alpha', obs
            assert obs['fenced_after_serve'] is False, obs
            assert obs['after_release'] == {'owned': False, 'db_closed': True, 'contender': 'ACQUIRED', 'fenced': True}, obs
            assert writes and all(w['owned'] and not w['after_close'] and not w['connection_absent'] for w in writes), obs
            assert closes and all(not c['thread_live'] for c in closes), obs
        finally:
            release.set(); upgrade.join(20)
            for contender in contenders:
                _let_go(contender)


def test_a_deferred_title_handed_over_after_the_profile_retired_is_refused(tmp_path, monkeypatch):
    from agent import title_generator as tg
    from gateway.run import _profile_runtime_scope
    from gateway.runtime_ownership import process_ownership
    modelled = []; writes = []; closes = []; contenders = []
    monkeypatch.setattr(tg, 'title_upgrade_must_wait_for_turn', lambda runtime: True)   # the self-hosted (deferred) branch
    monkeypatch.setattr(tg, 'generate_title', lambda *a, **k: modelled.append(1) or 'Late deferred title')
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home = homes['alpha']; db = runner.session_authorities.for_home(home).db
        _make_agent(monkeypatch, home)
        db.create_session('deferred', source='test')
        with _profile_runtime_scope(home):
            upgrade = tg.maybe_auto_title(db, 'deferred', 'Investigate the deferred title fence',
                                          main_runtime={'model': 'test/model', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9'})
        assert upgrade is not None and upgrade.ident is None and upgrade not in tg._UPGRADE_THREADS
        assert tg._UPGRADE_STORES.get(upgrade) == str(db.db_path)
        instant = db.get_session_title('deferred')
        real_write = db._execute_write; real_close = db.close

        def write(fn, *a, **kw):
            writes.append({'thread': threading.current_thread().name, 'owned': process_ownership.owns(home), 'after_close': bool(closes)})
            return real_write(fn, *a, **kw)

        def close(*a, **kw):
            closes.append(True)
            return real_close(*a, **kw)
        monkeypatch.setattr(db, '_execute_write', write); monkeypatch.setattr(db, 'close', close)
        try:
            with _external_teardown():
                alpha = call(runner._unserve_profile('alpha', home))
            contenders.append(_contender(home)); contender = _read(contenders[-1])
            tg.start_title_upgrade(upgrade)      # what the turn's finalizer does once its model answered (synchronous refusal)
            with _external_teardown():
                beta = call(runner._unserve_profile('beta', homes['beta']))
            observed = {'alpha': alpha, 'contender': contender, 'closed': bool(closes), 'started': upgrade.ident is not None,
                        'registered': upgrade in tg._UPGRADE_THREADS, 'fenced': _root(home) in tg._FENCED_HOMES,
                        'modelled': len(modelled), 'writes': writes, 'beta': beta, 'instant': instant}
            (tmp_path / 'observation.json').write_text(json.dumps(observed, indent=2, default=str))
            assert alpha is True and contender == 'ACQUIRED' and observed['closed'], observed
            assert not observed['started'] and not observed['registered'] and observed['fenced'], observed
            assert observed['modelled'] == 0 and not [w for w in writes if w['thread'] == 'auto-title'], observed
            assert beta is True and instant, observed
        finally:
            for contender in contenders:
                _let_go(contender)


def test_the_fence_refuses_only_its_home_until_it_is_lifted(tmp_path):
    from agent import title_generator as tg
    from gateway import run_shutdown_budget as budget
    alpha = tmp_path / 'alpha'; beta = tmp_path / 'beta'; alpha.mkdir(); beta.mkdir()
    assert budget.title_upgrades_live_under(alpha, 0.0, fence=True) == 0
    try:
        assert _root(alpha) in tg._FENCED_HOMES and _root(beta) not in tg._FENCED_HOMES
        refused = threading.Thread(target=lambda: None, name='auto-title', daemon=True)
        tg._UPGRADE_STORES[refused] = str(alpha / 'state.db')
        tg.start_title_upgrade(refused)
        assert refused.ident is None and refused not in tg._UPGRADE_THREADS
        gate, other = _held(tg, beta / 'state.db')      # another home starts normally
        try:
            assert other.ident is not None and tg.join_title_upgrades_under(alpha, 0.0) == 0
            assert budget.title_upgrades_live_under(beta, 0.05, fence=True) == 1    # a live writer is never fenced over
            assert _root(beta) not in tg._FENCED_HOMES
        finally:
            gate.set(); other.join(10)
        budget.unfence_title_home(alpha)
        allowed = threading.Thread(target=lambda: None, name='auto-title', daemon=True)
        tg._UPGRADE_STORES[allowed] = str(alpha / 'state.db')
        tg.start_title_upgrade(allowed); allowed.join(10)
        assert allowed.ident is not None
    finally:
        budget.unfence_title_home(alpha)


# A3-SH-TITLE-03 (review6390, t_60889c77): a failed re-serve lifted the fence, its initialization failed, and the release
# read the home's titles without fencing it. An old deferred upgrade handed over afterwards then reopened alpha's store and
# wrote it while an independent contender held alpha's lock. The cases below follow the review's
# test_review_q07_lifetime_extra.py and test_review_q07_failed_reserve_live_title.py shapes: production unserve/serve/release,
# real deferred upgrade and real deferred-start consumer, separate contender processes. Only the model answer, the authority
# initialization failure and the unrelated MCP/memory/log teardown are substituted.


def _failing_initialize(monkeypatch, home, before_failure=None):
    from gateway import session_authority
    real = session_authority.initialize_session_authority

    async def initialize(*args, **kwargs):
        if kwargs.get('profile_id') == str(home):
            if before_failure is not None:
                before_failure()
            raise RuntimeError('injected authority initialization failure')
        return await real(*args, **kwargs)
    monkeypatch.setattr(session_authority, 'initialize_session_authority', initialize)
    return lambda: monkeypatch.setattr(session_authority, 'initialize_session_authority', real)


@pytest.mark.parametrize('fail_build', [True, False], ids=['failed-reserve', 'served-then-unserved'])
def test_a_deferred_title_stays_refused_after_a_reserve_attempt_releases_the_home(tmp_path, monkeypatch, fail_build):
    from agent import title_generator as tg
    from agent.turn_context import start_deferred_title_upgrade
    from gateway.run import _profile_runtime_scope
    from gateway.runtime_ownership import process_ownership
    modelled = []; writes = []; contenders = []; obs = {}
    monkeypatch.setattr(tg, 'title_upgrade_must_wait_for_turn', lambda runtime: True)   # the self-hosted (deferred) branch
    monkeypatch.setattr(tg, 'generate_title', lambda *a, **k: modelled.append(1) or 'Late deferred title after reserve')
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home = homes['alpha']; db = runner.session_authorities.for_home(home).db
        child, child_db = _make_agent(monkeypatch, home)
        assert child_db is db
        db.create_session('deferred-reserve', source='test')
        with _profile_runtime_scope(home):
            upgrade = tg.maybe_auto_title(db, 'deferred-reserve', 'Investigate the title fence across a failed reserve')
        assert upgrade is not None and upgrade.ident is None
        child._deferred_title_upgrade = upgrade      # the turn's finalizer hands it over later
        with _external_teardown():
            assert call(runner._unserve_profile('alpha', home)) is True
        if fail_build:
            _failing_initialize(monkeypatch, home)
        obs['serve_error'] = call(runner._serve_profile_runtime('alpha', home))
        if fail_build:
            assert 'injected authority initialization failure' in obs['serve_error'], obs
        else:
            assert obs['serve_error'] is None, obs
            with _external_teardown():
                assert call(runner._unserve_profile('alpha', home)) is True
        contenders.append(_contender(home))
        obs['before_late_start'] = {'owned': process_ownership.owns(home), 'contender': _read(contenders[-1]),
                                    'authority': runner.session_authorities.for_home(home) is not None,
                                    'fenced': _root(home) in tg._FENCED_HOMES}
        real_write = db._execute_write

        def write(fn, *a, **kw):
            writes.append({'thread': threading.current_thread().name, 'owned': process_ownership.owns(home)})
            return real_write(fn, *a, **kw)
        monkeypatch.setattr(db, '_execute_write', write)
        try:
            start_deferred_title_upgrade(child)
            if upgrade.ident is not None:
                upgrade.join(20)
            with _external_teardown():
                obs['beta_unserve'] = call(runner._unserve_profile('beta', homes['beta']))
            contenders.append(_contender(homes['beta'])); obs['beta_contender'] = _read(contenders[-1])
            obs.update(started=upgrade.ident is not None, registered=upgrade in tg._UPGRADE_THREADS,
                       modelled=len(modelled), writes=writes)
            (tmp_path / 'observation.json').write_text(json.dumps(obs, indent=2, default=str))
            assert obs['before_late_start'] == {'owned': False, 'contender': 'ACQUIRED', 'authority': False, 'fenced': True}, obs
            assert not obs['started'] and not obs['registered'] and obs['modelled'] == 0 and writes == [], obs
            assert obs['beta_unserve'] is True and obs['beta_contender'] == 'ACQUIRED', obs
        finally:
            for contender in contenders:
                _let_go(contender)
            db.close()


@pytest.mark.platforms('linux')
def test_a_failed_reserve_keeps_an_already_started_title_owned_then_releases_and_titles_again(tmp_path, monkeypatch):
    """The live-writer side of TITLE-03: an old deferred upgrade that started while the failed serve still owned alpha is a
    live writer. The failure keeps alpha's lock (contender refused) and every write stays owned; beta releases on its own.
    Once the title retired, a later serve and unserve titles the new generation and finally releases alpha."""
    from agent import title_generator as tg
    from agent.turn_context import start_deferred_title_upgrade
    from gateway.run import _profile_runtime_scope
    from gateway.runtime_ownership import process_ownership
    gate = threading.Event(); entered = threading.Event(); writes = []; contenders = []; obs = {}
    titles = iter(['Old generation live title', 'New generation title'])

    def model(*args, **kwargs):
        entered.set(); assert gate.wait(60)
        return next(titles)
    monkeypatch.setattr(tg, 'title_upgrade_must_wait_for_turn', lambda runtime: True)
    monkeypatch.setattr(tg, 'generate_title', model)
    with _gateway(tmp_path, monkeypatch, served=('alpha', 'beta')) as (runner, homes, call):
        home = homes['alpha']; db = runner.session_authorities.for_home(home).db
        child, child_db = _make_agent(monkeypatch, home)
        assert child_db is db
        db.create_session('live-old', source='test')
        with _profile_runtime_scope(home):
            upgrade = tg.maybe_auto_title(db, 'live-old', 'Investigate a title started during a failed reserve')
        assert upgrade is not None and upgrade.ident is None
        child._deferred_title_upgrade = upgrade
        with _external_teardown():
            assert call(runner._unserve_profile('alpha', home)) is True

        def started_during_build():
            start_deferred_title_upgrade(child)
            assert entered.wait(10)
            obs['during_build'] = {'owned': process_ownership.owns(home), 'writer_alive': upgrade.is_alive()}
        restore = _failing_initialize(monkeypatch, home, started_during_build)
        real_write = db._execute_write

        def write(fn, *a, **kw):
            writes.append({'thread': threading.current_thread().name, 'owned': process_ownership.owns(home)})
            return real_write(fn, *a, **kw)
        monkeypatch.setattr(db, '_execute_write', write)
        try:
            obs['serve_error'] = call(runner._serve_profile_runtime('alpha', home))
            contenders.append(_contender(home))
            obs['after_failed_build'] = {'owned': process_ownership.owns(home), 'writer_alive': upgrade.is_alive(),
                                         'contender': _read(contenders[-1]), 'fenced': _root(home) in tg._FENCED_HOMES,
                                         'authority': runner.session_authorities.for_home(home) is not None}
            with _external_teardown():
                obs['beta_unserve'] = call(runner._unserve_profile('beta', homes['beta']))
            contenders.append(_contender(homes['beta'])); obs['beta_contender'] = _read(contenders[-1])
            gate.set(); upgrade.join(20)
            obs['old_title'] = db.get_session_title('live-old')
            obs['writer_alive_after_join'] = upgrade.is_alive()
            restore()
            monkeypatch.setattr(tg, 'title_upgrade_must_wait_for_turn', lambda runtime: False)
            obs['reserve_again'] = call(runner._serve_profile_runtime('alpha', home))
            fresh = runner.session_authorities.for_home(home).db
            fresh.create_session('live-new', source='test')
            with _profile_runtime_scope(home):
                new = tg.maybe_auto_title(fresh, 'live-new', 'Investigate the next generation title')
            assert new is not None
            new.join(20)
            obs['new_title'] = fresh.get_session_title('live-new')
            with _external_teardown():
                obs['last_unserve'] = call(runner._unserve_profile('alpha', home))
            contenders.append(_contender(home)); obs['last_contender'] = _read(contenders[-1])
            obs['writes'] = writes
            (tmp_path / 'observation.json').write_text(json.dumps(obs, indent=2, default=str))
            assert obs['during_build'] == {'owned': True, 'writer_alive': True}, obs
            assert 'injected authority initialization failure' in obs['serve_error'], obs
            assert obs['after_failed_build'] == {'owned': True, 'writer_alive': True, 'contender': 'REFUSED',
                                                 'fenced': False, 'authority': False}, obs
            assert obs['beta_unserve'] is True and obs['beta_contender'] == 'ACQUIRED', obs
            assert not obs['writer_alive_after_join'] and obs['old_title'] == 'Old generation live title', obs
            assert writes and all(w['owned'] for w in writes), obs
            assert obs['reserve_again'] is None and obs['new_title'] == 'New generation title', obs
            assert obs['last_unserve'] is True and obs['last_contender'] == 'ACQUIRED', obs
        finally:
            gate.set()
            if upgrade.ident is not None:
                upgrade.join(20)
            for contender in contenders:
                _let_go(contender)
            db.close()


def test_a_failed_initialization_restores_the_fence_unless_a_writer_started(tmp_path, monkeypatch):
    """``serve_profile_runtime`` lifts the fence before it builds; a build that raises puts it back at once (before any
    release runs), and leaves the home unfenced only for an upgrade that started meanwhile (a live writer it must keep)."""
    import asyncio
    from types import SimpleNamespace
    from agent import title_generator as tg
    from gateway import run_runtime
    from gateway import run_shutdown_budget as budget
    alpha = (tmp_path / 'alpha'); alpha.mkdir()
    runner = SimpleNamespace(session_authorities=SimpleNamespace(for_home=lambda home: None))
    started = []

    async def build(runner, name, home, *, register):
        assert _root(home) not in tg._FENCED_HOMES      # lifted for the attempt
        if started:
            gate, thread = _held(tg, alpha / 'state.db'); started.append((gate, thread))
        raise RuntimeError('injected build failure')
    monkeypatch.setattr(run_runtime, '_build_profile_authority', build)
    assert budget.title_upgrades_live_under(alpha, 0.0, fence=True) == 0     # a retired alpha
    try:
        with pytest.raises(RuntimeError, match='injected build failure'):
            asyncio.run(run_runtime.serve_profile_runtime(runner, 'alpha', alpha))
        assert _root(alpha) in tg._FENCED_HOMES
        started.append(None)
        with pytest.raises(RuntimeError, match='injected build failure'):
            asyncio.run(run_runtime.serve_profile_runtime(runner, 'alpha', alpha))
        gate, thread = started[-1]
        try:
            assert thread.is_alive() and _root(alpha) not in tg._FENCED_HOMES
        finally:
            gate.set(); thread.join(10)
    finally:
        budget.unfence_title_home(alpha)
