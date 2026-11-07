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
same defect)."""
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
