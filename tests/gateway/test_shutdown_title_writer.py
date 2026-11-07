"""A3-SH-TITLE-01: a retired turn's auto-title upgrade thread is a SessionDB writer for the shutdown close gate.

The real-conversation pair below is review run6017's ``test_review_q04_title_writer_v3.py`` (t_60889c77) kept as a maintained
case: a real initialized hosted runner, its registry SessionDB, a real ``run_conversation`` on the gateway executor, the real
title spawn/persistence, the real stop, the production lock release and a separate contender process. Only the model answer
and the title model call are synthetic (the latter held on an event). While the title thread can still write, the stop must
keep the store open and the profile owned (contender REFUSED, every write owned and before any close); once it retired before
the stop, the normal close and release must still happen (contender ACQUIRED)."""
import json
import sys
import threading
import time

import pytest

from tests.agent.test_run_agent import _mock_response
from tests.gateway.test_hosted_group_members import _gateway
from tests.gateway.test_profile_store_release import _let_go
from tests.gateway.test_shutdown_async_delegation_writers import _make_agent
from tests.gateway.test_shutdown_hosted_room_writers import _contender, _probe


@pytest.mark.parametrize('retire_before_stop', [False, True])
def test_title_writer_cannot_outlive_store_ownership(tmp_path, monkeypatch, retire_before_stop):
    from agent import title_generator as tg
    from gateway.run_runtime import managed_turn_count
    from gateway.run_shutdown_budget import final_sweep_live
    from gateway.runtime_ownership import process_ownership
    from gateway.status import release_gateway_runtime_lock
    release = threading.Event(); entered = threading.Event(); writes = []; closes = []; contenders = []; obs = {}
    upgrade = None

    def model(*args, **kwargs):
        entered.set(); assert release.wait(100)
        return 'Bounded independent shutdown witness'
    monkeypatch.setattr(tg, 'generate_title', model)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        home = homes['alpha']; db = runner.session_authorities.for_home(home).db
        child, child_db = _make_agent(monkeypatch, home)
        assert child_db is db
        child.platform = 'api_server'
        child.context_compressor.threshold_tokens = 10**9
        child._interruptible_api_call = lambda _kwargs: _mock_response(content='Completed independent turn', finish_reason='stop')
        sid = child.session_id
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
        turn = call(runner._run_in_executor_with_context(child.run_conversation, 'Investigate bounded shutdown ownership'))
        assert turn['final_response'] == 'Completed independent turn', turn
        assert entered.wait(15)
        upgrades = [t for t in tg._UPGRADE_THREADS if t.is_alive()]
        assert len(upgrades) == 1, upgrades
        upgrade = upgrades[0]
        monkeypatch.setattr(db, 'close', close)
        try:
            if retire_before_stop:
                release.set(); upgrade.join(20); assert not upgrade.is_alive()
            call(runner.stop()); release_gateway_runtime_lock()
            contender = _contender(home); contenders.append(contender)
            obs['at_stop'] = {'writer_alive': upgrade.is_alive(), 'db_closed': db._conn is None,
                              'owned': process_ownership.owns(home), 'retained': getattr(process_ownership, '_retained_for', None),
                              'managed_live': managed_turn_count(runner), 'sweep_live': final_sweep_live(runner),
                              'contender': json.loads(contender.stdout.readline())['contender']}
            release.set(); upgrade.join(20)
            obs.update(writes=writes, closes=closes, writer_alive_after_join=upgrade.is_alive(),
                       connection_reopened=db._conn is not None, title=db.get_session_title(sid))
            (tmp_path / 'observation.json').write_text(json.dumps(obs, indent=2))
            assert not upgrade.is_alive(), obs
            assert all(w['owned'] and not w['after_close'] for w in writes), obs
            if not retire_before_stop:
                assert obs['at_stop']['owned'] and not obs['at_stop']['db_closed'], obs
                assert obs['at_stop']['contender'] == 'REFUSED', obs
                assert obs['title'] == 'Bounded independent shutdown witness', obs
            else:
                assert not obs['at_stop']['owned'] and obs['at_stop']['db_closed'], obs
                assert obs['at_stop']['contender'] == 'ACQUIRED', obs
        finally:
            release.set(); upgrade.join(20)
            for contender in contenders:
                _let_go(contender)
            db.close()


def _held_upgrade(tg):
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, args=(30,), name='auto-title', daemon=True)
    tg.start_title_upgrade(thread)
    return gate, thread


def test_join_title_upgrades_counts_a_live_upgrade_within_its_bound():
    from agent import title_generator as tg
    gate, thread = _held_upgrade(tg)
    try:
        t0 = time.monotonic(); live = tg.join_title_upgrades(0.3); waited = time.monotonic() - t0
        assert live == 1 and 0.25 <= waited < 5.0, (live, waited)
    finally:
        gate.set(); thread.join(10)
    assert tg.join_title_upgrades(5.0) == 0


def test_an_unstarted_registered_upgrade_counts_as_live_and_is_not_joined():
    from agent import title_generator as tg
    thread = threading.Thread(target=lambda: None, name='auto-title', daemon=True)
    tg._UPGRADE_THREADS.add(thread)   # registered, about to start (start_title_upgrade adds before start)
    try:
        assert tg.join_title_upgrades(0.0) == 1
    finally:
        tg._UPGRADE_THREADS.discard(thread)


def test_the_close_gate_reads_title_upgrades_only_where_the_generator_is_loaded(monkeypatch):
    from agent import title_generator as tg
    from gateway import run_shutdown_budget as budget
    gate, thread = _held_upgrade(tg)
    try:
        assert budget.title_upgrades_live(0.05) == 1
        assert budget.join_descendants_after_starters(0.05) >= 1   # the final descendant read sees it too
        monkeypatch.delitem(sys.modules, 'agent.title_generator')   # a process that never titled started none
        assert budget.title_upgrades_live(0.05) == 0
    finally:
        gate.set(); thread.join(10)
