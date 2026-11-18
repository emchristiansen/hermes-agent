"""TITLE-CANCEL-01 regression, unloaded title generator (curator run6683's normal-entry witness
test_curator_unloaded_title_registry_r2.py, adopted unchanged except this docstring). In a fresh test process the
title generator is genuinely unloaded when the retirement starts; its fencing read is paused before the lookup, the
retirement is cancelled, the profile re-served and the generator first imported, then the obsolete read released: it
must not fence the successor, whose real title upgrade must start and run."""
import asyncio
import os
import sys
import threading
from tests.gateway.test_hosted_group_members import _gateway


def test_unloaded_registry_at_retirement_cannot_become_successor_fence(tmp_path, monkeypatch):
    from gateway import run_runtime as rr
    from gateway import run_shutdown_budget as budget
    entered, release, finished = (threading.Event() for _ in range(3))
    selected = []
    def trace(frame, event, arg):
        if frame.f_code is not budget.title_upgrades_live_under.__code__:
            return None
        if event == 'call' and not entered.is_set():
            selected.append(threading.get_ident())
            entered.set()
            assert release.wait(20)
        if event == 'return' and threading.get_ident() in selected:
            finished.set()
        return trace
    threading.settrace(trace)
    with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
        assert 'agent.title_generator' not in sys.modules, 'unloaded-registry schedule not established'
        async def scenario():
            attempt = asyncio.create_task(rr.unserve_profile_runtime(runner, homes['alpha']))
            assert await asyncio.to_thread(entered.wait, 10)
            attempt.cancel()
            try:
                await attempt
            except asyncio.CancelledError:
                pass
            successor = await rr.serve_profile_runtime(runner, 'alpha', homes['alpha'])
            assert 'agent.title_generator' not in sys.modules
            from agent import title_generator as tg
            root = os.path.normcase(os.path.realpath(homes['alpha']))
            assert root not in tg._FENCED_HOMES
            release.set()
            assert await asyncio.to_thread(finished.wait, 10)
            ran = threading.Event()
            def model(*args, **kwargs):
                ran.set()
                return 'Unloaded registry successor witness'
            monkeypatch.setattr(tg, 'generate_title', model)
            from gateway.session_authorities import owner_scope
            successor.db.create_session('unloaded-title-probe', source='test')
            with owner_scope(successor):
                title = tg.maybe_auto_title(successor.db, 'unloaded-title-probe', 'Investigate unloaded title registry race')
            assert title is not None
            if title.ident is not None:
                title.join(5)
            observed = {'successor_fenced':root in tg._FENCED_HOMES,'title_started':title.ident is not None,'title_ran':ran.is_set()}
            print('UNLOADED_OBSERVATION',observed)
            assert ran.is_set(), observed
        try:
            call(scenario())
        finally:
            release.set()
            threading.settrace(None)
