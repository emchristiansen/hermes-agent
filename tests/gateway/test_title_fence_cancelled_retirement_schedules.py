"""TITLE-CANCEL-01 regression, three schedules, anchored by CONTENT (not absolute line numbers): a cancelled retirement's
title-fence read must not fence the re-served successor, whether its thread is paused BEFORE the generator's read body
runs (pre-capture: the curator's counterexample to draft 2a61a5b0), at ``if fence and live == 0`` (post-capture:
review6658's line-123 schedule), or before it even looks the title generator up, with the generator NOT imported when the
retirement started and first imported by the successor (not-imported: the curator's counterexample to draft 83ab8ef8).
Real initialized gateway, real unserve/cancel/serve, real ``maybe_auto_title`` under the successor's owner scope; only the auxiliary model answer is substituted. Derived from review6658's
test_review_cancelled_title_fence.py (kept verbatim alongside as test_title_fence_cancelled_retirement.py)."""
import asyncio
import importlib
import inspect
import json
import os
import sys
import threading

from tests.gateway.test_hosted_group_members import _gateway
import pytest


def _anchor(fn, needle):
    lines, first = inspect.getsourcelines(fn)
    hits = [first + i for i, line in enumerate(lines) if needle in line]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


@pytest.mark.parametrize('schedule', ['pre-capture', 'post-capture', 'not-imported'])
@pytest.mark.parametrize('cancel_first', [False, True], ids=['joined-control', 'cancelled-retirement'])
def test_cancelled_retirement_cannot_fence_a_successfully_reserved_successor(tmp_path, monkeypatch, cancel_first,
                                                                              schedule):
    from agent import title_generator as tg
    from gateway import run_runtime as rr
    from gateway import run_shutdown_budget as budget
    from gateway.runtime_ownership import process_ownership

    # Trace only schedules an existing worker (the retirement's fencing read); no registry, retirement, title or
    # serve semantics replaced. Pause points are found by source content, not absolute line numbers.
    if schedule == 'not-imported':
        target = budget.title_upgrades_live_under
        pause_line = _anchor(target, 'join = getattr(sys.modules.get("agent.title_generator"), "join_title_upgrades_under"')
    else:
        target = tg.join_title_upgrades_under
        pause_line = _anchor(target, 'root = os.path.normcase(' if schedule == 'pre-capture' else 'if fence and live == 0:')
    target_code = target.__code__
    at_fence = threading.Event()
    resume = threading.Event()
    prior_finished = threading.Event()
    armed = threading.Event()
    release_title = threading.Event()
    caught, imported_at_pause = [], []
    def trace(frame, event, arg):
        if frame.f_code is not target_code:
            return None
        if (event == 'line' and armed.is_set() and not at_fence.is_set()
                and frame.f_lineno == pause_line):
            caught.append(threading.get_ident())
            imported_at_pause.append('agent.title_generator' in sys.modules)
            at_fence.set()
            assert resume.wait(20)
        if event == 'return' and frame.f_code is target_code and threading.get_ident() in caught:
            prior_finished.set()
        return trace

    threading.settrace(trace)
    try:
        with _gateway(tmp_path, monkeypatch, served=('alpha',)) as (runner, homes, call):
            home = homes['alpha']
            old = runner.session_authorities.for_home(home)
            writer = threading.Thread(target=release_title.wait, name='auto-title', daemon=True)
            tg._UPGRADE_STORES[writer] = str(old.db.db_path)
            tg.start_title_upgrade(writer)
            release_title.set()
            writer.join(5)
            if schedule == 'not-imported':
                import agent
                # The retirement starts with it unimported. Restore BOTH the sys.modules entry and the package attribute
                # at teardown: a re-import rebinds ``agent.title_generator``, and a stale attribute would hand later
                # tests a module object the production ``sys.modules`` lookups never call (harness isolation).
                monkeypatch.setattr(agent, 'title_generator', tg)
                monkeypatch.delitem(sys.modules, 'agent.title_generator')
            armed.set()

            async def scenario():
                attempt = asyncio.create_task(rr.unserve_profile_runtime(runner, home))
                assert await asyncio.to_thread(at_fence.wait, 10)
                if schedule == 'not-imported':
                    assert imported_at_pause == [False], imported_at_pause  # the schedule really ran unimported
                if cancel_first:
                    attempt.cancel()
                    try:
                        await attempt
                    except asyncio.CancelledError:
                        pass
                else:
                    resume.set()
                    assert await attempt
                # Retiring authority stays registered; real serve finishes it and
                # initializes a new authority, including the production unfence.
                successor = await rr.serve_profile_runtime(runner, 'alpha', home)
                if schedule == 'not-imported':
                    gen = importlib.import_module('agent.title_generator')  # the successor imports it first
                else:
                    gen = tg
                root = os.path.normcase(os.path.realpath(home))
                before = root in gen._FENCED_HOMES
                resume.set()
                assert await asyncio.to_thread(prior_finished.wait, 10)
                ran = threading.Event()
                def title_model(*args, **kwargs):
                    ran.set()
                    return 'Successor model title witness'
                monkeypatch.setattr(gen, 'generate_title', title_model)
                from gateway.session_authorities import owner_scope
                successor.db.create_session('successor-title-probe', source='test')
                with owner_scope(successor):
                    new_title = gen.maybe_auto_title(successor.db, 'successor-title-probe',
                                                    'Investigate cancelled profile retirement')
                assert new_title is not None
                if new_title.ident is not None:
                    new_title.join(5)
                observed = dict(successor_is_new=successor is not old,
                                successor_retiring=getattr(successor, 'retiring', False),
                                reserved=process_ownership.owns(home),
                                fenced_after_serve=before,
                                fenced_after_cancelled_worker=root in gen._FENCED_HOMES,
                                successor_title_started=new_title.ident is not None,
                                successor_title_ran=ran.is_set(), schedule=schedule)
                print('OBSERVATION', json.dumps(observed))
                (tmp_path / 'observation.json').write_text(json.dumps(observed, indent=2))
                assert observed['successor_is_new'] and not observed['successor_retiring'], observed
                assert observed['reserved'] and not observed['fenced_after_serve'], observed
                assert not observed['fenced_after_cancelled_worker'], observed
                assert observed['successor_title_started'] and observed['successor_title_ran'], observed
                assert successor.db.get_session_title('successor-title-probe') == 'Successor model title witness', observed
            call(scenario())
    finally:
        release_title.set()
        resume.set()
        armed.clear()
        threading.settrace(None)
