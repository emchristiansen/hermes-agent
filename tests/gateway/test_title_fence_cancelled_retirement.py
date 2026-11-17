"""TITLE-CANCEL-01 regression (review6658 probe test_review_cancelled_title_fence.py, sha256 b51b3503..., adopted unchanged
except this docstring): a cancelled retirement's title fence must not fence the re-served successor. The trace pauses at
``join_title_upgrades_under``'s ``if fence and live == 0`` line (123), between the completed join and the fence publication."""
import asyncio
import json
import os
import threading
import sys

from tests.gateway.test_hosted_group_members import _gateway
import pytest


@pytest.mark.parametrize('cancel_first', [False, True], ids=['joined-control', 'cancelled-retirement'])
def test_cancelled_retirement_cannot_fence_a_successfully_reserved_successor(tmp_path, monkeypatch, cancel_first):
    from agent import title_generator as tg
    from gateway import run_runtime as rr
    from gateway.runtime_ownership import process_ownership

    # Trace only schedules an existing worker between its completed join and its
    # fence publication. No registry, retirement, title or serve semantics replaced.
    at_fence = threading.Event()
    resume = threading.Event()
    prior_finished = threading.Event()
    armed = threading.Event()
    release_title = threading.Event()
    caught = []
    def trace(frame, event, arg):
        if frame.f_code is not tg.join_title_upgrades_under.__code__:
            return None
        if (event == 'line' and armed.is_set() and not at_fence.is_set()
                and frame.f_lineno == 123):
            caught.append(threading.get_ident())
            at_fence.set()
            assert resume.wait(20)
        if event == 'return' and frame.f_code is tg.join_title_upgrades_under.__code__ and threading.get_ident() in caught:
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
            armed.set()

            async def scenario():
                attempt = asyncio.create_task(rr.unserve_profile_runtime(runner, home))
                assert await asyncio.to_thread(at_fence.wait, 10)
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
                root = os.path.normcase(os.path.realpath(home))
                before = root in tg._FENCED_HOMES
                resume.set()
                assert await asyncio.to_thread(prior_finished.wait, 10)
                ran = threading.Event()
                def title_model(*args, **kwargs):
                    ran.set()
                    return 'Successor model title witness'
                monkeypatch.setattr(tg, 'generate_title', title_model)
                from gateway.session_authorities import owner_scope
                successor.db.create_session('successor-title-probe', source='test')
                with owner_scope(successor):
                    new_title = tg.maybe_auto_title(successor.db, 'successor-title-probe',
                                                    'Investigate cancelled profile retirement')
                assert new_title is not None
                if new_title.ident is not None:
                    new_title.join(5)
                observed = dict(successor_is_new=successor is not old,
                                successor_retiring=getattr(successor, 'retiring', False),
                                reserved=process_ownership.owns(home),
                                fenced_after_serve=before,
                                fenced_after_cancelled_worker=root in tg._FENCED_HOMES,
                                successor_title_started=new_title.ident is not None,
                                successor_title_ran=ran.is_set())
                print('OBSERVATION', json.dumps(observed))
                (tmp_path / 'observation.json').write_text(json.dumps(observed, indent=2))
                assert observed['successor_title_ran'], observed
            call(scenario())
    finally:
        release_title.set()
        resume.set()
        armed.clear()
        threading.settrace(None)
