"""A4-E5-04 regression (E5 r6): ownership handed from an API-server worker to a nested writer after the
executor quiesce read the detached registry must still keep the shared SessionDB open.

Adopted from the two independent-review probes of A4r5 (run4687; /home/eric/scratch/a4-review-e5-2c7cafff41/tree/tests/
gateway/test_review_late_api_{nested,registration}.py), unchanged except this docstring and the observation file, written
under tmp_path. Real packaged API submission/counting (``_submit_api_worker``), real child entry/conversation/timeout and
deferred close, real ``_stop_quiesce_and_close_session_dbs``; the HTTP handler is omitted (``ctx.api_live=0`` models the
cancelled-handler case), provider work and scheduling are synthetic. Each test has an early-registration control and the
late-registration case: (1) the counted API starter begins its nested child after the detached snapshot; (2) the child is
already in its model wait before shutdown and the starter is paused just before its first registration.
"""
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.gateway.test_shutdown_async_delegation_writers import _make_agent, _probe_tools_only
from tests.gateway.restart_test_helpers import make_restart_runner
from tools import async_delegation as ad, detached_writers as dw
from tools import delegate_tool as dt, delegate_tool_child_run as cr
from gateway.platforms.api_server_runs import _submit_api_worker, api_worker_live_count
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from gateway import run_shutdown_budget as budget

@pytest.mark.asyncio
@pytest.mark.parametrize('late', [False, True], ids=['registered-before-snapshot','registered-after-snapshot'])
async def test_api_nested_transfer_must_not_escape_close(monkeypatch, tmp_path, late):
    from hermes_state_registry import close_all
    ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    child, db = _make_agent(monkeypatch, tmp_path)
    child.context_compressor.threshold_tokens = 10**9
    child._credential_pool = None; child._subagent_id = None
    parent_entered, begin_child, release_child, parent_done, child_closed = [threading.Event() for _ in range(5)]
    nested, writes, closes, result = [], [], [], []
    real_write, real_close, real_child_close, real_defer, real_join = db._execute_write, db.close, child.close, cr._defer_close_after_timeout, dw.join
    def model(_kwargs):
        assert release_child.wait(40)
        raise InterruptedError('review held nested model unwinds')
    child._interruptible_api_call = model
    def write(fn, *a, **kw):
        writes.append({'after_close': bool(closes), 'thread': threading.current_thread().name})
        return real_write(fn,*a,**kw)
    def close(*a,**kw):
        closes.append({'nested_live':real_join(0),'api_live':api_worker_live_count(),'child_closed':child_closed.is_set()})
        return real_close(*a,**kw)
    def childclose(*a,**kw):
        try:return real_child_close(*a,**kw)
        finally:child_closed.set()
    def defer(c,f):
        nested.append(f); return real_defer(c,f)
    def parent():
        parent_entered.set(); assert begin_child.wait(30)
        r=dt._run_single_child(0,'late nested child',child=child,parent_agent=SimpleNamespace())
        result.append(r);return r
    snapshots=[]
    def observed_join(timeout):
        n=real_join(timeout)
        snapshots.append({'detached_result':n,'api_live':api_worker_live_count()})
        if late:
            begin_child.set()
            assert parent_done.wait(30)
            assert api_worker_live_count()==0
        return n
    monkeypatch.setattr(db,'_execute_write',write);monkeypatch.setattr(db,'close',close)
    monkeypatch.setattr(child,'close',childclose);monkeypatch.setattr(cr,'_defer_close_after_timeout',defer)
    monkeypatch.setattr(cr._ChildRun,'seed_workspace',lambda self:None)
    monkeypatch.setattr(dt,'_get_child_timeout',lambda:5.0)
    runner,_=make_restart_runner();runner._session_db=db
    ctx=GatewayShutdownMixin._StopContext(deferred_count=lambda:0,started_at=time.monotonic())
    ctx.api_live=0
    with patch.object(child,'_cleanup_task_resources'), _probe_tools_only():
        future=_submit_api_worker(asyncio.get_running_loop(),parent)
        async def collect():
            try:await future
            finally:parent_done.set()
        collect_task=asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(parent_entered.wait,30)
            assert api_worker_live_count()==1
            if not late:
                begin_child.set();await collect_task
            monkeypatch.setattr(dw,'join',observed_join)
            await asyncio.to_thread(GatewayRunner._stop_quiesce_and_close_session_dbs,runner,30.,ctx)
            at_stop={'connection_closed':db._conn is None,'api_live':api_worker_live_count(),
                     'detached_live':real_join(0),'nested_done':nested[0].done() if nested else None}
        finally:
            begin_child.set();release_child.set()
            await collect_task
            if nested:
                await asyncio.to_thread(nested[0].result,30)
                assert await asyncio.to_thread(child_closed.wait,30)
    obs={'late':late,'snapshots':snapshots,'at_stop':at_stop,'closes':closes,
         'result_status':result[0]['status'],'post_close_writes':[w for w in writes if w['after_close']],
         'integrity':db._read_one('PRAGMA integrity_check')[0]}
    (tmp_path/('api-transfer-'+str(late)+'.json')).write_text(json.dumps(obs,indent=2))
    db.close();close_all();ad._reset_for_tests();dw._reset_for_tests();budget._reset_maintenance_for_tests()
    assert not at_stop['connection_closed'],obs
    assert not obs['post_close_writes'],obs

@pytest.mark.asyncio
@pytest.mark.parametrize('late',[False,True],ids=['registered-before-snapshot','registered-after-snapshot'])
async def test_started_api_child_registration_transfer(monkeypatch,tmp_path,late):
    from hermes_state_registry import close_all
    ad._reset_for_tests();dw._reset_for_tests();budget._reset_maintenance_for_tests()
    child,db=_make_agent(monkeypatch,tmp_path)
    child.context_compressor.threshold_tokens=10**9
    child._credential_pool=None;child._subagent_id=None
    registering,allow_registration,model_entered,release_child,parent_done,child_closed=[threading.Event() for _ in range(6)]
    nested,writes,closes,result,snapshots=[],[],[],[],[]
    real_write,real_close,real_child_close,real_defer,real_join,real_track=db._execute_write,db.close,child.close,cr._defer_close_after_timeout,dw.join,dw.track
    calls=[]
    def track():
        calls.append('track')
        if len(calls)==1:
            registering.set()
            assert allow_registration.wait(30)
        return real_track()
    def model(_kwargs):
        model_entered.set();assert release_child.wait(40)
        raise InterruptedError('already-running child unwinds after shutdown')
    child._interruptible_api_call=model
    def write(fn,*a,**kw):
        writes.append({'after_close':bool(closes),'thread':threading.current_thread().name})
        return real_write(fn,*a,**kw)
    def close(*a,**kw):
        closes.append({'nested_live':real_join(0),'api_live':api_worker_live_count(),'child_closed':child_closed.is_set()})
        return real_close(*a,**kw)
    def childclose(*a,**kw):
        try:return real_child_close(*a,**kw)
        finally:child_closed.set()
    def defer(c,f):
        nested.append(f);return real_defer(c,f)
    def parent():
        r=dt._run_single_child(0,'already running nested child',child=child,parent_agent=SimpleNamespace())
        result.append(r);return r
    def observed_join(timeout):
        n=real_join(timeout)
        snapshots.append({'detached_result':n,'api_live':api_worker_live_count(),'model_already_entered':model_entered.is_set()})
        if late:
            allow_registration.set()
            assert parent_done.wait(30)
            assert api_worker_live_count()==0
        return n
    monkeypatch.setattr(dw,'track',track)
    monkeypatch.setattr(db,'_execute_write',write);monkeypatch.setattr(db,'close',close)
    monkeypatch.setattr(child,'close',childclose);monkeypatch.setattr(cr,'_defer_close_after_timeout',defer)
    monkeypatch.setattr(cr._ChildRun,'seed_workspace',lambda self:None)
    monkeypatch.setattr(dt,'_get_child_timeout',lambda:5.0)
    runner,_=make_restart_runner();runner._session_db=db
    ctx=GatewayShutdownMixin._StopContext(deferred_count=lambda:0,started_at=time.monotonic());ctx.api_live=0
    with patch.object(child,'_cleanup_task_resources'),_probe_tools_only():
        future=_submit_api_worker(asyncio.get_running_loop(),parent)
        async def collect():
            try:await future
            finally:parent_done.set()
        collect_task=asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(registering.wait,30)
            assert await asyncio.to_thread(model_entered.wait,30)
            assert api_worker_live_count()==1 and real_join(0)==0
            cr._signal_child_stop(child,'synthetic gateway shutdown')
            if not late:
                allow_registration.set();await collect_task
            monkeypatch.setattr(dw,'join',observed_join)
            await asyncio.to_thread(GatewayRunner._stop_quiesce_and_close_session_dbs,runner,30.,ctx)
            at_stop={'connection_closed':db._conn is None,'api_live':api_worker_live_count(),'detached_live':real_join(0),'nested_done':nested[0].done() if nested else None}
        finally:
            allow_registration.set();release_child.set();await collect_task
            if nested:
                await asyncio.to_thread(nested[0].result,30)
                assert await asyncio.to_thread(child_closed.wait,30)
    obs={'late':late,'snapshots':snapshots,'at_stop':at_stop,'closes':closes,'result_status':result[0]['status'],
         'post_close_writes':[w for w in writes if w['after_close']],'integrity':db._read_one('PRAGMA integrity_check')[0]}
    (tmp_path/('api-registration-'+str(late)+'.json')).write_text(json.dumps(obs,indent=2))
    db.close();close_all();ad._reset_for_tests();dw._reset_for_tests();budget._reset_maintenance_for_tests()
    assert not at_stop['connection_closed'],obs
    assert not obs['post_close_writes'],obs
