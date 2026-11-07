"""Round-5 witnesses of review t_f95f82f7/run3748 W1-CONFIRM-REUSE-07: a confirmation vouches for
the run whose admission it records, never for whatever run its key names later. Run A is admitted
and completes while its confirmation keeps failing; terminal retention then releases the key and
a same-key request reserves run B, whose real authority admission is held undecided. Neither A's
owed confirmation (healed by an ordinary read, or by the closing drain) nor a recognition of A's
admission that was already in flight may admit B. Real aiohttp handlers, SQLite stores, retention
pruning and session authority; only the turn body is replaced, retention is accelerated on B's
store instance, and scheduling hooks hold the real admission lock and admission read."""
import asyncio
import sqlite3
import threading
from unittest.mock import patch

import pytest

from tests.gateway.test_loop_offload_io_confirm import _authority_fixture, _deny_confirmations, _ObservedLock

_TRIGGERS = ["owed-retry-read", "owed-retry-close", "recognition", "replacement-admitted-control"]


def _rows(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM run_idempotency")]
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", _TRIGGERS)
async def test_a_confirmation_never_admits_the_run_that_reused_its_key(tmp_path, monkeypatch, trigger):
    from gateway.platforms.api_server_authority_runs import run_admission
    from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _runs_client
    executed = []
    fx = await _authority_fixture(tmp_path, monkeypatch, executed)
    a = fx.adapter
    store_a = a._run_idempotency_store
    _deny_confirmations(store_a, None)  # A's confirmation keeps failing: owed
    clients, adapters, lock, gate, recognizing = [], [a], None, None, None

    async def client(adapter):
        c = _runs_client(adapter)
        await c.start_server()
        clients.append(c)
        return c
    body = {"input": "AUTHORITY", "session_id": "api-session"}
    headers = {**_AUTH_HEADERS, "Idempotency-Key": "reuse"}
    try:
        first = await (await client(a)).post("/v1/runs", json=body, headers=headers)
        a_id = (await first.json())["run_id"]
        await asyncio.wait_for(fx.executing.wait(), 30)
        await fx.authority.sessions["api-session"].task
        await asyncio.gather(*list(a._active_run_tasks.values()))
        await asyncio.wrap_future(store_a.submit(lambda: None))
        rows_a = _rows(tmp_path / "idem.db")
        b, c = fx.adapter_on_store(), fx.adapter_on_store()
        adapters += [b, c]
        cb, cc = await client(b), await client(c)
        if trigger == "recognition":  # C's recognition of A is in flight when the key is reused
            entered, release, loop = asyncio.Event(), threading.Event(), asyncio.get_running_loop()

            def _held_read(adapter, run_id):
                value = run_admission(adapter, run_id)
                if adapter is c and run_id == a_id:
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(30), "the admission read was never released"
                return value
            gate = patch("gateway.platforms.api_server_authority_runs.run_admission", side_effect=_held_read)
            gate.start()
            recognizing = asyncio.ensure_future(cc.post("/v1/runs", json=body, headers=headers))
            await asyncio.wait_for(entered.wait(), 30)
        b._run_idempotency_store.RETENTION_SECONDS = 0  # A's terminal row is now prunable
        lock = fx.authority._api_admissions = _ObservedLock()
        await lock.acquire()
        lock.armed = True
        second = asyncio.ensure_future(cb.post("/v1/runs", json=body, headers=headers))
        await asyncio.wait_for(lock.entered.wait(), 30)  # B reserved; its admission is undecided
        replacement = _rows(tmp_path / "idem.db")
        before = await cc.post("/v1/runs", json=body, headers=headers)
        before_body = await before.json()
        if trigger == "recognition":
            release.set()
            after = await asyncio.wait_for(recognizing, 30)
        else:
            store_a._conn.set_authorizer(None)  # A's store heals: its owed confirmation is retried
            if trigger == "owed-retry-close":
                await store_a.aclose()
            else:
                await asyncio.wrap_future(store_a.submit(store_a.owns_run, replacement[0]["scope"], a_id))
            after = await cc.post("/v1/runs", json=body, headers=headers)
        after_body = await after.json()
        after_rows = _rows(tmp_path / "idem.db")
        b_admitted_before_decision = run_admission(c, replacement[0]["run_id"]) is not None
        if trigger != "replacement-admitted-control":
            fx.authority.runner._draining = True  # B's real admission is refused
        lock.release()
        decided = await asyncio.wait_for(second, 30)
        decided_body = await decided.json()
        if trigger == "replacement-admitted-control":
            await fx.authority.sessions["api-session"].task
            await asyncio.gather(*list(b._active_run_tasks.values()))
            await asyncio.wrap_future(b._run_idempotency_store.submit(lambda: None))
            replay = await cc.post("/v1/runs", json=body, headers=headers)
            replay_body = await replay.json()
        final_rows = _rows(tmp_path / "idem.db")
    finally:
        if gate is not None:
            gate.stop()
        if lock is not None and lock.locked():
            lock.release()
        for each in clients:
            await each.close()
        for each in adapters:
            each._response_store.close()
            await each._run_idempotency_store.aclose()
        fx.authority.db.close()
        fx.sessions.close_all_db_handles()
    b_id = replacement[0]["run_id"]
    assert first.status == 202 and [(r["run_id"], r["admitted"]) for r in rows_a] == [(a_id, 0)]
    assert [(r["run_id"], r["admitted"]) for r in replacement] == [(b_id, 0)] and b_id != a_id
    assert before.status == 503 and before_body["error"]["code"] == "idempotency_key_in_flight"
    assert not b_admitted_before_decision
    # The stale confirmation (or recognition) of A never admits B: still provisional, still 503.
    assert [(r["run_id"], r["admitted"]) for r in after_rows] == [(b_id, 0)], after_rows
    assert after.status == 503 and after_body["error"]["code"] == "idempotency_key_in_flight", after_body
    if trigger == "replacement-admitted-control":
        # B's own admission still confirms B and replays it.
        assert decided.status == 202 and decided_body["run_id"] == b_id
        assert executed == ["AUTHORITY", "AUTHORITY"]
        assert [(r["run_id"], r["admitted"]) for r in final_rows] == [(b_id, 1)]
        assert replay.status == 202 and replay_body["replayed"] and replay_body["run_id"] == b_id
    else:
        # B is refused and released, never contradicting an acknowledgement; only A ever ran.
        assert decided.status == 503, decided_body
        assert final_rows == [] and executed == ["AUTHORITY"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [True, False], ids=["owed", "no-fault-control"])
@pytest.mark.parametrize("trigger", ["read", "close"])
async def test_an_owed_confirmation_never_admits_a_held_then_cancelled_replacement(tmp_path, trigger, fault):
    """Port of run3748's store ``probe-reuse``: no authority. A completes with its confirmation
    owed (or, the control, confirmed); B's reservation of the reused key commits on its store's
    worker and is held there while a fresh adapter asks (503). A's store heals and an ordinary read
    (or the closing drain) retries the owed confirmation: B must stay provisional and 503. B's
    handler is then cancelled and its release deletes B: nothing contradicts an acknowledgement."""
    import contextlib
    import json

    from aiohttp.test_utils import TestClient, TestServer

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    from tests.gateway.test_loop_offload_io import _agent, _runs_app
    db = tmp_path / "idem.db"

    def adapter():
        made = APIServerAdapter(PlatformConfig(enabled=True))
        made._run_idempotency_store.close()
        made._run_idempotency_store = RunIdempotencyStore(str(db))
        return made

    def rows():
        return [(r["run_id"], r["admitted"], json.loads(r["status_json"])["status"]) for r in _rows(db)]
    first, second, observer = adapter(), adapter(), adapter()
    old, new = first._run_idempotency_store, second._run_idempotency_store
    if fault:
        _deny_confirmations(old, None)
    calls, new_calls, observer_calls, handlers, seen = [], [], [], [], []
    parked, release = threading.Event(), threading.Event()
    headers = {"Idempotency-Key": "reuse"}
    try:
        async with contextlib.AsyncExitStack() as stack:
            stack.enter_context(patch.object(first, "_create_agent", return_value=_agent(calls)))
            c1 = await stack.enter_async_context(TestClient(TestServer(_runs_app(first))))
            a_id = (await (await c1.post("/v1/runs", json={"input": "x"}, headers=headers)).json())["run_id"]
            for task in list(first._active_run_tasks.values()):
                await task
            await asyncio.wrap_future(old.submit(lambda: None))
            initial = rows()
            new.RETENTION_SECONDS = -1  # A's terminal row is prunable by B's store
            reserve = new.reserve

            def held_reserve(*args, **kwargs):
                out = reserve(*args, **kwargs)
                seen.append(out)
                parked.set()
                assert release.wait(30), "the replacement's reservation was never released"
                return out
            new.reserve = held_reserve
            limited = second._concurrency_limited_response

            def record_handler():
                handlers.append(asyncio.current_task())
                return limited()
            stack.enter_context(patch.object(second, "_concurrency_limited_response", side_effect=record_handler))
            stack.enter_context(patch.object(second, "_create_agent", return_value=_agent(new_calls)))
            stack.enter_context(patch.object(observer, "_create_agent", return_value=_agent(observer_calls)))
            c2 = await stack.enter_async_context(TestClient(TestServer(_runs_app(second))))
            c3 = await stack.enter_async_context(TestClient(TestServer(_runs_app(observer))))
            pending = asyncio.ensure_future(c2.post("/v1/runs", json={"input": "x"}, headers=headers))
            assert await asyncio.to_thread(parked.wait, 30)
            b_id = seen[0][1]["run_id"]
            before = await c3.post("/v1/runs", json={"input": "x"}, headers=headers)
            old._conn.set_authorizer(None)
            if trigger == "read":
                await old.owns_run_async("irrelevant", a_id)
            else:
                await old.aclose()
            after_rows = rows()
            after = await c3.post("/v1/runs", json={"input": "x"}, headers=headers)
            after_body = await after.json()
            handlers[0].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handlers[0]
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            for _ in range(3):  # the release is scheduled from the reservation's completion
                await asyncio.wrap_future(new.submit(lambda: None))
                for _ in range(5):
                    await asyncio.sleep(0)
            final_rows = rows()
    finally:
        release.set()
        for each in (first, second, observer):
            await each.disconnect()
    assert seen[0][0] == "created" and b_id != a_id
    assert initial == [(a_id, 0 if fault else 1, "completed")]
    assert before.status == 503
    assert after_rows == [(b_id, 0, "queued")], after_rows
    assert after.status == 503 and after_body["error"]["code"] == "idempotency_key_in_flight", after_body
    assert final_rows == [] and calls == [1] and not new_calls and not observer_calls
