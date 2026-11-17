"""Round-4 witnesses of review t_f95f82f7/run3092 (W1): the confirmation that turns a provisional
reservation into an admission is owed until it commits, and it follows the real durable authority
admission even when the HTTP request that asked for it was cancelled. Real aiohttp handlers, real
SQLite, a real separate process and a real session authority; only the agent body is replaced."""
import asyncio
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_loop_offload_io import _agent, _runs_app
from tests.gateway.test_loop_offload_io_admission import _REPO, _drain, _registry

_FOREIGN_READER = '''
import asyncio, json, sys
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_loop_offload_io import _agent, _runs_app

async def main(db, out, run_id, key):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(db)
    calls = []
    try:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            async with TestClient(TestServer(_runs_app(adapter))) as client:
                got = await client.get("/v1/runs/" + run_id)
                result = {"get": {"status": got.status, "body": await got.json()}, "posts": []}
                for _ in range(2):
                    posted = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": key})
                    result["posts"].append({"status": posted.status, "body": await posted.json()})
                for task in list(adapter._active_run_tasks.values()):
                    await task
                result["executions"] = len(calls)
                json.dump(result, open(out, "w"))
    finally:
        await adapter.disconnect()

asyncio.run(main(*sys.argv[1:]))
'''


async def _foreign_reader(tmp_path, run_id, key, label):
    """A GET and two exact retries from another OS process with its own adapter and connection."""
    script, out = tmp_path / "foreign_reader.py", tmp_path / f"foreign-{label}.json"
    script.write_text(_FOREIGN_READER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO), os.environ.get("PYTHONPATH")]))}
    done = await asyncio.to_thread(
        subprocess.run, [sys.executable, str(script), str(tmp_path / "idem.db"), str(out), run_id, key],
        cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def _admitted(path):
    conn = sqlite3.connect(str(path))
    try:
        return [tuple(r) for r in conn.execute("SELECT run_id, admitted, status_json FROM run_idempotency")]
    finally:
        conn.close()


def _deny_confirmations(store, budget):
    """SQLite refuses the next *budget* writes of the ``admitted`` column (None: every one)."""
    denied = []

    def _authorize(action, table, column, _db, _trigger):
        if action == sqlite3.SQLITE_UPDATE and table == "run_idempotency" and column == "admitted" and (
                budget is None or len(denied) < budget):
            denied.append(column)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    store._conn.set_authorizer(_authorize)
    return denied


def _replays(observed, run_id):
    return all(p["status"] == 202 and p["body"]["replayed"] and p["body"]["run_id"] == run_id
               for p in observed["posts"])


# ---------------------------------------------------------------------------------------------
# W1-CONFIRM-ERROR-05: a failed confirmation is owed, not lost
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [True, False], ids=["one-confirmation-error", "no-fault-control"])
async def test_one_failed_confirmation_never_strands_completed_work_from_another_process(tmp_path, caplog, fault):
    """Port of run3092 ``confirm-error``: SQLite refuses exactly one write of the ``admitted``
    column; the reservation and every status write are healthy. Once the run has completed, a
    separate process must GET it and replay it (twice), with no second execution."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    store = adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    denied = _deny_confirmations(store, 1 if fault else 0)
    calls = []
    try:
        with caplog.at_level("ERROR"), patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            async with TestClient(TestServer(_runs_app(adapter))) as client:
                posted = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "held"})
                run_id = (await posted.json())["run_id"]
                for task in list(adapter._active_run_tasks.values()):
                    await task
                await _drain(store)
                store._conn.set_authorizer(None)  # the fault is over
                local = await (await client.get(f"/v1/runs/{run_id}")).json()
                foreign = await _foreign_reader(tmp_path, run_id, "held", "confirm")
    finally:
        await adapter.disconnect()
    assert posted.status == 202 and calls == [1] and local["status"] == "completed"
    assert denied == (["admitted"] if fault else [])
    assert foreign["get"]["status"] == 200 and foreign["get"]["body"]["status"] == "completed", foreign
    assert _replays(foreign, run_id), foreign
    assert foreign["executions"] == 0
    assert [(r, a) for r, a, _ in _admitted(tmp_path / "idem.db")] == [(run_id, 1)]
    assert not _registry(adapter)
    if fault:
        assert "still owed" in caplog.text


@pytest.mark.asyncio
async def test_a_confirmation_failing_past_the_terminal_write_stays_owed_until_it_lands(tmp_path):
    """Every confirmation attempt fails until after the run's terminal status committed. Meanwhile
    another process answers a truthful retryable 503 (never a replay of a row it cannot know is
    admitted); once the store heals, the owner's next store operation lands the owed confirmation
    and the other process replays the one completed run."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    store = adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    denied = _deny_confirmations(store, None)
    calls = []
    try:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            async with TestClient(TestServer(_runs_app(adapter))) as client:
                posted = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "owed"})
                run_id = (await posted.json())["run_id"]
                for task in list(adapter._active_run_tasks.values()):
                    await task
                await _drain(store)
                attempts = len(denied)
                pending = await _foreign_reader(tmp_path, run_id, "owed", "while-failing")
                stored_while_failing = _admitted(tmp_path / "idem.db")
                store._conn.set_authorizer(None)
                other = await client.post("/v1/runs", json={"input": "y"}, headers={"Idempotency-Key": "other"})
                await _drain(store)
                for task in list(adapter._active_run_tasks.values()):
                    await task
                healed = await _foreign_reader(tmp_path, run_id, "owed", "healed")
    finally:
        await adapter.disconnect()
    assert calls == [1, 1] and other.status == 202
    assert attempts >= 2, "the confirmation was retried with the run's own later writes"
    assert [(r, a) for r, a, _ in stored_while_failing] == [(run_id, 0)]
    assert json.loads(stored_while_failing[0][2])["status"] == "completed"
    assert {p["status"] for p in pending["posts"]} == {503}
    assert {p["body"]["error"]["code"] for p in pending["posts"]} == {"idempotency_key_in_flight"}
    assert pending["executions"] == 0
    assert healed["get"]["status"] == 200 and healed["get"]["body"]["status"] == "completed"
    assert _replays(healed, run_id) and healed["executions"] == 0


# ---------------------------------------------------------------------------------------------
# W1-CANCEL-CONFIRM-06: the confirmation follows the durable admission, not its HTTP observer
# ---------------------------------------------------------------------------------------------


class _ObservedLock(asyncio.Lock):
    """The authority's real API-admission lock, reporting when an admission starts to wait on it."""

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.armed = False

    async def acquire(self):
        if self.armed:
            self.entered.set()
        return await super().acquire()


async def _authority_fixture(tmp_path, monkeypatch, executed):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB
    from tests.gateway.test_api_unknown_resolution import _adapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    executing = asyncio.Event()

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()["result"] = {"final_response": "ACK", "messages": []}
        executing.set()
        return "ACK"

    runner = SimpleNamespace(_session_db=db, session_store=sessions, _draining=False, _handle_message=answer)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="confirm-owner")

    def adapter_on_store():
        adapter = _adapter(runner, db)
        adapter._run_idempotency_store.close()
        adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
        return adapter
    first = adapter_on_store()
    runner._adapter_for_source = lambda source: first
    return SimpleNamespace(authority=authority, db=db, sessions=sessions, executing=executing,
                           adapter=first, adapter_on_store=adapter_on_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [True, False], ids=["observer-cancelled", "uncancelled-control"])
async def test_a_cancelled_requests_real_authority_admission_is_replayed_by_a_fresh_adapter(
        tmp_path, monkeypatch, cancelled):
    """Port of run3092 ``probe_authority``: the real shielded authority admission waits on the
    authority's real API-admission lock; the HTTP handler is cancelled meanwhile (or not, the
    control); the lock is released, the admission commits and its work runs once. A fresh adapter
    on the same authority and store must replay the completed run, not answer 503 forever."""
    from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _runs_client
    executed = []
    fx = await _authority_fixture(tmp_path, monkeypatch, executed)
    adapter, authority = fx.adapter, fx.authority
    lock = authority._api_admissions = _ObservedLock()
    await lock.acquire()
    lock.armed = True
    handlers = []
    limited = adapter._concurrency_limited_response

    def _record():
        handlers.append(asyncio.current_task())
        return limited()
    body = {"input": "AUTHORITY", "session_id": "api-session"}
    headers = {**_AUTH_HEADERS, "Idempotency-Key": "authority-held"}
    client, client2, other = _runs_client(adapter), None, None
    await client.start_server()
    try:
        with patch.object(adapter, "_concurrency_limited_response", side_effect=_record):
            first = asyncio.ensure_future(client.post("/v1/runs", json=body, headers=headers))
            await asyncio.wait_for(lock.entered.wait(), 30)
            run_id = _admitted(tmp_path / "idem.db")[0][0]
            if cancelled:
                handlers[0].cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await handlers[0]
            pending = list(authority._mutation_tasks)
            lock.release()
            await asyncio.gather(*pending)
            first_result = (await asyncio.gather(first, return_exceptions=True))[0]
            await asyncio.wait_for(fx.executing.wait(), 30)
            await authority.sessions["api-session"].task
            for task in list(adapter._active_run_tasks.values()):
                await task
            await _drain(adapter._run_idempotency_store)
            other = fx.adapter_on_store()
            client2 = _runs_client(other)
            await client2.start_server()
            replay = await client2.post("/v1/runs", json=body, headers=headers)  # before any GET
            replay_body = await replay.json()
            got = await client2.get(f"/v1/runs/{run_id}", headers=_AUTH_HEADERS)
            got_body = await got.json()
            again = await client2.post("/v1/runs", json=body, headers=headers)
            again_body = await again.json()
            await _drain(other._run_idempotency_store)
            stored = _admitted(tmp_path / "idem.db")
    finally:
        if lock.locked():
            lock.release()
        if client2 is not None:
            await client2.close()
        await client.close()
        for each in (adapter, other):
            if each is not None:
                each._response_store.close()
                await each._run_idempotency_store.aclose()
        authority.db.close()
        fx.sessions.close_all_db_handles()
    if cancelled:
        assert handlers[0].cancelled(), first_result
    else:
        assert first_result.status == 202
    assert executed == ["AUTHORITY"], "the admitted work runs exactly once"
    assert got.status == 200 and got_body["status"] == "completed", got_body
    for response, payload in ((replay, replay_body), (again, again_body)):
        assert response.status == 202, payload
        assert payload["replayed"] and payload["run_id"] == run_id and payload["status"] == "completed"
    assert [(r, a) for r, a, _ in stored] == [(run_id, 1)], "the durable admission confirmed the row"


@pytest.mark.asyncio
async def test_a_provisional_row_without_an_authority_admission_is_still_refused(tmp_path, monkeypatch):
    """The recognition never invents an admission: while the owner's real authority admission is
    still waiting (nothing admitted yet), a fresh adapter's exact retry is a retryable 503 and the
    row stays provisional; once the admission commits, the same retry replays the one run."""
    from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _runs_client
    executed = []
    fx = await _authority_fixture(tmp_path, monkeypatch, executed)
    adapter, authority = fx.adapter, fx.authority
    lock = authority._api_admissions = _ObservedLock()
    await lock.acquire()
    lock.armed = True
    body = {"input": "UNDECIDED", "session_id": "api-session"}
    headers = {**_AUTH_HEADERS, "Idempotency-Key": "undecided"}
    client, client2, other = _runs_client(adapter), None, None
    await client.start_server()
    try:
        first = asyncio.ensure_future(client.post("/v1/runs", json=body, headers=headers))
        await asyncio.wait_for(lock.entered.wait(), 30)
        other = fx.adapter_on_store()
        client2 = _runs_client(other)
        await client2.start_server()
        refused = await client2.post("/v1/runs", json=body, headers=headers)
        refused_body = await refused.json()
        stored_while_undecided = _admitted(tmp_path / "idem.db")
        lock.release()
        admitted = await first
        run_id = (await admitted.json())["run_id"]
        await asyncio.wait_for(fx.executing.wait(), 30)
        await authority.sessions["api-session"].task
        for task in list(adapter._active_run_tasks.values()):
            await task
        await _drain(adapter._run_idempotency_store)
        replay = await client2.post("/v1/runs", json=body, headers=headers)
        replay_body = await replay.json()
    finally:
        if lock.locked():
            lock.release()
        if client2 is not None:
            await client2.close()
        await client.close()
        for each in (adapter, other):
            if each is not None:
                each._response_store.close()
                await each._run_idempotency_store.aclose()
        authority.db.close()
        fx.sessions.close_all_db_handles()
    assert refused.status == 503 and refused_body["error"]["code"] == "idempotency_key_in_flight", refused_body
    assert [a for _, a, _ in stored_while_undecided] == [0]
    assert admitted.status == 202 and executed == ["UNDECIDED"]
    assert replay.status == 202 and replay_body["replayed"] and replay_body["run_id"] == run_id


@pytest.mark.asyncio
async def test_an_owed_confirmation_lands_in_order_and_at_the_closing_drain(tmp_path):
    """Store contract of the owed confirmation: it is retried before the run's later status writes
    (so a confirmation never trails the terminal status it vouches for once the store is healthy),
    and the closing drain lands one still owed."""
    store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    for key, run_id in (("ordered", "run_ordered"), ("drained", "run_drained")):
        await asyncio.wrap_future(store.submit(
            store.reserve, "s", key, "fp", run_id, {"status": "queued"}, provisional=True))
    denied = _deny_confirmations(store, 1)
    order = []
    store._conn.set_trace_callback(lambda sql: order.append(
        "confirm" if "SET admitted=1" in sql else "status" if "SET status_json" in sql else None))
    store.enqueue_confirmation("s", "ordered", "run_ordered")  # refused once: owed
    await asyncio.wrap_future(store.submit(lambda: None))
    store.enqueue_status("run_ordered", {"status": "completed"})  # retried first, then the status
    await asyncio.wrap_future(store.submit(lambda: None))
    after_status = _admitted(tmp_path / "idem.db")
    _deny_confirmations(store, None)
    store.enqueue_confirmation("s", "drained", "run_drained")
    await asyncio.wrap_future(store.submit(lambda: None))
    still_owed = dict((r, a) for r, a, _ in _admitted(tmp_path / "idem.db"))["run_drained"]
    store._conn.set_authorizer(None)
    await store.aclose()
    assert denied == ["admitted"]
    updates = [u for u in order if u]
    assert updates[:2] == ["confirm", "status"], updates  # the refused attempt never executed
    assert {r: (a, json.loads(s)["status"]) for r, a, s in after_status}["run_ordered"] == (1, "completed")
    assert still_owed == 0
    assert {r: a for r, a, _ in _admitted(tmp_path / "idem.db")} == {"run_ordered": 1, "run_drained": 1}
