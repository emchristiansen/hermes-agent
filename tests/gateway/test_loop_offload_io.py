"""The event loop never waits for run-idempotency commits, completion-admission history reads or
the peer receipt re-read (W1/W2/R1 of the loop-stall report).

Every blocking boundary is probed with a deterministic barrier instead of wall-clock timing: on
the loop thread the probe records ``on-loop`` (it must not wait there: that wait IS the defect);
off the loop it proves the loop is free by having the loop run a callback while the boundary is
still parked, and it can hold the boundary until the test releases it.
"""
import asyncio
import contextlib
import functools
import hashlib
import json
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


class LoopBarrier:
    """Call inside a blocking boundary (see the module docstring). Build it on the loop thread."""

    def __init__(self, hold_timeout: float = 60.0):
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.seen: list[tuple[str, str]] = []
        self.parked = threading.Event()
        self.release = threading.Event()
        self.hold_timeout = hold_timeout

    def __call__(self, label: str, *, hold: bool = False) -> None:
        if threading.get_ident() == self.loop_thread:
            self.seen.append((label, "on-loop"))
            return
        ran = threading.Event()
        try:
            self.loop.call_soon_threadsafe(ran.set)
            responsive = ran.wait(60)
        except RuntimeError:  # loop already closed: nothing left to block
            responsive = True
        self.seen.append((label, "off-loop" if responsive else "loop-unresponsive"))
        if hold:
            self.parked.set()
            self.release.wait(self.hold_timeout)

    def on_loop(self) -> list[str]:
        return [label for label, where in self.seen if where != "off-loop"]

    async def parked_before(self, task: asyncio.Future | None = None, timeout: float = 30.0) -> bool:
        """True once a held boundary is parked off-loop; False if *task* finished first (the
        boundary ran on the loop and so could never park)."""
        waiter = asyncio.ensure_future(asyncio.to_thread(self.parked.wait, timeout))
        pending = {waiter} | ({task} if task is not None else set())
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if waiter in done:
            return waiter.result()
        self.parked.set()  # unblock the helper thread
        await waiter
        return False


class _ConnProbe:
    """The run store's sqlite3 connection, with every commit passing the barrier."""

    def __init__(self, conn, barrier: LoopBarrier):
        self._conn = conn
        self.barrier = barrier
        self.last_sql = ""
        self.hold_on = None  # one-shot predicate on the SQL the held commit follows
        self.fail_on = None
        self.statuses: list[str] = []

    def execute(self, sql, params=()):
        self.last_sql = sql
        if sql.startswith("UPDATE run_idempotency SET status_json"):
            self.statuses.append(json.loads(params[0])["status"])
        return self._conn.execute(sql, params)

    def commit(self):
        hold = self.hold_on is not None and self.hold_on(self.last_sql)
        if hold:
            self.hold_on = None
        self.barrier("commit:" + self.last_sql.split()[0], hold=hold)
        if self.fail_on is not None and self.fail_on(self.last_sql):
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    @property
    def total_changes(self):
        return self._conn.total_changes


_SCOPE = hashlib.sha256(b"default\0unauthenticated-test-listener").hexdigest()
def _INSERT(sql):
    return sql.startswith("INSERT")


def _STATUS(sql):
    return sql.startswith("UPDATE run_idempotency SET status_json")


def _probed_adapter(tmp_path, barrier):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    store = adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    probe = store._conn = _ConnProbe(store._conn, barrier)
    return adapter, store, probe


def _runs_app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _agent(calls):
    agent = MagicMock()

    def run(**_kwargs):
        calls.append(1)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


def _committed_run_ids(tmp_path, key):
    conn = sqlite3.connect(str(tmp_path / "idem.db"))
    try:
        return [row[0] for row in conn.execute(
            "SELECT run_id FROM run_idempotency WHERE idempotency_key=?", (key,))]
    finally:
        conn.close()


async def _flushed_status(store, run_id):
    """The stored status once every write published before this call has landed."""
    record = await asyncio.to_thread(store.status_for_run, _SCOPE, run_id)
    return record and record["status"]


async def _settled(client, run_id):
    for _ in range(400):
        body = await (await client.get(f"/v1/runs/{run_id}")).json()
        if body.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} never settled")


# ---------------------------------------------------------------------------------------------
# W1 - run-idempotency store
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyed_run_lifecycle_never_commits_on_the_event_loop(tmp_path):
    barrier = LoopBarrier()
    adapter, store, _probe = _probed_adapter(tmp_path, barrier)
    calls: list[int] = []
    headers = {"Idempotency-Key": "lifecycle"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            first = await client.post("/v1/runs", json={"input": "hello"}, headers=headers)
            run_id = (await first.json())["run_id"]
            assert (await _settled(client, run_id))["status"] == "completed"
            replay = await client.post("/v1/runs", json={"input": "hello"}, headers=headers)
            replay_body = await replay.json()
            conflict = await client.post("/v1/runs", json={"input": "other"}, headers=headers)
    assert first.status == replay.status == 202
    assert replay_body["run_id"] == run_id and replay.headers["Idempotency-Replayed"] == "true"
    assert conflict.status == 409
    assert calls == [1]
    assert (await _flushed_status(store, run_id))["status"] == "completed"
    commits = [label for label, _ in barrier.seen]
    assert len(commits) >= 4, commits  # lookup, reserve, running, terminal (+ replay lookups)
    assert barrier.on_loop() == [], f"run-store commits ran on the event loop: {barrier.on_loop()}"


@pytest.mark.asyncio
async def test_reservation_commits_before_admission_and_concurrent_retries_replay_it(tmp_path):
    barrier = LoopBarrier()
    adapter, _store, probe = _probed_adapter(tmp_path, barrier)
    calls: list[int] = []
    admitted_with_row: list[list[str]] = []
    activate = adapter._activate_admitted_request

    def _activate():
        admitted_with_row.append(_committed_run_ids(tmp_path, "held"))
        return activate()

    headers = {"Idempotency-Key": "held"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)), \
                patch.object(adapter, "_activate_admitted_request", side_effect=_activate):
            probe.hold_on = _INSERT
            first = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            parked = await barrier.parked_before(first)
            assert parked, f"the reservation commit ran on the event loop: {barrier.on_loop()}"
            # Parked in the reservation's commit: nothing is admitted, nothing answered, and the
            # loop still serves a same-key retry and a conflicting payload (both queue behind it).
            assert not first.done() and admitted_with_row == [] and calls == []
            assert _committed_run_ids(tmp_path, "held") == []
            retry = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            conflict = asyncio.ensure_future(client.post("/v1/runs", json={"input": "y"}, headers=headers))
            await asyncio.sleep(0)
            assert not retry.done() and not conflict.done()
            barrier.release.set()
            first, retry, conflict = await asyncio.gather(first, retry, conflict)
            first_body, retry_body = await first.json(), await retry.json()
            await _settled(client, first_body["run_id"])
    assert first.status == retry.status == 202
    assert first_body["replayed"] is False and retry_body["replayed"] is True
    assert retry_body["run_id"] == first_body["run_id"]
    assert conflict.status == 409
    assert admitted_with_row == [[first_body["run_id"]]], "admitted before the reservation committed"
    assert calls == [1]
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_cancelled_admission_releases_its_reservation(tmp_path):
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    calls: list[int] = []
    handlers: list[asyncio.Task] = []
    limited = adapter._concurrency_limited_response

    def _record_handler():
        handlers.append(asyncio.current_task())
        return limited()

    headers = {"Idempotency-Key": "cancelled"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)), \
                patch.object(adapter, "_concurrency_limited_response", side_effect=_record_handler):
            probe.hold_on = _INSERT
            request = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            assert await barrier.parked_before(request), barrier.on_loop()
            handlers[0].cancel()
            for _ in range(200):
                if handlers[0].done():
                    break
                await asyncio.sleep(0.01)
            # The handler is gone while its reservation is still committing.
            assert handlers[0].cancelled()
            assert not any(s.get("status") == "queued" for s in adapter._run_statuses.values())
            barrier.release.set()
            with contextlib.suppress(Exception):
                await request
            # Once the reservation's commit is done its release is queued on the store worker;
            # a lookup queued after that sees the key free again.
            await asyncio.wrap_future(store.submit(lambda: None))
            outcome, _ = await asyncio.wrap_future(store.submit(store.lookup, _SCOPE, "cancelled", "-"))
            assert outcome == "missing"
            fresh = await client.post("/v1/runs", json={"input": "x"}, headers=headers)
            body = await fresh.json()
            await _settled(client, body["run_id"])
    assert fresh.status == 202 and body["replayed"] is False
    assert calls == [1], "only the admitted retry ran"


@pytest.mark.asyncio
async def test_overlapping_same_key_retry_never_replays_a_cancelled_reservation(tmp_path, monkeypatch):
    """Review t_f95f82f7/run2676 W1-CANCEL-REPLAY-01, through the real POST/GET handlers and SQLite.

    The first request's reservation commit is parked; a same-key retry arrives and is in flight
    (it looked the key up, or joined the pending reservation); the first handler is cancelled; the
    commit is released, so the cancellation releases the row. The retry must never be acknowledged
    as a replay of that released run (202 for a run that GET reports 404 and nobody executes)."""
    from gateway.platforms import api_server_runs
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    loop = asyncio.get_running_loop()
    calls: list[int] = []
    handlers: list[asyncio.Task] = []
    retry_in_flight = asyncio.Event()
    limited = adapter._concurrency_limited_response
    submit = store.submit

    def _record_handler():
        handlers.append(asyncio.current_task())
        return limited()

    def _record_submit(fn, *args, **kwargs):
        future = submit(fn, *args, **kwargs)
        name = getattr(fn.func if isinstance(fn, functools.partial) else fn, "__name__", "")
        if name == "lookup" and barrier.parked.is_set():
            loop.call_soon_threadsafe(retry_in_flight.set)  # the retry's lookup is queued
        return future
    store.submit = _record_submit
    join = getattr(api_server_runs, "_join_key_reservation", None)

    async def _record_join(entry):
        retry_in_flight.set()  # the retry is waiting for the pending reservation
        return await join(entry)
    monkeypatch.setattr(api_server_runs, "_join_key_reservation", _record_join, raising=False)

    headers = {"Idempotency-Key": "overlap"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)), \
                patch.object(adapter, "_concurrency_limited_response", side_effect=_record_handler):
            probe.hold_on = _INSERT
            first = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            assert await barrier.parked_before(first), f"the reservation commit ran on the loop: {barrier.on_loop()}"
            retry = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            await asyncio.wait_for(retry_in_flight.wait(), 30)
            handlers[0].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handlers[0]
            barrier.release.set()
            _first, response = await asyncio.gather(first, retry, return_exceptions=True)
            assert not isinstance(response, BaseException), repr(response)
            body = await response.json()
            await asyncio.wrap_future(submit(lambda: None))  # every queued store job has landed
            for task in list(adapter._active_run_tasks.values()):
                await task
            if response.status == 202:
                polled = await client.get(f"/v1/runs/{body['run_id']}")
                polled_body = await polled.json()
    rows = [tuple(r) for r in sqlite3.connect(str(tmp_path / "idem.db")).execute(
        "SELECT idempotency_key, run_id FROM run_idempotency")]
    if response.status == 202:
        # Acknowledged: it names a run that exists, ran exactly once and owns the key durably.
        assert polled.status == 200, (body, polled_body)
        assert polled_body["status"] == "completed"
        assert calls == [1]
        assert rows == [("overlap", body["run_id"])]
        assert body["replayed"] is False, "a replay of the cancelled request"
    else:
        # Not admitted: a truthful retryable answer, nothing executed, nothing reserved.
        assert response.status == 503 and body["error"]["code"] == "idempotency_key_in_flight", body
        assert calls == [] and rows == []
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_a_retry_joining_an_admitted_reservation_replays_it(tmp_path):
    """The uncancelled control of the witness above: the joined request replays the admitted run."""
    barrier = LoopBarrier()
    adapter, _store, probe = _probed_adapter(tmp_path, barrier)
    calls: list[int] = []
    headers = {"Idempotency-Key": "joined"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            probe.hold_on = _INSERT
            first = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
            assert await barrier.parked_before(first)
            retries = [asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers=headers))
                       for _ in range(3)]
            for _ in range(20):
                await asyncio.sleep(0)
            barrier.release.set()
            responses = await asyncio.gather(first, *retries)
            bodies = [await r.json() for r in responses]
            await _settled(client, bodies[0]["run_id"])
    assert {r.status for r in responses} == {202}
    assert {b["run_id"] for b in bodies} == {bodies[0]["run_id"]}
    assert [b["replayed"] for b in bodies] == [False, True, True, True]
    assert calls == [1]


@pytest.mark.asyncio
async def test_bounded_cleanup_cancellation_never_drops_the_store_close(tmp_path):
    """Review t_f95f82f7/run2676 W1-CLOSE-CANCEL-02 through the real deadline-bounded consumer."""
    from gateway.run_adapters import GatewayAdapterLifecycleMixin
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    raw = probe._conn
    run_id = "run_bounded"
    status = adapter._set_run_status(run_id, "running")
    await asyncio.to_thread(store.reserve, _SCOPE, "bounded", "fp", run_id, dict(status))
    adapter._run_idempotency_ids.add(run_id)
    probe.hold_on = _STATUS
    adapter._set_run_status(run_id, "interrupted", error="shutdown")
    assert await barrier.parked_before()
    # The real teardown bound gives up and cancels disconnect() while the commit is still held.
    completed = await GatewayAdapterLifecycleMixin()._await_adapter_cleanup_with_timeout(adapter.disconnect(), 0.2)
    assert completed is False
    for _ in range(5):
        await asyncio.sleep(0)  # deliver the detached child's cancellation
    barrier.release.set()
    # A later cleanup joins the same close and returns only once it really happened.
    await asyncio.wait_for(adapter.disconnect(), 60)
    with pytest.raises(sqlite3.ProgrammingError):
        raw.execute("SELECT 1")
    await store.aclose()  # idempotent: joins the finished close
    await asyncio.to_thread(store._executor.shutdown, wait=True)  # off-loop join of the worker
    assert not any(t.is_alive() for t in store._executor._threads)
    conn = sqlite3.connect(str(tmp_path / "idem.db"))
    try:
        stored = json.loads(conn.execute("SELECT status_json FROM run_idempotency").fetchone()[0])
    finally:
        conn.close()
    assert stored["status"] == "interrupted"
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_close_is_one_operation_every_caller_joins_including_its_error(tmp_path):
    barrier = LoopBarrier()
    # Success: concurrent and repeated closes (async and sync) all observe the one close.
    (tmp_path / "ok").mkdir()
    _adapter, store, probe = _probed_adapter(tmp_path / "ok", barrier)
    raw = probe._conn
    await asyncio.gather(store.aclose(), store.aclose())
    await store.aclose()
    await asyncio.to_thread(store.close)
    with pytest.raises(sqlite3.ProgrammingError):
        raw.execute("SELECT 1")
    # Failure: the close's error is what every caller observes, never "already closed".
    (tmp_path / "err").mkdir()
    _adapter2, failing, probe2 = _probed_adapter(tmp_path / "err", barrier)
    real_close = probe2.close

    def _broken_close():
        real_close()
        raise sqlite3.OperationalError("close failed")
    probe2.close = _broken_close
    with pytest.raises(sqlite3.OperationalError, match="close failed"):
        await failing.aclose()
    with pytest.raises(sqlite3.OperationalError, match="close failed"):
        await failing.aclose()
    with pytest.raises(sqlite3.OperationalError, match="close failed"):
        await asyncio.to_thread(failing.close)
    for closed in (store, failing):
        await asyncio.to_thread(closed._executor.shutdown, wait=True)
        assert not any(t.is_alive() for t in closed._executor._threads)
    with pytest.raises(RuntimeError):
        failing.enqueue_status("run_x", {"status": "failed"})
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_status_snapshots_land_in_publication_order_across_threads(tmp_path):
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    run_id = "run_order"
    status = adapter._set_run_status(run_id, "queued")
    adapter._run_owners[run_id] = _SCOPE
    await asyncio.to_thread(store.reserve, _SCOPE, "order", "fp", run_id, dict(status))
    adapter._run_idempotency_ids.add(run_id)

    probe.hold_on = _STATUS
    adapter._set_run_status(run_id, "running")
    assert await barrier.parked_before(), f"the status commit ran on the event loop: {barrier.on_loop()}"
    # The worker is parked inside the "running" commit (holding the store lock). An agent thread's
    # approval prompt and the loop's terminal status both publish without waiting for it.
    await asyncio.to_thread(adapter._set_run_status, run_id, "waiting_for_approval",
                            last_event="approval.request", approval={"request_id": "a"})
    adapter._set_run_status(run_id, "completed", output="done")
    assert not barrier.release.is_set()
    barrier.release.set()
    assert (await _flushed_status(store, run_id))["status"] == "completed"
    published = ["running", "waiting_for_approval", "completed"]
    written = probe.statuses
    assert written[-1] == "completed" and written[0] == "running", written
    assert [published.index(s) for s in written] == sorted(published.index(s) for s in written), written
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_a_status_publication_never_waits_for_another_threads_commit(tmp_path):
    """The loop used to futex-wait on the store lock a worker thread held across its fsync."""
    barrier = LoopBarrier(hold_timeout=30)
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    run_id = "run_lock"
    status = adapter._set_run_status(run_id, "queued")
    await asyncio.to_thread(store.reserve, _SCOPE, "lock", "fp", run_id, dict(status))
    adapter._run_idempotency_ids.add(run_id)
    probe.hold_on = _STATUS
    holder = threading.Thread(target=store.update_status, args=(run_id, {"status": "running"}))
    holder.start()
    assert await barrier.parked_before()
    adapter._set_run_status(run_id, "waiting_for_approval", approval={"request_id": "b"})  # loop thread
    returned_while_held = holder.is_alive()
    barrier.release.set()
    holder.join(30)
    assert returned_while_held, "the loop thread waited for a commit another thread held"
    assert (await _flushed_status(store, run_id))["status"] == "waiting_for_approval"


def test_unchanged_retention_writes_nothing_and_keeps_its_boolean(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    try:
        horizon = time.time() + 3600
        store.reserve(_SCOPE, "k", "fp", "run_r", {"status": "running"}, retention_until=horizon)
        conn = store._conn
        before = conn.total_changes
        assert store.extend_retention(_SCOPE, "run_r", horizon - 60) is True  # never lowered
        assert store.extend_retention(_SCOPE, "run_r", horizon) is True
        assert store.lookup(_SCOPE, "k", "fp", retention_until=horizon)[0] == "reused"
        assert store.status_for_run(_SCOPE, "run_r", retention_until=horizon)["status"]["status"] == "running"
        assert conn.total_changes == before, "an unchanged retention horizon rewrote the row"
        assert store.extend_retention(_SCOPE, "run_r", horizon + 60) is True
        assert conn.total_changes == before + 1
        stored = conn.execute("SELECT retention_until FROM run_idempotency WHERE run_id='run_r'").fetchone()[0]
        assert stored == horizon + 60
        assert store.extend_retention(_SCOPE, "run_missing", horizon + 120) is False
        assert store.extend_retention("other-scope", "run_r", horizon + 120) is False
        assert store.extend_retention(_SCOPE, "run_r", 0) is False
        assert conn.execute("SELECT retention_until FROM run_idempotency WHERE run_id='run_r'").fetchone()[0] \
            == horizon + 60
    finally:
        store.close()


@pytest.mark.asyncio
async def test_disconnect_drains_queued_statuses_without_blocking_the_loop(tmp_path, caplog):
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    run_id = "run_shutdown"
    status = adapter._set_run_status(run_id, "running")
    await asyncio.to_thread(store.reserve, _SCOPE, "shutdown", "fp", run_id, dict(status))
    adapter._run_idempotency_ids.add(run_id)
    probe.hold_on = _STATUS
    from gateway.platforms import api_server_runs
    api_server_runs._mark_shutdown_interrupted_runs(adapter, {run_id})
    assert await barrier.parked_before(), f"the shutdown status commit ran on the loop: {barrier.on_loop()}"
    disconnect = asyncio.ensure_future(adapter.disconnect())
    for _ in range(20):  # yield to it repeatedly: its close is queued behind the parked commit
        await asyncio.sleep(0)
    assert not disconnect.done(), "disconnect closed the store before its queued status landed"
    barrier.release.set()
    await asyncio.wait_for(disconnect, 60)
    conn = sqlite3.connect(str(tmp_path / "idem.db"))
    try:
        stored = json.loads(conn.execute(
            "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()[0])
    finally:
        conn.close()
    assert stored["status"] == "interrupted"
    assert not any(t.is_alive() for t in store._executor._threads), "the store worker outlived close"
    # After close a publication is refused and logged, never raised into the caller.
    with caplog.at_level("ERROR"):
        adapter._set_run_status(run_id, "failed", error="late")
    assert "failed to persist idempotent run status" in caplog.text
    with pytest.raises(RuntimeError):
        store.enqueue_status(run_id, {"status": "failed"})
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_store_errors_keep_their_caller_contracts(tmp_path, caplog):
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    # A failed reservation commit is the handler's error, exactly as before (HTTP 500).
    probe.fail_on = _INSERT
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent([])):
            response = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "boom"})
    assert response.status == 500
    assert _committed_run_ids(tmp_path, "boom") == []
    # A failed status commit is logged, never raised into the publisher.
    probe.fail_on = _STATUS
    run_id = "run_err"
    status = adapter._set_run_status(run_id, "queued")
    await asyncio.to_thread(store.reserve, _SCOPE, "err", "fp", run_id, dict(status))
    adapter._run_idempotency_ids.add(run_id)
    with caplog.at_level("ERROR"):
        assert adapter._set_run_status(run_id, "completed")["status"] == "completed"
        await _flushed_status(store, run_id)
    assert "failed to persist idempotent run status" in caplog.text
    # The store stays usable and the next publication lands.
    probe.fail_on = None
    adapter._set_run_status(run_id, "completed", output="again")
    stored = await _flushed_status(store, run_id)
    assert stored["status"] == "completed" and stored["output"] == "again"


@pytest.mark.asyncio
async def test_dead_owner_hydration_commits_interrupted_off_loop(tmp_path):
    seed = RunIdempotencyStore(str(tmp_path / "idem.db"))
    seed.reserve(_SCOPE, "stale", "fp", "run_stale", {"run_id": "run_stale", "status": "running"},
                 owner_pid=999_999_999, owner_started=1)
    seed.close()
    barrier = LoopBarrier()
    adapter, _store, probe = _probed_adapter(tmp_path, barrier)
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        body = await (await client.get("/v1/runs/run_stale")).json()
        again = await (await client.get("/v1/runs/run_stale")).json()
    assert body["status"] == again["status"] == "interrupted"
    assert probe.statuses == ["interrupted"]
    assert barrier.on_loop() == []


# ---------------------------------------------------------------------------------------------
# W2 - completion_admission
# ---------------------------------------------------------------------------------------------


async def _owner(tmp_path, monkeypatch):
    from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
    from gateway.platforms.event import MessageEvent
    runner, _adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text="human", source=source, message_id="human"))
    key = runner.session_store._generate_session_key(source)
    return runner, authority, key, receipt.ref.session_id


def _identity(runner, evt):
    from gateway.session_automation import producer_identity
    return producer_identity(runner, evt)


def _seed(authority, root, key, request_id, payload, *, terminal=True):
    from hermes_state_runtime import admit_session_input
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id="automation:" + key,
                              session_id=root, request_id=request_id, payload=payload)
    if terminal:
        authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status='terminal', outcome='completed' WHERE admission_id=?",
            (row["admission_id"],)))
    return row


def _native(identity, identities=None, text="done"):
    automation = {"identity": identity, "owner": "x"}
    if identities is not None:
        automation["identities"] = sorted(set(identities))
    return {"text": text, "native_text_v1": {"automation": automation}}


def _local(identity, identities=None, text="done"):
    descriptor = {"identity": identity, "owner": "x", "route": "r", "target": "t"}
    if identities is not None:
        descriptor["identities"] = sorted(set(identities))
    return {"text": text, "local_automation_v1": descriptor}


def _delegation(key, name):
    return {"type": "async_delegation", "session_key": key, "delegation_id": name}


@pytest.mark.asyncio
async def test_completion_admission_matches_exact_parsed_identity_in_both_shapes(tmp_path, monkeypatch):
    from gateway.session_automation import completion_admission
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    primary, sibling, unicode_evt, lone, missing = (
        _delegation(key, n) for n in ("prim-1", "sib-2", "café/\"q\"", "lone-4", "missing-5"))
    ids = {name: _identity(runner, evt) for name, evt in
           (("p", primary), ("s", sibling), ("u", unicode_evt), ("l", lone), ("m", missing))}
    # Unrelated history whose text names an identity's verbatim run: a prefilter hit, never a match.
    for i in range(40):
        _seed(authority, root, key, f"noise-{i}", _native(f"noise-{i}", text=f"mentions prim-1 sib-2 {i}"))
    batch = _seed(authority, root, key, ids["p"], _native(ids["p"], [ids["p"], ids["s"], ids["u"]]))
    later_dup = _seed(authority, root, key, ids["s"], _local(ids["s"]))  # a later row naming the sibling
    local_row = _seed(authority, root, key, ids["l"], _local(ids["l"]))
    assert completion_admission(runner, primary)["admission_id"] == batch["admission_id"]
    # Coalesced sibling: the earliest row carrying it, not the later exact-primary duplicate.
    assert completion_admission(runner, sibling)["admission_id"] == batch["admission_id"]
    assert completion_admission(runner, unicode_evt)["admission_id"] == batch["admission_id"]
    assert completion_admission(runner, lone)["admission_id"] == local_row["admission_id"]
    assert completion_admission(runner, missing) is None  # a miss is "not admitted", never success
    assert later_dup["seq"] > batch["seq"]
    # Isolation: an unknown route has no ledger, and another live route's ledger never answers
    # for this one (nor this one's for it).
    assert completion_admission(runner, _delegation("agent:main:discord:dm:unknown", "prim-1")) is None
    from gateway.platforms.event import MessageEvent
    from tests.gateway.test_automation_compression_media import _admit_human
    other_source = runner.adapters[next(iter(runner.adapters))].build_source(
        chat_id="43", chat_type="dm", user_id="42")
    other = await _admit_human(runner, authority, MessageEvent(text="h2", source=other_source, message_id="h2"))
    other_key = runner.session_store._generate_session_key(other_source)
    assert other_key != key and other.ref.session_id != root
    only_there = _delegation(other_key, "only-there-6")
    there = _seed(authority, other.ref.session_id, other_key, _identity(runner, only_there),
                  _native(_identity(runner, only_there)))
    assert completion_admission(runner, only_there)["admission_id"] == there["admission_id"]
    assert completion_admission(runner, _delegation(key, "only-there-6")) is None
    assert completion_admission(runner, _delegation(other_key, "prim-1")) is None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_completion_admission_decodes_only_prefiltered_rows_without_a_sort(tmp_path, monkeypatch):
    import hermes_state_runtime
    from gateway.session_automation import completion_admission
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    for i in range(120):
        _seed(authority, root, key, f"history-{i}", _native(f"history-{i}", text="x" * 2000))
    wanted = _delegation(key, "needle-delegation-0042")
    row = _seed(authority, root, key, _identity(runner, wanted), _native(_identity(runner, wanted)))
    decoded = []
    real_row = hermes_state_runtime._row
    monkeypatch.setattr(hermes_state_runtime, "_row", lambda r: decoded.append(1) or real_row(r))
    assert completion_admission(runner, wanted)["admission_id"] == row["admission_id"]
    assert completion_admission(runner, _delegation(key, "never-admitted-9999")) is None
    assert len(decoded) <= 2, f"decoded {len(decoded)} rows of a {122}-row history"
    with authority.db._read_ctx() as conn:
        plan = " ".join(str(tuple(r)) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM session_admissions "
            "WHERE target_session_id=? AND instr(payload_json, ?) > 0", (root, "needle")))
    assert "TEMP B-TREE" not in plan and "session_admissions_pending" in plan, plan
    await asyncio.sleep(0)


def _spy_admission(monkeypatch, barrier):
    from gateway import session_automation
    real = session_automation.completion_admission

    def _spy(runner, evt):
        barrier("completion_admission")
        return real(runner, evt)
    monkeypatch.setattr(session_automation, "completion_admission", _spy)


@pytest.mark.asyncio
async def test_delivery_and_lost_ack_replay_run_the_lookup_off_loop(tmp_path, monkeypatch):
    from tests.gateway.test_completion_admission import pending
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    barrier = LoopBarrier()
    _spy_admission(monkeypatch, barrier)
    event = dict(pending(key, "offloop-delivery"), parent_session_id=root)
    assert await runner._deliver_async_delegation_group([event]) is True
    admitted = [r for r in list_session_admissions(authority.db, session_id=root, pending_only=False)
                if r["request_id"] == _identity(runner, event)]
    assert len(admitted) == 1
    # Lost ACK: the same completion again is acknowledged from the committed row, not re-admitted.
    replay = dict(pending(key, "offloop-delivery"), parent_session_id=root)
    assert await runner._deliver_async_delegation_group([replay]) is True
    assert [r for r in list_session_admissions(authority.db, session_id=root, pending_only=False)
            if r["request_id"] == _identity(runner, event)] == admitted
    assert len(barrier.seen) >= 3, barrier.seen  # group check + delivery check, then the replay
    assert barrier.on_loop() == [], f"completion_admission ran on the event loop: {barrier.seen}"
    await asyncio.sleep(0)


def _batch_entry(loop, session_id, started_at, key):
    evt = {"type": "completion", "session_id": session_id, "started_at": started_at, "session_key": key,
           "output": "ok", "exit_code": 0}
    return (f"[{session_id} done]", evt, loop.create_future())


@pytest.mark.asyncio
async def test_batch_flush_checks_every_identity_off_loop_and_settles_on_cancel(tmp_path, monkeypatch):
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    barrier = LoopBarrier()
    _spy_admission(monkeypatch, barrier)
    # Both identities are already admitted (coalesced primary + sibling): nothing is delivered again.
    entries = [_batch_entry(loop, "proc_a", 1.0, key), _batch_entry(loop, "proc_b", 2.0, key)]
    ids = [_identity(runner, evt) for _t, evt, _f in entries]
    _seed(authority, root, key, ids[0], _native(ids[0], ids))
    runner._completion_notification_batch_window = 0
    runner._completion_notification_batches[("k",)] = list(entries)
    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_deliver_completion_notification", deliver)
    await runner._flush_process_completion_batch(("k",))
    assert [f.result() for _t, _e, f in entries] == [True, True]
    deliver.assert_not_awaited()
    assert len(barrier.seen) == 2 and barrier.on_loop() == [], barrier.seen

    # Shutdown cancels a flush parked in the lookup: its waiters settle as retryable at once.
    held = LoopBarrier()
    from gateway import session_automation
    real = session_automation.completion_admission

    def _parked(r, evt):
        held("completion_admission", hold=True)
        return real(r, evt)
    monkeypatch.setattr(session_automation, "completion_admission", _parked)
    entries = [_batch_entry(loop, "proc_c", 3.0, key)]
    runner._completion_notification_batches[("k2",)] = list(entries)
    flush = asyncio.ensure_future(runner._flush_process_completion_batch(("k2",)))
    assert await held.parked_before(flush), f"the batch lookup ran on the event loop: {held.seen}"
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert entries[0][2].result() is False
    held.release.set()
    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_errors_keep_each_callsite_contract(tmp_path, monkeypatch, caplog):
    runner, _authority, key, _root = await _owner(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    from gateway import session_automation

    def _broken(_runner, _evt):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(session_automation, "completion_admission", _broken)
    entries = [_batch_entry(loop, "proc_e", 5.0, key)]
    runner._completion_notification_batch_window = 0
    runner._completion_notification_batches[("err",)] = list(entries)
    with caplog.at_level("ERROR"):
        await runner._flush_process_completion_batch(("err",))
    assert entries[0][2].result() is False  # retryable, as before
    assert "Coalesced process completion delivery failed" in caplog.text
    from tests.gateway.test_completion_admission import pending
    with pytest.raises(sqlite3.OperationalError):
        await runner._deliver_async_delegation_group([pending(key, "err-group")])


@pytest.mark.asyncio
async def test_process_watcher_lost_ack_recheck_runs_off_loop_in_the_owner_scope(monkeypatch, tmp_path):
    import tools.process_registry as pr_module
    from gateway import session_automation
    from gateway import run_notifications as rn
    from tests.gateway.test_background_process_notifications import _FakeRegistry, _build_runner, _watcher_dict

    exited = SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="echo", started_at=1.0)
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry([exited] * 20, consumed=False))
    monkeypatch.setattr(rn, "_MAX_COMPLETION_INJECTION_ATTEMPTS", 1)
    runner = _build_runner(monkeypatch, tmp_path, "concise")
    runner._enqueue_process_completion_notification = AsyncMock(return_value=False)
    runner.session_authority = object()
    barrier = LoopBarrier()
    inside = []

    @contextlib.asynccontextmanager
    async def _scope(evt):
        inside.append(evt)
        try:
            yield
        finally:
            inside.pop()
    monkeypatch.setattr(runner, "_completion_event_scope", _scope)

    def _admission(_runner, evt):
        barrier("watcher recheck")
        assert inside and inside[-1] is evt
        return {"admission_id": "committed"}
    monkeypatch.setattr(session_automation, "completion_admission", _admission)
    await runner._run_process_watcher({**_watcher_dict(), "check_interval": 0, "notify_on_complete": True})
    runner.adapters[next(iter(runner.adapters))].send.assert_not_awaited()  # lost ACK: no second notice
    assert len(barrier.seen) == 1 and barrier.on_loop() == [], barrier.seen


# ---------------------------------------------------------------------------------------------
# R1 - await_peer_receipt
# ---------------------------------------------------------------------------------------------


def _peer_record():
    return {"status": "queued", "delivery_id": "d1", "admission_id": "adm-1", "session_id": "s1",
            "profile_home": "/nonexistent", "message": "hi"}


def _peer_runner():
    authority = SimpleNamespace(waiters={}, _bot_reply_watchers={})
    return SimpleNamespace(session_authority=authority), authority


@pytest.mark.asyncio
async def test_receipt_settled_between_waiter_registration_and_read_is_not_lost(monkeypatch):
    from gateway import session_bot
    from gateway.platforms.api_server_bot_chat import await_live_delivery
    runner, authority = _peer_runner()
    barrier = LoopBarrier()
    state = {"live": True, "settled": False}
    monkeypatch.setattr(session_bot, "peer_wait_admission",
                        lambda _a, _r: ("adm-1", False) if state["live"] else (None, False))

    def _result(_authority, _record):
        barrier("receipt read", hold=not state["settled"])
        return {"status": "settled" if state["settled"] else "queued", "reply": "ok" if state["settled"] else ""}
    monkeypatch.setattr(session_bot, "_result", _result)

    waiting = asyncio.ensure_future(await_live_delivery(SimpleNamespace(gateway_runner=runner), None,
                                                        _peer_record(), None))
    assert await barrier.parked_before(waiting), f"the receipt re-read ran on the event loop: {barrier.seen}"
    # The owner settles while the (stale) read is in flight: the waiter registered before the read
    # is the one it resolves, so the wait (no timeout, no should_stop) still ends.
    waiter = authority.waiters.pop("adm-1")
    state.update(live=False, settled=True)
    waiter.set_result(None)
    barrier.release.set()
    receipt = await asyncio.wait_for(waiting, 30)
    assert receipt["status"] == "settled" and receipt["reply"] == "ok"
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_receipt_wait_cancellation_deadline_and_errors(monkeypatch):
    from gateway import session_bot
    from gateway.platforms.api_server_bot_chat import await_peer_receipt
    from hermes_state_runtime import RuntimeStoreError
    _runner, authority = _peer_runner()
    barrier = LoopBarrier()
    monkeypatch.setattr(session_bot, "peer_wait_admission", lambda _a, _r: ("adm-1", False))
    mode = {"hold": True, "raise": False}

    def _result(_authority, _record):
        barrier("receipt read", hold=mode["hold"])
        if mode["raise"]:
            raise RuntimeStoreError("storage_unavailable")
        return {"status": "queued"}
    monkeypatch.setattr(session_bot, "_result", _result)

    # Cancelled while the read is parked: the cancel lands at once, not after the read.
    waiting = asyncio.ensure_future(await_peer_receipt(authority, _peer_record(), None))
    assert await barrier.parked_before(waiting), barrier.seen
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    barrier.release.set()
    # Deadline: a pending receipt is answered as it stands once the timeout lapses.
    mode["hold"] = False
    receipt = await asyncio.wait_for(await_peer_receipt(authority, _peer_record(), 0.05), 30)
    assert receipt["status"] == "queued"
    # should_stop answers the current receipt.
    receipt = await asyncio.wait_for(
        await_peer_receipt(authority, _peer_record(), None, should_stop=lambda: True), 30)
    assert receipt["status"] == "queued"
    # A storage error from the read propagates to the caller as before.
    mode["raise"] = True
    with pytest.raises(RuntimeStoreError):
        await asyncio.wait_for(await_peer_receipt(authority, _peer_record(), 1), 30)
    assert barrier.on_loop() == []
