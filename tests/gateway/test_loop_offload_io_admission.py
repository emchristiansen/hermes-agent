"""Round-3 witnesses of review t_f95f82f7/run2812 (W1): a reservation is acknowledged as admitted
only once it is (in any process), a failed release stays owed, a terminal event follows its durable
status, and one failed status write never consumes another run's. Real HTTP handlers and SQLite;
every interleaving is held by a barrier, never timed."""
import asyncio
import contextlib
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_loop_offload_io import (
    _INSERT, _SCOPE, LoopBarrier, _agent, _probed_adapter, _runs_app, _settled,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _rows(path):
    conn = sqlite3.connect(str(path))
    try:
        return [tuple(r) for r in conn.execute("SELECT idempotency_key, run_id FROM run_idempotency")]
    finally:
        conn.close()


def _registry(adapter):
    return getattr(adapter, "_run_key_reservations", {})


async def _drain(store):
    """Every store job queued before this call (and the loop callbacks it scheduled) has run.
    (A store without a worker -- the pre-offload baseline -- has nothing queued.)"""
    for _ in range(2):
        if hasattr(store, "submit"):
            await asyncio.wrap_future(store.submit(lambda: None))
        for _ in range(5):
            await asyncio.sleep(0)


def _capture_handlers(adapter, handlers):
    limited = adapter._concurrency_limited_response

    def _record():
        handlers.append(asyncio.current_task())
        return limited()
    return patch.object(adapter, "_concurrency_limited_response", side_effect=_record)


# ---------------------------------------------------------------------------------------------
# W1-CANCEL-REPLAY-01-XPROC: another process's durable-store consumer
# ---------------------------------------------------------------------------------------------

_OBSERVER = '''
import asyncio, json, sys
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_loop_offload_io import _agent, _runs_app

async def main(db, out, key):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(db)
    calls = []
    try:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            async with TestClient(TestServer(_runs_app(adapter))) as client:
                response = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": key})
                body = await response.json()
                for task in list(adapter._active_run_tasks.values()):
                    await task
                json.dump({"status": response.status, "body": body, "executions": len(calls)}, open(out, "w"))
    finally:
        await adapter.disconnect()

asyncio.run(main(*sys.argv[1:]))
'''


def _observe(tmp_path, label, key):
    """One same-key POST from a separate gateway process on the same durable store."""
    script, out = tmp_path / "observer.py", tmp_path / f"observer-{label}.json"
    script.write_text(_OBSERVER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO), os.environ.get("PYTHONPATH")]))}
    done = subprocess.run([sys.executable, str(script), str(tmp_path / "idem.db"), str(out), key],
                          cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


class _TwoPhaseCommit:
    """The store connection, parking the first reservation commit before and after it lands."""

    def __init__(self, conn):
        self._conn = conn
        self._sql = ""
        self._armed = True
        self.before, self.after = threading.Event(), threading.Event()
        self.release_before, self.release_after = threading.Event(), threading.Event()

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, params=()):
        self._sql = sql
        return self._conn.execute(sql, params)

    def commit(self):
        if not (self._armed and self._sql.startswith("INSERT")):
            return self._conn.commit()
        self._armed = False
        self.before.set()
        assert self.release_before.wait(60), "pre-commit hold never released"
        self._conn.commit()
        self.after.set()
        assert self.release_after.wait(60), "post-commit hold never released"
        return None

    def open_all(self):
        self.release_before.set()
        self.release_after.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [True, False], ids=["cancelled", "uncancelled-control"])
async def test_another_process_never_replays_a_reservation_its_owner_may_release(tmp_path, cancelled):
    """Review run2812 W1-CANCEL-REPLAY-01-XPROC. The owner's reservation has committed but its
    request is not decided (cancelled meanwhile, or still admitting); a second gateway process
    POSTs the same key. It must not acknowledge the reservation as admitted work: it answers a
    retryable 503 (or, on code that admits synchronously, replays the run that then executes).
    Afterwards the key is either free (cancelled: the next request runs it once) or replays the
    one admitted run (control)."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    store = adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    conn = store._conn = _TwoPhaseCommit(store._conn)
    loop, handlers, calls, seen = asyncio.get_running_loop(), [], [], {}

    def _drive():
        try:
            assert conn.before.wait(60), "the reservation never committed"
            if cancelled:
                loop.call_soon_threadsafe(handlers[0].cancel)
            conn.release_before.set()
            assert conn.after.wait(60)
            seen["held"] = _observe(tmp_path, "held", "xproc")
        except BaseException as exc:  # reported below, on the test's thread
            seen["error"] = exc
        finally:
            conn.open_all()

    driver = threading.Thread(target=_drive)
    driver.start()
    try:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            with _capture_handlers(adapter, handlers), \
                    patch.object(adapter, "_create_agent", return_value=_agent(calls)):
                first = await asyncio.gather(
                    client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "xproc"}),
                    return_exceptions=True)
                first_body = None if isinstance(first[0], BaseException) else await first[0].json()
                await asyncio.to_thread(driver.join, 180)
                assert "error" not in seen, seen.get("error")
                for task in list(adapter._active_run_tasks.values()):
                    await task
                await _drain(store)
                rows_after_owner = _rows(tmp_path / "idem.db")
                registry_after_owner = dict(_registry(adapter))
                seen["after"] = await asyncio.to_thread(_observe, tmp_path, "after", "xproc")
                owner_calls = list(calls)
    finally:
        conn.open_all()
        await asyncio.to_thread(driver.join, 180)
        await adapter.disconnect()
    held, after = seen["held"], seen["after"]
    # Code that commits on the loop (the pre-offload baseline) decides the request before the
    # cancel can land: that is the admitted case, judged as the control.
    if cancelled and handlers[0].cancelled():
        assert held["status"] == 503 and held["body"]["error"]["code"] == "idempotency_key_in_flight", held
        assert held["executions"] == 0
        assert owner_calls == [] and rows_after_owner == [] and not registry_after_owner
        # The key was released: the next request (in the other process) is admitted and runs once.
        assert after["status"] == 202 and after["body"]["replayed"] is False and after["executions"] == 1
    else:
        run_id = first_body["run_id"]
        assert first[0].status == 202
        in_flight = held["status"] == 503 and held["body"]["error"]["code"] == "idempotency_key_in_flight"
        replay = held["status"] == 202 and held["body"]["run_id"] == run_id and held["body"]["replayed"]
        assert in_flight or replay, held
        assert held["executions"] == 0 and owner_calls == [1]
        assert rows_after_owner == [("xproc", run_id)]
        assert after["status"] == 202 and after["body"] == {**after["body"], "run_id": run_id, "replayed": True}
        assert after["executions"] == 0


# ---------------------------------------------------------------------------------------------
# W1-RELEASE-ERROR-03: a release that did not delete the row is still owed
# ---------------------------------------------------------------------------------------------


def _deny_deletes(raw, budget):
    """SQLite itself refuses the next *budget* DELETEs of the reservation table (None: all)."""
    denied = []

    def _authorize(action, arg1, _arg2, _db, _trigger):
        if action == sqlite3.SQLITE_DELETE and arg1 == "run_idempotency" and (budget is None or len(denied) < budget):
            denied.append(arg1)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    raw.set_authorizer(_authorize)
    return denied


async def _cancel_while_committing(adapter, probe, barrier, client, handlers, key):
    probe.hold_on = _INSERT
    first = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": key}))
    assert await barrier.parked_before(first), barrier.on_loop()
    return first


@pytest.mark.asyncio
async def test_an_overlapping_retry_never_replays_a_reservation_whose_release_failed(tmp_path, monkeypatch):
    """Port of the run2812 ``release_error`` witness: the first DELETE of the cancelled request's
    reservation is refused by SQLite while a same-key retry waits for it. The retry must not be
    acknowledged as a replay (202 ``queued`` for a run nobody executes): the owed release is
    retried, and the retry is admitted and runs once."""
    from gateway.platforms import api_server_runs
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    denied = _deny_deletes(probe._conn, 1)
    calls, handlers, joined = [], [], asyncio.Event()
    join = api_server_runs._join_key_reservation

    async def _record_join(entry):
        joined.set()
        return await join(entry)
    monkeypatch.setattr(api_server_runs, "_join_key_reservation", _record_join)
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with _capture_handlers(adapter, handlers), patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            first = await _cancel_while_committing(adapter, probe, barrier, client, handlers, "owed")
            retry = asyncio.ensure_future(client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "owed"}))
            await asyncio.wait_for(joined.wait(), 30)
            handlers[0].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handlers[0]
            barrier.release.set()
            _first, response = await asyncio.gather(first, retry, return_exceptions=True)
            body = await response.json()
            if response.status == 202:
                await _settled(client, body["run_id"])
            await _drain(store)
    assert denied == ["run_idempotency"], "the witness needs exactly one refused DELETE"
    assert not (response.status == 202 and body.get("status") == "queued" and not calls), body
    assert response.status == 202 and body["replayed"] is False, body
    assert calls == [1]
    assert _rows(tmp_path / "idem.db") == [("owed", body["run_id"])]
    assert not _registry(adapter)
    assert barrier.on_loop() == []


@pytest.mark.asyncio
async def test_a_failed_release_is_owed_until_a_later_request_completes_it(tmp_path):
    """One refused DELETE, then healthy store traffic: the reservation stays owed (never replayed,
    never forgotten), other keys are unaffected, and the next same-key request completes the
    release and is admitted fresh. While every DELETE fails, same-key requests get a retryable 503."""
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(tmp_path, barrier)
    denied = _deny_deletes(probe._conn, None)
    calls, handlers = [], []
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with _capture_handlers(adapter, handlers), patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            first = await _cancel_while_committing(adapter, probe, barrier, client, handlers, "owed")
            handlers[0].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handlers[0]
            barrier.release.set()
            with contextlib.suppress(Exception):
                await first
            await _drain(store)
            entry = _registry(adapter)[(_SCOPE, "owed")]
            assert entry.failed and not entry.releasing and len(denied) == 1
            # Healthy traffic meanwhile: another key is admitted, run and replayed normally.
            other = await client.post("/v1/runs", json={"input": "y"}, headers={"Idempotency-Key": "other"})
            other_id = (await other.json())["run_id"]
            assert (await _settled(client, other_id))["status"] == "completed"
            # Still refused: never a replay, a truthful retryable answer; the row is still owed.
            refused = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "owed"})
            refused_body = await refused.json()
            assert refused.status == 503 and refused_body["error"]["code"] == "idempotency_key_in_flight"
            assert refused.headers.get("Retry-After")
            assert len(denied) > 1 and ("owed", entry_run(tmp_path)) in _rows(tmp_path / "idem.db")
            # The store heals: the next same-key request completes the release and is admitted.
            probe._conn.set_authorizer(None)
            fresh = await client.post("/v1/runs", json={"input": "x"}, headers={"Idempotency-Key": "owed"})
            body = await fresh.json()
            await _settled(client, body["run_id"])
            await _drain(store)
    assert fresh.status == 202 and body["replayed"] is False, body
    assert calls == [1, 1], "the other key and the fresh admission ran; the cancelled request never"
    assert sorted(_rows(tmp_path / "idem.db")) == sorted([("other", other_id), ("owed", body["run_id"])])
    assert not _registry(adapter)
    assert barrier.on_loop() == []


def entry_run(tmp_path):
    return dict(_rows(tmp_path / "idem.db")).get("owed")


@pytest.mark.asyncio
async def test_a_refused_admission_whose_release_fails_stays_refused_and_owed(tmp_path, monkeypatch):
    """The refused-admission cleanup path: the session authority refuses the admission (state.db
    writer held, 503 ``storage_unavailable``) and the reservation's DELETE is refused once. The
    refusal stays the answer, nothing is replayed, and the exact retry completes the release and
    runs the work once."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB
    from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _adapter, _runs_client
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()["result"] = {"final_response": "ACK", "messages": []}
        return "ACK"

    runner = SimpleNamespace(_session_db=db, session_store=sessions, _draining=False, _handle_message=answer)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="release-owner")
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter
    adapter._run_idempotency_store.close()
    store = adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    denied = _deny_deletes(store._conn, 1)
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.3)
    client = _runs_client(adapter)
    await client.start_server()
    headers = {**_AUTH_HEADERS, "Idempotency-Key": "refused-owed"}
    body = {"input": "REFUSED", "session_id": "api-session"}
    holder = sqlite3.connect(tmp_path / "state.db", isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        refused = await client.post("/v1/runs", json=body, headers=headers)
        assert refused.status == 503, await refused.text()
        assert (await refused.json())["error"]["code"] == "storage_unavailable"
        await _drain(store)
        assert denied == ["run_idempotency"]
        entry = next(iter(_registry(adapter).values()))
        assert entry.failed, "the refused admission's release is still owed"
        assert [k for k, _ in _rows(tmp_path / "idem.db")] == ["refused-owed"]
        holder.execute("ROLLBACK")
        retry = await client.post("/v1/runs", json=body, headers=headers)
        retry_body = await retry.json()
        assert retry.status == 202 and retry_body["replayed"] is False, retry_body
        await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 30)
        await _drain(store)
        assert executed == ["REFUSED"]
        assert _rows(tmp_path / "idem.db") == [("refused-owed", retry_body["run_id"])]
        assert not _registry(adapter)
    finally:
        holder.close()
        for task in adapter._active_run_tasks.values():
            task.cancel()
        await client.close()
        adapter._response_store.close()
        await store.aclose()
        authority.db.close()
        sessions.close_all_db_handles()


# ---------------------------------------------------------------------------------------------
# W1-TERMINAL-DURABILITY-03: a terminal event follows its committed status
# ---------------------------------------------------------------------------------------------


def _hold_terminal_commit(probe, terminal="completed"):
    """Hold only the commit of the run's *terminal* status write; every other write proceeds."""
    execute = probe.execute

    def _execute(sql, params=()):
        if sql.startswith("UPDATE run_idempotency SET status_json") and json.loads(params[0])["status"] == terminal:
            probe.hold_on = lambda s: s.startswith("UPDATE run_idempotency SET status_json")
        return execute(sql, params)
    probe.execute = _execute


def _durable(tmp_path, run_id):
    conn = sqlite3.connect(str(tmp_path / "idem.db"))
    try:
        return json.loads(conn.execute(
            "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()[0])
    finally:
        conn.close()


async def _run_over(adapter, run_id):
    """The run's task has returned: its terminal status is published in memory."""
    task = adapter._active_run_tasks.get(run_id)
    if task is not None:
        await task
    assert adapter._run_statuses[run_id]["status"] == "completed"


def _published(adapter, run_id):
    stream = adapter._run_streams.get(run_id)
    return [] if stream is None else [e and e.get("event") for _, e in stream.backlog]


@pytest.mark.asyncio
async def test_the_sse_consumer_never_sees_a_terminal_event_before_its_status_is_durable(tmp_path):
    """Port of run2812's ``probe-sse``: only the ``completed`` status commit is held. The run has
    finished, yet its stream carries no terminal event (and no close) and an SSE consumer cannot
    finish; releasing the commit publishes both, and the event then agrees with durable state."""
    barrier = LoopBarrier()
    adapter, _store, probe = _probed_adapter(tmp_path, barrier)
    _hold_terminal_commit(probe)
    app = _runs_app(adapter)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    calls, order = [], []
    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(calls)):
            posted = await client.post("/v1/runs", json={"input": "hi"}, headers={"Idempotency-Key": "durable"})
            run_id = (await posted.json())["run_id"]
            assert await barrier.parked_before(), barrier.on_loop()
            await _run_over(adapter, run_id)
            stream = await client.get(f"/v1/runs/{run_id}/events")  # attached, headers sent

            async def _consume():
                text = await stream.text()
                order.append(("sse-finished", barrier.release.is_set()))
                return text
            consumer = asyncio.ensure_future(_consume())
            for _ in range(50):
                await asyncio.sleep(0)
            held_publications = _published(adapter, run_id)
            held_durable = _durable(tmp_path, run_id)["status"]
            consumer_done_while_held = consumer.done()
            order.append(("release", True))
            barrier.release.set()
            text = await asyncio.wait_for(consumer, 60)
            durable_after = _durable(tmp_path, run_id)["status"]
    assert calls == [1]
    assert held_durable in {"queued", "running"}  # write-behind: the newest non-terminal status
    assert "run.completed" not in held_publications and None not in held_publications, held_publications
    assert not consumer_done_while_held
    assert order == [("release", True), ("sse-finished", True)]
    assert "run.completed" in text and durable_after == "completed"
    assert barrier.on_loop() == []


_CRASHING_OWNER = '''
import asyncio, json, os, pathlib, sys
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
from tests.gateway.test_loop_offload_io import LoopBarrier, _agent, _probed_adapter, _runs_app
from tests.gateway.test_loop_offload_io_admission import _hold_terminal_commit, _published, _run_over

async def main(root):
    root = pathlib.Path(root)
    barrier = LoopBarrier()
    adapter, store, probe = _probed_adapter(root, barrier)
    _hold_terminal_commit(probe)
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent([])):
            posted = await client.post("/v1/runs", json={"input": "hi"}, headers={"Idempotency-Key": "crash"})
            run_id = (await posted.json())["run_id"]
            assert await barrier.parked_before()
            await _run_over(adapter, run_id)
            for _ in range(50):
                await asyncio.sleep(0)
            (root / "crashed.json").write_text(json.dumps(
                {"run_id": run_id, "published": _published(adapter, run_id),
                 "live": adapter._run_statuses[run_id]["status"]}))
            os._exit(0)  # the process dies with the terminal commit still held: nothing drains

asyncio.run(main(sys.argv[1]))
'''


@pytest.mark.asyncio
async def test_a_crash_before_the_terminal_commit_never_contradicts_what_clients_saw(tmp_path):
    """Crash/reopen witness: the owner dies while its ``completed`` commit is held. No client was
    told ``run.completed``; reopening the store reports the run ``interrupted``, consistently."""
    from gateway.platforms.api_server_runs import _load_durable_record
    script = tmp_path / "crashing_owner.py"
    script.write_text(_CRASHING_OWNER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO), os.environ.get("PYTHONPATH")]))}
    done = await asyncio.to_thread(subprocess.run, [sys.executable, str(script), str(tmp_path)],
                                   cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    crashed = json.loads((tmp_path / "crashed.json").read_text())
    assert crashed["live"] == "completed"
    assert "run.completed" not in crashed["published"] and None not in crashed["published"]
    store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    try:
        assert _durable(tmp_path, crashed["run_id"])["status"] in {"queued", "running"}
        hydrated = await asyncio.wrap_future(store.submit(
            _load_durable_record, store, _SCOPE, crashed["run_id"], 0))
    finally:
        await store.aclose()
    assert hydrated["status"] == "interrupted" and "output" not in hydrated


@pytest.mark.asyncio
async def test_a_failed_terminal_write_keeps_its_disposition_and_still_publishes(tmp_path, caplog):
    """A real write error is logged, as before, and does not strand the stream: the terminal
    event and the close still reach the consumer once the write has failed."""
    barrier = LoopBarrier()
    adapter, _store, probe = _probed_adapter(tmp_path, barrier)
    probe.fail_on = lambda sql: sql.startswith("UPDATE run_idempotency SET status_json") and bool(
        adapter._run_statuses) and next(iter(adapter._run_statuses.values()))["status"] == "completed"
    app = _runs_app(adapter)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    with caplog.at_level("ERROR"):
        async with TestClient(TestServer(app)) as client:
            with patch.object(adapter, "_create_agent", return_value=_agent([])):
                posted = await client.post("/v1/runs", json={"input": "hi"}, headers={"Idempotency-Key": "fails"})
                run_id = (await posted.json())["run_id"]
                await _run_over(adapter, run_id)
                text = await asyncio.wait_for((await client.get(f"/v1/runs/{run_id}/events")).text(), 60)
                polled = await (await client.get(f"/v1/runs/{run_id}")).json()
    assert "run.completed" in text
    assert polled["status"] == "completed"
    assert "failed to persist idempotent run status" in caplog.text


# ---------------------------------------------------------------------------------------------
# W1-STATUS-BATCH-04: one failing publication never consumes unrelated pending writes
# ---------------------------------------------------------------------------------------------


def _batch_adapter(store):
    from gateway.platforms import api_server_runs
    adapter = SimpleNamespace(_run_statuses={}, _run_idempotency_ids=set(), _run_shutdown_requested_at=None,
                              _run_streams={}, _run_idempotency_store=store)
    adapter._set_run_status = lambda *a, **kw: api_server_runs._set_run_status(adapter, *a, **kw)
    return adapter


async def _two_publications_behind_a_lookup(store, adapter):
    """Run ids ``bad`` then ``good`` publish while the worker is busy and a lookup is queued: the
    lookup's read-your-writes drain carries both pending writes in one batch."""
    for run_id in ("bad", "good"):
        await asyncio.to_thread(store.reserve, "s", run_id, "fp", run_id, {"status": "queued"})
        adapter._run_idempotency_ids.add(run_id)
    parked, release = threading.Event(), threading.Event()
    blocker = store.submit(lambda: (parked.set(), release.wait(60)))
    assert await asyncio.to_thread(parked.wait, 30)
    lookup = store.submit(store.lookup, "s", "good", "fp")
    adapter._set_run_status("bad", "completed", output="bad")
    adapter._set_run_status("good", "completed", output="good")
    release.set()
    await asyncio.wrap_future(blocker)
    await asyncio.wrap_future(lookup)
    await asyncio.wrap_future(store.submit(lambda: None))


def _stored(path):
    conn = sqlite3.connect(str(path))
    try:
        return {r: json.loads(s)["status"] for r, s in conn.execute("SELECT run_id, status_json FROM run_idempotency")}
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_a_row_local_write_failure_never_consumes_another_runs_status(tmp_path, caplog):
    """Port of run2812's ``batch_failure``: SQLite refuses only run ``bad``'s update (trigger)."""
    store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    adapter = _batch_adapter(store)
    store._conn.execute(
        "CREATE TRIGGER fail_bad BEFORE UPDATE OF status_json ON run_idempotency "
        "WHEN NEW.run_id='bad' BEGIN SELECT RAISE(ABORT, 'bad row denied'); END")
    store._conn.commit()
    try:
        with caplog.at_level("ERROR"):
            await _two_publications_behind_a_lookup(store, adapter)
    finally:
        await store.aclose()
    assert _stored(tmp_path / "idem.db") == {"bad": "queued", "good": "completed"}
    assert "failed to persist idempotent run status bad" in caplog.text


@pytest.mark.asyncio
async def test_a_transient_commit_failure_never_consumes_another_runs_status(tmp_path, caplog):
    """The transient equivalent: the batch's first status commit (run ``bad``) fails once; the
    good write is still attempted and lands, and bad's next publication lands too."""
    barrier = LoopBarrier()
    _adapter, store, probe = _probed_adapter(tmp_path, barrier)
    adapter = _batch_adapter(store)
    failures = []

    def _fail_once(sql):
        if sql.startswith("UPDATE run_idempotency SET status_json") and not failures:
            failures.append(sql)
            return True
        return False
    probe.fail_on = _fail_once
    try:
        with caplog.at_level("ERROR"):
            await _two_publications_behind_a_lookup(store, adapter)
        stored = _stored(tmp_path / "idem.db")
        adapter._set_run_status("bad", "completed", output="bad again")
        await asyncio.wrap_future(store.submit(lambda: None))
        healed = _stored(tmp_path / "idem.db")
    finally:
        await store.aclose()
    assert len(failures) == 1
    assert stored == {"bad": "queued", "good": "completed"}
    assert healed == {"bad": "completed", "good": "completed"}
    assert "failed to persist idempotent run status bad" in caplog.text
    assert barrier.on_loop() == []
