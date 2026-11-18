"""msg-retry-on-13f-9: an uncertain send stays retryable, and nothing strands the messages after it.

MR-DELIVERY-15: a live text send that STARTED but whose confirmation outlasted the wait was "assumed delivered",
so the queue finished it delivered and pruned its payload before the adapter's late refusal.
MR-DELIVERY-16: an explicitly ambiguous media acknowledgment (``raw_response={'ambiguous': True}``) did the same.
MR-RETRY-14: the capped backoff computed ``30 * 2**(attempts-1)`` before the cap, so a row retried 1025 times
raised OverflowError in every drain and blocked every other queued delivery.

Real job store, execution ledger, delivery queue, scheduler, live DeliveryRouter and settlement; only the business
run and the platform's outbound calls are local doubles (the reviewer's msgretry8-r19546 scheduler probe and
retry-boundary probe, ported). The business run is counted: a message retry never re-runs it.
Eric (canonical ea298f05): "I don't give a fuck whether the same message is sent twice. I just want all messages
to be delivered at least once." Duplicates below are accepted, never a failure.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from cron import delivery_queue, executions, jobs, scheduler, scheduler_delivery
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("cron: {wrap_response: false}\n")
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    clock = {"now": 3_000_000.0}
    monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: clock["now"])
    monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_SECS", 0.2)
    state = SimpleNamespace(business=[], standalone=[], bodies={}, clock=clock, home=tmp_path)

    def run(job, **kwargs):
        state.business.append(job["id"])
        return True, "raw", state.bodies.get(job["prompt"], "body-" + job["prompt"]), None

    async def standalone(platform, pconfig, chat_id, text, **kwargs):
        # A satellite profile has no platform credential: standalone cannot stand in for the live adapter.
        state.standalone.append((chat_id, text))
        return {"error": "no platform credential in this profile"}

    monkeypatch.setattr(scheduler, "run_job", run)
    monkeypatch.setattr("gateway.outbound.send._send_to_platform", standalone)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    state.loop = loop
    yield state
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()
    delivery_queue._ACTIVE_DELIVERIES.clear()
    delivery_queue._UNSTARTED_DELIVERIES.clear()


def _run(prompt, chat):
    job = jobs.create_job(prompt=prompt, schedule="every 1h", deliver=f"discord:{chat}")
    scheduler.run_one_job(job)
    return job, executions.latest_execution(job["id"])["id"]


def _drain(env, adapter):
    return scheduler.drain_delivery_queue({Platform.DISCORD: adapter}, env.loop)


def _flush(env):
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), env.loop).result(5)


@pytest.mark.parametrize("late", ["refused", "delivered-ack-lost"])
def test_a_started_unconfirmed_text_send_stays_queued_and_a_later_message_still_delivers(env, late):
    """A starts on the live adapter and outlasts the confirmation wait; B confirms. A is not delivered: its row
    keeps its payload and is retried after its backoff (standalone cannot send it here). Whatever A's first send
    finally did (late refusal, or late delivery whose ack was lost), the retry delivers it, so A reaches the
    recipient at least once (twice when the late send had landed: accepted). Each business run happened once."""
    gate, started = asyncio.Event(), threading.Event()
    received, attempts = [], []

    async def send(chat_id, content, metadata=None):
        attempts.append(chat_id)
        if chat_id == "unconfirmed" and attempts.count("unconfirmed") == 1:
            started.set()
            await gate.wait()
            if late == "refused":
                return SendResult(success=False, error="refused after the confirmation deadline")
        received.append((chat_id, content))
        return SendResult(success=True, message_id=f"m{len(attempts)}")

    adapter = SimpleNamespace(send=send)
    a, a_exec = _run("uncertain-A", "unconfirmed")
    b, b_exec = _run("healthy-B", "healthy")
    _drain(env, adapter)
    assert started.is_set()
    row = delivery_queue.get_status(a_exec)
    assert row["status"] == "pending" and row["content"] == "body-uncertain-A", row
    assert "unconfirmed" in row["error"] and row["retry_after"] > env.clock["now"]
    assert jobs.get_job(a["id"])["last_status"] == "delivery_retrying"
    assert executions.get_execution(a_exec)["delivery_outcome"] == "retrying"
    assert delivery_queue.get_status(b_exec)["status"] == "delivered"
    assert ("healthy", "body-healthy-B") in received
    assert env.standalone == [("unconfirmed", "body-uncertain-A")]  # the fallback was tried; it could not send

    env.loop.call_soon_threadsafe(gate.set)  # A's first send finishes late
    _flush(env)
    assert _drain(env, adapter) == 0  # still waiting out its backoff, never given up
    env.clock["now"] = row["retry_after"]
    assert _drain(env, adapter) == 1
    assert delivery_queue.get_status(a_exec)["status"] == "delivered"
    saved = jobs.get_job(a["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(a_exec)["delivery_outcome"] == "delivered"
    copies = received.count(("unconfirmed", "body-uncertain-A"))
    assert copies == (2 if late == "delivered-ack-lost" else 1)  # at least once; a duplicate is accepted
    assert sorted(env.business) == sorted([a["id"], b["id"]])  # message retries never re-ran the business


@pytest.mark.parametrize("unconfirmed", [
    SendResult(success=True, raw_response={"ambiguous": True}),
    None,
], ids=["ambiguous-ack", "no-result"])
def test_an_unconfirmed_attachment_stays_queued_with_its_body_and_media_until_confirmed(env, unconfirmed):
    """Text confirms; the attachment's result is unconfirmed (lost ack, or no result at all). The message stays
    queued with its body and MEDIA reference, retried after its backoff: text and attachment are sent again
    (a duplicate, accepted) and the confirmed retry delivers it. One business run."""
    media = env.home / "report.pdf"
    media.write_bytes(b"%PDF-1.4 harmless fixture")
    texts, media_calls = [], []

    async def send(chat_id, content, metadata=None):
        texts.append((chat_id, content))
        return SendResult(success=True, message_id=f"t{len(texts)}")

    async def send_document(chat_id, file_path, metadata=None):
        media_calls.append((chat_id, file_path))
        return unconfirmed if len(media_calls) == 1 else SendResult(success=True, message_id="doc")

    adapter = SimpleNamespace(send=send, send_document=send_document)
    env.bodies["media"] = f"body\nMEDIA:{media}"
    job, eid = _run("media", "report")
    _drain(env, adapter)
    row = delivery_queue.get_status(eid)
    assert len(media_calls) == 1
    assert row["status"] == "pending" and row["content"] == f"body\nMEDIA:{media}", row
    assert str(media) in row["error"]
    assert jobs.get_job(job["id"])["last_status"] == "delivery_retrying"
    assert executions.get_execution(eid)["delivery_outcome"] == "retrying"
    env.clock["now"] = row["retry_after"]
    assert _drain(env, adapter) == 1
    assert delivery_queue.get_status(eid)["status"] == "delivered"
    assert executions.get_execution(eid)["delivery_outcome"] == "delivered"
    assert texts == [("report", "body")] * 2 and media_calls == [("report", str(media))] * 2
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
    assert env.business == [job["id"]]


# MR-RETRY-14 (the reviewers' test_retry_boundary.py / test_long_retry.py / test_gateway_progress.py, ported):
# bounded arithmetic at any retry count; the old message keeps its payload and error and is retried forever.

@pytest.fixture
def queue_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: clock["now"])
    monkeypatch.setattr("cron.delivery_outcome.settle_quietly", lambda *a, **k: None)
    sent, refuse = [], {"A"}

    def transport(job, content, **kwargs):  # the scheduler's _deliver_result: a harmless double
        sent.append(job["id"])
        return f"refused {job['id']}" if job["id"] in refuse else None
    monkeypatch.setattr(scheduler, "_deliver_result", transport)
    yield SimpleNamespace(clock=clock, sent=sent, refuse=refuse)
    delivery_queue._ACTIVE_DELIVERIES.clear()
    delivery_queue._UNSTARTED_DELIVERIES.clear()


def _set_attempts(execution_id, attempts):
    with delivery_queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET attempts=? WHERE execution_id=?", (attempts, execution_id))


def _gateway_drain():
    """The gateway housekeeping tick's ordinary non-waiting drain; returns once its sends finished."""
    from gateway import run
    from tests.waiting import wait_until
    run._drain_restart_safe_cron_deliveries({}, None)
    wait_until(lambda: not delivery_queue._ACTIVE_DELIVERIES, timeout=10)


@pytest.mark.parametrize("attempts", [1024, 10**6])
def test_a_long_refused_message_keeps_a_bounded_backoff_never_blocks_b_and_delivers_on_retry(queue_only, attempts):
    delivery_queue.enqueue("A", {"id": "A"}, "still owed")
    _set_attempts("A", attempts)
    scheduler.drain_delivery_queue({}, None)
    row = delivery_queue.get_status("A")
    assert (row["status"], row["content"], row["attempts"], row["error"]) == ("pending", "still owed", attempts + 1,
                                                                             "refused A")
    assert row["retry_after"] == 1_000_000.0 + delivery_queue.RETRY_BACKOFF_MAX_SECONDS
    delivery_queue.enqueue("B", {"id": "B"}, "later owed")
    _gateway_drain()
    assert delivery_queue.get_status("B")["status"] == "delivered"
    queue_only.refuse.clear()
    queue_only.clock["now"] = row["retry_after"]
    _gateway_drain()
    assert delivery_queue.get_status("A")["status"] == "delivered"
    assert queue_only.sent == ["A", "B", "A"]


def test_an_abandoned_send_at_a_large_retry_count_is_queued_again_b_progresses_and_a_delivers(queue_only):
    """recover_abandoned (every drain's first step) uses the same backoff: a dead owner's large-count send is
    queued again with its payload, the later message is claimed in the same drain, and A's retry delivers."""
    delivery_queue.enqueue("A", {"id": "A"}, "still owed")
    _set_attempts("A", 5000)
    claimed = delivery_queue.claim_next()
    with delivery_queue._lock:
        delivery_queue._ACTIVE_DELIVERIES.discard(claimed["execution_id"])  # its send thread died
    delivery_queue.enqueue("B", {"id": "B"}, "later owed")
    _gateway_drain()
    row = delivery_queue.get_status("A")
    assert (row["status"], row["content"], row["attempts"]) == ("pending", "still owed", 5001)
    assert row["retry_after"] == 1_000_000.0 + delivery_queue.RETRY_BACKOFF_MAX_SECONDS
    assert delivery_queue.get_status("B")["status"] == "delivered" and queue_only.sent == ["B"]
    queue_only.refuse.clear()
    queue_only.clock["now"] = row["retry_after"]
    _gateway_drain()
    assert delivery_queue.get_status("A")["status"] == "delivered" and queue_only.sent == ["B", "A"]


@pytest.mark.parametrize(("claims", "delay"), [(1, 30.0), (2, 60.0), (7, 1920.0), (8, 3600.0), (13, 3600.0)])
def test_the_backoff_doubles_from_its_base_up_to_its_cap(queue_only, claims, delay):
    """Control (also green before the repair): the ordinary small-count backoff and its cap are unchanged, the
    payload stays, and a later message still delivers."""
    delivery_queue.enqueue("A", {"id": "A"}, "still owed")
    _set_attempts("A", claims - 1)
    scheduler.drain_delivery_queue({}, None)
    row = delivery_queue.get_status("A")
    assert (row["status"], row["content"], row["retry_after"]) == ("pending", "still owed", 1_000_000.0 + delay)
    delivery_queue.enqueue("B", {"id": "B"}, "later owed")
    scheduler.drain_delivery_queue({}, None)
    assert delivery_queue.get_status("B")["status"] == "delivered"
