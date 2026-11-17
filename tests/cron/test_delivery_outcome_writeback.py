"""A queued cron delivery's real outcome is written back onto the run that queued it.

The in-process run hands its send to the durable queue and books ``delivery_queued`` /
ledger ``queued``. Once a gateway drain sends or refuses it, the job and the ledger must
say what actually happened (main's inline ``ok``/``delivered`` and ``delivery_failed``/
``failed`` + error) — including when the drain wins the race against the run's own
bookkeeping write.
"""
import pytest

from cron import executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.mark.parametrize("drain_timing", ["after_run", "before_bookkeeping"])
@pytest.mark.parametrize("transport_ok", [True, False], ids=["sent", "refused"])
def test_drained_delivery_outcome_is_written_back(tmp_path, monkeypatch, transport_ok, drain_timing):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    sent = []

    async def send(platform, pconfig, chat_id, text, **kwargs):
        sent.append(text)
        return {"success": transport_ok, "message_id": "r",
                "error": None if transport_ok else "transport refused"}

    monkeypatch.setattr("gateway.outbound.send._send_to_platform", send)
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
    if drain_timing == "before_bookkeeping":
        from cron import delivery_queue
        real_enqueue = delivery_queue.enqueue

        def enqueue_then_drain(*args, **kwargs):
            queued = real_enqueue(*args, **kwargs)
            assert scheduler.drain_delivery_queue({}, None) == 1
            return queued

        monkeypatch.setattr(delivery_queue, "enqueue", enqueue_then_drain)

    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="telegram:fixture")
    scheduler.run_one_job(job)
    if drain_timing == "after_run":
        assert sent == [], "the run only queues; the gateway drain sends"
        assert scheduler.drain_delivery_queue({}, None) == 1

    assert len(sent) == 1
    saved = jobs.get_job(job["id"])
    row = executions.latest_execution(job["id"])
    if transport_ok:
        assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
        assert row["delivery_outcome"] == "delivered"
    else:
        # msg-retry-on-13f-6 (ea298f05): a refused attempt is retried; the run says so, with the error.
        assert saved["last_status"] == "delivery_retrying"
        assert "transport refused" in saved["last_delivery_error"]
        assert row["delivery_outcome"] == "retrying"
        # msg-retry-on-13f-7 (MR-ACCOUNT-10): the job carries the exact failed-attempt error the queue recorded,
        # and the queued payload is kept for the retry.
        from cron import delivery_queue as queue_now
        queued_row = queue_now.get_status(row["id"])
        assert queued_row["status"] == "pending" and queued_row["retry_after"]
        assert saved["last_delivery_error"] == queued_row["error"]
        assert "the result" in queued_row["content"] and queued_row["job_json"] != "{}"
        # The retry delivers: the final outcome is written back over the retrying one.
        from cron import delivery_queue
        transport_ok_now = {"ok": True}

        async def send_ok(platform, pconfig, chat_id, text, **kwargs):
            sent.append(text)
            return {"success": transport_ok_now["ok"], "message_id": "r2", "error": None}

        monkeypatch.setattr("gateway.outbound.send._send_to_platform", send_ok)
        later = delivery_queue._now_seconds() + delivery_queue.RETRY_BACKOFF_MAX_SECONDS + 1
        monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: later)
        assert scheduler.drain_delivery_queue({}, None) == 1
        assert len(sent) == 2  # the duplicate attempt is accepted
        saved = jobs.get_job(job["id"])
        row = executions.latest_execution(job["id"])
        assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
        assert row["delivery_outcome"] == "delivered"
