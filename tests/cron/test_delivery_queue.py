"""Durable at-least-once delivery handoff for restart-safe cron workers (a send its gateway died in
is sent again; one its own gateway returned from is not)."""

from __future__ import annotations

import sqlite3
from unittest.mock import Mock

import pytest


def test_pending_delivery_is_claimed_and_sent_once(tmp_path, monkeypatch):
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    queue.enqueue("exec-1", {"id": "job-1"}, "brief")
    send = Mock(return_value=None)

    assert queue.drain(send) == 1
    assert queue.drain(send) == 0
    send.assert_called_once_with({"id": "job-1"}, "brief", False)
    status = queue.get_status("exec-1")
    assert status["status"] == "delivered"
    assert status["job_json"] == "{}"
    assert status["content"] == ""


def test_pending_deliveries_are_claimed_in_instant_order_across_dst_fall_back(
    tmp_path, monkeypatch
):
    """01:10-05:00 is 20 minutes after 01:50-04:00 but sorts first as text."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    new_york = ZoneInfo("America/New_York")
    monkeypatch.setattr(
        queue, "_hermes_now", lambda: datetime(2026, 11, 1, 1, 50, tzinfo=new_york, fold=0)
    )
    queue.enqueue("exec-z-earlier", {"id": "job-1"}, "first")
    monkeypatch.setattr(
        queue, "_hermes_now", lambda: datetime(2026, 11, 1, 1, 10, tzinfo=new_york, fold=1)
    )
    queue.enqueue("exec-a-later", {"id": "job-2"}, "second")

    assert queue.claim_next()["execution_id"] == "exec-z-earlier"
    assert queue.claim_next()["execution_id"] == "exec-a-later"


def test_terminal_delivery_retention_is_bounded(tmp_path, monkeypatch):
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 2, raising=False)
    for index in range(4):
        execution_id = f"exec-{index}"
        queue.enqueue(execution_id, {"id": f"job-{index}"}, f"brief-{index}")
        assert queue.claim_next()["execution_id"] == execution_id
        assert queue._finish(execution_id, error=None)

    # Pruning may discard verbose outcome rows, but never the durable
    # idempotency tombstone for an execution that could be replayed later.
    pruned = queue.get_status("exec-0")
    assert pruned is not None
    assert pruned["status"] == "delivered"
    assert queue.get_status("exec-1")["status"] == "delivered"
    assert queue.get_status("exec-2")["status"] == "delivered"
    assert queue.get_status("exec-3")["status"] == "delivered"

    # Pruning may discard verbose outcome rows, but never the durable
    # idempotency tombstone for an execution that could be replayed later.
    queue.enqueue("exec-0", {"id": "job-replayed"}, "duplicate brief")
    send = Mock(return_value=None)
    assert queue.drain(send) == 0
    send.assert_not_called()


def test_failure_delivery_lane_survives_durable_handoff(tmp_path, monkeypatch):
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    queue.enqueue(
        "exec-failure",
        {"id": "job-failure", "failure_deliver": "local"},
        "failed",
        for_failure=True,
    )
    send = Mock(return_value=None)

    assert queue.drain(send) == 1
    send.assert_called_once_with(
        {"id": "job-failure", "failure_deliver": "local"},
        "failed",
        True,
    )


def test_legacy_queue_schema_adds_failure_lane_before_enqueue(tmp_path, monkeypatch):
    import cron.delivery_queue as queue

    db = tmp_path / "deliveries.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE deliveries (
                 execution_id TEXT PRIMARY KEY,
                 job_json TEXT NOT NULL,
                 content TEXT NOT NULL,
                 status TEXT NOT NULL,
                 owner_process_id TEXT,
                 owner_pid INTEGER,
                 owner_started_at INTEGER,
                 created_at TEXT NOT NULL,
                 finished_at TEXT,
                 error TEXT
               )"""
        )
    monkeypatch.setattr(queue, "DELIVERY_DB", db)

    queue.enqueue(
        "exec-migrated",
        {"id": "job-migrated"},
        "failed",
        for_failure=True,
    )

    assert queue.get_status("exec-migrated")["for_failure"] == 1


def test_wait_timeout_leaves_inflight_delivery_to_its_live_gateway(
    tmp_path, monkeypatch
):
    """Mid-send in a live gateway: a deferral, never fenced unknown and never sent a second time
    while that gateway still owns it (msg-retry-on-13f-1; was: marked unknown without retry)."""
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    queue.enqueue("exec-inflight", {"id": "job-inflight"}, "result")
    assert queue.claim_next() is not None

    error = queue.enqueue_and_wait(
        "exec-inflight", {"id": "job-inflight"}, "result", timeout=0
    )

    assert error is None
    status = queue.get_status("exec-inflight")
    assert status is not None
    assert status["status"] == "delivering"
    send = Mock(return_value=None)
    assert queue.drain(send) == 0
    send.assert_not_called()


def test_dead_delivery_owner_is_queued_again_and_resent(
    tmp_path, monkeypatch
):
    """At least once (msg-retry-on-13f-1; was: unknown and not retried): a duplicate is acceptable."""
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    queue.enqueue("exec-1", {"id": "job-1"}, "brief")
    assert queue.claim_next() is not None
    monkeypatch.setattr(queue, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(queue, "_owner_is_live", lambda _pid, _started: False)

    assert queue.recover_abandoned() == 1  # queued again, not fenced
    monkeypatch.setattr(queue, "_now_seconds", lambda: 10.0 ** 12)  # past its retry backoff
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-1"}, "brief", False)
    assert queue.get_status("exec-1")["status"] == "delivered"


def test_delivery_failure_keeps_its_payload_and_is_retried(
    tmp_path, monkeypatch
):
    """msg-retry-on-13f-5 (MR-RETRY-08; recorded contract change, canonical decision ea298f05: "I just want all
    messages to be delivered at least once."). Was: a returned send error was terminal ``failed``, never
    retried, payload pruned. Now the payload stays and the send is retried after its backoff; a duplicate is
    accepted."""
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(queue, "_now_seconds", lambda: clock["now"])
    queue.enqueue("exec-1", {"id": "job-1"}, "brief")
    error = "request failed: https://example.test/?token=TOKEN123"
    send = Mock(side_effect=[error, None])

    assert queue.drain(send) == 1
    status = queue.get_status("exec-1")
    assert status["status"] == "pending" and status["content"] == "brief"
    assert status["job_json"] != "{}" and status["error"] == error
    assert queue.drain(send) == 0  # waiting out its backoff, never given up
    clock["now"] += queue.RETRY_BACKOFF_MAX_SECONDS + 1
    assert queue.drain(send) == 1
    assert send.call_count == 2
    assert queue.get_status("exec-1")["status"] == "delivered"


def test_wait_timeout_leaves_unclaimed_delivery_queued_for_next_gateway(
    tmp_path, monkeypatch
):
    """A row nobody claimed was never attempted: it is not uncertain, so a
    gateway outage longer than the worker's wait budget must not lose it."""
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    job = {"id": "job-3", "deliver": "origin"}

    error = queue.enqueue_and_wait("exec-3", job, "result", timeout=0)

    # Deferred, not failed: the worker must not record delivery_failed.
    assert error is None
    status = queue.get_status("exec-3")
    assert status is not None
    assert status["status"] == "pending"
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with(job, "result", False)
    assert queue.get_status("exec-3")["status"] == "delivered"


def test_same_gateway_resends_a_send_whose_outcome_it_could_not_record(
    tmp_path, monkeypatch
):
    """The send returned but its outcome (delivered or failed) was never recorded: uncertain, so it is
    sent again after its backoff (msg-retry-on-13f-1; was: unknown, not resent). A duplicate is fine."""
    import cron.delivery_queue as queue

    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    queue.enqueue("exec-4", {"id": "job-4"}, "result")
    send = Mock(return_value=None)
    original_finish = queue._finish
    monkeypatch.setattr(
        queue,
        "_finish",
        Mock(side_effect=OSError("database temporarily unavailable")),
    )

    with pytest.raises(OSError, match="temporarily unavailable"):
        queue.drain(send)

    monkeypatch.setattr(queue, "_finish", original_finish)
    assert queue.drain(send) == 0  # queued again, waiting out its backoff
    send.assert_called_once()
    assert queue.get_status("exec-4")["status"] == "pending"
    monkeypatch.setattr(queue, "_now_seconds", lambda: 10.0 ** 12)
    assert queue.drain(send) == 1
    assert send.call_count == 2
    assert queue.get_status("exec-4")["status"] == "delivered"
