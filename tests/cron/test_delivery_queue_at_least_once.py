"""A cron delivery is a MESSAGE: a send whose outcome is uncertain is sent again (at least once).

recovery-exercise-1 d-cron-delivering-1: a gateway that claimed a delivery was killed mid-send; the next
drain fenced the row ``unknown`` ("not retried") and the message was never delivered. Eric's rule: "A
replay is completely fine. I don't give a fuck whether the same message is sent twice. I just want all
messages to be delivered at least once." The row is queued again and a later drain sends it; a duplicate
is acceptable. It is never given up: each retry waits a capped backoff, and a row waiting out its backoff
stays visible as ``pending`` and never holds back an independent later delivery.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    import cron.delivery_queue as queue
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    clock = {"now": 1_000_000.0}
    # raising=False: on a queue without a retry clock (tree-13f) the tests still run and fail on behaviour.
    monkeypatch.setattr(queue, "_now_seconds", lambda: clock["now"], raising=False)
    queue.clock = clock
    yield queue
    del queue.clock


def _dead_pid():
    gone = subprocess.Popen([sys.executable, '-c', 'pass'])
    gone.wait()
    return gone.pid


def _killed_mid_send(queue, execution_id):
    """Claim ``execution_id`` as a gateway that then died mid-send: a real reaped pid owns the row."""
    claimed = queue.claim_next()
    assert claimed is not None and claimed["execution_id"] == execution_id
    queue._ACTIVE_DELIVERIES.discard(execution_id)
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET owner_process_id='dead-gateway', owner_pid=? WHERE execution_id=?",
                     (_dead_pid(), execution_id))


def _after_backoff(queue):
    queue.clock["now"] += getattr(queue, "RETRY_BACKOFF_MAX_SECONDS", 3600.0) + 1


def test_a_send_its_gateway_died_in_is_sent_again(queue):
    queue.enqueue("exec-1", {"id": "job-1"}, "brief")
    _killed_mid_send(queue, "exec-1")

    send = Mock(return_value=None)
    assert queue.drain(send) == 0  # queued again, waiting out its backoff
    assert queue.get_status("exec-1")["status"] == "pending"
    _after_backoff(queue)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-1"}, "brief", False)
    assert queue.get_status("exec-1")["status"] == "delivered"


def test_a_send_that_keeps_killing_its_gateway_is_never_given_up(queue):
    """Past any number of uncertain claims it stays pending (visible, eligible) with a capped backoff."""
    queue.enqueue("exec-2", {"id": "job-2"}, "brief")
    for _ in range(12):
        _after_backoff(queue)
        _killed_mid_send(queue, "exec-2")
        assert queue.recover_abandoned() == 1
        row = queue.get_status("exec-2")
        assert row["status"] == "pending"
        assert row["retry_after"] <= queue.clock["now"] + getattr(queue, "RETRY_BACKOFF_MAX_SECONDS", 3600.0)
    assert row["attempts"] == 12
    _after_backoff(queue)
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-2"}, "brief", False)


def test_a_backed_off_delivery_never_delays_an_independent_later_one(queue):
    queue.enqueue("exec-stuck", {"id": "job-a"}, "first")
    _killed_mid_send(queue, "exec-stuck")
    queue.recover_abandoned()
    queue.enqueue("exec-later", {"id": "job-b"}, "second")

    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-b"}, "second", False)
    assert queue.get_status("exec-later")["status"] == "delivered"
    assert queue.get_status("exec-stuck")["status"] == "pending"  # still eligible, not given up
    _after_backoff(queue)
    assert queue.drain(send) == 1
    assert queue.get_status("exec-stuck")["status"] == "delivered"


def test_a_wait_timeout_requeues_a_dead_gateways_send(queue):
    queue.enqueue("exec-3", {"id": "job-3"}, "result")
    _killed_mid_send(queue, "exec-3")

    # Deferred, not failed: a later drain sends it.
    assert queue.enqueue_and_wait("exec-3", {"id": "job-3"}, "result", timeout=0) is None
    assert queue.get_status("exec-3")["status"] == "pending"
    _after_backoff(queue)
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-3"}, "result", False)


def test_a_wait_timeout_leaves_a_live_gateways_send_to_it(queue):
    queue.enqueue("exec-4", {"id": "job-4"}, "result")
    assert queue.claim_next()["owner_pid"] == os.getpid()  # this live process is mid-send

    assert queue.enqueue_and_wait("exec-4", {"id": "job-4"}, "result", timeout=0) is None
    assert queue.get_status("exec-4")["status"] == "delivering"
    assert queue._finish("exec-4", error=None)
    assert queue.get_status("exec-4")["status"] == "delivered"



def test_a_held_send_never_delays_a_later_ready_message_or_a_new_arrival(tmp_path, monkeypatch):
    """MR-DISPATCH-06 / DISPATCH-PRIORITY-1: each eligible row gets its own in-flight send. Through the normal
    scheduler.drain_delivery_queue: B is sent while A is held; C, enqueued during the hold, is sent by the next
    (non-waiting) drain; A's own outcome is recorded when its send returns."""
    import threading
    from cron import delivery_queue as queue, scheduler
    monkeypatch.setattr(queue, 'DELIVERY_DB', tmp_path / 'deliveries.db')
    queue.enqueue('a', {'id': 'a'}, 'A')
    queue.enqueue('b', {'id': 'b'}, 'B')
    entered, release = threading.Event(), threading.Event()
    sent = {'b': threading.Event(), 'c': threading.Event()}

    def send(job, content, **kwargs):
        if job['id'] == 'a':
            entered.set()
            assert release.wait(20)
        else:
            sent[job['id']].set()
    monkeypatch.setattr(scheduler, '_deliver_result', send)
    try:
        assert scheduler.drain_delivery_queue({}, None, wait=False) == 2
        assert entered.wait(10) and sent['b'].wait(10)
        queue.enqueue('c', {'id': 'c'}, 'C')  # arrives while A is still in flight
        assert scheduler.drain_delivery_queue({}, None, wait=False) == 1
        assert sent['c'].wait(10)
        assert queue.get_status('a')['status'] == 'delivering'
    finally:
        release.set()
    from tests.waiting import wait_until
    wait_until(lambda: queue.get_status('a')['status'] != 'delivering', timeout=10, what="A's own outcome recorded")
    assert [queue.get_status(k)['status'] for k in 'abc'] == ['delivered'] * 3


@pytest.mark.parametrize("wait", [True, False], ids=["waiting", "non-waiting"])
@pytest.mark.parametrize("failure", ["construct", "start"])
def test_a_send_thread_that_fails_to_launch_leaves_no_stranded_message(queue, monkeypatch, failure, wait):
    """MR-DISPATCH-07: the claimed row's send thread cannot be built or started. Nothing was sent, so the claim is
    released: the next drain of this live gateway sends it at once (no backoff, no counted attempt), and the
    failure itself is surfaced to the caller."""
    import threading
    from agent import memory_provider
    queue.enqueue("a", {"id": "a"}, "A")
    sent = []

    def send(job, content, for_failure):
        sent.append(content)
    real_spawn, real_start = memory_provider.spawn_context_thread, threading.Thread.start

    def broken_spawn(*args, **kwargs):
        raise RuntimeError("injected thread construction failure")

    def broken_start(self):
        if self.name.startswith("cron-delivery-"):
            raise RuntimeError("injected thread start failure")
        return real_start(self)
    if failure == "construct":
        monkeypatch.setattr(memory_provider, "spawn_context_thread", broken_spawn)
    else:
        monkeypatch.setattr(threading.Thread, "start", broken_start)
    with pytest.raises(RuntimeError, match="injected thread"):
        queue.drain(send, wait=wait)
    assert sent == [] and "a" not in queue._ACTIVE_DELIVERIES
    monkeypatch.setattr(memory_provider, "spawn_context_thread", real_spawn)
    monkeypatch.setattr(threading.Thread, "start", real_start)
    assert queue.drain(send) == 1  # the same clock: no backoff for a send that never started
    assert sent == ["A"]
    status = queue.get_status("a")
    assert status["status"] == "delivered" and status["attempts"] == 1


@pytest.mark.parametrize("error", ["recipient did not acknowledge send", "timed out waiting for the platform"])
def test_a_negative_or_timed_out_send_is_retried_while_later_messages_advance(queue, error):
    """MR-RETRY-08: A's send returns an error (negative / timeout). A keeps its payload and is queued again
    after its backoff; B (and C, arriving later) are delivered meanwhile; A's retry delivers it. A duplicate is
    accepted."""
    queue.enqueue("a", {"id": "a"}, "A")
    queue.enqueue("b", {"id": "b"}, "B")
    sent, results = [], {"A": [error, None]}

    def send(job, content, for_failure):
        sent.append(content)
        return results[content].pop(0) if content in results else None
    assert queue.drain(send) == 2
    a = queue.get_status("a")
    assert a["status"] == "pending" and a["content"] == "A" and a["error"] == error and a["retry_after"]
    assert queue.get_status("b")["status"] == "delivered"
    queue.enqueue("c", {"id": "c"}, "C")
    assert queue.drain(send) == 1 and queue.get_status("c")["status"] == "delivered"  # A's backoff blocks nobody
    _after_backoff(queue)
    assert queue.drain(send) == 1
    assert queue.get_status("a")["status"] == "delivered"
    assert sorted(sent) == ["A", "A", "B", "C"]


def test_a_forever_failing_message_never_delays_a_healthy_one_queued_after_it(queue):
    """msg-retry-on-13f-6: A's send fails forever (e.g. no configured target). It is retried indefinitely, its
    backoff capped at RETRY_BACKOFF_MAX_SECONDS (1 h), its error kept on the row, its payload never pruned. Each
    healthy message queued after it, in every round, is delivered by the first drain after it arrives."""
    queue.enqueue("a", {"id": "a"}, "A")
    sent = []

    def send(job, content, for_failure):
        sent.append(content)
        return "no delivery target configured" if content == "A" else None
    for round_ in range(12):
        healthy = f"h{round_}"
        queue.enqueue(healthy, {"id": healthy}, healthy.upper())
        before = queue.clock["now"]
        queue.drain(send)
        assert queue.get_status(healthy)["status"] == "delivered", round_  # never waits behind A
        a = queue.get_status("a")
        assert a["status"] == "pending" and a["content"] == "A" and a["error"] == "no delivery target configured"
        assert 0 < a["retry_after"] - before <= queue.RETRY_BACKOFF_MAX_SECONDS
        queue.clock["now"] = a["retry_after"]  # A's next retry is due
    assert a["retry_after"] - before == queue.RETRY_BACKOFF_MAX_SECONDS  # the cap holds after many attempts
    assert sent.count("A") == 12 and all(f"H{i}" in sent for i in range(12))
