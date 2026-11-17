"""MR-DIRECT-19: script outputs and raised-run notices are durable MESSAGES.

Real run_job / _run_no_agent_job / private inert Python subprocess / job store /
ledger / queue / DeliveryRouter. Only the wire, retry clock and (for the raised
lane) the post-script document-header seam are fixtures. No provider is used.
Duplicates are accepted; a retry must never execute the script again.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cron import delivery_queue, executions, jobs, scheduler
from gateway.config import Platform
from gateway.platforms.base import SendResult
from tests.cron.test_delivery_uncertain_retry import env

_REAL_RUN_JOB = scheduler.run_job


def _script_job(env, name, *, output=None, failure_deliver=None):
    scripts = env.home / "scripts"
    scripts.mkdir(exist_ok=True)
    counter = env.home / (name + ".count")
    script = scripts / (name + ".py")
    script.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(counter)!r})\n"
        "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1')\n"
        f"print({('body-' + name if output is None else output)!r})\n"
    )
    job = jobs.create_job(
        prompt=name, schedule="every 1h", deliver="discord:" + name,
        no_agent=True, script=str(script), interpreter=sys.executable,
    )
    if failure_deliver:
        jobs.update_job(job["id"], {"failure_deliver": failure_deliver})
        job = jobs.get_job(job["id"])
    return job, counter


def _new_owner_retry(env, eid):
    """Fresh process, original store, real queue/router; only its outbound wire is fake."""
    command = '''
import asyncio, json, os, sys, threading
from types import SimpleNamespace
from cron import delivery_queue, executions, jobs, scheduler, scheduler_delivery, scheduler_bookkeeping
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway import config
from gateway.platforms.base import SendResult
cfg = GatewayConfig()
cfg.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
config.load_gateway_config = lambda: cfg
row = delivery_queue.get_status(os.environ['RETRY_EXECUTION'])
delivery_queue._now_seconds = lambda: row['retry_after']
received = []
async def send(chat_id, content, metadata=None):
    received.append((chat_id, content))
    return SendResult(success=True, message_id='fresh-owner-confirmed')
loop = asyncio.new_event_loop()
thread = threading.Thread(target=loop.run_forever)
thread.start()
try:
    scheduler.drain_delivery_queue({Platform.DISCORD: SimpleNamespace(send=send)}, loop)
finally:
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()
print(json.dumps({'received': received, 'pid': os.getpid(), 'owner': delivery_queue._PROCESS_ID,
                  'scheduler': scheduler.__file__, 'executable': sys.executable,
                  'sources': {m.__name__: m.__file__
                    for m in [delivery_queue, executions, jobs, scheduler, scheduler_delivery, scheduler_bookkeeping]},
                  'queue': delivery_queue.get_status(row['execution_id'])}))
'''
    child_env = dict(os.environ, HERMES_HOME=str(env.home), RETRY_EXECUTION=eid)
    result = subprocess.run([sys.executable, "-c", command], env=child_env, capture_output=True, text=True, check=False)
    print("NEW_OWNER_RAW", result.returncode, result.stdout, result.stderr)
    assert result.returncode == 0
    observation = json.loads(result.stdout)
    assert observation["pid"] != os.getpid()
    assert observation["owner"] != delivery_queue._PROCESS_ID
    for name, source in observation["sources"].items():
        assert os.path.samefile(source, sys.modules[name].__file__)
    assert observation["executable"] == sys.executable
    return [tuple(item) for item in observation["received"]]


@pytest.mark.parametrize("lane", ["output", "raised-business"])
@pytest.mark.parametrize("first_result", ["refused", "ack-lost"])
@pytest.mark.parametrize("drain_first", [False, True], ids=["book-first", "send-first"])
def test_real_script_message_survives_new_owner_without_business_replay(
    env, monkeypatch, lane, first_result, drain_first,
):
    monkeypatch.setattr(scheduler, "run_job", _REAL_RUN_JOB)
    header = scheduler._job_doc_header

    def make_header(name, *args, **kwargs):
        if name == "A" and lane == "raised-business":
            raise RuntimeError("fixture business crash after real script")
        return header(name, *args, **kwargs)

    monkeypatch.setattr(scheduler, "_job_doc_header", make_header)
    attempts, received = [], []
    healthy = {"value": False}
    target_a = "failure-A" if lane == "raised-business" else "A"

    async def send(chat_id, content, metadata=None):
        attempts.append((chat_id, content))
        if chat_id == target_a and not healthy["value"]:
            if first_result == "ack-lost":
                received.append((chat_id, content))
                return SendResult(success=False, error="fixture acknowledgment lost after delivery")
            return SendResult(success=False, error="fixture wire refusal")
        received.append((chat_id, content))
        return SendResult(success=True, message_id="confirmed")

    adapters = {Platform.DISCORD: SimpleNamespace(send=send)}
    real_enqueue = delivery_queue.enqueue

    def enqueue(*args, **kwargs):
        row = real_enqueue(*args, **kwargs)
        if drain_first:
            scheduler.drain_delivery_queue(adapters, env.loop)
        return row

    monkeypatch.setattr(delivery_queue, "enqueue", enqueue)
    a, ac = _script_job(env, "A", failure_deliver="discord:failure-A")
    scheduler.run_one_job(a, adapters=adapters, loop=env.loop)
    eid = executions.latest_execution(a["id"])["id"]
    scheduler.drain_delivery_queue(adapters, env.loop)
    first = delivery_queue.get_status(eid)
    print("DIRECT_INITIAL " + json.dumps(dict(
        lane=lane, first_result=first_result, drain_first=drain_first,
        counter=ac.read_text() if ac.exists() else None, queue=first,
        job=jobs.get_job(a["id"]), ledger=executions.get_execution(eid),
        scheduler=scheduler.__file__, producer=_REAL_RUN_JOB.__code__.co_filename,
    ), sort_keys=True))
    assert ac.read_text() == "1"
    assert first is not None and first["status"] == "pending", first
    assert first["content"] and first["error"] and first["for_failure"] == (lane == "raised-business")
    queued = json.loads(first["job_json"])
    assert queued["execution_id"] == eid
    before = jobs.get_job(a["id"])
    if lane == "raised-business":
        assert before["last_status"] == "error"
        assert before["last_error"] == "fixture business crash after real script"
        assert before["last_delivery_error"]
        assert executions.get_execution(eid)["status"] == "failed"
    else:
        assert first["content"] == "body-A"
        assert before["last_status"] == "delivery_retrying"
    assert executions.get_execution(eid)["delivery_outcome"] == "retrying"

    # Later messages progress before a fresh process reopens the store and
    # claims A. Recovery never calls the producer or scheduler.run_one_job.
    jobs.update_job(a["id"], {"deliver": "discord:WRONG", "failure_deliver": "discord:WRONG"})
    healthy["value"] = True
    counters = [ac]
    for name in ("B", "C", "D"):
        job, counter = _script_job(env, name)
        counters.append(counter)
        scheduler.run_one_job(job, adapters=adapters, loop=env.loop)
        scheduler.drain_delivery_queue(adapters, env.loop)
        assert (name, "body-" + name) in received
    assert delivery_queue.get_status(eid)["status"] == "pending"
    received.extend(_new_owner_retry(env, eid))
    final = delivery_queue.get_status(eid)
    saved = jobs.get_job(a["id"])
    ledger = executions.get_execution(eid)
    print("DIRECT_REAL " + json.dumps(dict(
        lane=lane, first_result=first_result, first=first, final=final,
        attempts=attempts, received=received, counters=[p.read_text() for p in counters],
        saved=saved, ledger=ledger,
    ), sort_keys=True, default=str))
    assert final["status"] == "delivered" and ledger["delivery_outcome"] == "delivered"
    assert received.count((target_a, first["content"])) == (2 if first_result == "ack-lost" else 1)
    assert all(p.read_text() == "1" for p in counters)
    assert saved["last_delivery_error"] is None
    assert saved["repeat"]["completed"] == 1
    if lane == "raised-business":
        assert (saved["last_status"], saved["last_error"], saved["failure_streak"]) == (
            "error", "fixture business crash after real script", 1)
        assert ledger["status"] == "failed" and ledger["error"] == saved["last_error"]
    else:
        assert saved["last_status"] == "ok" and ledger["status"] == "completed"


@pytest.mark.parametrize("output", ["", "[SILENT]", '{"wakeAgent": false}'])
def test_silent_script_has_no_delivery_obligation(env, monkeypatch, output):
    monkeypatch.setattr(scheduler, "run_job", _REAL_RUN_JOB)
    job, counter = _script_job(env, "silent", output=output)
    scheduler.run_one_job(job)
    eid = executions.latest_execution(job["id"])["id"]
    assert counter.read_text() == "1"
    assert delivery_queue.get_status(eid) is None
    assert jobs.get_job(job["id"])["last_status"] == "ok"
    assert executions.get_execution(eid)["delivery_outcome"] == "suppressed"


@pytest.mark.parametrize("ack", [None, SendResult(success=True, raw_response={"ambiguous": True})],
                         ids=["no-media-result", "ambiguous-media"])
def test_real_script_partial_media_retries_body_attachment_and_original_target(env, monkeypatch, ack):
    monkeypatch.setattr(scheduler, "run_job", _REAL_RUN_JOB)
    media = env.home / "report.pdf"
    media.write_bytes(b"%PDF-1.4 inert private report")
    content = f"body-A\nMEDIA:{media}"
    job, counter = _script_job(env, "A", output=content)
    texts, documents = [], []

    async def send(chat_id, text, metadata=None):
        texts.append((chat_id, text))
        return SendResult(success=True, message_id="text-confirmed")

    async def send_document(chat_id, file_path, metadata=None):
        documents.append((chat_id, file_path))
        return ack if len(documents) == 1 else SendResult(success=True, message_id="media-confirmed")

    adapters = {Platform.DISCORD: SimpleNamespace(send=send, send_document=send_document)}
    scheduler.run_one_job(job, adapters=adapters, loop=env.loop)
    scheduler.drain_delivery_queue(adapters, env.loop)
    eid = executions.latest_execution(job["id"])["id"]
    row = delivery_queue.get_status(eid)
    assert row and row["status"] == "pending" and row["content"] == content, row
    assert media.read_bytes() == b"%PDF-1.4 inert private report"
    b, bc = _script_job(env, "B")
    scheduler.run_one_job(b, adapters=adapters, loop=env.loop)
    scheduler.drain_delivery_queue(adapters, env.loop)
    assert ("B", "body-B") in texts
    env.clock["now"] = row["retry_after"]
    scheduler.drain_delivery_queue(adapters, env.loop)
    assert delivery_queue.get_status(eid)["status"] == "delivered"
    assert texts.count(("A", "body-A")) == 2  # duplicate text accepted
    assert documents == [("A", str(media))] * 2
    assert counter.read_text() == bc.read_text() == "1"


def test_unknown_worker_notice_retries_without_resolving_or_reexecuting_business(env, monkeypatch):
    from cron.scheduler_worker_failure import record_unknown_worker_outcome

    job, counter = _script_job(env, "unknown")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True, force=True)
    eid = executions.create_execution(job["id"], source="test")["id"]
    executions.give_up_execution(eid, reason="fixture uncertain business")
    claimed["execution_id"] = eid
    healthy, received = {"value": False}, []

    async def send(chat_id, text, metadata=None):
        if not healthy["value"]:
            return SendResult(success=False, error="fixture unknown-notice refusal")
        received.append(text)
        return SendResult(success=True, message_id="confirmed")

    adapters = {Platform.DISCORD: SimpleNamespace(send=send)}
    assert record_unknown_worker_outcome(claimed, adapters=adapters, loop=env.loop)
    scheduler.drain_delivery_queue(adapters, env.loop)
    row = delivery_queue.get_status(eid)
    assert row and row["status"] == "pending", row
    assert executions.get_execution(eid)["delivery_outcome"] == "retrying"
    healthy["value"] = True
    env.clock["now"] = row["retry_after"]
    scheduler.drain_delivery_queue(adapters, env.loop)
    ledger, saved = executions.get_execution(eid), jobs.get_job(job["id"])
    assert ledger["status"] == "unknown" and ledger["error"] == "fixture uncertain business"
    assert ledger["delivery_outcome"] == "delivered" and received == [row["content"]]
    assert saved["last_status"] == "error" and saved["last_error"] == ledger["error"]
    assert saved["last_delivery_error"] is None and saved["repeat"]["completed"] == 0
    assert not counter.exists()  # uncertain business is NEVER a retry instruction
    assert record_unknown_worker_outcome(claimed, adapters=adapters, loop=env.loop)
    assert scheduler.drain_delivery_queue(adapters, env.loop) == 0
    assert saved == jobs.get_job(job["id"])


@pytest.mark.parametrize("lane", ["output", "raised-business"])
def test_direct_notice_publication_rechecks_the_real_fire_fence(env, monkeypatch, lane):
    monkeypatch.setattr(scheduler, "run_job", _REAL_RUN_JOB)
    job, counter = _script_job(env, "A")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True, force=True)
    original_lost = scheduler._FireOwnership.lost
    header = scheduler._job_doc_header

    def make_header(*args, **kwargs):
        if lane == "raised-business":
            raise RuntimeError("fixture post-script failure")
        return header(*args, **kwargs)

    def lost(ownership):
        verdict = original_lost(ownership)
        if not verdict:
            jobs.update_job(job["id"], {"fire_claim": dict(claimed["fire_claim"], by="replacement")})
        return verdict

    monkeypatch.setattr(scheduler, "_job_doc_header", make_header)
    # The heartbeat is real; the first post-business ownership sample races
    # with a replacement. The subsequent publication fence must refuse.
    monkeypatch.setattr(scheduler._FireOwnership, "lost", lost)
    scheduler.run_one_job(claimed)
    eid = executions.latest_execution(job["id"])["id"]
    assert counter.read_text() == "1"
    assert delivery_queue.get_status(eid) is None
    assert env.standalone == []
    assert jobs.get_job(job["id"])["fire_claim"]["by"] == "replacement"


def test_cancelled_no_agent_execution_does_not_publish_or_retry_a_message(env, monkeypatch):
    from cron.cancellation import request_cancel

    monkeypatch.setattr(scheduler, "run_job", _REAL_RUN_JOB)
    job, counter = _script_job(env, "cancel")
    eid = executions.create_execution(job["id"], source="test")["id"]
    job["execution_id"] = eid
    request_cancel(eid, reason="fixture operator cancel")
    scheduler.run_one_job(job)
    assert executions.get_execution(eid)["status"] == "cancelled"
    # Existing run_one_job cancellation is observed after the subprocess returns;
    # this verifies message exclusion, not preemptive cancellation of business.
    assert counter.read_text() == "1" and delivery_queue.get_status(eid) is None
