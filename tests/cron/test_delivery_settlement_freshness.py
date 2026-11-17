"""msg-retry-on-13f-8 (review run18667): delivery settlement is truthful for a failed business run and fresh within
one execution. Real job, ledger and queue storage, the normal scheduler path, a harmless fake job and send, and an
accelerated queue clock.

- MR-ACCOUNT-12: a failed run's failure notice that is refused and retried keeps the run's own ``error`` status and
  shows the notice's retry error in ``last_delivery_error`` (and in the ordinary CLI consumers of the saved record);
  actual delivery clears it.
- MR-ACCOUNT-13: monotonic attempt freshness without a global serial send. A settlement held after reading an older
  attempt never replaces a newer attempt's error, a final settlement stays final, and a held settlement never
  overwrites a newer run's outcome; the held message itself is still retried and delivered.
Eric (canonical ea298f05): "I just want all messages to be delivered at least once."
"""
import threading

import pytest

from cron import delivery_outcome, delivery_queue, executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_cli import cron as cli


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    state = {"attempt": 0, "fail": True, "clock": 3_000_000.0, "sent": []}
    monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: state["clock"])

    async def send(platform, pconfig, chat_id, text, **kwargs):
        state["attempt"] += 1
        state["sent"].append((chat_id, text))
        refused = state["fail"]
        return {"success": not refused, "message_id": f"m{state['attempt']}",
                "error": f"settlement refusal {state['attempt']}" if refused else None}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    return state


def _hold_settlement(monkeypatch, marker):
    """Hold the job settlement whose observed error contains *marker* until released (it has already read the
    queue error it was called with)."""
    held, release, real = threading.Event(), threading.Event(), delivery_outcome._settle_job_record

    def settle(job_id, execution_id, error=None, **kwargs):
        if error and marker in error and kwargs.get("retrying"):
            held.set()
            assert release.wait(10), "the held settlement was never released"
        return real(job_id, execution_id, error, **kwargs)

    monkeypatch.setattr(delivery_outcome, "_settle_job_record", settle)
    return held, release


def _drain_on_thread(errors):
    def drain():
        try:
            scheduler.drain_delivery_queue({}, None)
        except BaseException as exc:  # health: allow BLE001 -- surfaced by the test's own assertion
            errors.append(repr(exc))
    thread = threading.Thread(target=drain, name="held-settlement-drain")
    thread.start()
    return thread


@pytest.mark.parametrize("timing", ["drain-after-bookkeeping", "drain-before-bookkeeping"])
def test_a_failed_runs_refused_failure_notice_shows_its_latest_retry_error_until_delivered(env, monkeypatch, timing):
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (False, "raw", "", "the business run failed"))
    if timing == "drain-before-bookkeeping":
        real_enqueue = delivery_queue.enqueue

        def enqueue(*args, **kwargs):
            row = real_enqueue(*args, **kwargs)
            scheduler.drain_delivery_queue({}, None)
            return row

        monkeypatch.setattr(delivery_queue, "enqueue", enqueue)
    job = jobs.create_job(prompt="failing business", schedule="every 1h", deliver="telegram:report",
                          failure_deliver="telegram:alerts")
    scheduler.run_one_job(job)
    if timing == "drain-after-bookkeeping":
        scheduler.drain_delivery_queue({}, None)
    monkeypatch.setattr(delivery_queue, "enqueue", delivery_queue.enqueue)
    execution = executions.latest_execution(job["id"])["id"]
    for attempt in (1, 2):
        row = delivery_queue.get_status(execution)
        saved, ledger = jobs.get_job(job["id"]), executions.get_execution(execution)
        assert row["status"] == "pending" and row["content"] and row["job_json"] != "{}"
        assert f"settlement refusal {attempt}" in row["error"] and row["attempts"] == attempt
        assert (ledger["status"], ledger["delivery_outcome"]) == ("failed", "retrying")
        assert saved["last_status"] == "error" and "the business run failed" in saved["last_error"]
        assert saved["last_delivery_error"] == row["error"]  # the failure notice's own latest retry error
        # The ordinary CLI consumers of this saved record show the business failure AND the notice's error.
        assert "the business run failed" in cli._last_run_display(saved)
        assert any(row["error"] in line for line in cli._job_warnings(saved))
        issues = cli._cron_doctor_issues_for_job(saved)
        assert any(i.startswith("last run failed") and "the business run failed" in i for i in issues)
        assert any(row["error"] in i for i in issues)
        env["clock"] = row["retry_after"]
        if attempt == 2:
            env["fail"] = False
        scheduler.drain_delivery_queue({}, None)
    saved, ledger = jobs.get_job(job["id"]), executions.get_execution(execution)
    assert delivery_queue.get_status(execution)["status"] == "delivered"
    assert (ledger["status"], ledger["delivery_outcome"]) == ("failed", "delivered")
    assert (saved["last_status"], saved["last_delivery_error"]) == ("error", None)
    assert "the business run failed" in saved["last_error"]
    assert not any("not delivered" in line for line in cli._job_warnings(saved))
    assert not any("not delivered" in i for i in cli._cron_doctor_issues_for_job(saved))
    assert [chat for chat, _ in env["sent"]] == ["alerts"] * 3
    assert env["sent"][0] == env["sent"][1] == env["sent"][2]  # the same notice each time; duplicates accepted


def test_a_held_settlement_never_replaces_a_newer_attempts_error(env, monkeypatch):
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the report", None))
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="telegram:report")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    held, release = _hold_settlement(monkeypatch, "settlement refusal 1")
    errors = []
    thread = _drain_on_thread(errors)
    try:
        assert held.wait(10), "the first failed attempt was not settled"
        env["clock"] = delivery_queue.get_status(execution)["retry_after"]
        scheduler.drain_delivery_queue({}, None)  # the next due send fails too
        newer = delivery_queue.get_status(execution)
        assert "settlement refusal 2" in newer["error"] and newer["attempts"] == 2
        assert jobs.get_job(job["id"])["last_delivery_error"] == newer["error"]
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    row, saved = delivery_queue.get_status(execution), jobs.get_job(job["id"])
    assert row["attempts"] == 2 and "settlement refusal 2" in row["error"]
    assert (saved["last_status"], saved["last_delivery_error"]) == ("delivery_retrying", row["error"])
    assert executions.get_execution(execution)["delivery_outcome"] == "retrying"
    env["fail"] = False
    env["clock"] = row["retry_after"]
    scheduler.drain_delivery_queue({}, None)
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(execution)["delivery_outcome"] == "delivered"


@pytest.mark.parametrize("business_ok", [True, False], ids=["business-ok", "business-failed"])
def test_a_final_settlement_stays_final_after_a_held_retrying_settlement(env, monkeypatch, business_ok):
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (
        (True, "raw", "the report", None) if business_ok else (False, "raw", "", "the business run failed")))
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="telegram:report",
                          failure_deliver="telegram:alerts")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    held, release = _hold_settlement(monkeypatch, "settlement refusal 1")
    errors = []
    thread = _drain_on_thread(errors)
    try:
        assert held.wait(10), "the first failed attempt was not settled"
        env["fail"] = False
        env["clock"] = delivery_queue.get_status(execution)["retry_after"]
        scheduler.drain_delivery_queue({}, None)  # the retry delivers: the final settlement
        assert delivery_queue.get_status(execution)["status"] == "delivered"
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok" if business_ok else "error", None)
    assert executions.get_execution(execution)["delivery_outcome"] == "delivered"
    assert len(env["sent"]) == 2 and env["sent"][0] == env["sent"][1]  # the duplicate is accepted


def test_a_held_settlement_never_overwrites_a_newer_runs_outcome(env, monkeypatch):
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the report", None))
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="telegram:report")
    scheduler.run_one_job(job)
    first = executions.latest_execution(job["id"])["id"]
    held, release = _hold_settlement(monkeypatch, "settlement refusal 1")
    errors = []
    thread = _drain_on_thread(errors)
    try:
        assert held.wait(10), "the first failed attempt was not settled"
        env["fail"] = False
        scheduler.run_one_job(jobs.get_job(job["id"]))  # a newer run of the same job
        second = executions.latest_execution(job["id"])["id"]
        assert second != first
        scheduler.drain_delivery_queue({}, None)  # the first send is in its backoff: only the newer one is sent
        assert executions.get_execution(second)["delivery_outcome"] == "delivered"
        assert (jobs.get_job(job["id"])["last_status"], jobs.get_job(job["id"])["last_delivery_error"]) == ("ok", None)
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)  # the newer run's outcome stands
    assert executions.get_execution(second)["delivery_outcome"] == "delivered"
    assert executions.get_execution(first)["delivery_outcome"] == "retrying"
    row = delivery_queue.get_status(first)
    assert row["status"] == "pending" and row["content"]  # the older message is still retried, never dropped
    env["clock"] = row["retry_after"]
    scheduler.drain_delivery_queue({}, None)
    assert executions.get_execution(first)["delivery_outcome"] == "delivered"
    assert (jobs.get_job(job["id"])["last_status"], jobs.get_job(job["id"])["last_delivery_error"]) == ("ok", None)
