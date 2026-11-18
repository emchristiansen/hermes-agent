"""msg-retry-on-13f-7 (review18625 MR-ACCOUNT-10): a failed send attempt is accounted truthfully across the queue,
the job record, the execution ledger and the CLI. It is retrying with its exact error, never terminal, never pruned,
and it settles to the final outcome when a retry delivers. Writes are fenced to their own execution. Real job,
ledger and queue storage; only the outbound platform send is a local double.
Eric (canonical ea298f05): "I just want all messages to be delivered at least once."
"""
import pytest

from cron import delivery_queue, executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    clock = {"now": 2_000_000.0}
    monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: clock["now"])
    state = {"sent": [], "fail": set(), "attempt": 0, "clock": clock}

    async def send(platform, pconfig, chat_id, text, **kwargs):
        state["attempt"] += 1
        state["sent"].append(text)
        refused = any(marker in text for marker in state["fail"])
        return {"success": not refused, "message_id": f"m{state['attempt']}",
                "error": f"transport refused attempt {state['attempt']}" if refused else None}

    monkeypatch.setattr("gateway.outbound.send._send_to_platform", send)
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "body-" + job["prompt"], None))
    return state


def _run(prompt):
    job = jobs.create_job(prompt=prompt, schedule="every 1h", deliver="telegram:fixture")
    scheduler.run_one_job(job)
    return job, executions.latest_execution(job["id"])["id"]


def test_a_permanent_failure_stays_retrying_with_its_latest_error_while_a_healthy_job_delivers(env):
    """Repeated permanent failure: every round the job reads ``delivery_retrying`` with the exact error of its
    latest attempt (equal to the queue row's error), the ledger reads ``retrying``, the payload is kept, and the
    backoff stays bounded (at most 1 h). A healthy job queued after it is delivered by the first drain each round,
    with its own outcome ``ok``/``delivered``."""
    env["fail"].add("body-failing")
    failing, failing_exec = _run("failing")
    errors = []
    for round_ in range(6):
        healthy, healthy_exec = _run(f"healthy{round_}")
        before = env["clock"]["now"]
        scheduler.drain_delivery_queue({}, None)
        row = delivery_queue.get_status(failing_exec)
        saved = jobs.get_job(failing["id"])
        assert row["status"] == "pending" and "body-failing" in row["content"], round_
        assert saved["last_status"] == "delivery_retrying", round_
        assert saved["last_delivery_error"] == row["error"] and "transport refused attempt" in row["error"]
        assert executions.get_execution(failing_exec)["delivery_outcome"] == "retrying"
        assert 0 < row["retry_after"] - before <= delivery_queue.RETRY_BACKOFF_MAX_SECONDS
        errors.append(row["error"])
        good = jobs.get_job(healthy["id"])
        assert (good["last_status"], good["last_delivery_error"]) == ("ok", None), round_
        assert executions.get_execution(healthy_exec)["delivery_outcome"] == "delivered"
        env["clock"]["now"] = row["retry_after"]  # the failing send's next retry is due
    assert len(set(errors)) == 6  # each round's error is that round's own attempt
    env["fail"].clear()
    scheduler.drain_delivery_queue({}, None)
    saved = jobs.get_job(failing["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(failing_exec)["delivery_outcome"] == "delivered"
    assert sum("body-failing" in text for text in env["sent"]) == 7  # six refused attempts and the delivered one


def test_an_older_executions_retry_never_overwrites_a_newer_runs_outcome(env):
    """Stale-execution fence: a retrying write-back, and later a final one, for execution 1 leave the job record of
    the newer execution 2 and its ledger row unchanged; execution 1's own ledger row settles."""
    env["fail"].add("body-same")
    job = jobs.create_job(prompt="same", schedule="every 1h", deliver="telegram:fixture")
    scheduler.run_one_job(job)
    first = executions.latest_execution(job["id"])["id"]
    scheduler.drain_delivery_queue({}, None)
    assert jobs.get_job(job["id"])["last_status"] == "delivery_retrying"
    env["fail"].clear()
    scheduler.run_one_job(jobs.get_job(job["id"]))
    second = executions.latest_execution(job["id"])["id"]
    assert second != first
    scheduler.drain_delivery_queue({}, None)  # the first send is still in its backoff: only the second is sent
    assert (jobs.get_job(job["id"])["last_status"], jobs.get_job(job["id"])["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(second)["delivery_outcome"] == "delivered"
    env["fail"].add("body-same")
    env["clock"]["now"] = delivery_queue.get_status(first)["retry_after"]
    scheduler.drain_delivery_queue({}, None)  # the first execution's retry fails again: a stale retrying write-back
    assert delivery_queue.get_status(first)["status"] == "pending"
    assert (jobs.get_job(job["id"])["last_status"], jobs.get_job(job["id"])["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(second)["delivery_outcome"] == "delivered"
    assert executions.get_execution(first)["delivery_outcome"] == "retrying"
    env["fail"].clear()
    env["clock"]["now"] = delivery_queue.get_status(first)["retry_after"]
    scheduler.drain_delivery_queue({}, None)  # and its final write-back
    assert executions.get_execution(first)["delivery_outcome"] == "delivered"
    assert (jobs.get_job(job["id"])["last_status"], jobs.get_job(job["id"])["last_delivery_error"]) == ("ok", None)
    assert executions.get_execution(second)["delivery_outcome"] == "delivered"


def test_the_cli_shows_a_retrying_delivery_with_its_full_error():
    """``cron list`` shows the retry and the complete error (no truncation); ``cron doctor`` reports the delivery
    error and does not call the run itself failed."""
    from hermes_cli import cron as cli
    error = "transport refused: " + "detail " * 40 + "END-OF-ERROR"
    job = {"id": "j", "name": "n", "last_status": "delivery_retrying", "last_delivery_error": error,
           "last_error": None}
    shown = cli._last_run_display(job)
    assert "retried" in shown and error in shown
    issues = cli._cron_doctor_issues_for_job(dict(job, last_delivery_error="transport refused attempt 3"))
    assert not any(i.startswith("last run failed") for i in issues)
    assert any("transport refused attempt 3" in i for i in issues)
