"""A SIGTERM stop exits 1 by contract, whatever the SessionDB close did.

A stop that the gateway did not plan (a bare SIGTERM: ``systemctl restart``/``stop``, a
container restart, OOM, kill) carries no restart request, so the exit verdict is a failure
exit (code 1). That lets ``Restart=on-failure`` revive the gateway. The verdict only reads the
runner's failure flag, its explicit exit code and the restart request. It does not read the
SessionDB close.

Production 2026-11-06 19:43:24Z: on the RESTART=both connect landing, the old interpreter
logged "Exiting with code 1 (signal-initiated shutdown without restart request)" and systemd
recorded status=1/FAILURE. In that same stop the executor quiesce kept SessionDB open for 5
live workers. These tests pin that the exit code comes from the signal verdict alone. A close
skipped for a live executor worker does not change it, and neither does a close that went ahead
while a settled final sweep's thread was still unwinding (QUIESCE-COMPOSITION-I1: the sweep's
work is done once its result is in, so that thread is not a writer).
"""

import logging
import threading
from types import SimpleNamespace

import pytest

import gateway.run as gw_mod
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.run_shutdown import _resolve_gateway_exit_verdict
from tests.gateway.test_shutdown_executor_quiesce import (
    _FakeGateway,
    _hold_final_sweep_thread,
    _probe_close,
)


def _verdict_runner(*, restart_requested=False, via_service=False, exit_code=None, failure=False):
    return SimpleNamespace(
        should_exit_with_failure=failure, exit_reason=None, exit_code=exit_code,
        _restart_requested=restart_requested, _restart_via_service=via_service,
    )


def test_a_signal_stop_without_a_restart_request_is_a_failure_exit(caplog):
    caplog.set_level(logging.INFO, logger="gateway.run")
    assert _resolve_gateway_exit_verdict(_verdict_runner(), signal_initiated_shutdown=True) is False
    assert any(
        "Exiting with code 1 (signal-initiated shutdown without restart request)" in r.getMessage()
        for r in caplog.records
    )
    # The same stop without the signal is a clean exit; a requested restart is not a failure.
    assert _resolve_gateway_exit_verdict(_verdict_runner(), signal_initiated_shutdown=False) is True
    assert _resolve_gateway_exit_verdict(
        _verdict_runner(restart_requested=True), signal_initiated_shutdown=True
    ) is True
    with pytest.raises(SystemExit) as exc:
        _resolve_gateway_exit_verdict(
            _verdict_runner(restart_requested=True, via_service=True), signal_initiated_shutdown=True
        )
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE


class _VerdictGateway(_FakeGateway):
    should_exit_with_failure = gw_mod.GatewayRunner.should_exit_with_failure
    exit_reason = gw_mod.GatewayRunner.exit_reason
    exit_code = gw_mod.GatewayRunner.exit_code


def _arm_executor_worker(gw):
    release, started = threading.Event(), threading.Event()

    def _stuck():
        started.set()
        release.wait(5.0)

    future = gw._executor.submit(_stuck)
    assert started.wait(2.0), "worker never started"
    return release, future


@pytest.mark.asyncio
async def test_a_kept_open_session_db_does_not_decide_the_exit_code(monkeypatch):
    events = []
    gw = _VerdictGateway(events)
    release, future = _arm_executor_worker(gw)
    monkeypatch.setattr(gw_mod, "_EXECUTOR_QUIESCE_TIMEOUT", 0.0)
    _probe_close(gw, events, [])
    try:
        await gw_mod.GatewayRunner.stop(gw)
        closed = list(events)
    finally:
        release.set()
        future.result(timeout=5)

    assert closed == [], f"SessionDB closed despite a live executor worker: {closed}"
    assert gw.should_exit_with_failure is False and gw.exit_code is None
    assert _resolve_gateway_exit_verdict(gw, signal_initiated_shutdown=False) is True
    assert _resolve_gateway_exit_verdict(gw, signal_initiated_shutdown=True) is False


@pytest.mark.asyncio
async def test_a_settled_final_sweep_still_unwinding_closes_and_does_not_decide_the_exit_code(
    monkeypatch,
):
    events = []
    gw = _VerdictGateway(events)
    threads = _hold_final_sweep_thread(monkeypatch, exit_on_join=False)
    _probe_close(gw, events, threads)
    try:
        await gw_mod.GatewayRunner.stop(gw)
        closed = list(events)
    finally:
        for thread in threads:
            thread.release()

    assert len(threads) == 1, threads
    assert closed == ["closing_flag:True", "sweep_thread_alive:True", "close:session_db"], (
        f"a settled final sweep held the close: {closed}"
    )
    assert gw.should_exit_with_failure is False and gw.exit_code is None
    assert _resolve_gateway_exit_verdict(gw, signal_initiated_shutdown=False) is True
    assert _resolve_gateway_exit_verdict(gw, signal_initiated_shutdown=True) is False
