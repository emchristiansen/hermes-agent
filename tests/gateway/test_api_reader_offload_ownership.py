"""API state.db readers stay owned by the gateway runner until their thread finishes (R2B-LIFETIME-01).

``_admitted_by_authority`` (``run_admission``) and ``await_peer_receipt`` (``peer_wait_admission`` and
``_receipt``) read state.db on the loop's default executor. Cancelling their awaiter does not stop that
thread, so the read must stay registered with the runner's deferred-worker owner, the one the SessionDB
close gate counts: shutdown must leave the database open while the read is live, the read must finish
on the open handle (no reopen of a closed writer), and the ownership must retire once the thread has
really finished. Real SessionDB, real runner accounting, the real close gate and the real ``stop()``;
the reader is the real function, paused just before it runs (derived from reviewer t_00ab1243/run5486's
probes ``test_review_reader_lifetime.py`` / ``test_review_reader_full_stop.py``).
"""
import asyncio
import contextlib
import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner
from tests.gateway.test_loop_offload_io import _owner, _seed
from tests.gateway.test_notification_offload_ownership import _Park, _awaiting, _stop_ctx, _until

SITES = ("recognized-admission", "peer-wait-admission", "peer-receipt")


def _start(site, runner, authority, key, root, tmp_path, monkeypatch, *, timeout=30):
    """Park the real reader of *site* before it runs and start its real caller."""
    from gateway.platforms import api_server_authority_runs as ar
    from gateway.platforms import api_server_bot_chat as bot
    from gateway.platforms import api_server_runs as runs
    from gateway import session_bot
    if site == "recognized-admission":
        park = _Park(monkeypatch, authority.db, inside=False, target=(ar, "run_admission"))
        adapter = SimpleNamespace(gateway_runner=runner, _run_admission_aliases={})
        return park, asyncio.create_task(runs._admitted_by_authority(adapter, "absent-run"))
    row = _seed(authority, root, key, "peer-" + site, {"text": "peer"}, terminal=False)
    record = {"status": "queued", "admission_id": row["admission_id"], "session_id": root,
              "delivery_id": "peer-" + site, "profile_home": str(tmp_path), "message": "peer"}
    target = (session_bot, "peer_wait_admission") if site == "peer-wait-admission" else (bot, "_receipt")
    park = _Park(monkeypatch, authority.db, inside=False, target=target)
    return park, asyncio.create_task(bot.await_peer_receipt(authority, record, timeout))


async def _cancel_while_parked(park, task, runner):
    """Reach the experiment (real reader parked before its DB operation), cancel its caller; observations."""
    assert await _until(park.parked.is_set), "real reader not reached"
    obs = {"chain": _awaiting(task), "deferred_before_cancel": runner._active_deferred_agent_worker_count()}
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    obs["deferred_after_cancel"] = runner._active_deferred_agent_worker_count()
    return obs


async def _release(park, runner, authority, obs, entry):
    """After the shutdown *entry* ran: record whether it closed under the live read, release the read,
    record its completion, reopen events and the ownership afterwards. Prints the per-case marker."""
    obs.update(entry=entry, open_after_entry=authority.db.__dict__.get("_conn") is not None,
               closes_while_live=[c for c in park.closes if c["reader_live"]])
    park.release.set()
    obs["reader_finished"] = await _until(park.done.is_set)
    await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
    obs["deferred_after_finish"] = runner._active_deferred_agent_worker_count()
    obs["events"] = list(park.events)
    print("REACHED-EXPERIMENT", obs, flush=True)  # reader parked at the real call, shutdown entry ran
    return obs


def _assert_owned(obs):
    assert obs["chain"][-1] == "owned_offload", obs
    assert obs["deferred_before_cancel"] == 1 and obs["deferred_after_cancel"] == 1, obs  # owned past the cancel
    assert obs["open_after_entry"] and obs["closes_while_live"] == [], obs  # never closed under the live read
    assert obs["reader_finished"], obs
    assert [e for e in obs["events"] if e[0] == "reopen"] == [], obs  # the read used the open handle
    assert obs["events"][-1] == ("released", False), obs
    assert obs["deferred_after_finish"] == 0, obs  # retired once the thread really finished


@pytest.mark.asyncio
@pytest.mark.parametrize("site", SITES)
async def test_cancelled_api_reader_keeps_close_gate_open_until_it_finishes(tmp_path, monkeypatch, site):
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    park, task = _start(site, runner, authority, key, root, tmp_path, monkeypatch)
    try:
        obs = await _cancel_while_parked(park, task, runner)
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30, _stop_ctx(runner))
        obs = await _release(park, runner, authority, obs, "close-gate")
        _assert_owned(obs)
        # Once the reader has really finished, the same gate closes the database as usual.
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30, _stop_ctx(runner))
        assert park.closes and not any(c["reader_live"] for c in park.closes)
    finally:
        park.release.set()
        await _until(park.done.is_set)
        park.real_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("site", SITES)
async def test_cancelled_api_reader_survives_full_stop_without_reopen(tmp_path, monkeypatch, site):
    import gateway.run as gw
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    park, task = _start(site, runner, authority, key, root, tmp_path, monkeypatch)
    try:
        obs = await _cancel_while_parked(park, task, runner)
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        runner.adapters = {}
        runner._restart_drain_timeout = 0.05
        with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
            await asyncio.wait_for(runner.stop(), 60)  # bounded: the live read does not hold the stop
        _assert_owned(await _release(park, runner, authority, obs, "full-stop"))
    finally:
        park.release.set()
        await _until(park.done.is_set)
        park.real_close()


_PROBE = contextvars.ContextVar("api_reader_probe", default=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("site", SITES)
async def test_api_reader_runs_off_loop_in_callers_context_and_retires(tmp_path, monkeypatch, site):
    """Uncancelled path: same answers as before, read off the loop with the caller's contextvars
    (profile/secret scope ride the same copy), retired after. (Ownership while it runs is asserted from
    the loop in the cancellation cases above: the executor thread may start before the loop thread has
    registered the future in the same synchronous step, so an in-thread count is not meaningful; the
    close gate runs on the loop thread and cannot interleave there.)"""
    from gateway.platforms import api_server_authority_runs as ar
    from gateway.platforms import api_server_bot_chat as bot
    from gateway.platforms import api_server_runs as runs
    from gateway import session_bot
    runner, authority, key, root = await _owner(tmp_path, monkeypatch)
    module, name = {"recognized-admission": (ar, "run_admission"),
                    "peer-wait-admission": (session_bot, "peer_wait_admission"),
                    "peer-receipt": (bot, "_receipt")}[site]
    real, seen = getattr(module, name), []
    loop_thread = threading.get_ident()

    def observed(*a, **kw):
        seen.append((threading.get_ident() != loop_thread, _PROBE.get()))
        return real(*a, **kw)
    monkeypatch.setattr(module, name, observed)
    _PROBE.set("caller-scope")
    if site == "recognized-admission":
        adapter = SimpleNamespace(gateway_runner=runner, _run_admission_aliases={})
        from hermes_state_runtime import admit_session_input
        admit_session_input(authority.db, epoch=authority.epoch, principal_id="api", session_id=root,
                            request_id="admitted-run", payload={"text": "x"})
        assert await runs._admitted_by_authority(adapter, "admitted-run") is True
        assert await runs._admitted_by_authority(adapter, "absent-run") is False
    else:
        row = _seed(authority, root, key, "peer-ok-" + site, {"text": "peer"}, terminal=False)
        record = {"status": "queued", "admission_id": row["admission_id"], "session_id": root,
                  "delivery_id": "peer-ok-" + site, "profile_home": str(tmp_path), "message": "peer"}
        current = await bot.await_peer_receipt(authority, record, 0.2)
        assert current["status"] in ("queued", "claimed")  # still pending when the wait lapses
    print("REACHED-EXPERIMENT", site, seen, flush=True)  # the real reader ran
    assert seen and all(s == (True, "caller-scope") for s in seen), seen
    assert runner._active_deferred_agent_worker_count() == 0
