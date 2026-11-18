"""Busy-message snapshot reads remain owned after their actual receiving claim ends.

The real authority/admission and store-close gate are exercised. A deterministic
barrier parks only the physical SQLite reader; an inert finite turn controls its
natural completion. Cancellation must not let the physical read escape ownership.
"""
import asyncio
import contextlib

import pytest

from gateway import session_busy_ingress, session_finite
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority
from gateway.session_contract import SessionRef
from tests.gateway.test_loop_offload_io import _owner
from tests.gateway.test_notification_offload_ownership import _Park, _stop_ctx, _until
from tests.gateway.test_prompt_queued_transfer import BufferedAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_steering", [False, True], ids=["natural-completion", "cancelled-pass"])
async def test_busy_reader_outlives_claim_but_not_its_physical_owner(tmp_path, monkeypatch, cancel_steering):
    schedule = SessionAuthority._schedule
    runner, authority, _key, root = await _owner(tmp_path, monkeypatch)
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
    db, live = authority.db, authority.sessions[root]
    park = _Park(monkeypatch, db, inside=False, target=(session_busy_ingress, "list_session_admissions"))
    finish = asyncio.Event()
    executed = []

    async def finite_turn(owner, ref, row):
        executed.append(row["request_id"])
        owner.adopt_agent(ref.session_id, row["generation"], BufferedAgent())
        await finish.wait()
        return "receiving claim finished"

    monkeypatch.setattr(session_finite, "execute_finite_admission", finite_turn)
    authority._schedule(SessionRef(authority.profile_id, root))
    steering = None
    try:
        assert await _until(park.parked.is_set), "busy admission reader did not reach its real boundary"
        steering = live.steering
        finish.set()
        await asyncio.wait_for(live.task, 30)
        assert len(executed) == 1 and root not in authority.executing
        if cancel_steering:
            steering.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await steering
        else:
            assert not steering.done()

        # Let the claim's unrelated completion bookkeeping really finish. The
        # sole remaining work must be this reader, not an incidental close fence.
        assert await _until(lambda: runner._active_work_count() == runner._active_deferred_agent_worker_count())
        deferred = runner._active_deferred_agent_worker_count()
        active = runner._active_work_count()
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))
        closes_while_parked = list(park.closes)
        open_while_parked = db.__dict__.get("_conn") is not None

        # Always release and observe the real read before asserting the safety
        # contract: the failing baseline exposes its close/reopen, not a mock call.
        park.release.set()
        assert await _until(park.done.is_set)
        if not cancel_steering:
            await asyncio.wait_for(steering, 30)
        assert await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
        reopened = [event for event in park.events if event[0] == "reopen"]
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 30.0, _stop_ctx(runner))

        assert deferred == 1, {"deferred": deferred, "closes": closes_while_parked, "events": park.events}
        assert active >= 1
        assert closes_while_parked == [] and open_while_parked
        assert reopened == []
        assert park.closes and not any(item["reader_live"] for item in park.closes)
        assert db.__dict__.get("_conn") is None
    finally:
        finish.set()
        park.release.set()
        if steering is not None and not steering.done():
            steering.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await steering
        if not live.task.done():
            await asyncio.wait_for(live.task, 30)
        if park.parked.is_set():
            await _until(park.done.is_set)
        await _until(lambda: runner._active_deferred_agent_worker_count() == 0)
        park.real_close()
