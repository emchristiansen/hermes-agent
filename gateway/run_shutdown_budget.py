"""One deadline for a stop: every bounded wait after the settle draws on the shutdown watchdog's leash.

``_stop_impl`` records ``runner._stop_deadline`` (arm time + the effective leash) when it arms the
watchdog. Each later wait is clamped to what is left after the reserve of the phases that still
follow it, in stop order: agent finalize/cleanup → adapter teardown → final tool kill → executor
quiesce → SessionDB close → exit-state writes. A fixed reserve for "the rest of the tail" did not
hold once a later phase was itself slow (a 4s adapter disconnect still overran a 7s leash), so the
reserves are derived from the bounded steps that follow, and those steps are clamped in turn.
Outside a stop (no deadline) every wait keeps its own timeout unchanged.
"""

from __future__ import annotations

import time

# SessionDB close and the exit-state writes, after the executor quiesce (the quiesce kept 1s for them).
STOP_CLOSE_RESERVE_SECONDS = 1.0
# The share of its 2s bound the executor quiesce is guaranteed after the adapters and the final tool
# kill: it waits for a live writer as long as the deadline allows, and a writer still live when it
# gives up keeps SessionDB open (#101093), so a shorter quiesce is safe where a cut close is not.
STOP_QUIESCE_FLOOR_SECONDS = 0.5


def stop_wait_budget(runner, timeout: float, *, reserve: float) -> float:
    """``timeout``, clamped during a stop to the leash left after ``reserve`` (never negative)."""
    deadline = getattr(runner, "_stop_deadline", None)  # None outside a stop and on bare test doubles
    if deadline is None:
        return timeout
    return max(0.0, min(timeout, deadline - time.monotonic() - reserve))


def quiesce_reserve() -> float:
    """What the adapter teardown and the final tool kill leave: the quiesce floor + the close."""
    return STOP_QUIESCE_FLOOR_SECONDS + STOP_CLOSE_RESERVE_SECONDS


def agent_wait_reserve(runner) -> float:
    """What the agent finalize/cleanup waits leave: one adapter's full teardown bound (each further
    adapter wait is clamped to ``quiesce_reserve`` itself), then the quiesce and the close."""
    from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
    timeout_fn = getattr(runner, "_adapter_disconnect_timeout_secs", None)  # absent on bare test doubles
    adapter = timeout_fn() if callable(timeout_fn) else _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
    return adapter + quiesce_reserve()
