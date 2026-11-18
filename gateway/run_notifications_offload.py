"""Off-loop state.db work of the notification paths, owned until its thread has really finished.

The notification paths keep their blocking ``state.db`` reads/writes off the event loop (the loop-stall
W2 work). ``asyncio.to_thread`` alone is not enough at shutdown: the stop cancels the completion-batch
flushes and the watchers, and cancelling the coroutine that awaits ``to_thread`` does not stop its
default-executor thread. The executor quiesce never sees that executor, and ``asyncio.run`` joins it
only after ``stop()`` has closed SessionDB. A read left running by a cancelled flush could therefore
outlive ``SessionDB.close()`` and reopen the closed writer, minting a WAL generation behind the close's
checkpoint (#94736, #101064).

``owned_offload`` runs the call on the same default executor, still off the loop, and registers the
executor future through the runner's existing deferred-worker ownership (``_track_deferred_agent_worker``).
That owner is the one the SessionDB close gate already counts for default-executor work (``deferred_count``);
the drain, the shutdown watchdog snapshot and the active-work accounting count it too. The future stays
registered until the thread has finished the call: the caller awaits only a relay of it, so cancelling the
caller leaves the ownership in place. A caller cancelled before the call started turns the call into a
no-op, so it never touches the store. A call already running finishes, and while it runs the close is
skipped, as for any other live writer. The caller's contextvars (profile scope, secret scope) are carried
into the thread, as ``to_thread`` does. Results and exceptions reach a caller that is still waiting
unchanged; for an abandoned call, nobody waits for the result and a failure is logged at debug level and
consumed. (``asyncio.shield`` is not used: since 3.12 it reports a shielded future that fails after its
caller was cancelled to the loop's exception handler, an ERROR for a read nobody waits for.)
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger("gateway.run")


def _log_abandoned_failure(future: asyncio.Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.debug("Off-loop notification call failed after its caller was cancelled",
                     exc_info=future.exception())


async def owned_offload(runner: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``await asyncio.to_thread(fn, *args, **kwargs)``, with the thread owned by *runner* until it has finished.
    A *runner* without the deferred-worker owner (``None``, a bare double) gets plain ``to_thread`` semantics."""
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    abandoned = threading.Event()

    def call() -> Any:
        if abandoned.is_set():  # the caller was cancelled before this started: touch nothing
            return None
        return context.run(fn, *args, **kwargs)

    future = loop.run_in_executor(None, call)
    track = getattr(runner, "_track_deferred_agent_worker", None)
    if callable(track):
        track(future, None)  # no agent to interrupt: the call is bounded by its own store timeouts
    waiter = loop.create_future()  # cancelling the caller cancels this relay, never the executor future

    def relay(done: asyncio.Future) -> None:
        if waiter.done():
            return
        if done.cancelled():
            waiter.cancel()
        elif done.exception() is not None:
            waiter.set_exception(done.exception())
        else:
            waiter.set_result(done.result())
    future.add_done_callback(relay)
    try:
        return await waiter
    except asyncio.CancelledError:
        abandoned.set()
        future.add_done_callback(_log_abandoned_failure)
        raise
