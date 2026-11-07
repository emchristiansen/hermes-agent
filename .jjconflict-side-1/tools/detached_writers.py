"""Writers that outlive the unit or turn that started them, counted for the gateway's SessionDB close.

A delegated child's conversation runs on its own daemon worker and shares the gateway's SessionDB. When
the parent's wait ends first (inactivity timeout, stale verdict) the parent returns, and so does the
gateway turn or async-delegation unit around it, while the conversation is still unwinding and its
``child.close()`` is deferred to that worker's exit. Both still write. ``track()`` counts such a writer
until its ``settle()`` runs; the stop's executor quiesce waits for them (``join``) and keeps SessionDB
open for any still live (#101093).

Registration happens on the starting thread, before the counted unit or turn that started it can
finish, so a read taken after that starter was joined or sampled at zero sees it: the executor quiesce
joins the gateway pools and async-delegation workers before this registry, and the close gate reads it
again after sampling the starters it cannot join (API-server workers, cron, deferred and managed workers,
the final sweep; A4-E5-04). Not fenced: it extends an already-counted writer.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, wait as wait_futures
from typing import Callable

__all__ = ["join", "track"]

_lock = threading.Lock()
_live: set[Future] = set()


def track() -> Callable[[], None]:
    """Count one detached writer; returns its idempotent ``settle()``."""
    lifetime: Future = Future()
    with _lock:
        _live.add(lifetime)

    def settle() -> None:
        with _lock:
            _live.discard(lifetime)
        if not lifetime.done():
            lifetime.set_result(None)
    return settle


def join(timeout: float) -> int:
    """Wait up to ``timeout`` seconds for every detached writer; returns how many are still live.
    Re-snapshots until none is left: a writer started by a settling one registered before it settled."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _lock:
            pending = set(_live)
        remaining = deadline - time.monotonic()
        if not pending or remaining <= 0:
            return len(pending)
        wait_futures(pending, timeout=remaining)


def _reset_for_tests() -> None:
    with _lock:
        _live.clear()
