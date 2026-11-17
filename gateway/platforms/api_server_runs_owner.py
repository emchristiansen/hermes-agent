"""Durable run records whose owner may be gone: the stale-owner ``interrupted`` status and the replay lookup's owner
check. Moved out of ``gateway/platforms/api_server_runs.py`` (code-health FILE_LINES after the v0.23.1 merge)."""

import time
from typing import Any

from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES


def _stale_owner_status(record: dict[str, Any]) -> dict[str, Any] | None:
    """The ``interrupted`` status a durable non-terminal record whose owner is gone becomes."""
    from gateway.platforms import api_server_runs as runs  # its ``_owner_alive`` (tests patch it there)
    status = dict(record["status"])
    if status.get("status") in TERMINAL_STATUSES or runs._owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)):
        return None
    status.update(
        status="interrupted", error="The gateway restarted before this run settled.",
        last_event="run.interrupted", updated_at=time.time())
    return status


def _lookup_for_replay(store, scope, key, fingerprint, retention_until):
    """Store-worker lookup; a provisional row also says whether its owner still lives."""
    from gateway.platforms import api_server_runs as runs  # its ``_owner_alive`` (tests patch it there)
    outcome, record = store.lookup(scope, key, fingerprint, retention_until=retention_until)
    if record is not None and record.get("admitted") is False:
        record["owner_alive"] = runs._owner_alive(record["owner_pid"], record["owner_started"])
    return outcome, record
