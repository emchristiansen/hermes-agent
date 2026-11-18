"""Write a queued cron delivery's terminal outcome back onto the run that queued it.

An agent run hands its send to ``cron.delivery_queue`` and books ``last_status =
"delivery_queued"`` / ledger ``delivery_outcome = "queued"``. Once a gateway drain sends,
refuses, suppresses or fences the send ``unknown``, the job record and the execution
ledger must carry the real result (main's inline ``ok``/``delivered`` and
``delivery_failed``/``failed`` + ``last_delivery_error``); otherwise ``hermes cron list``
says "delivery still in progress" forever and refused sends never surface.

Both sides call :func:`settle_queued_delivery`: the drain right after it terminalizes a
row, and the run right after its own bookkeeping (a drain can finish before the run's
``mark_job_run`` lands). Every write is fenced on the execution id, so the call is
idempotent and never rewrites a later run's status. Within one execution the job write is
derived from the queue row as it is when the job-store lock is held, so the latest durable
attempt wins and a final outcome stays final (msg-retry-on-13f-8, MR-ACCOUNT-13).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Queue terminal status -> ledger delivery_outcome. ``unknown`` (owner died mid-send, never
# retried) is surfaced as a failure: the user cannot be told the result arrived.
_LEDGER_OUTCOME = {
    "delivered": "delivered", "suppressed": "suppressed", "failed": "failed", "unknown": "failed",
}


def _row_outcome(row: Optional[dict]) -> Optional[tuple[bool, Optional[str]]]:
    """``(final, delivery_error)`` for a queue row as it is now: final once terminal (with its error when it
    failed), not final while a failed attempt waits for its retry (that attempt's error). None when there is
    nothing to account: no row, or a send not yet attempted."""
    if not row:
        return None
    if row["status"] in _LEDGER_OUTCOME:
        if row["status"] in ("failed", "unknown"):
            return True, str(row.get("error") or f"delivery {row['status']}")
        return True, None
    if row["status"] in ("pending", "delivering") and row.get("error"):
        return False, str(row["error"])
    return None


def _settle_job_record(job_id: str, execution_id: str, observed_error: Optional[str] = None, *,
                       retrying: bool = False) -> bool:
    """Stamp the delivery outcome onto the job only while *execution_id* is still its current run: the ledger's
    latest attempt for the job AND the job's latest recorded completion.

    What is written comes from the queue row re-read under the job-store lock, never from the caller's
    snapshot (*observed_error*, *retrying*). Job writes are serialized by that lock and the row only moves
    forward, so a delayed caller cannot replace a newer attempt's error, and once the row is terminal every
    later call writes that final outcome again (a no-op): a final settlement stays final (MR-ACCOUNT-13).

    ``mark_job_run`` appends *execution_id* to ``canonical_completions`` in the same store write that books
    the run's status, so past that fence the run's bookkeeping has landed. A delivery handoff status
    (``delivery_queued``/``delivery_retrying``) becomes ``delivery_retrying`` while a failed attempt waits and
    ``ok``/``delivery_failed`` once final. Any other status is the run's own result (``error`` for a failed
    business run) and is kept; the delivery's error is shown beside it in ``last_delivery_error`` and cleared
    when the send is delivered (MR-ACCOUNT-12)."""
    from cron.delivery_queue import get_status
    from cron.executions import latest_execution
    from cron.jobs import _with_job, save_jobs
    from cron.jobs_error_text import sanitize_persisted_error

    latest = latest_execution(job_id)
    if latest is None or latest.get("id") != execution_id:
        return False

    def apply(jobs, _i, job):
        completions = job.get("canonical_completions") or []
        if not completions or completions[-1] != execution_id:
            return False
        row = get_status(execution_id)  # the latest durable attempt, read under the job-store lock
        outcome = _row_outcome(row)
        if outcome is None:
            return True
        final, delivery_error = outcome
        if (final, delivery_error) != (not retrying, observed_error):
            logger.debug("Cron delivery %s: settling the current queue state (final=%s, error=%r), not the "
                         "caller's earlier view (final=%s, error=%r)", execution_id, final, delivery_error,
                         not retrying, observed_error)
        # The queue row is redacted but not length-bounded; jobs.json gets the same cap as every
        # other last_delivery_error writer, and the no-op check compares the stored form.
        delivery_error = sanitize_persisted_error(delivery_error)
        if final and delivery_error and "content" not in row and job.get("last_delivery_error"):
            delivery_error = job["last_delivery_error"]  # a pruned row keeps no error text; keep the exact one
        status = job.get("last_status")
        if status in ("delivery_queued", "delivery_retrying"):
            status = ("delivery_failed" if delivery_error else "ok") if final else "delivery_retrying"
        if (status, delivery_error) == (job.get("last_status"), job.get("last_delivery_error")):
            return True
        job["last_status"] = status
        job["last_delivery_error"] = delivery_error
        save_jobs(jobs)
        return True

    return bool(_with_job(job_id, apply, missing=False))


def _terminal(execution_id: str) -> Optional[tuple[str, Optional[str]]]:
    """``(ledger_outcome, delivery_error)`` once the queue row is terminal, else None."""
    from cron.delivery_queue import get_status

    row = get_status(str(execution_id))
    outcome = _row_outcome(row)
    if outcome is None or not outcome[0]:
        return None
    return _LEDGER_OUTCOME[row["status"]], outcome[1]


def _retrying(execution_id: str) -> Optional[str]:
    """The last attempt's error while a failed send waits for its retry (queue row not terminal), else None."""
    from cron.delivery_queue import get_status

    outcome = _row_outcome(get_status(str(execution_id)))
    return outcome[1] if outcome is not None and not outcome[0] else None


def settled_outcome(execution_id: str) -> Optional[str]:
    """The ledger outcome a drain already reached for *execution_id* (None while queued)."""
    terminal = _terminal(execution_id)
    return terminal[0] if terminal else None


def settle_queued_delivery(job_id: Optional[str], execution_id: Optional[str]) -> bool:
    """Apply the queue's terminal outcome for *execution_id*; False while still in flight. A failed attempt
    waiting for its retry writes the truthful ``retrying`` outcome (with its error) and returns False."""
    if not job_id or not execution_id:
        return False
    terminal = _terminal(str(execution_id))
    if terminal is None:
        error = _retrying(str(execution_id))
        if error is not None:
            from cron.executions import settle_delivery_outcome

            settle_delivery_outcome(str(execution_id), "retrying")
            _settle_job_record(str(job_id), str(execution_id), error, retrying=True)
        return False
    from cron.executions import settle_delivery_outcome

    settle_delivery_outcome(str(execution_id), terminal[0])
    _settle_job_record(str(job_id), str(execution_id), terminal[1])
    return True


def settle_quietly(job_id: Optional[str], execution_id: Optional[str]) -> None:
    """Bookkeeping failure must not undo a terminal at-most-once send or fail its run."""
    try:
        settle_queued_delivery(job_id, execution_id)
    except Exception:
        logger.warning("Cron delivery %s: outcome write-back failed", execution_id, exc_info=True)
