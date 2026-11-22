"""One uncounted owner restart per card before any attempt of it is actually admitted.

A Kanban submitter (``kanban_worker_client``) waits ``OWNER_WAIT_S`` for an owner that is
``draining`` or ``starting``. When the wait runs out it exits 1, and the crash sweep books a counted
crash, even though nothing was submitted. If the owner was restarting, that spends the card's failure
budget on the host's restart. This module grants exactly one exception per card. The grace then stays
spent until the owner actually admits an attempt of the card again.

* **Pre-submission only.** The caller asks only when every connect attempt failed in
  ``connect_gateway``'s ensure phase. No connection was made, so no ``kanban.run`` was sent.
* **Owner replacement evidence.** The control home's ``gateway_state.json`` named a live gateway
  incarnation, ``(pid, start_time)``, when the submitter started (:func:`live_owner`). By the
  deadline it names another incarnation, and the first one has ended (:func:`replacement`). An
  unchanged, missing, malformed, other-home or uncomparable record is no evidence. This is not a
  forecast that the owner will return.
* **At most one grace.** The grace is spent when :data:`GRACE_EVENT` is recorded on the run, in the
  same board transaction that checks none was spent since the card's last ACTUAL admission
  (:func:`record_grace`): the newest ``owner_admitted`` marker whose admission the owner's queue
  accepted (:func:`accepted`). The marker alone is not enough, since the owner commits it before
  submitting. Nothing but an actual admission opens a new window. Counted failures, rate limits,
  reclaims, unblocks, ``ready``, completion and reopen never write that admission. Event GC keeps
  :data:`GRACE_EVENT` rows (``kanban_db_workspace._prune_terminal_events``), so an aged, pruned and
  reopened card still has its grace spent. A pruned marker or a retired-and-reclaimed admission
  only narrows the window (fewer graces).
* **Consumer fence.** The crash sweep books a 79 exit on an unadmitted run as ``owner_restarted``
  only when that run carries the grace event under its claim
  (``kanban_owner_recovery.presubmit_restart_fence``).

Every unreadable or ambiguous reading fails closed: no grace, the ordinary counted exit.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

GRACE_EVENT = 'owner_restart_grace'


def control_home():
    """The home whose gateway answers this profile's clients, the same target ``connect_gateway``
    ensures: the multiplexer root that serves it, else the profile home itself."""
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import gateway_profile_home
    from hermes_cli.gateway_runtime_multiplex import multiplexer_serves_home
    home = gateway_profile_home(get_hermes_home())
    return multiplexer_serves_home(home) or home


def baseline():
    """``(control home, its live owner incarnation)`` when an attempt starts; ``(None, None)`` when
    the control home cannot be resolved."""
    try:
        home = control_home()
    except Exception:  # fail closed: no control home, no baseline
        logger.warning('owner control home unresolved; no owner-restart grace', exc_info=True)
        return None, None
    return home, live_owner(home)


def _recorded_owner(home):
    """``(pid, start_time)`` the home's ``gateway_state.json`` names for its own gateway, else None."""
    from gateway.status import read_runtime_status
    from hermes_cli.gateway_runtime import _canonical_home
    record = read_runtime_status(Path(home) / 'gateway_state.json')
    if not isinstance(record, dict):
        return None
    pid, start = record.get('pid'), record.get('start_time')
    if type(pid) is not int or pid <= 0 or type(start) is not int:
        return None
    if 'hermes_home' in record:  # absent: a legacy record; present: a non-empty string naming this home
        named = record['hermes_home']
        if not isinstance(named, str) or not named or _canonical_home(named) != _canonical_home(home):
            return None
    return pid, start


def _incarnation(owner):
    """True when ``owner`` is running now, False when it has provably ended, None when unknown."""
    from gateway.status import get_process_start_time, start_time_fingerprints_match
    from hermes_cli import kanban_db as kb
    pid, start = owner
    if not kb._pid_alive(pid):
        return False
    current = get_process_start_time(pid)
    return None if current is None else start_time_fingerprints_match(start, current)


def live_owner(home):
    """The live gateway incarnation the home's record names, else None."""
    try:
        owner = _recorded_owner(home)
        return owner if owner is not None and _incarnation(owner) is True else None
    except Exception:  # fail closed: an unreadable record or identity is no baseline
        logger.warning('owner identity of %s unreadable; no owner-restart grace', home, exc_info=True)
        return None


def replacement(baseline, home):
    """The incarnation that replaced ``baseline`` on ``home``, else None. ``baseline`` must have
    ended, and the record must name a different one."""
    if baseline is None:
        return None
    try:
        now = _recorded_owner(home)
        if now is None:
            return None
        from gateway.status import start_time_fingerprints_match
        if now[0] == baseline[0] and start_time_fingerprints_match(baseline[1], now[1]):
            return None
        return now if _incarnation(baseline) is False else None
    except Exception:  # fail closed: an unreadable record or identity is no evidence
        logger.warning('owner identity of %s unreadable; no owner-restart grace', home, exc_info=True)
        return None


def record_grace(board_db, params, old, new, home) -> bool:
    """Spend the card's grace on the submitter's exact run. True only when it was recorded now."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    try:
        task_id, run_id, claim = params['task_id'], params['run_id'], params['claim_lock']
        path = Path(board_db)
        if not path.is_file() or type(run_id) is not int:
            return False
        with connect_closing(path) as conn, kb.write_txn(conn):
            task = conn.execute('SELECT status, current_run_id, claim_lock FROM tasks WHERE id=?', (task_id,)).fetchone()
            if task is None or (task[0], task[1], task[2]) != ('running', run_id, claim):
                return False
            markers = conn.execute("SELECT run_id, payload FROM task_events WHERE task_id=? AND kind='owner_admitted' "
                                   "ORDER BY run_id DESC, id DESC", (task_id,)).fetchall()
            if any(marker_run is None or marker_run >= run_id for marker_run, _payload in markers):
                return False  # this run carries a marker: not a pre-submission attempt
            admitted = next((marker_run for marker_run, payload in markers if accepted(payload)), 0)
            if conn.execute('SELECT 1 FROM task_events WHERE task_id=? AND kind=? AND run_id > ?',
                            (task_id, GRACE_EVENT, admitted)).fetchone():
                return False  # spent since the last actual admission
            kb._append_event(conn, task_id, GRACE_EVENT, {
                'claim_lock': claim, 'pid': os.getpid(), 'control_home': str(home),
                'old_owner': {'pid': old[0], 'start_time': old[1]},
                'new_owner': {'pid': new[0], 'start_time': new[1]}}, run_id=run_id)
        return True
    except Exception:  # fail closed: a board that cannot take the write grants nothing
        logger.warning('owner-restart grace not recorded on %s', board_db, exc_info=True)
        return False


def accepted(marker_payload) -> bool:
    """Whether the owner's queue actually accepted the admission an ``owner_admitted`` marker names.

    ``session_kanban.run_task`` commits the marker in its board transaction BEFORE
    ``authority.submit`` queues the turn, so a marker alone is no admission: an owner that dies or
    fails between the two leaves a marker over nothing. Accepted means the marker's own owner ledger
    (``db``) holds its ``(session_id, request_id)`` admission, live in ``session_admissions`` or
    retired into its terminal tombstone. Any unreadable reading is no admission (fail closed: the
    window does not reopen)."""
    import json
    import sqlite3
    from contextlib import closing
    try:
        binding = json.loads(marker_payload)
        db, sid, request = binding['db'], binding['session_id'], binding['request_id']
        if not all(isinstance(value, str) and value for value in (db, sid, request)) or not Path(db).is_file():
            return False
        with closing(sqlite3.connect(Path(db).as_uri() + '?mode=ro', uri=True, timeout=5)) as ledger:
            if ledger.execute('SELECT 1 FROM session_admissions WHERE target_session_id=? AND request_id=?',
                              (sid, request)).fetchone():
                return True
            from hermes_state_terminal import ADMISSION_PREFIX
            return ledger.execute(
                "SELECT 1 FROM state_meta WHERE key GLOB ? AND json_valid(value) "
                "AND json_extract(value, '$.target_session_id')=? AND json_extract(value, '$.request_id')=?",
                (ADMISSION_PREFIX + '*', sid, request)).fetchone() is not None
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        logger.warning('owner admission behind marker unreadable; treated as not admitted', exc_info=True)
        return False


def grace_backs_run(conn, task_id, claim_lock) -> bool:
    """Whether a 79 exit of the task's current run may book ``owner_restarted``: the run was admitted
    (the owner's own accounting decides, unchanged), or it carries its grace under ``claim_lock``."""
    return run_backing(conn, task_id, claim_lock) is not None


def run_backing(conn, task_id, claim_lock):
    """What backs a 79 exit of the task's current run: ``'admitted'`` (the run carries an
    ``owner_admitted`` marker, so a turn may have been sent and the owner's own accounting decides),
    ``'grace'`` (no marker, but the run carries its pre-submission grace under ``claim_lock``: nothing
    was sent), else None."""
    import json
    run = conn.execute('SELECT current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
    if run is None or run[0] is None:
        return None
    if conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
                    (task_id, run[0])).fetchone():
        return 'admitted'
    for (payload,) in conn.execute('SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind=?',
                                   (task_id, run[0], GRACE_EVENT)).fetchall():
        try:
            if json.loads(payload).get('claim_lock') == claim_lock:
                return 'grace'
        except (ValueError, TypeError, AttributeError):
            continue
    return None
