"""Read-only canonical admission fences for dispatcher reclaim paths."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
import psutil


def _owner_receipt(conn, task_id):
    """The parsed ``owner_admitted`` receipt of the task's CURRENT run, or None.

    The receipt is the owner's pointer to its own admission ledger. No receipt means the run
    never reached an owner (a plain dispatcher worker): nothing about it is ambiguous.
    """
    marker = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind='owner_admitted' ORDER BY e.id DESC LIMIT 1",
        (task_id,)).fetchone()
    if marker is None:
        return None
    try:
        receipt = json.loads(marker['payload'])
    except (ValueError, TypeError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def _admission_row(receipt):
    """``(status, outcome)`` from the owner ledger ``receipt`` names, or None when unreadable.

    None is "the owner said nothing we can read" — never "the owner said the attempt ended".
    """
    if not receipt:
        return None
    try:
        with closing(sqlite3.connect(Path(receipt['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            row = owner.execute('SELECT status,outcome FROM session_admissions WHERE target_session_id=? AND request_id=?',
                                (receipt['session_id'], receipt['request_id'])).fetchone()
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
        return None
    return (row[0], row[1]) if row is not None else None


def owner_reclaim_paused(conn, task_id):
    """Missing/ambiguous owner receipts never authorize another attempt."""
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return False
    try:
        row = _admission_row(receipt)
        if row is not None and row[0] == 'terminal':
            # Interrupted still pauses the generic reclaim paths; owner_attempt_provably_over decides
            # when nothing can still be running it.
            return row[1] == 'interrupted'
        if row is None or row[0] in {'queued', 'unknown'}:
            return True
        from hermes_cli.process_identity import worker_birth
        return worker_birth(psutil.Process(receipt['pid'])) != receipt['birth']
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return True


def _latest_run_event(conn, task_id, kind):
    """The newest ``kind`` payload on the task's CURRENT run, or ``{}``.

    A payload that cannot be read is not a payload that says the attempt is over, so
    callers treat ``{}`` as "no evidence" rather than as an absence.
    """
    row = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind=? ORDER BY e.id DESC LIMIT 1",
        (task_id, kind)).fetchone()
    try:
        payload = json.loads(row['payload']) if row else {}
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _host_local_lock(lock) -> bool:
    """True when ``lock`` was issued by this host — the only pids we can judge."""
    from hermes_cli import kanban_db as kb
    return bool(lock) and str(lock).startswith(kb._host_prefix())


def bound_interpreter_gone(conn, task_id):
    """True when the owner bound an interpreter to the task's current run and it is dead."""
    bound = _latest_run_event(conn, task_id, 'worker_bound')
    if not bound:
        return False
    from hermes_cli.kanban_db_dispatch import _worker_alive
    return not _worker_alive(bound.get('pid'), bound.get('started_at'))


def _owner_process_gone(receipt) -> bool:
    """True when the receipt's own pid is gone, or its number now belongs to another process.

    The identity rule ``owner_reclaim_paused`` already applies: a live pid whose ``birth``
    differs is a recycled number, not the owner. A zombie keeps its birth until reaped but has
    already exited, so liveness goes through the dispatcher's zombie-aware ``_pid_alive``. A live pid we cannot compare (no ``birth``
    recorded) and an unreadable one are never proof of death.
    """
    try:
        pid = int(receipt['pid'])
    except (KeyError, TypeError, ValueError):
        return False
    from hermes_cli import kanban_db as kb
    if not kb._pid_alive(pid):
        return True  # absent, or an exited zombie still waiting to be reaped
    try:
        from hermes_cli.process_identity import worker_birth
        current = worker_birth(psutil.Process(pid))
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False
    birth = receipt.get('birth')
    return birth is not None and current != birth


def _owner_settled_admission(receipt) -> bool:
    """True when the receipt's admission is ``terminal`` in the owner's own ledger.

    A live owner settles an attempt it can no longer run (a lost managed worker, before or after
    binding) ``terminal``/``interrupted`` and runs it no further; terminal says the owner is done
    executing it, not what its effects were. An unreadable ledger is never proof."""
    admission = _admission_row(receipt)
    return admission is not None and admission[0] == 'terminal'


def _run_worker_pids(conn, task_row, task_id):
    """Every worker pid recorded for the task's current run, with its spawn fingerprint."""
    recorded = [(task_row['worker_pid'], task_row['worker_started_at'])]
    for kind in ('spawned', 'worker_bound'):
        payload = _latest_run_event(conn, task_id, kind)
        recorded.append((payload.get('pid'), payload.get('started_at')))
    return [(int(pid), fingerprint) for pid, fingerprint in recorded if pid]


def _run_heartbeat(conn, task_id):
    """The newest heartbeat of the task's CURRENT run (epoch), or None.

    ``heartbeat_worker`` stamps the run row and the task row together. The task row's stamp
    outlives its run (a crash booking does not clear it), so it speaks for the current run only
    when it was stamped after this run was claimed: a previous run's heartbeat is no evidence that
    this attempt is alive."""
    from hermes_cli import kanban_db as kb
    row = conn.execute(
        "SELECT r.last_heartbeat_at AS run_beat, r.started_at AS run_started, t.last_heartbeat_at AS task_beat "
        "FROM tasks t JOIN task_runs r ON r.id = t.current_run_id WHERE t.id=?", (task_id,)).fetchone()
    if row is None:
        return None
    beats = [kb._to_epoch(row['run_beat'])]
    task_beat, started = kb._to_epoch(row['task_beat']), kb._to_epoch(row['run_started'])
    if task_beat is not None and (started is None or task_beat >= started):
        beats.append(task_beat)  # an unreadable run clock never discards a heartbeat
    beats = [b for b in beats if b is not None]
    return max(beats) if beats else None


def owner_attempt_provably_over(conn, task_id) -> bool:
    """True when an unresolved owner receipt cannot still be executing this attempt.

    ``owner_reclaim_paused`` is about effects: an admission that may still be in flight is
    never repeated. It must not also mean the run keeps its claim forever. A paused run whose
    owner died, or whose owner settled the admission after losing its worker, may never have
    written a ``worker_bound`` event, so ``bound_interpreter_gone`` cannot speak for it. This
    asks the question that matters — can anything still be executing this attempt? — from
    evidence the board already keeps:

      * the run never bound an interpreter (a bound run is the crash sweep's to book);
      * the claim is host-local and EXPIRED (a live claim is the owner still working);
      * no heartbeat of THIS run inside ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (a previous
        run's heartbeat on the task row is not this attempt's, ``_run_heartbeat``);
      * the receipt's own process is gone or recycled, or that owner has SETTLED the admission
        (``status='terminal'``: a live owner fences a managed worker it lost before ``worker_bound``);
      * every worker pid recorded for the run (``tasks.worker_pid``, its ``spawned`` and
        ``worker_bound`` events) is dead or recycled, and there is at least one.

    A settled admission outranks the owner's liveness: the owner is usually the long-lived
    gateway, alive by design, and it settles a lost managed worker's admission without
    extending the claim. Settlement proves the owner stopped, not that the dispatcher's own
    worker or the claim did, so the claim must still have lapsed and every recorded worker be dead.

    Missing evidence is never proof: no receipt, no recorded worker pid, a live pid whose
    identity cannot be compared, or an unreadable clock all return False and the pause holds.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _worker_alive
    if _latest_run_event(conn, task_id, 'worker_bound'):
        return False  # a bound run is the crash sweep's to book (bound_interpreter_gone)
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return False  # no receipt: this pause did not come from an owner attempt
    row = conn.execute(
        "SELECT claim_lock, claim_expires, worker_pid, worker_started_at "
        "FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or not _host_local_lock(row['claim_lock']):
        return False
    pids = _run_worker_pids(conn, row, task_id)
    if not pids or any(_worker_alive(pid, fingerprint) for pid, fingerprint in pids):
        return False
    now = int(time.time())
    expires = kb._to_epoch(row['claim_expires'])
    if expires is None or expires >= now:
        return False
    heartbeat = _run_heartbeat(conn, task_id)
    if heartbeat is not None and now - heartbeat <= kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS:
        return False
    return _owner_settled_admission(receipt) or _owner_process_gone(receipt)


def owner_attempt_note(conn, task_id, lost=None) -> str:
    """One clause naming why the owner says the attempt ended, or ``""``.

    The reclaim paths put this in the run error so the board says what happened instead of
    a bare ``stale_lock=``: a managed worker that died (its owner recorded ``worker_lost``,
    passed as ``lost`` when the caller already read it) is a different fault from a worker
    that never started.
    """
    if lost is None:
        lost = _latest_run_event(conn, task_id, 'worker_lost')
    reason = str(lost.get('reason') or '')
    if reason:
        return f"managed worker lost: {reason[:80]}"
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return ""
    admission = _admission_row(receipt)
    if admission is not None and admission[0] == 'terminal':
        return f"owner admission settled {admission[1] or 'terminal'}"
    return ""


def reclaim_error(conn, task_id, claim_lock) -> str:
    """A stale-lock reclaim's run error, naming how the attempt ended when the owner recorded it."""
    note = owner_attempt_note(conn, task_id)
    return f"stale_lock={claim_lock}" + (f" [{note}]" if note else "")


def apply_attempt_note(conn, task_id, pid, dead) -> None:
    """Fold the owner's own record of how the attempt ended into ONE dead worker's booking.

    A bound interpreter whose ``worker_result`` says exit 0 after its managed execution was
    lost reads as a protocol violation. The owner's ``worker_lost`` evidence outranks it: the
    turn never finished, so it books as a crash and charges the breaker, and the
    protocol-violation streak (which reads the persisted payload) does not count it.
    """
    lost = _latest_run_event(conn, task_id, "worker_lost")
    note = owner_attempt_note(conn, task_id, lost)
    if not note:
        return
    dead.event_payload["attempt_note"] = note
    if lost:
        dead.event_payload["worker_lost"] = lost
        if dead.protocol_violation:
            from hermes_cli.kanban_db_dispatch import _PROTOCOL_VIOLATION_ERROR
            dead.protocol_violation = False
            dead.event_payload.pop("protocol_violation", None)
            dead.event_payload["exit_kind"] = dead.kind
            dead.event_kind = "crashed"
            # Keep the worker's last output that _classify_dead_worker appended.
            dead.error_text = (f"pid {pid} exited cleanly after its managed worker was lost."
                               + dead.error_text.removeprefix(_PROTOCOL_VIOLATION_ERROR))
    # Before the worker's own output: the stored error is capped, and the output may be long.
    from hermes_cli.kanban_db_dispatch import WORKER_OUTPUT_SEP
    head, sep, output = dead.error_text.partition(WORKER_OUTPUT_SEP)
    dead.error_text = f"{head} [{note}]{sep}{output}"


def _owner_restart_record(conn, task_id):
    """The owner's ``owner_restarted`` record (exit 79) that classifies the task's CURRENT run under
    its live claim, with that task row; ``(None, row)`` when there is none."""
    from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
    from hermes_cli.kanban_db_workspace import OWNER_RECORDED_RESULT_KEY
    task = conn.execute('SELECT current_run_id, claim_lock, worker_pid FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task is None or task['current_run_id'] is None or not task['claim_lock']:
        return None, task
    result = attempt_result(conn, task_id, task['current_run_id'])
    if (OWNER_RECORDED_RESULT_KEY not in result or result.get('exit_code') != KANBAN_OWNER_RESTART_EXIT_CODE
            or result.get('claim_lock') != task['claim_lock']):
        return None, task
    return result, task


def owner_restart_recorded(conn, task_id) -> bool:
    """True when the owner has committed ``owner_restarted`` as the cause of the task's current run
    under its live claim (``session_kanban.record_owner_restart_result``), bound or not.

    That cause is the attempt's accounting (:func:`attempt_result`): only the crash sweep books such
    a run, ``owner_restarted`` and uncounted, once its worker is dead. An ordinary reclaimer that
    finds it after its own termination step leaves it there
    (:func:`owner_reclaim_blocked_after_kill`)."""
    return _owner_restart_record(conn, task_id)[0] is not None


def owner_restart_accounted(conn, task_id) -> bool:
    """True when the owner accounted for the task's current run as lost to its own end, and that
    run never bound an interpreter.

    The unbound twin of :func:`bound_interpreter_gone`: an owner that ends between admitting an
    attempt (``owner_admitted``) and its interpreter's ``worker_bound`` settles it ``interrupted``
    and records ``owner_restarted`` under the run's own claim
    (``session_kanban.record_owner_restart_result``). Nothing can still execute it once that owner
    is gone, or has settled the admission ``terminal``; the crash sweep then books it, uncounted,
    as soon as the run's worker is dead, as it books a bound run.

    The record names the run's worker (``tasks.worker_pid``) when the dispatcher had registered it.
    The dispatcher spawns before it registers (``_set_worker_pid``), so the owner can record first,
    with no pid; that record is still this run's under this claim, and it speaks for the worker the
    run registers later (:func:`managed_run_exit_code`). Until one is registered, no sweep can prove
    the worker dead, so the run is held, uncounted (see ``owner_reclaim_blocked``)."""
    if _latest_run_event(conn, task_id, 'worker_bound'):
        return False  # a bound run is bound_interpreter_gone's
    result, task = _owner_restart_record(conn, task_id)
    if result is None or result.get('pid') not in (None, task['worker_pid']):
        return False
    receipt = _latest_run_event(conn, task_id, 'owner_admitted')
    return bool(receipt) and (_owner_process_gone(receipt) or _owner_settled_admission(receipt))


def owner_reclaim_blocked(conn, task_id, *, crash_sweep=False) -> bool:
    """True when a reclaim path must leave this task alone.

    The pause protects the effects of an attempt that may still be in flight, and stops
    holding the claim as soon as nothing can still be executing it. A run whose bound
    interpreter is dead stays the crash sweep's alone (``crash_sweep=True``), so it is booked
    ``crashed`` and re-queued once rather than ``reclaimed`` by whichever path ticks first; so
    does an unbound run its ended owner accounted for (``owner_restart_accounted``), which the
    crash sweep books ``owner_restarted`` and a stale-claim reclaim would count as a failure. An
    accounted run whose worker pid never gets registered is never swept (the crash sweep needs a
    dead registered worker) and is held here, uncounted, like any admitted run without a recorded
    worker pid (``owner_attempt_provably_over``); an operator ``reclaim`` releases it. A
    run that never bound one is released by any path once ``owner_attempt_provably_over``:
    the shape that used to be unreclaimable forever, because ``bound_interpreter_gone`` was the
    only override and it needs the ``worker_bound`` event that run never wrote.
    """
    if not owner_reclaim_paused(conn, task_id):
        return False
    accounted = owner_restart_accounted(conn, task_id)
    if crash_sweep and (accounted or bound_interpreter_gone(conn, task_id)):
        return False
    return accounted or not owner_attempt_provably_over(conn, task_id)


def owner_reclaim_blocked_after_kill(conn, task_id, *, signalled: bool) -> bool:
    """The under-lock re-check for a reclaim path, after its termination step.

    When this host just signalled the run's worker (``signalled``), killing it takes its bound
    interpreter down too (``_watch_submitter``) and the owner settles the admission ``unknown``:
    a pause this path caused itself, which would leave a timed-out run for the crash sweep to
    misbook ``crashed``. Only an interpreter still LIVE (one that bound after the first look)
    keeps the claim. Without a local signal (a foreign-host claim) the bound interpreter's pid
    means nothing here, so a "dead" reading is no proof and the plain gate applies.

    That exception is for a pause the kill caused, never for a cause the owner committed: a run
    whose owner recorded ``owner_restarted`` (:func:`owner_restart_recorded`), possibly while this
    path was terminating its worker, stays the crash sweep's to book uncounted; this path must not
    book it ``timed_out``/``reclaimed`` and count it.
    """
    if owner_restart_recorded(conn, task_id):
        return True
    return owner_reclaim_blocked(conn, task_id, crash_sweep=signalled)


def attempt_result(conn, task_id, run_id):
    """The ``worker_result`` that classifies run ``run_id``'s attempt for accounting, ``{}`` when none.

    A run can carry two results. A replacement owner that settles the attempt's lost turn records
    ``owner_restarted`` (exit 79, ``recorded_by_owner_pid``) only when no result exists yet
    (``session_kanban.record_owner_restart_result``). The bound interpreter, which may outlive its
    owner, records its own result when its turn ends. That later result stays the interpreter's own
    fact (and ends its workspace binding, ``kanban_db_workspace``), but the owner already accounted
    for the attempt: its record decides, so a restart stays uncounted whichever arrives first at a
    reader. Without an owner record, the latest result decides, as before."""
    from hermes_cli.kanban_db_workspace import OWNER_RECORDED_RESULT_KEY
    results = []
    for (payload,) in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result' ORDER BY id",
            (task_id, run_id)).fetchall():
        try:
            result = json.loads(payload) if payload else {}
        except ValueError:
            result = {}
        results.append(result if isinstance(result, dict) else {})
    owner = [r for r in results if OWNER_RECORDED_RESULT_KEY in r]
    return owner[0] if owner else (results[-1] if results else {})


def managed_run_exit_code(conn, task_id, worker_pid, claim_lock):
    """The exit code a managed run's authority recorded for the task's CURRENT run, else None.

    Managed interpreters are children of the authority, not the dispatcher, so their result is
    read from the run's ``worker_result`` event (it survives a dispatcher restart and cannot be
    reaped here). It speaks for the run only under the same claim and from a pid of this run:
    the dispatcher's own worker, or the owner-side interpreter ``worker_bound`` recorded under
    that claim — ``session_kanban._record_worker_result`` writes the interpreter's pid, never the
    spawned submitter's ``tasks.worker_pid``. The run's result is its accounting result
    (:func:`attempt_result`). The owner's record for a run that had not bound, made before the
    dispatcher registered the run's worker, carries no pid (``owner_restart_accounted``); the only
    worker that run can register later is its own (``_set_worker_pid`` and ``adopt_worker_pid``
    register only on the run's own row), so under the same claim that record speaks for ``worker_pid``.
    It still does once an interpreter binds the run after the record (it had adopted before the
    owner ended and bound after): the owner's record was first, so it is the run's accounting
    whatever the interpreter binds or records later, and the crash sweep has already required that
    interpreter dead when the owner's pause holds the run (``bound_interpreter_gone``)."""
    from hermes_cli.kanban_db_workspace import OWNER_RECORDED_RESULT_KEY
    def latest(kind):
        row = conn.execute(
            "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
            "WHERE t.id=? AND e.task_id=t.id AND e.kind=? ORDER BY e.id DESC LIMIT 1", (task_id, kind)).fetchone()
        try:
            payload = json.loads(row['payload']) if row else {}
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    run = conn.execute('SELECT current_run_id FROM tasks WHERE id=?', (task_id,)).fetchone()
    result = attempt_result(conn, task_id, run[0]) if run and run[0] is not None else {}
    bound = latest('worker_bound')
    pids = {worker_pid, *([bound.get('pid')] if bound.get('claim_lock') == claim_lock else [])}
    if OWNER_RECORDED_RESULT_KEY in result and worker_pid is not None:
        pids.add(None)  # recorded before this run's worker was registered, or before it bound
    return result.get('exit_code') if result.get('claim_lock') == claim_lock and result.get('pid') in pids else None


def presubmit_restart_fence(conn, row, exit_code, dead):
    """``dead`` as classified, unless it books ``owner_restarted`` from the worker's own 79 (reaped or
    from its log: ``exit_code``, the run's recorded result, is None) for a run its owner never
    admitted and that carries no pre-submission grace under its claim (``kanban_presubmit_grace``).
    Such a 79 speaks for nothing (a stale trailer, a forged exit) and is an ordinary counted exit.
    Admitted runs keep the owner's own accounting, unchanged. A grace-backed run keeps its booking and
    counters too, but not the classifier's "its turn was settled interrupted" text: no turn was sent,
    so its error says what actually happened (COMPOSE-ACCOUNT-N1)."""
    if not dead.owner_restarted or exit_code is not None:
        return dead
    from hermes_cli.kanban_presubmit_grace import run_backing
    backing = run_backing(conn, row['id'], row['claim_lock'])
    if backing == 'admitted':
        return dead
    pid = dead.event_payload.get('pid')
    if backing == 'grace':
        from dataclasses import replace
        text = (f"pid {pid}: the attempt's profile owner was replaced before the attempt submitted its turn; "
                "nothing was sent, so no turn was settled or replayed (the card's one pre-submission owner-restart "
                "grace) — requeued without counting a failure")
        output = dead.event_payload.get('worker_output')
        if output:
            text += f" Worker's last output: {output!r}"
        return replace(dead, error_text=text)
    from hermes_cli.kanban_db_retry_policy import DeadWorker
    return DeadWorker('nonzero_exit', dead.code, f'pid {pid} exited with code {dead.code}', 'crashed',
                      {'pid': pid, 'claimer': row['claim_lock'], 'exit_kind': 'nonzero_exit', 'exit_code': dead.code})
