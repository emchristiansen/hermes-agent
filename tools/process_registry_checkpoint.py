"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[list[dict[str, Any]]] = None, settled=()):
        """Write running process metadata to the checkpoint file atomically. ``settled``: on-disk
        entries this writer resolved without tracking them (a custody scope a recovery reaped).

        Every interpreter of the profile (owner, each managed worker) rewrites this file from its own
        table, so the snapshot is taken under the file lock (one taken before it could land after a
        newer writer's and undo its transition), and a finish is never undone: an on-disk custody
        entry outranks a writer's running copy of the same process."""
        from hermes_cli.active_sessions import _FileLock
        from tools.environments.sandbox_scope import hosted_scope
        from tools.process_registry import _checkpoint_path
        from utils import atomic_json_write

        # A hosted caller only changes its own agent's records, which are never persisted here, so
        # it has no host change to write: it neither takes the lock nor rewrites this host's file.
        if hosted_scope() is not None:
            return
        try:
            path = _checkpoint_path()
            with _FileLock(path.with_name(path.name + ".lock")):
                try:
                    on_disk = [e for e in json.loads(path.read_text(encoding="utf-8-sig")) if isinstance(e, dict)]
                except (OSError, ValueError):
                    on_disk = []
                entries, known = self._checkpoint_snapshot(extra_entries, settled)
                custody = {e.get("session_id"): e for e in on_disk if e.get("scope_custody") and e.get("systemd_unit")}
                entries = [entry if entry.get("scope_custody") else custody.get(entry.get("session_id"), entry)
                           for entry in entries]
                atomic_json_write(path, entries + self._foreign_handoffs(on_disk, known))
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def _checkpoint_snapshot(self, extra_entries, settled):
        """This registry's entries (running, or finished with its scope in custody) and the IDs it decides."""
        from agent.runtime_session_store import is_worker_process
        from tools.process_registry import _CHECKPOINT_FIELDS
        from tools.process_registry_awaiting import completion_settled
        # Every interpreter on the home writes this file; ``hermes update --force`` sweeps only what
        # the gateway it killed owns (update_cmd_force_stop): a managed worker's jobs are its owner's.
        owner_pid = os.getppid() if is_worker_process() else os.getpid()  # windows-footgun: ok — getppid exists on Windows
        with self._lock:
            entries = []
            # A finished session stays while its scope is in custody (process_registry_scope_custody).
            for s in (*self._running.values(), *self._finished.values()):
                # A hosted agent's process is not this host's to persist: its record (command,
                # cwd, owner) stays in memory, whoever triggers the write (scoped or not).
                if s.sandbox_key:
                    continue
                # A settling exit (its outcome still being recorded) stays a running entry, scope and all.
                outcome_settled = completion_settled(s)  # the session lock, taken inside the registry lock
                custody = bool(outcome_settled and s._scope_stop_pending and s.systemd_unit)
                if outcome_settled and not custody:
                    continue
                # Backfill the start time so recovery can detect PID recycling
                # even for sessions spawned before this field existed.
                if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                    s.host_start_time = self._safe_host_start_time(s.pid)
                entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                # Redact inline credentials before persisting (~/.hermes/processes.json).
                # Recovery uses command only for display (adoption re-validates the
                # PID, never re-runs it), so masking is lossless.
                # See #77484.
                entry["command"] = redact_sensitive_text(s.command, code_file=True)
                entry["owner_task_id"] = s.owner_task_id or s.task_id
                entry["owner_pid"] = owner_pid
                if custody:
                    entry.update(scope_custody=True, exit_code=s.exit_code,
                                 completion_reason=s.completion_reason, termination_source=s.termination_source)
                entries.append(entry)
            if extra_entries:
                tracked_ids = {item.get("session_id") for item in entries}
                entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
            return entries, {*self._running, *self._finished, *settled, *(item.get("session_id") for item in entries)}

    def _foreign_handoffs(self, on_disk, known) -> list[dict[str, Any]]:
        """On-disk worker handoffs this registry never tracked whose log still decides an outcome:
        the process is alive (same start time) or its recorded exit is waiting to be read (a worker's
        output-log handoff is its only path to the next interpreter); and scopes another interpreter
        holds in custody (only a recovery may release those)."""
        from pathlib import Path
        from tools.process_registry_output_log import _exit_path, adoptable_log
        return [entry for entry in on_disk if entry.get("session_id") not in known
                and ((entry.get("scope_custody") and entry.get("systemd_unit"))
                     or ((log := adoptable_log(entry.get("output_log"))) and Path(log).exists()
                         and (_exit_path(log).exists() or self._detached_host_fate(
                             entry.get("pid"), entry.get("host_start_time")) == "running")))]

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions."""
        from tools.process_registry import _checkpoint_path
        from tools.process_registry_scope_custody import completion_fence

        # This host adopts what a crashed predecessor left running, so the jobs it starts must
        # survive its own crash for the next one: they stay out of the owner-death supervisor.
        self._jobs_outlive_host = True
        checkpoint_path = _checkpoint_path()
        if not checkpoint_path.exists():
            return 0
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return 0
        recovered = 0
        unresolved_scope_entries: list[dict[str, Any]] = []
        released_scopes: set[str] = set()
        for entry in entries:
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            if not pid:
                continue
            # The registry is process-global, so every profile's checkpoint carries every live
            # process; a multiplexer recovering several homes must adopt each session once.
            with self._lock:
                already_tracked = entry.get("session_id") in self._running or (
                    entry.get("session_id") in self._finished and entry.get("scope_custody"))
            if already_tracked:
                continue
            if pid_scope != "host":  # in-sandbox PIDs mean nothing once the env handle is gone
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60], pid, pid_scope)
                continue
            # A receipt exists only for a finished command: it outranks an older running snapshot.
            if (decided := self._entry_with_receipt(entry)) is None:
                released_scopes.add(entry["session_id"])  # finished, scope released: nothing to do
                continue
            # Alive and still ours: re-attach. A start-time probe that cannot
            # be read is not proof the PID was reused — dropping it would leave
            # a live child unsupervised, and marking it exited would invent a
            # completion. A positive mismatch means the number was recycled:
            # do not adopt it and do not signal it.
            # A custody entry's command already finished (its receipt is durable): only its scope
            # matters, and its root PID, possibly recycled since, is never consulted or signalled.
            probed = None if decided.get("scope_custody") else self._detached_host_fate(pid, entry.get("host_start_time"))
            # The command's owner (another interpreter) may have made its finish durable since that
            # look: decide, and act, on what is durable now, with no finish landing in between.
            with completion_fence() as fenced:
                if (decided := self._durable_entry(decided)) is None:
                    released_scopes.add(entry["session_id"])
                    continue
                outcome = self._recover_entry(decided, probed, fenced)
            if outcome == "adopted":
                recovered += 1
            elif outcome == "unresolved":
                unresolved_scope_entries.append(decided)
            elif outcome == "released":
                released_scopes.add(entry["session_id"])
        self._write_checkpoint(extra_entries=unresolved_scope_entries, settled=released_scopes)
        return recovered

    def _recover_entry(self, entry: dict[str, Any], probed: Optional[str], fenced: bool) -> str:
        """Act on one checkpoint entry as decided under ``completion_fence``; ``probed`` is the root PID's
        fate (None: not probed). Returns adopted, held, released, unresolved or skipped."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log
        from tools.process_registry_results import forget_result_scope
        pid, custody = entry.get("pid"), bool(entry.get("scope_custody"))
        fate = "gone" if custody else probed
        # A worker spawn's output log and recorded exit outlive both the spawner and the PID,
        # so its reader decides the outcome (a recycled PID is never signalled: fate != running).
        log = adoptable_log(entry.get("output_log"))
        if log and not custody:
            fate = "running"
        if fate == "reused":
            logger.info(
                "Not recovering session %s: pid %d is alive but its "
                "start time no longer matches — PID was recycled onto "
                "an unrelated process; refusing to adopt or signal it.",
                entry.get("session_id", "?"), pid)
            return "skipped"
        if fate != "running":
            # Descendants of a finished command still run in its scope: keep custody, no stop.
            if custody and self._readopt_scope_custody(entry):
                return "held"
            systemd_unit = entry.get("systemd_unit", "")
            if systemd_unit and not (fenced and _stop_systemd_unit(systemd_unit)):
                logger.warning(
                    "Could not reap persisted scope %s for dead wrapper pid %s; "
                    "retaining checkpoint entry for the next startup",
                    systemd_unit, pid)
                return "unresolved"
            if custody:  # verified empty and reaped: the receipt no longer names the scope
                forget_result_scope(entry["session_id"])
                return "released"
            return "skipped"
        fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
        fields.update(
            command=entry.get("command", "unknown"),
            owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
            started_at=entry.get("started_at", time.time()), sandbox_key=entry.get("sandbox_key", ""))
        # detached: no Popen handle, but can report status + kill; a worker spawn's output log
        # (and recorded exit) outlives its spawner, so tail it from the start.
        fields["output_log"] = log
        session = ProcessSession(id=entry["session_id"], detached=True, **fields)
        if session.output_log:
            self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
        else:
            with self._lock:
                self._running[session.id] = session
        logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
        # Re-enqueue watcher so gateway can resume notifications
        if session.watcher_interval > 0:
            self.pending_watchers.append({
                "session_id": session.id,
                "check_interval": session.watcher_interval,
                "session_key": session.session_key,
                **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                "notify_on_complete": session.notify_on_complete,
                "parent_session_id": session.parent_session_id,
            })
        return "adopted"
