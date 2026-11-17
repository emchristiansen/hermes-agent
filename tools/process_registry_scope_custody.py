"""Custody of a background process's systemd scope after its tracked root exits.

A root (the ``-lic`` shell systemd-run started in the scope) can exit while descendants it started
(a ``setsid`` daemon, a double fork) still run in the same scope. The root's exit status is the
command's receipt, not proof that the scope is empty, and ``--collect`` only removes a scope once it
is empty. So a session keeps its unit (``_scope_stop_pending``) until the scope is verified empty or
has been stopped, and that custody is durable: the checkpoint keeps a ``scope_custody`` entry and
the completed-result receipt keeps ``systemd_unit``, so the next interpreter (the session's next
managed worker, the owner at startup) can still reach the scope. Holding custody never stops
anything by itself: an explicit kill, ``kill_all`` (the gateway's shutdown sweep among them) or a
later recovery that finds the scope empty releases it.

Every interpreter of a profile may recover the profile's checkpoint (each managed-worker admission
does) while another is still finishing or closing one of its commands. So every transition that can
retain a scope (a finish: scope check, receipt, checkpoint; the close of an adopted entry whose root
is gone or recycled) and a recovery's decision about the same command (adopt, hold or stop) are each
made under ``completion_fence``: a recovery acts on the custody that is durable when it acts, and a
transition that comes after a recovery decided to stop the scope finds it stopped. A transition
that cannot take the fence makes no durable change: it stays pending and is retried.
"""
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger("tools.process_registry")

_CGROUP_ROOT = Path("/sys/fs/cgroup")
# A finish whose completion fence cannot be taken retries with this backoff (seconds).
_FENCE_RETRY_FIRST_SECONDS, _FENCE_RETRY_MAX_SECONDS = 0.05, 2.0
# systemd's SD_MESSAGE_UNIT_OUT_OF_MEMORY: "A process of <unit> unit has been killed by the OOM killer."
_UNIT_OOM_MESSAGE_ID = "fe6faa94e7774663a0da52717891d8ef"


class _ScopeUnknown(Exception):
    """systemd cannot be asked (no ``systemctl``, no user bus) or no cgroup hierarchy is mounted."""


def _scope_cgroup(unit: str) -> Optional[Path]:
    """*unit*'s kernel cgroup directory; None once the unit is garbage-collected or has no cgroup."""
    from tools.process_registry import systemd_user_bus_env
    binary = shutil.which("systemctl")
    if binary is None:
        raise _ScopeUnknown("no systemctl")
    try:
        out = subprocess.run(
            [binary, "--user", "show", "-p", "LoadState", "-p", "ControlGroup", unit], capture_output=True,
            text=True, timeout=15, stdin=subprocess.DEVNULL, env=systemd_user_bus_env(), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("systemctl --user show %s failed: %s", unit, exc)
        raise _ScopeUnknown(str(exc)) from exc
    if out.returncode != 0:
        raise _ScopeUnknown(f"systemctl exited {out.returncode}")
    props = dict(line.partition("=")[::2] for line in out.stdout.splitlines())
    if props.get("LoadState") == "not-found" or not props.get("ControlGroup"):
        return None
    # v2 (unified), then hybrid, then the v1 named hierarchy.
    base = next((b for b in (_CGROUP_ROOT, _CGROUP_ROOT / "unified", _CGROUP_ROOT / "systemd")
                 if (b / "cgroup.procs").exists()), None)
    if base is None:
        raise _ScopeUnknown("no cgroup hierarchy")
    return base / props["ControlGroup"].lstrip("/")


def scope_populated(unit: str) -> Optional[bool]:
    """True while *unit*'s cgroup holds a live process, False once it is verifiably empty or gone,
    None when that cannot be established (no ``systemctl``, no user bus): keep custody then."""
    if not unit:
        return False
    try:
        cgroup = _scope_cgroup(unit)
    except _ScopeUnknown:
        return None
    if cgroup is None:
        return False  # garbage-collected, or loaded without a cgroup: nothing runs in it
    # The kernel's view, not ActiveState: systemd notices an emptied scope asynchronously.
    # cgroup.procs never lists zombies.
    try:
        return bool((cgroup / "cgroup.procs").read_text(encoding="utf-8-sig").split())
    except FileNotFoundError:
        return False  # the cgroup was removed after the query
    except OSError:
        return None


def scope_oom_killed(unit: str) -> Optional[bool]:
    """Whether the kernel OOM killer killed a process in *unit*'s scope; None when not established.

    Neither the root's exit status nor ``termination_source`` (Hermes's own kills) shows a cgroup OOM:
    under the default ``OOMPolicy=stop`` systemd then SIGTERMs the rest of the scope, so the root
    reports an ordinary 143/-15. While the scope's cgroup exists its ``memory.events`` counter decides;
    once ``--collect`` removed it, systemd's journal record of the kill, logged before that stop, does.
    A journal without the record is no proof (it may be volatile or rate-limited): None, never False."""
    try:
        cgroup = _scope_cgroup(unit)
    except _ScopeUnknown:
        cgroup = None
    if cgroup is not None:
        try:
            events = (cgroup / "memory.events").read_text(encoding="utf-8-sig").split()
            return int(dict(zip(events[::2], events[1::2]))["oom_kill"]) > 0
        except FileNotFoundError:
            pass  # removed after the query: the journal still has the kill
        except (OSError, ValueError, KeyError):
            return None
    return _journal_records_oom_kill(unit)


def _journal_records_oom_kill(unit: str) -> Optional[bool]:
    binary = shutil.which("journalctl")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, "--user", "--no-pager", "--quiet", "-o", "cat", "--output-fields=MESSAGE_ID", "-u", unit,
             f"MESSAGE_ID={_UNIT_OOM_MESSAGE_ID}"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("journalctl --user -u %s failed: %s", unit, exc)
        return None
    return True if out.returncode == 0 and out.stdout.strip() else None


@contextmanager
def completion_fence():
    """The profile's lock ordering a command's durable finish against a recovery's decision about it,
    across interpreters (``flock``, so across threads too). Never taken while holding the checkpoint
    lock or a registry lock. Yields False when it could not be taken: the caller then makes no
    custody transition and stops nothing (each caller defines its pending/retry path)."""
    from hermes_cli.active_sessions import _FileLock
    from tools.process_registry import _checkpoint_path
    path = _checkpoint_path()
    lock = _FileLock(path.with_name(path.name + ".finish.lock"))
    try:
        lock.__enter__()
    except (OSError, RuntimeError) as exc:  # cannot open the lock file / flock refused
        logger.debug("Process completion fence %s unavailable: %s", lock.path, exc)
        yield False
        return
    try:
        yield True
    finally:
        lock.__exit__(None, None, None)


class ProcessScopeCustodyMixin:
    """The subclass supplies ``_lock``, ``_finished``, ``_write_checkpoint``, ``_move_to_finished``,
    ``_release_finished_handles``, ``_prune_uncollected_detached`` and ``_close_reused_detached``.
    See ProcessRegistry."""

    @staticmethod
    def _sweepable(session) -> bool:
        """Still something to stop: the root runs, or its scope is in custody. ``kill_all`` and the turn
        baseline (``snapshot_running_ids``) share it, so a turn's timeout spares an older retained scope."""
        return not session.exited or session._scope_stop_pending

    @staticmethod
    def _observe_scope_oom(session, exit_code) -> Optional[bool]:
        """Before a scoped root's nonzero exit is recorded: did the kernel OOM-kill a process in its
        scope? Read-only and slow (systemctl, journal), so taken outside every lock; the finisher hands
        it to ``mark_exited``, which records it with the exit or, after a raced kill, not at all. An
        exit 0 is not checked."""
        return scope_oom_killed(session.systemd_unit) if session.systemd_unit and exit_code != 0 else None

    @staticmethod
    def _hold_scope_if_populated(session) -> None:
        """The root exited (or its PID is gone): keep the scope in custody unless it is verifiably
        empty. Called before the session is persisted as finished, so its receipt and checkpoint
        carry the unit."""
        if session.systemd_unit:
            session._scope_stop_pending = scope_populated(session.systemd_unit) is not False

    def _finish_with_scope(self, session, mark_exited, *, block: bool = True) -> bool:
        """A reader-observed exit: hold the scope if it is populated, record the exit (``mark_exited``,
        under the session lock) and persist the outcome, as one step against a recovery's decision.
        While the fence cannot be taken nothing is recorded: the session stays running with its output
        and the finish is retried, here with backoff (``block``, a reader thread) until it lands or
        another path (a kill) finalised the session, else by the caller's next poll. True: finished.
        Abandoning it to that other path still releases the handles this caller owns (a PTY the kill
        left to its live reader), as the duplicate finish would: never through an unfenced publish."""
        delay = _FENCE_RETRY_FIRST_SECONDS
        while True:
            with completion_fence() as fenced:
                if fenced:
                    self._hold_scope_if_populated(session)
                    with session._lock:
                        mark_exited()
                        deferred = session._kills_in_flight > 0  # a kill that signalled it publishes this exit
                    if not deferred:
                        self._move_to_finished(session)
                    return True
            if delay == _FENCE_RETRY_FIRST_SECONDS:
                logger.log(logging.WARNING if block else logging.DEBUG, "Completion of %s is pending: the "
                           "process completion fence is unavailable; retrying", session.id)
            if session._finish_claimed:  # the kill owns the result; only the local handles are ours
                self._release_finished_handles(session)
                return False
            if not block:
                return False
            time.sleep(delay)
            delay = min(delay * 2, _FENCE_RETRY_MAX_SECONDS)

    def _close_detached(self, session, fate: str):
        """An adopted (Popen-less) entry whose root is ``gone`` or whose PID was ``reused``: hold the scope
        if it is populated and close the entry with no collected exit (no receipt, no completion), under
        ``completion_fence`` like a finish. Fence unavailable: unchanged, a later refresh retries."""
        with completion_fence() as fenced:
            if not fenced:
                return session
            self._hold_scope_if_populated(session)
            if fate == "gone":
                return self._prune_uncollected_detached(session)
            self._close_reused_detached(session)
            return session

    def _stop_scope(self, session) -> bool:
        """Stop the session's systemd scope: True when it has none or the stop worked. Every kill
        path stops the scope through here, so a failed stop always leaves it in custody for a retry,
        and the custody of a finished session is persisted (or released) with the outcome."""
        if not session.systemd_unit:
            return True
        from tools.process_registry import _stop_systemd_unit
        held = session._scope_stop_pending
        stopped = _stop_systemd_unit(session.systemd_unit)
        session._scope_stop_pending = not stopped
        if session.exited and held != session._scope_stop_pending:
            from tools.process_registry_awaiting import completion_settled
            from tools.process_registry_results import forget_result_scope, save_completed_result
            if not completion_settled(session):
                pass  # no receipt yet: the owner's publication persists the outcome with this custody
            elif stopped:
                forget_result_scope(session.id)
            else:
                save_completed_result(session)
            self._write_checkpoint()
        return stopped

    @staticmethod
    def _entry_with_receipt(entry: dict) -> Optional[dict]:
        """A recovered checkpoint entry as its durable receipt decides. Only a finished command has a
        receipt, so a running entry with one is an older writer's snapshot: it becomes a custody entry
        while the receipt still names the scope, and None (nothing to adopt or stop) once it names none.
        No receipt, or one written before scope custody existed: the entry stands."""
        if entry.get("scope_custody"):
            return entry
        from tools.process_registry_results import load_result_record
        record = load_result_record(str(entry.get("session_id", "")))
        if record is None or "systemd_unit" not in record:
            return entry
        if not record["systemd_unit"]:
            return None
        return {**entry, "scope_custody": True, "systemd_unit": record["systemd_unit"],
                **{key: record.get(key) for key in ("exit_code", "completion_reason", "termination_source")}}

    def _durable_entry(self, entry: dict) -> Optional[dict]:
        """``entry`` as what is durable now decides it (call under ``completion_fence``): the on-disk
        custody entry of the same process if its finish was persisted meanwhile, else its receipt."""
        from tools.process_registry import _checkpoint_path
        try:
            on_disk = json.loads(_checkpoint_path().read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            on_disk = []
        held = next((e for e in on_disk if isinstance(e, dict) and e.get("session_id") == entry.get("session_id")
                     and e.get("scope_custody") and e.get("systemd_unit")), None)
        return self._entry_with_receipt(held or entry)

    def _readopt_scope_custody(self, entry: dict) -> bool:
        """Recovery of a ``scope_custody`` checkpoint entry: its command already finished and its
        receipt is durable. While the scope still holds processes (or that is unknown), track it as
        a finished session in custody (reachable by kill, ``kill_all`` and the next checkpoint) and
        return True; never signal it here. False: the scope is verifiably empty, so the caller reaps
        it on the dead-PID path."""
        unit = entry.get("systemd_unit", "")
        if not unit or scope_populated(unit) is False:
            return False
        from tools.process_registry import ProcessSession, _CHECKPOINT_DEFAULTS, _CHECKPOINT_FIELDS
        from tools.process_registry_output_log import adoptable_log
        from tools.process_registry_results import load_result_record
        record = load_result_record(entry["session_id"]) or {}
        fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
        fields.update(command=entry.get("command", "unknown"), output_log=adoptable_log(entry.get("output_log")),
                      owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                      scope_unit=record.get("scope_unit", fields["scope_unit"]))
        # The receipt decides the finished fields: a later re-save (a failed stop) must not erase them.
        session = ProcessSession(
            id=entry["session_id"], detached=True, exited=True, _scope_stop_pending=True,
            output_buffer=record.get("output", ""), **fields,
            **{key: record.get(key, entry.get(key, default)) for key, default in (
                ("exit_code", None), ("completion_reason", "exited"), ("termination_source", ""),
                ("scope_oom_killed", None))})
        session._completion_event.set()
        with self._lock:
            self._finished.setdefault(session.id, session)
        logger.info("Holding scope %s of finished process %s: descendants still run in it", unit, session.id)
        return True
