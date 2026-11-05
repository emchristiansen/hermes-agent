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
"""
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger("tools.process_registry")

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def scope_populated(unit: str) -> Optional[bool]:
    """True while *unit*'s cgroup holds a live process, False once it is verifiably empty or gone,
    None when that cannot be established (no ``systemctl``, no user bus): keep custody then."""
    if not unit:
        return False
    from tools.process_registry import systemd_user_bus_env
    binary = shutil.which("systemctl")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, "--user", "show", "-p", "LoadState", "-p", "ControlGroup", unit], capture_output=True,
            text=True, timeout=15, stdin=subprocess.DEVNULL, env=systemd_user_bus_env(), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("systemctl --user show %s failed: %s", unit, exc)
        return None
    if out.returncode != 0:
        return None
    props = dict(line.partition("=")[::2] for line in out.stdout.splitlines())
    if props.get("LoadState") == "not-found" or not props.get("ControlGroup"):
        return False  # garbage-collected, or loaded without a cgroup: nothing runs in it
    # The kernel's view, not ActiveState: systemd notices an emptied scope asynchronously.
    # cgroup.procs never lists zombies. v2 (unified), then hybrid, then the v1 named hierarchy.
    base = next((b for b in (_CGROUP_ROOT, _CGROUP_ROOT / "unified", _CGROUP_ROOT / "systemd")
                 if (b / "cgroup.procs").exists()), None)
    if base is None:
        return None
    procs = base / props["ControlGroup"].lstrip("/") / "cgroup.procs"
    try:
        return bool(procs.read_text(encoding="utf-8-sig").split())
    except FileNotFoundError:
        return False  # the cgroup was removed after the query
    except OSError:
        return None


class ProcessScopeCustodyMixin:
    """The subclass supplies ``_lock``, ``_finished``, ``_write_checkpoint``. See ProcessRegistry."""

    @staticmethod
    def _hold_scope_if_populated(session) -> None:
        """The root exited (or its PID is gone): keep the scope in custody unless it is verifiably
        empty. Called before the session is persisted as finished, so its receipt and checkpoint
        carry the unit."""
        if session.systemd_unit:
            session._scope_stop_pending = scope_populated(session.systemd_unit) is not False

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
            from tools.process_registry_results import forget_result_scope, save_completed_result
            if stopped:
                forget_result_scope(session.id)
            else:
                save_completed_result(session)
            self._write_checkpoint()
        return stopped

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
                      owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""))
        session = ProcessSession(
            id=entry["session_id"], detached=True, exited=True, _scope_stop_pending=True,
            output_buffer=record.get("output", ""), **fields,
            **{key: record.get(key, entry.get(key, default)) for key, default in (
                ("exit_code", None), ("completion_reason", "exited"), ("termination_source", ""))})
        session._completion_event.set()
        with self._lock:
            self._finished.setdefault(session.id, session)
        logger.info("Holding scope %s of finished process %s: descendants still run in it", unit, session.id)
        return True
