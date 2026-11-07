"""BG-SCOPE-04: recovery decides a dead root's scope on what is durable when it acts, not when it looked.

Two interpreters share a profile checkpoint: an owner whose root just exited while a descendant still
runs in its scope, and a recovery (another managed worker's admission) that read the owner's running
entry. Each is a separate OS process using the real registry, receipts, checkpoint and file locks; only
the PID probe, the scope's membership and ``systemctl stop`` are stand-ins, so nothing is signalled.
A completion that became durable before the recovery acts must be honoured (scope retained, result
kept, no second completion); one that comes after a recovery has decided to stop must wait for the
stop, then find the scope empty.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SID, UNIT = "proc_fence000004", "hermes-worker-proc_fence000004.scope"

ACTOR = r"""
import json, os, sys, time
from pathlib import Path
from unittest.mock import patch

role, schedule, logged, sync = sys.argv[1], sys.argv[2], sys.argv[3] == "log", Path(sys.argv[4])
SID, UNIT, action, fault = sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
import tools.process_registry_scope_custody as custody
from hermes_cli.active_sessions import _FileLock
from tools.process_registry import ProcessRegistry, ProcessSession, _checkpoint_path
from tools.process_registry_output_log import output_dir
from tools.process_registry_results import load_result_record


def mark(name):
    (sync / name).write_text(repr(time.time()))


def wait(name):
    deadline = time.monotonic() + 60
    while not (sync / name).exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{role}: timed out waiting for {name}")
        time.sleep(0.01)


def durable():
    # What another interpreter can see now: custody markers on disk and the receipt's unit.
    try:
        checkpoint = json.loads(_checkpoint_path().read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        checkpoint = []
    receipt = load_result_record(SID)
    return {"custody": [bool(e.get("scope_custody")) for e in checkpoint if e.get("session_id") == SID],
            "receipt_unit": None if receipt is None else receipt.get("systemd_unit")}


stops, signals, lock_failures = [], [], []


def stop(unit):
    at_entry = durable()
    if schedule == "stop-race":  # hold the stop until the owner is acting, then give it time to land
        mark("stopping")
        wait("acting")
        time.sleep(1.0)
    stops.append({"unit": unit, "at_entry": at_entry, "at_effect": durable()})
    mark("stopped")
    return True


def populated(unit):  # the descendant runs in the scope until a stop empties it
    return not (sync / "stopped").exists()


real_enter = _FileLock.__enter__


def failing_enter(lock):  # only this actor's completion fence fails (the owner's: until the stop healed it)
    if str(lock.path).endswith(".finish.lock") and not (role == "owner" and (sync / "stopped").exists()):
        lock_failures.append(time.time())
        raise PermissionError("injected: this actor cannot open the completion fence")
    return real_enter(lock)


def kill(pid, sig):  # no signal may reach the (possibly recycled) root PID
    signals.append([pid, int(sig)])


with patch.object(custody, "scope_populated", side_effect=populated), \
        patch("tools.process_registry._stop_systemd_unit", side_effect=stop), \
        patch("os.kill", side_effect=kill), patch("os.killpg", side_effect=kill), \
        patch.object(_FileLock, "__enter__", failing_enter if fault == role + "-lock" else real_enter):
    registry = ProcessRegistry()
    if role == "owner":
        refresh = action.startswith("refresh")
        session = ProcessSession(id=SID, command="daemonize", task_id="owner", owner_task_id="owner",
                                 pid=999_999_999, pid_scope="host", host_start_time=123, started_at=time.time(),
                                 systemd_unit=UNIT, output_buffer="completed\n", notify_on_complete=True,
                                 detached=refresh)  # refresh: an earlier adoption of a pipe-backed command
        if logged:  # a worker spawn: its log and recorded exit outlive it
            output_dir().mkdir(parents=True, exist_ok=True)
            (output_dir() / f"{SID}.log").write_text("completed\n", encoding="utf-8")
            (output_dir() / f"{SID}.exit").write_text("0\n", encoding="utf-8")
            session.output_log = str(output_dir() / f"{SID}.log")
        registry._running[SID] = session
        registry._write_checkpoint()
        mark("checkpointed")
        wait({"before": "checkpointed", "during": "probed", "reread": "reread", "stop-race": "stopping"}[schedule])
        mark("acting")
        with patch.object(registry, "_detached_host_fate", return_value=action.partition("-")[2] or "gone"):
            registry.get(SID) if refresh else registry._finish_exited(session, 0)
        mark("acted")
        print(json.dumps({"stops": stops, "signals": signals, "lock_failures": len(lock_failures),
                          "durable": durable(), "receipt": load_result_record(SID),
                          "completions": registry.completion_queue.qsize(),
                          "session": {"exited": session.exited, "exit_code": session.exit_code,
                                      "custody": session._scope_stop_pending}}))
        raise SystemExit(0)

    probes = []

    def fate(pid, start):
        probes.append(pid)
        if len(probes) == 1:
            mark("probed")
            if schedule == "during":  # the owner's transition lands at the PID-probe seam
                wait("acted")
        return "gone"

    real_reread = registry._durable_entry

    def reread(entry):  # recovery has re-read what is durable; the owner acts before it decides
        decided = real_reread(entry)
        if schedule == "reread":
            mark("reread")
            wait("acting")
            time.sleep(1.0)
        return decided

    with patch.object(registry, "_detached_host_fate", side_effect=fate), \
            patch.object(registry, "_durable_entry", side_effect=reread):
        wait("acted" if schedule == "before" else "checkpointed")
        recovered = registry.recover_from_checkpoint()
        if logged and SID in registry._running:  # a replayed log-backed adoption finishes on its reader
            registry._running[SID]._completion_event.wait(10)
        held = registry._finished.get(SID)
        result = {"recovered": recovered, "stops_in_recovery": list(stops),
                  "completions": registry.completion_queue.qsize(), "durable_after_recovery": durable(),
                  "held": held and {"exit_code": held.exit_code, "output": held.output_buffer,
                                    "custody": held._scope_stop_pending, "unit": held.systemd_unit}}
        wait("acted")
        result["explicit_kill"] = registry.kill_process(SID)["status"] if held else None
        result.update(stops=stops, signals=signals, durable=durable(), receipt=load_result_record(SID))
    print(json.dumps(result))
"""


def _race(tmp_path, schedule, logged, action="finish", fault="none"):
    home, sync = tmp_path / "home", tmp_path / "sync"
    home.mkdir()
    sync.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_HOME", "PYTHONPATH")}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(ROOT), HERMES_SESSION_ID="fence-session")
    actors = {role: subprocess.Popen(
        [sys.executable, "-c", ACTOR, role, schedule, "log" if logged else "pipe", str(sync), SID, UNIT, action, fault],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8") for role in ("owner", "recovery")}
    results = {}
    try:
        for role, actor in actors.items():
            out, err = actor.communicate(timeout=120)
            assert actor.returncode == 0 and out.strip(), (role, actor.returncode, err[-3000:])
            results[role] = json.loads(out.strip().splitlines()[-1])
    finally:
        for actor in actors.values():
            if actor.poll() is None:
                actor.kill()
                actor.communicate()
    marks = {path.name: float(path.read_text()) for path in sync.iterdir()}
    return results["owner"], results["recovery"], marks


@pytest.mark.parametrize(("schedule", "logged"), [("during", False), ("during", True), ("before", False)],
                         ids=["completion-at-probe-pipe", "completion-at-probe-log-backed", "completion-first"])
def test_recovery_honours_a_completion_that_became_durable_before_it_acts(tmp_path, schedule, logged):
    """The reviewer's schedule: recovery read a running entry and found no receipt; the owner then made
    its completion durable (receipt + custody checkpoint) before recovery reached the dead root. Recovery
    must hold the populated scope as the finished command's, never stop it or replay the completion; a
    later explicit stop still reaches it."""
    owner, recovery, _ = _race(tmp_path, schedule, logged)
    assert owner["receipt"]["exit_code"] == 0 and owner["receipt"]["systemd_unit"] == UNIT
    assert owner["durable"] == {"custody": [True], "receipt_unit": UNIT}
    assert recovery["stops_in_recovery"] == []  # no stop on the strength of the earlier look
    assert recovery["recovered"] == 0 and recovery["completions"] == 0  # not re-adopted as running
    assert recovery["held"] == {"exit_code": 0, "output": "completed\n", "custody": True, "unit": UNIT}
    assert recovery["durable_after_recovery"] == {"custody": [True], "receipt_unit": UNIT}
    assert recovery["explicit_kill"] == "already_exited"
    [stop] = recovery["stops"]
    assert stop["unit"] == UNIT and stop["at_effect"] == {"custody": [True], "receipt_unit": UNIT}
    assert recovery["durable"] == {"custody": [], "receipt_unit": ""}
    assert recovery["receipt"]["exit_code"] == 0 and recovery["receipt"]["output"] == "completed\n"


def test_a_completion_racing_a_decided_stop_waits_for_it_and_releases_the_scope(tmp_path):
    """The other order: recovery found no completion and is stopping the dead root's scope when the owner
    finishes. The owner's completion must not become durable custody while that stop is in flight; it
    lands after the stop, finds the scope empty and releases it."""
    owner, recovery, marks = _race(tmp_path, "stop-race", False)
    [stop] = recovery["stops"]
    no_custody = {"custody": [False], "receipt_unit": None}
    assert stop == {"unit": UNIT, "at_entry": no_custody, "at_effect": no_custody}
    assert marks["stopped"] < marks["acted"]
    assert owner["stops"] == []
    assert owner["receipt"]["exit_code"] == 0 and owner["receipt"]["systemd_unit"] == ""
    assert owner["durable"] == {"custody": [], "receipt_unit": ""}
    assert recovery["held"] is None and recovery["completions"] == 0


NO_CUSTODY = {"custody": [False], "receipt_unit": None}


@pytest.mark.parametrize("fate", ["gone", "reused"])
def test_a_detached_refresh_that_retains_the_scope_first_is_honoured(tmp_path, fate):
    """BG-SCOPE-05, custody first: an earlier adoption's ordinary ``get()`` finds the root gone (or its PID
    recycled) while the scope is populated, and retains the scope before recovery re-reads. Recovery holds
    it as a closed entry, stops nothing, invents no receipt or completion, and signals no PID; a later
    explicit stop still reaches it."""
    owner, recovery, _ = _race(tmp_path, "during", False, action=f"refresh-{fate}")
    assert owner["session"] == {"exited": True, "exit_code": None, "custody": True}
    assert owner["durable"] == {"custody": [True], "receipt_unit": None}
    assert owner["receipt"] is None and owner["completions"] == 0
    assert recovery["stops_in_recovery"] == [] and recovery["recovered"] == 0 and recovery["completions"] == 0
    assert recovery["held"] == {"exit_code": None, "output": "", "custody": True, "unit": UNIT}
    assert recovery["explicit_kill"] == "already_exited"
    assert [stop["unit"] for stop in recovery["stops"]] == [UNIT]
    assert recovery["durable"] == {"custody": [], "receipt_unit": None} and recovery["receipt"] is None
    assert owner["signals"] == [] and recovery["signals"] == []


def test_a_sequential_detached_refresh_before_recovery_is_honoured(tmp_path):
    """Control: the refresh completes before recovery even reads the checkpoint."""
    owner, recovery, _ = _race(tmp_path, "before", False, action="refresh-gone")
    assert owner["durable"] == {"custody": [True], "receipt_unit": None}
    assert recovery["stops_in_recovery"] == [] and recovery["held"]["custody"] is True


@pytest.mark.parametrize("fate", ["gone", "reused"])
def test_a_detached_refresh_cannot_retain_a_scope_inside_a_recovery_decision(tmp_path, fate):
    """BG-SCOPE-05, recovery first (the reviewer's schedule): recovery re-read a running entry under its
    fence; the refresh starts before recovery acts. It must not publish custody until recovery's stop is
    done, then find the scope empty and close the entry without custody, receipt or completion."""
    owner, recovery, marks = _race(tmp_path, "reread", False, action=f"refresh-{fate}")
    [stop] = recovery["stops"]
    assert stop == {"unit": UNIT, "at_entry": NO_CUSTODY, "at_effect": NO_CUSTODY}
    assert marks["stopped"] < marks["acted"]
    assert owner["session"] == {"exited": True, "exit_code": None, "custody": False}
    assert owner["durable"] == {"custody": [], "receipt_unit": None} and owner["completions"] == 0
    assert recovery["held"] is None and recovery["completions"] == 0
    assert owner["signals"] == [] and recovery["signals"] == []


def test_a_finish_that_cannot_take_the_fence_waits_instead_of_retaining_the_scope(tmp_path):
    """BG-SCOPE-06: recovery holds a working fence and re-read a running entry; the owner's own fence
    acquisition fails. The finish must not publish its receipt or custody meanwhile. It stays pending and
    retries, and lands once the fence is available again (after the stop): exit and output kept, one
    completion, the emptied scope released."""
    owner, recovery, marks = _race(tmp_path, "reread", False, fault="owner-lock")
    [stop] = recovery["stops"]
    assert stop == {"unit": UNIT, "at_entry": NO_CUSTODY, "at_effect": NO_CUSTODY}
    assert owner["lock_failures"] >= 1 and marks["stopped"] < marks["acted"]
    assert owner["receipt"]["exit_code"] == 0 and owner["receipt"]["output"] == "completed\n"
    assert owner["receipt"]["systemd_unit"] == "" and owner["completions"] == 1
    assert owner["durable"] == {"custody": [], "receipt_unit": ""}


def test_a_recovery_that_cannot_take_the_fence_stops_nothing(tmp_path):
    """Control: recovery's own fence fails while the owner (with a working fence) finishes and retains the
    scope between recovery's read and its decision. Recovery stops nothing and custody survives."""
    owner, recovery, _ = _race(tmp_path, "reread", False, fault="recovery-lock")
    assert recovery["stops"] == [] and recovery["held"] is None
    assert owner["durable"] == {"custody": [True], "receipt_unit": UNIT}
    assert recovery["durable"] == {"custody": [True], "receipt_unit": UNIT}
