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

ACTOR = r'''
import json, sys, time
from pathlib import Path
from unittest.mock import patch

role, schedule, logged, sync = sys.argv[1], sys.argv[2], sys.argv[3] == "log", Path(sys.argv[4])
SID, UNIT = sys.argv[5], sys.argv[6]
import tools.process_registry_scope_custody as custody
from tools.process_registry import ProcessRegistry, ProcessSession, _checkpoint_path
from tools.process_registry_output_log import output_dir
from tools.process_registry_results import load_result_record


def mark(name):
    (sync / name).write_text(repr(time.time()))


def wait(name):
    deadline = time.monotonic() + 30
    while not (sync / name).exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{role}: timed out waiting for {name}")
        time.sleep(0.01)


def durable():
    """What another interpreter can see now: custody markers on disk and the receipt's unit."""
    try:
        checkpoint = json.loads(_checkpoint_path().read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        checkpoint = []
    receipt = load_result_record(SID)
    return {"custody": [bool(e.get("scope_custody")) for e in checkpoint if e.get("session_id") == SID],
            "receipt_unit": None if receipt is None else receipt.get("systemd_unit")}


stops = []


def stop(unit):
    at_entry = durable()
    if schedule == "stop-race":  # hold the stop until the owner is finishing, then give it time to land
        mark("stopping")
        wait("finishing")
        time.sleep(1.0)
    stops.append({"unit": unit, "at_entry": at_entry, "at_effect": durable()})
    mark("stopped")
    return True


def populated(unit):  # the descendant runs in the scope until a stop empties it
    return not (sync / "stopped").exists()


with patch.object(custody, "scope_populated", side_effect=populated), \
        patch("tools.process_registry._stop_systemd_unit", side_effect=stop):
    registry = ProcessRegistry()
    if role == "owner":
        session = ProcessSession(id=SID, command="daemonize", task_id="owner", owner_task_id="owner",
                                 pid=999_999_999, pid_scope="host", host_start_time=123, started_at=time.time(),
                                 systemd_unit=UNIT, output_buffer="completed\n", notify_on_complete=True)
        if logged:  # a worker spawn: its log and recorded exit outlive it
            output_dir().mkdir(parents=True, exist_ok=True)
            (output_dir() / f"{SID}.log").write_text("completed\n", encoding="utf-8")
            (output_dir() / f"{SID}.exit").write_text("0\n", encoding="utf-8")
            session.output_log = str(output_dir() / f"{SID}.log")
        registry._running[SID] = session
        registry._write_checkpoint()
        mark("checkpointed")
        wait({"before": "checkpointed", "during": "probed", "stop-race": "stopping"}[schedule])
        mark("finishing")
        registry._finish_exited(session, 0)
        mark("finished")
        print(json.dumps({"stops": stops, "durable": durable(), "receipt": load_result_record(SID)}))
        raise SystemExit(0)

    probes = []

    def fate(pid, start):
        probes.append(pid)
        if len(probes) == 1:
            mark("probed")
            if schedule == "during":  # the owner's completion lands at the PID-probe seam
                wait("finished")
        return "gone"

    with patch.object(registry, "_detached_host_fate", side_effect=fate):
        wait("finished" if schedule == "before" else "checkpointed")
        recovered = registry.recover_from_checkpoint()
        if logged and SID in registry._running:  # a replayed log-backed adoption finishes on its reader
            registry._running[SID]._completion_event.wait(10)
        held = registry._finished.get(SID)
        result = {"recovered": recovered, "stops_in_recovery": list(stops),
                  "completions": registry.completion_queue.qsize(), "durable_after_recovery": durable(),
                  "held": held and {"exit_code": held.exit_code, "output": held.output_buffer,
                                    "custody": held._scope_stop_pending, "unit": held.systemd_unit}}
        if schedule != "stop-race":
            wait("finished")
            result["explicit_kill"] = registry.kill_process(SID)["status"] if held else None
        result.update(stops=stops, durable=durable(), receipt=load_result_record(SID))
    print(json.dumps(result))
'''


def _race(tmp_path, schedule, logged):
    home, sync = tmp_path / "home", tmp_path / "sync"
    home.mkdir()
    sync.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("HERMES_HOME", "PYTHONPATH")}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(ROOT), HERMES_SESSION_ID="fence-session")
    actors = {role: subprocess.Popen(
        [sys.executable, "-c", ACTOR, role, schedule, "log" if logged else "pipe", str(sync), SID, UNIT],
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
    assert marks["stopped"] < marks["finished"]
    assert owner["stops"] == []
    assert owner["receipt"]["exit_code"] == 0 and owner["receipt"]["systemd_unit"] == ""
    assert owner["durable"] == {"custody": [], "receipt_unit": ""}
    assert recovery["held"] is None and recovery["completions"] == 0
