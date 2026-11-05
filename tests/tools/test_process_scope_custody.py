"""A finished root's scope stays in durable custody until it is verified empty or stopped.

The root's exit is the command's receipt, not proof that its systemd scope is empty: a ``setsid``
descendant can outlive it there. The unit stays on the session, in the checkpoint and on the receipt
until the scope is empty or stopped, so kill / kill_all in this or a later interpreter still reach
it; holding custody never stops anything. The real-systemd path is in test_worker_spawn_unit_scope.
"""
import json
import time

import pytest

import tools.process_registry_scope_custody as custody
from hermes_constants import get_hermes_home
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import load_result_record

SID = "proc_custody0001"
UNIT = f"hermes-worker-{SID}.scope"


@pytest.fixture()
def stops(monkeypatch):
    calls, results = [], []
    monkeypatch.setattr("tools.process_registry._stop_systemd_unit",
                        lambda unit: calls.append(unit) or (results.pop(0) if results else True))
    return calls, results


def _populated(monkeypatch, value):
    monkeypatch.setattr(custody, "scope_populated", lambda unit: value)


def _checkpoint():
    return json.loads((get_hermes_home() / "processes.json").read_text(encoding="utf-8-sig"))


def _finished_root(registry, monkeypatch, populated):
    _populated(monkeypatch, populated)
    session = ProcessSession(id=SID, command="daemonize", task_id="t1", pid=999_999_999, pid_scope="host",
                             started_at=time.time(), systemd_unit=UNIT, output_buffer="done\n")
    registry._running[SID] = session
    registry._finish_exited(session, 0)
    return session


def test_a_populated_scope_stays_in_durable_custody_until_kill_all_stops_it(monkeypatch, stops):
    registry = ProcessRegistry()
    session = _finished_root(registry, monkeypatch, True)
    assert session.exited and session.exit_code == 0 and session._scope_stop_pending
    [entry] = _checkpoint()
    assert (entry["session_id"], entry["scope_custody"], entry["systemd_unit"], entry["exit_code"]) == (
        SID, True, UNIT, 0)
    assert load_result_record(SID)["systemd_unit"] == UNIT
    assert stops[0] == []  # holding custody stops nothing

    assert registry.kill_all(source="gateway_shutdown") == 1
    assert stops[0] == [UNIT]
    assert _checkpoint() == [] and load_result_record(SID)["systemd_unit"] == ""
    assert load_result_record(SID)["exit_code"] == 0  # the receipt itself is untouched


def test_a_verified_empty_scope_is_released_at_finish(monkeypatch, stops):
    registry = ProcessRegistry()
    session = _finished_root(registry, monkeypatch, False)
    assert not session._scope_stop_pending
    assert _checkpoint() == [] and load_result_record(SID)["systemd_unit"] == ""
    assert registry.kill_all(source="gateway_shutdown") == 0 and stops[0] == []


def test_an_unknown_scope_state_keeps_custody(monkeypatch, stops):
    session = _finished_root(ProcessRegistry(), monkeypatch, None)
    assert session._scope_stop_pending and _checkpoint()[0]["scope_custody"] is True


def test_a_failed_stop_keeps_custody_for_the_next_interpreter(monkeypatch, stops):
    registry = ProcessRegistry()
    _finished_root(registry, monkeypatch, True)
    stops[1].append(False)
    result = registry.kill_process(SID)
    assert result["status"] == "already_exited" and result["scope_stop_failed"] == UNIT
    assert _checkpoint()[0]["scope_custody"] is True and load_result_record(SID)["systemd_unit"] == UNIT

    successor = ProcessRegistry()  # a later interpreter of the same profile
    assert successor.recover_from_checkpoint() == 0
    restored = successor._finished[SID]
    assert restored.exited and restored.exit_code == 0 and restored.output_buffer == "done\n"
    assert stops[0] == [UNIT]  # recovery of a populated scope does not stop it
    assert successor.kill_process(SID)["status"] == "already_exited"
    assert stops[0] == [UNIT, UNIT]
    assert _checkpoint() == [] and load_result_record(SID)["systemd_unit"] == ""


def test_recovery_reaps_an_emptied_custody_scope_on_the_dead_pid_path(monkeypatch, stops):
    _finished_root(ProcessRegistry(), monkeypatch, True)
    _populated(monkeypatch, False)
    stops[1].append(False)  # the reap fails once: the entry is retained for the next recovery
    successor = ProcessRegistry()
    assert successor.recover_from_checkpoint() == 0 and SID not in successor._finished
    assert stops[0] == [UNIT] and _checkpoint()[0]["scope_custody"] is True

    assert ProcessRegistry().recover_from_checkpoint() == 0
    assert stops[0] == [UNIT, UNIT]
    assert _checkpoint() == [] and load_result_record(SID)["systemd_unit"] == ""


def test_another_interpreters_checkpoint_write_keeps_a_custody_entry(monkeypatch, stops):
    _finished_root(ProcessRegistry(), monkeypatch, True)
    ProcessRegistry()._write_checkpoint()  # an unrelated interpreter rewrites the shared file
    assert [entry["session_id"] for entry in _checkpoint()] == [SID]


def test_a_receipt_written_before_scope_custody_still_loads(monkeypatch):
    from tools.process_registry_results import load_completed_results
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    directory = get_hermes_home() / "logs" / "process-results"
    directory.mkdir(parents=True)
    record = {"id": SID, "command": "c", "cwd": None, "task_id": "", "owner_task_id": "", "session_key": "",
              "parent_session_id": "owner-session", "started_at": 1.0, "exit_code": 0,
              "completion_reason": "exited", "termination_source": "", "notify_on_complete": False, "output": ""}
    (directory / f"{SID}.json").write_text(json.dumps(record))
    loaded = load_completed_results(SID)[SID]
    assert loaded.systemd_unit == "" and not loaded._scope_stop_pending
    (directory / f"{SID}.json").write_text(json.dumps({**record, "systemd_unit": UNIT}))
    loaded = load_completed_results(SID)[SID]
    assert loaded.systemd_unit == UNIT and loaded._scope_stop_pending


def test_scope_populated_reads_the_kernel_cgroup(monkeypatch, tmp_path):
    class Shown:
        returncode = 0
        stdout = "LoadState=loaded\nControlGroup=/app.slice/x.scope\n"
    monkeypatch.setattr(custody.shutil, "which", lambda _name: "/bin/systemctl")
    monkeypatch.setattr(custody.subprocess, "run", lambda *a, **k: Shown)
    monkeypatch.setattr(custody, "_CGROUP_ROOT", tmp_path)
    (tmp_path / "cgroup.procs").write_text("1\n")
    scope = tmp_path / "app.slice" / "x.scope"
    scope.mkdir(parents=True)
    (scope / "cgroup.procs").write_text("4242\n")
    assert custody.scope_populated("x.scope") is True
    (scope / "cgroup.procs").write_text("")
    assert custody.scope_populated("x.scope") is False
    (scope / "cgroup.procs").unlink()
    assert custody.scope_populated("x.scope") is False
    Shown.stdout = "LoadState=not-found\nControlGroup=\n"
    assert custody.scope_populated("x.scope") is False
    Shown.returncode = 1  # no user bus: unknown, so custody is kept
    assert custody.scope_populated("x.scope") is None
