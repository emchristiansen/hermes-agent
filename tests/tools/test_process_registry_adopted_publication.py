"""An adopted worker command is complete only once its result is durable (C1-COMP-01).

A registered worker starts a log-backed command, checkpoints it and exits; a fresh interpreter adopts the checkpoint
and its log reader records the exit (``mark_exited``) and then publishes it (``_move_to_finished``: the receipt, the
completion). Between the two, nothing public may report the command as completed: a finite parent that saw ``exited``
could act on it, or exit, before the receipt exists. Real worker, checkpoint, adoption, command, exit record and
reader; only the publication step is held, at its entry, and then run unchanged.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tools.process_registry import ProcessRegistry
from tools.process_registry_results import load_result_record

ROOT = Path(__file__).resolve().parents[2]

SPAWNER = """
import json, os
from agent import runtime_session_store
runtime_session_store._worker_process = True
from tools.process_registry import process_registry
session = process_registry.spawn_local({command!r})
print(json.dumps(session.id), flush=True)
os._exit(0)
"""

# What the command does once released, and the outcome its reader must publish: (exit_code, reason, source, output).
OUTCOMES = {
    "succeeds": ("printf READY", (0, "exited", "", "READY")),
    "fails": ("printf FAILED; exit 3", (3, "exited", "", "FAILED")),
    # SIGKILL skips the exit trap: no status is recorded, so the adopter cannot know the exit.
    "unknown": ("printf GONE; kill -9 $$", (-1, "lost", "backend_lost", "GONE")),
}


class _HeldPublication(ProcessRegistry):
    """The real registry, its publication held at entry until released."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def _move_to_finished(self, session):
        self.entered.set()
        assert self.release.wait(30), "publication barrier expired"
        return super()._move_to_finished(session)


@pytest.fixture
def adopted(tmp_path, monkeypatch):
    """Adopt a real worker's command in a fresh registry; returns (registry, session id, gate file)."""
    if sys.platform == "win32":
        pytest.skip("log-backed worker commands are POSIX shell commands")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    gate = tmp_path / "go"
    started = []

    def adopt(then: str):
        command = f"while [ ! -f {gate} ]; do sleep 0.05; done; {then}"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        spawned = subprocess.run([sys.executable, "-c", SPAWNER.format(command=command)], cwd=ROOT, env=env,
                                 capture_output=True, text=True, timeout=60, check=True)
        session_id = json.loads(spawned.stdout.strip().splitlines()[-1])
        registry = _HeldPublication()
        assert registry.recover_from_checkpoint() == 1
        started.append(registry)
        gate.touch()
        assert registry.entered.wait(20), "the adopted reader never reached publication"
        return registry, session_id

    yield adopt
    for registry in started:
        registry.release.set()
        for session in [*registry._running.values(), *registry._finished.values()]:
            if session._reader_thread is not None:
                session._reader_thread.join(10)


def _assert_pending(registry: ProcessRegistry, session_id: str) -> None:
    waited = registry.wait(session_id, timeout=1)
    assert waited["status"] == "timeout" and "exit_code" not in waited, waited
    polled = registry.poll(session_id)
    assert polled["status"] == "settling" and "exit_code" not in polled, polled
    assert not registry.is_completion_consumed(session_id)
    assert load_result_record(session_id) is None


def _assert_published(registry: ProcessRegistry, session_id: str, expected) -> None:
    exit_code, reason, source, output = expected
    waited = registry.wait(session_id, timeout=10)
    receipt = load_result_record(session_id)  # immediately: no wait for it beyond the completed wait
    assert receipt is not None, waited
    for result in (waited, receipt, registry.poll(session_id)):
        assert (result["exit_code"], result["completion_reason"], result["termination_source"]) == (
            exit_code, reason, source), result
    # The command's own output is in both. (A login shell's explicit ``exit`` appends ``logout``, and the receipt keeps
    # the raw terminal bytes that the completion output strips, so they are not compared byte for byte.)
    assert waited["status"] == "exited" and output in waited["output"], waited
    assert output in receipt["output"], (receipt, waited)


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_an_adopted_command_is_reported_complete_only_once_its_receipt_is_durable(adopted, outcome):
    then, expected = OUTCOMES[outcome]
    registry, session_id = adopted(then)
    session = registry.get(session_id)
    assert session.exited and not session._completion_event.is_set()  # the reader recorded it; publication held
    _assert_pending(registry, session_id)
    registry.release.set()
    _assert_published(registry, session_id, expected)
    assert registry.is_completion_consumed(session_id)


def test_a_kill_during_publication_reports_settling_then_the_published_outcome(adopted):
    registry, session_id = adopted(OUTCOMES["succeeds"][0])
    killed = registry.kill_process(session_id, source="test.kill", consume_output=False)
    assert killed["status"] == "settling" and "exit_code" not in killed, killed
    assert load_result_record(session_id) is None
    registry.release.set()
    _assert_published(registry, session_id, OUTCOMES["succeeds"][1])
    again = registry.kill_process(session_id, source="test.kill", consume_output=False)
    assert (again["status"], again["exit_code"], again["completion_reason"]) == ("already_exited", 0, "exited"), again
