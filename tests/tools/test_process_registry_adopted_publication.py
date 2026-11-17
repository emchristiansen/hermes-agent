"""An adopted worker command is complete only once its result is durable (C1-COMP-01).

A registered worker starts a log-backed command, checkpoints it and exits; a fresh interpreter adopts the checkpoint
and its log reader records the exit (``mark_exited``) and then publishes it (``_move_to_finished``: the receipt, the
completion). Between the two, nothing public may report the command as completed: a finite parent that saw ``exited``
could act on it, or exit, before the receipt exists. Real worker, checkpoint, adoption, command, exit record and
reader; only the publication step is held, at its entry, and then run unchanged.

The exit record itself is one step under the session lock, yet sets ``exited`` before the outcome (C2-COMP-02): held
there, with the reader's lock held, no query may report or consume an exit without its outcome, and a wait still
times out or yields.
"""

import inspect
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.interrupt import request_yield
from tools.process_registry import ProcessRegistry, ProcessSession
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


class _HeldExitRecord:
    """Traces the real reader's ``mark_exited`` and holds it just after its ``exited = True``, the session lock
    held, until released. Nothing it executes is replaced."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        lines, start = inspect.getsourcelines(ProcessSession.mark_exited)
        self.line = start + 1 + next(i for i, line in enumerate(lines) if line.strip() == "self.exited = True")

    def trace(self, frame, event, arg):
        return self._hold if frame.f_code is ProcessSession.mark_exited.__code__ else None

    def _hold(self, frame, event, arg):
        if event == "line" and frame.f_lineno == self.line and not self.entered.is_set():
            self.entered.set()
            if not self.release.wait(30):
                raise RuntimeError("exit-record barrier expired")
        return self._hold


@pytest.fixture
def adopted(tmp_path, monkeypatch):
    """Adopt a real worker's command in a fresh registry; returns (registry, session id, gate file)."""
    if sys.platform == "win32":
        pytest.skip("log-backed worker commands are POSIX shell commands")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    gate = tmp_path / "go"
    started = []

    def adopt(then: str, held_exit: _HeldExitRecord | None = None):
        command = f"while [ ! -f {gate} ]; do sleep 0.05; done; {then}"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        spawned = subprocess.run([sys.executable, "-c", SPAWNER.format(command=command)], cwd=ROOT, env=env,
                                 capture_output=True, text=True, timeout=60, check=True)
        session_id = json.loads(spawned.stdout.strip().splitlines()[-1])
        registry = _HeldPublication()
        if held_exit is not None:
            threading.settrace(held_exit.trace)  # the reader thread recovery starts
            holds.append(held_exit)
        assert registry.recover_from_checkpoint() == 1
        threading.settrace(None)
        started.append(registry)
        gate.touch()
        held = held_exit.entered if held_exit is not None else registry.entered
        assert held.wait(20), "the adopted reader never reached the held step"
        return registry, session_id

    holds = []
    yield adopt
    threading.settrace(None)
    for held_exit in holds:
        held_exit.release.set()
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


def test_an_exit_being_recorded_is_neither_reported_nor_consumed(adopted):
    held_exit = _HeldExitRecord()
    registry, session_id = adopted(OUTCOMES["fails"][0], held_exit)
    session = registry.get(session_id)
    assert session.exited and not session._outcome_final  # inside the real exit record, its lock held
    started = time.monotonic()
    waited = registry.wait(session_id, timeout=1)  # the writer's lock does not hold the wait past its timeout
    assert waited["status"] == "timeout" and "exit_code" not in waited, waited
    request_yield(threading.current_thread().ident)
    yielded = registry.wait(session_id, timeout=60)  # nor past a steer's yield
    assert yielded["status"] == "interrupted" and "exit_code" not in yielded, yielded
    assert time.monotonic() - started < 10
    assert not registry.is_completion_consumed(session_id) and load_result_record(session_id) is None
    # Queries issued now read once the record is complete: the outcome is recorded, its publication still held.
    answers = {}
    queries = [threading.Thread(target=lambda name=name, query=query: answers.__setitem__(name, query()))
               for name, query in (("poll", lambda: registry.poll(session_id)),
                                   ("list", lambda: registry.list_sessions()))]
    for query in queries:
        query.start()
    held_exit.release.set()
    assert registry.entered.wait(20), "the reader never reached publication"
    for query in queries:
        query.join(20)
    listed = [entry for entry in answers["list"] if entry["session_id"] == session_id]
    for answer in (answers["poll"], *listed):
        assert answer["status"] == "settling" and "exit_code" not in answer, answer
    assert len(listed) == 1, answers["list"]
    _assert_pending(registry, session_id)
    registry.release.set()
    _assert_published(registry, session_id, OUTCOMES["fails"][1])
    assert registry.is_completion_consumed(session_id)
