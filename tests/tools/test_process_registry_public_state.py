"""The process tool's ``list`` and ``log`` report one coherent observation of a session: its state and the exit
fields shown with it never come from different moments. A reader that records an exit while a kill still holds
its claim must not make a concurrent query show that pending exit as ``exited`` (MEMCG-R8-13)."""

import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import tools.process_registry as pr
from tools.process_registry_results import load_result_record

SID = "proc_publicstate"


def _query(surface: str) -> dict:
    response = json.loads(pr._handle_process({"action": surface, "session_id": SID}))
    return next(p for p in response["processes"] if p["session_id"] == SID) if surface == "list" else response


@pytest.fixture
def held_kill(monkeypatch):
    """A real kill A of a running session, held inside its physical signal (the only substituted effect)."""
    registry = pr.ProcessRegistry()
    session = pr.ProcessSession(
        id=SID, command="fixture", task_id="t", started_at=time.time(), notify_on_complete=True,
        parent_session_id="owner", output_buffer="partial output")
    session.process = SimpleNamespace(pid=999999999, stdout=None, stderr=None, stdin=None, poll=lambda: None)
    registry._running[SID] = session
    monkeypatch.setattr(pr, "process_registry", registry)
    entered, release, out = threading.Event(), threading.Event(), {}

    def signal(*_args):
        entered.set()
        assert release.wait(30)

    monkeypatch.setattr(registry, "_terminate_host_pid", signal)
    kill = threading.Thread(target=lambda: out.update(kill=registry.kill_process(SID, source="kill.a", consume_output=False)))
    kill.start()
    assert entered.wait(10)
    yield registry, session, out
    release.set()
    kill.join(10)
    assert not kill.is_alive()
    final = registry.poll(SID)
    assert (final["status"], final["exit_code"], final["completion_reason"]) == ("exited", -15, "killed")
    assert (out["kill"]["status"], out["kill"]["completion_reason"]) == ("killed", "killed")
    for surface in ("list", "log"):
        assert _query(surface)["status"] == "exited"
    assert _query("list")["exit_code"] == -15


def _assert_not_final(observed: dict) -> None:
    assert observed["status"] in ("running", "settling") and "exit_code" not in observed, observed
    assert "exited_at" not in observed and "completion_reason" not in observed, observed


@pytest.mark.parametrize("surface", ["list", "log"])
def test_a_query_racing_a_deferred_finish_shows_one_coherent_state(held_kill, surface):
    """The query pauses inside its state read, just after learning nothing is still settling the exit. The real
    reader then records exit 143 from another thread, deferring to kill A. The query's answer must be one
    observation: either from before the exit (``running``) or after it (``settling``), never ``exited``."""
    registry, session, _out = held_kill
    paused, resume, result = threading.Event(), threading.Event(), {}

    def tracer(frame, event, arg):
        if (event == "return" and frame.f_code is pr.ProcessRegistry._reader_finalizing.__code__ and not arg
                and frame.f_back.f_code.co_name == "_run_state" and frame.f_back.f_locals.get("session") is session):
            paused.set()
            assert resume.wait(10)
        return tracer

    def query():
        sys.settrace(tracer)
        try:
            result["observed"] = _query(surface)
        except BaseException as exc:  # surfaced below
            result["error"] = repr(exc)
        finally:
            sys.settrace(None)

    querier = threading.Thread(target=query)
    querier.start()
    assert paused.wait(10)
    # The finish runs in its own thread: a correct query holds the session's observation, so the finish waits
    # for it; an incoherent one lets the finish record the exit under the paused query.
    finisher = threading.Thread(target=lambda: registry._finish_exited(session, 143))
    finisher.start()
    finisher.join(1.0)
    resume.set()
    querier.join(10)
    finisher.join(10)
    assert not querier.is_alive() and not finisher.is_alive()
    assert "error" not in result, result
    _assert_not_final(result["observed"])
    assert session._kills_in_flight == 1 and not session._finish_claimed
    pending = registry.poll(SID)
    assert pending["status"] == "settling" and "exit_code" not in pending
    _assert_not_final(_query(surface))
    assert load_result_record(SID) is None


@pytest.mark.parametrize("surface", ["list", "log"])
def test_a_pending_exit_stays_settling_on_list_and_log(held_kill, surface):
    """Control without interleaving: before the exit the session is running; after a deferred finish it is
    settling, with partial output and no exit fields, until kill A publishes."""
    registry, session, _out = held_kill
    before = _query(surface)
    assert before["status"] == "running"
    registry._finish_exited(session, 143)
    observed = _query(surface)
    assert observed["status"] == "settling"
    _assert_not_final(observed)
    assert "partial output" in (observed["output_preview"] if surface == "list" else observed["output"])
    assert load_result_record(SID) is None
