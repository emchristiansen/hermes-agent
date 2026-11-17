"""A command's receipt says which scope it ran in and whether the kernel OOM-killed in that scope.

Production 2026-11-09: four commands died of cgroup OOM kills in their own ``hermes-worker-*.scope``
(kernel ``CONSTRAINT_MEMCG``, systemd "Failed with result 'oom-kill'"), yet their receipts showed only
exit 143/-15 with an empty ``termination_source`` and an empty ``systemd_unit``. The latter is custody
(released once the scope is empty), so nothing on the receipt said the command had a scope at all, and
nothing said why it ended. ``scope_unit`` is provenance, kept apart from custody through the receipt,
a cold load and every exit report. Exit codes stay as observed; nothing is normalized to 137.
"""
import json
import time

import pytest

import tools.process_registry_scope_custody as custody
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification
from tools.process_registry_results import _result_path, load_completed_results, load_result_record

SID = "proc_scopeoom0001"
UNIT = f"hermes-worker-{SID}.scope"


class _Commands:
    """``systemctl --user show`` and ``journalctl --user`` as the host would answer them."""

    def __init__(self, monkeypatch, tmp_path):
        self.show = "LoadState=not-found\nControlGroup=\n"
        self.journal = ""
        self.calls = []
        monkeypatch.setattr(custody.shutil, "which", lambda name: f"/bin/{name}")
        monkeypatch.setattr(custody.subprocess, "run", self._run)
        monkeypatch.setattr(custody, "_CGROUP_ROOT", tmp_path)
        tmp_path.mkdir()
        (tmp_path / "cgroup.procs").write_text("1\n")
        self.cgroup = tmp_path / "app.slice" / UNIT

    def _run(self, argv, **_kwargs):
        self.calls.append(argv[0])
        stdout = self.show if argv[0] == "/bin/systemctl" else self.journal

        class Done:
            returncode = 0
        Done.stdout = stdout
        return Done

    def live_scope(self, oom_kill: int):
        self.cgroup.mkdir(parents=True, exist_ok=True)
        (self.cgroup / "cgroup.procs").write_text("")
        (self.cgroup / "memory.events").write_text(
            f"low 0\nhigh 0\nmax 9\noom {oom_kill}\noom_kill {oom_kill}\noom_group_kill 0\n")
        self.show = f"LoadState=loaded\nControlGroup=/app.slice/{UNIT}\n"


@pytest.fixture
def host(monkeypatch, tmp_path):
    return _Commands(monkeypatch, tmp_path / "cgroupfs")


def test_the_scope_cgroup_counter_decides_while_it_exists(host):
    host.live_scope(oom_kill=2)
    assert custody.scope_oom_killed(UNIT) is True
    host.live_scope(oom_kill=0)
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"  # not consulted while the cgroup answers
    assert custody.scope_oom_killed(UNIT) is False
    assert "/bin/journalctl" not in host.calls


def test_a_collected_scope_is_decided_by_systemds_journal_record(host):
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    assert custody.scope_oom_killed(UNIT) is True
    host.journal = ""  # no record is not proof of no kill
    assert custody.scope_oom_killed(UNIT) is None


def _finish(registry, exit_code):
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", pid=999_999_999,
                             pid_scope="host", started_at=time.time(), systemd_unit=UNIT, scope_unit=UNIT,
                             output_buffer="evaluating...\n", notify_on_complete=True,
                             parent_session_id="owner-session")
    registry._running[SID] = session
    registry._finish_exited(session, exit_code)
    return session


def test_an_oom_killed_scope_is_on_the_receipt_and_the_completion(host, monkeypatch):
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"  # the scope is already collected at the finish
    registry = ProcessRegistry()
    _finish(registry, 143)

    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["termination_source"], receipt["completion_reason"]) == (
        143, "", "exited")
    assert receipt["systemd_unit"] == ""  # custody released: the scope is gone
    assert (receipt["scope_unit"], receipt["scope_oom_killed"]) == (UNIT, True)

    notice = registry.completion_queue.get_nowait()
    assert (notice["scope_unit"], notice["scope_oom_killed"], notice["exit_code"]) == (UNIT, True, 143)
    assert "kernel OOM killer" in format_process_notification(notice)

    # A cold interpreter reads the receipt: custody is empty, provenance and the OOM are not.
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    cold = ProcessRegistry()
    for report in (cold.poll(SID), cold.wait(SID, timeout=1)):
        assert (report["exit_code"], report["scope_unit"], report["scope_oom_killed"]) == (143, UNIT, True), report


def test_a_clean_exit_records_its_scope_without_asking_about_oom(host):
    registry = ProcessRegistry()
    _finish(registry, 0)
    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["scope_unit"], receipt["scope_oom_killed"]) == (0, UNIT, None)
    assert "/bin/journalctl" not in host.calls
    notice = registry.completion_queue.get_nowait()
    assert "scope_oom_killed" not in notice
    assert "OOM" not in format_process_notification(notice)


def test_a_receipt_written_before_scope_provenance_still_loads(monkeypatch, host):
    registry = ProcessRegistry()
    _finish(registry, 1)
    path = _result_path(SID)
    record = json.loads(path.read_text())
    del record["scope_unit"], record["scope_oom_killed"]
    path.write_text(json.dumps(record))
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    loaded = load_completed_results(SID)[SID]
    assert (loaded.scope_unit, loaded.scope_oom_killed) == ("", None)  # unknown, not inferred
    report = ProcessRegistry().poll(SID)
    assert "scope_unit" not in report and "scope_oom_killed" not in report


def test_a_kill_that_wins_during_the_slow_observation_keeps_its_receipt_whole(host, monkeypatch):
    """The OOM lookup runs outside every lock. A kill that commits meanwhile owns the result: the finish
    must not add the observation to the session or the durable receipt afterwards."""
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", pid=999_999_999,
                             pid_scope="host", started_at=time.time(), systemd_unit=UNIT, scope_unit=UNIT,
                             parent_session_id="owner-session")
    registry._running[SID] = session

    def observed_while_killed(unit):
        with session._lock:  # the kill's commit, as kill_process records it
            session.exited, session.exit_code = True, -15
            session.completion_reason, session.termination_source = "killed", "process.kill"
        registry._move_to_finished(session)
        return True
    monkeypatch.setattr(custody, "scope_oom_killed", observed_while_killed)

    registry._finish_exited(session, 143)

    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["scope_oom_killed"]) == (-15, "killed", None)
    assert (session.exit_code, session.scope_oom_killed) == (-15, None)


def test_the_orphaned_pipe_reconcile_records_the_scope_oom_too(host):
    """``_reconcile_local_exit`` finishes a root whose pipe a descendant still holds without the reader:
    that finisher observes the scope like the reader's."""
    from unittest.mock import MagicMock
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix build", task_id="t1", pid=999_999_999, pid_scope="host",
                             started_at=time.time(), systemd_unit=UNIT, scope_unit=UNIT,
                             parent_session_id="owner-session")
    session.process = MagicMock(stdout=None)
    session.process.poll.return_value = 143
    registry._running[SID] = session

    registry._reconcile_local_exit(session)

    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["scope_unit"], receipt["scope_oom_killed"]) == (143, UNIT, True)


class _Stream:
    closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(("first", "second"), [(True, None), (None, True), (False, True)])
def test_competing_finishers_report_the_one_outcome_the_winner_persisted(host, monkeypatch, first, second):
    """Two normal finishers (the reader, the orphaned-pipe reconcile) each observe the scope outside every
    lock; their answers can differ (the cgroup counter, then a journal not yet written). The first to
    record the exit persists the receipt and the notice, so the later one must not rewrite the session:
    warm poll/wait, a cold load and the queued notice all report the same cause (MEMCG-R3-04)."""
    import threading
    registry = ProcessRegistry()
    stream = _Stream()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", pid=999_999_999,
                             pid_scope="host", started_at=time.time(), systemd_unit=UNIT, scope_unit=UNIT,
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = type("Popen", (), {"stdout": stream, "stderr": None, "stdin": None})()
    registry._running[SID] = session
    observing, winner_done = threading.Event(), threading.Event()

    def observe(unit):
        if threading.current_thread().name == "loser":
            observing.set()
            winner_done.wait(10)  # its slow lookup returns only after the winner persisted
            return second
        return first
    monkeypatch.setattr(custody, "scope_oom_killed", observe)

    loser = threading.Thread(target=registry._finish_exited, args=(session, 143), name="loser", daemon=True)
    loser.start()
    assert observing.wait(10)
    registry._finish_exited(session, 143)
    winner_done.set()
    loser.join(10)
    assert not loser.is_alive()

    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["scope_unit"], receipt["scope_oom_killed"]) == (143, UNIT, first)
    assert receipt["systemd_unit"] == "" and not session._scope_stop_pending  # custody released: scope empty
    assert stream.closed  # the finished session's pipe handle is released
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty()  # published once, by the winner
    assert (notice["exit_code"], notice.get("scope_oom_killed")) == (143, first)
    assert ("kernel OOM killer" in format_process_notification(notice)) is (first is True)
    for report in (registry.poll(SID), registry.wait(SID, timeout=1)):
        assert (report["exit_code"], report.get("scope_oom_killed")) == (143, first), report
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    cold = ProcessRegistry()
    for report in (cold.poll(SID), cold.wait(SID, timeout=1)):
        assert (report["exit_code"], report.get("scope_oom_killed")) == (143, first), report


@pytest.mark.parametrize("observed", [True, False, None], ids=["oom", "no-oom", "unknown"])
def test_a_lost_adopted_command_keeps_its_unknown_exit_and_the_known_scope_oom(host, monkeypatch, tmp_path, observed):
    """An adopted log-backed command vanishes without a recorded status: the actual log reader finalizes it lost,
    exit -1 (unknown). A positive scope OOM observation is still a known cause, so the single notice carries it
    beside the lost status, as the coalesced batch does; false and unknown add nothing (MEMCG-R4-05)."""
    from gateway.run import GatewayRunner
    if observed is True:
        host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"  # the scope is already collected
    elif observed is False:
        host.live_scope(oom_kill=0)
    log = tmp_path / "proc.log"
    log.write_bytes(b"evaluating...\n")  # and no proc.exit beside it
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", pid=999_999_999, pid_scope="host",
                             detached=True, started_at=time.time(), systemd_unit=UNIT, scope_unit=UNIT,
                             output_log=str(log), notify_on_complete=True, parent_session_id="owner-session")
    registry._running[SID] = session
    monkeypatch.setattr(registry, "_detached_host_fate", lambda *_args: "exited")

    registry._log_reader_loop(session)

    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["termination_source"],
            receipt["scope_unit"], receipt["scope_oom_killed"]) == (-1, "lost", "backend_lost", UNIT, observed)
    notice = registry.completion_queue.get_nowait()
    assert (notice["exit_code"], notice["completion_reason"], notice.get("scope_oom_killed")) == (-1, "lost", observed)
    oom = f"the kernel OOM killer killed a process in its systemd scope {UNIT}"
    single = format_process_notification(notice)
    assert "marked lost because the process backend disappeared" in single and "(exit code -1)" in single
    assert (oom in single) is (observed is True), single
    plain = dict(notice, session_id="proc_plain", scope_oom_killed=None, scope_unit="")
    batch = GatewayRunner._format_coalesced_process_completions([("", notice, None), ("", plain, None)])
    line, = [text for text in batch.splitlines() if text.startswith(f"- {SID}:")]
    assert "exit_code=-1, reason=lost" in line and (oom in line) is (observed is True), line
    assert "OOM" not in next(text for text in batch.splitlines() if text.startswith("- proc_plain:"))
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    for reporter in (registry, ProcessRegistry()):
        for report in (reporter.poll(SID), reporter.wait(SID, timeout=1)):
            assert (report["exit_code"], report.get("scope_oom_killed")) == (-1, observed), report


class _Popen:
    """A local child the kill's real Popen branch signals; the physical signal is the seam."""
    pid, stdout, stderr, stdin = 999_999_999, None, None, None

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("consume_output", [True, False], ids=["explicit", "bulk"])
@pytest.mark.parametrize("order", ["exited-before-kill", "exit-during-signal", "exit-after-kill",
                                   "exit-during-failed-scope-stop"])
def test_a_kill_and_the_normal_finisher_report_one_outcome(host, monkeypatch, order, consume_output):
    """The real ``kill_process`` against the reader's real finish, in every order. A kill that finds the command
    exited reports that outcome; a kill of a running command owns the outcome, and a finish recorded while its
    signal is in flight (typically the signal's own effect) defers publication to it, keeping its scope OOM
    observation; a kill that cannot stop the scope leaves the finish its own. Either way one notice, the receipt,
    the kill's result and warm/cold poll/wait agree, and the scope's stop and custody hold (MEMCG-R4-06)."""
    import tools.process_registry as module
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", started_at=time.time(),
                             systemd_unit=UNIT, scope_unit=UNIT, output_buffer="evaluating...\n",
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = _Popen()
    registry._running[SID] = session
    stops = []

    def reader_finishes():  # the reader observes the exit and records it (its OOM lookup included)
        session.process.returncode = 143
        registry._finish_exited(session, 143)

    def stop(unit):
        stops.append(unit)
        if order == "exit-during-failed-scope-stop":
            reader_finishes()
            return False
        return True
    monkeypatch.setattr(module, "_stop_systemd_unit", stop)

    def signal(pid, *_args):
        assert pid == session.process.pid  # the real dispatcher chose its local-Popen branch
        if order == "exit-during-signal":
            reader_finishes()
        elif order != "exit-during-failed-scope-stop":
            session.process.returncode = -15
    monkeypatch.setattr(registry, "_terminate_host_pid", signal)

    if order == "exited-before-kill":
        reader_finishes()
    result = registry.kill_process(SID, consume_output=consume_output)
    if order == "exit-after-kill":
        registry._finish_exited(session, -15)  # the reader's late finish changes nothing

    expected = {"exited-before-kill": (143, "exited", True), "exit-during-signal": (-15, "killed", True),
                "exit-after-kill": (-15, "killed", None),
                "exit-during-failed-scope-stop": (143, "exited", True)}[order]
    status = {"exited-before-kill": "already_exited", "exit-during-failed-scope-stop": "error"}.get(order, "killed")
    assert result["status"] == status, result
    if status == "error":
        assert (result["scope_stop_failed"], result["process_running"]) == (UNIT, False), result
        assert session._scope_stop_pending  # custody kept for the retry
    else:
        assert result["completion_reason"] == expected[1] and "evaluating" in result["output"], result
        assert not session._scope_stop_pending
    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["scope_oom_killed"]) == expected
    assert receipt["systemd_unit"] == (UNIT if status == "error" else "")
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty()  # one publication
    assert (notice["exit_code"], notice["completion_reason"], notice.get("scope_oom_killed")) == expected
    assert ("kernel OOM killer" in format_process_notification(notice)) is (expected[2] is True)
    assert stops == [UNIT]
    assert registry.is_completion_consumed(SID) is (consume_output and status != "error")
    assert session._completion_event.is_set() and not session._kills_in_flight
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    for reporter in (registry, ProcessRegistry()):
        for report in (reporter.poll(SID), reporter.wait(SID, timeout=1)):
            got = (report["exit_code"], report.get("completion_reason", "exited"), report.get("scope_oom_killed"))
            assert got == expected, report


@pytest.mark.parametrize("consume_output", [True, False], ids=["explicit", "bulk"])
@pytest.mark.parametrize("b_fails", ["refused", "incomplete"])
def test_an_overlapping_kill_that_fails_never_releases_another_kills_claim(host, monkeypatch, b_fails, consume_output):
    """MEMCG-R5-07: kill A claims the running command and is held inside its physical signal; kill B also claims
    it and fails (its signal is refused, or the command survives it). B releases only its own claim, so the reader
    finishing after B still defers to A, and A publishes one outcome: the queued notice, the receipt, both results
    and warm/cold poll/wait agree, and the scope is stopped by the kills that reached it, never left in custody."""
    import threading
    import tools.process_registry as module
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", started_at=time.time(),
                             systemd_unit=UNIT, scope_unit=UNIT, output_buffer="evaluating...\n",
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = _Popen()
    registry._running[SID] = session
    stops = []
    monkeypatch.setattr(module, "_stop_systemd_unit", lambda unit: stops.append(unit) or True)
    monkeypatch.setattr(registry, "_KILL_SETTLE_SECONDS", 0.05)
    a_signalling, a_release = threading.Event(), threading.Event()

    def signal(pid, *_args):
        assert pid == session.process.pid  # the real dispatcher chose its local-Popen branch
        if threading.current_thread().name == "kill-a":
            a_signalling.set()
            assert a_release.wait(10)
        elif b_fails == "refused":
            raise PermissionError(1, "Operation not permitted")
        # "incomplete": B's signal lands nowhere; the root still runs, so B finds a survivor
    monkeypatch.setattr(registry, "_terminate_host_pid", signal)

    results = {}
    kill_a = threading.Thread(name="kill-a", daemon=True, target=lambda: results.__setitem__(
        "a", registry.kill_process(SID, consume_output=consume_output)))
    kill_a.start()
    try:
        assert a_signalling.wait(10)
        results["b"] = registry.kill_process(SID, consume_output=consume_output)
        assert results["b"]["status"] == "error", results["b"]
        assert session._kills_in_flight == 1  # A's claim survives B's failure
        session.process.returncode = 143
        registry._finish_exited(session, 143)  # the reader records the exit while A still signals
        assert registry.completion_queue.empty() and not session._completion_event.is_set()
    finally:
        a_release.set()
        kill_a.join(10)
    assert not kill_a.is_alive()

    expected = (-15, "killed", True)  # A's kill, with the reader's scope OOM observation kept
    assert results["a"]["status"] == "killed" and results["a"]["completion_reason"] == "killed", results["a"]
    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["scope_oom_killed"]) == expected
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty()  # one publication
    assert (notice["exit_code"], notice["completion_reason"], notice.get("scope_oom_killed")) == expected
    assert stops == ([UNIT] if b_fails == "refused" else [UNIT, UNIT]) and not session._scope_stop_pending
    assert receipt["systemd_unit"] == ""
    assert registry.is_completion_consumed(SID) is consume_output
    assert session._kills_in_flight == 0 and session._completion_event.is_set()
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    for reporter in (registry, ProcessRegistry()):
        for report in (reporter.poll(SID), reporter.wait(SID, timeout=1)):
            got = (report["exit_code"], report.get("completion_reason", "exited"), report.get("scope_oom_killed"))
            assert got == expected, report


@pytest.mark.parametrize("consume_output", [True, False], ids=["explicit", "bulk"])
@pytest.mark.parametrize("b_arrives", ["late-unsettled", "late-settles", "early"])
def test_a_second_kill_never_reports_an_unsettled_exit_as_final(host, monkeypatch, b_arrives, consume_output):
    """MEMCG-R6-09: kill A claims the running command and is held inside its physical signal; the reader records
    the exit (deferred to A). A kill B arriving now must not return that provisional exit as a terminal result:
    while A has not settled it says ``settling`` (no exit fields, nothing consumed); if A settles within its short
    wait it reports A's outcome. B admitted before the exit (early) runs its own kill. Either way one notice, the
    receipt, every terminal result and warm/cold poll/wait agree, and custody is released."""
    import threading
    import tools.process_registry as module
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", started_at=time.time(),
                             systemd_unit=UNIT, scope_unit=UNIT, output_buffer="evaluating...\n",
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = _Popen()
    registry._running[SID] = session
    stops = []
    monkeypatch.setattr(module, "_stop_systemd_unit", lambda unit: stops.append(unit) or True)
    a_signalling, a_release = threading.Event(), threading.Event()

    def signal(pid, *_args):
        assert pid == session.process.pid  # the real dispatcher chose its local-Popen branch
        if threading.current_thread().name == "kill-a":
            a_signalling.set()
            assert a_release.wait(10)
        else:
            session.process.returncode = -15  # B's signal lands
    monkeypatch.setattr(registry, "_terminate_host_pid", signal)

    results = {}
    kill_a = threading.Thread(name="kill-a", daemon=True, target=lambda: results.__setitem__(
        "a", registry.kill_process(SID, source="kill.a", consume_output=consume_output)))
    kill_a.start()
    try:
        assert a_signalling.wait(10)
        if b_arrives == "early":
            results["b"] = registry.kill_process(SID, source="kill.b", consume_output=consume_output)
        session.process.returncode = session.process.returncode or 143
        registry._finish_exited(session, 143)  # the reader records the exit while A still signals
        if b_arrives == "late-settles":
            threading.Timer(0.1, a_release.set).start()  # A settles within B's wait
        if b_arrives != "early":
            results["b"] = registry.kill_process(SID, source="kill.b", consume_output=consume_output)
    finally:
        a_release.set()
        kill_a.join(10)
    assert not kill_a.is_alive()

    source = "kill.b" if b_arrives == "early" else "kill.a"
    expected = (-15, "killed", source)
    assert (results["a"]["status"], results["a"]["termination_source"]) == ("killed", source), results["a"]
    b = results["b"]
    if b_arrives == "late-unsettled":
        assert b["status"] == "settling" and "exit_code" not in b and "completion_reason" not in b, b
    elif b_arrives == "late-settles":
        assert (b["status"], b["exit_code"], b["completion_reason"], b["termination_source"]) == (
            "already_exited", *expected), b
    else:
        assert (b["status"], b["completion_reason"], b["termination_source"]) == ("killed", "killed", source), b
    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["termination_source"]) == expected
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty()  # one publication
    assert (notice["exit_code"], notice["completion_reason"], notice["termination_source"]) == expected
    assert set(stops) == {UNIT} and not session._scope_stop_pending and receipt["systemd_unit"] == ""
    # consumed only by a kill that returned the settled output (a settling B consumes nothing)
    assert registry.is_completion_consumed(SID) is consume_output
    assert session._kills_in_flight == 0 and session._completion_event.is_set()
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    for reporter in (registry, ProcessRegistry()):
        for report in (reporter.poll(SID), reporter.wait(SID, timeout=1)):
            got = (report["exit_code"], report.get("completion_reason"), report.get("termination_source"))
            assert got == expected, report


@pytest.mark.parametrize("consume_output", [True, False], ids=["explicit", "bulk"])
@pytest.mark.parametrize("a_stop", ["a-stops", "a-stop-fails"])
@pytest.mark.parametrize("b_stop", ["b-stop-fails", "b-stops"])
def test_a_pending_outcome_is_never_persisted_or_reported_as_final(host, monkeypatch, b_stop, a_stop, consume_output):
    """MEMCG-R7-11/R7-12: kill A owns the running command and is held inside its physical signal; the reader records
    143 with a positive scope OOM (deferred to A). Until A settles, nothing may present that exit as final: the real
    process tool's poll and list say ``settling`` without exit fields; a late kill B whose scope stop fails keeps the
    custody (checkpointed as a running entry) but writes no completed receipt, so a cold interpreter's poll/wait see
    no terminal result. Then A settles once: killed if its stop works; else its failed stop publishes the reader's
    own outcome with the scope still in custody, and a later stop releases it. Notice, receipt, results and
    warm/cold queries agree."""
    import json
    import threading
    import tools.process_registry as module
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    monkeypatch.setattr(module, "process_registry", registry)
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", started_at=time.time(),
                             systemd_unit=UNIT, scope_unit=UNIT, output_buffer="evaluating...\n",
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = _Popen()
    registry._running[SID] = session
    stop_plan = [b_stop == "b-stops", a_stop == "a-stops", True]  # B's stop, A's stop, a later retry
    stops = []
    monkeypatch.setattr(module, "_stop_systemd_unit", lambda unit: stops.append(unit) or stop_plan[len(stops) - 1])
    a_signalling, a_release = threading.Event(), threading.Event()

    def signal(pid, *_args):
        assert pid == session.process.pid  # the real dispatcher chose its local-Popen branch
        a_signalling.set()
        assert a_release.wait(10)
    monkeypatch.setattr(registry, "_terminate_host_pid", signal)

    def tool_poll():
        return json.loads(module._handle_process({"action": "poll", "session_id": SID}))

    results = {}
    kill_a = threading.Thread(name="kill-a", daemon=True, target=lambda: results.__setitem__(
        "a", registry.kill_process(SID, source="kill.a", consume_output=consume_output)))
    kill_a.start()
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    try:
        assert a_signalling.wait(10)
        session.process.returncode = 143
        registry._finish_exited(session, 143)  # the reader records the exit; publication defers to A
        for pending in (tool_poll(), registry.list_sessions()[0]):
            assert pending["status"] == "settling" and "exit_code" not in pending, pending
            assert "completion_reason" not in pending and "termination_source" not in pending, pending
        assert "evaluating" in tool_poll()["output_preview"]  # partial output stays visible
        results["b"] = registry.kill_process(SID, source="kill.b", consume_output=consume_output)
        assert results["b"]["status"] == "settling" and "exit_code" not in results["b"], results["b"]
        assert ("scope_stop_failed" in results["b"]) is (b_stop == "b-stop-fails")
        assert load_result_record(SID) is None  # no completed receipt for a pending outcome
        cold = ProcessRegistry()
        for report in (cold.poll(SID), cold.wait(SID, timeout=1)):
            assert report["status"] != "exited" and "exit_code" not in report, report
        if b_stop == "b-stop-fails":  # custody kept recoverable, as a running entry
            entry, = [e for e in json.loads(module._checkpoint_path().read_text()) if e["session_id"] == SID]
            assert entry["systemd_unit"] == UNIT and "scope_custody" not in entry and "exit_code" not in entry
        assert registry.completion_queue.empty()
    finally:
        a_release.set()
        kill_a.join(10)
    assert not kill_a.is_alive()

    if a_stop == "a-stops":
        expected = (-15, "killed", "kill.a", True)
        assert results["a"]["status"] == "killed", results["a"]
        assert load_result_record(SID)["systemd_unit"] == "" and not session._scope_stop_pending
        assert registry.is_completion_consumed(SID) is consume_output
    else:
        expected = (143, "exited", "", True)  # the kill never completed: the reader's own outcome, scope held
        assert (results["a"]["status"], results["a"]["scope_stop_failed"]) == ("error", UNIT), results["a"]
        assert load_result_record(SID)["systemd_unit"] == UNIT and session._scope_stop_pending
        retry = registry.kill_process(SID)
        assert retry["status"] == "already_exited" and "scope_stop_failed" not in retry, retry
        assert load_result_record(SID)["systemd_unit"] == "" and not session._scope_stop_pending
    assert len(stops) == (3 if a_stop == "a-stop-fails" else 2)
    receipt = load_result_record(SID)
    got = lambda r: (r["exit_code"], r.get("completion_reason"), r.get("termination_source"), r.get("scope_oom_killed"))
    assert got(receipt) == expected
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty() and got(notice) == expected  # one publication
    assert tool_poll()["status"] == "exited" and got(tool_poll()) == expected
    for reporter in (registry, ProcessRegistry()):
        for report in (reporter.poll(SID), reporter.wait(SID, timeout=1)):
            assert got(report) == expected, report


@pytest.mark.parametrize("consume_output", [True, False], ids=["explicit", "bulk"])
def test_a_failed_late_scope_stop_writes_no_receipt_for_a_pending_outcome(host, monkeypatch, consume_output):
    """MEMCG-R7-11 alone (no poll surface involved): while kill A holds the reader's deferred 143, a late kill B
    whose scope stop fails keeps custody but must not persist that pending exit as a completed receipt, so a cold
    interpreter sees no terminal result; A then publishes the one final outcome and the receipt follows it."""
    import json
    import threading
    import tools.process_registry as module
    host.journal = custody._UNIT_OOM_MESSAGE_ID + "\n"
    registry = ProcessRegistry()
    session = ProcessSession(id=SID, command="nix flake check", task_id="t1", started_at=time.time(),
                             systemd_unit=UNIT, scope_unit=UNIT, output_buffer="evaluating...\n",
                             notify_on_complete=True, parent_session_id="owner-session")
    session.process = _Popen()
    registry._running[SID] = session
    stop_plan = [False, True]  # B's stop fails, A's stop works
    stops = []
    monkeypatch.setattr(module, "_stop_systemd_unit", lambda unit: stops.append(unit) or stop_plan[len(stops) - 1])
    a_signalling, a_release = threading.Event(), threading.Event()

    def signal(pid, *_args):
        a_signalling.set()
        assert a_release.wait(10)
    monkeypatch.setattr(registry, "_terminate_host_pid", signal)
    results = {}
    kill_a = threading.Thread(name="kill-a", daemon=True, target=lambda: results.__setitem__(
        "a", registry.kill_process(SID, source="kill.a", consume_output=consume_output)))
    kill_a.start()
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    try:
        assert a_signalling.wait(10)
        session.process.returncode = 143
        registry._finish_exited(session, 143)
        registry.kill_process(SID, source="kill.b", consume_output=consume_output)
        assert session._scope_stop_pending  # custody kept for the retry
        assert load_result_record(SID) is None, load_result_record(SID)
        cold = ProcessRegistry()
        for report in (cold.poll(SID), cold.wait(SID, timeout=1)):
            assert "exit_code" not in report, report
        entry, = [e for e in json.loads(module._checkpoint_path().read_text()) if e["session_id"] == SID]
        assert entry["systemd_unit"] == UNIT and "exit_code" not in entry
    finally:
        a_release.set()
        kill_a.join(10)
    receipt = load_result_record(SID)
    assert (receipt["exit_code"], receipt["completion_reason"], receipt["scope_oom_killed"]) == (-15, "killed", True)
    assert receipt["systemd_unit"] == "" and results["a"]["status"] == "killed"
    notice = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty() and (notice["exit_code"], notice["completion_reason"]) == (-15, "killed")
