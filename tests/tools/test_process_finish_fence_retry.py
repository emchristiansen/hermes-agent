"""BG-SCOPE-07: a reader that abandons its finish still releases the handles it owns.

A kill that lands while the PTY reader is finishing publishes the killed result, claims the completion
and leaves the PTY to the still-alive reader (closing it from another thread can block on the reader's
read). If the reader's completion fence then fails, it must not publish anything itself (the kill owns
the result), but it must still close the PTY on its way out. Real POSIX PTY and child, real registry,
receipts and file locks; only the completion fence's entry is scheduled (and, where named, fails).
"""
import sys
import threading

import pytest

from hermes_cli.active_sessions import _FileLock
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import load_result_record


@pytest.mark.platforms("posix")  # a real POSIX PTY
@pytest.mark.parametrize("fence", ["fails", "healthy", "heals"])
def test_a_kill_during_the_readers_finish_leaves_one_result_and_a_closed_pty(monkeypatch, fence):
    """``fails``: the reader's fence entry fails after the kill claimed the completion. ``healthy``: it
    succeeds (the duplicate finish). ``heals``: it fails before the kill, the kill lands while the reader
    waits to retry, and the retry succeeds."""
    pytest.importorskip("ptyprocess")
    from ptyprocess import PtyProcess
    registry = ProcessRegistry()
    pty = PtyProcess.spawn([sys.executable, "-c", "print('preserved tail')"])
    session = ProcessSession(id="proc_fenceretry07", command="pty probe", pid=pty.pid, _pty=pty,
                             notify_on_complete=True)
    registry._running[session.id] = session
    entered, killed = threading.Event(), threading.Event()
    entries = []
    real_enter = _FileLock.__enter__

    def enter(lock):
        if not str(lock.path).endswith(".finish.lock"):
            return real_enter(lock)
        entries.append(fence)
        if fence == "heals" and len(entries) == 1:
            raise PermissionError("injected: completion fence unavailable")
        entered.set()  # ``heals``: the reader is now in its retry, so the kill lands between the two
        assert killed.wait(10), "the kill did not happen"
        if fence == "fails":
            raise PermissionError("injected: completion fence unavailable")
        return real_enter(lock)

    monkeypatch.setattr(_FileLock, "__enter__", enter)
    errors = []

    def reader():
        try:
            registry._pty_reader_loop(session)
        except BaseException as exc:  # surfaced below
            errors.append(repr(exc))

    thread = threading.Thread(target=reader, name="pty-reader", daemon=True)
    session._reader_thread = thread
    try:
        thread.start()
        assert entered.wait(10), "the reader did not reach the fence the kill must precede"
        result = registry.kill_process(session.id, consume_output=False)
        assert not pty.closed  # left to the live reader
        killed.set()
        thread.join(10)
        assert not errors and not thread.is_alive()
        assert result["status"] == "killed"
        assert pty.closed, "the reader exited without releasing the PTY it owns"
        assert (session.exit_code, session.completion_reason) == (-15, "killed")
        assert "preserved tail" in session.output_buffer
        receipt = load_result_record(session.id)
        assert (receipt["exit_code"], receipt["completion_reason"]) == (-15, "killed")
        assert "preserved tail" in receipt["output"]
        assert registry.completion_queue.qsize() == 1
        assert len(entries) == (2 if fence == "heals" else 1)
    finally:
        killed.set()
        thread.join(10)
        if not pty.closed:
            pty.close(force=True)
