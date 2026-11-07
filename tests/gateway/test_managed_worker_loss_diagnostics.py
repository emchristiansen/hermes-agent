"""A lost managed turn says how it ended.

2026-11-04 17:19:53: 18 managed workers were lost within a second of their submitters exiting,
and the owner logged only ``Managed worker lost: EOFError``. The bound worker's submitter watch
``os._exit``-s without a word, so nothing told that exit apart from an outside kill. The watch now
says why on stderr (drained into managed-worker.log), and the loss line carries the child's pid
and exit status.
"""
import os
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from gateway.session_managed_worker import _describe_exit


def test_exit_status_names_the_signal():
    assert _describe_exit(None) == 'running'
    assert _describe_exit(1) == '1'
    assert _describe_exit(-9) == '-9 (SIGKILL)'
    assert _describe_exit(-15) == '-15 (SIGTERM)'


def _bound_worker(tmp_path, *, full_stderr=False, no_threads=False):
    """A real process running the real submitter watch over a scratch run row whose submitter
    has exited and been reaped; ``full_stderr`` first fills its stderr pipe."""
    gone = subprocess.Popen([sys.executable, '-c', 'pass'])
    gone.wait()
    db = tmp_path / 'kanban.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE task_runs (id INTEGER PRIMARY KEY, worker_pid INTEGER, worker_started_at TEXT)')
        conn.execute('INSERT INTO task_runs VALUES (7, ?, NULL)', (gone.pid,))
    fill = textwrap.dedent('''
        os.set_blocking(2, False)
        try:
            while True:
                os.write(2, b'x' * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(2, True)
    ''') if full_stderr else ''
    if no_threads:  # the watch itself runs; any thread started after it cannot be
        fill += textwrap.dedent('''
            import threading
            _start = threading.Thread.start
            def _refuse(self):
                if self.name != 'kanban-submitter-watch':
                    raise RuntimeError("can't start new thread")
                _start(self)
            threading.Thread.start = _refuse
        ''')
    script = textwrap.dedent('''
        import os, time
        from pathlib import Path
        from gateway.session_kanban import _watch_submitter
    ''') + fill + textwrap.dedent(f'''
        _watch_submitter(Path({str(db)!r}), 7)
        time.sleep(20)
    ''')
    return gone.pid, script


@pytest.mark.platforms("linux", "macos")
def test_the_submitter_watch_says_why_it_ends_the_bound_worker(tmp_path):
    gone_pid, script = _bound_worker(tmp_path)
    began = time.monotonic()
    bound = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30, check=False,
                           env={**os.environ, 'PYTHONPATH': os.getcwd()})
    assert bound.returncode == 1
    assert time.monotonic() - began < 15
    assert f'kanban run 7: submitter pid {gone_pid} exited; ending this bound worker' in bound.stderr


@pytest.mark.platforms("linux", "macos")
def test_a_full_stderr_pipe_does_not_hold_back_the_exit(tmp_path):
    _gone_pid, script = _bound_worker(tmp_path, full_stderr=True)
    began = time.monotonic()
    bound = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
    try:
        # Nothing reads the pipe until the worker has exited.
        assert bound.wait(timeout=15) == 1
    finally:
        bound.kill()
        bound.communicate()
    assert time.monotonic() - began < 15


@pytest.mark.platforms("linux", "macos")
def test_the_watch_still_exits_when_the_note_cannot_be_written(tmp_path):
    """A process out of threads cannot start the note's writer; the exit must still happen."""
    _gone_pid, script = _bound_worker(tmp_path, no_threads=True)
    began = time.monotonic()
    bound = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30, check=False,
                           env={**os.environ, 'PYTHONPATH': os.getcwd()})
    assert bound.returncode == 1
    assert time.monotonic() - began < 15
