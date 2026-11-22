"""A failed managed turn exits with its error, not a buffered-stdin fatal abort."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.platforms('posix')
def test_failed_worker_exits_while_owner_control_pipe_remains_open(tmp_path):
    code = textwrap.dedent('''
        import sys
        import threading
        from types import SimpleNamespace
        from agent import managed_worker as worker
        from tests.waiting import wait_until

        controls = worker.WorkerControls(SimpleNamespace(send=lambda *a, **kw: None), 'test-route')
        def reading():
            frame = sys._current_frames().get(controls.reader.ident)
            while frame is not None:
                if frame.f_code.co_name == 'read_frame':
                    return True
                frame = frame.f_back
            return False
        wait_until(reading, timeout=5, what='control reader waiting for owner')
        print('reader-ready', flush=True)
        raise RuntimeError('worker_receipt_timeout')
    ''')
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'))
    process = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Do not use communicate(): it closes stdin, hiding the production failure.
        assert process.stdout.readline() == b'reader-ready\n'
        assert process.wait(timeout=10) == 1
        stderr = process.stderr.read().decode()
        assert 'RuntimeError: worker_receipt_timeout' in stderr
        assert '_enter_buffered_busy' not in stderr
        assert 'Fatal Python error' not in stderr
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


@pytest.mark.platforms('posix')
def test_bootstrap_read_ahead_preserves_the_finish_control(tmp_path):
    code = textwrap.dedent('''
        import sys
        from agent import managed_worker as worker
        worker.hello = lambda: {}
        worker.validate_bootstrap = lambda frame: frame
        def execute(frame, channel, *, control_stream=None):
            controls = (worker.WorkerControls(channel, 'route', control_stream)
                        if control_stream is not None else worker.WorkerControls(channel, 'route'))
            if not controls.finish.wait(5):
                raise RuntimeError('finish control lost after bootstrap read-ahead')
            controls.reader.join(timeout=5)
            if controls.reader.is_alive():
                raise RuntimeError('reader did not stop after finish')
            channel.send('finished')
        worker.execute = execute
        raise SystemExit(worker.main())
    ''')
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'))
    process = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(b'{}\n{"type":"finish"}\n')
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
        assert b'"type":"finished"' in process.stdout.read()
        assert not process.stderr.read()
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
