"""The read pool's descriptor-headroom count stays cheap in a busy multi-threaded process.

Regression for the 2026-11-04 gateway stalls: every read open refreshes the count, and the
count listed /proc/self/fd. A listing hands the GIL off once per entry, so beside CPU-bound
threads a ~1,200-descriptor gateway spent seconds per count; twenty pool threads and the event
loop queued behind it, and owner replies missed managed workers' receipt budget.
"""
import os
import threading
import time

import pytest

import hermes_state_readpool as readpool

_EXTRA_FDS = 1000
_SPINNERS = 4


def _spin(started, stop):
    started.release()
    count = 0
    while not stop.is_set():
        count += 1


@pytest.fixture
def extra_fds():
    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(_EXTRA_FDS)]
    try:
        yield fds
    finally:
        for fd in fds:
            os.close(fd)


@pytest.mark.platforms("linux")
def test_fd_count_counts_each_open_descriptor():
    before = readpool._open_fd_count()
    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(25)]
    try:
        assert readpool._open_fd_count() == before + 25
    finally:
        for fd in fds:
            os.close(fd)
    assert readpool._open_fd_count() == before


@pytest.mark.platforms("linux")
def test_fd_count_stays_cheap_beside_cpu_bound_threads(extra_fds):
    started, stop = threading.Semaphore(0), threading.Event()
    spinners = [threading.Thread(target=_spin, args=(started, stop), daemon=True) for _ in range(_SPINNERS)]
    for spinner in spinners:
        spinner.start()
    try:
        for _ in spinners:
            assert started.acquire(timeout=10)
        began = time.monotonic()
        count = readpool._open_fd_count()
        elapsed = time.monotonic() - began
    finally:
        stop.set()
        for spinner in spinners:
            spinner.join(10)
    assert count >= _EXTRA_FDS
    assert elapsed < 2.0, f'counting {count} descriptors took {elapsed:.1f}s beside {_SPINNERS} busy threads'
