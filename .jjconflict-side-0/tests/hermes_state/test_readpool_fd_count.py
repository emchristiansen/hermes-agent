"""The read pool's descriptor-headroom count stays cheap in a busy multi-threaded process.

Regression for the 2026-11-04 gateway stalls: every read open refreshes the count, and the
count listed /proc/self/fd. A listing hands the GIL off once per entry, so beside CPU-bound
threads a ~1,200-descriptor gateway spent seconds per count; twenty pool threads and the event
loop queued behind it, and owner replies missed managed workers' receipt budget.

The cheap path is the kernel sizing /proc/self/fd (Linux 6.2+). Where it reports 0 or the
stat fails, the listing is still the supported fallback, so only the functional contract
applies there, not the timing one.
"""
import contextlib
import errno
import os
import threading
import time

import pytest

import hermes_state_readpool as readpool

_EXTRA_FDS = 1000
_SPINNERS = 4
# Descriptors the test process must keep free beyond what a test opens (pytest, logging, the
# listing fallback's own directory handle).
_SPARE_FDS = 64


def _spin(started, stop):
    started.release()
    count = 0
    while not stop.is_set():
        count += 1


def _kernel_sizes_fd_dir() -> bool:
    try:
        return os.stat("/proc/self/fd").st_size > 0
    except OSError:
        return False


def _listed_fd_count() -> int:
    """The fallback's own measure: a listing, which counts the handle it opens for itself."""
    return len(os.listdir("/proc/self/fd"))


@contextlib.contextmanager
def _open_devnulls(count):
    """``count`` open descriptors, all closed on exit. Each descriptor is owned by the cleanup
    from the moment it opens, so running out partway (EMFILE/ENFILE) closes the ones already
    opened before the test is skipped for want of headroom."""
    soft = readpool._fd_soft_limit()
    if soft is not None and _listed_fd_count() + count + _SPARE_FDS > soft:
        pytest.skip(f"needs {count} spare descriptors under RLIMIT_NOFILE {soft}")
    fds = []
    try:
        try:
            for _ in range(count):
                fds.append(os.open(os.devnull, os.O_RDONLY))
        except OSError as exc:
            if exc.errno not in (errno.EMFILE, errno.ENFILE):
                raise
            pytest.skip(f"descriptor table exhausted after {len(fds)} of {count} opens")
        yield fds
    finally:
        for fd in fds:
            os.close(fd)


@pytest.fixture
def extra_fds():
    with _open_devnulls(_EXTRA_FDS) as fds:
        yield fds


@pytest.mark.platforms("linux")
def test_partial_descriptor_acquisition_closes_what_it_opened(monkeypatch):
    real_open = os.open
    opened = []

    def open_until_exhausted(path, flags, *args):
        if len(opened) == 10:
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE))
        fd = real_open(path, flags, *args)
        opened.append(fd)
        return fd

    before = _listed_fd_count()
    soft = readpool._fd_soft_limit()
    if soft is not None and before + 10 + 2 > soft:  # the 10 real opens, plus the listing handles
        pytest.skip(f"needs 10 spare descriptors under RLIMIT_NOFILE {soft}")
    # The injected EMFILE is the exhaustion under test; the helper's own headroom skip would
    # pre-empt it under a low limit.
    monkeypatch.setattr(readpool, "_fd_soft_limit", lambda: None)
    monkeypatch.setattr(os, "open", open_until_exhausted)
    with pytest.raises(pytest.skip.Exception, match="after 10 of 25"):
        with _open_devnulls(25):
            pass
    monkeypatch.undo()
    assert len(opened) == 10
    assert _listed_fd_count() == before


@pytest.mark.platforms("linux")
def test_fd_count_counts_each_open_descriptor():
    before = readpool._open_fd_count()
    with _open_devnulls(25):
        assert readpool._open_fd_count() == before + 25
    assert readpool._open_fd_count() == before


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("stat_result", ["zero_size", "stat_fails"])
def test_fd_count_falls_back_to_the_listing(monkeypatch, stat_result):
    real_stat = os.stat

    def stat_without_fd_size(path, *args, **kwargs):
        if path != "/proc/self/fd":
            return real_stat(path, *args, **kwargs)
        if stat_result == "stat_fails":
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        sized = real_stat(path, *args, **kwargs)
        return os.stat_result((*sized[:6], 0, *sized[7:]))

    monkeypatch.setattr(os, "stat", stat_without_fd_size)
    before = readpool._open_fd_count()
    assert before == _listed_fd_count()
    with _open_devnulls(25):
        assert readpool._open_fd_count() == before + 25
    assert readpool._open_fd_count() == before


@pytest.mark.parametrize("errno_value, expected", [(errno.EMFILE, -1), (errno.ENFILE, -1), (errno.ENOENT, None)])
def test_fd_count_reports_exhaustion_and_absence(monkeypatch, errno_value, expected):
    def unsized(path, *args, **kwargs):
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    def unlistable(path):
        raise OSError(errno_value, os.strerror(errno_value), path)

    monkeypatch.setattr(os, "stat", unsized)
    monkeypatch.setattr(os, "listdir", unlistable)
    assert readpool._open_fd_count() == expected


@pytest.mark.platforms("linux")
@pytest.mark.skipif(not _kernel_sizes_fd_dir(), reason="kernel does not size /proc/self/fd (Linux < 6.2)")
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
