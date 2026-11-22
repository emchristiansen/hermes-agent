"""SH-01 STORE-01: reading the unfinished-close set must never finalize a SessionDB under its lock.

``SessionDB.__del__`` closes the handle, and ``close()`` records itself in ``hermes_state``'s
unfinished-close set under ``_unsettled_closes_lock``. If the observer (``unsettled_closes``, read by
the gateway stop's close verdict) drops the last reference to such a handle while it holds that lock,
the finalizer re-enters the lock on the same thread: with a non-reentrant lock the verdict never
returns and every later close blocks.

The two schedule-sensitive cases run in a disposable subprocess, so a deadlock fails the case (the
subprocess reports a blocked call and exits with ``os._exit``) instead of wedging the test process.
The schedule is chosen with ``sys.settrace`` only: the SessionDB, its failed close, the registry's
quiet close, the lock, the observer and the finalizer are all real. Controls: an empty and a
non-empty snapshot, the per-directory filter, and a close that finishes leaving the set.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parents[2]

# A handle whose close did not finish: the real retired-generation settlement raises (injected), and the
# real registry quiet-close swallows it, so the handle stays open and is recorded as unfinished.
_SCENARIO = r'''
import json, os, sys, threading, weakref
from pathlib import Path
import hermes_state as hs
import hermes_state_registry as reg
from hermes_state_dbfile import RetiredGenerationCaptureError
mode, path = sys.argv[1], sys.argv[2]
db = hs.SessionDB(Path(path))
def failed_capture():
    raise RetiredGenerationCaptureError("injected capture failure")
db._db_wal_generation_lost = True
db._settle_lost_generation_locked = failed_capture
reg.release_or_close(db)
out = {"before": hs.unsettled_closes()}
# Instrumentation only (the real finalizer still runs): is the unfinished-close lock held by this thread
# when the finalizer starts?
seen = {}
original_del = hs.SessionDB.__del__
def recording_del(self):
    lock = hs._unsettled_closes_lock
    seen.setdefault("finalizer_under_lock", lock._is_owned() if hasattr(lock, "_is_owned") else lock.locked())
    original_del(self)
hs.SessionDB.__del__ = recording_del
target = weakref.ref(db)
holder = [db]
del db
dropped, done = threading.Event(), threading.Event()

def trace(frame, event, arg):
    # Once the observer's own frame holds the handle, drop the only other reference: the observer's
    # reference is now the last one, and the finalizer runs wherever the observer lets go of it.
    if holder and event == "line" and frame.f_code.co_filename == hs.__file__:
        if frame.f_locals.get("db") is holder[0]:
            holder.clear()
            dropped.set()
    return trace

def observe():
    try:
        if mode == "verdict":
            from gateway.run_shutdown_budget import stores_closed
            sys.settrace(trace)
            stores_closed(0.0, [])
        elif mode == "held-lock":
            # A finalizer that runs on a thread already inside the lock (a collection while the
            # observer or a marker holds it) must not block on the same lock.
            with hs._unsettled_closes_lock:
                holder.clear()
                dropped.set()
        else:
            raise ValueError(mode)
    finally:
        sys.settrace(None)
    done.set()

thread = threading.Thread(target=observe, daemon=True)
thread.start()
out["dropped"] = dropped.wait(30)
out["returned"] = done.wait(30)
out["finalized"] = target() is None
out["finalizer_under_lock"] = seen.get("finalizer_under_lock")
lock_free = hs._unsettled_closes_lock.acquire(timeout=5)
if lock_free:
    hs._unsettled_closes_lock.release()
out["lock_free"] = lock_free
out["after"] = hs.unsettled_closes() if lock_free else None
from gateway.runtime_ownership import process_ownership
out["retention"] = process_ownership._retained_for
print("OBSERVATION " + json.dumps(out), flush=True)
os._exit(0)  # a deadlocked daemon thread must not hold this process at interpreter exit
'''


def _scenario(mode, tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    env = {**os.environ, 'HOME': str(home), 'HERMES_HOME': str(home / 'hermes'), 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': os.pathsep.join([str(_ROOT), os.environ.get('PYTHONPATH', '')])}
    db_path = tmp_path / 'unsettled.db'
    proc = subprocess.run([sys.executable, '-c', _SCENARIO, mode, str(db_path)], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=180, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.startswith('OBSERVATION ')]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
    observed = json.loads(lines[-1][len('OBSERVATION '):])
    print('OBSERVATION', mode, observed)
    (tmp_path / f'observation-{mode}.json').write_text(json.dumps(observed))  # retained basetemp
    return str(db_path), observed


def test_the_close_verdict_finishes_when_it_holds_the_last_reference(tmp_path):
    """The stop's close verdict reads the set while the handle's other owner lets go: the verdict
    returns, retains for the handle it saw, the finalizer runs, and the lock is free afterwards."""
    db_path, observed = _scenario('verdict', tmp_path)
    assert observed['before'] == [db_path], observed
    assert observed['dropped'], ('the schedule did not reach the observer holding the handle', observed)
    assert observed['returned'], ('the close verdict did not return: a finalizer re-entered its lock', observed)
    assert observed['lock_free'], observed
    assert observed['finalized'], observed
    assert observed['finalizer_under_lock'] is False, ('the observer finalized the handle under its lock', observed)
    assert observed['retention'] == f'SessionDB close did not finish: {db_path}', observed
    assert observed['after'] == [], observed  # a handle that no longer exists is no longer reported


def test_a_finalizer_inside_the_lock_does_not_block(tmp_path):
    """The last reference to an unfinished handle drops on a thread that is inside the lock: its
    ``__del__`` -> ``close()`` -> marker must not wait for the lock this thread already holds."""
    db_path, observed = _scenario('held-lock', tmp_path)
    assert observed['before'] == [db_path], observed
    assert observed['dropped'], observed
    assert observed['returned'], ('a finalizer blocked on the lock its own thread holds', observed)
    assert observed['lock_free'] and observed['finalized'], observed
    assert observed['finalizer_under_lock'] is True, observed  # the schedule this case is about
    assert observed['after'] == [], observed


def _unfinished_close(db):
    """The real registry quiet-close of *db* while its retired-generation capture fails (injected)."""
    import hermes_state_registry as reg
    from hermes_state_dbfile import RetiredGenerationCaptureError

    def failed_capture():
        raise RetiredGenerationCaptureError('injected capture failure')
    db._db_wal_generation_lost = True
    db._settle_lost_generation_locked = failed_capture
    reg.release_or_close(db)


def _finished_close(db):
    del db._settle_lost_generation_locked
    db._db_wal_generation_lost = False
    db.close()


def test_snapshot_controls_empty_nonempty_and_settled(tmp_path, monkeypatch):
    """Ordinary use: nothing unfinished; one unfinished close (reported); a later close of the same
    handle that finishes removes it; a close that finished at once is never reported."""
    import hermes_state as hs
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    (tmp_path / 'store').mkdir()
    (tmp_path / 'other').mkdir()
    db_path, other_path = str(tmp_path / 'store' / 'state.db'), str(tmp_path / 'other' / 'state.db')

    def mine():
        return [p for p in hs.unsettled_closes() if p.startswith(str(tmp_path))]
    observed = {'empty': mine()}
    clean = hs.SessionDB(Path(other_path))
    clean.close()
    observed['clean'] = mine()
    db = hs.SessionDB(Path(db_path))
    _unfinished_close(db)
    observed.update(nonempty=mine(), conn_open=db._conn is not None)
    _finished_close(db)
    observed.update(after_close=mine(), conn_after=db._conn is not None)
    print('OBSERVATION', observed)
    assert observed['empty'] == [] and observed['clean'] == [], observed
    assert observed['nonempty'] == [db_path] and observed['conn_open'], observed
    assert observed['after_close'] == [] and not observed['conn_after'], observed


def test_snapshot_filtered_by_directory(tmp_path, monkeypatch):
    """``unsettled_closes(under)`` (the per-profile release's question) names only stores in *under*."""
    import hermes_state as hs
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    store, sibling, other = tmp_path / 'p' / 'alpha', tmp_path / 'p' / 'alphabet', tmp_path / 'p' / 'beta'
    dbs = {}
    for home in (store, sibling, other):
        home.mkdir(parents=True)
        dbs[home] = hs.SessionDB(home / 'state.db')
    _unfinished_close(dbs[store])
    _unfinished_close(dbs[sibling])
    observed = {'alpha': hs.unsettled_closes(store), 'beta': hs.unsettled_closes(other),
                'parent': hs.unsettled_closes(tmp_path / 'p')}
    for home in (store, sibling):
        _finished_close(dbs[home])
    dbs[other].close()
    observed['after'] = hs.unsettled_closes(tmp_path / 'p')
    print('OBSERVATION', observed)
    assert observed['alpha'] == [str(store / 'state.db')], observed  # not the 'alphabet' sibling
    assert observed['beta'] == [], observed
    assert observed['parent'] == sorted([str(store / 'state.db'), str(sibling / 'state.db')]), observed
    assert observed['after'] == [], observed
