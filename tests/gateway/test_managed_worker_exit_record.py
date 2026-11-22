"""The owner durably records a finished managed worker's lifecycle with the admission's settled result
(``managed_exit``): the actual ``process.wait`` returncode of the reserved interpreter, and the result/ACK/finish/
exit times, correlated by the Kanban card/run (request identity), admission, generation and execution.
An exit the owner did not observe (wait timeout, a launcher handle) stays explicitly ``unknown`` and is never
inferred from the turn's result. Real child process, real SQLite authority and settlement write."""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionActor, SessionRef

REQUEST = 'kanban:' + json.dumps(['theseus-active', 't_exitrec', 42], separators=(',', ':'))

CHILD = """import json, os, signal, sys, time, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]})
shape = sys.argv[1]
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if 'version' in frame:
        if shape == 'killed-before-result':
            os.kill(os.getpid(), signal.SIGKILL)
        send({'type': 'result', 'result': {'final_response': 'EXIT_RECORD_ANSWER', 'model': 'm'}})
    elif frame == {'type': 'finish'}:
        time.sleep(0.05)  # measurable ACK -> finished interval
        send({'type': 'finished'})
        break
out.close()
if shape == 'clean':
    sys.exit(0)
if shape == 'nonzero':
    os._exit(3)
if shape == 'killed-after-finished':
    os.kill(os.getpid(), signal.SIGKILL)
time.sleep(60)  # 'lingering': outlives the owner's 10 s exit wait
"""

# A launcher handle: the Popen child starts the real interpreter as its own child and exits with its code,
# so the handle's returncode is the launcher's, never the reserved interpreter's.
LAUNCHER = """import subprocess, sys
sys.exit(subprocess.call([sys.executable, '-c', sys.argv[1], 'clean']))
"""


def _setup(tmp_path, monkeypatch, shape):
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    spawned = []

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            argv = ([sys.executable, '-c', LAUNCHER, CHILD] if shape == 'launcher'
                    else [sys.executable, '-c', CHILD, shape])
            super().__init__(argv, **kw)
            spawned.append(self)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    return managed, db, authority, spawned


def _run(managed, db, authority, tmp_path, expect_unknown=False):
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state_runtime import claim_session_input
    actor = SessionActor('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission(REQUEST, ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, authority.sessions['s'], row)
        async with asyncio.timeout(40):
            if expect_unknown:
                with pytest.raises(managed.ManagedExecutionUnknown):
                    await managed.execute_managed(authority, ref, row, policy)
                return row, None
            return row, await managed.execute_managed(authority, ref, row, policy)
    return asyncio.run(scenario())


def _settle_and_read(db, authority, row, response):
    """The ordinary settlement write, then the stored row read back from state.db."""
    from gateway.session_results import admission_result, finish_result
    finish_result(db, epoch=authority.epoch, row=row, response=response, outcome='completed',
                  result=authority.pending_results[row['admission_id']])
    return admission_result(db, row['admission_id'])


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(('shape', 'expected'), [
    ('clean', {'status': 'exited', 'returncode': 0}),
    ('nonzero', {'status': 'exited', 'returncode': 3}),
    ('killed-after-finished', {'status': 'signalled', 'returncode': -9, 'signal': 'SIGKILL'}),
    ('lingering', {'status': 'unknown', 'returncode': None, 'reason': 'wait_timeout'}),
    ('launcher', {'status': 'unknown', 'returncode': None, 'reason': 'handle_is_not_the_worker'}),
])
def test_finished_worker_exit_is_recorded_durably_and_only_as_observed(tmp_path, monkeypatch, shape, expected):
    managed, db, authority, spawned = _setup(tmp_path, monkeypatch, shape)
    with db:
        row, response = _run(managed, db, authority, tmp_path)
        assert response == 'EXIT_RECORD_ANSWER'
        stored = _settle_and_read(db, authority, row, response)
        status = db._read_one('SELECT status, outcome FROM session_admissions WHERE admission_id=?',
                              (row['admission_id'],))
    assert tuple(status) == ('terminal', 'completed')
    record = stored['managed_exit']
    assert stored['result']['final_response'] == 'EXIT_RECORD_ANSWER'
    # Correlation: card/run from the request identity, admission, generation, execution, owner epoch.
    assert record['request_id'] == REQUEST
    assert record['kanban'] == {'board': 'theseus-active', 'task_id': 't_exitrec', 'run_id': 42}
    assert (record['admission_id'], record['generation'], record['owner_epoch']) == (
        row['admission_id'], row['generation'], authority.epoch)
    assert record['execution_id'] == 'admission-worker:' + row['admission_id']
    assert record['handle_pid'] == spawned[0].pid
    assert (record['worker_pid'] == record['handle_pid']) is (shape != 'launcher')
    # The exit is exactly what the owner observed, never inferred from the completed turn.
    assert {k: record['exit'].get(k) for k in expected} == expected, record
    # Ordering of the owner's own observations: result accepted -> ACK sent -> finished seen -> exit seen.
    assert record['result_received_at'] <= record['ack_sent_at'] <= record['finish_observed_at']
    assert record['finish_observed_at'] - record['ack_sent_at'] >= 0.04
    if record['exit']['observed_at'] is not None:
        assert record['exit']['observed_at'] >= record['finish_observed_at']
    else:
        assert record['exit']['status'] == 'unknown'
    if shape == 'launcher':
        assert record['exit']['handle_returncode'] == 0 and record['exit']['handle_observed_at'] is not None
    assert spawned[0].poll() is not None, 'close() did not reap the worker'


@pytest.mark.platforms("linux")
def test_control_a_worker_killed_before_its_result_records_no_exit_and_stays_unknown(tmp_path, monkeypatch):
    """Failure/kill path control: the loss path is unchanged. The admission is fenced ``unknown``, nothing
    settles, and no lifecycle record (least of all an exit 0) is produced for the lost worker."""
    managed, db, authority, spawned = _setup(tmp_path, monkeypatch, 'killed-before-result')
    with db:
        row, _ = _run(managed, db, authority, tmp_path, expect_unknown=True)
        status = db._read_one('SELECT status FROM session_admissions WHERE admission_id=?', (row['admission_id'],))[0]
        from hermes_state_terminal import RESULT_PREFIX as _RESULT_PREFIX
        stored = db._read_one('SELECT value FROM state_meta WHERE key=?', (_RESULT_PREFIX + row['admission_id'],))
    assert status == 'unknown'
    assert 'managed_exit' not in (authority.pending_results.get(row['admission_id']) or {})
    assert stored is None or 'managed_exit' not in stored[0]
    assert spawned[0].poll() == -9


def test_kanban_attempt_is_read_from_the_request_identity_only():
    from gateway.session_managed_worker import _kanban_attempt
    assert _kanban_attempt(REQUEST) == {'board': 'theseus-active', 'task_id': 't_exitrec', 'run_id': 42}
    for other in (None, 'turn', 'kanban:', 'kanban:{"a": 1, "b": 2, "c": 3}', 'kanban:[1,2]', 'kanban:not-json'):
        assert _kanban_attempt(other) is None
