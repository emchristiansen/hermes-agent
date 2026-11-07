"""The at-most-one-grace ledger guard (``kanban_presubmit_grace``), case by case, on disposable boards.

A card's pre-submission owner-restart grace is spent by one ``owner_restart_grace`` event, recorded
only in the board transaction that finds none since the card's last ACTUAL admission: an
``owner_admitted`` marker whose admission the owner's queue accepted. These cases cover duplicate and
concurrent publication, a failed transaction, stale attempts, unreadable records, the consumer's
claim check, and which admissions reopen the window. The aged-GC-and-reopen lifecycle runs end to
end in ``test_kanban_presubmit_owner_restart``.

Real: the board, its write transactions and the grace module. In the last test, also the owner's own
admission (``gateway.session_kanban.run_task``) on a disposable SessionStore, as in
``tests/gateway/test_kanban_finite_admission.py``. Scripted: owner incarnations are plain tuples
(``record_grace`` does not inspect them), owner ledgers are hand-built SQLite files except in the last
test, and one test injects a failure after the event append.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

GRACE_EVENT = 'owner_restart_grace'
OLD, NEW = (101, 1000), (102, 2000)


class _Board:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
        monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'kanban'))
        (tmp_path / 'profile').mkdir()
        self.tmp = tmp_path
        kb.init_db()
        self.conn = kbc.connect()
        self.db = self.conn.execute('PRAGMA database_list').fetchone()[2]
        self.tid = kb.create_task(self.conn, title='card', assignee='w')
        self.params = self.claim()

    def claim(self):
        task = kb.claim_task(self.conn, self.tid)
        assert task is not None
        return dict(board='default', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)

    def next_run(self):
        assert kb.reclaim_task(self.conn, self.tid, reason='test')
        self.params = self.claim()
        return self.params

    def grace(self, params=None):
        from hermes_cli.kanban_presubmit_grace import record_grace
        return record_grace(self.db, params or self.params, OLD, NEW, self.tmp)

    def graces(self):
        return [r[0] for r in self.conn.execute('SELECT run_id FROM task_events WHERE task_id=? AND kind=? ORDER BY id',
                                                (self.tid, GRACE_EVENT))]

    def event(self, kind, payload, run_id):
        with kb.write_txn(self.conn):
            self.conn.execute('INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,0)',
                              (self.tid, run_id, kind, payload if isinstance(payload, str) else json.dumps(payload)))

    def ledger(self, name, *, live=(), retired=()):
        """A hand-built owner ledger holding ``live`` and ``retired`` ``(session_id, request_id)`` admissions."""
        from hermes_state_terminal import ADMISSION_PREFIX
        path = self.tmp / name
        path.unlink(missing_ok=True)
        with closing(sqlite3.connect(path)) as ledger, ledger:
            ledger.execute('CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, status TEXT)')
            ledger.execute('CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)')
            for sid, request in live:
                ledger.execute("INSERT INTO session_admissions VALUES (?,?,'terminal')", (sid, request))
            for i, (sid, request) in enumerate(retired):
                ledger.execute('INSERT INTO state_meta VALUES (?,?)', (ADMISSION_PREFIX + f'a{i}', json.dumps(
                    {'admission_id': f'a{i}', 'target_session_id': sid, 'request_id': request, 'status': 'terminal'})))
        return str(path)


@pytest.fixture
def board(tmp_path, monkeypatch):
    b = _Board(tmp_path, monkeypatch)
    yield b
    b.conn.close()


def test_a_duplicate_publication_records_one_grace(board):
    assert [board.grace(), board.grace()] == [True, False]
    assert board.graces() == [board.params['run_id']]


def test_concurrent_publications_record_one_grace(board):
    """Eight submitters of the same attempt race on separate connections: one wins."""
    barrier, results = threading.Barrier(8), []

    def publish():
        barrier.wait()
        results.append(board.grace())
    threads = [threading.Thread(target=publish) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert sorted(results) == [False] * 7 + [True]
    assert board.graces() == [board.params['run_id']]


@pytest.mark.parametrize('failure', ['fails-after-the-append', 'board-locked'])
def test_a_failed_transaction_spends_nothing(board, monkeypatch, failure):
    if failure == 'fails-after-the-append':
        append = kb._append_event

        def append_then_fail(*args, **kwargs):
            append(*args, **kwargs)
            raise RuntimeError('injected after the grace append')
        monkeypatch.setattr(kb, '_append_event', append_then_fail)
        assert board.grace() is False
        monkeypatch.setattr(kb, '_append_event', append)
    else:
        monkeypatch.setenv('HERMES_KANBAN_BUSY_TIMEOUT_MS', '50')
        monkeypatch.setattr(kbc, '_BUSY_MAX_RETRIES', 0)
        with closing(sqlite3.connect(board.db, isolation_level=None)) as writer:
            writer.execute('BEGIN IMMEDIATE')  # another writer holds the board
            assert board.grace() is False
            writer.execute('ROLLBACK')
    assert board.graces() == []  # rolled back: the grace is still unspent
    assert board.grace() is True and len(board.graces()) == 1


def _block(board, params):
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (board.tid,))
    return params


# Each stale or malformed attempt, as a change to the submitter's ``params`` (and the board).
_STALE = {
    'other-run': lambda board, p: dict(p, run_id=p['run_id'] + 1),
    'other-claim': lambda board, p: dict(p, claim_lock='elsewhere:1'),
    'run-id-text': lambda board, p: dict(p, run_id=str(p['run_id'])),
    'run-id-bool': lambda board, p: dict(p, run_id=True),
    'not-running': _block,
    'unknown-task': lambda board, p: dict(p, task_id='t_00000000'),
    'missing-field': lambda board, p: {k: v for k, v in p.items() if k != 'claim_lock'},
}


@pytest.mark.parametrize('stale', [*_STALE, 'missing-board'])
def test_a_stale_or_malformed_attempt_spends_nothing(board, stale):
    from hermes_cli.kanban_presubmit_grace import record_grace
    params = _STALE.get(stale, lambda board, p: p)(board, dict(board.params))
    db = str(board.tmp / 'no-such-board.db') if stale == 'missing-board' else board.db
    assert record_grace(db, params, OLD, NEW, board.tmp) is False
    assert board.graces() == []


def _live():
    from gateway.status import get_process_start_time
    return os.getpid(), get_process_start_time(os.getpid())


@pytest.mark.parametrize('record', ['garbage', 'list', 'pid-bool', 'pid-text', 'pid-zero', 'start-float',
                                    'start-none', 'start-text', 'other-home', 'home-not-text', 'home-null',
                                    'home-empty'])
def test_an_unreadable_owner_record_is_no_evidence(board, record):
    from hermes_cli.kanban_presubmit_grace import _recorded_owner, live_owner, replacement
    home = board.tmp / 'profile'
    pid, start = _live()
    good = {'pid': pid, 'start_time': start, 'hermes_home': str(home.resolve())}
    (home / 'gateway_state.json').write_text(json.dumps(good))
    assert live_owner(home) == (pid, start)  # the control: a well-formed record of a live owner
    bad = {'garbage': '{not json', 'list': json.dumps([good]), 'pid-bool': json.dumps(dict(good, pid=True)),
           'pid-text': json.dumps(dict(good, pid=str(pid))), 'pid-zero': json.dumps(dict(good, pid=0)),
           'start-float': json.dumps(dict(good, start_time=float(start))),
           'start-none': json.dumps(dict(good, start_time=None)),
           'start-text': json.dumps(dict(good, start_time=str(start))),
           'other-home': json.dumps(dict(good, hermes_home=str(board.tmp / 'elsewhere'))),
           'home-not-text': json.dumps(dict(good, hermes_home=7)),
           'home-null': json.dumps(dict(good, hermes_home=None)),
           'home-empty': json.dumps(dict(good, hermes_home=''))}[record]
    (home / 'gateway_state.json').write_text(bad)
    assert _recorded_owner(home) is None and live_owner(home) is None
    assert replacement(OLD, home) is None and replacement(None, home) is None


def test_a_live_baseline_is_never_replaced(board):
    from hermes_cli.kanban_presubmit_grace import replacement
    home = board.tmp / 'profile'
    (home / 'gateway_state.json').write_text(json.dumps({'pid': 1, 'start_time': 5, 'hermes_home': str(home)}))
    assert replacement(_live(), home) is None  # the baseline still runs: no restart happened


def test_a_spent_grace_with_an_unreadable_payload_still_counts(board):
    board.event(GRACE_EVENT, '{not json', board.params['run_id'])
    board.next_run()
    assert board.grace() is False
    assert len(board.graces()) == 1


def test_the_consumer_honours_only_the_current_runs_grace_under_its_claim(board):
    from hermes_cli.kanban_presubmit_grace import grace_backs_run
    run, claim = board.params['run_id'], board.params['claim_lock']

    def backs(lock=claim):
        return grace_backs_run(board.conn, board.tid, lock)
    assert not backs()
    board.event(GRACE_EVENT, {'claim_lock': 'elsewhere:1'}, run)
    board.event(GRACE_EVENT, '{not json', run)
    board.event(GRACE_EVENT, ['claim_lock', claim], run)
    assert not backs()  # another claim's or an unreadable grace backs nothing
    board.event(GRACE_EVENT, {'claim_lock': claim}, run)
    assert backs() and not backs('nobody:2')
    nxt = board.next_run()
    assert not grace_backs_run(board.conn, board.tid, nxt['claim_lock'])  # a previous run's grace
    assert not grace_backs_run(board.conn, board.tid, claim)
    board.event('owner_admitted', {'session_id': 's'}, nxt['run_id'])
    assert grace_backs_run(board.conn, board.tid, nxt['claim_lock'])  # the owner's own accounting decides


@pytest.mark.parametrize('admission, reopens', [
    ('accepted', True), ('accepted-then-retired', True), ('marker-only', False), ('other-request', False),
    ('ledger-missing', False), ('payload-unreadable', False), ('payload-fields-missing', False)])
def test_only_an_accepted_admission_reopens_the_window(board, admission, reopens):
    """grace on run 1; run 2 carries an ``owner_admitted`` marker; run 3 asks again."""
    assert board.grace() is True
    run2 = board.next_run()['run_id']
    ledgers = {  # the marker's owner ledger, per case
        'accepted': lambda: board.ledger('owner.db', live=[('s', 'r')]),
        'accepted-then-retired': lambda: board.ledger('owner.db', retired=[('s', 'r')]),
        'marker-only': lambda: board.ledger('owner.db'),
        'other-request': lambda: board.ledger('owner.db', live=[('s', 'r-other')]),
        'ledger-missing': lambda: str(board.tmp / 'gone.db'),
    }
    marker = {'db': ledgers.get(admission, ledgers['accepted'])(), 'session_id': 's', 'request_id': 'r', 'pid': 1,
              'birth': 0}
    if admission == 'payload-fields-missing':
        del marker['request_id']
    board.event('owner_admitted', '{not json' if admission == 'payload-unreadable' else marker, run2)
    assert board.grace() is False  # a run that carries a marker is never a pre-submission attempt
    board.next_run()
    assert board.grace() is reopens
    assert len(board.graces()) == (2 if reopens else 1)


def test_an_older_accepted_admission_does_not_reopen_past_a_newer_spent_grace(board):
    """Window = since the newest ACCEPTED admission: an unaccepted newer marker leaves the older window."""
    run1 = board.params['run_id']
    board.event('owner_admitted', {'db': board.ledger('owner.db', live=[('s', 'r1')]), 'session_id': 's',
                                   'request_id': 'r1'}, run1)
    board.next_run()
    assert board.grace() is True  # first grace since the run-1 admission
    run3 = board.next_run()['run_id']
    board.event('owner_admitted', {'db': board.ledger('owner.db', live=[('s', 'r1')]), 'session_id': 's',
                                   'request_id': 'r3'}, run3)  # a marker over nothing
    board.next_run()
    assert board.grace() is False
    assert len(board.graces()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('submit', ['fails-after-the-marker', 'accepted'])
async def test_the_real_owner_marker_reopens_the_window_only_with_queue_acceptance(tmp_path, monkeypatch, submit):
    """The owner's actual entry point, ``run_task``: its board transaction commits ``owner_admitted``, THEN
    ``authority.submit`` queues the turn. A submit that fails in between leaves a real marker over no
    admission, and the grace window stays closed. Disposable owner as in
    ``tests/gateway/test_kanban_finite_admission.py`` (no FIFO schedule: nothing is executed)."""
    from gateway import run
    from gateway.session_controls import AuthorityConnection
    from gateway.session_kanban import run_task
    from hermes_cli.kanban_presubmit_grace import accepted, record_grace
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_session_follow_profile_config import _owner

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    for name in ('HERMES_KANBAN_DB', 'HERMES_DELEGATED_CHILD', 'HERMES_SINGLE_QUERY_SESSION', 'HERMES_YOLO_MODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps(
        {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}))
    kb.create_board('owned')
    conn = kbc.connect(board='owned')
    db = conn.execute('PRAGMA database_list').fetchone()[2]
    tid = kb.create_task(conn, title='card', assignee='default', workspace_kind='dir', workspace_path=str(tmp_path))
    kb.recompute_ready(conn)

    def claim():
        task = kb.claim_task(conn, tid)
        return dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)

    def again():
        assert kb.reclaim_task(conn, tid, reason='test')
        return claim()

    store, _runner, authority, dashboard = _owner(tmp_path)
    authority.profile_id = str(tmp_path)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})

    async def owner_admits(params):
        if submit == 'fails-after-the-marker':
            async def submit_fails(*args, **kwargs):
                raise RuntimeError('owner ended between the marker and the queue')
            monkeypatch.setattr(authority, 'submit', submit_fails)
            with pytest.raises(RuntimeError, match='between the marker'):
                await run_task(connection, params)
            return None
        return (await run_task(connection, params))['session_id']

    try:
        assert record_grace(db, claim(), OLD, NEW, tmp_path) is True
        admitted = again()
        sid = await owner_admits(admitted)
        payload = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
                               (tid, admitted['run_id'])).fetchone()[0]  # the marker committed either way
        binding = json.loads(payload)
        rows = list_session_admissions(authority.db, session_id=binding['session_id'], pending_only=False)
        assert [r['request_id'] for r in rows] == ([] if sid is None else [binding['request_id']])
        assert accepted(payload) is (sid is not None)
        later = again()
        assert record_grace(db, later, OLD, NEW, tmp_path) is (sid is not None)
        spent = conn.execute('SELECT run_id FROM task_events WHERE task_id=? AND kind=? ORDER BY id',
                             (tid, GRACE_EVENT)).fetchall()
        assert len(spent) == (1 if sid is None else 2)
    finally:
        await connection.close()
        await dashboard.close()
        store.close_all_db_handles()
        conn.close()
