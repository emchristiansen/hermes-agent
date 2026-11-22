"""A card attempt whose owner restarted before anything was submitted gets ONE uncounted grace
until the owner next actually admits an attempt of that card.

The submitter (``kanban_worker_client``) waits ``OWNER_WAIT_S`` for an owner that is ``draining`` or
``starting``. On 208e122b, once that wait expires it always exits 1 and the crash sweep counts a
crash, even when the owner's own restart outlasted the wait and nothing was submitted. With the
grace (``kanban_presubmit_grace``):

* The control home's ``gateway_state.json`` must show that the owner incarnation live when the
  attempt started was replaced.
* The card's grace must be unspent since its last ``owner_admitted``.

When both hold, the submitter records the grace on its exact run and exits 79, and the crash sweep
books ``owner_restarted`` only for a run that carries that grace.

What is real and what is scripted (no live gateway, service, credential or replay; no stress):

* Real: ``kanban_worker_client.main`` runs as a subprocess spawned by the real ``dispatch_once``, with
  its stderr in the task's worker log. The real crash sweep reads that log's trailer. The board is a
  disposable SQLite board. Owner incarnations are real sleeping child processes.
* Scripted inside the submitter subprocess: ``connect_gateway`` (as in ``test_kanban_worker_owner_wait``),
  ``OWNER_WAIT_S`` (shortened), the owner's "restart" (it kills the old incarnation and rewrites the
  scratch ``gateway_state.json``), and, where a mode says so, the RPC answers. The ``admitted`` mode
  writes the ``owner_admitted`` marker straight into the board: it models only the marker window.
* Real owner admission: ``test_real_owner_admission_reopens_the_grace_after_a_counted_failure``
  admits through the owner's own entry point and transaction (``gateway.session_kanban.run_task``)
  on a disposable SessionStore, with nothing executed (the FIFO schedule is a no-op).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from gateway.status import get_process_start_time
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# Linux-only: the module starts real processes, kills them with signal.SIGKILL (owner replacement
# and cleanup) and reads /proc/<pid>/status to wait for the killed owner to become a zombie.
pytestmark = pytest.mark.platforms("linux")

REPO = Path(__file__).resolve().parents[2]
FAILURE_LIMIT = 3
GRACE_EVENT = 'owner_restart_grace'  # the board event that spends a card's grace

# The submitter, unchanged, with its owner connection scripted by ``PRESUBMIT_SCENARIO``.
_SUBMITTER = r'''
import asyncio, contextlib, json, os, signal, sqlite3, sys, time
sys.path.insert(0, sys.argv[1])
scenario = json.loads(os.environ['PRESUBMIT_SCENARIO'])
from hermes_cli import gateway_client, kanban_worker_client as worker
worker.OWNER_WAIT_S = scenario.get('wait', 0.3)
state = {'replaced': False}

def replace_owner():
    """The owner restarts while this attempt waits: the old incarnation ends, a new one records itself."""
    swap = scenario.get('replace')
    if not swap or state['replaced']:
        return
    state['replaced'] = True
    try:
        os.kill(swap['old'], signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone
    for _ in range(200):
        try:
            with open(f"/proc/{swap['old']}/status") as status:
                if any(line.startswith('State:') and 'Z' in line for line in status):
                    break
        except FileNotFoundError:
            break
        time.sleep(0.01)
    with open(swap['record'], 'w') as record:
        json.dump(swap['new_record'], record)

def sent(method):
    with open(scenario['sent'], 'a') as log:
        log.write(method + '\n')

class Client:
    def __init__(self):
        self.events = asyncio.Queue()
    async def rpc(self, method, **params):
        sent(method)
        mode = scenario['mode']
        if mode == 'lost':
            raise gateway_client.GatewayClientError('Gateway disconnected; turn outcome is unknown.')
        if mode == 'lost_unavailable':
            raise gateway_client.GatewayUnavailableError('Gateway draining: not_ready', 'draining')
        if method == 'kanban.run':  # 'admitted': the owner's admission marker, as admit_claim writes it
            with sqlite3.connect(params['db']) as board:
                board.execute("INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
                    (params['task_id'], params['run_id'], 'owner_admitted', json.dumps({
                        'db': scenario['ledger'], 'session_id': 's', 'request_id': 'r',
                        'pid': os.getpid(), 'birth': 0}), int(time.time())))
            return {'session_id': 's', 'receipt': {'status': 'terminal', 'admission_id': 'a'}}
        return {'status': 'terminal', 'admission_id': 'a'}

@contextlib.asynccontextmanager
async def connect(ready_timeout=None):
    with open(scenario['connects'], 'a') as log:
        log.write('connect\n')
    replace_owner()
    if scenario['mode'] in ('unavailable', 'refused'):
        state_name = 'inaccessible' if scenario['mode'] == 'refused' else scenario.get('state', 'starting')
        raise gateway_client.GatewayUnavailableError(f'Gateway {state_name}: not_ready', state_name)
    yield Client()

gateway_client.connect_gateway = connect
if scenario['mode'] == 'stale79':  # a 79 trailer nothing recorded: e.g. left by an earlier run
    print('\n[kanban-worker-exit] rc=79', file=sys.stderr, flush=True)
    sys.exit(79)
if scenario['mode'] == 'standin':  # stands in for a submitter whose admitted attempt then failed
    print('\n[kanban-worker-exit] rc=1', file=sys.stderr, flush=True)
    sys.exit(1)
sys.exit(worker.main())
'''


class _Host:
    """A disposable board, profile home and owner incarnations; spawns real submitters."""

    def __init__(self, tmp_path, monkeypatch):
        self.home = tmp_path / 'profile'
        self.home.mkdir()
        monkeypatch.setenv('HERMES_HOME', str(self.home))
        monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'kanban'))
        monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
        from hermes_cli import profiles
        monkeypatch.setattr(profiles, 'profile_exists', lambda name: True)
        kbd._recent_worker_exits.clear()
        kb.init_db()
        self.conn = kbc.connect()
        self.db = self.conn.execute('PRAGMA database_list').fetchone()[2]
        self.tid = kb.create_task(self.conn, title='card', assignee='w')
        self.record = self.home / 'gateway_state.json'
        self.sent = tmp_path / 'sent.log'
        self.connects = tmp_path / 'connects.log'
        self.ledger = tmp_path / 'owner-state.db'
        with sqlite3.connect(self.ledger) as ledger:  # the admitting owner's ledger: settled, not paused
            ledger.execute('CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)')
            ledger.execute("INSERT INTO session_admissions VALUES ('s', 'r', 'terminal', 'failed')")
        self.owners: list[subprocess.Popen] = []
        self.submitters: list[subprocess.Popen] = []
        self.scenario: dict = {}

    def owner(self):
        """A live owner incarnation: a real process and its ``(pid, start_time)``."""
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
        self.owners.append(proc)
        for _ in range(100):
            start = get_process_start_time(proc.pid)
            if start is not None:
                return {'pid': proc.pid, 'start_time': start, 'hermes_home': str(self.home.resolve()),
                        'gateway_state': 'draining', 'kind': 'hermes-gateway'}
        raise AssertionError('no start time for the owner process')

    def record_owner(self, record):
        self.record.write_text(json.dumps(record))

    def replacing(self, old, new):
        return {'old': old['pid'], 'record': str(self.record), 'new_record': new}

    def attempt(self, mode, **extra):
        """One dispatcher tick (it books the previous attempt and claims the next) spawning one real
        submitter, which runs to its exit. Returns ``(tick result, submitter exit, run id)``."""
        self.scenario = dict(mode=mode, sent=str(self.sent), connects=str(self.connects), ledger=str(self.ledger),
                             **extra)
        res = kbd.dispatch_once(self.conn, spawn_fn=self._spawn, failure_limit=FAILURE_LIMIT)
        if not self.submitters or self.submitters[-1].returncode is not None:
            return res, None, None
        proc = self.submitters[-1]
        proc.wait(timeout=120)
        return res, proc.returncode, kb.get_task(self.conn, self.tid).current_run_id

    def _spawn(self, task, workspace, board=None):
        env = dict(os.environ, PRESUBMIT_SCENARIO=json.dumps(self.scenario),
                   HERMES_KANBAN_TASK=task.id, HERMES_KANBAN_RUN_ID=str(task.current_run_id),
                   HERMES_KANBAN_CLAIM_LOCK=task.claim_lock, HERMES_KANBAN_DB=self.db,
                   HERMES_KANBAN_BOARD='default')
        env.pop('HERMES_TUI_GATEWAY_URL', None)
        log = kb.worker_log_path(task.id, board=board)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'a') as out:
            proc = subprocess.Popen([sys.executable, '-c', _SUBMITTER, str(REPO)], env=env, cwd=REPO,
                                    stdout=out, stderr=out)
        self.submitters.append(proc)
        return proc.pid

    def settle(self):
        """The tick after the last attempt: books it, claims nothing more."""
        return kbd.dispatch_once(self.conn, spawn_fn=lambda *a, **k: None, failure_limit=FAILURE_LIMIT, max_spawn=0)

    def outcomes(self):
        return [r[0] for r in self.conn.execute('SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id', (self.tid,))]

    def graces(self):
        return [r[0] for r in self.conn.execute('SELECT run_id FROM task_events WHERE task_id=? AND kind=? ORDER BY id',
                                                (self.tid, GRACE_EVENT))]

    def task(self):
        return kb.get_task(self.conn, self.tid)

    def close(self):
        for proc in self.submitters + self.owners:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=30)
        self.conn.close()


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    h = _Host(tmp_path, monkeypatch)
    yield h
    h.close()


def test_owner_replaced_before_submission_is_one_uncounted_restart(host):
    old, new = host.owner(), host.owner()
    host.record_owner(old)
    _res, rc, run = host.attempt('unavailable', replace=host.replacing(old, new), wait=3.0)
    assert rc == 79
    res = host.settle()
    assert res.owner_restarted == [host.tid] and not res.crashed
    assert host.outcomes() == ['owner_restarted']
    assert (host.task().status, host.task().consecutive_failures) == ('ready', 0)
    grace = json.loads(host.conn.execute('SELECT payload FROM task_events WHERE kind=?', (GRACE_EVENT,)).fetchone()[0])
    assert host.graces() == [run]
    assert (grace['old_owner']['pid'], grace['new_owner']['pid']) == (old['pid'], new['pid'])
    assert not host.sent.exists()  # nothing was ever submitted
    assert len(host.connects.read_text().split()) >= 2  # the ordinary pre-connection retry still runs


def test_a_presubmission_grace_records_that_nothing_was_sent(host):
    """COMPOSE-ACCOUNT-N1: the run's persisted error says what happened. The owner was replaced before
    anything was sent, so no turn exists that could have been "settled interrupted". The booking,
    counters and grace are exactly those of ``test_owner_replaced_before_submission_is_one_uncounted_restart``,
    and the submitter's own words are kept."""
    old, new = host.owner(), host.owner()
    host.record_owner(old)
    _res, rc, run = host.attempt('unavailable', replace=host.replacing(old, new), wait=3.0)
    assert rc == 79
    host.settle()
    outcome, status, error = host.conn.execute('SELECT outcome, status, error FROM task_runs WHERE id=?', (run,)).fetchone()
    assert (outcome, status) == ('owner_restarted', 'owner_restarted')
    assert (host.task().status, host.task().consecutive_failures) == ('ready', 0) and host.graces() == [run]
    assert not host.sent.exists()  # nothing was ever submitted
    assert 'settled interrupted' not in error
    assert "owner was replaced before the attempt submitted its turn; nothing was sent" in error
    assert 'Kanban attempt not submitted' in error  # the submitter's stderr, as the classifier appends it


_DROP = object()  # removes the key from an owner record
_GRANTED = (79, ['owner_restarted'], 0, 1)  # (exit, outcomes, consecutive failures, graces)
_COUNTED = (1, ['crashed'], 1, 0)


@pytest.mark.parametrize('old_edit, new_edit, expected', [
    pytest.param({}, {}, _GRANTED, id='matching'),
    pytest.param({'hermes_home': _DROP}, {'hermes_home': _DROP}, _GRANTED, id='absent'),
    pytest.param({'hermes_home': None}, {}, _COUNTED, id='null-baseline'),
    pytest.param({}, {'hermes_home': None}, _COUNTED, id='null-replacement'),
    pytest.param({}, {'hermes_home': '/elsewhere/hermes-home'}, _COUNTED, id='wrong-home'),
])
def test_the_owner_records_home_must_be_absent_or_name_this_home(host, old_edit, new_edit, expected):
    """``hermes_home`` absent (a legacy record) is no objection; present, it must name this home. An
    explicit null in either record is no evidence: no grace, the ordinary counted failure."""
    def edited(record, edit):
        return {key: value for key, value in {**record, **edit}.items() if value is not _DROP}
    old, new = edited(host.owner(), old_edit), edited(host.owner(), new_edit)
    host.record_owner(old)
    _res, rc, _run = host.attempt('unavailable', replace=host.replacing(old, new))
    host.settle()
    assert not host.sent.exists()
    assert (rc, host.outcomes(), host.task().consecutive_failures, len(host.graces())) == expected


@pytest.mark.parametrize('evidence', ['no-record', 'malformed', 'baseline-already-dead', 'unchanged-owner'])
def test_missing_or_unprovable_replacement_still_counts(host, evidence):
    old, new = host.owner(), host.owner()
    extra = {}
    if evidence == 'malformed':
        host.record_owner(dict(old, pid=str(old['pid'])))
        extra['replace'] = dict(host.replacing(old, new), new_record=dict(new, start_time=None))
    elif evidence == 'baseline-already-dead':
        host.owners[0].kill()
        host.owners[0].wait()
        host.record_owner(old)  # names an incarnation that had ended before the attempt started
        extra['replace'] = host.replacing(old, new)
    elif evidence == 'unchanged-owner':
        host.record_owner(old)  # the same incarnation keeps draining
    _res, rc, _run = host.attempt('unavailable', **extra)
    assert rc == 1
    host.settle()
    assert host.outcomes() == ['crashed'] and host.task().consecutive_failures == 1
    assert host.graces() == []


@pytest.mark.parametrize('ending', ['keeps-draining', 'dies-without-replacement'])
def test_an_owner_that_never_returns_trips_the_failure_limit(host, ending):
    old = host.owner()
    host.record_owner(old)
    if ending == 'dies-without-replacement':
        host.owners[0].kill()
        host.owners[0].wait()
    codes = [host.attempt('unavailable')[1] for _ in range(FAILURE_LIMIT)]
    res = host.settle()
    assert codes == [1] * FAILURE_LIMIT
    assert host.outcomes() == ['crashed'] * FAILURE_LIMIT
    assert host.tid in res.auto_blocked and host.task().status == 'blocked'
    assert host.graces() == []


def test_a_crash_looping_owner_gets_one_grace_then_trips_the_failure_limit(host):
    """A new owner pid on every attempt is replacement evidence every time; the grace is spent once."""
    owners = [host.owner() for _ in range(FAILURE_LIMIT + 2)]
    host.record_owner(owners[0])
    codes = []
    for i in range(FAILURE_LIMIT + 1):  # 4 successive attempts
        codes.append(host.attempt('unavailable', replace=host.replacing(owners[i], owners[i + 1]))[1])
    res = host.settle()
    assert codes == [79] + [1] * FAILURE_LIMIT
    assert host.outcomes() == ['owner_restarted'] + ['crashed'] * FAILURE_LIMIT
    assert len(host.graces()) == 1
    assert host.tid in res.auto_blocked and host.task().status == 'blocked'
    assert host.task().consecutive_failures == FAILURE_LIMIT


def test_an_intervening_counted_failure_or_manual_wake_does_not_replenish_the_grace(host):
    owners = [host.owner() for _ in range(3)]
    host.record_owner(owners[0])
    assert host.attempt('unavailable', replace=host.replacing(owners[0], owners[1]))[1] == 79
    assert host.attempt('refused')[1] == 1  # a counted failure, nothing admitted
    host.settle()
    with kb.write_txn(host.conn):  # an operator parks and wakes the card: still nothing admitted
        host.conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (host.tid,))
    assert kb.unblock_task(host.conn, host.tid)
    _res, rc, _run = host.attempt('unavailable', replace=host.replacing(owners[1], owners[2]))
    host.settle()
    assert rc == 1
    assert host.outcomes() == ['owner_restarted', 'crashed', 'crashed']
    assert len(host.graces()) == 1


def test_an_admission_marker_opens_a_new_grace_window(host):
    """Marker-window coverage only: the ``admitted`` mode writes ``owner_admitted`` directly (scripted).
    The owner's real admission is ``test_real_owner_admission_reopens_the_grace_after_a_counted_failure``."""
    owners = [host.owner() for _ in range(3)]
    host.record_owner(owners[0])
    assert host.attempt('unavailable', replace=host.replacing(owners[0], owners[1]))[1] == 79
    _res, rc, admitted = host.attempt('admitted')  # the owner admits; the attempt then fails counted
    assert rc == 1 and host.sent.read_text().split() == ['kanban.run']
    _res, rc, later = host.attempt('unavailable', replace=host.replacing(owners[1], owners[2]))
    host.settle()
    assert rc == 79
    assert host.outcomes() == ['owner_restarted', 'crashed', 'owner_restarted']
    assert host.graces() == [host.graces()[0], later] and host.graces()[0] < admitted < later


@pytest.mark.parametrize('answer', ['lost', 'lost_unavailable'])
def test_an_ambiguous_answer_after_the_send_is_terminal_never_resubmitted_never_a_grace(host, answer):
    """``kanban.run`` was sent, then its answer was lost: the owner may have admitted it.

    * ``lost``: the RPC hang-up the real client raises (``GatewayClientError``).
    * ``lost_unavailable``: SYNTHETIC defensive coverage. The real client raises
      ``GatewayUnavailableError`` only in ``connect_gateway``'s ensure phase, never after a send. This
      mode scripts one anyway, to pin that a connection makes any later failure terminal.

    The wait window is 3 s against a 1 s retry pace, so a retrying client would send ``kanban.run``
    two or more times. Exactly one send, the ordinary counted exit, and no grace, even though
    replacement evidence is present."""
    old, new = host.owner(), host.owner()
    host.record_owner(old)
    _res, rc, _run = host.attempt(answer, replace=host.replacing(old, new), wait=3.0)
    host.settle()
    assert rc == 1
    assert host.sent.read_text().split() == ['kanban.run']  # exactly one send: never resubmitted
    assert host.connects.read_text().split() == ['connect']
    assert host.outcomes() == ['crashed'] and host.task().consecutive_failures == 1
    assert host.graces() == []


def test_a_79_trailer_without_a_recorded_grace_is_a_counted_crash(host):
    _res, rc, run = host.attempt('stale79')
    host.settle()
    assert rc == 79
    assert host.outcomes() == ['crashed'] and host.task().consecutive_failures == 1
    # Control for COMPOSE-ACCOUNT-N1: an unbacked 79 keeps its plain counted-exit error.
    (error,) = host.conn.execute('SELECT error FROM task_runs WHERE id=?', (run,)).fetchone()
    assert error.startswith('pid ') and ' exited with code 79' in error and 'settled' not in error


@pytest.mark.asyncio
async def test_real_owner_admission_reopens_the_grace_after_a_counted_failure(host, monkeypatch):
    """grace, then an ACTUAL owner admission, then a counted failure, then a later replacement earns a
    new grace.

    * Real: the owner's admission entry point (``gateway.session_kanban.run_task``), both of its
      steps, checked separately: the ``owner_admitted`` marker its board transaction commits FIRST,
      then the queue acceptance (``authority.submit``: the ``session_admissions`` row and receipt).
      The marker alone reopens nothing (``test_kanban_presubmit_grace_ledger``). Also real: the
      owner's FIFO claim (``claim_session_input``), its loss fence under a live owner
      (``_fence_lost_worker``) and settlement (``settle_unattended_unknown``) on a disposable
      SessionStore, ledger and home, as in ``tests/gateway/test_kanban_finite_admission.py``; the real
      submitters, dispatcher ticks, stale-claim reclaim and crash sweep.
    * Scripted: nothing is executed (the authority's FIFO schedule is a no-op, so there is no
      external work). The owner's worker registration is a row insert. The attempt-2 submitter is a
      stand-in process that exits 1 with its trailer. ``gateway_state.json`` is reset to the scenario's
      owner of record after the in-process owner ran. Its claim's expiry is backdated, so the real
      ``release_stale_claims`` books that attempt: ``reclaimed`` and counted (the owner pid that
      admitted it is this live test process, so only the expired claim plus the owner's terminal
      settlement prove the attempt over; the crash sweep leaves an owner-admitted run alone).
    """
    from gateway import run
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    from gateway.session_kanban import run_task
    from gateway.session_managed_worker import _fence_lost_worker
    from gateway.session_unattended_unknown import owner_ending
    from gateway.session_unattended_unknown import settle_unattended_unknown
    from hermes_cli.kanban_presubmit_grace import accepted
    from hermes_state_runtime import claim_session_input, find_session_admission
    from tests.gateway.test_session_follow_profile_config import _owner

    (host.home / 'config.yaml').write_text(json.dumps(
        {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}))
    monkeypatch.setattr(run, '_hermes_home', host.home)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(host.home))
    store, _runner, authority, dashboard = _owner(host.home)
    authority.profile_id = str(host.home)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})
    try:
        owners = [host.owner() for _ in range(3)]
        host.record_owner(owners[0])
        _res, rc, first = host.attempt('unavailable', replace=host.replacing(owners[0], owners[1]))
        assert rc == 79

        _res, rc, admitted = host.attempt('standin')
        task = host.task()
        params = dict(board='default', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock,
                      db=host.db)

        def marker():
            return host.conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
                                     "AND kind='owner_admitted'", (host.tid, admitted)).fetchone()
        assert marker() is None
        answer = await run_task(connection, params)  # the owner's actual admission
        sid, receipt = answer['session_id'], answer['receipt']
        binding = json.loads(marker()[0])  # step 1: the board marker
        assert (binding['session_id'], binding['pid']) == (sid, os.getpid())
        assert binding['db'] == str(Path(authority.db.db_path).resolve())
        queued = find_session_admission(authority.db, principal_id=connection.actor.subject, session_id=sid,
                                        request_id=binding['request_id'])  # step 2: the queue accepted it
        assert queued is not None and queued['admission_id'] == receipt['admission_id']
        assert (queued['status'], receipt['status']) == ('queued', 'queued')
        assert accepted(marker()[0])
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        authority.db._execute_write(lambda c: c.execute(
            "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
            "VALUES('execution',?,'compute',?,?,'registered','test')", (sid, authority.epoch, row['generation'])))
        await _fence_lost_worker(authority, row, {'epoch': authority.epoch, 'generation': row['generation'],
                                                  'execution_id': 'execution'},
                                 owner_ending(authority))  # main binds the loss's cause where it is found
        assert await settle_unattended_unknown(authority, SessionRef(authority.profile_id, sid))
        with kb.write_txn(host.conn):  # the claim lapses with nothing heartbeating it
            host.conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (int(time.time()) - 1, host.tid))
        # The in-process owner may record ITSELF (this test process) in the home's gateway_state.json;
        # the scenario's owner of record stays the incarnation the first restart left, owners[1].
        host.record_owner(owners[1])

        _res, rc, later = host.attempt('unavailable', replace=host.replacing(owners[1], owners[2]))
        host.settle()
        assert rc == 79
        assert host.outcomes() == ['owner_restarted', 'reclaimed', 'owner_restarted']
        assert host.task().consecutive_failures == 1  # the admitted attempt's failure was counted
        assert host.graces() == [first, later] and first < admitted < later
    finally:
        await connection.close()
        await dashboard.close()
        store.close_all_db_handles()


def test_an_aged_pruned_and_reopened_card_keeps_its_grace_spent(host):
    """The supported lifecycle that used to erase a spent grace with no admission: grace, then the card
    is completed by hand (``complete_task`` from ``ready``), its events age past the cutoff, the real
    event GC (``kb.gc_events``) runs, and the dashboard reopens it (``_set_status_direct``, the
    PATCH ``/tasks/{id}`` status path). Only the event age is backdated."""
    from plugins.kanban.dashboard.plugin_api import _set_status_direct
    owners = [host.owner() for _ in range(3)]
    host.record_owner(owners[0])
    assert host.attempt('unavailable', replace=host.replacing(owners[0], owners[1]))[1] == 79
    host.settle()
    assert host.task().status == 'ready'
    assert kb.complete_task(host.conn, host.tid, result='done by hand', summary='done by hand')
    old = int(time.time()) - 40 * 24 * 3600
    with kb.write_txn(host.conn):
        host.conn.execute('UPDATE task_events SET created_at=? WHERE task_id=?', (old, host.tid))
    events = host.conn.execute('SELECT COUNT(*) FROM task_events WHERE task_id=?', (host.tid,)).fetchone()[0]
    assert kb.gc_events(host.conn) > 0  # the default 30-day retention prunes this card's history
    assert host.conn.execute('SELECT COUNT(*) FROM task_events WHERE task_id=?', (host.tid,)).fetchone()[0] < events
    assert len(host.graces()) == 1  # ...but keeps the spent grace
    assert _set_status_direct(host.conn, host.tid, 'ready')
    _res, rc, _run = host.attempt('unavailable', replace=host.replacing(owners[1], owners[2]))
    host.settle()
    assert rc == 1
    assert host.outcomes() == ['owner_restarted', 'completed', 'crashed']  # complete_task books its own run
    assert host.task().consecutive_failures == 1 and len(host.graces()) == 1
