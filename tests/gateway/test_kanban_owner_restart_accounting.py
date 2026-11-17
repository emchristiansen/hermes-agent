"""An attempt its owner recorded as ``owner_restarted`` stays uncounted when its interpreter ends later.

ACCOUNT-01 (t_f96cc165): a replacement owner's boot recovery records exit 79 for an attempt its
predecessor left in flight; the bound interpreter, which may outlive its owner, then records its own
result (here 130). Both result readers took the latest one, so the submitter returned 130 and the
crash sweep booked the run ``crashed`` and counted a failure. The owner's record now decides the
attempt's accounting (``kanban_owner_recovery.attempt_result``); the interpreter's result stays on the
run as its own fact and still ends its workspace binding. Real SessionStore, board SQLite, boot
recovery, a separate bound interpreter (``bind_worker_context`` / ``_record_worker_result``), the
submitter's result reader and the dispatcher's crash sweep. (Ported from the t_f96cc165 reviewer's
characterization, whose late case expected the counted crash.)
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway import session_kanban
from gateway.session_authority import initialize_session_authority
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as ws
from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
from tests.gateway.test_kanban_owner_restart_durable import _consumed, _owner, _start_turn

pytest_plugins = ('tests.gateway.test_kanban_owner_restart_durable',)  # its ``card`` fixture

# The bound interpreter: binds as the owner binds it, then either records its own end (``late``)
# or exits without a result.
_INTERPRETER = r'''
import json, sys
sys.path.insert(0, sys.argv[2])
from gateway import session_kanban
context = json.loads(sys.argv[1])
session_kanban.bind_worker_context({'policy': {'source': 'kanban', 'kanban_json': json.dumps(context)}})
print('BOUND', flush=True)
if sys.stdin.readline().strip() == 'late':
    session_kanban._record_worker_result(context, 130, 'interrupted after its owner accounted for it')
else:
    session_kanban._TURNS_DONE.set()
'''


def _bind_interpreter(context):
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen([sys.executable, '-c', _INTERPRETER, json.dumps(context), str(repo)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    if line.strip() != 'BOUND':
        proc.kill()
        raise AssertionError(('interpreter did not bind', line, proc.communicate(timeout=30)))
    return proc


def _end_interpreter(proc, late):
    try:
        out, err = proc.communicate('late\n' if late else 'exit\n', timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, (out, err)


def _results(conn, params):
    return [json.loads(p) for (p,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result' ORDER BY id",
        (params['task_id'], params['run_id']))]


def _open_binding_pids(conn, params):
    return {json.loads(payload)['pid'] for task_id, _path, payload in conn.execute(ws._OPEN_BINDINGS_SQL)
            if task_id == params['task_id']}


@pytest.mark.asyncio
@pytest.mark.parametrize('late', [False, True])
async def test_owner_restart_stays_uncounted_when_the_interpreter_ends_later(card, tmp_path, late):
    store, runner, conn, params, board_db = card
    # This live test process stands in for the attempt's submitter (the fixture's is a dead pid).
    with kb.write_txn(conn):
        for table, key in (('tasks', params['task_id']), ('task_runs', params['run_id'])):
            conn.execute(f'UPDATE {table} SET worker_pid = ?, worker_started_at = ? WHERE id = ?',
                         (os.getpid(), kbd._process_fingerprint(os.getpid()), key))
    ws.set_workspace_path(conn, params['task_id'], tmp_path)  # a dir workspace: accounting only
    context = dict(params, db=board_db, workspace=str(tmp_path), workspaces_root=str(tmp_path), profile='default',
                   branch=None, tenant=None, goal_mode=False, max_runtime_seconds=None)
    admission = _start_turn(await _owner(runner, 'original'), params)
    proc = await asyncio.to_thread(_bind_interpreter, context)
    try:
        await initialize_session_authority(runner, profile_id='p', instance_id='replacement')
        assert session_kanban.worker_exit_code(board_db, params) == KANBAN_OWNER_RESTART_EXIT_CODE
        assert proc.pid in _open_binding_pids(conn, params)  # the owner's record does not end it
    finally:
        await asyncio.to_thread(_end_interpreter, proc, late)
    # The submitter stand-in exits: the crash sweep may now book the run.
    with kb.write_txn(conn):
        for table, key in (('tasks', params['task_id']), ('task_runs', params['run_id'])):
            conn.execute(f'UPDATE {table} SET worker_pid = ? WHERE id = ?', (proc.pid, key))
    observed = _consumed(store._db, conn, params, board_db, admission['admission_id'])
    assert observed == (('terminal', 'interrupted'), KANBAN_OWNER_RESTART_EXIT_CODE, 'owner_restarted', 0)
    # The interpreter's own end is still on the run, as its own fact, and ends its binding.
    assert [(r['exit_code'], r['pid'], 'recorded_by_owner_pid' in r) for r in _results(conn, params)] == (
        [(KANBAN_OWNER_RESTART_EXIT_CODE, proc.pid, True)] + ([(130, proc.pid, False)] if late else []))
    assert (proc.pid in _open_binding_pids(conn, params)) is not late
