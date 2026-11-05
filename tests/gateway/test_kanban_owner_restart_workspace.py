"""An owner's restart accounting does not end its bound interpreter's use of the workspace.

t_2c446913 INT-CWD-01: a replacement owner's boot recovery records the lost attempt's
``worker_result`` (exit 79, the bound interpreter's pid, ``recorded_by_owner_pid``) while that
interpreter may still be running: a tool in flight gets a grace period after owner-pipe EOF. Workspace
cleanup read any same run/pid result as "this worker is done", so the interpreter's own
``kanban_complete`` (and parent, shared and event-retention cleanup) removed its live cwd. Only the
interpreter's own result ends its binding now; an owner-recorded one leaves it to the process liveness
check. Real SessionStore, board SQLite, boot recovery, completion, cleanup, gc and prompt builder.
"""
import argparse
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
from hermes_state_runtime import get_session_admission
from tests.gateway.test_kanban_completion_workspace import _ended_process_identity
from tests.gateway.test_kanban_owner_restart_durable import _owner, _start_turn

pytest_plugins = ('tests.gateway.test_kanban_owner_restart_durable',)  # its ``card`` fixture


def _workspace(conn, params):
    workspace = Path(ws.resolve_workspace(kb.get_task(conn, params['task_id'])))
    ws.set_workspace_path(conn, params['task_id'], workspace)
    (workspace / 'AGENTS.md').write_text('restart sentinel\n')
    return workspace


def _bind(conn, params, pid, started_at):
    with kb.write_txn(conn):
        kb._append_event(conn, params['task_id'], 'worker_bound',
                         {'pid': pid, 'claim_lock': params['claim_lock'], 'started_at': started_at},
                         run_id=params['run_id'])


def _bind_this_interpreter(conn, params, workspace, monkeypatch):
    """This live test process is the attempt's bound interpreter, working in its workspace."""
    _bind(conn, params, os.getpid(), kbd._process_fingerprint(os.getpid()))
    monkeypatch.chdir(workspace)


async def _owner_restarts(runner, params, board_db, store):
    """The attempt's owner ends mid-turn; a replacement owner's boot recovery settles it."""
    admission = _start_turn(await _owner(runner, 'first'), params)
    await initialize_session_authority(runner, profile_id='p', instance_id='replacement')
    receipt = get_session_admission(store._db, admission_id=admission['admission_id'])
    assert (receipt['status'], receipt['outcome']) == ('terminal', 'interrupted')
    result = session_kanban.worker_result(board_db, params)
    assert result['exit_code'] == KANBAN_OWNER_RESTART_EXIT_CODE and result['recorded_by_owner_pid'] == os.getpid()
    return result


def _complete_as_the_worker(monkeypatch, params, board_db):
    from tools import kanban_tools
    monkeypatch.setenv('HERMES_KANBAN_DB', board_db)
    monkeypatch.setenv('HERMES_KANBAN_TASK', params['task_id'])
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(params['run_id']))
    completed = json.loads(kanban_tools._handle_complete({'summary': 'completed by the bound worker'}))
    assert completed['ok'], completed


def _turn_goes_on():
    """What the interpreter's turn does next: rebuild its prompt from its cwd, write relatively."""
    from agent.prompt_builder import build_context_files_prompt
    assert 'restart sentinel' in build_context_files_prompt(cwd=None)
    Path('late.txt').write_text('still in the bound worker turn')


def _worker_ends(params, board_db, tmp_path, monkeypatch):
    """The interpreter's own finalizer: its result, then the deferred cleanup."""
    monkeypatch.chdir(tmp_path)
    session_kanban._record_worker_result(dict(params, db=board_db), 0, 'finished')


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['live-normal', 'live-restart', 'dead-restart'])
async def test_boot_recovery_does_not_release_a_live_workspace(card, monkeypatch, tmp_path, mode):
    """The reviewer's composition probe: own completion after (or without) a replacement owner's
    boot recovery. A live interpreter keeps its cwd until its own end; a dead one is reclaimed."""
    store, runner, conn, params, board_db = card
    workspace = _workspace(conn, params)
    if mode == 'dead-restart':
        _bind(conn, params, *_ended_process_identity())
    else:
        _bind_this_interpreter(conn, params, workspace, monkeypatch)
    if mode == 'live-normal':
        _start_turn(await _owner(runner, 'first'), params)
    else:
        await _owner_restarts(runner, params, board_db, store)
    expected = None if mode == 'dead-restart' else 'live_worker'
    assert ws._workspace_in_use_by_other(conn, params['task_id'], workspace) == expected
    _complete_as_the_worker(monkeypatch, params, board_db)
    if mode == 'dead-restart':
        assert not workspace.exists()  # nothing runs there any more: reclaimed at once
        return
    try:
        assert workspace.is_dir()
        _turn_goes_on()
    finally:
        _worker_ends(params, board_db, tmp_path, monkeypatch)
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_parent_and_shared_cleanup_keep_an_owner_settled_live_workspace(card, monkeypatch, tmp_path):
    """The owner-settled interpreter's workspace also survives a child's parent sweep and another
    card's cleanup of the same directory, until the interpreter itself ends."""
    store, runner, conn, params, board_db = card
    workspace = _workspace(conn, params)
    child = kb.create_task(conn, title='consumer', assignee='w', parents=[params['task_id']],
                           workspace_kind='dir', workspace_path=str(workspace))
    sharer = kb.create_task(conn, title='sharer', assignee='w', workspace_kind='scratch', workspace_path=str(workspace))
    _bind_this_interpreter(conn, params, workspace, monkeypatch)
    await _owner_restarts(runner, params, board_db, store)
    _complete_as_the_worker(monkeypatch, params, board_db)  # own cleanup: the child still needs it
    monkeypatch.delenv('HERMES_KANBAN_TASK')
    assert kb.complete_task(conn, sharer, summary='shared cleanup')  # sharer's own scratch removal
    assert workspace.is_dir()
    assert kb.complete_task(conn, child, summary='child done')  # child's sweep of its done parent
    assert workspace.is_dir()
    try:
        _turn_goes_on()
    finally:
        _worker_ends(params, board_db, tmp_path, monkeypatch)
    assert not workspace.exists()


def _bindings(conn, params):
    return [json.loads(p) for (p,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound'", (params['task_id'],))]


@pytest.mark.asyncio
@pytest.mark.parametrize('worker', ['live', 'dead'])
async def test_event_retention_keeps_an_owner_settled_live_binding(card, monkeypatch, tmp_path, worker):
    """Positive-retention ``kanban gc`` over the archived card keeps a live interpreter's binding
    (and so its workspace) despite the owner's result, and reclaims both once it is gone."""
    from hermes_cli import kanban_ops
    store, runner, conn, params, board_db = card
    workspace = _workspace(conn, params)
    identity = (os.getpid(), kbd._process_fingerprint(os.getpid())) if worker == 'live' else _ended_process_identity()
    _bind(conn, params, *identity)
    await _owner_restarts(runner, params, board_db, store)
    monkeypatch.chdir(tmp_path)
    assert kb.archive_task(conn, params['task_id'], signal_fn=lambda *_: None)
    with kb.write_txn(conn):
        conn.execute('UPDATE task_events SET created_at = created_at - 31 * 86400 WHERE task_id = ?', (params['task_id'],))
    args = argparse.Namespace(event_retention_days=30, log_retention_days=0)
    for _ in range(2):
        assert kanban_ops._cmd_gc(args) == 0
    kept = [(b['pid'], b.get('started_at')) for b in _bindings(conn, params)]
    if worker == 'dead':
        assert identity not in kept
        assert not workspace.exists()
        return
    assert kept == [identity]  # the live binding, and nothing else, outlives retention
    assert workspace.is_dir()
    _worker_ends(params, board_db, tmp_path, monkeypatch)  # archived card: its own end reclaims
    assert not workspace.exists()


# A separate real interpreter binds through ``bind_worker_context`` (fingerprint, submitter watch)
# and enters ``kanban_complete``; only the call into ``complete_task`` waits for the test's release,
# so a replacement owner's boot recovery can land while the completion is in flight.
_HELD_WORKER = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from gateway import session_kanban
from hermes_cli import kanban_db as kb
context = json.loads(sys.argv[1])
session_kanban.bind_worker_context({'policy': {'source': 'kanban', 'kanban_json': json.dumps(context)}})
from tools import kanban_tools
original = kb.complete_task
def entered(*args, **kwargs):
    print('ENTERED', flush=True)
    assert sys.stdin.readline().strip() == 'release'
    return original(*args, **kwargs)
kb.complete_task = entered
completed = json.loads(kanban_tools._handle_complete({'summary': 'real bound worker completed'}))
seen = {'pid': os.getpid(), 'completion_ok': completed.get('ok')}
try:
    from agent.prompt_builder import build_context_files_prompt
    seen['prompt'] = 'restart sentinel' in build_context_files_prompt(cwd=None)
    Path('after.txt').write_text('still using cwd')
    seen['workspace_exists'] = Path(context['workspace']).is_dir()
except FileNotFoundError as exc:
    seen['cwd_error'] = type(exc).__name__
print('OBSERVATION ' + json.dumps(seen), flush=True)
os.chdir(Path(context['db']).parent)
'''


def _enter_held_worker(context):
    """Start the held interpreter and wait until it is inside ``complete_task``."""
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen([sys.executable, '-c', _HELD_WORKER, json.dumps(context), str(repo)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    if line.strip() != 'ENTERED':
        proc.kill()
        raise AssertionError(('child ended before the completion barrier', line, proc.communicate(timeout=30)))
    return proc


def _release_held_worker(proc):
    """Let its completion proceed; what its turn saw afterwards."""
    try:
        stdout, stderr = proc.communicate('release\n', timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, (stdout, stderr)
    seen = json.loads(next(x.removeprefix('OBSERVATION ') for x in stdout.splitlines() if x.startswith('OBSERVATION ')))
    return seen, stderr


@pytest.mark.asyncio
@pytest.mark.parametrize('restart', [False, True])
async def test_bound_process_inside_completion_during_boot(card, restart):
    """The reviewer's process probe: the owner restarts while a real bound interpreter is inside
    ``kanban_complete``; its completion must not remove the cwd it goes on using."""
    _store, runner, conn, params, board_db = card
    workspace = _workspace(conn, params)
    # This live test process stands in for the attempt's submitter, which the bound interpreter's
    # watch thread polls (the fixture's recorded submitter is a dead pid).
    with kb.write_txn(conn):
        for table, key in (('tasks', params['task_id']), ('task_runs', params['run_id'])):
            conn.execute(f'UPDATE {table} SET worker_pid = ?, worker_started_at = ? WHERE id = ?',
                         (os.getpid(), kbd._process_fingerprint(os.getpid()), key))
    context = dict(params, db=board_db, workspace=str(workspace), workspaces_root=str(workspace.parent),
                   profile='default', branch=None, tenant=None, goal_mode=False, max_runtime_seconds=None)
    _start_turn(await _owner(runner, 'original'), params)
    proc = await asyncio.to_thread(_enter_held_worker, context)
    try:
        assert ws._workspace_in_use_by_other(conn, params['task_id'], workspace) == 'live_worker'
        if restart:
            await initialize_session_authority(runner, profile_id='p', instance_id='replacement')
            assert session_kanban.worker_result(board_db, params)['exit_code'] == KANBAN_OWNER_RESTART_EXIT_CODE
            assert ws._workspace_in_use_by_other(conn, params['task_id'], workspace) == 'live_worker'
    finally:
        seen, stderr = await asyncio.to_thread(_release_held_worker, proc)
    assert seen == {'pid': proc.pid, 'completion_ok': True, 'prompt': True, 'workspace_exists': True}, (seen, stderr)
