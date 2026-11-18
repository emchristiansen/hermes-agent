"""At-least-once ``kanban.run``: a resent request for the same attempt never runs the card twice.

The dispatcher worker re-sends ``kanban.run`` when its answer was lost (a dropped socket, a restart;
hermes_cli/kanban_worker_client.py). Duplicates are allowed; duplicate execution is not. The owner
keys the attempt (board, task, run) to one session and one durable admission request id, so a resend
on a new connection, while the first turn still runs, or after it settled returns the original
admission, and the drain executes the turn once.
"""
import asyncio
from contextlib import closing
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until

_CONFIG = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}


@pytest.fixture
def kanban_owner(tmp_path, monkeypatch):
    """A real owner authority and a claimed card on its board (as tests/gateway/test_kanban_finite_admission.py)."""
    import json
    from gateway import run
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from tests.gateway.test_session_follow_profile_config import _owner

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    for name in ('HERMES_KANBAN_DB', 'HERMES_DELEGATED_CHILD', 'HERMES_SINGLE_QUERY_SESSION', 'HERMES_YOLO_MODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))

    def build(config):
        (tmp_path / 'config.yaml').write_text(json.dumps(config))
        kb.create_board('owned')
        with closing(connect(board='owned')) as conn:
            task_id = kb.create_task(conn, title='Headless work', assignee='default',
                                    workspace_kind='dir', workspace_path=str(tmp_path))
            kb.recompute_ready(conn)
            task = kb.claim_task(conn, task_id)
        store, runner, authority, dashboard = _owner(tmp_path)
        authority.profile_id = str(tmp_path)
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        connection = _second_connection(authority)

        async def close():
            await connection.close()
            await dashboard.close()
            store.close_all_db_handles()

        params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
        return SimpleNamespace(runner=runner, authority=authority, connection=connection, params=params, close=close)
    return build


def _second_connection(authority):
    """A redial: a new socket to the same owner, authenticated as the same dispatcher."""
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})


def _real_drain(owner, monkeypatch, gate=None):
    """Restore the authority's real FIFO scheduling and count executions per admission."""
    from gateway.session_authority import SessionAuthority
    authority = owner.authority
    monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    executed = []

    async def execute(authority_, ref, row):
        executed.append(row['admission_id'])
        if gate is not None:
            await gate.wait()
        return 'card done'

    monkeypatch.setattr('gateway.session_ingress.execute_admission', execute)
    return executed


async def _drained(authority, sid):
    """The session's FIFO drain has nothing scheduled: a copy that had been admitted anew would have run by now."""
    def idle():
        live = authority.sessions.get(sid)
        return live is None or live.task is None or live.task.done()
    await async_wait_until(idle, timeout=10, what='the card session drain to go idle')


async def _settle(owner, sid, admission_id):
    """Wait for the admission's receipt to turn terminal (the drain executed and settled it)."""
    from gateway.session_contract import SessionRef
    authority = owner.authority
    ref = SessionRef(authority.profile_id, sid)

    async def terminal():
        receipt = await authority.receipt(owner.connection.actor, ref, admission_id)
        return receipt if receipt.status == 'terminal' else None
    return await async_wait_until(terminal, timeout=10, what='the card turn to settle')


def test_a_resend_after_the_turn_settled_returns_the_original_admission_and_never_reruns(kanban_owner, monkeypatch):
    from gateway.session_kanban import run_task
    from hermes_state_runtime import list_session_admissions

    owner = kanban_owner(_CONFIG)
    authority = owner.authority
    executed = _real_drain(owner, monkeypatch)

    async def scenario():
        redial = _second_connection(authority)
        try:
            first = await run_task(owner.connection, owner.params)  # its answer is "lost"
            sid, admission_id = first['session_id'], first['receipt']['admission_id']
            await _settle(owner, sid, admission_id)
            assert executed == [admission_id]
            resent = await run_task(redial, owner.params)  # the worker's resend on a new socket
            assert resent['session_id'] == sid
            assert resent['receipt']['admission_id'] == admission_id
            await _drained(authority, sid)
            assert executed == [admission_id], 'a resend must never execute the card again'
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert [row['admission_id'] for row in rows] == [admission_id]
        finally:
            await redial.close()
            await owner.close()

    asyncio.run(scenario())


def test_a_resend_while_the_first_turn_runs_joins_it(kanban_owner, monkeypatch):
    from gateway.session_kanban import run_task
    from hermes_state_runtime import list_session_admissions

    owner = kanban_owner(_CONFIG)
    authority = owner.authority

    async def scenario():
        gate = asyncio.Event()
        executed = _real_drain(owner, monkeypatch, gate)
        redial = _second_connection(authority)
        try:
            first, resent = await asyncio.gather(run_task(owner.connection, owner.params),
                                                 run_task(redial, owner.params))
            sid, admission_id = first['session_id'], first['receipt']['admission_id']
            assert resent['session_id'] == sid and resent['receipt']['admission_id'] == admission_id
            again = await run_task(redial, owner.params)  # a third copy mid-turn
            assert again['receipt']['admission_id'] == admission_id
            gate.set()
            await _settle(owner, sid, admission_id)
            await _drained(authority, sid)
            assert executed == [admission_id], 'concurrent copies of one attempt run it once'
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert [row['admission_id'] for row in rows] == [admission_id]
        finally:
            gate.set()
            await redial.close()
            await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('cut', ['before_resume', 'before_submit'])
def test_a_resend_after_an_owner_side_cut_admits_the_attempt_once(kanban_owner, monkeypatch, cut):
    """The first request died inside the owner (session created and claim marked, no admission yet):
    the resend completes that same attempt, once."""
    from gateway.session_kanban import run_task
    from hermes_state_runtime import list_session_admissions

    owner = kanban_owner(_CONFIG)
    authority = owner.authority
    executed = _real_drain(owner, monkeypatch)

    class Cut(Exception):
        pass

    async def scenario():
        redial = _second_connection(authority)
        try:
            if cut == 'before_resume':
                real = owner.connection.resume

                async def resume(ref, params):
                    raise Cut
                monkeypatch.setattr(owner.connection, 'resume', resume)
            else:
                real = authority.submit

                async def submit(actor, submission, **kwargs):
                    raise Cut
                monkeypatch.setattr(authority, 'submit', submit)
            with pytest.raises(Cut):
                await run_task(owner.connection, owner.params)
            monkeypatch.setattr(owner.connection if cut == 'before_resume' else authority,
                                'resume' if cut == 'before_resume' else 'submit', real)
            resent = await run_task(redial, owner.params)
            sid, admission_id = resent['session_id'], resent['receipt']['admission_id']
            await _settle(owner, sid, admission_id)
            again = await run_task(redial, owner.params)
            assert again['receipt']['admission_id'] == admission_id
            await _drained(authority, sid)
            assert executed == [admission_id]
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert [row['admission_id'] for row in rows] == [admission_id]
        finally:
            await redial.close()
            await owner.close()

    asyncio.run(scenario())
