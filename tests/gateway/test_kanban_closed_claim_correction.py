"""A Kanban session's correction stays task-bound after its claim ends (NC11-CLIENT-01).

The first ``session.steer`` reaches the running card attempt and is kept on its task (``kept:
task_comment``); its reply is lost. The claim then ends. The client's retry names the same, now
closed, generation: the owner keeps it on the card's task again (a duplicate, accepted) and answers
the kept ACK, so no client turns it into a fresh, task-unbound ``prompt.submit``; its comment says it
reached no turn. ``replaced`` here is the session's execution (its next claim runs); a board claim
replaced by a new run is ``test_..._after_the_board_claim_was_replaced``. A non-Kanban session's stale
correction, a generation the session never had, another session's generation and a caller without
``session:control`` are refused.

Real authority, SQLite store, board and AuthorityConnection dispatch; the worker is a fixture. The
session's frozen policy is written as the bare record the owner reads (``policy.kanban_json``), once the
claim has ended, not as a full local lineage receipt.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
from tests.gateway.test_prompt_queued_controls import _call, _connection
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit


def _card_session(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    board = kbc.connect(tmp_path / 'board.sqlite')
    tid = kb.create_task(board, title='bound card')
    task = kb.claim_task(board, tid)
    context = {'db': str(tmp_path / 'board.sqlite'), 'task_id': tid, 'run_id': task.current_run_id}

    def correct(*args):
        verdict = asyncio.get_running_loop().create_future()
        verdict.set_result('rejected')
        return verdict
    worker = SimpleNamespace(policy=SimpleNamespace(kanban_json=json.dumps(context)), correct=correct)
    return db, authority, board, tid, worker


def _freeze_card_policy(db, worker):
    db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (
        POLICY_PREFIX + 's', json.dumps({'policy': {'kanban_json': worker.policy.kanban_json}}))))


def _kept(board, tid):
    return [c.body for c in kb.list_comments(board, tid) if 'TASK_BOUND_RETRY' in c.body]


@pytest.mark.asyncio
@pytest.mark.parametrize('claim', ['closed', 'replaced'])
async def test_a_retried_card_correction_after_its_claim_ended_is_kept_on_the_task(tmp_path, monkeypatch, claim):
    """``replaced``: the session's next claim is running when the retry names the ended one."""
    db, authority, board, tid, worker = _card_session(tmp_path, monkeypatch)
    with db:
        await _submit(authority, 'original-head')
        await _submit(authority, 'next-attempt')
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority.executing['s'] = row['generation']
        authority._managed_workers = {'s': worker}
        conn = _connection(authority)
        first = await _call(conn, 'session.steer', text='TASK_BOUND_RETRY', execution_generation=row['generation'])
        assert first['result']['kept'] == 'task_comment', first
        settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        authority.executing.pop('s')
        authority._managed_workers.clear()
        if claim == 'replaced':
            authority.executing['s'] = claim_session_input(db, epoch=authority.epoch, session_id='s')['generation']
        _freeze_card_policy(db, worker)
        for verb in ('session.steer', 'session.redirect'):
            retry = await _call(conn, verb, text='TASK_BOUND_RETRY', execution_generation=row['generation'])
            assert retry['result'] == {'text': 'TASK_BOUND_RETRY', 'execution_generation': row['generation'],
                                       'authority_epoch': authority.epoch, 'status': 'rejected',
                                       'delivered': False, 'kept': 'task_comment'}, retry
    assert len(_kept(board, tid)) == 3  # the first keep and both retries: duplicates, never a loss
    ended = [body for body in _kept(board, tid) if 'received after the turn it named had ended' in body]
    assert len(ended) == 2 and all('it was not sent to that turn' in body for body in ended)
    assert 'TASK_BOUND_RETRY' in kb.build_worker_context(board, tid)
    assert [r['request_id'] for r in list_session_admissions(db, session_id='s', pending_only=False)] == [
        'original-head', 'next-attempt']


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['not-a-card-session', 'never-this-sessions-generation'])
async def test_any_other_stale_correction_is_still_refused(tmp_path, monkeypatch, case):
    db, authority, board, tid, worker = _card_session(tmp_path, monkeypatch)
    with db:
        await _submit(authority, 'original-head')
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        if case != 'not-a-card-session':
            _freeze_card_policy(db, worker)
        generation = row['generation'] + (0 if case == 'not-a-card-session' else 7)
        reply = await _call(_connection(authority), 'session.steer', text='TASK_BOUND_RETRY',
                            execution_generation=generation)
    assert reply['error']['data']['reason'] == 'stale_generation', reply
    assert _kept(board, tid) == []


@pytest.mark.asyncio
async def test_a_generation_of_another_session_is_refused_for_this_card_session(tmp_path, monkeypatch):
    """Session ``s2`` is a card session of another task; ``s``'s past generation named against it is
    not one of its own claims: refused, nothing kept on either task."""
    from gateway.session_authority import LiveSession
    db, authority, board, tid, worker = _card_session(tmp_path, monkeypatch)
    other = kb.create_task(board, title='another card')
    other_run = kb.claim_task(board, other).current_run_id
    with db:
        await _submit(authority, 'original-head')
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        db.create_session('s2', source='test')
        authority.sessions['s2'] = LiveSession(authority.sessions['s'].source, 'route-2')
        _freeze_card_policy(db, worker)
        db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (
            POLICY_PREFIX + 's2', json.dumps({'policy': {'kanban_json': json.dumps(
                {'db': str(tmp_path / 'board.sqlite'), 'task_id': other, 'run_id': other_run})}}))))
        reply = await _call(_connection(authority), 'session.steer', session_id='s2', text='TASK_BOUND_RETRY',
                            execution_generation=row['generation'])
    assert reply['error']['data']['reason'] == 'stale_generation', reply
    assert _kept(board, tid) == [] and _kept(board, other) == []


@pytest.mark.asyncio
async def test_a_correction_without_session_control_is_refused_before_anything_is_kept(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    db, authority, board, tid, worker = _card_session(tmp_path, monkeypatch)
    with db:
        await _submit(authority, 'original-head')
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        _freeze_card_policy(db, worker)
        reader = AuthorityConnection(authority, object(), {'user_id': 'human', 'capabilities': ['session:read', 'session:submit']})
        reply = await _call(reader, 'session.steer', text='TASK_BOUND_RETRY', execution_generation=row['generation'])
    assert reply['error']['data']['reason'] == 'permission_denied', reply
    assert _kept(board, tid) == []


@pytest.mark.asyncio
async def test_a_retried_card_correction_after_the_board_claim_was_replaced(tmp_path, monkeypatch):
    """The board claim itself is replaced: the card's run ends and the task is claimed by a new run (a
    new attempt, in its own session). The old session's retry is kept on the same task, above the new
    run's ``comment_floor``, so the new attempt's worker is owed it on its first poll and its context;
    nothing is admitted to the old session."""
    db, authority, board, tid, worker = _card_session(tmp_path, monkeypatch)
    old_run = json.loads(worker.policy.kanban_json)['run_id']
    with db:
        await _submit(authority, 'original-head')
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority.executing['s'] = row['generation']
        authority._managed_workers = {'s': worker}
        conn = _connection(authority)
        first = await _call(conn, 'session.steer', text='TASK_BOUND_RETRY', execution_generation=row['generation'])
        assert first['result']['kept'] == 'task_comment', first
        settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        authority.executing.pop('s')
        authority._managed_workers.clear()
        board.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?", (tid,))
        board.commit()
        new_run = kb.claim_task(board, tid).current_run_id
        assert new_run != old_run
        _freeze_card_policy(db, worker)
        retry = await _call(conn, 'session.steer', text='TASK_BOUND_RETRY', execution_generation=row['generation'])
    assert retry['result']['kept'] == 'task_comment', retry
    floor = json.loads(kb._latest_event(board, tid, 'claimed', new_run)['payload'])['comment_floor']
    retried = [c for c in kb.list_comments(board, tid) if 'received after the turn it named had ended' in c.body]
    assert len(retried) == 1 and retried[0].id > floor and f'run {old_run}' in retried[0].body
    assert 'TASK_BOUND_RETRY' in kb.build_worker_context(board, tid)
    assert [r['request_id'] for r in list_session_admissions(db, session_id='s', pending_only=False)] == ['original-head']


@pytest.mark.asyncio
async def test_a_card_correction_before_its_worker_attaches_is_kept_not_handed_back(tmp_path, monkeypatch):
    """The card's claim runs but no agent or worker is attached yet (the build window): the owner
    would answer ``delivered: False``, which every client re-admits as a fresh, task-unbound prompt.
    A Kanban session keeps it on the card's task instead, above the run's ``comment_floor`` (its
    worker's first poll reads it) and answers the kept ACK; nothing is admitted. A non-Kanban
    session's reply is unchanged (test_session_steer_stale_busy_refusal)."""
    from types import SimpleNamespace as NS
    from gateway.session_busy_controls import correct
    from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF
    from tests.gateway.test_session_steer_stale_busy_refusal import _claim_without_agent
    db, authority = _authority(tmp_path, monkeypatch)
    board = kbc.connect(tmp_path / 'board.sqlite')
    tid = kb.create_task(board, title='bound card')
    run = kb.claim_task(board, tid).current_run_id
    context = {'db': str(tmp_path / 'board.sqlite'), 'task_id': tid, 'run_id': run}
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        _freeze_card_policy(db, NS(policy=NS(kanban_json=json.dumps(context))))
        generation = authority._handle(REF).execution_generation
        reply = await correct(NS(authority=authority, actor=ACTOR), REF, {
            'session_id': 's', 'text': 'TASK_BOUND_RETRY', 'execution_generation': generation}, verb='redirect')
        assert reply == {'text': 'TASK_BOUND_RETRY', 'execution_generation': generation,
                         'authority_epoch': authority.epoch, 'status': 'rejected', 'delivered': False,
                         'kept': 'task_comment'}, reply
        # The bare fixture record is no full local lineage receipt; the finishing turn reads one.
        db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + 's',)))
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    floor = json.loads(kb._latest_event(board, tid, 'claimed', run)['payload'])['comment_floor']
    kept = [c for c in kb.list_comments(board, tid) if 'TASK_BOUND_RETRY' in c.body]
    assert len(kept) == 1 and kept[0].id > floor and f'run {run} while it was running' in kept[0].body
    assert [r['request_id'] for r in list_session_admissions(db, session_id='s', pending_only=False)] == ['building']
