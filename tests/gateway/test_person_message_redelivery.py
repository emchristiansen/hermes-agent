"""A person's message whose turn its owner's end cut short runs again at the next owner: at least once.

recovery-exercise-1 (a-cgstop-3, b-eofrace-1, b-eofdelay-1, c-cgkill-1): the in-flight person turn was
settled ``interrupted`` ("will not be run again. Send your message again") or fenced ``unknown`` at boot,
and the message was never answered. Eric's rule: "A replay is completely fine. I don't give a fuck whether
the same message is sent twice. I just want all messages to be delivered at least once."

The message is queued again in its own FIFO place and runs as an ordinary turn (a duplicate answer is
acceptable). A turn that reached a tool may have had side effects (a business effect): it is never re-run
silently and keeps the owner's notice for the person, as before. Real SessionStore SQLite, two real owner
epochs; the turn is lost the way a dead owner leaves it (claimed, its user row written).
"""
import pytest

from gateway.session_contract import SessionRef
from hermes_state_runtime import claim_session_input
from tests.gateway.test_unknown_automation_settles import _person_head, _runner, _statuses
from tests.waiting import async_wait_until


async def _owner(runner, name):
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, initialize_session_authority
    authority = await initialize_session_authority(runner, profile_id='p', instance_id=name)
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='gw10'), 's')
    return authority


async def _lost_person_turn(tmp_path, monkeypatch, executed, *, tool_step, stopped):
    """Owner 1 admits a person's HEAD and FOLLOWER, claims HEAD and writes its user row (with a tool call
    and result when ``tool_step``). ``stopped``: owner 1 settles it as its stop does (interrupted,
    shutdown cause); otherwise it dies with the turn started. Owner 2 then boots on the same store."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import Submission
    from tests.gateway.test_unknown_automation_settles import Peer
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    db.create_session('s', source='telegram')
    first = await _owner(runner, 'owner-1')
    first._schedule = lambda ref: None
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    head_id = await _person_head(first, actor, ref)
    follower = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = claim_session_input(db, epoch=first.epoch, session_id='s')
    assert claimed['admission_id'] == head_id
    db.append_message('s', 'user', 'HEAD')
    if tool_step:
        db.append_message('s', 'assistant', None, tool_calls=[{'id': 'c1', 'type': 'function', 'function': {
            'name': 'terminal', 'arguments': '{"command": "touch effect"}'}}])
        db.append_message('s', 'tool', 'done', tool_call_id='c1', tool_name='terminal')
    if stopped:
        from gateway.run import _INTERRUPT_REASON_GATEWAY_SHUTDOWN
        from gateway.session_results import finish_result
        captured = {'result': {'final_response': '', 'interrupted': True,
                               'interrupt_message': _INTERRUPT_REASON_GATEWAY_SHUTDOWN}, 'usage': {}}
        settled, _ = finish_result(db, epoch=first.epoch, row=claimed, response='', outcome='interrupted',
                                   result=captured)
        assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    second = await _owner(runner, 'owner-2')
    return store, second, head_id, follower.admission_id


async def _drained(authority):
    authority._schedule(SessionRef('p', 's'))
    live = authority.sessions['s']
    await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')


def _notices(db):
    return [m['content'] for m in db.get_messages('s') if m.get('display_kind') == 'failed_turn']


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, True], ids=['owner_died', 'owner_stopped'])
async def test_a_persons_lost_message_runs_again_at_the_next_owner(tmp_path, monkeypatch, stopped):
    executed = []
    store, authority, head_id, follower_id = await _lost_person_turn(
        tmp_path, monkeypatch, executed, tool_step=False, stopped=stopped)
    try:
        await _drained(authority)
        # The lost message ran again (its first run was never answered), then the follower, once each.
        assert executed == ['HEAD', 'FOLLOWER']
        assert _statuses(authority.db)[head_id] == ('terminal', 'completed')
        assert _statuses(authority.db)[follower_id] == ('terminal', 'completed')
        assert any('will run again' in notice for notice in _notices(authority.db)), _notices(authority.db)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, True], ids=['owner_died', 'owner_stopped'])
async def test_a_lost_turn_that_reached_a_tool_is_not_run_again(tmp_path, monkeypatch, stopped):
    """Business-effect control: a tool step may have changed the world, so nothing re-runs it."""
    executed = []
    store, authority, head_id, follower_id = await _lost_person_turn(
        tmp_path, monkeypatch, executed, tool_step=True, stopped=stopped)
    try:
        await _drained(authority)
        assert 'HEAD' not in executed
        if stopped:
            # Settled interrupted with the stop's own notice ("Send your message again"); the follower runs.
            assert _statuses(authority.db)[head_id] == ('terminal', 'interrupted')
            assert executed == ['FOLLOWER']
            assert not any('will run again' in notice for notice in _notices(authority.db))
        else:
            # Fenced unknown: the person's FIFO pauses for them, exactly as before.
            assert _statuses(authority.db)[head_id] == ('unknown', None)
            assert executed == []
            assert _statuses(authority.db)[follower_id] == ('queued', None)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_redelivered_managed_turns_worker_outbox_is_never_replayed(tmp_path, monkeypatch):
    """The lost turn ran in a managed worker. Its message runs again, but the worker's own outbox (its
    transcript/usage writes: persistence, not messages) is never replayed: the lost execution is
    retired with the redelivery, so neither its adoption nor its claim on the session survives."""
    import hashlib
    from hermes_state_runtime import RuntimeStoreError, adopt_worker_execution
    executed = []
    real_claim = claim_session_input

    def claim_with_worker(db, *, epoch, session_id, **kw):
        row = real_claim(db, epoch=epoch, session_id=session_id, **kw)
        if row and row['payload']['text'] == 'HEAD':  # the claim the dead owner's worker ran
            db._execute_write(lambda c: c.execute(
                "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
                "VALUES(?,?,'compute',?,?,'running',?)", ('admission-worker:' + row['admission_id'], session_id, epoch,
                                                          row['generation'], hashlib.sha256(b's3cret').hexdigest())))
        return row
    monkeypatch.setitem(globals(), 'claim_session_input', claim_with_worker)  # _lost_person_turn's claim
    store, authority, head_id, _follower_id = await _lost_person_turn(
        tmp_path, monkeypatch, executed, tool_step=False, stopped=False)
    try:
        worker = authority.db._read_one("SELECT status, generation FROM worker_executions WHERE execution_id=?",
                                        ('admission-worker:' + head_id,))
        assert worker[0] == 'terminal'
        with pytest.raises(RuntimeStoreError):
            adopt_worker_execution(authority.db, epoch=authority.epoch, execution_id='admission-worker:' + head_id,
                                   session_id='s', generation=worker[1], adoption_secret='s3cret')
        await _drained(authority)
        assert executed == ['HEAD', 'FOLLOWER']
    finally:
        store.close_all_db_handles()
