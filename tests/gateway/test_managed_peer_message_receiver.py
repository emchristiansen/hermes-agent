"""The managed worker's private peer-message receiver, at its own entry points.

Child (``agent.managed_worker.WorkerControls``): frames go through the real control reader (a pipe),
the board fence is the real one over a real claimed card attempt, and the agent's steer buffer is
the real ``AIAgent`` buffer.

- An attempt is buffered under its own ticket only while intake is open, the agent exists, the frame
  names this interpreter's exact destination (profile, session, generation, board, task, run) and the
  bound claim is the card's live run. Duplicates stay duplicates. Nothing is latched or kept on the
  board.
- Intake check, buffering and close share one lock: an attempt after close is refused; one before
  close is settled by the close.
- Settlement is honest: ``unconsumed`` when withdrawn unread, ``consumed`` only after a completed,
  flushed turn that left nothing unrun, nothing otherwise (unknown).

Owner (``gateway.session_managed_worker.ManagedWorker``): before ``ready`` and for any other
destination nothing is queued; verdicts and settlements reach the forwarded attempt; a lost child
leaves its outcome unknown.

Order (NC15-PEER-01): the child writes an attempt's verdict before its settlement can be decided, so a
close racing a verdict waits for it; the owner still resolves an attempt whose settlement overtook its
verdict, and a waiter that gave up neither loses its settlement nor fails the owner's frame loop.
"""
import asyncio
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import managed_worker as worker
from gateway.session_managed_worker import ManagedWorker, _control_frame
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
from hermes_state_runtime import RuntimeStoreError
from tests.agent.test_steer import _bare_agent
from tests.waiting import wait_until

PROFILE = '/profiles/p'
PRIVATE = 'PRIVATE_PEER_BODY'


class Channel:
    def __init__(self):
        self.frames = []

    def send(self, kind, **payload):
        self.frames.append({'type': kind, **payload})

    def verdicts(self):
        return [(f['id'], f['n'], f['status']) for f in self.frames if f['type'] == 'peer_message.verdict']


@pytest.fixture
def card(tmp_path, monkeypatch):
    """A real board with a claimed card attempt bound to this interpreter."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title='peer', assignee='default')
        task = kb.claim_task(conn, task_id, claimer='private-host:owner')
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'worker_bound', {'pid': os.getpid(), 'claim_lock': task.claim_lock},
                             run_id=task.current_run_id)
    return {'profile': PROFILE, 'session_id': 'physical-s', 'generation': 4, 'board_db': str(path), 'db': str(path),
            'task_id': task_id, 'run_id': task.current_run_id, 'claim_lock': task.claim_lock}


@pytest.fixture
def child(card):
    """WorkerControls reading a real pipe, with a real steer buffer adopted."""
    read_fd, write_fd = os.pipe()
    channel = Channel()
    controls = worker.WorkerControls(channel, 'route', os.fdopen(read_fd, 'rb'))
    controls.accept_peer_messages(card)
    agent = _bare_agent()
    controls.adopt(agent)
    pipe = os.fdopen(write_fd, 'wb')

    def send(frame):
        before = len(channel.verdicts())
        pipe.write(worker.encode_frame(frame))
        pipe.flush()
        wait_until(lambda: len(channel.verdicts()) > before, timeout=10, what='peer verdict')
        return channel.verdicts()[-1][2]

    yield SimpleNamespace(controls=controls, agent=agent, channel=channel, send=send, card=card)
    pipe.write(worker.encode_frame({'type': 'finish'}))
    pipe.flush()
    controls.reader.join(timeout=10)
    pipe.close()


def frame(card, n=1, delivery_id='d1', text=PRIVATE, **changes):
    return {'type': 'peer_message', 'id': delivery_id, 'n': n, 'text': text,
            'author': {'id': 'bot-1', 'name': 'Peer', 'is_bot': True},
            **{key: card[key] for key in worker._PEER_DESTINATION}, **changes}


def board_cells(card):
    with connect_closing(Path(card['db'])) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return [str(cell) for table in tables for row in conn.execute(f'SELECT * FROM "{table}"') for cell in row]


COMPLETED = {'completed': True}


def test_an_attempt_is_buffered_under_its_own_ticket_and_a_duplicate_stays_a_duplicate(child):
    assert child.send(frame(child.card, n=1)) == 'steered'
    assert child.send(frame(child.card, n=2)) == 'steered'
    drained = child.agent._drain_pending_steer()
    assert drained.count(PRIVATE) == 2
    # Both drained by a completed turn that closed intake with nothing unrun: both consumed.
    assert child.controls.settle(dict(COMPLETED), may_continue=False) is None
    assert child.controls.peer_settlements(COMPLETED) == [('d1', 1, 'consumed'), ('d1', 2, 'consumed')]
    assert not any(PRIVATE in cell for cell in board_cells(child.card))


@pytest.mark.parametrize('key, value', [
    ('profile', '/profiles/other'), ('session_id', 'other-s'), ('generation', 5),
    ('board_db', '/elsewhere/kanban.db'), ('task_id', 't_other'), ('run_id', 999),
])
def test_an_attempt_naming_any_other_destination_is_rejected_unbuffered(child, key, value):
    assert child.send(frame(child.card, **{key: value})) == 'rejected'
    assert child.agent._drain_pending_steer() is None
    assert child.controls.peer_tickets == {}


def test_before_the_agent_exists_an_attempt_is_rejected_never_latched(card):
    read_fd, write_fd = os.pipe()
    channel = Channel()
    controls = worker.WorkerControls(channel, 'route', os.fdopen(read_fd, 'rb'))
    controls.accept_peer_messages(card)
    with os.fdopen(write_fd, 'wb') as pipe:
        pipe.write(worker.encode_frame(frame(card)) + worker.encode_frame({'type': 'finish'}))
    controls.reader.join(timeout=10)
    assert channel.verdicts() == [('d1', 1, 'rejected')]
    assert controls.latched == [] and controls.peer_tickets == {}


def test_a_worker_with_no_card_attempt_takes_no_peer_message(card):
    read_fd, write_fd = os.pipe()
    channel = Channel()
    controls = worker.WorkerControls(channel, 'route', os.fdopen(read_fd, 'rb'))
    controls.adopt(_bare_agent())
    with os.fdopen(write_fd, 'wb') as pipe:
        pipe.write(worker.encode_frame(frame(card)) + worker.encode_frame({'type': 'finish'}))
    controls.reader.join(timeout=10)
    assert channel.verdicts() == [('d1', 1, 'rejected')]


@pytest.mark.parametrize('change', ['reclaimed', 'expired', 'other_interpreter'])
def test_the_board_fence_rejects_an_attempt_once_the_bound_claim_is_not_the_live_run(child, change):
    card = child.card
    with connect_closing(Path(card['db'])) as conn, kb.write_txn(conn):
        if change == 'reclaimed':
            conn.execute("UPDATE tasks SET claim_lock = 'someone-else' WHERE id = ?", (card['task_id'],))
        elif change == 'expired':
            conn.execute('UPDATE tasks SET claim_expires = 1 WHERE id = ?', (card['task_id'],))
            conn.execute('UPDATE task_runs SET claim_expires = 1 WHERE id = ?', (card['run_id'],))
        else:
            kb._append_event(conn, card['task_id'], 'worker_bound', {'pid': os.getpid() + 1, 'claim_lock': card['claim_lock']},
                             run_id=card['run_id'])
    assert child.send(frame(card)) == 'rejected'
    assert child.agent._drain_pending_steer() is None


def test_an_attempt_before_close_is_settled_by_it_and_one_after_close_is_refused(child):
    # Frame before close: buffered; the close withdraws it unread (unconsumed, never run).
    assert child.send(frame(child.card, n=1)) == 'steered'
    assert child.controls.settle(dict(COMPLETED), may_continue=False) is None
    # Frame after close: refused, nothing buffered that no settlement could withdraw.
    assert child.send(frame(child.card, n=2)) == 'rejected'
    assert child.agent._drain_pending_steer() is None
    assert child.controls.peer_settlements(COMPLETED) == [('d1', 1, 'unconsumed')]


def test_close_waits_for_an_attempt_holding_intake_and_then_settles_it(child, monkeypatch):
    """The intake lock orders a racing close after an attempt that is mid-fence: the close sees it."""
    from gateway import session_kanban
    entered, release = threading.Event(), threading.Event()
    real = session_kanban.steer_peer_if_bound

    def slow(context, steer):
        entered.set()
        release.wait(10)
        return real(context, steer)
    monkeypatch.setattr(session_kanban, 'steer_peer_if_bound', slow)
    attempt = threading.Thread(target=child.send, args=(frame(child.card),))
    attempt.start()
    assert entered.wait(10)
    closed = []
    closer = threading.Thread(target=lambda: closed.append(child.controls.settle(dict(COMPLETED), may_continue=False)))
    closer.start()
    closer.join(0.3)
    assert closer.is_alive()  # blocked behind the attempt holding intake
    release.set()
    attempt.join(10)
    closer.join(10)
    assert child.channel.verdicts() == [('d1', 1, 'steered')]
    assert child.controls.peer_settlements(COMPLETED) == [('d1', 1, 'unconsumed')]


@pytest.mark.parametrize('result', [
    {'completed': False, 'failed': True},  # e.g. the final usage flush failed
    {'completed': True, 'interrupted': True},
])
def test_a_drained_attempt_is_consumed_only_after_a_completed_flushed_turn(child, result):
    assert child.send(frame(child.card, n=1)) == 'steered'
    child.agent._drain_pending_steer()
    assert child.send(frame(child.card, n=2)) == 'steered'
    assert child.controls.settle(dict(result), may_continue=False) is None
    # Attempt 1 ran in a turn that did not complete cleanly: unknown, so no settlement. Attempt 2 unread.
    assert child.controls.peer_settlements(result) == [('d1', 2, 'unconsumed')]


def test_unrun_correction_text_leaves_a_drained_attempt_unknown(child):
    assert child.send(frame(child.card)) == 'steered'
    child.agent._drain_pending_steer()
    child.agent.steer('a correction nothing ran')
    assert child.controls.settle(dict(COMPLETED), may_continue=False) is None
    assert child.controls.peer_settlements(COMPLETED) == []


def test_a_follow_up_turn_never_carries_an_unread_private_attempt(child):
    assert child.send(frame(child.card)) == 'steered'
    child.agent.steer('ordinary late steer')
    follow_up = child.controls.settle(dict(COMPLETED), may_continue=True)
    assert follow_up is not None and PRIVATE not in follow_up and 'ordinary late steer' in follow_up
    assert child.controls.settle(dict(COMPLETED), may_continue=False) is None
    assert child.controls.peer_settlements(COMPLETED) == [('d1', 1, 'unconsumed')]


def test_a_stopped_turn_settles_its_dropped_attempt_unconsumed(child):
    assert child.send(frame(child.card)) == 'steered'
    child.controls.stopped.set()
    assert child.controls.settle(dict(COMPLETED), may_continue=True) is None
    assert child.controls.peer_settlements({'completed': False, 'interrupted': True}) == [('d1', 1, 'unconsumed')]


# Owner side -----------------------------------------------------------------------------------


class Message(SimpleNamespace):
    pass


def owner_worker(card, *, ready=True):
    managed = ManagedWorker(SimpleNamespace(pid=1))
    managed.ready = ready
    managed.session_id = 'owner-s'
    managed.scope = {'profile_id': PROFILE, 'session_id': 'physical-s', 'generation': 4}
    managed.policy = SimpleNamespace(kanban_json=json.dumps({'db': card['db'], 'task_id': card['task_id'],
                                                             'run_id': card['run_id']}))
    return managed


def message(card, **changes):
    fields = dict(delivery_id='d1', text=PRIVATE, sender=SimpleNamespace(transport='api_key', author={'id': 'bot-1'}),
                  profile=PROFILE, session_id='owner-s', board_db=card['db'], task_id=card['task_id'],
                  run_id=card['run_id'], generation=4)
    return Message(**{**fields, **changes})


def queued(managed):
    frames = []
    while not managed.commands.empty():
        frames.append(managed.commands.get_nowait())
    return frames


def test_owner_forwards_only_its_exact_destination_after_ready(card):
    async def run():
        managed = owner_worker(card)
        future = managed.peer_message(message(card), 1)
        assert queued(managed) == [{'type': 'peer_message', 'id': 'd1', 'n': 1, 'text': PRIVATE, 'author': {'id': 'bot-1'},
                                    'profile': PROFILE, 'session_id': 'physical-s', 'generation': 4,
                                    'board_db': card['db'], 'task_id': card['task_id'], 'run_id': card['run_id']}]
        assert not future.done()
        for change in ({'profile': '/profiles/other'}, {'profile': None}, {'session_id': 'other'}, {'generation': 5},
                       {'board_db': '/elsewhere'}, {'task_id': 't_x'}, {'run_id': 999}):
            refused = managed.peer_message(message(card, delivery_id='d2', **change), 1)
            assert refused.result() == 'rejected'
        unready = owner_worker(card, ready=False)
        assert unready.peer_message(message(card), 1).result() == 'rejected'
        assert queued(managed) == [] and queued(unready) == []
    asyncio.run(run())


def test_owner_verdict_then_settlement_runs_settle_off_the_loop_once(card):
    async def run():
        managed = owner_worker(card)
        settled = []
        future = managed.peer_message(message(card), 1, settle=lambda outcome: settled.append(
            (outcome, threading.current_thread() is threading.main_thread())))
        assert managed.answer_peer_message({'type': 'peer_message.verdict', 'id': 'd1', 'n': 1, 'status': 'steered'})
        assert await future == 'steered'
        assert await managed.settle_peer_message({'type': 'peer_settled', 'id': 'd1', 'n': 1, 'outcome': 'consumed'})
        assert await managed.settle_peer_message({'type': 'peer_settled', 'id': 'd1', 'n': 1, 'outcome': 'unconsumed'})
        assert settled == [('consumed', False)]
        # A refused attempt is never settled.
        rejected = managed.peer_message(message(card), 2, settle=lambda outcome: settled.append(outcome))
        managed.answer_peer_message({'type': 'peer_message.verdict', 'id': 'd1', 'n': 2, 'status': 'rejected'})
        assert await rejected == 'rejected'
        await managed.settle_peer_message({'type': 'peer_settled', 'id': 'd1', 'n': 2, 'outcome': 'consumed'})
        assert settled == [('consumed', False)]
    asyncio.run(run())


@pytest.mark.parametrize('unread', [True, False])
def test_owner_leaves_a_lost_childs_attempt_unknown_and_an_unread_one_rejected(card, unread):
    async def run():
        managed = owner_worker(card)
        future = managed.peer_message(message(card), 1)
        managed.abandon_corrections(unread=unread)
        if unread:
            assert await future == 'rejected'
        else:
            with pytest.raises(RuntimeStoreError, match='peer_message_outcome_unknown'):
                await future
    asyncio.run(run())


def test_owner_raises_only_when_nothing_was_queued(card):
    async def run():
        managed = owner_worker(card)
        for index in range(managed.commands.maxsize):
            managed.commands.put_nowait({'type': 'filler', 'i': index})
        with pytest.raises(RuntimeStoreError, match='worker_control_backpressure'):
            managed.peer_message(message(card), 1)
        assert managed.peer_messages == {}
    asyncio.run(run())


# Verdict and settlement order (NC15-PEER-01) ------------------------------------------------------


def wire_frames(stream):
    stream.seek(0)
    frames = []
    while True:
        try:
            frames.append(worker.read_frame(stream))
        except EOFError:
            return frames


@pytest.mark.parametrize('drained', [False, True])
def test_a_close_racing_a_verdict_waits_for_it_and_the_wire_keeps_each_verdict_before_its_settlement(card, drained):
    """Real reader pipe, real ``WorkerChannel`` frames, real board fence and steer buffer, real owner
    frame handling: A's verdict is paused mid-publication while the turn thread closes intake."""
    async def run():
        owner = owner_worker(card)
        settled = []
        b = owner.peer_message(message(card, delivery_id='b'), 1, settle=lambda outcome: settled.append(('b', outcome)))
        a = owner.peer_message(message(card), 1, settle=lambda outcome: settled.append(('a', outcome)))
        publishing, release = threading.Event(), threading.Event()

        class Paused(worker.WorkerChannel):
            def send(self, kind, **payload):
                if kind == 'peer_message.verdict' and payload['id'] == 'd1':
                    publishing.set()
                    assert release.wait(10)
                super().send(kind, **payload)

        stream = io.BytesIO()
        channel = Paused(stream)
        read_fd, write_fd = os.pipe()
        controls = worker.WorkerControls(channel, 'route', os.fdopen(read_fd, 'rb'))
        controls.accept_peer_messages(card)
        agent = _bare_agent()
        controls.adopt(agent)
        with os.fdopen(write_fd, 'wb') as pipe:
            try:
                for command in queued(owner):  # B, then A
                    pipe.write(worker.encode_frame(command))
                pipe.flush()
                assert publishing.wait(10)
                if drained:
                    assert PRIVATE in agent._drain_pending_steer()
                closed = []
                closer = threading.Thread(target=lambda: closed.append(controls.settle(dict(COMPLETED), may_continue=False)))
                closer.start()
                closer.join(0.3)
                assert closer.is_alive()  # the close cannot settle A while A's verdict is unwritten
            finally:
                release.set()
            closer.join(10)
            assert closed == [None]
            for delivery_id, n, outcome in controls.peer_settlements(COMPLETED):
                channel.send('peer_settled', id=delivery_id, n=n, outcome=outcome)
            retry = owner.peer_message(message(card), 2, settle=lambda outcome: settled.append(('a2', outcome)))
            pipe.write(worker.encode_frame(queued(owner)[0]) + worker.encode_frame({'type': 'finish'}))
            pipe.flush()
            controls.reader.join(10)
            assert not controls.reader.is_alive()
        wire = wire_frames(stream)
        outcome = 'consumed' if drained else 'unconsumed'
        assert [(f['type'], f['id'], f['n'], f.get('status', f.get('outcome'))) for f in wire] == [
            ('peer_message.verdict', 'b', 1, 'steered'), ('peer_message.verdict', 'd1', 1, 'steered'),
            ('peer_settled', 'b', 1, outcome), ('peer_settled', 'd1', 1, outcome),
            ('peer_message.verdict', 'd1', 2, 'rejected')]  # the retry after close: refused, never buffered
        for frame in wire:
            assert await _control_frame(None, None, None, owner, frame)
        owner.abandon_corrections(unread=True)  # the normal finish
        assert (a.result(), b.result(), retry.result()) == ('steered', 'steered', 'rejected')
        assert settled == [('b', outcome), ('a', outcome)]
        assert owner.peer_messages == {}
    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['consumed', 'unconsumed'])
def test_owner_settlement_overtaking_its_verdict_resolves_the_attempt_and_settles_it_once(card, outcome):
    async def run():
        owner = owner_worker(card)
        settled = []
        a = owner.peer_message(message(card), 1, settle=lambda result: settled.append(('a', result)))
        b = owner.peer_message(message(card, delivery_id='b'), 1, settle=lambda result: settled.append(('b', result)))
        retry = owner.peer_message(message(card), 2, settle=lambda result: settled.append(('a2', result)))
        for frame in ({'type': 'peer_settled', 'id': 'd1', 'n': 1, 'outcome': outcome},
                      {'type': 'peer_message.verdict', 'id': 'd1', 'n': 1, 'status': 'steered'},
                      {'type': 'peer_message.verdict', 'id': 'b', 'n': 1, 'status': 'steered'},
                      {'type': 'peer_message.verdict', 'id': 'd1', 'n': 2, 'status': 'steered'},
                      {'type': 'peer_settled', 'id': 'b', 'n': 1, 'outcome': 'consumed'},
                      {'type': 'peer_settled', 'id': 'd1', 'n': 2, 'outcome': 'unconsumed'},
                      {'type': 'peer_settled', 'id': 'd1', 'n': 1, 'outcome': outcome}):  # a duplicate settles nothing
            assert await _control_frame(None, None, None, owner, frame)
        owner.abandon_corrections(unread=True)
        assert (a.result(), b.result(), retry.result()) == ('steered', 'steered', 'steered')
        assert settled == [('a', outcome), ('b', 'consumed'), ('a2', 'unconsumed')]
    asyncio.run(run())


def test_owner_a_waiter_that_gave_up_keeps_its_settlement_and_the_frame_loop(card):
    async def run():
        owner = owner_worker(card)
        settled = []
        a = owner.peer_message(message(card), 1, settle=lambda result: settled.append(('a', result)))
        b = owner.peer_message(message(card, delivery_id='b'), 1, settle=lambda result: settled.append(('b', result)))
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(a, 0.01)  # an unshielded caller's timeout cancels the future itself
        assert a.cancelled()
        for frame in ({'type': 'peer_message.verdict', 'id': 'd1', 'n': 1, 'status': 'steered'},
                      {'type': 'peer_settled', 'id': 'd1', 'n': 1, 'outcome': 'consumed'},
                      {'type': 'peer_message.verdict', 'id': 'b', 'n': 1, 'status': 'steered'},
                      {'type': 'peer_settled', 'id': 'b', 'n': 1, 'outcome': 'unconsumed'}):
            assert await _control_frame(None, None, None, owner, frame)
        assert b.result() == 'steered'
        assert settled == [('a', 'consumed'), ('b', 'unconsumed')]
    asyncio.run(run())
