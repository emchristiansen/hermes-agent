"""Every message reaching a busy session steers into its running turn (Eric, 2026-11-04).

"I want literally every message to always be a steering message." A message admitted while a claim
runs goes to that claim's agent through the existing queued-steer transfer, without any client asking
for it, and is consumed at the turn's next tool/result boundary; an idle session still runs it as its
own turn at once. Real authority, admission ledger, transfer and settlement; the turn body adopts the
real steer seam (``BufferedAgent``) and its boundary is the canonical
``apply_pending_steer_to_tool_results``.
"""
import asyncio

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from gateway import session_finite
from tests.gateway.test_prompt_queued_controls import _call, _connection, _ledger, interactive
from tests.gateway.test_prompt_queued_transfer import BufferedAgent, _drained
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit
from tests.gateway.test_session_bot_retry import KEY, _admissions, _settle_quietly, bot  # the fixture
from tests.waiting import async_wait_until


def _boundary(agent):
    """The turn's tool/result boundary: the steer, if any, appended after a tool result."""
    messages = [{'role': 'assistant', 'tool_calls': []}, {'role': 'tool', 'content': 'fixture result'}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    return messages[-1]['content'] if messages[-1]['role'] == 'user' else None


def _status(db, request_id):
    return next((status, outcome) for rid, status, outcome in _ledger(db) if rid == request_id)


class HeldTurn:
    """Each claimed turn adopts ``agent`` (``adopt``: only once ``adopt_now`` is set), records its text,
    and the first one stays running until ``boundary`` is set, then crosses one boundary."""

    def __init__(self, monkeypatch, *, adopt_late=False):
        self.agent, self.ran, self.taken = BufferedAgent(), [], []
        self.running, self.adopt_now, self.boundary = asyncio.Event(), asyncio.Event(), asyncio.Event()
        if not adopt_late:
            self.adopt_now.set()

        async def execute(authority, ref, row):
            self.ran.append(row['payload']['text'])
            if len(self.ran) > 1:
                return 'done'
            self.running.set()
            await asyncio.wait_for(self.adopt_now.wait(), 30)  # the turn is still building its agent
            authority.adopt_agent(ref.session_id, row['generation'], self.agent)
            await asyncio.wait_for(self.boundary.wait(), 30)
            self.taken.append(_boundary(self.agent))
            return 'done'
        monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)


async def _transferred(db, request_id):
    await async_wait_until(lambda: _status(db, request_id) == ('terminal', 'steering'), timeout=10,
                           what=f'{request_id} to transfer to the running turn')


@pytest.mark.asyncio
async def test_a_busy_prompt_steers_into_the_running_turn_without_any_steer_request(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = HeldTurn(monkeypatch)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.running.wait(), 30)
        await _submit(authority, 'follow-up', 'and the staging logs')
        await _transferred(db, 'follow-up')
        turn.boundary.set()
        await _drained(authority)
    assert turn.ran == ['running'], 'never its own turn after the running one'
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'steered')]
    [content] = turn.taken
    assert 'and the staging logs' in content


@pytest.mark.asyncio
async def test_a_prompt_admitted_before_the_turn_adopted_its_agent_steers_at_adoption(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = HeldTurn(monkeypatch, adopt_late=True)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.running.wait(), 30)
        await _submit(authority, 'early')
        await asyncio.sleep(0.05)
        assert _status(db, 'early') == ('queued', None), 'no agent to reach yet: it waits for adoption'
        turn.adopt_now.set()
        await _transferred(db, 'early')
        turn.boundary.set()
        await _drained(authority)
    assert turn.ran == ['running']
    assert _status(db, 'early') == ('terminal', 'steered')
    [content] = turn.taken
    assert 'early' in content


@pytest.mark.asyncio
async def test_an_idle_session_runs_a_prompt_as_its_own_turn_at_once(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = HeldTurn(monkeypatch)
    turn.boundary.set()
    with db:
        await _submit(authority, 'first')
        await _drained(authority)
        await _submit(authority, 'second')
        await _drained(authority)
    assert turn.ran == ['first', 'second']
    assert _ledger(db) == [('first', 'terminal', 'completed'), ('second', 'terminal', 'completed')]


@pytest.mark.asyncio
async def test_a_message_the_turn_never_read_runs_once_as_its_own_turn(tmp_path, monkeypatch):
    """Arrives after the turn's last boundary: queued again in its own place, run exactly once."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    agent, ran, release, done_boundary = BufferedAgent(), [], asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        if len(ran) > 1:
            return 'done'
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        _boundary(agent)  # the last boundary, before the message arrives
        done_boundary.set()
        await asyncio.wait_for(release.wait(), 30)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(done_boundary.wait(), 30)
        await _submit(authority, 'late')
        await _transferred(db, 'late')
        release.set()
        await _drained(authority)
        await async_wait_until(lambda: len(ran) == 2, timeout=10, what='the returned message ran')
        await _drained(authority)
    assert ran == ['running', 'late']
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('late', 'terminal', 'completed')]


@pytest.mark.asyncio
async def test_a_whole_job_row_still_runs_as_its_own_turn(tmp_path, monkeypatch):
    """A session launched by a job producer (here cron) reads each turn's own result: not a message."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db, source='cron')
    turn = HeldTurn(monkeypatch)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.running.wait(), 30)
        await _submit(authority, 'job')
        await asyncio.sleep(0.1)
        assert _status(db, 'job') == ('queued', None)
        turn.boundary.set()
        await _drained(authority)
        await async_wait_until(lambda: len(turn.ran) == 2, timeout=10, what='the job ran')
        await _drained(authority)
    assert turn.ran == ['running', 'job']
    assert _status(db, 'job') == ('terminal', 'completed')


@pytest.mark.asyncio
@pytest.mark.parametrize('asked', ['queue', 'interrupt'])
async def test_every_busy_preference_reads_steer(tmp_path, monkeypatch, asked):
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        conn = _connection(authority)
        written = await _call(conn, 'config.set', key='busy', value=asked)
        read = await _call(conn, 'config.get', key='busy')
    assert written.get('result', {}).get('value') == 'steer', written
    assert read.get('result', {}).get('value') == 'steer', read


@pytest.mark.asyncio
async def test_a_bot_chat_delivery_to_a_busy_bot_chat_steers_as_its_sender_reached_it(bot, monkeypatch):
    """The legacy Desktop relay's chain past its WS hop: ``session_bot.deliver`` with the server-derived
    bot author -> ``_admit`` -> ``admit_automation``. The second delivery reaches the running turn; its
    sender's receipt reports what the ledger knows: no reply of its own (``ambiguous``), never a
    second turn."""
    from gateway.session_bot import _result, deliver
    from tools.bot_live_delivery import read_delivery_result
    agent, ran, running, boundary, taken = BufferedAgent(), [], asyncio.Event(), asyncio.Event(), []

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        if len(ran) > 1:
            return 'own turn'
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        running.set()
        await asyncio.wait_for(boundary.wait(), 30)
        taken.append(_boundary(agent))
        return 'pong'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    author = {'id': 'relay:abc', 'name': 'Desktop relay', 'is_bot': True}
    first = asyncio.create_task(deliver(bot.connection, dict(id='a' * 32, profile='default', message='ping')))
    await asyncio.wait_for(running.wait(), 30)
    second = await deliver(bot.connection, dict(id=KEY, profile='default', message='second', author=author))
    await async_wait_until(lambda: any(r['request_id'] == 'bot:' + KEY and r['status'] == 'terminal'
                                       for r in _admissions(bot)), timeout=10, what='the delivery to transfer')
    boundary.set()
    await first
    await _settle_quietly(bot)
    rows = {r['request_id']: (r['status'], r['outcome']) for r in _admissions(bot)}
    receipt = _result(bot.authority, await asyncio.to_thread(read_delivery_result, bot.home, KEY))

    assert ran == ['ping'], 'the delivery never ran as its own turn'
    assert rows == {'bot:' + 'a' * 32: ('terminal', 'completed'), 'bot:' + KEY: ('terminal', 'steered')}
    [content] = taken
    assert 'second' in content, 'the delivered text, at the running turn boundary'
    assert second['admission_id'] and second['status'] in {'queued', 'claimed'}, second
    # The truthful limit of the existing receipt contract: no reply belongs to a steered delivery.
    assert (receipt['status'], receipt['reply']) == ('ambiguous', ''), receipt


@pytest.mark.asyncio
async def test_a_client_steer_of_a_row_the_pass_already_transferred_replays_it_once(tmp_path, monkeypatch):
    """An older client still sends ``prompt.steer_queued`` for the row it showed: it answers with the
    committed transfer, and the turn receives the text once."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = HeldTurn(monkeypatch)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.running.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        await _transferred(db, 'follow-up')
        reply = await _call(_connection(authority), 'prompt.steer_queued', admission_id=queued.admission_id)
        turn.boundary.set()
        await _drained(authority)
    assert 'error' not in reply and reply['result']['admission_id'] == queued.admission_id, reply
    [content] = turn.taken
    assert content.count('follow-up') == 1
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'steered')]
