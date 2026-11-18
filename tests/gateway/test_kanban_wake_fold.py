"""A busy session holds at most one queued Kanban wake, carrying every notice still waiting.

Each Kanban notice used to become its own queued admission. On 2026-11-04 the Interlocutor's
local session held 31 queued notices (from 144 notify+wake subscriptions, the backlog released when
local wakes started being delivered); at about a minute a turn, every slot a turn freed was
refilled within one 5 s notifier tick, so its ``ADMISSION_QUEUE_MAX`` FIFO stayed full and refused
human input. A new wake now folds the route's queued wakes into its own admission, oldest notice
first, cancelling them in the same transaction; a human input already queued keeps its place.
"""
import asyncio
import json

import pytest
import pytest_asyncio

from tests.waiting import async_wait_until


@pytest_asyncio.fixture
async def busy_local(tmp_path, monkeypatch):
    """A local session whose first turn blocks until ``release`` is set."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    turns, release = [], asyncio.Event()

    async def answer(event):
        turns.append(event.text)
        if len(turns) == 1:
            await release.wait()
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", answer)
    viewer = SessionActor("test-owner", "default", frozenset(
        {"session:create", "session:submit", "session:read"}), "viewer")
    ref = create_local_session(authority, viewer, dict(request_id="cli", source="cli", model="fixture", toolsets=[]))
    live = authority.sessions[ref.session_id]
    adapter = runner._adapter_for_source(live.source)
    # What KanbanNotifier.wake() builds from a LOCAL subscription row.
    source = SessionSource(platform=Platform.LOCAL, chat_id=live.source.chat_id, chat_type="dm",
                           user_id=live.source.user_id)

    class Busy:
        pass
    busy = Busy()
    busy.__dict__.update(runner=runner, db=db, authority=authority, ref=ref, live=live, adapter=adapter,
                         source=source, turns=turns, release=release)
    try:
        yield busy
    finally:
        release.set()
        db.close()


def _rows(busy):
    from hermes_state_runtime import list_session_admissions
    return list_session_admissions(busy.db, session_id=busy.ref.session_id, pending_only=False)


def _queued_wakes(busy):
    return [row for row in _rows(busy) if row['status'] == 'queued'
            and json.loads(row['request_id'])[:1] == ['kanban-wake']]


async def _occupy(busy):
    from gateway.wake import deliver_wake
    await deliver_wake(busy.adapter, text="first", source=busy.source)
    await async_wait_until(lambda: busy.turns == ["first"], timeout=10, what="the blocking first turn")


async def _human(busy, text):
    """A client input admitted like the local viewer's submit, not a producer wake."""
    from hermes_state_runtime import admit_session_input
    row = admit_session_input(busy.db, epoch=busy.authority.epoch, principal_id="human:test",
                              session_id=busy.ref.session_id, request_id="human-" + text,
                              payload={"text": text})
    return row


@pytest.mark.asyncio
async def test_queued_wakes_fold_into_one_admission_oldest_first(busy_local):
    from gateway.wake import deliver_wake
    busy = busy_local
    await _occupy(busy)
    for n in (1, 2, 3):
        await deliver_wake(busy.adapter, text=f"kanban: t_{n} completed", source=busy.source)

    waiting = _queued_wakes(busy)
    assert len(waiting) == 1
    assert waiting[0]['payload']['text'] == "kanban: t_1 completed\n\nkanban: t_2 completed\n\nkanban: t_3 completed"
    cancelled = [row for row in _rows(busy) if row['outcome'] == 'cancelled']
    assert len(cancelled) == 2
    # Supersession receipt: the survivor names every notice it carries, each exactly once.
    carried = waiting[0]['payload']['local_automation_v1']['identities']
    assert sorted(carried) == sorted([row['request_id'] for row in cancelled] + [waiting[0]['request_id']])

    busy.release.set()
    await async_wait_until(lambda: len(busy.turns) == 2, timeout=10, what="the folded wake turn")
    assert busy.turns[1] == waiting[0]['payload']['text']


@pytest.mark.asyncio
async def test_a_human_input_queued_between_wakes_keeps_its_place(busy_local):
    from gateway.wake import deliver_wake
    busy = busy_local
    await _occupy(busy)
    await deliver_wake(busy.adapter, text="kanban: t_1 completed", source=busy.source)
    human = await _human(busy, "eric asks")
    await deliver_wake(busy.adapter, text="kanban: t_2 completed", source=busy.source)

    queued = [row for row in _rows(busy) if row['status'] == 'queued']
    assert queued[0]['admission_id'] == human['admission_id']
    assert [row['payload']['text'] for row in queued[1:]] == ["kanban: t_1 completed\n\nkanban: t_2 completed"]


@pytest.mark.asyncio
async def test_a_diagnostic_wake_never_folds_into_a_result_wake(busy_local):
    from gateway.wake import deliver_wake
    busy = busy_local
    await _occupy(busy)
    await deliver_wake(busy.adapter, text="result notice", source=busy.source)
    await deliver_wake(busy.adapter, text="diagnostic notice", source=busy.source, notification_category="diagnostic")
    await deliver_wake(busy.adapter, text="second result", source=busy.source)

    assert sorted(row['payload']['text'] for row in _queued_wakes(busy)) == [
        "diagnostic notice", "result notice\n\nsecond result"]


@pytest.mark.asyncio
async def test_a_wake_whose_folded_row_was_claimed_first_cancels_nothing(busy_local, monkeypatch):
    from gateway import session_automation
    from gateway.wake import WakeNotAccepted, deliver_wake
    busy = busy_local
    await _occupy(busy)
    await deliver_wake(busy.adapter, text="kanban: t_1 completed", source=busy.source)
    fold = session_automation._fold_queued_kanban_wakes

    async def claimed_meanwhile(authority, entry, identity, text, category):
        folded = await fold(authority, entry, identity, text, category)
        with busy.db._lock:  # the drain starts the folded wake between the read and the write
            busy.db._conn.execute("UPDATE session_admissions SET status='started',owner_epoch=?,generation=1 "
                                  "WHERE admission_id=?", (busy.authority.epoch, *folded[1]))
            busy.db._conn.commit()
        return folded
    monkeypatch.setattr(session_automation, '_fold_queued_kanban_wakes', claimed_meanwhile)

    with pytest.raises(WakeNotAccepted) as refused:
        await deliver_wake(busy.adapter, text="kanban: t_2 completed", source=busy.source)
    assert refused.value.reason == 'stale_generation'  # transient: the notifier rewinds and retries
    texts = {row['payload']['text']: (row['status'], row['outcome']) for row in _rows(busy)}
    assert texts["kanban: t_1 completed"] == ('started', None)
    assert "kanban: t_1 completed\n\nkanban: t_2 completed" not in texts


@pytest.mark.asyncio
async def test_a_refused_fold_cancels_nothing_and_leaves_the_queue_as_it_was(busy_local, monkeypatch):
    """A fold whose write is refused (here the owner admission gate) rolls back every cancellation."""
    from gateway.wake import WakeNotAccepted, deliver_wake
    from hermes_state_runtime import RuntimeStoreError
    busy = busy_local
    await _occupy(busy)
    for n in (1, 2):
        await deliver_wake(busy.adapter, text=f"kanban: t_{n} completed", source=busy.source)
    before = [(row['admission_id'], row['status'], row['outcome']) for row in _rows(busy)]
    gate = busy.authority._admission_gate

    def refusing(authorize=None, **kwargs):
        inner = gate(authorize, **kwargs)

        def guard(conn):
            inner(conn)  # the fold's cancellations run here, inside the write transaction
            raise RuntimeStoreError('runtime_draining')
        return guard
    monkeypatch.setattr(busy.authority, '_admission_gate', refusing)

    with pytest.raises(WakeNotAccepted):
        await deliver_wake(busy.adapter, text="kanban: t_3 completed", source=busy.source)
    assert [(row['admission_id'], row['status'], row['outcome']) for row in _rows(busy)] == before


@pytest.mark.asyncio
async def test_a_wake_folds_a_full_fifo_of_legacy_wakes(busy_local, monkeypatch):
    """The 2026-11-04 shape: a FIFO already full (ADMISSION_QUEUE_MAX) of unfolded wakes admitted
    one per notice, plus a human input behind some of them. The next wake needs only the room its
    fold leaves, so it is admitted, and the human input moves to the head."""
    from gateway import session_automation
    from gateway.wake import deliver_wake
    from hermes_state_runtime import ADMISSION_QUEUE_MAX
    busy = busy_local
    await _occupy(busy)
    fold = session_automation._fold_queued_kanban_wakes

    async def unfolded(authority, entry, identity, text, category):  # the pre-fold producer
        return text, (), ()
    monkeypatch.setattr(session_automation, '_fold_queued_kanban_wakes', unfolded)
    for n in range(ADMISSION_QUEUE_MAX - 1):
        if n == 10:
            human = await _human(busy, "eric asks")
        await deliver_wake(busy.adapter, text=f"kanban: t_{n} completed", source=busy.source)
    assert len([row for row in _rows(busy) if row['status'] == 'queued']) == ADMISSION_QUEUE_MAX
    monkeypatch.setattr(session_automation, '_fold_queued_kanban_wakes', fold)

    await deliver_wake(busy.adapter, text="kanban: t_new completed", source=busy.source)
    queued = [row for row in _rows(busy) if row['status'] == 'queued']
    assert [row['admission_id'] for row in queued][:1] == [human['admission_id']]
    assert len(queued) == 2
    assert queued[1]['payload']['text'] == "\n\n".join(
        [*(f"kanban: t_{n} completed" for n in range(ADMISSION_QUEUE_MAX - 1)), "kanban: t_new completed"])
    assert len(queued[1]['payload']['local_automation_v1']['identities']) == ADMISSION_QUEUE_MAX


@pytest.mark.asyncio
async def test_a_wake_with_nothing_to_fold_still_respects_a_full_fifo(busy_local):
    """No bypass: the cap still refuses a wake when the FIFO is full of inputs it cannot fold."""
    from gateway.wake import WakeNotAccepted, deliver_wake
    from hermes_state_runtime import ADMISSION_QUEUE_MAX
    busy = busy_local
    await _occupy(busy)
    for n in range(ADMISSION_QUEUE_MAX):
        await _human(busy, f"input {n}")
    with pytest.raises(WakeNotAccepted) as refused:
        await deliver_wake(busy.adapter, text="kanban: t_1 completed", source=busy.source)
    assert refused.value.reason == 'queue_full'
    assert len([row for row in _rows(busy) if row['status'] == 'queued']) == ADMISSION_QUEUE_MAX
