"""Reviewer continuation through the real owner: ``run_task`` seeds a review session with the
review conversation it continues, once, before admission, and tells the reviewer what it got."""
import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_CONFIG = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
_SENTINEL = 'REVIEW_SENTINEL_7f3a'


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import kanban_db as kb
    from tests.gateway.test_session_follow_profile_config import _owner

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    for name in ('HERMES_KANBAN_DB', 'HERMES_DELEGATED_CHILD', 'HERMES_SINGLE_QUERY_SESSION', 'HERMES_YOLO_MODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps(_CONFIG))
    kb.create_board('owned')
    store, _runner, authority, dashboard = _owner(tmp_path)
    authority.profile_id = str(tmp_path)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})

    async def close():
        await connection.close()
        await dashboard.close()
        store.close_all_db_handles()
    yield SimpleNamespace(authority=authority, db=authority.db, connection=connection, close=close, home=tmp_path)


def _board():
    from hermes_cli.kanban_db_connect import connect
    return closing(connect(board='owned'))


def _handed_off(title, metadata=None, reviewer='rev'):
    """A card whose implementer handed it to review with *metadata*."""
    from hermes_cli import kanban_db as kb
    with _board() as conn:
        tid = kb.create_task(conn, title=title, assignee='impl', workspace_kind='dir', workspace_path=os.environ['HERMES_HOME'])
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, tid)
        ok, reason = kb.request_review(conn, tid, summary='done', metadata=metadata, reviewer=reviewer,
                                       expected_run_id=task.current_run_id, with_reason=True)
        assert ok, reason
    return tid


def _claim_review(tid):
    from hermes_cli import kanban_db as kb
    with _board() as conn:
        task = kb.claim_review_task(conn, tid)
    return dict(board='owned', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)


def _complete(params):
    from hermes_cli import kanban_db as kb
    with _board() as conn:
        assert kb.complete_task(conn, params['task_id'], summary='ok', expected_run_id=params['run_id'])


def _settle(authority, sid):
    """Run the queued card turn to completion in the store, as a finished review would leave it."""
    from hermes_state_runtime import claim_session_input, settle_session_input
    claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=claimed['admission_id'],
                         generation=claimed['generation'], outcome='completed')


def _transcript(db, sid):
    return [(m['role'], m.get('content'), m.get('tool_calls')) for m in db.get_messages(sid)]


def _admitted_text(db, sid):
    from hermes_state_runtime import list_session_admissions
    [row] = list_session_admissions(db, session_id=sid, pending_only=False)
    return row['payload']['text']


def _receipt(db, sid):
    from hermes_state_local import local_receipt
    return local_receipt(db, sid)


async def _source_review(owner, *, settle=True, complete=True, messages=True):
    """A spec card reviewed by 'rev' in this owner: its review session holds the sentinel."""
    from gateway.session_kanban import run_task
    spec = _handed_off('spec')
    params = _claim_review(spec)
    sid = (await run_task(owner.connection, params))['session_id']
    if messages:
        owner.db.append_message(sid, 'user', 'work kanban task ' + spec)
        owner.db.append_message(sid, 'assistant', None, tool_calls=[{'id': 'c1', 'type': 'function',
            'function': {'name': 'kanban_show', 'arguments': '{}'}}])
        owner.db.append_message(sid, 'tool', '{"task": "spec"}', tool_call_id='c1', tool_name='kanban_show')
        owner.db.append_message(sid, 'assistant', f'{_SENTINEL}: the spec needs a retry budget')
    if settle:
        _settle(owner.authority, sid)
    if complete:
        _complete(params)
    return spec, params, sid


def test_review_inherits_the_named_spec_review_conversation_once(owner):
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            spec, spec_params, src = await _source_review(owner)
            before = _transcript(owner.db, src)
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            first = await run_task(owner.connection, params)
            retry = await run_task(owner.connection, params)
            assert retry == first
            dest = first['session_id']
            assert _transcript(owner.db, dest) == before == _transcript(owner.db, src)
            seed = _receipt(owner.db, dest)['review_seed']
            assert seed['outcome'] == 'inherited' and seed['count'] == 4 and seed['source_busy'] is False
            assert (seed['source_task'], seed['source_run'], seed['source_session']) == (spec, spec_params['run_id'], src)
            text = _admitted_text(owner.db, dest)
            assert text.startswith(f'work kanban task {impl}\n\n')
            assert 'ended before this run' in text and f"task {impl} run {params['run_id']}" in text
            assert 'fresh verdict' in text
            row = owner.db.get_session(dest)
            assert row['parent_session_id'] is None and row['message_count'] == 4
            # The worker's first-turn read, turn recovery and the goal loop all re-read this.
            assert _SENTINEL in json.dumps(owner.db.get_messages_as_conversation(dest))
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_busy_or_running_source_is_described_as_a_snapshot(owner):
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            spec, _params, src = await _source_review(owner, settle=False, complete=False)
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            dest = (await run_task(owner.connection, params))['session_id']
            seed = _receipt(owner.db, dest)['review_seed']
            assert seed['source_busy'] is True and seed['source_run_ended'] is False
            text = _admitted_text(owner.db, dest)
            assert 'snapshot' in text and 'may still be active' in text and str(seed['boundary']) in text
            assert 'ended' not in text and 'closed' not in text
            # The source keeps going; a retry neither copies the new rows nor rewords the note.
            owner.db.append_message(src, 'assistant', 'LATE_ROW')
            retry = await run_task(owner.connection, params)
            assert retry['session_id'] == dest and 'LATE_ROW' not in json.dumps(_transcript(owner.db, dest))
            assert _admitted_text(owner.db, dest) == text
        finally:
            await owner.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('guard', ['turn_lease', 'compression_lock'])
def test_settled_source_under_a_transcript_guard_is_a_recorded_snapshot(owner, guard):
    """A settled, completed source whose transcript guard is held (a live turn lease or a
    compression lock) is busy: recorded in the seed and described as a snapshot, never refused."""
    from gateway.session_kanban import run_task

    acquire, release = {
        'turn_lease': (owner.db.try_acquire_session_turn_lease, owner.db.release_session_turn_lease),
        'compression_lock': (owner.db.try_acquire_compression_lock, owner.db.release_compression_lock),
    }[guard]

    async def scenario():
        src = None
        try:
            spec, _params, src = await _source_review(owner)
            assert acquire(src, 'held-source-guard')
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            first = await run_task(owner.connection, params)
            dest = first['session_id']
            seed = _receipt(owner.db, dest)['review_seed']
            assert seed['outcome'] == 'inherited' and seed['count'] == 4
            assert seed['source_busy'] is True and seed['source_run_ended'] is True
            assert _transcript(owner.db, dest) == _transcript(owner.db, src)
            text = _admitted_text(owner.db, dest)
            assert 'snapshot' in text and 'may still be active' in text and str(seed['boundary']) in text
            assert 'ended' not in text and 'closed' not in text
            assert f"task {impl} run {params['run_id']}" in text and 'fresh verdict' in text
            # A retry returns the stored record: no second copy, no rewording.
            owner.db.append_message(src, 'assistant', 'LATE_ROW')
            assert await run_task(owner.connection, params) == first
            assert 'LATE_ROW' not in json.dumps(_transcript(owner.db, dest))
            assert _admitted_text(owner.db, dest) == text
        finally:
            if src is not None:
                release(src, 'held-source-guard')
            await owner.close()
    asyncio.run(scenario())


def test_empty_source_and_unavailable_sources_review_cold_but_admit(owner):
    from gateway.session_kanban import run_task
    from hermes_cli import kanban_db as kb

    async def scenario():
        try:
            spec, _params, _src = await _source_review(owner, messages=False)
            empty = _handed_off('empty', {'reviewer_context': 'continue', 'review_continues_task': spec})
            dest = (await run_task(owner.connection, _claim_review(empty)))['session_id']
            seed = _receipt(owner.db, dest)['review_seed']
            assert (seed['outcome'], seed['count'], seed['boundary']) == ('source_empty', 0, 0)
            assert 'not attached (source_empty' in _admitted_text(owner.db, dest)

            lone = _handed_off('lone', {'reviewer_context': 'continue'})
            dest = (await run_task(owner.connection, _claim_review(lone)))['session_id']
            assert _receipt(owner.db, dest)['review_seed'] == {'outcome': 'unavailable', 'reason': 'no_predecessor'}
            assert 'not attached (no_predecessor)' in _admitted_text(owner.db, dest)

            for reason, marker in (('different_owner_store', {'db': '/elsewhere/state.db', 'session_id': 'x'}),
                                   ('source_session_missing', {'db': None, 'session_id': 'local-missing'})):
                other = _handed_off('forged-' + reason)
                forged = _claim_review(other)
                with _board() as conn, kb.write_txn(conn):
                    kb._append_event(conn, other, 'owner_admitted', dict(marker, db=marker['db'] or str(
                        Path(owner.db.db_path).resolve()), request_id='kanban:x'), run_id=forged['run_id'])
                _complete(forged)
                card = _handed_off('cold-' + reason, {'reviewer_context': 'continue', 'review_continues_task': other})
                dest = (await run_task(owner.connection, _claim_review(card)))['session_id']
                assert _receipt(owner.db, dest)['review_seed'] == {'outcome': 'unavailable', 'reason': reason}
                assert owner.db.get_messages(dest) == []
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_no_opt_in_review_is_unchanged(owner):
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            await _source_review(owner)
            plain = _handed_off('plain', {'tests_run': ['x']})
            params = _claim_review(plain)
            dest = (await run_task(owner.connection, params))['session_id']
            assert 'review_seed' not in _receipt(owner.db, dest) and owner.db.get_messages(dest) == []
            assert _admitted_text(owner.db, dest) == f'work kanban task {plain}'
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_admission_failure_after_seed_retries_without_a_second_copy(owner, monkeypatch):
    from gateway.session_kanban import run_task
    from hermes_cli import kanban_db as kb
    from hermes_state_runtime import RuntimeStoreError

    async def scenario():
        try:
            spec, _params, src = await _source_review(owner)
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            real = kb._append_event

            def crash(conn, task_id, kind, *args, **kwargs):
                if kind == 'owner_admitted':
                    raise RuntimeStoreError('storage_unavailable')
                return real(conn, task_id, kind, *args, **kwargs)
            monkeypatch.setattr(kb, '_append_event', crash)
            with pytest.raises(RuntimeStoreError):
                await run_task(owner.connection, params)
            monkeypatch.setattr(kb, '_append_event', real)
            dest = (await run_task(owner.connection, params))['session_id']
            assert _transcript(owner.db, dest) == _transcript(owner.db, src)
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_concurrent_launches_seed_once(owner):
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            spec, _params, src = await _source_review(owner)
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            results = await asyncio.gather(run_task(owner.connection, params), run_task(owner.connection, params),
                                           return_exceptions=True)
            done = [r for r in results if isinstance(r, dict)]
            assert done and len({r['session_id'] for r in done}) == 1
            assert _transcript(owner.db, done[0]['session_id']) == _transcript(owner.db, src)
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_unanswered_verdict_call_at_the_source_tail_is_repaired_not_replayed(owner):
    from agent.replay_cleanup import canonicalize_replay_history
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            spec, _params, src = await _source_review(owner)
            owner.db.append_message(src, 'assistant', None, tool_calls=[{'id': 'v1', 'type': 'function',
                'function': {'name': 'kanban_complete', 'arguments': '{"summary": "spec ok"}'}}])
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            dest = (await run_task(owner.connection, _claim_review(impl)))['session_id']
            history = canonicalize_replay_history(owner.db.get_messages_as_conversation(dest))
            answered = {m.get('tool_call_id') for m in history if m['role'] == 'tool'}
            calls = [c['id'] for m in history if m['role'] == 'assistant' for c in m.get('tool_calls') or ()]
            assert set(calls) <= answered, 'every replayed call is answered (or stubbed), none left to run'
            assert _SENTINEL in json.dumps(history)
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_two_cards_continuing_one_review_get_independent_copies(owner):
    from gateway.session_kanban import run_task

    async def scenario():
        try:
            spec, _params, src = await _source_review(owner)
            dests = []
            for title in ('impl-1', 'impl-2'):
                card = _handed_off(title, {'reviewer_context': 'continue', 'review_continues_task': spec})
                dests.append((await run_task(owner.connection, _claim_review(card)))['session_id'])
            owner.db.append_message(dests[0], 'assistant', 'ONLY_IN_FIRST')
            assert _transcript(owner.db, dests[1]) == _transcript(owner.db, src)
            assert 'ONLY_IN_FIRST' not in json.dumps(_transcript(owner.db, src))
        finally:
            await owner.close()
    asyncio.run(scenario())


def _restart(owner, monkeypatch):
    """A fresh owner over the same home and store, as after the owner process died: new epoch, nothing in memory."""
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_session_follow_profile_config import _owner
    store, _runner, authority, dashboard = _owner(owner.home)
    authority.profile_id = str(owner.home)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})
    previous = owner.close

    async def close():
        await connection.close()
        await dashboard.close()
        store.close_all_db_handles()
        await previous()
    return SimpleNamespace(authority=authority, db=authority.db, connection=connection, close=close, home=owner.home)


@pytest.mark.parametrize('crash', ['before_seed', 'after_seed'])
def test_crash_between_session_creation_and_admission_retries_to_the_no_crash_payload(owner, monkeypatch, crash):
    """The owner dies after creating the review session, before (or just after) seeding it; the
    retry of the same attempt on a restarted owner reuses that session, seeds once and admits
    exactly what an undisturbed attempt admits; the card then completes once."""
    import hermes_state_review_seed
    from gateway.session_kanban import run_task
    from hermes_cli import kanban_db as kb
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions

    async def scenario():
        nonlocal owner
        try:
            spec, _params, src = await _source_review(owner)
            # The undisturbed twin: same source, same handoff, no crash.
            twin = _claim_review(_handed_off('twin', {'reviewer_context': 'continue', 'review_continues_task': spec}))
            twin_dest = (await run_task(owner.connection, twin))['session_id']
            impl = _handed_off('impl', {'reviewer_context': 'continue', 'review_continues_task': spec})
            params = _claim_review(impl)
            seeded = []
            real_seed, real_append = hermes_state_review_seed.seed_review_continuity, kb._append_event

            def die_before_seed(*args, **kwargs):
                raise RuntimeStoreError('storage_unavailable')

            def die_before_admission(conn, task_id, kind, *args, **kwargs):
                if kind == 'owner_admitted':
                    raise RuntimeStoreError('storage_unavailable')
                return real_append(conn, task_id, kind, *args, **kwargs)
            if crash == 'before_seed':
                monkeypatch.setattr(hermes_state_review_seed, 'seed_review_continuity', die_before_seed)
            else:
                monkeypatch.setattr(kb, '_append_event', die_before_admission)
            with pytest.raises(RuntimeStoreError):
                await run_task(owner.connection, params)
            monkeypatch.setattr(hermes_state_review_seed, 'seed_review_continuity', real_seed)
            monkeypatch.setattr(kb, '_append_event', real_append)
            from gateway.session_local_recovery import local_identity
            request_id = 'kanban:' + json.dumps(['owned', impl, params['run_id']], separators=(',', ':'))
            dest = local_identity(owner.authority.profile_id, 'owner', request_id)
            assert owner.db.get_session(dest) is not None, 'the crash left the created session behind'
            assert ('review_seed' in _receipt(owner.db, dest)) == (crash == 'after_seed')
            assert len(owner.db.get_messages(dest)) == (0 if crash == 'before_seed' else 4)
            assert list_session_admissions(owner.db, session_id=dest, pending_only=False) == []
            before_retry = _receipt(owner.db, dest).get('review_seed')

            owner = _restart(owner, monkeypatch)

            def counted(*args, **kwargs):
                seeded.append(kwargs['session_id'])
                return real_seed(*args, **kwargs)
            monkeypatch.setattr(hermes_state_review_seed, 'seed_review_continuity', counted)
            first = await run_task(owner.connection, params)
            again = await run_task(owner.connection, params)
            assert first['session_id'] == again['session_id'] == dest and again == first
            seed = _receipt(owner.db, dest)['review_seed']
            if before_retry is not None:
                assert seed == before_retry, 'the retry re-seeded a recorded session'
            assert _transcript(owner.db, dest) == _transcript(owner.db, src) == _transcript(owner.db, twin_dest)
            twin_seed = _receipt(owner.db, twin_dest)['review_seed']
            assert {k: v for k, v in seed.items() if k != 'dest_last_row'} == {
                k: v for k, v in twin_seed.items() if k != 'dest_last_row'}
            [admitted] = list_session_admissions(owner.db, session_id=dest, pending_only=False)
            [twin_admitted] = list_session_admissions(owner.db, session_id=twin_dest, pending_only=False)
            normal = lambda payload, p: json.loads(json.dumps(payload).replace(p['task_id'], 'TASK').replace(
                f"run {p['run_id']}", 'run RUN').replace(f'"run_id": {p["run_id"]}', '"run_id": "RUN"'))
            assert normal(admitted['payload'], params) == normal(twin_admitted['payload'], twin)
            assert 'ended before this run' in admitted['payload']['text']
            assert seeded == [dest, dest]  # both retries reached the seed; only the first copied
            _complete(params)
            with _board() as conn:
                events = [r['kind'] for r in conn.execute(
                    "SELECT kind FROM task_events WHERE task_id=? AND kind IN ('owner_admitted', 'completed')", (impl,))]
                assert events == ['owner_admitted', 'completed']
                assert kb.get_task(conn, impl).status == 'done'
        finally:
            await owner.close()
    asyncio.run(scenario())
