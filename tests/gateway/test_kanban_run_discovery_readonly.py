"""``session.kanban_run`` is a read: it never creates a board and never restores a session."""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError


def _discovery_connection(tmp_path):
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'), profile_id='p')
    return SimpleNamespace(authority=authority, actor=SimpleNamespace(capabilities={'session:read'}))


def test_absent_board_is_not_created_by_a_read_and_is_not_found(monkeypatch, tmp_path):
    from gateway.session_kanban_discovery import kanban_run
    from hermes_cli import kanban_db as kb
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'private-board-root'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    params = {'board': 'never-created', 'task_id': 'absent-card', 'run_id': 123}
    target = kb.kanban_db_path(board=params['board'])
    assert not target.exists()
    with pytest.raises(RuntimeStoreError, match='not_found'):
        asyncio.run(kanban_run(_discovery_connection(tmp_path), None, params))
    assert not target.exists(), f'read created board DB: {target}'
    assert not target.parent.exists(), f'read created board directory: {target.parent}'


def test_explicit_db_pin_without_a_board_gets_the_same_refusal(monkeypatch, tmp_path):
    from gateway.session_kanban_discovery import kanban_run
    pinned = tmp_path / 'pinned' / 'kanban.db'
    monkeypatch.setenv('HERMES_KANBAN_DB', str(pinned))
    params = {'board': 'default', 'task_id': 'absent-card', 'run_id': 1}
    with pytest.raises(RuntimeStoreError, match='not_found'):
        asyncio.run(kanban_run(_discovery_connection(tmp_path), None, params))
    assert not pinned.exists() and not pinned.parent.exists()


def test_file_without_the_kanban_schema_is_not_a_board_and_is_left_unchanged(monkeypatch, tmp_path):
    import sqlite3
    from gateway.session_kanban_discovery import kanban_run
    pinned = tmp_path / 'other.db'
    sqlite3.connect(pinned).close()
    before = pinned.read_bytes()
    monkeypatch.setenv('HERMES_KANBAN_DB', str(pinned))
    with pytest.raises(RuntimeStoreError, match='not_found'):
        asyncio.run(kanban_run(_discovery_connection(tmp_path), None,
                               {'board': 'default', 'task_id': 'x', 'run_id': 1}))
    assert pinned.read_bytes() == before


def _current_card(db, tmp_path, monkeypatch, session_id='s'):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'board-home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    with connect_closing() as board:
        tid = kb.create_task(board, title='Real current card')
        run = kb.claim_task(board, tid)
        kb._append_event(board, tid, 'owner_admitted', {'db': str(db.db_path), 'session_id': session_id},
                         run_id=run.current_run_id)
    return {'board': 'default', 'task_id': tid, 'run_id': run.current_run_id}


def test_session_absent_from_live_sessions_is_never_restored_by_a_read(monkeypatch, tmp_path):
    from gateway import session_kanban_discovery as discovery
    from gateway.session_contract import SessionActor
    from tests.gateway.test_session_authority_cancel_settlement import _authority
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        authority.sessions.pop('s')
        restored = []
        import gateway.session_route_recovery as recovery

        def restore(*args, **kwargs):
            restored.append(True)
            raise RuntimeError('RESTORE_REACHED_BY_READ')
        monkeypatch.setattr(recovery, 'restore_routed_session', restore)
        params = _current_card(db, tmp_path, monkeypatch)
        connection = SimpleNamespace(authority=authority, actor=SessionActor(
            'sender', authority.profile_id, frozenset({'session:read', 'session:operator'}), 'cli'))
        with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
            asyncio.run(discovery.kanban_run(connection, None, params))
        assert restored == [], 'discovery tried restoring a session absent from live sessions'
        assert 's' not in authority.sessions
