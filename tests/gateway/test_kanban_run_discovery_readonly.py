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


def _wal_board(tmp_path, monkeypatch):
    """A real board in WAL mode with a current, running, owner-admitted card (owner db is a stand-in path)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'wal-home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    owner_db = tmp_path / 'state.db'
    with connect_closing() as board:
        assert board.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        tid = kb.create_task(board, title='WAL card')
        run = kb.claim_task(board, tid)
        kb._append_event(board, tid, 'owner_admitted', {'db': str(owner_db), 'session_id': 's'},
                         run_id=run.current_run_id)
    return kb.kanban_db_path(), {'board': 'default', 'task_id': tid, 'run_id': run.current_run_id}


def _schema(path):
    import sqlite3
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        return conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    finally:
        conn.close()


def test_closed_wal_board_content_and_schema_are_unchanged_by_a_read(monkeypatch, tmp_path):
    """Sidecars (-wal/-shm) may appear as for any reader of a WAL database; the database is not modified."""
    import hashlib
    from gateway.session_kanban_discovery import _read_claim
    db_path, params = _wal_board(tmp_path, monkeypatch)
    for suffix in ('-wal', '-shm'):  # the writer is closed: a clean close leaves no sidecars
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
    before, schema = hashlib.sha256(db_path.read_bytes()).hexdigest(), _schema(db_path)
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'))
    assert _read_claim(authority, params) == ('s', str(db_path))
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert _schema(db_path) == schema


def test_read_with_an_open_idle_writer_sees_current_wal_data_and_changes_no_content(monkeypatch, tmp_path):
    """An idle writer's committed rows live in the -wal: the read must see them (so not ``immutable=1``)."""
    import sqlite3
    from gateway.session_kanban_discovery import _read_claim
    db_path, params = _wal_board(tmp_path, monkeypatch)
    writer = sqlite3.connect(db_path)
    try:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute("UPDATE tasks SET status='done' WHERE id=?", (params['task_id'],))
        writer.commit()  # committed, idle, and only in the -wal
        rows_before = writer.execute('SELECT id, status FROM tasks ORDER BY id').fetchall()
        authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'))
        with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):  # the WAL's 'done' is what it read
            _read_claim(authority, params)
        assert writer.execute('SELECT id, status FROM tasks ORDER BY id').fetchall() == rows_before
        writer.execute("UPDATE tasks SET status='running' WHERE id=?", (params['task_id'],))
        writer.commit()
        assert _read_claim(authority, params) == ('s', str(db_path))  # and the next commit is seen too
    finally:
        writer.close()


def test_discovery_lookup_does_not_reset_a_retired_routes_idle_timer(monkeypatch, tmp_path):
    from gateway import session_kanban_discovery as discovery
    from gateway.session_contract import SessionActor
    from tests.gateway.test_session_authority_cancel_settlement import _authority
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        params = _current_card(db, tmp_path, monkeypatch)
        connection = SimpleNamespace(authority=authority, actor=SessionActor(
            'sender', authority.profile_id, frozenset({'session:read', 'session:operator'}), 'cli'))

        async def lookup():
            timer = asyncio.get_running_loop().call_later(3600, lambda: None)
            live = authority.sessions['s']
            live.retire_timer = timer
            authority.retired_routes['s'] = live.route
            try:
                await discovery.kanban_run(connection, None, params)
            except RuntimeStoreError:
                pass  # whatever the answer, the route's timer is untouched by the lookup
            assert live.retire_timer is timer and not timer.cancelled(), 'discovery reset the retirement timer'
            timer.cancel()
        asyncio.run(lookup())
