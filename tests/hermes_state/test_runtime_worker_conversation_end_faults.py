"""The managed-worker reset path under receipt failure, a lost acknowledgement and a failed resume selection."""
import sqlite3
from contextlib import contextmanager
import pytest


def seed(db, sid='owned'):
    db.append_messages_batch(sid, [{'role': 'user', 'content': 'before reset'}])


def test_receipt_insert_failure_rolls_back_reset(worker):
    db, store = worker
    seed(db)
    before = db.get_session('owned')
    db._execute_write(lambda c: c.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON worker_receipts BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='injected receipt failure'):
        store.end_conversation_in_place('owned')
    assert db.get_session('owned') == before
    assert [m['content'] for m in db.get_messages('owned')] == ['before reset']
    assert not db._read_all('SELECT * FROM worker_receipts')
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == 0
    db._execute_write(lambda c: c.execute('DROP TRIGGER reject_receipt'))
    store.retry_pending()
    assert db.get_messages('owned') == []
    assert db.get_session('owned')['system_prompt'] is None
    assert len(db._read_all('SELECT * FROM worker_receipts')) == 1


def test_lost_ack_exact_replay_preserves_new_rows(worker):
    db, store = worker
    seed(db)
    rpc = store.rpc
    def lose_ack(method, **params):
        result = rpc(method, **params)
        raise TimeoutError('injected lost ack after commit')
    store.rpc = lose_ack
    with pytest.raises(TimeoutError):
        store.end_conversation_in_place('owned')
    seed(db)
    store.rpc = rpc
    store.retry_pending()
    assert [m['content'] for m in db.get_messages('owned')] == ['before reset']
    assert len(db._read_all('SELECT * FROM worker_receipts')) == 1
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == 1


def test_strict_worker_resume_must_raise_on_selection_read_failure(worker, monkeypatch):
    from hermes_state_worker_compression import CompressionSnapshot
    db, store = worker
    seed(db)
    real_read = CompressionSnapshot._read_ctx
    class FailingConnection:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, params=()):
            if sql == 'SELECT 1 FROM messages WHERE session_id = ? LIMIT 1':
                raise sqlite3.OperationalError('injected resume-selection read failure')
            return self.conn.execute(sql, params)
    @contextmanager
    def failing_read(view):
        with real_read(view) as conn:
            yield FailingConnection(conn)
    monkeypatch.setattr(CompressionSnapshot, '_read_ctx', failing_read)
    # The owner contract really is strict under the identical SQL failure.
    owner_read = db._read_ctx
    @contextmanager
    def failing_owner_read():
        with owner_read() as conn:
            yield FailingConnection(conn)
    with monkeypatch.context() as owner_patch:
        owner_patch.setattr(db, '_read_ctx', failing_owner_read)
        with pytest.raises(sqlite3.OperationalError, match='injected resume-selection read failure'):
            db.resolve_resume_session_id('owned', strict=True)
    with pytest.raises(sqlite3.OperationalError, match='injected resume-selection read failure'):
        store.resolve_resume_session_id('owned', strict=True)


def test_forget_stored_prompt_refuses_failed_resume_selection(worker, monkeypatch):
    from types import SimpleNamespace
    from agent.home_guidance import forget_stored_prompt
    from hermes_state_worker_compression import CompressionSnapshot
    db, store = worker
    seed(db)
    real_read = CompressionSnapshot._read_ctx
    failures = []
    class FailingConnection:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, params=()):
            if sql == 'SELECT 1 FROM messages WHERE session_id = ? LIMIT 1':
                failures.append(params)
                raise sqlite3.OperationalError('injected resume-selection read failure')
            return self.conn.execute(sql, params)
    @contextmanager
    def failing_read(view):
        with real_read(view) as conn:
            yield FailingConnection(conn)
    monkeypatch.setattr(CompressionSnapshot, '_read_ctx', failing_read)
    try:
        forget_stored_prompt(SimpleNamespace(_session_db=store, session_id='owned'))
    except sqlite3.OperationalError:
        assert db.get_messages('owned')
    else:
        pytest.fail(f'reset reported success after {len(failures)} failed selection reads; '
                    f'active_messages={len(db.get_messages("owned"))}')


def test_strict_worker_resume_normal_path(worker):
    db, store = worker
    seed(db)
    assert store.resolve_resume_session_id('owned', strict=True) == 'owned'


def test_forget_stored_prompt_ends_an_ordinary_and_a_compression_lineage_conversation(worker):
    from types import SimpleNamespace
    from agent.home_guidance import forget_stored_prompt
    db, store = worker
    seed(db)
    forget_stored_prompt(SimpleNamespace(_session_db=store, session_id='owned'))
    assert db.get_messages('owned') == []
    # A compressed conversation: the session every resume reads is the tip, and it ends too.
    seed(db)
    db.end_session('owned', 'compression')
    db.create_session('tip', 'cli', parent_session_id='owned')
    seed(db, 'tip')
    forget_stored_prompt(SimpleNamespace(_session_db=store, session_id='owned'))
    assert db.get_messages('owned') == [] and db.get_messages('tip') == []
