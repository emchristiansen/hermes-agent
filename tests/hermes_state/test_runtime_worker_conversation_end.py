"""A managed worker ends its own conversation in place through the owner, as SessionDB does."""
import pytest


def _conversation(db, sid):
    db.append_messages_batch(sid, [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}])


def test_worker_store_ends_the_conversation_exactly_as_the_owner_does(worker):
    db, store = worker
    db.create_session('twin', 'cli', system_prompt='prefix')
    _conversation(db, 'owned')
    _conversation(db, 'twin')
    store.end_conversation_in_place('owned')
    db.end_conversation_in_place('twin')

    def state(sid):
        row = db.get_session(sid)
        return (row['system_prompt'], [(m['role'], m['content']) for m in db.get_messages(sid)],
                [(m['role'], m['content']) for m in db.get_messages(sid, include_inactive=True)])
    ended = state('owned')
    assert ended == state('twin')
    assert ended[0] is None and ended[1] == [] and ended[2] == [('user', 'q'), ('assistant', 'a')]


def test_worker_cannot_end_a_conversation_outside_its_lineage(worker):
    from hermes_state_runtime import RuntimeStoreError
    db, store = worker
    _conversation(db, 'foreign')
    with pytest.raises(RuntimeStoreError) as exc:
        store.end_conversation_in_place('foreign')
    assert exc.value.reason == 'permission_denied'
    assert [m['content'] for m in db.get_messages('foreign')] == ['q', 'a']
    assert db.get_session('foreign')['system_prompt'] == 'secret'
