"""Reviewer witnesses of assessment t_70278867/run2763 (SC-RECEIPT-02/03), maintained.

Ported from the review's disposable archive with only the JSON observation directory moved from the
reviewer's scratch path to pytest's ``tmp_path`` and two unused unpacked names underscored for the
linter; every assertion is the reviewer's."""
import json
import sqlite3
from types import SimpleNamespace
import pytest
from tests.agent import test_steer_preflight_compression as support
from tests.agent.test_merged_turn_receipt import _run, _assert_receipt_names_stored_rows


def receipts(agent, history, result):
    from gateway.run_turn_runner import _persisted_turn
    from tui_gateway.prompt_turn import _persisted_turn_receipt
    st = SimpleNamespace(agent=agent, history=history, result=result, compression_count=0)
    return {'gateway': _persisted_turn(agent, history, result, 0),
            'tui': _persisted_turn_receipt(st, result['final_response'], 'complete')}

@pytest.mark.parametrize('shape', ['preflight', 'uncompressed', 'in_loop', 'rotation'])
def test_real_turn_both_producers(monkeypatch, tmp_path, shape):
    if shape == 'rotation':
        make = support._make_agent
        def rotating(*args):
            agent, db = make(*args)
            agent.compression_in_place = False
            return agent, db
        monkeypatch.setattr(support, '_make_agent', rotating)
    options = {'compress': shape in ('preflight', 'rotation'), 'loop_compress': shape == 'in_loop'}
    history, agent, db, result, _payloads, rows = _run(monkeypatch, tmp_path, **options)
    try:
        got = receipts(agent, history, result)
        current = db.get_messages(agent.session_id)
        (tmp_path / f'real-{shape}.json').write_text(json.dumps({'session_id':agent.session_id,
            'receipts':got, 'messages':result['messages'], 'current_rows':current,
            'original_session_rows':rows, 'turn_message':agent._turn_user_message}, indent=2))
        assert result['completed'] is True
        for receipt in got.values():
            if shape in ('preflight', 'uncompressed'):
                _assert_receipt_names_stored_rows(receipt, rows, result)
            elif shape == 'in_loop':
                assert receipt is not None and not receipt['complete']
                assert 'user_row_id' not in receipt and 'user_row_ids' not in receipt
            else:
                assert agent.session_id != support.SESSION
            if receipt:
                by_id = {r['id']: r for r in current}
                assert all(i in by_id for i in receipt['row_ids'])
                assert all(i in by_id for i in receipt.get('user_row_ids', []))
                if 'user_row_id' in receipt:
                    prompt = by_id[receipt['user_row_id']]
                    assert support.INBOUND in prompt['content'] and support.BG_NOTICE not in prompt['content']
        assert got['gateway'] == got['tui']
    finally:
        db.close()

@pytest.mark.parametrize('compacted', [False, True])
def test_folded_prompt_cannot_publish_gapped_user_id_list(tmp_path, compacted):
    from hermes_state import SessionDB
    from agent.context_compressor import stamp_db_persisted_markers
    from agent.agent_runtime_helpers import _merge_consecutive_users
    db = SessionDB(db_path=tmp_path / 'folded.db')
    db.create_session(session_id='folded', source='local')
    def persist(rows):
        db.append_messages_batch('folded', rows)
        stamp_db_persisted_markers(rows)
        return rows
    try:
        prompt, steer = persist([dict(role='user', content='prompt'), dict(role='user', content='steer', display_kind='steer')])
        turn = dict(prompt)
        merged, count = _merge_consecutive_users([prompt, steer])
        assert count == 1
        if compacted:
            db.archive_and_compact('folded', merged)
            stamp_db_persisted_markers(merged)
        tail = persist([dict(role='assistant', content='response'), dict(role='user', content='later steer', display_kind='steer'), dict(role='assistant', content='done')])
        agent = SimpleNamespace(_session_db=db, session_id='folded', _persist_user_message_idx=0,
            _turn_user_message=turn, _persist_user_message_override=turn['content'],
            context_compressor=SimpleNamespace(compression_count=int(compacted)))
        result = {'messages':merged+tail, 'final_response':'done'}
        got = receipts(agent, [], result)
        (tmp_path / f'folded-{compacted}.json').write_text(json.dumps({'receipts':got,'rows':db.get_messages('folded'),'messages':merged+tail,'original_identities':{'prompt':{'row_id':turn['_row_id'],'uid':turn['message_uid']},'steer':{'row_id':steer['_row_id'],'uid':steer['message_uid']},'later':{'row_id':tail[1]['_row_id'],'uid':tail[1]['message_uid']}}},indent=2))
        for receipt in got.values():
            assert receipt and not receipt['complete']
            assert 'user_row_ids' not in receipt, 'absorbed steer occurrence omitted from positional list'
            if compacted:
                assert 'user_row_id' not in receipt, 'composite must not identify original prompt alone'
    finally:
        db.close()

@pytest.mark.parametrize('producer', ['gateway','tui'])
def test_identity_read_failure_omits_optional_receipt(monkeypatch, tmp_path, producer):
    history, agent, db, result, _payloads, _rows = _run(monkeypatch, tmp_path)
    try:
        assert result['completed'] is True
        assert receipts(agent, history, result)[producer]['user_row_id']
        def failed_read(*args):
            raise sqlite3.OperationalError('reviewer synthetic identity read failure')
        monkeypatch.setattr(db, 'get_message_identity', failed_read)
        from gateway.run_turn_runner import _persisted_turn
        from tui_gateway.prompt_turn import _persisted_turn_receipt
        if producer == 'gateway':
            receipt = _persisted_turn(agent, history, result, 0)
        else:
            st = SimpleNamespace(agent=agent, history=history, result=result, compression_count=0)
            receipt = _persisted_turn_receipt(st, result['final_response'], 'complete')
        assert receipt is None
    finally:
        db.close()
