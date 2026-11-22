"""SC-RECEIPT-01: the persisted-turn receipt after the alternation repair folded this turn's inbound row
into an older unanswered user row (the background-process notice).

The re-anchor names the merged dict as this turn's row, but that dict's ``_row_id`` is the OLDER
notice row; the inbound's own stored row is in its merge witness (``_absorbed_row_ids`` /
``_absorbed_message_uids``). ``message.complete.persisted_turn`` must name the inbound's own row, or
omit what it cannot prove, and every id it lists must be the SQLite row it claims. Both shapes: the
compressed one (compaction made replacement dicts, so the caller's history holds none of them) and
the uncompressed one (the folded-into dict IS the caller's history dict).

Real turn loop, compressor, steer drains, sequence repair, SessionDB and the gateway receipt producer
(``gateway.run_turn_runner._persisted_turn``); only the summary/model calls are stubbed.
"""
from tests.agent import test_steer_preflight_compression as support


def _run(monkeypatch, tmp_path, **kwargs):
    history = []
    original = support._history

    def capture(db):
        value = original(db)
        history.extend(value)
        return value
    monkeypatch.setattr(support, '_history', capture)
    agent, db, result, payloads, rows = support._run_turn(monkeypatch, tmp_path, **kwargs)
    return history, agent, db, result, payloads, rows


def _assert_receipt_names_stored_rows(receipt, rows, result):
    by_id = {r['id']: r for r in rows}
    assert receipt is not None and receipt['complete'] is False
    prompt = by_id[receipt['user_row_id']]
    assert support.INBOUND in prompt['content'] and support.BG_NOTICE not in prompt['content']
    assert prompt['display_kind'] != 'steer'
    assert by_id[receipt['final_assistant_row_id']]['content'] == result['final_response']
    assert all(row_id in by_id for row_id in receipt['row_ids'])
    assert not any(support.BG_NOTICE in (by_id[row_id]['content'] or '') for row_id in receipt['row_ids'])
    assert [by_id[row_id]['role'] for row_id in receipt['user_row_ids']] == ['user'] * 3
    assert receipt['user_row_ids'][0] == receipt['user_row_id']
    steer_contents = [by_id[row_id]['content'] for row_id in receipt['user_row_ids'][1:]]
    assert support.WAKE in steer_contents[0] and support.FOLLOW in steer_contents[1]


def test_merged_turn_keeps_payloads_and_returns_no_false_receipt(monkeypatch, tmp_path):
    """The reviewer's reproduction (t_89ebbb1d), maintained: compressed replacement-dict shape."""
    history, agent, db, result, _payloads, rows = _run(monkeypatch, tmp_path)
    try:
        from gateway.run_turn_runner import _persisted_turn
        receipt = _persisted_turn(agent, history, result, 0)
        by_id = {r['id']: r for r in rows}
        assert receipt is not None and receipt['complete'] is False
        assert support.INBOUND in by_id[receipt['user_row_id']]['content']
        assert by_id[receipt['user_row_id']]['display_kind'] != 'steer'
        assert by_id[receipt['final_assistant_row_id']]['content'] == result['final_response']
        assert all(row_id in by_id for row_id in receipt['row_ids'])
        assert all(by_id[row_id]['role'] == 'user' for row_id in receipt['user_row_ids'])
        assert sum(support.INBOUND in (r.get('content') or '') for r in rows) == 1
        assert sum(support.WAKE in (r.get('content') or '') for r in rows) == 1
        assert sum(support.FOLLOW in (r.get('content') or '') for r in rows) == 1
        assert any(support.BG_NOTICE in (r.get('content') or '') for r in rows)
        assert result['final_response'] == 'Handled.'
        _assert_receipt_names_stored_rows(receipt, rows, result)
    finally:
        db.close()


def test_uncompressed_merged_row_receipt_names_the_inbound_row(monkeypatch, tmp_path):
    """Uncompressed shape: the turn is folded into the caller's own history dict. Not mistaken for a
    returned old transcript, and the folded dict's (notice) row is never listed as this turn's."""
    history, agent, db, result, _payloads, rows = _run(monkeypatch, tmp_path, compress=False)
    try:
        from gateway.run_turn_runner import _persisted_turn
        assert agent.context_compressor.compression_count == 0
        anchor = result['messages'][agent._persist_user_message_idx]
        assert any(anchor is before for before in history)  # the shape under test
        _assert_receipt_names_stored_rows(_persisted_turn(agent, history, result, 0), rows, result)
    finally:
        db.close()


def test_in_loop_compaction_composite_row_omits_the_prompt_identity(monkeypatch, tmp_path):
    """Compaction AFTER the merge stores the merged dict as one new composite row (notice + inbound) and
    retires both originals: no single stored row is this turn's prompt, so the prompt and user-row list
    are omitted, the composite row is never listed, and every listed id is an active SQLite row."""
    history, agent, db, result, _payloads, rows = _run(monkeypatch, tmp_path, compress=False, loop_compress=True)
    try:
        from gateway.run_turn_runner import _persisted_turn
        assert agent.context_compressor.compression_count >= 1
        by_id = {r['id']: r for r in rows}
        anchor = result['messages'][agent._persist_user_message_idx]
        composite = by_id[anchor['_row_id']]  # the shape under test
        assert support.BG_NOTICE in composite['content'] and support.INBOUND in composite['content']
        receipt = _persisted_turn(agent, history, result, 0)
        assert receipt is not None and receipt['complete'] is False
        assert 'user_row_id' not in receipt and 'user_row_ids' not in receipt
        assert all(row_id in by_id for row_id in receipt['row_ids']) and composite['id'] not in receipt['row_ids']
        assert by_id[receipt['final_assistant_row_id']]['content'] == result['final_response']
    finally:
        db.close()


def test_unproven_folded_prompt_is_omitted_not_guessed():
    """A folded anchor's own id is the OLDER row: without SQLite confirming the single absorbed row
    (active, a user row, the witnessed uid) the prompt and user-row list are omitted, never guessed; a
    single-row final reply keeps its partial receipt, and SQLite has the last word on every id."""
    from agent.persisted_turn_receipt import persisted_turn_receipt, verify_receipt_rows

    folded = {'role': 'user', 'content': 'notice\n\nask', '_row_id': 38, '_db_persisted': True,
              '_merged_turn_prefix': 'notice', '_absorbed_row_ids': [39], '_absorbed_message_uids': ['u39'],
              '_merged_durable_rows': 2, 'message_uid': 'u38'}
    reply = {'role': 'assistant', 'content': 'done', '_row_id': 40, '_db_persisted': True, 'message_uid': 'u40'}
    turn = {'role': 'user', 'content': 'ask'}

    def receipt(**kwargs):
        return persisted_turn_receipt([folded, reply], 0, [], 'done', 'complete', compression_unchanged=True,
                                      prefix_row_matches=lambda a, b: a is b, **kwargs)
    partial = {'row_ids': [40], 'complete': False, 'final_assistant_row_id': 40}
    assert receipt() == partial  # no proof at all
    assert receipt(turn_message=turn) == partial  # no store to confirm the absorbed row
    rows = {39: ('user', 'u39'), 40: ('assistant', 'u40'), 38: ('user', 'u38')}
    assert receipt(turn_message=turn, row_identity=rows.get) == {
        'row_ids': [39, 40], 'complete': False, 'user_row_id': 39, 'user_row_ids': [39],
        'final_assistant_row_id': 40}
    # Retired into a composite row, or another message at that id: no single row is the prompt.
    assert receipt(turn_message=turn, row_identity={40: ('assistant', 'u40')}.get) == partial
    assert receipt(turn_message=turn, row_identity={**rows, 39: ('user', 'other')}.get) == partial
    # A dict standing for another message is never this turn's prompt.
    other = {'role': 'user', 'content': 'old', '_row_id': 37, '_db_persisted': True, 'message_uid': 'u37'}
    assert 'user_row_id' not in persisted_turn_receipt(
        [other, reply], 0, [], 'done', 'complete', compression_unchanged=True,
        prefix_row_matches=lambda a, b: a is b, turn_message={'role': 'user', 'content': 'x', 'message_uid': 'u9'})

    proven = {'row_ids': [39, 40], 'complete': True, 'user_row_id': 39, 'user_row_ids': [39],
              'final_assistant_row_id': 40}
    assert verify_receipt_rows(proven, {39: ('user', 'u39'), 40: ('user', 'u40')}.get) == {
        'row_ids': [39, 40], 'complete': False, 'user_row_id': 39, 'user_row_ids': [39]}
    assert verify_receipt_rows(proven, {40: ('assistant', 'u40')}.get) == {
        'row_ids': [40], 'complete': False, 'final_assistant_row_id': 40}
