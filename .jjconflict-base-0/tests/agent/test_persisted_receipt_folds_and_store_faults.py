"""SC-RECEIPT-02/03: the persisted-turn receipt after an alternation fold, and under store failure.

SC-RECEIPT-02. ``_merge_consecutive_users`` folds two stored user rows into one live dict. Whichever
row leads, the dict takes ONE slot in the receipt's positional ``user_row_ids`` while it stands for
two occurrences, and in-place compaction then stores it as one NEW composite row under the leading
row's ``message_uid``. The desktop binder pairs bubbles positionally even for a partial receipt, so
a list missing the absorbed occurrence hands an earlier bubble a later row's id. Matrix: notice- or
prompt-leading fold, before or after in-place compaction, each followed by a later standalone steer;
both real producers (``gateway.run_turn_runner._persisted_turn`` and
``tui_gateway.prompt_turn._persisted_turn_receipt``). Only a proven one-to-one mapping may be
published; the proven final assistant survives every omission.

SC-RECEIPT-03. The receipt is optional: a ``sqlite3.Error`` while proving an absorbed row
(``_prompt_row_id``) or while verifying the listed rows (``verify_receipt_rows``) must omit it, and
the turn must still complete. Driven through the gateway's real ``TurnRunner.run_sync`` and
settlement ``completion_payload``, and the TUI's real ``_run_prompt_submit`` ``message.complete``.

Real SessionDB, merge repair, in-place compaction and persistence-marker stamping; the agent is a
constructed envelope around them (no model loop: full-turn shapes live in test_merged_turn_receipt).
"""
from __future__ import annotations

import sqlite3
import traceback
from types import SimpleNamespace

import pytest

SID = 'fold'
NOTICE = '[IMPORTANT: Background process proc_1 completed (exit 0).]'
FOLDS = [(leading, compacted) for leading in ('notice', 'prompt') for compacted in (False, True)]


def _persisted(db):
    from agent.context_compressor import stamp_db_persisted_markers

    def persist(rows):
        db.append_messages_batch(SID, rows)
        stamp_db_persisted_markers(rows)
        return rows
    return persist


def fold_witness(db, leading: str, compacted: bool) -> dict:
    """Store the fold shape and return ``{agent, result, originals, bubbles}``.

    ``originals`` maps each optimistic bubble the desktop shows for this turn to the row id and uid
    of its own ORIGINAL stored occurrence (what a correct binding must name, or omit)."""
    from agent.agent_runtime_helpers import _merge_consecutive_users
    from agent.context_compressor import stamp_db_persisted_markers

    db.create_session(session_id=SID, source='local')
    persist = _persisted(db)
    if leading == 'prompt':
        prompt, steer = persist([dict(role='user', content='prompt'),
                                 dict(role='user', content='steer', display_kind='steer')])
        pair, absorbed = [prompt, steer], {'steer': steer}
    else:
        notice, prompt = persist([dict(role='user', content=NOTICE), dict(role='user', content='prompt')])
        pair, absorbed = [notice, prompt], {}
    turn = dict(prompt)  # the turn's own dict, as recorded before the repair mutates the leading one
    originals = {'prompt': prompt, **absorbed}
    originals = {key: {'row_id': row['_row_id'], 'uid': row['message_uid']} for key, row in originals.items()}
    merged, count = _merge_consecutive_users(pair)
    assert count == 1 and len(merged) == 1
    if compacted:
        db.archive_and_compact(SID, merged)
        stamp_db_persisted_markers(merged)
    tail = persist([dict(role='assistant', content='response'),
                    dict(role='user', content='later steer', display_kind='steer'),
                    dict(role='assistant', content='done')])
    originals['later'] = {'row_id': tail[1]['_row_id'], 'uid': tail[1]['message_uid']}
    agent = SimpleNamespace(
        _session_db=db, session_id=SID, _persist_user_message_idx=0, _turn_user_message=turn,
        _persist_user_message_override=turn['content'],
        context_compressor=SimpleNamespace(compression_count=int(compacted)))
    bubbles = ['prompt', 'steer', 'later'] if leading == 'prompt' else ['prompt', 'later']
    return {'agent': agent, 'result': {'messages': merged + tail, 'final_response': 'done'},
            'originals': originals, 'bubbles': bubbles, 'final_row_id': tail[2]['_row_id'],
            'composite': merged[0]}


def both_receipts(agent, result) -> dict:
    from gateway.run_turn_runner import _persisted_turn
    from tui_gateway.prompt_turn import _persisted_turn_receipt
    st = SimpleNamespace(agent=agent, history=[], result=result, compression_count=0)
    return {'gateway': _persisted_turn(agent, [], result, 0),
            'tui': _persisted_turn_receipt(st, result['final_response'], 'complete')}


def _active_rows(db):
    return {row['id']: row for row in db.get_messages(SID)}


@pytest.mark.parametrize('leading,compacted', FOLDS)
def test_fold_publishes_only_one_to_one_user_occurrences(tmp_path, leading, compacted):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'fold.db')
    try:
        w = fold_witness(db, leading, compacted)
        got = both_receipts(w['agent'], w['result'])
        rows, originals = _active_rows(db), w['originals']
        assert got['gateway'] == got['tui']
        receipt = got['gateway']
        # Every omission keeps the proven final assistant, and the receipt never claims the whole turn.
        assert receipt is not None and receipt['complete'] is False
        assert receipt['final_assistant_row_id'] == w['final_row_id']
        assert all(row_id in rows for row_id in receipt['row_ids'])
        # A composite row is never this turn's prompt or any single occurrence.
        if compacted:
            assert w['composite']['_row_id'] not in receipt['row_ids']
        if leading == 'notice' and not compacted:
            # Positive control: the only fold with a proven one-to-one mapping (the notice is a pre-turn
            # row; the prompt's own stored row is active, a user row, with the turn's uid).
            assert receipt['user_row_id'] == originals['prompt']['row_id']
            assert receipt['user_row_ids'] == [originals['prompt']['row_id'], originals['later']['row_id']]
            for key, row_id in zip(w['bubbles'], receipt['user_row_ids']):
                assert (rows[row_id]['role'], rows[row_id]['message_uid']) == ('user', originals[key]['uid'])
        else:
            assert 'user_row_ids' not in receipt, 'a fold without a one-to-one mapping published a list'
            assert 'user_row_id' not in receipt, 'a composite or unproven row published as the prompt'
    finally:
        db.close()


def test_prompt_leading_fold_without_turn_uid_or_as_the_turn_dict_is_not_the_prompt():
    """The same rule when the turn's dict carries no uid, or IS the mutated leading dict (the live
    repair folds in place): the fold's own row id never stands for the prompt, the list is omitted."""
    from agent.persisted_turn_receipt import persisted_turn_receipt

    composite = {'role': 'user', 'content': 'prompt\n\nsteer', '_row_id': 3, '_db_persisted': True,
                 '_merged_turn_prefix': 'prompt', '_absorbed_row_ids': [2], '_absorbed_message_uids': ['u2'],
                 '_merged_durable_rows': 2, 'message_uid': 'u1'}
    tail = [{'role': 'assistant', 'content': 'response', '_row_id': 4, '_db_persisted': True},
            {'role': 'user', 'content': 'later steer', 'display_kind': 'steer', '_row_id': 5, '_db_persisted': True},
            {'role': 'assistant', 'content': 'done', '_row_id': 6, '_db_persisted': True}]
    rows = {2: ('user', 'u2'), 3: ('user', 'u1'), 4: ('assistant', None), 5: ('user', None), 6: ('assistant', None)}
    expected = {'row_ids': [4, 5, 6], 'complete': False, 'final_assistant_row_id': 6}
    for turn in ({'role': 'user', 'content': 'prompt', 'message_uid': 'u1', '_row_id': 1},
                 {'role': 'user', 'content': 'prompt'}, composite):
        assert persisted_turn_receipt(
            [composite, *tail], 0, [], 'done', 'complete', compression_unchanged=False,
            prefix_row_matches=lambda a, b: a is b, turn_message=turn, turn_override='prompt',
            row_identity=rows.get) == expected


class _FaultingStore:
    """The real SessionDB, with ``get_message_identity`` raising in one receipt phase only. The
    phase is read from the live call stack, so the test proves which phase actually failed."""

    def __init__(self, db, phase: str):
        self._db, self._phase, self.calls = db, phase, []

    def get_message_identity(self, session_id, row_id):
        names = {frame.name for frame in traceback.extract_stack()}
        phase = 'proof' if '_prompt_row_id' in names else 'verify' if 'verify_receipt_rows' in names else '?'
        failed = phase == self._phase
        self.calls.append((phase, row_id, failed))
        if failed:
            raise sqlite3.OperationalError(f'injected {phase} identity read failure')
        return self._db.get_message_identity(session_id, row_id)

    def __getattr__(self, name):
        return getattr(self._db, name)


def _assert_phase_hit(store, phase):
    assert any(p == phase and failed for p, _row, failed in store.calls), store.calls
    if phase == 'verify':  # the proof read ran and succeeded first: this is the verification phase
        assert any(p == 'proof' and not failed for p, _row, failed in store.calls), store.calls


def _gateway_turn(agent_state: dict, result: dict):
    """One turn through the gateway's real ``TurnRunner.run_sync`` and the settlement's
    ``completion_payload`` (the authority's message.complete)."""
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_results import execution_result, prepare_result
    from gateway.session_settlement_recovery import completion_payload
    from gateway.turn_context import TurnContext
    from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _RecordingAgent

    class _FoldTurnAgent(_RecordingAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            vars(self).update(agent_state)

        def run_conversation(self, message, **kwargs):
            return dict(result)

    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id='sid', user_id='u'),
        message='prompt', history=[], session_id='sid', session_key='key', user_config={},
        AIAgent=_FoldTurnAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    captured = {}
    token = execution_result.set(captured)
    try:
        turn = TurnRunner(_gateway_runner(), ctx)
        turn._approval_owner = (SimpleNamespace(sessions={'sid': SimpleNamespace(tool_progress_mode=None)},
                                                profile_id='p', adopt_agent=lambda *a: None), 'sid', 1)
        delivered = turn.run_sync()
    finally:
        execution_result.reset(token)
    row = {'admission_id': 'a1', 'request_id': 'r1', 'payload': {'text': 'prompt'}}
    outcome, _stored = prepare_result(row, delivered['final_response'], 'completed', captured)
    return delivered, completion_payload(row, {'outcome': outcome}, delivered['final_response'], captured)


def _tui_turn(monkeypatch, tmp_path, agent_state: dict, result: dict) -> list:
    """One turn through the TUI's real ``_run_prompt_submit``: the emitted message.complete frames."""
    from tests.tui_gateway.test_none_agent_turn_guard import _session, _turn_env
    from tui_gateway import server
    emitted = _turn_env(monkeypatch, tmp_path)
    agent = SimpleNamespace(run_conversation=lambda *a, **k: dict(result), clear_interrupt=lambda: None,
                            **agent_state)
    server._run_prompt_submit('rid', 'ui-sid', _session(agent), 'prompt')
    return [payload for kind, _sid, payload in emitted if kind == 'message.complete']


def _fault_witness(tmp_path, phase):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'fault.db')
    w = fold_witness(db, 'notice', False)  # the proven shape: it needs both proof and verification reads
    state = dict(vars(w['agent']))
    store = _FaultingStore(db, phase) if phase else db
    state['_session_db'] = store
    state['context_compressor'] = SimpleNamespace(compression_count=0, last_prompt_tokens=0, context_length=200_000)
    expected = {'row_ids': [w['originals']['prompt']['row_id'], *[m['_row_id'] for m in w['result']['messages'][1:]]],
                'complete': False, 'user_row_id': w['originals']['prompt']['row_id'],
                'user_row_ids': [w['originals']['prompt']['row_id'], w['originals']['later']['row_id']],
                'final_assistant_row_id': w['final_row_id']}
    return db, store, state, w['result'], expected


@pytest.mark.parametrize('phase', [None, 'proof', 'verify'])
def test_gateway_completion_survives_a_store_failure_and_omits_the_receipt(tmp_path, phase):
    db, store, state, result, expected = _fault_witness(tmp_path, phase)
    try:
        delivered, complete = _gateway_turn(state, result)
        assert delivered['final_response'] == 'done'
        assert (complete['status'], complete['outcome'], complete['text']) == ('complete', 'completed', 'done')
        assert 'error_surface' not in complete
        if phase is None:  # positive control: the same turn publishes the proven receipt
            assert delivered['persisted_turn'] == expected
            assert complete['persisted_turn'] == {**expected, 'submission_id': 'r1'}
        else:
            _assert_phase_hit(store, phase)
            assert 'persisted_turn' not in delivered and 'persisted_turn' not in complete
    finally:
        db.close()


@pytest.mark.parametrize('phase', [None, 'proof', 'verify'])
def test_tui_completion_survives_a_store_failure_and_omits_the_receipt(monkeypatch, tmp_path, phase):
    db, store, state, result, expected = _fault_witness(tmp_path, phase)
    try:
        [complete] = _tui_turn(monkeypatch, tmp_path, state, result)
        assert complete['status'] == 'complete' and complete['text'] == 'done'
        assert 'error' not in complete
        if phase is None:  # positive control
            assert complete['persisted_turn'] == expected
        else:
            _assert_phase_hit(store, phase)
            assert 'persisted_turn' not in complete
    finally:
        db.close()


@pytest.mark.parametrize('phase', ['proof', 'verify'])
def test_the_one_boundary_covers_both_phases_for_both_producers(tmp_path, phase):
    """Direct producer calls: neither producer raises, both return None, in either phase."""
    db, store, state, result, _expected = _fault_witness(tmp_path, phase)
    try:
        got = both_receipts(SimpleNamespace(**state), result)
        assert got == {'gateway': None, 'tui': None}
        _assert_phase_hit(store, phase)
    finally:
        db.close()
