"""msg-retry-on-13f-3: every notice states the branch the turn actually takes, the same on the live
``message.complete.warning`` and in the transcript boundary, and the next owner's boot does what it says.

Branches (``gateway.session_message_redelivery``):
  S1  a stopping owner settles a person's turn that reached no tool: owes the redelivery; the next owner's
      boot queues it again ("... will run again once Hermes reconnects / is back").
  S2  a stopping owner settles a turn that ran a tool (a business turn): never re-run ("... will not be run
      again. Send your message again ...").
  U   an unattended row: never re-run here; the neutral automation copy.
  D   the owner died (no settlement): the next owner's boot fences it unknown and, with no tool step, queues it
      again with REDELIVERY_NOTICE as its transcript boundary (no live frame: no viewer of a dead owner).
Eric's rule: "A replay is completely fine. I don't give a fuck whether the same message is sent twice. I just
want all messages to be delivered at least once."
"""
import pytest

from tests.gateway.test_restart_interrupt_notice import _notices, _settle_on_real_store

CAUSES = [('Gateway restarting', 'reconnects'), ('Gateway shutting down', 'is back')]


def _warning(frames):
    (complete,) = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    return complete.get('warning')


def _next_owner_boot(db):
    """The next owner's real boot order (``initialize_session_authority``): epoch, fence (which owes each lost
    no-tool turn its redelivery in the same write), the unattended settlement, then the redelivery."""
    from gateway.session_message_redelivery import redeliver_recovered_messages
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    epoch = begin_runtime_epoch(db, instance_id='next-owner')
    recover_session_inputs(db, epoch=epoch)
    settle_recovered_unattended(db, epoch)
    return epoch, redeliver_recovered_messages(db, epoch)


def _status(db, row):
    from hermes_state_runtime import get_session_admission
    current = get_session_admission(db, admission_id=row['admission_id'])
    return current['status'], current['outcome']


@pytest.mark.parametrize(('cause', 'when'), CAUSES)
def test_s1_no_tool_person_turn_both_notices_say_it_runs_again_and_the_next_owner_runs_it(tmp_path, cause, when):
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    db, _authority, row, frames, settlement = _settle_on_real_store(tmp_path, cause, tool_step=False)
    with db:
        (notice,) = _notices(db)
        assert notice.startswith(cause + ':') and 'will run again once Hermes ' + when in notice
        assert 'not be run again' not in notice and 'Send your message again' not in notice
        assert _warning(frames) == notice == settlement['settled']['owner_stop_notice']
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (REDELIVER_PREFIX + row['admission_id'],))
        assert _status(db, row) == ('terminal', 'interrupted')
        _epoch, queued = _next_owner_boot(db)
        assert queued == 1 and _status(db, row) == ('queued', None)  # what the notice promised happens


@pytest.mark.parametrize(('cause', 'when'), CAUSES)
def test_s2_business_turn_both_notices_say_it_will_not_run_again_and_the_next_owner_does_not(tmp_path, cause, when):
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    from gateway.session_settlement_recovery import RESTART_INTERRUPT_NOTICE, SHUTDOWN_INTERRUPT_NOTICE
    expected = RESTART_INTERRUPT_NOTICE if cause == 'Gateway restarting' else SHUTDOWN_INTERRUPT_NOTICE
    db, _authority, row, frames, settlement = _settle_on_real_store(tmp_path, cause, tool_step=True)
    with db:
        assert _notices(db) == [expected] and _warning(frames) == expected
        assert 'will not be run again' in expected and settlement['settled']['owner_stop_notice'] == expected
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (REDELIVER_PREFIX + row['admission_id'],)) is None
        _epoch, queued = _next_owner_boot(db)
        assert queued == 0 and _status(db, row) == ('terminal', 'interrupted')  # never re-run: as promised


@pytest.mark.parametrize('tool_step', [False, True], ids=['no-tool', 'tool'])
@pytest.mark.parametrize(('cause', 'when'), CAUSES)
def test_a_settlement_retry_publishes_the_notice_its_first_write_committed(tmp_path, cause, when, tool_step):
    """``finish_result`` re-run over the committed row (settle_with_retry after a commit that reported failure)."""
    from gateway.session_results import finish_result
    db, authority, row, frames, _settlement = _settle_on_real_store(tmp_path, cause, tool_step=tool_step)
    with db:
        settled, _response = finish_result(
            db, epoch=authority.epoch, row=row, response='partial reply', outcome='interrupted',
            result={'result': {'final_response': 'partial reply', 'messages': [], 'interrupted': True,
                               'interrupt_message': cause}})
        assert settled['owner_stop_notice'] == _warning(frames) == _notices(db)[0]
        assert len(_notices(db)) == 1


@pytest.mark.parametrize(('cause', 'when'), CAUSES)
def test_u_unattended_turn_keeps_the_neutral_copy_and_no_redelivery(tmp_path, cause, when):
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    db, _authority, row, frames, settlement = _settle_on_real_store(
        tmp_path, cause, principal='cron-owner', tool_step=False)
    with db:
        assert _notices(db) == [AUTOMATION_INTERRUPTED_NOTICE] and _warning(frames) == AUTOMATION_INTERRUPTED_NOTICE
        assert settlement['settled']['owner_stop_notice'] == AUTOMATION_INTERRUPTED_NOTICE
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (REDELIVER_PREFIX + row['admission_id'],)) is None


@pytest.mark.parametrize('tool_step', [False, True], ids=['no-tool', 'tool'])
def test_d_owner_died_the_boot_boundary_matches_what_the_boot_did(tmp_path, tool_step):
    from gateway.session_message_redelivery import REDELIVERY_NOTICE
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    admit_session_input(db, epoch=epoch, principal_id='person', session_id='s', request_id='r',
                        payload={'text': 'please work'})
    row = claim_session_input(db, epoch=epoch, session_id='s')
    db.append_message('s', 'user', 'please work')
    if tool_step:
        db.append_message('s', 'tool', 'done', tool_call_id='c1', tool_name='terminal')
    with db:  # the owner dies here: no settlement, no live frame
        _epoch, queued = _next_owner_boot(db)
        if tool_step:
            assert queued == 0 and _status(db, row)[0] == 'unknown' and REDELIVERY_NOTICE not in _notices(db)
        else:
            assert queued == 1 and _status(db, row) == ('queued', None) and _notices(db) == [REDELIVERY_NOTICE]


@pytest.mark.parametrize('retry_cause', ['Gateway shutting down', None], ids=['changed-cause', 'missing-cause'])
@pytest.mark.parametrize(('principal', 'tool_step'), [('person', False), ('person', True), ('cron-owner', False)],
                         ids=['no-tool', 'tool', 'unattended'])
def test_a_retry_with_a_changed_or_missing_capture_publishes_the_committed_notice(tmp_path, principal, tool_step,
                                                                                    retry_cause):
    """MR-NOTICE-05: the committed row is read back; its notice comes from durable state, not the retry's capture."""
    from gateway.session_contract import SessionRef
    from gateway.session_settlement_recovery import commit_and_publish
    db, authority, row, frames, _settlement = _settle_on_real_store(tmp_path, 'Gateway restarting',
                                                                    principal=principal, tool_step=tool_step)
    with db:
        (committed,) = _notices(db)
        retry = {'result': {'final_response': 'different retry text', 'messages': [], 'interrupted': True}}
        if retry_cause:
            retry['result']['interrupt_message'] = retry_cause
        settlement = {}
        commit_and_publish(authority, authority.sessions['s'], SessionRef('owned', 's'), row,
                           'different retry text', 'interrupted', retry, settlement)
        completes = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert _notices(db) == [committed]
        assert [c.get('warning') for c in completes] == [committed, committed]
        assert settlement['settled']['owner_stop_notice'] == committed


@pytest.mark.parametrize('tool_step', [False, True], ids=['no-tool', 'tool'])
def test_a_commit_reported_as_failed_is_retried_and_publishes_the_committed_notice(tmp_path, monkeypatch, tool_step):
    """MR-NOTICE-05: the settlement write commits, then its caller sees a storage error (the reply is lost);
    the real settle_with_retry retries, reads the committed row back and publishes that write's notice once."""
    import asyncio
    from gateway import session_results
    from gateway.session_settlement_recovery import settle_with_retry
    from tests.gateway.test_restart_interrupt_notice import _settle_on_real_store as settle
    real = session_results.settle_session_input
    calls = []

    def committed_then_lost(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result['status'])
        if len(calls) == 1:
            raise OSError('reply lost after commit')
        return result
    monkeypatch.setattr(session_results, 'settle_session_input', committed_then_lost)
    monkeypatch.setattr('gateway.session_settlement_recovery._SETTLE_RETRY_DELAYS_S', (0.0, 0.0))
    import gateway.session_settlement_recovery as recovery
    real_commit = recovery.commit_and_publish
    captured = {}

    def no_first_commit(authority, live, ref, row, response, outcome, cap, settlement):
        captured.update(authority=authority, live=live, ref=ref, row=row, cap=cap)
    monkeypatch.setattr(recovery, 'commit_and_publish', no_first_commit)
    db, authority, row, frames, _settlement = settle(tmp_path, 'Gateway restarting', tool_step=tool_step)
    monkeypatch.setattr(recovery, 'commit_and_publish', real_commit)
    with db:
        settlement = {}
        cap = captured['cap']
        asyncio.run(settle_with_retry(authority, captured['live'], captured['ref'], row, 'partial reply',
                                      'interrupted', cap, settlement))
        (notice,) = _notices(db)
        completes = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert calls == ['terminal'] and len(completes) == 1
        assert completes[0].get('warning') == notice == settlement['settled']['owner_stop_notice']
        assert ('will run again' in notice) == (not tool_step)


@pytest.mark.parametrize(('cause', 'when'), CAUSES)
@pytest.mark.parametrize(('principal', 'tool_step'), [('person', False), ('person', True), ('cron-owner', False)],
                         ids=['no-tool', 'tool', 'unattended'])
def test_the_fallback_after_a_reported_settlement_error_publishes_the_committed_notice(
        tmp_path, monkeypatch, cause, when, principal, tool_step):
    """MR-NOTICE-09: the settlement committed, then its caller saw an error, so ``settle_and_deliver``'s outer
    fallback re-reads the stored row and publishes it. The live warning is the committed notice (durable
    owed-redelivery decision), equal to the transcript boundary, never 'will not be run again' for a message that
    runs again. The direct publisher on the stored row agrees."""
    import asyncio
    import gateway.session_settlement_recovery as recovery
    from gateway.session_contract import SessionRef
    from gateway.session_results import admission_result
    from hermes_state_runtime import get_session_admission
    db, authority, row, frames, _settlement = _settle_on_real_store(tmp_path, cause, principal=principal,
                                                                    tool_step=tool_step)
    with db:
        (committed,) = _notices(db)
        current = get_session_admission(db, admission_id=row['admission_id'])
        captured = admission_result(db, row['admission_id'])
        frames.clear()
        recovery.publish_terminal(authority, SessionRef('owned', 's'), row, current,
                                  captured['result'].get('final_response') or '', captured)
        assert _warning(frames) == committed
        frames.clear()

        async def reported_failure(*args, **kwargs):
            raise OSError('reported failure after the write already committed')
        monkeypatch.setattr(recovery, 'settle_with_retry', reported_failure)
        asyncio.run(recovery.settle_and_deliver(authority, authority.sessions['s'], SessionRef('owned', 's'), row,
                                                'partial reply', 'interrupted'))
        assert _warning(frames) == committed == _notices(db)[0]
        assert ('will run again' in committed) == (principal == 'person' and not tool_step)
        assert 'will not be run again' not in committed or tool_step
