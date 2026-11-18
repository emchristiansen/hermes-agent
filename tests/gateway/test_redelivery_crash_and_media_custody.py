"""msg-retry-on-13f-9: an owed redelivery survives a cut-short boot and keeps its accepted media.

MR-RECOVERY-17: ``initialize_session_authority`` fenced a lost turn ``started``→``unknown`` in its own commit while
only a volatile list read before it named the no-tool turns to queue again. A boot that failed after the fence
left the next boot nothing to find: the head stayed ``unknown`` forever and its follower stayed queued behind it.
The fence now owes the redelivery in the same write, so the next boot still runs the head, then the follower.
MR-MEDIA-18: a genuine pre-bootstrap no-tool stop settles the turn ``terminal`` and owes its redelivery, but the
settlement's media collector treated every terminal row's native attachments as unheld and deleted the PNG; the
next boot queued the turn and ``restore_attachments`` refused ``storage_unavailable``. An owed redelivery now
holds its inputs until the next owner queues it.

Real SQLite session store, epochs, claims, fence, settlement and media collector; the turn executor is a local
double that only records what it was given (the reviewer's msgretry8-r19546 boot-cut and gateway probe-v2
scenarios, ported). A tool-bearing lost turn is still never run again.
Eric (canonical ea298f05): "I just want all messages to be delivered at least once."
"""
import hashlib
from pathlib import Path

import pytest

from gateway import session_authority as sa
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import LiveSession
from gateway.session_contract import SessionRef
from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission
from tests.gateway.test_unknown_automation_settles import _runner


def _cut_after_fence(monkeypatch):
    """The boot's fence commits, then the boot fails (before its redelivery step)."""
    real = sa.recover_session_inputs

    def committed_then_cut(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError('boot interrupted after the durable recovery fence')
    monkeypatch.setattr(sa, 'recover_session_inputs', committed_then_cut)
    return real


async def _lost_head(runner, db, *, tool_step):
    first = await sa.initialize_session_authority(runner, profile_id='p', instance_id='first')
    first.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    head = admit_session_input(db, epoch=first.epoch, principal_id='person', session_id='s',
                               request_id='head', payload={'text': 'HEAD'})
    follow = admit_session_input(db, epoch=first.epoch, principal_id='person', session_id='s',
                                 request_id='follow', payload={'text': 'FOLLOW'})
    claim_session_input(db, epoch=first.epoch, session_id='s')
    db.append_message('s', 'user', 'HEAD')
    if tool_step:
        db.append_message('s', 'tool', 'an already-recorded effect', tool_call_id='call-1', tool_name='terminal')
    return head, follow  # the first owner dies here, its turn in flight


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_step', [False, True], ids=['no-tool', 'tool-evidence'])
@pytest.mark.parametrize('cut', [False, True], ids=['uninterrupted', 'cut-after-fence'])
async def test_a_lost_no_tool_turn_runs_again_even_after_a_boot_cut_short_after_its_fence(
        tmp_path, monkeypatch, tool_step, cut):
    """The next full boot runs the lost no-tool head, then its follower, whether or not a boot before it failed
    after the fence. A lost turn that reached a tool stays ``unknown`` and is never run again;
    its independent follower completes, cut or not."""
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    executed = []
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    try:
        db.create_session('s', source='telegram')
        head, follow = await _lost_head(runner, db, tool_step=tool_step)
        if cut:
            real = _cut_after_fence(monkeypatch)
            with pytest.raises(OSError, match='boot interrupted'):
                await sa.initialize_session_authority(runner, profile_id='p', instance_id='cut-boot')
            assert get_session_admission(db, admission_id=head['admission_id'])['status'] == 'unknown'
            monkeypatch.setattr(sa, 'recover_session_inputs', real)
        boot = await sa.initialize_session_authority(runner, profile_id='p', instance_id='next-boot')
        retained = get_session_admission(db, admission_id=head['admission_id'])
        boot.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
        boot._schedule(SessionRef('p', 's'))
        await boot.sessions['s'].task
        statuses = [get_session_admission(db, admission_id=row['admission_id'])['status'] for row in (head, follow)]
        if tool_step:
            assert statuses == ['unknown', 'terminal'] and executed == ['FOLLOW']
            assert get_session_admission(db, admission_id=head['admission_id']) == retained
            assert retained['status'] == 'unknown' and retained['outcome'] is None
            assert db._read_one('SELECT 1 FROM state_meta WHERE key=?',
                                (REDELIVER_PREFIX + head['admission_id'],)) is None
            assert get_session_admission(db, admission_id=follow['admission_id'])['outcome'] == 'completed'
        else:
            assert statuses == ['terminal', 'terminal'] and executed == ['HEAD', 'FOLLOW']
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('disposition', ['operator-resolved', 'stale-claim'])
async def test_an_owed_lost_turn_the_owner_no_longer_owes_is_never_replayed(tmp_path, monkeypatch, disposition):
    """The fence's owed marker names its claim and status: a lost turn an operator resolved after the cut boot
    (``resolve_unknown_session_input``), or a marker of another claim (epoch/generation), never runs again; the
    follower then runs."""
    import json
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    from hermes_state_runtime import begin_runtime_epoch, resolve_unknown_session_input
    executed = []
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    try:
        db.create_session('s', source='telegram')
        head, follow = await _lost_head(runner, db, tool_step=False)
        real = _cut_after_fence(monkeypatch)
        with pytest.raises(OSError, match='boot interrupted'):
            await sa.initialize_session_authority(runner, profile_id='p', instance_id='cut-boot')
        monkeypatch.setattr(sa, 'recover_session_inputs', real)
        lost = get_session_admission(db, admission_id=head['admission_id'])
        key = REDELIVER_PREFIX + head['admission_id']
        assert json.loads(db._read_one('SELECT value FROM state_meta WHERE key=?', (key,))[0])['status'] == 'unknown'
        if disposition == 'operator-resolved':
            resolve_unknown_session_input(db, epoch=begin_runtime_epoch(db, instance_id='operator'),
                                          admission_id=head['admission_id'], generation=lost['generation'])
        else:
            db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(
                {'epoch': lost['owner_epoch'], 'generation': lost['generation'] + 1, 'status': 'unknown'}), key)))
        boot = await sa.initialize_session_authority(runner, profile_id='p', instance_id='next-boot')
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (key,)) is None  # the stale marker is dropped
        retained = get_session_admission(db, admission_id=head['admission_id'])
        boot.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
        boot._schedule(SessionRef('p', 's'))
        await boot.sessions['s'].task
        statuses = [get_session_admission(db, admission_id=row['admission_id'])['status'] for row in (head, follow)]
        if disposition == 'operator-resolved':  # resolved: never replayed; the FIFO advances
            assert statuses == ['terminal', 'terminal'] and executed == ['FOLLOW']
        else:  # not this claim's marker: HEAD stays fenced; independent FOLLOW completes
            assert statuses == ['unknown', 'terminal'] and executed == ['FOLLOW']
            assert get_session_admission(db, admission_id=head['admission_id']) == retained == lost
            assert retained['outcome'] is None
            assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (key,)) is None
            assert get_session_admission(db, admission_id=follow['admission_id'])['outcome'] == 'completed'
    finally:
        store.close_all_db_handles()


async def _stop_before_spawn_with_media(tmp_path, monkeypatch, executed):
    """A person's PNG turn the owner stops before bootstrap (the actual pre-bootstrap producer), settled by the
    actual outer settlement and its media collector. Returns (store, runner, admission, reference, digest)."""
    from gateway.session_ingress_media import capture_native_media
    from gateway.session_managed_worker import _stopped_before_spawn
    from gateway.session_settlement_recovery import settle_and_deliver
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    db.create_session('s', source='telegram')
    first = await sa.initialize_session_authority(runner, profile_id='p', instance_id='first')
    live = first.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='m'), 's')
    source = tmp_path / 'photo.png'
    source.write_bytes(b'\x89PNG\r\n\x1a\ninert image fixture')
    (reference,) = capture_native_media([source])
    admission = admit_session_input(db, epoch=first.epoch, principal_id='person', session_id='s',
                                    request_id='image', payload={'text': 'read my attached image', 'attachments_v1':
                                                                 {'media': [reference], 'media_types': ['image/png']}})
    row = claim_session_input(db, epoch=first.epoch, session_id='s')
    ref = SessionRef('p', 's')
    first._stamp_claim(ref, live, row)
    first._managed_spawning = {'s': row['generation']}
    first.pending_stops['s'] = row['generation']
    first.pending_stop_causes['s'] = (row['generation'], 'Gateway shutting down')
    assert _stopped_before_spawn(first, ref, row)
    await settle_and_deliver(first, live, ref, row, '', 'interrupted')
    await first._media_release_task
    return store, runner, admission, reference, hashlib.sha256(source.read_bytes()).hexdigest()


async def _settled_alias(db, tmp_path):
    """A second person's message carrying the same PNG (same name and bytes, so the same retained file),
    settled ``completed`` (owes nothing)."""
    from gateway.session_ingress_media import capture_native_media
    from hermes_state_runtime import begin_runtime_epoch, settle_session_input
    (tmp_path / 'again').mkdir()
    copy = tmp_path / 'again' / 'photo.png'
    copy.write_bytes((tmp_path / 'photo.png').read_bytes())
    epoch = begin_runtime_epoch(db, instance_id='alias-owner')
    db.create_session('alias', source='telegram')
    admission = admit_session_input(db, epoch=epoch, principal_id='person', session_id='alias', request_id='again',
                                    payload={'text': 'again', 'attachments_v1': {
                                        'media': capture_native_media([copy]), 'media_types': ['image/png']}})
    row = claim_session_input(db, epoch=epoch, session_id='alias')
    settle_session_input(db, epoch=epoch, admission_id=admission['admission_id'], generation=row['generation'],
                         outcome='completed')
    return admission


@pytest.mark.asyncio
async def test_an_owed_redelivery_keeps_its_attachment_until_the_next_owner_runs_it(tmp_path, monkeypatch):
    from gateway.session_ingress_media import release_admission_media, sweep_settled_media
    from gateway.session_message_redelivery import REDELIVER_PREFIX
    executed, seen = [], []
    store, runner, admission, reference, digest = await _stop_before_spawn_with_media(tmp_path, monkeypatch, executed)
    db = store._db
    answer = runner._handle_message

    async def record_media(event):
        seen.extend(hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in event.media_urls)
        return await answer(event)
    runner._handle_message = record_media
    try:
        current = get_session_admission(db, admission_id=admission['admission_id'])
        assert (current['status'], current['outcome']) == ('terminal', 'interrupted')
        assert db._read_one('SELECT 1 FROM state_meta WHERE key=?', (REDELIVER_PREFIX + admission['admission_id'],))
        assert Path(reference['path']).exists(), 'the settlement collector deleted input owed to a redelivery'
        # Every other collector of terminal inputs holds it too: a repeat release and the age sweep.
        assert release_admission_media(db, admission['admission_id']) == 0
        assert sweep_settled_media(db, now=10**12) == 0 and Path(reference['path']).exists()
        # An alias: another, ordinary terminal admission of the same bytes (the same content-addressed file)
        # releases its own reference; the owed row still holds the file.
        alias = await _settled_alias(db, tmp_path)
        assert get_session_admission(db, admission_id=alias['admission_id'])['payload']['attachments_v1'][
            'media'] == [reference]
        assert release_admission_media(db, alias['admission_id']) == 0 and Path(reference['path']).exists()
        boot = await sa.initialize_session_authority(runner, profile_id='p', instance_id='second')
        assert get_session_admission(db, admission_id=admission['admission_id'])['status'] == 'queued'
        boot.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='m'), 's')
        boot._schedule(SessionRef('p', 's'))
        await boot.sessions['s'].task
        assert executed == ['read my attached image'] and seen == [digest]  # the same bytes, run again
        assert get_session_admission(db, admission_id=admission['admission_id'])['status'] == 'terminal'
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_ordinary_terminal_turn_that_owes_no_redelivery_still_releases_its_attachment(tmp_path, monkeypatch):
    """Control (green before the repair too): a settled turn that owes nothing does not hold its bytes; the
    ordinary collection deletes them."""
    from gateway.session_ingress_media import release_admission_media
    store, _ = _runner(tmp_path, monkeypatch, [])
    db = store._db
    try:
        (tmp_path / 'photo.png').write_bytes(b'\x89PNG\r\n\x1a\ninert image fixture')
        admission = await _settled_alias(db, tmp_path)
        (reference,) = get_session_admission(db, admission_id=admission['admission_id'])['payload'][
            'attachments_v1']['media']
        assert Path(reference['path']).exists()
        assert release_admission_media(db, admission['admission_id']) == 1 and not Path(reference['path']).exists()
    finally:
        store.close_all_db_handles()
