"""Bot producer discovery and authority admission must agree after a local reset."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_state_runtime import list_session_admissions

pytest_plugins = ['tests.gateway.test_session_bot_retry', 'tests.gateway.test_peer_dm_bot_chat_headless']


@pytest.mark.parametrize('compress_after_reset', [False, True])
@pytest.mark.asyncio
async def test_discovered_bot_delivery_reaches_reset_transcript_once(bot, monkeypatch, compress_after_reset):
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from tests.gateway.test_session_bot_retry import _settle_quietly
    from tools import bot_live_delivery as mailbox

    authority = bot.authority
    actor = replace(bot.connection.actor,
                    capabilities=bot.connection.actor.capabilities | {'session:control'})
    row = authority.db.get_session(bot.chat)
    reset = await mutate_session(authority, actor, SessionRef(authority.profile_id, bot.chat), dict(
        session_id=bot.chat, request_id='reset-for-delivery', expected_revision=row['runtime_revision'],
        expected_generation=row['runtime_generation'], operation='reset', payload={}))
    physical = reset['target_session_id']
    assert physical != bot.chat
    if compress_after_reset:
        from tests.gateway.test_local_resume_compression_child import _compress
        _compress(authority.db, physical, 'compressed-reset', 'gui')
        physical = 'compressed-reset'
    loop = asyncio.get_running_loop()
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                        lambda *args, **kwargs: SimpleNamespace(
                            state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
    monkeypatch.setattr(mailbox, 'authority_delivery', lambda home, params:
                        asyncio.run_coroutine_threadsafe(deliver(bot.connection, params), loop).result())
    key = 'a' * 32

    def send():
        owner = mailbox.find_canonical_live_owner(bot.home)
        return mailbox.deliver_to_live_owner(bot.home, owner, 'after reset', delivery_id=key)

    first = await asyncio.to_thread(send)
    await _settle_quietly(bot)
    again = await asyncio.to_thread(send)
    await _settle_quietly(bot)
    assert first['admission_id'] == again['admission_id']
    assert again['status'] == 'settled'
    rows = list_session_admissions(authority.db, session_id=bot.chat, pending_only=False)
    assert [r['request_id'] for r in rows] == ['bot:' + key]
    live = authority.sessions[bot.chat]
    assert authority.runner.session_store.lookup_by_session_key(live.route).session_id == physical


@pytest.mark.asyncio
async def test_discovery_pin_cannot_cross_a_later_reset_or_wrong_chat(bot, monkeypatch):
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError
    from tools import bot_live_delivery as mailbox

    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                        lambda *args, **kwargs: SimpleNamespace(
                            state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
    stale = mailbox.find_canonical_live_owner(bot.home)
    row = bot.authority.db.get_session(bot.chat)
    actor = replace(bot.connection.actor,
                    capabilities=bot.connection.actor.capabilities | {'session:control'})
    await mutate_session(bot.authority, actor, SessionRef('default', bot.chat), dict(
        session_id=bot.chat, request_id='later-reset', expected_revision=row['runtime_revision'],
        expected_generation=row['runtime_generation'], operation='reset', payload={}))
    for pin in (stale['session_id'], 'unrelated-chat'):
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver(bot.connection, dict(id='b' * 32, profile='default',
                                              message='must not move', session_id=pin))
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []
    assert mailbox.read_delivery_result(bot.home, 'b' * 32) is None


@pytest.mark.parametrize('reset_first', [False, True])
@pytest.mark.asyncio
async def test_discovery_leaves_classic_compression_adoption_to_authority(bot, reset_first):
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from tests.gateway.test_local_receipt_untracked_compression import _compact_like_0_22_2
    from tools.bot_live_delivery import _bot_chat_tip

    physical = bot.chat
    if reset_first:
        actor = replace(bot.connection.actor,
                        capabilities=bot.connection.actor.capabilities | {'session:control'})
        row = bot.authority.db.get_session(bot.chat)
        reset = await mutate_session(bot.authority, actor, SessionRef('default', bot.chat), dict(
            session_id=bot.chat, request_id='reset-before-classic', expected_revision=row['runtime_revision'],
            expected_generation=row['runtime_generation'], operation='reset', payload={}))
        physical = reset['target_session_id']
    _compact_like_0_22_2(bot.authority.db, physical, 'classic-child')
    assert _bot_chat_tip(bot.home) == 'classic-child'


@pytest.mark.parametrize('compress_after_reset', [False, True])
@pytest.mark.asyncio
async def test_legacy_titled_ancestor_reaches_adopted_reset_once(headless, monkeypatch, compress_after_reset):
    from pathlib import Path
    from gateway.session_bot import deliver, _target
    from gateway.session_mutations import mutate_session
    from tests.gateway.test_peer_dm_bot_chat_headless import _local_owner
    from tests.gateway.test_session_bot_retry import _settle_quietly
    from tests.gateway.test_local_receipt_untracked_compression import _compact_like_0_22_2
    from tests.gateway.test_local_resume_compression_child import _compress
    from tools import bot_live_delivery as mailbox

    authority = headless.authority
    home = Path(authority.db.db_path).parent
    authority.db.create_session('legacy-ancestor', 'tui', model='fixture')
    authority.db.set_session_title('legacy-ancestor', 'Bot Chat')
    _compact_like_0_22_2(authority.db, 'legacy-ancestor', 'legacy-adopted')
    connection = _local_owner(authority)
    try:
        ref, _, _ = await _target(authority, connection.actor)
        assert ref.session_id == 'legacy-adopted'
        assert authority.db.get_session_by_title('Bot Chat')['id'] == 'legacy-ancestor'
        row = authority.db.get_session(ref.session_id)
        reset = await mutate_session(authority, connection.actor, ref, dict(
            session_id=ref.session_id, request_id='legacy-reset', expected_revision=row['runtime_revision'],
            expected_generation=row['runtime_generation'], operation='reset', payload={}))
        physical = reset['target_session_id']
        if compress_after_reset:
            _compress(authority.db, physical, 'legacy-reset-compressed', 'gui')
            physical = 'legacy-reset-compressed'
        loop = asyncio.get_running_loop()
        monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                            lambda *args, **kwargs: SimpleNamespace(
                                state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
        monkeypatch.setattr(mailbox, 'authority_delivery', lambda home, params:
                            asyncio.run_coroutine_threadsafe(deliver(connection, params), loop).result())
        key = 'f' * 32

        def send():
            owner = mailbox.find_canonical_live_owner(home)
            return mailbox.deliver_to_live_owner(home, owner, 'legacy reset delivery', delivery_id=key)

        first = await asyncio.to_thread(send)
        await _settle_quietly(headless)
        again = await asyncio.to_thread(send)
        await _settle_quietly(headless)
        assert mailbox.find_canonical_live_owner(home)['session_id'] == physical
        assert first['admission_id'] == again['admission_id'] and again['status'] == 'settled'
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert [r['request_id'] for r in rows] == ['bot:' + key]
        assert [text for _, _, text in headless.executed] == ['legacy reset delivery']
        live = authority.sessions[ref.session_id]
        assert authority.runner.session_store.lookup_by_session_key(live.route).session_id == physical
    finally:
        await connection.close()
