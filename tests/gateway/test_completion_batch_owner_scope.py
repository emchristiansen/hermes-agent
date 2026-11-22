"""A completion batch runs in its events' pinned owner scope, not in the context that created it.

INT-R2-PROFILE-01 (review t_00fdc6c6/run4793): a root-scoped consumer enqueued a served profile's
completions. The flush task inherited the ROOT context, so its one off-loop admission read ran against the
launch authority and recognised nothing. The per-event delivery wrapper did restore the scope: it found the
old (lost-ACK) completion admitted and returned True. That success ended the candidate loop and marked the
fresh sibling delivered, though it was never admitted. The enqueue now pins the owner (its scope context
and home): the flush, its admission reads, recognition and delivery all run there, and distinct owners
never share a batch.

The first test is the reviewer's probe, adopted (only the evidence output moved into the test). It drives
real profile authorities, stores and the production enqueue -> flush -> admission -> delivery path. Only
the batch-window sleep is held, so both entries occupy one batch.
"""
import asyncio
import threading

import pytest


async def _multiplex(tmp_path, monkeypatch, names=("alpha",)):
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authority import SessionAuthority
    from gateway.config import Platform, PlatformConfig, GatewayConfig
    from plugins.platforms.discord.adapter import DiscordAdapter
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=names)
    monkeypatch.setenv('HERMES_BASE_HOME', str(root))
    for name, home in homes:
        (home / '.env').write_text(f'REVIEW_TEST_KEY={name}\nDISCORD_ALLOWED_USERS=42\n')
        (home / 'config.yaml').write_text('terminal:\n  env: local\n  cwd: ' + str(home) + '\n')
    process_ownership.reserve([h for _, h in homes])
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = _runner(root, homes)
    await initialize_gateway_runtime(runner)
    adapters = {}
    for name, home in homes:
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
        adapter.profile_name = None if name == 'default' else name
        adapter.gateway_runner = runner
        if name == 'default':
            runner._wire_adapter_handlers(adapter)
        else:
            runner._configure_profile_adapter(adapter, name, Platform.DISCORD)
        adapters[name] = adapter
    runner.adapters = {Platform.DISCORD: adapters['default']}
    runner._profile_adapters = {n: {Platform.DISCORD: a} for n, a in adapters.items() if n != 'default'}
    runner._profile_configs = {n: GatewayConfig() for n, h in homes if n != 'default'}
    return root, homes, runner, adapters


async def _close(runner, homes):
    from gateway.runtime_ownership import process_ownership
    await runner._cancel_process_completion_batch_tasks()
    runner.close_all_session_db_handles()
    runner.session_store.close_all_db_handles()
    for _, h in homes:
        process_ownership.release(h)


async def _old_and_fresh(runner, adapters, homes, name, chat_id):
    """Admit a human input and one completion (OLD) in *name*'s own scope; return (authority, root, old, fresh)."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress_context import native_callback
    from gateway import session_automation
    home = dict(homes)[name]
    authority = runner.session_authorities.for_home(home)
    source = adapters[name].build_source(chat_id=chat_id, chat_type='dm', user_id='42')
    async with _async_profile_runtime_scope(home):
        event = MessageEvent(text='human', source=source, message_id=f'human-{name}')
        with native_callback(runner, event, home, profile=None if name == 'default' else name):
            receipt = await authority.admit_native(event)
        key = runner.session_store._generate_session_key(source)
        old = dict(type='completion', session_id=f'proc_old_{name}', started_at=1.0, session_key=key,
                   platform='discord', chat_type='dm', chat_id=chat_id, user_id='42',
                   parent_session_id=receipt.ref.session_id, exit_code=0, output=f'OLD-{name}')
        fresh = dict(old, session_id=f'proc_fresh_{name}', started_at=2.0, output=f'FRESH-{name}')
        assert await runner._deliver_completion_notification(f'OLD-{name}', old) is True
        assert session_automation.completion_admission(runner, old) is not None
        assert session_automation.completion_admission(runner, fresh) is None
    return home, authority, receipt.ref.session_id, old, fresh


def _hold_batch_window(monkeypatch, runner):
    entered, release = asyncio.Event(), asyncio.Event()
    original_sleep = asyncio.sleep

    async def sleep(delay):
        if delay == 0.173:
            entered.set()
            await release.wait()
        else:
            await original_sleep(delay)
    runner._completion_notification_batch_window = 0.173
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    return entered, release


async def _admitted_in(runner, home, evt):
    from gateway.run import _async_profile_runtime_scope
    from gateway import session_automation
    async with _async_profile_runtime_scope(home):
        return session_automation.completion_admission(runner, evt) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('ambient', ['root', 'owner', 'root-no-observer'])
async def test_multiplex_completion_batch_lost_ack_plus_fresh(tmp_path, monkeypatch, ambient):
    from gateway.run import _async_profile_runtime_scope
    from gateway import session_automation
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret
    from tools.terminal_scope import terminal_env
    from gateway.session_authorities import active_authority
    from hermes_state_runtime import list_session_admissions
    from tests.waiting import async_wait_until
    _root, homes, runner, adapters = await _multiplex(tmp_path, monkeypatch)
    real = session_automation.completion_admission
    observed = []

    def observed_read(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            return type(exc).__name__

    def observe(r, event):
        result = real(r, event)
        observed.append({'process': event['session_id'], 'home': str(get_hermes_home()),
                         'authority': active_authority(r).profile_id,
                         'secret': observed_read(get_secret, 'REVIEW_TEST_KEY'),
                         'cwd': observed_read(terminal_env, 'TERMINAL_CWD'),
                         'thread': threading.get_ident(), 'admitted': result is not None})
        return result
    try:
        alpha, authority, root_sid, old, fresh = await _old_and_fresh(runner, adapters, homes, 'alpha', '42')
        if ambient != 'root-no-observer':
            monkeypatch.setattr(session_automation, 'completion_admission', observe)
        entered, release = _hold_batch_window(monkeypatch, runner)

        async def exercise():
            one = asyncio.create_task(runner._enqueue_process_completion_notification('OLD', old))
            await asyncio.wait_for(entered.wait(), 30)
            two = asyncio.create_task(runner._enqueue_process_completion_notification('FRESH', fresh))
            await async_wait_until(lambda: any(len(v) == 2 for v in runner._completion_notification_batches.values()),
                                   timeout=30, what='two batch entries')
            release.set()
            return await asyncio.gather(one, two)
        if ambient == 'owner':
            async with _async_profile_runtime_scope(alpha):
                outcome = await exercise()
        else:
            outcome = await exercise()
        fresh_admitted = await _admitted_in(runner, alpha, fresh)
        rows = list_session_admissions(authority.db, session_id=root_sid, pending_only=False)
        evidence = {'ambient': ambient, 'outcome': outcome, 'observed': observed, 'fresh_admitted': fresh_admitted,
                    'payload_texts': [r['payload'].get('text') for r in rows],
                    'fresh_marked_delivered':
                        runner._completion_delivery_identity(fresh) in runner._completion_deliveries_delivered}
        assert outcome == [True, True], evidence
        assert fresh_admitted, evidence  # every acknowledged fresh identity is in alpha's own ledger
        assert all(x['home'] == str(alpha) and x['authority'] == str(alpha)
                   and x['secret'] == 'alpha' and x['cwd'] == str(alpha) for x in observed), evidence
    finally:
        await _close(runner, homes)


@pytest.mark.asyncio
async def test_root_scoped_batches_of_two_served_owners_stay_isolated(tmp_path, monkeypatch):
    from tests.waiting import async_wait_until
    _root, homes, runner, adapters = await _multiplex(tmp_path, monkeypatch, names=('alpha', 'beta'))
    try:
        legs = {name: await _old_and_fresh(runner, adapters, homes, name, chat)
                for name, chat in (('alpha', '42'), ('beta', '43'))}
        _entered, release = _hold_batch_window(monkeypatch, runner)
        # Interleaved across the two owners: alpha OLD, beta OLD, alpha FRESH, beta FRESH.
        order = [(legs[n][3], 'OLD') for n in ('alpha', 'beta')] + [(legs[n][4], 'FRESH') for n in ('alpha', 'beta')]
        tasks = []
        for evt, text in order:
            tasks.append(asyncio.create_task(runner._enqueue_process_completion_notification(text, evt)))
            await asyncio.sleep(0)
        await async_wait_until(lambda: sorted(map(len, runner._completion_notification_batches.values())) == [2, 2],
                               timeout=30, what='two two-entry batches')
        assert len({key[-1] for key in runner._completion_notification_batches}) == 2  # one pinned owner each
        release.set()
        assert await asyncio.gather(*tasks) == [True] * 4
        for name, (home, _auth, _sid, _old, fresh) in legs.items():
            other = 'beta' if name == 'alpha' else 'alpha'
            assert await _admitted_in(runner, home, fresh), name
            assert not await _admitted_in(runner, legs[other][0], fresh), (name, 'admitted in', other)
    finally:
        await _close(runner, homes)


@pytest.mark.asyncio
async def test_default_profile_batch_lost_ack_plus_fresh_in_a_multiplexed_gateway(tmp_path, monkeypatch):
    """The launch profile's own route under the same root-scoped consumer: unchanged, and admitted at root."""
    from tests.waiting import async_wait_until
    _root, homes, runner, adapters = await _multiplex(tmp_path, monkeypatch)
    try:
        home, _auth, _sid, old, fresh = await _old_and_fresh(runner, adapters, homes, 'default', '44')
        entered, release = _hold_batch_window(monkeypatch, runner)
        one = asyncio.create_task(runner._enqueue_process_completion_notification('OLD', old))
        await asyncio.wait_for(entered.wait(), 30)
        two = asyncio.create_task(runner._enqueue_process_completion_notification('FRESH', fresh))
        await async_wait_until(lambda: any(len(v) == 2 for v in runner._completion_notification_batches.values()),
                               timeout=30, what='two batch entries')
        release.set()
        assert await asyncio.gather(one, two) == [True, True]
        assert await _admitted_in(runner, home, fresh)
        assert not await _admitted_in(runner, dict(homes)['alpha'], fresh)
    finally:
        await _close(runner, homes)
