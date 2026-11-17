"""Crash/restart scenarios for the existing real shared-authority peer."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path

from gateway.platforms.event import MessageEvent
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until, wait_until


def rows(authority, sid):
    return list_session_admissions(authority.db, session_id=sid, pending_only=False)


async def verify_envelope_guards(runner, source):
    from gateway.session_contract import SessionActor, Submission
    from hermes_state_runtime import RuntimeStoreError
    authority = runner.session_authority
    ref = authority.register(source)
    before = rows(authority, ref.session_id)
    actor = SessionActor('authenticated-viewer', 'default', frozenset({'session:submit'}), 'fixture')
    try:
        await authority.submit(actor, Submission('forged-source', ref, {
            'text': 'FORGED', 'native_text_v1': {'source': {'internal': True}}}, 'queue'))
    except RuntimeStoreError as exc:
        assert exc.reason == 'invalid_params'
    else:
        raise AssertionError('client supplied its own trusted source')
    unsupported = [
        MessageEvent(text='MEDIA', source=source, media_urls=['/unretained/image.png']),
        MessageEvent(text='INTERNAL', source=source, internal=True),
        MessageEvent(text='RELAY', source=replace(source, delivered_via_upstream_relay=True)),
        MessageEvent(text='ROLE', source=replace(source, role_authorized=True)),
    ]
    for event in unsupported:
        try:
            await authority.admit_native(event)
        except RuntimeStoreError as exc:
            assert exc.reason == 'invalid_params'
        else:
            raise AssertionError('unsupported envelope ACKed without recoverable trust')
        assert not event._gateway_accepted
    assert rows(authority, ref.session_id) == before


async def prepare_crash(runner, adapter, source, peer):
    authority = runner.session_authority
    peer.blocked.clear()
    peer.release.clear()
    await adapter.handle_message(MessageEvent(text='BLOCK_FIFO_CRASH', source=source, message_id='crash-started'))
    assert await asyncio.to_thread(peer.blocked.wait, 5)
    sid = authority.register(source).session_id
    follower = asyncio.create_task(adapter.handle_message(MessageEvent(  # explicit: plain text would steer
        text='/queue UNKNOWN_FOLLOWER', source=source, message_id='unknown-follower')))
    await async_wait_until(lambda: any(r['request_id'] == 'unknown-follower' for r in rows(authority, sid)),
                           timeout=5, what='unknown-follower committed')
    # The receive coroutine is released at durable commit (serial IRC/Signal loops keep
    # reading); the follower itself stays queued behind the started head until the crash.
    await asyncio.wait_for(follower, 5)
    assert [r['status'] for r in rows(authority, sid) if r['request_id'] == 'unknown-follower'] == ['queued']
    # Trusted ingress commits before it schedules execution. Freeze THIS scheduling boundary,
    # not a claim/storage predicate: the owner dies with the safe row committed and unclaimed.
    safe_source = replace(source, chat_id='recover-chat', thread_id='recover-thread')
    authority._schedule = lambda ref: None
    accepted = await authority.admit_native(MessageEvent(text='RECOVER_EXACTLY_ONCE', source=safe_source,
                                                  message_id='crash-queued', reply_to_message_id='quote-7'))
    evidence = {'blocked_sid': sid, 'safe_sid': accepted.ref.session_id,
                'blocked': rows(authority, sid), 'safe': rows(authority, accepted.ref.session_id)}
    assert evidence['safe'][-1]['status'] == 'queued'
    assert any(r['status'] == 'started' for r in evidence['blocked'])
    # Published by rename: the parent SIGKILLs this process the moment the file EXISTS, so a plain
    # write_text (create+truncate, then write) could be killed between the two and leave an empty file.
    ready = Path(os.environ['HERMES_HOME'], 'crash-ready.json')
    ready.with_suffix('.tmp').write_text(json.dumps(evidence))
    os.replace(ready.with_suffix('.tmp'), ready)
    # Hold the loop frozen until the parent's SIGKILL; end early if the parent itself went away.
    parent = os.getppid()
    wait_until(lambda: os.getppid() != parent, timeout=60, interval=.05, what='the parent to SIGKILL this owner')
    raise AssertionError('parent exited without killing the actual owner')


async def recover_probe(runner, adapter, source, peer):
    from gateway.session_contract import SessionRef
    authority = runner.session_authority
    before = json.loads(Path(os.environ['HERMES_HOME'], 'crash-ready.json').read_text())
    safe_sid, blocked_sid = before['safe_sid'], before['blocked_sid']
    peer.release.set()  # the redelivered lost turn runs again: its model call must not wait on the crash gate
    safe_source = replace(source, chat_id='recover-chat', thread_id='recover-thread')
    binding = (safe_sid, safe_source, adapter)
    initial = rows(authority, safe_sid)
    initial_blocked = rows(authority, blocked_sid)
    # Current authorization, not yesterday's accepted boolean, must gate claim.
    os.environ['TELEGRAM_ALLOWED_USERS'] = 'different-user'
    denied = await authority.recover_native_sessions([binding])
    assert denied[safe_sid] == 'permission_denied', denied
    assert rows(authority, safe_sid) == initial
    os.environ['TELEGRAM_ALLOWED_USERS'] = 'fixture-user'
    ambiguous = await authority.recover_native_sessions([binding, binding])
    assert ambiguous[safe_sid] == 'admission_conflict', ambiguous
    assert rows(authority, safe_sid) == initial
    unavailable = runner.adapters.pop(source.platform)
    try:
        absent = await authority.recover_native_sessions([binding])
        assert absent[safe_sid] == 'not_found', absent
        assert rows(authority, safe_sid) == initial
    finally:
        runner.adapters[source.platform] = unavailable
    wrong = await authority.recover_native_sessions([(safe_sid, replace(safe_source, chat_id='wrong-chat'), adapter)])
    assert wrong[safe_sid] == 'admission_conflict', wrong
    assert rows(authority, safe_sid) == initial
    assert not peer.requests
    recovered = await authority.recover_native_sessions([binding, (blocked_sid, source, adapter)])
    # At-least-once (gateway/session_message_redelivery.py): the lost person's message, which reached no tool,
    # is queued again in its own FIFO place and runs again (a duplicate answer is acceptable; a lost message
    # is not), and the input queued behind it is not stranded: the session is ready, not blocked.
    assert recovered[blocked_sid] == 'ready', recovered
    for sid in (safe_sid, blocked_sid):
        task = authority.sessions[sid].task
        if task is not None:
            await asyncio.wait_for(task, 35)
    safe = rows(authority, safe_sid)
    blocked = rows(authority, blocked_sid)
    assert safe[0]['status'] == 'terminal' and safe[0]['outcome'] == 'completed', safe
    assert safe[0]['payload'] == before['safe'][0]['payload']
    lost = next(r for r in blocked if r['request_id'] == 'crash-started')
    assert (lost['status'], lost['outcome']) == ('terminal', 'completed'), lost  # same admission, a new claim
    assert lost['admission_id'] == next(r for r in before['blocked'] if r['request_id'] == 'crash-started')['admission_id']
    assert lost['seq'] < next(r for r in blocked if r['request_id'] == 'unknown-follower')['seq']
    follower = next(r for r in blocked if r['request_id'] == 'unknown-follower')
    assert (follower['status'], follower['outcome']) == ('terminal', 'completed'), follower
    assert follower['payload'] == next(r for r in before['blocked'] if r['request_id'] == 'unknown-follower')['payload']
    assert authority.sessions[safe_sid].source.platform == source.platform
    # (The blocked session's agent now exists because its redelivered turn ran; that it ran once, not twice, is
    # the asked() count below.)
    already_completed = initial[0]['status'] == 'terminal'
    def settled(request_id):
        return next(r for r in initial_blocked if r['request_id'] == request_id)['status'] == 'terminal'

    def asked(text):  # model calls whose newest user message is ``text``
        return sum(next((m.get('content', '') for m in reversed(request['messages']) if m.get('role') == 'user'), '') == text
                   for request in peer.requests)

    redelivered, follower_done = settled('crash-started'), settled('unknown-follower')
    assert len(peer.requests) == (0 if already_completed else 1) + (0 if redelivered else 1) + (0 if follower_done else 1), peer.requests
    assert asked('BLOCK_FIFO_CRASH') == (0 if redelivered else 1) and asked('UNKNOWN_FOLLOWER') == (0 if follower_done else 1)
    if not already_completed:
        assert asked('RECOVER_EXACTLY_ONCE') == 1
        safe_deliveries = [d for d in adapter.deliveries if d['chat_id'] == 'recover-chat']
        assert safe_deliveries and any('LOCAL_ACK' in d['content'] for d in safe_deliveries)
        # The other session's redelivered and follower turns answer in its own chat, never the safe row's.
        assert {d['chat_id'] for d in adapter.deliveries} <= {'recover-chat', 'fixture-chat'}
    evidence = {'safe': safe, 'blocked': blocked, 'model_calls': len(peer.requests),
                'deliveries': adapter.deliveries, 'recovery': recovered}
    Path(os.environ['HERMES_HOME'], 'recovery-receipt.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))
