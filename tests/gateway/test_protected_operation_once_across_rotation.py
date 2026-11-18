"""MR-PROOF-03 (t_e44cb2d0/run18372; reviewer18496 refinement): a business turn's protected operation runs ONCE
through the normal route, with the protected tool evidence ROTATED AWAY by the production compactor itself.

Real daemons (plain ``gateway.run``, no fixture patch) share one HERMES_HOME and a loopback model:

1. ``COUNT_ONCE``: the model asks for one ``terminal`` call that appends to a counter file. The real agent persists
   the call, and the real terminal tool runs it, so the counter is 1. The model then asks for ``terminal`` filler
   calls with distinct sizeable outputs. Its usage reports the honest size of each request, so the agent's own
   post-tool compaction (``trigger="post_tool"``, ``compression.in_place: false``) fires and ROTATES the business
   transcript. Default settings otherwise.
   Measured scope:
   - The FIRST production compaction keeps the protected call and result. The default ``protect_first_n=3``
     protects the head of a session's first compaction, and the test records that, as
     ``requests_after_a_compaction_still_carrying_protected``.
   - The fillers continue until a later production compaction (head protection decays) has dropped them. The first
     request whose provider-visible history lacks the call parks.
   - The compactor's summary may still NAME the counter file as text (for example its deterministic summary's
     "Relevant Files"). That is recorded as ``protected_command_text_after_compaction``, not asserted away.
2. ``PARK_NO_TOOL`` (positive control, another session): its first request parks before any tool.
3. The owner is killed (SIGKILL), and both turns are lost with it.
4. A fresh ``gateway.run`` boots (real ``initialize_session_authority``). The no-tool message is queued again and
   ACTUALLY runs again. The business turn stays ``unknown`` and is never re-run.
5. The client's ordinary retry of the business input (``prompt.submit``, same input id) and a third boot do not
   re-run it either.

At the kill and at every later boundary the test asserts:
- the protected call (``count-once``) and its result are ABSENT, as structured history, from the provider-visible
  request after the compaction that dropped them and from the current (rotated) physical transcript;
- the counter stays 1;
- the business turn's model requests do not grow;
- the no-tool control genuinely advances once.
Nothing strips history by hand: what is absent was removed by the production compactor.

Eric's rule: "A replay is completely fine. I don't give a fuck whether the same message is sent twice. I just want
all messages to be delivered at least once." A message re-runs; a business effect is never replayed.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.waiting import async_wait_until

MAX_FILLERS = 24  # filler calls continue until a production compaction has removed the protected call (bounded)
FILLER_LINES = 1_700  # distinct ~12K-char outputs (repeated identical output is collapsed in context)
PROTECTED = 'count-once'


def _text(message):
    content = message.get('content')
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    return content or ''


def _carries_protected(messages):
    """The protected call or its result, as structured history (the evidence the reviewer asks to be absent)."""
    for m in messages:
        if m.get('role') == 'tool' and m.get('tool_call_id') == PROTECTED:
            return True
        if any(c.get('id') == PROTECTED for c in m.get('tool_calls') or []):
            return True
    return False


class CounterModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        users = [m for m in messages if m['role'] == 'user']
        last = _text(users[-1]) if users else ''
        message, finish = {'role': 'assistant', 'content': 'ACK_' + last.rsplit('\n\n', 1)[-1][:40]}, 'stop'
        if messages and not body.get('tools'):  # the compactor's summary request (no tool schema)
            if not self.server.summaries:
                self.server.business_at_summary = len(self.server.business)
            self.server.summaries.append(body)
            message = {'role': 'assistant', 'content': 'EARLIER_TURN_SUMMARY: the user asked for a counted step.'}
        elif any('COUNT_ONCE' in _text(m) for m in users) and not last.endswith('PARK_NO_TOOL'):
            self.server.business.append(body)
            step = len(self.server.business) - 1
            carries = _carries_protected(messages)
            if self.server.summaries and carries:  # a compaction ran and its history still carries the call
                self.server.retained_after_compaction += 1
            if step == 0:
                call = ('count-once', f'printf x >> {self.server.counter}')
            elif (carries or not self.server.summaries) and step <= MAX_FILLERS:
                first = step * 1_000_000
                call = (f'filler-{step}', f'seq {first} {first + FILLER_LINES - 1}')
            else:
                call = None  # the first request whose provider-visible history lacks the call parks
                self.server.parked_at = step
            if call:
                message = {'role': 'assistant', 'content': '', 'tool_calls': [{
                    'id': call[0], 'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps(
                        {'command': call[1], 'timeout': 10})}}]}
                finish = 'tool_calls'
            else:
                self.server.parked_business.set()
                self.server.release.wait(120)
        elif last.endswith('PARK_NO_TOOL'):
            self.server.no_tool.append(body)
            if len(self.server.no_tool) == 1:
                self.server.no_tool_parked.set()
                self.server.release.wait(120)
        prompt_tokens = len(json.dumps(messages)) // 4  # honest size: the agent's real-usage pressure
        usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': 5, 'total_tokens': prompt_tokens + 5}
        payload = json.dumps({'id': 'once', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                              'usage': usage}).encode()
        kind = 'application/json'
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            payload = ('data: ' + json.dumps({'id': 'once', 'choices': [{'index': 0, 'delta': delta,
                       'finish_reason': finish}], 'usage': usage}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.mark.platforms("linux")
def test_a_business_turns_protected_operation_runs_once_across_rotation_restarts_and_retry(tmp_path):
    from gateway.session_message_redelivery import REDELIVERY_NOTICE
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    counter = tmp_path / 'protected-operation.count'
    peer = ThreadingHTTPServer(('127.0.0.1', 0), CounterModel)
    peer.counter, peer.business, peer.no_tool, peer.summaries = counter, [], [], []
    peer.business_at_summary, peer.parked_at, peer.retained_after_compaction = None, None, 0
    peer.parked_business, peer.no_tool_parked, peer.release = threading.Event(), threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'once', 'base_url': url, 'context_length': 64_000},
           # Production compaction, rotating (not in place), triggered by the agent's own post-tool pressure.
           'compression': {'enabled': True, 'in_place': False, 'threshold_tokens': 16_000, 'protect_last_n': 2},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal']}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    sessions, observed = {}, {}

    def db():
        return sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)

    def admissions():
        with db() as c:
            return dict(c.execute('SELECT request_id,status FROM session_admissions'))

    def count():
        return len(counter.read_text()) if counter.exists() else 0

    def durable(sql, *args):
        with db() as c:
            return c.execute(sql, args).fetchone()[0]

    def transcript_tip():
        """The protected call's physical transcript, the current tip of its rotation chain and the tip's rows."""
        with db() as c:
            origin = c.execute("SELECT session_id FROM messages WHERE tool_call_id=?", (PROTECTED,)).fetchone()[0]
            tip = origin
            while (child := c.execute('SELECT id FROM sessions WHERE parent_session_id=? ORDER BY started_at DESC',
                                      (tip,)).fetchone()) is not None:
                tip = child[0]
            rows = [{'role': r, 'tool_call_id': t, 'tool_calls': json.loads(tc) if tc else None}
                    for r, t, tc in c.execute('SELECT role,tool_call_id,tool_calls FROM messages WHERE session_id=?',
                                              (tip,))]
        return origin, tip, rows

    def boundary(name):
        """Assertions at one retry boundary."""
        assert peer.business_at_summary is not None, ('the production compactor never ran', len(peer.business))
        assert peer.parked_at is not None, ('no compaction removed the protected call', len(peer.summaries))
        post = [b['messages'] for b in peer.business[peer.parked_at:]]
        origin, tip, rows = transcript_tip()
        observed[name] = {
            'counter': count(), 'business_requests': len(peer.business), 'no_tool_requests': len(peer.no_tool),
            'summaries': len(peer.summaries), 'origin': origin, 'tip': tip,
            'requests_after_a_compaction_still_carrying_protected': peer.retained_after_compaction,
            'post_compaction_requests_carrying_protected': sum(map(_carries_protected, post)),
            'tip_carries_protected': _carries_protected(rows),
            'protected_command_text_after_compaction': any(
                str(counter) in json.dumps(m) for m in post),
        }
        o = observed[name]
        assert o['counter'] == 1, observed
        assert o['summaries'] >= 1 and tip != origin, observed  # the production compactor summarised and rotated
        assert o['post_compaction_requests_carrying_protected'] == 0 and not o['tip_carries_protected'], observed

    async def first(desc, proc):
        async with websocket(home, desc) as ws:
            for name in ('biz', 'msg'):
                created = await rpc(ws, 'session.create', request_id=name, source='cli', cwd=str(tmp_path),
                                    model='once', toolsets=['terminal'], ignore_rules=True)
                assert 'result' in created, created
                sessions[name] = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sessions['biz'], input_id='biz', text='COUNT_ONCE')
            assert await asyncio.to_thread(peer.parked_business.wait, 90), (
                'the post-compaction request never arrived', len(peer.business), len(peer.summaries))
            assert count() == 1  # the real terminal tool ran the protected operation once
            await rpc(ws, 'prompt.submit', session_id=sessions['msg'], input_id='msg', text='PARK_NO_TOOL')
            assert await asyncio.to_thread(peer.no_tool_parked.wait, 60), 'the no-tool turn never started'
            assert admissions()['biz'] == 'started' and admissions()['msg'] == 'started'
            proc.kill()  # the owner dies with both turns in flight
            await asyncio.to_thread(proc.wait, 10)

    async def recovered(desc, *, retry):
        async with websocket(home, desc) as ws:
            await async_wait_until(lambda: admissions().get('msg') == 'terminal', timeout=60, interval=.05,
                                   what='the no-tool message ran again')
            assert durable('SELECT count(*) FROM messages WHERE content=?', REDELIVERY_NOTICE) == 1
            assert durable("SELECT count(*) FROM messages WHERE role='assistant' AND content='ACK_PARK_NO_TOOL'") == 1
            biz = await rpc(ws, 'session.resume', session_id=sessions['biz'])
            assert [p['status'] for p in biz['result']['pending']] == ['unknown'], biz
            if retry:  # the client's ordinary retry of the same input is the same admission: never run again
                observed['retry_reply'] = await rpc(ws, 'prompt.submit', session_id=sessions['biz'], input_id='biz',
                                                    text='COUNT_ONCE')

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(first(desc, proc))
        boundary('after_kill')
        assert observed['after_kill']['no_tool_requests'] == 1, observed
        business_requests = observed['after_kill']['business_requests']
        assert business_requests == peer.parked_at + 1, observed  # the parked request is the last one
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(recovered(desc, retry=True))
        boundary('after_boot_and_retry')
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(recovered(desc, retry=False))
        boundary('after_second_boot')
        observed['admissions'] = admissions()
        print('PROTECTED_ONCE', json.dumps(observed, default=str))
        # The business turn's model requests never grew after the kill; the no-tool message reached the model
        # exactly once more (its redelivery) and not again after that.
        for name in ('after_boot_and_retry', 'after_second_boot'):
            assert (observed[name]['business_requests'], observed[name]['no_tool_requests']) == (
                business_requests, 2), observed
        assert observed['admissions']['biz'] == 'unknown' and observed['admissions']['msg'] == 'terminal', observed
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
