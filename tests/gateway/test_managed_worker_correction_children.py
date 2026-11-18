"""Corrections that meet live work a managed worker started, on the real composed path (owner
JSON-RPC -> worker control pipe -> the worker's ``AIAgent``).

- A redirect while a delegated child runs takes the tool-boundary path (``queued``): the child is not
  abandoned, its result is consumed once as the ``delegate_task`` result, and the correction reaches
  the parent's next model request. No second child is spawned and the child's goal is never replayed.
- A redirect while the model weighs a background process's handle takes the model path
  (``redirected``): the stale response is fenced, the process is neither killed nor re-spawned, and
  the corrected request waits on the SAME handle for its natural result, consumed once.
"""
import asyncio
import json
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_managed_worker_corrections import (
    MANAGED, _generation, _kept, _managed_exit, _managed_session, _transcript, _turns, _user_text,
)
from tests.gateway.test_session_busy_controls import admissions, owner, settled
from tests.waiting import async_wait_until


def _rows(home, sql, *args):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return db.execute(sql, args).fetchall()


def _call(name, args, ident):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': ident, 'index': 0, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}


def _reply(handler, body, message):
    finish = 'tool_calls' if message.get('tool_calls') else 'stop'
    choice = {'index': 0, 'delta': message, 'finish_reason': finish}
    frame = {'id': 'children', 'model': 'm', 'choices': [choice],
             'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
    if body.get('stream'):
        payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
    else:
        choice['message'] = choice.pop('delta')
        payload, kind = json.dumps(frame).encode(), 'application/json'
    try:
        handler.send_response(200)
        handler.send_header('Content-Type', kind)
        handler.send_header('Content-Length', str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


class DelegateModel(BaseHTTPRequestHandler):
    """``DELEGATE_BLOCK`` delegates once: one child (the single-goal form) or a ``tasks`` batch of
    ``server.labels``. Each child's ``CHILD_GOAL_<label>`` request parks until ``release`` and
    answers ``CHILD_RESULT_<label>``; any parent request that already holds a tool result answers
    ``PARENT_DONE``. Each request is recorded as parent or child."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        users = [_user_text(m) for m in messages if m['role'] == 'user']
        goal = re.search(r'CHILD_GOAL_(\w+)', json.dumps(messages))
        if users and users[0] == 'DELEGATE_BLOCK':
            self.server.parent.append(body)
            goals = [f'CHILD_GOAL_{label} answer with your result marker' for label in self.server.labels]
            if any(m['role'] == 'tool' for m in messages):
                message = {'role': 'assistant', 'content': 'PARENT_DONE'}
            elif len(goals) == 1:
                message = _call('delegate_task', {'goal': goals[0]}, 'd1')
            else:
                message = _call('delegate_task', {'tasks': [{'goal': g} for g in goals]}, 'd1')
        elif goal:
            self.server.child.append(goal.group(1))
            self.server.release.wait(60)
            message = {'role': 'assistant', 'content': f'CHILD_RESULT_{goal.group(1)}'}
        else:
            message = {'role': 'assistant', 'content': 'OTHER'}
        _reply(self, body, message)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('labels', [['42'], ['A1', 'B2']], ids=['one-child', 'two-children'])
def test_redirect_during_a_delegated_child_keeps_the_child_and_consumes_its_result_once(tmp_path, labels):
    """Two children put ``_run_children_parallel``'s parent-interrupt abandonment check on the path:
    a tool-boundary redirect must not trip it."""
    config = {**MANAGED, 'platform_toolsets': {'cli': ['delegation', 'todo']}}
    with owner(tmp_path, config, handler=DelegateModel) as (home, peer, desc):
        peer.parent, peer.child, peer.labels = [], [], labels
        url = f'http://127.0.0.1:{peer.server_port}/v1'

        def child_states():
            return _rows(home, "SELECT status FROM worker_executions WHERE kind='child'")

        async def probe():
            async with websocket(home, desc) as ws:
                created = await rpc(ws, 'session.create', request_id='parent', source='cli', cwd=str(home),
                                    model='m', provider='custom', base_url=url, api_key='loopback-only',
                                    toolsets=['delegation', 'todo'], ignore_rules=True)
                sid = created['result']['session_id']
                assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn',
                                             text='DELEGATE_BLOCK')
                await async_wait_until(lambda: len(peer.child) >= len(labels), timeout=60, interval=.05,
                                       what='every child parked')
                generation = await _generation(ws, sid)
                corrected = await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation,
                                      text='CHILD_REDIRECT')
                # The tool batch is running: the honest verdict is a tool-boundary steer, never an abort.
                assert corrected.get('result', {}).get('status') == 'queued', corrected
                # Still parked, still running, and the parent has not moved past the batch.
                assert child_states() == [('running',)] * len(labels), child_states()
                assert sorted(peer.child) == sorted(labels) and len(peer.parent) == 1, (peer.parent, peer.child)
                peer.release.set()
                await settled(home)
                return sid, (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
        sid, messages = asyncio.run(probe())
        tools = [m['content'] for m in messages if m['role'] == 'tool']
        assert len(tools) == 1, tools
        for label in labels:
            assert tools[0].count(f'CHILD_RESULT_{label}') == 1, (label, tools)
        result = json.loads(tools[0])
        assert [r.get('status') for r in result.get('results', [])] == ['completed'] * len(labels), result
        # The correction was consumed by the parent's next model request, which finished the turn.
        assert len(peer.parent) == 2 and 'CHILD_REDIRECT' in json.dumps(peer.parent[1]['messages']), peer.parent
        assert 'CHILD_REDIRECT' not in json.dumps(peer.parent[0]['messages'])
        assert messages[-1]['content'] == 'PARENT_DONE', messages
        # Each child asked for its goal once: no replay and no extra spawn.
        assert sorted(peer.child) == sorted(labels), peer.child
        children = _rows(home, "SELECT w.status, s.source, s.parent_session_id, s.end_reason FROM worker_executions w "
                               "JOIN sessions s ON s.id=w.session_id WHERE w.kind='child'")
        assert children == [('terminal', 'subagent', sid, 'agent_close')] * len(labels), children
        answers = sorted(c for (child_sid,) in _rows(home, "SELECT session_id FROM worker_executions WHERE kind='child'")
                         for r, c in _transcript(home, child_sid) if r == 'assistant')
        assert answers == sorted(f'CHILD_RESULT_{label}' for label in labels), answers
        assert _turns(home) == [('turn', 'terminal')], _kept(home)
        # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
        assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
        assert _managed_exit(home) == ['terminal'] * (len(labels) + 1)


class BackgroundModel(BaseHTTPRequestHandler):
    """``BG_START`` starts the gated background command. The next request (it carries the handle)
    parks until ``release``, then answers with a stale tool call that would touch ``marker``. Once
    ``REDIRECT_WAIT`` reached the model, it waits on the handle; with the exit in hand, ``BG_FINAL``."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        if messages:
            self.server.requests.append(body)
        seen = json.dumps(messages).replace('\\"', '"')
        procs = re.findall(r'proc_[0-9a-f]{12}', seen)
        message = {'role': 'assistant', 'content': 'BG_FINAL'}
        if messages and messages[-1]['role'] == 'user' and _user_text(messages[-1]) == 'BG_START':
            message = _call('terminal', {'command': self.server.command, 'background': True}, 'start')
        elif 'REDIRECT_WAIT' in seen and procs and '"exit_code": 7' not in seen:
            self.server.waited.append(procs[0])
            message = _call('tool_call', {'calls': [{'name': 'process_manage', 'arguments': {
                'action': 'wait', 'session_id': procs[0], 'timeout': 60}}]}, 'wait')
        elif procs and not self.server.parked:
            self.server.parked.append(procs[0])
            self.server.blocked.set()
            self.server.release.wait(60)
            message = dict(_call('terminal', {'command': f'touch {self.server.marker}', 'timeout': 5}, 'stale'),
                           content='STALE_TEXT')
        _reply(self, body, message)
        if message.get('content') == 'STALE_TEXT':
            self.server.answered.set()


@pytest.mark.platforms("linux")
def test_background_job_survives_a_model_redirect_and_its_result_is_consumed_once(tmp_path):
    config = {**MANAGED, 'approvals': {'mode': 'off'}}
    counter, gate, artifact, marker = (tmp_path / n for n in ('counter', 'gate', 'artifact', 'stale-tool-ran'))
    with owner(tmp_path, config, handler=BackgroundModel) as (home, peer, desc):
        peer.parked, peer.waited, peer.answered, peer.marker = [], [], threading.Event(), marker
        # One line per spawn; it stays running until the test opens the gate, so the redirect
        # certainly meets an unresolved process.
        peer.command = (f'echo spawned >> {counter}; i=0; while [ ! -e {gate} ] && [ $i -lt 1200 ]; '
                        f'do sleep 0.05; i=$((i+1)); done; echo ARTIFACT_$((6*7)) > {artifact}; cat {artifact}; '
                        'exit 7')
        (tmp_path / 'user' / '.hushlogin').touch()  # keep login-shell banners out of the command output

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BG_START')
                assert await asyncio.to_thread(peer.blocked.wait, 60), peer.requests
                corrected = await rpc(ws, 'session.redirect', session_id=sid,
                                      execution_generation=await _generation(ws, sid), text='REDIRECT_WAIT')
                peer.release.set()  # the stale answer arrives the moment the ACK does
                assert corrected.get('result', {}).get('status') == 'redirected', corrected
                assert counter.read_text() == 'spawned\n' and not artifact.exists()  # still unresolved
                gate.touch()
                assert await asyncio.to_thread(peer.answered.wait, 60)
                await settled(home)
                return sid, (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
        sid, messages = asyncio.run(probe())
        tools = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        assert len(tools) == 2, tools
        started, waited = tools
        proc = started['session_id']
        assert peer.parked == [proc] and peer.waited == [proc], (proc, peer.parked, peer.waited)
        assert waited['status'] == 'exited' and waited['exit_code'] == 7, waited
        assert 'ARTIFACT_42' in waited['output'], waited
        assert artifact.read_text() == 'ARTIFACT_42\n'
        assert counter.read_text() == 'spawned\n'  # never re-spawned
        assert not marker.exists(), 'a fenced response dispatched its tool call'
        corrected = [r for r in peer.requests if 'REDIRECT_WAIT' in json.dumps(r['messages'])]
        assert corrected and proc in json.dumps(corrected[0]['messages']), corrected
        rows = _transcript(home, sid)
        assert not any('STALE_TEXT' in content for _, content in rows), rows
        assert sum('"exit_code": 7' in content.replace('\\"', '"') for r, content in rows if r == 'tool') == 1, rows
        assert rows[-1] == ('assistant', 'BG_FINAL'), rows
        assert _turns(home) == [('turn', 'terminal')], _kept(home)
        # Each correction was kept before its ACK, and its turn confirmed running it: never run twice.
        assert _kept(home) and {kept[:2] for kept in _kept(home)} == {('terminal', 'steered')}, _kept(home)
        assert _managed_exit(home) == ['terminal']
