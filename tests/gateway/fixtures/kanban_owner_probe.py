"""Actual dispatcher argv, ordinary daemon, loopback tool round, disposable board."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from local_recovery_probe import daemon, websocket, rpc


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        if messages and not self.server.blocked.is_set():
            self.server.blocked.set()
            assert self.server.release.wait(65)
        if messages and self.server.mode in {'rate_limit', 'billing'}:
            payload = json.dumps({'error': {'message': 'Rate limit exceeded' if self.server.mode == 'rate_limit' else 'Insufficient quota',
                'type': 'rate_limit_error' if self.server.mode == 'rate_limit' else 'insufficient_quota',
                'code': 'rate_limit_exceeded' if self.server.mode == 'rate_limit' else 'insufficient_quota'}}).encode()
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        has_result = any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'OWNED_COMPLETE'}
        finish = 'stop'
        if messages and body.get('tools') and self.server.mode in REVIEW_MODES:
            message, finish = review_turn(messages, self.server)
        elif messages and not body.get('tools'):
            message['content'] = json.dumps({'verdict': 'done', 'reason': 'Owned acceptance satisfied'})
        elif messages and (not has_result or messages[-1]['role'] == 'user'):
            continuation = has_result
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'owned-complete',
                'type': 'function', 'function': {'name': 'kanban_complete' if continuation else 'kanban_comment', 'arguments': json.dumps({'summary': 'Owned probe complete'} if continuation else {'body': 'Owned first turn'})}}]}
            finish = 'tool_calls'
        payload = json.dumps({'id': 'owned', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'owned', 'choices': [{'index': 0, 'delta': message, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


REVIEW_SENTINEL = 'REVIEW_SENTINEL_c41d'
REVIEW_MODES = {'review_continue', 'review_continue_compressed', 'review_continue_first_turn_compaction'}
COMPACTION_MARK = 'COMPACTION_SUMMARY_9e1'


def current_card(messages):
    """The card a worker request serves: its newest user turn is that run's ``work kanban task``."""
    last = max(i for i, m in enumerate(messages) if m['role'] == 'user')
    return re.search(r'work kanban task (t_[0-9a-f]+)', json.dumps(messages[last]['content'])).group(1), messages[last + 1:]


def review_turn(messages, server):
    """The first reviewed card's reviewer leaves the sentinel and its verdict; the continuing
    reviewer only gives its own verdict. Any request after a tool result ends the turn."""
    card, since = current_card(messages)
    server.first_card = server.first_card or card
    if any(m['role'] == 'tool' for m in since):
        return {'role': 'assistant', 'content': 'OWNED_COMPLETE'}, 'stop'
    summary = f'{card} reviewed'
    return {'role': 'assistant', 'content': REVIEW_SENTINEL if card == server.first_card else None, 'tool_calls': [{
        'index': 0, 'id': 'verdict-' + card, 'type': 'function',
        'function': {'name': 'kanban_complete', 'arguments': json.dumps({'summary': summary})}}]}, 'tool_calls'


def compress_review(home, card, receipt):
    """Compact card's finished review session the way a context compaction publishes it: the
    summary plus the protected tail become a new physical transcript, the original stays behind
    in the closed parent. Only the summary text is the fixture's; publication is the store's own."""
    from agent.context_compressor import SUMMARY_PREFIX
    from hermes_cli.kanban_db_connect import connect
    from hermes_state import SessionDB
    with closing(connect(board='owned')) as conn:
        sid = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='owner_admitted' "
                                      "ORDER BY id DESC LIMIT 1", (card,)).fetchone()[0])['session_id']
    db = SessionDB(db_path=home / 'state.db')
    try:
        original = db.get_messages_as_conversation(sid)
        # A's own prompt turn is the pre-compaction-only row: the summary stands in for it.
        assert original[0]['role'] == 'user' and f'work kanban task {card}' in json.dumps(original[0]), original
        assert f'work kanban task {card}' not in json.dumps(original[1:]), original
        tail = original[1:]  # the sentinel turn with its verdict call, the call's result, the final reply
        handoff = [{'role': 'user', 'content': f'{SUMMARY_PREFIX}\n{COMPACTION_MARK}: the spec was read.'}, *tail]
        assert db.try_acquire_compression_lock(sid, 'probe-compaction')
        try:
            db.publish_compression_child(parent_session_id=sid, child_session_id=sid + '-compacted', source='cli',
                                         messages=handoff, compression_lock_holder='probe-compaction')
        finally:
            db.release_compression_lock(sid, 'probe-compaction')
        receipt.update(a_session=sid, a_compacted=sid + '-compacted', a_original_rows=len(original),
                       a_tip_rows=len(db.get_messages(sid + '-compacted')))
    finally:
        db.close()


def review_continue(peer, home, workspace, receipt, clients, wait_for):
    """Card A's dispatched reviewer leaves a sentinel; card B hands off asking to continue A's
    review. B's reviewer's FIRST model request must carry A's conversation, and B's verdict must
    close B's run alone (A's old verdict call is history, never re-run)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    hook_calls = lambda: len((home / 'hook-effect').read_text().splitlines()) if (home / 'hook-effect').exists() else 0

    def reviewed(title, metadata=None):
        with closing(connect(board='owned')) as conn:
            tid = kb.create_task(conn, title=title, assignee='default', workspace_kind='dir',
                                 workspace_path=str(workspace), skills=['owned-skill'])
            kb.recompute_ready(conn)
            task = kb.claim_task(conn, tid)
            ok, reason = kb.request_review(conn, tid, summary='implemented', metadata=metadata, reviewer='default',
                                           expected_run_id=task.current_run_id, with_reason=True)
            assert ok, reason
            launched = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
            assert [s[0] for s in launched.spawned] == [tid], launched
            run_id = kb.get_task(conn, tid).current_run_id
            wait_for(lambda: kb.get_task(conn, tid).status == 'done')
            wait_for(lambda: all(c.poll() is not None for c in clients))
            dispatch.reap_worker_zombies()
            return tid, run_id

    a, _a_run = reviewed('SPEC_CARD')
    if peer.mode == 'review_continue_compressed':
        compress_review(home, a, receipt)
    elif peer.mode == 'review_continue_first_turn_compaction':
        # B's worker loads this; any first request is over the threshold, so preflight compaction runs on
        # B's first turn with the inherited rows as its only compressible middle (legacy tail = the prompt).
        cfg = json.loads((home / 'config.yaml').read_text())
        cfg['compression'] = {'threshold_tokens': 1000, 'protect_first_n': 0, 'protect_last_n': 1, 'tail_mode': 'legacy'}
        (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    after_a = (hook_calls(), len(peer.requests))
    b, b_run = reviewed('IMPL_CARD', {'reviewer_context': 'continue', 'review_continues_task': a})
    b_requests = [r for r in peer.requests[after_a[1]:] if r.get('tools')]
    first = json.dumps(b_requests[0]['messages']) if b_requests else ''
    with closing(connect(board='owned')) as conn:
        runs = {tid: [dict(r) for r in conn.execute('SELECT id, outcome FROM task_runs WHERE task_id=? ORDER BY id', (tid,))]
                for tid in (a, b)}
        completed = {tid: conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='completed'",
                                       (tid,)).fetchone()[0] for tid in (a, b)}
        statuses = {tid: kb.get_task(conn, tid).status for tid in (a, b)}
    if b_requests:
        wire = b_requests[0]['messages']
        calls = [c['id'] for m in wire if m['role'] == 'assistant' for c in m.get('tool_calls') or ()]
        results = [m.get('tool_call_id') for m in wire if m['role'] == 'tool']
        receipt.update(b_first_calls=calls, b_first_results=results,
                       b_first_note=json.dumps([m for m in wire if m['role'] == 'user'][-1]['content']),
                       b_first_has_compaction=COMPACTION_MARK in first, b_first_has_pre_compaction=f'work kanban task {a}' in first)
    with closing(connect(board='owned')) as conn:
        b_sid = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
                                        (b, b_run)).fetchone()[0])['session_id']
    import sqlite3
    with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
        from hermes_state_local import POLICY_PREFIX
        receipt['b_seed'] = json.loads(db.execute('SELECT value FROM state_meta WHERE key=?',
                                                  (POLICY_PREFIX + b_sid,)).fetchone()[0]).get('review_seed')
        receipt['b_rows'] = [list(r) for r in db.execute(
            'SELECT session_id, role, active, compacted, substr(coalesce(content, \'\'), 1, 80) FROM messages '
            'WHERE session_id=? OR session_id IN (SELECT id FROM sessions WHERE parent_session_id=?) ORDER BY id', (b_sid, b_sid))]
    after = peer.requests[after_a[1]:]
    receipt['b_request_shapes'] = [dict(tools=bool(r.get('tools')), messages=len(r['messages']),
        compaction='[CONTEXT COMPACTION' in json.dumps(r['messages']), sentinel=REVIEW_SENTINEL in json.dumps(r['messages']),
        note='fresh verdict' in json.dumps(r['messages'])) for r in after]
    receipt['b_active_compaction_rows'] = sum(1 for row in receipt['b_rows'] if row[2] and row[4].startswith('[CONTEXT COMPACTION'))
    receipt.update(b_requests=len(after), b_summary_requests=sum(1 for r in after if not r.get('tools')),
                   b_last_tool_request_note='fresh verdict' in json.dumps(b_requests[-1]['messages']) if b_requests else False,
                   b_last_tool_request_names_current=f'task {b} run {b_run}' in json.dumps(b_requests[-1]['messages']) if b_requests else False)
    receipt.update(card_a=a, card_b=b, statuses=statuses, completed_events=completed, runs=runs,
        b_first_has_sentinel=REVIEW_SENTINEL in first, b_first_has_note='fresh verdict' in first,
        b_first_names_current=f'task {b} run {b_run}' in first, b_first_has_a_verdict_call=f'verdict-{a}' in first,
        b_tool_calls=hook_calls() - after_a[0], a_tool_calls=after_a[0])


def main():
    root = Path(__file__).resolve().parents[3]
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True, exist_ok=True)
    Path(os.environ['HOME']).mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_KANBAN_HOME'] = str(home)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.mode = os.environ.get('KANBAN_PROBE_MODE', 'complete')
    peer.blocked, peer.release = threading.Event(), threading.Event()
    peer.first_card = None
    if peer.mode in REVIEW_MODES:
        peer.release.set()  # two sequential dispatched reviews; nothing waits on the first request
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    workspace = home / 'workspace'
    workspace.mkdir(exist_ok=True)
    skill = home / 'skills' / 'owned-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: owned-skill\ndescription: Owned fixture\n---\nKANBAN_SKILL_SENTINEL\n', encoding='utf-8')
    hook = home / 'hook.py'
    hook.write_text('with open(' + repr(str(home / 'hook-effect')) + ', "a") as f:\n    f.write("accepted\\n")\n', encoding='utf-8')
    cfg = {'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file']},
        'terminal': {'backend': 'local'}, 'agent': {'api_max_retries': 1}, 'kanban': {'dispatch_in_gateway': False},
        'hooks': {'post_tool_call': [{'command': f'{sys.executable} {hook}'}]}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    # Pin the installed launcher to this checkout for base/fixed argv comparisons.
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = None if peer.mode in REVIEW_MODES else kb.create_task(conn, title='KANBAN_TASK_SENTINEL', body='Acceptance: finish owned card', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'],
            goal_mode=peer.mode in {'complete', 'redirect'}, goal_max_turns=2,
            max_runtime_seconds=3600 if peer.mode == 'timeout' else None)
    env = dict(os.environ)
    receipt, clients = {}, []
    original_popen = subprocess.Popen
    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
            if peer.mode == 'timeout':
                assert peer.blocked.wait(55), 'managed worker did not reach loopback'
                with closing(connect(board='owned')) as snapshot:
                    bound = snapshot.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound' "
                                             "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
                    receipt['managed_pid'] = json.loads(bound[0])['pid'] if bound else None
        return child
    def wait_for(predicate):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError(('deadline', receipt))
    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            subprocess.Popen = spawn
            try:
                if peer.mode in REVIEW_MODES:
                    review_continue(peer, home, workspace, receipt, clients, wait_for)
                    return
                with closing(connect(board='owned')) as conn:
                    launched = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
                    assert [s[0] for s in launched.spawned] == [tid], launched
                    task = kb.get_task(conn, tid)
                    params = dict(task_id=tid, board='owned', run_id=task.current_run_id, claim_lock=task.claim_lock)
                    assert peer.blocked.wait(55), 'model never reached'
                    active = kb.get_task(conn, tid)
                    receipt.update(client_pid=clients[0].pid, stored_pid=active.worker_pid, run_id=task.current_run_id)
                    if peer.mode == 'redirect':
                        # The live attempt's managed worker is parked in its first provider call: correct it
                        # through the canonical route; the SAME run, claim and worker must keep the task.
                        marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
                                              "AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                              (tid, task.current_run_id)).fetchone()
                        sid = json.loads(marker[0])['session_id']
                        async def correct():
                            async with websocket(home, desc) as ws:
                                generation = (await rpc(ws, 'session.resume', session_id=sid))['result']['execution_generation']
                                return await rpc(ws, 'session.redirect', session_id=sid, execution_generation=generation,
                                                 text='KANBAN_CORRECTION')
                        receipt['redirect_status'] = asyncio.run(correct()).get('result', {}).get('status')
                        wait_for(lambda: 'KANBAN_CORRECTION' in json.dumps(peer.requests))
                        after = kb.get_task(conn, tid)
                        receipt['claim_kept'] = ((after.current_run_id, after.claim_lock, after.worker_pid)
                                                 == (task.current_run_id, task.claim_lock, active.worker_pid))
                        receipt['correction_before_release'] = not peer.release.is_set()
                        # Kept on the task before it was sent (the owner's durable receipt), whole and
                        # captioned as accepted, and in front of any next attempt of this task.
                        receipt['correction_comment'] = any(
                            'KANBAN_CORRECTION' in c.body and f'accepted for run {task.current_run_id}' in c.body
                            for c in kb.list_comments(conn, tid))
                        receipt['correction_in_next_context'] = 'KANBAN_CORRECTION' in kb.build_worker_context(conn, tid)
                    if peer.mode == 'timeout':
                        # The spawned worker stays the run's handle; the owner's interpreter dies with it.
                        receipt['pid_preserved'] = active.worker_pid == clients[0].pid != receipt['managed_pid']
                        # Deterministically elapse this owned attempt; keep real timeout/kill logic.
                        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time()) - 3601, task.current_run_id))
                        conn.commit()
                        swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                        receipt['timed_out'] = swept.timed_out
                        wait_for(lambda: not kb._pid_alive(receipt['managed_pid']))
                        receipt['managed_dead'] = True
                        assert receipt['pid_preserved'] and receipt['managed_dead'], receipt
                    elif peer.mode == 'crash':
                        import signal
                        # Windows has no SIGKILL; os.kill there is TerminateProcess for any signal.
                        os.kill(active.worker_pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                    peer.release.set()
                    if peer.mode != 'timeout':
                        wait_for(lambda: not kb._pid_alive(active.worker_pid))
                    wait_for(lambda: not kb._pid_alive(clients[0].pid))
                    dispatch.reap_worker_zombies()
                    receipt['client_exit'] = dispatch._classify_worker_exit(clients[0].pid)
                    dispatch._recent_worker_exits.clear()
                    swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                    task = kb.get_task(conn, tid)
                    run = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (params['run_id'],)).fetchone())
                    from gateway.session_kanban import worker_exit_code
                    assert worker_exit_code(kb.kanban_db_path(board='owned'), params | {'claim_lock': 'foreign'}) == 1
                    receipt['booked_exit'] = worker_exit_code(kb.kanban_db_path(board='owned'), params)
                    import sqlite3
                    with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                        receipt['turn_outcomes'] = db.execute('SELECT status, outcome FROM session_admissions').fetchall()
                    receipt.update(task_status=task.status, failures=task.consecutive_failures, run_outcome=run['outcome'],
                        run_metadata=kb._json_dict(run['metadata']), rate_limited=swept.rate_limited,
                        cooldown=dispatch.check_respawn_guard(conn, tid))
                    if peer.mode in {'rate_limit', 'billing'}:
                        assert task.status == 'ready' and task.consecutive_failures == 0 and run['outcome'] == 'rate_limited', receipt
                        assert receipt['client_exit'] == ('rate_limited', kb.KANBAN_RATE_LIMIT_EXIT_CODE), receipt
                        assert receipt['cooldown'] == 'rate_limit_cooldown', receipt
                    elif peer.mode == 'crash':
                        # Killing the spawned worker ends the attempt: one crash, re-queued, as before the owner.
                        assert task.status == 'ready' and task.consecutive_failures == 1 and run['outcome'] == 'crashed', receipt
                    elif peer.mode == 'timeout':
                        assert task.status == 'ready' and task.consecutive_failures == 1 and run['outcome'] == 'timed_out', receipt
            finally:
                subprocess.Popen = original_popen
            if peer.mode not in {'complete', 'redirect'}:
                return
            assert receipt['client_exit'] == ('clean_exit', 0), receipt
            async def retry():
                async with websocket(home, desc) as ws:
                    first = await rpc(ws, 'kanban.run', **params)
                    second = await rpc(ws, 'kanban.run', **params)
                    assert 'result' in first and 'result' in second, (first, second)
                    return first['result']['session_id'], first['result']['session_id'] == second['result']['session_id']
            sid, same = asyncio.run(retry())
            import sqlite3
            with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                receipt['source'] = db.execute('SELECT source FROM sessions WHERE id=?', (sid,)).fetchone()[0]
                receipt['admissions'] = db.execute('SELECT count(*) FROM session_admissions').fetchone()[0]
            with closing(connect(board='owned')) as conn:
                receipt['task_status'] = kb.get_task(conn, tid).status
            wire = json.dumps(peer.requests)
            turns = [r for r in peer.requests if r.get('tools')]
            prefixes = [json.dumps([m for m in r['messages'] if m['role'] in {'system', 'developer'}]) for r in turns]
            receipt.update(goal_continuation=len(turns) >= 4, stable_prefix=len(set(prefixes)) == 1,
                retry_same_session=same, tools='kanban_complete' in wire,
                task_context='KANBAN_TASK_SENTINEL' in wire, skill_context='KANBAN_SKILL_SENTINEL' in wire,
                hook_effect=(home / 'hook-effect').exists())
    finally:
        peer.release.set()
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown()
        peer.server_close()
        (home / 'model-requests.json').write_text(json.dumps(peer.requests, indent=2), encoding='utf-8')
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
