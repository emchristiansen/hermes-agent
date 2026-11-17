"""Caller -> live finite Kanban session discovery and redirect, through the real dispatcher, owner daemon,
managed worker and the ``hermes kanban redirect`` caller. Owned private board and loopback model; the model
fence (every provider call parks until released) keeps each original live child, handle and result in place."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
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
            self.server.release.wait(120)  # the model fence: every provider call parks until released
        has_result = any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'OWNED_COMPLETE'}
        finish = 'stop'
        if messages and body.get('tools') and not has_result:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'owned-complete', 'type': 'function',
                'function': {'name': 'kanban_complete', 'arguments': json.dumps({'summary': 'Owned discovery probe complete'})}}]}
            finish = 'tool_calls'
        payload = json.dumps({'id': 'owned', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'owned', 'choices': [{'index': 0, 'delta': message, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    root = Path(__file__).resolve().parents[3]
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True, exist_ok=True)
    Path(os.environ['HOME']).mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_KANBAN_HOME'] = str(home)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.release = threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    workspace = home / 'workspace'
    workspace.mkdir(exist_ok=True)
    cfg = {'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file']},
        'terminal': {'backend': 'local'}, 'agent': {'api_max_retries': 1}, 'kanban': {'dispatch_in_gateway': False}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    kb.create_board('owned')
    tasks = {}
    with closing(connect(board='owned')) as conn:
        for name in ('A', 'B'):
            tasks[name] = kb.create_task(conn, title=f'TASK_{name}_SENTINEL', body=f'Finish card {name}', assignee='default',
                workspace_kind='dir', workspace_path=str(workspace), goal_mode=False)
    env = dict(os.environ)
    receipt = {}
    clients = []
    original_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
        return child

    def wait_for(predicate, what):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError(('deadline', what, receipt))

    def asked(sentinel, text=None):
        return [r for r in peer.requests if sentinel in json.dumps(r) and (text is None or text in json.dumps(r['messages'][-3:]))]

    def cli(*words):
        run = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', '--board', 'owned', *words], env=env, cwd=root,
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
        try:
            out = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            out = {'unparsed': run.stdout, 'stderr': run.stderr[-400:]}
        return run.returncode, out

    async def discover(desc, **params):
        async with websocket(home, desc) as ws:
            return await rpc(ws, 'session.kanban_run', **params)

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='owned')) as conn:
                    launched = dispatch.dispatch_once(conn, board='owned', max_spawn=2)
                    assert sorted(s[0] for s in launched.spawned) == sorted(tasks.values()), launched
                    run = {n: kb.get_task(conn, t) for n, t in tasks.items()}
                    wait_for(lambda: len(asked('TASK_A_SENTINEL')) >= 1 and len(asked('TASK_B_SENTINEL')) >= 1, 'both runs reach the model')
                    pids = {n: run[n].worker_pid for n in run}
                    # (1) two simultaneous runs: each exact run binds its own session and generation.
                    found = {n: asyncio.run(discover(desc, board='owned', task_id=tasks[n], run_id=run[n].current_run_id)) for n in run}
                    assert all('result' in f for f in found.values()), found
                    sid = {n: found[n]['result']['session_id'] for n in found}
                    gen = {n: found[n]['result']['execution_generation'] for n in found}
                    markers = {}
                    for n in run:
                        marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                              (tasks[n], run[n].current_run_id)).fetchone()
                        markers[n] = json.loads(marker[0])['session_id']
                    receipt['two_runs_exact'] = sid['A'] != sid['B'] and sid == markers
                    # (2) unqualified / unauthorized callers are refused.
                    stale = asyncio.run(discover(desc, board='owned', task_id=tasks['A'], run_id=run['A'].current_run_id + 100))
                    receipt['unqualified_run'] = stale.get('error', {}).get('message')
                    other = asyncio.run(discover(desc, board='owned', task_id=tasks['A'], run_id=run['A'].current_run_id, profile='no-such-profile'))
                    receipt['wrong_profile'] = other.get('error', {}).get('message')
                    from tests.gateway.test_normal_runtime_boot import control
                    from websockets.asyncio.client import connect as ws_connect
                    async def unauthorized():
                        grant = control(home, 'session-ticket', {'profile_id': str(home), 'instance_id': desc['instance_id'], 'purpose': 'worker-adoption'})
                        try:
                            async with ws_connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
                                                  subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']]) as ws:
                                return await rpc(ws, 'session.kanban_run', board='owned', task_id=tasks['A'], run_id=run['A'].current_run_id)
                        except Exception as exc:  # the socket itself may refuse a grant with no session capability
                            return {'error': {'message': 'connection_refused:' + type(exc).__name__}}
                    receipt['unauthorized'] = asyncio.run(unauthorized()).get('error', {}).get('message')
                    # (3) the actual caller redirects exactly run A while the model fence holds.
                    code, out = cli('redirect', tasks['A'], 'CORRECTION_FOR_A')
                    receipt['cli_a'] = [code, out.get('status'), out.get('session_id') == sid['A'], out.get('execution_generation') == gen['A']]
                    wait_for(lambda: asked('TASK_A_SENTINEL', 'CORRECTION_FOR_A'), 'A sees its correction')
                    receipt['b_untouched'] = not asked('TASK_B_SENTINEL', 'CORRECTION_FOR_A')
                    # (4) lost ACK for A, then B, then a resend of A: B progresses, A is duplicated or retried, never stranded.
                    async def lost_ack():
                        async with websocket(home, desc) as ws:
                            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 'lost', 'method': 'session.redirect',
                                'params': {'session_id': sid['A'], 'execution_generation': gen['A'], 'text': 'LOST_ACK_A'}}))
                            # the connection closes without reading the answer
                    asyncio.run(lost_ack())
                    code_b, out_b = cli('redirect', tasks['B'], 'AFTER_LOST_ACK_B')
                    receipt['cli_b'] = [code_b, out_b.get('status')]
                    wait_for(lambda: asked('TASK_B_SENTINEL', 'AFTER_LOST_ACK_B'), 'B progresses past the unacknowledged A')
                    code_a2, out_a2 = cli('redirect', tasks['A'], 'LOST_ACK_A')
                    receipt['resend_a'] = [code_a2, out_a2.get('status')]
                    wait_for(lambda: asked('TASK_A_SENTINEL', 'LOST_ACK_A'), 'A reaches its model after the resend')
                    # (5) a replacement run: A's old run is no longer bindable, and the successor is its own session.
                    kb.reclaim_task(conn, tasks['A'], reason='replace for discovery test')
                    wait_for(lambda: kb.get_task(conn, tasks['A']).status == 'ready', 'A reclaimed')
                    import signal
                    try:
                        os.kill(pids['A'], signal.SIGKILL)  # the replaced attempt's own worker ends with its claim
                    except ProcessLookupError:
                        pass
                    wait_for(lambda: not kb._pid_alive(pids['A']), 'old A worker gone')
                    dispatch.reap_worker_zombies()
                    receipt['respawn_guard'] = dispatch.check_respawn_guard(conn, tasks['A'])
                    relaunched = dispatch.dispatch_once(conn, board='owned', max_spawn=2)
                    for _ in range(40):  # the reclaim's own sweep and respawn guard settle first
                        if relaunched.spawned:
                            break
                        time.sleep(.25)
                        relaunched = dispatch.dispatch_once(conn, board='owned', max_spawn=2)
                    assert [s[0] for s in relaunched.spawned] == [tasks['A']], (relaunched, kb.get_task(conn, tasks['A']).status)
                    replacement = kb.get_task(conn, tasks['A'])
                    receipt['replacement_run_differs'] = replacement.current_run_id != run['A'].current_run_id
                    wait_for(lambda: len(asked('TASK_A_SENTINEL')) >= 1 and replacement.worker_pid is not None, 'replacement starts')
                    old = asyncio.run(discover(desc, board='owned', task_id=tasks['A'], run_id=run['A'].current_run_id))
                    receipt['old_run_after_replacement'] = old.get('error', {}).get('message')
                    for _ in range(80):  # the replacement's owner admission lands once its worker connects
                        new = asyncio.run(discover(desc, board='owned', task_id=tasks['A'], run_id=replacement.current_run_id))
                        if 'result' in new:
                            break
                        time.sleep(.25)
                    receipt['replacement_discovery'] = new.get('result', new)
                    receipt['replacement_session_differs'] = new.get('result', {}).get('session_id') not in (None, sid['A'])
                    code_old, out_old = cli('redirect', tasks['A'], 'STALE_RUN_TEXT', '--run', str(run['A'].current_run_id))
                    receipt['stale_cli'] = [code_old, out_old.get('status'), out_old.get('reason')]
                    async def stale_direct():
                        async with websocket(home, desc) as ws:
                            return await rpc(ws, 'session.redirect', session_id=sid['A'], execution_generation=gen['A'], text='STALE_DIRECT_TEXT')
                    direct = asyncio.run(stale_direct())
                    receipt['stale_direct'] = direct.get('error', {}).get('message') or direct.get('result', {}).get('status', 'kept')
                    # (6) the original B child, handle and result survive the whole exchange; release the fence.
                    receipt['b_pid_kept'] = kb.get_task(conn, tasks['B']).worker_pid == pids['B']
                    peer.release.set()
                    wait_for(lambda: kb.get_task(conn, tasks['B']).status == 'done', 'B completes through its original child')
                    receipt['b_done_once'] = conn.execute("SELECT count(*) FROM task_runs WHERE task_id=?", (tasks['B'],)).fetchone()[0] == 1
                    import sqlite3
                    with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                        receipt['b_admissions'] = db.execute('SELECT count(*), group_concat(DISTINCT outcome) FROM session_admissions WHERE target_session_id=?', (sid['B'],)).fetchone()
            finally:
                subprocess.Popen = original_popen
    finally:
        peer.release.set()
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown()
        peer.server_close()
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
