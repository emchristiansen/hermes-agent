"""TR-3 (RB4765 r3c11h, aa04): TR2-01 (reviewer t_70a14ad0/run9210) through a REAL process.

``python -m hermes_cli.main peer run [--exact] spark`` runs as a subprocess of the tree under test (``hermes_cli`` as
imported here), with its own HERMES_HOME (a fixture ``config.yaml`` naming a loopback peer) and a fixture key in the
environment (no credential read, no live peer). The fixture peer records the RAW request bytes; the ``input`` JSON field
is compared with the bytes given on stdin.

- ``latin1``: PYTHONIOENCODING=latin1 (the reviewer's condition); ``utf8``: a normal C.UTF-8 locale.
- ``--exact`` + valid UTF-8 stdin: the request carries exactly those code points (TR-2: transformed under latin1).
- ``--exact`` + invalid UTF-8 stdin: exit 2, nothing sent (TR-2: accepted under latin1).
- default (no ``--exact``): unchanged from the baseline, i.e. the process's own stdin decoding, stripped.
- ``--exact -- <argv body>``: argv is not stdin; exact under both encodings.
Expected: RED on TR-2 (dev-tr-2, tr2-7a70) in the latin1 exact cases, GREEN on TR-3 (dev-tr-3, tr3-7a70)."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import hermes_cli

TREE = Path(hermes_cli.__file__).resolve().parent.parent
VALID = '  été \U0001F916 　paragraph\n\n'.encode('utf-8')
INVALID = b'  caf\xe9 \xff\n'


class _Peer(BaseHTTPRequestHandler):
    posts: list = []

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/v1/capabilities':
            return self._json({'features': {'runs_idempotency': {'supported': True, 'durable': True,
                                                                 'retention_seconds': 86400}}})
        if self.path.startswith('/api/sessions'):
            return self._json({'object': 'list', 'data': [{'id': 'bc_1', 'title': 'Bot Chat'}]})
        return self._json({'error': {'message': 'not found'}}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        type(self).posts.append((self.path, raw))
        if self.path == '/v1/runs':
            return self._json({'run_id': 'run_1', 'status': 'started', 'replayed': False}, 202)
        return self._json({'error': {'message': 'not found'}}, 404)

    def log_message(self, *args):
        pass


@pytest.fixture()
def peer(tmp_path):
    _Peer.posts = []
    server = HTTPServer(('127.0.0.1', 0), _Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / 'hermes'
    home.mkdir()
    (home / 'config.yaml').write_text(f'bot_peers:\n  spark:\n    url: http://127.0.0.1:{server.server_port}\n')
    try:
        yield home
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _run(home, tmp_path, encoding, args, stdin):
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(tmp_path), 'HERMES_HOME': str(home),
           'TMPDIR': str(tmp_path), 'PYTHONPATH': str(TREE), 'PYTHONDONTWRITEBYTECODE': '1',
           'HERMES_PEER_SPARK_KEY': 'fixture-key-0000000000', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    if encoding == 'latin1':
        env['PYTHONIOENCODING'] = 'latin1'
    proc = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'peer', 'run', '--idempotency-key', 'k1', *args],
                          input=stdin, env=env, cwd=str(tmp_path), capture_output=True, timeout=120, check=False)
    sent = [json.loads(raw)['input'] for path, raw in _Peer.posts if path == '/v1/runs']
    return proc.returncode, sent, proc.stderr.decode('utf-8', 'replace')[-400:]


@pytest.mark.parametrize('encoding', ['latin1', 'utf8'])
def test_tr3_exact_stdin_valid_utf8_is_byte_exact(peer, tmp_path, encoding):
    rc, sent, err = _run(peer, tmp_path, encoding, ['--exact', 'spark'], VALID)
    print('TR3 exact valid', encoding, rc, [s.encode('utf-8', 'surrogatepass') for s in sent], err)
    assert rc == 0 and [s.encode('utf-8') for s in sent] == [VALID], (rc, sent, err)


@pytest.mark.parametrize('encoding', ['latin1', 'utf8'])
def test_tr3_exact_stdin_invalid_utf8_is_refused_and_nothing_sent(peer, tmp_path, encoding):
    rc, sent, err = _run(peer, tmp_path, encoding, ['--exact', 'spark'], INVALID)
    print('TR3 exact invalid', encoding, rc, sent, err)
    assert rc == 2 and sent == [], (rc, sent, err)


@pytest.mark.parametrize('encoding', ['latin1', 'utf8'])
def test_tr3_exact_argv_body_is_byte_exact(peer, tmp_path, encoding):
    body = VALID.decode('utf-8')
    rc, sent, err = _run(peer, tmp_path, encoding, ['--exact', 'spark', '--', body], b'')
    assert rc == 0 and sent == [body], (rc, sent, err)


@pytest.mark.parametrize('encoding', ['latin1', 'utf8'])
def test_tr3_default_stdin_is_unchanged_from_the_baseline(peer, tmp_path, encoding):
    """No --exact: the process's own stdin decoding, stripped (the baseline's behaviour, on TR-2 and TR-3 alike)."""
    rc, sent, err = _run(peer, tmp_path, encoding, ['spark'], VALID)
    expected = VALID.decode('latin1' if encoding == 'latin1' else 'utf-8').strip()
    assert rc == 0 and sent == [expected], (rc, sent, err)
