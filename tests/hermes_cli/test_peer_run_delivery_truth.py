"""``hermes peer run`` says truthfully whether its run request was delivered.

On 2026-11-12 every POST /v1/runs to the interlocutor took 30-494 s server-side and was admitted (202), while
the client's 30 s read timeout reported "Could not reach peer ... timed out" (procs 635a57da915f and
02ecfdfc0066; both messages were in the peer's FIFO).
[Additive correction to the sentence above, which is kept as written: "every POST ... took 30-494 s" overstates
it. stable-merge-2/peer-ingress-evidence-1/post-v1-runs-latency-since-1900.txt (sha256 b7508bdd...) line 9
counts theseus-interlocutor n=106 with 102 at >=30 s, so 4 of those calls took under 30 s (min 6 s, median
55 s, max 494 s).] review18499 then showed that an exception's class is
no proof either (PEER-TRUTH-01: a complete POST answered 302 to a refusing hop read as "not delivered"), and
that an unusable 202 lost the typed uncertainty (PEER-TRUTH-02). The client now decides from what its real
transport did (http.client's own connect/send audit events, across every redirect hop):
- exit 3 ``not_delivered`` ONLY when the first hop's connection failed before any request byte was written
  (or a setup request before the run failed, so the run was never attempted);
- exit 4 ``possibly_admitted`` whenever request bytes may have been written and the outcome is unconfirmed:
  a lost answer, a redirect hop's failure or error, an HTTP 5xx, or an answer that binds no run (malformed or
  missing run ID, shown verbatim). A first-hop 4xx stays the peer's rejection (exit 1).
The idempotency key is printed BEFORE the run is sent and kept in every outcome. Resending the message is
always allowed, immediately; key reuse is optional and never a receipt or dedup gate. Every peer below is a
real local HTTP server (or a port nothing listens on): no transport, redirect or exception is substituted.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_cmd

_CLIENT_TIMEOUT_S = 0.6


def _closed_port() -> int:
    with socket.socket() as probe:  # bound, never listening: a connect is refused
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        return

    def _send(self, code, body: bytes, headers=()):
        self.send_response(code)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload).encode(), [("Content-Type", "application/json")])

    def do_GET(self):
        if self.path.startswith("/v1/capabilities"):
            return self._json(200, {"features": {"runs_idempotency": {"supported": True, "durable": True}}})
        if self.path.startswith("/api/sessions"):
            return self._json(200, {"data": [{"id": "bot-chat-1", "title": peer_cmd.BOT_CHAT_TITLE}]})
        return self._json(404, {"error": {"message": "nope"}})

    def do_POST(self):
        server = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        key = self.headers.get("Idempotency-Key")
        server.received.append(key)
        mode = server.mode
        if mode in ("ok", "lost_202"):
            server.admitted.append((key, body["input"]))
        if mode in ("redirect_refused", "redirect_404"):
            target = server.redirect if mode == "redirect_refused" else "/nowhere"
            return self._send(302, b"", [("Location", target)])
        if mode == "close":  # the whole request was read; the peer closes without any answer
            self.close_connection = True
            return None
        if mode in server.bodies:
            code, raw = server.bodies[mode]
            return self._send(code, raw)
        if mode != "ok":  # admitted or not, the answer comes after the client gave up (released at teardown)
            server.release.wait(timeout=_CLIENT_TIMEOUT_S * 4)
        try:
            self._json(202, {"run_id": "run_1", "status": "started"})
        except OSError:
            pass  # the client is gone: the 202 is lost, as in the access log's "202 0"
        return None


@pytest.fixture
def peer(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Peer)
    server.mode, server.received, server.admitted, server.release = "ok", [], [], threading.Event()
    server.redirect = f"http://127.0.0.1:{_closed_port()}/hop"
    server.bodies = {
        "bad_json": (202, b"{"),
        "missing_id": (202, b'{"status": "started"}'),
        "non_object": (202, b"[1, 2]"),
        "undecodable": (202, b"\xff\xfe{"),
        "server_error": (503, b'{"error": "busy"}'),
        "conflict": (409, b'{"error": {"message": "conflict"}}'),
    }
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _point_at(monkeypatch, f"http://127.0.0.1:{server.server_port}")
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()


def _point_at(monkeypatch, url):
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"box": {"url": url}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "secret-key-123456")
    real_request = peer_cmd._request

    def _short(url, key, **kw):  # the client's 30 s budget, shortened; the urllib path is unchanged
        kw.setdefault("timeout", _CLIENT_TIMEOUT_S)
        return real_request(url, key, **kw)

    monkeypatch.setattr(peer_cmd, "_request", _short)


def _run(capsys, *, key=None):
    rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="run", target="box", message="deliver me", json=True,
                                           idempotency_key=key, exact=False))
    out = capsys.readouterr()
    return rc, (json.loads(out.out) if out.out.strip() else None), out.err


def test_an_answered_run_exits_0_with_its_run_id_and_the_key_printed_before_sending(peer, capsys):
    rc, payload, err = _run(capsys, key="ticket-7")
    assert rc == 0 and payload["run_id"] == "run_1" and payload["idempotency_key"] == "ticket-7"
    assert "idempotency_key: ticket-7" in err and peer.received == ["ticket-7"]


def test_a_lost_202_is_reported_possibly_admitted_and_the_peer_did_admit_it(peer, capsys):
    peer.mode = "lost_202"
    rc, payload, err = _run(capsys)
    key = payload["idempotency_key"]
    assert rc == peer_cmd.EXIT_POSSIBLY_ADMITTED == 4
    assert payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert peer.admitted == [(key, "deliver me")]  # it WAS delivered: "could not reach" would be false
    assert err.index(f"idempotency_key: {key}") < err.index("not confirmed")  # key out before the loss
    assert "Could not reach" not in err and "always allowed, immediately" in err


def test_an_after_send_timeout_without_admission_is_still_only_possibly_admitted(peer, capsys):
    peer.mode = "hang"
    rc, payload, _err = _run(capsys, key="ticket-8")
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["idempotency_key"] == "ticket-8"
    assert peer.received == ["ticket-8"] and peer.admitted == []  # the client cannot know which: "possibly"


def test_a_setup_failure_before_the_run_is_not_delivered_and_keeps_the_key(monkeypatch, capsys):
    _point_at(monkeypatch, f"http://127.0.0.1:{_closed_port()}")  # even the session lookup cannot connect
    rc, payload, err = _run(capsys, key="ticket-9")
    assert rc == peer_cmd.EXIT_NOT_DELIVERED == 3
    assert payload["status"] == "not_delivered" and payload["idempotency_key"] == "ticket-9"
    assert "was not attempted" in err


def test_a_first_hop_connect_refused_on_the_run_post_is_not_delivered(monkeypatch, capsys):
    """The run POST itself goes to a port nothing listens on: one connect attempted, no byte written."""
    _point_at(monkeypatch, f"http://127.0.0.1:{_closed_port()}")
    monkeypatch.setattr(peer_cmd, "_peer_run_durability", lambda base, key: True)  # setup only, no transport
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bot-chat-1")
    rc, payload, err = _run(capsys, key="ticket-10")
    assert rc == 3 and payload["status"] == "not_delivered"
    assert "connection was refused or its host unresolvable, so no request byte was written" in err


@pytest.mark.parametrize("mode", ["redirect_refused", "redirect_404"])
def test_a_complete_post_followed_by_a_failing_redirect_hop_is_possibly_admitted(peer, capsys, mode):
    """review18499 PEER-TRUTH-01: the receiver read the whole POST; the next hop's failure is no verdict."""
    peer.mode = mode
    rc, payload, err = _run(capsys, key="ticket-11")
    assert peer.received == ["ticket-11"]
    assert rc == 4 and payload["status"] == "possibly_admitted" and "after a redirect" in err
    assert "never sent" not in err


def test_a_peer_that_reads_the_request_then_closes_is_possibly_admitted(peer, capsys):
    peer.mode = "close"
    rc, payload, _err = _run(capsys, key="ticket-12")
    assert peer.received == ["ticket-12"] and rc == 4 and payload["status"] == "possibly_admitted"


@pytest.mark.parametrize(("mode", "raw", "shape"), [
    ("bad_json", "{", "is not JSON"),
    ("missing_id", '{"status": "started"}', "carries no run_id"),
    ("non_object", "[1, 2]", "is not a JSON object"),
    ("undecodable", "\\xff\\xfe{", "is not valid UTF-8 text"),  # review19670 PEER-TRUTH-10: + exact bytes
])
def test_an_answer_that_binds_no_run_is_typed_possibly_admitted_with_the_body_verbatim(peer, capsys, mode, raw,
                                                                                      shape):
    """review18499 PEER-TRUTH-02: no fabricated run ID, no success, no "unreachable"; the answer as it came."""
    peer.mode = mode
    rc, payload, err = _run(capsys, key="ticket-13")
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["idempotency_key"] == "ticket-13"
    assert payload["response"] == {"http_status": 202, "body": raw} and shape in payload["error"]
    assert payload.get("response_hex") == (peer.bodies[mode][1].hex() if mode == "undecodable" else None)
    assert f"\n{raw}\n" in err  # stderr shows the complete body verbatim too


def test_a_server_error_on_the_run_post_is_possibly_admitted_with_its_body(peer, capsys):
    peer.mode = "server_error"
    rc, payload, _err = _run(capsys, key="ticket-14")
    assert rc == 4 and payload["response"] == {"http_status": 503, "body": '{"error": "busy"}'}


def test_a_first_hop_4xx_stays_the_peers_rejection(peer, capsys):
    peer.mode = "conflict"
    rc, payload, err = _run(capsys, key="ticket-15")
    assert rc == 1 and payload is None and "rejected the request (HTTP 409)" in err


_PHASES_SCRIPT = textwrap.dedent("""
    import json, socket, sys, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    installs, real_add = [], sys.addaudithook
    def counting(hook):  # counts and delegates: the real hook is installed
        installs.append(hook)
        real_add(hook)
    sys.addaudithook = counting
    from hermes_cli.subcommands import peer

    hold, arrived = threading.Event(), threading.Event()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/slow":
                arrived.set()
                hold.wait(5)
            out = b'{"run_id": "r"}'
            self.send_response(202)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    live = f"http://127.0.0.1:{srv.server_port}"

    def tracked(url):
        with peer._RunSendPhase() as ph:
            try:
                peer._request(url, "k", method="POST", body={}, raw=True, timeout=5)
            except OSError:
                pass
        return [ph.connects, ph.sends]

    serial = [tracked(live + "/fast" if i % 2 == 0 else f"http://127.0.0.1:{dead}/x") for i in range(6)]
    slow = {}
    t = threading.Thread(target=lambda: slow.setdefault("p", tracked(live + "/slow")))
    t.start()
    assert arrived.wait(5)  # the slow POST is in flight on its own thread while the refused one runs
    refused_meanwhile = tracked(f"http://127.0.0.1:{dead}/x")
    hold.set()
    t.join(10)
    srv.shutdown()
    print(json.dumps({"installs": len(installs), "serial": serial, "refused_meanwhile": refused_meanwhile,
                      "slow": slow.get("p"), "outside": [peer._SEND_PHASE.__dict__.get("current")]}))
""")


def test_the_audit_hook_is_installed_once_and_counts_stay_per_call_and_per_thread():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    done = subprocess.run([sys.executable, "-c", _PHASES_SCRIPT], env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["installs"] == 1  # N tracked POSTs, one process-wide hook
    for i, (connects, sends) in enumerate(result["serial"]):
        assert connects == 1 and ((sends >= 1) if i % 2 == 0 else (sends == 0)), result
    assert result["refused_meanwhile"] == [1, 0] and result["slow"][0] == 1 and result["slow"][1] >= 1, result
    assert result["outside"] == [None]  # no tracker survives its block
