"""PEER-TRUTH-11/12: real main/parser/security/urllib/socket response evidence.

The fixture owns the wire and only substitutes private peer configuration and a dummy
key. Short Content-Length replies must retain available status/bytes without ever
parsing a partial handle. No response reader, transport or exception is substituted.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli.subcommands import peer as peer_cmd

INVALID = bytes.fromhex("7b226572726f72223a225cff227d")
LITERAL = bytes.fromhex("7b226572726f72223a225c5c786666227d")


class _EvidencePeer(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _answer(self, code, body, excess=0):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body) + excess))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self):
        body = (b'{"features":{"runs_idempotency":{"supported":true,"durable":true}}}'
                if self.path == "/v1/capabilities" else
                b'{"data":[{"id":"evidence-chat","title":"Bot Chat"}]}')
        self._answer(200, body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        key = self.headers.get("Idempotency-Key")
        self.server.received.append({"key": key, "body": raw.decode("utf-8"),
                                     "key_before_post": key in self.server.stderr.getvalue()})
        if self.server.hold:
            self.send_response(self.server.code)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.flush()
            # A real read timeout after headers exposes no partial-byte attribute.
            self.server.release.wait(10)
            self.close_connection = True
            return
        self._answer(self.server.code, self.server.body, self.server.excess)


@pytest.fixture
def evidence_peer(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EvidencePeer)
    server.daemon_threads = True
    server.code, server.body, server.excess, server.hold = 202, b'{"run_id":"healthy"}', 0, False
    server.release = threading.Event()
    server.received, server.observations = [], []
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"evidence": {"url": f"http://127.0.0.1:{server.server_port}"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda _name: "evidence-dummy-key")
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()
    worker.join(5)
    assert not worker.is_alive()


@pytest.fixture
def invoke_main(evidence_peer, monkeypatch, request):
    from hermes_cli.main import main

    def invoke(key="evidence-key"):
        out, err = io.StringIO(), io.StringIO()
        evidence_peer.stderr = err
        monkeypatch.setattr(sys, "argv", ["hermes", "peer", "run", "evidence", "message-" + key,
                                          "--json", "--idempotency-key", key])
        rc = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main()
            except SystemExit as exc:
                rc = exc.code
        payload = json.loads(out.getvalue()) if out.getvalue().strip() else None
        evidence_peer.observations.append({"rc": rc, "stdout": out.getvalue(), "stderr": err.getvalue(),
                                           "status": evidence_peer.code,
                                           "wire_hex": None if evidence_peer.hold else evidence_peer.body.hex(),
                                           "excess": evidence_peer.excess, "hold": evidence_peer.hold,
                                           "received": list(evidence_peer.received)})
        return rc, payload, err.getvalue()

    yield invoke
    destination = os.environ.get("REVIEW_OBSERVATIONS")
    if destination:
        name = request.node.name.replace("/", "_") + ".json"
        (Path(destination) / name).write_text(json.dumps(evidence_peer.observations, indent=2))


def _uncertain(rc, payload, err, key="evidence-key"):
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["idempotency_key"] == key
    assert "never sent" not in err and "no request byte was written" not in err
    assert "always allowed, immediately" in err


@pytest.mark.parametrize("code", [202, 503])
@pytest.mark.parametrize("body", [b'{"run_id":"accepted-but-', INVALID, b'{"run_id":"looks-complete"}', b""],
                         ids=["prefix", "invalid", "complete-looking", "empty"])
def test_incomplete_response_preserves_exact_available_evidence(evidence_peer, invoke_main, code, body):
    evidence_peer.code, evidence_peer.body, evidence_peer.excess = code, body, 100
    rc, payload, err = invoke_main()
    _uncertain(rc, payload, err)
    assert payload["response"] == {"http_status": code, "body": body.decode("utf-8", "backslashreplace")}
    assert payload.get("response_hex") == (body.hex() if body == INVALID else None)
    assert payload["response_complete"] is False
    assert "IncompleteRead" in payload["response_read_error"]
    assert "100 more expected" in payload["response_read_error"]
    assert "incomplete" in err.lower() and "complete body, verbatim" not in err
    assert evidence_peer.received == [{"key": "evidence-key", "body": json.dumps(
        {"input": "message-evidence-key", "session_id": "evidence-chat"}), "key_before_post": True}]


@pytest.mark.parametrize("code", [202, 503])
def test_read_failure_with_no_exposed_bytes_is_not_an_empty_body(evidence_peer, invoke_main, monkeypatch, code):
    evidence_peer.code, evidence_peer.hold = code, True
    real_request = peer_cmd._request

    def bounded_request(url, key, **kwargs):
        # Only the time budget changes; the real urllib/socket response reader is exercised.
        return real_request(url, key, **{**kwargs, "timeout": 2})

    monkeypatch.setattr(peer_cmd, "_request", bounded_request)
    rc, payload, err = invoke_main()
    _uncertain(rc, payload, err)
    assert payload["response"] == {"http_status": code, "body": None}
    assert payload["response_complete"] is False
    assert "TimeoutError" in payload["response_read_error"]
    assert "response_hex" not in payload
    assert "unavailable" in err.lower()


def test_invalid_and_literal_error_bodies_keep_distinct_exact_identities(evidence_peer, invoke_main):
    identities = []
    for body in (INVALID, LITERAL):
        evidence_peer.code, evidence_peer.body = 503, body
        rc, payload, err = invoke_main()
        _uncertain(rc, payload, err)
        assert payload["response"] == {"http_status": 503, "body": body.decode("utf-8", "backslashreplace")}
        exact = bytes.fromhex(payload["response_hex"]) if "response_hex" in payload else payload["response"]["body"].encode("utf-8")
        assert exact == body
        identities.append(exact)
        assert "response_read_error" not in payload and "response_complete" not in payload
    assert identities[0] != identities[1]


@pytest.mark.parametrize("code", [202, 503])
def test_complete_empty_response_is_valid_empty_evidence(evidence_peer, invoke_main, code):
    evidence_peer.code, evidence_peer.body = code, b""
    rc, payload, err = invoke_main()
    _uncertain(rc, payload, err)
    assert payload["response"] == {"http_status": code, "body": ""}
    assert "response_read_error" not in payload and "response_complete" not in payload


@pytest.mark.parametrize("handle", ["healthy", "\\xff", "λ:7/a b"])
def test_complete_healthy_handles_remain_successful(evidence_peer, invoke_main, handle):
    evidence_peer.body = json.dumps({"run_id": handle}, ensure_ascii=False).encode("utf-8")
    rc, payload, _err = invoke_main()
    assert rc == 0 and payload["run_id"] == handle
    assert evidence_peer.received[0]["key_before_post"]


@pytest.mark.parametrize("code", [401, 403, 409])
def test_first_hop_rejection_stays_distinct(evidence_peer, invoke_main, code):
    evidence_peer.code, evidence_peer.body = code, b'{"error":{"message":"fixture rejection"}}'
    rc, payload, err = invoke_main()
    assert rc == 1 and payload is None and f"rejected the request (HTTP {code})" in err


def test_partial_a_never_gates_b_c_or_a_message_retry(evidence_peer, invoke_main):
    evidence_peer.body, evidence_peer.excess = b'{"run_id":"partial-a"}', 100
    rc, payload, err = invoke_main("A")
    _uncertain(rc, payload, err, "A")
    evidence_peer.body, evidence_peer.excess = b'{"run_id":"healthy"}', 0
    for key in ("B", "C", "A", "A-new-key"):
        rc, payload, _err = invoke_main(key)
        assert rc == 0 and payload["idempotency_key"] == key
    assert [r["key"] for r in evidence_peer.received] == ["A", "B", "C", "A", "A-new-key"]
    assert all(r["key_before_post"] for r in evidence_peer.received)
