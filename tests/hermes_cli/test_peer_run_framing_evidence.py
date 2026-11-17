"""PEER-TRUTH-13/14: real public main and stdlib framing, with private fixture peers only."""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli.subcommands import peer


class _FramingPeer(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        body = (b'{"features":{"runs_idempotency":{"supported":true,"durable":true}}}'
                if self.path == "/v1/capabilities" else
                b'{"data":[{"id":"framing-chat","title":"Bot Chat"}]}')
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        key = self.headers.get("Idempotency-Key")
        self.server.received.append({"body_hex": body.hex(), "key": key,
                                     "key_before_post": key in self.server.stderr.getvalue()})
        self.wfile.write(self.server.wire)
        self.wfile.flush()
        self.close_connection = True


def _complete(code=202, body=b'{"run_id":"good"}'):
    return (f"HTTP/1.1 {code} Fixture\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)


def _chunked(code, chunks):
    return f"HTTP/1.1 {code} Fixture\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode() + chunks


@pytest.fixture
def framing(monkeypatch, request):
    from hermes_cli.main import main

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FramingPeer)
    server.daemon_threads = True
    server.received, observations = [], []
    server.wire, server.stderr = _complete(), io.StringIO()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(peer, "_load_peers", lambda: {"framing": {"url": f"http://127.0.0.1:{server.server_port}"}})
    monkeypatch.setattr(peer, "_peer_secret", lambda _: "framing-dummy-key")

    def invoke(wire, key="framing-key", message="same-logical-message"):
        server.wire = wire
        out, err = io.StringIO(), io.StringIO()
        server.stderr = err
        monkeypatch.setattr(sys, "argv", ["hermes", "peer", "run", "framing", message,
                                          "--json", "--idempotency-key", key])
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main()
            except SystemExit as exc:
                rc = exc.code
            else:
                rc = 0
        payload = json.loads(out.getvalue()) if out.getvalue().strip() else None
        observations.append({"wire_hex": wire.hex(), "rc": rc, "stdout": out.getvalue(),
                             "stderr": err.getvalue(), "received": list(server.received)})
        return rc, payload, err.getvalue()

    yield server, invoke, observations
    server.shutdown()
    server.server_close()
    worker.join(5)
    assert not worker.is_alive()
    destination = os.environ.get("REVIEW_OBSERVATIONS")
    if destination:
        (Path(destination) / (request.node.name.replace("/", "_") + ".json")).write_text(
            json.dumps(observations, indent=2))


def _uncertain(result, status, key="framing-key"):
    rc, payload, err = result
    assert rc == 4
    assert payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["idempotency_key"] == key
    assert payload["response"]["http_status"] == status
    assert "rejected the request" not in err and "never sent" not in err
    assert "always allowed, immediately" in err
    return payload


@pytest.mark.parametrize("code", [202, 503])
@pytest.mark.parametrize("tail", [b"tail", b"\xff", b""], ids=["text", "binary", "empty"])
def test_nested_chunked_partials_are_distinct_exact_observations(framing, code, tail):
    server, invoke, observations = framing
    wire = _chunked(code, b"4\r\nhead\r\n8\r\n" + tail)
    payload = _uncertain(invoke(wire), code)
    assert payload["response_complete"] is False
    assert payload["response"] == {"http_status": code, "body": "head"}
    # These are separate exception observations, NOT an invented complete body.
    parts = payload["response_body_observations"]
    assert [(p["source"], bytes.fromhex(p["body_hex"])) for p in parts] == [
        ("read_error.partial", b"head"), ("read_error.__cause__.partial", tail)]
    assert "not a reconstructed body" in observations[-1]["stderr"]
    assert server.received[0]["key_before_post"]
    assert len(server.received) == 1
    # Acquire the real reader's independently exposed structure, without replacing it.
    con = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        con.request("POST", "/v1/runs", b"reader-fixture", {"Idempotency-Key": "direct"})
        response = con.getresponse()
        with pytest.raises(http.client.IncompleteRead) as caught:
            response.read()
        outer = caught.value
        assert outer.partial == b"head" and outer.__cause__.partial == tail
        observations.append({"direct_status": response.status, "outer_hex": outer.partial.hex(),
                             "cause_hex": outer.__cause__.partial.hex(), "wire_hex": wire.hex()})
    finally:
        con.close()


@pytest.mark.parametrize("info", [101, 102, 103, 199])
@pytest.mark.parametrize("final", [False, True], ids=["no-final", "final202"])
def test_informational_answer_is_not_a_rejection(framing, info, final):
    server, invoke, _ = framing
    wire = f"HTTP/1.1 {info} Informational\r\nLink: </fixture>\r\n\r\n".encode()
    if final:
        wire += _complete()
    payload = _uncertain(invoke(wire), info)
    assert payload["response"]["body"] == ""  # the stdlib exposes no 1xx body
    assert "informational" in payload["error"].lower()
    assert len(server.received) == 1  # no second POST to infer the first outcome
    assert server.received[0]["key_before_post"]


@pytest.mark.parametrize("code", [401, 403, 409])
def test_definite_first_hop_rejection_still_exits_one(framing, code):
    server, invoke, _ = framing
    rc, payload, err = invoke(_complete(code, b'{"error":{"message":"fixture rejection"}}'))
    assert rc == 1 and payload is None and f"rejected the request (HTTP {code})" in err
    assert len(server.received) == 1


@pytest.mark.parametrize("code", [202, 503])
def test_complete_chunked_and_normal_answers_keep_compatible_shapes(framing, code):
    _, invoke, _ = framing
    for wire in (_complete(code), _chunked(code, b'11\r\n{"run_id":"good"}\r\n0\r\n\r\n')):
        rc, payload, _ = invoke(wire)
        if code == 202:
            assert rc == 0 and payload["run_id"] == "good"
        else:
            assert rc == 4 and payload["response"] == {"http_status": 503, "body": '{"run_id":"good"}'}
        assert "response_body_observations" not in payload
        assert "response_read_error" not in payload and "response_complete" not in payload


def test_uncertain_a_does_not_gate_later_messages_or_identical_a_retries(framing):
    server, invoke, _ = framing
    _uncertain(invoke(b"HTTP/1.1 103 Early Hints\r\n\r\n" + _complete(), "A", "logical-A"), 103, "A")
    for key, message in [("B", "logical-B"), ("C", "logical-C"), ("A", "logical-A"), ("A-new", "logical-A")]:
        rc, payload, _ = invoke(_complete(), key, message)
        assert rc == 0 and payload["idempotency_key"] == key
    assert [r["key"] for r in server.received] == ["A", "B", "C", "A", "A-new"]
    assert [json.loads(bytes.fromhex(r["body_hex"]))["input"] for r in server.received] == [
        "logical-A", "logical-B", "logical-C", "logical-A", "logical-A"]
    assert all(r["key_before_post"] for r in server.received)


def test_arbitrary_causal_partials_are_not_concatenated_or_deduplicated_by_value():
    # A boundary-only graph control complements the REAL reader/main tests above.
    outer = http.client.IncompleteRead(b"same")
    cause = http.client.IncompleteRead(b"same")
    context = http.client.IncompleteRead(b"\x00\xff")
    outer.__cause__, outer.__context__ = cause, context
    cause.__cause__ = outer  # cycle must not strand reporting
    error = peer._ResponseReadError(202, outer)
    assert error.partial == b"same"
    assert [(p.source, p.body) for p in error.observations] == [
        ("read_error.partial", b"same"), ("read_error.__cause__.partial", b"same"),
        ("read_error.__context__.partial", b"\x00\xff")]
