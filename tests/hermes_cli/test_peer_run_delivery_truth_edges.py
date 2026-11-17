"""``hermes peer run`` delivery truth at the edges review19553 found (PEER-TRUTH-03..06).

- 03: a supported custom urllib handler may write the request WITHOUT http.client's send audit event; a receiver
  that read the complete POST and closed is still ``possibly_admitted`` (4), never "never sent" (3); so is one that
  writes the POST and then fails ANOTHER connection with a refusal. Exit 3 needs the stock transport (no installed
  custom handler) with its first connection itself refused (or its host unresolvable): no TCP connection, no byte.
- 04: the text claims only what is known: an unobserved send phase is "unknown whether ... was sent", a started
  write "none, some or all of its bytes"; the JSON stays ``possibly_admitted``.
- 05: a redirect the client does not follow (307/308 for a POST, a refused Location) answered the complete POST:
  4 with the response, not the peer's rejection (1).
- 06: a run_id that is not a non-empty string (object, list, bool, blank) binds no usable run: 4, run_id null, the
  answer verbatim; a legitimate string id stays 0.
Real local HTTP servers and real urllib openers (installed with ``install_opener``); nothing substitutes the
classifier, the transport or an exception.
"""
from __future__ import annotations

import http.client
import socket
import urllib.request

import pytest

from tests.hermes_cli import test_peer_run_delivery_truth as truth

_point_at, _run = truth._point_at, truth._run
peer = truth.peer  # the real local HTTP peer fixture, shared


@pytest.fixture
def opener():
    yield lambda *handlers: urllib.request.install_opener(urllib.request.build_opener(*handlers))
    urllib.request.install_opener(None)


class _UnauditedConnection(http.client.HTTPConnection):
    def send(self, data):  # a supported custom transport: real socket, no ``http.client.send`` audit event
        if self.sock is None:
            self.connect()
        self.sock.sendall(data)


class _UnauditedHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_UnauditedConnection, req) if req.full_url.endswith("/v1/runs") else super().http_open(req)


class _PrefixSocket:
    def __init__(self, real):
        self.real = real

    def sendall(self, data):  # writes only the first 12 bytes, then the write side fails
        self.real.sendall(data[:12])
        self.real.shutdown(socket.SHUT_WR)
        raise BrokenPipeError("synthetic partial write")

    def __getattr__(self, name):
        return getattr(self.real, name)


class _PrefixConnection(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        self.sock = _PrefixSocket(self.sock)


class _PrefixHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PrefixConnection, req) if req.full_url.endswith("/v1/runs") else super().http_open(req)


class _RefuseBeforeConnect(urllib.request.BaseHandler):
    def http_request(self, req):
        if req.full_url.endswith("/v1/runs"):
            raise OSError("local request policy refused before connection")
        return req


def test_a_custom_handler_that_writes_unaudited_and_is_closed_on_is_possibly_admitted(peer, capsys, opener):
    """PEER-TRUTH-03: the receiver read the complete POST; no send audit event was seen."""
    opener(_UnauditedHandler())
    peer.mode = "close"
    rc, payload, err = _run(capsys, key="edge-03")
    assert peer.received == ["edge-03"]
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert "never sent" not in err and "unknown whether the run request was sent" in err


def test_a_refusal_before_any_connection_is_honestly_unknown_not_sent(peer, capsys, opener):
    """PEER-TRUTH-04: nothing observed: conservatively 4, and the text does not claim it was sent."""
    opener(_RefuseBeforeConnect())
    rc, payload, err = _run(capsys, key="edge-04a")
    assert peer.received == [] and rc == 4 and payload["status"] == "possibly_admitted"
    assert "unknown whether the run request was sent" in err and "was sent and" not in err


def test_a_partial_write_is_possibly_admitted_and_said_to_be_possibly_partial(peer, capsys, opener):
    opener(_PrefixHandler())
    rc, payload, err = _run(capsys, key="edge-04b")
    assert rc == 4 and payload["status"] == "possibly_admitted"
    assert "a write of the run request was started (none, some or all" in err and "never sent" not in err


@pytest.mark.parametrize("code", [307, 308])
def test_an_unfollowed_post_redirect_is_possibly_admitted_with_its_answer(peer, capsys, code):
    """PEER-TRUTH-05: urllib does not follow a 307/308 for a POST; the complete POST was read."""
    peer.bodies["redirect"] = (code, b"redirect-witness")
    peer.mode = "redirect"
    rc, payload, err = _run(capsys, key=f"edge-05-{code}")
    assert peer.received == [f"edge-05-{code}"]
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["response"] == {"http_status": code, "body": "redirect-witness"}
    assert "a redirect that was not followed" in err and "rejected the request" not in err


@pytest.mark.parametrize(("raw", "shape"), [
    (b'{"run_id": {"bogus": 7}, "status": "started"}', "is not a non-empty string"),
    (b'{"run_id": ["r"]}', "is not a non-empty string"),
    (b'{"run_id": true}', "is not a non-empty string"),
    (b'{"run_id": "   "}', "is not a non-empty string"),
    (b'{"run_id": ""}', "carries no run_id"),
], ids=["object", "list", "bool", "blank", "empty"])
def test_an_unusable_run_id_binds_no_run_and_fabricates_no_handle(peer, capsys, raw, shape):
    """PEER-TRUTH-06."""
    peer.bodies["bad_id"] = (202, raw)
    peer.mode = "bad_id"
    rc, payload, err = _run(capsys, key="edge-06")
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["response"] == {"http_status": 202, "body": raw.decode()} and shape in payload["error"]
    assert f"\n{raw.decode()}\n" in err  # the answer verbatim on stderr too


def test_a_legitimate_string_run_id_is_still_the_answer(peer, capsys):
    peer.bodies["odd_id"] = (202, b'{"run_id": "run:7/a b", "status": "started"}')
    peer.mode = "odd_id"
    rc, payload, _err = _run(capsys, key="edge-06-ok")
    assert rc == 0 and payload["run_id"] == "run:7/a b" and payload["status"] == "started"


def test_an_unresolvable_first_hop_is_still_not_delivered(monkeypatch, capsys):
    import hermes_cli.subcommands.peer as peer_cmd
    _point_at(monkeypatch, "http://no-such-host.invalid:9")
    monkeypatch.setattr(peer_cmd, "_peer_run_durability", lambda base, key: True)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bot-chat-1")
    rc, payload, err = _run(capsys, key="edge-dns")
    assert rc == 3 and payload["status"] == "not_delivered" and "no request byte was written" in err


class _WriteThenRefusedHandler(urllib.request.HTTPHandler):
    """Writes the complete POST unaudited, then fails a DIFFERENT connection (refused), with no stdlib event."""

    def __init__(self, refused_port):
        super().__init__()
        self.refused_port = refused_port

    def http_open(self, req):
        if not req.full_url.endswith("/v1/runs"):
            return super().http_open(req)
        response = self.do_open(_UnauditedConnection, req)
        response.read()
        socket.create_connection(("127.0.0.1", self.refused_port), timeout=1)  # raises ConnectionRefusedError
        return response


def test_a_custom_handler_that_writes_then_fails_another_connection_is_possibly_admitted(peer, capsys, opener):
    opener(_WriteThenRefusedHandler(truth._closed_port()))
    rc, payload, err = _run(capsys, key="edge-03b")
    assert peer.received == ["edge-03b"]  # the receiver read the complete POST
    assert rc == 4 and payload["status"] == "possibly_admitted" and "never sent" not in err


def test_a_stock_first_hop_refusal_stays_not_delivered_with_a_stock_opener_installed(monkeypatch, capsys, opener):
    import hermes_cli.subcommands.peer as peer_cmd
    opener(urllib.request.HTTPHandler())  # an installed opener made only of stock urllib handlers
    _point_at(monkeypatch, f"http://127.0.0.1:{truth._closed_port()}")
    monkeypatch.setattr(peer_cmd, "_peer_run_durability", lambda base, key: True)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bot-chat-1")
    rc, payload, _err = _run(capsys, key="edge-stock-refused")
    assert rc == 3 and payload["status"] == "not_delivered"

