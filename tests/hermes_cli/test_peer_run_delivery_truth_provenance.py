"""``hermes peer run`` delivery truth at the review19670 boundaries (PEER-TRUTH-09/10).

- 09: "never sent" (3) needs the transport's actual writing behavior to be stock, not only its class. A genuine
  ``urllib.request.HTTPHandler`` INSTANCE whose ``http_open`` is replaced per instance (it survives the policy's
  ``copy.copy`` clone) can write the complete POST without a send event and then fail another connection with a
  refusal: that is ``possibly_admitted`` (4), installed before the run or during its send phase. A truly stock
  refused first hop stays 3, and an unmodified handler instance is still stock.
- 10: the answer's bytes are parsed only if they are valid UTF-8. A raw ``\\`` followed by an invalid byte must not
  become the same parser input as a legitimate literal backslash: invalid UTF-8 is 4, run_id null, with the
  readable body AND the exact bytes (hex); a valid literal-backslash id stays 0 and is the handle.
Real local HTTP servers, real urllib openers, real ``install_opener``; nothing substitutes the classifier, the
transport or an exception.
"""
from __future__ import annotations

import urllib.request

import pytest

from tests.hermes_cli import test_peer_run_delivery_truth as truth
from tests.hermes_cli import test_peer_run_delivery_truth_edges as edges
from tests.hermes_cli import test_peer_run_delivery_truth_race as race

_point_at, _run = truth._point_at, truth._run
peer = truth.peer  # the real local HTTP peer fixture, shared
opener = edges.opener


def _instance_writer(refused_port):
    """A genuine stock-class handler instance whose own ``http_open`` writes unaudited, then fails elsewhere."""
    handler = urllib.request.HTTPHandler()
    custom = edges._WriteThenRefusedHandler(refused_port)
    handler.http_open = lambda req: custom.http_open(req)
    return handler


def test_a_stock_class_instance_with_its_own_http_open_is_not_judged_stock(peer, capsys, opener):
    opener(_instance_writer(truth._closed_port()))
    rc, payload, err = _run(capsys, key="pt09-installed")
    assert peer.received == ["pt09-installed"]  # the receiver read the complete POST
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["idempotency_key"] == "pt09-installed"
    assert "never sent" not in err and "no request byte was written" not in err


def test_such_an_instance_installed_during_the_send_phase_is_judged_too(peer, capsys, monkeypatch, opener):
    urllib.request.install_opener(None)
    thread = race._switch_opener_during_send_phase(monkeypatch, _instance_writer(truth._closed_port()))
    rc, payload, _err = _run(capsys, key="pt09-race")
    thread.join(5)
    assert peer.received == ["pt09-race"]
    assert rc == 4 and payload["status"] == "possibly_admitted"


@pytest.mark.parametrize("installed", ["none", "unmodified-instance"])
def test_a_truly_stock_refused_first_hop_is_still_not_delivered(monkeypatch, capsys, opener, installed):
    import hermes_cli.subcommands.peer as peer_cmd
    if installed == "none":
        urllib.request.install_opener(None)
    else:
        opener(urllib.request.HTTPHandler())  # data attributes only, as constructed
    _point_at(monkeypatch, f"http://127.0.0.1:{truth._closed_port()}")
    monkeypatch.setattr(peer_cmd, "_peer_run_durability", lambda base, key: True)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bot-chat-1")
    rc, payload, _err = _run(capsys, key=f"pt09-stock-{installed}")
    assert rc == 3 and payload["status"] == "not_delivered"


def test_the_stock_judgement_is_per_handler_instance():
    import hermes_cli.subcommands.peer as peer_cmd
    plain, patched = urllib.request.HTTPHandler(), urllib.request.HTTPHandler()
    patched.http_open = lambda req: None
    assert peer_cmd._stock_opener(urllib.request.build_opener(plain))
    assert not peer_cmd._stock_opener(urllib.request.build_opener(patched))


INVALID = bytes.fromhex("7b2272756e5f6964223a225cff227d")  # {"run_id":"\<0xff>"}: not UTF-8
LITERAL = bytes.fromhex("7b2272756e5f6964223a225c5c786666227d")  # {"run_id":"\\xff"}: id is backslash x f f


@pytest.mark.parametrize("body", [INVALID, b'{"run_id": "run_9", "note": "\xc3\x28"}'], ids=["id", "metadata"])
def test_an_answer_that_is_not_valid_utf8_binds_no_run(peer, capsys, body):
    peer.bodies["raw"] = (202, body)
    peer.mode = "raw"
    rc, payload, err = _run(capsys, key="pt10-invalid")
    assert peer.received == ["pt10-invalid"]
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["idempotency_key"] == "pt10-invalid"
    assert payload["response_hex"] == body.hex()  # the exact bytes, never only a lossy rendering
    assert payload["response"] == {"http_status": 202, "body": body.decode("utf-8", "backslashreplace")}
    assert "not valid UTF-8" in payload["error"] and body.hex() in err


def test_a_legitimate_literal_backslash_id_is_still_the_handle(peer, capsys):
    peer.bodies["raw"] = (202, LITERAL)
    peer.mode = "raw"
    rc, payload, _err = _run(capsys, key="pt10-literal")
    assert rc == 0 and payload["run_id"] == "\\xff"


def test_the_two_bodies_are_not_one_answer(peer, capsys):
    results = []
    for body in (INVALID, LITERAL):
        peer.bodies["raw"] = (202, body)
        peer.mode = "raw"
        rc, payload, _err = _run(capsys, key="pt10-pair")
        results.append((rc, payload["run_id"]))
    assert results == [(4, None), (0, "\\xff")]
