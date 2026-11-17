"""``hermes peer run`` delivery truth at the review19626 boundaries (PEER-TRUTH-07/08).

- 07: "never sent" (3) is evidence about the transport the run POST actually went through. An opener installed by
  another thread AFTER the send phase began (and before the credential-safe opener was built) must be the one judged:
  a custom handler that writes the complete POST unaudited and then fails another connection is
  ``possibly_admitted`` (4), never 3. A stock first hop that is refused stays 3.
- 08: a string run_id the follow-up requests cannot encode (an escaped lone surrogate) binds no usable run: 4,
  run_id null, the answer verbatim; any encodable string (astral Unicode, punctuation) stays 0 and works for status.
Real local HTTP servers, real urllib openers and a real competing ``install_opener``; nothing substitutes the
classifier, the transport or an exception. The only seam schedules the thread switch around the real ``_turn_body``.
"""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from tests.hermes_cli import test_peer_run_delivery_truth as truth
from tests.hermes_cli import test_peer_run_delivery_truth_edges as edges

_point_at, _run = truth._point_at, truth._run
peer = truth.peer  # the real local HTTP peer fixture, shared
opener = edges.opener


def _switch_opener_during_send_phase(monkeypatch, handler):
    """Install ``handler``'s opener from another thread after the send phase began, before the POST's opener exists."""
    import hermes_cli.subcommands.peer as peer_cmd
    real_turn_body = peer_cmd._turn_body
    arrived, switched = threading.Event(), threading.Event()

    def switcher():
        assert arrived.wait(5)
        urllib.request.install_opener(urllib.request.build_opener(handler))
        switched.set()

    def turn_body(*args, **kwargs):
        arrived.set()
        assert switched.wait(5)
        return real_turn_body(*args, **kwargs)

    monkeypatch.setattr(peer_cmd, "_turn_body", turn_body)
    thread = threading.Thread(target=switcher)
    thread.start()
    return thread


def test_a_custom_opener_installed_during_the_send_phase_is_the_transport_judged(peer, capsys, monkeypatch, opener):
    urllib.request.install_opener(None)  # stock when the send phase begins (the fixture resets it at teardown)
    thread = _switch_opener_during_send_phase(monkeypatch, edges._WriteThenRefusedHandler(truth._closed_port()))
    rc, payload, err = _run(capsys, key="race-07")
    thread.join(5)
    assert peer.received == ["race-07"]  # the receiver read the complete POST
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert "never sent" not in err and "no request byte was written" not in err


def test_a_stock_opener_installed_during_the_send_phase_keeps_a_refused_first_hop_not_delivered(monkeypatch, capsys,
                                                                                               opener):
    import hermes_cli.subcommands.peer as peer_cmd
    urllib.request.install_opener(None)
    _point_at(monkeypatch, f"http://127.0.0.1:{truth._closed_port()}")
    monkeypatch.setattr(peer_cmd, "_peer_run_durability", lambda base, key: True)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bot-chat-1")
    thread = _switch_opener_during_send_phase(monkeypatch, urllib.request.HTTPHandler())
    rc, payload, _err = _run(capsys, key="race-07-stock")
    thread.join(5)
    assert rc == 3 and payload["status"] == "not_delivered"


def test_repeated_runs_judge_each_post_by_its_own_opener(peer, capsys, opener):
    opener(edges._WriteThenRefusedHandler(truth._closed_port()))
    rc_custom, _payload, _err = _run(capsys, key="race-07-a")
    urllib.request.install_opener(None)
    rc_stock, payload, _err = _run(capsys, key="race-07-b")
    assert (rc_custom, rc_stock) == (4, 0) and payload["run_id"]


@pytest.mark.parametrize("raw", [b'{"run_id": "\\ud800"}', b'{"run_id": "\\udfff"}', b'{"run_id": "a\\udc80b"}'],
                         ids=["lone-high", "lone-low", "embedded-low"])
def test_a_run_id_the_follow_up_requests_cannot_encode_binds_no_run(peer, capsys, raw):
    """PEER-TRUTH-08."""
    peer.bodies["bad_id"] = (202, raw)
    peer.mode = "bad_id"
    rc, payload, err = _run(capsys, key="race-08")
    assert rc == 4 and payload["status"] == "possibly_admitted" and payload["run_id"] is None
    assert payload["response"] == {"http_status": 202, "body": raw.decode()}
    assert "cannot be used in a run request" in payload["error"] and f"\n{raw.decode()}\n" in err


@pytest.mark.parametrize("run_id", ["run:7/a b", "运行-𝔘𝔫𝔦𝔠𝔬𝔡𝔢-😀", "%2F..?#"], ids=["punctuation", "astral", "reserved"])
def test_any_encodable_run_id_is_the_answer_and_its_follow_up_path_encodes(peer, capsys, run_id):
    import hermes_cli.subcommands.peer as peer_cmd
    peer.bodies["odd_id"] = (202, json.dumps({"run_id": run_id, "status": "started"}).encode())
    peer.mode = "odd_id"
    rc, payload, _err = _run(capsys, key="race-08-ok")
    assert rc == 0 and payload["run_id"] == run_id
    segment = peer_cmd._run_path(run_id)  # exactly what status/stop put in /v1/runs/<id>
    assert "/" not in segment and segment.isascii()
