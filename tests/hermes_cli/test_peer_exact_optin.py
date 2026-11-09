"""TR-02 / TR-2 (RB4765 r3c11h, aa04): Walker wiring-r4's OPT-IN ``--exact`` operand (peer-exact-ingress.patch
80551b8e) at the request boundary, plus named delta D1. Every case drives the REAL CLI parser
(``hermes_cli.main._build_cli_parser`` + ``_parse_cli_args``) and the real request construction (``_turn_body`` ->
``_request`` -> urllib) to a local HTTP peer that records RAW request bytes. No live peer, no credential read.

- The adapters' exact invocation (wiring-r4 ``PEER_RUN_EXACT_COMMAND``):
  ``hermes peer run --idempotency-key K --json --exact <target> -- <body>``; the plain one has no ``--exact``.
- DEFAULT (no ``--exact``) must stay byte-compatible with the baseline: these cases pass on the baseline too.
- ``*_d1`` cases test delta D1 (``--exact`` refuses NUL / non-UTF-8 with exit 2, nothing sent); they fail on the
  operand-only tree by design (the operand sends such a body).
Expected: exact-mode cases RED on the baselines (no ``--exact``: usage error), GREEN on TR-2."""
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_cmd


class _Peer(BaseHTTPRequestHandler):
    posts: list = []

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/capabilities":
            return self._json({"features": {"runs_idempotency": {"supported": True, "durable": True,
                                                                 "retention_seconds": 86400}}})
        if self.path.startswith("/api/sessions"):
            return self._json({"object": "list", "data": [{"id": "bc_1", "title": "Bot Chat"}]})
        return self._json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).posts.append((self.path, dict(self.headers), raw))
        if self.path.endswith("/chat"):
            return self._json({"object": "hermes.session.chat.completion", "session_id": "bc_1",
                               "message": {"role": "assistant", "content": "ack"}})
        if self.path == "/v1/runs":
            return self._json({"run_id": "run_1", "status": "started", "replayed": False}, 202)
        return self._json({"error": {"message": "not found"}}, 404)

    def log_message(self, *args):
        pass


@pytest.fixture()
def peer(monkeypatch):
    _Peer.posts = []
    server = HTTPServer(("127.0.0.1", 0), _Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": f"http://127.0.0.1:{server.server_port}"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "fixture-key-0000000000")
    monkeypatch.setattr("agent.swarm_capabilities.peer_session_author", lambda: None, raising=False)
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    try:
        yield _Peer
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _tty(monkeypatch, raw=None):
    """stdin: a TTY (no body), or a pipe of exactly ``raw`` bytes."""
    stream = SimpleNamespace(isatty=lambda: raw is None, buffer=io.BytesIO(raw or b""),
                             read=lambda: (raw or b"").decode("utf-8"))
    monkeypatch.setattr("sys.stdin", stream)


def _cli(argv):
    """The real top-level parser and its routing; returns ``args.func(args)`` (a usage error: its exit code)."""
    from hermes_cli.main import _build_cli_parser, _parse_cli_args
    parser, subparsers = _build_cli_parser()
    try:
        args = _parse_cli_args(parser, subparsers, argv)
    except SystemExit as exc:
        return exc.code
    return args.func(args)


def _sent(peer, action):
    path = "/v1/runs" if action == "run" else "/chat"
    key = "input" if action == "run" else "message"
    bodies = [json.loads(raw) for p, _, raw in peer.posts if p.endswith(path)]
    return [b[key] for b in bodies]


def _exact_argv(body, key="k1", action="run"):
    keyargs = ["--idempotency-key", key] if action == "run" else []
    return ["peer", action, *keyargs, "--json", "--exact", "spark", "--", body]


def _plain_argv(body, key="k1"):
    return ["peer", "run", "--idempotency-key", key, "--json", "spark", "--", body]


EXACT = {
    "lead-trail-spaces": "  leading and trailing  ",
    "newlines": "line one\nline two\n",
    "paragraph": "\n\nparagraph\n\n",
    "crlf": "\r\nCRLF body\r\n",
    "tabs-vt-ff": "\t\x0b\x0cbody\t",
    "unicode-ws": "\u00a0nbsp\u2003em\u3000ideographic\u2028ls\u2029ps\u0085nel\ufeff",
    "help-trailing-space": "--help ",
    "emoji-ws": " \U0001F916 \u00e9t\u00e9 ",
}
PLAIN = {"help": "--help", "h": "-h", "dashdash": "--", "dash": "-", "exact-flag": "--exact", "json-flag": "--json",
         "emoji": "Message from \U0001F916 dixie: e\u0301", "inner-ws": "a  b\nc"}


@pytest.mark.parametrize("name", sorted(EXACT))
def test_tr2_adapter_exact_invocation_is_byte_exact(peer, monkeypatch, name):
    body = EXACT[name]
    _tty(monkeypatch)
    rc = _cli(_exact_argv(body, f"k-{name}"))
    sent = _sent(peer, "run")
    print("TR2 exact", name, repr(body), "->", rc, [repr(s) for s in sent])
    assert rc == 0 and sent == [body]
    assert hashlib.sha256(sent[0].encode("utf-8")).digest() == hashlib.sha256(body.encode("utf-8")).digest()


@pytest.mark.parametrize("name", sorted(PLAIN))
def test_tr2_adapter_plain_invocation_is_unchanged(peer, monkeypatch, name):
    """Bodies that stripping leaves unchanged go WITHOUT --exact (wiring-r4); identical on baseline and TR-2."""
    body = PLAIN[name]
    _tty(monkeypatch)
    assert _cli(_plain_argv(body, f"k-{name}")) == 0 and _sent(peer, "run") == [body]


@pytest.mark.parametrize("name", sorted(EXACT))
def test_tr2_default_path_is_byte_compatible(peer, monkeypatch, name):
    """No --exact: the baseline's strip, exactly (passes on the baseline too)."""
    body = EXACT[name]
    _tty(monkeypatch)
    rc = _cli(_plain_argv(body, f"k-{name}"))
    assert rc == 0 and _sent(peer, "run") == [body.strip()]


@pytest.mark.parametrize("blank", [" ", "\n\t ", "\u3000"], ids=["space", "nl-tab", "ideographic"])
@pytest.mark.parametrize("exact", [False, True], ids=["default", "exact"])
def test_tr2_blank_bodies_are_refused_either_way(peer, monkeypatch, capsys, blank, exact):
    """A blank body is no message: refused (exit 2), nothing sent, with or without --exact (operand: unchanged)."""
    _tty(monkeypatch)
    rc = _cli(_exact_argv(blank) if exact else _plain_argv(blank))
    assert rc == 2 and peer.posts == [], capsys.readouterr().err


def test_tr2_default_stdin_is_stripped_as_before(peer, monkeypatch):
    _tty(monkeypatch, b"  piped body \n")
    assert _cli(["peer", "run", "spark"]) == 0 and _sent(peer, "run") == ["piped body"]


def test_tr2_exact_stdin_is_byte_exact(peer, monkeypatch):
    _tty(monkeypatch, b"  piped body \n")
    assert _cli(["peer", "run", "--exact", "spark"]) == 0 and _sent(peer, "run") == ["  piped body \n"]


def test_tr2_dm_exact_is_byte_exact(peer, monkeypatch):
    _tty(monkeypatch)
    assert _cli(_exact_argv("  dm body\n", action="dm")) == 0 and _sent(peer, "dm") == ["  dm body\n"]


def test_tr2_help_lists_exact_and_a_help_body_after_end_of_options_is_sent(peer, monkeypatch, capsys):
    _tty(monkeypatch)
    rc = _cli(["peer", "run", "--help"])
    out = capsys.readouterr().out
    assert rc == 0 and "--exact" in out and peer.posts == []
    assert _cli(_exact_argv("--help\n")) == 0 and _sent(peer, "run") == ["--help\n"]


def test_tr2_lost_ack_a_then_later_b_then_resend_a_byte_identical(peer, monkeypatch):
    """At-least-once at the request boundary: A (whitespace, --exact) under ka, a later B under kb, then A resent
    under ka. Three requests in order; A's two request bodies are byte-identical (a duplicate is acceptable)."""
    a, b = "  A: the uncertain one \n", "B: the later one"
    _tty(monkeypatch)
    assert _cli(_exact_argv(a, "ka")) == 0
    assert _cli(_plain_argv(b, "kb")) == 0
    assert _cli(_exact_argv(a, "ka")) == 0
    runs = [(h.get("Idempotency-Key"), raw) for p, h, raw in peer.posts if p == "/v1/runs"]
    assert [k for k, _ in runs] == ["ka", "kb", "ka"]
    assert runs[0][1] == runs[2][1]
    assert [json.loads(raw)["input"] for _, raw in runs] == [a, b, a]


@pytest.mark.parametrize("case", ["nul-stdin", "surrogate-arg", "surrogate-stdin"])
def test_tr2_exact_refuses_undeliverable_bytes_d1(peer, monkeypatch, capsys, case):
    """Delta D1: --exact refuses NUL / non-UTF-8 (exit 2, nothing sent). The operand alone sends them."""
    if case == "nul-stdin":
        _tty(monkeypatch, b" a\x00b ")
        argv = ["peer", "run", "--exact", "spark"]
    elif case == "surrogate-stdin":  # an undecodable stdin byte under UTF-8 mode's surrogateescape
        stream = SimpleNamespace(isatty=lambda: False, read=lambda: " caf\udce9 ")
        monkeypatch.setattr("sys.stdin", stream)
        argv = ["peer", "run", "--exact", "spark"]
    else:  # an undecodable argv byte arrives as a lone surrogate (surrogateescape)
        _tty(monkeypatch)
        argv = _exact_argv(" caf\udce9 ")
    rc = _cli(argv)
    err = capsys.readouterr().err
    print("TR2 D1", case, rc, err.strip())
    assert rc == 2 and peer.posts == [], (rc, err)
