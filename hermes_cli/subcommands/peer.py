"""``hermes peer`` — bot-to-bot DMs across machines/gateways.

A *peer* is another Hermes gateway running the ``api_server`` platform; its stock
API is the transport (no new server surface). ``dm`` resolves the remote canonical
"Bot Chat" session (creating it when missing) and runs ONE synchronous turn — the
cross-machine twin of ``hermes -p <bot> chat --in ~ -c "Bot Chat"``. ``run``/``status``
/``stop`` do the same turn through the async Runs API. Peer labels/URLs live in
config.yaml (``bot_peers``); the key lives in ``~/.hermes/.env`` as
``HERMES_PEER_<NAME>_KEY``. ``<peer>/<profile>`` targets the ``/p/<profile>/`` mirror.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import NamedTuple
from tools.ansi_strip import sanitize_display_text

BOT_CHAT_TITLE = "Bot Chat"
_PEER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# The @handle Bot Mode shows for a host's default profile, and the 404 a peer returns for a
# /p/<profile>/ prefix naming no profile it has (UNKNOWN_PROFILE_CODE / UNKNOWN_PROFILE_ERROR in
# gateway/platforms/api_server_profile_prefix.py; a 'hermes' profile it has but does not serve differs).
DEFAULT_PROFILE_HANDLE = "hermes"
_UNKNOWN_PROFILE_CODE = "profile_unknown"
_UNKNOWN_PROFILE_ERROR = "Unknown or unconfigured profile"

# One synchronous agent turn can legitimately take minutes.
DM_TIMEOUT_S = 600
LIST_TIMEOUT_S = 30

# ``peer run`` exit codes beyond 0 ok / 1 peer error / 2 usage (see the parser epilog):
EXIT_NOT_DELIVERED = 3  # positively never sent: the first connection was refused or unresolvable, so no byte went out
EXIT_POSSIBLY_ADMITTED = 4  # sent (fully, partly or unknown), its outcome unconfirmed: it may be admitted and running

_RESEND_NOTE = ("Resending the message is always allowed, immediately; a duplicate is fine. Reusing the "
                "idempotency key is optional (it lets the peer recognise the same run); it is not a receipt "
                "and not a dedup gate.")

_SEND_PHASE = threading.local()
_SEND_AUDIT_LOCK = threading.Lock()
_send_audit_installed = False


def _send_audit(event: str, _args: tuple) -> None:
    # The stdlib's own events: ``http.client.connect`` opens each hop's connection, ``http.client.send``
    # precedes every write of request bytes to its socket (headers and body, on every redirect hop).
    if event == "http.client.send" or event == "http.client.connect":
        phase = getattr(_SEND_PHASE, "current", None)
        if phase is not None:
            if event == "http.client.send":
                phase.sends += 1
            else:
                phase.connects += 1


class _RunSendPhase:
    """What the real transport did for ONE run POST on this thread, observed through http.client's audit
    events across every redirect hop. Nothing in the transport, redirect handling or exceptions changes.

    http.client raises ``http.client.send`` BEFORE each ``sendall``, and a failed ``sendall`` cannot say how
    much it wrote, so a seen send event means request bytes MAY have been sent: exit 4. Exit 3 (``never_sent``)
    needs the stock transport, the first hop's connection attempted (``http.client.connect``), no send event, and
    urllib reporting that connection refused or its host unresolvable. Any later hop (a redirect), an installed
    custom handler or an unobserved phase is 4: an absent event alone is no evidence.

    The audit hook is installed once per process behind a lock-guarded flag (``sys.addaudithook`` cannot be
    removed). It is O(1), cannot raise, and does nothing for other events or when no POST is tracked on the
    current thread; per-POST counts live in a thread-local, so concurrent and repeated calls never mix."""

    def __init__(self) -> None:
        self.connects = 0
        self.sends = 0
        self._outer = None
        # Unknown until ``use`` binds the opener this POST actually goes through (PEER-TRUTH-07): a sample of the
        # mutable global opener taken at another moment says nothing about the transport the request used.
        self.stock_transport = False

    def use(self, opener):
        """Record the transport of ``opener``, the exact opener the run POST is then sent through."""
        self.stock_transport = _stock_opener(opener)
        return opener

    def __enter__(self) -> _RunSendPhase:
        global _send_audit_installed
        with _SEND_AUDIT_LOCK:
            if not _send_audit_installed:
                sys.addaudithook(_send_audit)  # process-wide and permanent; inert outside a tracked POST
                _send_audit_installed = True
        self._outer = getattr(_SEND_PHASE, "current", None)
        _SEND_PHASE.current = self
        return self

    def __exit__(self, *_exc) -> bool:
        _SEND_PHASE.current = self._outer
        return False

    def never_sent(self, exc: BaseException) -> bool:
        """Positive evidence only, and only for the stock transport, whose every write is preceded by its send
        event and whose first-connection failure IS the error urllib raises: exactly one connection attempted,
        no send event, and that error a refused connection or an unresolvable host (no TCP connection, so no
        request byte left). An installed custom handler or processor may write without any event, or fail some
        other connection after writing (review19553 PEER-TRUTH-03), so with one the outcome stays unknown."""
        return (self.stock_transport and self.connects == 1 and self.sends == 0
                and _connection_never_established(exc))

    @property
    def redirected(self) -> bool:
        return self.connects > 1


def _connection_never_established(exc: BaseException) -> bool:
    """The stock transport's own report of its connection failing to open: urllib's URLError whose reason is a
    refused connection or an unresolvable host (no deeper cause chain is trusted)."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, (ConnectionRefusedError, socket.gaierror))


def _stock_opener(opener) -> bool:
    """True when every handler of ``opener`` (the credential-safe opener a request is sent through) runs only stock
    code, so every write is the stock transport's and is preceded by its send event: the director is exactly urllib's
    ``OpenerDirector`` and each handler is exactly one of urllib's own handler classes or Hermes' redirect/sanitizer
    policy classes (class identity, not a module name a subclass can claim) AND carries no per-instance behavior. A
    handler instance can replace what the director calls on it (an instance ``http_open``, as ``ProxyHandler``
    itself does), and such a callable survives the policy's ``copy.copy`` clone (review19670 PEER-TRUTH-09); a stock
    class proves nothing about it. Anything else: its writes are not observable here, so no zero-write proof."""
    if type(opener) is not urllib.request.OpenerDirector:
        return False
    return all(_stock_handler(handler) for handler in getattr(opener, "handlers", ()))


def _stock_handler(handler) -> bool:
    from hermes_cli import urllib_security
    cls = type(handler)
    if not (getattr(urllib.request, cls.__name__, None) is cls
            or cls in (urllib_security.SafeCredentialRedirectHandler, urllib_security._CrossOriginRequestSanitizer)):
        return False
    return not any(name != "parent" and (callable(value) or callable(getattr(cls, name, None)))
                   for name, value in vars(handler).items())


def _run_path(run_id: str) -> str:
    """A run handle as the ``/v1/runs/<id>`` path segment every follow-up request (``status``/``stop``) uses."""
    return urllib.parse.quote(run_id, safe="")


def _usable_run_id(run_id) -> bool:
    """A run handle the follow-up requests can use: a non-blank string that ``_run_path`` can encode (UTF-8). No
    other grammar is imposed (``run:7/a b`` or any Unicode is fine); an escaped lone surrogate is not (PEER-TRUTH-08)."""
    if not (isinstance(run_id, str) and run_id.strip()):
        return False
    try:
        _run_path(run_id.strip())
    except UnicodeEncodeError:
        return False
    return True


def _peer_key_env(name: str) -> str:
    return f"HERMES_PEER_{name.upper().replace('-', '_')}_KEY"


def _load_peers() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    peers = cfg.get("bot_peers")
    return peers if isinstance(peers, dict) else {}


def _save_peers(peers: dict) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config() or {}
    cfg["bot_peers"] = peers
    save_config(cfg)


def _peer_secret(name: str) -> str:
    """The peer's API key: profile-scoped secret store first, raw env fallback."""
    env_name = _peer_key_env(name)
    try:
        from agent.secret_scope import get_secret

        return (get_secret(env_name, "") or "").strip()
    except Exception:
        import os

        return (os.environ.get(env_name) or "").strip()


class _ReadPartial(NamedTuple):
    """Exact bytes exposed by one read exception; source names provenance, not body order."""

    source: str
    body: bytes


def _read_partials(error: Exception) -> tuple[_ReadPartial, ...]:
    """Preserve structured partials without guessing that causal bytes are disjoint or contiguous.

    In particular http.client's chunked reader wraps its completed prefix around a cause
    holding the incomplete current chunk (PEER-TRUTH-13). Other readers can overlap or
    repeat fragments: keep distinct observations, never concatenate/deduplicate by value.
    Walk exception identities once, including context, so cycles cannot strand reporting.
    """
    pending, seen, observations = [("read_error", error)], set(), []
    while pending:
        source, current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, http.client.IncompleteRead):
            observations.append(_ReadPartial(source + ".partial", current.partial))
        if current.__context__ is not None:
            pending.append((source + ".__context__", current.__context__))
        if current.__cause__ is not None:
            pending.append((source + ".__cause__", current.__cause__))
    return tuple(observations)


class _ResponseReadError(Exception):
    """An observed HTTP answer whose body could not be completely read (PEER-TRUTH-11).

    ``partial`` is exactly what the reader exposed, including b''; None means no bytes
    were exposed, not an empty peer body. This answer must never reach the run parser.
    ``observations`` also preserves nested exception partials without claiming that
    they belong to a single contiguous body or can safely be concatenated.
    """

    def __init__(self, http_status: int, error: Exception) -> None:
        self.http_status = http_status
        self.partial = error.partial if isinstance(error, http.client.IncompleteRead) else None
        self.observations = _read_partials(error)
        self.read_error = f"{type(error).__name__}: {error}"
        super().__init__(self.read_error)

    def supplemental_observations(self) -> list[dict] | None:
        """Only add the separate evidence field when the legacy outer partial cannot hold it all."""
        if len(self.observations) > 1 or (self.observations and self.partial is None):
            return [{"source": part.source, "body_hex": part.body.hex()} for part in self.observations]
        return None


def _read_response(response, http_status: int) -> bytes:
    """Return a complete body or preserve the observed status and available partial bytes."""
    try:
        return response.read()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise _ResponseReadError(http_status, exc) from exc


def _request(
    url: str, key: str, *, method: str = "GET", body: dict | None = None,
    timeout: int = LIST_TIMEOUT_S, headers: dict[str, str] | None = None, raw: bool = False, on_opener=None):
    """The parsed JSON object; with ``raw``, the complete answer's ``(HTTP status, body bytes)`` exactly as received
    (decoding, and judging whether they are text at all, is the caller's). ``on_opener(opener)`` sees the exact opener the request is then sent
    through, built once from the installed policy (the same opener ``open_credentialed_url`` would build)."""
    from hermes_cli.urllib_security import _secure_opener_from_installed_policy, open_credentialed_url
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "hermes-peer-dm"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    # The peer URL is user-registered (``hermes peer add``); a redirect to a
    # different origin must not carry the Authorization: Bearer key with it —
    # a compromised/MITM'd peer could otherwise harvest it. open_credentialed_url
    # strips non-safelisted headers across a cross-origin redirect.
    factory = None
    if on_opener is not None:
        opener = on_opener(_secure_opener_from_installed_policy(url))
        for name, value in getattr(opener, "_hermes_initial_addheaders", ()):
            if not req.has_header(name):
                req.add_header(name, value)
        factory = lambda _redirect_handler: opener  # the one opener this request is sent through
    try:
        with open_credentialed_url(req, timeout=timeout, opener_factory=factory) as resp:
            if raw:
                return resp.status, _read_response(resp, resp.status)
            payload = resp.read().decode("utf-8", "replace")
    except http.client.HTTPException as exc:
        # BadStatusLine / UnknownProtocol carry the peer's raw status line; uncaught, the
        # traceback prints it to the terminal unsanitized. RemoteDisconnected is also an
        # OSError: keep it on the callers' "could not reach" transport path.
        if isinstance(exc, OSError):
            raise
        raise RuntimeError(f"Peer sent a malformed HTTP response: {sanitize_display_text(str(exc)[:200])}") from exc
    try:
        parsed = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError(f"Peer returned non-JSON response: {sanitize_display_text(payload[:200])}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Peer returned a non-object JSON response")
    return parsed


def _base_url(peer: dict, profile: str | None) -> str:
    url = str(peer.get("url") or "").rstrip("/")
    if profile:
        # Multiplex mirror: same handlers, scoped to the named profile.
        return f"{url}/p/{urllib.parse.quote(profile, safe='')}"
    return url


def _peer_lacks_profile(exc: Exception) -> bool:
    """True only for the peer's own "no such profile" verdict: the 404 its /p/<profile>/ prefix
    middleware returns before any handler runs. A handler's 404 (unknown run or session) carries
    ``{"error": {"message": ...}}`` and a transport failure is no verdict at all. The verdict is its
    ``code``; only a body with no ``code`` (a peer older than it) is read by its fixed sentence."""
    if not isinstance(exc, urllib.error.HTTPError) or exc.code != 404:
        return False
    try:
        body = json.loads(exc.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return False
    if not isinstance(body, dict):
        return False
    if "code" in body:
        return body["code"] == _UNKNOWN_PROFILE_CODE
    return body.get("error") == _UNKNOWN_PROFILE_ERROR


def _resolve_base(peer: dict, profile: str | None, key: str, peer_name: str = "") -> str:
    """The mirror a target addresses. Bot Mode shows a peer's default profile as @hermes
    (#118076); an upgraded peer aliases ``/p/hermes`` to its default itself, unless it has a real
    pre-reservation profile named ``hermes``, which keeps the handle. The client never reroutes on
    its own: an older peer answers the same generic 404 for "no such profile" and for a real
    ``hermes`` profile it does not serve (parked, standalone), so guessing ``/p/default`` there
    delivers into the wrong agent. That ambiguous answer becomes an error naming the explicit
    ``<peer>/default`` address instead."""
    base = _base_url(peer, profile)
    if profile != DEFAULT_PROFILE_HANDLE:
        return base
    try:
        _request(f"{base}/health", key)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        if _peer_lacks_profile(exc):
            name = peer_name or "<peer>"
            raise LookupError(
                f"Peer '{name}' answered that it serves no '{DEFAULT_PROFILE_HANDLE}' profile. That answer "
                f"is ambiguous: an up-to-date gateway gives it when it does not serve its default profile, "
                f"and an older one also when it has a '{DEFAULT_PROFILE_HANDLE}' profile it does not serve. "
                f"If the peer serves its default profile, address that bot as {name}/default.") from exc
    return base


def _find_bot_chat(base: str, key: str) -> str | None:
    """The remote canonical Bot Chat's session id, or None.

    Bot Mode always HIDES canonical chats, so the plain listing (which
    excludes hidden sessions) misses an existing Bot Chat and the caller
    would try to create a duplicate that the peer's UNIQUE(title) guard
    rejects (issue #91583). Newer peers support an exact-title lookup with
    ``include_hidden=1``; older peers ignore the unknown query params and
    return the ordinary visible listing, so this single request degrades
    to exactly the previous behavior against them.
    """
    query = urllib.parse.urlencode({"limit": 200, "title": BOT_CHAT_TITLE, "include_hidden": 1})
    listing = _request(f"{base}/api/sessions?{query}", key)
    for session in listing.get("data") or []:
        if isinstance(session, dict) and (session.get("title") or "").strip() == BOT_CHAT_TITLE:
            return str(session.get("id") or "") or None
    return None


def _ensure_bot_chat(base: str, key: str) -> str:
    existing = _find_bot_chat(base, key)
    if existing:
        return existing
    try:
        created = _request(
            f"{base}/api/sessions", key, method="POST",
            body={"title": BOT_CHAT_TITLE, "source": "bot_peer_dm"})
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        if exc.code == 400 and "title" in detail.lower():
            # Older peer (no title/include_hidden lookup support): its
            # canonical Bot Chat exists but is hidden, so we couldn't see it
            # and the create collided with the UNIQUE(title) guard.
            raise RuntimeError(
                f"Peer already has a '{BOT_CHAT_TITLE}' session but it is hidden and the "
                f"peer's gateway is too old to expose hidden sessions to this lookup "
                f"(HTTP 400: {detail}). Update the peer's hermes-agent, or unhide the "
                f"session there: PATCH /api/sessions/<id> {{\"hidden\": false}}.") from exc
        raise
    # Real api_server wraps the row: {"object": "hermes.session", "session": {...}}.
    session = created.get("session") if isinstance(created.get("session"), dict) else created
    session_id = str(session.get("id") or session.get("session_id") or "")
    if not session_id:
        raise RuntimeError("Peer did not return a session id for the new Bot Chat")
    return session_id


def _parse_target(target: str) -> tuple[str, str | None]:
    """``<peer>`` or ``<peer>/<profile>`` → (peer, profile|None)."""
    raw = (target or "").strip()
    peer, _, profile = raw.partition("/")
    peer = peer.strip()
    profile = profile.strip() or None
    if not peer:
        raise ValueError("Peer name required (hermes peer dm <peer>[/<agent>] ...)")
    if profile and not _PROFILE_RE.match(profile):
        raise ValueError(f"Invalid agent/profile name: {profile!r}")
    if profile:
        # Profile ids are lowercase (normalize_profile_name); 'box/Hermes' must address the
        # peer's real 'hermes' profile on every peer, including those that compare prefixes exactly.
        from hermes_cli.profiles import normalize_profile_name
        profile = normalize_profile_name(profile)
    return peer, profile


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
        parsed = json.loads(body)
        message = parsed.get("error", {}).get("message") if isinstance(parsed, dict) else None
        return sanitize_display_text(str(message or body[:200]))
    except Exception:
        return sanitize_display_text(str(exc))


def _resolve_peer_target(target: str) -> tuple[str, str | None, dict, str]:
    """Resolve a registered target to ``(name, profile, config, key)``."""
    peer_name, profile = _parse_target(target)
    peer = _load_peers().get(peer_name)
    if not isinstance(peer, dict) or not peer.get("url"):
        raise LookupError(f"No peer named '{peer_name}'. Run: hermes peer list")
    key = _peer_secret(peer_name)
    if not key:
        raise PermissionError(
            f"No API key for peer '{peer_name}'. Set it: hermes peer add {peer_name} "
            f"--url <url> --key <key> (or add {_peer_key_env(peer_name)}=<key> to ~/.hermes/.env)")
    return peer_name, profile, peer, key


def _message_from_args(args) -> str:
    """The message (argument, else stdin), or "" when it is blank.

    With ``--exact`` a non-blank message is sent byte-for-byte; otherwise surrounding
    whitespace is stripped, as before.
    """
    message = getattr(args, "message", None) or ""
    if not message.strip() and not sys.stdin.isatty():
        message = _exact_stdin() if getattr(args, "exact", False) else sys.stdin.read()
    if not message.strip():
        return ""
    return message if getattr(args, "exact", False) else message.strip()


def _exact_stdin() -> str:
    """TR-3 (TR2-01): ``--exact`` stdin as raw bytes decoded as UTF-8 here, independent of the process's stdin text
    encoding. Invalid bytes become lone surrogates, which ``_exact_deliverable`` refuses (exit 2, nothing sent)."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # an already-decoded stream (embedded callers): its text is what it is
        return sys.stdin.read()
    return buffer.read().decode("utf-8", "surrogateescape")


def _exact_deliverable(message: str) -> bool:
    """TR-2 D1: whether ``--exact`` can carry ``message`` byte-for-byte (no NUL, encodable as UTF-8)."""
    if "\x00" in message:
        return False
    try:
        message.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _peer_run_durability(base: str, key: str) -> bool | None:
    """Return durable support, or None when an older peer cannot advertise it."""
    try:
        capabilities = _request(f"{base}/v1/capabilities", key)
    except Exception:
        return None
    features = capabilities.get("features")
    if not isinstance(features, dict):
        return None
    contract = features.get("runs_idempotency")
    if not isinstance(contract, dict) or not contract.get("supported"):
        return None
    return bool(contract.get("durable"))


def _peer_failure(peer_name: str, exc: Exception) -> int:
    """Print a peer HTTP rejection or transport failure to stderr; always exit 1."""
    if isinstance(exc, urllib.error.HTTPError):
        detail = _http_error_detail(exc)
        print(f"Peer '{peer_name}' rejected the request (HTTP {exc.code}): {detail}",
              file=sys.stderr)
    else:
        print(f"Could not reach peer '{peer_name}': {sanitize_display_text(str(exc))}", file=sys.stderr)
    return 1


def _run_not_confirmed(args, peer_name: str, profile: str | None, session_id: str, idempotency_key: str,
                       *, sent: bool, reason: str, error: str | None = None,
                       response: dict | None = None, response_hex: str | None = None,
                       response_body_observations: list[dict] | None = None,
                       response_read_error: str | None = None, write_started: bool = False,
                       unsent_why: str = "its connection was refused or its host unresolvable, so no request "
                                         "byte was written") -> int:
    """Report a run whose admission is not confirmed, truthfully, and keep its key.

    ``sent=False`` (exit 3, ``not_delivered``) only with positive evidence that no byte of the run request
    was ever written; otherwise exit 4, ``possibly_admitted``. ``response`` is the peer's actual answer
    (HTTP status and the available body, verbatim) when one arrived but could not confirm the run; when that body is
    not valid UTF-8 its rendering is not injective, so ``response_hex`` carries its exact bytes alongside. If the
    read failed, ``response_read_error`` is separate evidence, not fabricated peer content, and None body means
    unavailable bytes, not empty bytes. ``response_body_observations`` retains separately sourced nested partials
    as exact hex, not a reconstructed body. The text claims only what is known (PEER-TRUTH-04): an answer means the
    request was sent; a started write (its send
    event precedes ``sendall``) may have sent none, some or all of it; otherwise whether it was sent is unknown."""
    status = "possibly_admitted" if sent else "not_delivered"
    if sent:
        what = ("the run request was sent and answered" if response is not None else
                "a write of the run request was started (none, some or all of its bytes may have gone out)"
                if write_started else "it is unknown whether the run request was sent;")
        print(f"Peer '{peer_name}': {what} but its outcome is not confirmed ({reason}); it may already be admitted "
              f"and running. idempotency_key: {idempotency_key}. {_RESEND_NOTE}", file=sys.stderr)
    else:
        print(f"Could not reach peer '{peer_name}': {reason}. The run request was never sent: {unsent_why}. "
              f"idempotency_key: {idempotency_key}. {_RESEND_NOTE}", file=sys.stderr)
    if response is not None:
        body = response["body"]
        kind = ("body unavailable" if body is None else
                "incomplete body, available bytes only" if response_read_error else "body verbatim")
        print(f"Peer response (HTTP {response['http_status']}), {kind}:", file=sys.stderr)
        if body is not None:
            print(body, file=sys.stderr)
        if response_hex is not None:
            print(f"(not valid UTF-8; its exact bytes, hex: {response_hex})", file=sys.stderr)
        if response_read_error is not None:
            print(f"Response body read failed: {response_read_error}", file=sys.stderr)
    if response_body_observations is not None:
        print("Separate read-exception byte observations (not a reconstructed body):", file=sys.stderr)
        for observation in response_body_observations:
            print(f"{observation['source']}: hex {observation['body_hex']}", file=sys.stderr)
    if getattr(args, "json", False):
        payload = {"peer": peer_name, "profile": profile, "session_id": session_id or None, "run_id": None,
                   "status": status, "idempotency_key": idempotency_key, "error": error or reason}
        if response is not None:
            payload["response"] = response
        if response_hex is not None:
            payload["response_hex"] = response_hex
        if response_read_error is not None:
            payload["response_complete"] = False
            payload["response_read_error"] = response_read_error
        if response_body_observations is not None:
            payload["response_body_observations"] = response_body_observations
        print(json.dumps(payload))
    return EXIT_POSSIBLY_ADMITTED if sent else EXIT_NOT_DELIVERED


def _raw_response(http_status: int, raw: bytes | None) -> tuple[dict, str | None]:
    """An answer that could not confirm the run, for display and evidence: ``(response, response_hex)``. The
    response's ``body`` is a readable rendering (undecodable bytes as ``\\xNN``, which is not injective: a literal
    backslash text can render the same), so the exact received bytes are ALSO returned, hex-encoded, whenever the
    body is not valid UTF-8 (otherwise None). None is unavailable, distinct from b'' (valid empty)."""
    if raw is None:
        return {"http_status": http_status, "body": None}, None
    try:
        return {"http_status": http_status, "body": raw.decode("utf-8")}, None
    except UnicodeDecodeError:
        return {"http_status": http_status, "body": raw.decode("utf-8", "backslashreplace")}, raw.hex()


def _http_error_response(exc: urllib.error.HTTPError) -> tuple[dict, str | None, str | None, list[dict] | None]:
    """The same byte-preserving evidence boundary for HTTP errors and ordinary answers."""
    try:
        raw = _read_response(exc, exc.code)
    except _ResponseReadError as read_exc:
        response, response_hex = _raw_response(read_exc.http_status, read_exc.partial)
        return response, response_hex, read_exc.read_error, read_exc.supplemental_observations()
    response, response_hex = _raw_response(exc.code, raw)
    return response, response_hex, None, None


def _emit(args, payload: dict, text_lines: list[str]) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload))
    else:
        for line in text_lines:
            print(sanitize_display_text(line))
    return 0


def _peer_add(args) -> int:
    name = (args.name or "").strip().lower()
    if not _PEER_NAME_RE.match(name):
        print(f"Invalid peer name: {name!r} (lowercase, digits, -, _; max 64)", file=sys.stderr)
        return 2
    url = (args.url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        print("Peer --url must be an http(s) gateway base URL, e.g. http://spark.lan:8377", file=sys.stderr)
        return 2
    peers = _load_peers()
    peers[name] = {"url": url.rstrip("/"), **({"note": args.note.strip()} if getattr(args, "note", "") else {})}
    _save_peers(peers)
    key = (getattr(args, "key", "") or "").strip()
    if key:
        from hermes_cli.config import save_env_value

        save_env_value(_peer_key_env(name), key)
        print(f"Peer '{name}' saved ({url}) — key stored as {_peer_key_env(name)} in ~/.hermes/.env")
    else:
        print(
            f"Peer '{name}' saved ({url}). No key given — set the peer's API_SERVER_KEY with:\n"
            f"  hermes peer add {name} --url {url} --key <key>\n"
            f"  (or add {_peer_key_env(name)}=<key> to ~/.hermes/.env)")
    return 0


def _peer_remove(args) -> int:
    name = (args.name or "").strip().lower()
    peers = _load_peers()
    if name not in peers:
        print(f"No peer named '{name}'.", file=sys.stderr)
        return 1
    peers.pop(name)
    _save_peers(peers)
    print(f"Peer '{name}' removed (its {_peer_key_env(name)} entry in .env is kept; delete it manually if unused).")
    return 0


def _peer_list(args) -> int:
    peers = _load_peers()
    if not peers:
        print("No peers registered. Add one: hermes peer add <name> --url http://host:port --key <API_SERVER_KEY>")
        return 0
    for name in sorted(peers):
        entry = peers[name] if isinstance(peers[name], dict) else {}
        has_key = "key set" if _peer_secret(name) else f"NO KEY ({_peer_key_env(name)} unset)"
        note = f" — {entry.get('note')}" if entry.get("note") else ""
        print(f"{name}\t{entry.get('url', '?')}\t[{has_key}]{note}")
    return 0


def _peer_run_ctl(args, action: str, peer_name: str, profile: str | None, base: str,
                  key: str) -> int:
    """``status`` / ``stop`` on an asynchronous run."""
    run_id = (getattr(args, "run_id", None) or "").strip()
    if not run_id:
        print("Run ID required.", file=sys.stderr)
        return 2
    stop = action == "stop"
    try:
        result = _request(
            f"{base}/v1/runs/{_run_path(run_id)}" + ("/stop" if stop else ""),
            key, method="POST" if stop else "GET", body={} if stop else None)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        return _peer_failure(peer_name, exc)
    if getattr(args, "json", False):
        print(json.dumps({"peer": peer_name, "profile": profile, **result}))
        return 0
    print(f"{run_id}: {sanitize_display_text(str(result.get('status', 'unknown')))}")
    if not stop and result.get("output"):
        print(sanitize_display_text(str(result["output"])))
    elif not stop and result.get("error"):
        print(sanitize_display_text(str(result["error"])), file=sys.stderr)
    return 0


def _turn_body(message: str, *, message_key: str, **extra) -> dict:
    """Request body with the bound swarm session author, else dispatcher-supplied attribution."""
    from agent.turn_author import turn_author_from_env

    body = {message_key: message, **extra}
    from agent.swarm_capabilities import peer_session_author
    author = peer_session_author() or turn_author_from_env()
    if author is not None:
        body["author"] = author
    return body


def _peer_run(args, message: str, peer_name: str, profile: str | None, base: str, key: str) -> int:
    idempotency_key = (getattr(args, "idempotency_key", None) or f"peer-{uuid.uuid4().hex}").strip()
    if (not idempotency_key or len(idempotency_key) > 255
            or re.search(r"[\r\n\x00]", idempotency_key)):
        print("Idempotency key must be 1-255 characters without control newlines.", file=sys.stderr)
        return 2
    session_id = ""
    try:
        if _peer_run_durability(base, key) is not True:
            print(
                "Warning: this peer does not advertise restart-durable run replay; keep the run ID to follow "
                "this run across a gateway restart. Resending the message is always allowed.", file=sys.stderr)
        session_id = _ensure_bot_chat(base, key)
    except urllib.error.HTTPError as exc:
        return _peer_failure(peer_name, exc)
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        return _run_not_confirmed(
            args, peer_name, profile, session_id, idempotency_key, sent=False, reason=str(exc),
            error=f"{type(exc).__name__}: {exc}",
            unsent_why="a setup request before it failed, so the run request was not attempted")
    except RuntimeError as exc:
        return _peer_failure(peer_name, exc)
    # Kept BEFORE the run is sent, so a lost answer still leaves the caller this run's key.
    print(f"peer run: sending to '{peer_name}' with idempotency_key: {idempotency_key}", file=sys.stderr)
    # Delivery truth comes from what the real transport did (``_RunSendPhase``), never from an exception's
    # class: only a first-hop connection that failed before any request byte was written is "not delivered".
    phase = _RunSendPhase()
    try:
        with phase:
            http_status, raw = _request(
                f"{base}/v1/runs", key, method="POST",
                body=_turn_body(message, message_key="input", session_id=session_id),
                headers={"Idempotency-Key": idempotency_key}, raw=True, on_opener=phase.use)
    except urllib.error.HTTPError as exc:
        if phase.redirected or exc.code >= 500 or 100 <= exc.code < 200 or 300 <= exc.code < 400:
            # A later hop's answer is not the receiver's verdict on the original POST; a 5xx does not say whether
            # the run was admitted before it; and a redirect the client did not follow (307/308 for a POST, a
            # refused Location) answered the complete POST without rejecting it (PEER-TRUTH-05).
            # Stock urllib can surface 103 without consuming the final response. An
            # informational answer is not a rejection or a usable admission verdict.
            # Do not repeat the POST or invent the unobserved final response.
            where = ("after a redirect" if phase.redirected else
                     "an informational response, not a final admission verdict" if 100 <= exc.code < 200 else
                     "a server error" if exc.code >= 500
                     else "a redirect that was not followed")
            response, response_hex, read_error, observations = _http_error_response(exc)
            return _run_not_confirmed(
                args, peer_name, profile, session_id, idempotency_key, sent=True,
                reason=f"HTTP {exc.code} {where}", response=response,
                error=f"HTTP {exc.code}: {where}" if 100 <= exc.code < 200 else f"HTTPError: {exc}",
                response_hex=response_hex, response_read_error=read_error, response_body_observations=observations)
        return _peer_failure(peer_name, exc)
    except _ResponseReadError as exc:
        response, response_hex = _raw_response(exc.http_status, exc.partial)
        return _run_not_confirmed(
            args, peer_name, profile, session_id, idempotency_key, sent=True,
            reason=f"HTTP {exc.http_status} response body could not be completely read", error=exc.read_error,
            response=response, response_hex=response_hex, response_read_error=exc.read_error,
            response_body_observations=exc.supplemental_observations())
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        sent = not phase.never_sent(exc)
        if sent:
            reason = (f"{exc} after a redirect" if phase.redirected
                      else f"{exc} after a write of the request was started" if phase.sends
                      else f"{exc}; the transport's send phase was not observed")
        else:
            reason = str(exc)
        return _run_not_confirmed(args, peer_name, profile, session_id, idempotency_key, sent=sent,
                                  reason=reason, error=f"{type(exc).__name__}: {exc}", write_started=phase.sends > 0)
    try:
        text = raw.decode("utf-8")  # strict: a display form of undecodable bytes is never parser input
    except UnicodeDecodeError:
        text = None
    try:
        result = json.loads(text) if text is not None else None
        parsed = text is not None
    except ValueError:
        result, parsed = None, False
    run_id = result.get("run_id") if isinstance(result, dict) else None
    if not _usable_run_id(run_id):
        # The peer answered but did not bind a usable run handle: that is not "not admitted", and no handle is
        # made up from another type (PEER-TRUTH-06). Show its answer as it came.
        shape = ("its body is not valid UTF-8 text (PEER-TRUTH-10)" if text is None else
                 "its body is not JSON" if not parsed else
                 "its body is not a JSON object" if not isinstance(result, dict) else
                 "it carries no run_id" if result.get("run_id") in (None, "") else
                 "its run_id is not a non-empty string" if not (isinstance(run_id, str) and run_id.strip()) else
                 "its run_id cannot be used in a run request (not encodable as UTF-8)")
        response, response_hex = _raw_response(http_status, raw)
        return _run_not_confirmed(
            args, peer_name, profile, session_id, idempotency_key, sent=True,
            reason=f"the peer answered HTTP {http_status} but {shape}", response=response, response_hex=response_hex)
    payload = {
        "peer": peer_name, "profile": profile, "session_id": session_id, "run_id": run_id,
        "status": result.get("status") or "started", "idempotency_key": idempotency_key,
        "replayed": bool(result.get("replayed", False))}
    replay = " (replayed)" if payload["replayed"] else ""
    return _emit(args, payload, [
        f"{run_id}: {payload['status']}{replay}", f"session_id: {session_id}",
        f"idempotency_key: {idempotency_key}"])


def _peer_dm(args, message: str, peer_name: str, profile: str | None, base: str, key: str) -> int:
    session_id = ""
    try:
        session_id = _ensure_bot_chat(base, key)
        result = _request(
            f"{base}/api/sessions/{urllib.parse.quote(session_id, safe='')}/chat", key,
            method="POST", body=_turn_body(message, message_key="message"), timeout=DM_TIMEOUT_S)
    except RuntimeError as exc:
        print(f"Peer '{peer_name}': {sanitize_display_text(str(exc))}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Resolving a session does not prove admission of the following POST.
        # A lost response can leave delivery uncertain; retrying a message is allowed.
        if session_id and isinstance(exc, TimeoutError):
            print(f"No reply from peer '{peer_name}' within {DM_TIMEOUT_S}s "
                  f"(session {sanitize_display_text(str(session_id))}); delivery may have been accepted. "
                  "Retry promptly if the reply is still needed; duplicate messages do not authorize repeated effects.",
                  file=sys.stderr)
            return 1
        return _peer_failure(peer_name, exc)
    if result.get("object") == "hermes.session.chat.queued":
        # The peer's Bot Chat is open in its Desktop and that turn outlasted the peer's wait: the
        # message is queued in that chat; keep its delivery handle for follow-up.
        queued_in = result.get("session_id") or session_id
        return _emit(args, {"peer": peer_name, "profile": profile, "session_id": queued_in,
                            "status": result.get("status") or "queued", "delivery_id": result.get("delivery_id")},
                     [f"Peer '{peer_name}' has its Bot Chat open, so the message went into that chat (session "
                      f"{queued_in}) and is answered there. The reply cannot come back on this call. Retry promptly if delivery fails, times out or remains uncertain; duplicate messages do not authorize repeated effects."])
    msg = result.get("message")
    reply = str(msg.get("content") or "") if isinstance(msg, dict) else ""
    # A successful bare silence marker is a delivery decision, not a message:
    # the turn stays in the peer's own transcript, the sending agent never
    # sees NO_REPLY/[SILENT] as a real reply. Same rule as the gateway's live
    # Bot Chat completion, the Desktop bot_relay.deliver RPC and the one-shot
    # local `hermes chat -Q` transport (tools/bot_mode_dm.py) — this is the
    # 4th Bot Mode delivery door and was missing the same check.
    from gateway.response_filters import is_intentional_silence_response, strip_trailing_silence_marker
    reply = "" if is_intentional_silence_response(reply) else strip_trailing_silence_marker(reply)
    payload = {"peer": peer_name, "profile": profile,
               "session_id": result.get("session_id") or session_id, "reply": reply}
    return _emit(args, payload, [reply or "(no reply)"])


_REGISTRY_ACTIONS = {
    "add": _peer_add, "set": _peer_add, "remove": _peer_remove, "rm": _peer_remove,
    "list": _peer_list, "ls": _peer_list, None: _peer_list}


def cmd_peer(args) -> int:
    action = getattr(args, "peer_action", None)
    if action in _REGISTRY_ACTIONS:
        return _REGISTRY_ACTIONS[action](args)
    if action not in {"dm", "run", "status", "stop"}:
        print("Unknown peer action. See: hermes peer --help", file=sys.stderr)
        return 2
    try:
        peer_name, profile, peer, key = _resolve_peer_target(args.target)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (LookupError, PermissionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        base = _resolve_base(peer, profile, key, peer_name)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if action in {"status", "stop"}:
        return _peer_run_ctl(args, action, peer_name, profile, base, key)
    message = _message_from_args(args)
    if not message:
        print("Message required (argument or stdin).", file=sys.stderr)
        return 2
    if getattr(args, "exact", False) and not _exact_deliverable(message):
        print("--exact: the message contains NUL or is not valid UTF-8, so it cannot be sent byte-for-byte; "
              "nothing was sent.", file=sys.stderr)
        return 2
    handler = _peer_run if action == "run" else _peer_dm
    return handler(args, message, peer_name, profile, base, key)


def build_peer_parser(subparsers) -> None:
    """Attach the ``peer`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "peer", help="Bot-to-bot DMs across machines (peer Hermes gateways)",
        description="Register other Hermes gateways as peers and message their agents. "
            "'hermes peer dm <peer>[/<agent>] \"...\"' delivers into the remote "
            "agent's canonical Bot Chat over the peer's API server and prints "
            "the reply — the cross-machine twin of 'hermes -p <bot> chat'. "
            "The peer must run the api_server platform; its API_SERVER_KEY is "
            "stored locally as a credential in ~/.hermes/.env.",
        epilog=(
            "Examples:\n"
            "  hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>\n"
            "  hermes peer list\n"
            '  hermes peer dm spark "Message from 🤖 dixie (@dixie): disk status?"\n'
            '  hermes peer dm spark/researcher "..."   # named profile on a multiplexed peer\n'
            "  hermes peer run spark --idempotency-key ticket-123 < long-task.txt\n"
            "  hermes peer status spark run_abc123\n"
            "  hermes peer stop spark run_abc123\n"
            "  hermes peer remove spark\n"
            "\n"
            "Exit codes: 0 ok, 1 delivery/peer error, 2 usage error.\n"
            "'peer run' also: 3 not delivered (positively never sent: the first connection was refused or its\n"
            "host unresolvable, or a setup request before it failed), 4 possibly admitted (request bytes may\n"
            "have been sent and the outcome is unconfirmed: a lost or unusable answer, an unfollowed or failing\n"
            "redirect, an HTTP 5xx, an unobserved send; the run may be running). Both print the idempotency key, which\n"
            "'peer run' also prints before sending. Resending the message is always allowed, immediately; a\n"
            "duplicate is fine. Reusing the key is optional; it is not a receipt and not a dedup gate."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    peer_sub = parser.add_subparsers(dest="peer_action")

    add_p = peer_sub.add_parser("add", aliases=["set"], help="Register (or update) a peer gateway")
    add_p.add_argument("name", help="Peer name (lowercase slug, e.g. spark, homelab)")
    add_p.add_argument("--url", required=True, help="Peer gateway base URL, e.g. http://spark.lan:8377")
    add_p.add_argument("--key", default="", help="The peer's API_SERVER_KEY (stored in ~/.hermes/.env)")
    add_p.add_argument("--note", default="", help="Optional description")

    peer_sub.add_parser("list", aliases=["ls"], help="List registered peers")

    rm_p = peer_sub.add_parser("remove", aliases=["rm"], help="Remove a peer")
    rm_p.add_argument("name", help="Peer name")

    def _remote(name: str, help: str, *, run_id: bool):
        sp = peer_sub.add_parser(name, help=help)
        sp.add_argument("target", help="<peer> or <peer>/<agent> (named profile on a multiplexed peer)")
        if run_id:
            sp.add_argument("run_id", help="Run ID returned by 'hermes peer run'")
        else:
            sp.add_argument("message", nargs="?", default=None, help="Message text (or stdin)")
            sp.add_argument(
                "--exact", action="store_true", default=False,
                help="Send the message byte-for-byte, without stripping surrounding whitespace")
            if name == "run":
                sp.add_argument(
                    "--idempotency-key", default=None, help="Stable retry key (generated when omitted)")
        sp.add_argument("--json", action="store_true", default=False, help="Emit a JSON result")

    _remote("dm", "Message an agent on a peer gateway and print its reply", run_id=False)
    _remote("run", "Start a long peer turn asynchronously and return its run ID", run_id=False)
    _remote("status", "Read the status and final output of an asynchronous peer run", run_id=True)
    _remote("stop", "Stop one asynchronous peer run without affecting another turn", run_id=True)

    parser.set_defaults(func=cmd_peer)
