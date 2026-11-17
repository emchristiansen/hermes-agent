"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process; each profile's
``.env`` keys **cannot** be unioned into ``os.environ`` (profile A's keys would
leak into profile B's turns and subprocesses). This module is a fail-closed,
context-local secret scope: ``set_secret_scope(mapping)`` installs the active
profile's secrets for the current task (a contextvar, so it propagates into the
agent's worker thread via ``copy_context()``); ``get_secret(name)`` reads from
it and, when multiplexing is active with no scope set, RAISES rather than
falling back to ``os.environ``. Design: ``website/docs/developer-guide/multiplexing-gateway.md``.
"""
from __future__ import annotations

import codecs
import os
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Callable, Dict, Mapping, NamedTuple, Optional, Tuple


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False
# Launch home pinned by set_multiplex_active(True) itself (None: no auto-pin outstanding).
_AUTO_PINNED_HOME = None

# Context-local counterpart: a task serving a profile OTHER than the process's own, inside a
# process that is not a multiplexer as a whole — the desktop backend's cron ticker firing a
# sibling profile's job. Every isolation keyed on ``is_multiplex_active()`` (the routed-dotenv
# guard, ``get_secret``'s fail-closed miss, subprocess scrubbing, passthrough) applies inside
# it while the process's own turns keep single-profile semantics. A contextvar, so it reaches
# the pool worker together with the home override via ``copy_context()``.
_MULTIPLEX_CONTEXT: ContextVar[bool] = ContextVar("_MULTIPLEX_CONTEXT", default=False)


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed).

    Activation also pins the launch home for routed-profile decisions
    (``hermes_constants.pin_process_hermes_home``) unless an embedding host already pinned one:
    from here on "is this task routed" compares the override against the home the process was
    launched with, not against whatever a host later mirrors into ``os.environ["HERMES_HOME"]``.
    Deactivation releases only the pin activation itself created — a transient toggle
    (``gateway_migrate._multiplex_read_mode``, a cron worker restoring the caller's mode) must not
    drop the host's explicit pin (#119242)."""
    global _MULTIPLEX_ACTIVE, _AUTO_PINNED_HOME
    from hermes_constants import (
        get_routing_process_hermes_home,
        pin_process_hermes_home,
        process_hermes_home_is_pinned,
    )
    _MULTIPLEX_ACTIVE = bool(active)
    if _MULTIPLEX_ACTIVE:
        if not process_hermes_home_is_pinned():
            _AUTO_PINNED_HOME = get_routing_process_hermes_home()
            pin_process_hermes_home(_AUTO_PINNED_HOME)
    elif _AUTO_PINNED_HOME is not None:
        if get_routing_process_hermes_home() == _AUTO_PINNED_HOME:
            pin_process_hermes_home(None)
        _AUTO_PINNED_HOME = None


def set_multiplex_context(active: bool) -> Token:
    """Run the current task under multiplex semantics regardless of the process flag.
    Returns a reset token; pair with :func:`reset_multiplex_context` in a ``finally``."""
    return _MULTIPLEX_CONTEXT.set(bool(active))


def reset_multiplex_context(token: Token) -> None:
    _MULTIPLEX_CONTEXT.reset(token)


def is_multiplex_active() -> bool:
    """True in a multiplexing process, or for a task running under multiplex semantics."""
    return _MULTIPLEX_ACTIVE or _MULTIPLEX_CONTEXT.get()


class _BoundScope(NamedTuple):
    """An installed secret scope plus the home it was built for, when the binder
    declared one — the provenance ``serves_routed_profile`` needs when the binding
    deliberately skips the HERMES_HOME override (kanban spawn-env builds, MCP
    owner scopes)."""

    mapping: Mapping[str, str]
    profile_home: Optional[str]


def serves_routed_profile() -> bool:
    """True when the current task runs for a profile other than the process's own: always under
    multiplexing, else when a HERMES_HOME override names another home (dashboard/desktop backend,
    per-profile cron ticker) or a secret scope stamped with a foreign home is bound. The MCP
    registry scope and the check_fn cache key both follow this predicate so a served profile's
    view never aliases the launch profile's (#111151). A host that mirrors the turn's profile into
    ``HERMES_HOME`` pins its own home with ``hermes_constants.pin_process_hermes_home`` so the
    mirror cannot flip this predicate."""
    if is_multiplex_active():
        return True
    from hermes_constants import get_hermes_home_override, get_routing_process_hermes_home, hermes_home_key
    own = hermes_home_key(get_routing_process_hermes_home())
    bound = _SECRET_SCOPE.get()
    if bound is not None and bound.profile_home and hermes_home_key(bound.profile_home) != own:
        return True
    override = get_hermes_home_override()
    return override is not None and hermes_home_key(override) != own


_SECRET_SCOPE: ContextVar[Optional[_BoundScope]] = ContextVar("_SECRET_SCOPE", default=None)
# A launch-profile body that entered before this process started multiplexing (no scope bound,
# single-profile semantics): the launch profile's own mapping to read once fail-closed is armed
# mid-body. Set only by ``tui_gateway.launch_profile_policy.launch_body``; never ``os.environ``.
_LAUNCH_BODY_SCOPE: ContextVar[Optional[Callable[[], Mapping[str, str]]]] = ContextVar(
    "_LAUNCH_BODY_SCOPE", default=None)


def set_launch_body_scope(resolver: Callable[[], Mapping[str, str]]) -> Token:
    return _LAUNCH_BODY_SCOPE.set(resolver)


def reset_launch_body_scope(token: Token) -> None:
    _LAUNCH_BODY_SCOPE.reset(token)


class UnscopedSecretError(RuntimeError):
    """A secret was read in multiplex mode with no scope installed.

    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the global allowlist.

    ``str(exc)`` is the ONE sentence an end user can act on; the developer diagnosis
    (which secret, which doc) rides ``__notes__`` so tracebacks and logs keep it.
    """

    def __init__(self, secret_name: str = "", developer_detail: str = ""):
        # Older callers passed the whole developer sentence positionally
        # (``UnscopedSecretError("get_secret('X') called with no scope ...")``); a secret
        # name never contains whitespace, so treat such a string as the detail.
        if secret_name and not developer_detail and any(ch.isspace() for ch in secret_name):
            secret_name, developer_detail = "", secret_name
        what = f"this profile's {secret_name}" if secret_name else "this profile's API key"
        super().__init__(
            f"Hermes could not read {what} (an internal profile-scoping bug on the multiplexed "
            "gateway, not your configuration). Run `hermes gateway restart`; if it keeps happening, "
            "report it with `hermes debug share`."
        )
        self.secret_name = secret_name
        self.developer_detail = developer_detail
        if developer_detail:
            self.add_note(developer_detail)


def set_secret_scope(secrets: Optional[Mapping[str, str]], *, profile_home: Optional[str] = None) -> Token:
    """Install the active profile's secret mapping; ``None`` clears. Returns a reset token.

    ``profile_home`` stamps the home the mapping was built for so
    ``serves_routed_profile`` detects a foreign-home scope even when the binder
    deliberately skips the HERMES_HOME override."""
    if secrets is None:
        return _SECRET_SCOPE.set(None)
    return _SECRET_SCOPE.set(_BoundScope(secrets, str(profile_home) if profile_home else None))


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """The active secret mapping, or None when no scope is installed."""
    bound = _SECRET_SCOPE.get()
    return bound.mapping if bound is not None else None


def current_secret_scope_home() -> Optional[str]:
    """The home the active scope was stamped with, or None when unstamped/unbound."""
    bound = _SECRET_SCOPE.get()
    return bound.profile_home if bound is not None else None


# Genuinely-global env vars: process/deployment settings, NOT profile secrets.
# They keep reading os.environ even in multiplex mode (routing them through the
# fail-closed path would wrongly crash). Keep this tight — when in doubt a
# value is a profile secret. Membership is exact name OR prefix.
_GLOBAL_ENV_EXACT = frozenset({
    # Hermes runtime / deployment
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_GATEWAY_LOCK_DIR",
    "HERMES_MAX_ITERATIONS", "HERMES_API_TIMEOUT",
    "HERMES_REDACT_SECRETS", "HERMES_NOUS_TIMEOUT_SECONDS",
    "_HERMES_GATEWAY",
    # OS / interpreter
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "PWD", "SHELL", "TMPDIR",
    "VIRTUAL_ENV", "PYTHONPATH", "SSL_CERT_FILE",
    # Kanban paths (per-board, not per-profile-secret)
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD",
    # API-server LISTENER settings — deployment config (compose/systemd env),
    # which the scoped runner reload must keep seeing or containers silently
    # lose the api_server platform. API_SERVER_KEY is a credential: NOT here.
    # See #64674, #69379.
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
    # Relay-connector ROUTING stamps injected by managed deploys. Every reader
    # (gateway.config, relay_url()/registration/self-provision) must resolve
    # the SAME value or the adapter registers while the platform is absent
    # from config. GATEWAY_RELAY_SECRET/_ID/_DELIVERY_KEY and IDP_* are auth
    # material and deliberately stay profile-scoped.
    "GATEWAY_RELAY_URL", "GATEWAY_RELAY_ENDPOINT",
    "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS",
    "GATEWAY_RELAY_PLATFORMS", "GATEWAY_RELAY_BOT_IDS",
    "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
    "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_DISPLAY_NAME",
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # tuning knobs (batch delays, fallback toggles) — NOT the token
    "TERMINAL_",          # terminal/sandbox backend settings
)


def _is_global_env(name: str) -> bool:
    """True for genuinely process-global (non-profile-secret) env vars."""
    return name in _GLOBAL_ENV_EXACT or name.startswith(_GLOBAL_ENV_PREFIXES)


def _environ_or(name: str, default: Optional[str]) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    Global vars always read ``os.environ``. With a scope installed, a miss returns
    ``default`` under multiplexing (never another profile's ``os.environ`` value)
    but falls through to ``os.environ`` otherwise — single-profile deployments
    inject credentials via the process env (systemd, ``op run``), so the scope
    must stay a ``.env`` overlay, not a blindfold (otherwise cron 401s). With no
    scope: multiplex INACTIVE reads ``os.environ``; ACTIVE raises (fail closed).
    """
    if _is_global_env(name):
        return _environ_or(name, default)
    bound = _SECRET_SCOPE.get()
    if bound is not None:
        val = bound.mapping.get(name)
        if val is not None:
            return val
        return default if serves_routed_profile() else _environ_or(name, default)
    if is_multiplex_active():
        launch_scope = _LAUNCH_BODY_SCOPE.get()
        if launch_scope is not None:
            return launch_scope().get(name, default)
        raise UnscopedSecretError(
            name,
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See website/docs/developer-guide/multiplexing-gateway.md "
            f"(Workstream A).",
        )
    return _environ_or(name, default)


def secret_source_label(name: str) -> str:
    """``" (from Bitwarden)"``-style suffix naming the external secret source (a secret-source plugin)
    that supplied env var *name*; ``""`` for ``.env`` / shell values. Public for plugins' status lines."""
    from hermes_cli.env_loader import format_secret_source_suffix

    return format_secret_source_suffix(name)


def get_secret_str(name: str, default: str = "") -> str:
    """``get_secret`` for callers that want a ``str``: ``default`` only when the secret is genuinely
    unset. Still raises ``UnscopedSecretError`` — swallowing it hides a spawn-site bug."""
    val = get_secret(name, default)
    return default if val is None else val


def _strip_inline_comment(value: str) -> str:
    """Strip a dotenv-style inline comment (python-dotenv semantics): quoted values
    scan to the matching close quote (backslash-aware for double quotes) and drop a
    trailing ``# ...``, else stay untouched; unquoted values truncate only at a
    ``#`` PRECEDED BY WHITESPACE (``foo#bar`` survives, ``value # c`` → ``value``)."""
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2  # skip the escaped character
                continue
            if ch == quote:
                return value[: i + 1] if value[i + 1:].lstrip().startswith("#") else value
            i += 1
        return value  # unterminated quote: leave as-is
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _parse_env_value(raw_value: str) -> str:
    """Parse the small .env value subset Hermes writes itself (bare, 'single', or "double" with
    ``\\"`` / ``\\\\`` escapes)."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted = value[1:-1]
        parsed: list[str] = []
        i = 0
        while i < len(quoted):
            escaped = quoted[i] == "\\" and quoted[i + 1:i + 2] in ('"', "\\")
            parsed.append(quoted[i + 1] if escaped else quoted[i])
            i += 2 if escaped else 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


# Per-path memo of parsed ``.env`` files. ``build_profile_secret_scope()`` runs on every gateway
# turn, cron fire, MCP/browser adoption and housekeeping drain, and each call used to re-read and
# re-parse the whole file.
#
# FRESHNESS: every call still OPENS and READS the file (a few hundred bytes) and keys on its exact
# bytes; only the decode + parse is skipped. A stat identity is not enough: a kernel that stamps
# mtime/ctime from the coarse (jiffy) clock leaves (mtime, size, inode, ctime) unchanged across an
# in-place same-size rewrite inside one tick, and ``/reload`` then served the old value (CI run
# 39146191939). The open keeps close-to-open revalidation on NFS, and a vanished/unreadable file
# fails the open and is never cached (a transient EACCES must not become "this profile has no
# secrets"). ``invalidate_env_file_cache()`` is the explicit knob; ``hermes_cli.config.
# invalidate_env_cache()`` calls it for Hermes's own .env writers.
_ENV_FILE_CACHE: OrderedDict[str, tuple[bytes, dict[str, str]]] = OrderedDict()
_ENV_FILE_CACHE_LOCK = threading.Lock()
_ENV_FILE_CACHE_MAX = 64  # one entry per profile home in practice


def invalidate_env_file_cache(env_path: Optional[Path] = None) -> None:
    """Drop one path from the ``load_env_file()`` memo, or all of them."""
    with _ENV_FILE_CACHE_LOCK:
        if env_path is None:
            _ENV_FILE_CACHE.clear()
        else:
            _ENV_FILE_CACHE.pop(str(env_path), None)


def _decode_env_bytes(raw: bytes) -> str:
    """BOM stripped; invalid UTF-8 falls back to latin-1 exactly as
    ``env_loader._load_dotenv_with_fallback`` installs it into ``os.environ``."""
    raw = raw.removeprefix(codecs.BOM_UTF8)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_env_text(text: str) -> dict[str, str]:
    """Tokenize already-read ``.env`` text. See :func:`load_env_file`."""
    secrets: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            secrets[key] = _parse_env_value(_strip_inline_comment(value))
    return secrets


def load_env_file(env_path: Path) -> dict[str, str]:
    """THE ``.env`` tokenizer: every reader (profile scope, ``hermes_cli.config.load_env``, the dashboard
    scrub, skill secret capture, managed .env, setup prompts) parses through here so no two boundaries
    disagree on which keys/values a file defines. Dict only — never touches ``os.environ``. ``export``
    prefix, ``#`` comments, quote escapes reversed; a BOM is stripped so it doesn't prefix the first key.
    Invalid UTF-8 decodes as latin-1, exactly like ``env_loader._load_dotenv_with_fallback`` installs it
    into ``os.environ``. Absent/unreadable → ``{}``.

    Memoised per path on the file's exact bytes (see the cache comment above). Always
    returns a fresh dict: callers mutate what they get back (``build_profile_secret_scope`` layers
    external secrets over it).
    """
    try:
        return read_env_file(env_path)
    except OSError:
        return {}


def read_env_file(env_path: Path, *, dir_fd: Optional[int] = None) -> dict[str, str]:
    """:func:`load_env_file` for a caller that must tell "absent" from "unreadable": the ``OSError``
    of the one open/read propagates (``FileNotFoundError`` when there is no file) instead of
    collapsing to ``{}``, so a fail-closed reader needs no separate exists()/read probe.

    *dir_fd*: an open descriptor of ``env_path.parent``; the file is opened by name relative to
    that directory, so its bytes come from the directory the caller already identified even if a
    path component is retargeted meanwhile (see :func:`_read_profile_home`)."""
    return _read_env_file(env_path, dir_fd)[0]


def _read_env_file(env_path: Path, dir_fd: Optional[int] = None, *,
                   want_stat: bool = False) -> tuple[dict[str, str], Optional[os.stat_result]]:
    """:func:`read_env_file`, plus (*want_stat*) the ``fstat`` of the file the bytes were read from,
    so a caller can name the exact file that supplied them (:func:`_path_read_profile_home`)."""
    key = str(env_path)
    opener = None if dir_fd is None else (lambda name, flags: os.open(name, flags, dir_fd=dir_fd))
    try:
        with open(env_path.name if dir_fd is not None else env_path, "rb", buffering=0, opener=opener) as handle:
            raw = handle.read()
            file_stat = os.fstat(handle.fileno()) if want_stat else None
    except OSError:
        # Gone or unreadable: drop any entry so a stale map cannot outlive the file.
        invalidate_env_file_cache(env_path)
        raise

    with _ENV_FILE_CACHE_LOCK:
        cached = _ENV_FILE_CACHE.get(key)
        if cached is not None and cached[0] == raw:
            _ENV_FILE_CACHE.move_to_end(key)
            return dict(cached[1]), file_stat
    secrets = _parse_env_text(_decode_env_bytes(raw))
    with _ENV_FILE_CACHE_LOCK:
        _ENV_FILE_CACHE[key] = (raw, dict(secrets))
        _ENV_FILE_CACHE.move_to_end(key)
        while len(_ENV_FILE_CACHE) > _ENV_FILE_CACHE_MAX:
            _ENV_FILE_CACHE.popitem(last=False)
    return secrets, file_stat


class _ProfileHomeRead(NamedTuple):
    """A profile home's ``.env`` mapping and its launch-home verdict, taken from ONE directory."""

    env: dict[str, str]
    is_process_home: bool


# Directory descriptors + ``dir_fd`` opens (POSIX). Elsewhere (Windows) the home is resolved afresh.
_HOME_DIR_FD = (getattr(os, "O_DIRECTORY", 0) != 0 and os.open in os.supports_dir_fd)
# O_PATH (Linux) needs only search permission on the home, like reading ``<home>/.env`` by path.
_HOME_DIR_FLAGS = (getattr(os, "O_PATH", 0) or os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0) | getattr(
    os, "O_CLOEXEC", 0)


def _same_directory(home_stat: os.stat_result) -> bool:
    """Is the directory *home_stat* describes the launch home? Identity is the inode, so the
    answer is whatever the launch path names NOW (one stat, no per-component walk)."""
    from hermes_constants import get_routing_process_hermes_home
    try:
        launch = os.stat(get_routing_process_hermes_home())
    except OSError:
        return False
    return (home_stat.st_dev, home_stat.st_ino) == (launch.st_dev, launch.st_ino)


def _is_launch_env_file(env_stat: Optional[os.stat_result]) -> bool:
    """Is *env_stat* (the ``fstat`` of a ``.env`` just read, or None when none could be read) the
    launch home's own ``.env`` as its path names it NOW? None matches only a launch home with no
    ``.env`` at all; anything that cannot be confirmed answers False."""
    from hermes_constants import get_routing_process_hermes_home
    try:
        launch = os.stat(get_routing_process_hermes_home() / ".env")
    except FileNotFoundError:
        return env_stat is None
    except OSError:
        return False
    return env_stat is not None and (env_stat.st_dev, env_stat.st_ino) == (launch.st_dev, launch.st_ino)


def _path_read_profile_home(home: Path) -> _ProfileHomeRead:
    """:func:`_read_profile_home` where no directory descriptor is available (no ``dir_fd`` support,
    or a home that refuses one): ``.env`` is read by path, so no opened directory names its source.

    The launch-home verdict is therefore granted only when it is bound to the bytes actually read:
    the home resolves to the launch home both BEFORE and AFTER the read, AND the file read is the
    launch home's own ``.env`` (its ``fstat`` identity equals a fresh stat of ``<launch>/.env``; when
    no file could be read, the launch home must have none either). Anything else fails closed: the
    files are returned without launch authority. A path resolved only after the read (92f9d132's
    order inverted) would pair a file read from one home with whatever the path names next."""
    env_path = home / ".env"
    if not _resolved_is_process_home(home):
        return _ProfileHomeRead(load_env_file(env_path), False)
    try:
        env, env_stat = _read_env_file(env_path, want_stat=True)
    except OSError:
        env, env_stat = {}, None
    return _ProfileHomeRead(env, _resolved_is_process_home(home) and _is_launch_env_file(env_stat))


def _read_profile_home(hermes_home: Path) -> _ProfileHomeRead:
    """Read ``<home>/.env`` and decide "is this the launch home?" from the SAME opened directory.

    The verdict grants launch-only authority — the launch env's credentials
    (``owner_secret_scope``) and the bridged ``allow_all_users`` grant (below) — while the mapping
    is the profile's data. Taking both from one directory descriptor makes the mix that a path
    re-resolved (or a cached identity) could produce unrepresentable: a home path retargeted
    between the two reads, or after an identity was remembered, cannot pair one home's files with
    another home's authority. A home that cannot be opened has no ``.env`` to read: a launch home
    that does not exist yet keeps its env-only credentials as before, but only while its path still
    names no directory (a retarget landing after the failed open cannot attach them). Without a
    descriptor (no ``dir_fd`` support, or a home that refuses one) see :func:`_path_read_profile_home`."""
    home = Path(hermes_home)
    if not _HOME_DIR_FD:
        return _path_read_profile_home(home)
    try:
        fd = os.open(home, _HOME_DIR_FLAGS)
    except PermissionError:
        # Searchable but unreadable home on a platform without O_PATH: read it by path.
        return _path_read_profile_home(home)
    except OSError:
        return _ProfileHomeRead({}, _resolved_is_process_home(home) and not home.is_dir())
    try:
        home_stat = os.fstat(fd)
        try:
            env = read_env_file(home / ".env", dir_fd=fd)
        except OSError:
            env = {}
    finally:
        os.close(fd)
    return _ProfileHomeRead(env, _same_directory(home_stat))


class ProfileSecretScope(dict):
    """:func:`build_profile_secret_scope`'s mapping, carrying the launch-home verdict its files were
    read under (:func:`_read_profile_home`: one opened directory). Launch-only authority (the launch
    env's credentials, ``owner_secret_scope``) is added on THIS verdict and no other, so it cannot
    come from a different identity check than the bytes. Any other mapping (a copy, a merged dict, a
    test double) carries no verdict and gets no launch authority: see :func:`read_as_launch_home`."""

    __slots__ = ("read_as_launch_home",)

    def __init__(self, secrets: Mapping[str, str], read_as_launch_home: bool):
        super().__init__(secrets)
        self.read_as_launch_home = read_as_launch_home


def read_as_launch_home(scope: Mapping[str, str]) -> bool:
    """Were *scope*'s files read from the launch home (as :func:`build_profile_secret_scope` saw it)?
    False for any mapping that does not carry that producer's verdict."""
    return isinstance(scope, ProfileSecretScope) and scope.read_as_launch_home is True


def build_profile_secret_scope(hermes_home: Path) -> ProfileSecretScope:
    """Build a profile's secret mapping from ``<home>/.env`` plus its external
    secret sources. Global vars are NOT copied in — ``get_secret`` reads those
    from ``os.environ`` — so the scope holds only profile secrets.

    The result carries the launch-home verdict its ``.env`` was read under
    (:class:`ProfileSecretScope`); the bridged ``allow_all_users`` grant below and the caller's
    launch-env layer both use that verdict, never a separate identity check."""
    home_read = _read_profile_home(Path(hermes_home))
    secrets = home_read.env
    try:
        from hermes_cli.env_loader import get_secret_source_values
        external_secrets = get_secret_source_values(Path(hermes_home))
    except Exception:
        external_secrets = {}
    secrets.update((k, v) for k, v in external_secrets.items() if not _is_global_env(k))
    # The DEFAULT profile's config.yaml allow_all_users grant lives only in os.environ (bridged by
    # gateway.config_loader); scoped gate readers under multiplex never fall to os.environ, so seed it
    # into that profile's own mapping. A secondary never inherits it (#80099 class).
    from gateway.config_loader import bridged_allow_all_users
    bridged = bridged_allow_all_users()
    if bridged is not None and home_read.is_process_home:
        secrets.setdefault("GATEWAY_ALLOW_ALL_USERS", bridged)
    # Administrator-managed ``.env`` LAST, with override: the launch process applies it that way
    # (``env_loader._apply_managed_env``) so policy beats a user's own value. Under multiplex
    # semantics ``get_secret`` never reads ``os.environ`` on a scope miss, so a scope built from
    # the profile files alone would drop a managed-only credential and let the user's value win a
    # managed-vs-user collision (#111187 review). Every multiplex-authoritative scope — gateway
    # turn, routed cron fire, external worker — is built here, so managed authority is composed
    # once, not restored by each consumer.
    from hermes_cli.managed_scope import load_managed_env  # fail-open: {} when no managed scope

    secrets.update((k, v) for k, v in load_managed_env().items() if not _is_global_env(k))
    return ProfileSecretScope(secrets, home_read.is_process_home)


def _resolved_is_process_home(hermes_home: Path) -> bool:
    from hermes_constants import get_routing_process_hermes_home
    try:
        return Path(hermes_home).resolve() == get_routing_process_hermes_home().resolve()
    except OSError:
        return False


def _is_process_home(hermes_home: Path) -> bool:
    """Is *hermes_home* the profile this process serves as its own, as the filesystem names it NOW?
    The launch-home identity of ``serves_routed_profile()`` (a host that mirrors a served profile into
    ``HERMES_HOME`` would otherwise seed the launch profile's bridged allow-all grant into that
    profile's scope), decided afresh: never a remembered resolution, which a retargeted alias would
    turn into another home's authority. Directory identity (two stats) where the home exists; the
    fresh path comparison where it does not. A caller that also reads the home's files takes the
    verdict from that read instead (:func:`read_as_launch_home` on :func:`build_profile_secret_scope`'s
    result), so both come from one directory."""
    if _HOME_DIR_FD:
        try:
            return _same_directory(os.stat(hermes_home))
        except OSError:
            pass
    return _resolved_is_process_home(Path(hermes_home))


def refresh_installed_secret_scope(hermes_home: Path) -> bool:
    """Fold a fresh build of *hermes_home*'s secrets into the INSTALLED scope, in place.

    A scope is frozen when installed, but a fire can learn of new values afterwards: a routed cron
    fire's first agent build discovers plugin secret sources, and under multiplex semantics the
    reload that follows is hydrate-only (never ``os.environ``), so nothing else would carry those
    values into the scope this fire already holds. The caller names the home the installed scope
    was built for. True when a scope was updated; False when none is installed."""
    bound = _SECRET_SCOPE.get()
    scope = bound.mapping if bound is not None else None
    if not isinstance(scope, dict):
        return False
    # REPLACE, don't merge: the rebuild is the profile's current truth, so a name a source has
    # stopped supplying (rotated, revoked, source removed) must disappear from the fire's scope
    # rather than survive as the stale value dict.update() would keep.
    rebuilt = build_profile_secret_scope(hermes_home)
    # Update first, then drop what is gone: a concurrent reader never sees an emptied scope.
    scope.update(rebuilt)
    for name in [n for n in scope if n not in rebuilt]:
        scope.pop(name, None)
    return True
