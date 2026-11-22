"""Operator display names for pooled credentials, kept outside the credential store.

``auth.json`` row labels are written back from every process's in-memory pool snapshot, so a
label edited on disk is reverted by the next stale flush. Display names therefore live in a
separate, secret-free file that no Hermes code writes, beside the ``auth.json`` that owns the
rows (the profile's own store, or the global root when the profile borrows root's rows)::

    <owning store>/credential_display_names.json
    {"version": 1,
     "providers": {"openai-codex": {"<row id>": {"display_name": "user@example.com",
                                                "expected_account_id": "<chatgpt_account_id>",
                                                "expected_subject": "<sub>"}}}}

A display name is an operator assertion. It is shown only when the row's access-token principal
(``chatgpt_account_id`` and ``sub`` claims, decoded locally without signature checks or network)
MATCHES both ``expected_account_id`` and ``expected_subject``. The subject is required because
members of one ChatGPT workspace share ``chatgpt_account_id``; an account-only match could put
one member's name on another member's row. The match is a consistency check against the
mapping, not provider authentication and not proof of the email. Identifiers are opaque and
compared exactly: nothing is trimmed into a match, and an expected identifier with surrounding
whitespace is rejected as malformed. Any other outcome keeps the stored label and says why. Rows are matched by stable row id only, never by position or label.
Only ``openai-codex`` has a principal to match today; names for other providers are rejected
rather than shown unmatched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DISPLAY_NAMES_FILENAME = "credential_display_names.json"

_CLAIM_MATCH_PROVIDERS = frozenset({"openai-codex"})


@dataclass(frozen=True)
class DisplayNameSpec:
    """One parsed mapping row; ``problem`` is set when a required field is missing."""

    display_name: str
    expected_account_id: str
    expected_subject: str
    problem: Optional[str] = None


@dataclass(frozen=True)
class DisplayNames:
    path: Path
    providers: dict[str, dict[str, DisplayNameSpec]]
    error: Optional[str] = None  # file present but unusable: names are NOT applied


# Per-path cache: (file signature, parsed). One process can serve several profiles.
_cache: dict[str, tuple[tuple[int, int], DisplayNames]] = {}


def owning_store_dir(provider: str) -> Path:
    """Directory of the ``auth.json`` that owns *provider*'s pool rows for the current scope."""
    from hermes_cli.auth import _global_auth_file_path
    from hermes_constants import get_hermes_home
    global_auth = _global_auth_file_path()
    if global_auth is None:  # classic mode: the active home is the root
        return get_hermes_home()
    from agent.credential_pool import _profile_owns_pool_provider
    return get_hermes_home() if _profile_owns_pool_provider(provider) else global_auth.parent


def _identifier_problem(field: str, value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return f"mapping has no {field}"
    if value != value.strip():
        return f"mapping {field} has surrounding whitespace"
    return None


def _parse_spec(spec: Any) -> DisplayNameSpec:
    if not isinstance(spec, dict):
        return DisplayNameSpec("", "", "", "mapping entry is not an object")
    name, account, subject = (spec.get(k) for k in ("display_name", "expected_account_id", "expected_subject"))
    name = name.strip() if isinstance(name, str) else ""
    problem = ("mapping has no display_name" if not name else
               _identifier_problem("expected_account_id", account) or
               _identifier_problem("expected_subject", subject))
    if problem:
        return DisplayNameSpec(name, "", "", problem)
    return DisplayNameSpec(name, account, subject)


def _raw_principal(access_token: Any) -> Optional[tuple[str, str]]:
    """Unmodified ``(chatgpt_account_id, sub)`` claims, or None when either is missing or blank.

    Deliberately not ``_codex_principal_identity``: that helper trims claims for credential
    selection, and trimming here would turn unequal identifiers into a display-name match.
    """
    from hermes_cli.auth import _decode_jwt_claims
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth")
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    subject = claims.get("sub")
    if not (isinstance(account_id, str) and account_id.strip() and isinstance(subject, str) and subject.strip()):
        return None
    return account_id, subject


def _parse(raw: Any) -> tuple[dict[str, dict[str, DisplayNameSpec]], Optional[str]]:
    providers = raw.get("providers") if isinstance(raw, dict) and raw.get("version") == 1 else None
    if not isinstance(providers, dict):
        return {}, "not a version-1 display-name mapping"
    parsed: dict[str, dict[str, DisplayNameSpec]] = {}
    for provider, rows in providers.items():
        if not isinstance(rows, dict):
            return {}, f"provider {provider!r} is not an object"
        parsed[str(provider).strip().lower()] = {str(row_id): _parse_spec(spec) for row_id, spec in rows.items()}
    return parsed, None


def load_display_names(provider: str) -> DisplayNames:
    """Mapping for *provider*'s owning store. Never raises; an unusable file sets ``error``."""
    path = owning_store_dir(provider) / DISPLAY_NAMES_FILENAME
    key = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        _cache.pop(key, None)
        return DisplayNames(path, {})
    except OSError as exc:
        return DisplayNames(path, {}, f"cannot stat ({exc})")
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        providers, error = _parse(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        providers, error = {}, f"cannot read ({exc.__class__.__name__})"
    if error:
        logger.warning("credential display names: %s ignored: %s", path, error)
    names = DisplayNames(path, providers, error)
    _cache[key] = (signature, names)
    return names


def _rejection(provider: str, spec: DisplayNameSpec, access_token: Any) -> Optional[str]:
    """Why a mapped name must not be shown for this row, or None when the full principal matches."""
    if spec.problem:
        return spec.problem
    if provider not in _CLAIM_MATCH_PROVIDERS:
        return "no account claim to match for this provider"
    identity = _raw_principal(access_token)
    if identity is None:
        return "row lacks account or subject claim"
    account_id, subject = identity
    if account_id != spec.expected_account_id:
        return "account claim mismatch"
    if subject != spec.expected_subject:
        return "subject claim mismatch"
    return None


def display_name(entry: Any, fallback: Optional[str] = None) -> str:
    """Name to show for a pool entry: the principal-matched display name, else the caller's old text.

    *fallback* is what the caller printed before display names existed (default: ``label``,
    or the short id when unlabeled), so unmapped rows render exactly as before. A mapped row
    whose check fails keeps that text plus a bracketed reason.
    """
    if fallback is None:
        fallback = getattr(entry, "label", "") or str(getattr(entry, "id", ""))[:8]
    provider = str(getattr(entry, "provider", "") or "").strip().lower()
    try:
        names = load_display_names(provider)
    except Exception:  # cosmetic: a scope lookup failure must never break a log line or listing
        logger.debug("credential display names: lookup failed for %s", provider, exc_info=True)
        return fallback
    spec = names.providers.get(provider, {}).get(str(getattr(entry, "id", "")))
    if spec is None:
        return fallback
    reason = _rejection(provider, spec, getattr(entry, "access_token", None))
    if reason is None:
        return spec.display_name
    return f"{fallback} [display name rejected: {reason}]"


def listing_notes(provider: str, row_ids: Iterable[str]) -> list[str]:
    """Lines for ``hermes auth list``: an unusable mapping file, and mapped ids not in the pool."""
    names = load_display_names(provider)
    if names.error:
        return [f"display names NOT applied: {names.path} ignored ({names.error})"]
    present = {str(r) for r in row_ids}
    return [f"display name for unknown {provider} row id={row_id} ignored"
            for row_id in sorted(names.providers.get(provider.strip().lower(), {})) if row_id not in present]
