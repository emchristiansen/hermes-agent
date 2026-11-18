"""``hermes auth principals``: read-only principal projection for explicitly selected Codex pool rows.

Grounds operator display-name mappings (``credential_display_names.json``) in the principal each
row actually holds, without exposing credentials. Rows come from the strict mode of the shared store
reader (``hermes_cli.auth_store_files``): the same profile-then-global-root view the pool loads,
without seeding, healing, refreshing, locking or writing. Each requested row id yields its stable id
plus the exact raw ``chatgpt_account_id`` and ``sub`` claims of its stored access token, decoded
locally without signature checks or network. That is a claim about stored bytes, not issuer
verification or proof the token is still valid. Missing, blank, whitespace-padded or ambiguous
identity is reported as unresolved with a reason, never guessed from labels, order or another
provider. A malformed, unreadable or unobservable store is an error (exit 1) naming the store that
failed, never an empty pool or a silent fall back to the global root. Tokens, all other claims and
raw store signatures are never printed.

The command promises no home initialization or writes, so ``hermes_cli.main`` runs it through
``main()`` before its config/logging imports (which scaffold the home); this module and the reader
it uses never import ``hermes_cli.config``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from hermes_cli import auth_store_files as store_files
from hermes_cli.auth_constants import AuthStoreUnreadable, DuplicateJsonMember, _decode_jwt_claims
from hermes_constants import get_hermes_home

PROVIDER = "openai-codex"
HELP = "Show the exact account/subject claims of selected openai-codex pool rows (read-only)"
_AUTH_CLAIMS = "https://api.openai.com/auth"
_PROFILE_CLAIMS = "https://api.openai.com/profile"
PROVENANCE = ("local unsigned decode of the stored access token; not issuer-verified, "
              "not a freshness or validity check")
SNAPSHOT = "store files compared before and after one read; not linearizable"

Signature = Optional[tuple[int, ...]]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The command's arguments, shared by the full ``hermes auth`` parser and the pre-import ``main``."""
    parser.add_argument("provider", help="Provider id (openai-codex only)")
    parser.add_argument(
        "--row", action="append", metavar="ID", help="Pool row id to project (repeatable; required)")


def _observe(path: Optional[Path]) -> Signature:
    """Change detector for one store file (never printed); None when it does not exist. Any other
    failure to observe it raises ``OSError``: an unobservable file is not an absent one."""
    if path is None:
        return None
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _identifier(field: str, value: Any) -> tuple[Optional[str], Optional[str]]:
    """(raw value, problem). Values are opaque: returned exactly, never trimmed."""
    if not isinstance(value, str) or not value.strip():
        return None, f"access token has no {field} claim"
    if value != value.strip():
        return value, f"{field} claim has surrounding whitespace"
    return value, None


def _email(claims: dict[str, Any]) -> dict[str, Any]:
    profile = claims.get(_PROFILE_CLAIMS)
    email = profile.get("email") if isinstance(profile, dict) else None
    if isinstance(email, str) and email.strip():
        return {"email": email, "email_source": f"access token claim {_PROFILE_CLAIMS}.email ({PROVENANCE})"}
    return {"email": None, "email_source": "not stored: no email claim in the stored access token"}


def _project_row(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"row_id": row_id, "resolved": False}
    matches = [row for row in rows if row.get("id") == row_id]
    if not matches:
        return {**result, "reason": "row id not in the pool"}
    if len(matches) > 1:
        return {**result, "reason": f"ambiguous: {len(matches)} rows share this id"}
    row = matches[0]
    try:
        claims = _decode_jwt_claims(row.get("access_token"), unique_members=True)
    except DuplicateJsonMember:
        return {**result, "reason": "ambiguous: the stored access token's claims repeat a member name"}
    if not claims:
        return {**result, "reason": "stored access token is missing or not a readable JWT"}
    auth_claims = claims.get(_AUTH_CLAIMS)
    account_id, account_problem = _identifier(
        "chatgpt_account_id",
        auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None)
    subject, subject_problem = _identifier("sub", claims.get("sub"))
    last_refresh = row.get("last_refresh")
    result.update({
        "account_id": account_id, "subject": subject, **_email(claims),
        "identity_source": PROVENANCE,
        "last_refresh": last_refresh if isinstance(last_refresh, str) else None,
    })
    problem = account_problem or subject_problem
    if problem:
        return {**result, "reason": problem}
    return {**result, "resolved": True}


def project_principals(row_ids: list[str]) -> tuple[dict[str, Any], bool]:
    """(projection, ok) for *row_ids* in the active scope's owning store. Never writes or refreshes.

    Both store files are observed before and after one strict read of the rows and the
    profile/root fallback decision; any change marks every row unresolved. That detects interleaved
    writes but is not a linearizable snapshot. On failure, ``store``/``path`` name the store that
    could not be observed or read ("unknown" when that cannot be established), and
    ``store_present``/``changed_during_read`` are null when they were not observed."""
    home = get_hermes_home()
    active_path = store_files.auth_file_path(home)
    global_path = store_files.global_auth_file_path(home)
    observed_from = _now()
    before: Optional[tuple[Signature, Signature]] = None
    after: Optional[tuple[Signature, Signature]] = None
    rows: list[Any] = []
    source = "active"
    failure: Optional[str] = None
    failed_path: Optional[Path] = None
    try:
        before = (_observe(active_path), _observe(global_path))
        rows, source = store_files.select_pool_rows(
            PROVIDER, lambda: store_files.load_auth_store_file(active_path, strict=True),
            lambda: store_files.load_global_store_strict(global_path))
        after = (_observe(active_path), _observe(global_path))
    except AuthStoreUnreadable as exc:
        failure, failed_path = f"auth store unreadable: {exc}", exc.path
    except OSError as exc:
        failed_path = Path(exc.filename) if isinstance(exc.filename, str) else None
        failure = (f"auth store could not be observed: {failed_path} ({exc.__class__.__name__})"
                   if failed_path is not None else
                   f"auth store could not be observed ({exc.__class__.__name__}; store unknown)")
    if failure is not None and before is not None and after is None:
        try:  # report an interleaved write honestly even when the read itself failed
            after = (_observe(active_path), _observe(global_path))
        except OSError:
            after = None
    changed = None if before is None or after is None else before != after

    def describe(path: Optional[Path], *, borrowed: bool) -> tuple[str, Optional[bool]]:
        index = 0 if path == active_path else 1 if path is not None and path == global_path else None
        if index is None:
            return "unknown", None
        present = None if before is None else before[index] is not None
        if index == 1:
            return ("global root (borrowed by this profile)" if borrowed else "global root"), present
        return ("profile" if global_path is not None else "active home (classic mode)"), present

    if failure is None:
        owner = global_path if source == "global" and global_path is not None else active_path
        store, present = describe(owner, borrowed=True)
    else:
        owner = failed_path
        store, present = describe(owner, borrowed=False)
    result: dict[str, Any] = {
        "provider": PROVIDER, "observed_from": observed_from, "observed_to": _now(),
        "store": store, "path": None if owner is None else str(owner), "store_present": present,
        "changed_during_read": changed, "snapshot": SNAPSHOT,
    }
    if failure is not None or changed:
        reason = failure or "auth store changed during the read; re-run"
        result["error"] = reason
        result["rows"] = [{"row_id": r, "resolved": False, "reason": reason} for r in row_ids]
        return result, False
    result["rows"] = [_project_row(rows, r) for r in row_ids]
    return result, True


def run(provider: Optional[str], row_ids: Optional[list[str]]) -> int:
    """Print the projection for the parsed arguments; exit status 0, or 1 on a store error/race."""
    if (provider or "").strip().lower() != PROVIDER:
        raise SystemExit(f"`hermes auth principals` supports only {PROVIDER} (got {provider!r}).")
    selected = list(dict.fromkeys(row_ids or []))
    if not selected:
        raise SystemExit("`hermes auth principals` needs at least one --row ID; rows are never enumerated.")
    result, ok = project_principals(selected)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ok else 1


def auth_principals_command(args) -> None:
    """``hermes auth`` dispatch entry (the full CLI path, e.g. inside container mode)."""
    if code := run(getattr(args, "provider", ""), getattr(args, "row", None)):
        raise SystemExit(code)


def main(argv: list[str]) -> int:
    """Pre-import entry: ``hermes auth principals ARGS`` before ``hermes_cli.main``'s home setup."""
    parser = argparse.ArgumentParser(prog="hermes auth principals", description=HELP)
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args.provider, args.row)
