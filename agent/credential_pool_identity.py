"""Account identity read from credential token claims, shared by the pool and its display names.

Claims are decoded without signature verification: these helpers label and compare credentials
Hermes already holds; they never decide whether a token is valid.
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_cli.auth import _decode_jwt_claims


def label_from_token(token: str, fallback: str) -> str:
    claims = _decode_jwt_claims(token)
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _codex_principal_identity(access_token: Any) -> Optional[tuple[str, str]]:
    """``(chatgpt_account_id, sub)`` of a Codex access token, or None when either claim is missing.

    Decoded without signature verification: this only decides whether two credentials Hermes
    already holds belong to the same principal, never whether a token is valid. Both claims are
    required because members of one ChatGPT workspace share ``chatgpt_account_id`` yet have their
    own subjects and quotas.
    """
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    subject = claims.get("sub") if isinstance(claims, dict) else None
    if not (isinstance(account_id, str) and account_id.strip() and isinstance(subject, str) and subject.strip()):
        return None
    return account_id.strip(), subject.strip()
