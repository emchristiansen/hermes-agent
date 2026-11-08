"""Auth-store file paths and parsing: THE reader of ``auth.json``, importable before the CLI's home setup.

``hermes_cli.config`` scaffolds the active home when it is imported (provider discovery loads
config.yaml through ``ensure_hermes_home``), and ``hermes_cli.auth`` imports it. A read-only command
that promises no initialization or writes (``hermes auth principals``) therefore cannot import
``hermes_cli.auth``. Both use this module: ``hermes_cli.auth`` delegates its store paths, parsing
and profile-then-root row rule here, and adds the runtime's memo and write paths on top.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.auth_constants import AUTH_STORE_VERSION, AuthStoreUnreadable, DuplicateJsonMember, unique_json_members

# The store's messages keep the logger they have always used.
logger = logging.getLogger("hermes_cli.auth")


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return left == right


def auth_file_path(home: Path) -> Path:
    path = home / "auth.json"
    # Seat belt: under pytest, refuse to touch the real user's auth store (tests that forgot to
    # monkeypatch HERMES_HOME or escaped the hermetic conftest). In production: one dict lookup.
    if (os.environ.get("PYTEST_CURRENT_TEST")
            and same_path(path, Path.home() / ".hermes" / "auth.json")):  # health: allow HX001 -- the seat belt names the real user's store
        raise RuntimeError(
            f"Refusing to touch real user auth store during test run: {path}. "
            "Set HERMES_HOME to a tmp_path in your test fixture, or run "
            "via scripts/run_tests.sh for hermetic CI-parity env.")
    return path


def global_auth_file_path(home: Path) -> Optional[Path]:
    """Global-root auth.json in profile mode; None when *home* and the global root are the same dir.

    Read-only fallback path, so no pytest seat belt here (it lives on ``auth_file_path()``)."""
    try:
        from hermes_constants import get_default_hermes_root
        global_root = get_default_hermes_root()
    except Exception:  # health: allow BLE001 -- moved verbatim from auth._global_auth_file_path: no root, no fallback
        return None
    return None if same_path(home, global_root) else global_root / "auth.json"


def is_real_root_under_pytest(global_path: Path) -> bool:
    """Seat belt for global-root reads: under pytest the real user's root store reads as absent."""
    if not (os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("HOME")):
        return False
    real_root = Path(os.environ["HOME"]) / ".hermes" / "auth.json"
    try:
        return os.path.normcase(os.path.abspath(global_path)) == os.path.normcase(os.path.abspath(real_root))
    except Exception:  # health: allow BLE001 -- moved verbatim from auth.py's seat belt: uncomparable is not the root
        return False


def empty_auth_store() -> dict[str, Any]:
    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _from_legacy_systems(systems: dict[str, Any]) -> dict[str, Any]:
    providers = {"nous": systems["nous_portal"]} if "nous_portal" in systems else {}
    return {**empty_auth_store(), "providers": providers,
            "active_provider": "nous" if providers else None}


def load_auth_store_file(auth_file: Path, *, strict: bool = False) -> dict[str, Any]:
    """Parse one ``auth.json``; a missing file is an empty store.

    Default (every runtime reader): an unreadable file raises ``OSError``; unparseable bytes are
    preserved as ``.json.corrupt`` and read as empty; an unrecognised layout reads as empty.
    ``strict`` (read-only projections) never writes or degrades: see ``_load_strict``."""
    if strict:
        return _load_strict(auth_file)
    if not auth_file.exists():
        return empty_auth_store()
    try:
        raw = json.loads(auth_file.read_text(encoding="utf-8-sig"))
    except OSError:
        # Exists but unreadable (EMFILE, EACCES, EIO, stalled mount): contents are not bad, and this
        # module read-modify-writes everywhere, so an empty store here is one _save_auth_store()
        # away from erasing every credential. Fail loudly.
        logger.warning(
            "auth: could not read %s, leaving the store on disk untouched "
            "rather than degrading to an empty one",
            auth_file, exc_info=True)
        raise
    except Exception as exc:  # health: allow BLE001 -- moved verbatim from auth._load_auth_store (legacy degrade)
        # Genuine corruption: unparseable JSON or non-UTF-8 bytes. Preserve a copy, but never
        # advertise a backup that was not written.
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        try:
            shutil.copy2(auth_file, corrupt_path)
            preserved = True
        except Exception:
            preserved = False
            logger.debug("auth: could not preserve a copy of the corrupt store at %s", corrupt_path,
                         exc_info=True)
        logger.warning(
            "auth: failed to parse %s (%s), starting with empty store. %s %s",
            auth_file, exc,
            "Corrupt file preserved at" if preserved else "A copy could NOT be preserved at",
            corrupt_path)
        return empty_auth_store()

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict) or isinstance(raw.get("credential_pool"), dict)):
        raw.setdefault("providers", {})
        if isinstance(raw.get("providers"), dict):
            from hermes_cli.auth_nous import _migrate_stale_nous_portal_url
            _migrate_stale_nous_portal_url(raw["providers"])
        return raw

    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):  # legacy "systems" format
        return _from_legacy_systems(raw["systems"])
    return empty_auth_store()


def _load_strict(auth_file: Path) -> dict[str, Any]:
    """Raise ``AuthStoreUnreadable`` (naming *auth_file* and the failure kind, never its contents)
    instead of degrading: the file cannot be read or parsed; any object in it repeats a member name
    (which value counts would be a parser accident: a second empty pool erasing the first, a second
    token replacing the first); it has no recognised layout; its
    ``providers`` or ``credential_pool`` is present but not an object; a pool slice is not a list of
    objects with string ids; or normalising its provider entries fails. Only a file that does not
    exist is an empty store, so a malformed store can never pass for one with no rows."""
    try:
        data = auth_file.read_bytes()
    except FileNotFoundError:
        return empty_auth_store()
    except OSError as exc:
        raise AuthStoreUnreadable(auth_file, f"could not be read ({exc.__class__.__name__})") from None
    try:
        raw = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_json_members)
    except DuplicateJsonMember:
        raise AuthStoreUnreadable(auth_file, "is ambiguous (an object repeats a member name)") from None
    except (ValueError, RecursionError) as exc:  # bad UTF-8 or JSON (both ValueError), or nesting too deep
        raise AuthStoreUnreadable(auth_file, f"is not parseable ({exc.__class__.__name__})") from None
    if not isinstance(raw, dict):
        raise AuthStoreUnreadable(auth_file, "has no recognised auth-store layout")
    for key in ("providers", "credential_pool"):
        if key in raw and not isinstance(raw[key], dict):
            raise AuthStoreUnreadable(auth_file, f"has a malformed {key} (not an object)")
    if "providers" not in raw and "credential_pool" not in raw:
        if isinstance(raw.get("systems"), dict):
            return _from_legacy_systems(raw["systems"])
        raise AuthStoreUnreadable(auth_file, "has no recognised auth-store layout")
    for rows in raw.get("credential_pool", {}).values():
        if not isinstance(rows, list):
            raise AuthStoreUnreadable(auth_file, "has a malformed credential_pool (a provider's rows are not a list)")
        if not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows):
            raise AuthStoreUnreadable(
                auth_file, "has a malformed credential_pool (a row is not an object with a string id)")
    raw.setdefault("providers", {})
    try:
        from hermes_cli.auth_nous import _migrate_stale_nous_portal_url
        _migrate_stale_nous_portal_url(raw["providers"], log=False)  # never echo unrelated stored values
    except Exception as exc:  # health: allow BLE001 -- strict boundary (PP-05): any normalisation failure is this store's error
        raise AuthStoreUnreadable(
            auth_file, f"has a provider entry that cannot be normalised ({exc.__class__.__name__})") from None
    return raw


def load_global_store_strict(global_path: Optional[Path]) -> dict[str, Any]:
    """Strict global-root read: no memo (it may hold the empty store a corrupt file degraded to)."""
    if global_path is None or is_real_root_under_pytest(global_path):
        return {}
    return load_auth_store_file(global_path, strict=True)


def _provider_rows(store: dict[str, Any], provider_id: str) -> list[Any]:
    pool = store.get("credential_pool")
    entries = pool.get(provider_id) if isinstance(pool, dict) else None
    return list(entries) if isinstance(entries, list) else []


def select_pool_rows(provider_id: str, load_active: Callable[[], dict[str, Any]],
                     load_global: Callable[[], dict[str, Any]]) -> tuple[list[Any], str]:
    """THE profile-then-root rule: the active store's rows for *provider_id* (``"active"``), or, only
    when it has none, the global root's (``"global"``; ``hermes auth add`` in a profile shadows root).
    The global store is loaded only when needed."""
    entries = _provider_rows(load_active(), provider_id)
    if entries:
        return entries, "active"
    return _provider_rows(load_global(), provider_id), "global"
