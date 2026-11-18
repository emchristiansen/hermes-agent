"""``hermes auth principals``: read-only exact principal projection for selected Codex pool rows.

Driven through the real ``hermes auth`` parser and dispatch on synthetic stores. Network, pool
loading/seeding, refresh and every auth-store write path fail the test if entered, and store
directories must be byte-identical afterwards (no ``.corrupt`` copies either). The fresh-process
tests at the end run the real ``hermes_cli.main`` entry point and compare whole synthetic homes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ACCT_A = "a9631cb8-8258-4371-a753-7567e79aa882"
ACCT_B = "bfe407e3-3c31-4ca5-bf99-19fc0db1c601"
SIG = "SYNTHETIC-SIGNATURE"
INCIDENTAL = "incidental-plan-value"


def _jwt(account_id=ACCT_A, subject="user-1", email="eric@foglinesupply.com") -> str:
    claims: dict = {"exp": 1, "plan_type": INCIDENTAL}
    if account_id is not None:
        claims["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id, "plan": INCIDENTAL}
    if subject is not None:
        claims["sub"] = subject
    if email is not None:
        claims["https://api.openai.com/profile"] = {"email": email, "email_verified": True}
    def enc(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{enc({'alg': 'RS256'})}.{enc(claims)}.{SIG}"


def _row(row_id: str, label: str, token, priority: int = 0) -> dict:
    return {"id": row_id, "label": label, "auth_type": "oauth", "priority": priority,
            "source": "manual:device_code", "access_token": token, "refresh_token": f"rt-secret-{row_id}",
            "base_url": "https://chatgpt.com/backend-api/codex", "last_refresh": "2026-11-01T00:00:00Z"}


def _store(rows: list[dict]) -> str:
    return json.dumps({"version": 1, "providers": {}, "credential_pool": {"openai-codex": rows}})


def _tree(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.fixture
def guarded(monkeypatch):
    """Fail on network, pool loading, refresh or any auth-store write."""
    import agent.credential_pool as pool_mod
    import hermes_cli.auth as auth_mod
    from hermes_cli import auth_store_files

    def forbidden(*_a, **_k):
        raise AssertionError("read-only projection entered a forbidden path")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("load_pool", "_seed_from_singletons"):
        monkeypatch.setattr(pool_mod, name, forbidden)
    for name in ("_save_auth_store", "write_credential_pool", "_save_private_json", "atomic_json_write",
                 "_auth_store_lock", "_file_lock"):
        monkeypatch.setattr(auth_mod, name, forbidden)
    monkeypatch.setattr(auth_store_files.shutil, "copy2", forbidden)
    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None)


@pytest.fixture
def home(tmp_path, monkeypatch, guarded):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "auth.json").write_text(_store([
        _row("8e990c", "foglinesupply", _jwt()),
        _row("63dfc7", "emchristiansen-mcortex", _jwt(ACCT_B, "user-2", "emchristiansen@mcortex.net"), 1),
    ]))
    return hermes_home


def _run(argv: list[str], capsys):
    from hermes_cli.auth_commands import auth_command
    from hermes_cli.subcommands.auth import build_auth_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_auth_parser(parser.add_subparsers(dest="command"), cmd_auth=auth_command)
    args = parser.parse_args(["auth", "principals", *argv])
    code = 0
    try:
        args.func(args)
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip().startswith("{") else out)


def _assert_no_secrets(text: str) -> None:
    for secret in (SIG, "rt-secret-", INCIDENTAL, "eyJ"):
        assert secret not in text


def test_selected_row_projects_exact_principal_and_nothing_else(home, capsys):
    before = _tree(home)
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 0 and _tree(home) == before
    assert set(out) == {"provider", "observed_from", "observed_to", "store", "path", "store_present",
                        "changed_during_read", "snapshot", "rows"}
    assert out["store"] == "active home (classic mode)" and out["changed_during_read"] is False
    assert out["rows"] == [{
        "row_id": "8e990c", "resolved": True, "account_id": ACCT_A, "subject": "user-1",
        "email": "eric@foglinesupply.com",
        "email_source": "access token claim https://api.openai.com/profile.email (local unsigned decode "
                        "of the stored access token; not issuer-verified, not a freshness or validity check)",
        "identity_source": "local unsigned decode of the stored access token; not issuer-verified, "
                           "not a freshness or validity check",
        "last_refresh": "2026-11-01T00:00:00Z"}]
    _assert_no_secrets(json.dumps(out))


def test_only_requested_rows_are_projected(home, capsys):
    code, out = _run(["openai-codex", "--row", "63dfc7"], capsys)
    assert code == 0 and [r["row_id"] for r in out["rows"]] == ["63dfc7"]
    assert ACCT_A not in json.dumps(out) and "foglinesupply" not in json.dumps(out)


@pytest.mark.parametrize("argv,message", [
    (["openai-codex"], "needs at least one --row ID"),
    (["anthropic", "--row", "8e990c"], "supports only openai-codex"),
])
def test_no_enumeration_and_no_other_provider(home, capsys, argv, message):
    code, out = _run(argv, capsys)
    assert message in str(code) and out == ""


@pytest.mark.parametrize("rows,request_id,reason,raw", [
    ([_row("8e990c", "foglinesupply", _jwt())], "foglinesupply", "row id not in the pool", None),
    ([_row("8e990c", "a", _jwt()), _row("8e990c", "b", _jwt(ACCT_B))], "8e990c",
     "ambiguous: 2 rows share this id", None),
    ([_row("8e990c", "a", None)], "8e990c", "stored access token is missing or not a readable JWT", None),
    ([_row("8e990c", "a", "not-a-jwt")], "8e990c", "stored access token is missing or not a readable JWT", None),
    ([_row("8e990c", "a", _jwt(None))], "8e990c", "access token has no chatgpt_account_id claim", None),
    ([_row("8e990c", "a", _jwt(subject=None))], "8e990c", "access token has no sub claim", None),
    ([_row("8e990c", "a", _jwt(subject="   "))], "8e990c", "access token has no sub claim", None),
    ([_row("8e990c", "a", _jwt(subject=" user-1 "))], "8e990c", "sub claim has surrounding whitespace",
     ("subject", " user-1 ")),
    ([_row("8e990c", "a", _jwt(ACCT_A + "\t"))], "8e990c",
     "chatgpt_account_id claim has surrounding whitespace", ("account_id", ACCT_A + "\t")),
])
def test_unresolved_rows_say_why(home, capsys, rows, request_id, reason, raw):
    (home / "auth.json").write_text(_store(rows))
    code, out = _run(["openai-codex", "--row", request_id], capsys)
    row = out["rows"][0]
    assert code == 0 and row["resolved"] is False and row["reason"] == reason
    if raw is not None:  # opaque identifiers are reported exactly, never trimmed
        assert row[raw[0]] == raw[1]
    _assert_no_secrets(json.dumps(out))


def test_missing_email_is_null_not_guessed_from_label(home, capsys):
    (home / "auth.json").write_text(_store([_row("8e990c", "eric@foglinesupply.com", _jwt(email=None))]))
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    row = out["rows"][0]
    assert code == 0 and row["resolved"] is True and row["email"] is None
    assert row["email_source"] == "not stored: no email claim in the stored access token"


def test_absent_store_is_distinct_from_unreadable(tmp_path, monkeypatch, guarded, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    (tmp_path / "empty-home").mkdir()
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 0 and out["store_present"] is False and out["changed_during_read"] is False
    assert out["store"] == "active home (classic mode)" and out["path"] == str(tmp_path / "empty-home" / "auth.json")
    assert out["rows"][0]["reason"] == "row id not in the pool"


@pytest.mark.parametrize("content,kind", [
    ("{not json", "is not parseable (JSONDecodeError)"),
    (b"\xff\xfe\x00garbage", "is not parseable (UnicodeDecodeError)"),
    ("[]", "has no recognised auth-store layout"),
    ("{}", "has no recognised auth-store layout"),
    pytest.param("[" * 200_000 + "]" * 200_000, "is not parseable (RecursionError)", id="nesting-too-deep"),
])
def test_unreadable_store_is_an_error_without_writes(home, capsys, content, kind):
    target = home / "auth.json"
    target.write_bytes(content if isinstance(content, bytes) else content.encode())
    before = _tree(home)
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 1 and out["error"] == f"auth store unreadable: {target} {kind}"
    assert out["rows"] == [{"row_id": "8e990c", "resolved": False, "reason": out["error"]}]
    assert (out["store"], out["path"], out["store_present"]) == ("active home (classic mode)", str(target), True)
    assert _tree(home) == before  # no .corrupt copy, no rewrite


def _profile_layout(tmp_path, monkeypatch, *, profile_store: str | None, root_store: str):
    root = tmp_path / "hroot"
    profile = root / "profiles" / "p1"
    profile.mkdir(parents=True)
    (root / "auth.json").write_text(root_store)
    if profile_store is not None:
        (profile / "auth.json").write_text(profile_store)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return root, profile


ROOT_ROWS = _store([_row("r00001", "root-login", _jwt(ACCT_A, "root-sub"))])


def test_profile_without_rows_borrows_root(tmp_path, monkeypatch, guarded, capsys):
    root, _profile = _profile_layout(tmp_path, monkeypatch, profile_store=None, root_store=ROOT_ROWS)
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 0 and out["store"] == "global root (borrowed by this profile)"
    assert out["path"] == str(root / "auth.json") and out["rows"][0]["subject"] == "root-sub"
    assert out["store_present"] is True


def test_profile_with_rows_never_leaks_root_rows(tmp_path, monkeypatch, guarded, capsys):
    _root, profile = _profile_layout(tmp_path, monkeypatch, root_store=ROOT_ROWS, profile_store=_store(
        [_row("p00001", "profile-login", _jwt(ACCT_B, "profile-sub"))]))
    code, out = _run(["openai-codex", "--row", "p00001", "--row", "r00001"], capsys)
    assert code == 0 and out["store"] == "profile" and out["path"] == str(profile / "auth.json")
    assert out["rows"][0]["subject"] == "profile-sub"
    assert out["rows"][1] == {"row_id": "r00001", "resolved": False, "reason": "row id not in the pool"}


def test_corrupt_profile_never_silently_borrows_root(tmp_path, monkeypatch, guarded, capsys):
    root, profile = _profile_layout(tmp_path, monkeypatch, root_store=ROOT_ROWS, profile_store="{oops")
    before = (_tree(root),)
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 1 and "auth store unreadable" in out["error"] and str(profile / "auth.json") in out["error"]
    assert (out["store"], out["path"], out["store_present"]) == ("profile", str(profile / "auth.json"), True)
    assert "root-sub" not in json.dumps(out) and (_tree(root),) == before


def test_corrupt_root_is_an_error_for_a_borrowing_profile(tmp_path, monkeypatch, guarded, capsys):
    root, _profile = _profile_layout(tmp_path, monkeypatch, root_store="{oops", profile_store=None)
    before = _tree(root)
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 1 and str(root / "auth.json") in out["error"] and _tree(root) == before
    # The failed store is root; the profile (absent) is never presented as the owner (PP-04).
    assert (out["store"], out["path"], out["store_present"]) == ("global root", str(root / "auth.json"), True)


def test_store_changed_during_read_marks_every_row_unresolved(home, monkeypatch, capsys):
    from hermes_cli import auth_store_files

    real = auth_store_files.select_pool_rows

    def interleaved_writer(*args):
        result = real(*args)
        (home / "auth.json").write_text(_store([_row("8e990c", "swapped", _jwt(ACCT_B, "other-sub"))]))
        return result

    monkeypatch.setattr(auth_store_files, "select_pool_rows", interleaved_writer)
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 1 and out["changed_during_read"] is True
    assert (out["store"], out["path"], out["store_present"]) == (
        "active home (classic mode)", str(home / "auth.json"), True)
    assert out["rows"] == [{"row_id": "8e990c", "resolved": False,
                            "reason": "auth store changed during the read; re-run"}]
    assert ACCT_A not in json.dumps(out) and "other-sub" not in json.dumps(out)


def test_legacy_reader_default_is_unchanged(tmp_path, monkeypatch):
    """The non-strict default still degrades a corrupt store to empty and preserves a copy."""
    import hermes_cli.auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text("{oops")
    assert auth_mod.read_credential_pool("openai-codex") == []
    assert (tmp_path / "auth.json.corrupt").read_text() == "{oops"
    (tmp_path / "auth.json").write_text(_store([_row("8e990c", "a", "t")]))
    assert [r["id"] for r in auth_mod.read_credential_pool("openai-codex")] == ["8e990c"]
    assert list(auth_mod.read_credential_pool()) == ["openai-codex"]


def test_strict_read_ignores_a_memoised_degraded_root(tmp_path, monkeypatch, capsys):
    """A legacy read memoises the empty store a corrupt root degrades to; strict must still fail."""
    import hermes_cli.auth as auth_mod

    root, _profile = _profile_layout(tmp_path, monkeypatch, root_store="{oops", profile_store=None)
    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None)
    from hermes_cli import auth_store_files
    monkeypatch.setattr(auth_store_files.shutil, "copy2", lambda *_a, **_k: None)  # legacy copy, not under test
    assert auth_mod.read_credential_pool("openai-codex") == []  # legacy: degraded and memoised
    assert auth_mod._global_auth_store_cache is not None
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 1 and str(root / "auth.json") in out["error"]


# ── PP-01: a malformed present pool/provider/row shape is an error, never "no rows" ─────────────────

_VALID = [_row("p00001", "profile-login", _jwt(ACCT_B, "profile-sub"))]
MALFORMED_PROFILE_STORES = {
    "pool is a string": {"providers": {}, "credential_pool": "broken"},
    "pool is a list": {"providers": {}, "credential_pool": []},
    "providers is a list": {"providers": [], "credential_pool": {"openai-codex": _VALID}},
    "provider slice is a string": {"providers": {}, "credential_pool": {"openai-codex": "broken"}},
    "provider slice is null": {"providers": {}, "credential_pool": {"openai-codex": None}},
    "provider slice is an object": {"providers": {}, "credential_pool": {"openai-codex": {}}},
    "row is null": {"providers": {}, "credential_pool": {"openai-codex": [None]}},
    "row is a string": {"providers": {}, "credential_pool": {"openai-codex": ["row"]}},
    "row has no id": {"providers": {}, "credential_pool": {"openai-codex": [{"label": "x"}]}},
    "row id is a number": {"providers": {}, "credential_pool": {"openai-codex": [{"id": 7}]}},
    "another provider's slice is malformed": {"providers": {}, "credential_pool": {"nous": "broken"}},
}


@pytest.mark.parametrize("name", sorted(MALFORMED_PROFILE_STORES))
def test_malformed_profile_pool_is_an_error_and_never_borrows_root(tmp_path, monkeypatch, guarded, capsys, name):
    root, profile = _profile_layout(tmp_path, monkeypatch, root_store=ROOT_ROWS,
                                    profile_store=json.dumps(MALFORMED_PROFILE_STORES[name]))
    before = (_tree(root), _tree(profile))
    code, out = _run(["openai-codex", "--row", "r00001", "--row", "p00001"], capsys)
    assert code == 1 and out["error"].startswith(f"auth store unreadable: {profile / 'auth.json'} has a malformed ")
    assert (out["store"], out["path"], out["store_present"]) == ("profile", str(profile / "auth.json"), True)
    assert [r["resolved"] for r in out["rows"]] == [False, False]
    assert "root-sub" not in json.dumps(out) and "profile-sub" not in json.dumps(out)
    assert (_tree(root), _tree(profile)) == before


@pytest.mark.parametrize("profile_store", [
    None,
    {"providers": {}},
    {"credential_pool": {}},
    {"providers": {}, "credential_pool": {"openai-codex": []}},
    {"providers": {}, "credential_pool": {"nous": [{"id": "n00001"}]}},
], ids=["absent", "no-pool", "empty-pool", "empty-slice", "other-provider-only"])
def test_legitimately_empty_profile_borrows_root(tmp_path, monkeypatch, guarded, capsys, profile_store):
    root, _profile = _profile_layout(tmp_path, monkeypatch, root_store=ROOT_ROWS,
                                     profile_store=None if profile_store is None else json.dumps(profile_store))
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 0 and out["store"] == "global root (borrowed by this profile)"
    assert out["path"] == str(root / "auth.json") and out["rows"][0]["subject"] == "root-sub"


def test_malformed_root_slice_is_an_error_for_a_borrowing_profile(tmp_path, monkeypatch, guarded, capsys):
    root, _profile = _profile_layout(tmp_path, monkeypatch, profile_store=None, root_store=json.dumps(
        {"providers": {}, "credential_pool": {"openai-codex": [None]}}))
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 1 and out["error"].startswith(f"auth store unreadable: {root / 'auth.json'} has a malformed ")
    assert (out["store"], out["path"], out["store_present"]) == ("global root", str(root / "auth.json"), True)


# ── PP-05: an unrelated provider entry that cannot be normalised is a structured strict failure ─────

@pytest.mark.parametrize("portal_base_url,kind", [
    (42, "AttributeError"), (["x"], "AttributeError"), ({"x": 1}, "AttributeError"), ("http://[", "ValueError"),
], ids=["int", "list", "dict", "invalid-url"])
def test_unnormalisable_provider_entry_is_a_strict_error(home, capsys, portal_base_url, kind):
    target = home / "auth.json"
    target.write_text(json.dumps({"version": 1, "providers": {"nous": {"portal_base_url": portal_base_url}},
                                  "credential_pool": {"openai-codex": [_row("8e990c", "a", _jwt())]}}))
    before = _tree(home)
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 1
    assert out["error"] == f"auth store unreadable: {target} has a provider entry that cannot be normalised ({kind})"
    assert out["rows"] == [{"row_id": "8e990c", "resolved": False, "reason": out["error"]}]
    assert ACCT_A not in json.dumps(out) and _tree(home) == before


@pytest.mark.parametrize("portal_base_url,exc", [(42, AttributeError), ("http://[", ValueError)])
def test_legacy_reader_normalisation_is_unchanged(tmp_path, monkeypatch, portal_base_url, exc):
    """The default reader keeps its inherited behaviour (it raises); only the strict path is structured."""
    import hermes_cli.auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {"portal_base_url": portal_base_url}}}))
    with pytest.raises(exc):
        auth_mod._load_auth_store()


# ── PP-03 / PP-04: observation and read failures are JSON failures naming the store, never absence ──

def _chmod(path, mode):
    original = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode)
    return original


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="permission bits bind non-root only")
def test_unobservable_store_directory_is_a_failure_not_absence(home, capsys):
    original = _chmod(home, 0)  # stat of auth.json now fails with PermissionError
    try:
        code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    finally:
        home.chmod(original)
    target = home / "auth.json"
    assert code == 1 and out["error"] == f"auth store could not be observed: {target} (PermissionError)"
    assert (out["store"], out["path"], out["store_present"], out["changed_during_read"]) == (
        "active home (classic mode)", str(target), None, None)
    assert out["rows"] == [{"row_id": "8e990c", "resolved": False, "reason": out["error"]}]


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="permission bits bind non-root only")
def test_unreadable_store_file_is_a_failure(home, capsys):
    target = home / "auth.json"
    original = _chmod(target, 0)
    try:
        code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    finally:
        target.chmod(original)
    assert code == 1 and out["error"] == f"auth store unreadable: {target} could not be read (PermissionError)"
    assert (out["store"], out["path"], out["store_present"], out["changed_during_read"]) == (
        "active home (classic mode)", str(target), True, False)


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="permission bits bind non-root only")
def test_store_unobservable_after_a_successful_read_is_a_failure(home, monkeypatch, capsys):
    from hermes_cli import auth_store_files

    real = auth_store_files.select_pool_rows
    originals = []

    def then_hide(*args):
        result = real(*args)
        originals.append(_chmod(home, 0))
        return result

    monkeypatch.setattr(auth_store_files, "select_pool_rows", then_hide)
    try:
        code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    finally:
        if originals:
            home.chmod(originals[0])
    target = home / "auth.json"
    assert code == 1 and out["error"] == f"auth store could not be observed: {target} (PermissionError)"
    assert (out["store_present"], out["changed_during_read"]) == (True, None)
    assert ACCT_A not in json.dumps(out)


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="permission bits bind non-root only")
def test_unobservable_root_is_named_as_root(tmp_path, monkeypatch, guarded, capsys):
    root, _profile = _profile_layout(tmp_path, monkeypatch, profile_store=None, root_store=ROOT_ROWS)
    lock = root / "locked"
    lock.mkdir()
    (root / "auth.json").rename(lock / "auth.json")
    (root / "auth.json").symlink_to(lock / "auth.json")  # stat follows into an unreadable directory
    original = _chmod(lock, 0)
    try:
        code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    finally:
        lock.chmod(original)
    assert code == 1 and out["error"] == f"auth store could not be observed: {root / 'auth.json'} (PermissionError)"
    assert (out["store"], out["path"], out["store_present"]) == ("global root", str(root / "auth.json"), None)


def test_absent_profile_and_root_is_success_with_honest_presence(tmp_path, monkeypatch, guarded, capsys):
    root = tmp_path / "hroot"
    (root / "profiles" / "p1").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "p1"))
    code, out = _run(["openai-codex", "--row", "r00001"], capsys)
    assert code == 0 and (out["store"], out["path"], out["store_present"]) == (
        "global root (borrowed by this profile)", str(root / "auth.json"), False)
    assert out["rows"] == [{"row_id": "r00001", "resolved": False, "reason": "row id not in the pool"}]


# ── PP-02: the real entry point initializes and writes nothing ─────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(root: Path) -> dict:
    """Every entry under *root*: file bytes and mode, directory mode, symlink target."""
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        key = p.relative_to(root).as_posix()
        if stat.S_ISLNK(st.st_mode):
            out[key] = ("link", os.readlink(p))
        elif stat.S_ISDIR(st.st_mode):
            out[key] = ("dir", stat.S_IMODE(st.st_mode))
        else:
            out[key] = ("file", stat.S_IMODE(st.st_mode), hashlib.sha256(p.read_bytes()).hexdigest())
    return out


# Temp environments a launch can inherit: a private TMPDIR; Hermes' own scratch export inherited
# (TMPDIR == HERMES_SCRATCH_DIR, which the scratch setup treats as its own and re-points); the same
# with TMP/TEMP too; and no temp variables at all (the scratch setup then picks its own directory).
TEMP_ENVIRONMENTS = ("custom", "inherited-scratch", "inherited-scratch-all", "unset")


def _fresh_cli(tmp_path: Path, argv: list[str], *, store: str | None, profile_flag: bool = False,
               flags: tuple[str, ...] = (), temp: str = "custom", root: str | None = None, entry: str = "module"):
    """Run ``python -m hermes_cli.main [-p p] FLAGS ARGV`` in a fresh interpreter on a synthetic HOME
    whose profile ``p`` lives under the default root, with the *temp* environment named in
    ``TEMP_ENVIRONMENTS``; the whole synthetic tree (files, directories, links, modes) is
    snapshotted before the interpreter starts and compared after it exits."""
    box = tmp_path / "box"
    user = box / "user"
    profile = user / ".hermes" / "profiles" / "p"
    profile.mkdir(parents=True)
    (box / "tmp").mkdir()
    for path, content in ((profile / "auth.json", store), (user / ".hermes" / "auth.json", root)):
        if content is not None:
            path.write_text(content)
            path.chmod(0o600)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTEST_"))
           and key not in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME")}
    env.update({"HOME": str(user), "XDG_CACHE_HOME": str(box / "cache"), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)})
    assert temp in TEMP_ENVIRONMENTS
    if temp != "unset":
        env["TMPDIR"] = str(box / "tmp")
    if temp.startswith("inherited-scratch"):
        env["HERMES_SCRATCH_DIR"] = str(box / "tmp")
    if temp == "inherited-scratch-all":
        env.update(TMP=str(box / "tmp"), TEMP=str(box / "tmp"))
    if not profile_flag:
        env["HERMES_HOME"] = str(profile)
    before = _snapshot(box)
    argv = (["-p", "p"] if profile_flag else []) + list(flags) + argv
    # ``module``: ``python -m hermes_cli.main``; ``console``: the console script's ``main()`` import.
    command = ([sys.executable, "-m", "hermes_cli.main", *argv] if entry == "module" else
               [sys.executable, "-c", f"import sys; sys.argv = {['hermes', *argv]!r}; "
                                      "from hermes_cli.main import main; main()"])
    result = subprocess.run(command, capture_output=True, text=True,
                            env=env, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, check=False)
    return result, before, _snapshot(box), profile


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("store,expect_resolved,owner", [
    (None, False, "global root (borrowed by this profile)"), ("valid", True, "profile")], ids=["absent", "valid"])
def test_fresh_cli_process_writes_nothing(tmp_path, store, expect_resolved, owner):
    content = None if store is None else _store([_row("chosen", "a", _jwt())])
    result, before, after, _profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"], store=content)
    assert result.returncode == 0, result.stderr
    assert after == before  # no home skeleton, SOUL.md, logs, .corrupt copy, mode or content change
    out = json.loads(result.stdout)
    assert out["rows"][0]["resolved"] is expect_resolved and out["store"] == owner
    _assert_no_secrets(result.stdout)


@pytest.mark.platforms("linux")
def test_fresh_cli_process_honours_the_profile_flag_and_writes_nothing(tmp_path):
    result, before, after, profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), profile_flag=True)
    assert result.returncode == 0, result.stderr
    assert after == before
    out = json.loads(result.stdout)
    assert out["path"] == str(profile / "auth.json") and out["rows"][0]["resolved"] is True


@pytest.mark.platforms("linux")
def test_fresh_cli_process_store_error_is_json_without_writes(tmp_path):
    result, before, after, profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"], store="{oops")
    assert result.returncode == 1 and after == before
    assert json.loads(result.stdout)["error"] == f"auth store unreadable: {profile / 'auth.json'} is not parseable (JSONDecodeError)"
    assert "Traceback" not in result.stderr


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("temp", TEMP_ENVIRONMENTS)
@pytest.mark.parametrize("profile_flag,flags", [
    (False, ()), (False, ("--cli",)), (False, ("--yolo",)), (False, ("--yolo", "--cli")),
    (True, ()), (True, ("--cli",)),
], ids=["plain", "cli", "yolo", "yolo-cli", "profile", "profile-cli"])
def test_fresh_cli_startup_writes_nothing_for_any_spelling_or_temp_environment(tmp_path, profile_flag, flags, temp):
    """PP-02: the whole startup (bootstrap scratch setup, profile scratch re-export, config/logging)
    is skipped for every accepted spelling, including an inherited scratch TMPDIR."""
    result, before, after, profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), profile_flag=profile_flag, flags=flags, temp=temp)
    assert result.returncode == 0, result.stderr
    assert after == before and result.stderr == ""
    out = json.loads(result.stdout)
    assert out["path"] == str(profile / "auth.json") and out["rows"][0]["resolved"] is True


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("store,root,code,owner", [
    (None, None, 0, "global root (borrowed by this profile)"),
    (None, _store([_row("chosen", "a", _jwt())]), 0, "global root (borrowed by this profile)"),
    (json.dumps({"providers": {}, "credential_pool": {"openai-codex": "broken"}}), None, 1, "profile"),
    (None, "{oops", 1, "global root"),
], ids=["absent", "root-fallback", "malformed-profile", "corrupt-root"])
def test_fresh_cli_stores_write_nothing_under_an_inherited_scratch_tmpdir(tmp_path, store, root, code, owner):
    result, before, after, _profile = _fresh_cli(
        tmp_path, ["--yolo", "auth", "principals", "openai-codex", "--row", "chosen"],
        store=store, root=root, temp="inherited-scratch")
    assert result.returncode == code and after == before, result.stderr
    out = json.loads(result.stdout)
    assert out["store"] == owner and ("error" in out) is bool(code)


# ── PP-06: strict normalisation never echoes an unrelated stored value on either stream ────────────

SECRET_PORTAL_URL = "https://SYNTHETIC-USER:SYNTHETIC-PASSWORD@api.nousresearch.com/private?access_token=SYNTHETIC-BEARER"
STALE_PORTAL_URL = "https://api.nousresearch.com"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("portal_base_url", [SECRET_PORTAL_URL, STALE_PORTAL_URL], ids=["credential-url", "stale-url"])
def test_fresh_cli_never_echoes_an_unrelated_stored_url(tmp_path, portal_base_url):
    store = json.dumps({"version": 1, "providers": {"nous": {"portal_base_url": portal_base_url}},
                        "credential_pool": {"openai-codex": [_row("chosen", "a", _jwt())]}})
    result, before, after, _profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"], store=store)
    assert result.returncode == 0 and after == before
    assert result.stderr == ""
    for value in (portal_base_url, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD", "SYNTHETIC-BEARER"):
        assert value not in result.stdout
    row = json.loads(result.stdout)["rows"][0]
    assert row["resolved"] is True and row["account_id"] == ACCT_A and row["subject"] == "user-1"
    _assert_no_secrets(result.stdout)


def test_legacy_reader_still_announces_the_stale_url_migration(tmp_path, monkeypatch, caplog):
    """The default reader's diagnostic is unchanged: only the strict projection is silent."""
    import hermes_cli.auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {"portal_base_url": STALE_PORTAL_URL}}}))
    with caplog.at_level("WARNING", logger="hermes_cli.auth"):
        store = auth_mod._load_auth_store()
    assert store["providers"]["nous"]["portal_base_url"] != STALE_PORTAL_URL
    assert [r.getMessage() for r in caplog.records] == [
        f"auth: migrating stale nous portal_base_url {STALE_PORTAL_URL} -> "
        f"{store['providers']['nous']['portal_base_url']}"]


def test_strict_reader_normalises_like_the_default_reader_without_a_diagnostic(tmp_path, caplog):
    from hermes_cli import auth_store_files

    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {"portal_base_url": SECRET_PORTAL_URL}}}))
    with caplog.at_level("DEBUG"):
        strict = auth_store_files.load_auth_store_file(tmp_path / "auth.json", strict=True)
    assert caplog.records == []
    with caplog.at_level("WARNING", logger="hermes_cli.auth"):
        default = auth_store_files.load_auth_store_file(tmp_path / "auth.json")
    assert strict == default and default["providers"]["nous"]["portal_base_url"] != SECRET_PORTAL_URL
    assert len(caplog.records) == 1


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("temp", ["custom", "inherited-scratch"])
def test_other_commands_keep_ordinary_startup(tmp_path, temp):
    """Only ``auth principals`` skips startup: ``hermes auth --help`` still scaffolds the home and,
    when the inherited TMPDIR is Hermes' own scratch export, re-points it at this home's scratch dir
    (a user's own TMPDIR is left alone)."""
    result, _before, after, _profile = _fresh_cli(tmp_path, ["auth", "--help"], store=None, temp=temp)
    assert result.returncode == 0, result.stderr
    assert "user/.hermes/profiles/p/SOUL.md" in after
    assert ("user/.hermes/profiles/p/cache/scratch" in after) is (temp == "inherited-scratch")


def test_projection_modules_never_import_the_home_initializing_config():
    probe = ("import sys\nimport hermes_cli.auth_principals, hermes_cli.auth_store_files\n"
             "print('hermes_cli.config' in sys.modules, 'hermes_cli.auth' in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=REPO_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


@pytest.mark.parametrize("argv,expected", [
    (["auth", "principals", "openai-codex"], True),
    (["--cli", "auth", "principals", "openai-codex"], True),
    (["--yolo", "--cli", "auth", "principals"], True),
    (["-p", "p", "--cli", "auth", "principals"], True),
    (["--profile=p", "auth", "principals"], True),
    (["--reasoning", "high", "-m", "x", "--skills", "a", "auth", "principals"], True),
    # Accepted abbreviations and value spellings: argparse, not a token scan, decides which word is the command.
    (["--reas", "high", "auth", "principals"], True),
    (["--mod", "x", "auth", "principals"], True),
    (["--prov", "x", "auth", "principals"], True),
    (["--mod=x", "--prov=y", "auth", "principals"], True),
    (["-mx", "auth", "principals"], True),
    (["auth", "principals", "-p", "p", "openai-codex"], True),  # -p is removed wherever it is, as the override does
    (["--p", "x", "auth", "principals"], False),  # ambiguous abbreviation: the parser rejects it
    (["--vers", "auth", "principals"], False),
    (["--one=hi", "auth", "principals"], False),
    (["-c", "auth", "principals"], False),  # -c takes "auth" as its session name
    # main joins a session name first, ended by the active memory provider's command (config): normal path
    (["-c", "my", "session", "auth", "principals"], False),
    (["--oneshot=", "auth", "principals"], True),  # main runs -z only for a non-empty prompt
    (["-z", "", "auth", "principals"], True),
    (["--one=", "auth", "principals"], True),

    (["--", "auth", "principals"], True),
    (["-p", "auth", "principals"], False),  # "auth" is the profile name; the command is "principals"
    (["auth", "list"], False),
    (["auth"], False),
    (["chat", "auth", "principals"], False),
    (["chat", "-q", "auth principals"], False),
    # The full CLI answers these itself (help, version, one-shot) or rejects them (exit 2).
    (["--help", "auth", "principals"], False),
    (["-h", "auth", "principals"], False),
    (["--yolo", "-h", "auth", "principals"], False),
    (["--version", "auth", "principals"], False),
    (["-V", "auth", "principals"], False),
    (["-z", "hello", "auth", "principals"], False),
    (["--oneshot=hello", "auth", "principals"], False),
    (["--not-a-hermes-flag", "auth", "principals"], False),
    (["--yolo=1", "auth", "principals"], False),
    (["-p", "p", "-p", "q", "auth", "principals"], False),  # the full CLI rejects it ('q' is not a command)
])
def test_no_init_command_classifier(argv, expected):
    from hermes_cli._parser import is_no_init_command

    assert is_no_init_command(argv) is expected


@pytest.mark.parametrize("argv,expected", [
    (["--cli", "auth", "principals"], True),
    (["-p", "q", "auth", "principals"], False),  # a -p left after the override is the parser's error
    (["--help", "auth", "principals"], False),
])
def test_no_init_command_after_the_profile_override(argv, expected):
    from hermes_cli._parser import is_no_init_command

    assert is_no_init_command(argv, profile_flag_consumed=True) is expected


def test_classifier_imports_no_config_and_writes_nothing(tmp_path):
    """Classifying runs in hermes_bootstrap for every command: it must only parse. A stray word in
    the options (a second ``-p NAME``) must not reach the full parser's command lookup, whose
    hints load the home-initializing config."""
    home = tmp_path / "user"
    home.mkdir()
    probe = ("import sys\nfrom hermes_cli._parser import is_no_init_command\n"
             "r = [is_no_init_command(a) for a in (['-p', 'p', '-p', 'q', 'auth', 'principals'],\n"
             "     ['--bogus', 'auth', 'principals'], ['--help', 'auth', 'principals'], ['--cli', 'auth', 'principals'])]\n"
             "print(r, 'hermes_cli.config' in sys.modules)\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("HERMES_", "PYTEST_"))}
    env.update(HOME=str(home), TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    before = _snapshot(tmp_path)
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=REPO_ROOT, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("] ") == ["[False, False, False, True", "False\n"]
    assert _snapshot(tmp_path) == before


@pytest.mark.platforms("linux")
def test_fresh_cli_second_profile_flag_is_rejected_not_projected(tmp_path):
    """``-p`` is consumed once before argparse; the full CLI rejects a second one (exit 2)."""
    result, _before, _after, _profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), profile_flag=True, flags=("-p", "p"))
    assert result.returncode == 2
    assert "'p' is not a `hermes` command." in result.stderr
    assert result.stdout == "" and ACCT_A not in result.stderr


def test_classifier_never_prints(capsys):
    from hermes_cli._parser import is_no_init_command

    for argv in (["--help", "auth", "principals"], ["--bogus", "auth", "principals"]):
        assert is_no_init_command(argv) is False
    assert capsys.readouterr() == ("", "")


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("flags", [("--help",), ("--yolo", "-h")], ids=["help", "yolo-h"])
def test_fresh_cli_top_level_help_is_help_not_a_projection(tmp_path, flags):
    """Global help before ``auth principals`` is answered by the full parser, as for any command."""
    result, _before, _after, _profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), flags=flags, temp="inherited-scratch")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: hermes ") and "<command>" in result.stdout
    assert ACCT_A not in result.stdout + result.stderr and '"rows"' not in result.stdout


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("flags", [("--not-a-hermes-flag",), ("--cli", "--bogus")], ids=["unknown", "cli-unknown"])
def test_fresh_cli_unknown_top_level_option_is_rejected_not_projected(tmp_path, flags):
    result, _before, _after, _profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), flags=flags, temp="inherited-scratch")
    assert result.returncode == 2
    assert f"unrecognized arguments: {flags[-1]}" in result.stderr
    assert result.stdout == "" and ACCT_A not in result.stderr


def _real_parser_reaches_auth_principals(argv: list[str]) -> bool:
    """What ``main._parse_cli_args`` does with *argv* (after the profile override), with a stand-in
    ``auth principals`` subcommand: session names coalesced, ``parse_args`` with subcommand routing
    required, then without it; no output, no exit."""
    from hermes_cli._parser import _coalesce_session_name_args, build_top_level_parser

    parser, subparsers, _chat = build_top_level_parser()
    auth = subparsers.add_parser("auth").add_subparsers(dest="auth_command")
    auth.add_parser("principals").add_argument("rest", nargs="*")

    def refuse(*_a, **_k):
        raise RuntimeError

    for p in (parser, *subparsers.choices.values()):
        p.print_help = p.print_usage = p.error = p.exit = refuse
    argv = _coalesce_session_name_args(argv)
    try:
        subparsers.required = True
        args = parser.parse_args(argv)
    except RuntimeError:
        try:
            subparsers.required = False
            args = parser.parse_args(argv)
        except RuntimeError:
            return False
    return (args.command, getattr(args, "auth_command", None)) == ("auth", "principals") \
        and not args.version and not args.oneshot  # main's own tests: ``if args.version``, ``if args.oneshot``


@pytest.mark.parametrize("argv", [
    ["auth", "principals", "x"], ["--reas", "high", "auth", "principals"], ["--mod", "x", "auth", "principals"],
    ["--prov", "x", "auth", "principals"], ["--reasoning", "high", "auth", "principals"], ["--mod=x", "auth", "principals"],
    ["-mx", "auth", "principals"], ["-t", "a,b", "--yolo", "auth", "principals"], ["--skills", "s", "-s", "t", "auth", "principals"],
    ["--res", "latest", "auth", "principals"], ["--in", "/d", "auth", "principals"], ["--usage", "f", "auth", "principals"],
    ["--", "auth", "principals"], ["-c", "auth", "principals"], ["-c", "name", "auth", "principals"],
    ["--p", "x", "auth", "principals"], ["--vers", "auth", "principals"], ["-z", "q", "auth", "principals"],
    ["-c", "my", "session", "auth", "principals"], ["--continue", "a", "b", "auth", "principals"],
    ["--oneshot", "", "auth", "principals"], ["--oneshot=", "auth", "principals"], ["-z", "", "auth", "principals"],
    ["--one=", "auth", "principals"], ["-zq", "auth", "principals"], ["-z", "q", "--version", "auth", "principals"],
    ["--cont", "a", "b", "auth", "principals"], ["-r", "a", "b", "auth", "principals"],
    ["--help", "auth", "principals"], ["--bogus", "auth", "principals"], ["chat", "auth", "principals"],
])
def test_classifier_agrees_with_the_real_top_level_parser(argv):
    """The fast path and the full parser must agree on whether *argv* is ``auth principals``. A session
    flag that captures a name takes the normal path: the active memory provider's ``hermes <name>``
    ends a name, and finding it reads config, which the classifier must not."""
    from hermes_cli._parser import _coalesce_session_name_args, is_no_init_command

    if _coalesce_session_name_args(argv, provider_commands=False, names_only=True):
        assert is_no_init_command(argv, profile_flag_consumed=True) is False
        return
    assert is_no_init_command(argv, profile_flag_consumed=True) is _real_parser_reaches_auth_principals(argv)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("entry", ["module", "console"])
@pytest.mark.parametrize("flags", [("--reas", "high"), ("--mod", "x"), ("--prov", "x"), ("--reasoning", "high"),
                                   ("--model=x",), ("-mx",), ("--reas", "high", "--cli"), ("--oneshot=",)],
                         ids=["reas", "mod", "prov", "reasoning", "model-eq", "attached", "reas-cli", "oneshot-empty"])
def test_fresh_cli_option_spellings_write_nothing(tmp_path, flags, entry):
    """PP-02 (review 7066): every accepted spelling of a value-taking option reaches the write-free path."""
    result, before, after, profile = _fresh_cli(
        tmp_path, ["auth", "principals", "openai-codex", "--row", "chosen"],
        store=_store([_row("chosen", "a", _jwt())]), flags=flags, temp="inherited-scratch", entry=entry)
    assert result.returncode == 0, result.stderr
    assert after == before and result.stderr == ""
    out = json.loads(result.stdout)
    assert out["path"] == str(profile / "auth.json") and out["rows"][0]["resolved"] is True


def test_pre_import_dispatch_passes_only_the_command_arguments(monkeypatch):
    from hermes_cli import _startup_fast, auth_principals

    seen = []
    monkeypatch.setattr(_startup_fast, "container_mode_may_be_active", lambda: False)
    monkeypatch.setattr(auth_principals, "main", lambda argv: seen.append(argv) or 7)
    assert _startup_fast.try_auth_principals(["--yolo", "--cli", "auth", "principals", "openai-codex", "--row", "x"]) == 7
    assert _startup_fast.try_auth_principals(["--reas", "high", "auth", "principals", "openai-codex", "--row", "y"]) == 7
    assert _startup_fast.try_auth_principals(["auth", "list"]) is None
    assert _startup_fast.try_auth_principals(["--help", "auth", "principals", "openai-codex"]) is None
    assert _startup_fast.try_auth_principals(["--bogus", "auth", "principals", "openai-codex"]) is None
    assert seen == [["openai-codex", "--row", "x"], ["openai-codex", "--row", "y"]]


def test_container_mode_takes_the_normal_path(tmp_path, monkeypatch):
    from hermes_cli import _startup_fast

    monkeypatch.setattr(_startup_fast, "container_mode_may_be_active", lambda: True)
    assert _startup_fast.try_auth_principals(["auth", "principals", "openai-codex", "--row", "x"]) is None


# ── Review t_96d1256c/run7088 CL-01/02/03: a repeated JSON member is ambiguous, never a parser's pick ──
def _raw_jwt(claims_json: str) -> str:
    """A JWT whose claims are exactly *claims_json* (which may repeat members, as json.dumps never would)."""
    def enc(text: str) -> str:
        return base64.urlsafe_b64encode(text.encode()).rstrip(b"=").decode()

    return f"{enc(json.dumps({'alg': 'RS256'}))}.{enc(claims_json)}.{SIG}"


_AUTH = '"https://api.openai.com/auth"'
_PROFILE = '"https://api.openai.com/profile"'


@pytest.mark.parametrize("claims_json", [
    # account id stated twice, both orders (the first one is the reserved/personal witness)
    f'{{{_AUTH}: {{"chatgpt_account_id": "personal", "chatgpt_account_id": "{ACCT_A}"}}, "sub": "s"}}',
    f'{{{_AUTH}: {{"chatgpt_account_id": "{ACCT_A}", "chatgpt_account_id": "personal"}}, "sub": "s"}}',
    f'{{{_AUTH}: {{"chatgpt_account_id": "{ACCT_A}"}}, "sub": "s1", "sub": "s2"}}',
    f'{{{_AUTH}: {{"chatgpt_account_id": "{ACCT_A}"}}, "sub": "s", '
    f'{_PROFILE}: {{"email": "a@x"}}, {_PROFILE}: {{"email": "b@x"}}}}',
    f'{{{_PROFILE}: {{"email": "a@x", "email": "b@x"}}, {_AUTH}: {{"chatgpt_account_id": "{ACCT_A}"}}, "sub": "s"}}',
    f'{{{_AUTH}: {{"chatgpt_account_id": "personal"}}, {_AUTH}: {{"chatgpt_account_id": "{ACCT_A}"}}, "sub": "s"}}',
    f'{{{_AUTH}: {{"chatgpt_account_id": "{ACCT_A}"}}, "sub": "s", "plan": 1, "plan": 2}}',  # any repeat is ambiguity
], ids=["account-personal-first", "account-personal-last", "sub", "profile-namespace", "email", "auth-namespace",
        "unrelated-member"])
def test_token_repeating_a_claim_member_is_ambiguous_not_resolved(home, capsys, claims_json):
    (home / "auth.json").write_text(_store([_row("8e990c", "a", _raw_jwt(claims_json))]))
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 0
    assert out["rows"] == [{"row_id": "8e990c", "resolved": False,
                            "reason": "ambiguous: the stored access token's claims repeat a member name"}]
    assert "personal" not in json.dumps(out) and ACCT_A not in json.dumps(out)


def test_legacy_jwt_decode_is_unchanged_for_repeated_members():
    from hermes_cli.auth_constants import _decode_jwt_claims

    token = _raw_jwt('{"sub": "first", "sub": "last"}')
    assert _decode_jwt_claims(token) == {"sub": "last"}  # default callers keep json's last-member result
    assert _decode_jwt_claims("not-a-jwt", unique_members=True) == {}


_POOL_ROW = json.dumps(_row("8e990c", "a", _jwt()))
_PERSONAL_ROW = json.dumps(_row("p00000", "personal", _jwt(ACCT_B, "personal-sub")))


@pytest.mark.parametrize("store_json", [
    # a nonempty pool then an empty one (last-member reading would erase the rows and borrow root), and reversed
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_PERSONAL_ROW}]}}, "credential_pool": {{}}}}',
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW}]}}}}',
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_PERSONAL_ROW}], "openai-codex": []}}}}',
    f'{{"version": 1, "providers": {{}}, "providers": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW}]}}}}',
    # a row stating its token twice, both orders
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW[:-1]}, '
    f'"access_token": "{_jwt(ACCT_B, "personal-sub")}"}}]}}}}',
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{{"access_token": '
    f'"{_jwt(ACCT_B, "personal-sub")}", {_POOL_ROW[1:]}]}}}}',
    f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW[:-1]}, "id": "other"}}]}}}}',
], ids=["pool-then-empty", "empty-then-pool", "provider-slice", "providers", "token-last", "token-first", "row-id"])
def test_store_repeating_a_member_is_a_strict_error_and_never_borrows_root(tmp_path, monkeypatch, guarded, capsys,
                                                                           store_json):
    json.loads(store_json)  # the fixture is valid JSON apart from the repeat
    root, profile = _profile_layout(tmp_path, monkeypatch, root_store=ROOT_ROWS, profile_store=store_json)
    before = (_tree(root), _tree(profile))
    code, out = _run(["openai-codex", "--row", "8e990c", "--row", "r00001"], capsys)
    target = profile / "auth.json"
    assert code == 1
    assert out["error"] == f"auth store unreadable: {target} is ambiguous (an object repeats a member name)"
    assert (out["store"], out["path"], out["store_present"]) == ("profile", str(target), True)
    assert [r["resolved"] for r in out["rows"]] == [False, False]
    assert "root-sub" not in json.dumps(out) and "personal-sub" not in json.dumps(out)
    assert (_tree(root), _tree(profile)) == before


def test_root_repeating_a_member_is_an_error_for_a_borrowing_profile(tmp_path, monkeypatch, guarded, capsys):
    root_json = f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW}]}}, "version": 1}}'
    root, _profile = _profile_layout(tmp_path, monkeypatch, root_store=root_json, profile_store=None)
    code, out = _run(["openai-codex", "--row", "8e990c"], capsys)
    assert code == 1 and (out["store"], out["path"]) == ("global root", str(root / "auth.json"))
    assert out["error"].endswith("is ambiguous (an object repeats a member name)")


def test_legacy_store_reader_is_unchanged_for_repeated_members(tmp_path, monkeypatch):
    import hermes_cli.auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None)
    (tmp_path / "auth.json").write_text(
        f'{{"version": 1, "providers": {{}}, "credential_pool": {{"openai-codex": [{_POOL_ROW}]}}, "credential_pool": {{}}}}')
    assert auth_mod.read_credential_pool("openai-codex") == []  # default readers keep json's last-member result


@pytest.mark.parametrize("marker", [True, False], ids=["clone-marker", "no-marker"])
def test_projection_import_and_read_start_no_process(tmp_path, marker):
    """CL-03: a profile's optional ``hermes-agent/.git`` clone must not make the read-only projection run
    ``git`` (the Codex user agent's version is resolved on first use, not when the constants import)."""
    home = tmp_path / "user"
    hermes_home = home / ".hermes" / "profiles" / "p"
    hermes_home.mkdir(parents=True)
    (hermes_home / "auth.json").write_text(_store([_row("8e990c", "a", _jwt())]))
    if marker:
        (hermes_home / "hermes-agent" / ".git").mkdir(parents=True)
    probe = ("import sys\n"
             "def guard(event, args):\n"
             "    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'os.exec')"
             " or event.startswith('socket.'):\n"
             "        raise AssertionError('forbidden: ' + event)\n"
             "sys.addaudithook(guard)\n"
             "from hermes_cli.auth_principals import project_principals\n"
             "out, ok = project_principals(['8e990c'])\n"
             "print(ok, out['rows'][0]['resolved'])\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("HERMES_", "PYTEST_"))}
    env.update(HOME=str(home), HERMES_HOME=str(hermes_home), TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=REPO_ROOT, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]


def test_codex_user_agent_is_still_the_version_on_first_use():
    """v0.23.1 builds the Codex User-Agent where the request is built (``hermes_user_agent``), never at import;
    the lazy ``auth_constants.CODEX_OAUTH_USER_AGENT`` it replaced is gone."""
    import hermes_cli.auth_constants as constants
    from hermes_cli.version_info import get_version_info, hermes_user_agent

    assert hermes_user_agent() == f"hermes-cli/{get_version_info().base_version}"
    with pytest.raises(AttributeError):
        constants.CODEX_OAUTH_USER_AGENT
