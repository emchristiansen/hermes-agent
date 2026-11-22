"""Operator display names for pooled credentials (agent/credential_display_names.py).

Display names live in a separate secret-free file keyed by stable row id, and are shown only
when the row's own access-token principal (account id and subject) matches the mapping's.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket

import pytest

ACCT_A = "a9631cb8-8258-4371-a753-7567e79aa882"
ACCT_B = "bfe407e3-3c31-4ca5-bf99-19fc0db1c601"


def _codex_jwt(account_id: str | None, subject: str | None = "user-1", email: str = "x@example.com") -> str:
    claims: dict = {"email": email}
    if subject is not None:
        claims["sub"] = subject
    if account_id is not None:
        claims["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id}
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


def _row(row_id: str, label: str, token: str, priority: int) -> dict:
    return {
        "id": row_id, "label": label, "auth_type": "oauth", "priority": priority,
        "source": "manual:device_code", "access_token": token, "refresh_token": f"rt-{row_id}",
        "base_url": "https://chatgpt.com/backend-api/codex", "last_refresh": "2026-11-01T00:00:00Z",
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _no_network(*_a, **_k):
        raise AssertionError("display names must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [
            _row("8e990c", "foglinesupply", _codex_jwt(ACCT_A), 0),
            _row("63dfc7", "emchristiansen-mcortex", _codex_jwt(ACCT_B, subject="user-2"), 1),
        ]},
    }, indent=2))
    import agent.credential_display_names as names
    monkeypatch.setattr(names, "_cache", {})
    return hermes_home


def _write_names(home, providers: dict) -> None:
    (home / "credential_display_names.json").write_text(json.dumps({"version": 1, "providers": providers}))


def _list(capsys) -> str:
    from hermes_cli.auth_commands import auth_list_command
    auth_list_command(type("Args", (), {"provider": "openai-codex"})())
    return capsys.readouterr().out


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_without_mapping_list_output_is_unchanged(home, capsys):
    out = _list(capsys)
    assert "#1  foglinesupply" in out and "#2  emchristiansen-mcortex" in out
    assert "display name" not in out


def test_verified_mapping_shows_full_name_and_writes_no_credentials(home, capsys):
    _write_names(home, {"openai-codex": {
        "8e990c": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
        "63dfc7": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_B, "expected_subject": "user-2"},
    }})
    auth_before, names_before = _sha(home / "auth.json"), _sha(home / "credential_display_names.json")
    out = _list(capsys)
    assert "#1  eric@foglinesupply.com" in out and "id=8e990c" in out
    assert "#2  emchristiansen@mcortex.net" in out and "id=63dfc7" in out
    assert _sha(home / "auth.json") == auth_before
    assert _sha(home / "credential_display_names.json") == names_before


def test_mismatched_account_is_rejected_not_shown(home, capsys):
    # Names swapped onto the wrong rows: neither email may appear on a row.
    _write_names(home, {"openai-codex": {
        "8e990c": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_B, "expected_subject": "user-2"},
        "63dfc7": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
    }})
    out = _list(capsys)
    assert "foglinesupply [display name rejected: account claim mismatch]" in out
    assert "emchristiansen-mcortex [display name rejected: account claim mismatch]" in out
    assert "@" not in out.split("credentials):", 1)[1].split("External CLI", 1)[0]


@pytest.mark.parametrize("spec,token,reason", [
    ({"display_name": "eric@foglinesupply.com"}, None, "mapping has no expected_account_id"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A}, None, "mapping has no expected_subject"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": " "}, None,
     "mapping has no expected_subject"),
    ({"display_name": "", "expected_account_id": ACCT_A, "expected_subject": "user-1"}, None, "mapping has no display_name"),
    ("not-an-object", None, "mapping entry is not an object"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"}, _codex_jwt(None), "row lacks account or subject claim"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
     _codex_jwt(ACCT_A, subject=None), "row lacks account or subject claim"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"}, "not-a-jwt", "row lacks account or subject claim"),
])
def test_incomplete_identity_is_rejected(home, capsys, spec, token, reason):
    if token is not None:
        store = json.loads((home / "auth.json").read_text())
        store["credential_pool"]["openai-codex"][0]["access_token"] = token
        (home / "auth.json").write_text(json.dumps(store))
    _write_names(home, {"openai-codex": {"8e990c": spec}})
    out = _list(capsys)
    assert f"foglinesupply [display name rejected: {reason}]" in out
    assert "eric@foglinesupply.com" not in out


def test_unknown_row_id_is_reported_and_never_matched_by_position(home, capsys):
    _write_names(home, {"openai-codex": {
        "zzz999": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
    }})
    out = _list(capsys)
    assert "display name for unknown openai-codex row id=zzz999 ignored" in out
    assert "#1  foglinesupply" in out


def test_provider_without_identity_check_is_rejected(home):
    from agent.credential_display_names import display_name
    from agent.credential_pool import PooledCredential
    _write_names(home, {"anthropic": {"a1": {"display_name": "me@example.com", "expected_account_id": "x",
                                                    "expected_subject": "y"}}})
    entry = PooledCredential(provider="anthropic", id="a1", label="claude", auth_type="oauth",
                             priority=0, source="manual", access_token="sk-ant-x")
    assert display_name(entry) == "claude [display name rejected: no account claim to match for this provider]"


@pytest.mark.parametrize("content,error", [
    ("{not json", "cannot read (JSONDecodeError)"),
    (json.dumps({"version": 2, "providers": {}}), "not a version-1 display-name mapping"),
    (json.dumps({"version": 1, "providers": {"openai-codex": ["8e990c"]}}), "provider 'openai-codex' is not an object"),
])
def test_unusable_file_is_reported_visibly_and_names_not_applied(home, capsys, caplog, content, error):
    (home / "credential_display_names.json").write_text(content)
    with caplog.at_level(logging.WARNING, logger="agent.credential_display_names"):
        out = _list(capsys)
    assert "#1  foglinesupply" in out
    assert f"display names NOT applied: {home / 'credential_display_names.json'} ignored ({error})" in out
    assert error in caplog.text


def test_bom_prefixed_file_is_read(home, capsys):
    # Windows tooling BOMs files it touches; the mapping must still apply.
    body = json.dumps({"version": 1, "providers": {"openai-codex": {
        "8e990c": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"}}}})
    (home / "credential_display_names.json").write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    out = _list(capsys)
    assert "#1  eric@foglinesupply.com" in out and "NOT applied" not in out


def test_stale_pool_flush_and_token_rotation_do_not_disturb_names(home, capsys):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import write_credential_pool

    stale = load_pool("openai-codex")  # snapshot taken before the names exist
    _write_names(home, {"openai-codex": {
        "8e990c": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
    }})
    names_before = _sha(home / "credential_display_names.json")

    # A token refresh on disk (same account, new token generation) ...
    store = json.loads((home / "auth.json").read_text())
    store["credential_pool"]["openai-codex"][0]["access_token"] = _codex_jwt(ACCT_A, email="rotated@example.com")
    store["credential_pool"]["openai-codex"][0]["refresh_token"] = "rt-rotated"
    (home / "auth.json").write_text(json.dumps(store))
    # ... then a stale writer flushes its old in-memory rows.
    write_credential_pool("openai-codex", [e.to_dict() for e in stale.entries()])

    assert _sha(home / "credential_display_names.json") == names_before
    rows = json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"]
    assert [r["label"] for r in rows] == ["foglinesupply", "emchristiansen-mcortex"]
    assert "#1  eric@foglinesupply.com" in _list(capsys)


def test_rotation_log_uses_verified_name(home, caplog):
    from agent.credential_pool import load_pool
    _write_names(home, {"openai-codex": {
        "63dfc7": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_B, "expected_subject": "user-2"},
    }})
    pool = load_pool("openai-codex")
    first = pool.select()
    assert first is not None and first.id == "8e990c"
    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        nxt = pool.mark_exhausted_and_rotate(status_code=429, credential_id="8e990c")
    assert nxt is not None and nxt.id == "63dfc7"
    assert "credential pool: rotated to emchristiansen@mcortex.net" in caplog.text


def _profile_layout(tmp_path, monkeypatch, *, profile_rows: list[dict] | None):
    """Custom root <tmp>/hroot with profile <tmp>/hroot/profiles/p1 (root = HERMES_HOME.parent.parent)."""
    import agent.credential_display_names as names
    monkeypatch.setattr(names, "_cache", {})
    root = tmp_path / "hroot"
    profile = root / "profiles" / "p1"
    profile.mkdir(parents=True)
    (root / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {"openai-codex": [
        _row("8e990c", "foglinesupply", _codex_jwt(ACCT_A), 0)]}}))
    _write_names(root, {"openai-codex": {
        "8e990c": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"}}})
    if profile_rows is not None:
        (profile / "auth.json").write_text(json.dumps({"version": 1, "providers": {},
                                                       "credential_pool": {"openai-codex": profile_rows}}))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return root, profile


def test_profile_borrowing_root_rows_uses_root_names(tmp_path, monkeypatch, capsys):
    root, _profile = _profile_layout(tmp_path, monkeypatch, profile_rows=None)
    from agent.credential_display_names import owning_store_dir
    assert owning_store_dir("openai-codex") == root
    assert "#1  eric@foglinesupply.com" in _list(capsys)


def test_profile_owning_rows_uses_only_its_own_names(tmp_path, monkeypatch, capsys):
    root, profile = _profile_layout(tmp_path, monkeypatch, profile_rows=[
        _row("p00001", "profile-login", _codex_jwt(ACCT_B, subject="user-2"), 0)])
    from agent.credential_display_names import owning_store_dir
    assert owning_store_dir("openai-codex") == profile
    out = _list(capsys)
    # Root's mapping must not leak into the profile: no root names, no root unknown-id noise.
    assert "#1  profile-login" in out and "eric@foglinesupply.com" not in out and "unknown" not in out
    _write_names(profile, {"openai-codex": {
        "p00001": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_B, "expected_subject": "user-2"}}})
    assert "#1  emchristiansen@mcortex.net" in _list(capsys)
    # ...and the profile's mapping does not leak back into root's own listing.
    monkeypatch.setenv("HERMES_HOME", str(root))
    out = _list(capsys)
    assert "#1  eric@foglinesupply.com" in out and "emchristiansen@mcortex.net" not in out


def _same_workspace(home) -> None:
    """Both rows are members of ONE workspace: same chatgpt_account_id, different subjects."""
    store = json.loads((home / "auth.json").read_text())
    store["credential_pool"]["openai-codex"][1]["access_token"] = _codex_jwt(ACCT_A, subject="user-2")
    (home / "auth.json").write_text(json.dumps(store))


def test_same_workspace_members_need_their_own_subject(home, capsys):
    _same_workspace(home)
    _write_names(home, {"openai-codex": {
        "8e990c": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
        "63dfc7": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_A, "expected_subject": "user-2"},
    }})
    out = _list(capsys)
    assert "#1  eric@foglinesupply.com" in out and "#2  emchristiansen@mcortex.net" in out


def test_same_workspace_swapped_names_are_rejected_by_subject(home, capsys):
    # The account claim matches on both rows; only the subject tells the members apart.
    _same_workspace(home)
    _write_names(home, {"openai-codex": {
        "8e990c": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_A, "expected_subject": "user-2"},
        "63dfc7": {"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
    }})
    out = _list(capsys)
    assert "foglinesupply [display name rejected: subject claim mismatch]" in out
    assert "emchristiansen-mcortex [display name rejected: subject claim mismatch]" in out
    assert "@" not in out.split("credentials):", 1)[1].split("External CLI", 1)[0]


def test_same_workspace_account_only_mapping_is_rejected(home, capsys):
    _same_workspace(home)
    _write_names(home, {"openai-codex": {
        "63dfc7": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_A},
    }})
    out = _list(capsys)
    assert "emchristiansen-mcortex [display name rejected: mapping has no expected_subject]" in out
    assert "emchristiansen@mcortex.net" not in out


@pytest.mark.parametrize("spec,shown", [
    ({"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_A, "expected_subject": "user-2"},
     "emchristiansen@mcortex.net"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A, "expected_subject": "user-1"},
     "emchristiansen-mcortex [display name rejected: subject claim mismatch]"),
    ({"display_name": "eric@foglinesupply.com", "expected_account_id": ACCT_A},
     "emchristiansen-mcortex [display name rejected: mapping has no expected_subject]"),
])
def test_same_workspace_selection_log_names_only_the_matching_member(home, caplog, spec, shown):
    from agent.credential_pool import load_pool
    _same_workspace(home)
    _write_names(home, {"openai-codex": {"63dfc7": spec}})
    pool = load_pool("openai-codex")
    assert pool.select().id == "8e990c"
    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        nxt = pool.mark_exhausted_and_rotate(status_code=429, credential_id="8e990c")
    assert nxt is not None and nxt.id == "63dfc7"
    assert f"credential pool: rotated to {shown}" in caplog.text
    if "rejected" in shown:
        assert spec["display_name"] not in caplog.text


def test_selection_log_rejects_row_without_subject(home, caplog):
    from agent.credential_pool import load_pool
    store = json.loads((home / "auth.json").read_text())
    store["credential_pool"]["openai-codex"][1]["access_token"] = _codex_jwt(ACCT_B, subject=None)
    (home / "auth.json").write_text(json.dumps(store))
    _write_names(home, {"openai-codex": {
        "63dfc7": {"display_name": "emchristiansen@mcortex.net", "expected_account_id": ACCT_B, "expected_subject": "user-2"},
    }})
    pool = load_pool("openai-codex")
    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        nxt = pool.mark_exhausted_and_rotate(status_code=429, credential_id="8e990c")
    assert nxt is not None and nxt.id == "63dfc7"
    assert "rotated to emchristiansen-mcortex [display name rejected: row lacks account or subject claim]" in caplog.text
    assert "emchristiansen@mcortex.net" not in caplog.text


def _set_token(home, index: int, token: str) -> None:
    store = json.loads((home / "auth.json").read_text())
    store["credential_pool"]["openai-codex"][index]["access_token"] = token
    (home / "auth.json").write_text(json.dumps(store))


# Identifiers are opaque: values that differ only by whitespace are different principals.
NONCANONICAL = [
    pytest.param((ACCT_A, " user-1 "), (ACCT_A, "user-1"), "subject claim mismatch", id="token-sub-padded"),
    pytest.param((ACCT_A, "user-1\n"), (ACCT_A, "user-1"), "subject claim mismatch", id="token-sub-newline"),
    pytest.param((" " + ACCT_A, "user-1"), (ACCT_A, "user-1"), "account claim mismatch", id="token-account-padded"),
    pytest.param((ACCT_A + "\t", "user-1"), (ACCT_A, "user-1"), "account claim mismatch", id="token-account-tab"),
    pytest.param((ACCT_A, "user-1"), (ACCT_A, " user-1 "), "mapping expected_subject has surrounding whitespace",
                 id="mapping-sub-padded"),
    pytest.param((ACCT_A, "user-1"), (ACCT_A + " ", "user-1"),
                 "mapping expected_account_id has surrounding whitespace", id="mapping-account-padded"),
    pytest.param((ACCT_A, "   "), (ACCT_A, "user-1"), "row lacks account or subject claim", id="token-sub-blank"),
    pytest.param(("  ", "user-1"), (ACCT_A, "user-1"), "row lacks account or subject claim", id="token-account-blank"),
]


@pytest.mark.parametrize("token,mapping,reason", NONCANONICAL)
def test_noncanonical_principal_is_rejected_in_listing(home, capsys, token, mapping, reason):
    _set_token(home, 0, _codex_jwt(*token))
    _write_names(home, {"openai-codex": {"8e990c": {
        "display_name": "wrong@example.invalid", "expected_account_id": mapping[0], "expected_subject": mapping[1]}}})
    out = _list(capsys)
    assert f"#1  foglinesupply [display name rejected: {reason}]" in out
    assert "wrong@example.invalid" not in out


@pytest.mark.parametrize("token,mapping,reason", NONCANONICAL)
def test_noncanonical_principal_is_rejected_in_selection_log(home, caplog, token, mapping, reason):
    from agent.credential_pool import load_pool
    # Row 2 keeps a principal distinct from row 1 (user-2), so rotation itself is unaffected.
    token, mapping = (tuple(v.replace("user-1", "user-2") for v in pair) for pair in (token, mapping))
    _set_token(home, 1, _codex_jwt(*token))
    _write_names(home, {"openai-codex": {"63dfc7": {
        "display_name": "wrong@example.invalid", "expected_account_id": mapping[0], "expected_subject": mapping[1]}}})
    pool = load_pool("openai-codex")
    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        nxt = pool.mark_exhausted_and_rotate(status_code=429, credential_id="8e990c")
    assert nxt is not None and nxt.id == "63dfc7"
    assert f"rotated to emchristiansen-mcortex [display name rejected: {reason}]" in caplog.text
    assert "wrong@example.invalid" not in caplog.text


def test_display_fence_leaves_selection_identity_helper_unchanged():
    # Credential selection keeps its inherited trimming; only the display fence compares exactly.
    from agent.credential_pool import _codex_principal_identity
    assert _codex_principal_identity(_codex_jwt(" " + ACCT_A + " ", " user-1 ")) == (ACCT_A, "user-1")
