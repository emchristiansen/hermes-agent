"""``browser_vault_fill(require_empty=True)``: a password field that already holds a value, or a page
without one, is refused before the backend resolves anything; the fill script re-checks emptiness in
the same evaluation as the write. Without the flag the fill is unchanged. Secrets are dummy values in a
temporary local vault; the page is a recording stand-in for the supervisor."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_login_classifier import PAGE_CLOCK_JS, build_fill_js
from agent.vault_store import VaultStore, normalize_origin

_PASSWORD = "dummy-password-9"
_ORIGIN = "https://mail.example"


def _controls(has_value: bool):
    return [
        {"autocomplete": "username", "formIndex": 0, "index": 0, "name": "user", "type": "email", "hasValue": True},
        {"autocomplete": "current-password", "formIndex": 0, "index": 1, "name": "pw", "type": "password",
         "hasValue": has_value},
    ]


class _Page:
    def __init__(self, controls, fill_result=None):
        self.controls, self.writes, self.cleared = controls, [], 0
        self.fill_result = fill_result or {"filled": 1}

    def focus_page(self, origin, accept=None):
        return {"ok": origin == _ORIGIN, "url": _ORIGIN + "/login"}

    def evaluate_runtime(self, expression, **_):
        if expression == PAGE_CLOCK_JS:  # the page-clock read before the fill script
            return {"ok": True, "result": json.dumps({"timeOrigin": 1.0, "now": 0.0})}
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": _ORIGIN + "/login"}
        if "expectedOrigin" in expression:
            self.writes.append(expression)
            return {"ok": True, "result": json.dumps(self.fill_result)}
        if 'removeAttribute("data-hermes-vault-slot")' in expression:
            self.cleared += 1
            return {"ok": True, "result": True}
        return {"ok": True, "result": json.dumps(self.controls)}


class _Recording(LocalLoginBackend):
    def __init__(self):
        self.resolved = []

    def resolve_password(self, handle):
        self.resolved.append("password")
        return super().resolve_password(handle)


@pytest.fixture()
def vault(tmp_path):
    from agent import redact

    store = VaultStore(base_dir=tmp_path / "vault")
    backend = _Recording()
    meta = store.add_item("login", "Mail", {"identifier_type": "email", "identifier": "a@mail.example",
                                            "password": _PASSWORD}, origin=_ORIGIN)
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.backend_for_handle", return_value=backend):
            yield meta, backend
    finally:
        redact.clear_vault_redaction_values()


def _on_page(page):
    """The page is the task's registered supervisor; ``supervisor_only()`` (as ``hermes vault fill`` runs)
    keeps every access on it."""
    from contextlib import ExitStack

    from tools import browser_vault_tool
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    stack = ExitStack()
    stack.enter_context(patch.object(SUPERVISOR_REGISTRY, "get", return_value=page))
    stack.enter_context(browser_vault_tool.supervisor_only())
    return stack


def _fill(meta, page, **kwargs):
    from tools import browser_vault_tool

    with _on_page(page):
        return json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", **kwargs))


def test_non_empty_password_field_is_refused_before_resolution(vault):
    meta, backend = vault
    page = _Page(_controls(has_value=True))
    out = _fill(meta, page, require_empty=True)
    assert out["success"] is False and out["error_type"] == "field_not_empty"
    assert backend.resolved == [] and page.writes == []
    assert page.cleared == 1  # the inspection stamps are removed on the refusal


def test_page_without_password_field_is_refused_before_resolution(vault):
    meta, backend = vault
    page = _Page(_controls(has_value=False)[:1])
    out = _fill(meta, page, require_empty=True)
    assert out["error_type"] == "no_password_field"
    assert backend.resolved == [] and page.writes == []


def test_empty_field_fills_with_in_page_recheck(vault):
    meta, backend = vault
    page = _Page(_controls(has_value=False))
    out = _fill(meta, page, require_empty=True)
    assert out["success"] is True and out["filled_fields"] == 1
    assert backend.resolved == ["password"]
    [write] = page.writes
    assert "field_not_empty" in write and "__REQUIRE_EMPTY__" not in write
    assert _PASSWORD not in json.dumps(out)


def test_value_appearing_before_the_write_is_reported_as_refused(vault):
    meta, _ = vault
    page = _Page(_controls(has_value=False), fill_result={"refused": "field_not_empty"})
    out = _fill(meta, page, require_empty=True)
    assert out["success"] is False and out["error_type"] == "field_not_empty"


def test_without_the_flag_a_filled_field_is_still_overwritten_as_before(vault):
    meta, backend = vault
    page = _Page(_controls(has_value=True))
    out = _fill(meta, page)
    assert out["success"] is True and backend.resolved == ["password"]


def test_fill_script_checks_emptiness_only_when_asked():
    fills = [{"index": 1, "token": "current-password", "value": "x"}]
    assert "const requireEmpty = true;" in build_fill_js(fills, _ORIGIN, "n", require_empty=True)
    assert "const requireEmpty = false;" in build_fill_js(fills, _ORIGIN, "n")
    assert normalize_origin(_ORIGIN) == _ORIGIN


def test_explicit_backend_is_the_only_store_consulted(vault, tmp_path):
    from tools import browser_vault_tool

    meta, backend = vault
    page = _Page(_controls(has_value=False))
    with patch("agent.vault_backends.backend_for_handle", side_effect=AssertionError("enumerated backends")), \
         _on_page(page):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", require_empty=True,
                                                               backend=backend))
    assert out["success"] is True and backend.resolved == ["password"]


def test_explicit_backend_refuses_a_handle_it_does_not_own(vault):
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from tools import browser_vault_tool

    meta, backend = vault
    page = _Page(_controls(has_value=False))
    with _on_page(page):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t",
                                                               backend=OnePasswordLoginBackend({})))
    assert out["error_type"] == "backend_mismatch" and backend.resolved == [] and page.writes == []


def test_supervisor_only_never_looks_up_another_page(vault):
    """No session lookup or on-demand attach, no plugin page backend and no preview page: only the
    registered supervisor."""
    from tools import browser_vault_tool
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    meta, backend = vault
    with patch.object(SUPERVISOR_REGISTRY, "get", return_value=None), \
         patch("tools.browser_session.session_for_task", side_effect=AssertionError("session looked up")), \
         patch("agent.browser_page_registry.backend_for_task", side_effect=AssertionError("page backend used")), \
         patch("tools.browser_vault_preview.active_page", side_effect=AssertionError("preview page used")), \
         browser_vault_tool.supervisor_only():
        assert browser_vault_tool._ensure_supervisor("t") is None
        assert browser_vault_tool._eval_js("t", "1")["success"] is False
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", require_empty=True,
                                                               backend=backend))
    assert out["success"] is False and backend.resolved == []


def test_explicit_locked_backend_is_refused_without_unlock_or_lookup(vault):
    from tools import browser_vault_tool

    meta, _ = vault

    class _Locked(_Recording):
        needs_unlock = True

        def is_unlocked(self):
            return False

    locked = _Locked()
    page = _Page(_controls(has_value=False))
    with patch.object(browser_vault_tool, "browser_vault_unlock", side_effect=AssertionError("unlock")), \
         patch("agent.vault_backends.backend_for_handle", side_effect=AssertionError("lookup")), \
         _on_page(page):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", require_empty=True,
                                                               backend=locked))
    assert out["error_type"] == "unlock_required" and locked.resolved == [] and page.writes == []
