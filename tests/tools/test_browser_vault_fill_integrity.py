"""The vault fill writes exactly the stored value, only into an actual empty password field, and leaves no
slot stamp behind on any exit.

* The fill script is rendered in one pass, so a value containing a template placeholder, quotes,
  backslashes, slashes or non-ASCII text reaches the page byte for byte.
* ``el.focus()`` runs page focus listeners synchronously; the target and its emptiness are checked again
  after it, immediately before the write.
* The password target is an actual ``type=password`` control, chosen once for the gate and the writer, so
  an ``autocomplete=current-password`` text box never causes a resolve.
* The inspection's stamps are removed on every exit after it: refusals, no field, exceptions, success.

The unit tests use a recording stand-in for the page. The ``integration`` tests run ``hermes vault fill``
(``hermes_cli.vault._cmd_fill``) against a real headless Chrome; they are opt-in like
``test_browser_supervisor.py`` (``HERMES_E2E_BROWSER=1`` and ``-m integration``). Secrets are dummy values
in a temporary local vault."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_login_classifier import (
    ClassifiedLoginControl,
    LoginControl,
    _render_js,
    best_password_control,
    build_clear_stamps_js,
    build_fill_js,
    build_inspection_js,
    select_password_fill,
)
from agent.vault_store import VaultStore
from tests.waiting import wait_until

# Every placeholder the fill and inspection templates use, plus quoting, escaping and non-ASCII text.
_HARD_VALUES = [
    "REVIEW-__REQUIRE_EMPTY__-tail",
    "a__NONCE__b",
    "__FILLS__",
    "x__EXPECTED_ORIGIN__y",
    "__UNKNOWN__",
    "q'uo\"te`s",
    "back\\slash\\\\double",
    "sl/ash</script>",
    "uni-é中\U0001f511-\u2028-\u2029-end",
    "$1$&$'$`",
    " lead and trail ",
]


def _fills_literal(js: str) -> list:
    [line] = [ln for ln in js.splitlines() if ln.strip().startswith("const fills = ")]
    return json.loads(line.strip()[len("const fills = "):].rstrip(";"))


@pytest.mark.parametrize("value", _HARD_VALUES)
def test_fill_script_carries_the_full_value_and_nothing_else_changes(value):
    fills = [{"index": 3, "token": "current-password", "value": value}]
    js = build_fill_js(fills, "https://mail.example", "0123abcd", require_empty=True)
    assert _fills_literal(js) == fills
    plain = build_fill_js([{"index": 3, "token": "current-password", "value": "PLAIN"}], "https://mail.example",
                          "0123abcd", require_empty=True)
    # The only bytes that differ are the value's own JSON literal: no placeholder in the value was replaced.
    assert js == plain.replace(json.dumps("PLAIN"), json.dumps(value))
    assert "\u2028" not in js and "\u2029" not in js  # line separators stay escaped inside the JS literal


@pytest.mark.parametrize("value", _HARD_VALUES)
def test_origin_and_nonce_are_data_too(value):
    js = build_fill_js([], value, value)
    assert f"const expectedOrigin = {json.dumps(value)};" in js
    assert f"const nonce = {json.dumps(value)};" in js
    assert f"const nonce = {json.dumps(value)};" in build_inspection_js(value)


def test_template_and_values_must_match():
    with pytest.raises(ValueError):
        _render_js("(() => __A__)()", a=1, b=2)
    with pytest.raises(ValueError):
        _render_js("(() => __A__ + __B__)()", a=1)


def test_stamp_cleanup_is_scoped_to_its_own_inspection():
    js = build_clear_stamps_js("feedc0de")
    assert 'return retireInspection("feedc0de");' in js
    assert 'const prefix = nonce + ":";' in js and "startsWith(prefix)" in js
    # The retained original nodes are cleaned too, connected or not, before the tree walk.
    assert js.index("registry.nodes.forEach") < js.index('querySelectorAll("*")', js.index("function retireInspection"))


def test_fill_script_rechecks_after_focus_before_the_write():
    js = build_fill_js([{"index": 0, "token": "current-password", "value": "v"}], "https://m.example", "n",
                       require_empty=True)
    focus = js.index("el.focus();")
    write = js.index("setter.set.call(el, f.value)")
    assert focus < js.index("const still = el.isConnected", focus) < write
    assert focus < js.index('if (requireEmpty && current(el).length > 0) return JSON.stringify({ refused: "field_not_empty" });', focus) < write
    assert js.index("} finally {") > write
    assert js.index("retireInspection(nonce);", js.index("} finally {")) > write and "startsWith(prefix)" in js


def _ctrl(**kw):
    base = dict(autocomplete="", form_index=0, index=0, label="", name="", type="text")
    base.update(kw)
    return LoginControl(**base)


def test_a_text_box_claiming_current_password_is_never_the_password_target():
    text = ClassifiedLoginControl(_ctrl(index=0, autocomplete="current-password"), 100, "current-password")
    real = ClassifiedLoginControl(_ctrl(index=1, type="password"), 90, "current-password")
    assert best_password_control([text]) is None and select_password_fill([text], "p") == []
    assert best_password_control([text, real]) is real
    assert select_password_fill([text, real], "p") == [{"index": 1, "token": "current-password", "value": "p"}]


# ── browser_vault_fill on a recording page ──────────────────────────────────

_ORIGIN = "https://mail.example"
_PASSWORD = "dummy-password-integrity"


class _Page:
    """Answers the supervisor calls browser_vault_fill makes; records writes and stamp clean-ups."""

    def __init__(self, controls, fill_result=None):
        self.controls, self.writes, self.cleared, self.inspections = controls, [], [], []
        self.fill_result = fill_result or {"filled": 1}

    def focus_page(self, origin, accept=None):
        return {"ok": origin == _ORIGIN, "url": _ORIGIN + "/login"}

    def evaluate_runtime(self, expression, **_):
        from agent.vault_login_classifier import PAGE_CLOCK_JS

        if expression == PAGE_CLOCK_JS:  # the page-clock read before the fill script
            return {"ok": True, "result": json.dumps({"timeOrigin": 1.0, "now": 0.0})}
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": _ORIGIN + "/login"}
        if "expectedOrigin" in expression:
            self.writes.append(expression)
            return {"ok": True, "result": json.dumps(self.fill_result)}
        if "return retireInspection(" in expression:
            self.cleared.append(expression)
            return {"ok": True, "result": 0}
        self.inspections.append(expression)
        return {"ok": True, "result": json.dumps(self.controls)}


class _Recording(LocalLoginBackend):
    def __init__(self, raises=None):
        self.resolved, self.raises = [], raises

    def resolve_password(self, handle):
        self.resolved.append(handle)
        if self.raises:
            raise self.raises
        return super().resolve_password(handle)


@pytest.fixture()
def item(tmp_path):
    from agent import redact

    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "Mail", {"identifier_type": "email", "identifier": "a@mail.example",
                                            "password": _PASSWORD}, origin=_ORIGIN)
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store):
            yield meta
    finally:
        redact.clear_vault_redaction_values()


def _fill(meta, page, backend):
    from tools import browser_vault_tool
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    with patch.object(SUPERVISOR_REGISTRY, "get", return_value=page), browser_vault_tool.supervisor_only():
        return json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", require_empty=True,
                                                                backend=backend))


_EMAIL = {"autocomplete": "username", "formIndex": 0, "index": 0, "name": "e", "type": "email", "hasValue": True}
_PW = {"autocomplete": "current-password", "formIndex": 0, "index": 1, "name": "p", "type": "password",
       "hasValue": False}
_FAKE_PW = {"autocomplete": "current-password", "formIndex": 0, "index": 0, "name": "e", "type": "text",
            "hasValue": False}


@pytest.mark.parametrize(
    "controls, fill_result, error_type, resolves",
    [
        ([], None, None, False),  # no login control at all
        ([_EMAIL], None, "no_password_field", False),
        ([_FAKE_PW], None, "no_password_field", False),  # text box claiming current-password
        ([_EMAIL, dict(_PW, hasValue=True)], None, "field_not_empty", False),
        ([_EMAIL, _PW], {"refused": "field_not_empty"}, "field_not_empty", True),  # value appeared in the page
        ([_EMAIL, _PW], {"refused": "target_changed"}, "target_changed", True),
        ([_EMAIL, _PW], {"refused": "origin_changed", "found": "https://other.example"}, "origin_changed", True),
        ([_EMAIL, _PW], {"refused": "value_mismatch", "intact": True, "cleared": True}, "value_mismatch", True),
        ([_EMAIL, _PW], {"refused": "value_mismatch", "intact": True, "cleared": False}, "value_mismatch", True),
        ([_EMAIL, _PW], {"filled": 0}, None, True),
        ([_EMAIL, _PW], {"filled": 1}, None, True),
    ],
)
def test_every_exit_after_inspection_removes_its_stamps(item, controls, fill_result, error_type, resolves):
    page, backend = _Page(controls, fill_result), _Recording()
    out = _fill(item, page, backend)
    assert out.get("error_type") == error_type
    assert out["success"] is (fill_result == {"filled": 1})
    assert ("next" in out) is out["success"]  # no "Submit" hint when nothing was filled
    assert backend.resolved == ([item.id] if resolves else [])
    assert len(page.writes) == (1 if resolves else 0)
    [inspection] = page.inspections
    if fill_result is None:  # the fill script never ran: the tool removes this inspection's stamps
        [cleanup] = page.cleared
        [nonce] = re.findall(r'const nonce = "([0-9a-f]+)";', inspection)
        assert f'return retireInspection("{nonce}");' in cleanup  # only this inspection's stamps
    else:  # the fill script ran to completion and removed them in its own finally
        assert page.cleared == []
    assert _PASSWORD not in json.dumps(out)


@pytest.mark.parametrize(
    "fill_result, field_cleared",
    [({"refused": "value_mismatch", "intact": True, "cleared": True}, True),
     ({"refused": "value_mismatch", "intact": True, "cleared": False}, False),  # the page refilled the field
     ({"refused": "value_mismatch", "intact": False, "cleared": False}, False),  # original removed or replaced
     ({"refused": "value_mismatch", "intact": False, "cleared": True}, False),  # inconsistent: never trusted
     ({"refused": "value_mismatch", "cleared": True}, False),  # no intact report: not established
     ({"refused": "value_mismatch"}, False)],  # no report of the final state: not established
)
def test_a_mismatch_says_nothing_is_filled_only_when_the_page_reported_the_field_empty(item, fill_result,
                                                                                     field_cleared):
    """The rollback's own input/change notifications run page code; the fill script reports whether the field
    is empty after them, and only then does the result say that nothing is filled."""
    out = _fill(item, _Page([_EMAIL, _PW], fill_result), _Recording())
    assert out["success"] is False and out["error_type"] == "value_mismatch" and "next" not in out
    assert out["field_cleared"] is field_cleared
    assert ("nothing is filled" in out["error"]) is field_cleared
    if not field_cleared:
        assert "not established" in out["error"]
        refilled = fill_result.get("cleared") is False and fill_result.get("intact") is True
        assert ("the field is not empty" in out["error"]) is refilled
    assert _PASSWORD not in json.dumps(out)


def test_an_exception_while_resolving_still_removes_the_stamps(item):
    page, backend = _Page([_EMAIL, _PW]), _Recording(raises=RuntimeError("backend exploded"))
    with pytest.raises(RuntimeError):
        _fill(item, page, backend)
    assert len(page.cleared) == 1 and page.writes == []


# ── hermes vault fill against a real Chrome ─────────────────────────────────

_FORM = (b'<html><body><form onsubmit="window.submits++;return false">'
         b'<input id="e" type="email" value="human@example.invalid">'
         b'<input id="p" type="password" autocomplete="current-password"><button>Submit</button></form>'
         b'<script>window.submits=0</script></body></html>')

_STATE = ("JSON.stringify({slots:document.querySelectorAll('[data-hermes-vault-slot]').length,"
          "email:document.getElementById('e').value,pw:document.getElementById('p').value,submits:window.submits})")


def _chrome_binary():
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(candidate):
            return shutil.which(candidate)
    return None


def _integration(fn):
    for mark in (
        pytest.mark.integration,
        pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                           reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in"),
        pytest.mark.skipif(_chrome_binary() is None, reason="Chrome/Chromium not installed"),
    ):
        fn = mark(fn)
    return fn


@pytest.fixture
def browser():
    """A headless Chrome on a fresh profile with one login page served from 127.0.0.1."""

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_FORM)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    # Chrome gets a complete disposable environment of its own (HOME, XDG directories, TMPDIR, no D-Bus
    # session, no keyring, no credentials), not the runner's: a crash report, keyring entry or profile
    # write can only land in this directory.
    root = Path(tempfile.mkdtemp(prefix="hvf-"))
    env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "TZ": "UTC",
           "DBUS_SESSION_BUS_ADDRESS": "disabled:"}
    for var, sub in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                     ("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state"), ("XDG_RUNTIME_DIR", "runtime"),
                     ("TMPDIR", "tmp")):
        (root / sub).mkdir(mode=0o700)
        env[var] = str(root / sub)
    profile = root / "profile"
    proc = subprocess.Popen(
        [_chrome_binary(), "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--disable-extensions",
         "--disable-crash-reporter", "--disable-breakpad", "--password-store=basic", origin + "/login"],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        http, page = wait_until(lambda: _login_target(proc, profile, origin), timeout=15,
                                what="Chrome exposing the login page over CDP")
        wait_until(lambda: _ready(page), timeout=15, what="the login form")
        yield SimpleNamespace(http=http, origin=origin, page=page)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        server.shutdown()
        server.server_close()
        shutil.rmtree(root, ignore_errors=True)


def _login_target(proc, profile, origin):
    """``(http endpoint, page target)`` once Chrome lists the login page, else None."""
    if proc.poll() is not None:
        pytest.fail(f"Chrome exited ({proc.returncode}) before exposing CDP")
    try:
        port = int((profile / "DevToolsActivePort").read_text().splitlines()[0])
        http = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(http + "/json/list", timeout=1) as r:
            targets = json.load(r)
    except (OSError, ValueError, IndexError):  # not written yet / not listening yet
        return None
    page = next((t for t in targets if t.get("type") == "page" and t["url"].startswith(origin)), None)
    return (http, page) if page else None


def _ready(page):
    try:
        return _evaluate(page, "document.readyState === 'complete' && !!document.getElementById('p')") is True
    except (OSError, AssertionError):
        return False


def _evaluate(page, expression):
    from websockets.sync.client import connect

    with connect(page["webSocketDebuggerUrl"], open_timeout=5) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                            "params": {"expression": expression, "returnByValue": True}}))
        while True:
            msg = json.loads(ws.recv(timeout=5))
            if msg.get("id") == 1:
                assert "exceptionDetails" not in msg["result"], msg
                return msg["result"]["result"].get("value")


def _state(browser):
    return json.loads(_evaluate(browser.page, _STATE))


def _cli_fill(browser, store, password, capsys, during_resolve=None):
    """Run ``hermes vault fill`` (the real CLI command, supervisor and fill) on the login page."""
    from agent.redact import clear_vault_redaction_values
    from hermes_cli import vault as vault_cli

    meta = store.add_item("login", "Mail", {"identifier_type": "email", "identifier": "a@x.invalid",
                                            "password": password}, origin=browser.origin)
    resolved = []
    original = LocalLoginBackend.resolve_password

    def resolve(self, handle):
        resolved.append(handle)
        value = original(self, handle)
        if during_resolve:
            _evaluate(browser.page, during_resolve)
        return value

    args = SimpleNamespace(handle=meta.id, cdp=browser.http, target_id=browser.page["id"], backend="local")
    with ExitStack() as stack:
        stack.enter_context(patch("agent.vault_store.get_vault_store", return_value=store))
        stack.enter_context(patch.object(LocalLoginBackend, "resolve_password", resolve))
        with pytest.raises(SystemExit) as exc:
            vault_cli._cmd_fill(args)
    clear_vault_redaction_values()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return exc.value.code, out, resolved


@pytest.fixture
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


@_integration
@pytest.mark.parametrize("value", ["REVIEW-full-value-ordinary-7", *_HARD_VALUES])
def test_cli_fills_the_full_stored_value(browser, store, capsys, value):
    _evaluate(browser.page, "document.getElementById('p').value=''")
    code, out, _ = _cli_fill(browser, store, value, capsys)
    assert code == 0 and out["success"] is True and out["filled_fields"] == 1
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": value, "submits": 0}
    assert value not in json.dumps(out)



@_integration
def test_cli_does_not_overwrite_a_value_a_focus_listener_writes(browser, store, capsys):
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "window.focusFired=0;document.getElementById('p').addEventListener('focus',"
                            "function(){window.focusFired++;this.value='HUMAN-focus-value'},{once:true})")
    code, out, _ = _cli_fill(browser, store, "vault-value-1", capsys)
    assert code == 1 and out["error_type"] == "field_not_empty"
    assert _evaluate(browser.page, "window.focusFired") == 1
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "HUMAN-focus-value",
                               "submits": 0}



@_integration
def test_cli_refuses_when_a_focus_listener_changes_the_target(browser, store, capsys):
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "document.getElementById('p').addEventListener('focus',"
                            "function(){this.type='text'},{once:true})")
    code, out, _ = _cli_fill(browser, store, "vault-value-2", capsys)
    assert code == 1 and out["error_type"] == "target_changed"
    state = _state(browser)
    assert state["pw"] == "" and state["slots"] == 0
    _evaluate(browser.page, "document.getElementById('p').type='password'")



@_integration
def test_cli_refuses_a_value_typed_during_resolution_and_leaves_no_stamps(browser, store, capsys):
    _evaluate(browser.page, "document.getElementById('p').value=''")
    code, out, resolved = _cli_fill(browser, store, "vault-value-3", capsys,
                                    during_resolve="document.getElementById('p').value='HUMAN-during-resolution'")
    assert code == 1 and out["error_type"] == "field_not_empty" and len(resolved) == 1
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "HUMAN-during-resolution",
                               "submits": 0}



@_integration
@pytest.mark.parametrize(
    "setup, error_type",
    [
        # no usable control at all
        ("document.getElementById('p').disabled=true;document.getElementById('e').type='hidden'", None),
        # the only current-password control is a text box
        ("document.getElementById('p').disabled=true;document.getElementById('e').type='text';"
         "document.getElementById('e').autocomplete='current-password';document.getElementById('e').value=''",
         "no_password_field"),
    ],
)
def test_cli_without_an_actual_password_field_resolves_nothing_and_leaves_no_stamps(browser, store, capsys, setup,
                                                                                    error_type):
    _evaluate(browser.page, "document.getElementById('p').value='';" + setup)
    code, out, resolved = _cli_fill(browser, store, "vault-value-4", capsys)
    assert code == 1 and out["success"] is False and out.get("error_type") == error_type
    assert resolved == [] and "next" not in out
    assert _evaluate(browser.page, "document.querySelectorAll('[data-hermes-vault-slot]').length") == 0
    assert _evaluate(browser.page, "document.getElementById('p').value") == ""



@_integration
def test_cli_fills_an_empty_password_field_inside_an_open_shadow_root(browser, store, capsys):
    """The exact-target fill keeps this tree's shadow-root reach: the inspection, the emptiness gate, the
    recheck after focus and the stamp cleanup all see a password control rendered inside an open shadow
    root, and the composed input event reaches a listener on its host."""
    _evaluate(browser.page, "document.getElementById('p').remove();const h=document.createElement('div');h.id='host';"
                            "document.querySelector('form').appendChild(h);const r=h.attachShadow({mode:'open'});"
                            "r.innerHTML='<input id=\"sp\" type=\"password\" autocomplete=\"current-password\">';"
                            "window.hostInputs=0;h.addEventListener('input',function(){window.hostInputs++})")
    shadow = "document.getElementById('host').shadowRoot"
    code, out, _ = _cli_fill(browser, store, "shadow-value-5", capsys)
    assert code == 0 and out["success"] is True and out["filled_fields"] == 1
    assert _evaluate(browser.page, f"{shadow}.getElementById('sp').value") == "shadow-value-5"
    assert _evaluate(browser.page, "window.hostInputs") == 1
    assert _evaluate(browser.page, f"{shadow}.querySelectorAll('[data-hermes-vault-slot]').length") == 0
    assert _evaluate(browser.page, "window.submits") == 0
    _evaluate(browser.page, f"{shadow}.getElementById('sp').value='HUMAN-shadow-value'")
    code, out, resolved = _cli_fill(browser, store, "shadow-value-6", capsys)
    assert code == 1 and out["error_type"] == "field_not_empty" and resolved == []
    assert _evaluate(browser.page, f"{shadow}.getElementById('sp').value") == "HUMAN-shadow-value"
    assert _evaluate(browser.page, f"{shadow}.querySelectorAll('[data-hermes-vault-slot]').length") == 0


@_integration
@pytest.mark.parametrize("value", ["line-one\nline-two", "carriage\rreturn", "crlf\r\nvalue", "trailing-newline\n"])
def test_cli_never_reports_a_value_the_field_normalized_and_clears_it(browser, store, capsys, value):
    """A single-line input strips CR/LF: the field then holds something other than the stored value. The
    fill compares the field's value with the exact stored value after writing; on a mismatch it clears
    what it wrote, removes its stamps and refuses (never success, never a Submit hint)."""
    _evaluate(browser.page, "document.getElementById('p').value=''")
    code, out, resolved = _cli_fill(browser, store, value, capsys)
    assert code == 1 and out["success"] is False and out["error_type"] == "value_mismatch", out
    assert len(resolved) == 1 and "next" not in out and value not in json.dumps(out)
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}


_REINSERT = ("const p=document.getElementById('p');const parent=p.parentNode,next=p.nextSibling;"
             "p.remove();parent.insertBefore(p,next);")
_REPLACE = "const p=document.getElementById('p');p.replaceWith(p.cloneNode(true));"


@_integration
@pytest.mark.parametrize("change", [_REINSERT, _REPLACE], ids=["detached_and_reinserted", "replaced_by_a_clone"])
def test_cli_refuses_a_stamped_field_that_left_the_document_before_the_write(browser, store, capsys, change):
    """Between stamping (inspection) and writing, the stamped password node is detached and reinserted, or
    replaced by a clone that carries the stamp. Neither is the verified connected original: nothing is
    written, the refusal is target_changed and no stamp remains."""
    _evaluate(browser.page, "document.getElementById('p').value=''")
    code, out, resolved = _cli_fill(browser, store, "vault-value-7", capsys, during_resolve=change)
    assert code == 1 and out["success"] is False and out["error_type"] == "target_changed", out
    assert len(resolved) == 1 and "next" not in out
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}


@_integration
def test_cli_refuses_when_a_focus_listener_detaches_and_reinserts_the_field(browser, store, capsys):
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "document.getElementById('p').addEventListener('focus',function(){const parent=this.parentNode,"
                            "next=this.nextSibling;this.remove();parent.insertBefore(this,next)},{once:true})")
    code, out, _ = _cli_fill(browser, store, "vault-value-8", capsys)
    assert code == 1 and out["error_type"] == "target_changed", out
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}


# ── stamp cleanup reaches the inspected nodes even while they are detached (review23311 V2) ──────────────

_REGISTRIES = "Object.getOwnPropertyNames(window).filter(x=>x.startsWith('hermesVaultSlots:'))"


@_integration
def test_stamp_cleanup_survives_transient_input_detachment(browser, store, capsys):
    """The page's input listener detaches the filled field and queues its reinsertion: the fill script's
    cleanup runs while the field is detached, and must still remove this inspection's stamp from it.

    Deliberate behaviour change (Fable FA-1): this used to report success. The field left the page between
    the write and the script's return, so it is not the intact inspected field; even though the microtask
    puts it back holding the value, the fill is reported as not established (no Submit hint)."""
    _evaluate(browser.page, "document.getElementById('p').addEventListener('input',function(){const p=this;"
                            "const parent=p.parentNode;p.remove();queueMicrotask(()=>parent.appendChild(p))},{once:true})")
    code, out, _ = _cli_fill(browser, store, "DETACHED-SECRET", capsys)
    actual = _state(browser)
    assert code == 1 and out["success"] is False and out["error_type"] == "not_established", out
    assert "next" not in out and "DETACHED-SECRET" not in json.dumps(out)
    assert actual["slots"] == 0, actual
    assert actual["pw"] == "DETACHED-SECRET"  # back in the page, but left it in between: not established
    assert _evaluate(browser.page, f"{_REGISTRIES}.length") == 0


@_integration
@pytest.mark.parametrize("mode", ["fill-finally", "explicit-clear"])
def test_cleanup_retains_detached_node_reference(browser, mode):
    """Both cleanups (the fill script's finally and the tool's explicit stamp removal) reach a stamped node
    that is detached at cleanup time, through the inspection's retained node references."""
    nonce = "reviewDetached"
    _evaluate(browser.page, build_inspection_js(nonce))
    if mode == "fill-finally":
        _evaluate(browser.page, "window.old=document.getElementById('p');"
                                "old.addEventListener('input',function(){this.remove()},{once:true})")
        out = json.loads(_evaluate(browser.page, build_fill_js(
            [{"index": 1, "token": "current-password", "value": "DETACHED-SECRET"}], browser.origin, nonce,
            require_empty=True)))
        assert out == {"refused": "not_established"}, out  # FA-1: the field left the page (was {"filled": 1})
    else:
        _evaluate(browser.page, "window.old=document.getElementById('p');old.remove()")
        _evaluate(browser.page, build_clear_stamps_js(nonce))
    data = json.loads(_evaluate(browser.page, "JSON.stringify({stamp:old.getAttribute('data-hermes-vault-slot'),"
                                              f"value:old.value,registries:{_REGISTRIES},submits:window.submits}})"))
    assert data["registries"] == [] and data["submits"] == 0, data
    assert data["stamp"] is None, data


@_integration
def test_clear_only_own_nonce(browser):
    _evaluate(browser.page, build_inspection_js("first"))
    _evaluate(browser.page, build_inspection_js("second"))
    _evaluate(browser.page, build_clear_stamps_js("first"))
    assert _evaluate(browser.page, "document.getElementById('p').getAttribute('data-hermes-vault-slot')") == "second:1"
    assert _evaluate(browser.page, "!!window['hermesVaultSlots:second']") is True
    _evaluate(browser.page, build_clear_stamps_js("second"))
    assert _evaluate(browser.page, "document.querySelectorAll('[data-hermes-vault-slot]').length") == 0


@_integration
def test_detached_cleanup_keeps_another_inspections_stamp(browser):
    """A node detached after a second inspection re-stamped it: removing the first inspection's stamps (which
    still holds that node) leaves the second's marker; the second's own cleanup then removes it."""
    _evaluate(browser.page, build_inspection_js("first"))
    _evaluate(browser.page, build_inspection_js("second"))
    _evaluate(browser.page, "window.old=document.getElementById('p');old.remove()")
    _evaluate(browser.page, build_clear_stamps_js("first"))
    assert _evaluate(browser.page, "old.getAttribute('data-hermes-vault-slot')") == "second:1"
    assert _evaluate(browser.page, f"JSON.stringify({_REGISTRIES})") == '["hermesVaultSlots:second"]'
    _evaluate(browser.page, build_clear_stamps_js("second"))
    assert _evaluate(browser.page, "old.getAttribute('data-hermes-vault-slot')") is None
    assert _evaluate(browser.page, f"{_REGISTRIES}.length") == 0


# ── a mismatch reports the field's actual state after the rollback's notifications (review23311 V3) ───────

def _mismatch_state(browser):
    return dict(_state(browser), registries=_evaluate(browser.page, f"{_REGISTRIES}.length"))


@_integration
def test_mismatch_rollback_reports_a_field_the_page_refills_as_not_empty(browser, store, capsys):
    """An input listener rewrites the field on every input event, so it runs during the write's notification
    and again during the rollback's. The refusal stands, without a Submit hint, and must not claim that
    nothing is filled: the field is not empty."""
    _evaluate(browser.page, "document.getElementById('p').value='';window.inputs=0;document.getElementById('p')"
                            ".addEventListener('input',function(){window.inputs++;this.value='PAGE-REWRITE'})")
    code, out, resolved = _cli_fill(browser, store, "before\nafter", capsys)
    assert code == 1 and out["success"] is False and out["error_type"] == "value_mismatch", out
    assert len(resolved) == 1 and "next" not in out and out["field_cleared"] is False
    assert "nothing is filled" not in out["error"] and "not empty" in out["error"], out
    assert _evaluate(browser.page, "window.inputs") == 2  # the write's notification, then the rollback's
    assert _mismatch_state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "PAGE-REWRITE",
                                        "submits": 0, "registries": 0}


@_integration
def test_mismatch_rollback_reports_a_field_a_change_listener_refills_as_not_empty(browser, store, capsys):
    """Only the rollback's change notification puts a value back (a page restoring an emptied field)."""
    _evaluate(browser.page, "document.getElementById('p').value='';document.getElementById('p')"
                            ".addEventListener('change',function(){if(this.value==='')this.value='PAGE-RESTORE'})")
    code, out, _ = _cli_fill(browser, store, "line-one\nline-two", capsys)
    assert code == 1 and out["error_type"] == "value_mismatch" and "next" not in out, out
    assert out["field_cleared"] is False and "nothing is filled" not in out["error"]
    assert _mismatch_state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "PAGE-RESTORE",
                                        "submits": 0, "registries": 0}


@_integration
def test_mismatch_from_an_insertion_listener_is_cleared_and_says_so(browser, store, capsys):
    """A listener rewrites the value during the write's input notification only; the rollback's notifications
    leave the field empty, and only then does the refusal say that nothing is filled."""
    _evaluate(browser.page, "document.getElementById('p').value='';document.getElementById('p')"
                            ".addEventListener('input',function(){this.value=this.value+'-PAGE'},{once:true})")
    code, out, _ = _cli_fill(browser, store, "exact-value-9", capsys)
    assert code == 1 and out["error_type"] == "value_mismatch" and "next" not in out, out
    assert out["field_cleared"] is True and "nothing is filled" in out["error"]
    assert _mismatch_state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "",
                                        "submits": 0, "registries": 0}


# ── a rollback that invalidates the original field never claims that nothing is filled (review23329) ──────

_REPLACE_ON_ROLLBACK_CHANGE = ("window.old=document.getElementById('p');window.events=[];old.addEventListener('change',"
                               "function(){events.push(this.value);if(this.value===''){const replacement=this.cloneNode(true);"
                               "replacement.value=%s;this.replaceWith(replacement)}})")


@_integration
def test_rollback_change_replaces_original_with_nonempty_clone(browser, store, capsys):
    """review23329's repro: the rollback's change listener replaces the emptied original with a clone that
    holds a value. The original is empty but no longer the field, so emptiness is not established."""
    _evaluate(browser.page, _REPLACE_ON_ROLLBACK_CHANGE % "'REVIEW-REPLACEMENT'")
    code, out, _ = _cli_fill(browser, store, "before\nafter", capsys)
    state = _state(browser)
    observed = {"result": out, "state": state, "original": _evaluate(browser.page, "old.value"),
                "events": json.loads(_evaluate(browser.page, "JSON.stringify(events)"))}
    assert code == 1 and out["error_type"] == "value_mismatch" and "next" not in out, out
    assert state["submits"] == 0 and state["slots"] == 0, state
    assert not (out["field_cleared"] and state["pw"] != ""), observed
    assert not ("nothing is filled" in out["error"] and state["pw"] != ""), observed
    assert out["field_cleared"] is False and "not established" in out["error"], observed
    assert state["pw"] == "REVIEW-REPLACEMENT" and observed["original"] == "", observed


@_integration
def test_rollback_change_replacing_original_with_an_empty_clone_is_still_not_established(browser, store, capsys):
    """The replacement is empty too, but the original was detached during the rollback: what the page now
    shows is not the field the fill cleared, so emptiness is still not established."""
    _evaluate(browser.page, _REPLACE_ON_ROLLBACK_CHANGE % "''")
    code, out, _ = _cli_fill(browser, store, "before\nafter", capsys)
    state = _state(browser)
    assert code == 1 and out["error_type"] == "value_mismatch" and "next" not in out, out
    assert out["field_cleared"] is False and "nothing is filled" not in out["error"], out
    assert "not established" in out["error"], out
    assert _evaluate(browser.page, "old.isConnected") is False
    assert state == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}
    assert _evaluate(browser.page, f"{_REGISTRIES}.length") == 0


# ── the mismatch report is built after the cleanup's own synchronous page callbacks (review23339) ──────────
# The reviewer's probes (test_review_rollback_identity.py, SHA256 692f12b7…f45f4e), carried verbatim, plus
# marked extra assertions on the exact report. A customized built-in input observing the stamp attribute runs
# page code synchronously when the cleanup removes the stamp, after the rollback's notifications.

from tools.browser_vault_tool import _fill_outcome


@_integration
@pytest.mark.parametrize('mutation', ['move','shadow-replace','clone-reinsert','ancestor-move','shadow-host-reinsert'])
def test_rollback_invalidations(browser,mutation):
    setup = "window.old=document.getElementById('p');window.live=old;"
    if mutation in ['shadow-replace','shadow-host-reinsert']:
        setup += "window.host=document.createElement('div');document.body.append(host);window.root=host.attachShadow({mode:'open'});root.append(old);"
    else:
        setup += "window.dest=document.createElement('div');document.body.append(dest);"
    body = {
        'move': 'dest.append(this);',
        'shadow-replace': "live=this.cloneNode(true);live.value='REVIEW-REPLACEMENT';this.replaceWith(live);",
        'clone-reinsert': "live=this.cloneNode(true);live.value='REVIEW-REPLACEMENT';this.replaceWith(live);live.after(this);",
        'ancestor-move': 'dest.append(this.parentNode);',
        'shadow-host-reinsert': 'host.remove();document.body.append(host);',
    }[mutation]
    setup += "old.addEventListener('change',function(){if(this.value===''){"+body+"}});"
    _evaluate(browser.page,setup)
    nonce='identity23339'
    controls=json.loads(_evaluate(browser.page,build_inspection_js(nonce)))
    index=next(x['index'] for x in controls if x['type']=='password')
    fill=build_fill_js([{'index':index,'token':'current-password','value':'before\nafter'}],browser.origin,nonce,require_empty=True)
    observed=json.loads(_evaluate(browser.page,"(()=>{const result="+fill+";return JSON.stringify({result:JSON.parse(result),original:old.value,connected:old.isConnected,current:live.value,slots:[old,live].filter(x=>x.hasAttribute('data-hermes-vault-slot')).length,submits:window.submits,registries:Object.getOwnPropertyNames(window).filter(x=>x.startsWith('hermesVaultSlots:')).length})})()"))
    out=json.loads(_fill_outcome(observed['result'],None,None,browser.origin,[]))
    print('IDENTITY_OBSERVATION='+json.dumps(dict(observed,mutation=mutation,tool=out)))
    assert observed['result']=={'refused':'value_mismatch','intact':False,'cleared':False}
    assert out['success'] is False and out['error_type']=='value_mismatch' and out['field_cleared'] is False
    assert 'not established' in out['error'] and 'nothing is filled' not in out['error'] and 'next' not in out
    assert observed['slots']==observed['submits']==observed['registries']==0
    assert 'REVIEW-REPLACEMENT' not in json.dumps(out)

@_integration
@pytest.mark.parametrize('mutation',['refill','replace'])
def test_finally_cleanup_preserves_report_truth_at_script_return(browser,mutation):
    effect = "this.value='REVIEW-CLEANUP';" if mutation=='refill' else "const clone=document.createElement('input');clone.id='p';clone.type='password';clone.value='REVIEW-CLEANUP';this.replaceWith(clone);"
    setup = "customElements.define('review-password',class extends HTMLInputElement {static get observedAttributes(){return ['data-hermes-vault-slot'];} attributeChangedCallback(name,oldValue,newValue){if(oldValue!==null && newValue===null){"+effect+"}}},{extends:'input'});const old=document.getElementById('p');const custom=document.createElement('input',{is:'review-password'});custom.id='p';custom.name=old.name;custom.type='password';custom.autocomplete='current-password';old.replaceWith(custom);window.reviewOriginal=custom;"
    _evaluate(browser.page,setup)
    nonce='finally23339'
    controls=json.loads(_evaluate(browser.page,build_inspection_js(nonce)))
    index=next(x['index'] for x in controls if x['type']=='password')
    fill=build_fill_js([{'index':index,'token':'current-password','value':'before\nafter'}],browser.origin,nonce,require_empty=True)
    observed=json.loads(_evaluate(browser.page,"(()=>{const result="+fill+";return JSON.stringify({result:JSON.parse(result),original:window.reviewOriginal.value,connected:window.reviewOriginal.isConnected,current:document.getElementById('p').value,submits:window.submits,registries:Object.getOwnPropertyNames(window).filter(x=>x.startsWith('hermesVaultSlots:')).length})})()"))
    out=json.loads(_fill_outcome(observed['result'],None,None,browser.origin,[]))
    print('FINALLY_OBSERVATION='+json.dumps(dict(observed,mutation=mutation,tool=out)))
    assert observed['current']=='REVIEW-CLEANUP',observed
    assert observed['submits']==observed['registries']==0
    assert out['error_type']=='value_mismatch' and out['success'] is False and 'next' not in out
    assert out['field_cleared'] is False and 'nothing is filled' not in out['error'], observed
    # The report is built after cleanup: a refilled original is "not empty", a replaced one "not established".
    assert observed['result']=={'refused':'value_mismatch','intact':mutation=='refill','cleared':False}, observed
    assert ('the field is not empty' if mutation=='refill' else 'not established') in out['error'], out
    assert 'REVIEW-CLEANUP' not in json.dumps(out)

@_integration
@pytest.mark.parametrize('mutation',['refill','replace'])
def test_cli_finally_cleanup_report(browser,store,capsys,mutation):
    from tests.tools.test_browser_vault_fill_integrity import _cli_fill
    effect = "this.value='REVIEW-CLEANUP';" if mutation=='refill' else "const clone=document.createElement('input');clone.id='p';clone.type='password';clone.value='REVIEW-CLEANUP';this.replaceWith(clone);"
    setup = "customElements.define('review-password',class extends HTMLInputElement {static get observedAttributes(){return ['data-hermes-vault-slot'];} attributeChangedCallback(name,oldValue,newValue){if(oldValue!==null && newValue===null){"+effect+"}}},{extends:'input'});const old=document.getElementById('p');const custom=document.createElement('input',{is:'review-password'});custom.id='p';custom.name=old.name;custom.type='password';custom.autocomplete='current-password';old.replaceWith(custom);window.reviewOriginal=custom;"
    _evaluate(browser.page,setup)
    code,out,_=_cli_fill(browser,store,'before\nafter',capsys)
    state=json.loads(_evaluate(browser.page,"JSON.stringify({original:window.reviewOriginal.value,connected:window.reviewOriginal.isConnected,current:document.getElementById('p').value,slots:document.querySelectorAll('[data-hermes-vault-slot]').length,submits:window.submits})"))
    print('CLI_FINALLY_OBSERVATION='+json.dumps(dict(state,result=out)))
    assert code==1 and out['success'] is False and out['error_type']=='value_mismatch' and 'next' not in out
    assert state['current']=='REVIEW-CLEANUP' and state['slots']==state['submits']==0,state
    assert out['field_cleared'] is False and 'nothing is filled' not in out['error'],out
    # Maintained addition: the tool's wording matches the report built after cleanup.
    assert ('the field is not empty' if mutation=='refill' else 'not established') in out['error'], out
    assert 'REVIEW-CLEANUP' not in json.dumps(out)


# ── a password counts as filled only while the intact original holds the exact value (Fable FA-1) ──────────

_FRESH_FIELD = ("const n=document.createElement('input');n.type='password';n.id='p';"
                "n.setAttribute('autocomplete','current-password');")
_STAMPS_ANYWHERE = ("(function w(r){let n=0;for(const e of r.querySelectorAll('*')){"
                    "if(e.hasAttribute('data-hermes-vault-slot'))n++;if(e.shadowRoot)n+=w(e.shadowRoot)}return n})(document)")


@_integration
@pytest.mark.parametrize("listener, page_field", [
    ("old.addEventListener('input',function(){" + _FRESH_FIELD + "this.replaceWith(n)},{once:true})", ""),
    ("old.addEventListener('change',function(){" + _FRESH_FIELD + "this.replaceWith(n)},{once:true})", ""),
    ("old.addEventListener('input',function(){this.remove()},{once:true})", None),
    ("old.addEventListener('change',function(){" + _FRESH_FIELD + "n.value='PAGE-OTHER';this.replaceWith(n)},{once:true})",
     "PAGE-OTHER"),
    ("old.addEventListener('change',function(){this.type='text'},{once:true})", None),
    ("void 0", "FA1-vault-value"),
], ids=["input-replaces-with-empty", "change-replaces-with-empty", "input-removes-for-good",
        "change-replaces-with-other-value", "change-turns-into-text", "control-no-listener"])
def test_success_counts_only_an_intact_field_holding_the_value(browser, store, capsys, listener, page_field):
    """Fable FA-1, through the real `hermes vault fill`: a write-time listener that replaces, removes or
    retypes the original leaves the fill not established (no success, no Submit hint); the control fills."""
    _evaluate(browser.page, "window.old=document.getElementById('p');old.value='';" + listener)
    code, out, resolved = _cli_fill(browser, store, "FA1-vault-value", capsys)
    assert len(resolved) == 1 and "FA1-vault-value" not in json.dumps(out), out
    page = json.loads(_evaluate(browser.page, "JSON.stringify({pw:(document.querySelector('input[type=password]')||{value:null}).value,"
                                              f"stamps:{_STAMPS_ANYWHERE},registries:{_REGISTRIES}.length,submits:window.submits}})"))
    if listener == "void 0":
        assert code == 0 and out["success"] is True and out["filled_fields"] == 1 and "next" in out, out
    else:
        assert code == 1 and out["success"] is False and out["error_type"] == "not_established", out
        assert "next" not in out and "not established" in out["error"], out
    assert page["pw"] == page_field and page["stamps"] == 0 and page["registries"] == 0 and page["submits"] == 0, page


# ── a fill script cannot write late: page-clock budget and passive dialog awareness (Fable FA-2) ───────────

def test_the_page_budget_is_strictly_shorter_than_the_reply_wait():
    """The script's page-clock budget ends well before the command stops waiting for its reply, so a script
    that would write after the command has given up (a dialog, a paused renderer) is always past its budget."""
    from agent.vault_login_classifier import FILL_PAGE_BUDGET_S
    from tools.browser_supervisor import EVAL_REPLY_WAIT_S
    from tools.browser_vault_tool import _PAGE_CLOCK_WAIT_S

    assert FILL_PAGE_BUDGET_S + 2.0 <= EVAL_REPLY_WAIT_S
    assert _PAGE_CLOCK_WAIT_S < EVAL_REPLY_WAIT_S
    import inspect

    from tools.browser_supervisor import CDPSupervisor
    assert inspect.signature(CDPSupervisor.evaluate_runtime).parameters["timeout"].default == EVAL_REPLY_WAIT_S


def test_the_budget_is_the_last_gate_before_the_write():
    from agent.vault_login_classifier import FILL_PAGE_BUDGET_S

    anchor = {"timeOrigin": 1700000000123.25, "now": 4321.5}
    js = build_fill_js([{"index": 0, "token": "current-password", "value": "v"}], "https://m.example", "n",
                       require_empty=True, anchor=anchor)
    assert f"const anchor = {json.dumps(anchor)};" in js and f"const budgetMs = {FILL_PAGE_BUDGET_S * 1000.0};" in js
    focus, write = js.index("el.focus();"), js.index("setter.set.call(el, f.value)")
    gate = js.index('if (late()) return JSON.stringify({ refused: "expired", written: filled });', focus)
    assert js.index('if (requireEmpty && current(el).length > 0) return JSON.stringify({ refused: "field_not_empty" });',
                    focus) < gate < write
    assert js.index('if (late()) return JSON.stringify({ refused: "expired", written: 0 });') < focus  # and at start
    assert "const anchor = null;" in build_fill_js([], "https://m.example", "n")  # direct use without a clock


def test_the_supervisor_records_dialogs_passively():
    """Page.javascriptDialogOpening/Closed are recorded per page session, with or without the dialog bridge;
    with the bridge off nothing is answered, queued or archived."""
    import asyncio

    from tools.browser_supervisor import CDPSupervisor

    sup = CDPSupervisor("t", "ws://127.0.0.1:1/devtools/browser/x", attach_target_id="T1", dialog_bridge=False)
    sup._page_session_id = "S1"
    assert sup.open_dialog() is None
    asyncio.run(sup._on_dialog_opening({"type": "confirm", "message": "m"}, "S2"))  # another session
    assert sup.open_dialog() is None
    asyncio.run(sup._on_dialog_opening({"type": "alert", "message": "m"}, "S1"))
    assert sup.open_dialog()["type"] == "alert"
    assert sup.snapshot().pending_dialogs == () and sup.snapshot().recent_dialogs == ()  # never acted on
    asyncio.run(sup._on_dialog_closed({"result": True}, "S1"))
    assert sup.open_dialog() is None


class _BlockedPage(_Page):
    """A recording page whose dialog state and clock answers a test controls."""

    def __init__(self, controls, fill_result=None, *, dialog=None, clock=True, fill_error=None):
        super().__init__(controls, fill_result)
        self.dialog, self.clock, self.fill_error, self.waits = dialog, clock, fill_error, []

    def open_dialog(self):
        return self.dialog

    def evaluate_runtime(self, expression, **kw):
        from agent.vault_login_classifier import PAGE_CLOCK_JS

        self.waits.append((expression[:24], kw.get("timeout")))
        if expression == PAGE_CLOCK_JS and not self.clock:
            return {"ok": False, "error": "TimeoutError: "}
        if "expectedOrigin" in expression and self.fill_error:
            self.writes.append(expression)
            return {"ok": False, "error": self.fill_error}
        return super().evaluate_runtime(expression, **kw)


def test_a_dialog_open_before_the_fill_refuses_before_anything_is_inspected(item):
    page, backend = _BlockedPage([_EMAIL, _PW], dialog={"type": "alert"}), _Recording()
    out = _fill(item, page, backend)
    assert out["error_type"] == "dialog_open" and out["success"] is False and "next" not in out
    assert "alert dialog is open" in out["error"] and "nothing was inspected, resolved, sent or written" in out["error"]
    assert page.inspections == [] and backend.resolved == [] and page.writes == []


def test_a_dialog_opening_during_resolution_means_the_value_is_never_sent(item):
    page = _BlockedPage([_EMAIL, _PW])

    class _OpensDialog(_Recording):
        def resolve_password(self, handle):
            page.dialog = {"type": "confirm"}
            return super().resolve_password(handle)

    backend = _OpensDialog()
    out = _fill(item, page, backend)
    assert out["error_type"] == "dialog_open" and "the value was not sent; nothing was written" in out["error"]
    assert backend.resolved == [item.id] and page.writes == [] and _PASSWORD not in json.dumps(out)
    [cleanup] = page.cleared  # the stamps are still removed, with the short wait (the page is blocked)
    assert (cleanup[:24], 2.0) in page.waits


def test_a_page_that_does_not_answer_the_clock_read_never_gets_the_value(item):
    page, backend = _BlockedPage([_EMAIL, _PW]), _Recording()
    original = page.evaluate_runtime
    clock_reads = []

    def evaluate_runtime(expression, **kw):  # answers before inspection, then stops answering
        from agent.vault_login_classifier import PAGE_CLOCK_JS

        if expression == PAGE_CLOCK_JS:
            clock_reads.append(kw.get("timeout"))
            page.clock = len(clock_reads) == 1
        return original(expression, **kw)

    page.evaluate_runtime = evaluate_runtime
    out = _fill(item, page, backend)
    assert out["error_type"] == "page_unresponsive" and "nothing was written" in out["error"], out
    assert clock_reads == [2.0, 2.0] and backend.resolved == [item.id] and page.writes == []


def test_an_unanswered_fill_script_reports_the_truth_not_a_bare_timeout(item):
    page, backend = _BlockedPage([_EMAIL, _PW], fill_error="TimeoutError: "), _Recording()
    out = _fill(item, page, backend)
    assert out["error_type"] == "fill_timeout" and out["success"] is False and "next" not in out
    assert out["error"] != "TimeoutError: " and "not established" in out["error"]
    assert "no dialog was recorded open" in out["error"] and "cannot write later" in out["error"]
    assert len(page.writes) == 1 and len(page.cleared) == 1


@_integration
@pytest.mark.parametrize("case", ["past-the-budget", "another-document", "within-the-budget"])
def test_a_script_run_later_than_its_budget_writes_nothing(browser, case):
    """The real fill script on a real page: a script that first runs more than FILL_PAGE_BUDGET_S after its
    anchor (or in another document) writes nothing and still retires its inspection."""
    from agent.vault_login_classifier import FILL_PAGE_BUDGET_S

    anchor_js = {
        "past-the-budget": f"({{timeOrigin: performance.timeOrigin, now: performance.now() - {FILL_PAGE_BUDGET_S * 1000 + 500}}})",
        "another-document": "({timeOrigin: performance.timeOrigin + 1, now: performance.now()})",
        "within-the-budget": "({timeOrigin: performance.timeOrigin, now: performance.now()})",
    }[case]
    expected = {"filled": 1} if case == "within-the-budget" else {"refused": "expired", "written": 0}
    nonce = "budget"
    _evaluate(browser.page, "document.getElementById('p').value=''")
    _evaluate(browser.page, build_inspection_js(nonce))
    anchor = _evaluate(browser.page, f"JSON.stringify({anchor_js})")
    js = build_fill_js([{"index": 1, "token": "current-password", "value": "BUDGET-value"}], browser.origin, nonce,
                       require_empty=True, anchor=json.loads(anchor))
    assert json.loads(_evaluate(browser.page, js)) == expected
    state = _state(browser)
    assert state["pw"] == ("BUDGET-value" if "filled" in expected else "") and state["slots"] == 0, state
    assert _evaluate(browser.page, f"{_REGISTRIES}.length") == 0


class _PersonAtTheBrowser:
    """A second CDP connection to the tab, standing in for the person at the browser: it sees native dialogs
    open and closes them (the vault fill never does)."""

    def __init__(self, page):
        from websockets.sync.client import connect

        self.ws, self.next_id, self.events = connect(page["webSocketDebuggerUrl"], open_timeout=5), 0, []
        self.call("Page.enable")

    def _recv(self, until, timeout):
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                msg = json.loads(self.ws.recv(timeout=max(0.05, deadline - _time.monotonic())))
            except TimeoutError:
                return None
            if "method" in msg:
                self.events.append(msg["method"])
            if until(msg):
                return msg
        return None

    def call(self, method, params=None, timeout=10):
        self.next_id += 1
        call_id = self.next_id
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        return self._recv(lambda m: m.get("id") == call_id, timeout)

    def wait_open(self, timeout=10):
        if "Page.javascriptDialogOpening" in self.events:
            return True
        return self._recv(lambda m: m.get("method") == "Page.javascriptDialogOpening", timeout) is not None

    def close_dialog(self):
        self.events = [e for e in self.events if e != "Page.javascriptDialogOpening"]
        return self.call("Page.handleJavaScriptDialog", {"accept": True})

    def close_after(self, delay):
        """In the background: wait for a dialog to open, keep it open ``delay`` seconds, then close it."""
        import time as _time

        def run():
            if self.wait_open(30):
                _time.sleep(delay)  # health: allow HX013 -- how long a person keeps the dialog open is the behaviour under test (within vs past the page-clock budget)
                self.close_dialog()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def close(self):
        self.ws.close()


def _supervisor_saw_a_dialog():
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    with SUPERVISOR_REGISTRY._lock:
        supervisors = list(SUPERVISOR_REGISTRY._by_task.values())
    return any(s.open_dialog() for s in supervisors)


def _page_after(browser, person):
    """The page once the person has closed any dialog and the renderer is free again."""
    return wait_until(lambda: _settled(browser), timeout=15, what="the page settling after the dialog closed")


def _settled(browser):
    state = json.loads(_evaluate(browser.page, "JSON.stringify({pw:document.getElementById('p').value,"
                                              f"stamps:{_STAMPS_ANYWHERE},registries:{_REGISTRIES}.length,"
                                              "submits:window.submits})"))
    return state if state["stamps"] == 0 and state["registries"] == 0 else None


@_integration
@pytest.mark.parametrize("kind", ["alert", "confirm"])
def test_a_dialog_during_resolution_is_refused_and_nothing_is_written_after_it_closes(browser, store, capsys, kind):
    """Fable FA-2: the page opens a native dialog after inspection, while the value is resolved. The value is
    never sent (a plain dialog_open refusal); when the person closes the dialog after the command has
    exited, nothing is written and no stamp remains."""
    _evaluate(browser.page, "document.getElementById('p').value=''")
    person = _PersonAtTheBrowser(browser.page)
    try:
        sent = []
        from tools import browser_vault_tool
        real_secret = browser_vault_tool._eval_js_secret

        def during_resolve_dialog():
            person.call("Runtime.evaluate", {"expression": f"setTimeout(()=>{{window.dialogResult={kind}('FA-2')}},0);1"})
            assert person.wait_open(5)
            wait_until(_supervisor_saw_a_dialog, timeout=5, what="the supervisor recording the dialog")

        original = LocalLoginBackend.resolve_password

        def resolve(self, handle):
            value = original(self, handle)
            during_resolve_dialog()
            return value

        with patch.object(LocalLoginBackend, "resolve_password", resolve), \
             patch.object(browser_vault_tool, "_eval_js_secret", lambda *a, **k: sent.append(a) or real_secret(*a, **k)):
            code, out, _ = _cli_fill(browser, store, "FA2-vault-value", capsys)
        assert code == 1 and out["error_type"] == "dialog_open" and "next" not in out, out
        assert f"{kind} dialog is open" in out["error"] and "the value was not sent" in out["error"], out
        assert sent == []
        person.close_dialog()
        state = _page_after(browser, person)
        assert state == {"pw": "", "stamps": 0, "registries": 0, "submits": 0}, state
    finally:
        person.close()


@_integration
def test_a_dialog_already_open_refuses_plainly_and_nothing_is_resolved(browser, store, capsys):
    _evaluate(browser.page, "document.getElementById('p').value=''")
    person = _PersonAtTheBrowser(browser.page)
    try:
        person.call("Runtime.evaluate", {"expression": "setTimeout(()=>{alert('already open')},0);1"})
        assert person.wait_open(5)
        code, out, resolved = _cli_fill(browser, store, "FA2-open-before", capsys)
        print("FA2_OPEN_BEFORE=" + json.dumps(out))
        assert code == 1 and out["success"] is False and "next" not in out, out
        # The exact-target supervisor cannot even attach (Runtime.enable waits on the blocked renderer), or it
        # attaches and the passive record / clock read refuses: either way a plain refusal, nothing resolved.
        assert out["error_type"] in ("dialog_open", "page_unresponsive", "attach_failed"), out
        assert ("Nothing was resolved, sent or written" in out["error"]
                or "nothing was inspected, resolved, sent or written" in out["error"]), out
        assert resolved == [], resolved
        if out["error_type"] == "attach_failed":
            assert "did not answer in time" in out["error"], out
        person.close_dialog()
        state = _page_after(browser, person)
        assert state == {"pw": "", "stamps": 0, "registries": 0, "submits": 0}, state
    finally:
        person.close()


@_integration
def test_a_dialog_opened_and_closed_within_the_budget_still_fills(browser, store, capsys):
    """A focus listener opens an alert in the middle of the fill script; the person closes it well within the
    page-clock budget, and the script goes on to fill."""
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "document.getElementById('p').addEventListener('focus',function(){alert('quick')},{once:true})")
    person = _PersonAtTheBrowser(browser.page)
    try:
        closer = person.close_after(0.3)
        code, out, _ = _cli_fill(browser, store, "FA2-quick-value", capsys)
        closer.join(10)
        assert code == 0 and out["success"] is True and out["filled_fields"] == 1, out
        assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "FA2-quick-value", "submits": 0}
    finally:
        person.close()


@_integration
def test_a_dialog_held_past_the_budget_inside_the_script_writes_nothing(browser, store, capsys):
    """The alert opens inside the fill script (a focus listener) and stays open past the page-clock budget,
    but is closed before the command stops waiting: the last gate refuses, nothing is written."""
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "document.getElementById('p').addEventListener('focus',function(){alert('slow')},{once:true})")
    person = _PersonAtTheBrowser(browser.page)
    from agent.vault_login_classifier import FILL_PAGE_BUDGET_S

    try:
        closer = person.close_after(FILL_PAGE_BUDGET_S + 1.5)
        code, out, _ = _cli_fill(browser, store, "FA2-slow-value", capsys)
        closer.join(10)
        assert code == 1 and out["error_type"] == "expired" and "Nothing was written" in out["error"], out
        assert "next" not in out
        assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}
    finally:
        person.close()


@_integration
@pytest.mark.parametrize("kind", ["alert", "confirm"])
def test_a_dialog_closed_after_the_command_exits_cannot_cause_a_late_write(browser, store, capsys, kind):
    """The dialog opens inside the fill script and is closed only after the command has given up waiting:
    the command says what it knows (no bare TimeoutError), and when the script finally resumes it is past its
    budget, so nothing is written and it retires its inspection."""
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            f"document.getElementById('p').addEventListener('focus',function(){{{kind}('late')}},{{once:true}})")
    person = _PersonAtTheBrowser(browser.page)
    try:
        code, out, _ = _cli_fill(browser, store, "FA2-late-value", capsys)
        assert code == 1 and out["success"] is False and out["error_type"] == "fill_timeout" and "next" not in out, out
        assert f"{kind} dialog is open" in out["error"] and "not established" in out["error"], out
        assert "cannot write later" in out["error"] and out["error"] != "TimeoutError: "
        assert person.wait_open(1)
        person.close_dialog()
        state = _page_after(browser, person)
        assert state == {"pw": "", "stamps": 0, "registries": 0, "submits": 0}, state
    finally:
        person.close()
