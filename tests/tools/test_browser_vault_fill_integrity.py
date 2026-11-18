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
import sys
import tempfile
import threading
import urllib.request
import contextlib
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

_REGISTRIES = "Object.getOwnPropertyNames(window).filter(x=>x.startsWith('hermesVaultSlots:'))"

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
    # N-d: each field's budget gate also runs before its focus listeners
    assert js.rindex('if (late()) return report({ refused: "expired" });', 0, focus) > js.index("for (const f of fills)")
    assert focus < js.index('if (requireEmpty && current(el).length > 0) return report({ refused: "field_not_empty" });', focus) < write
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


_FILLED = {"filled": 1, "written": 1, "roles": ["current-password"]}


class _Page:
    """Answers the supervisor calls browser_vault_fill makes; records writes and stamp clean-ups."""

    def __init__(self, controls, fill_result=None):
        self.controls, self.writes, self.cleared, self.inspections = controls, [], [], []
        self.fill_result = fill_result or _FILLED

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
        ([_EMAIL, _PW], {"filled": 1}, "partial_fill", True),  # N-k: no written count reported, so not complete
        ([_EMAIL, _PW], _FILLED, None, True),
    ],
)
def test_every_exit_after_inspection_removes_its_stamps(item, controls, fill_result, error_type, resolves):
    page, backend = _Page(controls, fill_result), _Recording()
    out = _fill(item, page, backend)
    assert out.get("error_type") == error_type
    assert out["success"] is (fill_result == _FILLED)
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
        yield SimpleNamespace(http=http, origin=origin, page=page, worlds=[])
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


def _evaluate_in(page, expression, unique_id):
    """``expression`` in the isolated world ``unique_id`` (a world's uniqueId reaches it from any connection);
    None when that world is gone (its document was replaced)."""
    from websockets.sync.client import connect

    with connect(page["webSocketDebuggerUrl"], open_timeout=5) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": expression, "returnByValue": True, "uniqueContextId": unique_id}}))
        while True:
            msg = json.loads(ws.recv(timeout=5))
            if msg.get("id") == 1:
                if "error" in msg and "Cannot find context with specified id" in msg["error"]["message"]:
                    return None
                assert "exceptionDetails" not in msg["result"], msg
                return msg["result"]["result"].get("value")


def _registries(browser):
    """Inspection registries left anywhere: in the page's own world and in every isolated world a command of
    this test created (where the CLI keeps its registry)."""
    count = _evaluate(browser.page, f"{_REGISTRIES}.length")
    for world in browser.worlds:
        assert world and world.get("unique_id"), world  # the supervisor obtained the world's uniqueId
        count += _evaluate_in(browser.page, f"{_REGISTRIES}.length", world["unique_id"]) or 0
    return count


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
    from tools.browser_supervisor import CDPSupervisor

    real_world = CDPSupervisor.create_isolated_world

    def world(self, name, **kw):  # records each command's isolated world, so tests can look inside it
        made = real_world(self, name, **kw)
        browser.worlds.append(made.get("world"))
        return made

    with ExitStack() as stack:
        stack.enter_context(patch("agent.vault_store.get_vault_store", return_value=store))
        stack.enter_context(patch.object(LocalLoginBackend, "resolve_password", resolve))
        stack.enter_context(patch.object(CDPSupervisor, "create_isolated_world", world))
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
    assert _registries(browser) == 0


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
        # FA-1: the field left the page (was {"filled": 1}); FB-1: the report says what was written
        assert out == {"refused": "not_established", "exposed_cleared": 0, "written": 1,
                       "roles": ["current-password"]}, out
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
    return dict(_state(browser), registries=_registries(browser))


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
    assert _registries(browser) == 0


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
    # FB-1 (this round): every report now also carries written/roles; the reviewer's exact dict is checked on the
    # other keys, and nothing is left written (the password was cleared again).
    assert {k:v for k,v in observed['result'].items() if k not in ('written','roles')}=={'refused':'value_mismatch','intact':False,'cleared':False}
    assert observed['result']['written']==0 and observed['result']['roles']==[]
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
    assert observed['result']=={'refused':'value_mismatch','intact':mutation=='refill','cleared':False,
                                'written':0,'roles':[]}, observed
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
                                              f"stamps:{_STAMPS_ANYWHERE},submits:window.submits}})"))
    page["registries"] = _registries(browser)
    if listener == "void 0":
        assert code == 0 and out["success"] is True and out["filled_fields"] == 1 and "next" in out, out
    else:
        assert code == 1 and out["success"] is False and out["error_type"] == "not_established", out
        assert "next" not in out and "not established" in out["error"], out
        if "type='text'" in listener:  # FB-3: the emptied field is not reported as left written
            assert "No other field was left written." in out["error"] and "1 field was" not in out["error"], out
        else:
            assert "1 field was written (or a write was attempted) before this and was not cleared: the password field." in out["error"], out
    assert page["pw"] == page_field and page["stamps"] == 0 and page["registries"] == 0 and page["submits"] == 0, page
    if "type='text'" in listener:  # N-a: the retyped original no longer shows the stored value, and the result says so
        assert _evaluate(browser.page, "old.isConnected && old.type==='text' && old.value===''") is True
        assert out["exposed_cleared"] is True and "it was emptied without page events" in out["error"], out
    elif listener != "void 0":
        assert out["exposed_cleared"] is False, out


# ── a fill script released late: page-clock budget and passive dialog awareness (Fable FA-2) ──────────────

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
    gate = js.index('if (late()) return report({ refused: "expired" });', focus)
    assert js.index('if (requireEmpty && current(el).length > 0) return report({ refused: "field_not_empty" });',
                    focus) < gate < write
    assert js.index('if (late()) return report({ refused: "expired" });') < js.index("for (const f of fills)")  # at start
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
            return {"ok": False, "error": self.fill_error, "timed_out": True}
        return super().evaluate_runtime(expression, **kw)


def test_a_dialog_open_before_the_fill_refuses_before_anything_is_inspected(item):
    """N-b: the passive dialog record is consulted before the origin probes (no tab focusing or URL reads that
    would each wait on a blocked renderer)."""
    page, backend = _BlockedPage([_EMAIL, _PW], dialog={"type": "alert"}), _Recording()
    focused = []
    page.focus_page = lambda origin, accept=None: focused.append(origin) or {"ok": True, "url": _ORIGIN + "/login"}
    out = _fill(item, page, backend)
    assert out["error_type"] == "dialog_open" and out["success"] is False and "next" not in out
    assert "alert dialog is open" in out["error"] and "nothing was resolved, sent or written" in out["error"]
    assert focused == [] and page.inspections == [] and backend.resolved == [] and page.writes == []


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
    """DEADLINE-R01: an unanswered script may still complete later on the tab; the result says so and asks for
    the tab to be checked. Nothing tries to stop the script (no Runtime.terminateExecution: it cannot tell our
    script from the page's own)."""
    page, backend = _BlockedPage([_EMAIL, _PW], fill_error="TimeoutError: "), _Recording()
    out = _fill(item, page, backend)
    assert out["error_type"] == "fill_timeout" and out["success"] is False and "next" not in out
    assert out["error"] != "TimeoutError: " and "not established" in out["error"] and "check the tab" in out["error"]
    assert "may still complete" in out["error"] and out["may_complete_later"] is True
    assert "cannot write" not in out["error"] and "no dialog was recorded open" in out["error"]
    assert "termination_requested" not in out and "terminat" not in out["error"].lower()
    assert len(page.writes) == 1 and len(page.cleared) == 1


@pytest.mark.parametrize("error, error_type", [
    ("ConnectionError: CDP connection closed before reply", "fill_unconfirmed"),
    ("Uncaught Error: PAGE-CONTROLLED-TEXT", "fill_unconfirmed"),
    ("Execution was terminated", "fill_unconfirmed"),
    ("RuntimeError: CDP error on id=9: {'code': -32000, 'message': 'Inspected target navigated or closed'}",
     "fill_unconfirmed"),
    ("RuntimeError: CDP error on id=9: {'code': -32000, 'message': 'Cannot find context with specified id'}",
     "fill_unconfirmed"),  # no isolated world in use: never read as "the script did not start"
])
def test_every_failure_after_sending_is_not_established(item, error, error_type):
    """FB-2: whatever ends the wait after the fill script was sent, the result says the script may still
    complete and the field's state is not established; it never says nothing was written, and never repeats
    the failure's own text (a page exception's message is page text)."""
    page, backend = _BlockedPage([_EMAIL, _PW]), _Recording()
    original = page.evaluate_runtime

    def evaluate_runtime(expression, **kw):
        if "expectedOrigin" in expression:
            page.writes.append(expression)
            return {"ok": False, "error": error}
        return original(expression, **kw)

    page.evaluate_runtime = evaluate_runtime
    out = _fill(item, page, backend)
    assert out["error_type"] == error_type and out["success"] is False and "next" not in out, out
    assert out["may_complete_later"] is True and "not established" in out["error"] and "check the tab" in out["error"]
    assert "Nothing was written" not in out["error"] and "nothing was written" not in out["error"]
    assert "PAGE-CONTROLLED-TEXT" not in json.dumps(out) and "CDP error" not in out["error"], out
    assert len(page.writes) == 1 and len(page.cleared) == 1


def test_a_lost_world_before_the_script_started_is_the_only_nothing_written(item):
    """FB-2 / point 1: only "Cannot find context with specified id" for the command's isolated world (the page
    moved to a new document before the script started) means nothing was written; the stamp cleanup finding the
    world gone says the new document carries none of the stamps."""
    page, backend = _BlockedPage([_EMAIL, _PW]), _Recording()
    original, seen = page.evaluate_runtime, []
    gone = {"ok": False, "error": "RuntimeError: CDP error on id=9: {'code': -32000, 'message': "
                                  "'Cannot find context with specified id'}"}
    page.create_isolated_world = lambda name, **kw: {"ok": True, "world": {"session_id": "S", "context_id": 7,
                                                                           "unique_id": "u-7"}}

    def evaluate_runtime(expression, **kw):
        seen.append(kw.get("world"))
        if "expectedOrigin" in expression:
            page.writes.append(expression)
            return gone
        if "return retireInspection(" in expression:
            return gone
        return original(expression, **kw)

    page.evaluate_runtime = evaluate_runtime
    out = _fill(item, page, backend)
    assert out["error_type"] == "page_navigated" and out["success"] is False and "next" not in out, out
    assert "the script never ran. Nothing was written" in out["error"], out
    assert "carries none of this inspection's stamps" in out["stamps"], out
    assert seen and all(w == {"session_id": "S", "context_id": 7, "unique_id": "u-7"} for w in seen), seen


def test_a_failure_before_sending_keeps_its_own_refusal(item):
    """A supervisor that never sent the script (loop stopped, inactive) is a plain refusal, not "may complete"."""
    page, backend = _BlockedPage([_EMAIL, _PW]), _Recording()
    original = page.evaluate_runtime
    page.evaluate_runtime = lambda expression, **kw: ({"ok": False, "error": "supervisor is not active", "sent": False}
                                                      if "expectedOrigin" in expression else original(expression, **kw))
    out = _fill(item, page, backend)
    assert out["success"] is False and out["error_type"] == "supervisor_required" and "may_complete_later" not in out


def test_role_words_are_a_closed_table_and_page_text_is_never_repeated():
    """FB-4: an unknown role is "another field"; origin_changed no longer repeats the page's reported origin."""
    from tools.browser_vault_tool import _in_page_refusal

    out = _in_page_refusal({"refused": "target_changed", "written": 3,
                            "roles": ["one-time-code", "cc-number", "<img src=x onerror=alert(1)>"]}, _ORIGIN)
    assert out["error"].endswith("not cleared: a one-time-code box, a card field, another field."), out
    assert "<img" not in json.dumps(out)
    moved = _in_page_refusal({"refused": "origin_changed", "found": "https://PAGE-TEXT.example", "written": 0,
                              "roles": []}, _ORIGIN)
    assert "PAGE-TEXT" not in json.dumps(moved) and _ORIGIN in moved["error"] and "Nothing was written." in moved["error"]


@pytest.mark.parametrize("report, expected, success", [
    ({"filled": 6, "written": 6, "roles": ["one-time-code"] * 6}, 6, True),
    ({"filled": 5, "written": 6, "roles": ["one-time-code"] * 6}, 6, False),  # five of six boxes hold a value
    ({"filled": 5, "written": 5, "roles": ["one-time-code"] * 5}, 6, False),  # a box was skipped
    ({"filled": 1}, 1, False),  # no written count: not established
    ({"filled": 1, "written": 1, "roles": ["one-time-code"]}, 1, True),
])
def test_a_code_is_entered_only_when_every_box_was_written_and_holds_a_value(report, expected, success):
    """N-k: success needs written == filled == the number of fills."""
    from tools.browser_vault_tool import _code_write_outcome

    out = _code_write_outcome(report, _ORIGIN, "user", expected)
    assert out["success"] is success and ("next" in out) is success, out
    if not success and report["filled"]:
        assert out["error_type"] == "partial_fill" and f"Only {report['filled']} of the {expected}" in out["error"], out


def test_enter_code_without_a_saved_login_checks_the_dialog_record_first():
    """N-j: no saved login means no origin guard ran; a dialog recorded open refuses before any tab probe or
    inspection."""
    from tools import browser_vault_tool
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    page = _BlockedPage([], dialog={"type": "prompt"})
    probes = []
    page.focus_page = lambda origin, accept=None: probes.append(origin) or {"ok": True, "url": _ORIGIN}
    with patch.object(SUPERVISOR_REGISTRY, "get", return_value=page), browser_vault_tool.supervisor_only():
        out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
    assert out["error_type"] == "dialog_open" and "prompt dialog is open" in out["error"], out
    assert probes == [] and page.inspections == [] and page.waits == []


def test_the_isolated_world_is_created_without_universal_access():
    """Point 4: Page.createIsolatedWorld never sets grantUniveralAccess; the world's uniqueId comes from the
    context-created event and is what evaluations target."""
    import asyncio

    from tools.browser_supervisor import CDPSupervisor
    from tools.browser_supervisor_worlds import world_params

    sup = CDPSupervisor("t", "ws://127.0.0.1:1/devtools/browser/x", attach_target_id="T1", dialog_bridge=False)
    calls = []

    async def cdp(method, params=None, **kw):
        calls.append((method, params, kw.get("session_id")))
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "F1"}}}}
        sup._on_context_created({"context": {"id": 11, "uniqueId": "u-11", "name": "hermes-vault:n1"}}, "S1")
        sup._on_context_created({"context": {"id": 12, "uniqueId": "u-12", "name": "page-world"}}, "S1")
        return {"result": {"executionContextId": 11}}

    sup._cdp = cdp
    world = asyncio.run(sup._create_world("S1", "hermes-vault:n1", timeout=1.0))
    assert world == {"session_id": "S1", "context_id": 11, "unique_id": "u-11"}
    assert calls[1] == ("Page.createIsolatedWorld", {"frameId": "F1", "worldName": "hermes-vault:n1"}, "S1")
    assert world_params(world) == {"uniqueContextId": "u-11"}
    assert world_params(dict(world, unique_id=None)) == {"contextId": 11}
    assert list(sup._world_unique_ids) == [("S1", 11)]  # only the vault's own worlds are recorded
    sup._on_contexts_cleared({}, "S1")
    assert sup._world_unique_ids == {}
    assert not hasattr(sup, "terminate_execution")


def test_a_timeout_is_classified_by_the_supervisors_marker_not_by_text():
    from tools.browser_supervisor import _err
    from tools.browser_vault_tool import _timed_out

    assert _err(TimeoutError())["timed_out"] is True
    assert "timed_out" not in _err(RuntimeError("navigation timeout in page"))
    assert _timed_out({"success": False, "error": "x", "timed_out": True}) is True
    assert _timed_out({"success": False, "error": "TimeoutError: navigation timeout"}) is False


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
    expected = ({"filled": 1, "written": 1, "roles": ["current-password"]} if case == "within-the-budget"
                else {"refused": "expired", "written": 0, "roles": []})
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
                                              f"stamps:{_STAMPS_ANYWHERE},submits:window.submits}})"))
    state["registries"] = _registries(browser)
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
def test_a_dialog_before_the_last_gate_closed_after_exit_is_caught_by_the_budget(browser, store, capsys, kind):
    """The dialog opens inside the fill script BEFORE its last gate (a focus listener) and is closed only after
    the command has given up waiting: the command says the script may still complete (no bare TimeoutError);
    when the script resumes, the last gate is past its budget, so nothing is written and it retires its
    inspection. A dialog held AFTER the last gate is not caught: see the setter-alert witness below."""
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            f"document.getElementById('p').addEventListener('focus',function(){{{kind}('late')}},{{once:true}})")
    person = _PersonAtTheBrowser(browser.page)
    try:
        code, out, _ = _cli_fill(browser, store, "FA2-late-value", capsys)
        assert code == 1 and out["success"] is False and out["error_type"] == "fill_timeout" and "next" not in out, out
        assert f"{kind} dialog is open" in out["error"] and "not established" in out["error"], out
        assert "may still complete" in out["error"] and out["error"] != "TimeoutError: "
        assert "cannot write" not in out["error"] and out["may_complete_later"] is True, out
        assert person.wait_open(1)
        person.close_dialog()
        state = _page_after(browser, person)
        assert state == {"pw": "", "stamps": 0, "registries": 0, "submits": 0}, state
    finally:
        person.close()


# ── refusals after earlier writes say what was written (Fable FB-1) ─────────────────────────────────────────

_CODE_BOXES = ("const f=document.querySelector('form');for(const id of ['b0','b1']){const b=document.createElement('input');"
               "b.id=id;b.name=id;b.type='text';b.maxLength=1;b.autocomplete='one-time-code';f.appendChild(b)}")


def _inspect(browser, nonce):
    return [LoginControl.from_dict(c) for c in json.loads(_evaluate(browser.page, build_inspection_js(nonce)))]


@_integration
def test_a_code_box_change_after_the_first_box_was_written_says_one_was_written(browser):
    """Split code boxes: box 0 is written; its input listener detaches and reattaches box 1, so box 1 is no
    longer the inspected node. The refusal reports that one box was written (and left), not 'Nothing'."""
    from agent.vault_login_classifier import build_otp_fills, classify_otp_controls
    from tools.browser_vault_tool import _code_write_outcome

    _evaluate(browser.page, _CODE_BOXES + ";document.getElementById('b0').addEventListener('input',function(){"
                            "const b1=document.getElementById('b1'),p=b1.parentNode;b1.remove();p.appendChild(b1)},{once:true})")
    fills = build_otp_fills(classify_otp_controls(_inspect(browser, "fb1otp")), "12")
    assert [f["value"] for f in fills] == ["1", "2"]
    raw = json.loads(_evaluate(browser.page, build_fill_js(fills, browser.origin, "fb1otp")))
    assert raw == {"refused": "target_changed", "written": 1, "roles": ["one-time-code"]}, raw
    out = _code_write_outcome(raw, browser.origin, "user", len(fills))
    assert out["success"] is False and out["error_type"] == "target_changed" and "next" not in out, out
    assert "1 field was written (or a write was attempted) before this and was not cleared: a one-time-code box." in out["error"], out
    assert "Nothing was written" not in out["error"] and "1" not in out["error"].replace("1 field", "")
    assert _evaluate(browser.page, "[document.getElementById('b0').value,document.getElementById('b1').value].join('|')") == "1|"
    assert _evaluate(browser.page, f"{_STAMPS_ANYWHERE}") == 0


@_integration
def test_an_identifier_change_after_the_password_was_written_says_the_password_was_written(browser):
    """with_identifier: the password is written first; its input listener then replaces the identifier field.
    The refusal says the password field was written (no value), not 'Nothing was written'."""
    from tools.browser_vault_tool import _in_page_refusal

    _evaluate(browser.page, "document.getElementById('p').value='';document.getElementById('p').addEventListener('input',"
                            "function(){const e=document.getElementById('e');e.replaceWith(e.cloneNode(true))},{once:true})")
    controls = _inspect(browser, "fb1id")
    pw = next(c for c in controls if c.type == "password")
    ident = next(c for c in controls if c.type == "email")
    fills = [{"index": pw.index, "token": "current-password", "value": "FB1-password"},
             {"index": ident.index, "token": "email", "value": "fb1@example.invalid"}]
    raw = json.loads(_evaluate(browser.page, build_fill_js(fills, browser.origin, "fb1id")))
    assert raw == {"refused": "target_changed", "written": 1, "roles": ["current-password"]}, raw
    out = _in_page_refusal(raw, browser.origin)
    assert out["error_type"] == "target_changed" and out["success"] is False, out
    assert "1 field was written (or a write was attempted) before this and was not cleared: the password field." in out["error"], out
    assert "FB1-password" not in json.dumps(out) and "Nothing was written" not in out["error"]
    assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "FB1-password", "submits": 0}


@_integration
def test_a_code_box_written_then_emptied_by_the_page_is_written_but_not_held(browser):
    from agent.vault_login_classifier import build_otp_fills, classify_otp_controls
    from tools.browser_vault_tool import _code_write_outcome

    _evaluate(browser.page, "const f=document.querySelector('form');const b=document.createElement('input');b.id='otp';"
                            "b.name='otp';b.type='text';b.autocomplete='one-time-code';f.appendChild(b);"
                            "b.addEventListener('input',function(){this.value=''})")
    fills = build_otp_fills(classify_otp_controls(_inspect(browser, "fb1held")), "482913")
    raw = json.loads(_evaluate(browser.page, build_fill_js(fills, browser.origin, "fb1held")))
    assert raw == {"filled": 0, "written": 1, "roles": ["one-time-code"]}, raw
    out = _code_write_outcome(raw, browser.origin, "user", len(fills))
    assert out["success"] is False and "next" not in out and "482913" not in json.dumps(out), out
    assert "1 code box was written, but none held a value after the page's own events" in out["error"], out


# ── the deadline is not an execution fence (DEADLINE-R01, t_ce72d12b/run23376) ──────────────────────────────
# Production `hermes vault fill` (_cmd_fill) runs in a separate OS process; only the vault store is a dummy.

_CLI_CHILD = """
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from agent.vault_store import VaultStore
from hermes_cli.vault import _cmd_fill
store = VaultStore(base_dir=Path(sys.argv[3]))
meta = store.add_item('login', 'Deadline dummy', {'identifier_type': 'email', 'identifier': 'a@x.invalid',
                                                  'password': 'DEADLINE-DUMMY'}, origin=sys.argv[4])
with patch('agent.vault_store.get_vault_store', return_value=store):
    _cmd_fill(SimpleNamespace(handle=meta.id, cdp=sys.argv[1], target_id=sys.argv[2], backend='local'))
"""


def _cli_process(browser, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    return subprocess.Popen([sys.executable, "-c", _CLI_CHILD, browser.http, browser.page["id"], str(tmp_path / "vault"),
                             browser.origin], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _cli_result(proc):
    out, err = proc.communicate(timeout=60)
    return proc.returncode, json.loads(out.strip().splitlines()[-1]), err


class _Cdp:
    """A separate CDP connection to the tab (here: the DevTools debugger a person might have open)."""

    def __init__(self, page):
        from websockets.sync.client import connect

        self.ws, self.next_id = connect(page["webSocketDebuggerUrl"], open_timeout=5), 0

    def recv(self, timeout=30):
        return json.loads(self.ws.recv(timeout=timeout))

    def call(self, method, params=None, timeout=30):
        self.next_id += 1
        call_id = self.next_id
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            msg = self.recv(timeout)
            if msg.get("id") == call_id:
                assert "error" not in msg, msg
                return msg["result"]

    def paused(self, timeout=30):
        while True:
            msg = self.recv(timeout)
            if msg.get("method") == "Debugger.paused":
                return msg["params"]


def _truthful_timeout(code, out):
    assert code == 1 and out["success"] is False and out["error_type"] == "fill_timeout" and "next" not in out, out
    assert out["may_complete_later"] is True and "may still complete" in out["error"], out
    assert "check the tab" in out["error"] and "not established" in out["error"] and "cannot write" not in out["error"]
    assert "termination_requested" not in out and "terminat" not in out["error"].lower(), out
    assert "DEADLINE-DUMMY" not in json.dumps(out)


def _filled_exactly(code, out):
    assert code == 0 and out["success"] is True and out["filled_fields"] == 1, out
    assert "DEADLINE-DUMMY" not in json.dumps(out)


@_integration
def test_a_debugger_pause_after_the_last_gate_is_reported_truthfully(browser, tmp_path):
    """DEADLINE-R01 witness A: a debugger pauses the production fill script on its write, after its last
    budget check. The isolated world does not help here: the command reports that the script may still
    complete, and when the debugger resumes it, the write happens after the command returned (a limit)."""
    d = _Cdp(browser.page)
    proc = None
    try:
        d.call("Debugger.enable")
        entry = d.call("Debugger.setInstrumentationBreakpoint", {"instrumentation": "beforeScriptExecution"})["breakpointId"]
        proc = _cli_process(browser, tmp_path)
        while True:  # stop each script at entry until the secret-bearing fill script appears
            frame = d.paused()["callFrames"][0]
            source = d.call("Debugger.getScriptSource", {"scriptId": frame["location"]["scriptId"]})["scriptSource"]
            if "const fills =" in source and "DEADLINE-DUMMY" in source:
                line = next(i for i, s in enumerate(source.splitlines()) if "setter.set.call(el, f.value)" in s)
                bp = d.call("Debugger.setBreakpoint", {"location": {"scriptId": frame["location"]["scriptId"],
                                                                    "lineNumber": line}})["breakpointId"]
                d.call("Debugger.removeBreakpoint", {"breakpointId": entry})
                d.call("Debugger.resume")
                d.paused()
                break
            d.call("Debugger.resume")
        code, out, _ = _cli_result(proc)
        _truthful_timeout(code, out)
        d.call("Debugger.removeBreakpoint", {"breakpointId": bp})
        d.call("Debugger.resume")
        after = _settled_after(browser)
        print("DEADLINE_A=" + json.dumps({"result": out, "after": after}))
        # Measured: resumed past its last gate, the script writes the exact value after the command returned
        # (the result said it may), and its own finally removes the stamps.
        assert after == {"pw": "DEADLINE-DUMMY", "submits": 0, "stamps": 0}, after
    finally:
        try:
            d.call("Debugger.disable", timeout=5)
        finally:
            d.ws.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()


_LATE_WRITE_WINDOW_S = 4.0  # longer than FILL_PAGE_BUDGET_S: a late write would have landed by then


def _no_late_write(browser, extra):
    """The page state right after the command and again once a late write would have landed."""
    import time as _time

    first = _settled_after(browser, extra=extra)
    _time.sleep(_LATE_WRITE_WINDOW_S)  # health: allow HX013 -- the observation window for a write after the command returned
    return first, _settled_after(browser, extra=extra)


@_integration
def test_a_page_clock_that_blocks_at_the_last_gate_is_reported_truthfully(browser, tmp_path):
    """DEADLINE-R01 witness B: page code replaces performance.now with one that blocks. The fill script reads the
    clock of its isolated world, which page code cannot replace: the page's function is never called, the fill
    completes with the exact value, and nothing is written later."""
    _evaluate(browser.page, """(() => {
      const realNow = performance.now.bind(performance); let calls = 0;
      window.clockCalls = []; window.pwWrites = [];
      document.getElementById('p').addEventListener('input', () => window.pwWrites.push(Date.now()));
      performance.now = function () {
        const snap = realNow(); calls++; window.clockCalls.push(calls);
        if (calls >= 1) { const until = Date.now() + 18000; while (Date.now() < until) {} }
        return snap;
      };
    })()""")
    proc = _cli_process(browser, tmp_path)
    code, out, _ = _cli_result(proc)
    _filled_exactly(code, out)
    first, later = _no_late_write(browser, "clock:window.clockCalls,writes:window.pwWrites.length")
    print("DEADLINE_B=" + json.dumps({"result": out, "after": first, "later": later}))
    assert first == later == {"pw": "DEADLINE-DUMMY", "submits": 0, "stamps": 0, "clock": [], "writes": 1}, (first, later)


@_integration
def test_a_page_setter_that_blocks_before_the_write_is_reported_truthfully(browser, tmp_path):
    """DEADLINE-R01 witness C (page clock untouched): page code wraps HTMLInputElement.prototype.value's setter
    and blocks inside it. The isolated world has its own prototype, so the page's wrapper is never called: the
    fill completes with the exact value and nothing is written later."""
    _evaluate(browser.page, """(() => {
      const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      window.pwWrites = []; window.wrapperCalls = 0;
      document.getElementById('p').addEventListener('input', () => window.pwWrites.push(Date.now()));
      Object.defineProperty(HTMLInputElement.prototype, 'value', {...native, set(value) {
        window.wrapperCalls++;
        if (value === 'DEADLINE-DUMMY') { const until = Date.now() + 18000; while (Date.now() < until) {} }
        native.set.call(this, value);
      }});
    })()""")
    proc = _cli_process(browser, tmp_path)
    code, out, _ = _cli_result(proc)
    _filled_exactly(code, out)
    first, later = _no_late_write(browser, "writes:window.pwWrites.length,wrapper:window.wrapperCalls")
    print("DEADLINE_C=" + json.dumps({"result": out, "after": first, "later": later}))
    assert first == later == {"pw": "DEADLINE-DUMMY", "submits": 0, "stamps": 0, "writes": 1, "wrapper": 0}, (first, later)


@_integration
def test_a_page_setter_that_opens_a_dialog_before_the_write_is_reported_truthfully(browser, tmp_path):
    """DEADLINE-R01, formerly the limit: the page's wrapped setter would open an alert after the last gate. In the
    isolated world the wrapper is never called, so no dialog opens and the fill completes at once."""
    _evaluate(browser.page, """(() => {
      const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      window.pwWrites = []; window.wrapperCalls = 0;
      document.getElementById('p').addEventListener('input', () => window.pwWrites.push(Date.now()));
      Object.defineProperty(HTMLInputElement.prototype, 'value', {...native, set(value) {
        window.wrapperCalls++;
        if (value === 'DEADLINE-DUMMY') alert('setter');
        native.set.call(this, value);
      }});
    })()""")
    person = _PersonAtTheBrowser(browser.page)
    try:
        proc = _cli_process(browser, tmp_path)
        code, out, _ = _cli_result(proc)
        _filled_exactly(code, out)
        assert not person.wait_open(1)
        after = _settled_after(browser, extra="writes:window.pwWrites.length,wrapper:window.wrapperCalls")
        print("DEADLINE_SETTER_ALERT=" + json.dumps({"result": out, "after": after}))
        assert after == {"pw": "DEADLINE-DUMMY", "submits": 0, "stamps": 0, "writes": 1, "wrapper": 0}, after
    finally:
        person.close()


def _settled_after(browser, extra=""):
    """The page state once the page answers again (a separate connection with a long wait)."""
    from websockets.sync.client import connect

    expression = ("JSON.stringify({pw:document.getElementById('p').value,submits:window.submits,"
                  f"stamps:{_STAMPS_ANYWHERE}" + ("," + extra if extra else "") + "})")
    with connect(browser.page["webSocketDebuggerUrl"], open_timeout=5) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expression,
                                                                             "returnByValue": True}}))
        while True:
            msg = json.loads(ws.recv(timeout=30))
            if msg.get("id") == 1:
                return json.loads(msg["result"]["result"]["value"])


# ── the CDP isolated world (Fable, review-queue 14, UPDATE / UPDATE 2) ──────────────────────────────────────

class _WorldConn:
    """A direct CDP connection to the tab that creates isolated worlds and evaluates in them."""

    def __init__(self, page):
        from websockets.sync.client import connect

        self.ws, self.next_id = connect(page["webSocketDebuggerUrl"], open_timeout=5), 0

    def call(self, method, params=None):
        self.next_id += 1
        call_id = self.next_id
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=10))
            if msg.get("id") == call_id:
                assert "error" not in msg, msg
                return msg["result"]

    def world(self, name):
        frame = self.call("Page.getFrameTree")["frameTree"]["frame"]["id"]
        return self.call("Page.createIsolatedWorld", {"frameId": frame, "worldName": name})["executionContextId"]

    def eval(self, context_id, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "contextId": context_id,
                                                "returnByValue": True})
        assert "exceptionDetails" not in result, result
        return result["result"].get("value")


@_integration
def test_a_second_command_cannot_see_the_first_commands_registry(browser):
    """Point 5: the registry lives on the isolated world's window, invisible to the page. A new command's world
    does not see it; a reused world name returns the same world (measured), and there the registry is keyed by
    nonce, so another inspection's fill finds no registry of its own and writes nothing."""
    _evaluate(browser.page, "document.getElementById('p').value=''")
    w = _WorldConn(browser.page)
    try:
        first = w.world("hermes-vault:first")
        index = next(c["index"] for c in json.loads(w.eval(first, build_inspection_js("first"))) if c["type"] == "password")
        assert w.eval(first, f"JSON.stringify({_REGISTRIES})") == '["hermesVaultSlots:first"]'
        assert _evaluate(browser.page, f"{_REGISTRIES}.length") == 0  # the page cannot see it
        second = w.world("hermes-vault:second")
        assert second != first and w.eval(second, f"{_REGISTRIES}.length") == 0
        w.eval(second, build_inspection_js("second"))  # re-stamps the shared DOM with "second:<index>"
        again = w.world("hermes-vault:first")
        assert again == first  # the same name is the same world
        fill = build_fill_js([{"index": index, "token": "current-password", "value": "REUSED-WORLD"}], browser.origin,
                             "second", require_empty=True)
        assert json.loads(w.eval(again, fill)) == {"refused": "target_changed", "written": 0, "roles": []}
        assert w.eval(first, f"JSON.stringify({_REGISTRIES})") == '["hermesVaultSlots:first"]'  # untouched
        w.eval(first, build_clear_stamps_js("first"))
        w.eval(second, build_clear_stamps_js("second"))
        assert w.eval(first, f"{_REGISTRIES}.length") == 0 and w.eval(second, f"{_REGISTRIES}.length") == 0
        assert _state(browser) == {"slots": 0, "email": "human@example.invalid", "pw": "", "submits": 0}
    finally:
        w.ws.close()


@_integration
def test_a_lying_value_getter_cannot_hide_a_value_from_the_inspection(browser, store, capsys):
    """Point 6: the page's value getter claims every field is empty; the inspection reads the field in the
    isolated world, sees the person's value and refuses before anything is resolved."""
    _evaluate(browser.page, """(() => {
      const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      document.getElementById('p').value = 'HUMAN-VALUE';
      Object.defineProperty(HTMLInputElement.prototype, 'value', {...native, get() { return ''; }});
    })()""")
    code, out, resolved = _cli_fill(browser, store, "LIE-vault-value", capsys)
    assert code == 1 and out["error_type"] == "field_not_empty" and resolved == [], out
    assert _evaluate(browser.page, "document.getElementById('p').value") == ""  # the page's lie, in its own world
    w = _WorldConn(browser.page)
    try:
        assert w.eval(w.world("observer"), "document.getElementById('p').value") == "HUMAN-VALUE"  # untouched
    finally:
        w.ws.close()
    assert _registries(browser) == 0 and _evaluate(browser.page, _STAMPS_ANYWHERE) == 0


@_integration
def test_a_lying_value_getter_cannot_fake_the_final_comparison(browser, store, capsys):
    """Point 6: the page changes the written value and its getter claims the field holds exactly the stored
    value; the final comparison runs in the isolated world and refuses (value_mismatch)."""
    _evaluate(browser.page, """(() => {
      const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      const p = document.getElementById('p'); p.value = '';
      p.addEventListener('input', function () { native.set.call(this, native.get.call(this) + '-PAGE'); }, {once: true});
      Object.defineProperty(HTMLInputElement.prototype, 'value', {...native, get() { return 'LIE-exact-value'; }});
    })()""")
    code, out, resolved = _cli_fill(browser, store, "LIE-exact-value", capsys)
    assert code == 1 and out["error_type"] == "value_mismatch" and len(resolved) == 1 and "next" not in out, out
    assert out["field_cleared"] is True and "LIE-exact-value" not in json.dumps(out), out
    assert _registries(browser) == 0 and _evaluate(browser.page, _STAMPS_ANYWHERE) == 0


@_integration
def test_a_forged_json_stringify_cannot_forge_the_report(browser, store, capsys):
    """Point 6: the page replaces JSON.stringify with one that always reports a fill, and turns the field into a
    text box on focus. The fill script serializes with the isolated world's own JSON: the true refusal."""
    _evaluate(browser.page, "document.getElementById('e').focus();document.getElementById('p').value='';"
                            "document.getElementById('p').addEventListener('focus',function(){this.type='text'},{once:true});"
                            "JSON.stringify=function(){return '{\"filled\":1,\"written\":1,\"roles\":[\"current-password\"]}'}")
    code, out, resolved = _cli_fill(browser, store, "FORGED-JSON-value", capsys)
    assert code == 1 and out["success"] is False and out["error_type"] == "target_changed" and len(resolved) == 1, out
    assert "Nothing was written." in out["error"], out
    assert _evaluate(browser.page, "document.getElementById('p').value") == ""
    _evaluate(browser.page, "document.getElementById('p').type='password'")


def _navigate(browser, url):
    _evaluate(browser.page, f"location.href={json.dumps(url)};1")
    wait_until(lambda: _evaluate_quiet(browser.page, "location.href") == url and _ready(browser.page), timeout=15,
               what=f"the tab showing {url}")


def _evaluate_quiet(page, expression):
    try:
        return _evaluate(page, expression)
    except (OSError, AssertionError, TimeoutError):
        return None


@_integration
@pytest.mark.parametrize("site", ["same-site", "cross-site"])
def test_a_navigation_before_the_fill_script_starts(browser, store, capsys, site):
    """Point 1 / navigation context loss: the tab moves to a new document after the clock read and before the
    fill script is sent, so the script is evaluated in the old document's isolated world, which is gone; it
    never runs in the new document. Same site (same renderer): "Cannot find context with specified id", the
    one reply that means the script never started, so nothing was written and the new document has no stamps.
    Cross site (another renderer process): measured, the reply to the world's uniqueContextId is a different
    error ("uniqueContextId not found"), which the strict rule does not read as "never started": the result
    says not established. Either way nothing was written to the new document."""
    from tools import browser_vault_tool

    port = browser.origin.rsplit(":", 1)[1]
    url = (browser.origin if site == "same-site" else f"http://localhost:{port}") + "/next"
    _evaluate(browser.page, "document.getElementById('p').value=''")
    real_secret, replies = browser_vault_tool._eval_js_secret, []

    def secret(*a, **k):
        _navigate(browser, url)
        reply = real_secret(*a, **k)
        replies.append(reply)
        return reply

    with patch.object(browser_vault_tool, "_eval_js_secret", secret):
        code, out, resolved = _cli_fill(browser, store, "NAV-vault-value", capsys)
    print("NAV_BEFORE_START=" + json.dumps({"site": site, "result": out, "reply_error": replies[0].get("error")}))
    assert code == 1 and len(resolved) == 1 and "next" not in out and out["success"] is False, out
    if site == "same-site":
        assert "Cannot find context with specified id" in replies[0]["error"], replies
        assert out["error_type"] == "page_navigated" and "Nothing was written" in out["error"], out
        assert "carries none of this inspection's stamps" in out["stamps"], out
    else:
        assert "uniqueContextId not found" in replies[0]["error"], replies
        assert out["error_type"] == "fill_unconfirmed" and out["may_complete_later"] is True, out
        assert "not established" in out["error"] and "Nothing was written" not in out["error"], out
    assert _evaluate(browser.page, "location.href") == url
    assert _evaluate(browser.page, f"JSON.stringify([document.getElementById('p').value,{_STAMPS_ANYWHERE}])") == '["",0]'


@_integration
def test_a_navigation_while_the_fill_script_runs_is_not_established(browser, tmp_path):
    """Point 1: a navigation while the script is running (held by a debugger) is NOT "nothing written". Measured:
    the reply is "Inspected target navigated or closed", and the command says the script may still complete
    and the field's state is not established."""
    d = _Cdp(browser.page)
    proc = None
    try:
        d.call("Debugger.enable")
        entry = d.call("Debugger.setInstrumentationBreakpoint", {"instrumentation": "beforeScriptExecution"})["breakpointId"]
        proc = _cli_process(browser, tmp_path)
        while True:
            frame = d.paused()["callFrames"][0]
            source = d.call("Debugger.getScriptSource", {"scriptId": frame["location"]["scriptId"]})["scriptSource"]
            if "const fills =" in source and "DEADLINE-DUMMY" in source:
                break
            d.call("Debugger.resume")
        d.call("Debugger.removeBreakpoint", {"breakpointId": entry})
        d.ws.send(json.dumps({"id": 9000, "method": "Page.navigate", "params": {"url": browser.origin + "/moved"}}))
        code, out, err = _cli_result(proc)
        print("NAV_MID_SCRIPT=" + json.dumps({"result": out}))
        assert code == 1 and out["success"] is False and out["error_type"] == "fill_unconfirmed", (out, err[-2000:])
        assert out["may_complete_later"] is True and "not established" in out["error"] and "check the tab" in out["error"]
        assert "navigated or closed while the fill script was running" in out["error"], out
        assert "Nothing was written" not in out["error"] and "DEADLINE-DUMMY" not in json.dumps(out)
    finally:
        with contextlib.suppress(Exception):
            d.call("Debugger.disable", timeout=5)
        d.ws.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()


@_integration
def test_a_setter_that_writes_and_then_throws_counts_as_written(browser):
    """N-h, in the page's own world (a plugin page backend has no isolated world): the page's setter writes and
    then throws. The role is recorded before the setter is called, so the report says written or attempted."""
    _evaluate(browser.page, """(() => {
      const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      document.getElementById('p').value = '';
      Object.defineProperty(HTMLInputElement.prototype, 'value', {...native, set(v) {
        native.set.call(this, v); if (v === 'NH-value') throw new Error('PAGE-THROW');
      }});
    })()""")
    nonce = "nhthrow"
    index = next(c["index"] for c in json.loads(_evaluate(browser.page, build_inspection_js(nonce))) if c["type"] == "password")
    raw = json.loads(_evaluate(browser.page, build_fill_js(
        [{"index": index, "token": "current-password", "value": "NH-value"}], browser.origin, nonce, require_empty=True)))
    assert raw == {"filled": 0, "written": 1, "roles": ["current-password"]}, raw
    assert _evaluate(browser.page, "document.getElementById('p').value") == "NH-value"
    assert _evaluate(browser.page, _STAMPS_ANYWHERE) == 0
