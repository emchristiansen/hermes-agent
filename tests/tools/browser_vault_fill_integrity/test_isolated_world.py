"""The vault fill and enter_code in the command's CDP isolated world, against a real headless Chrome
(opt-in like ``tests/tools/test_browser_vault_fill_integrity.py``, whose helpers these use; its fixtures come
through this package's conftest):
registry isolation, page tampering, navigation before and while the script runs, and enter_code's exits."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_login_classifier import build_clear_stamps_js, build_fill_js, build_inspection_js
from tests.tools.test_browser_vault_fill_integrity import (
    _GONE_CROSS_SITE,
    _GONE_SAME_SITE,
    _REGISTRIES,
    _STAMPS_ANYWHERE,
    _Cdp,
    _cli_fill,
    _cli_process,
    _cli_result,
    _evaluate,
    _integration,
    _navigate,
    _registries,
    _state,
)
from tools.browser_vault_tool import _fill_outcome


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




@_integration
@pytest.mark.parametrize("site", ["same-site", "cross-site"])
def test_a_navigation_before_the_fill_script_starts(browser, store, capsys, site):
    """Point 1 / navigation context loss: the tab moves to a new document after the clock read and before the
    fill script is sent, so the script is evaluated in the old document's isolated world, which is gone; it
    never runs in the new document. Same site (same renderer): "Cannot find context with specified id", the
    one reply that means the script never started, so nothing was written and the new document has no stamps.
    Cross site (another renderer process): measured, the reply to the world's uniqueContextId is -32602
    "uniqueContextId not found", and the expression never runs there either; by the agreed rule it reads as
    never started too."""
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
    expected = _GONE_SAME_SITE if site == "same-site" else _GONE_CROSS_SITE
    assert replies[0]["protocol_error"] == expected and replies[0]["addressed"] == "uniqueContextId", replies
    assert out["error_type"] == "page_navigated" and "Nothing was written" in out["error"], out
    assert "carries none of this inspection's stamps" in out["stamps"], out
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
    fills = [{"index": index, "token": "current-password", "value": "NH-value"}]
    out = json.loads(_fill_outcome(raw, SimpleNamespace(name="local"), SimpleNamespace(kind="login", has_otp=False),
                                   browser.origin, fills))  # FC-1: the tool result names the write
    assert out["success"] is False and "next" not in out and "NH-value" not in json.dumps(out), out
    assert out["error"].startswith("1 field was written (or a write was attempted) and was not cleared: the password field."), out
    assert _evaluate(browser.page, "document.getElementById('p').value") == "NH-value"
    assert _evaluate(browser.page, _STAMPS_ANYWHERE) == 0


@_integration
def test_a_navigation_before_the_second_gate_is_reported_as_a_navigation(browser, store, capsys):
    """FC-2 in Chromium: the tab moves to a new document after inspection and before the gate that precedes
    sending. The value is not sent, nothing is written, and the refusal says the page moved (not that it did
    not answer)."""
    from tools import browser_vault_tool

    _evaluate(browser.page, "document.getElementById('p').value=''")
    real, calls = browser_vault_tool._send_gate, []

    def gate(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            _navigate(browser, browser.origin + "/next")
        return real(*a, **k)

    sent = []
    real_secret = browser_vault_tool._eval_js_secret
    with patch.object(browser_vault_tool, "_send_gate", gate), \
         patch.object(browser_vault_tool, "_eval_js_secret", lambda *a, **k: sent.append(1) or real_secret(*a, **k)):
        code, out, resolved = _cli_fill(browser, store, "FC2-vault-value", capsys)
    assert code == 1 and out["error_type"] == "page_navigated" and len(resolved) == 1 and sent == [], out
    assert "the value was not sent; nothing was written" in out["error"] and "did not answer" not in out["error"], out
    assert "carries none of this inspection's stamps" in out["stamps"], out
    assert _evaluate(browser.page, f"JSON.stringify([document.getElementById('p').value,{_STAMPS_ANYWHERE}])") == '["",0]'


_OTP_BOX = ("const f=document.querySelector('form');const b=document.createElement('input');b.id='otp';b.name='otp';"
            "b.setAttribute('autocomplete','one-time-code');f.appendChild(b);")


def _enter_code(browser, prompt):
    """browser_vault_enter_code (no saved login) through an exact-target supervisor on the test tab, with
    ``prompt`` as the user's code prompt; returns the tool result and the worlds created."""
    from agent.vault_backends import unlock
    from tools import browser_vault_tool
    from tools.browser_session_cdp import resolve_cdp_override
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    task_id = "enter-code-" + browser.page["id"][:8]
    SUPERVISOR_REGISTRY.get_or_start(task_id, resolve_cdp_override(browser.http), attach_target_id=browser.page["id"],
                                     dialog_bridge=False)
    unlock.set_code_prompt_callback(prompt)
    try:
        with patch.object(unlock, "can_prompt_here", lambda prompt=None: True), browser_vault_tool.supervisor_only():
            return json.loads(browser_vault_tool.browser_vault_enter_code(task_id=task_id))
    finally:
        unlock.set_code_prompt_callback(None)
        SUPERVISOR_REGISTRY.stop(task_id)


@_integration
def test_enter_code_fills_exactly_and_retires_its_inspection(browser):
    _evaluate(browser.page, _OTP_BOX)
    out = _enter_code(browser, lambda site, hint: "739415")
    assert out["success"] is True and out["filled_fields"] == 1 and "739415" not in json.dumps(out), out
    assert _evaluate(browser.page, f"JSON.stringify([document.getElementById('otp').value,{_STAMPS_ANYWHERE}])") == '["739415",0]'


@_integration
def test_enter_code_declined_leaves_no_stamps(browser):
    """FC-6: the user declines the code; the inspection's stamps, registry and observer are retired anyway."""
    _evaluate(browser.page, _OTP_BOX)
    out = _enter_code(browser, lambda site, hint: "")
    assert out["error_type"] == "code_declined", out
    assert _evaluate(browser.page, f"JSON.stringify([document.getElementById('otp').value,{_STAMPS_ANYWHERE}])") == '["",0]'


@_integration
def test_enter_code_navigation_while_the_user_types_is_a_navigation(browser):
    """FC-2, the enter_code case: the tab moves to a new document while the user is asked for the code. The code
    is not sent, nothing is written, the refusal says the page moved, and the new document has no stamps."""
    _evaluate(browser.page, _OTP_BOX)

    def prompt(site, hint):
        _navigate(browser, browser.origin + "/next")
        return "739415"

    out = _enter_code(browser, prompt)
    assert out["success"] is False and out["error_type"] == "page_navigated", out
    assert "the value was not sent; nothing was written" in out["error"] and "did not answer" not in out["error"], out
    assert "carries none of this inspection's stamps" in out["stamps"] and "739415" not in json.dumps(out), out
    assert _evaluate(browser.page, f"JSON.stringify([location.pathname,{_STAMPS_ANYWHERE}])") == '["/next",0]'
