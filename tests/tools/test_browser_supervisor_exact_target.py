"""Target-scoped CDP supervisor: ``attach_target_id`` attaches exactly one page target and
``dialog_bridge=False`` installs nothing on any tab, so a supervisor on a browser whose other tabs
belong to someone else leaves those tabs untouched. No real Chrome: ``_cdp`` is a recording fake."""

from __future__ import annotations

import asyncio

import pytest

from tools import browser_supervisor as bs

_BRIDGE_METHODS = {"Page.addScriptToEvaluateOnNewDocument", "Fetch.enable"}

_TARGETS = [
    {"targetId": "OTHER", "type": "page", "url": "https://other.example/inbox", "attached": False},
    {"targetId": "LOGIN", "type": "page", "url": "https://mail.example/login", "attached": True},
    {"targetId": "WORKER", "type": "service_worker", "url": "https://mail.example/sw.js", "attached": False},
]


class _FakeCDP:
    """Answers the CDP calls the supervisor makes and records (method, params, session) for each."""

    def __init__(self, targets=_TARGETS, probe_value=True):
        self.targets, self.probe_value, self.calls = targets, probe_value, []

    async def __call__(self, method, params=None, *, session_id=None, timeout=10.0):
        self.calls.append((method, params or {}, session_id))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": self.targets}}
        if method == "Target.getTargetInfo":
            info = next((t for t in self.targets if t["targetId"] == params["targetId"]), None)
            return {"result": {"targetInfo": info}} if info else {"result": {}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": f"S-{params['targetId']}"}}
        if method == "Target.createTarget":
            return {"result": {"targetId": "NEW"}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": self.probe_value}}}
        return {"result": {}}

    def methods(self):
        return [m for m, _, _ in self.calls]

    def attached(self):
        return [p["targetId"] for m, p, _ in self.calls if m == "Target.attachToTarget"]


def _supervisor(monkeypatch, cdp, **kwargs):
    sup = bs.CDPSupervisor(task_id="exact", cdp_url="ws://127.0.0.1:1/devtools/browser/x", **kwargs)
    monkeypatch.setattr(sup, "_cdp", cdp)

    async def quiet(method, params, *, session_id, timeout, what):
        await cdp(method, params, session_id=session_id, timeout=timeout)

    monkeypatch.setattr(sup, "_cdp_quiet", quiet)
    return sup


def test_exact_target_attaches_only_that_target_and_installs_no_bridge(monkeypatch):
    cdp = _FakeCDP()
    sup = _supervisor(monkeypatch, cdp, attach_target_id="LOGIN", dialog_bridge=False)
    asyncio.run(sup._attach_initial_page())
    assert cdp.attached() == ["LOGIN"]
    assert sup._page_session_id == "S-LOGIN"
    assert not _BRIDGE_METHODS & set(cdp.methods())
    assert all(p.get("expression") is None for m, p, _ in cdp.calls if m == "Runtime.evaluate")
    assert all(s in (None, "S-LOGIN") for _, _, s in cdp.calls)


@pytest.mark.parametrize("target_id", ["MISSING", "WORKER"])
def test_exact_target_missing_or_not_a_page_fails_closed(monkeypatch, target_id):
    cdp = _FakeCDP()
    sup = _supervisor(monkeypatch, cdp, attach_target_id=target_id, dialog_bridge=False)
    with pytest.raises(bs.SupervisorTargetError):
        asyncio.run(sup._attach_initial_page())
    assert cdp.methods() == ["Target.getTargets"]  # no attach, no createTarget, no fallback tab


def test_default_supervisor_is_unchanged_first_page_and_bridge(monkeypatch):
    cdp = _FakeCDP()
    sup = _supervisor(monkeypatch, cdp)
    asyncio.run(sup._attach_initial_page())
    assert cdp.attached() == ["OTHER"]
    assert _BRIDGE_METHODS <= set(cdp.methods())


def test_bridge_off_alone_still_attaches_first_page_but_installs_nothing(monkeypatch):
    cdp = _FakeCDP()
    sup = _supervisor(monkeypatch, cdp, dialog_bridge=False)
    asyncio.run(sup._attach_initial_page())
    assert not _BRIDGE_METHODS & set(cdp.methods())


@pytest.mark.parametrize("bad", ["", "   ", 7])
def test_attach_target_id_must_be_a_real_id(bad):
    with pytest.raises(ValueError):
        bs.CDPSupervisor(task_id="t", cdp_url="ws://x", attach_target_id=bad)


class TestFocusExactTarget:
    def _ready(self, monkeypatch, **cdp_kwargs):
        cdp = _FakeCDP(**cdp_kwargs)
        sup = _supervisor(monkeypatch, cdp, attach_target_id="LOGIN", dialog_bridge=False)
        sup._page_session_id = "S-LOGIN"
        return sup, cdp

    def test_on_origin_with_form_focuses_without_new_attach(self, monkeypatch):
        sup, cdp = self._ready(monkeypatch)
        out = asyncio.run(sup._focus_exact_target("https://mail.example", "probe()", timeout=1))
        assert out == {"ok": True, "url": "https://mail.example/login"}
        assert "Target.attachToTarget" not in cdp.methods() and "Target.getTargets" not in cdp.methods()

    def test_other_origin_fails_even_if_another_tab_matches(self, monkeypatch):
        sup, cdp = self._ready(monkeypatch)
        out = asyncio.run(sup._focus_exact_target("https://other.example", "probe()", timeout=1))
        assert out["ok"] is False
        assert "Target.attachToTarget" not in cdp.methods()

    def test_form_probe_false_fails(self, monkeypatch):
        sup, _ = self._ready(monkeypatch, probe_value=False)
        assert asyncio.run(sup._focus_exact_target("https://mail.example", "probe()", timeout=1))["ok"] is False

    def test_target_gone_fails(self, monkeypatch):
        sup, _ = self._ready(monkeypatch, targets=[t for t in _TARGETS if t["targetId"] != "LOGIN"])
        assert asyncio.run(sup._focus_exact_target("https://mail.example", None, timeout=1))["ok"] is False

    def test_not_attached_fails(self, monkeypatch):
        sup, _ = self._ready(monkeypatch)
        sup._page_session_id = None
        assert asyncio.run(sup._focus_exact_target("https://mail.example", None, timeout=1))["ok"] is False


def test_bridge_off_makes_no_automatic_dialog_choice(monkeypatch):
    cdp = _FakeCDP()
    sup = _supervisor(monkeypatch, cdp, attach_target_id="LOGIN", dialog_bridge=False, dialog_policy="auto_accept")
    asyncio.run(sup._on_dialog_opening({"type": "confirm", "message": "Delete?"}, "S-LOGIN"))
    assert sup.snapshot().pending_dialogs == () and sup._dialog_watchdogs == {}
    assert "Page.handleJavaScriptDialog" not in cdp.methods()


class _Running:
    """A registered supervisor stand-in that looks healthy and records stop()."""

    def __init__(self, cdp_url, attach_target_id=None, dialog_bridge=True):
        self.cdp_url, self.attach_target_id, self.dialog_bridge = cdp_url, attach_target_id, dialog_bridge
        self.stopped = False

        class _Alive:
            def is_alive(self):
                return True

            def is_running(self):
                return True

        self._thread = self._loop = _Alive()

    def stop(self, timeout=5.0):
        self.stopped = True


@pytest.fixture
def registry():
    reg = bs._SupervisorRegistry()
    yield reg


@pytest.mark.parametrize(
    "existing, request_kwargs",
    [
        ({"cdp_url": "ws://a"}, {"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": False}),
        ({"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": False}, {"cdp_url": "ws://a"}),
        ({"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": False},
         {"cdp_url": "ws://b", "attach_target_id": "LOGIN", "dialog_bridge": False}),
        ({"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": False},
         {"cdp_url": "ws://a", "attach_target_id": "OTHER", "dialog_bridge": False}),
        ({"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": False},
         {"cdp_url": "ws://a", "attach_target_id": "LOGIN", "dialog_bridge": True}),
    ],
)
def test_scoped_mismatch_refuses_and_leaves_the_running_supervisor(registry, existing, request_kwargs):
    running = _Running(**existing)
    registry._by_task["t"] = running
    url = request_kwargs.pop("cdp_url")
    with pytest.raises(bs.SupervisorScopeConflict):
        registry.get_or_start("t", url, **request_kwargs)
    assert running.stopped is False and registry.get("t") is running


def test_same_scope_reuses_the_running_supervisor(registry):
    running = _Running("ws://a", "LOGIN", False)
    registry._by_task["t"] = running
    assert registry.get_or_start("t", "ws://a", attach_target_id="LOGIN", dialog_bridge=False) is running


def test_legacy_endpoint_change_still_replaces(registry, monkeypatch):
    running = _Running("ws://a")
    registry._by_task["t"] = running
    started = []
    monkeypatch.setattr(bs.CDPSupervisor, "start", lambda self, timeout=15.0: started.append(self.cdp_url))
    new = registry.get_or_start("t", "ws://b")
    assert running.stopped is True and started == ["ws://b"] and registry.get("t") is new
