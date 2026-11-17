"""``hermes vault fill``: one exact target, no dialog bridge, only the named backend, nothing resolved
or written when the attach is refused, and the supervisor always stopped. No browser: the registry and
the fill are recording fakes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import vault as vault_cli
from tools import browser_supervisor as bs

_WS = "ws://127.0.0.1:9/devtools/browser/abc"


class _Registry:
    def __init__(self, raises=None):
        self.started, self.stopped, self.raises = [], [], raises

    def get_or_start(self, task_id, cdp_url, **kwargs):
        self.started.append((task_id, cdp_url, kwargs))
        if self.raises:
            raise self.raises

    def stop(self, task_id):
        self.stopped.append(task_id)


@pytest.fixture
def wired(monkeypatch):
    calls = {"fill": []}

    def fake_fill(handle, task_id, **kwargs):
        calls["fill"].append((handle, task_id, kwargs))
        return json.dumps({"success": True, "filled_fields": 1})

    monkeypatch.setattr("tools.browser_session_cdp.resolve_cdp_override", lambda url: _WS)
    monkeypatch.setattr("tools.browser_vault_tool.browser_vault_fill", fake_fill)
    monkeypatch.setattr("tools.browser_vault_tool._fenced_page_op", lambda task_id, fn: fn())
    return calls


def _run(monkeypatch, registry, capsys, handle="vault_abc", backend="local"):
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    args = SimpleNamespace(handle=handle, cdp="http://127.0.0.1:9", target_id="T1", backend=backend)
    with pytest.raises(SystemExit) as exc:
        vault_cli._cmd_fill(args)
    return exc.value.code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_fill_attaches_the_exact_target_without_bridge_and_stops(monkeypatch, capsys, wired):
    registry = _Registry()
    code, out = _run(monkeypatch, registry, capsys)
    assert code == 0 and out["success"] is True
    [(task_id, url, kwargs)] = registry.started
    assert url == _WS and kwargs == {"attach_target_id": "T1", "dialog_bridge": False}
    assert registry.stopped == [task_id]
    [(handle, fill_task, fill_kwargs)] = wired["fill"]
    assert handle == "vault_abc" and fill_task == task_id
    assert fill_kwargs["require_empty"] is True and fill_kwargs["backend"].name == "local"


def test_scope_conflict_refuses_before_any_fill(monkeypatch, capsys, wired):
    registry = _Registry(raises=bs.SupervisorScopeConflict("busy"))
    code, out = _run(monkeypatch, registry, capsys)
    assert code == 1 and out["error_type"] == "attach_failed"
    assert wired["fill"] == []


def test_handle_of_another_backend_is_refused_before_attach(monkeypatch, capsys, wired):
    registry = _Registry()
    code, out = _run(monkeypatch, registry, capsys, handle="op:abc", backend="local")
    assert code == 1 and out["error_type"] == "backend_mismatch"
    assert registry.started == [] and wired["fill"] == []


def test_unresolvable_endpoint_attaches_nothing(monkeypatch, capsys, wired):
    monkeypatch.setattr("tools.browser_session_cdp.resolve_cdp_override", lambda url: url)
    registry = _Registry()
    code, out = _run(monkeypatch, registry, capsys)
    assert code == 1 and out["error_type"] == "cdp_unresolved"
    assert registry.started == [] and wired["fill"] == []
