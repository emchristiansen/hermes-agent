"""``delegation.inherit_parent``: a DEFAULT child follows the parent's effective (running) model and reasoning, on the
parent's own route, with ``delegation.parent_child_models`` mapping a parent model to its child model; explicit child
selections and other profiles keep their own routes."""

import json
import os
import threading
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

INHERIT = {"inherit_parent": True, "parent_child_models": {"astra-model": "sol-model"}}


def _write_profile(home, *, host, model, key, effort, delegation=None):
    home.mkdir(parents=True, exist_ok=True)
    cfg = {"model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"},
           "agent": {"reasoning_effort": effort}}
    if delegation is not None:
        cfg["delegation"] = delegation
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\n", encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home and a ``worker`` profile; the parent runs AS ``worker`` (HERMES_HOME), whose config file names
    terra-model at high effort while the parent itself runs something else (a card override or session pick)."""
    launch = tmp_path / "os-home" / ".hermes"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key", effort="medium")
    worker = launch / "profiles" / "worker"
    _write_profile(worker, host="worker.invalid", model="terra-model", key="worker-key", effort="high")
    other = launch / "profiles" / "other"
    _write_profile(other, host="other.invalid", model="other-model", key="other-key", effort="low")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(worker))
    monkeypatch.setenv("PROFILE_KEY", "worker-key")
    return launch, worker, other


def _parent(model, effort, provider="custom", base_url="https://parent-route.invalid/v1", key="parent-route-key"):
    """The parent's EFFECTIVE runtime: model, effort, provider, endpoint and key as it is actually running."""
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider, parent.requested_provider = base_url, key, provider, provider
    parent.api_mode, parent.model, parent.platform = "chat_completions", model, "cli"
    parent.reasoning_config = None if effort is None else {"enabled": True, "effort": effort}
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._credential_pool = MagicMock(name="parent-pool")
    parent.fallback_providers = []
    return parent


def _resolve(config, parent, tasks=1):
    from tools.delegate_tool_routes import resolve_task_routes
    routes, error = resolve_task_routes([{"goal": f"task {i}"} for i in range(tasks)], config, parent)
    assert error is None
    return routes


def _build(route, parent):
    """The real child construction path (`_build_child_agent`), with only `AIAgent` itself captured."""
    from tools.delegate_tool import _build_child_agent, _route_overrides
    seen = {}

    def construct(**kwargs):
        seen.update(kwargs)
        return MagicMock(session_id="child-1")

    with patch("run_agent.AIAgent", side_effect=construct):
        _build_child_agent(task_index=0, goal="child", context=None, toolsets=None, model=route.creds["model"],
                           max_iterations=5, task_count=1, parent_agent=parent, **_route_overrides(route))
    return seen


def _built(seen):
    return {k: seen.get(k) for k in ("model", "provider", "base_url", "api_key", "reasoning_config")}


def test_astra_parent_overriding_its_profile_gets_sol_children_at_its_own_effort(homes):
    """A card override runs the parent on astra-model/low although its profile file says terra-model/high."""
    parent = _parent("astra-model", "low")
    (route,) = _resolve({"profile": "worker", **INHERIT}, parent)
    assert route.error is None and route.profile_home is None
    assert _built(_build(route, parent)) == {
        "model": "sol-model", "provider": "custom", "base_url": "https://parent-route.invalid/v1",
        "api_key": "parent-route-key", "reasoning_config": {"enabled": True, "effort": "low"}}


def test_non_astra_parent_gets_its_own_running_model_and_effort(homes):
    """A session pick (terra -> mid-model at medium) is what default children follow, not the profile file."""
    parent = _parent("mid-model", "medium")
    (route,) = _resolve({"profile": "worker", **INHERIT}, parent)
    built = _build(route, parent)
    assert _built(built) == {"model": "mid-model", "provider": "custom", "base_url": "https://parent-route.invalid/v1",
                             "api_key": "parent-route-key", "reasoning_config": {"enabled": True, "effort": "medium"}}


def test_parent_on_its_profile_default_matches_the_previous_profile_route(homes):
    parent = _parent("terra-model", "high", base_url="https://worker.invalid/v1", key="worker-key")
    (inherited,) = _resolve({"profile": "worker", **INHERIT}, parent)
    (legacy,) = _resolve({"profile": "worker"}, parent)
    a, b = _built(_build(inherited, parent)), _built(_build(legacy, parent))
    assert a == b == {"model": "terra-model", "provider": "custom", "base_url": "https://worker.invalid/v1",
                      "api_key": "worker-key", "reasoning_config": {"enabled": True, "effort": "high"}}


def test_without_opt_in_the_profile_route_is_unchanged(homes):
    """No ``inherit_parent`` (or not exactly ``true``): the profile file still decides, as before."""
    parent = _parent("astra-model", "low")
    for config in ({"profile": "worker"}, {"profile": "worker", "inherit_parent": "yes",
                                           "parent_child_models": {"astra-model": "sol-model"}}):
        (route,) = _resolve(config, parent)
        assert route.profile_home is not None
        built = _build(route, parent)
        assert (built["model"], built["base_url"], built["reasoning_config"]) == (
            "terra-model", "https://worker.invalid/v1", {"enabled": True, "effort": "high"})


def test_parent_without_effort_passes_none_not_the_profile_level(homes):
    parent = _parent("mid-model", None)
    (route,) = _resolve({"profile": "worker", **INHERIT}, parent)
    assert route.reasoning_effort.config is None
    assert _build(route, parent)["reasoning_config"] is None


def test_no_profile_parent_maps_astra_to_sol_on_its_own_route(homes, monkeypatch):
    """The default home delegates without a profile: same rule, parent's own provider/endpoint/key."""
    launch, _, _ = homes
    monkeypatch.setenv("HERMES_HOME", str(launch))
    parent = _parent("astra-model", "medium")
    (route,) = _resolve(dict(INHERIT), parent)
    assert _built(_build(route, parent)) == {
        "model": "sol-model", "provider": "custom", "base_url": "https://parent-route.invalid/v1",
        "api_key": "parent-route-key", "reasoning_config": {"enabled": True, "effort": "medium"}}


@pytest.mark.parametrize("field,value", [
    ("model", "pinned-model"), ("provider", "openrouter"), ("base_url", "https://pinned.invalid/v1"),
    ("reasoning_effort", "low"),
])
def test_explicit_child_selection_keeps_its_route(homes, field, value):
    """A pinned model, provider, endpoint or effort is an explicit choice; inheritance never overrides it."""
    from tools.delegate_tool_inherit import inherited_route
    parent = _parent("astra-model", "medium")
    route = {"profile": "worker", field: value}
    assert inherited_route(route, {**route, **INHERIT}, parent, "default") is None


def test_explicit_profile_model_pin_still_resolves_on_the_profile(homes):
    parent = _parent("astra-model", "low")
    (route,) = _resolve({"profile": "worker", "model": "pinned-model", **INHERIT}, parent)
    assert route.profile_home is not None
    built = _build(route, parent)
    assert (built["model"], built["base_url"], built["api_key"], built["reasoning_config"]) == (
        "pinned-model", "https://worker.invalid/v1", "worker-key", {"enabled": True, "effort": "high"})


def test_another_profile_keeps_its_own_route(homes):
    _, _, other = homes
    parent = _parent("astra-model", "low")
    (route,) = _resolve({"profile": "other", **INHERIT}, parent)
    assert route.profile_home == other
    built = _build(route, parent)
    assert (built["model"], built["base_url"], built["reasoning_config"]) == (
        "other-model", "https://other.invalid/v1", {"enabled": True, "effort": "low"})


def test_a_non_default_backend_keeps_its_route(homes):
    from tools.delegate_tool_inherit import inherited_route
    assert inherited_route({}, dict(INHERIT), _parent("astra-model", "low"), "options") is None


@pytest.mark.parametrize("model", [None, "", "  ", 7])
def test_parent_without_a_model_is_refused_alone(homes, model):
    parent = _parent("x", "low")
    parent.model = model
    (route,) = _resolve({"profile": "worker", **INHERIT}, parent)
    assert route.creds == {} and "no model for its children" in route.error


@pytest.mark.parametrize("mapping", [["astra-model"], {"astra-model": ""}, {"astra-model": 5}, "sol-model"])
def test_malformed_parent_child_models_is_refused(homes, mapping):
    parent = _parent("astra-model", "low")
    (route,) = _resolve({"profile": "worker", "inherit_parent": True, "parent_child_models": mapping}, parent)
    assert route.creds == {} and "parent_child_models" in route.error


def test_managed_worker_frozen_config_inherits_the_kanban_override(homes, monkeypatch):
    """A Kanban worker binds its frozen session config; its own-profile default child still inherits the card's
    override model/effort (a different profile stays refused there, unchanged)."""
    from agent import safe_worker_policy
    _, worker, _ = homes
    frozen = yaml.safe_load((worker / "config.yaml").read_text(encoding="utf-8"))
    frozen["delegation"] = {"profile": "worker", **INHERIT}
    monkeypatch.setattr(safe_worker_policy, "_policy", safe_worker_policy._SafeWorkerPolicy(
        os.getpid(), False, False, json.dumps(frozen)))
    parent = _parent("astra-model", "medium")
    (route,) = _resolve(frozen["delegation"], parent)
    assert route.error is None
    assert _build(route, parent)["model"] == "sol-model"
    (other,) = _resolve({"profile": "other", **INHERIT}, parent)
    assert "managed worker" in (other.error or "")


def test_batch_children_share_one_inherited_route(homes):
    parent = _parent("astra-model", "low")
    routes = _resolve({"profile": "worker", **INHERIT}, parent, tasks=3)
    assert {r.creds["model"] for r in routes} == {"sol-model"}
    assert all(r.reasoning_effort.config == {"enabled": True, "effort": "low"} for r in routes)
