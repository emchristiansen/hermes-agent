"""Default children that follow the parent's EFFECTIVE model and reasoning (``delegation.inherit_parent``).

A profile route resolves the child from its profile's config file, so a parent running on a different
model or effort than that file names (a Kanban card's model override, a session ``/model`` or
``/reasoning`` pick) launched children on the file's values instead. With ``delegation.inherit_parent:
true`` a default child takes the parent's running model and reasoning instead, on the parent's own
route (same profile, provider, endpoint and account); ``delegation.parent_child_models`` maps a parent
model to the child model it delegates to (``{gpt-6-astra: gpt-6.1-sol}``), at the parent's effort.

Only DEFAULT children inherit. A route carrying any explicit child selection (a model or provider pin,
an endpoint, an effort, a speed tier, another profile) or a non-default delegation backend keeps its
route unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

from agent.delegation_provider import DelegationRouteError

#: Route keys that make a route an explicit child selection rather than a default child.
EXPLICIT_ROUTE_KEYS = ("model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
                       "reasoning_effort", "service_tier")


def _parent_child_models(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("parent_child_models")
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping) or not all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in raw.items()):
        raise DelegationRouteError(
            "delegation.parent_child_models must map raw parent model IDs to raw child model IDs.")
    return {k.strip(): v.strip() for k, v in raw.items()}


def _is_own_profile(name: Any) -> bool:
    from hermes_constants import get_hermes_home, hermes_home_key
    from tools.delegate_tool_routes import _profile_home
    return hermes_home_key(_profile_home(name)) == hermes_home_key(get_hermes_home())


def inherited_route(route: Mapping[str, Any], config: Mapping[str, Any], parent_agent: Any,
                    backend: str) -> Optional[tuple[dict[str, Any], Optional[dict]]]:
    """``(route, reasoning config)`` for a default child of ``parent_agent``, or ``None`` when the route keeps
    its own resolution (inheritance off, a non-default backend, or an explicit child selection).

    The returned route has no ``profile`` key (the child runs in the parent's own profile scope, which is the
    same home) and names only a mapped child model, so the default resolver takes the parent's provider,
    endpoint and key. A parent without a usable model refuses rather than guessing one."""
    if backend != "default" or config.get("inherit_parent") is not True:
        return None
    if any(route.get(k) not in (None, "", {}) for k in EXPLICIT_ROUTE_KEYS):
        return None
    profile = route.get("profile")
    if profile not in (None, "") and not _is_own_profile(profile):
        return None
    parent_model = getattr(parent_agent, "model", None)
    if not isinstance(parent_model, str) or not parent_model.strip():
        raise DelegationRouteError("delegation.inherit_parent: the parent has no model for its children to inherit.")
    parent_model = parent_model.strip()
    child_model = _parent_child_models(config).get(parent_model, parent_model)
    child = {k: v for k, v in route.items() if k not in ("profile", "model")}
    if child_model != parent_model:
        # A mapped child model is a pin (it never borrows the parent's fallback chain); the parent's own model
        # stays a pure inherit, exactly as a child of a parent without a delegation profile.
        child["model"] = child_model
    return child, copy.deepcopy(getattr(parent_agent, "reasoning_config", None))
