"""A worker's provider-wall evidence, and the rate-limit guard that lets one card probe an exhausted pool.

A worker that meets a provider wall exits ``KANBAN_RATE_LIMIT_EXIT_CODE`` and its card is requeued
uncounted; ``check_respawn_guard`` spaces that card's retries by the rate-limit cooldown. The exit
code alone says nothing about which wall, so when a profile's whole Codex pool was exhausted every
walled card of that profile re-spawned once per cooldown, each through a claim, an owner admission,
a session and an interpreter that died at credential resolution before any model call (t_d4ba87d4:
3,647 walled runs on 55 cards between 01:40 and 09:22).

Evidence. The bound interpreter records what it met as ``provider_wall`` in its ``worker_result``
(:func:`wall_from_exception` before any turn ran, :func:`wall_from_result` for a turn that ended on a
provider failure); the crash sweep copies it onto the rate-limited run (:func:`attach_wall`).

Coalescing. Only a POOL verdict speaks for another card: the credential resolver found every pool
row of the provider cooling and no other credential (``AuthError.exhausted == 'credential_pool'``).
Two walls share that verdict only when their workers resolved the same profile home, provider, model
and endpoint with no pinned credential or fallback chain, and the route inputs (the profile's and
the managed scope's config files, :func:`route_inputs`) and the cards' overrides are unchanged,
and each card's next attempt runs under the profile its wall did (its current assignee, however it
changed: reassignment, review request or return); anything else is unrepresentable here and keeps
the per-card cooldown. A walled card past its own cooldown is then held only while a sibling already holds a
fresher verdict for that pool: a probe started after this card's wall that is still within one
cooldown, or a wall that such a probe met within the last cooldown. Whichever card the dispatcher
reaches first probes; its credential resolution rereads the pool and asks upstream about an early
reset, so newly eligible capacity (an added or re-enabled credential, a reopened window) ends the
walls and every held card runs within one cooldown. A reset time is never used to hold: the earliest
reset one worker saw is not a horizon for every route. Nothing here counts a failure or blocks a card.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ``TRANSIENT_PROVIDER_REASONS`` that are a usage limit; overloaded/server_error/timeout are not.
QUOTA_TURN_REASONS = frozenset({"rate_limit", "upstream_rate_limit", "billing"})
POOL_SCOPE = "credential_pool"
ROUTE_PROBED = "rate_limit_route_probed"
_EVIDENCE_CHARS = 500
# Route inputs whose change makes an earlier attempt's route unknown for the next one: the files the
# effective worker config is read from (``hermes_cli.config_effective``), the profile's and the
# managed scope's, whose directory is recorded too (``managed_dir``).
_ROUTE_INPUT_FILES = ("config.yaml", ".env")
_MANAGED_PREFIX = "managed/"
_ROUTE_INPUT_KEYS = frozenset({*_ROUTE_INPUT_FILES, "managed_dir", *(_MANAGED_PREFIX + n for n in _ROUTE_INPUT_FILES)})
# ``model:`` keys that pin an endpoint credential (``gateway.session_route_identity.ROUTE_BOUND_KEYS``).
_PINNED_KEYS = ("api_key", "api", "api_key_from", "key_env", "api_key_env")


@dataclass(frozen=True)
class WallRoute:
    """The route the walled worker resolved, as far as a pool verdict needs it."""

    home: str
    provider: str
    model: Optional[str]
    base_url: str
    pinned: bool
    fallback: bool
    inputs: dict  # ``route_inputs(home)`` stamped with the attempt's policy, unchanged when it walled

    @classmethod
    def parse(cls, value: Any) -> Optional[WallRoute]:
        if not isinstance(value, dict) or set(value) != _ROUTE_FIELDS:
            return None
        inputs = value["inputs"]
        if (not _text(value["home"]) or not _text(value["provider"]) or not _optional_text(value["model"])
                or not isinstance(value["base_url"], str) or type(value["pinned"]) is not bool
                or type(value["fallback"]) is not bool or not isinstance(inputs, dict)
                or set(inputs) != _ROUTE_INPUT_KEYS or not _optional_text(inputs["managed_dir"])
                or not all(_stat_shape(v) for k, v in inputs.items() if k != "managed_dir")):
            return None
        return cls(**value)


@dataclass(frozen=True)
class ProviderWall:
    reason: str  # "quota" (credential resolution) or the turn's ``failure_reason``
    phase: str  # "credentials": no turn ran; "turn": the turn ended on it
    provider: Optional[str]
    model: Optional[str]
    retry_at: Optional[float]  # the provider's own hint; evidence only, never a hold
    evidence: str
    scope: Optional[str] = None  # POOL_SCOPE: every pool row of ``provider`` was cooling
    route: Optional[WallRoute] = None

    def payload(self) -> dict:
        return asdict(self)

    @classmethod
    def parse(cls, value: Any) -> Optional[ProviderWall]:
        """The recorded wall, or None when absent or not exactly this shape (legacy runs, other
        writers, corruption): an unreadable wall is unknown, which keeps the per-card cooldown.
        Only the credential resolver's pool verdict carries a scope: a scope on any other wall
        contradicts its own reason or phase, so the whole record is unknown."""
        if not isinstance(value, dict) or set(value) != _WALL_FIELDS:
            return None
        retry_at, route = value["retry_at"], value["route"]
        if (not _text(value["reason"]) or value["phase"] not in ("credentials", "turn")
                or not _optional_text(value["provider"]) or not _optional_text(value["model"])
                or not isinstance(value["evidence"], str) or not _optional_text(value["scope"])
                or not (retry_at is None or _finite_number(retry_at))):
            return None
        parsed_route = None if route is None else WallRoute.parse(route)
        if route is not None and parsed_route is None:
            return None
        wall = cls(**{**value, "route": parsed_route})
        return wall if wall.scope is None or wall._pool_verdict() else None

    def is_quota(self) -> bool:
        """A usage limit, as opposed to an overloaded, failing or timed-out provider."""
        return self.reason == "quota" or self.reason in QUOTA_TURN_REASONS

    def _pool_verdict(self) -> bool:
        """The credential resolver found every pool row cooling: the only wall that has a scope."""
        return self.scope == POOL_SCOPE and self.reason == "quota" and self.phase == "credentials"

    def pool_route(self) -> Optional[WallRoute]:
        """The route whose pool this wall exhausted, when the verdict is representable."""
        route = self.route
        if not self._pool_verdict() or route is None or route.pinned or route.fallback:
            return None
        return route


_ROUTE_FIELDS = frozenset(f.name for f in fields(WallRoute))
_WALL_FIELDS = frozenset(f.name for f in fields(ProviderWall))


def _finite_number(value: Any) -> bool:
    """A JSON number usable as a float: not a bool, NaN, an infinity or an int beyond float range
    (JSON decodes ``1e999`` to inf and arbitrarily long integers exactly)."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _optional_text(value: Any) -> bool:
    return value is None or _text(value)


def _named(value: Any) -> Optional[str]:
    return value if _text(value) else None


def _stat_shape(value: Any) -> bool:
    """A recorded ``_stamp``: None, or the four integers of ``utils.file_signature``. An older record's
    three-part stamp (no ctime) does not parse, so its wall is unknown rather than read as certain."""
    return value is None or (isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value))


def route_inputs(home: str) -> Optional[dict]:
    """What the profile's effective worker config is read from, as its ``_stamp`` (None: absent) for its ``config.yaml`` and ``.env`` and for the managed scope's, under ``managed_dir``
    as the config loader resolves it (``managed_scope.get_managed_dir``; None: no managed scope).
    Never a credential value. None when the route may depend on an input no file stamp sees, so
    the route is unknown and keeps the per-card cooldown: a ``${...}`` reference in either
    config.yaml can resolve from the process environment, and a config.yaml that does not parse
    is not what the loader serves (it recovers the profile's last-known-good settings, from this
    process or ``backups/config``, and ignores a broken managed file)."""
    from hermes_cli.config import _ENV_REF_RE
    from hermes_cli.managed_scope import get_managed_dir
    from utils import fast_safe_load
    managed = get_managed_dir()
    inputs: dict = {"managed_dir": str(managed) if managed is not None else None}
    for name in _ROUTE_INPUT_FILES:
        for key, root in ((name, Path(home)), (_MANAGED_PREFIX + name, managed)):
            inputs[key] = _stamp(root / name) if root is not None else None
            if name != "config.yaml" or inputs[key] is None:
                continue
            try:
                text = (root / name).read_text(encoding="utf-8-sig")
                if _ENV_REF_RE.search(text):
                    return None
                fast_safe_load(text)  # the loaders' own parser
            except Exception:  # health: allow BLE001 -- unreadable or unparseable: the loader serves something else
                return None
    return inputs


def _stamp(path: Path) -> Optional[list]:
    """The config loader's own change key for ``path`` (``utils.file_signature``: mtime, size, inode
    and ctime, so an in-place edit that restores the mtime still reads as a change); None: absent."""
    from utils import file_signature
    try:
        st = os.stat(path)
    except (OSError, ValueError):  # ValueError: a recorded home with an embedded NUL
        return None
    return list(file_signature(st))


def _route_of(frame: dict, provider: Optional[str]) -> Optional[WallRoute]:
    """The route the attempt's worker config resolves (``gateway.session_route_identity.resolved_route``),
    or None when it is not representable here: it does not resolve to the walled provider, or the
    profile's route inputs were not stamped unchanged across the policy's capture and are not still
    unchanged now (the worker may have followed edited config: its actual route is then unknown)."""
    from gateway.session_route_identity import resolved_route
    policy, home = frame.get("policy") or {}, frame.get("home")
    if not home or not provider or not isinstance(policy, dict):
        return None
    try:
        config = json.loads(policy.get("config_json") or "{}")
        request = json.loads(policy.get("request_json") or "{}")
        kanban = json.loads(policy.get("kanban_json") or "{}")
    except (TypeError, ValueError):
        return None
    if not all(isinstance(v, dict) for v in (config, request, kanban)):
        return None
    captured = kanban.get("route_inputs")
    if captured is None or captured != route_inputs(str(home)):
        return None
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    try:
        identity, base_url = resolved_route(model_cfg, config)
    except Exception:  # health: allow BLE001 -- a route this cannot resolve is unknown; never mask the wall
        logger.debug("provider wall: route not resolvable", exc_info=True)
        return None
    if identity != provider:
        return None
    construct = request.get("construct_v1") if isinstance(request.get("construct_v1"), dict) else {}
    # The bootstrap clears ``credential_ref`` and hands a launch key over as the frame's ``api_key``.
    pinned = (bool(frame.get("api_key")) or bool(policy.get("credential_ref"))
              or any(model_cfg.get(key) for key in _PINNED_KEYS))
    fallback = bool(config.get("fallback_providers") or config.get("fallback_model") or construct.get("fallback_model"))
    return WallRoute(home=str(home), provider=provider, model=_named(policy.get("model")),
                     base_url=base_url, pinned=pinned, fallback=fallback, inputs=captured)


def _retry_at(value: Any) -> Optional[float]:
    """A provider's reset hint as a finite float, else None (the hint is evidence, never a hold)."""
    return float(value) if _finite_number(value) else None


def wall_from_exception(exc: BaseException, frame: dict) -> Optional[ProviderWall]:
    """The quota wall an exception raised before any turn result names, else None."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, AuthError):
            break
        exc = exc.__cause__ or exc.__context__
    if not isinstance(exc, AuthError) or not is_rate_limited_auth_error(exc):
        return None
    policy = frame.get("policy") or {}
    retry_after = exc.retry_after if _finite_number(exc.retry_after) else None
    provider = _named(exc.provider)
    return ProviderWall(
        reason="quota", phase="credentials", provider=provider, model=_named(policy.get("model")),
        retry_at=_retry_at(time.time() + retry_after) if retry_after is not None else None,
        evidence=f"{type(exc).__name__}: {exc}"[-_EVIDENCE_CHARS:],
        scope=getattr(exc, "exhausted", None),
        route=_route_of(frame, provider))


def wall_from_result(result: Any, agent: Any) -> Optional[ProviderWall]:
    """The provider failure a finished turn's result records, else None (the turn did not end on one)."""
    if not isinstance(result, dict) or not isinstance(result.get("failure_reason"), str):
        return None
    resets_at = result.get("failure_resets_at")
    return ProviderWall(
        reason=result["failure_reason"] or "unknown", phase="turn",
        provider=_named(getattr(agent, "provider", None)), model=_named(getattr(agent, "model", None)),
        retry_at=_retry_at(resets_at),
        evidence=str(result.get("final_response") or result.get("error") or "")[-_EVIDENCE_CHARS:])


def attach_wall(conn, row, payload: dict) -> None:
    """Copy the attempt's recorded wall onto its rate-limited booking (``payload`` becomes the run's
    metadata). Only the result that booked this run speaks: the current run's accounting result,
    under this claim."""
    from hermes_cli.kanban_owner_recovery import attempt_result
    task = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (row["id"],)).fetchone()
    if task is None or task["current_run_id"] is None:
        return
    result = attempt_result(conn, row["id"], task["current_run_id"])
    wall = ProviderWall.parse(result.get("provider_wall"))
    if wall is not None and result.get("claim_lock") == row["claim_lock"]:
        payload["provider_wall"] = wall.payload()


def rate_limit_guard(conn, task_id: str, latest_run, now: int, cooldown: int) -> Optional[str]:
    """Guard reason for a card whose LATEST run is ``rate_limited``, else None.

    Its own cooldown first (``rate_limit_cooldown``, as before; 0 respawns at once); after it, a
    pool wall waits while a sibling holds a fresher verdict for the same pool (``ROUTE_PROBED``)."""
    from hermes_cli import kanban_db as kb
    if cooldown <= 0:
        return None
    ended = kb._to_epoch(latest_run["ended_at"])
    if ended is not None and now - ended < cooldown:
        return "rate_limit_cooldown"
    route = _pool_route(latest_run["metadata"])
    if (ended is None or route is None or not _runs_as(conn, task_id, latest_run["profile"])
            or not _route_unchanged(conn, task_id, route, latest_run["started_at"])):
        return None
    return ROUTE_PROBED if _fresher_pool_verdict(conn, task_id, latest_run["profile"], route, ended, now, cooldown) else None


def _pool_route(metadata) -> Optional[WallRoute]:
    from hermes_cli import kanban_db as kb
    wall = ProviderWall.parse(kb._json_dict(metadata).get("provider_wall"))
    return wall.pool_route() if wall is not None else None


def _known_profile(profile) -> bool:
    """A profile an attempt can actually run under. Unassigned (``None``) and legacy blank rows name
    no profile, so two of them never match each other; ``default`` is a real profile and does."""
    return isinstance(profile, str) and profile.strip() != ""


def _runs_as(conn, task_id: str, profile) -> bool:
    """Whether the card's next attempt runs under ``profile``: a claim opens its run under the
    card's assignee as it stands then, so a wall another profile met (a reviewer's, before the card
    went back to its implementer, or any other reassignment) says nothing about the next attempt.
    Read from the card itself, whichever transition changed it."""
    row = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()
    return row is not None and _known_profile(profile) and row["assignee"] == profile


def _route_unchanged(conn, task_id: str, route: WallRoute, since) -> bool:
    """The card's next attempt resolves the route its wall (started at ``since``) recorded: no
    override or assignee change on the card since, and the profile's route inputs untouched."""
    from hermes_cli import kanban_db as kb
    if kb._to_epoch(since) is None:
        return False
    # SQLite sorts TEXT above every INTEGER: a text stamp always matches, which reads as a change.
    changed = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id=? AND kind IN ('model_override_set', 'assigned') "
        "AND created_at >= ? LIMIT 1", (task_id, kb._to_epoch(since))).fetchone()
    return changed is None and route_inputs(route.home) == route.inputs


def _fresher_pool_verdict(conn, task_id: str, profile, route: WallRoute, ended: int, now: int,
                          cooldown: int) -> bool:
    """Whether a sibling probe of the same pool, started after this card's wall, holds the pool's
    current verdict. A probe is an attempt whose previous run walled on this route; only probes speak,
    so ordinary sibling work neither holds nor releases this card. A probe that did not wall (it ended
    any other way, its owner's death included, or is still running a cooldown after its start: it
    resolved a credential, or no longer answers for the pool) is a passage, so a held card is never
    held by a probe that will not report. A
    wall within the last cooldown holds unless an even newer passage started; a probe still in flight
    within one cooldown of its start holds only while no passage exists, so once one probe gets
    through every held card runs together instead of one per cooldown."""
    from hermes_cli import kanban_db as kb
    # SQLite sorts TEXT above every INTEGER, so a text stamp is always fetched and decided parsed.
    rows = conn.execute(
        "SELECT id, task_id, outcome, started_at, ended_at, metadata, profile FROM task_runs "
        "WHERE profile IS ? AND task_id != ? AND started_at > ? AND (ended_at IS NULL OR ended_at >= ?)",
        (profile, task_id, ended, now - cooldown)).fetchall()
    walls, pending, passed = [], [], []
    for run in rows:
        started, run_ended = kb._to_epoch(run["started_at"]), kb._to_epoch(run["ended_at"])
        if (started is None or started <= ended
                or _probe_route(conn, run["task_id"], run["id"], run["profile"]) != route):
            continue
        if run["ended_at"] is None:
            (pending if now - started < cooldown else passed).append(started)
        elif run["outcome"] != "rate_limited":
            passed.append(started)
        elif run_ended is not None and now - run_ended < cooldown and _pool_route(run["metadata"]) == route:
            walls.append(started)
    if passed:
        return any(started > max(passed) for started in walls)
    return bool(walls or pending)


def _probe_route(conn, task_id: str, run_id: int, profile) -> Optional[WallRoute]:
    """The pool a sibling's attempt (run ``run_id`` under ``profile``) probes: the route its previous
    attempt walled on, when that wall was met under the same profile and nothing has changed the
    sibling's route since (else unknown: it holds no one)."""
    prior = conn.execute(
        "SELECT outcome, started_at, metadata, profile FROM task_runs WHERE task_id=? AND id<? "
        "AND ended_at IS NOT NULL ORDER BY id DESC LIMIT 1", (task_id, run_id)).fetchone()
    if (prior is None or prior["outcome"] != "rate_limited" or not _known_profile(profile)
            or prior["profile"] != profile):
        return None
    route = _pool_route(prior["metadata"])
    return route if route is not None and _route_unchanged(conn, task_id, route, prior["started_at"]) else None
