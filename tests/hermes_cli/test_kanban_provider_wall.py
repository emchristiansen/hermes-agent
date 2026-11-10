"""An exhausted credential pool is probed by one card per cooldown, never re-spawned by every card.

Regression for t_d4ba87d4: with every openai-codex pool row of one profile cooling, 55 cards each
re-spawned once per rate-limit cooldown for 7.7 hours (3,647 walled runs), every attempt a claim, an
owner admission, a session and an interpreter dying at credential resolution before any model call,
and the walled runs recorded nothing about the wall. Through the real worker-result producer
(``session_kanban.record_start_failure`` / ``run_worker_turns`` with the real Codex pool-exhaustion
``AuthError``), the real dead-worker sweep and the real dispatcher tick (``dispatch_once`` with a
spawn stub, so claims are real); real board SQLite, only the clock and the profile check are faked.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_provider_wall as pw
from hermes_cli.auth_codex import _codex_quota_exhausted_error
from hermes_cli.kanban_db_provider_wall import ROUTE_PROBED, route_inputs

COOLDOWN = 300
PROBED = ROUTE_PROBED
COOLING = "rate_limit_cooldown"
CODEX_CONFIG = {"model": {"provider": "openai-codex", "default": "gpt-test"}}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    offset = [0]
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + offset[0])
    return offset


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_assignees_spawnable, clock):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", str(COOLDOWN))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """The walled workers' profile home: its route inputs are what a wall's route is stamped with."""
    home = tmp_path / "profiles" / "w"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model:\n  provider: openai-codex\n  default: gpt-test\n", encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")
    return home


@pytest.fixture
def managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A managed scope (``HERMES_MANAGED_DIR``, the config loader's own seam) whose config.yaml
    pins no route yet: the effective worker config is the profile's under this overlay."""
    root = tmp_path / "managed"
    root.mkdir()
    (root / "config.yaml").write_text("agent:\n  max_turns: 50\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(root))
    return root


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _db(conn) -> str:
    return conn.execute("PRAGMA database_list").fetchone()[2]


class Board:
    """Real dispatcher ticks; ``spawn`` decides each spawned worker's pid (dead, or this live process)."""

    def __init__(self, conn, clock, profile: Path):
        self.conn, self.clock, self.profile = conn, clock, profile
        self.spawned: list[list[str]] = []
        self.alive: set[str] = set()  # cards whose next spawn keeps running (got past credentials)

    def spawn(self, task, workspace, board=None):
        self.spawned[-1].append(task.id)
        return os.getpid() if task.id in self.alive else _dead_pid()

    def tick(self, seconds: int = 60):
        """One real tick a little later: sweep the dead, then select and claim the ready."""
        self.clock[0] += seconds
        self.spawned.append([])
        res = kbd.dispatch_once(self.conn, spawn_fn=self.spawn)
        return res, dict(res.respawn_guarded)

    def run(self, tid):
        return kb.get_task(self.conn, tid)

    def bind(self, tid):
        """The attempt's interpreter binds (a dead pid: the crash sweep may reap it)."""
        task = self.run(tid)
        bound = _dead_pid()
        with kb.write_txn(self.conn):
            kb._append_event(self.conn, tid, "worker_bound", {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"},
                             run_id=task.current_run_id)
        return task, bound

    def frame(self, tid, *, policy_extra=None, config=None, capture=True, frame_extra=None):
        task = self.run(tid)
        context = {"db": _db(self.conn), "task_id": tid, "run_id": task.current_run_id, "claim_lock": task.claim_lock,
                   "route_inputs": route_inputs(str(self.profile)) if capture else None}
        policy = {"model": "gpt-test", "config_json": json.dumps(config or CODEX_CONFIG),
                  "request_json": "{}", "kanban_json": json.dumps(context), **(policy_extra or {})}
        return {"home": str(self.profile), "policy": policy, **(frame_extra or {})}

    def pool_wall(self, tid, monkeypatch, *, exc=None, frame=None):
        """The attempt dies at credential resolution: the real producer books exit 75 and the wall."""
        _task, bound = self.bind(tid)
        frame = frame or self.frame(tid)
        exc = exc or _codex_quota_exhausted_error(109934, exhausted="credential_pool")
        with monkeypatch.context() as m:
            m.setattr(os, "getpid", lambda: bound)
            session_kanban.record_start_failure(frame, exc)

    def turn_wall(self, tid, monkeypatch, result):
        """The attempt's turn ran and ended on a provider failure."""
        _task, bound = self.bind(tid)
        frame = self.frame(tid)
        agent = SimpleNamespace(provider="openai-codex", model="gpt-test")
        with monkeypatch.context() as m:
            m.setattr(os, "getpid", lambda: bound)
            m.setattr(session_kanban, "_run_task_turns", lambda *_a: result)
            session_kanban.run_worker_turns(agent, frame, [])

    def latest(self, tid):
        return self.conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()


@pytest.fixture
def board(conn, clock, profile):
    return Board(conn, clock, profile)


def _next_spawn(board):
    """Tick until a tick spawns something; that tick's spawns."""
    for _ in range(COOLDOWN // 60 + 3):
        board.tick()
        if board.spawned[-1]:
            return board.spawned[-1]
    raise AssertionError(f"nothing spawned within a cooldown: {board.spawned}")


def _wall_round(board, monkeypatch, cards):
    """Every card spawned on the last tick meets the exhausted pool."""
    for tid in board.spawned[-1]:
        assert tid in cards
        board.pool_wall(tid, monkeypatch)


def test_pool_wall_evidence_names_the_wall_its_attempt_met(board, monkeypatch):
    tid = kb.create_task(board.conn, title="owner", assignee="w")
    board.tick()
    walled_at = time.time()
    board.pool_wall(tid, monkeypatch)
    res, guarded = board.tick()
    assert tid in res.rate_limited and guarded[tid] == COOLING
    run = board.latest(tid)
    wall = json.loads(run["metadata"])["provider_wall"]
    assert run["outcome"] == "rate_limited"
    assert (wall["reason"], wall["phase"], wall["provider"], wall["scope"]) == (
        "quota", "credentials", "openai-codex", "credential_pool")
    assert "retry after 109934s" in wall["evidence"] and wall["route"]["home"] == str(board.profile)
    assert wall["route"]["inputs"] == route_inputs(str(board.profile)) and wall["route"]["model"] == "gpt-test"
    # The provider's own hint survives as evidence (it holds nothing: see the probe tests below).
    assert abs(wall["retry_at"] - (walled_at + 109934)) < 30


@pytest.mark.parametrize(("result", "reason", "quota"), [
    ({"failed": True, "failure_reason": "rate_limit", "failure_resets_at": 1_900_000_000.0,
      "final_response": "HTTP 429: The usage limit has been reached. Limit resets at 11:01"}, "rate_limit", True),
    ({"failed": True, "failure_reason": "overloaded", "final_response": "HTTP 529: overloaded"}, "overloaded", False),
])
def test_turn_wall_evidence_tells_quota_from_other_provider_failures(board, monkeypatch, result, reason, quota):
    from hermes_cli.kanban_db_provider_wall import ProviderWall
    tid = kb.create_task(board.conn, title="owner", assignee="w")
    board.tick()
    board.turn_wall(tid, monkeypatch, result)
    res, _ = board.tick()
    assert tid in res.rate_limited
    wall = ProviderWall.parse(json.loads(board.latest(tid)["metadata"])["provider_wall"])
    assert (wall.reason, wall.phase, wall.provider, wall.is_quota()) == (reason, "turn", "openai-codex", quota)
    assert wall.evidence == result["final_response"] and wall.pool_route() is None
    assert wall.retry_at == result.get("failure_resets_at")


def test_exhausted_pool_is_probed_by_one_card_per_cooldown(board, monkeypatch):
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(4)]
    board.tick()
    assert sorted(board.spawned[-1]) == sorted(cards)
    _wall_round(board, monkeypatch, cards)
    probes = []
    for _ in range(20):  # 20 minutes: four cooldowns
        _res, guarded = board.tick()
        _wall_round(board, monkeypatch, cards)
        probes.append(len(board.spawned[-1]))
        assert all(guarded.get(t) in (COOLING, PROBED) for t in cards if t not in board.spawned[-1])
    # Before: every card re-spawned once per cooldown (4 per round). Now ONE probe per round, a round
    # being the cooldown after the previous probe's wall was reaped (5 one-minute ticks, then the reap).
    rounds = [n for n, spawned in enumerate(probes) if spawned]
    assert max(probes) == 1 and rounds == [5, 11, 17], probes
    for tid in cards:
        assert board.run(tid).status == "ready" and board.run(tid).consecutive_failures == 0
        assert all(e.kind not in ("gave_up", "crashed") for e in kb.list_events(board.conn, tid))


def test_a_probe_past_the_wall_releases_every_held_card(board, monkeypatch):
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    _wall_round(board, monkeypatch, cards)
    board.alive.update(cards)  # capacity returns: the next attempt resolves a credential and works
    probe = _next_spawn(board)
    assert len(probe) == 1
    held = [t for t in cards if t not in probe]
    _res, guarded = board.tick()
    assert all(guarded.get(t) == PROBED for t in held)  # its verdict is pending, for one cooldown
    # The probe finishes its work: the pool's verdict is now "usable", the held cards run at once.
    kb.complete_task(board.conn, probe[0], summary="worked", expected_run_id=board.run(probe[0]).current_run_id)
    _res, guarded = board.tick()
    assert sorted(board.spawned[-1]) == sorted(held) and not any(t in guarded for t in held)


def test_a_long_running_probe_releases_the_held_cards_after_one_cooldown(board, monkeypatch):
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    _wall_round(board, monkeypatch, cards)
    board.alive.update(cards)
    probe = _next_spawn(board)[0]
    started = board.clock[0]
    released = []
    while not released:
        board.tick()
        released = board.spawned[-1]
    assert sorted(released) == sorted(t for t in cards if t != probe)
    assert board.clock[0] - started <= COOLDOWN + 60


def _parallel_walls(board, monkeypatch, frames):
    """Two cards wall together, each through its own frame; returns the two cards."""
    a, b = (kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in "ab")
    board.tick()
    for tid, frame in zip((a, b), frames(board, a, b)):
        board.pool_wall(tid, monkeypatch, frame=frame)
    return a, b


def _cooled(board, a, b, monkeypatch):
    """After both cooldowns: the first card probes; does the second wait for it?"""
    first = _next_spawn(board)
    _wall_round(board, monkeypatch, (a, b))
    return first


# ``Board.frame`` arguments for a route a pool verdict cannot be shared over.
UNREPRESENTABLE_ROUTES = {
    "pinned": {"policy_extra": {"credential_ref": "cred-1"}},
    "launch_key": {"frame_extra": {"api_key": "sk-launch"}},
    "config_fallback": {"config": {**CODEX_CONFIG, "fallback_providers": [{"provider": "openrouter"}]}},
    "turn_fallback": {"policy_extra": {"request_json": json.dumps(
        {"construct_v1": {"fallback_model": [{"provider": "openrouter", "model": "x"}]}})}},
    "other_model": {"policy_extra": {"model": "gpt-other"}},
    "other_endpoint": {"config": {"model": {**CODEX_CONFIG["model"], "base_url": "https://other.example/v1"}}},
    "uncaptured": {"capture": False},
}


@pytest.mark.parametrize("variant", list(UNREPRESENTABLE_ROUTES))
def test_routes_this_guard_cannot_represent_keep_each_cards_own_cooldown(board, monkeypatch, variant):
    """Each card walls on a route a pool verdict cannot be shared over: both cards still retry after
    their own cooldown, exactly as before (unknown stays runnable)."""
    def frames(board, a, b):
        return board.frame(a, **UNREPRESENTABLE_ROUTES[variant]), board.frame(b)
    a, b = _parallel_walls(board, monkeypatch, frames)
    first = _cooled(board, a, b, monkeypatch)
    assert sorted(first) == sorted((a, b)), (variant, first)


def test_a_single_credential_quota_error_is_not_a_pool_verdict(board, monkeypatch):
    """A token-endpoint 429 for one credential says nothing about the rest of the pool."""
    a, b = (kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in "ab")
    board.tick()
    board.pool_wall(a, monkeypatch, exc=_codex_quota_exhausted_error(60))  # token refresh 429: no scope
    board.pool_wall(b, monkeypatch)
    assert sorted(_cooled(board, a, b, monkeypatch)) == sorted((a, b))


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# A change after the walls, and how many held cards it releases at once (the rest stay held).
ROUTE_CHANGES = {
    "config_edit": (lambda board, managed, held: _write(board.profile / "config.yaml",
                                                         "model:\n  provider: openrouter\n  default: x\n"), 2),
    "override": (lambda board, managed, held: kb.set_model_override(board.conn, held[0], "gpt-other", "openai-codex"), 1),
    "managed_fallback": (lambda board, managed, held: _write(managed / "config.yaml",
                                                              "fallback_providers:\n  - provider: openrouter\n    model: x\n"), 2),
    "managed_model": (lambda board, managed, held: _write(managed / "config.yaml",
                                                           "model:\n  provider: openrouter\n  default: x\n"), 2),
    "managed_env": (lambda board, managed, held: _write(managed / ".env", "OPENAI_BASE_URL=https://other.example/v1\n"), 2),
    "managed_removed": (lambda board, managed, held: (managed / "config.yaml").unlink(), 2),
    "unchanged": (lambda board, managed, held: None, 0),
}


@pytest.mark.parametrize("change", list(ROUTE_CHANGES))
def test_a_route_change_after_the_wall_lets_the_card_run(board, managed, monkeypatch, change):
    """The held card's next attempt would not resolve the walled route: it runs as soon as its own
    cooldown allows. The effective worker config is the profile's config.yaml and .env under the
    managed scope's, so a managed-only edit releases every held card just as a profile edit does.
    Control: with nothing changed, the fresh verdict of the probe still holds both."""
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    _wall_round(board, monkeypatch, cards)
    probe = _next_spawn(board)
    assert len(probe) == 1
    _wall_round(board, monkeypatch, cards)  # the probe walls again: a fresh verdict for the pool
    held = [t for t in cards if t not in probe]
    apply, released = ROUTE_CHANGES[change]
    apply(board, managed, held)
    _res, guarded = board.tick()  # both held cards are past their own cooldown
    assert sorted(board.spawned[-1]) == sorted(held[:released]), (change, guarded)
    assert all(guarded.get(t) == PROBED for t in held[released:]), (change, guarded)


# A supported transition on the walled review card, and whether it leaves the card's next attempt
# under the profile that met the wall (its reviewer ``w``): then the fresh probe still holds it.
REVIEW_TRANSITIONS = {
    "unchanged": (lambda conn, tid: None, False),
    "reopened_to_implementer": (lambda conn, tid: kb.reopen_review_task(conn, tid), True),
    "reassigned_reviewer": (lambda conn, tid: kb.assign_task(conn, tid, "v"), True),
    # Away and back to the same reviewer: the same profile, but the card's ``assigned`` events since
    # its wall conservatively make its route unknown.
    "away_and_back": (lambda conn, tid: (kb.assign_task(conn, tid, "v"), kb.assign_task(conn, tid, "w")), True),
}


@pytest.mark.parametrize("transition", list(REVIEW_TRANSITIONS))
def test_a_card_that_leaves_the_walled_profile_is_not_held_by_its_pool(board, managed, monkeypatch, transition):
    """Through the real review lifecycle and dispatch: an implementer requests review from ``w``,
    the review attempts wall on ``w``'s pool, one probes again. A card whose next attempt no longer
    runs under ``w`` (returned to its implementer, reassigned) runs as soon as its own cooldown is
    over, whichever transition moved it; the review card left with ``w`` stays held behind the probe."""
    cards = [kb.create_task(board.conn, title=f"review {n}", assignee="implementer") for n in range(3)]
    for tid in cards:
        assert kb.request_review(board.conn, tid, summary="durable candidate", reviewer="w")
    board.tick()
    assert sorted(board.spawned[-1]) == sorted(cards)  # the reviewer claims all three
    _wall_round(board, monkeypatch, cards)
    probe = _next_spawn(board)
    assert len(probe) == 1
    _wall_round(board, monkeypatch, cards)  # a fresh verdict for w's pool
    target = next(t for t in cards if t not in probe)
    assert board.latest(target)["profile"] == "w" and board.run(target).status == "review"
    apply, released = REVIEW_TRANSITIONS[transition]
    apply(board.conn, target)
    _res, guarded = board.tick()  # the target is past its own cooldown
    assert (target in board.spawned[-1]) is released, (transition, board.run(target).assignee, guarded)
    assert guarded.get(target) == (None if released else PROBED), (transition, guarded)


def test_a_config_that_reads_the_environment_has_no_shared_route(board, monkeypatch):
    """A ``${...}`` reference in config.yaml may resolve from the process environment, which no file
    stamp sees: the route is unknown, so both walled cards retry on their own cooldown."""
    _write(board.profile / "config.yaml",
           "model:\n  provider: openai-codex\n  default: gpt-test\n  base_url: ${CODEX_BASE_URL}\n")
    a, b = _parallel_walls(board, monkeypatch, lambda board, a, b: (board.frame(a), board.frame(b)))
    board.tick()  # reaped: both walls booked, as pool verdicts without a route
    for tid in (a, b):
        wall = json.loads(board.latest(tid)["metadata"])["provider_wall"]
        assert wall["scope"] == "credential_pool" and wall["route"] is None
    assert sorted(_cooled(board, a, b, monkeypatch)) == sorted((a, b))
    assert route_inputs(str(board.profile)) is None


def test_a_changed_route_sibling_probe_holds_no_one(board, monkeypatch):
    """A sibling running after its route changed probes an unknown route: no verdict for the old one."""
    rerouted = kb.create_task(board.conn, title="rerouted", assignee="w", priority=1)  # dispatched first
    walled = kb.create_task(board.conn, title="still walled", assignee="w")
    board.tick()
    board.pool_wall(rerouted, monkeypatch)
    board.pool_wall(walled, monkeypatch)
    kb.set_model_override(board.conn, rerouted, "gpt-other", "openai-codex")
    board.alive.add(rerouted)
    assert _next_spawn(board) == [rerouted, walled]


def test_an_unchanged_sibling_probe_holds_the_pool(board, monkeypatch):
    """Control for the test above: the same order without the re-route holds the second card."""
    first = kb.create_task(board.conn, title="first", assignee="w", priority=1)
    second = kb.create_task(board.conn, title="second", assignee="w")
    board.tick()
    board.pool_wall(first, monkeypatch)
    board.pool_wall(second, monkeypatch)
    board.alive.add(first)
    assert _next_spawn(board) == [first]
    _res, guarded = board.tick()
    assert board.spawned[-1] == [] and guarded[second] == PROBED


def test_a_sibling_that_left_the_walled_profile_probes_nothing(board, monkeypatch):
    """Two review cards wall on reviewer ``w``'s pool; one is reopened to its implementer (the real
    ``reopen_review_task``, which records no ``assigned`` event). Its next attempt runs as the
    implementer, so it probes nothing for ``w``'s pool: the card left with ``w`` runs too."""
    left = kb.create_task(board.conn, title="reopened", assignee="implementer", priority=1)
    stayed = kb.create_task(board.conn, title="still with w", assignee="implementer")
    for tid in (left, stayed):
        assert kb.request_review(board.conn, tid, summary="durable candidate", reviewer="w")
    board.tick()
    board.pool_wall(left, monkeypatch)
    board.pool_wall(stayed, monkeypatch)
    board.tick()  # the sweep reaps both walls; each card is within its own cooldown
    assert board.spawned[-1] == [] and board.run(left).status == "review"
    assert kb.reopen_review_task(board.conn, left)
    assert board.run(left).assignee == "implementer" and board.latest(left)["profile"] == "w"
    board.alive.add(left)
    assert _next_spawn(board) == [left, stayed]
    assert board.latest(left)["profile"] == "implementer" and board.latest(stayed)["profile"] == "w"


# The card's assignee and the profile its wall was met under: only a real, equal profile shares the
# pool's verdict. Unassigned and legacy blank rows name no profile, even when both sides agree.
RUNS_AS = {
    "same": ("w", "w", True),
    "explicit_default": ("default", "default", True),
    "other": ("implementer", "w", False),
    "unassigned_vs_default": (None, "default", False),
    "default_vs_unknown": ("default", None, False),
    "both_unknown": (None, None, False),
    "both_blank": ("", "", False),
    "both_whitespace": ("  ", "  ", False),
}


@pytest.mark.parametrize("case", list(RUNS_AS))
def test_only_a_known_equal_profile_runs_as_the_walled_one(conn, case):
    assignee, walled_as, expected = RUNS_AS[case]
    tid = kb.create_task(conn, title=case, assignee="w")
    with kb.write_txn(conn):  # legacy rows: the normal writers reject blanks
        conn.execute("UPDATE tasks SET assignee=? WHERE id=?", (assignee, tid))
    assert pw._runs_as(conn, tid, walled_as) is expected
    assert pw._runs_as(conn, "t_missing", walled_as) is False


def test_unrelated_work_runs_every_tick_while_a_pool_is_held(board, monkeypatch):
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    _wall_round(board, monkeypatch, cards)
    _next_spawn(board)
    _wall_round(board, monkeypatch, cards)
    fresh = kb.create_task(board.conn, title="same profile, never walled", assignee="w")
    other = kb.create_task(board.conn, title="other profile", assignee="v")
    _res, guarded = board.tick()
    assert sorted(board.spawned[-1]) == sorted((fresh, other))
    assert all(guarded.get(t) in (COOLING, PROBED) for t in cards)


def test_an_older_three_part_route_stamp_is_unknown_and_keeps_the_per_card_cooldown(board, monkeypatch):
    """A wall an older runtime recorded stamped its route inputs without ctime: it cannot say an
    in-place, mtime-restoring edit did not happen, so it is unknown, never upgraded to a match."""
    a, b = (kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in "ab")
    board.tick()
    board.pool_wall(a, monkeypatch)
    board.pool_wall(b, monkeypatch)
    board.tick()
    run = board.latest(a)
    meta = json.loads(run["metadata"])
    inputs = meta["provider_wall"]["route"]["inputs"]
    assert any(isinstance(v, list) and len(v) == 4 for v in inputs.values())
    meta["provider_wall"]["route"]["inputs"] = {k: v[:3] if isinstance(v, list) else v for k, v in inputs.items()}
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(meta), run["id"]))
    assert sorted(_cooled(board, a, b, monkeypatch)) == sorted((a, b))


@pytest.mark.parametrize("damage", [
    {"route": {"home": 7}}, {"route": {"inputs": {"config.yaml": "x", ".env": None}}}, {"retry_at": float("nan")},
    {"retry_at": True}, {"phase": "later"}, {"extra": 1},
])
def test_an_unreadable_recorded_wall_is_unknown_and_keeps_the_per_card_cooldown(board, monkeypatch, damage):
    a, b = (kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in "ab")
    board.tick()
    board.pool_wall(a, monkeypatch)
    board.pool_wall(b, monkeypatch)
    board.tick()
    run = board.latest(a)
    meta = json.loads(run["metadata"])
    wall = meta["provider_wall"]
    for key, value in damage.items():
        if key == "route":
            wall["route"] = {**wall["route"], **value}
        else:
            wall[key] = value
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(meta), run["id"]))
    assert sorted(_cooled(board, a, b, monkeypatch)) == sorted((a, b))


def _damage_latest_wall(board, tid, damage):
    """An old writer or corruption rewrote the card's latest recorded wall (``route`` merges)."""
    run = board.latest(tid)
    meta = json.loads(run["metadata"])
    wall = meta["provider_wall"]
    for key, value in damage.items():
        wall[key] = {**wall[key], **value} if key == "route" else value
    with kb.write_txn(board.conn):
        board.conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(meta), run["id"]))


@pytest.mark.parametrize("damage", [
    {"retry_at": 10**400}, {"retry_at": -(10**400)}, {"retry_at": float("inf")},
    {"route": {"home": "/profiles/w\x00"}}, {"route": {"inputs": {"config.yaml": [10**400, 1, 1, 1], ".env": None}}},
], ids=["huge_int", "huge_negative_int", "infinity", "nul_home", "huge_stat"])
def test_a_recorded_wall_with_values_beyond_its_types_never_stops_the_dispatcher(board, monkeypatch, damage):
    """Valid JSON the wall cannot represent (an integer beyond float range, ``Infinity``, a home no
    path can name) is an unknown wall. The first real tick that reads it, once the card's own cooldown
    is over, dispatches the unrelated work too; the damaged card runs on its own cooldown, as before,
    and its attempt probes nothing, so its sibling runs as well."""
    walled = kb.create_task(board.conn, title="damaged wall", assignee="w", priority=1)  # read first
    sibling = kb.create_task(board.conn, title="sibling wall", assignee="w")
    board.tick()
    board.pool_wall(walled, monkeypatch)
    board.pool_wall(sibling, monkeypatch)
    board.tick()  # reaped: both walls booked
    _damage_latest_wall(board, walled, damage)
    fresh = kb.create_task(board.conn, title="same profile, never walled", assignee="w")
    other = kb.create_task(board.conn, title="other profile", assignee="v")
    _res, guarded = board.tick(COOLDOWN)
    assert sorted(board.spawned[-1]) == sorted((walled, sibling, fresh, other)), guarded


@pytest.mark.parametrize("contradiction", [
    None, {"reason": "overloaded", "phase": "turn"}, {"phase": "turn"}, {"reason": "rate_limit"},
    {"scope": "credential_pool_v2"},
], ids=["control", "overloaded_turn", "turn", "turn_reason", "unknown_scope"])
def test_only_a_credential_pool_quota_verdict_waits_for_a_sibling_probe(board, monkeypatch, contradiction):
    """A pool scope on a wall whose reason or phase is not the credential resolver's quota verdict
    (an old writer's or a corrupted record) is unknown: with a sibling probe in flight it still runs
    after its own cooldown. The control, a real pool verdict, waits for that probe."""
    probe = kb.create_task(board.conn, title="probe", assignee="w", priority=1)  # dispatched first
    walled = kb.create_task(board.conn, title="walled", assignee="w")
    board.tick()
    board.pool_wall(probe, monkeypatch)
    board.pool_wall(walled, monkeypatch)
    board.tick()  # reaped: both walls booked
    if contradiction is not None:
        _damage_latest_wall(board, walled, contradiction)
    board.alive.add(probe)
    # Both cooldowns end on the same tick: the card that does not spawn with the probe waits for it.
    assert _next_spawn(board) == ([probe] if contradiction is None else [probe, walled])


GOOD_BACKUP = "backups/config/config.yaml.good.20261109-000000"
CODEX_AT = "model:\n  provider: openai-codex\n  default: gpt-test\n  base_url: {}\n"


# How the profile's effective config is read (primary config.yaml, last-known-good backup), what
# changes after the walls, and whether the held peers run once their own cooldown is over.
RECOVERY_ROUTES = {
    # Broken primary: the loader serves the backup, whose endpoint reads the environment.
    "recovered_env": ({"config.yaml": "model: [\n", GOOD_BACKUP: CODEX_AT.format("${C4_ROUTE_URL}")},
                      lambda board, m: m.setenv("C4_ROUTE_URL", "https://new.example/v1"), True, "https://new.example/v1"),
    # Broken primary: the loader serves the backup, which is replaced by another endpoint.
    "recovered_backup_change": ({"config.yaml": "model: [\n", GOOD_BACKUP: CODEX_AT.format("https://old.example/v1")},
                                lambda board, m: _write(board.profile / GOOD_BACKUP, CODEX_AT.format("https://new.example/v1")),
                                True, "https://new.example/v1"),
    # Control: a valid primary with a literal endpoint and nothing changed still coalesces.
    "valid_unchanged": ({"config.yaml": CODEX_AT.format("https://old.example/v1"),
                         GOOD_BACKUP: CODEX_AT.format("https://other.example/v1")},
                        lambda board, m: None, False, "https://old.example/v1"),
}


def _loader_frames(board, monkeypatch):
    """The attempt frame for a card through the real effective loader and the real attempt policy
    producer, with route inputs stamped around the config read as a worker does."""
    from dataclasses import asdict
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli.config_effective import load_user_config_effective
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(board.profile))
    actor = Principal("owner", str(board.profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(board.profile)), actor=actor, native_owner=True)

    def policy_frame(tid):
        task = board.run(tid)
        params = dict(board="default", db=_db(board.conn), task_id=tid, run_id=task.current_run_id,
                      claim_lock=task.claim_lock)
        stamped = route_inputs(str(board.profile))
        policy, _ = build_kanban_policy(connection, params, load_user_config_effective(board.profile / "config.yaml"),
                                        route_inputs=stamped)
        return {"home": str(board.profile), "policy": asdict(policy)}
    return policy_frame


def _edit_in_place(path: Path, text: str, *, keep_mtime: bool) -> None:
    """Rewrite ``path`` in place (same inode); ``keep_mtime`` restores its modification time, as
    ``cp -p``, ``rsync -t`` or a restore does. Only ctime still records the edit."""
    before = path.stat()
    with open(path, "r+", encoding="utf-8") as f:
        f.write(text)
        f.truncate()
    if keep_mtime:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert (after.st_mtime_ns, after.st_size, after.st_ino) == (before.st_mtime_ns, before.st_size, before.st_ino)
        assert after.st_ctime_ns != before.st_ctime_ns


# An edit of the walled route's profile config after the walls, and whether the held peers (past
# their own cooldown) run: the loader serves the new endpoint whenever the file was edited.
LOADER_EDITS = {
    "unchanged": (None, False),
    "ordinary_edit": (False, True),
    "mtime_preserving_edit": (True, True),
}


@pytest.mark.parametrize("case", list(LOADER_EDITS))
def test_an_edit_the_loader_sees_releases_the_held_peers_even_with_its_mtime_restored(board, monkeypatch, case):
    """COMPOSE-ROUTE-01, through the real effective loader, policy producer, wall producer, sweep and
    dispatch: a same-length in-place endpoint edit whose mtime is restored changes what the loader
    serves (its ``file_signature`` includes ctime), so the walled route no longer applies and both
    peers run once their own cooldown is over. Control: nothing edited, the fresh probe holds both."""
    from hermes_cli.config_effective import load_user_config_effective
    keep_mtime, released = LOADER_EDITS[case]
    config_file = board.profile / "config.yaml"
    _write(config_file, CODEX_AT.format("https://old.example/v1"))
    policy_frame = _loader_frames(board, monkeypatch)
    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    for tid in cards:
        board.pool_wall(tid, monkeypatch, frame=policy_frame(tid))
    probe = _next_spawn(board)
    assert len(probe) == 1
    board.pool_wall(probe[0], monkeypatch, frame=policy_frame(probe[0]))  # a fresh verdict for the pool
    held = [t for t in cards if t not in probe]
    if keep_mtime is not None:
        _edit_in_place(config_file, CODEX_AT.format("https://new.example/v1"), keep_mtime=keep_mtime)
    endpoint = "https://new.example/v1" if released else "https://old.example/v1"
    assert load_user_config_effective(config_file)["model"]["base_url"] == endpoint  # what the next attempt reads
    _res, guarded = board.tick()  # the peers are past their own cooldown; the probe is within its own
    assert sorted(board.spawned[-1]) == (sorted(held) if released else []), (case, guarded)
    assert guarded.get(probe[0]) == COOLING
    assert all(guarded.get(t) == (None if released else PROBED) for t in held), (case, guarded)


@pytest.mark.parametrize("case", list(RECOVERY_ROUTES))
def test_a_route_the_loader_recovers_is_unknown_and_never_held(board, monkeypatch, case):
    """Through the real effective loader, the real attempt policy producer, the real wall producer and
    real dispatch: a profile whose config.yaml does not parse is served the last-known-good settings
    (this process's or ``backups/config``'s), which no route stamp names, so its walls share no
    route and every card retries on its own cooldown, whatever the recovered endpoint becomes. The
    control, a valid primary, still holds both peers behind the fresh probe."""
    from hermes_cli.config_effective import load_user_config_effective
    files, change, released, endpoint = RECOVERY_ROUTES[case]
    for name, text in files.items():
        (board.profile / name).parent.mkdir(parents=True, exist_ok=True)
        _write(board.profile / name, text)
    monkeypatch.setenv("C4_ROUTE_URL", "https://old.example/v1")
    config_file = board.profile / "config.yaml"
    assert load_user_config_effective(config_file)["model"]["base_url"] == "https://old.example/v1"
    policy_frame = _loader_frames(board, monkeypatch)

    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    for tid in cards:
        board.pool_wall(tid, monkeypatch, frame=policy_frame(tid))
    first = _next_spawn(board)  # the round after the walls: one probe, or every card on its own cooldown
    for tid in first:
        board.pool_wall(tid, monkeypatch, frame=policy_frame(tid))  # a fresh verdict for the pool
    change(board, monkeypatch)
    assert load_user_config_effective(config_file)["model"]["base_url"] == endpoint  # what the next attempt reads
    if not released:
        held = [t for t in cards if t not in first]
        _res, guarded = board.tick()  # the peers are past their own cooldown
        assert len(first) == 1 and board.spawned[-1] == [] and all(guarded.get(t) == PROBED for t in held), guarded
        return
    assert sorted(first) == sorted(cards), (case, first)  # no pool hold at all
    assert sorted(_next_spawn(board)) == sorted(cards), case  # and none after the recovered route changed


@pytest.mark.parametrize("edited", ["profile", "managed"])
def test_the_route_stamp_comes_from_the_policy_capture_not_the_failure(tmp_path, monkeypatch, edited):
    """Through the real effective-config loader and attempt policy producer: route inputs stamped
    before the config read are kept only when unchanged after the policy is built, and a wall names
    that route only while they are still unchanged when it fails; an edit on either side of the
    capture makes the route unknown, whether to the profile's config.yaml or to the managed
    scope's, which the real loader overlays (here: a fallback chain the policy did not see)."""
    from contextlib import closing
    from dataclasses import asdict
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.kanban_db_provider_wall import wall_from_exception
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(tmp_path))
    from hermes_cli.config_effective import load_user_config_effective
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  provider: openai-codex\n  default: gpt-test\n", encoding="utf-8")
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    managed_file = managed_dir / "config.yaml"
    managed_file.write_text("agent:\n  max_turns: 50\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("owned")
    with closing(connect(board="owned")) as conn:
        task_id = kb.create_task(conn, title="owner", assignee="default", workspace_kind="dir",
                                 workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal("owner", str(tmp_path), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board="owned", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    pool = _codex_quota_exhausted_error(109934, exhausted="credential_pool")

    def wall(policy):
        return wall_from_exception(pool, {"home": str(tmp_path), "policy": asdict(policy)})

    target = config_file if edited == "profile" else managed_file
    edits = iter(("fallback_providers:\n  - provider: openrouter\n    model: x\n", "# edited again\n"))

    def edit():
        target.write_text(target.read_text(encoding="utf-8") + next(edits), encoding="utf-8")

    stamped = route_inputs(str(tmp_path))
    config = load_user_config_effective(config_file)
    assert "fallback_providers" not in config
    policy, _ = build_kanban_policy(connection, params, config, route_inputs=stamped)
    assert json.loads(policy.kanban_json)["route_inputs"] == stamped
    assert wall(policy).pool_route() is not None
    edit()  # after the capture, before the attempt meets the wall: its actual route is unknown
    assert wall(policy).route is None and wall(policy).scope == "credential_pool"
    assert load_user_config_effective(config_file)["fallback_providers"] == [{"provider": "openrouter", "model": "x"}]
    before = route_inputs(str(tmp_path))
    config = load_user_config_effective(config_file)
    edit()  # between the stamp and the policy build: the stamp does not vouch for this policy
    policy, _ = build_kanban_policy(connection, params, config, route_inputs=before)
    assert json.loads(policy.kanban_json)["route_inputs"] is None and wall(policy).route is None
    assert stamped["managed_dir"] == str(managed_dir)  # the stamp names the scope the loader read


def _codex_pool(home: Path, *, usable: bool) -> None:
    """The profile's own auth store: two openai-codex pool rows frozen by a usage limit, and, once
    capacity returns, a third usable row (an added credential). Placeholder tokens, never JWTs, so the
    resolver's early-reset probe stays offline."""
    cooling = time.time() + 109934
    rows = [{"id": f"row{n}", "source": "manual:codex", "auth_type": "oauth", "access_token": f"at-{n}",
             "refresh_token": f"rt-{n}", "last_status": "exhausted", "last_error_code": 429,
             "last_error_reason": "usage_limit_reached", "last_error_reset_at": cooling} for n in range(2)]
    if usable:
        rows.append({"id": "row-added", "source": "manual:codex", "auth_type": "oauth",
                     "access_token": "at-added", "refresh_token": "rt-added"})
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {"openai-codex": rows}}),
                                    encoding="utf-8")


def test_a_pool_exhausted_at_agent_construction_is_probed_once_and_released_by_new_capacity(board, monkeypatch, tmp_path):
    """The incident's own branch, composed: each attempt runs the managed worker's real construction
    step (``agent.managed_worker._construct_agent`` -> ``worker_api_key`` -> the real runtime resolver
    -> ``resolve_codex_runtime_credentials`` on the profile's auth store), whose exhausted pool raises
    the real AuthError that the real call site books. Repeated rounds probe once each; when a usable
    row appears the next probe resolves it, and every held card runs on the following tick."""
    from agent.managed_worker import _construct_agent
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    _codex_pool(board.profile, usable=False)
    frozen = {**CODEX_CONFIG, "fallback_providers": []}
    policy = SimpleNamespace(request_json="{}", effective=lambda: (frozen, "gpt-test"))
    constructed: list[str] = []

    def attempt(tid):
        """One attempt: construction fails at credentials (booked exit 75) or builds the agent. The
        spawn already chose its interpreter's fate from the pool (``board.alive``); the real
        construction must agree."""
        task = board.run(tid)
        bound = os.getpid() if tid in board.alive else _dead_pid()
        with kb.write_txn(board.conn):
            kb._append_event(board.conn, tid, "worker_bound",
                             {"pid": bound, "claim_lock": task.claim_lock, "started_at": "x"}, run_id=task.current_run_id)
        with monkeypatch.context() as m:
            m.setenv("HERMES_HOME", str(board.profile))
            m.setattr(os, "getpid", lambda: bound)
            try:
                _construct_agent(board.frame(tid), lambda **kw: SimpleNamespace(**kw), policy, api_key=None)
            except Exception as exc:  # health: allow BLE001 -- the attempt crashes exactly as the worker does
                return exc
        constructed.append(tid)
        return None

    cards = [kb.create_task(board.conn, title=f"owner {n}", assignee="w") for n in range(3)]
    board.tick()
    errors = [attempt(t) for t in board.spawned[-1]]
    assert all("quota exhausted" in str(e) for e in errors), errors
    wall = json.loads(_cooled_latest(board, cards[0]))
    assert (wall["reason"], wall["phase"], wall["scope"]) == ("quota", "credentials", "credential_pool")
    assert wall["route"] is not None and "retry after" in wall["evidence"]
    probes = []
    for _ in range(12):  # two rounds
        board.tick()
        probes.append(list(board.spawned[-1]))
        for tid in board.spawned[-1]:
            assert attempt(tid) is not None
    assert [len(p) for p in probes if p] == [1, 1], probes
    _codex_pool(board.profile, usable=True)  # capacity returns: auth.json is not a route input
    board.alive.update(cards)  # so every later attempt resolves a credential and keeps running
    probe = _next_spawn(board)
    assert len(probe) == 1 and attempt(probe[0]) is None
    held = [t for t in cards if t not in probe]
    _res, guarded = board.tick()
    assert all(guarded.get(t) == PROBED for t in held)  # the probe's verdict is pending
    kb.complete_task(board.conn, probe[0], summary="worked", expected_run_id=board.run(probe[0]).current_run_id)
    board.tick()
    assert sorted(board.spawned[-1]) == sorted(held)
    assert all(attempt(t) is None for t in held) and sorted(constructed) == sorted(cards)
    for tid in cards:
        assert board.run(tid).consecutive_failures == 0
        assert all(e.kind not in ("gave_up", "crashed") for e in kb.list_events(board.conn, tid))


def _cooled_latest(board, tid) -> str:
    """The next tick reaps the round; the card's booked wall, as recorded."""
    board.tick()
    return json.dumps(json.loads(board.latest(tid)["metadata"])["provider_wall"])
