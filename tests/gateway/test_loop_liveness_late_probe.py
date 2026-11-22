"""Loop-liveness watchdog: a probe answering after its own wait expired still clears the strikes.

Each test scripts the loop so that every callback runs at a known point relative to the
watchdog's probe waits, instead of racing wall-clock sleeps.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Callable, Optional
from unittest.mock import patch

from gateway.shutdown_watchdog import start_loop_liveness_watchdog

PROBE_TIMEOUT_S = 0.02
JOIN_TIMEOUT_S = 10.0


class ScriptedLoop:
    """Records probe callbacks without running them. ``actions[n]`` runs inside the n-th
    ``call_soon_threadsafe``: after probe n is issued and before its wait begins."""

    def __init__(self, actions: Optional[dict[int, Callable[[ScriptedLoop], None]]] = None):
        self.callbacks: list[Callable[[], None]] = []
        self.actions = actions or {}

    def call_soon_threadsafe(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)
        if action := self.actions.get(len(self.callbacks)):
            action(self)

    def run(self, probe: int) -> None:
        """Run probe ``probe``'s callback (1-based), as the loop would when it reaches it."""
        self.callbacks[probe - 1]()


class Run:
    """Drive one watchdog generation to completion with every irreversible step recorded."""

    def __init__(self, loop, *, max_strikes: int, probe_timeout: float = PROBE_TIMEOUT_S,
                 on_dump: Optional[Callable[[], None]] = None,
                 on_mark: Optional[Callable[[], None]] = None):
        self.loop, self.max_strikes, self.probe_timeout = loop, max_strikes, probe_timeout
        self.on_dump, self.on_mark = on_dump, on_mark
        self.exits: list[tuple[int, int]] = []  # (exit code, probes issued when it fired)
        self.dumps: list[dict] = []
        self.marks: list[tuple] = []
        self.warnings: list[str] = []
        self.handle = None
        self._ready = threading.Event()

    def stop(self) -> None:
        assert self._ready.wait(timeout=2.0)
        self.handle.stop()

    def _issued(self) -> int:
        return len(self.loop.callbacks)

    def _dump(self, _path, *, delay_s, snapshot, event) -> None:
        self.dumps.append({"delay_s": delay_s, "event": event, "probes": self._issued(), **snapshot})
        if self.on_dump is not None:
            self.on_dump()

    def _mark(self, *args) -> None:
        self.marks.append(args)
        if self.on_mark is not None:
            self.on_mark()

    def _exit(self, code: int) -> None:
        self.exits.append((code, self._issued()))
        self.stop()  # a patched exit returns; stop the thread as a real exit would end it

    def __call__(self) -> Run:
        with (
            patch("gateway.shutdown_watchdog._write_watchdog_dump", side_effect=self._dump),
            patch("gateway.shutdown_watchdog._mark_exited_quietly", side_effect=self._mark),
            patch("gateway.shutdown_watchdog._hard_exit", side_effect=self._exit),
            patch("gateway.shutdown_watchdog.logger.warning",
                  side_effect=lambda msg, *a: self.warnings.append(msg)),
        ):
            self.handle = start_loop_liveness_watchdog(
                self.loop, probe_interval=0.001, probe_timeout=self.probe_timeout,
                max_strikes=self.max_strikes)
            assert self.handle is not None
            self._ready.set()
            self.handle.join(timeout=JOIN_TIMEOUT_S)
        assert not self.handle.is_alive(), "watchdog thread did not finish"
        return self


def test_first_probe_answering_during_second_wait_clears_its_strike():
    """Eric's case: probe 1's wait expires (strike 1); during probe 2's wait probe 1's callback runs
    while probe 2's own has not. That proves the loop recovered, so no restart follows."""
    run = Run(ScriptedLoop({2: lambda loop: loop.run(1), 3: lambda _loop: run.stop()}),
              max_strikes=2)
    run()
    assert run.exits == []
    assert run.dumps == []


def test_late_answer_clears_all_accumulated_strikes_and_only_later_misses_count():
    """Probes 1 and 2 miss; probe 1 answers during probe 3's wait. Both strikes clear, and the
    restart needs three genuinely later misses (probes 4, 5, 6)."""
    run = Run(ScriptedLoop({3: lambda loop: loop.run(1)}), max_strikes=3)()
    assert run.exits == [(75, 6)]
    assert [(d["strikes"], d["probes"]) for d in run.dumps] == [(3, 6)]
    assert len(run.marks) == 1 and run.marks[0][0] == 75


def test_answer_between_probes_resets_before_the_next_probe_and_its_miss_counts_fresh():
    """Probe 1 misses, then answers before probe 2 is issued. Probe 2's miss is a fresh strike 1,
    so the second strike is probe 3: exit after exactly three probes."""
    installed = threading.Event()
    loop = ScriptedLoop({1: lambda _loop: installed.wait(timeout=2.0)})
    run = Run(loop, max_strikes=2)
    gap_waits = []

    def start() -> None:
        assert run._ready.wait(timeout=2.0)
        stop_event = run.handle._stop_event
        original_wait = stop_event.wait

        def wait_answering_probe_1_first(timeout=None):
            gap_waits.append(timeout)
            if len(gap_waits) == 1:  # the interval wait after probe 1's miss was counted
                loop.run(1)
            return original_wait(timeout)

        stop_event.wait = wait_answering_probe_1_first
        installed.set()

    threading.Thread(target=start, daemon=True).start()
    run()
    assert gap_waits, "the gap wait was never intercepted"
    assert run.exits == [(75, 3)]
    assert [d["strikes"] for d in run.dumps] == [2]


def test_no_callbacks_at_all_escalates_after_max_strikes():
    run = Run(ScriptedLoop(), max_strikes=3)()
    assert run.exits == [(75, 3)]
    assert [(d["strikes"], d["event"]) for d in run.dumps] == [(3, "loop_liveness_watchdog_fired")]
    assert run.dumps[0]["delay_s"] == PROBE_TIMEOUT_S * 3
    assert run.marks == [(75, "loop_liveness_watchdog")]
    assert run.warnings == []


def test_duplicate_callbacks_of_an_already_observed_probe_are_not_new_evidence():
    """Probe 1's late answer clears strike 1 once. Re-running its callback during probes 3 and 4
    proves nothing new, so their misses still restart the gateway."""
    run = Run(ScriptedLoop({2: lambda loop: loop.run(1), 3: lambda loop: loop.run(1),
                            4: lambda loop: loop.run(1)}), max_strikes=2)()
    assert run.exits == [(75, 4)]


def test_stale_generation_callbacks_cannot_clear_a_new_generation_strike():
    """A replaced watchdog's late probe callbacks running now say nothing about the current
    generation: the new watchdog still escalates on its own two misses."""
    old_loop = ScriptedLoop({2: lambda _loop: old.stop()})
    old = Run(old_loop, max_strikes=1000)
    old()
    assert len(old_loop.callbacks) == 2 and old.exits == []

    def answer_old_generation(_loop) -> None:
        for callback in old_loop.callbacks:
            callback()

    new = Run(ScriptedLoop({1: answer_old_generation, 2: answer_old_generation}), max_strikes=2)()
    assert new.exits == [(75, 2)]


def test_answer_during_the_stack_dump_withdraws_the_restart():
    """Escalation has begun (critical log, dump) when the struck probe answers. The observation
    before the irreversible stamp is the commit point, so the restart is withdrawn; a later genuine
    miss escalates afresh."""
    loop = ScriptedLoop()
    dumps_seen = []

    def answer_during_first_dump() -> None:
        dumps_seen.append(len(loop.callbacks))
        if len(dumps_seen) == 1:
            loop.run(1)

    run = Run(loop, max_strikes=1, on_dump=answer_during_first_dump)()
    assert dumps_seen == [1, 2]
    assert len(run.warnings) == 1 and "restart withdrawn" in run.warnings[0]
    assert run.exits == [(75, 2)]
    assert len(run.marks) == 1


def test_answer_after_the_commit_point_cannot_stop_the_restart():
    """The unavoidable race: once the final observation found nothing, an answer arriving while
    the exit is stamped no longer changes the decision."""
    loop = ScriptedLoop()
    run = Run(loop, max_strikes=1, on_mark=lambda: loop.run(1))()
    assert run.exits == [(75, 1)]
    assert run.warnings == []


class GatedRealLoop:
    """A real asyncio loop on its own thread; ``before[n]`` runs just before probe n is handed to
    the real ``call_soon_threadsafe``, so blockers can be queued around the probes in FIFO order."""

    def __init__(self, before: dict[int, Callable[[], None]]):
        self.loop = asyncio.new_event_loop()
        self.before = before
        self.callbacks: list = []  # issue count, for Run's bookkeeping
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def call_soon_threadsafe(self, callback) -> None:
        self.callbacks.append(callback)
        if action := self.before.get(len(self.callbacks)):
            action()
        self.loop.call_soon_threadsafe(callback)

    def block_until(self, gate: threading.Event) -> None:
        self.loop.call_soon_threadsafe(gate.wait, JOIN_TIMEOUT_S)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=JOIN_TIMEOUT_S)
        self.loop.close()


def test_real_loop_late_first_probe_during_second_wait_prevents_restart():
    """Real asyncio dispatch: the loop is blocked through probe 1's wait, then runs probe 1 during
    probe 2's wait and blocks again before reaching probe 2. Probe 1's late answer clears strike 1."""
    gate1, gate2 = threading.Event(), threading.Event()
    run_ref: dict = {}

    def during_probe_2() -> None:
        loop.block_until(gate2)  # queued after probe 1, before probe 2
        gate1.set()  # loop now runs probe 1, then blocks on gate2: probe 2's wait expires

    def during_probe_3() -> None:
        gate2.set()

    loop = GatedRealLoop({2: during_probe_2, 3: during_probe_3,
                          4: lambda: run_ref["run"].stop()})
    try:
        loop.block_until(gate1)
        run = run_ref["run"] = Run(loop, max_strikes=2, probe_timeout=0.5)
        run()
    finally:
        gate1.set()
        gate2.set()
        loop.close()
    assert run.exits == []
    assert run.dumps == []


def test_real_loop_that_never_recovers_still_restarts():
    gate = threading.Event()
    loop = GatedRealLoop({})
    try:
        loop.block_until(gate)
        run = Run(loop, max_strikes=2, probe_timeout=0.1)()
    finally:
        gate.set()
        loop.close()
    assert run.exits == [(75, 2)]


def test_answer_observation_order_is_the_lock_order():
    """The linearization boundary: an observation reports exactly the answers recorded before it.
    An answer recorded after an observation is reported by the next one, and only once; a repeated
    or older callback is never new evidence; another generation's answers are separate."""
    from gateway.shutdown_watchdog import _ProbeAnswers

    answers, other = _ProbeAnswers(), _ProbeAnswers()
    first, second = answers.issue(), answers.issue()
    assert (first.sequence, second.sequence) == (1, 2)
    assert answers.observe_new() is False
    first.set()
    assert first.is_set()
    assert answers.observe_new() is True  # recorded before this observation
    assert answers.observe_new() is False  # each answer is reported once
    first.set()  # duplicate
    assert answers.observe_new() is False
    second.set()
    first.set()  # an older probe after a newer one
    assert answers.observe_new() is True
    assert answers.observe_new() is False
    other.issue().set()
    assert answers.observe_new() is False and other.observe_new() is True
