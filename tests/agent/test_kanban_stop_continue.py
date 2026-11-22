"""The stop nudge sends an unfinished worker back to work; it does not offer block as an exit.

Cryo t_7752959d runs 25385, 25387 and 25388: each worker ended its turn in plain text with work
left, the nudge listed ``kanban_block`` beside ``kanban_complete`` as an equal way out, and the
next call blocked the card ``transient`` with no obstacle. The nudge now says to continue
unfinished work and keeps ``kanban_block`` for a concrete obstacle outside the worker's control.
"""

from __future__ import annotations

import pytest

from agent.kanban_stop import build_kanban_stop_nudge


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_7752959d")
    return monkeypatch


_STOPPED_EARLY = [
    {"role": "user", "content": "work kanban task t_7752959d"},
    {"role": "assistant", "content": "Execution is ending here; the next step is the frontend build."},
]


def _nudge(**kw):
    nudge = build_kanban_stop_nudge(messages=_STOPPED_EARLY, **kw)
    assert nudge is not None
    return nudge


def test_unfinished_work_is_told_to_continue(worker):
    nudge = _nudge()
    assert "If authorized work remains that you can do now, keep doing it" in nudge
    assert "Ending your turn is not a reason to stop" in nudge
    # Continuing comes before any handoff or block.
    assert nudge.index("keep doing it") < nudge.index("kanban_complete(") < nudge.index("kanban_block(")


def test_block_is_only_for_a_concrete_external_obstacle(worker):
    nudge = _nudge()
    assert "`kanban_block(reason=...)` only for a concrete obstacle outside your control" in nudge
    assert "name the missing input, who provides it and when work can resume" in nudge
    assert "Unfinished work, a long task or the end of a turn is not such an obstacle" in nudge
    assert "if you are blocked" not in nudge


def test_handoff_and_bound_are_unchanged(worker):
    assert "t_7752959d" in _nudge()
    assert build_kanban_stop_nudge(messages=_STOPPED_EARLY, attempts=2) is None
    blocked = _STOPPED_EARLY + [{"role": "tool", "name": "kanban_block", "content": "ok"}]
    assert build_kanban_stop_nudge(messages=blocked) is None
