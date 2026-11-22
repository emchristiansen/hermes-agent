"""The executor's persistence step for results produced under a caller's ``output_control``
(tools/output_control.py).

A controlled envelope is bounded IN PLACE to the caller's inline budget and this model's
per-result limit: its metadata and recovery location survive, and its payload enters the
ordinary archive (verbatim) only when the caller named no destination and the payload must be cut.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from agent.message_sanitization import coalesce_tool_call_id
from tools.budget_config import BudgetConfig
from tools.output_control import (
    CONTROLLED_MESSAGE_FLAG, bound_envelope, control_for_call, dump_envelope, parse_envelope,
)
from tools.tool_result_storage import _content_text_size, archived_result_size_bound, persist_exact_text


def persist_controlled_result(tool_name: str, args: Any, result: Any, *, tool_call_id: str,
                              messages: list, budget: BudgetConfig, env: Callable[[], Any],
                              agent: Any = None, overhead: int = 0) -> tuple[str, str] | None:
    """``(bounded result, message flag)`` for a controlled envelope; ``None`` for every other
    result, which takes the ordinary persistence path.

    The bound is final when written: the transcript row is flushed right after, and a later
    aggregate pass could only change the live copy. So the result gets this model's per-result
    limit and the turn budget still unused after reserving room for every call still to come in
    *agent*'s current batch (``with_controlled_batch``), less *overhead*: what the executor adds
    around the result in its message (``finish_controlled_message``). The flag (message key
    ``CONTROLLED_MESSAGE_FLAG``) tells ``enforce_turn_budget`` to leave the row as written; its
    value says where the complete content lives (``"destination"``, or ``"default"``: the
    ordinary archive)."""
    control = control_for_call(tool_name, args)
    controlled = parse_envelope(result) if control is not None else None
    if controlled is None:
        return None
    to_destination = control.destination is not None
    # Reserve room for every later call of the batch, so the batch fits the turn budget live
    # whatever those calls return, while the aggregate pass never touches a controlled row and
    # every controlled row stays as flushed (``_reserve``). The payload yields to that reserve;
    # the mandatory receipt may draw on it (``receipt_limit``) before it counts as over budget.
    later = _later_calls(agent, tool_call_id)
    unused = max(0, budget.turn_budget - _batch_result_chars(messages) - overhead)
    share = unused // (1 + len(later))
    threshold = max(0, budget.resolve_threshold(tool_name) - overhead)
    bounded = bound_envelope(
        controlled, inline_chars=control.inline_chars,
        limit=min(threshold, max(0, unused - sum(_reserve(call, share, env, budget) for call in later))),
        archive=None if to_destination else (lambda text: persist_exact_text(text, tool_call_id, env())),
        receipt_limit=min(threshold, unused),
    )
    return dump_envelope(bounded), "destination" if to_destination else "default"


def finish_controlled_message(message: dict, persisted: str, controlled: tuple[str, str],
                              rebound: Callable[[int], tuple[str, str] | None]) -> str:
    """Mark the executor's composed *message* as a controlled row and keep it within its bound:
    hints, notices or wrappers placed around the bounded result count against the same room, so
    the result is bounded again (*rebound*) with that overhead set aside and substituted in
    place. Returns *persisted* (the result plus what was appended to it) with the same change."""
    bounded, flag = controlled
    message[CONTROLLED_MESSAGE_FLAG] = flag
    content = message.get("content")
    if not isinstance(content, str) or bounded not in content or len(content) <= len(bounded):
        return persisted
    refit = rebound(len(content) - len(bounded))[0]
    message["content"] = content.replace(bounded, refit, 1)
    return persisted.replace(bounded, refit, 1)


def with_controlled_batch(execute: Callable[..., None]) -> Callable[..., None]:
    """Decorate a batch executor ``(agent, assistant_message, ...)``: while it runs, the whole
    batch is known to :func:`persist_controlled_result` (the outermost executor's batch, since
    the sequential/segmented ones run sub-batches through the others)."""
    @functools.wraps(execute)
    def run(agent: Any, assistant_message: Any, *args: Any, **kwargs: Any) -> None:
        if getattr(agent, _BATCH_ATTR, None) is not None:
            return execute(agent, assistant_message, *args, **kwargs)
        setattr(agent, _BATCH_ATTR, list(assistant_message.tool_calls or ()))
        try:
            return execute(agent, assistant_message, *args, **kwargs)
        finally:
            setattr(agent, _BATCH_ATTR, None)
    return run


_BATCH_ATTR = "_output_control_batch"


def _later_calls(agent: Any, tool_call_id: str) -> list:
    """The model tool calls after *tool_call_id* in *agent*'s current batch."""
    batch = getattr(agent, _BATCH_ATTR, None) or []
    ids = [coalesce_tool_call_id(call) for call in batch]
    if tool_call_id not in ids:
        return []
    return batch[ids.index(tool_call_id) + 1:]


def _reserve(tool_call: Any, share: int, env: Callable[[], Any], budget: BudgetConfig) -> int:
    """Room for a later call's result: its equal *share*, capped at its per-result threshold
    (above which an ordinary result is archived before its flush), and never less than the
    archived form the aggregate pass may reduce an ordinary result to. Every call is reserved
    so, controlled or not: even an accepted control can be refused at run time (an unavailable
    destination) with an ordinary error result."""
    name = getattr(getattr(tool_call, "function", None), "name", "") or ""
    room = int(min(share, budget.resolve_threshold(name)))
    return max(room, archived_result_size_bound(coalesce_tool_call_id(tool_call), env(), budget))


def _batch_result_chars(messages: list) -> int:
    """Text characters of the tool results already appended for the current assistant turn."""
    total = 0
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        total += _content_text_size(message.get("content", ""))
    return total
