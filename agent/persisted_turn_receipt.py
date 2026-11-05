"""``message.complete.persisted_turn``: the committed row addresses of the agent's current turn.

Shared by every host that publishes a turn's completion (the TUI gateway's in-process turn and
the session authority's TurnRunner turn), so a client sees one receipt contract whichever host
ran the turn. Clients key on it to bind a streamed reply to its stored row (``SessionMessage.row_id``)
instead of comparing words.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.conversation_compression_archive import MERGED_DURABLE_ROWS
from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MERGED_TURN_PREFIX, message_uid_or_none, uid_list


def committed_row_id(message: Any) -> int | None:
    row_id = message.get("_row_id") if isinstance(message, dict) else None
    return row_id if message.get(_DB_PERSISTED_MARKER) and type(row_id) is int and row_id > 0 else None


def _folds_other_rows(message: Any) -> bool:
    """The alternation repair folded another row's text into this dict: its ``_row_id`` names only one
    of the stored rows its content spans, so it is no single row's identity."""
    return isinstance(message, dict) and (
        message.get(MERGED_TURN_PREFIX) is not None or bool(uid_list(message.get(ABSORBED_MESSAGE_UIDS)))
        or int(message.get(MERGED_DURABLE_ROWS) or 0) > 1)


def _single_row_id(message: Any) -> int | None:
    """``committed_row_id`` of a dict that stands for exactly its own stored row, else None."""
    return None if _folds_other_rows(message) else committed_row_id(message)


RowIdentity = Callable[[int], "tuple[str, str | None] | None"]


def row_identity_lookup(session_db: Any, session_id: Any) -> RowIdentity | None:
    """``row id -> (role, message_uid)`` of an ACTIVE stored row of ``session_id`` (None when absent);
    None when there is no store to ask. Anything but that pair reads as absent."""
    get = getattr(session_db, "get_message_identity", None)
    if not callable(get) or not session_id:
        return None

    def lookup(row_id: int):
        found = get(session_id, row_id)
        return found if isinstance(found, tuple) and len(found) == 2 and isinstance(found[0], str) else None
    return lookup


def _prompt_row_id(
    anchor: Any, turn_message: Any, turn_override: Any, row_identity: RowIdentity | None,
) -> tuple[int | None, bool]:
    """``(row id, folded)``: this turn's stored prompt row, proven by persistence-minted identity
    (``message_uid``, ``_row_id``) and SQLite, never searched by text. ``turn_message`` is the dict the
    turn appended (``agent._turn_user_message``), ``turn_override`` its clean persisted text.

    - Unfolded anchor: its own row, unless both carry uids and they differ (another message).
    - A folded anchor that is the turn's own message (it absorbed later text): its own row.
    - The repair folded this turn into an older unanswered row (the anchor holds ``prefix + "\n\n" +``
      the turn's text, live or override-cleaned; its own ``_row_id`` is the OLDER row): ``folded``. The
      prompt is the single row the merge witness absorbed, only when its uid is the turn's (when the turn
      dict carries one) and SQLite holds it active as a user row with that uid. Retired into a composite
      row, or anything else: no single row is the prompt, None.
    - Otherwise unproven: None (the identity is omitted, never guessed)."""
    from agent.turn_context import holds_merged_turn

    turn_uid = message_uid_or_none(turn_message) if isinstance(turn_message, dict) else None
    anchor_uid = message_uid_or_none(anchor)
    if not _folds_other_rows(anchor):
        if turn_uid is not None and anchor_uid is not None and anchor_uid != turn_uid:
            return None, False
        return committed_row_id(anchor), False
    if turn_uid is not None and anchor_uid == turn_uid:
        return committed_row_id(anchor), False
    texts = (turn_message.get("content") if isinstance(turn_message, dict) else None, turn_override)
    if not any(holds_merged_turn(anchor, text) for text in texts):
        return None, False
    absorbed, uids = list(anchor.get("_absorbed_row_ids") or ()), uid_list(anchor.get(ABSORBED_MESSAGE_UIDS))
    if (len(absorbed) == 1 and len(uids) == 1 and type(absorbed[0]) is int and turn_uid in (None, uids[0])
            and row_identity is not None and row_identity(absorbed[0]) == ("user", uids[0])):
        return absorbed[0], True
    return None, True


def verify_receipt_rows(receipt: dict | None, row_identity: RowIdentity | None) -> dict | None:
    """Keep only what SQLite confirms: every listed id an active stored row, the prompt and steer ids
    user rows, the final id an assistant row. A failed identity is omitted (fail closed); an address
    SQLite does not hold is dropped and the receipt is no longer complete."""
    if not receipt or row_identity is None:
        return receipt
    ids = {*receipt.get("row_ids", ()), *receipt.get("user_row_ids", ())}
    ids.update(receipt[k] for k in ("user_row_id", "final_assistant_row_id") if k in receipt)
    try:
        roles = {rid: (row_identity(rid) or (None,))[0] for rid in ids}
    except sqlite3.Error:
        return None  # the store cannot confirm anything: publish no receipt
    out = dict(receipt)
    out["row_ids"] = [rid for rid in receipt.get("row_ids", ()) if roles.get(rid) is not None]
    if not out["row_ids"]:
        return None
    if out["row_ids"] != receipt.get("row_ids"):
        out["complete"] = False
    if ("user_row_id" in out and roles.get(out["user_row_id"]) != "user") or any(
            roles.get(rid) != "user" for rid in out.get("user_row_ids", ())):
        out.pop("user_row_id", None)
        out.pop("user_row_ids", None)
        out["complete"] = False
    if "final_assistant_row_id" in out and roles.get(out["final_assistant_row_id"]) != "assistant":
        out.pop("final_assistant_row_id")
        out["complete"] = False
    return out


def persisted_turn_receipt(
    messages: Any, start: Any, history: list, raw: Any, status: str, *,
    compression_unchanged: bool, prefix_row_matches: Callable[[Any, Any], bool], turn_message: Any = None,
    turn_override: Any = None, row_identity: RowIdentity | None = None,
) -> dict | None:
    """Report committed row addresses, never a text/timestamp search for a matching turn.

    ``start`` is the agent's persistence cursor (``_persist_user_message_idx``), re-anchored during
    compaction. A partial receipt can address its surviving rows, but cannot retire a client's
    entire streamed turn. Full coverage additionally requires the unchanged pre-turn prefix
    (``prefix_row_matches(history_row, turn_row)`` for every row before ``start``), no compaction
    and no redirected user boundary.

    The prompt id is proven by identity (``_prompt_row_id``): when the alternation repair folded this
    turn's message into an older unanswered row, the anchor's own ``_row_id`` is that OLDER row, so the
    turn's own committed row is reported instead, and no folded dict ever stands for one row id.
    """
    if (not isinstance(messages, list) or type(start) is not int or not 0 <= start < len(messages)
            or not isinstance(messages[start], dict) or messages[start].get("role") != "user"):
        return None
    tail = messages[start:]
    prompt_id, folded = _prompt_row_id(tail[0], turn_message, turn_override, row_identity)
    anchor_id = committed_row_id(tail[0])
    # A folded anchor IS an older row; the turn is new only when its own stored row is proven folded in.
    if not (folded and prompt_id is not None) and any(before is tail[0] or (anchor_id is not None and anchor_id == committed_row_id(before))
                          for before in history):
        return None  # preflight returned the old transcript, not a new turn
    row_ids = [rid for rid in [prompt_id, *map(_single_row_id, tail[1:])] if rid is not None]
    if not row_ids:
        return None
    receipt: dict = {"row_ids": row_ids, "complete": False}
    if prompt_id is not None:
        receipt["user_row_id"] = prompt_id
    # Every rendered user row of the turn in order (the prompt, then each steer/redirect row), so a
    # client binds its optimistic bubbles even when the turn holds more than one prompt row. Only a
    # fully proven list is published: a gap would shift every later pairing.
    users = [message for message in tail if message.get("role") == "user" and message.get("display_kind") != "hidden"]
    user_row_ids = [prompt_id if message is tail[0] else _single_row_id(message) for message in users]
    if users and None not in user_row_ids:
        receipt["user_row_ids"] = user_row_ids
    last = tail[-1]
    # Equality only verifies the structurally selected final row's body: it never selects an identity.
    if (status == "complete" and last.get("role") == "assistant" and not last.get("tool_calls")
            and last.get("content") == raw and (final_id := _single_row_id(last)) is not None):
        receipt["final_assistant_row_id"] = final_id
    prefix_unchanged = start == len(history) and all(
        prefix_row_matches(before, after) for before, after in zip(history, messages[:start]))
    receipt["complete"] = bool(
        prefix_unchanged and compression_unchanged and not folded and not _folds_other_rows(tail[0])
        and len(row_ids) == len(tail)
        and sum(message.get("role") == "user" for message in tail) == 1
        and "final_assistant_row_id" in receipt)
    return receipt
