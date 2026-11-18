"""``hermes kanban redirect|steer <task_id> <text>``: correct the LIVE attempt of a card.

Two owner calls on the caller's own identity, both through the gateway that owns the attempt:
``session.kanban_run`` (read-only, ``session:read``) binds the named run to its session and execution
generation, then ``session.redirect`` / ``session.steer`` (``session:control``) carries that generation to
the protected control point, which rechecks claim, run and generation. An attempt that ended, was
reclaimed or was replaced is reported stale and nothing is delivered to a successor run; a correction the
owner could not hand to a live agent is kept on the card's task and answered as kept. Sending the same
correction again may duplicate it, never lose it: this verb never waits on an earlier send.

Exit 0: delivered, or kept on the card and not delivered (the owner's rejected-but-kept answer). 3: refused
(stale run, denied, validation, rejected); a refusal makes no claim about retention, because the owner may
already have kept the text on the card before refusing. 4: the owner could not be reached or could not say
whether the text was applied (unreachable, timeout, correction_outcome_unknown). Only
correction_outcome_unknown says the owner retained the text; a transport failure may have happened before
the owner received it. In every exit-4 case a resend is safe: a message resend only, never a business replay.
"""
from __future__ import annotations

import argparse
import asyncio
import json


def _print(payload) -> None:
    print(json.dumps(payload, sort_keys=True))


#: The owner reasons that say a correction may or may not have been applied. Every other owner refusal
#: (``unknown_method``, ``invalid_timeout``, authorization, stale, validation, ...) is a definitive refusal of
#: this request and exits 3; it says nothing about retention, because the owner may already have kept the
#: text on the card before refusing. A transport timeout or lost connection is not an owner reason: it is a
#: plain ``GatewayClientError`` and is exit 4 (unreachable).
AMBIGUOUS_REASONS = frozenset({"correction_outcome_unknown"})


def _ambiguous(reason: str) -> bool:
    return reason in AMBIGUOUS_REASONS


async def _send(task_id: str, board: str, run_id: int, verb: str, text: str) -> tuple[int, dict]:
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, connect_gateway
    try:
        async with connect_gateway() as client:
            try:
                found = await client.rpc("session.kanban_run", board=board, task_id=task_id, run_id=run_id)
            except GatewayRPCError as exc:
                return 3, {"status": "refused", "stage": "discovery", "reason": str(exc)}
            where = {"run_id": run_id, "session_id": found["session_id"],
                     "execution_generation": found["execution_generation"]}
            try:
                reply = await client.rpc(f"session.{verb}", session_id=found["session_id"],
                                         execution_generation=found["execution_generation"], text=text)
            except GatewayRPCError as exc:
                if _ambiguous(str(exc)):
                    # The owner kept the text and could not say whether the turn took it: a resend is a
                    # harmless duplicate, a refusal verdict would stop callers from sending it.
                    return 4, {"status": "ambiguous", "reason": str(exc), "stage": verb, **where,
                               "note": "ambiguous; text retained; resend allowed"}
                return 3, {"status": "refused", "stage": verb, "reason": str(exc), **where}
    except GatewayClientError as exc:
        return 4, {"status": "unreachable", "reason": str(exc)}
    if reply.get("kept") and reply.get("delivered") is False:
        # The turn could not take it, so the owner kept it on the card's task for the next attempt: durable,
        # not delivered. The owner's own status for that is ``rejected``; the verdict is about retention.
        return 0, {"status": "kept", "note": "kept on card, not delivered", **where, "result": reply}
    status = reply.get("status", "kept" if reply.get("kept") else "delivered")
    return (3 if status == "rejected" else 0), {"status": status, **where, "result": reply}


def cmd_redirect(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    text = " ".join(args.text).strip()
    if not text:
        _print({"status": "refused", "reason": "empty correction"})
        return 3
    board = kb.get_current_board()
    run_id = args.run
    if run_id is None:
        with connect_closing() as conn:
            task = kb.get_task(conn, args.task_id)
        if task is None or task.status != "running" or task.current_run_id is None:
            _print({"status": "refused", "reason": "stale_kanban_claim", "task_id": args.task_id})
            return 3
        run_id = task.current_run_id
    code, payload = asyncio.run(_send(args.task_id, board, run_id, args.kanban_action, text))
    _print(payload)
    return code
