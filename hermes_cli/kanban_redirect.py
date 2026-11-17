"""``hermes kanban redirect|steer <task_id> <text>``: correct the LIVE attempt of a card.

Two owner calls on the caller's own identity, both through the gateway that owns the attempt:
``session.kanban_run`` (read-only, ``session:read``) binds the named run to its session and execution
generation, then ``session.redirect`` / ``session.steer`` (``session:control``) carries that generation to
the protected control point, which rechecks claim, run and generation. An attempt that ended, was
reclaimed or was replaced is reported stale and nothing is delivered to a successor run; a correction the
owner could not hand to a live agent is kept on the card's task and answered as kept. Sending the same
correction again may duplicate it, never lose it: this verb never waits on an earlier send.

Exit 0: delivered or kept. 3: refused (stale run, denied, rejected). 4: the owner could not be reached or
answered ambiguously (resend is safe).
"""
from __future__ import annotations

import argparse
import asyncio
import json


def _print(payload) -> None:
    print(json.dumps(payload, sort_keys=True))


async def _send(task_id: str, board: str, run_id: int, verb: str, text: str) -> tuple[int, dict]:
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, connect_gateway
    try:
        async with connect_gateway() as client:
            try:
                found = await client.rpc("session.kanban_run", board=board, task_id=task_id, run_id=run_id)
            except GatewayRPCError as exc:
                return 3, {"status": "refused", "stage": "discovery", "reason": str(exc)}
            try:
                reply = await client.rpc(f"session.{verb}", session_id=found["session_id"],
                                         execution_generation=found["execution_generation"], text=text)
            except GatewayRPCError as exc:
                return 3, {"status": "refused", "stage": verb, "reason": str(exc), "run_id": run_id,
                           "session_id": found["session_id"], "execution_generation": found["execution_generation"]}
    except GatewayClientError as exc:
        return 4, {"status": "unreachable", "reason": str(exc)}
    status = reply.get("status", "kept" if reply.get("kept") else "delivered")
    return (3 if status == "rejected" else 0), {
        "status": status, "run_id": run_id, "session_id": found["session_id"],
        "execution_generation": found["execution_generation"], "result": reply}


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
