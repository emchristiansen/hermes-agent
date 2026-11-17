"""Caller -> running finite Kanban session discovery: ``session.kanban_run``.

A caller that wants to steer or redirect the live attempt of a card names the run it means (board, task,
run id). The owner answers with that attempt's session and the execution generation that attempt is
running, so the caller can carry both into the existing ``session.steer`` / ``session.redirect``
(``session:control``), whose protected control point rechecks claim, run and generation.

Read-only: nothing is attached, resumed, restored or written. Internal finite sessions stay out of
``session.active_list`` and the pickers; this exact lookup is the only route to one, and it binds only
a claim that is current now:

- the task is ``running`` and ``current_run_id`` is the named run (a run that ended, was reclaimed or was
  replaced by a newer run is ``stale_kanban_claim``, never rebound to its successor);
- the run's ``owner_admitted`` marker names this owner's database and a session that is live here;
- that session's saved policy is the same board card and run (``card_context``);
- a started claim exists on it, and its generation is the one returned.

The caller must hold ``session:read`` and pass ``authorize`` for the discovered session under the caller's
own identity: no credential is read or minted here.
"""
import asyncio
import json
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def _read_claim(authority, params):
    """Worker thread: ``(session_id, db_path)`` of the run's owner admission iff the card's claim is current."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    db_path = kb.kanban_db_path(board=params['board'])
    with connect_closing(db_path) as conn:
        task = kb.get_task(conn, params['task_id'])
        if not (task and task.status == 'running' and task.current_run_id == params['run_id']):
            raise RuntimeStoreError('stale_kanban_claim')
        marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' "
                              "ORDER BY id DESC LIMIT 1", (params['task_id'], params['run_id'])).fetchone()
    if marker is None:
        raise RuntimeStoreError('not_found')
    payload = json.loads(marker[0])
    if not isinstance(payload.get('session_id'), str) or Path(str(payload.get('db'))).resolve() != Path(
            authority.db.db_path).resolve():
        raise RuntimeStoreError('not_found')  # admitted by another owner: not this endpoint's session
    return payload['session_id'], str(db_path)


async def kanban_run(connection, ref, params):
    required = {'board': str, 'task_id': str, 'run_id': int}
    if (set(params) - set(required) - {'profile'}
            or any(type(params.get(key)) is not kind for key, kind in required.items())):
        raise RuntimeStoreError('invalid_params')
    if 'session:read' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    authority = connection.authority
    if params.get('profile') is not None:
        from hermes_cli.profiles import profile_matches_home
        if not isinstance(params['profile'], str):
            raise RuntimeStoreError('invalid_params')
        if params['profile'] and not profile_matches_home(params['profile'], Path(authority.profile_id)):
            raise RuntimeStoreError('profile_mismatch')
    session_id, db_path = await asyncio.to_thread(_read_claim, authority, params)
    from gateway.session_contract import SessionRef
    from gateway.session_retirement import LISTING_READ
    ref = SessionRef(authority.profile_id, session_id)
    token = LISTING_READ.set(True)  # a read never restores a parked session
    try:
        authority.authorize(connection.actor, ref, 'session:read')
    finally:
        LISTING_READ.reset(token)
    live = authority.sessions.get(session_id)
    if live is None:
        raise RuntimeStoreError('stale_kanban_claim')
    with live.event_stream.lock:
        handle = authority._handle(ref)
    if handle.execution_state != 'running':
        raise RuntimeStoreError('stale_kanban_claim')
    from gateway.session_managed_worker import card_context
    card = await asyncio.to_thread(card_context, authority, session_id, handle.execution_generation)
    if (card is None or card.get('task_id') != params['task_id'] or card.get('run_id') != params['run_id']
            or Path(card['db']).resolve() != Path(db_path).resolve()):
        raise RuntimeStoreError('stale_kanban_claim')
    return {'board': params['board'], 'task_id': params['task_id'], 'run_id': params['run_id'],
            'session_id': session_id, 'execution_generation': handle.execution_generation,
            'owner': {'profile_id': authority.profile_id, 'instance_id': authority.instance_id,
                      'epoch': authority.epoch}}
