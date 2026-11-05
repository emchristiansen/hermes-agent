"""Every message reaching a busy session steers into its running turn.

Eric (2026-11-04): "I want literally every message to always be a steering message." A message
admitted while a claim runs is transferred to that claim's agent through the existing queued-steer
primitive (``session_queued_controls.transfer``): consumed at the turn's next tool/result boundary,
settled ``steered``; one no boundary read goes back to its own queue place and runs as its own turn.
An idle session still runs the message as its own turn at once (the drain is unchanged).

The pass runs after each admission while a claim executes, and when the claim adopts its agent (a
message admitted before that has no agent to reach yet). Messages are interactive plain text (what
``prompt.steer_queued`` already offers) and local automation text: Kanban and process notices, liveness
ticks and Bot Chat deliveries, including the Desktop relay's, and a messaging route's wakes; these reach
the turn framed with their sender, never as the user's text. A row whose route no longer belongs to this session is left to the
drain, which refuses it as before, and nothing steers while the runtime drains (a restart's queued
input waits for the next owner, as the drain makes it). Whole jobs keep their own turn:
attachments, finite/cron/Kanban/API/A2A/hosted runs and hidden prompts, whose producers read that
turn's own result. A messaging user's own text steers through the messaging busy path, before admission.
"""
import asyncio
import logging

from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

logger = logging.getLogger(__name__)


async def is_message(authority, ref, row):
    """True when the queued ``row`` is a message to steer, False when it is a whole job."""
    from gateway.session_queued_controls import automation, queued_controls
    if 'steer' in queued_controls(authority, row):
        return True
    payload = row['payload']
    text = payload.get('text')
    if row['status'] != 'queued' or automation(payload) is None or not isinstance(text, str) or not text.strip():
        return False
    try:  # the drain's own route check for the row: never misrouted
        if 'local_automation_v1' in payload:
            from gateway.session_automation import check_local_automation
            check_local_automation(authority, ref, row)
        else:
            from gateway.session_envelope import check_native_route
            live = authority.sessions[ref.session_id]
            await check_native_route(authority.runner, payload, authority.physical_target(ref), live.source,
                                     authority.runner._adapter_for_source(live.source))
    except RuntimeStoreError:
        return False
    return True


def steer_soon(authority, ref):
    """Run the pass for ``ref``'s session on the owner loop, one at a time: a request while a pass
    runs makes that pass read the queue again before it exits."""
    live = authority.sessions.get(ref.session_id)
    if live is None:
        return
    if live.steering is not None and not live.steering.done():
        live.steer_again = True
        return
    live.steering = asyncio.get_running_loop().create_task(_pass(authority, ref, live))
    live.steering.add_done_callback(_log_failure)


def _log_failure(task):
    if not task.cancelled() and task.exception() is not None:
        logger.error('busy steering pass failed', exc_info=task.exception())


async def _pass(authority, ref, live):
    """Transfer each queued message the running claim's agent can take, in admission order."""
    from gateway.session_queued_controls import _receiving_agent, transfer
    while True:
        live.steer_again = False
        generation = authority.executing.get(ref.session_id)
        with live.event_stream.lock:
            agent = None if generation is None else _receiving_agent(authority, ref.session_id, generation)
        if agent is None or ref.session_id in getattr(authority, '_managed_workers', {}):
            return  # idle (the drain runs them), no agent yet (adoption runs this again), or a managed worker
        rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
        for row in rows:
            if row['status'] != 'queued' or not await is_message(authority, ref, row):
                continue
            if authority.runner._draining or authority.retiring:
                # The drain's own gate, decided at each transfer with no await before ``transfer`` registers
                # it: once a restart drain began (whose queue-only input exists only from then), what is
                # queued waits for the next owner, as the drain makes it.
                return
            try:
                await transfer(authority, ref, row, generation)
            except RuntimeStoreError as exc:
                if exc.reason == 'admission_started':
                    continue  # the drain claimed it first: it runs as its own turn
                if exc.reason in {'session_idle', 'stale_generation', 'steer_unavailable'}:
                    return  # the claim ended (or has no agent): the drain runs what is left
                raise
        if not live.steer_again:
            return


def admitted(authority, ref):
    """After an admission (on the owner loop): steer it if a claim is running."""
    authority._steer_loop = asyncio.get_running_loop()
    if authority.executing.get(ref.session_id) is not None:
        steer_soon(authority, ref)


def adopted(authority, session_id):
    """From ``SessionAuthority.adopt_agent`` (the turn's thread): run the pass for the claim's new agent."""
    loop = getattr(authority, '_steer_loop', None)
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(steer_soon, authority, SessionRef(authority.profile_id, session_id))
