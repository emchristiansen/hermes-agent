"""Trusted dispatcher claims enter the ordinary owner's durable admission queue."""
import asyncio
import contextlib
from contextlib import closing
from dataclasses import asdict, replace
import threading
import json
from pathlib import Path
import sqlite3

from hermes_state_runtime import RuntimeStoreError


def build_kanban_policy(connection, params, config, *, route_inputs=None):
    """The attempt's policy from ``config``; ``route_inputs`` are the profile's route inputs as
    stamped before ``config`` was read (``kanban_db_provider_wall.route_inputs``), kept only when
    unchanged after the policy is built, so a wall can name the route its attempt actually used."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import resolve_profile_env
    from gateway.session_policy import build_policy
    authority, actor = connection.authority, connection.actor
    if not connection.native_owner or 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (set(params) not in ({'board', 'task_id', 'run_id', 'claim_lock'}, {'board', 'task_id', 'run_id', 'claim_lock', 'db'})
            or any(not isinstance(params[k], str) or not params[k] for k in ('board', 'task_id', 'claim_lock'))
            or type(params['run_id']) is not int):
        raise RuntimeStoreError('invalid_params')
    try:
        board = kb._require_slug(params['board'])
        path = Path(params['db']).resolve(strict=True) if 'db' in params else kb.kanban_db_path(board=board).resolve()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            task = kb.get_task(conn, params['task_id'])
            run = conn.execute('SELECT * FROM task_runs WHERE id=?', (params['run_id'],)).fetchone()
            if (task is None or run is None or task.status != 'running' or run['status'] != 'running'
                    or task.current_run_id != params['run_id'] or run['task_id'] != task.id
                    or task.claim_lock != params['claim_lock'] or run['claim_lock'] != task.claim_lock):
                raise RuntimeStoreError('stale_kanban_claim')
            if Path(resolve_profile_env(task.assignee)).resolve() != Path(authority.profile_id).resolve():
                raise RuntimeStoreError('profile_mismatch')
            bound = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                 (task.id, task.current_run_id)).fetchone()
            if bound and json.loads(bound[0])['request_id'] != 'kanban:' + json.dumps([board, task.id, task.current_run_id], separators=(',', ':')):
                raise RuntimeStoreError('stale_kanban_claim')
            context = kb.build_worker_context(conn, task.id)
            skills = list(task.skills or ())
            if kb._retry_status_for_run(conn, task.id, task.current_run_id) == 'review':
                # Same rule as the dispatcher's review lane: the reviewer always loads sdlc-review.
                skills = list(dict.fromkeys([*skills, 'sdlc-review']))
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise RuntimeStoreError('invalid_kanban_claim') from exc
    cwd = task.workspace_path
    params_policy = {'source': 'cli', 'cwd': cwd}
    for name, value in (('model', task.model_override), ('provider', task.provider_override), ('reasoning', task.reasoning_effort)):
        if value:
            params_policy[name] = value
    private = {}
    policy = build_policy(params_policy, config, private_secrets=private)
    if policy.model is None:
        from gateway.run import _resolve_gateway_model
        policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    context = dict(params, workspace=cwd, branch=task.branch_name, tenant=task.tenant,
        profile=task.assignee, workspaces_root=str(kb.workspaces_root(board=board).resolve()),
        skills=skills, goal_mode=task.goal_mode, goal_max_turns=task.goal_max_turns,
        goal_text='\n\n'.join(p for p in (task.title, task.body) if p), context=context,
        max_runtime_seconds=task.max_runtime_seconds, accept_hooks=True)
    from hermes_cli.kanban_db_provider_wall import route_inputs as current_route_inputs
    context['route_inputs'] = (route_inputs if route_inputs is not None
                               and route_inputs == current_route_inputs(authority.profile_id) else None)
    context['db'] = str(path)
    request = json.loads(policy.request_json)
    request.update(source='kanban', board=board, task_id=task.id, run_id=task.current_run_id)
    return replace(policy, source='kanban', platform='cli', toolsets=tuple(sorted(set(policy.toolsets) | {'kanban'})),
                   request_json=json.dumps(request, sort_keys=True), kanban_json=json.dumps(context)), private


async def run_task(connection, params):
    if not connection.native_owner or 'session:create' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.run import _load_gateway_config
    from gateway.session_local import create_local_session
    from gateway.session_contract import Submission
    from gateway.session_local_recovery import local_identity
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    authority = connection.authority
    # Resolved off-loop up front: no await separates the saved-claim check from session
    # creation. The Kanban claim check and its marker share one off-loop transaction; a
    # concurrent retry of the same attempt meets that marker (same session) or a stale claim.
    owner_db = str(await asyncio.to_thread(Path(authority.db.db_path).resolve))
    # The dispatcher attempt is the durable request identity, including after the
    # task closes. Retry checks the saved claim, never reinterprets updated cards.
    request_id = 'kanban:' + json.dumps([params.get('board'), params.get('task_id'), params.get('run_id')], separators=(',', ':'))
    sid = local_identity(authority.profile_id, connection.actor.subject, request_id)
    saved = local_receipt(authority.db, sid) if authority.db.get_session(sid) is not None else None
    if saved:
        policy = restore_policy(saved['policy'])
        context = json.loads(policy.kanban_json or '{}')
        if (not connection.native_owner or set(params) not in ({'board', 'task_id', 'run_id', 'claim_lock'}, {'board', 'task_id', 'run_id', 'claim_lock', 'db'})
                or any(params[k] != context.get(k) for k in params)):
            raise RuntimeStoreError('permission_denied')
        from gateway.session_local_recovery import restore_local_session
        ref = restore_local_session(authority, sid)
    else:
        from hermes_cli.kanban_db_provider_wall import route_inputs
        captured = route_inputs(authority.profile_id)  # before the config read it vouches for
        policy, private = build_kanban_policy(connection, params, _load_gateway_config(), route_inputs=captured)
        ref = create_local_session(authority, connection.actor, {'request_id': request_id},
                                   trusted_policy=policy, trusted_secrets=private)
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    import os
    import psutil
    from hermes_cli.process_identity import worker_birth
    context = json.loads(policy.kanban_json)

    def admit_claim():
        # The Kanban DB is the cross-profile dispatch bus (busy_timeout up to 120 s): the claim
        # check and its marker share one write transaction, off the owner loop.
        with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
            # Either this exact attempt was already admitted to this session (retry after a
            # crash between session creation and admission), or the claim is still current
            # now; a reclaimed/closed task never reaches resume/submit.
            marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
                                  (params['task_id'], params['run_id'])).fetchone()
            if marker:
                if json.loads(marker[0])['session_id'] != ref.session_id:
                    raise RuntimeStoreError('stale_kanban_claim')
            else:
                task = kb.get_task(conn, params['task_id'])
                if not (task and task.status == 'running' and task.current_run_id == params['run_id'] and task.claim_lock == params['claim_lock']):
                    raise RuntimeStoreError('stale_kanban_claim')
                kb._append_event(conn, task.id, 'owner_admitted',
                    {'db': owner_db, 'session_id': ref.session_id,
                     'request_id': request_id, 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                    run_id=task.current_run_id)
    from gateway.session_runtime_workers import tracked_write
    await tracked_write(authority, admit_claim)
    await connection.resume(ref, {})
    # Finite like the `chat -q` worker it replaced: nobody can answer an approval or clarify in a
    # card (approvals.single_query_mode applies). Never unattended: that is `-z`'s auto-approval.
    # A retry of an attempt admitted before cards were finite resubmits its committed payload,
    # so the store returns the original receipt instead of admission_conflict.
    from hermes_state_runtime import find_session_admission
    committed = None if not saved else await asyncio.to_thread(find_session_admission, authority.db,
        principal_id=connection.actor.subject, session_id=ref.session_id, request_id=request_id)
    payload: dict = {'text': f'work kanban task {params["task_id"]}'}
    if committed is None or 'finite' in committed['payload']:
        payload['finite'] = True
    # The dispatcher's trusted producer marker (#134889 KB-8): nobody waits on a card attempt, so one
    # lost across an owner restart settles interrupted instead of pausing the card session. An
    # attempt main committed without it resubmits unmarked (same digest, original receipt).
    card = None
    if committed is None or 'kanban_card_v1' in committed['payload']:
        card = {'board': context.get('board'), 'task_id': params['task_id'], 'run_id': params['run_id']}
    receipt = await authority.submit(connection.actor, Submission(request_id, ref, payload, 'queue'),
                                     _kanban_card=card)
    return {'session_id': ref.session_id, 'receipt': asdict(receipt)}


def bind_worker_context(frame):
    """Only the adopted private exec calls this, before tool discovery imports."""
    import os
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return
    if frame['policy']['source'] != 'kanban':
        raise ValueError('invalid_kanban_policy')
    env = {'HERMES_KANBAN_TASK': context['task_id'], 'HERMES_KANBAN_BOARD': context['board'],
        'HERMES_KANBAN_DB': context['db'], 'HERMES_KANBAN_RUN_ID': str(context['run_id']),
        'HERMES_KANBAN_CLAIM_LOCK': context['claim_lock'], 'HERMES_KANBAN_WORKSPACE': context['workspace'],
        'HERMES_KANBAN_WORKSPACES_ROOT': context['workspaces_root'], 'HERMES_PROFILE': context['profile'],
        'HERMES_SESSION_SOURCE': 'kanban'}
    for key, value in (('HERMES_KANBAN_BRANCH', context['branch']), ('HERMES_TENANT', context['tenant'])):
        if value:
            env[key] = value
    if context['goal_mode']:
        env['HERMES_KANBAN_GOAL_MODE'] = '1'
    from hermes_cli.kanban_db_dispatch import _worker_terminal_timeout_env
    for key in ('TERMINAL_TIMEOUT', 'TERMINAL_MAX_FOREGROUND_TIMEOUT'):
        value = _worker_terminal_timeout_env(context['max_runtime_seconds'], os.environ.get(key))
        if value:
            env[key] = value
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    with connect_closing(Path(context['db'])) as conn:
        with kb.write_txn(conn):
            task = kb.get_task(conn, context['task_id'])
            if (task is None or task.status != 'running' or task.current_run_id != context['run_id']
                    or task.claim_lock != context['claim_lock']):
                raise RuntimeStoreError('stale_kanban_claim')
            # The dispatcher's spawned worker stays the run's pid (reclaim, timeout and operator
            # kills target it; ``_watch_submitter`` ties this interpreter's life to it). The bound
            # interpreter is recorded beside it so a sweep can tell when the attempt is truly gone.
            from hermes_cli.kanban_db_dispatch import UNVERIFIED_WORKER_FINGERPRINT, _process_fingerprint
            fingerprint = _process_fingerprint(os.getpid()) or UNVERIFIED_WORKER_FINGERPRINT
            kb._append_event(conn, task.id, 'worker_bound', {'pid': os.getpid(), 'claim_lock': context['claim_lock'],
                             'started_at': fingerprint}, run_id=context['run_id'])
            claimed = conn.execute("SELECT payload, created_at FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' ORDER BY id DESC LIMIT 1",
                                   (task.id, context['run_id'])).fetchone()
    # Heartbeats extend the claim by the TTL the dispatcher claimed it with (a spawned worker
    # inherited it from the dispatcher's environment); this daemon's own default is not that.
    expires = json.loads(claimed[0]).get('expires') if claimed else None
    if isinstance(expires, int) and claimed[1]:
        env['HERMES_KANBAN_CLAIM_TTL_SECONDS'] = str(max(1, expires - int(claimed[1])))
    _watch_submitter(Path(context['db']), context['run_id'])
    os.environ.update(env)
    os.chdir(context['workspace'])


_TURNS_DONE = threading.Event()


def _watch_submitter(db, run_id):
    """The dispatcher's spawned worker is this attempt's handle: killing it (an operator, a
    timeout, an OOM kill) must end the attempt, as it did when the worker ran the agent itself.
    Exiting hard leaves no ``worker_result``, so the dead-worker sweep books this run's crash.
    The run row is re-read each tick: the dispatcher records the pid only after its spawn returns."""
    import os
    from hermes_cli.kanban_db_dispatch import _worker_alive

    def watch():
        with closing(sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            while not _TURNS_DONE.wait(1.0):
                try:
                    row = conn.execute('SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?', (run_id,)).fetchone()
                except sqlite3.Error:
                    continue
                if row and row[0] and row[0] != os.getpid() and not _worker_alive(row[0], row[1]):
                    # Exiting hard prints nothing: say why on stderr (the owner drains it into
                    # managed-worker.log), or the owner's EOF reads as an unexplained death.
                    try:
                        _note_on_stderr(f'kanban run {run_id}: submitter pid {row[0]} exited; ending this bound worker\n')
                    finally:
                        os._exit(1)  # whatever the note did
    threading.Thread(target=watch, name='kanban-submitter-watch', daemon=True).start()


def _note_on_stderr(text):
    """Write ``text`` to fd 2, waiting at most a second: a full pipe nobody drains must not hold
    back the exit that follows."""
    import os

    def write():
        with contextlib.suppress(OSError):
            os.write(2, text.encode())
    writer = threading.Thread(target=write, name='kanban-submitter-note', daemon=True)
    writer.start()
    writer.join(1)


def run_worker_turns(agent, frame, history, settle=None, finalize=None):
    """``settle`` (the child's ``WorkerControls.settle``) hands back steer text a turn could not
    drain, run as a follow-up turn of this admission (``_follow_up``), and closes correction intake.
    ``finalize`` (default: none) makes the turns' result the one the owner will hear (the managed
    child's final usage flush) before a card attempt books its exit from it."""
    from gateway.session_managed_worker import worker_turn_scope
    from gateway.session_surface import arm_surface_turn
    with worker_turn_scope(frame):
        # The agent consumes explicit one-turn attributes, not the ContextVar: arm them exactly as
        # the in-process runner does (voice route marker + surface note on the user message).
        arm_surface_turn(agent)
        return _run_worker_turns(agent, frame, history, settle, finalize or (lambda result: result))


def _follow_up(agent, result, settle, may_continue=lambda: True, **kwargs):
    """A steer that landed after a turn's final model call has no tool batch left to drain into. In
    process it becomes the next user turn (``run_busy._merge_leftover_steer``); here that turn runs
    inside this admission, whose worker and claim end with it, rather than leaving the text in a
    ``pending_steer`` the result frame does not carry. Text no turn may run here stays kept by the
    owner (``settle``), which received it durably before sending it.

    The admission's result is the last turn's, except ``user_intervened``: a follow-up turn takes the
    steered text as its own user message and so starts untouched, yet the admission did not run on
    its own input alone, and its final is not attributable to the original prompt."""
    followed = False
    while settle is not None and (text := settle(result, may_continue=may_continue())):
        result = agent.run_conversation(
            text, conversation_history=agent._session_db.get_messages_as_conversation(agent.session_id), **kwargs)
        followed = True
    if followed and isinstance(result, dict):
        result['user_intervened'] = True
    return result


@contextlib.contextmanager
def _worker_workspace(frame):
    """A managed worker's chat turn in its session's workspace, as the in-process turn runner binds it
    (#140665): the session id as the terminal task id, the route as the session key (persistent Docker
    keys the workspace container from it) and the frozen cwd as the session's mount. Yields the task
    id, or None for a frame with no session scope (nothing is bound then)."""
    session_id = (frame.get('scope') or {}).get('session_id')
    if not session_id:
        yield None
        return
    from gateway.session_authorities import served_profile_name
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import bind_turn_workspace, restore_policy
    from hermes_state_keys import profile_from_session_key
    policy = restore_policy(frame['policy'])
    route = frame.get('route') or session_id
    # The profile its in-process turns bind, so persistent Docker keys the container they and the
    # session's side agents use (#140665): a served secondary's name, which its routes carry
    # (``agent:<name>:``); none for the launch home, whatever its name (``hermes -p beta gateway``
    # keys ``agent:main:``, and only the default profile runs the multiplexer).
    profile = served_profile_name(frame.get('home'))
    tokens = set_session_vars(session_key=route, session_id=session_id,
                              profile=profile if profile_from_session_key(route) == profile else '')
    try:
        bind_turn_workspace(policy, session_id)
        yield session_id
    finally:
        clear_session_vars(tokens)


def _run_worker_turns(agent, frame, history, settle, finalize):
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        from gateway.session_display import worker_display_kwargs
        author = frame.get('turn_author')
        with _worker_workspace(frame) as task_id:
            kwargs = {**worker_display_kwargs(), **({'task_id': task_id} if task_id else {})}
            result = agent.run_conversation(frame['text'], conversation_history=history, **kwargs,
                                            **({'turn_author': author} if author is not None else {}))
            result = _follow_up(agent, result, settle, **kwargs)
        return finalize(result)
    from hermes_cli.kanban_db_provider_wall import wall_from_exception, wall_from_result
    from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code
    code, last_output, wall = 1, '', None
    try:
        # A follow-up runs only under this attempt's still-live claim, before the exit is booked,
        # and the exit is booked from the finalized result: a failed final flush books that failure.
        result = finalize(_follow_up(agent, _run_task_turns(agent, frame, history, context), settle,
                                     may_continue=lambda: _bound_claim_is_live(context)))
        code = turn_exit_code(result, kanban_worker=True)
        if isinstance(result, dict):
            # Redact before the tail cut: a cut through a PEM header or a token prefix
            # leaves a body the redactor no longer recognises.
            from hermes_cli import kanban_db as kb
            last_output = kb.redact_review_value(str(result.get('final_response') or ''))[-500:]
        wall = wall_from_result(result, agent)
        return result
    except Exception as exc:
        code = turn_exit_code(None, kanban_worker=True, **credential_failure_flags(exc))
        wall = wall_from_exception(exc, frame)
        raise
    finally:
        _record_worker_result(context, code, last_output, wall)


def record_start_failure(frame, exc):
    """The worker died before any turn ran (agent construction resolves credentials): book the
    one-shot CLI's credential exit (75 quota wall / 78 re-login / 1) instead of a bare crash."""
    context = json.loads(frame['policy'].get('kanban_json') or 'null')
    if context is None:
        return
    from hermes_cli.kanban_db_provider_wall import wall_from_exception
    from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code
    _record_worker_result(context, turn_exit_code(None, kanban_worker=True, **credential_failure_flags(exc)), '',
                          wall_from_exception(exc, frame))


def _record_worker_result(context, code, last_output, wall=None):
    _TURNS_DONE.set()
    import os
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    with connect_closing(Path(context['db'])) as conn:
        with kb.write_txn(conn):
            if _bound_worker_matches(conn, context):
                # Closing a run clears its claim/PID and replaces metadata; keep the
                # result in the immutable attempt event stream instead.
                # The provider wall a requeued attempt met is the evidence its card's retries are judged by.
                kb._append_event(conn, context['task_id'], 'worker_result',
                    {'pid': os.getpid(), 'claim_lock': context['claim_lock'], 'exit_code': code,
                     'last_output': last_output,
                     **({'provider_wall': wall.payload()} if wall is not None and code == kb.KANBAN_RATE_LIMIT_EXIT_CODE else {})},
                    run_id=context['run_id'])
        _remove_finished_workspace(conn, context)


def _remove_finished_workspace(conn, context):
    """Workspace cleanup never removes a directory while a live bound worker works at or under it
    (``_workspace_in_use_by_other``), so a done or archived card's cleanup, and the parent sweep it
    runs, was held back while this worker's turn went on. This worker's ``worker_result`` is
    recorded now, so it no longer counts as live: step out of any directory the cleanup may
    remove and run it again. Directories another live worker still uses stay until that worker's
    own end runs this."""
    import os
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, context['task_id'])
    if task is None or task.status not in ('done', 'archived'):
        return
    os.chdir(Path(context['db']).resolve().parent)  # the board directory, never removed by cleanup
    kb._cleanup_workspace(conn, context['task_id'])


def _unbound_attempt_claim(conn, task_id, run_id, session_id):
    """The claim of run ``run_id`` when it is still the task's running attempt and was admitted to
    ``session_id`` (its ``owner_admitted`` marker), else None. For an attempt whose interpreter
    never bound: no ``worker_bound`` claim exists to fence by, so the run's own claim and admission
    marker fence it instead."""
    if not isinstance(session_id, str):
        return None
    run = conn.execute("SELECT r.claim_lock FROM tasks t JOIN task_runs r ON r.id = t.current_run_id "
                       "WHERE t.id=? AND t.current_run_id=? AND t.status='running' AND r.status='running' "
                       "AND r.claim_lock = t.claim_lock", (task_id, run_id)).fetchone()
    marker = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' "
                          "ORDER BY id DESC LIMIT 1", (task_id, run_id)).fetchone()
    try:
        admitted = json.loads(marker[0]).get('session_id') if marker else None
    except (ValueError, TypeError, AttributeError):
        admitted = None
    return run[0] if run and run[0] and admitted == session_id else None


def record_owner_restart_result(card, board_db, session_id=None) -> bool:
    """The profile owner ended under a card attempt: record that as the run's result.

    Called only where the owner itself settles the attempt's lost turn ``interrupted`` because it
    is stopping or restarting (``session_unattended_unknown``): a drain that fenced the lost worker
    while this owner was ending, or the next owner's boot recovery of a turn its predecessor left in
    flight. A turn that ended any other way (its interpreter killed under a live owner, a cancel)
    was terminal before that and never reaches here, so it keeps its ordinary crash booking.

    Writes the ``worker_result`` the bound interpreter would have, under the bound pid and claim, with
    ``KANBAN_OWNER_RESTART_EXIT_CODE``: the submitter returns it and the dispatcher requeues without
    counting a failure. Only for the task's CURRENT run under its live claim, and never over a result
    the interpreter already recorded. ``board_db``: the board file the attempt's session policy names
    (``card_board_db``). False when nothing was recorded.

    An attempt the owner admitted but whose interpreter never bound (the owner ended between its
    ``owner_admitted`` marker and ``worker_bound``) was lost to the same restart. With
    ``session_id``, the admission's session, it is recorded under the run's own claim and the
    dispatcher's spawned worker pid, fenced by the run's marker naming that session
    (``_unbound_attempt_claim``); the crash sweep books it as it books a bound one
    (``kanban_owner_recovery.owner_restart_accounted``). The dispatcher registers that pid after
    the spawn, so it can still be NULL here: the record then carries no pid and speaks for the pid
    the run registers later (``kanban_owner_recovery.managed_run_exit_code``).

    This is the attempt's accounting, not the end of the interpreter's use of its workspace: the
    interpreter may still be running (a tool in flight outlives owner-pipe EOF by a grace period).
    ``OWNER_RECORDED_RESULT_KEY`` marks the result as the owner's, so workspace cleanup keeps
    honouring the binding until the interpreter records its own result or is gone."""
    import os
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_retry_policy import KANBAN_OWNER_RESTART_EXIT_CODE
    from hermes_cli.kanban_db_workspace import OWNER_RECORDED_RESULT_KEY
    if not isinstance(card, dict) or not isinstance(card.get('task_id'), str) or type(card.get('run_id')) is not int:
        return False
    path = Path(board_db)
    if not path.is_file():
        return False
    task_id, run_id = card['task_id'], card['run_id']
    with connect_closing(path) as conn, kb.write_txn(conn):
        row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' "
                           "ORDER BY id DESC LIMIT 1", (task_id, run_id)).fetchone()
        bound = json.loads(row[0]) if row else {}
        task = conn.execute('SELECT current_run_id, claim_lock, worker_pid FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is not None:
            claim, pid = bound.get('claim_lock'), bound.get('pid')
        else:
            claim, pid = _unbound_attempt_claim(conn, task_id, run_id, session_id), task[2] if task else None
        if (not claim or task is None or task[0] != run_id or task[1] != claim
                or conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result'",
                                (task_id, run_id)).fetchone()):
            return False
        kb._append_event(conn, task_id, 'worker_result',
            {'pid': pid, 'claim_lock': claim, 'exit_code': KANBAN_OWNER_RESTART_EXIT_CODE,
             'last_output': '', OWNER_RECORDED_RESULT_KEY: os.getpid()}, run_id=run_id)
    return True


def card_board_db(conn, row):
    """The board file of the card attempt ``row`` admitted, from its session's frozen policy (the same
    ``kanban_json`` its bound interpreter records into); None unless that policy is this attempt's."""
    from hermes_state_local import POLICY_PREFIX
    card = (row.get('payload') or {}).get('kanban_card_v1') or {}
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + row['target_session_id'],)).fetchone()
    try:
        kanban = json.loads(json.loads(saved[0])['policy'].get('kanban_json') or 'null') if saved else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if (not isinstance(kanban, dict) or kanban.get('task_id') != card.get('task_id')
            or kanban.get('run_id') != card.get('run_id') or not isinstance(kanban.get('db'), str)):
        return None
    return kanban['db']


def record_worker_loss(policy, reason, *, pid=None, bound=False):
    """Append a run-scoped ``worker_lost`` event for this attempt to the card's board.

    Dropped when the card has moved past the run (it says nothing about the current attempt).
    Never raises: the caller's own loss handling must not depend on the board write.
    """
    try:
        context = json.loads(policy.kanban_json or 'null')
        if not context:
            return
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli import kanban_db as kb
        run_id = int(context['run_id'])
        with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
            row = conn.execute('SELECT status, current_run_id FROM tasks WHERE id=?',
                               (context['task_id'],)).fetchone()
            if row is None or row[0] != 'running' or row[1] != run_id:
                return
            kb._append_event(conn, context['task_id'], 'worker_lost',
                             {'reason': str(reason)[:200], 'claim_lock': context['claim_lock'],
                              'run_id': run_id, 'pid': pid, 'bound': bool(bound)},
                             run_id=run_id)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('kanban worker_lost record failed')


def _bound_worker_matches(conn, context):
    import os
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_bound' ORDER BY id DESC LIMIT 1",
        (context['task_id'], context['run_id'])).fetchone()
    bound = json.loads(row[0]) if row else {}
    return (bound.get('pid'), bound.get('claim_lock')) == (os.getpid(), context['claim_lock'])


def keep_on_board(context, text, *, ended=False):
    """A correction for a running card attempt (``context``: its ``kanban_json``), kept as a comment
    on its task BEFORE the owner sends it to the attempt's worker. Whatever happens to that worker
    (it runs the text, its claim ends first, it crashes), ``build_worker_context`` hands the comment
    to the task's next attempt whole, however long or old (its ``KEPT_CORRECTION_EVENT``, written in
    the same transaction, exempts it from the comment caps), and an operator reads it. Never a model
    turn or a claim write. The
    author is this profile's runtime identity, as for every worker comment (#19713). The running
    worker's board-note injection skips only comments its own process wrote, so it may steer this one
    in as well: a duplicate delivery, accepted, never a substitute for the next attempt's copy.
    ``ended``: the turn the correction named had already ended when the owner received it, so it is
    sent to no run; the caption says so."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_cli.kanban_db_context import KEPT_CORRECTION_EVENT
    from tools.kanban_tools import _persisted_identity
    body = (f"Correction for run {context['run_id']} received after the turn it named had ended. Kept "
            f"here; it was not sent to that turn:\n{text}" if ended else
            f"Correction accepted for run {context['run_id']} while it was running. Kept here before "
            f"it was sent to that run; whether that run applied it is not recorded here:\n{text}")
    with connect_closing(Path(context['db'])) as conn:
        with kb.write_txn(conn):
            comment_id = kb.add_comment(conn, context['task_id'], author=_persisted_identity(), body=body)
            kb._append_event(conn, context['task_id'], KEPT_CORRECTION_EVENT, {'comment_id': comment_id},
                             run_id=context['run_id'])


def _bound_claim_is_live(context):
    """Retry proof for a managed worker: the frame's exact run/claim lease is live and unexpired,
    and THIS interpreter is the one ``bind_worker_context`` bound to it. The run's worker_pid is
    the dispatcher's spawned submitter here, so pid identity comes from ``worker_bound`` instead."""
    from agent.kanban_turn_recovery import worker_claim_is_live
    if not worker_claim_is_live(task_id=context['task_id'], db_path=context['db'], run_id=str(context['run_id']),
                                claim_lock=context['claim_lock'], verify_pid=False):
        return False
    try:
        with closing(sqlite3.connect(Path(context['db']).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            return _bound_worker_matches(conn, context)
    except sqlite3.Error:
        return False


def steer_peer_if_bound(context, steer):
    """A managed worker's private peer message: ``steer()`` (an in-memory buffer append that writes
    no board) runs only while this interpreter's bound claim (``context``: its ``peer_destination``)
    is the card's live current run, and the board's write lock is held across that check and the
    append. A reclaim, reassignment, completion or expiry sweep is a board write, so it commits either
    before the check (nothing is buffered) or after the text is buffered, never between. The check is
    the canonical ``worker_claim_is_live`` (exact run and claim lock, open run, unexpired lease) plus
    this interpreter's own ``worker_bound`` event. The caller holds its intake lock around this call,
    and nothing holding the board waits for that lock. True when ``steer()`` took the text."""
    from agent.kanban_turn_recovery import worker_claim_is_live
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    with connect_closing(Path(context['db'])) as conn, kb.write_txn(conn):
        if not (worker_claim_is_live(task_id=context['task_id'], db_path=context['db'], run_id=str(context['run_id']),
                                     claim_lock=context['claim_lock'], verify_pid=False)
                and _bound_worker_matches(conn, context)):
            return False
        return bool(steer())


def worker_result(path, params):
    """Read only the dispatcher's exact attempt; never infer success from admission settlement.
    The attempt's accounting result (``kanban_owner_recovery.attempt_result``): an owner's
    restart record outranks the interpreter's later result for the same run."""
    from hermes_cli.kanban_owner_recovery import attempt_result
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        result = attempt_result(conn, params['task_id'], params['run_id'])
        return result if result.get('claim_lock') == params['claim_lock'] else {}


def worker_exit_code(path, params):
    return worker_result(path, params).get('exit_code', 1)


# Who wrote a card attempt's prompts: the dispatcher, never the operator, so memory providers do not
# file "work kanban task t_..." and the worker's progress notes as the human's own turns (#137406).
# Not a ``bot:<profile>`` id: a Bot Mode profile named ``kanban`` would share its memory key.
KANBAN_TURN_AUTHOR = {'id': 'kanban:dispatcher', 'name': 'Kanban dispatcher', 'is_bot': True}


def _run_task_turns(agent, frame, history, context):
    from agent.skill_commands import build_preloaded_skills_prompt
    skills, loaded, missing = build_preloaded_skills_prompt(context['skills'], task_id=agent.session_id)
    if missing and not loaded:
        raise ValueError('kanban_skills_unavailable')
    prompt = '\n\n'.join(p for p in (skills, context['context'], frame['text']) if p)
    result = agent.run_conversation(prompt, conversation_history=history, turn_author=KANBAN_TURN_AUTHOR)

    def recover(nudge):
        nonlocal result
        prior = result.get('messages') if isinstance(result, dict) else None
        result = agent.run_conversation(nudge, conversation_history=prior or
                                        agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR)
    # Same in-place retry as the one-shot CLI worker (agent/kanban_turn_recovery.py), but the
    # carrier is this frame's bound claim, never the owner process's environment.
    from agent.kanban_turn_recovery import recover_failed_kanban_turns
    recover_failed_kanban_turns(recover, lambda: result, task_id=context['task_id'],
                                claim_check=lambda: _bound_claim_is_live(context))
    from hermes_cli.turn_exit import turn_exit_code
    # Only a settled turn may continue the goal loop: an exhausted or denied recovery reaches
    # the honest exit path without re-entering the model under an unprovable claim.
    if not context['goal_mode'] or turn_exit_code(result, kanban_worker=True) != 0:
        return result
    from hermes_cli.goals import run_kanban_goal_loop, DEFAULT_MAX_TURNS
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    def status():
        with connect_closing() as conn:
            return kb.goal_run_status(conn, context['task_id'], context['run_id'])
    def block(reason):
        with connect_closing() as conn:
            kb.block_task(conn, context['task_id'], reason=reason, expected_run_id=context['run_id'])
    def turn(prompt):
        nonlocal result
        result = agent.run_conversation(prompt, conversation_history=agent._session_db.get_messages_as_conversation(agent.session_id),
                                        turn_author=KANBAN_TURN_AUTHOR)
        return result.get('final_response', '')
    run_kanban_goal_loop(task_id=context['task_id'], goal_text=context['goal_text'], run_turn=turn,
        task_status_fn=status, block_fn=block, max_turns=context['goal_max_turns'] or DEFAULT_MAX_TURNS,
        first_response=result.get('final_response', ''))
    return result
