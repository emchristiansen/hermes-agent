"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_claim_freezes_task_policy_and_rejects_forgery(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')  # a code worktree
    from contextlib import closing
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned acceptance', body='Goal criteria', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True,
            goal_max_turns=3, model_override='loop-model', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']},
              'agent': {'coding_context': 'focus'}}  # the worktree must not collapse a worker into the coding posture
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'}, {'cwd': '/tmp'}):
        with pytest.raises(RuntimeStoreError):
            build_kanban_policy(connection, params | changed, config)
    connection.native_owner = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        build_kanban_policy(connection, params, config)
    from gateway.session_policy import build_policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'kanban', 'cwd': str(workspace)}, config)


@pytest.mark.parametrize('mode', ['complete', 'timeout', 'rate_limit', 'billing', 'crash', 'redirect', 'flush_failed',
                                  'review_continue', 'review_continue_compressed',
                                  'review_continue_first_turn_compaction'])
def test_real_dispatcher_lifecycle(tmp_path, mode):
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    path = str(repo)
    if mode == 'flush_failed':  # the managed child's final usage flush fails once
        from tests.gateway.test_managed_worker_correction_durability import _failing_store_site
        path = os.pathsep.join((str(_failing_store_site(tmp_path)), path))
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=path, KANBAN_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    if mode == 'flush_failed':
        # The attempt's turn ran, but its final usage flush failed: the owner settles it failed, and
        # the exit the attempt booked and its spawned worker's exit say so, never a clean success.
        assert receipt['turn_outcomes'] == [['terminal', 'failed']], receipt
        assert receipt['booked_exit'] == 1 and receipt['client_exit'] == ['nonzero_exit', 1], receipt
        return
    if mode == 'review_continue_first_turn_compaction':
        a, b = receipt['card_a'], receipt['card_b']
        assert receipt['b_seed']['outcome'] == 'inherited', receipt
        # B's first request is unanchored, so preflight defers; after B's verdict call the post-tool
        # preflight compacts B's session in place, summarising inherited rows, inside B's first turn ...
        shapes = receipt['b_request_shapes']
        assert shapes[0]['tools'] and shapes[0]['sentinel'] and not shapes[0]['compaction'], receipt
        assert receipt['b_summary_requests'] >= 1 and receipt['b_active_compaction_rows'] == 1, receipt
        # ... and the turn still runs normally: the post-compaction request keeps the note naming B's
        # current assignment, B's verdict is B's alone, and A's old verdict call never re-runs.
        assert shapes[-1]['tools'] and shapes[-1]['compaction'] and shapes[-1]['note'], receipt
        assert receipt['b_last_tool_request_note'] and receipt['b_last_tool_request_names_current'], receipt
        assert receipt['statuses'] == {a: 'done', b: 'done'}, receipt
        assert receipt['b_tool_calls'] == 1 and receipt['completed_events'] == {a: 1, b: 1}, receipt
        assert [r['outcome'] for r in receipt['runs'][b]][-1] == 'completed', receipt
        return
    if mode.startswith('review_continue'):
        a, b = receipt['card_a'], receipt['card_b']
        assert receipt['statuses'] == {a: 'done', b: 'done'}, receipt
        # The second reviewer's FIRST request carries the first review's conversation and the note.
        assert receipt['b_first_has_sentinel'] and receipt['b_first_has_a_verdict_call'], receipt
        assert receipt['b_first_has_note'] and receipt['b_first_names_current'], receipt
        # A's old verdict call is history only: one hook-observed tool call (B's own), one completion each.
        assert receipt['b_tool_calls'] == 1 and receipt['completed_events'] == {a: 1, b: 1}, receipt
        assert [r['outcome'] for r in receipt['runs'][b]][-1] == 'completed', receipt
        # Every replayed call in the first request is answered exactly once; no orphan either way.
        assert len(set(receipt['b_first_calls'])) == len(receipt['b_first_calls']), receipt
        assert sorted(receipt['b_first_calls']) == sorted(receipt['b_first_results']), receipt
        seed = receipt['b_seed']
        if mode == 'review_continue_compressed':
            # A was compacted before B seeded: B starts from the post-compaction tip (summary + tail),
            # never from A's pre-compaction rows, and the record names the tip it copied.
            assert receipt['b_first_has_compaction'] and not receipt['b_first_has_pre_compaction'], receipt
            assert (seed['source_session'], seed['source_physical']) == (receipt['a_session'], receipt['a_compacted']), receipt
            assert seed['count'] == receipt['a_tip_rows'], receipt
        else:
            assert seed['source_physical'] == seed['source_session'] and receipt['b_first_has_pre_compaction'], receipt
        assert seed['outcome'] == 'inherited' and f"({seed['count']} messages" in receipt['b_first_note'], receipt
        return
    if mode not in {'complete', 'redirect'}:
        assert receipt['task_status'] == 'ready'
        return
    if mode == 'redirect':
        # A canonical redirect of the live dispatcher attempt's managed worker, mid provider call: the
        # model request is cancelled and rebuilt around the correction while the SAME run, claim and
        # spawned worker keep the task, and that attempt still completes it as one admission.
        assert receipt['redirect_status'] == 'redirected', receipt
        assert receipt['claim_kept'] and receipt['correction_before_release'], receipt
        assert receipt['run_outcome'] == 'completed', receipt
        assert receipt['correction_comment'] and receipt['correction_in_next_context'], receipt
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    print(json.dumps(receipt))


def test_late_launcher_cannot_stamp_replacement_claim(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned race', assignee='default')
        old = kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason='replace claim', expected_run_id=old.current_run_id)
        kb.unblock_task(conn, tid)
        new = kb.claim_task(conn, tid)
        dispatch._set_worker_pid(conn, tid, 424242, run_id=old.current_run_id, claim_lock=old.claim_lock)
        assert kb.get_task(conn, tid).worker_pid is None
        assert conn.execute('SELECT worker_pid FROM task_runs WHERE id=?', (new.current_run_id,)).fetchone()[0] is None


def test_claim_reclaimed_before_admission_marker_is_refused(tmp_path, monkeypatch):
    """A claim that stops being current between policy/session creation and the marker
    transaction is refused: no owner_admitted row, no resume, no submit."""
    import asyncio
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned race', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        old = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    calls = []
    def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
        # Another process reclaims the task while the owner session is being created.
        with closing(connect(board='owned')) as conn:
            kb.block_task(conn, task_id, reason='replace claim', expected_run_id=old.current_run_id)
            kb.unblock_task(conn, task_id)
            kb.claim_task(conn, task_id)
        return SimpleNamespace(session_id='owner-session', profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_local.create_local_session', create_local_session)
    async def submit(actor, submission, **kwargs):
        calls.append(('submit', submission))
    async def resume(ref, params):
        calls.append(('resume', ref))
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=old.current_run_id, claim_lock=old.claim_lock)
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 0
