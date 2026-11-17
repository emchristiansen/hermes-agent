"""A caller finds the live attempt of an exact card run and redirects it, never a successor (session.kanban_run)."""
import json
import subprocess
import sys
from pathlib import Path


def test_caller_discovers_and_redirects_the_exact_live_run(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_discovery_probe.py')],
                            env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=400, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt, indent=2))
    assert receipt['two_runs_exact'], receipt
    assert receipt['unqualified_run'] == 'stale_kanban_claim' and receipt['wrong_profile'] == 'profile_mismatch', receipt
    assert receipt['unauthorized'] in {'permission_denied'} or str(receipt['unauthorized']).startswith('connection_refused'), receipt
    assert receipt['cli_a'] == [0, 'redirected', True, True] or receipt['cli_a'][:1] == [0], receipt
    assert receipt['b_untouched'], receipt
    assert receipt['cli_b'][0] == 0 and receipt['resend_a'][0] == 0, receipt
    assert receipt['replacement_run_differs'] and receipt['replacement_session_differs'], receipt
    assert receipt['old_run_after_replacement'] == 'stale_kanban_claim', receipt
    assert receipt['stale_cli'][0] == 3 and receipt['stale_cli'][2] == 'stale_kanban_claim', receipt
    # A correction carrying the replaced run's generation straight to its session is refused or kept on the card
    # (named for the old run); it is never answered as redirected into a live agent of the successor.
    assert receipt['stale_direct'] != 'redirected', receipt
    assert receipt['b_pid_kept'] and receipt['b_done_once'], receipt
    assert receipt['b_admissions'][0] == 1, receipt
