"""A background process a registered worker starts outlives the systemd service the worker runs in.

A supervised gateway's managed workers share the gateway unit's cgroup. What a worker starts is
meant to outlive it (the session's next worker or the owner at startup adopts the checkpoint entry
and reads its log and recorded exit), but stopping or restarting that unit killed it with the unit,
and the interactive ``-lic`` wrapper, which ignores SIGTERM, held the stop until systemd's SIGKILL
(production 2026-11-05 01:59: the gateway unit sat 90s in ``final-sigterm`` on one such shell).
Real systemd: the worker is the main process of a transient user service, so its exit stops the
unit and systemd signals everything still in that unit's cgroup.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMAND = 'sleep 6 && echo SURVIVED_$((40+2))'  # outlives the unit's 2s stop timeout


@pytest.mark.platforms("linux")
def test_worker_spawn_survives_the_stop_of_its_service_unit(tmp_path):
    from tools.process_registry import _systemd_run_user_scope_available, systemd_user_bus_env
    if not _systemd_run_user_scope_available():
        pytest.skip("needs a reachable systemd user manager (systemd-run --user --scope)")
    home = tmp_path / 'home'
    home.mkdir()
    spawner = textwrap.dedent(f'''
        import json, os
        from agent import runtime_session_store
        runtime_session_store._worker_process = True
        from tools.process_registry import process_registry
        session = process_registry.spawn_local({COMMAND!r}, cwd={str(tmp_path)!r})
        print(json.dumps([session.id, session.systemd_unit]), flush=True)
        os._exit(0)
    ''')
    unit = f'hermes-test-worker-unit-{uuid.uuid4().hex[:12]}'
    out = subprocess.run(
        ['systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect', f'--unit={unit}',
         '--property=TimeoutStopSec=2', f'--working-directory={ROOT}', f'--setenv=HERMES_HOME={home}',
         f'--setenv=PYTHONPATH={ROOT}', f'--setenv=PATH={os.environ.get("PATH", "")}',
         '--', sys.executable, '-c', spawner],
        env=systemd_user_bus_env(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding='utf-8', timeout=120, check=False)
    session_id, _scope = json.loads(out.stdout.strip().splitlines()[-1])
    adopter = textwrap.dedent(f'''
        import json
        from tools.process_registry import ProcessRegistry
        registry = ProcessRegistry()
        registry.recover_from_checkpoint()
        print(json.dumps(registry.wait({session_id!r}, timeout=30)))
    ''')
    env = {k: v for k, v in os.environ.items() if k not in ('HERMES_HOME', 'PYTHONPATH')}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, '-c', adopter], env=env, cwd=ROOT, capture_output=True, text=True,
                         encoding='utf-8', timeout=120, check=False)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result['status'] == 'exited', (result, out.stderr[-2000:])
    assert result['exit_code'] == 0 and result['output'].strip() == 'SURVIVED_42', result


DESCENDANT = '''import os, sys, time
if os.fork() == 0:
    os.setsid()
    open(sys.argv[1], 'w').write(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
time.sleep(1)
'''


@pytest.mark.platforms("linux")
def test_scope_custody_survives_the_root_exit_until_a_stop_empties_it(tmp_path, request):
    """BG-SCOPE-01: the root exits 0 while a ``setsid`` descendant still runs in its scope. The
    first adopter collects the receipt; custody of the scope stays durable (checkpoint + receipt);
    a second same-session interpreter recovers it without stopping it, a failed stop keeps it, and
    the next interpreter's ``kill_all`` (the gateway's shutdown sweep) stops it."""
    from tools.process_registry import _systemd_run_user_scope_available, systemd_user_bus_env
    if not _systemd_run_user_scope_available():
        pytest.skip("needs a reachable systemd user manager (systemd-run --user --scope)")
    home, child = tmp_path / 'home', tmp_path / 'child.pid'
    home.mkdir()
    command = shlex.join([sys.executable, '-c', DESCENDANT, str(child)])
    spawner = textwrap.dedent(f'''
        import json, os
        from agent import runtime_session_store
        runtime_session_store._worker_process = True
        from tools.process_registry import process_registry
        session = process_registry.spawn_local({command!r}, cwd={str(tmp_path)!r})
        print(json.dumps([session.id, session.systemd_unit]), flush=True)
        os._exit(0)
    ''')
    bus = systemd_user_bus_env()
    unit = f'hermes-test-worker-custody-{uuid.uuid4().hex[:12]}'
    out = subprocess.run(
        ['systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect', f'--unit={unit}',
         '--property=TimeoutStopSec=2', f'--working-directory={ROOT}', f'--setenv=HERMES_HOME={home}',
         '--setenv=HERMES_SESSION_ID=custody-session', f'--setenv=PYTHONPATH={ROOT}',
         f'--setenv=PATH={os.environ.get("PATH", "")}', '--', sys.executable, '-c', spawner],
        env=bus, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', timeout=120,
        check=False)
    session_id, scope = json.loads(out.stdout.strip().splitlines()[-1])
    request.node.live_system_guard_owned_units = (scope,)  # this test's own scope: cleanup may stop it
    env = {k: v for k, v in os.environ.items() if k not in ('HERMES_HOME', 'PYTHONPATH')}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(ROOT), HERMES_SESSION_ID='custody-session')

    def interpreter(body):
        code = textwrap.dedent('''
            import json
            from tools.process_registry import ProcessRegistry
            registry = ProcessRegistry()
            recovered = registry.recover_from_checkpoint()
        ''') + textwrap.dedent(body)
        result = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', timeout=120, check=False)
        return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else result.stderr

    def descendant_in_scope():
        try:
            pid = int(child.read_text(encoding='utf-8-sig'))
            return Path(f'/proc/{pid}/cgroup').read_text(encoding='utf-8-sig').strip().endswith(f'/{scope}')
        except (FileNotFoundError, ValueError):
            return False

    def durable():
        checkpoint = json.loads((home / 'processes.json').read_text(encoding='utf-8-sig'))
        receipt_path = home / 'logs' / 'process-results' / f'{session_id}.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        return [(e['session_id'], e.get('scope_custody'), e['systemd_unit']) for e in checkpoint], receipt['systemd_unit']

    try:
        first = interpreter(f'print(json.dumps(registry.wait({session_id!r}, timeout=30)))')
        assert first['status'] == 'exited' and first['exit_code'] == 0, first  # the command's receipt
        assert descendant_in_scope()
        assert durable() == ([(session_id, True, scope)], scope)
        failed = interpreter(f'''
            import tools.process_registry as module
            module._stop_systemd_unit = lambda _unit: False
            print(json.dumps([recovered, registry.kill_all(source='gateway_shutdown'),
                              registry.get({session_id!r}).systemd_unit]))
        ''')
        assert failed == [0, 0, scope], failed  # recovery and a failed stop both keep custody
        assert descendant_in_scope() and durable() == ([(session_id, True, scope)], scope)
        stopped = interpreter(f'''
            print(json.dumps([registry.kill_all(source='gateway_shutdown'), registry.kill_process({session_id!r})]))
        ''')
        assert stopped[0] == 1 and stopped[1]['status'] == 'already_exited', stopped
        assert 'scope_stop_failed' not in stopped[1] and stopped[1]['exit_code'] == 0, stopped
        assert not descendant_in_scope()
        assert durable() == ([], '')
    finally:
        subprocess.run(['systemctl', '--user', 'stop', scope], env=bus, capture_output=True, timeout=30, check=False)
