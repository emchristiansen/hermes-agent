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
