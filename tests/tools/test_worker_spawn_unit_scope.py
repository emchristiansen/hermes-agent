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


DESCENDANT = """import os, sys, time
if os.fork() == 0:
    os.setsid()
    open(sys.argv[1], 'w').write(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
time.sleep(1)
"""


class ScopedDaemon:
    """A registered worker, the main process of a transient user service, starts a command whose root
    exits 0 at once while a ``setsid`` descendant keeps running in the command's scope; the service
    stops with the worker. Fresh same-session interpreters then act on the profile's checkpoint."""

    def __init__(self, tmp_path, request, *, task_id=''):
        from tools.process_registry import systemd_user_bus_env
        self.home, self.child, self.bus = tmp_path / 'home', tmp_path / 'child.pid', systemd_user_bus_env()
        self.home.mkdir()
        command = shlex.join([sys.executable, '-c', DESCENDANT, str(self.child)])
        spawner = textwrap.dedent(f"""
            import json, os
            from agent import runtime_session_store
            runtime_session_store._worker_process = True
            from tools.process_registry import process_registry
            session = process_registry.spawn_local({command!r}, cwd={str(tmp_path)!r}, task_id={task_id!r})
            print(json.dumps([session.id, session.systemd_unit]), flush=True)
            os._exit(0)
        """)
        unit = f'hermes-test-worker-custody-{uuid.uuid4().hex[:12]}'
        out = subprocess.run(
            ['systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect', f'--unit={unit}',
             '--property=TimeoutStopSec=2', f'--working-directory={ROOT}', f'--setenv=HERMES_HOME={self.home}',
             '--setenv=HERMES_SESSION_ID=custody-session', f'--setenv=PYTHONPATH={ROOT}',
             f'--setenv=PATH={os.environ.get("PATH", "")}', '--', sys.executable, '-c', spawner],
            env=self.bus, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', timeout=120,
            check=False)
        self.id, self.scope = json.loads(out.stdout.strip().splitlines()[-1])
        request.node.live_system_guard_owned_units = (self.scope,)  # this test's own scope: cleanup may stop it
        self.env = {k: v for k, v in os.environ.items() if k not in ('HERMES_HOME', 'PYTHONPATH')}
        self.env.update(HERMES_HOME=str(self.home), PYTHONPATH=str(ROOT), HERMES_SESSION_ID='custody-session')

    def interpreter(self, body):
        code = textwrap.dedent("""
            import json
            from tools.process_registry import ProcessRegistry
            registry = ProcessRegistry()
            recovered = registry.recover_from_checkpoint()
        """) + textwrap.dedent(body)
        result = subprocess.run([sys.executable, '-c', code], env=self.env, cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8', timeout=120, check=False)
        return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else result.stderr

    def descendant_in_scope(self):
        try:
            pid = int(self.child.read_text(encoding='utf-8-sig'))
            return Path(f'/proc/{pid}/cgroup').read_text(encoding='utf-8-sig').strip().endswith(f'/{self.scope}')
        except (FileNotFoundError, ValueError):
            return False

    def durable(self):
        """(checkpoint entries as (id, scope_custody, unit), the receipt's unit)."""
        checkpoint = json.loads((self.home / 'processes.json').read_text(encoding='utf-8-sig'))
        receipt_path = self.home / 'logs' / 'process-results' / f'{self.id}.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        return [(e['session_id'], e.get('scope_custody'), e['systemd_unit']) for e in checkpoint], receipt['systemd_unit']

    def stop(self):
        subprocess.run(['systemctl', '--user', 'stop', self.scope], env=self.bus, capture_output=True, timeout=30,
                       check=False)


@pytest.fixture
def scoped_daemon(tmp_path, request):
    from tools.process_registry import _systemd_run_user_scope_available
    if not _systemd_run_user_scope_available():
        pytest.skip("needs a reachable systemd user manager (systemd-run --user --scope)")
    started = []
    yield lambda **kwargs: started.append(ScopedDaemon(tmp_path, request, **kwargs)) or started[-1]
    for daemon in started:
        daemon.stop()


@pytest.mark.platforms("linux")
def test_scope_custody_survives_the_root_exit_until_a_stop_empties_it(scoped_daemon):
    """BG-SCOPE-01: the first adopter collects the receipt; custody of the scope stays durable
    (checkpoint + receipt); a second same-session interpreter recovers it without stopping it, a
    failed stop keeps it, and the next interpreter's ``kill_all`` (the gateway's shutdown sweep) stops it."""
    job = scoped_daemon()
    first = job.interpreter(f'print(json.dumps(registry.wait({job.id!r}, timeout=30)))')
    assert first['status'] == 'exited' and first['exit_code'] == 0, first  # the command's receipt
    assert job.descendant_in_scope()
    assert job.durable() == ([(job.id, True, job.scope)], job.scope)
    failed = job.interpreter(f"""
        import tools.process_registry as module
        module._stop_systemd_unit = lambda _unit: False
        print(json.dumps([recovered, registry.kill_all(source='gateway_shutdown'),
                          registry.get({job.id!r}).systemd_unit]))
    """)
    assert failed == [0, 0, job.scope], failed  # recovery and a failed stop both keep custody
    assert job.descendant_in_scope() and job.durable() == ([(job.id, True, job.scope)], job.scope)
    stopped = job.interpreter(f"""
        print(json.dumps([registry.kill_all(source='gateway_shutdown'), registry.kill_process({job.id!r})]))
    """)
    assert stopped[0] == 1 and stopped[1]['status'] == 'already_exited', stopped
    assert 'scope_stop_failed' not in stopped[1] and stopped[1]['exit_code'] == 0, stopped
    assert not job.descendant_in_scope()
    assert job.durable() == ([], '')


@pytest.mark.platforms("linux")
def test_a_later_turn_timeout_spares_an_earlier_turns_retained_scope(scoped_daemon):
    """BG-SCOPE-02: a turn's timeout reaps only what that turn started. A retained scope from an
    earlier turn (default ``persist_on_release=False``) is in the turn's baseline, so the timeout
    stops the new job alone; a later explicit stop still reaches the retained scope."""
    job = scoped_daemon(task_id='turn-owner')
    turn = job.interpreter(f"""
        old = registry.wait({job.id!r}, timeout=30)
        baseline = registry.snapshot_running_ids('turn-owner')
        new = registry.spawn_local('sleep 30', task_id='turn-owner')
        killed = registry.kill_started_since('turn-owner', baseline, source='gateway_turn_timeout')
        print(json.dumps([old['exit_code'], sorted(baseline), killed, registry.poll(new.id),
                          registry.get({job.id!r}).persist_on_release]))
    """)
    old_exit, baseline, killed, new, persisted = turn
    assert (old_exit, baseline, killed, persisted) == (0, [job.id], 1, False), turn
    assert (new['exit_code'], new['completion_reason'], new['termination_source']) == (
        -15, 'killed', 'gateway_turn_timeout'), new
    assert job.descendant_in_scope()  # the earlier turn's daemon survived the later turn's timeout
    assert job.durable() == ([(job.id, True, job.scope)], job.scope)
    stopped = job.interpreter(f'print(json.dumps(registry.kill_process({job.id!r})))')
    assert stopped['status'] == 'already_exited' and 'scope_stop_failed' not in stopped, stopped
    assert not job.descendant_in_scope()
    assert job.durable() == ([], '')
