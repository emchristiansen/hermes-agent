"""Private owner-side reservation for a concrete already-started admission.

The producer retains the Popen handle and holds the child's bootstrap pipe until
this returns. Public worker.register remains idle-only for unreserved callers.
"""
import json
import os
import secrets
import subprocess

from gateway.session_admission import admission_fingerprint
from gateway.owner_identity import same_owner
from hermes_state_runtime import OWE_OWNER_RESTART, RuntimeStoreError, _admission, _epoch, _secret_digest, _session

# A venv launcher (uv's Windows python.exe, pip's exe shim) sits between the owner's
# handle and the interpreter; the real worker is never further away than this.
MAX_LAUNCHER_HOPS = 3


def verify_worker_identity(process, hello):
    """Return the verified ``(pid, birth)`` of the interpreter behind ``process``.

    ``Popen.pid`` may be a launcher trampoline, so the worker reports its own pid and
    birth in a ``hello`` frame; the OWNER proves that pid is alive, carries that birth,
    and descends from (or is) the reserved handle. The worker never decides equality.
    """
    import psutil
    if not isinstance(process, subprocess.Popen) or process.poll() is not None:
        raise RuntimeStoreError('worker_not_live')
    if (not isinstance(hello, dict) or hello.get('type') != 'hello' or type(hello.get('pid')) is not int
            or type(hello.get('birth')) not in (int, float) or not isinstance(hello.get('ancestors'), list)
            or any(type(p) is not int for p in hello['ancestors'])):
        raise RuntimeStoreError('invalid_worker_frame')
    pid = hello['pid']
    if pid <= 0 or pid == os.getpid():
        raise RuntimeStoreError('permission_denied')
    try:
        handle = psutil.Process(process.pid)
        if handle.ppid() != os.getpid() or handle.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeStoreError('permission_denied')
        worker = psutil.Process(pid)
        from hermes_cli.process_identity import worker_birth
        birth = worker_birth(worker)  # kernel start: stable across clock steps (#140580) and btime re-anchors
        if birth != hello['birth'] or not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeStoreError('permission_denied')
        # psutil's parent() already refuses a recycled ppid (parent born after child).
        chain = [p.pid for p in worker.parents()[:MAX_LAUNCHER_HOPS]]
    except psutil.Error as exc:
        raise RuntimeStoreError('worker_not_live') from exc
    if pid != process.pid and process.pid not in chain:
        raise RuntimeStoreError('permission_denied')
    return pid, birth


def owner_worker_subject():
    """The subject a managed worker's adoption ticket carries: the owner's own OS identity. The
    control channel mints a worker ticket only for a same-user peer (``uid:<n>`` / ``sid:<sid>``)."""
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import _process_sid
        return 'sid:' + _process_sid(os.getpid())
    return f'uid:{os.getuid()}'  # windows-footgun: ok — POSIX branch


# Spelling is not ownership: legacy admission ids are arbitrary text, including ':g2'.
# Exact links commit with each new reservation; legacy rows/receipts remain untouched.
ADMISSION_WORKER_PREFIX = 'admission-worker:'
CLAIM_SUFFIX = ':g'
_CLAIMS_KEY = 'gateway.admission_worker_claims.v1.'
_OWNER_KEY = 'gateway.admission_worker_owner.v1.'


def admission_worker_execution_ids(admission_id, generation):
    """Preferred legacy spellings, not an ownership proof or an exhaustive reader."""
    first = ADMISSION_WORKER_PREFIX + admission_id
    return first, f'{first}{CLAIM_SUFFIX}{generation}'


class WorkerOwnershipConflict(RuntimeStoreError):
    """Unreadable retained worker proof, never a queued input's policy verdict.

    Keep the existing wire refusal while letting the drain distinguish whose
    conflict this is. An independent accepted input must remain queued.
    """

    def __init__(self, detail):
        super().__init__('admission_conflict', field='worker_ownership', detail=detail)


def _identity_record(conn, key):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
        if value is None:
            raise ValueError('null identity record')
        return value
    except (TypeError, ValueError) as exc:
        raise WorkerOwnershipConflict('invalid retained identity record') from exc


def _claim_links(conn, admission_id):
    links = _identity_record(conn, _CLAIMS_KEY + admission_id)
    if links is None:
        return []
    if (not isinstance(links, list) or any(not isinstance(key, str) for key in links)
            or len(links) != len(set(links))):
        raise WorkerOwnershipConflict('invalid retained admission claim links')
    return links


def _execution_owner(conn, execution_id):
    owner = _identity_record(conn, _OWNER_KEY + execution_id)
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise WorkerOwnershipConflict('invalid retained execution owner')
    return owner


def _legacy_elsewhere(conn, admission_id, execution_id, session_id):
    """Exclude a legacy spelling only with positive foreign physical-scope proof.

    Merely finding another imported ID is not enough: a queued ':g2' admission
    must not hide a real generation-2 worker of its prefix admission. Ambiguous
    or no-longer-readable lineage stays conservatively fenced.
    """
    try:
        current = _admission(conn, admission_id)
    except RuntimeStoreError as exc:
        if exc.reason == 'not_found':
            return False
        raise
    def targets(row):
        try:
            lineage = json.loads(row['lineage_json'])
            if not isinstance(lineage, list) or any(not isinstance(key, str) for key in lineage):
                return None
            return {row['target_session_id'], *lineage}
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    own = targets(current)
    if own is None or session_id in own:
        return False
    spelling = execution_id[len(ADMISSION_WORKER_PREFIX):]
    alternatives = {spelling}
    parent, suffix, generation = spelling.rpartition(CLAIM_SUFFIX)
    if suffix and generation.isdigit():
        alternatives.add(parent)
    for other_id in alternatives - {admission_id}:
        try:
            other = _admission(conn, other_id)
        except RuntimeStoreError as exc:
            if exc.reason == 'not_found':
                continue
            raise
        foreign = targets(other)
        if foreign is not None and session_id in foreign:
            return True
    return False


def admission_worker_rows(conn, admission_id):
    """Exact new links plus unambiguous legacy roots, including physically moved workers.

    A legacy numeric suffix is not ownership proof: disjoint retained admission
    scopes disambiguate old first/retry spellings. Tombstones count too. Ambiguous
    legacy roots remain conservatively fenced. Explicit new ownership takes priority
    over all spellings; it never rewrites an old worker, receipt or adoption digest.
    """
    first, _ = admission_worker_execution_ids(admission_id, 0)
    links = _claim_links(conn, admission_id)
    rows = {}
    for execution_id in links:
        row = conn.execute('SELECT execution_id,status,generation FROM worker_executions WHERE execution_id=?',
                           (execution_id,)).fetchone()
        if row is None:
            # Terminal session retirement moves (never fabricates) the existing
            # worker receipt to a tombstone. Keep that retained fence readable.
            from hermes_state_terminal import WORKER_PREFIX
            retired = _identity_record(conn, WORKER_PREFIX + execution_id)
            if (not isinstance(retired, dict) or retired.get('execution_id') != execution_id
                    or retired.get('status') != 'terminal' or type(retired.get('generation')) is not int):
                raise WorkerOwnershipConflict('missing or inconsistent retained worker tombstone')
            row = (execution_id, retired['status'], retired['generation'])
        if _execution_owner(conn, execution_id) != admission_id:
            raise WorkerOwnershipConflict('retained worker owner does not match admission claim link')
        rows[execution_id] = row
    legacy = conn.execute("SELECT execution_id,status,generation,session_id FROM worker_executions "
                          "WHERE kind='compute' AND (execution_id=? OR substr(execution_id,1,?)=?)",
                          (first, len(first + CLAIM_SUFFIX), first + CLAIM_SUFFIX)).fetchall()
    for row in legacy:
        execution_id = row[0]
        if execution_id != first and not execution_id[len(first + CLAIM_SUFFIX):].isdigit():
            continue
        owner = _execution_owner(conn, execution_id)
        if owner is not None:
            if owner == admission_id:
                if execution_id not in links:
                    raise WorkerOwnershipConflict('retained worker owner has no admission claim link')
                rows[execution_id] = tuple(row[:3])
            continue
        if _legacy_elsewhere(conn, admission_id, execution_id, row[3]):
            continue
        rows[execution_id] = tuple(row[:3])
    return list(rows.values())


def _claim_execution_id(conn, admission_id, generation):
    """One reservation per claim, preserving every predecessor and unrelated identity."""
    prior = admission_worker_rows(conn, admission_id)
    if any(status != 'terminal' or type(earlier) is not int or earlier >= generation
           for _execution_id, status, earlier in prior):
        raise RuntimeStoreError('admission_conflict')
    first, claimed = admission_worker_execution_ids(admission_id, generation)
    candidate = claimed if prior else first
    # A legacy first id and another admission's retry may have identical spelling.
    # Allocate around an unrelated retained key, never delete it or narrow imports.
    from agent.managed_worker import outbox_name
    try:
        outbox_name(candidate)
    except ValueError:
        # Imports may contain arbitrary text; the execution key need not repeat it.
        candidate = ADMISSION_WORKER_PREFIX + secrets.token_hex(32)
    from hermes_state_terminal import WORKER_PREFIX
    while (conn.execute('SELECT 1 FROM worker_executions WHERE execution_id=?', (candidate,)).fetchone()
           or _execution_owner(conn, candidate) is not None
           or _identity_record(conn, WORKER_PREFIX + candidate) is not None):
        candidate = ADMISSION_WORKER_PREFIX + secrets.token_hex(32)
    return candidate


def _record_claim(conn, admission_id, execution_id):
    links = _claim_links(conn, admission_id)
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                 (_OWNER_KEY + execution_id, json.dumps(admission_id)))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (_CLAIMS_KEY + admission_id, json.dumps([*links, execution_id])))


def reserve_admission_worker(authority, *, admission_id, process, principal_id, hello, worker_subject=None):
    """Return a private bootstrap scope; never accept caller-selected kind/target.

    Only the owner process may reserve its live direct exec child. The admission
    supplies physical session and generation; its durable principal must match
    the producer. Do not publish the returned secret on any client event stream.

    The adoption claim binds ``worker_subject``, the identity the worker will authenticate
    as (``owner_worker_subject``), when given. An automation admission's principal is a route
    (``automation:<session key>``), never a ticket subject, so a claim bound to it refused every
    adopt and each bot DM or completion wake on a managed session ended interrupted (#144139).
    """
    authority._require_admission_open()
    pid, birth = verify_worker_identity(process, hello)
    secret = secrets.token_urlsafe(32)

    def write(conn):
        _epoch(conn, authority.epoch)
        admission = _admission(conn, admission_id)
        if admission['status'] != 'started' or admission['owner_epoch'] != authority.epoch:
            raise RuntimeStoreError('producer_not_started')
        if not same_owner(admission['principal_id'], principal_id):
            raise RuntimeStoreError('permission_denied')
        owner, generation = admission['target_session_id'], admission['generation']
        if _session(conn, owner)['runtime_generation'] != generation:
            raise RuntimeStoreError('stale_generation')
        from hermes_state_local_lineage import local_physical_target
        sid = local_physical_target(conn, owner)
        execution_id = _claim_execution_id(conn, admission_id, generation)
        if conn.execute('SELECT 1 FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone():
            raise RuntimeStoreError('admission_conflict')
        from hermes_state_runtime_workers import claim_worker_states
        if claim_worker_states(conn, owner, authority.epoch):
            raise RuntimeStoreError('stale_generation')
        # The private worker fence follows its physical assignment; admission/FIFO
        # generation and identity remain on the logical owner.
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation, sid))
        if sid != owner:
            conn.execute("UPDATE session_admissions SET lineage_json=json_insert(lineage_json,'$[#]',?) "
                         "WHERE admission_id=?", (sid, admission_id))
        scope = dict(profile_id=authority.profile_id, session_id=sid, execution_id=execution_id,
                     generation=generation, pid=pid, birth=birth, secret=secret)
        claim = admission_fingerprint(canonical_target=sid, payload=scope | {'principal': worker_subject or principal_id})
        conn.execute("INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
                     "VALUES(?,?,'compute',?,?,'registered',?)",
                     (execution_id, sid, authority.epoch, generation, _secret_digest(claim)))
        _record_claim(conn, admission_id, execution_id)
        return scope | {'epoch': authority.epoch}

    return authority.db._execute_write(write)


def lose_admission_worker(authority, row, scope, owner_restart=False):
    """Revoke a private assignment and pause its FIFO; no inference replay.

    This also fences a result already in the pipe at loss/shutdown. Retention
    requires a still-started admission even when the generation has not changed.
    ``owner_restart``: the owner was already ending when it found the worker lost, so a Kanban card
    attempt's cause (``OWNER_RESTART``) commits with this fence.
    """
    def write(conn):
        _epoch(conn, scope['epoch'])
        current = _admission(conn, row['admission_id'])
        if (current['status'] != 'started' or current['owner_epoch'] != scope['epoch']
                or current['generation'] != scope['generation']):
            raise RuntimeStoreError('stale_generation')
        worker = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                              (scope['execution_id'],)).fetchone()
        if worker is None or worker['generation'] != scope['generation'] or worker['owner_epoch'] != scope['epoch']:
            raise RuntimeStoreError('stale_generation')
        conn.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id=?", (scope['execution_id'],))
        from hermes_state_runtime_workers import compact_terminal_receipts
        compact_terminal_receipts(conn, scope['execution_id'])
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
        if owner_restart:
            conn.execute(OWE_OWNER_RESTART + ' AND admission_id=?', (row['admission_id'],))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (current['target_session_id'],))
    authority.db._execute_write(write)
