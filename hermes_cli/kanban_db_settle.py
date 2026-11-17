"""Administrative settlement of an assessment its own worker delivered after its run was ended.

t_234158d0, 2026-11-07: a stale ``schedule`` ended reviewer run 6493 while its worker kept going;
the worker then delivered a negative assessment and exited 0, but ``complete`` refuses a
``scheduled`` card, and every other route reruns the review or passes through ``ready`` where the
dispatcher may claim it. This settles exactly that state, ``scheduled`` -> ``done`` in one write
transaction, fenced on the exact latest terminal run and its own worker's clean result and
attributed to the calling orchestrator. Run N's row and events are left as they are and no run is
synthesized: the delivered result lands on the task and one ``completed`` event records the
settlement, its structured metadata and who made it; :func:`settled_handoff` hands that, not an
older completed run, to dependents. Only local-only assessments qualify: a PR/repository
completion contract keeps its acceptance gate (review run6639 SI-04).

Every absolute path the result names that resolves into the card's scratch workspace, however it
is spelled or wrapped, is attached byte-identically before cleanup removes the workspace, or the
settlement is refused with nothing changed (review run6746 SI-05, run6784 SI-09, run6860 SI-11/12):
a reference is accepted only when it names exactly one regular file as a whole word or a closed
quotation, and every other reference into the workspace refuses. Cleanup runs only once each
referenced report's bytes are proven held by this card's attachments. Only the copies this attempt
made, and only those no committed row holds, are removed when it fails (SI-07, SI-08); and a later
result edit is recorded beside the settlement rather than on an older completed run, its handoff
summary and metadata both reaching dependents (SI-06, SI-10).
"""
import hashlib
import os
import sqlite3
import time
from pathlib import Path

from hermes_cli.kanban_db_retry_policy import _last_event_payload, terminal_run_refusal

# A delivered NEGATIVE assessment only: there is deliberately no passing verdict here.
VERDICTS = ('changes_required', 'rejected')


def delivered_settlement_refusal(conn, task_id, expected_terminal_run):
    """Why ``task_id`` cannot be settled from run ``expected_terminal_run`` (bounded), or None."""
    task = conn.execute(
        'SELECT status,current_run_id,claim_lock,claim_expires,worker_pid FROM tasks WHERE id=?', (task_id,),
    ).fetchone()
    if task is None:
        return 'no such task'
    if task['status'] != 'scheduled':
        return f'status is {task["status"]}, not scheduled'
    contract = conn.execute('SELECT completion_contract FROM tasks WHERE id=?', (task_id,)).fetchone()[0]
    if contract and contract != 'local-only':
        return (f'its completion contract {contract[:120]!r} requires PR acceptance; '
                'settle-delivered settles only local-only assessments')
    if any(task[key] is not None for key in ('current_run_id', 'claim_lock', 'claim_expires', 'worker_pid')):
        return f'the task still holds a run or claim (current run {task["current_run_id"]})'
    if conn.execute('SELECT 1 FROM task_runs WHERE task_id=? AND ended_at IS NULL', (task_id,)).fetchone():
        return 'an attempt is still open'
    refusal = terminal_run_refusal(conn, task_id, expected_terminal_run)
    if refusal is not None:
        return refusal
    run = conn.execute('SELECT outcome,claim_lock FROM task_runs WHERE id=?', (expected_terminal_run,)).fetchone()
    if run['outcome'] != 'scheduled':
        return f'run {expected_terminal_run} ended {run["outcome"]}, not by a schedule'
    result = _last_event_payload(conn, task_id, expected_terminal_run, 'worker_result')
    if (not run['claim_lock'] or not isinstance(result, dict) or result.get('exit_code') != 0
            or result.get('claim_lock') != run['claim_lock']):
        return f'run {expected_terminal_run} has no clean result (exit 0) from its own claimed worker'
    if not _kb._parents_satisfied(conn, task_id):
        return 'a parent is not done'
    return None


def settle_delivered_assessment(conn, task_id, *, expected_terminal_run, verdict, result, author, metadata=None):
    """Orchestrator-only ``scheduled`` -> ``done`` for a delivered negative assessment.

    Returns ``(ok, reason)``. Malformed arguments raise ``ValueError``; nothing is written on refusal."""
    from agent.swarm_capabilities import worker_board_management_enabled
    if os.environ.get('HERMES_KANBAN_TASK') and not worker_board_management_enabled():
        return False, 'delivered-assessment settlement is orchestrator-only'
    if type(expected_terminal_run) is not int or expected_terminal_run < 1:
        raise ValueError('expected terminal run must be a positive run id')
    if verdict not in VERDICTS:
        raise ValueError(f'verdict must be one of {"/".join(VERDICTS)}')
    if any(not isinstance(value, str) or not value.strip() for value in (result, author)):
        raise ValueError('result and author must be nonempty')
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError('metadata must be a JSON object')
    # References are judged on the argument exactly as given: the stored result is trimmed, but a
    # trailing whitespace character may be the end of a report's name (SI-18).
    original, result, now = result, result.strip(), int(time.time())
    # Never mutate the caller's dict: staging rewrites its artifact paths (a retry must start clean).
    # ``_staged_artifacts`` is the stager's own output; a caller-supplied one never names our copies.
    if metadata is not None:
        metadata = {key: value for key, value in metadata.items() if key != '_staged_artifacts'}
    metadata = _named_scratch_reports(conn, task_id, original, metadata)
    reports = _scratch_sources(conn, task_id, metadata)
    copies = []
    try:
        with _kb.write_txn(conn):
            refusal = delivered_settlement_refusal(conn, task_id, expected_terminal_run)
            if refusal is not None:
                return False, refusal
            cur = conn.execute(
                "UPDATE tasks SET status='done', result=?, completed_at=?, block_kind=NULL, block_recurrences=0 "
                "WHERE id=? AND status='scheduled' AND current_run_id IS NULL AND claim_lock IS NULL "
                "AND worker_pid IS NULL", (result, now, task_id),
            )
            if cur.rowcount != 1:
                return False, 'it changed while being settled; re-read the card'
            if metadata:
                # As _stage_completion_artifacts, but the copies are known before any row is written.
                _kb._persist_scratch_completion_artifacts(conn, task_id, metadata)
                copies = [Path(path) for path in metadata.pop('_staged_artifacts', [])]
                for path in copies:
                    _kb._insert_completion_attachment(
                        conn, task_id, filename=path.name, stored_path=str(path), size=path.stat().st_size,
                        created_at=now, uploaded_by='kanban_settle_delivered')
            _require_held(reports, copies)  # every report's bytes are in this attempt's copies, or roll back
            # The structured metadata lives on the settlement record, not on run N (SI-03).
            _kb._append_event(conn, task_id, 'completed', {
                **_kb._completed_event_payload(result, result, [], metadata),
                'settlement': {'kind': 'delivered_assessment', 'actor': author.strip(), 'verdict': verdict,
                               'expected_terminal_run': expected_terminal_run, 'from_status': 'scheduled',
                               'metadata': metadata or None},
            })
    except BaseException:
        _discard_uncommitted_copies(conn, copies)
        raise
    # The same post-commit effects as an ordinary completion: dependents, counters, workspace, hooks.
    _kb._clear_failure_counter(conn, task_id)
    _kb.recompute_ready(conn)
    kept = ''
    try:  # cleanup only once the committed attachments are proven to hold every report (SI-11)
        _require_held(reports, [row[0] for row in conn.execute(
            'SELECT stored_path FROM task_attachments WHERE task_id=?', (task_id,))])
    except (_kb.ArtifactPreservationError, sqlite3.Error) as exc:
        kept = f'; scratch workspace kept: {exc}'
    else:
        _kb._cleanup_workspace(conn, task_id)
    _kb._fire_task_hook('kanban_task_completed', _kb.get_task(conn, task_id), task_id, None, summary=result)
    return True, f'settled from run {expected_terminal_run} as a delivered {verdict} assessment{kept}'


def _discard_uncommitted_copies(conn, copies):
    """Remove this attempt's own new copies that no committed attachment row holds (SI-07).

    ``copies`` are only the files this attempt created, never a registered attachment the metadata
    merely names (SI-08). A copy whose row did commit (an exception raised after COMMIT) is kept;
    when what committed cannot be read, every copy is kept: an orphan is recoverable, a row whose
    file was deleted is not."""
    if not copies:
        return
    try:
        if conn.in_transaction:
            return
        held = {row[0] for row in conn.execute(
            f"SELECT stored_path FROM task_attachments WHERE stored_path IN ({','.join('?' * len(copies))})",
            [str(path) for path in copies])}
    except sqlite3.Error:
        return
    _kb._discard_staged_copies([path for path in copies if str(path) not in held], copies[0].parent)


# A reference opened by one of these is quoted: its name is exactly the text up to the closer.
_DELIMITERS = {'`': '`', '"': '"', "'": "'", '<': '>'}
# Sentence punctuation that may end a BARE reference; a quoted name is never trimmed.
_TRAILING = '.,;:!?)]}'
_MAX_WORDS = 32  # a bare name longer than this is not resolved, so it refuses rather than guesses


def _canonical(text):
    """``text`` as the absolute path the filesystem resolves it to (``~``, ``.``, ``..``, repeated
    separators, symlinks), as staging and cleanup see it; None when it is not an absolute path."""
    expanded = os.path.expanduser(text)
    if not os.path.isabs(expanded):
        return None
    try:
        return Path(expanded).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _inside(readings, root):
    """``[(reading, canonical path)]`` for the readings that resolve into the workspace ``root``."""
    pairs = ((reading, _canonical(reading)) for reading in dict.fromkeys(readings) if reading)
    return [(reading, path) for reading, path in pairs if path is not None and path.is_relative_to(root)]


def _workspace_reference(result, start, root):
    """``(kind, path)`` for the reference starting at ``start``: ``('none', None)`` when no reading
    of it is in the workspace ``root``, ``('root', None)`` when it names only the workspace itself,
    ``('file', path)`` when its text identifies exactly one regular file, else ``('refuse', text)``.

    Identity, not existence, decides (review6909 SI-13/SI-14, review6997 SI-15): a quoted reference
    names exactly the text up to its closer, untrimmed; a bare one is accepted only when no other
    existing name is consistent with its words or with the original text they were cut or trimmed
    from (:func:`_extended`). Whatever is unaccounted for refuses; nothing is skipped or guessed."""
    line_end = result.find('\n', start)
    line_end = len(result) if line_end < 0 else line_end
    closer = _DELIMITERS.get(result[start - 1]) if start else None
    if closer:
        return _quoted_reference(result, start, line_end, closer, root)
    words = result[start:line_end]
    ends = [i for i, char in enumerate(words) if char.isspace()][:_MAX_WORDS] + [len(words)]
    # Each reading: a whole-word run (cut at whitespace/line end), and that run with trailing sentence
    # punctuation removed, marked as normalized; both are judged against the original text.
    readings = {}
    for end in ends:
        readings.setdefault(words[:end], False)
        readings.setdefault(words[:end].rstrip(_TRAILING), True)
    inside = [(reading, normalized, path) for reading, normalized in readings.items() if reading
              for path in [_canonical(reading)] if path is not None and path.is_relative_to(root)]
    if not inside:
        return 'none', None
    files = {path for _, _, path in inside if path != root and path.is_file()}
    if len(files) == 1:
        named = next(iter(files))
        if not any(_extended(result, start, reading, path, bare=True, normalized=normalized)
                   for reading, normalized, path in inside if path == named):
            return 'file', named
    elif not files and all(path == root or not path.exists() for _, _, path in inside) and \
            any(path == root for _, _, path in inside) and \
            not any(_continues_into(result, start + len(reading), root) for reading, _, path in inside if path == root):
        return 'root', None
    return 'refuse', words[:ends[-1]].strip()


def _continues_into(result, after, directory):
    """True when the ORIGINAL text right after a reading that resolved to ``directory`` (the workspace
    root) begins with the name of an existing entry in it: the shortened reading is then not a mention
    of the workspace itself but a cut-off reference into it, such as ``/ws/ REVIEW.md**`` (SI-17). An
    unreadable directory counts as a continuation."""
    try:
        entries = [entry.name for entry in directory.iterdir()]
    except OSError:
        return True
    original = result[after:]
    return any(entry and original.startswith(entry) for entry in entries)


def _quoted_reference(result, start, line_end, closer, root):
    """A quoted reference names exactly the text up to its first closer on the line, untrimmed. Its
    other possible endings, at any later closer in the whole text, across line breaks too (as for
    ``'FINAL'S.md'`` or ``'FINAL'S<newline>REVIEW.md'``, SI-16), must name nothing that exists; and an
    unclosed quotation into the workspace refuses."""
    closes = [i for i in range(start + 1, len(result)) if result[i] == closer]
    if not closes or closes[0] >= line_end:
        return _unclosed(result[start:line_end], root)
    name = result[start:closes[0]]
    inside = _inside([name], root)
    others = [path for _, path in _inside((result[start:i] for i in closes[1:]), root) if path.exists()]
    if not inside and not others:
        return 'none', None
    if inside and not others:
        path = inside[0][1]
        if path == root:
            # A quoted root is the workspace itself only when no existing entry's name starts at its
            # closer or right after it: `/ws/`REVIEW.md names the file `REVIEW.md (SI-19).
            if not any(_continues_into(result, at, root) for at in (closes[0], closes[0] + 1)):
                return 'root', None
            return 'refuse', result[start:line_end]
        if path.is_file() and not _extended(result, start, name, path, bare=False, normalized=False):
            return 'file', path
    return 'refuse', name


def _unclosed(words, root):
    """An opened quotation not closed on its line: its name is unknown, so any reading of its words
    that lands in the workspace refuses."""
    ends = [i for i, char in enumerate(words) if char.isspace()][:_MAX_WORDS] + [len(words)]
    return ('refuse', words.strip()) if _inside((words[:end] for end in ends), root) else ('none', None)


def _extended(result, start, reading, path, *, bare, normalized):
    """True when an existing entry beside the named file, other than it, could be what the reference
    actually names, so the reference is ambiguous (SI-13, SI-15). Judged on the ORIGINAL result text
    from the name's first character, before any cut or trim:

    * the original text continues into a longer existing entry (``FINAL. REVIEW.md**`` beside
      ``FINAL``; a quoted ``'FINAL'S\nREVIEW.md'`` whose first closer ends at ``FINAL``);
    * for a bare reading trimmed of trailing punctuation, any longer entry starting with its name;
    * for any bare reading, a longer entry starting with its name followed by whitespace.
    An unreadable directory counts as ambiguous."""
    expanded = Path(os.path.expanduser(reading))
    names = {expanded.name, path.name}
    original = result[start + reading.rfind('/') + 1:]
    parents = {path.parent, _canonical(str(expanded.parent))} - {None}
    for parent in parents:
        try:
            entries = [entry.name for entry in parent.iterdir()]
        except OSError:
            return True
        for entry in (entry for entry in entries if entry not in names):
            if original.startswith(entry) and len(entry) > min(map(len, names)):
                return True
            if bare and any(entry.startswith(name) and (normalized or entry[len(name)].isspace())
                            for name in names if len(entry) > len(name)):
                return True
    return False


def _named_scratch_reports(conn, task_id, result, metadata):
    """``metadata`` with every regular file the result names in the managed scratch workspace added
    to its ``artifacts``, so staging attaches it before cleanup (SI-01, SI-05, SI-09, SI-11..SI-14).

    Every ``/`` and ``~`` in the result may start an absolute reference, whatever precedes it; each
    reading is canonicalized and compared with the canonical workspace, so neither a different
    spelling nor surrounding markup hides a reference. A reference into the workspace is accepted
    only when its text identifies exactly one regular file (:func:`_workspace_reference`) and that
    exact name survives staging, which trims an artifact path's surrounding whitespace. Anything
    else into the workspace (a wrapper, a truncated, ambiguous or untrimmable name, a missing file,
    a directory) raises :class:`ArtifactPreservationError` before anything is written: settling
    would delete what the result points at without keeping it. Relative paths are not resolved."""
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None or not _kb._is_managed_scratch_path(workspace):
        return metadata
    root = _canonical(str(workspace))
    if root is None:
        return metadata
    found = []
    for start in (index for index, char in enumerate(result) if char in '/~'):
        kind, named = _workspace_reference(result, start, root)
        if kind == 'file' and str(named) != str(named).strip():
            kind, named = 'refuse', str(named)  # staging would attach a different, trimmed name (SI-14)
        if kind == 'file':
            found.append(str(named))
        elif kind == 'refuse':
            raise _kb.ArtifactPreservationError(
                f'the result names {named[:300]!r} under the scratch workspace, but it is not a regular file named '
                'exactly and unambiguously (as plain words, or in closed backticks/quotes), so it cannot be proven '
                'attached before cleanup; name the report exactly')
    if not found:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get('artifacts')
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    known = {_canonical(item) for item in merged if isinstance(item, str) and item.strip()}
    updated['artifacts'] = merged + [path for path in dict.fromkeys(found) if Path(path) not in known]
    return updated


def _scratch_sources(conn, task_id, metadata):
    """The canonical files in the managed scratch workspace that ``metadata`` names: the reports
    cleanup would delete, resolved as the stager resolves them. One that cannot be resolved refuses."""
    workspace = _kb._scratch_workspace(conn, task_id)
    raw = metadata.get('artifacts') if isinstance(metadata, dict) else None
    if workspace is None or not isinstance(raw, (list, tuple)) or not _kb._is_managed_scratch_path(workspace):
        return []
    root, sources = workspace.resolve(), []
    for item in (str(i).strip() for i in raw if isinstance(i, str) and i.strip()):
        try:
            path = Path(item).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise _kb.ArtifactPreservationError(f'cannot resolve declared artifact {item[:300]!r}: {exc}') from exc
        if path.is_relative_to(root):
            sources.append(path)
    return sources


def _digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _attached_as(source, copy):
    """True when ``copy`` is the stager's copy of ``source``: the same name, or the name made unique as
    ``_unique_attachment_path`` does (``stem_N.suffix``)."""
    name = Path(source.name).name or 'artifact'
    stem, suffix = Path(name).stem or 'artifact', Path(name).suffix
    return copy.name == name or (copy.name.startswith(f'{stem}_') and copy.name.endswith(suffix)
                                 and copy.name[len(stem) + 1:len(copy.name) - len(suffix)].isdigit())


def _require_held(sources, held):
    """Raise :class:`ArtifactPreservationError` unless each source is held in ``held`` under its own
    referenced name with its exact bytes: reference identity, not only a matching digest (SI-13)."""
    if not sources:
        return
    digests = {}
    for path in map(Path, held):
        try:
            digests.setdefault(_digest(path), []).append(path)
        except OSError:
            continue
    for source in sources:
        try:
            copies = digests.get(_digest(source), [])
        except OSError as exc:
            raise _kb.ArtifactPreservationError(f'cannot read report {str(source)[:300]!r}: {exc}') from exc
        if not any(_attached_as(source, copy) for copy in copies):
            raise _kb.ArtifactPreservationError(
                f'no attachment holds {str(source)[:300]!r} under its own name with its exact bytes')


def settled_handoff(conn, task_id):
    """``(result, settlement, settled_at)`` when the task's newest completion is a delivered-assessment
    settlement, else None: dependents then read the settled verdict, not an older completed run (SI-02).
    A result edit since the settlement supplies the current handoff ``summary`` and metadata, so the
    handoff never pairs the corrected result with the settled edition's metadata (SI-06) and an
    accepted correction summary reaches dependents as an ordinary edit's run summary does (SI-10)."""
    row = conn.execute(
        "SELECT id, payload, created_at FROM task_events WHERE task_id=? AND kind='completed' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    settlement = _kb._json_dict(row['payload']).get('settlement') if row else None
    if not isinstance(settlement, dict) or settlement.get('kind') != 'delivered_assessment':
        return None
    for edit in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='edited' AND id>? ORDER BY id DESC",
        (task_id, row['id']),
    ):
        correction = _kb._json_dict(edit['payload']).get('settlement_correction')
        if isinstance(correction, dict):
            settlement = {**settlement, 'metadata': correction.get('metadata'), 'summary': correction.get('summary')}
            break
    task = conn.execute('SELECT result FROM tasks WHERE id=?', (task_id,)).fetchone()
    return task['result'], settlement, row['created_at']


def settlement_identity_event_ids(conn, task_id):
    """The events :func:`settled_handoff` reads the task's current handoff from, when its newest
    completion is a delivered-assessment settlement: that ``completed`` event and every
    ``settlement_correction`` edit after it. Empty otherwise (no settlement, or a later ordinary
    completion superseded it). Event retention keeps these however old (KAN-RETENTION-01): without
    them the readers fall back to an older completed run and hand dependents its superseded result."""
    row = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id=? AND kind='completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    settlement = _kb._json_dict(row['payload']).get('settlement') if row else None
    if not isinstance(settlement, dict) or settlement.get('kind') != 'delivered_assessment':
        return []
    corrections = [
        edit['id'] for edit in conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id=? AND kind='edited' AND id>?", (task_id, row['id']))
        if isinstance(_kb._json_dict(edit['payload']).get('settlement_correction'), dict)
    ]
    return [row['id'], *corrections]


def settlement_correction(conn, task_id, summary, metadata):
    """For ``edit_task --result``: the record to append beside the newest completion when that is a
    delivered-assessment settlement, else None (the edit then targets the completed run as before).

    A settlement's handoff is its own record, not a run: correcting it must neither rewrite an older
    completed run nor synthesize one (SI-06). Omitted metadata keeps the current metadata, as an
    ordinary edit keeps its run's."""
    current = settled_handoff(conn, task_id)
    if current is None:
        return None
    return {'summary': summary, 'metadata': current[1].get('metadata') if metadata is None else metadata}


# Late-bound kernel namespace; imported LAST so this module is fully populated first.
from hermes_cli import kanban_db as _kb
