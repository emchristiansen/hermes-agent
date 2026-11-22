"""Declared handoff artifacts: which declared files a completion or review handoff copies into the
task's attachments, binding vetted outside files to the object that was checked, and the copies.

Split out of ``hermes_cli.kanban_db``. Facade-resident names are reached through ``_kb``, imported
inside each function that needs it (never at module level, which would form an import cycle with the
facade's own tail import of this module), so monkeypatching ``kanban_db.<name>`` keeps working and
this module imports cleanly on its own.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

def _cleaned_artifact_paths(metadata: Any) -> list[str]:
    """Non-blank string paths declared in ``metadata["artifacts"]``."""
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("artifacts")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(p).strip() for p in raw if isinstance(p, str) and str(p).strip()]


def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: Optional[dict], *, summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Legacy workers named deliverables only by absolute path in prose; add
    those that exist under the scratch workspace to ``metadata["artifacts"]``
    before cleanup can erase them."""
    from hermes_cli import kanban_db as _kb
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return metadata
    if not _kb._is_managed_scratch_path(workspace):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        if candidate.is_file():
            discovered.append(str(candidate))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated


@dataclass(frozen=True)
class VettedArtifact:
    """A declared completion artifact outside managed scratch that the caller's file-sending
    policy allowed (``tools.kanban_tools``), bound to the file object it vetted.

    ``path`` is the resolved host path; ``device``/``inode`` identify the regular file found there
    when it was vetted. :func:`_copy_vetted_artifact` copies only that same object, so a path
    swapped after vetting is refused rather than read."""

    path: Path
    device: int
    inode: int

    @classmethod
    def bind(cls, resolved: str) -> VettedArtifact:
        """Bind the regular file now at ``resolved`` (an already symlink-resolved path); raises
        :class:`_kb.ArtifactPreservationError` if it is not one."""
        from hermes_cli import kanban_db as _kb
        try:
            st = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise _kb.ArtifactPreservationError(f"declared artifact {resolved} cannot be read: {exc}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise _kb.ArtifactPreservationError(f"declared artifact is not a regular file: {resolved}")
        return cls(Path(resolved), st.st_dev, st.st_ino)


def external_completion_artifacts(conn: sqlite3.Connection, task_id: str, metadata: Any) -> list[str]:
    """Declared ``metadata["artifacts"]`` entries a handoff can only keep by copying them from
    outside the task's managed scratch workspace: neither inside that workspace (staged here
    unconditionally) nor already one of this task's stored attachments. The caller must vet each
    one and pass it as a :class:`VettedArtifact`, or refuse the handoff."""
    from hermes_cli import kanban_db as _kb
    declared = _cleaned_artifact_paths(metadata)
    if not declared:
        return []
    scratch_root: Optional[Path] = None
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is not None and _kb._is_managed_scratch_path(workspace):
        with contextlib.suppress(OSError):
            scratch_root = workspace.resolve()
    stored = {
        row["stored_path"] for row in conn.execute(
            "SELECT stored_path FROM task_attachments WHERE task_id = ?", (task_id,))
    }
    external: list[str] = []
    for artifact in dict.fromkeys(declared):
        try:
            resolved = Path(artifact).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            external.append(artifact)
            continue
        if scratch_root is not None and resolved.is_relative_to(scratch_root):
            continue
        if str(resolved) in stored:
            continue
        external.append(artifact)
    return external


def _persist_scratch_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict, *,
    vetted: Mapping[str, VettedArtifact] = MappingProxyType({}),
) -> None:
    """Copy declared completion artifacts into the task's attachments before cleanup removes
    them: every one inside the managed scratch workspace, and each outside one the caller vetted
    (keyed by its declared spelling). Any other declared path that is not already one of this
    task's stored attachments stays recorded as given, and is also listed with its reason in
    ``metadata["unattached_artifacts"]`` so the handoff never reads as delivered without it."""
    from hermes_cli import kanban_db as _kb
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return
    unattached = [
        {"path": path, "reason": _UNATTACHED_REASON}
        for path in external_completion_artifacts(conn, task_id, metadata) if path not in vetted
    ]
    if unattached:
        metadata["unattached_artifacts"] = unattached

    workspace_root: Optional[Path] = None
    board: Optional[str] = None
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is not None:
        is_managed, managed_board = _kb._managed_scratch_path_info(workspace)
        if is_managed:
            with contextlib.suppress(OSError):
                workspace_root = workspace.resolve()
                board = managed_board
    if workspace_root is None and not vetted:
        return

    attachment_dir = _kb.task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        _discard_staged_copies(used_destinations, attachment_dir)

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        vetted_artifact = vetted.get(artifact)
        if vetted_artifact is not None:
            source: Path = vetted_artifact.path
        else:
            if workspace_root is None:
                persisted.append(artifact)
                continue
            src = Path(artifact).expanduser()
            try:
                resolved_src = src.resolve()
            except OSError:
                persisted.append(artifact)
                continue

            if not resolved_src.is_relative_to(workspace_root):
                persisted.append(artifact)
                continue

            problem = None
            if not src.is_file():
                problem = f"declared scratch artifact is unavailable or not a regular file: {artifact}"
            elif resolved_src.stat().st_size > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
                problem = (
                    f"declared scratch artifact exceeds the "
                    f"{_kb.KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
                )
            if problem:
                _discard_copies()
                raise _kb.ArtifactPreservationError(problem)
            source = resolved_src

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = _kb._unique_attachment_path(attachment_dir, source.name, used_destinations)
            if vetted_artifact is not None:
                _copy_vetted_artifact(vetted_artifact, dest, artifact)
            else:
                _kb._copy_capped(source, dest, artifact)
        except Exception as exc:
            if dest is not None:
                with contextlib.suppress(OSError):
                    dest.unlink(missing_ok=True)
            _discard_copies()
            if isinstance(exc, _kb.ArtifactPreservationError):
                raise
            raise _kb.ArtifactPreservationError(
                f"could not preserve declared artifact {artifact}: {exc}"
            ) from exc
        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        # Only this call's own new copies: a registered attachment passed through is not ours to discard.
        copied = {str(dest.resolve()) for dest in used_destinations}
        metadata["_staged_artifacts"] = [path for path in persisted if path in copied]


_UNATTACHED_REASON = (
    "outside the task's managed scratch workspace and not vetted by the worker handoff tools, "
    "so it was recorded as a path only, not copied into the task's attachments"
)


def _copy_vetted_artifact(vetted: VettedArtifact, dest: Path, artifact: str) -> None:
    """Copy exactly the file object ``vetted`` bound: opened without following a final symlink,
    and refused unless it is still that regular file (same device and inode) within the cap."""
    from hermes_cli import kanban_db as _kb
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(vetted.path, flags)
    except OSError as exc:
        raise _kb.ArtifactPreservationError(f"declared artifact {artifact} cannot be opened: {exc}") from exc
    with os.fdopen(fd, "rb") as source_file:
        st = os.fstat(source_file.fileno())
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (vetted.device, vetted.inode):
            raise _kb.ArtifactPreservationError(
                f"declared artifact {artifact} changed after it was checked; nothing was copied")
        if st.st_size > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
            raise _kb.ArtifactPreservationError(
                f"declared artifact exceeds the {_kb.KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}")
        with dest.open("xb") as destination_file:
            copied = 0
            while chunk := source_file.read(1024 * 1024):
                copied += len(chunk)
                if copied > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
                    raise _kb.ArtifactPreservationError(
                        f"declared artifact grew beyond the size limit: {artifact}")
                destination_file.write(chunk)


def _discard_staged_copies(copies: Iterable[Path], attachment_dir: Path) -> None:
    """Remove staged attachment copies whose DB rows never committed; a leaked
    copy would make the retry stage ``name_1.ext`` next to an orphan."""
    for copied in copies:
        with contextlib.suppress(OSError):
            Path(copied).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        attachment_dir.rmdir()


def _copy_capped(src: Path, dest: Path, artifact: str) -> None:
    """Chunked copy that aborts if the file grows past the attachment cap mid-copy."""
    from hermes_cli import kanban_db as _kb
    with src.open("rb") as source_file, dest.open("xb") as destination_file:
        copied = 0
        while chunk := source_file.read(1024 * 1024):
            copied += len(chunk)
            if copied > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
                raise _kb.ArtifactPreservationError(
                    f"declared scratch artifact grew beyond the size limit: {artifact}"
                )
            destination_file.write(chunk)


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    stem, suffix = Path(safe_name).stem or "artifact", Path(safe_name).suffix
    candidate = directory / safe_name
    idx = 1
    while candidate in used or candidate.exists():
        candidate = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate

