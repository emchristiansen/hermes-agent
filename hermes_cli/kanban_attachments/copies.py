"""Copying a declared completion artifact into a task's attachments, within the one size cap.

Moved unchanged out of ``hermes_cli/kanban_db.py`` (its completion path calls these): the attachment
cap, the preservation error, a vetted outside artifact bound to the file object it was checked as,
the capped and vetted copies, the discard of staged copies whose rows never committed, and the
collision-free destination name.
"""

from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KANBAN_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024  # one cap for dashboard, tools and CLI


class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""


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
        :class:`ArtifactPreservationError` if it is not one."""
        try:
            st = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactPreservationError(f"declared artifact {resolved} cannot be read: {exc}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise ArtifactPreservationError(f"declared artifact is not a regular file: {resolved}")
        return cls(Path(resolved), st.st_dev, st.st_ino)


def _copy_vetted_artifact(vetted: VettedArtifact, dest: Path, artifact: str) -> None:
    """Copy exactly the file object ``vetted`` bound: opened without following a final symlink,
    and refused unless it is still that regular file (same device and inode) within the cap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(vetted.path, flags)
    except OSError as exc:
        raise ArtifactPreservationError(f"declared artifact {artifact} cannot be opened: {exc}") from exc
    with os.fdopen(fd, "rb") as source_file:
        st = os.fstat(source_file.fileno())
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (vetted.device, vetted.inode):
            raise ArtifactPreservationError(
                f"declared artifact {artifact} changed after it was checked; nothing was copied")
        if st.st_size > KANBAN_ATTACHMENT_MAX_BYTES:
            raise ArtifactPreservationError(
                f"declared artifact exceeds the {KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}")
        with dest.open("xb") as destination_file:
            copied = 0
            while chunk := source_file.read(1024 * 1024):
                copied += len(chunk)
                if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                    raise ArtifactPreservationError(
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
    with src.open("rb") as source_file, dest.open("xb") as destination_file:
        copied = 0
        while chunk := source_file.read(1024 * 1024):
            copied += len(chunk)
            if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                raise ArtifactPreservationError(
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
