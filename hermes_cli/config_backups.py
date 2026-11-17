"""Point-in-time copies of ``config.yaml``: one directory, one naming scheme, bounded count.

Every writer that wants a "before" copy of the user's config (setup wizard, corrupt-file
snapshot, model migrations) goes through :func:`backup_config`. Copies live in
``<HERMES_HOME>/backups/config/`` — ``backups/`` is already excluded from full backups, so they
never nest — as ``config.yaml.<reason>.<YYYYMMDD-HHMMSS>``. A copy identical to the newest one
for the same reason is skipped, and only the newest ``keep`` per reason survive, so repeated
``hermes setup`` runs or a gateway restarting against broken YAML cannot litter the home dir.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BACKUPS_SUBDIR = Path("backups") / "config"
DEFAULT_KEEP = 5

# Names earlier code wrote next to config.yaml (setup wizard, corrupt snapshot, xai migration).
# Moved into the backups dir on first use so they stop accumulating in the home root; hand-named
# copies (``config.yaml.bak-my-note``) are the user's and are never touched.
_LEGACY_SIBLING_GLOBS = ("config.yaml.bak.[0-9]*", "config.yaml.corrupt.*.bak", "config.yaml.bak-pre-migrate-*")


def backups_dir(config_path: Path) -> Path:
    return config_path.parent / BACKUPS_SUBDIR


def list_config_backups(config_path: Path, reason: Optional[str] = None) -> list[Path]:
    """Existing backups, newest first; filtered to one *reason* when given."""
    root = backups_dir(config_path)
    if not root.is_dir():
        return []
    prefix = f"{config_path.name}.{reason}." if reason else f"{config_path.name}."
    return sorted((p for p in root.iterdir() if p.is_file() and p.name.startswith(prefix)),
                  key=lambda p: p.name, reverse=True)


def backup_config(config_path: Path, reason: str, *, keep: int = DEFAULT_KEEP) -> Optional[Path]:
    """Copy *config_path* to the backups dir; return the new path, or None when skipped/failed.

    Skips when the file is missing/empty, or when the newest backup for *reason* already holds
    identical bytes. Never raises: a failed backup must not block the write it precedes.
    Skipped entirely for a config backend without file tooling (no local copies, D2).
    """
    from hermes_cli.config_backend import supports_file_tooling
    if not supports_file_tooling():
        return None
    try:
        if not config_path.is_file() or config_path.stat().st_size == 0:  # config-reader: ok — file tooling, gated above
            return None
        root = backups_dir(config_path)
        root.mkdir(parents=True, exist_ok=True)
        _sweep_legacy_siblings(config_path, root)
        existing = list_config_backups(config_path, reason)
        # Not filecmp.cmp: its process-wide cache keys on (size, mtime), so an equal-size edit
        # within one mtime tick of an earlier comparison is reported identical and never backed up.
        data = config_path.read_bytes()  # config-reader: ok — file tooling, gated above
        if existing and data == existing[0].read_bytes():
            return None
        dest = root / f"{config_path.name}.{reason}.{time.strftime('%Y%m%d-%H%M%S')}"
        if dest.is_symlink() or dest.exists():  # never write through a planted link
            return None
        _write_copy(config_path, dest, data)
        for stale in [dest, *existing][keep:]:
            stale.unlink(missing_ok=True)
        return dest
    except OSError as exc:
        logger.warning("Could not back up %s (%s): %s", config_path, reason, exc)
        return None


def _write_copy(config_path: Path, dest: Path, data: bytes) -> None:
    """``shutil.copy2`` for a small file, as plain writes of the bytes already read.

    Not ``copy2``/``copyfile``: on Linux it hands the data to ``copy_file_range(2)``, which a
    filesystem may serve as a block clone. OpenZFS with ``zfs_bclone_wait_dirty=1`` then WAITS for
    the transaction group holding the just-written source to sync before cloning it. A ``good``
    backup copies a config that was just edited, under ``config._CONFIG_LOCK`` and often on the
    event loop, so that wait (10-15 s seen under load) stalled every caller of the config. A plain
    write does not wait for that sync. Metadata is copied as ``copy2`` does (``shutil.copystat``); that
    copy is sequential, so a failure part-way can leave the new file with some metadata applied.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(dest, flags, 0o600)  # O_EXCL: never through a planted link
    try:  # raw writes: this function alone owns fd, so it is closed exactly once
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)
    shutil.copystat(config_path, dest)


def load_newest_good_backup(config_path: Path) -> Optional[dict]:
    """Parse the newest ``good`` backup (the file as it was at the last successful load).

    Returns the raw mapping, or None when there is no usable copy. Older ``good`` copies are not
    tried: a backup that fails to parse means the on-disk copy was damaged after the fact, and
    guessing further back would serve a config the user never saw as current.
    """
    from hermes_cli.config_backend import supports_file_tooling
    if not supports_file_tooling():
        return None
    newest = list_config_backups(config_path, "good")[:1]
    if not newest:
        return None
    try:
        from utils import fast_safe_load
        with newest[0].open(encoding="utf-8") as f:
            data = fast_safe_load(f)
    except Exception as exc:
        logger.warning("Last-known-good backup %s is unreadable: %s", newest[0], exc)
        return None
    return data if isinstance(data, dict) else None


def _sweep_legacy_siblings(config_path: Path, root: Path) -> None:
    for pattern in _LEGACY_SIBLING_GLOBS:
        for old in config_path.parent.glob(pattern):
            if not old.is_file() or old.is_symlink():
                continue
            try:
                old.replace(root / old.name)
            except OSError as exc:
                logger.debug("Could not move legacy backup %s: %s", old, exc)
