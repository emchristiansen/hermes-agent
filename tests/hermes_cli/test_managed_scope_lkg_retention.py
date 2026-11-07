"""A managed ``sessions.auto_prune: false`` must win over a recovered last-known-good config.

Regression for RETENTION-LKG-01: a long-lived process cached ``auto_prune: true``, the user's
config.yaml then became invalid YAML, and the administrator policy appeared. The in-memory
last-known-good fallback returned the stale mapping without the CURRENT managed overlay (and
cached it under the new managed signature), so real startup maintenance kept pruning: it deleted
the session row and its transcript. Everything below goes through the real ``load_config``,
``load_user_config_effective``, ``_run_state_db_auto_maintenance`` and ``SessionDB`` — nothing in
the loader, maintenance or store is mocked.
"""
import sqlite3
import textwrap

import pytest

USER = textwrap.dedent("""\
    sessions:
      auto_prune: true
      auto_archive: false
      retention_days: 90
      vacuum_after_prune: false
    model:
      default: synthetic/model
    """)
POLICY = "sessions:\n  auto_prune: false\n"
BROKEN = "sessions: [\n"


def _reset_caches():
    import hermes_cli.config as cfg
    import hermes_cli.config_effective as eff
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    cfg._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    eff._EFFECTIVE_CACHE.clear()
    eff._LAST_GOOD_USER_RAW.clear()
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_caches()
    yield home, managed
    _reset_caches()


def _auto_prune():
    from hermes_cli.config import load_config
    from hermes_cli.config_effective import load_user_config_effective

    return (load_config()["sessions"]["auto_prune"],
            load_user_config_effective()["sessions"]["auto_prune"])


def _maintain_witness(home):
    """Seed an eligible (backdated, ended) session + transcript, run the real maintenance, and
    report whether the row and the transcript survived."""
    from hermes_cli.cli_auto_maintenance import _run_state_db_auto_maintenance
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("old", source="cli")
        db.append_message("old", role="user", content="synthetic retention witness")
        db.end_session("old", "done")
        # Latch the unrelated one-time repairs so only the retention gate decides.
        db.set_meta("ghost_session_prune_v1", "1")
        db.set_meta("orphaned_compression_finalize_v1", "1")
        with sqlite3.connect(home / "state.db") as conn:
            conn.execute("UPDATE sessions SET started_at=1, ended_at=1, last_activity_at=1")
            conn.execute("UPDATE messages SET timestamp=1")
        (home / "sessions").mkdir(exist_ok=True)
        transcript = home / "sessions" / "old.jsonl"
        transcript.write_text('{"synthetic":true}\n')
        _run_state_db_auto_maintenance(db)
        return db.get_session("old") is not None, transcript.exists()
    finally:
        db.close()


def test_control_no_policy_maintenance_deletes(homes):
    """Positive control: without a policy the fixture really is pruned (row + transcript)."""
    home, _ = homes
    (home / "config.yaml").write_text(USER)
    assert _auto_prune() == (True, True)
    assert _maintain_witness(home) == (False, False)


def test_control_policy_from_start_preserves(homes):
    home, managed = homes
    (managed / "config.yaml").write_text(POLICY)
    (home / "config.yaml").write_text(USER)
    assert _auto_prune() == (False, False)
    assert _maintain_witness(home) == (True, True)


def test_control_warm_valid_yaml_policy_appears(homes):
    """Warm cache, valid YAML: adding the policy flips the cached true to false."""
    from hermes_cli.config import load_config

    home, managed = homes
    (home / "config.yaml").write_text(USER)
    assert load_config()["sessions"]["auto_prune"] is True
    (managed / "config.yaml").write_text(POLICY)
    assert _auto_prune() == (False, False)
    cfg = load_config()
    assert cfg["sessions"]["retention_days"] == 90
    assert cfg["model"]["default"] == "synthetic/model"
    assert _maintain_witness(home) == (True, True)


def test_policy_added_while_user_yaml_invalid_preserves_session(homes):
    """RETENTION-LKG-01: the in-memory last-known-good must get the CURRENT managed overlay."""
    from hermes_cli.config import load_config
    from hermes_cli.config_read_errors import FailedConfigRead

    home, managed = homes
    (home / "config.yaml").write_text(USER)
    assert load_config()["sessions"]["auto_prune"] is True  # long-lived process caches true
    (home / "config.yaml").write_text(BROKEN)
    (managed / "config.yaml").write_text(POLICY)

    cfg = load_config()
    loaders = _auto_prune()
    # Real maintenance in the same process; observed together so a failure shows the deletion.
    row_and_transcript = _maintain_witness(home)
    assert (loaders, row_and_transcript) == ((False, False), (True, True))
    assert isinstance(cfg, FailedConfigRead)  # still served from last-known-good, not defaults
    # Sibling leaves come from the recovered user config; only the managed leaf is replaced.
    assert cfg["sessions"]["retention_days"] == 90
    assert cfg["sessions"]["vacuum_after_prune"] is False
    assert cfg["model"]["default"] == "synthetic/model"
    # The cached fallback (hit on the next call) carries the same overlay.
    assert load_config()["sessions"]["auto_prune"] is False


def test_policy_changed_while_user_yaml_invalid_tracks_current_overlay(homes):
    """The recovered config reflects the overlay applied now, not the one at caching time:
    in-memory recovery agrees with the fresh-process (backup) recovery as the policy changes."""
    from hermes_cli.config import load_config

    home, managed = homes
    (managed / "config.yaml").write_text("sessions:\n  retention_days: 30\n")
    (home / "config.yaml").write_text(USER)
    assert load_config()["sessions"]["retention_days"] == 30
    (home / "config.yaml").write_text(BROKEN)
    (managed / "config.yaml").write_text(POLICY)
    cfg = load_config()
    assert cfg["sessions"]["auto_prune"] is False
    assert cfg["sessions"]["retention_days"] == 90  # the old managed 30 is no longer applied
    _reset_caches()  # restart-equivalent: recover from the on-disk good backup instead
    cold = load_config()
    assert cold["sessions"]["auto_prune"] is False
    assert cold["sessions"]["retention_days"] == 90


def test_control_cold_cache_backup_recovery_applies_policy(homes):
    """Fresh-process control: recovery from backups/config/ applies the current policy."""
    from hermes_cli.config import load_config
    from hermes_cli.config_read_errors import FailedConfigRead

    home, managed = homes
    (home / "config.yaml").write_text(USER)
    assert load_config()["sessions"]["auto_prune"] is True  # leaves the good backup
    (home / "config.yaml").write_text(BROKEN)
    (managed / "config.yaml").write_text(POLICY)
    _reset_caches()
    cfg = load_config()
    assert isinstance(cfg, FailedConfigRead)
    assert cfg["sessions"]["auto_prune"] is False
    assert cfg["sessions"]["retention_days"] == 90
    assert cfg["model"]["default"] == "synthetic/model"
    assert _maintain_witness(home) == (True, True)
