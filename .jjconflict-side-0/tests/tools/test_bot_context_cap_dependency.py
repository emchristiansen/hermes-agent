"""A cap-only configuration edit must refresh an eternal Bot Chat once."""

from tools.bot_mode_probe import (
    capability_fingerprint,
    epoch_line,
    stored_prompt_capability_stale,
)


def test_cap_only_change_invalidates_stored_prompt_once(tmp_path):
    home = tmp_path / "bot"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("context_file_max_chars: 138728\n", encoding="utf-8")
    before = capability_fingerprint(home)
    stored = "Bot Chat system prompt\n" + epoch_line(home)
    assert not stored_prompt_capability_stale(stored, home)

    config.write_text("context_file_max_chars: 138976\n", encoding="utf-8")
    assert capability_fingerprint(home) != before
    assert stored_prompt_capability_stale(stored, home)
    refreshed = "Bot Chat system prompt\n" + epoch_line(home)
    assert not stored_prompt_capability_stale(refreshed, home)
    assert capability_fingerprint(home) == capability_fingerprint(home)


def test_equal_loader_caps_share_dependency(tmp_path):
    from agent.prompt_builder import _get_context_file_max_chars
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "bot"
    home.mkdir()
    config = home / "config.yaml"
    token = set_hermes_home_override(str(home))
    try:
        # Preserve canonical loader semantics, including float truncation and bool-as-int.
        for spellings in (("120000", "120000.9"), ("true", "1"), ("0.5", "0.9"),
                          ("null", "0", "-4", "false", "'120000'", "[]", "{}")):
            caps, epochs = [], []
            for spelling in spellings:
                config.write_text(f"context_file_max_chars: {spelling}\n", encoding="utf-8")
                caps.append(_get_context_file_max_chars(200000))
                epochs.append(capability_fingerprint(home))
            assert len(set(caps)) == 1
            assert len(set(epochs)) == 1
    finally:
        reset_hermes_home_override(token)


def test_dynamic_policy_does_not_guess_a_session_window(tmp_path):
    from agent.prompt_builder import _get_context_file_max_chars
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "bot"
    home.mkdir()
    config = home / "config.yaml"
    token = set_hermes_home_override(str(home))
    try:
        config.write_text("context_file_max_chars: null\n", encoding="utf-8")
        dynamic = capability_fingerprint(home)
        floor = _get_context_file_max_chars()
        assert _get_context_file_max_chars(200000) > floor
        assert _get_context_file_max_chars(32768) == floor
        # Same as the unknown-window floor, but NOT the same policy for a large model.
        config.write_text(f"context_file_max_chars: {floor}\n", encoding="utf-8")
        assert capability_fingerprint(home) != dynamic
        assert _get_context_file_max_chars(200000) == floor
        config.write_text("model:\n  context_length: 200000\n", encoding="utf-8")
        larger = capability_fingerprint(home)
        config.write_text("model:\n  context_length: 32768\n", encoding="utf-8")
        assert capability_fingerprint(home) != larger
    finally:
        reset_hermes_home_override(token)


def test_cap_dependency_is_profile_scoped(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    a, b = (root / "profiles" / name for name in ("editor", "other"))
    for home in (root, a, b):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("context_file_max_chars: 30000\n", encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(str(a))
    try:
        before = {home: capability_fingerprint(home) for home in (root, a, b)}
        (a / "config.yaml").write_text("context_file_max_chars: 40000\n", encoding="utf-8")
        assert capability_fingerprint(a) != before[a]
        assert capability_fingerprint(b) == before[b]
        assert capability_fingerprint(root) == before[root]
        assert get_hermes_home() == a
    finally:
        reset_hermes_home_override(token)


def test_cap_dependency_obeys_managed_precedence_and_environment_expansion(tmp_path, monkeypatch):
    from agent.prompt_builder import _get_context_file_max_chars
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home, managed = tmp_path / "bot", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    config, overlay = home / "config.yaml", managed / "config.yaml"
    config.write_text("context_file_max_chars: 30000\n", encoding="utf-8")
    overlay.write_text("context_file_max_chars: 40000\n", encoding="utf-8")
    token = set_hermes_home_override(str(home))
    try:
        initial = capability_fingerprint(home)
        assert _get_context_file_max_chars(200000) == 40000
        config.write_text("context_file_max_chars: 50000\n", encoding="utf-8")
        assert capability_fingerprint(home) == initial
        overlay.write_text("context_file_max_chars: 60000\n", encoding="utf-8")
        assert capability_fingerprint(home) != initial
        assert _get_context_file_max_chars(200000) == 60000
        # Expanded numeric strings stay strings, hence use dynamic fallback like the loader.
        overlay.write_text("context_file_max_chars: ${CAP_FIXTURE}\n", encoding="utf-8")
        monkeypatch.setenv("CAP_FIXTURE", "70000")
        expanded = capability_fingerprint(home)
        fallback = _get_context_file_max_chars(200000)
        monkeypatch.setenv("CAP_FIXTURE", "80000")
        assert _get_context_file_max_chars(200000) == fallback
        assert capability_fingerprint(home) == expanded
    finally:
        reset_hermes_home_override(token)
