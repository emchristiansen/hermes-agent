"""Consumers whose contract says a config read failure fails closed also fail closed on ``FailedConfigRead``.

``load_config`` never raises on a broken config.yaml: it serves defaults (cold) or the last good config
(hot) as a ``FailedConfigRead`` so ordinary readers keep working. Two consumers promise a fixed
fail-closed value on a read failure and must not take a value from that fallback (the same pattern as
SKL-LKG-1 in ``gateway.kanban_watchers_common``):

- ``tools.file_tools_write_guards._protected_instruction_config``: "Config read failures keep the gate ON";
- ``hermes_cli.tools_config_cua._cua_autostart_opt_in``: "Unreadable config fails closed to the on-demand default".

The protected-instruction gate forces only its ON switch: extra patterns served by the fallback (the last good
user config, the current managed overlay) only add protection, so they keep gating writes (SKL-LKG-2). The
write/patch cases run the real ``write_file_tool`` / ``patch_tool`` entries with only the human approval prompt
replaced by a recording denial.

Real loader, real files; valid-config controls retained. The loader's last-good fallback itself is unchanged.
"""
import json

import pytest

from tests.hermes_cli.test_managed_scope_lkg_retention import _reset_caches

BROKEN = "security: [\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _reset_caches()
    yield home
    _reset_caches()


def _break(home, warm_yaml, read):
    """Optionally load *warm_yaml* first (hot last-good), then break the file; returns the fallback."""
    from hermes_cli.config import load_config
    from hermes_cli.config_read_errors import FailedConfigRead
    if warm_yaml is not None:
        (home / "config.yaml").write_text(warm_yaml)
        read()
    (home / "config.yaml").write_text(BROKEN)
    served = load_config()
    assert isinstance(served, FailedConfigRead)
    return served


@pytest.mark.parametrize("warm", [False, True], ids=["cold-defaults", "hot-last-good-off"])
def test_protected_instruction_gate_stays_on_for_failed_read(home, warm):
    from tools.file_tools_write_guards import _protected_instruction_config
    warm_yaml = ("security:\n  protected_instruction_files: false\n"
                 "  protected_instruction_extra_patterns: ['*.mdc']\n") if warm else None
    if warm:
        (home / "config.yaml").write_text(warm_yaml)
        assert _protected_instruction_config() == (False, ["*.mdc"])  # valid control: the user's choice
    served = _break(home, warm_yaml, _protected_instruction_config)
    observed = _protected_instruction_config()
    print("REACHED-CONSUMER protected_instruction", warm, type(served).__name__, observed, flush=True)
    if warm:
        assert served["security"]["protected_instruction_files"] is False  # last good still served
        assert observed == (True, ["*.mdc"])  # switch forced on, last good restriction kept
    else:
        assert observed == (True, [])


def test_protected_instruction_gate_valid_controls(home):
    from tools.file_tools_write_guards import _protected_instruction_config
    assert _protected_instruction_config() == (True, [])  # no file: default on
    (home / "config.yaml").write_text("security:\n  protected_instruction_extra_patterns: ['*.mdc']\n")
    assert _protected_instruction_config() == (True, ["*.mdc"])
    (home / "config.yaml").write_text("security:\n  protected_instruction_files: false\n")
    assert _protected_instruction_config() == (False, [])


PATTERN_YAML = ("security:\n  protected_instruction_files: true\n"
                "  protected_instruction_extra_patterns: ['*.mdc']\n")


@pytest.fixture
def denial(monkeypatch):
    """Replace only the human prompt; classification, config reading and the precheck stay real."""
    import tools.file_tools_write_guards as guards
    calls = []

    def deny(reasons, task_id="default"):
        calls.append(list(reasons))
        return "BLOCKED: recorded denial"
    monkeypatch.setattr(guards, "_request_protected_instruction_approval", deny)
    return calls


def _write(path, content):
    from tools.file_tools import write_file_tool
    return json.loads(write_file_tool(str(path), content))


def _patch(path, old, new):
    from tools.file_tools import patch_tool
    return json.loads(patch_tool(mode="replace", path=str(path), old_string=old, new_string=new))


@pytest.mark.parametrize("entry", ["write_file", "patch"])
@pytest.mark.parametrize("source", ["user-lkg", "current-managed"])
def test_extra_patterns_keep_gating_writes_after_failed_read(home, tmp_path, denial, source, entry):
    from hermes_cli.config import load_config
    managed = tmp_path / "managed"
    project = tmp_path / "project"
    project.mkdir()
    target = project / "rules.mdc"
    target.write_text("original instructions\n")
    (home / "config.yaml").write_text(PATTERN_YAML if source == "user-lkg" else "model:\n  default: x/y\n")
    load_config()
    if source == "current-managed":
        (managed / "config.yaml").write_text(PATTERN_YAML)
    attempt = (lambda: _write(target, "replaced\n")) if entry == "write_file" else (
        lambda: _patch(target, "original", "replaced"))

    assert "recorded denial" in attempt()["error"]  # valid control: the pattern gates the write
    assert denial == [["rules.mdc"]]
    denial.clear()
    served = _break(home, None, load_config)
    assert served["security"]["protected_instruction_extra_patterns"] == ["*.mdc"]
    res = attempt()
    print("REACHED-PRECHECK extra_pattern", source, entry, type(served).__name__, res, denial, flush=True)
    assert "recorded denial" in res["error"]
    assert denial == [["rules.mdc"]]
    assert target.read_text() == "original instructions\n"


@pytest.mark.parametrize("warm", [False, True], ids=["cold-defaults", "hot-last-good-off"])
def test_builtin_names_keep_gating_writes_after_failed_read(home, tmp_path, denial, warm):
    from hermes_cli.config import load_config
    warm_yaml = "security:\n  protected_instruction_files: false\n" if warm else None
    target = tmp_path / "project" / "AGENTS.md"
    target.parent.mkdir()
    if warm:
        (home / "config.yaml").write_text(warm_yaml)
        assert _write(target, "first\n").get("error") is None  # valid control: the user turned the gate off
        assert denial == []
    _break(home, warm_yaml, load_config)
    before = target.read_bytes() if target.exists() else None
    res = _write(target, "second\n")
    print("REACHED-PRECHECK builtin", warm, res, denial, flush=True)
    assert "recorded denial" in res["error"]
    assert denial == [["AGENTS.md"]]
    assert (target.read_bytes() if target.exists() else None) == before


def test_multi_file_patch_stays_all_or_nothing_after_failed_read(home, tmp_path, denial):
    """One protected target refuses the whole V4A patch; the ordinary file in it is not written either."""
    from hermes_cli.config import load_config
    from tools.file_tools import patch_tool
    project = tmp_path / "project"
    project.mkdir()
    notes, rules = project / "notes.txt", project / "rules.mdc"
    notes.write_text("ordinary\n")
    rules.write_text("original instructions\n")
    (home / "config.yaml").write_text(PATTERN_YAML)
    load_config()
    _break(home, None, load_config)

    def v4a(*files):
        return "*** Begin Patch\n" + "".join(
            f"*** Update File: {f}\n@@\n-{old}\n+{new}\n" for f, old, new in files) + "*** End Patch"
    res = json.loads(patch_tool(mode="patch", patch=v4a((notes, "ordinary", "changed"),
                                                        (rules, "original instructions", "replaced"))))
    print("REACHED-PRECHECK multi_file", res, denial, flush=True)
    assert "recorded denial" in res["error"]
    assert denial == [["rules.mdc"]]
    assert notes.read_text() == "ordinary\n"
    assert rules.read_text() == "original instructions\n"
    res = json.loads(patch_tool(mode="patch", patch=v4a((notes, "ordinary", "changed"))))
    assert res.get("error") is None  # control: the ordinary file alone still patches
    assert denial == [["rules.mdc"]]
    assert notes.read_text() == "changed\n"


def test_ordinary_files_stay_ungated_after_failed_read(home, tmp_path, denial):
    from hermes_cli.config import load_config
    target = tmp_path / "project" / "notes.txt"
    target.parent.mkdir()
    (home / "config.yaml").write_text(PATTERN_YAML)
    load_config()
    _break(home, None, load_config)
    res = _write(target, "ordinary\n")
    assert res.get("error") is None
    assert denial == []
    assert target.read_text() == "ordinary\n"


@pytest.mark.parametrize("warm", [False, True], ids=["cold-defaults", "hot-last-good-true"])
def test_cua_autostart_fails_closed_for_failed_read(home, warm):
    from hermes_cli.tools_config_cua import _cua_autostart_opt_in
    warm_yaml = "computer_use:\n  autostart: true\n" if warm else None
    if warm:
        (home / "config.yaml").write_text(warm_yaml)
        assert _cua_autostart_opt_in() is True  # valid control: explicit opt-in
    served = _break(home, warm_yaml, _cua_autostart_opt_in)
    observed = _cua_autostart_opt_in()
    print("REACHED-CONSUMER cua_autostart", warm, type(served).__name__, observed, flush=True)
    if warm:
        assert served["computer_use"]["autostart"] is True  # last good still served
    assert observed is False


def test_cua_autostart_valid_controls(home):
    from hermes_cli.tools_config_cua import _cua_autostart_opt_in
    assert _cua_autostart_opt_in() is False
    (home / "config.yaml").write_text("computer_use:\n  autostart: false\n")
    assert _cua_autostart_opt_in() is False
    (home / "config.yaml").write_text("computer_use:\n  autostart: true\n")
    assert _cua_autostart_opt_in() is True
