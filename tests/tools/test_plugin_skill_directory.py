"""Plugin directory registration must advertise a loadable, confined skill."""

import json

import pytest

from hermes_cli.plugins import get_plugin_manager
from tools import skills_tool  # registers the real skills handlers
from tools.registry import registry


@pytest.fixture(params=["directory", "directory-str", "file", "file-str"])
def plugin_skill(tmp_path, monkeypatch, request):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)
    (home / "config.yaml").write_text("plugins:\n  enabled: [directory-fixture]\n")
    plugin = home / "plugins" / "directory-fixture"
    root = plugin / "skills" / "guide"
    root.mkdir(parents=True)
    content = "---\nname: guide\ndescription: Fixture guide\n---\nFull guide body.\n"
    (root / "SKILL.md").write_text(content)
    (plugin / "plugin.yaml").write_text("name: directory-fixture\nversion: 1.0.0\n")
    locator = "root / 'SKILL.md'" if request.param.startswith("file") else "root"
    if request.param.endswith("-str"):
        locator = f"str({locator})"
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def register(ctx):\n"
        "    root = Path(__file__).parent / 'skills' / 'guide'\n"
        f"    ctx.register_skill('guide', {locator}, description='Fixture guide')\n"
    )
    manager = get_plugin_manager()
    manager.discover_and_load()
    try:
        yield manager, root, content
    finally:
        manager.unload()


def _call(tool, **arguments):
    return json.loads(registry.dispatch(tool, arguments))


def test_advertised_plugin_directory_loads_complete_skill(plugin_skill):
    manager, root, content = plugin_skill
    listed = _call("skills_list")
    row = next(s for s in listed["skills"] if s["name"] == "directory-fixture:guide")
    result = _call("skill_view", name=row["name"])
    assert result["success"] is True, result
    assert result["content"].endswith(content)
    assert result["skill_dir"] == str(root)
    assert manager.find_plugin_skill(row["name"]) == root / "SKILL.md"


@pytest.mark.platforms("posix")
def test_plugin_supporting_files_stay_inside_registered_skill(plugin_skill, tmp_path):
    _, root, _ = plugin_skill
    reference = root / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("Reference details.")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside skill.")
    (reference.parent / "outside.md").symlink_to(outside)
    name = "directory-fixture:guide"
    main = _call("skill_view", name=name)
    assert main["linked_files"] == {"references": ["references/guide.md"]}
    assert _call("skill_view", name=name, file_path="references/guide.md")["content"] == "Reference details."
    for path in ("../outside.md", str(outside), "references/outside.md", "references"):
        assert _call("skill_view", name=name, file_path=path)["success"] is False


def test_deleted_plugin_index_returns_missing_error(plugin_skill):
    manager, root, _ = plugin_skill
    (root / "SKILL.md").unlink()
    result = _call("skill_view", name="directory-fixture:guide")
    assert result["success"] is False
    assert "no longer exists" in result["error"]
    assert manager.find_plugin_skill("directory-fixture:guide") is None


def test_unreadable_plugin_index_returns_read_error(plugin_skill, monkeypatch):
    from pathlib import Path

    _, root, _ = plugin_skill
    read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == root / "SKILL.md":
            raise PermissionError("fixture read denied")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    result = _call("skill_view", name="directory-fixture:guide")
    assert result["success"] is False
    assert "Failed to read skill 'directory-fixture:guide'" in result["error"]
    assert "fixture read denied" in result["error"]


@pytest.mark.parametrize("shape", ["absent", "empty-directory", "nested-index", "directory-index"])
def test_registration_requires_an_immediate_regular_index(tmp_path, shape):
    from hermes_cli.plugins import PluginContext, PluginManifest

    root = tmp_path / "candidate"
    if shape != "absent":
        root.mkdir()
    if shape == "nested-index":
        (root / "nested").mkdir()
        (root / "nested" / "SKILL.md").write_text("Nested is not the index.")
    if shape == "directory-index":
        (root / "SKILL.md").mkdir()
    manager = get_plugin_manager()
    ctx = PluginContext(PluginManifest(name="invalid-fixture", version="1.0.0"), manager)
    with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
        ctx.register_skill("guide", root)
    assert manager.find_plugin_skill("invalid-fixture:guide") is None


def test_ordinary_file_skill_and_absolute_directory_lookup(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    (root / "ordinary.md").write_text("---\nname: ordinary\n---\nOrdinary file.\n")
    result = _call("skill_view", name="ordinary")
    assert result["success"] is True
    assert result["content"].endswith("Ordinary file.\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("Not registered or in a skill search root.")
    assert _call("skill_view", name=str(outside))["success"] is False
