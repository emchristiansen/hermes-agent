"""``${VAR}`` expansion in a profile scope uses the ONE read the scope came from (review22815 MERGE-FLAGS-2).

``build_profile_secret_scope`` reads ``<home>/.env`` and decides "is this the launch home?" from one opened
directory. Template expansion must expand those bytes under that verdict: re-deciding identity and reopening
the file by path let a home retargeted after the read put the launch home's file, resolved against the
launch env, into a scope whose verdict says it is NOT the launch home. Real file readers and resolver;
only the retarget timing is injected, as in the review's witness. Ordinary defaults, launch-home
resolution and managed precedence are kept.
"""
import pytest

from agent import secret_scope as ss


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import hermes_constants
    from gateway import config_loader
    from hermes_cli import env_loader, managed_scope

    launch, other = tmp_path / "launch", tmp_path / "other"
    launch.mkdir()
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(hermes_constants, "get_routing_process_hermes_home", lambda: launch)
    monkeypatch.setattr(managed_scope, "load_managed_env", dict)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    monkeypatch.setattr(env_loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(config_loader, "bridged_allow_all_users", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-LAUNCH-KEY")
    return launch, other, alias


def _retarget_after_read(monkeypatch, alias, to):
    original = ss._read_profile_home

    def read_then_retarget(home):
        result = original(home)
        alias.unlink()
        alias.symlink_to(to, target_is_directory=True)
        return result

    monkeypatch.setattr(ss, "_read_profile_home", read_then_retarget)


def test_a_secondary_takes_its_default_and_the_launch_home_its_key(homes):
    launch, other, alias = homes
    (launch / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\n")
    (other / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\n")
    secondary = ss.build_profile_secret_scope(alias)
    assert dict(secondary) == {"API_ALIAS": "fallback"} and ss.read_as_launch_home(secondary) is False
    own = ss.build_profile_secret_scope(launch)
    assert dict(own) == {"API_ALIAS": "SYNTHETIC-LAUNCH-KEY"} and ss.read_as_launch_home(own) is True


def test_a_home_retargeted_after_the_read_never_puts_launch_values_into_a_non_launch_scope(homes, monkeypatch):
    launch, other, alias = homes
    (launch / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\nLAUNCH_ONLY=launch-file\n")
    (other / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\n")
    _retarget_after_read(monkeypatch, alias, launch)
    raced = ss.build_profile_secret_scope(alias)
    assert ss.read_as_launch_home(raced) is False
    assert dict(raced) == {"API_ALIAS": "fallback"}  # the secondary's bytes, under the secondary's verdict


def test_a_launch_home_retargeted_after_the_read_keeps_its_own_read(homes, monkeypatch):
    launch, other, alias = homes
    alias.unlink()
    alias.symlink_to(launch, target_is_directory=True)
    (launch / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\n")
    (other / ".env").write_text("API_ALIAS=other-${OPENAI_API_KEY:-fallback}\n")
    _retarget_after_read(monkeypatch, alias, other)
    raced = ss.build_profile_secret_scope(alias)
    assert ss.read_as_launch_home(raced) is True
    assert dict(raced) == {"API_ALIAS": "SYNTHETIC-LAUNCH-KEY"}  # never the other home's file


def test_managed_values_still_take_precedence_and_expand(homes, monkeypatch, tmp_path):
    from hermes_cli import managed_scope

    launch, other, alias = homes
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("API_ALIAS=managed-${OPENAI_API_KEY:-dflt}\n")
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed)
    monkeypatch.setattr(managed_scope, "load_managed_env", lambda: {"API_ALIAS": "managed-${OPENAI_API_KEY:-dflt}"})
    (other / ".env").write_text("API_ALIAS=${OPENAI_API_KEY:-fallback}\n")
    assert dict(ss.build_profile_secret_scope(alias)) == {"API_ALIAS": "managed-dflt"}
    assert dict(ss.build_profile_secret_scope(launch)) == {"API_ALIAS": "managed-SYNTHETIC-LAUNCH-KEY"}
