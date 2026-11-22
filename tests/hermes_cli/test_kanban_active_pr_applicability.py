"""Which linked PRs hold a ready card under the ``active_pr`` respawn guard.

Cryo t_7752959d: another agent's comment linked a dependency PR that had already landed, and the
dispatcher held the card ``respawn_guarded {"reason": "active_pr"}`` on every tick, with no writer.
The guard exists so that a re-spawned worker does not open a second PR beside the one the card's
own work opened. So a PR only someone else linked holds nothing, while the card's own PR still
holds, whoever names it later. Checked at the dispatcher's guard on a real board.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

OWN_PR = "https://github.com/example/repo/pull/44"
DEPENDENCY_PR = "https://github.com/example/repo/pull/4420"


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _card(conn) -> str:
    return kb.create_task(conn, title="frontend repair", assignee="dev")


def test_a_dependency_pr_linked_by_another_agent_holds_nothing(conn) -> None:
    tid = _card(conn)
    kb.add_comment(conn, tid, author="default/session:composer",
                   body=f"The assembler repair LANDED on main as {DEPENDENCY_PR}, merge fa13b7c2.")
    assert kbd.check_respawn_guard(conn, tid) is None


def test_the_cards_own_pr_still_holds(conn) -> None:
    tid = _card(conn)
    kb.add_comment(conn, tid, author="dev", body=f"Opened {OWN_PR} for review.")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_anyone_naming_the_cards_own_pr_still_holds(conn) -> None:
    tid = _card(conn)
    kb.add_comment(conn, tid, author="dev/session:worker", body=f"Opened {OWN_PR.upper()} for review.")
    kb.add_comment(conn, tid, author="operator", body=f"Same PR: {OWN_PR}")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_a_pr_in_the_completion_contract_is_the_cards_own(conn) -> None:
    tid = _card(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET completion_contract = ? WHERE id = ?", (OWN_PR, tid))
    kb.add_comment(conn, tid, author="reviewer", body=f"Reviewing {OWN_PR}.")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_a_profile_that_ran_the_card_links_its_own_pr(conn) -> None:
    tid = _card(conn)
    kb.claim_task(conn, tid)
    profile = conn.execute("SELECT profile FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET assignee = 'next-owner' WHERE id = ?", (tid,))
    kb.add_comment(conn, tid, author=profile, body=f"Opened {OWN_PR}.")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
