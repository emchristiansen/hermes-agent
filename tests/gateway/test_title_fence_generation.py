"""TITLE-CANCEL-01 companion: the serve generation of the title fence keeps legitimate fencing for the CURRENT
generation. A fencing read with no serve in between still fences and refuses a later upgrade; a read bound to a
generation a serve has since replaced publishes nothing; the successor's own later read fences again; another home is
unaffected; and the generation exists (and advances) without the title generator being imported."""
import os
import sys
import threading

from agent import title_generator as tg
from gateway import run_shutdown_budget as budget


def _root(path):
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _unstarted_upgrade(home):
    ran = threading.Event()
    upgrade = threading.Thread(target=ran.set, name='auto-title', daemon=True)
    tg._UPGRADE_STORES[upgrade] = str(home / 'state.db')
    return upgrade, ran


def test_a_fencing_read_of_the_current_generation_still_fences(tmp_path):
    home = tmp_path / 'alpha'; home.mkdir()
    budget.unfence_title_home(home)
    assert budget.title_upgrades_live_under(home, 0.0, fence=True) == 0
    assert _root(home) in tg._FENCED_HOMES
    upgrade, ran = _unstarted_upgrade(home)
    tg.start_title_upgrade(upgrade)
    assert upgrade.ident is None and not ran.is_set()  # refused: the retiring generation's fence applies
    budget.unfence_title_home(home)


def test_a_read_bound_to_a_replaced_generation_publishes_no_fence_but_the_successors_read_does(tmp_path):
    home, other = tmp_path / 'alpha', tmp_path / 'beta'; home.mkdir(); other.mkdir()
    bound = budget.title_fence_generation(home)  # a retirement binds before its thread handoff
    budget.unfence_title_home(home)              # a re-serve starts the successor generation
    assert budget.title_upgrades_live_under(home, 0.0, fence=True, generation=bound) == 0
    assert _root(home) not in tg._FENCED_HOMES   # the obsolete read did not fence the successor
    upgrade, ran = _unstarted_upgrade(home)
    tg.start_title_upgrade(upgrade)
    assert ran.wait(5)                           # the successor's upgrade starts
    upgrade.join(5)
    assert budget.title_upgrades_live_under(home, 5.0, fence=True) == 0  # the successor's own retirement read
    assert _root(home) in tg._FENCED_HOMES
    assert budget.title_upgrades_live_under(other, 0.0, fence=True) == 0
    assert _root(other) in tg._FENCED_HOMES      # another home: unaffected by alpha's generation
    budget.unfence_title_home(home); budget.unfence_title_home(other)


def test_the_generation_needs_no_title_generator_import(tmp_path, monkeypatch):
    home = tmp_path / 'gamma'; home.mkdir()
    monkeypatch.delitem(sys.modules, 'agent.title_generator')
    before = budget.title_fence_generation(home)
    budget.unfence_title_home(home)
    assert budget.title_fence_generation(home) == before + 1
    assert 'agent.title_generator' not in sys.modules  # a stop never imports it
