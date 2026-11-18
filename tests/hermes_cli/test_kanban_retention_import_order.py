"""The workspace-retention modules import cleanly first, in a fresh interpreter, in any order.

``kanban_db_workspace`` tail-imports ``kanban_db_retention``, and both tail-import the
``kanban_db`` facade, which imports the workspace module back: whichever is imported first decides
whether the others find their names defined yet. In-process tests cannot see this, because the
facade is already loaded by the time they run, so every order is checked in a new interpreter.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

FIRST = {
    "facade": "import hermes_cli.kanban_db",
    "worker tools": "import tools.kanban_tools",
    "workspace module": "import hermes_cli.kanban_db_workspace",
    "retention module": "from hermes_cli.kanban_db_retention import settle_record_only",
    "board module": "import hermes_cli.kanban_db_boards",
}
THEN = (
    "import hermes_cli.kanban_db as kb\n"
    "import hermes_cli.kanban_db_workspace as w\n"
    "import hermes_cli.kanban_db_retention as r\n"
    "assert w.retention_covering is r.retention_covering and w.removal_fence is r.removal_fence\n"
    "assert r._kb is kb and w._kb is kb\n"
    "print('imports-ok')\n"
)


@pytest.mark.parametrize("first", sorted(FIRST))
def test_retention_modules_import_first_in_a_fresh_interpreter(first):
    proc = subprocess.run(
        [sys.executable, "-c", f"{FIRST[first]}\n{THEN}"],
        cwd=REPO, capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0 and "imports-ok" in proc.stdout, (
        f"importing the {first} first failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
