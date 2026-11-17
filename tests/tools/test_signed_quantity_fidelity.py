"""Signed accounting deltas survive real read_file presentation and kanban persistence.

The phone-number pass of ``redact_sensitive_text`` masked ``+1109262336``-shaped byte deltas: in
``read_file`` output of an intact report, and in a ``kanban_schedule`` reason before it was stored.
These drive the real handlers on an isolated filesystem and board and read the stored bytes back;
the phone-shaped controls prove contact masking still runs through the same paths, including the
independent review's contact witnesses (IND-PHONE-02..11).
"""
import json

import pytest

PHONE = "+15551234567"  # reserved fictional 555 number
DELTAS = ("+1109262336", "+1109278720", "+2941231104")

TABLE = (
    "| Measure (bytes) |13:46:08 before|13:46:35 after|Shared-host change|\n"
    "|---|---:|---:|---:|\n"
    "|/home usable|538819559424|539928821760|+1109262336|\n"
    "|rpool/home AVAILABLE|538819592192|539928870912|+1109278720|\n"
    "|Pool FREE|671258910720|674200141824|+2941231104|\n"
    "|Pool FREEING|0|0|0|\n"
)
SUMMARY = (
    "OriginalsharedwindowHOME+1109262336B/pool+2941231104/FREEING0 separate nonadditive; "
    "pool delta +2941231104 bytes; dataset+1109278720B."
)
CONTACTS = f"| Name | Phone |\n|---|---:|\n| Alice | {PHONE} |\n"  # right-aligned, no unit


@pytest.fixture(autouse=True)
def _redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def test_read_file_presents_signed_table_and_summary_exactly(tmp_path, monkeypatch):
    import tools.file_tools as ft
    import tools.terminal_tool as tt

    original = tt._get_env_config
    monkeypatch.setattr(tt, "_get_env_config",
                        lambda: {**original(), "env_type": "local", "cwd": str(tmp_path)})
    task_id = "signed-quantity-read"
    report = tmp_path / "RESULT.md"
    body = f"# Accounting\n\n{TABLE}\n{SUMMARY}\n\n{CONTACTS}"
    try:
        assert json.loads(ft.write_file_tool(str(report), body, task_id)).get("error") is None
        assert report.read_text() == body
        content = json.loads(ft.read_file_tool(str(report), task_id=task_id))["content"]
    finally:
        tt.clear_session_cwd(task_id)

    lines = content.splitlines()
    for n, line in enumerate(body.splitlines(), 1):
        if PHONE in line:
            assert f"{n}|" + line not in lines and PHONE not in content
        else:
            assert f"{n}|{line}" in lines
    assert report.read_text() == body  # reading never rewrites the file


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with a running task; returns the task id."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="signed-quantity", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _call(tool: str, args: dict) -> dict:
    from tools import kanban_tools  # registers the handlers
    from tools.registry import registry
    return json.loads(registry.get_entry(tool).handler(args))


def _stored_run_summary(tid: str) -> str:
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        return conn.execute(
            "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()[0]


@pytest.mark.parametrize("tool", ["kanban_schedule", "kanban_complete"])
def test_kanban_summary_persists_signed_deltas_exactly(worker_env, tool):
    text = f"{SUMMARY}\n\n{TABLE.rstrip()}"  # kanban_schedule strips the reason
    field = "reason" if tool == "kanban_schedule" else "summary"
    out = _call(tool, {field: text})
    assert out["ok"] is True
    if tool == "kanban_schedule":
        assert out["reason"] == text

    assert _stored_run_summary(worker_env) == text
    shown = _call("kanban_show", {"task_id": worker_env})
    assert shown["runs"][-1]["summary"] == text
    assert all(d in text for d in DELTAS)


def test_kanban_summary_still_masks_contact_numbers(worker_env):
    out = _call("kanban_schedule", {"reason": f"{SUMMARY}\n\n{CONTACTS}"})
    stored = _stored_run_summary(worker_env)
    assert PHONE not in stored and PHONE not in out["reason"]
    assert stored.startswith(SUMMARY)



P = "+" + "12025550123"  # the independent review's synthetic witness
M = "+120****0123"           # its masked form
Q = "+" + "12025550124"  # a second synthetic contact value
QM = "+120****0124"
# input -> exactly what every consumer must show
CONTACT_WITNESSES = {
    "IND-PHONE-02": (f"Phone: {P} bytes=4096", f"Phone: {M} bytes=4096"),
    "IND-PHONE-03": (f"| Measure (bytes) | Phone number change |\n|---|---:|\n| owner | {P} |",
                     f"| Measure (bytes) | Phone number change |\n|---|---:|\n| owner | {M} |"),
    "phone-column": (f"| Measure (bytes) | Phone | change |\n|---|---:|---:|\n| disk | {P} | 0 |",
                     f"| Measure (bytes) | Phone | change |\n|---|---:|---:|\n| disk | {M} | 0 |"),
    "cross-occurrence": (
        f"| Measure (bytes) | change | Phone |\n|---|---:|---:|\n| disk | {P} | {P} |",
        f"| Measure (bytes) | change | Phone |\n|---|---:|---:|\n| disk | {P} | {M} |"),
    "contact-table-unit-cell": (f"| Measure (bytes) | Phone |\n|---|---:|\n| owner | {P} bytes |",
                                f"| Measure (bytes) | Phone |\n|---|---:|\n| owner | {M} bytes |"),
    "incident-glued": ("HOME+1109262336B/pool+2941231104/FREEING0",
                       "HOME+1109262336B/pool+2941231104/FREEING0"),
    "IND-PHONE-04": (
        f"| Measure (bytes) | Change |\n|---|---:|\n| pool | 0 |\n## Contacts | directory\n"
        f"| Name | Phone |\n|---|---:|\n| owner | {P} |",
        f"| Measure (bytes) | Change |\n|---|---:|\n| pool | 0 |\n## Contacts | directory\n"
        f"| Name | Phone |\n|---|---:|\n| owner | {M} |"),
    "IND-PHONE-05": (f"| Name | Phone |\n|---|---:|\n| owner | {P} bytes | note |",
                     f"| Name | Phone |\n|---|---:|\n| owner | {M} bytes | note |"),
    "IND-PHONE-06": (f"| Name | Cellphone (bytes) |\n|---|---:|\n| owner | {P} |",
                     f"| Name | Cellphone (bytes) |\n|---|---:|\n| owner | {M} |"),
    "IND-PHONE-07": (f'{{"phone": "{P} bytes"}}', f'{{"phone": "{M} bytes"}}'),
    "IND-QUANTITY-01": (f"pool{P}/FREEING0", f"pool{P}/FREEING0"),
    "IND-PHONE-08": (f'Phone: {P} / {Q} bytes\n{{"phone": "{P} / {Q} bytes"}}',
                     f'Phone: {M} / {QM} bytes\n{{"phone": "{M} / {QM} bytes"}}'),
    "IND-PHONE-09": (f"| Name | Phone |\n|---|---:|\nowner | {P} bytes\n\nName | Phone\n---|---:\n| owner | {P} bytes |",
                     f"| Name | Phone |\n|---|---:|\nowner | {M} bytes\n\nName | Phone\n---|---:\n| owner | {M} bytes |"),
    "IND-QUANTITY-03": (f"| Measure (bytes) | Change |\n|---|---:|\npool | {P}\n\nMeasure (bytes) | Change\n---|---:\n| pool | {P} |",
                        f"| Measure (bytes) | Change |\n|---|---:|\npool | {P}\n\nMeasure (bytes) | Change\n---|---:\n| pool | {P} |"),
    "IND-PHONE-10": (json.dumps({"phone": [P + " bytes"]}) + "\n" + json.dumps(json.dumps({"phone": P + " bytes"})),
                     json.dumps({"phone": [M + " bytes"]}) + "\n" + json.dumps(json.dumps({"phone": M + " bytes"}))),
    "json-field-boundary": (json.dumps({"phone": P + " bytes", "delta": Q + " bytes"}),
                            json.dumps({"phone": M + " bytes", "delta": Q + " bytes"})),
    "IND-QUANTITY-04": (f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {P} | {Q} bytes |",
                        f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {M} | {Q} bytes |"),
    "IND-PHONE-11": (f"Phone: ({P} bytes) / {Q} bytes\nPhone: ({P} bytes); freed {Q} bytes",
                     f"Phone: ({M} bytes) / {QM} bytes\nPhone: ({M} bytes); freed {Q} bytes"),
    "joined-identifier": ("noSupport/new09f71717+5323013 FULLread; build 1.2.3+7654321",
                          "noSupport/new09f71717+5323013 FULLread; build 1.2.3+7654321"),
    "IND-QUANTITY-02": (
        "HOME / pool FREE net movements: r7 **+820772864 / +1340882944 bytes**; "
        "r5 **+545652736 / −69820416 bytes**.",
        "HOME / pool FREE net movements: r7 **+820772864 / +1340882944 bytes**; "
        "r5 **+545652736 / −69820416 bytes**."),
}


def _through(consumer: str, text: str, tmp_path, monkeypatch, tid: str) -> str:
    """What the consumer shows: read_file content without its gutter, or the stored+shown text."""
    if consumer == "read_file":
        import tools.file_tools as ft
        import tools.terminal_tool as tt
        original = tt._get_env_config
        monkeypatch.setattr(tt, "_get_env_config",
                            lambda: {**original(), "env_type": "local", "cwd": str(tmp_path)})
        path, task_id = tmp_path / "witness.md", "witness-read"
        try:
            assert json.loads(ft.write_file_tool(str(path), text, task_id)).get("error") is None
            content = json.loads(ft.read_file_tool(str(path), task_id=task_id))["content"]
        finally:
            tt.clear_session_cwd(task_id)
        assert path.read_text() == text  # reading never rewrites the file
        lines = content.splitlines()
        assert all(line.startswith(f"{n}|") for n, line in enumerate(lines, 1))
        return "\n".join(line.split("|", 1)[1] for line in lines)
    if consumer == "kanban_comment":
        from hermes_cli import kanban_db_connect as kbc
        out = _call(consumer, {"task_id": tid, "body": text})
        assert out["ok"] is True
        with kbc.connect() as conn:
            stored = conn.execute("SELECT body FROM task_comments WHERE id = ?",
                                  (out["comment_id"],)).fetchone()[0]
        assert _call("kanban_show", {"task_id": tid})["comments"][-1]["body"] == stored
        return stored
    field = "reason" if consumer == "kanban_schedule" else "summary"
    assert _call(consumer, {field: text})["ok"] is True
    stored = _stored_run_summary(tid)
    assert _call("kanban_show", {"task_id": tid})["runs"][-1]["summary"] == stored
    return stored


@pytest.mark.parametrize("consumer", ["read_file", "kanban_schedule", "kanban_complete",
                                      "kanban_comment"])
@pytest.mark.parametrize("name", sorted(CONTACT_WITNESSES))
def test_contact_witnesses_through_real_consumers(name, consumer, worker_env, tmp_path, monkeypatch):
    text, expected = CONTACT_WITNESSES[name]
    assert _through(consumer, text, tmp_path, monkeypatch, worker_env) == expected
