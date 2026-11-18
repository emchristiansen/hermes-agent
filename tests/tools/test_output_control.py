"""Caller-controlled prospective output persistence (tools/output_control.py).

Every payload here is a synthetic marker. The checks run the real producers (foreground
``terminal`` through LocalEnvironment's streaming collector, ``kanban_show`` over a real board)
and, for the composed path, the real agent executor + session DB — nothing replaces the
collector, the handler or the persistence step being tested."""

import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import output_control as oc
from tools.terminal_tool import terminal_tool

MARKER = "SYNTHETIC-OC-MARKER"


@pytest.fixture(autouse=True)
def _release_terminal_environments():
    """Tear down the terminal environments these tests create, so later modules start clean."""
    yield
    from tools.terminal_tool_lifecycle import cleanup_all_environments
    cleanup_all_environments()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return home


@pytest.fixture
def outdir(tmp_path):
    d = tmp_path / "caller-out"
    d.mkdir()
    return Path(os.path.realpath(d))


def _default_spill_hits(home: Path) -> list[Path]:
    """Files in Hermes' default spill locations that carry the synthetic marker."""
    roots = [home / "cache", Path(tempfile.gettempdir()) / "hermes-results"]
    hits = []
    for root in roots:
        if root.exists():
            for f in root.rglob("*"):
                if f.is_file() and MARKER.encode() in f.read_bytes():
                    hits.append(f)
    return hits


def _payload_script(tmp_path: Path, rows: int = 3000) -> tuple[str, bytes]:
    """A command printing a deterministic synthetic payload, and the exact bytes it prints.
    It carries an ANSI escape and a credential-shaped marker so any transformation shows."""
    lines = [f"\x1b[1m{MARKER}-head\x1b[0m sk-proj-SYNTHETICa1B2c3D4e5F6g7H8i9J0SYNTHETIC"]
    lines += [f"{MARKER}-row-{i:05d} " + "z" * 60 for i in range(rows)]
    lines.append(f"{MARKER}-tail")
    expected = ("\n".join(lines) + "\n").encode()
    script = tmp_path / "emit.py"
    script.write_text("import sys\nsys.stdout.write(open(sys.argv[1], encoding='utf-8').read())\n")
    data = tmp_path / "payload.txt"
    data.write_bytes(expected)
    return f"{sys.executable} {script} {data}", expected


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("raw, fragment", [
    ("nope", "must be an object"),
    ({"destination": "relative/out.log"}, "absolute path"),
    ({"destination": "/a/../b"}, "normalized"),
    ({"select": []}, "non-empty list"),
    ({"select": ["task"]}, "JSON pointers"),
    ({"select": ["/a~2b"]}, "RFC 6901"),
    ({"select": ["/a~"]}, "RFC 6901"),
    ({"inline_chars": -1}, "0 to"),
    ({"inline_chars": oc.MAX_INLINE_CHARS + 1}, "0 to"),
    ({"inline_chars": True}, "0 to"),
    ({"bogus": 1}, "unknown field"),
])
def test_malformed_control_is_refused(raw, fragment):
    with pytest.raises(oc.OutputControlError, match=fragment):
        oc.OutputControl.parse(raw)


def test_pointer_resolution_follows_rfc6901():
    doc = {"a/b": {"m~n": [10, 20]}, "list": [1, 2, 3], "": "empty-key"}
    control = oc.OutputControl.parse({"select": [
        "/a~1b/m~0n/1", "/list/0", "/list/01", "/list/٣", "/list/3", "/", ""]})
    projected = control.project(doc)
    assert projected["selected"] == {"/a~1b/m~0n/1": 20, "/list/0": 1, "/": "empty-key", "": doc}
    # Leading zeros, non-ASCII digits and out-of-range indices name nothing.
    assert projected["missing"] == ["/list/01", "/list/٣", "/list/3"]


# --- terminal: real producer / collector -----------------------------------

@pytest.mark.platforms("linux")
def test_terminal_destination_receives_exact_stream_and_default_spill_is_unused(home, outdir, tmp_path):
    command, expected = _payload_script(tmp_path)
    dest = outdir / "out.log"
    r = json.loads(terminal_tool(command, task_id="oc-1", output_control=oc.OutputControl.parse(
        {"destination": str(dest), "inline_chars": 500})))
    assert r["exit_code"] == 0 and r["error"] is None
    assert dest.read_bytes() == expected  # byte-exact: no ANSI strip, no redaction, no cap
    meta = r["output_control"]
    assert meta == {"destination": str(dest), "chars": len(expected.decode()), "bytes": len(expected),
                    "sha256": hashlib.sha256(expected).hexdigest(), "complete": True, "inline_complete": False}
    assert r["full_output_path"] == str(dest)
    assert len(r["output"]) < 700
    assert (dest.stat().st_mode & 0o777) == 0o600
    assert _default_spill_hits(home) == []
    assert not (home / "cache" / "terminal-output").exists()


@pytest.mark.parametrize("text", [f"{MARKER}-no-trailing-newline", f"{MARKER}-pipe\n", ""])
def test_terminal_destination_holds_only_the_commands_own_bytes(home, outdir, text):
    """The wrapper's in-band trailer (cwd / pipeline-status marker lines) is protocol, consumed
    before the destination is committed — even when the command's output lacks a newline."""
    dest = outdir / "small.log"
    r = json.loads(terminal_tool(f"printf %s {shlex.quote(text)} | cat", task_id="oc-8",
                                 output_control=oc.OutputControl.parse({"destination": str(dest)})))
    assert r["exit_code"] == 0 and dest.read_bytes() == text.encode()
    assert r["output_control"]["complete"] is True


@pytest.mark.platforms("linux")
def test_terminal_without_destination_keeps_default_spill_with_caller_window(home, tmp_path):
    command, _expected = _payload_script(tmp_path)
    r = json.loads(terminal_tool(command, task_id="oc-2", output_control=oc.OutputControl.parse(
        {"inline_chars": 800})))
    assert r["exit_code"] == 0
    # Called directly (no executor) the payload is the ordinary window; the executor applies the
    # caller's inline budget (test_inline_chars_bounds_terminal_output_without_destination).
    assert len(r["output"]) <= 2000 + 200 and r["output_control"]["inline_complete"] is False
    spill = Path(r["full_output_path"])
    assert spill.parent == home / "cache" / "terminal-output" and MARKER in spill.read_text()


def test_terminal_existing_destination_refused_before_command_runs(home, outdir, tmp_path):
    dest = outdir / "taken.log"
    dest.write_text("caller data")
    sentinel = tmp_path / "ran"
    r = json.loads(terminal_tool(f"touch {sentinel}", task_id="oc-3", output_control=oc.OutputControl.parse(
        {"destination": str(dest)})))
    assert "already exists" in r["error"] and "not run" in r["error"]
    assert not sentinel.exists() and dest.read_text() == "caller data"


@pytest.mark.parametrize("make_parent", ["missing", "symlink"])
def test_terminal_unavailable_destination_refused_before_command_runs(home, outdir, tmp_path, make_parent):
    sentinel = tmp_path / "ran"
    if make_parent == "missing":
        dest = outdir / "no-such-dir" / "out.log"
    else:
        link = outdir / "link"
        link.symlink_to(outdir)
        dest = link / "out.log"
    r = json.loads(terminal_tool(f"touch {sentinel}", task_id="oc-4", output_control=oc.OutputControl.parse(
        {"destination": str(dest)})))
    assert "not run" in r["error"] and not sentinel.exists() and not dest.exists()


@pytest.mark.parametrize("target", ["AGENTS.md", "SOUL.md", "hermes-sessions", "outside-safe-root"])
def test_terminal_protected_destination_refused_before_command_runs(home, outdir, tmp_path, monkeypatch, target):
    # The destination is written by Hermes itself, so it honours the guards write_file applies.
    sentinel = tmp_path / "ran"
    if target == "outside-safe-root":
        safe = tmp_path / "safe-root"
        safe.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe))
        dest = outdir / "out.log"
    elif target == "hermes-sessions":
        sessions = Path(os.path.realpath(home)) / "sessions"
        sessions.mkdir()
        dest = sessions / "out.log"
    else:
        dest = outdir / target
    r = json.loads(terminal_tool(f"touch {sentinel}", task_id="oc-4p", output_control=oc.OutputControl.parse(
        {"destination": str(dest)})))
    assert "protected path" in r["error"] and "not run" in r["error"]
    assert not sentinel.exists() and not dest.exists()


def test_terminal_handler_refuses_malformed_control_and_background(home, tmp_path):
    from tools.terminal_tool import _handle_terminal
    sentinel = tmp_path / "ran"
    bad = json.loads(_handle_terminal({"command": f"touch {sentinel}", "output_control": {"select": ["~"]}}))
    bg = json.loads(_handle_terminal({"command": f"touch {sentinel}", "background": True,
                                      "output_control": {"inline_chars": 10}}))
    assert "not run" in bad["error"] and "foreground commands only" in bg["error"]
    assert not sentinel.exists()


@pytest.mark.platforms("linux")
def test_terminal_select_projects_json_stdout_into_destination(home, outdir, tmp_path):
    doc = {"meta": {"id": f"{MARKER}-id"}, "rows": [f"{MARKER}-{i}-" + "q" * 100 for i in range(5000)]}
    src = tmp_path / "doc.json"
    src.write_text(json.dumps(doc))
    dest = outdir / "sel.json"
    r = json.loads(terminal_tool(f"cat {src}", task_id="oc-5", output_control=oc.OutputControl.parse(
        {"destination": str(dest), "select": ["/meta/id", "/rows/2", "/absent"], "inline_chars": 0})))
    expected = json.dumps({"selected": {"/meta/id": f"{MARKER}-id", "/rows/2": doc["rows"][2]},
                           "missing": ["/absent"]}, ensure_ascii=False)
    assert dest.read_text() == expected
    assert r["output"] == "" and r["output_control"]["complete"] and r["output_control"]["missing_count"] == 1
    assert _default_spill_hits(home) == []  # the 500K-char raw stream was never written to disk


@pytest.mark.platforms("linux")
def test_terminal_select_on_non_json_is_explicitly_incomplete(home, outdir, tmp_path):
    command, _ = _payload_script(tmp_path)
    dest = outdir / "sel.json"
    r = json.loads(terminal_tool(command, task_id="oc-6", output_control=oc.OutputControl.parse(
        {"destination": str(dest), "select": ["/x"], "inline_chars": 100})))
    assert "not one JSON document" in r["error"]
    assert r["output_control"]["complete"] is False and r["output_control"]["bytes"] == 0
    # Nothing was selected, so none of the unselected stream is shown or stored.
    assert r["output"] == "" and MARKER not in json.dumps(r) and _default_spill_hits(home) == []


@pytest.mark.platforms("linux")
def test_terminal_midstream_write_failure_reports_partial_and_reroutes_nothing(home, outdir, tmp_path, monkeypatch):
    command, expected = _payload_script(tmp_path)
    real_write, budget = os.write, {"left": 4096}

    def failing_write(fd, data):  # the disk fills after 4 KiB
        if fd >= 0 and os.readlink(f"/proc/self/fd/{fd}").startswith(str(outdir)):
            if budget["left"] <= 0:
                raise OSError(28, "No space left on device")
            n = real_write(fd, bytes(data[:budget["left"]]))
            budget["left"] -= n
            return n
        return real_write(fd, data)

    monkeypatch.setattr(oc.os, "write", failing_write)
    dest = outdir / "out.log"
    r = json.loads(terminal_tool(command, task_id="oc-7", output_control=oc.OutputControl.parse(
        {"destination": str(dest), "inline_chars": 500})))
    meta = r["output_control"]
    assert meta["complete"] is False and meta["bytes"] == 4096 and "No space left" in meta["error"]
    assert dest.read_bytes() == expected[:4096] and meta["chars"] == len(expected[:4096].decode())
    assert r["output"] == "" and "not stored anywhere else" in r["error"]
    assert _default_spill_hits(home) == []


# --- kanban_show: real handler over a real board ----------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path, home):
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=f"{MARKER}-title", body=f"{MARKER}-body", assignee="test-worker")
        for i in range(60):  # a large comment history: far over the default 100K spill threshold
            kb.add_comment(conn, tid, "someone", f"{MARKER}-comment-{i} " + "c" * 3000)
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def test_kanban_show_projection_to_destination_is_startup_usable(worker_env, outdir, home):
    from tools import kanban_tools as kt
    full = json.loads(kt._handle_show({}))
    dest = outdir / "show.json"
    out = kt._handle_show({"output_control": {
        "select": ["/task/title", "/task/body", "/comments/59/body"], "destination": str(dest),
        "inline_chars": 4000}})
    expected = json.dumps({"selected": {"/task/title": full["task"]["title"], "/task/body": full["task"]["body"],
                                        "/comments/59/body": full["comments"][59]["body"]}, "missing": []})
    assert dest.read_text() == expected
    result = json.loads(out)
    assert result["full_output_path"] == str(dest)
    # Startup usability: the inline payload already is the complete selection (title + body).
    assert result["output"] == expected and len(out) < 5000
    assert result["output_control"]["complete"] is True and result["output_control"]["inline_complete"] is True
    assert result["output_control"]["missing_count"] == 0
    assert _default_spill_hits(home) == []


def test_kanban_show_full_result_to_destination_matches_unprojected_output(worker_env, outdir):
    from tools import kanban_tools as kt
    dest = outdir / "full.json"
    out = kt._handle_show({"output_control": {"destination": str(dest), "inline_chars": 0}})
    reference = json.loads(kt._handle_show({}))
    assert json.loads(dest.read_text()) == reference and len(dest.read_text()) > 150_000
    result = json.loads(out)
    assert result["output"] == "" and result["output_control"]["inline_complete"] is False


@pytest.mark.parametrize("control, fragment", [
    ({"destination": "rel.json"}, "absolute path"),
    ({"select": ["/a~9"]}, "RFC 6901"),
])
def test_kanban_show_malformed_control_is_a_tool_error(worker_env, control, fragment):
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_show({"output_control": control}))
    assert fragment in out["error"]


def test_kanban_show_unavailable_destination_refused_before_reading(worker_env, outdir, monkeypatch):
    from tools import kanban_tools as kt
    taken = outdir / "taken.json"
    taken.write_text("caller data")
    monkeypatch.setattr(kt, "_existing_task", MagicMock(side_effect=AssertionError("board was read")))
    for dest in (taken, outdir / "missing" / "x.json"):
        out = json.loads(kt._handle_show({"output_control": {"destination": str(dest)}}))
        assert "output_control.destination" in out["error"]
    assert taken.read_text() == "caller data"


def test_kanban_show_protected_destination_refused_before_reading(worker_env, outdir, monkeypatch):
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_existing_task", MagicMock(side_effect=AssertionError("board was read")))
    out = json.loads(kt._handle_show({"output_control": {"destination": str(outdir / "CLAUDE.md")}}))
    assert "protected path" in out["error"] and not (outdir / "CLAUDE.md").exists()


# --- composed path: real dispatch + executor + session DB ---------------------

def _run_tool_calls(agent, calls, mode="sequential"):
    from tests.agent.test_tool_call_incremental_persistence import _mock_tool_call
    tool_calls = [_mock_tool_call(name=n, arguments=json.dumps(a), call_id=f"call_{i}")
                  for i, (n, a) in enumerate(calls)]
    messages: list = []
    batch = SimpleNamespace(content="", tool_calls=tool_calls)
    if mode == "segmented":  # alternate parallel/sequential segments of two calls each
        from agent.tool_executor import execute_tool_calls_segmented
        segments = [("parallel" if n % 2 == 0 else "sequential", tool_calls[k:k + 2])
                    for n, k in enumerate(range(0, len(tool_calls), 2))]
        execute_tool_calls_segmented(agent, batch, messages, "task-oc", segments=segments)
    else:
        getattr(agent, f"_execute_tool_calls_{mode}")(batch, messages, "task-oc")
    return [m for m in messages if m.get("role") == "tool"]


@pytest.mark.platforms("linux")
def test_composed_executor_never_spills_destination_results_even_over_turn_budget(
        worker_env, outdir, home, tmp_path, monkeypatch):
    from tests.agent.test_tool_call_incremental_persistence import (
        _attach_real_session_db, _durable_messages, _make_agent)
    from tools.budget_config import BudgetConfig
    # Agent log-file setup is not under test; in a long suite run an earlier test can leave a
    # queued handler for the pre-sandbox home that the home-I/O guard rightly refuses to touch.
    monkeypatch.setattr("agent.agent_init._setup_logging", lambda _agent: None)
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    _attach_real_session_db(agent, db_path, "sess-oc")
    # A tight turn budget forces aggregate enforcement after the per-result flush.
    monkeypatch.setattr("agent.tool_executor._budget_for_agent",
                        lambda _agent: BudgetConfig(default_result_size=50_000, turn_budget=1_000, preview_size=200))
    monkeypatch.setattr(agent, "valid_tool_names", {"kanban_show", "terminal"}, raising=False)
    command, expected = _payload_script(tmp_path)
    show_dest, term_dest = outdir / "show.json", outdir / "term.log"
    msgs = _run_tool_calls(agent, [
        ("kanban_show", {"output_control": {"destination": str(show_dest), "inline_chars": 900}}),
        ("terminal", {"command": command, "output_control": {"destination": str(term_dest), "inline_chars": 900}}),
    ])
    assert len(msgs) == 2 and all(m.get(oc.CONTROLLED_MESSAGE_FLAG) == "destination" for m in msgs)
    assert term_dest.read_bytes() == expected and len(show_dest.read_text()) > 150_000
    assert _default_spill_hits(home) == []
    durable = [m for m in _durable_messages(db_path, "sess-oc") if m.get("role") == "tool"]
    assert len(durable) == 2
    # Under this tight budget a receipt may be compacted to name its file once (EI-011).
    for row, dest in zip(durable, (show_dest, term_dest)):
        result = json.loads(row["content"])
        assert result["output_control"]["destination"] == str(dest)
        assert result.get("full_output_path", str(dest)) == str(dest)


# --- native exposure: advertised definitions + real first-call dispatch -----------

def test_startup_tool_definitions_advertise_output_control_and_dispatch_accepts_first_call(
        worker_env, outdir, home):
    """Unmocked: the definitions a dispatcher-spawned worker is served at startup carry the
    parameter, and the real dispatch path (coercion, kanban argument check, handler) accepts
    a first kanban_show call shaped as a model sends it — including the object JSON-encoded."""
    import model_tools
    defs = {d["function"]["name"]: d["function"] for d in
            model_tools.get_tool_definitions(enabled_toolsets=["kanban", "terminal"], quiet_mode=False)}
    for name in ("kanban_show", "terminal"):
        prop = defs[name]["parameters"]["properties"]["output_control"]
        assert set(prop["properties"]) == {"destination", "select", "inline_chars"}
    dest = outdir / "first-call.json"
    first_call = {"output_control": json.dumps({"select": ["/task/title", "/task/body"],
                                                "destination": str(dest), "inline_chars": 2000})}
    out = model_tools.handle_function_call("kanban_show", first_call, task_id="oc-startup",
                                           tool_call_id="call_first")
    assert json.loads(out)["full_output_path"] == str(dest) and f"{MARKER}-body" in out
    assert json.loads(dest.read_text())["selected"]["/task/title"] == f"{MARKER}-title"
    term_dest = outdir / "first-term.log"
    tout = json.loads(model_tools.handle_function_call(
        "terminal", {"command": f"printf %s {MARKER}-T", "output_control": {"destination": str(term_dest)}},
        task_id="oc-startup", tool_call_id="call_term"))
    assert tout["exit_code"] == 0 and term_dest.read_text() == f"{MARKER}-T"
    assert _default_spill_hits(home) == []


# --- executor bounding of controlled envelopes (review findings EI-001..006) --------

def _agent(monkeypatch, tmp_path, *, result=8_000, turn=16_000):
    from tests.agent.test_tool_call_incremental_persistence import _attach_real_session_db, _make_agent
    from tools.budget_config import BudgetConfig
    monkeypatch.setattr("agent.agent_init._setup_logging", lambda _agent: None)
    agent = _make_agent()
    _attach_real_session_db(agent, tmp_path / "state.db", "sess-bound")
    monkeypatch.setattr("agent.tool_executor._budget_for_agent",
                        lambda _agent: BudgetConfig(default_result_size=result, turn_budget=turn, preview_size=1500))
    monkeypatch.setattr(agent, "valid_tool_names", {"kanban_show", "terminal"}, raising=False)
    return agent


def _durable_tool_rows(tmp_path):
    from tests.agent.test_tool_call_incremental_persistence import _durable_messages
    return [m for m in _durable_messages(tmp_path / "state.db", "sess-bound") if m.get("role") == "tool"]


def _cat(tmp_path, chars, name="payload.txt"):
    data = tmp_path / name
    data.write_text(f"{MARKER}-" + "x" * chars)
    return f"cat {data}", data.read_bytes()


@pytest.mark.parametrize("form", ["native", "encoded", "encoded-select"])
def test_executor_sees_the_coerced_control_the_handler_ran_under(home, outdir, tmp_path, monkeypatch, form):
    """EI-001: a JSON-encoded control (accepted by schema coercion) is the same control at the
    persistence boundary: the destination gets the payload and no default spill appears."""
    agent = _agent(monkeypatch, tmp_path)
    dest = outdir / f"{form}.out"
    if form == "encoded-select":
        src = tmp_path / "doc.json"
        src.write_text(json.dumps({"keep": f"{MARKER}-" + "k" * 25_000, "drop": "d"}))
        command, control = f"cat {src}", {"destination": str(dest), "select": ["/keep"], "inline_chars": 12_000}
    else:
        command, _ = _cat(tmp_path, 25_000)
        control = {"destination": str(dest), "inline_chars": 12_000}
    raw = json.dumps(control) if form != "native" else control
    (msg,) = _run_tool_calls(agent, [("terminal", {"command": command, "output_control": raw})])
    assert msg.get(oc.CONTROLLED_MESSAGE_FLAG) == "destination"
    assert dest.stat().st_size > 25_000 and _default_spill_hits(home) == []
    result = json.loads(msg["content"])
    assert len(msg["content"]) <= 8_000 and result["full_output_path"] == str(dest)


@pytest.mark.parametrize("inline", [0, 1, 40])
def test_inline_chars_bounds_the_payload_without_destination(worker_env, home, tmp_path, monkeypatch, inline):
    """EI-002: inline_chars bounds the shown payload; the full selection stays retrievable in the
    ordinary archive, and the durable transcript row carries no more than the immediate result."""
    agent = _agent(monkeypatch, tmp_path)
    (msg,) = _run_tool_calls(agent, [("kanban_show", {"output_control": {
        "select": ["/task/body", "/comments/0/body"], "inline_chars": inline}})])
    result = json.loads(msg["content"])
    assert len(result["output"]) <= inline + (len(result["output"]) and 120)  # head/tail notice only
    if inline == 0:
        assert result["output"] == "" and f"{MARKER}-body" not in msg["content"]
    archive = Path(result["full_output_path"])
    assert archive.parent == home / "cache" / "spillover" and f"{MARKER}-comment-0" in archive.read_text()
    assert msg.get(oc.CONTROLLED_MESSAGE_FLAG) == "default"
    (row,) = _durable_tool_rows(tmp_path)
    assert row["content"] == msg["content"]


@pytest.mark.parametrize("inline", [0, 1, 40])
def test_inline_chars_bounds_terminal_selection_without_destination(home, tmp_path, monkeypatch, inline):
    agent = _agent(monkeypatch, tmp_path)
    src = tmp_path / "doc.json"
    src.write_text(json.dumps({"keep": f"{MARKER}-" + "k" * 3_000, "drop": "d"}))
    (msg,) = _run_tool_calls(agent, [("terminal", {"command": f"cat {src}", "output_control": {
        "select": ["/keep"], "inline_chars": inline}})])
    result = json.loads(msg["content"])
    assert result["exit_code"] == 0 and len(result["output"]) <= inline + (len(result["output"]) and 120)
    assert json.loads(Path(result["full_output_path"]).read_text())["selected"]["/keep"].startswith(MARKER)


def test_aggregate_budget_shrinks_controlled_payloads_and_keeps_receipts(worker_env, home, outdir, tmp_path, monkeypatch):
    """EI-003: destination and ordinary results in one batch fit the real turn budget; destination
    bytes and receipts survive; destination payloads never enter the default archive."""
    agent = _agent(monkeypatch, tmp_path)
    command, expected = _cat(tmp_path, 7_000)
    calls = [("terminal", {"command": command, "output_control": {
        "destination": str(outdir / f"batch{i}.log"), "inline_chars": 7_000}}) for i in range(3)]
    calls.append(("kanban_show", {"output_control": {"destination": str(outdir / "show.json"), "inline_chars": 7_000}}))
    # An ordinary (uncontrolled) result may still be archived by the aggregate budget as before.
    calls.append(("terminal", {"command": "printf %s ORDINARY-UNCONTROLLED-RESULT"}))
    msgs = _run_tool_calls(agent, calls)
    assert sum(len(m["content"]) for m in msgs) <= 16_000
    for i, msg in enumerate(msgs[:3]):
        result = json.loads(msg["content"])
        assert result["exit_code"] == 0 and result["output_control"]["complete"] is True
        assert result["full_output_path"] == str(outdir / f"batch{i}.log")
        assert (outdir / f"batch{i}.log").read_bytes() == expected
    assert json.loads(msgs[3]["content"])["full_output_path"] == str(outdir / "show.json")
    assert _default_spill_hits(home) == []
    # Bounded before each flush by what was left of the turn budget: the durable payloads already
    # fit it, and every durable row keeps its receipt. (Metadata is never cut, so a row whose
    # share hit zero still carries it; the later aggregate pass may trim the live copies further.)
    # The controlled rows were bounded before their flush to shares of the budget, so they fit
    # it durably too. (The ordinary result keeps the existing aggregate handling.)
    rows = [json.loads(r["content"]) for r in _durable_tool_rows(tmp_path)[:4]]
    assert sum(len(json.dumps(r, ensure_ascii=False)) for r in rows) <= 16_000
    assert [r["full_output_path"] for r in rows] == [str(outdir / f"batch{i}.log") for i in range(3)] + [
        str(outdir / "show.json")]
    assert all(r["output_control"]["complete"] is True for r in rows)


@pytest.mark.parametrize("case", ["success", "nonzero", "delivery-failure"])
def test_per_result_bound_keeps_structured_receipt_and_status(home, outdir, tmp_path, monkeypatch, case):
    """EI-004: shrinking a terminal envelope cuts only its payload; the result stays JSON with the
    destination, exit status and completeness, and the transcript row matches."""
    agent = _agent(monkeypatch, tmp_path)
    command, expected = _cat(tmp_path, 25_000)
    if case == "nonzero":
        command += "; exit 3"
    if case == "delivery-failure":
        real_write = os.write

        def failing(fd, data):
            if os.readlink(f"/proc/self/fd/{fd}").startswith(str(outdir)):
                raise OSError(28, "No space left on device")
            return real_write(fd, data)
        monkeypatch.setattr(oc.os, "write", failing)
    dest = outdir / "receipt.log"
    (msg,) = _run_tool_calls(agent, [("terminal", {"command": command, "output_control": {
        "destination": str(dest), "inline_chars": 12_000}})])
    assert len(msg["content"]) <= 8_000
    result = json.loads(msg["content"])
    assert result["exit_code"] == (3 if case == "nonzero" else 0)
    assert result["output_control"]["destination"] == str(dest)
    assert result["output_control"]["complete"] is (case != "delivery-failure")
    if case != "delivery-failure":
        assert result["full_output_path"] == str(dest) and dest.read_bytes() == expected
        assert result["output_control"]["inline_complete"] is False
    (row,) = _durable_tool_rows(tmp_path)
    assert row["content"] == msg["content"] and _default_spill_hits(home) == []


def test_partial_write_counts_only_landed_characters(outdir, monkeypatch):
    """EI-006: a short write then failure reports the characters that actually landed."""
    sink = oc.Destination(outdir / "partial.out")
    real_write, calls = os.write, {"n": 0}

    def short_then_fail(fd, data):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data[:4]))  # "abc" + the first byte of "é", then fail
        raise OSError(28, "synthetic full")
    monkeypatch.setattr(oc.os, "write", short_then_fail)
    sink.write("abcé" + "z" * 10)
    receipt = sink.close()
    landed = (outdir / "partial.out").read_bytes()
    assert receipt.complete is False and receipt.bytes == len(landed) == 4
    assert receipt.chars == len(landed.decode("utf-8", errors="ignore")) == 3  # "abc"; half of "é"


def test_tiny_window_never_shows_wrapper_trailer(home, outdir, tmp_path):
    """A small caller window still lets the wrapper's marker lines be consumed as protocol."""
    expected = ("\n é中😀\r\n" * 1800 + "EOF\x1b[0m\n").encode()
    src = tmp_path / "utf8.bin"
    src.write_bytes(expected)
    dest = outdir / "utf8.out"
    r = json.loads(terminal_tool(f"cat {shlex.quote(str(src))}", task_id="oc-utf8",
                                 output_control=oc.OutputControl.parse({"destination": str(dest), "inline_chars": 100})))
    assert dest.read_bytes() == expected
    assert r["output_control"]["sha256"] == hashlib.sha256(expected).hexdigest()
    assert "__HERMES" not in r["output"] and not r["output"].rstrip().endswith("__")


@pytest.mark.parametrize("inline", [0, 5])
def test_tiny_window_keeps_session_cwd(home, outdir, tmp_path, inline):
    """EI-007: a tiny inline window still lets the wrapper's cwd marker reach _update_cwd, so a
    `cd` persists to the next command in the same session."""
    new = tmp_path / "changed-cwd"
    new.mkdir()
    first = json.loads(terminal_tool(f"cd {shlex.quote(str(new))}; printf ready", task_id=f"oc-cwd-{inline}",
                                     output_control=oc.OutputControl.parse(
                                         {"destination": str(outdir / f"cwd{inline}.out"), "inline_chars": inline})))
    assert first["exit_code"] == 0 and (outdir / f"cwd{inline}.out").read_text() == "ready"
    assert first["output"] == "ready"[:inline] or len(first["output"]) <= inline + 120
    nxt = json.loads(terminal_tool("pwd", task_id=f"oc-cwd-{inline}"))
    assert nxt["output"] == str(new)


# --- r4 repairs: EI-003 (durable), EI-009..012 --------------------------------

@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_destination_previews_fit_the_turn_budget_durably(home, outdir, tmp_path, monkeypatch, mode):
    """EI-003: each controlled result is bounded once, before its row is flushed, with a share
    of the unused turn budget left for the controlled calls still to come — so the complete
    serialized rows fit the budget live AND in SQLite, and the two agree."""
    agent = _agent(monkeypatch, tmp_path)
    command, expected = _cat(tmp_path, 7_000)
    msgs = _run_tool_calls(agent, [("terminal", {"command": command, "output_control": {
        "destination": str(outdir / f"{i}.out"), "inline_chars": 7_000}}) for i in range(3)], mode)
    durable = _durable_tool_rows(tmp_path)
    assert [r["content"] for r in durable] == [m["content"] for m in msgs]
    assert sum(len(m["content"]) for m in msgs) <= 16_000
    assert all((outdir / f"{i}.out").read_bytes() == expected for i in range(3))
    assert all("over_budget" not in json.loads(m["content"])["output_control"] for m in msgs)
    assert _default_spill_hits(home) == []


@pytest.mark.parametrize("text, inline", [("Q", 0), (f"{MARKER}-short", 0), (f"{MARKER}-" + "y" * 200, 10)])
def test_no_destination_terminal_keeps_the_complete_output_retrievable(home, tmp_path, monkeypatch, text, inline):
    """EI-009: without a destination the executor stores the complete stdout verbatim before
    cutting it to the caller's inline budget, and says the inline payload is partial."""
    agent = _agent(monkeypatch, tmp_path)
    msg, = _run_tool_calls(agent, [("terminal", {"command": f"printf %s {shlex.quote(text)}",
                                                 "output_control": {"inline_chars": inline}})])
    result = json.loads(msg["content"])
    assert Path(result["full_output_path"]).read_text() == text
    meta = result["output_control"]
    assert meta["chars"] == len(text) and meta["complete"] is True and meta["inline_complete"] is False
    assert len(result["output"]) <= inline + 200 and (inline or result["output"] == "")
    assert _durable_tool_rows(tmp_path)[0]["content"] == msg["content"]


def test_no_destination_archive_is_verbatim_even_for_envelope_shaped_json(home, tmp_path, monkeypatch):
    """EI-010: requested content shaped like an MCP envelope is archived byte-for-byte."""
    agent = _agent(monkeypatch, tmp_path)
    expected = json.dumps({"result": f"{MARKER}-" + "x" * 10_000})
    source = tmp_path / "raw.json"
    source.write_text(expected)
    msg, = _run_tool_calls(agent, [("terminal", {"command": f"cat {source}",
                                                 "output_control": {"inline_chars": 20_000}})])
    result = json.loads(msg["content"])
    assert Path(result["full_output_path"]).read_text() == expected
    assert result["output_control"]["chars"] == len(expected) and result["output_control"]["inline_complete"] is False


def test_long_missing_pointer_does_not_inflate_metadata(worker_env, home, tmp_path, monkeypatch):
    """EI-011: the pointers are part of the requested content, not repeated in metadata."""
    agent = _agent(monkeypatch, tmp_path)
    pointer = "/" + "M" * 9_000
    msg, = _run_tool_calls(agent, [("kanban_show", {"output_control": {"select": [pointer], "inline_chars": 0}})])
    result = json.loads(msg["content"])
    assert len(msg["content"]) <= 8_000 and result["output"] == ""
    assert result["output_control"]["missing_count"] == 1 and "over_budget" not in result["output_control"]
    assert json.loads(Path(result["full_output_path"]).read_text()) == {"selected": {}, "missing": [pointer]}


@pytest.mark.parametrize("turn", [1_000, 300])
def test_receipt_budget_is_met_compactly_or_stated(home, outdir, tmp_path, monkeypatch, turn):
    """EI-011: receipts are never cut. When the full form exceeds a result's share, the receipt
    drops only what it repeats (full_output_path == destination) and an empty payload's inline
    statistics; when even that cannot fit, ``over_budget`` says so — live and durable alike."""
    agent = _agent(monkeypatch, tmp_path, turn=turn)
    msgs = _run_tool_calls(agent, [("terminal", {"command": "printf abc", "output_control": {
        "destination": str(outdir / f"{i}.out"), "inline_chars": 0}}) for i in range(3)])
    results = [json.loads(m["content"]) for m in msgs]
    for i, r in enumerate(results):
        # A compact receipt names the file once (output_control.destination).
        assert r["output"] == "" and r["output_control"]["destination"] == str(outdir / f"{i}.out")
        assert r.get("full_output_path", str(outdir / f"{i}.out")) == str(outdir / f"{i}.out")
        assert r["output_control"]["complete"] is True and r["output_control"]["sha256"]
        assert (outdir / f"{i}.out").read_text() == "abc"
    assert [r["content"] for r in _durable_tool_rows(tmp_path)] == [m["content"] for m in msgs]
    total = sum(len(m["content"]) for m in msgs)
    # Independent lower bound: the mandatory receipt facts alone, compact JSON (paths vary in length).
    compact = max(len(oc.dump_envelope({"output": "", "exit_code": 0, "error": None, "output_control": {
        "destination": str(outdir / f"{i}.out"), "chars": 3, "bytes": 3, "sha256": "0" * 64, "complete": True}}))
        for i in range(3))
    if 3 * compact <= turn:
        assert total <= turn and not any("over_budget" in r["output_control"] for r in results)
    else:
        # Mandatory receipts that do not fit what is left are kept whole and say so.
        flagged = [(r, m) for r, m in zip(results, msgs) if "over_budget" in r["output_control"]]
        assert total > turn and flagged
        assert all(r["output_control"]["over_budget"] < len(m["content"]) for r, m in flagged)


@pytest.mark.parametrize("mode", ["sequential", "concurrent", "segmented"])
def test_mixed_batch_keeps_controlled_rows_durable_and_within_budget(home, outdir, tmp_path, monkeypatch, mode):
    """EI-003 with ordinary results in the same batch: every later call holds a share, so the
    controlled rows never need a post-flush trim — live and SQLite agree and the batch fits."""
    agent = _agent(monkeypatch, tmp_path)
    command, expected = _cat(tmp_path, 7_000)
    calls = []
    for i in range(3):
        calls.append(("terminal", {"command": command, "output_control": {
            "destination": str(outdir / f"{i}.out"), "inline_chars": 7_000}}))
        calls.append(("terminal", {"command": f"printf %s ORDINARY-{i}"}))
    msgs = _run_tool_calls(agent, calls, mode)
    durable = _durable_tool_rows(tmp_path)
    assert [r["content"] for r in durable] == [m["content"] for m in msgs]
    assert sum(len(r["content"]) for r in durable) <= 16_000
    assert all((outdir / f"{i}.out").read_bytes() == expected for i in range(3))
    assert [json.loads(m["content"])["output"] for m in msgs[1::2]] == [f"ORDINARY-{i}" for i in range(3)]


def test_kanban_show_reserves_destination_before_reading(worker_env, outdir, monkeypatch):
    """EI-012: the destination is created exclusively before the board is read; a competing
    writer can no longer take it in between, and a failed read removes the still-empty file."""
    from tools import kanban_tools as kt
    dest = outdir / "show.json"
    real = kt._existing_task
    seen = []

    def reader(*a, **kw):
        seen.append(dest.exists())
        return real(*a, **kw)

    monkeypatch.setattr(kt, "_existing_task", reader)
    out = json.loads(kt._handle_show({"output_control": {"destination": str(dest)}}))
    assert seen == [True] and out["full_output_path"] == str(dest) and json.loads(dest.read_text())["task"]
    failed = outdir / "failed.json"
    err = json.loads(kt._handle_show({"task_id": "t_does_not_exist", "output_control": {"destination": str(failed)}}))
    assert "error" in err and not failed.exists()


def test_huge_valid_array_index_is_simply_missing(worker_env, home, tmp_path, monkeypatch):
    """EI-014: a valid index token of any length resolves like any out-of-range index — missing,
    with the full pointer kept in the requested content and only a count in metadata."""
    assert oc.JsonPointer.parse("/list/" + "9" * 5_000).resolve({"list": [1, 2]}) == (False, None)
    assert oc.JsonPointer.parse("/list/1").resolve({"list": [1, 2]}) == (True, 2)
    assert oc.JsonPointer.parse("/list/10").resolve({"list": list(range(10))}) == (False, None)
    agent = _agent(monkeypatch, tmp_path)
    pointer = "/comments/" + "9" * 5_000
    msg, = _run_tool_calls(agent, [("kanban_show", {"output_control": {"select": [pointer], "inline_chars": 0}})])
    result = json.loads(msg["content"])
    assert result["output_control"]["missing_count"] == 1 and result["output"] == ""
    assert json.loads(Path(result["full_output_path"]).read_text()) == {"selected": {}, "missing": [pointer]}


@pytest.mark.parametrize("mode", ["sequential", "concurrent", "segmented"])
def test_controlled_rows_are_never_rewritten_after_flush(home, outdir, tmp_path, monkeypatch, mode):
    """EI-013: controlled previews are bounded before their flush with room reserved for a later
    ordinary result's archived form, so the aggregate pass archives only that ordinary result:
    the batch fits the turn budget live, and the controlled rows stay exactly as written, live
    and in SQLite. (Ordinary rows keep the existing aggregate handling.)"""
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 12_000, "max_lines": 2_000, "max_line_length": 12_000}})
    agent = _agent(monkeypatch, tmp_path, turn=6_000)
    command, expected = _cat(tmp_path, 7_000)
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("ORDINARY-" + "q" * 6_000)
    calls = [("terminal", {"command": command, "output_control": {
        "destination": str(outdir / f"{i}.out"), "inline_chars": 7_000}}) for i in range(3)]
    calls.append(("terminal", {"command": f"cat {ordinary}"}))
    msgs = _run_tool_calls(agent, calls, mode)
    durable = _durable_tool_rows(tmp_path)
    assert [r["content"] for r in durable[:3]] == [m["content"] for m in msgs[:3]]
    # The later ordinary result had room reserved for its archived form: the batch fits live.
    assert sum(len(m["content"]) for m in msgs) <= 6_000
    assert all((outdir / f"{i}.out").read_bytes() == expected for i in range(3))


@pytest.mark.parametrize("mode", ["sequential", "concurrent", "segmented"])
def test_refused_later_control_is_budgeted_like_any_result(worker_env, home, outdir, tmp_path, monkeypatch, mode):
    """EI-015: a later call whose control is malformed is refused with an ordinary error
    result; the room reserved for it does not depend on its nominal control, so the batch
    still fits the turn budget live and the controlled row stays as flushed."""
    agent = _agent(monkeypatch, tmp_path, turn=4_000)
    command, expected = _cat(tmp_path, 7_000)
    msgs = _run_tool_calls(agent, [
        ("terminal", {"command": command, "output_control": {"destination": str(outdir / "0.out"), "inline_chars": 7_000}}),
        ("kanban_show", {"output_control": {"select": ["M" * 9_000]}}),
    ], mode)
    assert "error" in json.loads(msgs[1]["content"]) or "persisted-output" in msgs[1]["content"]
    assert sum(len(m["content"]) for m in msgs) <= 4_000
    assert _durable_tool_rows(tmp_path)[0]["content"] == msgs[0]["content"]
    assert (outdir / "0.out").read_bytes() == expected


def test_hints_added_around_a_controlled_result_count_against_its_room(home, outdir, tmp_path, monkeypatch):
    """EI-016: what the executor appends to a controlled row (here a real subdirectory hint from
    an AGENTS.md beside the command's file) is part of the row: the bounded result makes room
    for it, the hint is delivered whole, and the row fits the budget live and in SQLite."""
    from agent.subdirectory_hints import SubdirectoryHintTracker
    project = tmp_path / "project"
    sub = project / "sub"
    sub.mkdir(parents=True)
    (sub / "AGENTS.md").write_text("SYNTHETIC-HINT-TEXT\n" + "H" * 9_000)
    source = sub / "data.txt"
    source.write_text(f"{MARKER}-" + "x" * 7_000)
    agent = _agent(monkeypatch, tmp_path, result=20_000, turn=16_000)
    agent._subdirectory_hints = SubdirectoryHintTracker(str(project), enabled=True)
    msg, = _run_tool_calls(agent, [("terminal", {"command": f"cat {source}", "output_control": {
        "destination": str(outdir / "out"), "inline_chars": 7_000}})])
    content = msg["content"]
    assert "SYNTHETIC-HINT-TEXT" in content and len(content) <= 16_000
    assert _durable_tool_rows(tmp_path)[0]["content"] == content
    receipt, _end = json.JSONDecoder().raw_decode(content)
    assert receipt["output_control"]["complete"] is True and (outdir / "out").read_text() == source.read_text()



def test_destination_is_refused_in_a_hosted_scope(tmp_path):
    """A hosted agent's tools act only through its own sandbox (tools/hosted_tool_policy.py), and
    a destination is a file on this host: it is refused before anything is created or run."""
    from agent.runtime_scope import bind_state
    from tests.tools.test_sandbox_scope_conformance import A
    target = tmp_path / "out.txt"
    with bind_state(A), pytest.raises(oc.OutputControlError, match="not available to a hosted agent"):
        oc.check_destination(target)
    assert not target.exists()
    oc.check_destination(target)  # the same path outside the hosted scope is accepted
