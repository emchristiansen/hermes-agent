"""Foreground result post-processing for the terminal tool (cwd dual-write,
sudo handling, transform hook, truncation, ANSI strip, redaction, exit-code
notes/hints, spill redaction, verification evidence) + exit-code tables. Lazy
``tools.terminal_tool`` lookups keep the origin's monkeypatch points authoritative.
"""

import json
import logging
import os
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tools.terminal_tool")


@contextmanager
def _quiet(label: str):
    """Best-effort block: swallow any exception, logging it at DEBUG."""
    try:
        yield
    except Exception:
        logger.debug("%s failed", label, exc_info=True)


# Signal-death notes keyed by signum, used for both ``-signum`` (subprocess)
# and ``128+signum`` (shell) encodings. Curated, not exhaustive, so a
# legitimate application exit code is never mislabeled; 130/SIGINT is owned
# by the executor's interrupt-marker path and excluded.
_SIGNAL_EXIT_NOTES: dict[int, str] = {
    3:  "SIGQUIT (quit from keyboard)",
    4:  "SIGILL (illegal instruction — corrupt binary or wrong architecture)",
    6:  "SIGABRT (abort — assertion failure, fatal runtime error, or glibc abort)",
    7:  "SIGBUS (bus error — misaligned or unmapped memory access)",
    8:  "SIGFPE (fatal arithmetic error, e.g. integer division by zero)",
    9:  "SIGKILL — often the kernel OOM killer on memory exhaustion, or an explicit kill -9",
    11: "SIGSEGV (segmentation fault — the program crashed)",
    13: "SIGPIPE (wrote to a closed pipe — e.g. output piped to a reader that exited)",
    15: "SIGTERM (terminated — kill/timeout or shutdown requested it to stop)",
    24: "SIGXCPU (CPU time limit exceeded)",
    25: "SIGXFSZ (file size limit exceeded)",
}


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Note for a signal-termination exit code, or None. Negative codes are
    definite (subprocess semantics); 128+signum is the shell convention and a
    program *can* exit 139 itself, so those notes hedge with "usually"."""
    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT — executor's interrupt-marker path owns it
            return None
        if note := _SIGNAL_EXIT_NOTES.get(signum):
            return f"Command terminated by signal {signum}: {note}"
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"
    if exit_code > 128 and (note := _SIGNAL_EXIT_NOTES.get(exit_code - 128)):
        return (f"Exit code {exit_code} usually means the command was "
                f"terminated by signal {exit_code - 128}: {note}")
    return None


# Informational non-zero exit codes per base command.
_EXIT_CODE_SEMANTICS: dict[str, dict[int, str]] = {
    **{key: {1: "No matches found (not an error)"} for key in ("grep", "egrep", "fgrep", "rg", "ag", "ack")},
    **{key: {1: "Files differ (expected, not an error)"} for key in ("diff", "colordiff")},
    **{key: {1: "Condition evaluated to false (expected, not an error)"} for key in ("test", "[")},
    "find": {1: "Some directories were inaccessible (partial results may still be valid)"},
    "curl": {6: "Could not resolve host", 7: "Failed to connect to host",
             22: "HTTP response code indicated error (e.g. 404, 500)", 28: "Operation timed out"},
    "git": {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
}

# Model-facing warning attached when the backend replaced its container/sandbox
# mid-command (out-of-band removal, terminal sandbox state). Persistent-filesystem
# state was restored, but background processes and anything outside the synced
# paths are gone (ported from lobehub/lobehub#19329).
_ENV_RECREATED_NOTE = (
    "The execution environment was recreated while running this command "
    "(the previous container/sandbox was gone). Persistent files were restored "
    "where the backend supports it, but background processes and any files "
    "outside persisted paths from earlier commands may be lost — verify state "
    "before relying on prior work."
)


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Note for a non-zero exit code that is informational rather than an
    error (grep=1 "no matches", diff=1 "files differ", signal deaths), so the
    model doesn't burn turns investigating it. None when 0 or a real error."""
    if exit_code == 0:
        return None
    if (signal_note := _interpret_signal_exit(exit_code)) is not None:
        # Signal terminations (ported from Kilo-Org/kilocode#12698, adapted to Python semantics). Two shapes
        # reach the model: * negative codes — subprocess.Popen reports a signal-killed process as
        # ``-signum`` (definite signal death), and * 128+signum — the conventional shell encoding when bash
        # reports a signal-killed child (heuristic: a program *can* ``exit 139``, so these notes say
        # "usually"). Without a note the model sees a bare ``exit_code=-9`` or ``137`` and burns turns
        # re-running or mis-diagnosing (137 = OOM kill is the big one). 130/SIGINT is deliberately absent:
        # the executor has bespoke interrupt-marker handling for rc=130.
        return signal_note
    # The last command of a pipeline/chain determines the exit code; base
    # command = its first word that isn't a VAR=val assignment, basename'd.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()
    base_cmd = next((w.split("/")[-1] for w in last_segment.split()
                     if "=" not in w or w.startswith("-")), "")
    return _EXIT_CODE_SEMANTICS.get(base_cmd, {}).get(exit_code)


def _sudo_annotations(command: str, output: str, env_type: str) -> tuple[str, bool, bool]:
    """Sudo failure handling -> (output, auth_failed, cache_cleared)."""
    import tools.terminal_tool as tt
    from tools.terminal_tool_sudo import (
        _handle_sudo_failure, _invalidate_cached_sudo_on_auth_failure, _no_sudo_user,
        _sudo_wrong_password_failure,
    )
    from utils import env_var_enabled
    output = _handle_sudo_failure(output, env_type)
    auth_failed = _sudo_wrong_password_failure(output)
    cache_cleared = _invalidate_cached_sudo_on_auth_failure(command, output)
    can_reprompt = cache_cleared and (
        tt._get_sudo_password_callback() is not None or env_var_enabled("HERMES_INTERACTIVE")
    ) and not _no_sudo_user()
    if can_reprompt:
        output += ("\n\n⚠️ Sudo authentication failed — cached password "
                   "cleared. You will be prompted again on the next sudo command.")
    return output, auth_failed, cache_cleared


def _apply_output_transform_hook(command, output, returncode, task_id, env_type) -> str:
    """Plugin output-transform seam (fail-open; first string result wins).
    Replacements are still subject to the output limit applied afterwards."""
    with _quiet("transform_terminal_output hook"):
        from hermes_cli.lifecycle import invoke_hook
        from tools.approval_context import _approval_tool_call_id
        # Concurrent terminal calls in one turn must gate per call, not collapse into one;
        # an empty id is treated as "no identity" by the hook gate.
        results = invoke_hook("transform_terminal_output", command=command, output=output,
                              returncode=returncode, task_id=task_id or "", env_type=env_type,
                              tool_call_id=_approval_tool_call_id.get())
        output = next((r for r in results if isinstance(r, str)), output)
    return output


def _truncate_head_tail(output: str, limit: Optional[int] = None) -> str:
    from tools.tool_output_limits import get_max_bytes
    from tools.tool_output_truncate import truncate_head_tail
    return truncate_head_tail(output, get_max_bytes() if limit is None else limit)


def _apply_output_control(control, sink, result: dict) -> tuple[str, list[tuple[str, Any]], Optional[int], bool, Optional[str]]:
    """The caller's ``output_control`` for one finished foreground command.

    Returns ``(inline_output, fields, window, selected, error)``. ``window`` bounds the inline
    head/tail cut (None = profile default); ``selected`` means *inline_output* is already the
    requested content and skips the transform hook and head/tail cut. ``fields`` always carry
    the ``output_control`` metadata, which makes the result a controlled envelope the executor
    bounds in place (``tools.output_control.bound_envelope``). The destination (when given)
    received the raw stream from byte 0 via the collector, or receives the projection here; it
    is never ANSI-stripped, redacted or rewritten afterwards."""
    from tools import output_control as oc
    raw = result.get("output", "")
    total = result.get("stream_total_chars", len(raw))
    if control.select is None:
        # The captured stream travels as the payload: the executor cuts it to the caller's inline
        # budget (``bound_envelope``), storing it verbatim first when nothing else holds it. With a
        # destination the window is already the caller's budget (the file holds everything);
        # without one, a stream that fit the collector travels whole, else the ordinary window.
        from tools.tool_output_limits import get_max_bytes
        if sink is not None:
            window = control.inline_chars if control.inline_chars is not None else get_max_bytes()
        else:
            window = max(1, len(raw)) if total <= len(raw) else get_max_bytes()
        error = None
        if sink is not None:
            receipt = sink.close()
            if control.inline_chars == 0:
                raw = ""
            meta: dict[str, Any] = {**receipt.fields(),
                                    "inline_complete": receipt.complete and receipt.chars <= min(len(raw), window)}
            fields: list[tuple[str, Any]] = [("full_output_path", str(receipt.path))]
            if not receipt.complete:
                error = oc.failed_delivery(receipt)["error"]
                raw = ""  # the payload stays where it landed; nothing inline to re-spill
        else:
            # A stream longer than the ordinary window overflowed into the default tee
            # (cache/terminal-output, up to its cap), which then is the recovery location.
            from tools.environments.base_output import _BoundedOutputCollector
            meta = {"chars": total, "complete": total <= _BoundedOutputCollector._SPILL_CAP_CHARS,
                    "inline_complete": total <= min(len(raw), window)}
            fields = []
        return raw, [*fields, (oc.PARAM, meta)], window, False, error

    failure = None
    if total > oc.SELECT_CAPTURE_CEILING_CHARS:
        failure = (f"output_control.select: stdout was {total:,} chars, over the "
                   f"{oc.SELECT_CAPTURE_CEILING_CHARS:,}-char selection ceiling; nothing was selected or stored. "
                   "Narrow the command's output (e.g. pipe through jq) and retry.")
    else:
        try:
            document = json.loads(raw)
        except ValueError as exc:
            failure = f"output_control.select: stdout is not one JSON document ({exc}); nothing was selected or stored."
    if failure is not None:
        meta = {"complete": False, "error": failure}
        if sink is not None:
            meta.update({k: v for k, v in sink.close().fields().items() if k in ("destination", "bytes")})
        # The caller asked for a selection only: the unselected stream is never shown or stored.
        return "", [(oc.PARAM, meta)], None, True, failure
    projection = control.project(document)
    text = json.dumps(projection, ensure_ascii=False)
    meta = {"chars": len(text), "missing_count": len(projection["missing"])}
    if sink is None:
        # The complete projection travels inline; the executor bounds it to the caller's and the
        # model's budgets, storing it in the ordinary archive first when it must be cut.
        return text, [(oc.PARAM, {**meta, "complete": True})], None, True, None
    sink.write(text)
    receipt = sink.close()
    if not receipt.complete:
        return "", [(oc.PARAM, receipt.fields())], None, True, oc.failed_delivery(receipt)["error"]
    preview = oc.cut_inline(text, control.inline_chars if control.inline_chars is not None
                            else oc.DEFAULT_PREVIEW_CHARS)
    return preview, [("full_output_path", str(receipt.path)),
                     (oc.PARAM, {**meta, **receipt.fields(), "inline_complete": preview == text})], None, True, None


def _failure_hint(command: str, returncode: int, output: str, exit_note) -> Optional[str]:
    """Recovery hints for well-known failure shapes (tools/terminal_hints.py);
    on rc=0, warn when a pipeline tail / `|| echo` may mask an upstream
    failure and the output carries strong failure indicators (advisory only)."""
    with _quiet("failure hint"):
        from tools import terminal_hints
        if returncode == 0:
            return terminal_hints.annotate_masked_success(command, output)
        if not exit_note:
            return terminal_hints.annotate_failure(command, returncode, output)
    return None


def _redact_spill_file(path, total_chars, command) -> list[tuple[str, Any]]:
    """Spill handle so the model can read the omitted middle instead of
    re-running. The collector wrote it raw; redact it with the same pass so no
    secret persists unmasked on disk. On failure drop the handle (and file)."""
    if not path:
        return []
    try:
        from agent.redact import redact_terminal_output
        from tools.ansi_strip import strip_ansi
        from tools.spill_safety import write_text_exclusive
        raw_spill = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        # lstat-checked unlink + exclusive create: the redacted copy can't
        # be diverted through a symlink planted since the collector's write.
        write_text_exclusive(Path(path), redact_terminal_output(strip_ansi(raw_spill), command),
                             private=True, overwrite=True, errors="replace")
    except Exception:
        logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
        with _quiet("spill unlink"):
            Path(path).unlink()
        return []
    note = ("Output exceeded the capture window (head+tail shown). "
            f"Full output ({total_chars:,} chars) saved to {path} — search it with "
            "search_files or page it with read_file instead of re-running the command.")
    return [("output_total_chars", total_chars), ("full_output_path", path), ("truncation_note", note)]


def _verification_evidence(command, cwd, session_id, returncode, output) -> Optional[dict]:
    with _quiet("verification evidence recording"):
        from agent.verification_evidence import record_terminal_result
        evidence = record_terminal_result(command=command, cwd=cwd, session_id=session_id,
                                          exit_code=returncode, output=output)
        if evidence:
            return {k: evidence.get(k) for k in ("status", "kind", "scope", "canonical_command")}
    return None


def finalize_foreground_result(
    *, command: str, result: dict, env: Any, env_type: str, effective_task_id: str,
    task_id: Optional[str], session_id: Optional[str], session_key: str,
    workdir: Optional[str], command_cwd: Optional[str], approval_note: Optional[str],
    output_control=None, sink=None,
) -> str:
    """Turn a raw ``env.execute`` result into the tool's JSON result string. ``output_control``
    (``tools.output_control.OutputControl``) and its opened destination ``sink`` apply the
    caller's prospective selection, inline budget and destination."""
    from tools.terminal_tool import record_session_cwd

    # Record the cwd this command finished in as THIS session's durable cwd —
    # only when the command actually reported it (an interrupted/killed command
    # emits no marker, and env.cwd then holds another session's directory), and
    # never for a transient per-command ``workdir`` (would hijack the session cwd
    # for every later command). Prefer the result's own cwd; env.cwd is shared
    # mutable compat state kept as fallback for third-party providers.
    observed_cwd = None
    if (result or {}).get("cwd_observed"):
        observed_cwd = (result or {}).get("cwd") or getattr(env, "cwd", None)
    if not workdir and observed_cwd:
        record_session_cwd(session_key, observed_cwd)

    output = result.get("output", "")
    returncode = result.get("returncode", 0)
    control_fields: list[tuple[str, Any]] = []
    window, selected, control_error = None, False, None
    if output_control is not None:
        output, control_fields, window, selected, control_error = _apply_output_control(output_control, sink, result)
    output, sudo_auth_failed, sudo_cache_cleared = _sudo_annotations(command, output, env_type)
    if not selected:
        output = _apply_output_transform_hook(command, output, returncode, effective_task_id, env_type)
        output = _truncate_head_tail(output, window) if output else ""
    # Strip ANSI so the model never copies escapes into file writes, then
    # redact secrets; redact_terminal_output is command-aware (env-dump
    # commands get the KEY=value pass, source/config dumps skip it).
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi
    output = strip_ansi(output)
    # For source/config dumps (MAX_TOKENS=100, "apiKey": "x" fixtures, postgresql:// f-string templates) the
    # ENV/JSON/template passes are skipped to avoid false positives (code_file=True). But for env-dump
    # commands (env/printenv/set/export/declare) the output IS a KEY=value credential dump, so
    # redact_terminal_output runs the ENV pass (code_file=False) to mask opaque tokens with no vendor
    # prefix. Real prefixes, auth headers, JWTs, private keys are masked in both modes. See issue #43025.
    output = redact_terminal_output(output.strip(), command) if output else ""

    exit_note = _interpret_exit_code(command, returncode)
    failure_hint = _failure_hint(command, returncode, output, exit_note)
    # cwd echo when the command changed directory (gated on the observation
    # flag so an interrupted command can't echo another session's cwd).
    changed_cwd = None
    with _quiet("cwd comparison"):
        if observed_cwd and command_cwd and os.path.realpath(str(observed_cwd)) != os.path.realpath(str(command_cwd)):
            changed_cwd = str(observed_cwd)
    # rc=130 is an interrupt only with the executor's marker — a command can
    # legitimately `exit 130` itself. An interrupted approved run keeps the
    # audit note but must never imply success.
    if approval_note and returncode == 130 and "[Command interrupted]" in output:
        approval_note = approval_note.rstrip(".") + ", then interrupted."

    result_dict = {"output": output, "exit_code": returncode, "error": control_error}
    # Optional fields in observable JSON key order; None means "omit". Spill
    # metadata is present only when output overflowed the capture window.
    optional_fields: list[tuple[str, Any]] = [
        ("cwd", changed_cwd),
        ("environment_recreated", _ENV_RECREATED_NOTE if result.get("environment_recreated") else None),
        *_redact_spill_file(result.get("full_output_path"), result.get("output_total_chars"), command),
        *control_fields,
        ("verification_evidence", _verification_evidence(
            command, command_cwd, session_id or task_id or effective_task_id or "default",
            returncode, output)),
        ("approval", approval_note or None),
        ("exit_code_meaning", exit_note or None),
        ("hint", failure_hint or None),
        ("sudo_auth_failed", True if sudo_auth_failed else None),
        ("sudo_cache_cleared", True if sudo_cache_cleared else None),
    ]
    for key, value in optional_fields:
        if value is not None:
            result_dict[key] = value
    return json.dumps(result_dict, ensure_ascii=False)
