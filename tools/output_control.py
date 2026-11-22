"""Caller-controlled output persistence: select, budget and destination chosen before any spill.

A tool that declares the ``output_control`` parameter (``OUTPUT_CONTROL_PROPERTY`` in its
schema) lets the caller decide, *before* the result materializes anywhere:

* ``select`` — RFC 6901 JSON pointers. The requested content becomes
  ``{"selected": {pointer: value, ...}, "missing": [pointer, ...]}`` instead of the whole
  JSON result; the unselected remainder is never written anywhere.
* ``inline_chars`` — how many characters of the requested content the model sees inline.
* ``destination`` — an absolute path of a NEW file that receives the complete requested
  content. Hermes' default spill locations (``cache/spillover``, ``cache/terminal-output``,
  sandbox ``hermes-results``) are then never used for this call: a destination that cannot
  be created fails the call before the tool runs, and a write that fails midway reports
  ``complete: false`` with the bytes that did land — the payload is never rerouted.

Without ``destination`` the requested content follows the ordinary persistence path, with
``inline_chars`` as its spill threshold, so it stays retrievable exactly as before.

Destination files hold the UTF-8 text the tool produced, unaltered: no ANSI stripping, no
redaction, no truncation. They are created ``0600`` with ``O_EXCL | O_NOFOLLOW``, so an
existing file or a planted symlink is refused, never overwritten or followed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
from tools.tool_result_storage import _OUTPUT_CONTROLLED

PARAM = "output_control"
# Private tool-message flag: a controlled envelope, bounded once by the executor before its row is
# flushed; the per-turn aggregate budget never rewrites or archives it afterwards.
CONTROLLED_MESSAGE_FLAG = _OUTPUT_CONTROLLED
# ``inline_chars`` ceiling: the largest per-result size any budget allows inline.
MAX_INLINE_CHARS = DEFAULT_RESULT_SIZE_CHARS
# A command's stdout is held in memory (never on disk) until ``select`` projects it; larger
# output cannot be projected and the call reports that instead of spilling the raw stream.
SELECT_CAPTURE_CEILING_CHARS = 5_000_000
# Inline payload of destination-delivered content when the caller gives no ``inline_chars``.
DEFAULT_PREVIEW_CHARS = 1_500

OUTPUT_CONTROL_PROPERTY: dict[str, Any] = {
    "type": "object",
    "description": (
        "Decide before the result is stored. destination: absolute path of a NEW file (parent "
        "must exist) that receives the complete requested content; Hermes then writes this "
        "result nowhere else. select: JSON pointers (e.g. \"/task/body\") — only those values "
        "are kept, as {\"selected\": {...}, \"missing\": [...]}. inline_chars: characters shown "
        "inline (0 = metadata only)."
    ),
    "properties": {
        "destination": {"type": "string"},
        "select": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "inline_chars": {"type": "integer", "minimum": 0, "maximum": MAX_INLINE_CHARS},
    },
    "additionalProperties": False,
}


# RFC 6901: "~" only as "~0" / "~1"; array indices are ASCII decimal without leading zeros
# (``str.isdigit`` would also accept other Unicode digits such as "\u0663").
_INVALID_ESCAPE = re.compile(r"~(?![01])")
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]*", re.ASCII)


class OutputControlError(ValueError):
    """The ``output_control`` argument is malformed, or its destination cannot be created."""


@dataclass(frozen=True)
class JsonPointer:
    text: str
    tokens: tuple[str, ...]

    @classmethod
    def parse(cls, text: Any) -> JsonPointer:
        if not isinstance(text, str) or (text and not text.startswith("/")):
            raise OutputControlError(
                f"output_control.select entries must be JSON pointers (\"\" or starting with \"/\"); got {text!r}")
        if _INVALID_ESCAPE.search(text):
            raise OutputControlError(f"output_control.select pointer {text!r}: \"~\" must be followed by 0 or 1 (RFC 6901)")
        tokens = tuple(t.replace("~1", "/").replace("~0", "~") for t in text.split("/")[1:])
        return cls(text, tokens)

    def resolve(self, document: Any) -> tuple[bool, Any]:
        """``(True, value)`` when the pointer names a value in *document*, else ``(False, None)``."""
        node = document
        for token in self.tokens:
            if isinstance(node, dict) and token in node:
                node = node[token]
            elif isinstance(node, list) and _ARRAY_INDEX.fullmatch(token) and _index_in_range(token, len(node)):
                node = node[int(token)]
            else:
                return False, None
        return True, node


def _index_in_range(token: str, length: int) -> bool:
    """Whether the canonical array-index *token* (no leading zeros) is below *length*, decided
    by its digit count first: an arbitrarily long valid index never reaches ``int()``."""
    digits = len(str(length))
    return len(token) < digits or (len(token) == digits and int(token) < length)


@dataclass(frozen=True)
class OutputControl:
    destination: Path | None
    select: tuple[JsonPointer, ...] | None
    inline_chars: int | None

    @classmethod
    def parse(cls, raw: Any) -> OutputControl:
        if not isinstance(raw, dict):
            raise OutputControlError("output_control must be an object with destination, select and/or inline_chars")
        unknown = sorted(set(raw) - set(OUTPUT_CONTROL_PROPERTY["properties"]))
        if unknown:
            raise OutputControlError(f"output_control: unknown field(s) {', '.join(unknown)}")
        destination = raw.get("destination")
        if destination is not None:
            if not isinstance(destination, str) or not destination or "\x00" in destination:
                raise OutputControlError("output_control.destination must be a non-empty path string")
            if not os.path.isabs(destination):
                raise OutputControlError(f"output_control.destination must be an absolute path; got {destination!r}")
            if os.path.normpath(destination) != destination:
                raise OutputControlError(
                    f"output_control.destination must be a normalized path (no \"..\", \".\" or repeated "
                    f"separators); got {destination!r}")
            destination = Path(destination)
        select = raw.get("select")
        if select is not None:
            if not isinstance(select, list) or not select:
                raise OutputControlError("output_control.select must be a non-empty list of JSON pointers")
            select = tuple(JsonPointer.parse(p) for p in select)
        inline = raw.get("inline_chars")
        if inline is not None and (isinstance(inline, bool) or not isinstance(inline, int)
                                   or not 0 <= inline <= MAX_INLINE_CHARS):
            raise OutputControlError(f"output_control.inline_chars must be an integer from 0 to {MAX_INLINE_CHARS}")
        return cls(destination, select, inline)

    def project(self, document: Any) -> dict:
        """The requested content for *document* under ``select`` (requires ``select``)."""
        assert self.select is not None
        selected, missing = {}, []
        for pointer in self.select:
            found, value = pointer.resolve(document)
            if found:
                selected[pointer.text] = value
            else:
                missing.append(pointer.text)
        return {"selected": selected, "missing": missing}


def parse_args(args: dict) -> OutputControl | None:
    """The call's control, ``None`` when absent; raises :class:`OutputControlError` when malformed."""
    raw = args.get(PARAM)
    return None if raw is None else OutputControl.parse(raw)


def control_for_call(tool_name: str, args: Any) -> OutputControl | None:
    """The control a completed call ran under, for the executor's persistence step.

    ``None`` unless *tool_name* declares the parameter and *args* carries a well-formed value —
    a malformed one was already refused by the tool, whose error result needs no special care."""
    if not isinstance(args, dict) or args.get(PARAM) is None:
        return None
    from tools.arg_coercion import coerce_tool_args
    from tools.registry import registry
    properties = (registry.get_schema(tool_name) or {}).get("parameters", {}).get("properties", {})
    if PARAM not in properties:
        return None
    try:
        # The dispatcher handed the handler coerced arguments (a JSON-encoded object becomes a
        # dict); normalize the same way so the handler and this boundary see one control.
        return OutputControl.parse(coerce_tool_args(tool_name, dict(args)).get(PARAM))
    except OutputControlError:  # malformed: the tool already refused it with an ordinary error
        return None


@dataclass(frozen=True)
class Receipt:
    """What landed in a destination: ``complete`` is False when a write failed midway."""
    path: Path
    chars: int
    bytes: int
    sha256: str
    complete: bool
    error: str | None = None

    def fields(self) -> dict:
        out = {"destination": str(self.path), "chars": self.chars, "bytes": self.bytes,
               "sha256": self.sha256, "complete": self.complete}
        if self.error:
            out["error"] = self.error
        return out


def check_destination(path: Path) -> None:
    """Raise :class:`OutputControlError` unless *path* can be created as a new file now.
    Lets a producer refuse before reading anything; :class:`Destination` re-checks and then
    creates exclusively, so a race only turns into the same refusal."""
    parent = path.parent
    if not parent.is_dir():
        raise OutputControlError(f"output_control.destination parent directory does not exist: {parent}")
    # O_NOFOLLOW guards only the last component; a symlinked ancestor could still redirect
    # the write, so the parent must already be its own canonical path.
    if os.path.realpath(parent) != str(parent):
        raise OutputControlError(
            f"output_control.destination parent must not involve symlinks: {parent} resolves to "
            f"{os.path.realpath(parent)}")
    if os.path.lexists(path):
        raise OutputControlError(f"output_control.destination already exists (never overwritten): {path}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise OutputControlError(f"output_control.destination parent directory is not writable: {parent}")
    protected = _protected_destination(path)
    if protected:
        raise OutputControlError(f"output_control.destination is a protected path: {protected}")


def _protected_destination(path: Path) -> str | None:
    """Why *path* is off limits to a Hermes-written file, else ``None``: the guards ``write_file``
    applies (credential/system paths, protected Hermes state, HERMES_WRITE_SAFE_ROOT, Hermes
    config, agent-instruction files). The approval-gated ones are refused outright — this write
    has no approval channel, and a destination is a new file, never an edit that needs one."""
    from agent.file_safety import get_write_denied_error, is_write_approval_required
    from tools.file_tools_write_guards import _check_sensitive_path, _protected_instruction_reason
    text = str(path)
    denied = get_write_denied_error(text, verb="Write") or _check_sensitive_path(text)
    if denied:
        return denied
    if is_write_approval_required(text):
        return f"{text} is an approval-gated file"
    instruction = _protected_instruction_reason(text)
    return f"{text} is a protected agent-instruction file ({instruction})" if instruction else None


class Destination:
    """An exclusively created caller file, written incrementally, hashed and size-checked on close."""

    def __init__(self, path: Path):
        check_destination(path)
        try:
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                               | getattr(os, "O_CLOEXEC", 0), 0o600)
        except FileExistsError:
            raise OutputControlError(f"output_control.destination already exists (never overwritten): {path}") from None
        except OSError as exc:
            raise OutputControlError(f"output_control.destination cannot be created: {path}: {exc.strerror or exc}") from None
        self.path = path
        self._identity = os.fstat(self._fd)
        self._hash = hashlib.sha256()
        self._chars = 0
        self._bytes = 0
        self._error: str | None = None
        self._receipt: Receipt | None = None
        self._hold = 0
        self._pending = ""

    def hold_back(self, chars: int) -> None:
        """Keep the last *chars* characters unwritten until :meth:`finish_tail`, so the producer
        can consume the trailer it appended to the stream (the shell wrapper's marker lines)."""
        self._hold = max(0, int(chars))

    def finish_tail(self, transform: Callable[[str], str]) -> None:
        """Write the held-back tail through *transform* (the stream has ended)."""
        pending, self._pending, self._hold = self._pending, "", 0
        self._write_now(transform(pending))

    def write(self, text: str) -> None:
        if not self._hold:
            self._write_now(text)
            return
        self._pending += text
        if len(self._pending) > self._hold:
            cut = len(self._pending) - self._hold
            ready, self._pending = self._pending[:cut], self._pending[cut:]
            self._write_now(ready)

    def _write_now(self, text: str) -> None:
        if self._error is not None or self._receipt is not None or not text:
            return
        data = text.encode("utf-8", errors="replace")
        done = 0
        try:
            while done < len(data):
                written = os.write(self._fd, memoryview(data)[done:])
                self._hash.update(memoryview(data)[done:done + written])
                done += written
            self._bytes += done
            self._chars += len(text)
        except OSError as exc:
            # Count exactly what landed: whole characters of the written prefix.
            self._bytes += done
            self._chars += len(data[:done].decode("utf-8", errors="ignore"))
            self._error = f"write failed after {self._bytes} bytes: {exc.strerror or exc}"

    def reset(self) -> None:
        """Discard everything written so far (the producer is about to start over)."""
        if self._receipt is not None:
            return
        try:
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
        except OSError as exc:
            self._error = f"reset failed: {exc.strerror or exc}"
            return
        self._hash, self._chars, self._bytes, self._error = hashlib.sha256(), 0, 0, None
        self._pending = ""

    def discard_unused(self) -> None:
        """The producer failed before delivering anything: close, and remove the file this
        object created — only while it is still that same empty file (never another one)."""
        if self._receipt is not None:
            return
        receipt = self.close()
        if receipt.bytes:
            return
        try:
            current = os.lstat(self.path)
            if (current.st_dev, current.st_ino) == (self._identity.st_dev, self._identity.st_ino) \
                    and current.st_size == 0:
                os.unlink(self.path)
        except OSError:
            pass

    def close(self) -> Receipt:
        if self._receipt is not None:
            return self._receipt
        if self._pending:
            pending, self._pending, self._hold = self._pending, "", 0
            self._write_now(pending)
        try:
            os.fsync(self._fd)
            on_disk = os.fstat(self._fd).st_size
            if self._error is None and on_disk != self._bytes:
                self._error = f"{on_disk} bytes on disk, expected {self._bytes}"
        except OSError as exc:
            self._error = self._error or f"flush failed: {exc.strerror or exc}"
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._receipt = Receipt(self.path, self._chars, self._bytes, self._hash.hexdigest(),
                                complete=self._error is None, error=self._error)
        return self._receipt


def open_destination(control: OutputControl | None) -> Destination | None:
    """Create the destination file now, before the tool runs; raises :class:`OutputControlError`."""
    return Destination(control.destination) if control is not None and control.destination is not None else None


def envelope(output: str, meta: dict, *, full_output_path: str | None = None, **extra: Any) -> dict:
    """The controlled result shape every declaring tool returns: ``output`` (the inline payload),
    ``full_output_path`` (where the complete requested content lives, when stored) and the
    ``output_control`` metadata. Terminal adds its own fields (``exit_code`` …) via *extra*."""
    out: dict[str, Any] = {"output": output}
    out.update(extra)
    if full_output_path is not None:
        out["full_output_path"] = full_output_path
    out[PARAM] = meta
    return out


def dump_envelope(data: dict) -> str:
    """The one serialized form of a bounded envelope; its length is what budgets measure."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def parse_envelope(result: Any) -> dict | None:
    """*result* as a controlled envelope dict, else ``None`` (plain results, tool errors)."""
    if not isinstance(result, str) or PARAM not in result:
        return None
    try:
        data = json.loads(result)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("output"), str) and isinstance(data.get(PARAM), dict):
        return data
    return None


def cut_inline(text: str, chars: int) -> str:
    """A bounded inline payload: head + notice + tail, never longer than *chars* plus the notice."""
    if chars <= 0:
        return ""
    from tools.tool_output_truncate import truncate_head_tail
    return truncate_head_tail(text, chars)


def bound_envelope(data: dict, *, inline_chars: int | None, limit: float,
                   archive: Callable[[str], str | None] | None, receipt_limit: float | None = None) -> dict:
    """Bound a controlled envelope IN PLACE OF archiving it: the inline payload shrinks to the
    caller's ``inline_chars`` and until the whole serialized result fits *limit*; metadata,
    ``full_output_path`` and every other field survive. A payload about to lose characters
    with nowhere else to live is first stored verbatim by *archive* (only offered when the
    caller named no destination); without a store the metadata says the inline payload is all
    there is. Metadata is never cut: when it alone exceeds *limit* the payload is empty and the
    receipt is compacted; it may then use up to *receipt_limit* (default *limit*: the room a
    caller held back for others, which a mandatory receipt may draw on), and only beyond that
    ``output_control.over_budget`` records the limit that could not be met."""
    original = data["output"]
    base_meta = dict(data[PARAM])
    stored = "full_output_path" in data or bool(base_meta.get("destination"))
    if not stored and archive is not None and base_meta.get("inline_complete", True):
        # Decide storage first so the measured result carries its final recovery fields.
        needs_cut = (inline_chars is not None and len(original) > inline_chars) or \
            len(dump_envelope(data)) > limit
        path = archive(original) if needs_cut else None
        if path:
            data = {**data, "full_output_path": path}
            base_meta.setdefault("chars", len(original))
            stored = True

    def render(payload: str, over_budget: int | None = None) -> dict:
        cut = payload != original
        meta = {**base_meta, "inline_chars": len(payload),
                "inline_complete": not cut and base_meta.get("inline_complete", True)}
        if cut and not stored:
            meta.update(complete=False, error="inline payload was cut and could not be stored elsewhere")
        if over_budget is not None:
            meta["over_budget"] = over_budget
        return {**data, "output": payload, PARAM: meta}

    def size(result: dict) -> int:
        return len(dump_envelope(result))

    target = len(original) if inline_chars is None else min(len(original), inline_chars)
    result = render(original if target >= len(original) else cut_inline(original, target))
    for _ in range(12):  # converges in a few rounds; JSON escaping makes the size non-linear
        if size(result) <= limit or not result["output"]:
            break
        target = max(0, min(target, len(result["output"])) - (size(result) - int(limit)) - 1)
        result = render(cut_inline(original, target))
    if size(result) > limit and not result["output"]:
        result = _compact(result)
    hard = limit if receipt_limit is None else max(limit, receipt_limit)
    if size(result) > hard:
        result = render(result["output"], over_budget=int(hard)) if result["output"] else \
            {**result, PARAM: {**result[PARAM], "over_budget": int(hard)}}
    return result


def _compact(result: dict) -> dict:
    """The same receipt without what it repeats or what an empty payload makes evident: the
    ``full_output_path`` that equals the destination, and the inline statistics."""
    meta = {key: value for key, value in result[PARAM].items() if key not in ("inline_chars", "inline_complete")}
    if meta.get("destination") is None or result.get("full_output_path") != meta["destination"]:
        return {**result, PARAM: meta}
    return {**{key: value for key, value in result.items() if key != "full_output_path"}, PARAM: meta}


def failed_delivery(receipt: Receipt, **extra: Any) -> dict:
    """Envelope for a destination write that did not complete. Carries no payload."""
    return envelope("", receipt.fields(), error=(
        f"output_control.destination write incomplete ({receipt.error}); {receipt.bytes} bytes "
        f"are in {receipt.path}. The result was not stored anywhere else."), **extra)


def render_json_result(payload: Any, control: OutputControl | None,
                       serialize: Callable[[Any], str] = json.dumps, *, sink: Destination | None = None) -> str:
    """Apply *control* to a JSON-producing tool's result object (``serialize(payload)`` when absent).

    Select-only controls return the projection as an ordinary result. With ``inline_chars`` or a
    destination the result is a controlled envelope; the executor then bounds it to the caller's
    and the model's budgets (``bound_envelope``). *sink* is the destination the producer opened
    before computing *payload* (``open_destination``); it is opened here when not given."""
    if control is None:
        return serialize(payload)
    requested = control.project(payload) if control.select is not None else payload
    text = serialize(requested)
    if control.destination is None and control.inline_chars is None:
        return text
    meta: dict[str, Any] = {"chars": len(text)}
    if control.select is not None:
        # The pointers themselves are in the requested content; repeating them here could
        # outgrow every budget (a pointer may be arbitrarily long).
        meta["missing_count"] = len(requested["missing"])
    if control.destination is None:
        return json.dumps(envelope(text, {**meta, "complete": True}), ensure_ascii=False)
    destination = sink if sink is not None else Destination(control.destination)
    destination.write(text)
    receipt = destination.close()
    if not receipt.complete:
        return json.dumps(failed_delivery(receipt), ensure_ascii=False)
    inline = control.inline_chars if control.inline_chars is not None else DEFAULT_PREVIEW_CHARS
    preview = cut_inline(text, inline)
    return json.dumps(envelope(preview, {**meta, **receipt.fields(), "inline_complete": preview == text},
                               full_output_path=str(receipt.path)), ensure_ascii=False)


@dataclass(frozen=True)
class StreamCapture:
    """How a foreground command's stream is captured under a control (``tools.environments``).

    ``window_chars`` is the retained head/tail window; ``sink`` receives every chunk from
    byte 0; ``default_spill`` keeps Hermes' ``cache/terminal-output`` overflow tee."""
    window_chars: int
    sink: Destination | None
    default_spill: bool

    @classmethod
    def for_control(cls, control: OutputControl, sink: Destination | None, default_window: int) -> StreamCapture:
        if control.select is not None:
            # Hold the raw stream in memory for projection; it is never written to disk.
            return cls(SELECT_CAPTURE_CEILING_CHARS, None, default_spill=False)
        if sink is None:
            # Nothing else will hold the stream: keep the ordinary window (and overflow tee) so
            # the executor can archive the complete requested output before any inline cut.
            return cls(max(1, default_window), None, default_spill=True)
        window = control.inline_chars if control.inline_chars is not None else default_window
        return cls(max(1, window), sink, default_spill=False)


def env_supports_stream_capture(env: Any) -> bool:
    """True when *env*'s ``execute`` honours ``capture=`` (the streaming collector path)."""
    if not getattr(env, "supports_stream_capture", False):
        return False
    try:
        params = inspect.signature(env.execute).parameters
    except (TypeError, ValueError):
        return False
    return "capture" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
