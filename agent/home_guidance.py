"""Home guidance: the user's own instruction file (``agent.home_guidance_file``) in the stable prompt tier.

Hermes otherwise never reads the user's harness-neutral home instructions (``~/.codex/AGENTS.md`` and the like). A profile can
carry a copy inside its SOUL.md, but the default home, profiles without that copy and delegate children without a profile route
(they load no SOUL) get none. This module gives each agent at most one complete edition, decided once per conversation and
frozen on the agent:

* :class:`Carried` — one complete ``<workstation_guidance ...>...</workstation_guidance>`` block, from the loaded SOUL.md (which
  then owns delivery: nothing is added), the configured file or the delegating parent.
* :class:`Unavailable` — the configured file could not be delivered whole, or the record of a continuing conversation's
  decision is unusable or missing; the prompt says so instead of carrying partial text.
* ``None`` — nothing is configured.

A stored prompt from before decisions were recorded (:data:`LEGACY`) carried whatever its SOUL.md held and added nothing;
restoring it recovers that edition, or its absence, from the prompt's identity slot (:func:`legacy_decision`), and the
conversation keeps it like any recorded decision.

The guidance is the user's or administrator's own operating text, not project context: troubleshooting and minimal-context
modes (``--ignore-rules``, ``--safe-mode``, a platform's ``skip_context_files``, delegates and internal agents) suppress project
rules, the SOUL persona, memory, plugins and profile config, and none of them names this file, so the decision does not depend
on them. What a bypass session may read is fixed where its frozen config is made (:func:`build_bypass_policy`); this module only
reads the config the agent is given.

Every built prompt records the decision in one renderer-owned line at the end of its trailer paragraph (the paragraph
``agent/surface_switch.py::identity_line_value`` reads; no identity, project, memory or guidance text comes after it). A
resumed session recovers the decision from that line alone: a carried block is located by offset and verified by length,
SHA-256 and block form, so text quoted inside guidance, SOUL.md or project files never reads as state. A rebuild of a
continuing conversation (compaction, a capability refresh, a model switch) therefore never re-reads the file or adopts a block
SOUL.md gains mid-session. A continuing conversation whose stored prompt is missing or unreadable cannot show its decision
and carries an explicit :class:`Unavailable` (none where nothing is configured); only a new conversation, or one reset in
place, decides afresh.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent import prompt_builder as _pb

logger = logging.getLogger(__name__)

# A marker is the tag name followed by whitespace, ``>``, ``/`` or the end of the text, so incomplete forms such as
# ``<workstation_guidance`` and ``<workstation_guidance/>`` count while longer names (``..._unavailable``) do not.
_OPEN = re.compile(r"<workstation_guidance(?=[\s>/]|\Z)")
_CLOSE_ANY = re.compile(r"</workstation_guidance(?=[\s>/]|\Z)")
_CLOSE = "</workstation_guidance>"
# A complete block: an opening tag of zero or more name="value" attributes (no quote, angle bracket or newline inside a
# value), then a payload with non-whitespace text, then the exact closing tag. Anything else carrying a marker is malformed.
_BLOCK = re.compile(r'<workstation_guidance(?:\s+[A-Za-z_][\w.-]*="[^"<>\n]*")*\s*>(.*?)</workstation_guidance>', re.DOTALL)
SETTING = "agent.home_guidance_file"
_SETTING_KEY = "home_guidance_file"
_UNSAFE_PATH_CHARS = frozenset('"<>\n\r\0')
_SUPERSEDES = "the complete home guidance below supersedes it."
_UNSET = object()
_STATE_PREFIX, _STATE_SUFFIX = "<!-- Hermes home guidance: ", " -->"
# The renderer-owned opening of a record line; a trailer line that starts with it is an attempted record.
_STATE_OPENING = _STATE_PREFIX.rstrip()
# Where the identity slot ends in every prompt this renderer builds: the opening both Hermes help paragraph variants
# share (unchanged since before decisions were recorded), right after the identity part.
_AFTER_IDENTITY = "\n\n" + os.path.commonprefix([_pb.HERMES_AGENT_HELP_GUIDANCE, _pb.HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS])
_CARRIED_ORIGINS, _UNAVAILABLE_ORIGINS = frozenset({"soul", "file", "parent"}), frozenset({"file", "parent"})
# The trusted file may fill at most this fraction of the model's window (``1 / _WINDOW_SHARE``), measured with the runtime's
# own characters-per-token estimate. The prompt prefix cannot be compacted, and compaction fires at half to three quarters of
# the window (``agent/context_compressor.py``), so a quarter leaves the conversation the larger share even when the estimate
# runs low. It never lowers the limit below the context-file limit, which a profile may raise for larger editions.
_WINDOW_SHARE = 4


@dataclass(frozen=True)
class Carried:
    """One complete block, delivered byte for byte. ``origin`` is ``soul``, ``file`` or ``parent``."""

    block: str
    origin: str

    def __post_init__(self) -> None:
        if self.origin not in _CARRIED_ORIGINS:
            raise ValueError(f"a carried edition's origin is soul, file or parent, not {self.origin!r}")


@dataclass(frozen=True)
class Unavailable:
    """The configured file was not delivered; ``source`` names the setting value, ``reason`` the failure. ``origin`` is
    ``file`` (this agent's own decision) or ``parent`` (inherited from the delegating agent)."""

    source: str
    reason: str
    origin: str = "file"

    def __post_init__(self) -> None:
        if self.origin not in _UNAVAILABLE_ORIGINS:
            raise ValueError(f"an unavailable edition's origin is file or parent, not {self.origin!r}")

    def notice(self) -> str:
        tag = (f'<workstation_guidance_unavailable source="{html.escape(self.source)}" '
               f'reason="{html.escape(self.reason)}"/>')
        return (f"{tag}\nHome guidance configured as {self.source} was not loaded, in whole or in part: {self.reason}. "
                "Do not assume its rules are present; report this gap to the user.")


class Legacy:
    """What :func:`persisted_state` reports for a stored prompt that records no decision: one written before decisions
    were recorded. It is a reading of a prompt, never a decision an agent carries: restoring such a prompt recovers the
    edition it actually carried (:func:`legacy_decision`). There is one instance, :data:`LEGACY`."""

    _instance: Optional[Legacy] = None

    def __new__(cls) -> Legacy:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "LEGACY"


LEGACY = Legacy()

HomeGuidance = Carried | Unavailable


def _checked(value: Any) -> Any:
    """*value* when it is a decision (``None``, :class:`Carried`, :class:`Unavailable`); anything else, :data:`LEGACY`
    included, is a caller error, never a guess."""
    if value is None or isinstance(value, (Carried, Unavailable)):
        return value
    raise TypeError(f"a home-guidance decision is None, Carried or Unavailable, not {type(value).__name__}")


@dataclass(frozen=True)
class SoulReading:
    """What one loaded SOUL.md says about home guidance. ``block`` is set only for exactly one complete block that the
    SOUL reader delivered byte for byte; ``problem`` explains any other marker use, including a block the reader
    truncated or altered (``agent/prompt_builder.py::load_soul_md`` caps and transforms what it returns)."""

    text: Optional[str]
    block: Optional[str] = None
    problem: Optional[str] = None


def scan_blocks(text: str) -> tuple[Optional[str], Optional[str]]:
    """``(block, None)`` for exactly one complete block, ``(None, problem)`` for any other marker use (an unmatched,
    nested, repeated or incomplete marker, a malformed opening tag, an empty payload), ``(None, None)`` when the text
    has no markers."""
    opens, closes = len(_OPEN.findall(text)), len(_CLOSE_ANY.findall(text))
    if not opens and not closes:
        return None, None
    if opens != 1 or closes != 1:
        return None, f"{opens} opening and {closes} closing workstation_guidance markers"
    m = _BLOCK.search(text)
    if m is None:
        return None, "a workstation_guidance tag that is not well formed or not in opening-then-closing order"
    if not m.group(1).strip():
        return None, "an empty workstation_guidance block"
    return m.group(0), None


def read_soul(text: Optional[str], path: Path) -> SoulReading:
    """Classify the SOUL.md text this build loaded. A block counts as complete only if it is identical to the one block in
    the file's exact text at *path* (read without newline translation), so a block the SOUL reader truncated, scanned,
    stripped or newline-normalized is never taken as delivered. The rest of SOUL.md is the reader's, unchanged."""
    block, problem = scan_blocks(text) if text else (None, None)
    if block is None:
        return SoulReading(text, None, problem)
    try:
        source = _pb._read_text_with_timeout(path, encoding="utf-8-sig", newline="")
    except (OSError, UnicodeDecodeError):
        source = None
    if source is None or scan_blocks(source)[0] != block:
        return SoulReading(text, None, "a workstation_guidance block that the SOUL.md reader did not deliver whole")
    return SoulReading(text, block, None)


def _own(agent: Any, name: str) -> Any:
    """An attribute this module or delegation set on *agent*; stand-in agents without it mean "not set"."""
    return getattr(agent, "__dict__", {}).get(name, _UNSET)


def _configured_value() -> Any:
    """The setting as the agent's config reader returns it (the profile's config, or a managed worker's frozen session
    config); ``""`` when it is absent or ``agent`` is not a mapping."""
    agent_cfg = _pb._config_readonly(SETTING).get("agent") or {}
    return agent_cfg.get(_SETTING_KEY, "") if isinstance(agent_cfg, dict) else ""


@dataclass(frozen=True)
class _Limit:
    """How long the file may be: the larger of the context-file limit and a quarter of the model's window, in characters."""

    chars: int
    context_file_cap: int
    window_chars: Optional[int]

    @classmethod
    def for_window(cls, context_length: Optional[int]) -> _Limit:
        from agent.model_metadata import CHARS_PER_TOKEN

        cap = _pb._get_context_file_max_chars(context_length)
        # Same reading as the context-file cap's dynamic branch: only a positive int is a known window.
        known = type(context_length) is int and context_length > 0
        window_chars = context_length * CHARS_PER_TOKEN if known else None
        return cls(max(cap, (window_chars or 0) // _WINDOW_SHARE), cap, window_chars)

    def refusal(self, resolved: str, chars: int) -> str:
        if self.window_chars is None:
            basis = "the context-file limit; the model's window is unknown"
        else:
            basis = (f"the larger of the context-file limit {self.context_file_cap} and a quarter of the model's "
                     f"{self.window_chars}-character window")
        return f"{resolved} has {chars} characters, over the home-guidance limit of {self.chars} ({basis}); it is never truncated"


def _read(path: str) -> tuple[Optional[str], Optional[str]]:
    """``(text, None)`` or ``(None, reason)``: strict UTF-8 without newline translation, so the text re-encodes to the
    file's exact bytes (CRLF included)."""
    try:
        text = _pb._read_text_with_timeout(Path(path), encoding="utf-8", newline="")
    except UnicodeDecodeError:
        return None, f"{path} is not valid UTF-8"
    except OSError as e:
        return None, f"{path} could not be read ({e.strerror or type(e).__name__})"
    if text is None:
        return None, f"reading {path} timed out"
    return text, None


def load_file(value: Any, context_length: Optional[int] = None) -> HomeGuidance:
    """Read the configured file on this host and wrap it in one exact block, or say why it cannot be delivered.

    ``source`` is the configured value, ``resolved`` its real path on this host, ``sha256`` the hash of the exact bytes.
    The file is never truncated: over the home-guidance limit (:class:`_Limit`) it is :class:`Unavailable`."""
    if not isinstance(value, str):
        return Unavailable(SETTING, f"the setting must be a path string, not {type(value).__name__}")
    if _UNSAFE_PATH_CHARS & set(value):
        return Unavailable(SETTING, "the configured path contains a quote, angle bracket, newline or NUL")
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        return Unavailable(value, "the path must be absolute or start with ~")
    resolved = os.path.realpath(path)
    if _UNSAFE_PATH_CHARS & set(resolved):
        return Unavailable(value, "its resolved path contains a quote, angle bracket, newline or NUL")
    text, problem = _read(resolved)
    if problem is None and not text.strip():  # a blank payload is not a complete block (``scan_blocks``)
        problem = f"{resolved} is empty or holds only whitespace"
    if problem is None and (_OPEN.search(text) or _CLOSE_ANY.search(text)):
        problem = f"{resolved} itself contains workstation_guidance markers"
    if problem is None:
        limit = _Limit.for_window(context_length)
        if len(text) > limit.chars:
            problem = limit.refusal(resolved, len(text))
    if problem is not None:
        return Unavailable(value, problem)
    if findings := _pb._scan_for_threats(text, scope="context"):
        logger.warning("Home guidance %s matched injection pattern(s) %s; loaded anyway because it is the user's own "
                       "configured file — review it if you did not write that text", resolved, ", ".join(findings))
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Carried(f'<workstation_guidance source="{value}" resolved="{resolved}" sha256="{sha}">\n{text}{_CLOSE}', "file")


def _resolve(agent: Any, soul: SoulReading, context_length: Optional[int]) -> Any:
    """The first build's decision: the recorded one of the conversation this agent continues, else SOUL.md's complete
    block, else what the delegating parent handed over, else the configured file."""
    persisted = _own(agent, "_home_guidance_persisted")
    if persisted is not _UNSET:
        return _checked(persisted)
    if soul.problem is not None:
        logger.warning("SOUL.md has %s; it does not count as delivered home guidance", soul.problem)
    if soul.block is not None:
        return Carried(soul.block, "soul")
    inherited = _own(agent, "_home_guidance_inherited")
    if inherited is not _UNSET and isinstance(_checked(inherited), (Carried, Unavailable)):
        return inherited
    value = _configured_value()
    if value is None or value == "":
        return None
    result = load_file(value, context_length)
    if isinstance(result, Unavailable):
        msg = f"⚠️  Home guidance {result.source} was not loaded: {result.reason}"
        logger.warning(msg)
        _pb.queue_prompt_warning(msg)
    return result


def _content(snapshot: Any) -> Optional[str]:
    if isinstance(snapshot, Carried):
        return snapshot.block
    if isinstance(snapshot, Unavailable):
        return snapshot.notice()
    return None


def _dump(record: dict) -> str:
    """One canonical single-line JSON object; ``<``, ``>`` and ``&`` are escaped so the line stays one HTML comment."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def state_line(agent: Any, stable: str) -> str:
    """The renderer-owned record of this build's decision, for the end of the trailer paragraph (``\\n`` + record).

    A carried block is recorded by its offset in the stable tier, which is the prompt's prefix, plus its length and
    SHA-256; the bytes themselves appear once, where the model reads them. ``""`` before any decision exists."""
    snapshot = _own(agent, "_home_guidance")
    if snapshot is _UNSET:
        return ""
    _checked(snapshot)
    if snapshot is None:
        record = {"state": "none"}
    elif isinstance(snapshot, Unavailable):
        record = {"state": "unavailable", "origin": snapshot.origin, "source": snapshot.source, "reason": snapshot.reason}
    else:
        at = stable.find(snapshot.block)
        if at < 0:  # the renderer always places the frozen block in the stable tier; never record a guess
            logger.error("Home guidance block missing from the stable prompt tier; its state is not recorded")
            return ""
        record = {"state": "carried", "origin": snapshot.origin, "at": at, "chars": len(snapshot.block),
                  "sha256": hashlib.sha256(snapshot.block.encode("utf-8")).hexdigest()}
    return f"\n{_STATE_PREFIX}{_dump({'v': 1, **record})}{_STATE_SUFFIX}"


def _recorded(prompt: str, record: dict) -> Optional[HomeGuidance]:
    state, origin = record.get("state"), record.get("origin")
    if state == "none" and set(record) == {"v", "state"}:
        return None
    if state == "unavailable" and origin in _UNAVAILABLE_ORIGINS and set(record) == {"v", "state", "origin", "source",
                                                                                   "reason"}:
        if isinstance(record["source"], str) and isinstance(record["reason"], str):
            return Unavailable(record["source"], record["reason"], origin)
    if state == "carried" and origin in _CARRIED_ORIGINS and set(record) == {"v", "state", "origin", "at", "chars", "sha256"}:
        at, chars = record["at"], record["chars"]
        if type(at) is int and type(chars) is int and 0 <= at:
            block = prompt[at:at + chars]
            if (len(block) == chars and hashlib.sha256(block.encode("utf-8")).hexdigest() == record["sha256"]
                    and scan_blocks(block) == (block, None)):
                return Carried(block, origin)
    raise ValueError("a home-guidance record that does not match the prompt it is stored in")


def persisted_state(prompt: str) -> Any:
    """The decision a stored prompt recorded: a recorded decision, :data:`LEGACY` when it records none (a prompt from
    before decisions were recorded), or an explicit :class:`Unavailable` when its record is present but unusable — a
    continuing conversation never decides afresh.

    Only the last line of the trailer paragraph is read, and only in its exact canonical form. That line is the
    renderer's own: one that starts like a record is an attempted record, so a damaged or cut-off one (its closing
    missing, or nothing but the opening left) is unusable, never a legacy prompt."""
    from agent.surface_switch import split_runtime_boundary

    line = split_runtime_boundary(prompt)[0].rsplit("\n\n", 1)[-1].rsplit("\n", 1)[-1]
    if not line.startswith(_STATE_OPENING):
        return LEGACY
    try:
        if not (line.startswith(_STATE_PREFIX) and line.endswith(_STATE_SUFFIX)
                and len(line) >= len(_STATE_PREFIX) + len(_STATE_SUFFIX)):
            raise ValueError("a home-guidance record line that is cut off or not closed")
        record = json.loads(line[len(_STATE_PREFIX):-len(_STATE_SUFFIX)])
        if not isinstance(record, dict) or record.get("v") != 1 or f"{_STATE_PREFIX}{_dump(record)}{_STATE_SUFFIX}" != line:
            raise ValueError("not the canonical home-guidance record")
        return _recorded(prompt, record)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Stored prompt's home-guidance record is unusable (%s); this conversation carries no edition", e)
        return Unavailable(SETTING, f"this conversation's stored home-guidance record is unusable ({e})")


def legacy_decision(prompt: str) -> Optional[HomeGuidance]:
    """The edition a stored prompt from before decisions were recorded (:data:`LEGACY`) actually carried. Such a build
    added nothing to SOUL.md, so the edition is the one complete block in the prompt's identity slot, or none when the
    slot holds no complete block. The slot is found by position — the prompt's first part, which ends where the
    renderer's Hermes help paragraph begins — so a block quoted in project, memory or skill text never counts. A prompt
    in which that boundary is missing or appears more than once cannot show its edition (:func:`_undetermined`)."""
    if prompt.count(_AFTER_IDENTITY) != 1:
        return _undetermined("this conversation began before home-guidance decisions were recorded, and its stored "
                             "prompt does not show which edition it carried")
    block, _problem = scan_blocks(prompt[:prompt.index(_AFTER_IDENTITY)])
    return None if block is None else Carried(block, "soul")


def _undetermined(reason: str) -> Optional[Unavailable]:
    """The decision of a continuing conversation that its stored prompt cannot show (*reason*): an explicit
    :class:`Unavailable`, so it takes up neither the configured file nor a block SOUL.md carries now. Where nothing is
    configured there is nothing it could take up, and a notice would name a setting that does not exist: it carries
    none (``None``, which also keeps a SOUL.md block out of its rebuilds)."""
    value = _configured_value()
    if value is None or value == "":
        return None
    logger.warning("Home guidance: %s; this conversation carries no edition", reason)
    return Unavailable(SETTING, reason)


def restore_home_guidance(agent: Any, prompt: Optional[str], *, continuing: bool = False,
                          unreadable: Optional[str] = None) -> None:
    """Adopt *prompt*, the stored prompt of the conversation this agent continues, as the source of its decision,
    whether or not those bytes are reused: its record replaces any decision made without it (a pre-turn build that ran
    before the prompt was adopted), so what this agent renders, records and hands its children is the decision of the
    prompt it sends. A prompt from before decisions were recorded continues with the edition it carried.

    Without a prompt, a conversation known to continue (*continuing*, :func:`continues_without_prompt`) or whose stored
    prompt could not be read (*unreadable*, the reason) cannot show its decision. Unless this agent already holds its
    own, it never decides afresh (:func:`_undetermined`): a missing prompt is no evidence of a new or reset
    conversation. Otherwise (a new session, or one reset in place) the first build decides."""
    if prompt:
        agent.__dict__.pop("_home_guidance", None)
        state = persisted_state(prompt)
        agent._home_guidance_persisted = legacy_decision(prompt) if state is LEGACY else state
        return
    if not (continuing or unreadable):
        return
    if _own(agent, "_home_guidance") is not _UNSET or _own(agent, "_home_guidance_persisted") is not _UNSET:
        return
    agent._home_guidance_persisted = _undetermined(
        f"this session's stored prompt, which records its home-guidance decision, could not be read ({unreadable})"
        if unreadable else "this conversation's stored prompt, which records its home-guidance decision, is missing")


def _is_reply(message: Any) -> bool:
    """A row a turn writes after it built its prompt: a tool result, or an assistant row other than those Hermes writes
    into a transcript without a turn of its conversation (``agent.message_metadata.written_without_a_turn``: the
    failed-turn notice, written also when no agent was built, and a delivery mirror of an outgoing message another
    session sent). A row folded from several keeps that exclusion only while every row it stands for had it
    (``absorb_turn_provenance``)."""
    from agent.message_metadata import written_without_a_turn

    role = message.get("role") if isinstance(message, dict) else None
    return role == "tool" or (role == "assistant" and not written_without_a_turn(message))


def _replies_to_a_user(messages: list, after: Optional[int] = None) -> bool:
    """Whether *messages* hold a reply (:func:`_is_reply`) to an earlier user message; with *after*, only a reply in a
    row whose id (``_row_id``) is greater counts, and one without an id cannot be placed."""
    asked = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            asked = True
        elif asked and _is_reply(message) and (after is None or (type(message.get("_row_id")) is int
                                                                 and message["_row_id"] > after)):
            return True
    return False


def _used_here(row: dict) -> bool:
    """Whether *row* shows a model call made here: Hermes counts each call and its tokens on the session's row
    (``hermes_state_usage.py``); an import writes none."""
    return any(type(row.get(column)) is int and row[column] > 0
               for column in ("api_call_count", "input_tokens", "output_tokens"))


def continues_without_prompt(row: Any, messages: Any) -> bool:
    """Whether *row*, a session row whose stored prompt is missing, holds a conversation already under way: its
    transcript *messages* hold a reply to a user message, which a turn writes after building the prompt that decided
    (:func:`_is_reply`). A transcript without one shows no decision: a first message stored before its agent is built,
    a cron or webhook brief, outgoing messages mirrored into a fresh session (with or without a brief before them), a
    request whose turn failed before any reply (its failed-turn notice). An ACP fork copies its parent's stored prompt
    instead. Opening turns a client seeds into a new chat cannot be told apart from a conversation under way and count
    as one: never a fresh decision over an existing reply. The stored transcript is read as the conversation's rows
    (:func:`stored_messages`): what compaction summarized away still shows its replies.

    A transcript imported from another tool (``origin_json`` ``imported_from``) was decided by no build here, so its
    first turn here decides. Its replies count only when they came after the import, which records its last message
    row (``hermes_state_portability.FOREIGN_IMPORT_THROUGH``): a reply whose row id (``_row_id``, from
    ``get_messages_as_conversation(include_row_ids=True)``) is later. A row without its id cannot be placed and does
    not count. An import recorded before that boundary existed cannot be split: it is under way once its row shows a
    model call made here (:func:`_used_here`) and it holds a reply. A compression child copies its parent's origin but
    is a continuation: its whole transcript counts."""
    if not isinstance(row, dict) or not isinstance(messages, list):
        return False
    try:
        origin = json.loads(row.get("origin_json") or "{}")
    except (TypeError, ValueError):
        origin = {}
    imported = origin.get("imported_from") if isinstance(origin, dict) else None
    if imported is None or row.get("parent_session_id"):
        return _replies_to_a_user(messages)
    from hermes_state_portability import FOREIGN_IMPORT_THROUGH

    through = imported.get(FOREIGN_IMPORT_THROUGH) if isinstance(imported, dict) else None
    if type(through) is not int or through < 0:
        return _used_here(row) and _replies_to_a_user(messages)
    return _replies_to_a_user(messages, after=through)


def foreign_import_view(db: Any, row: Any) -> Any:
    """*row* as :func:`continues_without_prompt` reads it: a CLI import of a foreign transcript carries no ``origin_json``
    (``hermes --resume`` refuses a row with one), so its ``imported_from`` record, kept beside the row
    (``hermes_state_portability.FOREIGN_IMPORT_META_PREFIX``), is shown in a copy. Any other row is returned as is."""
    if not isinstance(row, dict) or row.get("origin_json") or not isinstance(row.get("id"), str):
        return row
    from hermes_state_portability import FOREIGN_IMPORT_META_PREFIX
    get_meta = getattr(db, "get_meta", None)
    record = get_meta(FOREIGN_IMPORT_META_PREFIX + row["id"]) if callable(get_meta) else None
    try:
        imported = json.loads(record) if record else None
    except ValueError:
        return row
    return {**row, "origin_json": json.dumps({"imported_from": imported})} if isinstance(imported, dict) else row


def stored_messages(db: Any, session_id: str) -> list:
    """The session's stored transcript as :func:`continues_without_prompt` reads it, with each row's id: the rows of
    its conversation, live or summarized away by compaction (``active = 1 OR compacted = 1``). A compaction's summary
    need not keep a reply live (a context engine may return any valid list), but the rows it summarized still belong to
    the conversation; rows the user took back (undo, rewind) or a reset in place ended
    (:func:`forget_stored_prompt`) do not."""
    return db.get_messages_as_conversation(session_id, include_row_ids=True, include_compacted=True)


def adopt_stored_prompt(agent: Any) -> None:
    """Bind an agent to the conversation persisted for its session before anything can resolve a decision: an agent
    constructed for an existing session (an ACP or TUI cold restore, a gateway turn) or switched to one (/resume,
    /branch) adopts that session's stored prompt. A pre-turn command (/compress) can then only render that decision.
    Agents that render no prompt here (the gateway-backed CLI agent) are left alone.

    The stored prompt is the durable record of the session's current conversation: everything that continues the
    conversation keeps it — undo or rewind (even to an empty transcript), compression, a reconnect — and the one reset
    that starts a new conversation in the same session (ACP /reset) clears its messages and removes it
    (:func:`forget_stored_prompt`). So a stored prompt present means the conversation continues, and so does a row
    without one whose transcript is under way (:func:`continues_without_prompt`: a repaired prompt, a failed write):
    :func:`restore_home_guidance` marks that conversation, and one whose row cannot be read, as one whose decision is
    not shown. Any other session (a new one, a reset one, or one holding no reply) leaves the first build to decide."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id or not callable(getattr(agent, "_build_system_prompt", None)):
        return
    try:
        row = foreign_import_view(db, db.get_session(session_id))
        prompt = row.get("system_prompt") if isinstance(row, dict) else None
        prompt = prompt if isinstance(prompt, str) else None
        continuing = (isinstance(row, dict) and not prompt
                      and continues_without_prompt(row, stored_messages(db, session_id)))
    except Exception as e:
        logger.warning("Stored prompt of session %s could not be read for home guidance", session_id, exc_info=True)
        restore_home_guidance(agent, None, unreadable=type(e).__name__)
        return
    restore_home_guidance(agent, prompt, continuing=continuing)


def forget_stored_prompt(agent: Any, *session_ids: str) -> None:
    """A reset in place (ACP /reset) ended the conversation but keeps its session: drop the stored prompt that recorded
    the ended conversation's decision, so no later restore or reconnect adopts it, and take the ended conversation's
    rows out of the session's conversation (``SessionDB.end_conversation_in_place``), so neither its live rows (an
    agent that persists its own turns leaves them in place) nor those compaction summarized away read as a
    conversation under way. The next build stores the new prompt. *session_ids* are the session's rows besides the
    agent's own (the ACP-facing id after an internal rotation). For each, the conversation ends where it is read
    back: the id itself, the session every resume reads for it (``SessionDB.resolve_resume_session_id``: after a
    rotation the tip, even once a reconnect reopened the id and so cut it from its compression lineage), and every
    compression segment of both (``SessionDB.get_compression_lineage``: after more than one rotation, the
    intermediate ones too), so neither the model history nor the display lineage of a resume holds the ended
    conversation. Each segment ends in its own transaction, the agent's own first. A failed read of what a resume
    selects, or a failed write, raises: the caller must not report a reset whose old decision a reconnect would bring
    back; a retry ends the rest."""
    db = getattr(agent, "_session_db", None)
    if db is None:
        return
    segments = []
    for session_id in dict.fromkeys(filter(None, (getattr(agent, "session_id", None), *session_ids))):
        resumed = db.resolve_resume_session_id(session_id, strict=True)
        segments += [session_id, *reversed(db.get_compression_lineage(session_id)),
                     resumed, *reversed(db.get_compression_lineage(resumed))]
    for session_id in dict.fromkeys(segments):
        db.end_conversation_in_place(session_id)


def end_conversation(agent: Any) -> None:
    """Session boundary (``AIAgent.reset_session_state``: /new, /resume, /branch, ACP /reset), after the prompt that
    rendered the decision was dropped. The decision belongs to the conversation that ended, and so do its recorded and
    inherited inputs: all are dropped. Nothing is adopted here: /new and a reset in place (ACP /reset, which also drops
    its session's stored prompt) start a conversation whose first build decides; /resume and /branch then adopt the
    target session's stored prompt (:func:`adopt_stored_prompt`), whose recorded decision continues."""
    for name in ("_home_guidance", "_home_guidance_persisted", "_home_guidance_inherited"):
        agent.__dict__.pop(name, None)


def _recover(agent: Any) -> None:
    """Seed the decision recorded by the prompt this agent currently holds, when none is seeded yet: an agent can hold
    a stored prompt it never built (a same-model /btw or review fork copies only the parent's cached prompt)."""
    if _own(agent, "_home_guidance_persisted") is _UNSET:
        stored = _own(agent, "_cached_system_prompt")
        restore_home_guidance(agent, stored if isinstance(stored, str) else None)


def frozen(agent: Any) -> Any:
    """What this agent carries now: its own build's decision, else the one its current stored prompt recorded, else
    ``None``."""
    snapshot = _own(agent, "_home_guidance")
    if snapshot is not _UNSET:
        return snapshot
    _recover(agent)
    persisted = _own(agent, "_home_guidance_persisted")
    return None if persisted is _UNSET else persisted


def _render(snapshot: Any, soul: SoulReading) -> tuple[Optional[str], str]:
    """``(identity text, part)`` for one build. The frozen decision wins over a SOUL.md re-read for a rebuild: a complete
    block the SOUL now carries is replaced in place by the frozen content — or removed when the decision is "none" — so
    a session never changes or gains an edition mid-way. The rest of the re-read SOUL.md is used as before.

    A SOUL.md block the reader did not deliver byte for byte (newline-translated CRLF, say) is never delivered home
    guidance, but a conversation from before decisions were recorded carried exactly what that reader loaded
    (:func:`legacy_decision`): when the text's one complete block is the frozen edition, the text already shows it once."""
    content = _content(snapshot)
    if soul.block is not None:
        return soul.text.replace(soul.block, content or "", 1), ""
    if content is None:
        return soul.text, ""
    if soul.problem is None:
        return soul.text, content
    if isinstance(snapshot, Carried) and scan_blocks(soul.text)[0] == snapshot.block:
        return soul.text, ""
    tail = _SUPERSEDES if isinstance(snapshot, Carried) else "complete home guidance is unavailable."
    return soul.text, (f"Note: this profile's SOUL.md has {soul.problem}, which is not delivered home guidance; "
                       f"{tail}\n\n{content}")


def apply_home_guidance(agent: Any, soul_text: Optional[str], soul_path: Path,
                        context_length: Optional[int] = None) -> tuple[Optional[str], str]:
    """``(identity text, stable-tier part)`` for this build: resolved on the first build, then frozen on the agent. A
    first build while the agent holds a stored prompt (static-prefix reconstruction, a diagnostic) adopts that prompt's
    recorded decision rather than deciding afresh.

    *soul_text* is the SOUL.md text the identity slot loaded (``None`` when none was) and *soul_path* the file it came
    from. The part goes right after identity. The identity text changes only when the SOUL.md re-read for a rebuild
    carries a different edition than the frozen one."""
    soul = read_soul(soul_text, soul_path)
    snapshot = _own(agent, "_home_guidance")
    if snapshot is _UNSET:
        _recover(agent)
        snapshot = agent._home_guidance = _resolve(agent, soul, context_length)
    return _render(_checked(snapshot), soul)


def for_child(parent: Any) -> Optional[HomeGuidance]:
    """What a delegate child inherits: its parent's exact block or its parent's failure, never a fresh read of either. A
    parent that carries nothing (nothing configured, or a conversation older than its configuration, or one from before
    decisions were recorded whose SOUL.md held no block) hands nothing over, and the child — a new conversation —
    decides from its own inputs."""
    snapshot = _checked(frozen(parent))
    if isinstance(snapshot, Carried):
        return Carried(snapshot.block, "parent")
    if isinstance(snapshot, Unavailable):
        return Unavailable(snapshot.source, snapshot.reason, "parent")
    return None


def _capture_managed_setting(config: dict) -> None:
    """The bypass producer's capture (``gateway/session_local.py::_bypass_policy``): a --safe-mode / --ignore-user-config
    session never reads its profile, but the administrator's managed ``agent.home_guidance_file`` is not profile config
    (``website/docs/user-guide/managed-scope.md``: values a standard user cannot override). Copy that one pinned value, as
    the managed overlay expands it, into the frozen *config* the session is created with, so the session's worker reads it
    from its own snapshot and a later edit of the managed file never changes a running session. Nothing else of the
    managed layer is copied; without a managed pin *config* is unchanged."""
    from hermes_cli.managed_scope import apply_managed_overlay

    managed = apply_managed_overlay({})
    pinned = managed.get("agent") if isinstance(managed, dict) else None
    if not isinstance(pinned, dict) or _SETTING_KEY not in pinned:
        return
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = config["agent"] = {}
    agent_cfg[_SETTING_KEY] = pinned[_SETTING_KEY]


def build_bypass_policy(params: Any, config: dict, *args: Any, **kwargs: Any) -> Any:
    """``gateway.session_policy.build_policy`` for the bypass producer: the same call, with the administrator's pinned
    home-guidance setting captured into *config* first (:func:`_capture_managed_setting`). ``_bypass_policy`` binds this
    in place of ``build_policy``; every argument passes through unchanged, and the frozen policy is ``build_policy``'s
    own."""
    from gateway.session_policy import build_policy

    _capture_managed_setting(config)
    return build_policy(params, config, *args, **kwargs)
