"""Home guidance: the user's own instruction file (``agent.home_guidance_file``) in the stable prompt tier.

Hermes otherwise never reads the user's harness-neutral home instructions (``~/.codex/AGENTS.md`` and the like). A profile can
carry a copy inside its SOUL.md, but the default home, profiles without that copy and delegate children without a profile route
(they load no SOUL) get none. This module gives each agent at most one complete edition, decided once and frozen on the agent:

* :class:`Carried` — one complete ``<workstation_guidance ...>...</workstation_guidance>`` block, from the loaded SOUL.md (which
  then owns delivery: nothing is added), the configured file or the delegating parent.
* :class:`Unavailable` — the configured file could not be delivered whole; the prompt says so instead of carrying partial text.
* ``None`` — nothing applies: no file configured, safe mode, or a constructor that loads no user identity (ignore-rules).

Every built prompt records that decision in one renderer-owned line at the end of its trailer paragraph (the paragraph
``agent/surface_switch.py::identity_line_value`` reads; no identity, project, memory or guidance text comes after it). A
resumed session recovers the decision from that line alone: a carried block is located by offset and verified by length,
SHA-256 and block form, so text quoted inside guidance, SOUL.md or project files never reads as state. A rebuild therefore
never re-reads the file or adopts a block SOUL.md gains mid-session, while safe mode and ignore-rules constructors still get
nothing from a recovered decision.
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
_SETTING = "agent.home_guidance_file"
_UNSAFE_PATH_CHARS = frozenset('"<>\n\r\0')
_SUPERSEDES = "the complete home guidance below supersedes it."
_UNSET = object()
_STATE_PREFIX, _STATE_SUFFIX = "<!-- Hermes home guidance: ", " -->"
_CARRIED_ORIGINS, _UNAVAILABLE_ORIGINS = frozenset({"soul", "file", "parent"}), frozenset({"file", "parent"})


@dataclass(frozen=True)
class Carried:
    """One complete block, delivered byte for byte. ``origin`` is ``soul``, ``file`` or ``parent``."""

    block: str
    origin: str


@dataclass(frozen=True)
class Unavailable:
    """The configured file was not delivered; ``source`` names the setting value, ``reason`` the failure. ``origin`` is
    ``file`` (this agent's own read) or ``parent`` (inherited from the delegating agent)."""

    source: str
    reason: str
    origin: str = "file"

    def notice(self) -> str:
        tag = (f'<workstation_guidance_unavailable source="{html.escape(self.source)}" '
               f'reason="{html.escape(self.reason)}"/>')
        return (f"{tag}\nHome guidance configured as {self.source} was not loaded, in whole or in part: {self.reason}. "
                "Do not assume its rules are present; report this gap to the user.")


HomeGuidance = Carried | Unavailable


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
    agent_cfg = _pb._config_readonly(_SETTING).get("agent") or {}
    return agent_cfg.get("home_guidance_file", "") if isinstance(agent_cfg, dict) else ""


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
    The file is never truncated: over the context-file limit it is :class:`Unavailable`."""
    if not isinstance(value, str):
        return Unavailable(_SETTING, f"the setting must be a path string, not {type(value).__name__}")
    if _UNSAFE_PATH_CHARS & set(value):
        return Unavailable(_SETTING, "the configured path contains a quote, angle bracket, newline or NUL")
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
    cap = _pb._get_context_file_max_chars(context_length)
    if problem is None and len(text) > cap:
        problem = f"{resolved} has {len(text)} characters, over the context-file limit of {cap}; it is never truncated"
    if problem is not None:
        return Unavailable(value, problem)
    if findings := _pb._scan_for_threats(text, scope="context"):
        logger.warning("Home guidance %s matched injection pattern(s) %s; loaded anyway because it is the user's own "
                       "configured file — review it if you did not write that text", resolved, ", ".join(findings))
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Carried(f'<workstation_guidance source="{value}" resolved="{resolved}" sha256="{sha}">\n{text}{_CLOSE}', "file")


def _wants_user_identity(agent: Any) -> bool:
    """The existing SOUL rule (``agent/system_prompt.py::_identity_parts``): ignore-rules constructors want neither."""
    return bool(agent.load_soul_identity or not agent.skip_context_files)


def _permitted(agent: Any, snapshot: Optional[HomeGuidance]) -> Optional[HomeGuidance]:
    """*snapshot* as this agent may render it now. Safe mode allows nothing; a constructor that loads no user identity
    (ignore-rules) allows only what delegation handed it — an unprofiled child has those same flags and still gets its
    parent's edition."""
    from agent.safe_worker_policy import safe_worker_enabled

    if snapshot is None or safe_worker_enabled():
        return None
    return snapshot if snapshot.origin == "parent" or _wants_user_identity(agent) else None


def _resolve(agent: Any, soul: SoulReading, context_length: Optional[int]) -> Optional[HomeGuidance]:
    """The first build's decision: a resumed session's recorded one, else SOUL.md's complete block, else the delegating
    parent's, else the configured file — always within what :func:`_permitted` allows."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return None
    persisted = _own(agent, "_home_guidance_persisted")
    if persisted is not _UNSET:
        return _permitted(agent, persisted)
    if soul.problem is not None:
        logger.warning("SOUL.md has %s; it does not count as delivered home guidance", soul.problem)
    if soul.block is not None:
        return _permitted(agent, Carried(soul.block, "soul"))
    inherited = _own(agent, "_home_guidance_inherited")
    if inherited is not _UNSET and inherited is not None:
        return _permitted(agent, inherited)
    value = _configured_value()
    if value == "" or value is None or not _wants_user_identity(agent):
        return None
    result = load_file(value, context_length)
    if isinstance(result, Unavailable):
        msg = f"⚠️  Home guidance {result.source} was not loaded: {result.reason}"
        logger.warning(msg)
        _pb.queue_prompt_warning(msg)
    return result


def _content(snapshot: HomeGuidance) -> str:
    return snapshot.block if isinstance(snapshot, Carried) else snapshot.notice()


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
    """The decision a stored prompt recorded, or ``_UNSET`` when it records none (a prompt from before this feature).

    Only the last line of the trailer paragraph is read, and only in its exact canonical form."""
    from agent.surface_switch import split_runtime_boundary

    line = split_runtime_boundary(prompt)[0].rsplit("\n\n", 1)[-1].rsplit("\n", 1)[-1]
    if not (line.startswith(_STATE_PREFIX) and line.endswith(_STATE_SUFFIX)):
        return _UNSET
    try:
        record = json.loads(line[len(_STATE_PREFIX):-len(_STATE_SUFFIX)])
        if not isinstance(record, dict) or record.get("v") != 1 or f"{_STATE_PREFIX}{_dump(record)}{_STATE_SUFFIX}" != line:
            raise ValueError("not the canonical home-guidance record")
        return _recorded(prompt, record)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Stored prompt's home-guidance record is unusable (%s); the next build decides afresh", e)
        return _UNSET


def restore_home_guidance(agent: Any, prompt: Optional[str]) -> None:
    """Adopt *prompt*, the stored prompt of the conversation this agent continues, as the source of its decision,
    whether or not those bytes are reused: its record replaces any decision made without it (a pre-turn build that ran
    before the prompt was adopted), so what this agent renders, records and hands its children is the decision of the
    prompt it sends. Without a record (a prompt from before this feature) the next build decides. The decision still
    passes through :func:`_permitted` wherever it is used."""
    if not prompt:
        return
    agent.__dict__.pop("_home_guidance", None)
    state = persisted_state(prompt)
    if state is _UNSET:
        agent.__dict__.pop("_home_guidance_persisted", None)
    else:
        agent._home_guidance_persisted = state


def adopt_stored_prompt(agent: Any) -> None:
    """Bind an agent to the conversation persisted for its session before anything can resolve a decision: an agent
    constructed for an existing session (an ACP or TUI cold restore, a gateway turn) or switched to one (/resume,
    /branch) adopts that session's stored prompt. A pre-turn command (/compress) can then only render that decision.
    Agents that render no prompt here (the gateway-backed CLI agent) are left alone.

    The stored prompt is the durable record of the session's current conversation: everything that continues the
    conversation keeps it — undo or rewind (even to an empty transcript), compression, a reconnect — and the one reset
    that starts a new conversation in the same session removes it (:func:`forget_stored_prompt`). So a stored prompt
    present means the conversation continues; none (a new session, or a reset one) leaves the first build to decide."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id or not callable(getattr(agent, "_build_system_prompt", None)):
        return
    try:
        row = db.get_session(session_id)
    except Exception:
        logger.debug("Stored prompt of session %s unavailable for home guidance", session_id, exc_info=True)
        return
    prompt = row.get("system_prompt") if isinstance(row, dict) else None
    restore_home_guidance(agent, prompt if isinstance(prompt, str) else None)


def forget_stored_prompt(agent: Any, *session_ids: str) -> None:
    """A reset in place (ACP /reset) ended the conversation but keeps its session: drop the stored prompt that recorded
    the ended conversation's decision, so no later restore or reconnect adopts it. The next build stores the new one.
    *session_ids* are the session's rows besides the agent's own (the ACP-facing id after an internal rotation). A
    failed write raises: the caller must not report a reset whose old decision a reconnect would bring back."""
    db = getattr(agent, "_session_db", None)
    if db is None:
        return
    for session_id in dict.fromkeys((getattr(agent, "session_id", None), *session_ids)):
        if session_id:
            db.update_system_prompt(session_id, None)


def end_conversation(agent: Any) -> None:
    """Session boundary (``AIAgent.reset_session_state``: /new, /resume, /branch, ACP /reset), after the prompt that
    rendered the decision was dropped. The decision belongs to the conversation that ended, and so do its recorded and
    inherited inputs: all are dropped. Nothing is adopted here: a reset in place (ACP /reset) starts a new conversation
    and also drops its session's stored prompt (:func:`forget_stored_prompt`); a switch to a stored session adopts that
    session's prompt afterwards (:func:`adopt_stored_prompt`)."""
    for name in ("_home_guidance", "_home_guidance_persisted", "_home_guidance_inherited"):
        agent.__dict__.pop(name, None)


def _recover(agent: Any) -> None:
    """Seed the decision recorded by the prompt this agent currently holds, when none is seeded yet: an agent can hold
    a stored prompt it never built (a same-model /btw or review fork copies only the parent's cached prompt)."""
    if _own(agent, "_home_guidance_persisted") is _UNSET:
        stored = _own(agent, "_cached_system_prompt")
        restore_home_guidance(agent, stored if isinstance(stored, str) else None)


def frozen(agent: Any) -> Optional[HomeGuidance]:
    """What this agent carries now: its own build's decision, else the one its current stored prompt recorded (as this
    agent may render it), else ``None``."""
    snapshot = _own(agent, "_home_guidance")
    if snapshot is not _UNSET:
        return snapshot
    _recover(agent)
    persisted = _own(agent, "_home_guidance_persisted")
    return None if persisted is _UNSET else _permitted(agent, persisted)


def _render(snapshot: Optional[HomeGuidance], soul: SoulReading) -> tuple[Optional[str], str]:
    """``(identity text, part)`` for one build. The frozen result wins over a SOUL.md re-read for a rebuild: a complete
    block the SOUL now carries is replaced in place by the frozen content — or removed when the frozen result is "none"
    — so a session never changes or gains an edition mid-way. The rest of the re-read SOUL.md is used as before."""
    if soul.block is not None:
        return soul.text.replace(soul.block, "" if snapshot is None else _content(snapshot), 1), ""
    if snapshot is None:
        return soul.text, ""
    if soul.problem is None:
        return soul.text, _content(snapshot)
    tail = _SUPERSEDES if isinstance(snapshot, Carried) else "complete home guidance is unavailable."
    return soul.text, (f"Note: this profile's SOUL.md has {soul.problem}, which is not delivered home guidance; "
                       f"{tail}\n\n{_content(snapshot)}")


def apply_home_guidance(agent: Any, soul_text: Optional[str], soul_path: Path,
                        context_length: Optional[int] = None) -> tuple[Optional[str], str]:
    """``(identity text, stable-tier part)`` for this build: resolved on the first build, then frozen on the agent. A
    first build while the agent holds a stored prompt (static-prefix reconstruction, a diagnostic) adopts that prompt's
    recorded decision rather than reading afresh.

    *soul_text* is the SOUL.md text the identity slot loaded (``None`` when none was) and *soul_path* the file it came
    from. The part goes right after identity. The identity text changes only when the SOUL.md re-read for a rebuild
    carries a different edition than the frozen one."""
    soul = read_soul(soul_text, soul_path)
    snapshot = _own(agent, "_home_guidance")
    if snapshot is _UNSET:
        _recover(agent)
        snapshot = agent._home_guidance = _resolve(agent, soul, context_length)
    return _render(snapshot, soul)


def for_child(parent: Any) -> Optional[HomeGuidance]:
    """What a delegate child inherits: its parent's exact block or its parent's failure, never a fresh read."""
    snapshot = frozen(parent)
    if isinstance(snapshot, Carried):
        return Carried(snapshot.block, "parent")
    if isinstance(snapshot, Unavailable):
        return Unavailable(snapshot.source, snapshot.reason, "parent")
    return None
