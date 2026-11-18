"""Header-placeholder display fidelity for agent.redact and agent.redact_headers.

A secret-header or Authorization value that is ONLY a value-free placeholder (``Bearer %s\\n``,
``$OWNER_TOKEN``, ``${VAR}``, ``{{ token }}``, ``<YOUR_API_TOKEN>``) carries no credential, so it is
shown byte for byte (r15, 560fe9d754; display-fidelity intake 10123). Everything else keeps the
masking of the pass that found it: this module only ever says "keep the original text" or
"mask more" (a placeholder-led value with glued material), never "mask less" than that pass.
"""

import bisect
import json
import re

# A header value that is ONLY a value-free placeholder carries no credential material: a
# printf/%-format specifier (``Bearer %s\n``, ``%.*s``, ``%1$s``, ``% s``), a shell reference
# (``$OWNER_TOKEN``, ``${SERVICE_2026_TOKEN}``, ``$1``, ``${10}``), or a documented
# template/doc placeholder (``{token}``, ``{{ token }}``, ``{0}``, ``<YOUR_API_TOKEN>``),
# optionally followed by literal escape sequences (``\n``; ``\\n`` once JSON-encoded, e.g. in
# native session history). Masking those shows ``Bearer ***`` for source that on disk says
# ``Bearer %s`` and deletes the escape too, so readers draw false conclusions about the
# source (display-fidelity intake 10123).
#
# Exempt is ONE COMPLETE placeholder that starts the header credential slot and ends at a
# recognised boundary (see ``_AUTH_AFTER_RE`` / ``_SECRET_AFTER_RE``). The header regexes
# stop at whitespace, so a placeholder with inner whitespace (``{{ token }}``, ``% s``) is
# read from the original text past the slot -- for classification only; the emitted text
# is either the untouched original or the masking pass's own mask. Anything that can carry a
# literal value stays masked as before: defaults/modifiers (``${T:-value}``), command
# substitutions, dotted/indexed fields, wrappers whose content is not a documented
# placeholder name (``<TOKEN_AbCdEf…>``, ``<SyntheticSecret>``, ``<syn7short>``), and any
# value material glued before or after the placeholder (``%sAbCd``, ``{{ token }}AbCd``,
# ``$TOKEN\AbCd``, ``$TOKEN}AbCd``).
_FMT_ARG = r"(?:[1-9]\d?\$)?"
_FMT_SPEC = (rf"%{_FMT_ARG}(?:\((?P<fmt>[A-Za-z_]\w*)\))?[-+ #0]*(?:\d{{1,3}}|\*{_FMT_ARG})?"
             rf"(?:\.(?:\d{{0,3}}|\*{_FMT_ARG}))?[sdiouxXeEfFgGcrabqv]")
_VAR_REF = r"\$(?:\{(?P<braced>[A-Za-z_]\w*)\}|\{\d{1,2}\}|(?P<bare>[A-Za-z_]\w*)|[1-9@*])"
# Jinja/GitHub-style ``{{ name }}`` with any run of spaces/tabs (``\t``, ``\\t`` once
# JSON-encoded once or twice).
_TEMPLATE_WS = r"(?:[ \t]|\\+t)*"
_TEMPLATE_FIELD = (rf"\$?\{{\{{{_TEMPLATE_WS}(?P<jinja>[A-Za-z_][\w-]*){_TEMPLATE_WS}\}}\}}"
                   r"|\{(?:\d?|(?P<field>[A-Za-z_][\w-]*))\}")
_ANGLE_PLACEHOLDER = r"<(?P<angle>[A-Za-z_][\w-]*)>"
_CREDENTIAL_PLACEHOLDER_RE = re.compile(
    rf"(?:{_FMT_SPEC}|{_VAR_REF}|{_TEMPLATE_FIELD}|{_ANGLE_PLACEHOLDER})(?:\\+[nrt])*", re.ASCII)
# What may follow the placeholder, up to the next whitespace. A backslash alone is NOT a
# boundary: only a backslash run that escapes a closing quote is.
# Authorization: the slot itself stops at a quote and base never masked past it, so after
# an actual ``"``/``'`` anything may follow -- however long (a compact JSON line continues
# past the closing quote without whitespace; review6366 HF-11). Matched at the placeholder
# end: whitespace/end of text, or a (backslash-escaped) quote.
_AUTH_AFTER_RE = re.compile(r"(?!\S)|\\{0,127}[\"']", re.ASCII)
# x-api-key & co: base masks the whole ``\S+`` run, so only structural text may follow:
# ``);,`` and, AFTER an actual (escaped) quote, further quotes and JSON/structure closers
# ``)]},;`` (``{"content": "x-api-key: $TOKEN"}``). ``$TOKEN}`` / ``$TOKEN]x`` stay masked.
_SECRET_AFTER_RE = re.compile(r"[);,]*(?:\\*[\"'`](?:\\*[\"'`]|[)\]},;])*)?", re.ASCII)
# review6366 HF-11: compact JSON (no whitespace after ``,``) leaves the NEXT member/element
# inside the same ``\S+`` run as an x-api-key placeholder that ends a JSON string
# (``{"content":"x-api-key: $TOKEN","other":"PUBLIC"}``, ``["x-api-key: $TOKEN","PUBLIC"]``).
# That following source is not the credential slot. The header string ends at an actual
# closing ``"`` (backslash-escaped once per encoding level); after it and any ``]``/``}``
# closers, a ``,`` followed by the start of a JSON value -- a string opened at the SAME escape
# level as the last closing quote, ``{``, ``[``, a number, ``true``/``false``/``null`` -- is
# a member/element boundary: exactly where spaced JSON has whitespace and base's slot ended.
# The slot is kept through that ``,`` and the header scan resumes at the next member, so a
# following header value is classified (and masked) on its own. A quote followed by anything
# else (``$TOKEN"AbCd``, ``$TOKEN",AbCd``) is not a boundary and keeps the base mask.
# review6471 HF-12: equal escape counts do not make a quote structural. Inside ONE string,
# data quotes are escaped one level deeper than the string's own delimiters
# (``{"content":"x-api-key: $TOKEN\",\"AbCd"}`` is a single ``content`` string), so each
# closing quote must close a string that is actually open at the header, at exactly that
# string's escape level: ``_JsonStringScan`` tracks the strings open on the line (escape
# level of a quote = trailing 1-bits of its backslash run: ``"`` 0, ``\"`` 1, ``\\\"`` 2).
# review6556 HF-12/HF-13, review6640 HF-12/HF-14: escaping alone does not make a quote a
# string delimiter, at line level or one level deeper. A quote OPENS a string only where a
# JSON value can begin: at the start of the line (after a ``N|`` line-number gutter) or of
# the enclosing string's content, or after ``[`` ``{`` ``,`` ``:`` (spaces / escaped
# ``\n\r\t`` between). Elsewhere it is data (``5" tall``, ``echo "a"``), and so is every quote
# deeper than the innermost open string. And the ``,`` must separate members of a JSON
# array/object that is open there and is well-formed at that escape level: the innermost
# ``[``/``{`` open at the comma is decoded and checked with the standard JSON decoder. Only
# that container must be JSON, not the text around it, so a JSON literal in source code, a
# shell command or a numbered line keeps its members, while quote/comma data inside one string
# (``note = '5" tall x-api-key: $TOKEN","AbCd'``, ``":{\"x-api-key: %s\",null,AbCd"``) does not.
# review6751 HF-15/16/17: the container must OWN the comma -- the standard decoder, walking
# that container's members, must stop at this very ``,`` as a member separator -- so an
# unrelated parseable ``[]`` elsewhere never vouches for it. Raw offsets map into the decoded
# context exactly as the decoder decodes (a ``\uD83D\uDE00`` surrogate-pair escape is ONE
# character), and the decoded length must agree with the map. A closing quote closes the
# string open at its own escape level; strings the scan opened deeper (a data quote after
# ``:`` in ``":\"x-api-key: $TOKEN"``) were data inside it, exactly as the scan itself closes
# them. Numbers are validated lexically, never converted (no int-digit limit).
# review6810 HF-18: a newline ends every JSON STRING (none may contain one) but not a
# container: JSON allows whitespace, newlines included, between members, so the line-level
# ``[``/``{`` stay open across lines and a multi-line array/object is checked as a whole (with
# ``N|`` line-number gutters removed, as read_file/search_files render them). The comma must
# still be a member separator of that well-formed container.
# review6900 LG01 (HF-18 extent): a numbered read_file result is itself serialized (the tool's
# JSON response, a history record), so its gutters sit inside a JSON string after ESCAPED
# newlines (``"1|[\n2|\"x-api-key: %s\",0\n3|]"``). A gutter that starts a decoded line is
# skipped like whitespace when deciding where a value may begin, and is removed from a decoded
# string context before its container is checked -- the same rule as for raw lines.
# review7078 HF7078-01: pretty-printed JSON inside such a string puts ESCAPED whitespace (and,
# when it is a numbered reader result, a gutter after each escaped newline) between a closing
# quote, its closers, the member ``,`` and the next member (``\",\n3|  \"next\"``). That is the
# whitespace JSON allows between tokens, so the boundary chain and the next-member lookahead
# skip it; the decoder's member walk of the container still has to stop at that ``,``.
_JSON_ESCAPED_WS = r"(?:\\{1,127}[nrt](?:[ \t]{0,16}\d{1,12}\|)?|[ \t])"
_JSON_NEXT_MEMBER_RE = re.compile(
    rf"(?:(?P<q>\\{{0,127}})\"(?:{_JSON_ESCAPED_WS}|[\]}}])*){{1,16}},"
    rf"(?={_JSON_ESCAPED_WS}*(?:(?P=q)\"|[\[{{0-9-]|true\b|false\b|null\b))", re.ASCII)
_JSON_CLOSER_RE = re.compile(rf"(\\{{0,127}})\"((?:{_JSON_ESCAPED_WS}|[\]}}])*)", re.ASCII)
_JSON_QUOTE_OR_EOL_RE = re.compile(r"(?<!\\)(\\*)\"|\n")
_JSON_SCAN_RE = re.compile(r"(?<!\\)(\\*)\"|[\n\[\]{}]")
# One escape = one decoded character; a high+low surrogate-pair escape decodes to ONE
# character (as in the stdlib decoder), so it is matched as one escape.
_JSON_ESCAPE_RE = re.compile(
    r"\\(?:u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)", re.DOTALL)
# Lexical validation only: numbers stay strings (no int-digit limit, no float conversion).
_JSON_DECODER = json.JSONDecoder(parse_int=str, parse_float=str)
_JSON_WS_RE = re.compile(r"[ \t\n\r]*")
# No value-free placeholder (or its trailing structure) is longer than this; longer runs
# skip classification and mask as before (also bounds the backslash-run regexes above).
_PLACEHOLDER_MAX_SLOT = 128
_FOLLOWING_RUN_RE = re.compile(rf"\S{{0,{_PLACEHOLDER_MAX_SLOT + 1}}}")
# Shell/printf names are references by syntax: the value is never in the text.
_PLACEHOLDER_REFERENCE_GROUPS = frozenset({"fmt", "braced", "bare"})
# Wrapper contents (``<…>``, ``{…}``, ``{{…}}``) are literal text, so language acceptance
# alone does not make them value-free (review6184 HF-04, review6279 Q1). A wrapper name is
# exempt only if it splits (on ``_``/``-`` and camelCase boundaries) ENTIRELY into words of
# this closed list, 1-5 words, at least one a credential role. Casing never decides
# (``accessToken``, ``serviceToken``, ``API_KEY`` qualify; ``<SyntheticSecret>``,
# ``<TOKEN_AbCdEfGhIjKlMnOp>`` do not). The list is a fidelity heuristic, not proof that no
# credential is present; any other word (``<GITHUB_TOKEN>``, ``<YOUR_QWERTY_HERE>``) keeps
# the base masking (documented limitation).
_PLACEHOLDER_ROLE_WORDS = frozenset({
    "token", "tokens", "key", "apikey", "secret", "password", "passwd", "pass", "pw",
    "credential", "credentials", "jwt", "pat", "bearer"})
_PLACEHOLDER_QUALIFIER_WORDS = frozenset({
    "your", "my", "the", "a", "an", "api", "access", "auth", "client", "refresh", "session",
    "id", "personal", "service", "app", "bot", "user", "admin", "owner", "here", "value",
    "new", "this"})
_NAME_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", re.ASCII)


def _is_reference_name(name: str, max_len: int, max_alpha: int) -> bool:
    """``OWNER_TOKEN`` / ``MCP_N8N_API_KEY`` / ``SERVICE_2026_TOKEN`` read as names;
    ``Ab12Cd34`` does not.

    One policy for ``$NAME``, ``${NAME}`` and ``%(name)s``. Every ``_``-separated segment
    is a word (alphabetic, <= ``max_alpha``), a short single-case token with at most two
    digits (``N8N``, ``V2``, ``S3``), or an all-digit segment of at most four digits
    (``2026``; at most two such segments)."""
    if len(name) > max_len:
        return False
    digit_segments = 0
    for seg in filter(None, name.split("_")):
        if seg.isalpha() and len(seg) <= max_alpha:
            continue
        if seg.isdigit() and len(seg) <= 4:
            digit_segments += 1
            if digit_segments > 2:
                return False
            continue
        single_case = seg in (seg.upper(), seg.lower())
        if not (single_case and len(seg) <= 12 and sum(c.isdigit() for c in seg) <= 2):
            return False
    return True


def _is_wrapper_placeholder_name(name: str) -> bool:
    if len(name) > 40:
        return False
    words = []
    for seg in filter(None, re.split(r"[_-]+", name)):
        parts = _NAME_WORD_RE.findall(seg)
        if "".join(parts) != seg:
            return False
        words.extend(part.lower() for part in parts)
    return (1 <= len(words) <= 5
            and all(w in _PLACEHOLDER_ROLE_WORDS or w in _PLACEHOLDER_QUALIFIER_WORDS for w in words)
            and any(w in _PLACEHOLDER_ROLE_WORDS for w in words))


def _placeholder_names_ok(p: re.Match) -> bool:
    return all(_is_reference_name(v, 64, 32) if k in _PLACEHOLDER_REFERENCE_GROUPS
               else _is_wrapper_placeholder_name(v)
               for k, v in p.groupdict().items() if v)


def _is_credential_placeholder(value: str) -> bool:
    """True when ``value`` on its own is exactly one complete value-free placeholder."""
    p = _CREDENTIAL_PLACEHOLDER_RE.fullmatch(value or "")
    return bool(p) and _placeholder_names_ok(p)


def _slot_placeholder(m: re.Match, group: int) -> re.Match | None:
    """The ONE complete value-free placeholder (see above) that begins header slot ``group``
    of ``m``, matched in the original string (it may extend past the slot across inner
    whitespace), or None."""
    p = _CREDENTIAL_PLACEHOLDER_RE.match(m.string, m.start(group))
    return p if p and _placeholder_names_ok(p) else None


def auth_header_sub(m: re.Match, mask) -> str:
    """``agent.redact._AUTH_HEADER_RE`` replacement: the match unchanged when its credential slot
    (group 3) begins ONE complete value-free placeholder, else ``mask`` of the slot as before.
    The Authorization slot stops at a quote, so only the boundary right after the placeholder
    decides; whatever follows a real quote was never part of base's mask."""
    if (m.end(3) - m.start(3) <= _PLACEHOLDER_MAX_SLOT
            and (p := _slot_placeholder(m, 3)) is not None
            and _AUTH_AFTER_RE.match(m.string, p.end()) is not None):
        return m.group(0)
    return m.group(1) + (m.group(2) or "") + mask(m.group(3))


def _quote_level(run: int) -> int:
    """Escape level of a ``"`` preceded by ``run`` backslashes (pairs are literal backslashes)."""
    level = 0
    while run & 1:
        level += 1
        run >>= 1
    return level


def _line_gutter_start(text: str, j: int, lo: int) -> int | None:
    """Start of the line-number gutter (``_LINE_GUTTER_RE``) whose ``|`` is ``text[j]``, when
    it begins a line: at ``lo`` or right after an escaped newline (``\n``); else None."""
    k = j
    while k > lo and j - k < 12 and text[k - 1].isdecimal():
        k -= 1
    if k == j:
        return None
    g = k
    while g > lo and k - g < 16 and text[g - 1] in " \t":
        g -= 1
    if g == lo or (g - 2 >= lo and text[g - 1] == "n" and text[g - 2] == "\\"):
        return g
    return None


def _json_value_position(text: str, i: int, lo: int) -> bool:
    """True when a value may begin at ``i`` (an opening quote's escape start): only spaces,
    tabs, escaped ``\n``/``\r``/``\t`` and line-number gutters that start a line separate it
    from ``lo`` (start of the line or of the enclosing string's content) or from a preceding
    ``[`` ``{`` ``,`` ``:``."""
    while i > lo:
        c = text[i - 1]
        if c in " \t":
            i -= 1
        elif c == "|" and (g := _line_gutter_start(text, i - 1, lo)) is not None:
            i = g
        elif c in "nrt" and i - 2 >= lo and text[i - 2] == "\\":
            i -= 2
            while i > lo and text[i - 1] == "\\":
                i -= 1
        else:
            return c in "[{,:"
    return True


def _json_whole_content_string(text: str, start: int, level: int) -> bool:
    """For a level-``level`` quote that begins its enclosing string's content (``start`` =
    its end): True when that nested string closes at its own level and the enclosing string
    closes right after it, i.e. the content is one JSON string (``"\"[\\\"…\\\"]\""``),
    not a data quote (``"\"]x-api-key: …"``)."""
    for q in _JSON_QUOTE_OR_EOL_RE.finditer(text, start):
        if q.group(1) is None or _quote_level(len(q.group(1))) < level:
            return False
        if _quote_level(len(q.group(1))) == level:
            if len(q.group(1)) != (1 << level) - 1:
                return False
            e = _JSON_QUOTE_OR_EOL_RE.match(text, q.end())
            return e is not None and e.group(1) is not None and len(e.group(1)) == (1 << (level - 1)) - 1
    return False


# ``read_file``/``search_files`` render a line-number gutter (``     12|``) before each line.
_LINE_GUTTER_RE = re.compile(r"[ \t]{0,16}\d{1,12}\|")
_LINE_GUTTERS_RE = re.compile(r"^" + _LINE_GUTTER_RE.pattern, re.MULTILINE)


def _strip_line_gutters(doc: str) -> tuple[str, list]:
    """``doc`` without the gutters that start its lines, and the offset map (``_map_decoded``
    layer form) from ``doc`` into the result (empty when there is none)."""
    parts, ends, removed, last, n = [], [], [], 0, 0
    for g in _LINE_GUTTERS_RE.finditer(doc):
        parts.append(doc[last:g.start()])
        n += g.end() - g.start()
        ends.append(g.end())
        removed.append(n)
        last = g.end()
    if not ends:
        return doc, []
    parts.append(doc[last:])
    return "".join(parts), [(ends, removed)]


def _decode_json_layers(raw: str, layers: int) -> tuple[str, list] | None:
    """``raw`` (string content escaped ``layers`` times) decoded, with the per-layer
    ``(escape ends, removed counts)`` that map raw offsets into it; None if invalid or if the
    decoded length disagrees with the map."""
    maps = []
    for _ in range(layers):
        ends, removed, n = [], [], 0
        for e in _JSON_ESCAPE_RE.finditer(raw):
            n += len(e.group()) - 1
            ends.append(e.end())
            removed.append(n)
        try:
            decoded = json.loads('"' + raw + '"', strict=False)
        except (ValueError, RecursionError):
            return None
        if len(decoded) != len(raw) - n:
            return None
        raw = decoded
        maps.append((ends, removed))
    return raw, maps


def _map_decoded(i: int, maps: list) -> int:
    """Raw offset ``i`` (not inside an escape) as an offset into the decoded text."""
    for ends, removed in maps:
        k = bisect.bisect_right(ends, i)
        i -= removed[k - 1] if k else 0
    return i


def _json_member_commas(doc: str, i: int) -> frozenset[int] | None:
    """Offsets of the ``,`` that separate the members of the JSON array/object starting at
    ``doc[i]``, each member read by the standard decoder; None unless it is well-formed."""
    close = "]" if doc[i] == "[" else "}"
    commas = []
    j = _JSON_WS_RE.match(doc, i + 1).end()
    if doc.startswith(close, j):
        return frozenset()
    try:
        while True:
            if close == "}":
                if not doc.startswith('"', j):
                    return None
                j = _JSON_WS_RE.match(doc, _JSON_DECODER.raw_decode(doc, j)[1]).end()
                if not doc.startswith(":", j):
                    return None
                j = _JSON_WS_RE.match(doc, j + 1).end()
            j = _JSON_WS_RE.match(doc, _JSON_DECODER.raw_decode(doc, j)[1]).end()
            if doc.startswith(",", j):
                commas.append(j)
                j = _JSON_WS_RE.match(doc, j + 1).end()
            elif doc.startswith(close, j):
                return frozenset(commas)
            else:
                return None
    except (ValueError, RecursionError):
        return None


class _JsonStringScan:
    """JSON strings and ``[``/``{`` containers open at successive offsets of ``text`` (call
    ``depth`` with non-decreasing ``end``). A newline closes every string (JSON strings never
    contain one) but no container: line-level ``[``/``{`` stay open across lines. A quote at
    or below an open string's level closes down to it. A quote exactly one level deeper than
    the innermost open string (level 0 at line level) opens a nested
    string only when it is escaped exactly once per level and sits where a JSON value can
    begin (``_json_value_position``; at the very start of a string's content only when the
    content is that one string); otherwise it, and any deeper quote, is data. Brackets belong
    to the innermost open string's content (or the line level)."""

    __slots__ = ("decoded", "line", "line_brackets", "members", "open", "pos", "text")

    def __init__(self, text: str):
        self.text, self.pos = text, 0
        self.line = self._content_start(0)
        self.line_brackets: list[int] = []
        self.open: list[tuple[int, list[int]]] = []  # (content start, its open brackets)
        self.members: dict[int, frozenset[int] | None] = {}
        self.decoded: dict[int, tuple[str, list] | None] = {}

    def _content_start(self, i: int) -> int:
        g = _LINE_GUTTER_RE.match(self.text, i)
        return g.end() if g else i

    def depth(self, end: int) -> int:
        stack, text = self.open, self.text
        for q in _JSON_SCAN_RE.finditer(text, self.pos, end):
            if q.group(1) is None:
                c = q.group()
                if c == "\n":
                    stack.clear()
                    self.line = self._content_start(q.end())
                else:
                    brackets = stack[-1][1] if stack else self.line_brackets
                    if c in "[{":
                        brackets.append(q.start())
                    elif brackets:
                        brackets.pop()
                continue
            run = len(q.group(1))
            level = _quote_level(run)
            if level < len(stack):
                del stack[level:]
            elif level == len(stack) and run == (1 << level) - 1:
                lo = stack[-1][0] if stack else self.line
                if (_json_value_position(text, q.start(), lo)
                        and (not stack or q.start() > lo
                             or _json_whole_content_string(text, q.end(), level))):
                    stack.append((q.end(), []))
        self.pos = max(self.pos, end)
        return len(stack)

    def brackets(self, context: int) -> list[int]:
        """Open ``[``/``{`` at line level (context 0) or of string ``context - 1``'s content."""
        return self.open[context - 1][1] if context else self.line_brackets

    def _context(self, context: int) -> tuple[str, int, list] | None:
        """The text (context 0), or string ``context - 1``'s content decoded, without the
        line-number gutters that start its lines, with the raw offset of its start and the
        offset map (``_decode_json_layers`` layers, then ``_strip_line_gutters``)."""
        if context == 0:
            if -1 not in self.decoded:
                self.decoded[-1] = _strip_line_gutters(self.text)
            doc, maps = self.decoded[-1]
            return doc, 0, maps
        cs = self.open[context - 1][0]
        if cs not in self.decoded:
            self.decoded[cs] = None
            for q in _JSON_QUOTE_OR_EOL_RE.finditer(self.text, cs):
                if q.group(1) is None:
                    break
                if _quote_level(len(q.group(1))) < context:
                    layers = _decode_json_layers(
                        self.text[cs:q.end() - (1 << (context - 1))], context)
                    if layers is not None:
                        doc, gutters = _strip_line_gutters(layers[0])
                        self.decoded[cs] = (doc, layers[1] + gutters)
                    break
        if self.decoded[cs] is None:
            return None
        doc, maps = self.decoded[cs]
        return doc, cs, maps

    def member_comma(self, b: int, comma: int, context: int) -> bool:
        """True when the ``[``/``{`` at raw offset ``b`` in ``context`` starts a well-formed
        JSON value, once that context's escaping is decoded, and the ``,`` at raw offset
        ``comma`` is one of ITS member separators."""
        ctx = self._context(context)
        if ctx is None:
            return False
        doc, cs, maps = ctx
        i, j = _map_decoded(b - cs, maps), _map_decoded(comma - cs, maps)
        if not (i < j < len(doc) and doc[i] == self.text[b] and doc[j] == ","):
            return False
        if b not in self.members:
            self.members[b] = _json_member_commas(doc, i)
        commas = self.members[b]
        return commas is not None and j in commas


def _json_member_boundary_end(text: str, pos: int, scan: _JsonStringScan) -> int | None:
    """End of ``_JSON_NEXT_MEMBER_RE`` at ``pos`` when its closing quotes really close strings
    open there, each at exactly its own escape level and each shallower than the one before
    (review6471 HF-12; a closing quote closes any deeper string the scan opened, review6751
    HF-16), and its ``,`` is a member separator of the innermost ``[``/``{`` open in the
    remaining context, a well-formed JSON array/object at that level (review6640 HF-12/HF-14,
    review6751 HF-15/HF-17), else None."""
    c = _JSON_NEXT_MEMBER_RE.match(text, pos)
    if c is None:
        return None
    depth, i, closers = len(scan.open), pos, 0
    while i < c.end() - 1:
        q = _JSON_CLOSER_RE.match(text, i)
        run = len(q.group(1))
        level = _quote_level(run)
        if level >= depth or run != (1 << level) - 1:
            return None
        depth = level
        closers = q.group(2).count("]") + q.group(2).count("}")
        i = q.end()
    brackets = scan.brackets(depth)
    if (len(brackets) <= closers
            or not scan.member_comma(brackets[-1 - closers], c.end() - 1, depth)):
        return None
    return c.end()


# The base header slot: the whole non-whitespace run after ``<header>:`` (r15's ``\S+``).
_SLOT_RUN_RE = re.compile(r"\S*")


class PlaceholderSlots:
    """Placeholder decisions for secret-header values in one text (``agent.redact_headers``).

    ``kept_tail(start)`` is the end of the kept region when the value starting at ``start`` is
    ONE complete value-free placeholder that ends at a recognised boundary: the end of its
    non-whitespace run (``_SECRET_AFTER_RE``: structure only) or a compact JSON member boundary
    whose ``,`` its well-formed container owns (``_json_member_boundary_end``). Otherwise None,
    and the caller masks. Calls must use non-decreasing ``start`` for the JSON scan; an earlier
    ``start`` restarts it."""

    __slots__ = ("_scan", "text")

    def __init__(self, text: str):
        self.text, self._scan = text, None

    def placeholder(self, start: int) -> re.Match | None:
        p = _CREDENTIAL_PLACEHOLDER_RE.match(self.text, start)
        return p if p and _placeholder_names_ok(p) else None

    def kept_tail(self, start: int) -> int | None:
        text = self.text
        p = self.placeholder(start)
        if p is None:
            return None
        run_end = _SLOT_RUN_RE.match(text, start).end()
        following = _FOLLOWING_RUN_RE.match(text, p.end()).group()
        if (run_end - start <= _PLACEHOLDER_MAX_SLOT and len(following) <= _PLACEHOLDER_MAX_SLOT
                and _SECRET_AFTER_RE.fullmatch(following)):
            return p.end() + len(following)
        if self._scan is None or start < self._scan.pos:
            self._scan = _JsonStringScan(text)
        self._scan.depth(start)
        c_end = _json_member_boundary_end(text, p.end(), self._scan)
        if (c_end is None or c_end - p.end() > _PLACEHOLDER_MAX_SLOT
                or min(c_end, run_end) - start > _PLACEHOLDER_MAX_SLOT):
            return None
        return c_end

    def keeps(self, start: int, end: int) -> bool:
        """True when the slot ``[start, end)`` a masking pass found is wholly covered by a kept
        placeholder region, with at most whitespace after it inside the slot."""
        tail = self.kept_tail(start)
        return tail is not None and not self.text[tail:end].strip()


def slots_for(cache: list, text: str) -> PlaceholderSlots:
    """The ``PlaceholderSlots`` for ``text``, reused while a substitution pass works on it."""
    if not cache or cache[0].text is not text:
        cache[:] = [PlaceholderSlots(text)]
    return cache[0]
