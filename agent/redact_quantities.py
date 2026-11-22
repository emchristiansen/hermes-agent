"""Tell signed quantities and joined identifiers from E.164 phone numbers for the redactor's phone pass.

``+1109262336`` (bytes freed) and the ``+5323013`` of ``new09f71717+5323013`` (two identifiers
joined by ``+``) have the shape of an E.164 number. ``non_phone_offsets`` names the occurrences that
carry their own evidence of being something else, so ``agent.redact.redact_sensitive_text`` leaves
them intact and masks every other phone-shaped match.
"""

import bisect
import collections
import re

# E.164 phone numbers, 7-15 digits; the lookahead rejects hex strings / identifiers.
SIGNAL_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

_DATA_SIZE_UNIT = r"(?:[Bb]ytes?|[kKMGTPE]i?B)(?![A-Za-z0-9])"
_DATA_SIZE_UNIT_RE = re.compile(rf"(?<![A-Za-z0-9]){_DATA_SIZE_UNIT}")
# The unit of this number, or of the ``/``-separated list of signed integers it starts:
# ``+2941231104 bytes``, ``+820772864 / +1340882944 bytes``. Not a ``bytes=4096`` field that merely
# follows the number. _unit_follows walks the list one step at a time.
_LIST_STEP_RE = re.compile(r"[ \t]*/[ \t]*[+\-−]\d+")
_UNIT_AFTER_RE = re.compile(rf"[ \t]+{_DATA_SIZE_UNIT}(?![ \t]*=)")
# Storage-accounting labels a sign may be glued to (``pool+2941231104``), matched as the WHOLE
# glued letter run: ``alicepool`` is no label. The one supported producer preamble is the Storage
# summary's ``Originalsharedwindow`` before its first field (``OriginalsharedwindowHOME+...``).
_OPERATIONAL_LABEL = (r"(?:originalsharedwindow)?"
                      r"(?:home|pool|dataset|free|freeing|used|usable|available|avail|size|total|allocated)")
_OPERATIONAL_LABEL_RE = re.compile(rf"(?i){_OPERATIONAL_LABEL}")
# One field of a compact ``/``-joined run: label, optional sign, integer, optional attached unit.
_OPERATIONAL_FIELD_RE = re.compile(
    rf"(?i)[(\[]?{_OPERATIONAL_LABEL}[+-]?\d+(?:bytes?|[kmgtpe]i?b|b)?[)\].,;:]*")
# Contact data, including serialized compound names (``phoneNumber``, ``Cellphone``, ``mobile_no``):
# distinctive stems match inside a word, short ones only as words.
_CONTACT_NAME = (r"(?:[a-z]*(?:phone|mobile|msisdn|whatsapp|contact)[a-z]*"
                 r"|(?<![a-z])(?:tel|telephone|cell|fax|sms|signal|call)(?:s|ed|er|ing)?(?![a-z]))")
_CONTACT_NAME_RE = re.compile(rf"(?i){_CONTACT_NAME}")
# Where some _CONTACT_NAME_RE match occurs: its stem alone, without the optional surrounding letters
# whose ``[a-z]*`` makes a search over a long letter run quadratic. One pass finds them all (_Segments).
_CONTACT_STEM_RE = re.compile(
    r"(?i)phone|mobile|msisdn|whatsapp|contact"
    r"|(?<![a-z])(?:tel|telephone|cell|fax|sms|signal|call)(?:s|ed|er|ing)?(?![a-z])")
# Syntax between a contact label and its value: whitespace, Markdown emphasis, quotes and their
# JSON escapes, parentheses, array/object openers, XML tags, YAML list dashes and read_file's ``N|`` gutter.
_LABEL_GAP = r"(?:[\s*_\"'`\\(\[{<>-]|(?<=\n)\d+\|)*"
# A contact label directly before the number, through Markdown emphasis and serialized keys:
# ``Phone: ``, ``tel:``, ``Call``, ``**Phone:** ``, ``"phone": "``, ``phone_number = '``,
# ``"phone": ["``, ``{\\"phone\\": \\"``, ``"phone": {"work": "``, ``phone:\n  - ``.
_CONTACT_LABEL_RE = re.compile(
    rf"(?i){_CONTACT_NAME}(?:[ \t_-]*(?:number|num|no\.?)s?)?{_LABEL_GAP}(?:[:=#]{_LABEL_GAP})?"
    rf"(?:\{{[ \t]*(?:\\?[\"'])?[a-z0-9_ -]{{1,40}}(?:\\?[\"'])?[ \t]*:{_LABEL_GAP})*\Z")
# One earlier value of the list a contact label introduces (``Phone: +P / ``, ``"phone": ["+P", ``,
# ``phone:\n  - +P bytes\n  - ``): a number with optional sign, unit, emphasis, quotes or
# brackets, then a list separator (including a dash, pipe or line break).
_LIST_PREDECESSOR_RE = re.compile(
    rf"[*_\"'`\\(\[]*[+\-−]?\d[\d().\-]*(?:[ \t]+{_DATA_SIZE_UNIT})?[*_\"'`\\)\]]*[ \t]*"
    r"(?:[/,;&|—–-]|(?<![A-Za-z])(?:or|and)(?![A-Za-z])|\n(?:\d+\|)?(?:[ \t]*[-*](?=[ \t]))?)"
    r"[ \t*_\"'`\\(\[]*\Z")
_QUANTITY_CONTEXT = 200  # chars searched around a match for its token
_LABEL_CONTEXT = 60  # chars searched before a match for a contact label
# A blank line, behind read_file's optional ``N|`` gutter: it separates blocks of text (_Segments).
_BLANK_LINE_RE = re.compile(r"(?m)^(?:[ \t]*\d+\|)?[ \t\r]*$\n?")
# An XML name: it starts with a letter (any script), ``_`` or ``:`` (XML NameStartChar), never a
# digit, so ``<_phone>`` is an element while ``a <1`` is not.
_XML_NAME = r"(?:[^\W\d]|:)[^\s<>/]*"
# A quoted string to its closing quote or the end of its line, or a closer: a bracket (group 1)
# or an XML end tag (group 2, its name) (_closers). Triple quotes come first so ``"""`` is not
# read as an empty string.
_QUOTED_OR_CLOSER_RE = re.compile(
    r"\"\"\"|'''|\"(?:[^\"\\\n]|\\.)*\"?|'(?:[^'\\\n]|\\.)*'?"
    rf"|([\]}})])|</({_XML_NAME})[^<>\n]*>")
# An opener: a bracket (group 1) or the start of an XML start tag (group 2, its name) unless it
# visibly closes itself (``<br/>``). A tag whose end is not seen still counts.
_OPENER_RE = re.compile(rf"([\[{{(])|<({_XML_NAME})(?![^<>\n]*/>)")
# Delimiters of text that may run across blank lines, each opener with its closer: TOML/Python
# triple-quoted strings, XML comments and CDATA sections, and Markdown code fences at a line
# start (behind the gutter). Only a fence holds code whose brackets count (_Segments).
_DATA_BRACKETS = frozenset("[{")
_TEXT_SECTIONS = ("<!--", "<![CDATA[")  # XML comments and CDATA: text even inside a string
_LONG_CLOSER = {'"""': '"""', "'''": "'''", "<!--": "-->", "<![CDATA[": "]]>", "```": "```", "~~~": "~~~"}
_LONG_STRING_RE = re.compile(
    r"\"\"\"|'''|<!--|-->|<!\[CDATA\[|\]\]>|(?m:^(?:[ \t]*\d+\|)?[ \t]*(?:```|~~~))")
# A block's first line indented, or a YAML/Markdown list item (behind the gutter): it may continue
# an enclosing structure (``phone:\n- a\n\n- x1+P``).
_CONTINUATION_LINE_RE = re.compile(r"(?:[ \t]*\d+\|)?(?:[ \t]|-(?:[ \t]|$))", re.MULTILINE)

# GFM pipe tables, optionally behind read_file's ``N|`` gutter (see _table_cells).
_MD_GUTTER_RE = re.compile(r"[ \t]*\d+\|")
# Lines that start another block and so end a table: heading, block quote, code fence, list item.
_MD_BLOCK_START_RE = re.compile(
    r" {0,3}(?:#{1,6}(?:[ \t]|$)|>|```|~~~|[-*+][ \t]|\d{1,9}[.)][ \t])")
_MD_PIPE_RE = re.compile(r"(?<!\\)\|")
_MD_DELIMITER_CELL_RE = re.compile(r"[ \t]*:?-+:?[ \t]*")
_MD_CHANGE_RE = re.compile(r"(?i)(?<![a-z])(?:change|delta|diff|difference)s?(?![a-z])|Δ")
_MD_SIGNED_CELL_RE = re.compile(r"[ \t]*(\*\*|__)?\+\d+(?(1)\1)[ \t]*")


def _md_body(line: str, gutter: bool):
    """Where the row content begins (after read_file's gutter), or None without the gutter."""
    if not gutter:
        return 0
    m = _MD_GUTTER_RE.match(line)
    return m.end() if m else None


def _md_cells(line: str, base: int, gutter: bool):
    """``(offset, text)`` for each cell of a pipe-table row, or None if ``line`` is not one."""
    line = line.rstrip("\r\n")
    start = _md_body(line, gutter)
    if start is None:
        return None
    if "|" not in line[start:]:
        return None
    cells, pos = [], start
    for m in _MD_PIPE_RE.finditer(line, start):
        cells.append((base + pos, line[pos:m.start()]))
        pos = m.end()
    cells.append((base + pos, line[pos:]))
    if not cells[0][1].strip():  # leading pipe
        cells.pop(0)
    if cells and not cells[-1][1].strip():  # trailing pipe
        cells.pop()
    return cells


def _md_delimiter(lines, starts, i: int, gutter: bool):
    """The delimiter cells of line ``i`` if it is a delimiter row under a matching header."""
    if i < 1 or i >= len(lines):
        return None
    delimiter = _md_cells(lines[i], starts[i], gutter)
    if not delimiter or not all(_MD_DELIMITER_CELL_RE.fullmatch(c) for _, c in delimiter):
        return None
    header = _md_cells(lines[i - 1], starts[i - 1], gutter)
    if not header or len(header) != len(delimiter):
        return None
    return header, delimiter


def _table_cells(text: str):
    """Classify pipe-table cells: (``+`` offsets of quantity cells, spans masked unconditionally,
    the span of every body cell).

    A quantity column is right-aligned (``---:``) and its own header declares a quantity: a
    data-size unit (``Size (bytes)``), or a change (``Shared-host change``) in a table whose header
    declares a unit (``Measure (bytes)``). A header naming contact data (``Phone``, ``phoneNumber
    change``, ``Cellphone (bytes)``) makes every cell of its column masked, whatever the column's
    alignment or contents. A row whose cell count differs from the delimiter row cannot be mapped to
    columns, so the whole row is masked. As in GFM, rows may add or drop the outer pipes and a line
    without any pipe is a one-cell row; the table ends at a blank line, a line that starts another
    block (``## Contacts | directory``) or a row that heads a new table, so no schema carries over.
    """
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    signed, masked, cells = set(), [], []
    i = 1
    while i < len(lines):
        for gutter in (False, True):
            table = _md_delimiter(lines, starts, i, gutter)
            if table:
                break
        else:
            i += 1
            continue
        header, delimiter = table
        i += 1
        table_unit = any(_DATA_SIZE_UNIT_RE.search(h) for _, h in header)
        contact = [bool(_CONTACT_NAME_RE.search(h)) for _, h in header]
        quantity = [
            d.strip().endswith(":") and not d.strip().startswith(":") and not is_contact
            and bool(_DATA_SIZE_UNIT_RE.search(h) or (table_unit and _MD_CHANGE_RE.search(h)))
            for (_, h), (_, d), is_contact in zip(header, delimiter, contact)]
        while i < len(lines) and not _md_delimiter(lines, starts, i + 1, gutter):
            body = _md_body(lines[i], gutter)
            content = lines[i][body:].rstrip("\r\n") if body is not None else ""
            if not content.strip() or _MD_BLOCK_START_RE.match(content):
                break
            row = _md_cells(lines[i], starts[i], gutter) or [(starts[i] + body, content)]
            cells.extend((offset, offset + len(cell)) for offset, cell in row)
            if len(row) != len(delimiter):
                masked.append((starts[i], starts[i + 1]))
            else:
                for (offset, cell), is_quantity, is_contact in zip(row, quantity, contact):
                    if is_contact:
                        masked.append((offset, offset + len(cell)))
                    elif is_quantity and _MD_SIGNED_CELL_RE.fullmatch(cell):
                        signed.add(offset + cell.index("+"))
            i += 1
    return frozenset(signed), masked, cells


def _glued_label(text: str, start: int, window: int) -> str:
    """The letters directly before ``start`` (``pool`` in ``pool+2941231104``), or ``""``."""
    return re.search(r"[A-Za-z]*\Z", text[window:start]).group()


def _in_operational_run(text: str, m: re.Match, window: int) -> bool:
    """Whether ``m``'s ``/``-separated field and an adjacent one are complete operational fields.

    Only recognised fields count: in ``HOME+1109262336B/pool+2941231104/alice+P`` the ``pool``
    value has an operational neighbour, while ``alice+P`` is no operational field at all. The run
    is bounded by whitespace and pipes, so read_file's ``N|`` gutter and table borders are not
    part of a field.
    """
    left = re.search(r"[^\s|]*\Z", text[window:m.start()]).group()
    right = re.match(r"[^\s|]*", text[m.end():m.end() + _QUANTITY_CONTEXT]).group()
    fields = (left + m.group(1) + right).split("/")
    i = left.count("/")

    def operational(j: int) -> bool:
        return 0 <= j < len(fields) and bool(_OPERATIONAL_FIELD_RE.fullmatch(fields[j]))
    return operational(i) and (operational(i - 1) or operational(i + 1))


# The kind of each bracket: a closer only ever closes an opener of its own kind. An XML element's
# kind is its tag name, so ``</other>`` never closes ``<phone>``.
_KIND = {"[": "[", "]": "[", "{": "{", "}": "{", "(": "(", ")": "("}


def _kind(m: re.Match) -> str:
    return _KIND[m.group(1)] if m.group(1) else "<" + m.group(2)


def _openers(block: str) -> collections.Counter:
    """The openers of ``block`` by kind, counted everywhere, quoted or not."""
    return collections.Counter(_kind(m) for m in _OPENER_RE.finditer(block))


def _closers(block: str) -> collections.Counter:
    """The closers of ``block`` that are certainly outside a quoted string, by kind.

    A quoted string runs to its closing quote, honouring backslash escapes, or else to the end of
    its line (a JSON string cannot span lines); a closer inside ``"..."`` or ``'...'``, or after an
    unmatched quote on its line, is not counted. An apostrophe in prose therefore only hides
    closers, which keeps the open count high.
    """
    return collections.Counter(_kind(m) for m in _QUOTED_OR_CLOSER_RE.finditer(block)
                               if m.group(1) or m.group(2))


class _Segments:
    """The segments of one text: blocks of non-blank lines, joined where structure continues.

    A blank line ends a segment only when the text before it has closed every bracket and XML
    element it may have opened and the next line is neither indented nor a list item. Otherwise
    the next block may still belong to the same structured value: a pretty-printed JSON array
    (``"phone": [\n\n  "x1+P"``), an indented YAML block scalar (``phone: |\n\n  x1+P``), a YAML
    list (``phone:\n- a\n\n- x1+P``) or a ``<phone>`` element. The open count only errs high, so
    an unknown structure joins blocks (leaving more values masked) and never ends one early:
    every opener counts, while a closer counts only outside quotes (``"note ]}"`` closes nothing;
    _closers) and only against openers of its own kind (``<phone>\n])}`` leaves the element open).

    A triple-quoted string, XML comment, CDATA section or code fence is text, segmented like any
    other, but it belongs to the place that opened it: a value inside one is also near whatever
    is near its opening delimiter (``phone = \"\"\"\n\nnote\n\nx1+P``). Inside a fence brackets
    are code and count as usual. Inside a string, comment or CDATA section they count separately
    and only until it ends: a string's brackets and XML elements (serialized JSON or XML), a
    comment's or CDATA section's ``[`` and ``{`` only (``<!-- </phone> -->`` is text), and never
    a prose parenthesis.

    Built in one pass over the text when a digit-joined value first needs it; contact stems are
    found in one more pass.
    """

    def __init__(self, text: str):
        self.text, self.starts = text, None

    def _build(self) -> None:
        text, pos, blocks = self.text, 0, []
        for m in _BLANK_LINE_RE.finditer(text):
            if m.start() > pos:
                blocks.append((pos, m.start()))
            pos = max(pos, m.end())
        if pos < len(text):
            blocks.append((pos, len(text)))
        starts, block_starts, owners = [], [], []
        depth = collections.Counter()
        # Structure of data embedded in an open text region (a JSON or XML value inside a TOML
        # string), kept apart from ``depth`` so nothing inside the region can close or extend
        # structure outside it, and discarded when the region ends.
        inner = collections.Counter()
        long = None  # (closer, position, holds code) of the open long string or fence, if any
        section = None  # closer of an XML comment or CDATA section open inside a long string

        def count(piece: str) -> None:
            nonlocal depth, inner
            if long and not long[2]:
                # Text, not code: prose parentheses never merge. A string may hold serialized
                # data, so its brackets and elements nest; in a comment or CDATA section only
                # data brackets do, and tags there are text (``<!-- </phone> -->``).
                comment = section or long[0] in ("-->", "]]>")
                nests = (lambda k: k in _DATA_BRACKETS) if comment else (lambda k: k != "(")
                inner.update({k: n for k, n in _openers(piece).items() if nests(k)})
                inner.subtract({k: n for k, n in _closers(piece).items() if nests(k)})
                inner = +inner
                return
            depth.update(_openers(piece))
            depth.subtract(_closers(piece))
            depth = +depth  # an unmatched closer never goes below zero

        for a, b in blocks:
            if not starts or (not depth and not inner and not _CONTINUATION_LINE_RE.match(text, a)):
                starts.append(a)
                depth = collections.Counter()
            block_starts.append(a)
            owners.append(long[1] if long else None)
            cursor = a
            for m in _LONG_STRING_RE.finditer(text, a, b):
                count(text[cursor:m.start()])
                cursor = m.end()
                delimiter = m.group().lstrip(" \t0123456789|")
                if long is None:
                    if delimiter in _LONG_CLOSER:  # a stray closer (``-->``) opens nothing
                        long = (_LONG_CLOSER[delimiter], m.end(), delimiter in ("```", "~~~"))
                elif delimiter == long[0]:
                    long, section = None, None
                    inner = collections.Counter()
                elif long[0] in ('"""', "'''"):
                    if section is None and delimiter in _TEXT_SECTIONS:
                        section = _LONG_CLOSER[delimiter]
                    elif delimiter == section:
                        section = None
            count(text[cursor:b])
        self.starts, self.block_starts, self.owners = starts, block_starts, owners
        self.stems = [m.start() for m in _CONTACT_STEM_RE.finditer(text)]

    def _stem_near(self, pos: int) -> bool:
        """Whether a contact stem is in the segment holding ``pos`` or in the segment before it."""
        k = bisect.bisect_right(self.starts, pos) - 1
        lo = self.starts[max(0, k - 1)] if k >= 0 else 0
        hi = self.starts[k + 1] if k + 1 < len(self.starts) else len(self.text)
        n = bisect.bisect_left(self.stems, lo)
        return n < len(self.stems) and self.stems[n] < hi

    def contact_near(self, pos: int) -> bool:
        """Whether a contact name is near ``pos``, or near the opening of the long string or fence
        that ``pos`` lies in."""
        if self.starts is None:
            self._build()
        if self._stem_near(pos):
            return True
        owner = self.owners[bisect.bisect_right(self.block_starts, pos) - 1]
        return owner is not None and self._stem_near(owner)


def _digit_joined(text: str, start: int, segments: "_Segments") -> bool:
    """Whether the ``+`` at ``start`` joins two numbers rather than starting a phone number.

    E.164 writes ``+`` at the start of the number, never directly after a digit, so a ``+`` glued
    to a preceding digit joins two values: ``new09f71717+5323013``, ``1.2.3+7654321``, ``r7+P``.
    Contact context still wins, and is never cropped to a lookup window: any contact name in the
    value's segment or the segment before it (``phone1: x1+P``, ``{"phone": "<any text> x1+P"}``,
    ``Phone:\n\n1+P``, ``"phone": [\n\n  "x1+P"``) leaves the value to the ordinary
    classification below, as before.
    """
    return start > 0 and text[start - 1] in "0123456789" and not segments.contact_near(start)


def _contact_labelled(text: str, start: int, floor: int, memo: dict) -> bool:
    """Whether a contact label introduces the value at ``start`` or the list it continues.

    ``Phone: +P / +Q bytes`` labels ``+Q`` too: the earlier list values are skipped back to the
    label. A word between them (``Phone: +P; freed +Q bytes``) ends the contact value. Nothing
    before ``floor``, the start of the table cell holding the value, belongs to its field.

    Every value of one list reaches the same label, so ``memo`` (one per text) records the answer
    for each position walked: later values of a long list stop at the previous value's entry
    instead of walking the whole list again.
    """
    pos, walked = start, []
    while (pos, floor) not in memo:
        walked.append((pos, floor))
        m = _LIST_PREDECESSOR_RE.search(text, max(floor, pos - _QUANTITY_CONTEXT), pos)
        if not m:
            memo[pos, floor] = bool(
                _CONTACT_LABEL_RE.search(text, max(floor, pos - _LABEL_CONTEXT), pos))
            break
        pos = m.start()
    for key in walked:
        memo[key] = memo[pos, floor]
    return memo[pos, floor]


def _unit_follows(text: str, end: int, memo: dict) -> bool:
    """Whether a data-size unit follows the number ending at ``end`` or the list it starts.

    Each value of a long list shares the unit after the list, so ``memo`` (one per text) records
    the answer for each position stepped through rather than rescanning the rest of the list.
    """
    pos, stepped = end, []
    while pos not in memo:
        stepped.append(pos)
        m = _LIST_STEP_RE.match(text, pos)
        if not m:
            memo[pos] = bool(_UNIT_AFTER_RE.match(text, pos))
            break
        pos = m.end()
    for key in stepped:
        memo[key] = memo[pos]
    return memo[pos]


def non_phone_offsets(text: str) -> frozenset:
    """Offsets of each phone-shaped ``+digits`` that is evidently a signed quantity or joins two numbers.

    Contact context always masks it: a contact label directly before it or before the list it
    continues (``Phone: ``, ``tel:``, ``Call``, ``**Phone:**``, ``"phone": "``, ``Phone: +P / ``),
    a cell of a contact table column, or a table row that cannot be mapped to its columns. A sign
    glued to a label is a quantity only when the label is a storage-accounting one (``pool``,
    ``HOME``) and a data-size unit follows or the value is a field of a compact run such as
    ``HOME+1109262336B/pool+2941231104/FREEING0``; any other glued label (``Alice``, ``alice``)
    masks it. An unglued value is a quantity when it is a cell of a declared-quantity table
    column or a data-size unit follows it or the ``/`` list of signed integers it starts
    (``+820772864 / +1340882944 bytes``). Each occurrence needs its own evidence: equal digits
    elsewhere say nothing about what this one means. A lone ``+digits`` (``delta +P``) stays
    masked. A ``+`` glued directly after a digit joins two values and is no phone number unless a
    contact name is near it (``_digit_joined``).
    """
    table, masked, cells = (_table_cells(text) if "|" in text and "-" in text
                            else (frozenset(), [], []))
    masked.sort()
    masked_starts = [a for a, _ in masked]
    cell_starts = [a for a, _ in cells]  # table rows are in text order, so already sorted
    kept, labelled, units, segments = set(), {}, {}, _Segments(text)
    for m in SIGNAL_PHONE_RE.finditer(text):
        start = m.start()
        window = max(0, start - _QUANTITY_CONTEXT)
        span = bisect.bisect_right(masked_starts, start) - 1
        if span >= 0 and start < masked[span][1]:
            continue
        cell = bisect.bisect_right(cell_starts, start) - 1
        floor = cells[cell][0] if cell >= 0 and start < cells[cell][1] else 0
        if _contact_labelled(text, start, floor, labelled):
            continue
        if _digit_joined(text, start, segments):
            kept.add(start)
            continue
        unit_follows = _unit_follows(text, m.end(), units)
        label = _glued_label(text, start, window)
        if label:
            if _OPERATIONAL_LABEL_RE.fullmatch(label) and (
                    unit_follows or _in_operational_run(text, m, window)):
                kept.add(start)
        elif start in table or unit_follows:
            kept.add(start)
    return frozenset(kept)
