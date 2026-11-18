"""Tell signed quantities from E.164 phone numbers for the redactor's phone pass.

``+1109262336`` (bytes freed) has the shape of an E.164 number. ``non_phone_offsets`` names the
occurrences that carry their own evidence of being a quantity, so
``agent.redact.redact_sensitive_text`` leaves them intact and masks every other phone-shaped match.

``SIGNAL_PHONE_RE`` never matches a ``+`` glued to an ASCII word character or a dot (a version
suffix or two joined identifiers, #142574), so every value judged here stands on its own.
"""

import bisect
import re

# E.164 phone numbers, 7-15 digits; the lookahead rejects hex strings / identifiers. A ``+`` glued
# to a word or dot is a version suffix, not a phone: ``python-3.14.7+20260901-linux-x64`` (#142574).
SIGNAL_PHONE_RE = re.compile(r"(?<![0-9A-Za-z_.])(\+[1-9]\d{6,14})(?![A-Za-z0-9])")  # ASCII: CJK-glued numbers still mask

_DATA_SIZE_UNIT = r"(?:[Bb]ytes?|[kKMGTPE]i?B)(?![A-Za-z0-9])"
_DATA_SIZE_UNIT_RE = re.compile(rf"(?<![A-Za-z0-9]){_DATA_SIZE_UNIT}")
# The unit of this number, or of the ``/``-separated list of signed integers it starts:
# ``+2941231104 bytes``, ``+820772864 / +1340882944 bytes``. Not a ``bytes=4096`` field that merely
# follows the number. _unit_follows walks the list one step at a time.
_LIST_STEP_RE = re.compile(r"[ \t]*/[ \t]*[+\-−]\d+")
_UNIT_AFTER_RE = re.compile(rf"[ \t]+{_DATA_SIZE_UNIT}(?![ \t]*=)")
# Contact data, including serialized compound names (``phoneNumber``, ``Cellphone``, ``mobile_no``):
# distinctive stems match inside a word, short ones only as words.
_CONTACT_NAME = (r"(?:[a-z]*(?:phone|mobile|msisdn|whatsapp|contact)[a-z]*"
                 r"|(?<![a-z])(?:tel|telephone|cell|fax|sms|signal|call)(?:s|ed|er|ing)?(?![a-z]))")
_CONTACT_NAME_RE = re.compile(rf"(?i){_CONTACT_NAME}")
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
_QUANTITY_CONTEXT = 200  # chars searched before a match for the list it continues
_LABEL_CONTEXT = 60  # chars searched before a match for a contact label

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
    """Offsets of each phone-shaped ``+digits`` that is evidently a signed quantity.

    Contact context always masks it: a contact label directly before it or before the list it
    continues (``Phone: ``, ``tel:``, ``Call``, ``**Phone:**``, ``"phone": "``, ``Phone: +P / ``),
    a cell of a contact table column, or a table row that cannot be mapped to its columns. Otherwise
    a value is a quantity when it is a cell of a declared-quantity table column or a data-size unit
    follows it or the ``/`` list of signed integers it starts (``+820772864 / +1340882944 bytes``).
    Each occurrence needs its own evidence: equal digits elsewhere say nothing about what this one
    means. A lone ``+digits`` (``delta +P``) stays masked.
    """
    table, masked, cells = (_table_cells(text) if "|" in text and "-" in text
                            else (frozenset(), [], []))
    masked.sort()
    masked_starts = [a for a, _ in masked]
    cell_starts = [a for a, _ in cells]  # table rows are in text order, so already sorted
    kept, labelled, units = set(), {}, {}
    for m in SIGNAL_PHONE_RE.finditer(text):
        start = m.start()
        span = bisect.bisect_right(masked_starts, start) - 1
        if span >= 0 and start < masked[span][1]:
            continue
        cell = bisect.bisect_right(cell_starts, start) - 1
        floor = cells[cell][0] if cell >= 0 and start < cells[cell][1] else 0
        if _contact_labelled(text, start, floor, labelled):
            continue
        if start in table or _unit_follows(text, m.end(), units):
            kept.add(start)
    return frozenset(kept)
