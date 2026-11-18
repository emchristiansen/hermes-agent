"""The E.164 phone pass must tell a signed quantity from a phone number.

``+1109262336`` (bytes freed) is digit-for-digit the shape of an E.164 number, and the phone pass
masked it to ``+110****2336`` in read_file output and in kanban summaries before they were stored,
destroying accounting deltas. Each occurrence is a quantity only on its own evidence: followed by a
data-size unit, a field of a compact run of storage-accounting fields, or a lone cell of a
right-aligned Markdown column whose own header declares a quantity (a data-size unit, or a change in
a table whose header declares one). Contact context always masks it: a contact label directly
before it or a cell under a header naming contact data. A lone ``+digits``, or equal digits
elsewhere, are no evidence. A ``+`` glued to an ASCII word character or a dot is never a phone match
at all (#142574, ``tests/agent/test_redact_non_secret_shapes.py``), so glued values stay as written.
"""
import json

import pytest

from agent.redact import redact_sensitive_text, redact_terminal_output

PHONE = "+15551234567"  # reserved fictional 555 number
P = "+" + "12025550123"  # the independent review's synthetic witness
P_MASKED = "+120****0123"

# The accounting table and summary that were damaged, byte for byte.
TABLE = (
    "## Measured accounting — shared, not exclusive yield\n"
    "\n"
    "| Measure (bytes) |13:46:08 before|13:46:35 after|Shared-host change|\n"
    "|---|---:|---:|---:|\n"
    "|/home usable|538819559424|539928821760|+1109262336|\n"
    "|rpool/home AVAILABLE|538819592192|539928870912|+1109278720|\n"
    "|Pool FREE|671258910720|674200141824|+2941231104|\n"
    "|Pool FREEING|0|0|0|\n"
)
SUMMARY = (
    "OriginalsharedwindowHOME+1109262336B/pool+2941231104/FREEING0 separate nonadditive. "
    "My kanban_schedule tool payload supplied pool delta +2941231104 bytes"
)


@pytest.fixture(autouse=True)
def _redaction_enabled(monkeypatch):
    # agent.redact snapshots _REDACT_ENABLED at import time; force it on.
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _gutter(text: str, start: int = 1) -> str:
    """read_file's rendering: ``N|line``."""
    return "".join(f"{n}|{line}" for n, line in enumerate(text.splitlines(True), start))


@pytest.mark.parametrize("mode", [{}, {"file_read": True}, {"force": True}])
@pytest.mark.parametrize("text", [TABLE, _gutter(TABLE, 41), SUMMARY], ids=["table", "gutter", "summary"])
def test_signed_operational_integers_survive_unchanged(text, mode):
    assert redact_sensitive_text(text, **mode) == text


@pytest.mark.parametrize("text", [
    "HOME+1109262336B/pool+2941231104/FREEING0",  # compact run of accounting fields
    "OriginalsharedwindowHOME+1109262336B/pool+2941231104/FREEING0 separate nonadditive.",
    "pool+2941231104/FREEING0",          # two-field accounting run
    "pool+2941231104 bytes",             # accounting label with a unit
    "(HOME+1109262336B/pool+2941231104),",
    "freed +2941231104 bytes",          # data-size units
    "r7 **+820772864 / +1340882944 bytes**; r5 **+545652736 / −69820416 bytes**",  # signed list + unit
    "| Measure (bytes) | Change |\n|---|---:|\n| pool | +2941231104 |\n\n"
    "| Measure (bytes) | Change |\n|---|---:|\n| home | +1109262336 |",  # adjacent valid tables
    "freed +2941231104 bytes; and +2941231104 bytes",
    "grew +3000000000 GiB",
    "| k | delta (GiB) |\n|---|--:|\n| x | +12345678 |\n| y | **+3000000000** |",  # unit column
    "| Measure (bytes) | change |\n|---|---:|\n| pool | +2941231104 |",  # change, table unit
    "bare 1109262336 and 2941231104",   # unsigned integers were never touched
])
def test_quantity_contexts_are_not_phone_numbers(text):
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("text", [
    f"Call {PHONE} now",
    f"user_id={PHONE}",
    f'{{"phone": "{PHONE}"}}',
    f"{PHONE}, +15559876543",
    f"tel:{PHONE}",
    f"{PHONE} Bob",
    "delta +2941231104",  # lone signed integer in prose: indistinguishable, fail closed
    f"| Name | Phone |\n|---|---|\n| Alice | {PHONE} |",       # default-aligned column
    f"| Name | Phone |\n|:---|:---:|\n| Alice | {PHONE} |",     # left / centre column
    f"| Name | Phone |\n|---|---:|\n| Alice | {PHONE} |",       # right-aligned, no declared quantity
    _gutter(f"| Name | Phone |\n|---|---:|\n| Alice | {PHONE} |"),
    "| Measure | change |\n|---|---:|\n| pool | +2941231104 |",  # change, but no unit anywhere
    f"| Measure (bytes) | Phone |\n|---|---:|\n| disk owner | {PHONE} |",  # unit names another column
    *[form(f"| Measure (bytes) | Phone |\n|---|---:|\n| disk owner | {PHONE} |\n| unknown | {v} |")
      for form in (str, _gutter) for v in ("0", "-4096")],  # sentinels do not make Phone a quantity
    f"| Measure (bytes) | change |\n|---|---:|\n| a | 0 |\n| {PHONE} | {PHONE} | x |",  # row too long
    f"| Measure (bytes) | x | change |\n|---|---|---:|\n| a | b | 0 |\n| {PHONE} | x |",  # row too short
    f"| Name (bytes) | Phone |\n|---|---|\n| Alice | {PHONE} |",  # unit, column not right-aligned
    f"| Name (bytes) | Phone |\n|---|---:|\n| Alice | {PHONE} (cell) |",  # not a lone integer
    f"| Phone | Name (bytes) |\n|---|---:|\n| {PHONE} | Alice |",  # right-aligned is the other column
    f"| A | B (bytes) | C |\n|---|---:|\n| x | {PHONE} |",      # header/delimiter mismatch: no table
    f"| A | B (bytes) |\n|---|---:|\n| x | 1 |\n\n| x | {PHONE} |",  # blank line ends the table
    _gutter(f"| Name | Phone |\n|---|---|\n| Alice | {PHONE} |"),
])
def test_phone_numbers_stay_masked(text):
    out = redact_sensitive_text(text)
    assert "****" in out
    assert PHONE not in out and "+2941231104" not in out


@pytest.mark.parametrize("text", [
    f"Phone: {P} bytes=4096",            # IND-PHONE-02: unit-looking metadata after a contact
    f"tel:{P} bytes transferred: 4096",
    f"Contact: {P} Byte, Alice",
    f"{P} bytes=4096",                   # a following field, not this number's unit
    # IND-PHONE-04: a heading or new header ends the previous table's schema
    *[form(f"| Measure (bytes) | Change |\n|---|---:|\n| pool | 0 |\n{between}"
           f"| Name | Phone |\n|---|---:|\n| owner | {P} |")
      for form in (str, _gutter)
      for between in ("## Contacts | directory\n", "", "| x | y | z |\n")],
    # IND-PHONE-05: a row that cannot be mapped to columns gets no exemption
    *[form(f"| Name | Phone{extra_header} |\n|---|---:{extra_delim}|\n| owner | {P}{unit}{extra_cell} |")
      for form in (str, _gutter) for unit in ("", " bytes")
      for extra_header, extra_delim, extra_cell in ((" | Notes", "|---", ""), ("", "", " | note"))],
    # IND-PHONE-06: compound / serialized contact headers
    *[form(f"| Measure (bytes) | {header} |\n|---|---:|\n| owner | {P} |")
      for form in (str, _gutter)
      for header in ("phoneNumber change", "Cellphone (bytes)", "mobile_no delta", "CONTACT (bytes)")],
    # IND-PHONE-07: emphasized and quoted-key contact labels beat a following unit
    f"**Phone:** {P} bytes", f'{{"phone": "{P} bytes"}}', f"phone_number = '{P} bytes'",
    f"__Mobile__: {P} bytes", f"Phone: {P} / +4096 bytes",
    *[form(f"| Measure (bytes) | Phone |\n|---|{align}|\n| owner | {P} bytes |")
      for form in (str, _gutter) for align in ("---:", "---")],  # a contact cell beats a unit
    *[form(f"| Measure (bytes) | {header} |\n|---|---:|\n| owner | {P} |\n| x | 0 |")
      for form in (str, _gutter)
      for header in ("Phone number change", "Phone (stored as bytes)", "Mobile delta (bytes)")],
])
def test_contact_semantics_beat_quantity_words(text):  # independent review IND-PHONE-01..03
    out = redact_sensitive_text(text)
    assert P not in out and "+2941231104" not in out
    assert "****" in out


@pytest.mark.parametrize("text,expected", [
    # Equal digits in a quantity column do not make the Phone column a quantity.
    (f"| Measure (bytes) | change | Phone |\n|---|---:|---:|\n| disk | {P} | {P} |",
     f"| Measure (bytes) | change | Phone |\n|---|---:|---:|\n| disk | {P} | {P_MASKED} |"),
    (f"Phone: {P}; freed {P} bytes", f"Phone: {P_MASKED}; freed {P} bytes"),
])
def test_each_occurrence_is_judged_on_its_own_context(text, expected):
    assert redact_sensitive_text(text) == expected


def test_numeric_table_column_does_not_shield_credentials():
    secret = "ghp_" + "A" * 40  # synthetic
    text = f"| k | v (bytes) |\n|---|---:|\n| token | {secret} |\n| delta | +1109262336 |"
    out = redact_sensitive_text(text)
    assert secret not in out
    assert "| delta | +1109262336 |" in out


def test_quantity_column_beside_phone_column_in_one_table():
    text = (f"| Measure (bytes) | Phone | change |\n|---|---:|---:|\n"
            f"| disk | {PHONE} | 0 |\n| pool | +15559876543 | +2941231104 |\n")
    out = redact_sensitive_text(text)
    assert PHONE not in out and "+15559876543" not in out
    assert "| +2941231104 |" in out


@pytest.mark.parametrize("text,expected", [  # the same digits in each context
    (f"pool{P}/FREEING0", f"pool{P}/FREEING0"),            # accounting run
    (f"freed {P} bytes", f"freed {P} bytes"),              # unit
    (f"Phone: {P}", f"Phone: {P_MASKED}"),                 # contact label
    (f"delta {P}", f"delta {P_MASKED}"),                   # ambiguous
    (f"pool{P}x/FREEING0", f"pool{P}x/FREEING0"),          # not a phone match at all
])
def test_same_digits_in_each_context(text, expected):
    assert redact_sensitive_text(text) == expected


@pytest.mark.parametrize("text", [
    "pool+2941231104/FREEING0",
    "HOME+1109262336B/pool+2941231104/FREEING0",
])
def test_compact_run_survives_read_file_gutter(text):  # IND-QUANTITY-01
    gutter = _gutter(f"intro\n{text}\n| x |{text}|\n", 7)
    assert redact_sensitive_text(gutter, file_read=True) == gutter


Q = "+" + "12025550124"
Q_MASKED = "+120****0124"


@pytest.mark.parametrize("text,expected", [
    # IND-PHONE-08: a contact label scopes every value of its list, not only the first.
    (f"Phone: {P} / {Q} bytes", f"Phone: {P_MASKED} / {Q_MASKED} bytes"),
    (f'{{"phone": "{P} / {Q} bytes"}}', f'{{"phone": "{P_MASKED} / {Q_MASKED} bytes"}}'),
    (f"**Phone:** **{P}** / {Q} bytes", f"**Phone:** **{P_MASKED}** / {Q_MASKED} bytes"),
    (f"Phone: {P}, {Q} bytes", f"Phone: {P_MASKED}, {Q_MASKED} bytes"),
    (f"Phone: {P} or {Q} bytes", f"Phone: {P_MASKED} or {Q_MASKED} bytes"),
    (f"Phone: {P} / 4096 / {Q} bytes", f"Phone: {P_MASKED} / 4096 / {Q_MASKED} bytes"),
    (f"tel: {P} / {Q} / {P} bytes", f"tel: {P_MASKED} / {Q_MASKED} / {P_MASKED} bytes"),
    # A word between ends the contact value; the later quantity carries its own evidence.
    (f"Phone: {P}; freed {Q} bytes", f"Phone: {P_MASKED}; freed {Q} bytes"),
    # An unlabelled signed list is still a quantity.
    (f"pool delta {P} / {Q} bytes", f"pool delta {P} / {Q} bytes"),
])
def test_contact_label_scopes_its_whole_value_list(text, expected):
    assert redact_sensitive_text(text) == expected


_CONTACT_TABLES = [
    # IND-PHONE-09: GFM rows may drop or add the leading/trailing pipes of their header.
    f"| Name | Phone |\n|---|---:|\nowner | {P} bytes",
    f"Name | Phone\n---|---:\n| owner | {P} bytes |",
    f"| Name | Phone |\n|---|---:|\n| a | 0 |\nowner | {P} bytes",
    # A pipe-less line is still a row (here one cell of two): masked as unmappable.
    f"| Name | Phone |\n|---|---:|\n{P} bytes",
]
_QUANTITY_TABLES = [
    # IND-QUANTITY-03: the same pipe variations keep a declared quantity column.
    f"| Measure (bytes) | Change |\n|---|---:|\npool | {P}",
    f"Measure (bytes) | Change\n---|---:\n| pool | {P} |",
    # A blank line or a new block ends the table; a later line carries its own evidence.
    f"| Measure (bytes) | Change |\n|---|---:|\n| pool | 0 |\n\npool delta {P} / {Q} bytes",
    f"| Measure (bytes) | Change |\n|---|---:|\n| pool | 0 |\n## Totals\nfreed {P} bytes",
]


@pytest.mark.parametrize("form", [str, _gutter])
@pytest.mark.parametrize("text", _CONTACT_TABLES)
def test_contact_table_rows_mask_whatever_their_pipes(text, form):
    out = redact_sensitive_text(form(text), file_read=form is _gutter)
    assert P not in out and P_MASKED in out


@pytest.mark.parametrize("form", [str, _gutter])
@pytest.mark.parametrize("text", _QUANTITY_TABLES)
def test_quantity_table_rows_survive_whatever_their_pipes(text, form):
    assert redact_sensitive_text(form(text), file_read=form is _gutter) == form(text)


@pytest.mark.parametrize("text", [
    # IND-PHONE-10: serialized contact fields, however nested or encoded.
    json.dumps({"phone": [P + " bytes"]}),
    json.dumps({"phone": [P + " bytes", Q + " bytes"]}),
    json.dumps(json.dumps({"phone": P + " bytes"})),
    json.dumps({"phone": {"work": P + " bytes"}}),
    f"phone:\n  - {P} bytes\n  - {Q} bytes\n",
    f"Phone: {P} bytes, {Q} bytes",
    f"Phone: {P} | {Q} bytes", f"Phone: ({P}) / {Q} bytes", f"Phone: {P}\n{Q} bytes",
    f"Phone: {P} — {Q} bytes", f"phone_numbers: {P}; {Q}; {P} bytes", f"<phone>{P} bytes</phone>",
    json.dumps({"phones": [P, Q + " bytes"]}), json.dumps({"contact": {"mobile": P + " bytes"}}),
    f"1|phone:\n2|  - {P} bytes\n3|  - {Q} bytes\n",
])
def test_serialized_contact_fields_stay_masked(text):
    out = redact_sensitive_text(text)
    assert P not in out and Q not in out


def test_contact_field_ends_at_the_next_key():
    text = json.dumps({"phone": P + " bytes", "delta": Q + " bytes"})
    assert redact_sensitive_text(text) == text.replace(P, P_MASKED)


@pytest.mark.parametrize("form", [str, _gutter])
@pytest.mark.parametrize("text,expected", [
    # IND-QUANTITY-04: a contact label in one table cell does not reach into the next cell.
    (f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {P} | {Q} |",
     f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {P_MASKED} | {Q} |"),
    (f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {P} | {Q} bytes |",
     f"| Note | Change (bytes) |\n|---|---:|\n| Phone: {P_MASKED} | {Q} bytes |"),
    (f"Note | Change (bytes)\n---|---:\nPhone: {P} | {Q}",
     f"Note | Change (bytes)\n---|---:\nPhone: {P_MASKED} | {Q}"),
    # ... while a label inside the same cell still masks, and outside tables ``|`` still continues.
    (f"| Note | Change (bytes) |\n|---|---:|\n| owner | Phone: {P} bytes |",
     f"| Note | Change (bytes) |\n|---|---:|\n| owner | Phone: {P_MASKED} bytes |"),
    (f"Phone: {P} | {Q} bytes", f"Phone: {P_MASKED} | {Q_MASKED} bytes"),
    # IND-PHONE-11: a parenthesized first value is inside the label's field too.
    (f"Phone: ({P} bytes) / {Q} bytes", f"Phone: ({P_MASKED} bytes) / {Q_MASKED} bytes"),
    (f"Phone: ({P} bytes); freed {Q} bytes", f"Phone: ({P_MASKED} bytes); freed {Q} bytes"),
])
def test_contact_field_is_bounded_by_its_table_cell(text, expected, form):
    assert redact_sensitive_text(form(text), file_read=form is _gutter) == form(expected)


_N = 1000
_QUANTITY_LIST = " / ".join([P] * _N) + " bytes"
# IND-PERF-01 shapes: (input, exact expected output), each with _N phone-shaped values.
_LONG_INPUTS = {
    "quantity-list": (_QUANTITY_LIST, _QUANTITY_LIST),
    "contact-list": ("Phone: " + _QUANTITY_LIST, "Phone: " + " / ".join([P_MASKED] * _N) + " bytes"),
    "json-contact-array": (json.dumps({"phone": [P] * (_N - 1) + [P + " bytes"]}),
                           json.dumps({"phone": [P_MASKED] * (_N - 1) + [P_MASKED + " bytes"]})),
    "contact-then-quantity-list": (f"Phone: {P}; freed " + _QUANTITY_LIST,
                                   f"Phone: {P_MASKED}; freed " + _QUANTITY_LIST),
    "quantity-table": ("| Measure (bytes) | Change |\n|---|---:|\n" + f"| pool | {P} |\n" * _N,
                       "| Measure (bytes) | Change |\n|---|---:|\n" + f"| pool | {P} |\n" * _N),
    "contact-table": ("| Name | Phone |\n|---|---:|\n" + f"| owner | {P} bytes |\n" * _N,
                      "| Name | Phone |\n|---|---:|\n" + f"| owner | {P_MASKED} bytes |\n" * _N),
}


@pytest.mark.parametrize("name", sorted(_LONG_INPUTS))
def test_long_inputs_are_classified_exactly_in_linear_work(name, monkeypatch):
    """IND-PERF-01: each value is resolved from its neighbour, not by rescanning the list.

    Counts the list-walk regex calls rather than timing them, and checks every value's exact
    classification, so neither a slow nor a wrong answer passes.
    """
    import agent.redact_quantities as rq

    class Counting:
        def __init__(self, pattern):
            self.pattern, self.calls = pattern, 0

        def search(self, *args):
            self.calls += 1
            return self.pattern.search(*args)

        def match(self, *args):
            self.calls += 1
            return self.pattern.match(*args)

    walk = Counting(rq._LIST_PREDECESSOR_RE)
    step = Counting(rq._LIST_STEP_RE)
    monkeypatch.setattr(rq, "_LIST_PREDECESSOR_RE", walk)
    monkeypatch.setattr(rq, "_LIST_STEP_RE", step)
    text, expected = _LONG_INPUTS[name]
    assert redact_sensitive_text(text) == expected
    assert redact_sensitive_text(text, force=True) == expected
    assert walk.calls <= 6 * _N and step.calls <= 6 * _N  # two passes, a few calls per value


# A ``+`` glued directly after a digit joins two values; E.164 never writes one there.
_JOINED = [
    "DISPATCH/latest20/noSupport/new09f71717+5323013 FULLread",  # the damaged owner report
    "new09f71717+5323013",
    "build 1.2.3+7654321",              # SemVer build metadata
    "x9+7654321",
    f"ExactsharedHOME/pooldeltas r7{P}/HOME+1109262336B",  # digit-glued delta
    f"a1{P}{Q}",                        # each + follows a digit
    f"| Measure (bytes) | id |\n|---|---|\n| pool | r7{P} |",  # non-contact column
]


@pytest.mark.parametrize("mode", [{}, {"file_read": True}, {"force": True}])
@pytest.mark.parametrize("text", _JOINED)
def test_digit_joined_values_are_not_phone_numbers(text, mode):
    assert redact_sensitive_text(text, **mode) == text
    assert redact_sensitive_text(_gutter(text, 9), **mode) == _gutter(text, 9)


@pytest.mark.parametrize("text", _JOINED)
def test_digit_joined_values_survive_terminal_output(text):
    assert redact_terminal_output(text, "cat RESULT.md", force=True) == text


@pytest.mark.parametrize("prefix", [0, 50, 199, 200, 201, 300, 5000])
def test_digit_joined_value_after_a_long_unlabelled_token_is_kept(prefix):
    # No lookup window: without a contact name nearby, length alone never masks.
    text = "id " + "a" * prefix + "1" + P
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("text", [
    "id" + " " * 59 + "new09f71717+5323013",  # the word before the gap is inside the lookup
    "x" * 300 + " new09f71717+5323013",
    "callback " * 30 + "new09f71717+5323013",  # no contact word: ``call`` is only a whole word
])
def test_digit_joined_value_with_visible_context_is_kept(text):
    assert redact_sensitive_text(text) == text


def test_digit_joined_value_after_a_long_gap_without_contact_is_kept():
    text = "id" + " " * 300 + "new09f71717+5323013"
    assert redact_sensitive_text(text) == text


def _gutter_all(text: str) -> str:
    """read_file's rendering of every line, blank ones included (``N|``)."""
    return "\n".join(f"{n}|{line}" for n, line in enumerate(text.split("\n"), 1))


_PRESENTATIONS = {
    "normal": lambda s: redact_sensitive_text(s),
    "file_read": lambda s: redact_sensitive_text(s, file_read=True),
    "force": lambda s: redact_sensitive_text(s, force=True),
    "terminal": lambda s: redact_terminal_output(s, "cat RESULT.md", force=True),
    "gutter": lambda s: redact_sensitive_text(_gutter_all(s), file_read=True),
}


@pytest.mark.parametrize("mode", sorted(_PRESENTATIONS))
@pytest.mark.parametrize("text", [
    "DISPATCH/latest20/noSupport/new09f71717+5323013 FULLread",
    "build 1.2.3+7654321",
    "## Contacts\nPhone: " + P + "\n\nunrelated\n\nbuild r7+7654321",  # two blocks away
    'phone = """\nx\n"""\n\nnote\n\nbuild r7+7654321',  # the multi-line string closed earlier
    # A long report string (the original owner report's shape): prose inside it is segmented
    # as prose, and its parentheses are text, so a far-off contact word or ``(`` masks nothing.
    'report = """\nState.\n\ncalls44 consumed (see\n\nmore\n\nnoSupport/new09f71717+5323013 FULLread\n"""',
    # JSON inside the string scopes only itself: once closed, later prose is unrelated to it.
    'report = """\nState.\n\n{"phone": [\n\n"a"]}\n\nnote\n\nbuild r7+7654321\n"""',
    'report = """\nState.\n\n<phone>x</phone>\n\nnote\n\nbuild r7+7654321\n"""',  # closed element
    # Brackets and elements left open inside a string end with it.
    'report = """\nState.\n\n<phone>\n"""\n\nnote\n\nbuild r7+7654321',
    'report = """\nPhone: [\n"""\n\nnote\n\nbuild r7+7654321',
    "<a>x</a>\n<br/>\n\nnote\n\nbuild r7+7654321",  # closed and self-closing elements
    "<_phone>x</_phone>\n\nnote\n\nbuild r7+7654321",  # a closed underscore-named element
    "Phone: \"a ]}\" [x]\n\nnote\n\nbuild r7+7654321",  # every bracket closed outside quotes
])
def test_digit_joined_value_away_from_contacts_is_kept(text, mode):
    out = _PRESENTATIONS[mode](text)
    expected = text.replace("Phone: " + P, "Phone: " + P_MASKED)
    assert P not in expected
    assert out == (_gutter_all(expected) if mode == "gutter" else expected)


def test_original_owner_report_line_is_kept_exactly():
    # The damaged owner report sentence, verbatim, as read_file and the terminal present it.
    line = ("Originalhistoricalbi6Cargojob callbackowed/no newCargo/nextest. Completeoriginal18extents "
            "MANUAL44–73ONLY/auth3/CARRIER/separateCloud/DISPATCH/latest20/noSupport/new09f71717+5323013 "
            "FULLread/carryunchanged ALL3quotes ENTIRErecursive directive exactamendments;")
    assert redact_sensitive_text(line) == line
    assert redact_sensitive_text(_gutter(line, 9), file_read=True) == _gutter(line, 9)
    assert redact_terminal_output(line, "cat OWNER-UPDATE.toml", force=True) == line
