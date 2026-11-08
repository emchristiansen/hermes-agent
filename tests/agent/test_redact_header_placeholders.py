"""Header placeholders stay byte-for-byte on every display surface; credentials stay masked.

Regression for display-fidelity intake 10123: ``Authorization: Bearer %s\\n`` in a printf
template, ``$OWNER_TOKEN`` / ``${TOKEN}`` / ``<token>`` / ``{token}`` references were shown as
``Bearer ***`` (format specifier and escape deleted) by read_file, search_files, terminal
output (native-history readers) and kanban comment storage, so readers concluded the source
was wrong. Only a slot that is WHOLLY a placeholder is exempt; every value-bearing shape below
must stay masked. All values are fabricated. Review6101 HF-01/HF-02: shell defaults
(``${T:-value}``) and segmented/opaque wrappers (``<Ab12Cd34-Ef56…>``) carry values and stay masked.
Review6184 HF-04: a credential word plus an opaque tail (``<TOKEN_AbCdEf…>``) is a value too;
HF-05..07: ``${10}``, ``%.*s``/``%1$s`` and ``{{ token }}`` are value-free and stay verbatim.
Review6279 HF-08: a placeholder at the END of a JSON string keeps the JSON closers; HF-09:
``{{token }}`` / ``{{  token  }}`` / tab forms; HF-10: ``${SERVICE_2026_TOKEN}`` and ``% s``.
Wrapper names split (``_``/``-``/camelCase) entirely into the closed word list, so
``{accessToken}`` and ``{serviceToken}`` qualify; opaque tails and naked closers do not.
"""
import json
import re
import subprocess
import types

import pytest

BS = "\\"  # one literal backslash
SYN = "SYNTHETICopaque0123456789abcdefABCDEF4567"  # fabricated, credential-shaped


@pytest.fixture(autouse=True)
def _redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def R():
    import agent.redact as r
    return r


# ---------------------------------------------------------------- placeholders (RED -> GREEN)
PRINTF_TOK = f"  printf 'Authorization: Bearer %s{BS}n' \"$tok\" >\"$PRIV/connect.header\""
PRINTF_FN = f"bearer_header() {{ printf 'Authorization: Bearer %s{BS}n' \"$1\" >\"$2\"; }} # TOKEN FILE"
PLACEHOLDER_LINES = [
    PRINTF_TOK,
    PRINTF_FN,
    "curl -H \"Authorization: Bearer $OWNER_TOKEN\" https://x.invalid",
    "curl -H \"Authorization: Bearer ${CONNECT_TOKEN}\" https://x.invalid",
    "curl -H 'Authorization: Bearer <token>' https://x.invalid",
    "hdr = 'Authorization: Bearer {token}'.format(token=t)",
    "hdr = 'Authorization: Bearer %(token)s' % d",
    "Authorization: Bearer {{token}}",
    "Proxy-Authorization: Basic %s",
    "Authorization: %s",
    "curl -H 'x-api-key: $API_KEY' https://x.invalid",
    "curl -H \"Authorization: Bearer $1\" https://x.invalid",
    "curl -H \"Authorization: Bearer ${MCP_N8N_API_KEY}\" https://x.invalid",
    "Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}",
    "Authorization: Bearer $SOME_LONG_OWNER_TOKEN_VAR",
    "Authorization: Bearer <YOUR_API_TOKEN>",
    # review6184 HF-05/HF-06/HF-07: positional shell params, star/positional printf, spaced template
    "curl -H \"Authorization: Bearer ${1}\" https://x.invalid",
    "curl -H \"x-api-key: ${10}\" https://x.invalid",
    "printf 'Authorization: Bearer %.*s' \"$n\" \"$tok\"",
    "printf 'Authorization: Bearer %*s' 8 \"$tok\"",
    "printf 'x-api-key: %1$s' \"$tok\"",
    "Authorization: Bearer {{ token }}",
    "curl -H 'x-api-key: {{ api_key }}' https://x.invalid",
    "curl -H 'Authorization: Bearer <your-api-token>' https://x.invalid",
    # review6279 HF-09 / HF-10 and the closed-vocabulary camelCase wrapper names
    "curl -H \"Authorization: Bearer {{token }}\" https://x.invalid",
    "curl -H \"x-api-key: {{  token  }}\" https://x.invalid",
    "curl -H \"Authorization: Bearer {{\ttoken\t}}\" https://x.invalid",
    "curl -H \"x-api-key: ${SERVICE_2026_TOKEN}\" https://x.invalid",
    "curl -H \"Authorization: Bearer $SERVICE_2026_TOKEN\" https://x.invalid",
    "hdr = 'x-api-key: % s' % tok",
    "hdr = \"Authorization: Bearer {accessToken}\".format(accessToken=tok)",
    "curl -H \"Authorization: Bearer {{ accessToken }}\" https://x.invalid",
    "curl -H \"x-api-key: {serviceToken}\" https://x.invalid",
]


@pytest.mark.parametrize("line", PLACEHOLDER_LINES)
@pytest.mark.parametrize("surface", ["file_read", "terminal", "kanban_comment_write"])
def test_placeholder_header_value_displayed_verbatim(line, surface):
    r = R()
    if surface == "file_read":
        out = r.redact_sensitive_text(line, file_read=True)
    elif surface == "terminal":
        out = r.redact_terminal_output(line, "cat /synthetic/x.sh")
    else:  # tools/kanban_tools.py:961 body = _redact(...) -> :349 redact_sensitive_text(force=True)
        from tools import kanban_tools
        out = kanban_tools._redact(line)
    assert out == line


# Placeholder-shaped slots that can carry a literal value keep the base masking: a default
# (HF-01), a command substitution, a dotted/indexed field. The slot text must not survive.
VALUE_CAPABLE_SLOTS = [
    ("curl -H \"Authorization: Bearer ${TOKEN:-unset}\" https://x.invalid", "${TOKEN:-unset}"),
    ("curl -H \"Authorization: Bearer $(cat /run/synthetic/tok)\" https://x.invalid", "$(cat"),
    ("hdr = f'Authorization: Bearer {self.token}'", "{self.token}"),
    ("curl -H 'x-api-key: ${API_KEY:=fallback}' https://x.invalid", "${API_KEY:=fallback}"),
    # review6184 HF-04 boundary: wrapper words outside the closed placeholder vocabulary keep
    # the base masking even when they read like documentation (casing/keywords prove nothing)
    ("curl -H 'Authorization: Bearer <YOUR_GITHUB_TOKEN_HERE>' https://x.invalid", "<YOUR_GITHUB_TOKEN_HERE>"),
    ("curl -H 'x-api-key: <GITHUB_TOKEN>' https://x.invalid", "<GITHUB_TOKEN>"),
    # HF-07 boundary: literal text glued after the closing braces is not a bare placeholder
    ("Authorization: Bearer {{ token }}SYNTHETICvalue", "{{"),
    # review6279 controls: a backslash alone is not a boundary; naked closers are not structure;
    # digit segments are bounded; camelCase words must ALL be in the closed list
    (f"curl -H \"Authorization: Bearer {{{{ token }}}}{BS}AbCdEfGhIjKlMnOp\" https://x.invalid", "{{"),
    (f"curl -H \"x-api-key: $TOKEN{BS}AbCdEfGhIjKlMnOp\" https://x.invalid", "AbCdEfGhIjKlMnOp"),
    ("curl -H \"x-api-key: $TOKEN}AbCdEfGhIjKlMnOp\" https://x.invalid", "AbCdEfGhIjKlMnOp"),
    ("x-api-key: $TOKEN]", "$TOKEN]"),
    ("curl -H \"x-api-key: % sAbCdEfGhIjKl\" https://x.invalid", "%"),
    ("curl -H \"x-api-key: ${SERVICE_12345_TOKEN}\" https://x.invalid", "12345"),
    ("curl -H \"x-api-key: ${A_2026_2027_2028_TOKEN}\" https://x.invalid", "2028"),
    ("curl -H \"x-api-key: {tokenAbCdEfGhIjKl}\" https://x.invalid", "AbCdEfGhIjKl"),
    ("curl -H \"Authorization: Bearer {syntheticSecret}\" https://x.invalid", "syntheticSecret"),
    ("curl -H \"x-api-key: {{  TOKEN_AbCdEfGhIjKlMnOp  }}\" https://x.invalid", "{{"),
]


@pytest.mark.parametrize("line,slot", VALUE_CAPABLE_SLOTS)
@pytest.mark.parametrize("surface", ["file_read", "terminal", "kanban_comment_write"])
def test_value_capable_placeholder_shapes_stay_masked(line, slot, surface):
    r = R()
    if surface == "file_read":
        out = r.redact_sensitive_text(line, file_read=True)
    elif surface == "terminal":
        out = r.redact_terminal_output(line, "cat /synthetic/x.sh")
    else:
        from tools import kanban_tools
        out = kanban_tools._redact(line)
    assert slot not in out, out


def test_printf_escape_bytes_preserved_exactly():
    out = R().redact_sensitive_text(PRINTF_TOK, file_read=True)
    assert out.encode().hex() == PRINTF_TOK.encode().hex()
    assert "%s\\n'" in out and "***" not in out


def test_native_history_json_roundtrips_verbatim():
    rec = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write",
           "input": {"file_path": "/synthetic/chain.sh", "content": PRINTF_FN + "\n"}}]}}
    line = json.dumps(rec)
    assert "%s\\\\n" in line                       # JSON-escaped backslash
    for out in (R().redact_sensitive_text(line, file_read=True),
                R().redact_terminal_output(line, "python3 history.py")):
        assert out == line
        assert json.loads(out)["message"]["content"][0]["input"]["content"] == PRINTF_FN + "\n"


def test_actual_read_file_and_search_tools_show_disk_bytes(tmp_path):
    from tools import file_tools as F
    p = tmp_path / "s1.sh"
    p.write_bytes((PRINTF_TOK + "\n").encode())
    shown = json.loads(F.read_file_tool(str(p), task_id="r3t"))["content"]
    assert f"Bearer %s{BS}n'" in shown and "***" not in shown
    res = json.loads(F.search_tool("Authorization", path=str(p), task_id="r3t"))
    contents = [m["content"] for m in res["matches"]]
    assert contents and all(f"Bearer %s{BS}n'" in c and "***" not in c for c in contents)


# ---------------------------------------------------------------- protections (GREEN on both)
PROTECTED = [
    (f"Authorization: Bearer {SYN}", SYN[6:-4]),
    ("Authorization: Bearer syn7short", "syn7short"),
    ("Authorization: Basic U1lOVEhFVElDOnN5bnRoZXRpYw==", "OnN5bnRoZXRpYw"),
    (f"Authorization: Bearer %s{SYN}", SYN[6:-4]),
    (f"Authorization: Bearer ${{TOKEN}}{SYN}", SYN[6:-4]),
    (f"Authorization: Bearer <token>{SYN}", SYN[6:-4]),
    (f"Authorization: Bearer {{token}}{SYN}", SYN[6:-4]),
    ("Authorization: Bearer $2a$10$SYNTHETICbcryptSaltAndHashValue0123456", "SYNTHETICbcryptSalt"),
    ("Authorization: Bearer $AbCdEfGh0123456789XyZwVu", "AbCdEfGh0123456789XyZw"),
    (f"curl -H \"Authorization: Bearer {SYN}{BS}n\"", SYN[6:-4]),
    (f"curl -H 'x-api-key: {SYN}'", SYN[6:-4]),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.c3ludGhldGljc2ln", "eyJzdWIiOiJzeW50aGV0aWMifQ"),
    ('curl -H "Authorization: Bearer sk-abcdef1234567890"', "abcdef1234567890"),
    # placeholder-SHAPED wrappers around opaque material must not be exempted
    (f"Authorization: Bearer ${{TOKEN:-{SYN}}}", SYN[6:-4]),
    (f"Authorization: Bearer {{{SYN}}}", SYN[6:-4]),
    (f"Authorization: Bearer <{SYN}>", SYN[6:-4]),
    (f"Authorization: Bearer $({SYN})", SYN[6:-4]),
    (f"Authorization: Bearer %({SYN})s", SYN[6:-4]),
    ("Authorization: Bearer $deadbeef00112233445566778899aabbccddeeff", "00112233445566778899aabb"),
    (f"curl -H 'x-api-key: ${{K:-{SYN}}}'", SYN[6:-4]),
    # review6101 HF-01: short / segmented / alphabetic shell defaults carry the value
    *[(f"curl -H \"{fam}${{TOKEN:-{v}}}\" https://x.invalid", v)
      for fam in ("Authorization: Bearer ", "x-api-key: ")
      for v in ("syn7short", "Ab12Cd34_Ef56Gh78_Ij90Kl12_Mn34Op56", "SyntheticSecretPassword")],
    # review6101 HF-02: segmented opaque wrappers, plus underscore/dotted/alphabetic variants
    *[(f"curl -H \"{fam}{w}\" https://x.invalid", inner)
      for fam in ("Authorization: Bearer ", "x-api-key: ")
      for w, inner in (("<Ab12Cd34-Ef56Gh78-Ij90Kl12-Mn34Op56>", "Ab12Cd34-Ef56Gh78-Ij90Kl12-Mn34Op56"),
                       ("<Ab12Cd34_Ef56Gh78_Ij90Kl12_Mn34Op56>", "Ab12Cd34_Ef56Gh78_Ij90Kl12_Mn34Op56"),
                       ("<Ab12Cd34.Ef56Gh78.Ij90Kl12>", "Ab12Cd34.Ef56Gh78.Ij90Kl12"),
                       ("{Ab12Cd34Ef56Gh78Ij90Kl12}", "Ab12Cd34Ef56Gh78Ij90Kl12"),
                       ("<SyntheticSecretPassword>", "SyntheticSecretPassword"),
                       ("{{SyntheticSecretPassword}}", "SyntheticSecretPassword"))],
    # curator6149 ADJACENT (HF-02 family): short wrapped material is a value, not a reference
    *[(f"curl -H \"{fam}{w}\" https://x.invalid", w)
      for fam in ("Authorization: Bearer ", "x-api-key: ")
      for w in ("<syn7short>", "<ab12>", "<q7w2>", "<hunter>", "{ab12}", "{{q7w2}}", "<AB12>")],
    # review6184 HF-04: a credential word plus an opaque/mixed-case tail is a value
    *[(f"curl -H \"{fam}{w}\" https://x.invalid", inner)
      for fam in ("Authorization: Bearer ", "x-api-key: ")
      for w, inner in (("<TOKEN_AbCdEfGhIjKlMnOp>", "AbCdEfGhIjKlMnOp"),
                       ("{TOKEN_AbCdEfGhIjKlMnOp}", "AbCdEfGhIjKlMnOp"),
                       ("{{TOKEN_AbCdEfGhIjKlMnOp}}", "AbCdEfGhIjKlMnOp"),
                       ("<SyntheticSecret>", "SyntheticSecret"),
                       ("<token_qwertyuiopasdfgh>", "qwertyuiopasdfgh"),
                       ("<SECRET_QWERTYUIOPASDF>", "QWERTYUIOPASDF"),
                       ("<YOUR_AbCdEfGhIjKl_HERE>", "AbCdEfGhIjKl"))],
]


@pytest.mark.parametrize("line,secret_part", PROTECTED)
@pytest.mark.parametrize("surface", ["file_read", "terminal", "egress", "kanban_comment_write"])
def test_real_shaped_values_stay_masked(line, secret_part, surface):
    r = R()
    def _kanban():
        from tools import kanban_tools
        return kanban_tools._redact(line)
    out = {"file_read": lambda: r.redact_sensitive_text(line, file_read=True),
           "terminal": lambda: r.redact_terminal_output(line, "cat /synthetic/x"),
           "egress": lambda: r.redact_for_egress(line),
           "kanban_comment_write": _kanban}[surface]()
    assert secret_part not in out, out
    assert out != line


def test_vault_value_scrub_still_precedes_and_ignores_placeholder_exemption(monkeypatch):
    r = R()
    r.register_vault_redaction_value("$VAULTSYNTHETIC_OWNER")   # pathological: secret shaped like a var ref
    try:
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
        out = r.redact_sensitive_text("Authorization: Bearer $VAULTSYNTHETIC_OWNER", file_read=True)
        assert "VAULTSYNTHETIC_OWNER" not in out and "«redacted-vault-secret»" in out
    finally:
        r.clear_vault_redaction_values()


def test_already_masked_value_stays_masked():
    assert R().redact_sensitive_text("Authorization: Bearer ***", file_read=True) == "Authorization: Bearer ***"


# The reviewer's interaction contract (review6101 probe; review6141 verify.py for JSON/native
# history, HF-03) on the real producer paths: read_file, search_files, the terminal foreground
# finalizer, the kanban pre-write and egress. Each slot is checked raw and json.dumps-encoded
# (native-history shape), inside double and single quotes.
INTERACTION_SAMPLES = {
    "default_short": ("${TOKEN:-syn7short}", "syn7short", False),
    "default_segmented": ("${TOKEN:-Ab12Cd34_Ef56Gh78_Ij90Kl12_Mn34Op56}", "Ab12Cd34_Ef56Gh78_Ij90Kl12_Mn34Op56", False),
    "default_alpha": ("${TOKEN:-SyntheticSecretPassword}", "SyntheticSecretPassword", False),
    "wrapped_segmented": ("<Ab12Cd34-Ef56Gh78-Ij90Kl12-Mn34Op56>", "Ab12Cd34-Ef56Gh78-Ij90Kl12-Mn34Op56", False),
    "opaque_control": ("SyntheticOpaque0123456789AbCdEf", "SyntheticOpaque0123456789AbCdEf", False),
    "printf_control": (f"%s{BS}n", "", True),
    "var_control": ("$TOKEN", "", True),
    "braced_control": ("${TOKEN}", "", True),
    "wrapped_short_alnum": ("<syn7short>", "<syn7short>", False),
    "wrapped_tiny_alnum": ("<ab12>", "<ab12>", False),
    "wrapped_doc_control": ("<token>", "", True),
    "wrapped_doc_upper_control": ("<YOUR_API_TOKEN>", "", True),
    # review6184 HF-04 (value-bearing) and HF-05/06/07 (value-free)
    "hf04_angle_token_tail": ("<TOKEN_AbCdEfGhIjKlMnOp>", "AbCdEfGhIjKlMnOp", False),
    "hf04_brace_token_tail": ("{TOKEN_AbCdEfGhIjKlMnOp}", "AbCdEfGhIjKlMnOp", False),
    "hf04_jinja_token_tail": ("{{TOKEN_AbCdEfGhIjKlMnOp}}", "AbCdEfGhIjKlMnOp", False),
    "hf04_mixed_case_word": ("<SyntheticSecret>", "SyntheticSecret", False),
    "hf05_positional_1": ("${1}", "", True),
    "hf05_positional_10": ("${10}", "", True),
    "hf06_star_precision": ("%.*s", "", True),
    "hf06_star_width": ("%*s", "", True),
    "hf06_positional": ("%1$s", "", True),
    "hf07_spaced_template": ("{{ token }}", "", True),
    "control_fmt_prefixed_value": ("%sSYNTHETICvalue", "SYNTHETICvalue", False),
    # review6184 required controls: value material glued before/after a supported placeholder,
    # value-bearing modifiers/defaults, and wrapper words outside the closed vocabulary
    "hf04_doc_wrapper_opaque_suffix": ("<token>AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "hf04_doc_wrapper_opaque_prefix": ("AbCdEfGhIjKlMnOp<YOUR_API_TOKEN>", "AbCdEfGhIjKlMnOp", False),
    "hf04_spaced_template_opaque_name": ("{{ TOKEN_AbCdEfGhIjKlMnOp }}", "{{", False),
    "hf04_unlisted_wrapper_word": ("<YOUR_QWERTYUIOP_HERE>", "QWERTYUIOP", False),
    "hf05_positional_literal_suffix": ("${1}AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "hf05_positional_value_default": ("${1:-SyntheticSecretPassword}", "SyntheticSecretPassword", False),
    "hf06_star_literal_suffix": ("%.*sAbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "hf06_positional_literal_suffix": ("%1$sAbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "hf07_spaced_template_opaque_prefix": ("SyntheticOpaque0123456789AbCdEf{{ token }}", "SyntheticOpaque0123456789AbCdEf", False),
    # the header slot ends at the space after ``{{`` (base boundary); text glued after ``}}``
    # is not a bare placeholder, so the opener keeps the base mask
    "hf07_spaced_template_glued_suffix": ("{{ token }}AbCdEfGhIjKlMnOp", "{{", False),
    "hf07_spaced_template_api_key": ("{{ api_key }}", "", True),
    # review6279 HF-09 (whitespace templates), HF-10 (shell digit segment, printf space flag),
    # advisory camelCase wrapper names, and their value-bearing controls
    "hf09_template_no_leading_space": ("{{token }}", "", True),
    "hf09_template_double_spaces": ("{{  token  }}", "", True),
    "hf09_template_tabs": ("{{\ttoken\t}}", "", True),
    "hf09_template_no_trailing_space": ("{{ token}}", "", True),
    "hf09_template_backslash_opaque": ("{{ token }}" + BS + "AbCdEfGhIjKlMnOp", "{{", False),
    "hf09_template_naked_brace_suffix": ("{{token}}}AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "hf09_template_spaced_opaque_name": ("{{  TOKEN_AbCdEfGhIjKlMnOp  }}", "{{", False),
    "hf10_shell_digit_segment": ("${SERVICE_2026_TOKEN}", "", True),
    "hf10_shell_bare_digit_segment": ("$SERVICE_2026_TOKEN", "", True),
    "hf10_printf_space_flag": ("% s", "", True),
    "hf10_shell_long_digit_segment": ("${SERVICE_12345_TOKEN}", "SERVICE_12345_TOKEN", False),
    "hf10_shell_opaque_mixed": ("${Ab12Cd34Ef56Gh78}", "Ab12Cd34Ef56Gh78", False),
    "hf10_shell_digit_default": ("${SERVICE_2026_TOKEN:-SyntheticSecretPassword}", "SyntheticSecretPassword", False),
    # the header slot ends at the space after ``%`` (base boundary): base masks only ``%`` and
    # never the text after the space; glued material keeps exactly that base mask
    "hf10_printf_space_glued": ("% sAbCdEfGhIjKlMnOp", "%", False),
    "camel_vocab_access_token": ("{{ accessToken }}", "", True),
    "camel_vocab_service_token": ("{serviceToken}", "", True),
    "camel_opaque_tail": ("{tokenAbCdEfGhIjKl}", "AbCdEfGhIjKl", False),
    "backslash_opaque_suffix": ("$TOKEN" + BS + "AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
    "naked_brace_suffix": ("$TOKEN}AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
}


@pytest.mark.parametrize("name", sorted(INTERACTION_SAMPLES))
@pytest.mark.parametrize("family", ["Authorization: Bearer ", "x-api-key: "])
@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("encoding", ["raw", "json"])
def test_header_slot_contract_on_real_producer_paths(name, family, quote, encoding, tmp_path):
    from tools import file_tools, kanban_tools
    from tools.terminal_tool_result import finalize_foreground_result
    value, secret, preserve = INTERACTION_SAMPLES[name]
    raw = f"curl -H {quote}{family}{value}{quote} https://x.invalid"
    if encoding == "json":
        raw = json.dumps({"message": {"content": raw}})
    p = tmp_path / "probe.sh"
    p.write_text(raw + "\n")
    proc = subprocess.run(["cat", str(p)], capture_output=True, text=True, check=True)
    shown = {
        "read": json.loads(file_tools.read_file_tool(str(p), task_id="t"))["content"].removeprefix("1|"),
        "search": json.loads(file_tools.search_tool("Authorization|x-api-key", path=str(p),
                                                    task_id="t"))["matches"][0]["content"],
        "terminal": json.loads(finalize_foreground_result(
            command=f"cat {p}", result={"output": proc.stdout, "returncode": proc.returncode},
            env=types.SimpleNamespace(cwd=str(tmp_path)), env_type="local", effective_task_id="t",
            task_id="t", session_id="t", session_key="t", workdir=str(tmp_path),
            command_cwd=str(tmp_path), approval_note=None))["output"],
        "kanban": kanban_tools._redact(raw),
        "egress": R().redact_for_egress(raw),
    }
    for surface, out in shown.items():
        if preserve:
            assert out == raw, (surface, out)
        else:
            assert secret not in out, (surface, out)


# review6279 HF-08: the header is the END of a JSON string (object, array, nested, doubly
# encoded). A supported placeholder keeps its bytes AND the JSON closers after the real
# closing quote; value-bearing and naked-closer controls keep the base masking.
END_OF_JSON_SHAPES = {
    "object": lambda h: json.dumps({"content": h}),
    "array": lambda h: json.dumps([h]),
    "nested": lambda h: json.dumps({"message": {"content": [{"text": h}]}}),
    "double": lambda h: json.dumps(json.dumps({"content": h})),
}
END_OF_JSON_VALUES = {
    "var": ("$TOKEN", "", True),
    "positional": ("${1}", "", True),
    "printf_escape": ("%s" + BS + "n", "", True),
    "doc": ("<token>", "", True),
    "template_ws": ("{{  token  }}", "", True),
    "digit_segment": ("${SERVICE_2026_TOKEN}", "", True),
    "opaque_value": (SYN, SYN[6:-4], False),
    "default_value": ("${TOKEN:-SyntheticSecretPassword}", "SyntheticSecretPassword", False),
    "opaque_wrapper": ("<TOKEN_AbCdEfGhIjKlMnOp>", "AbCdEfGhIjKlMnOp", False),
    # material glued after a COMPLETE reference (``$TOKENAbCd…`` would be one shell name)
    "glued_suffix": ("${TOKEN}" + "AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp", False),
}


@pytest.mark.parametrize("name", sorted(END_OF_JSON_VALUES))
@pytest.mark.parametrize("shape", sorted(END_OF_JSON_SHAPES))
@pytest.mark.parametrize("family", ["Authorization: Bearer ", "x-api-key: "])
def test_end_of_json_string_contract_on_real_producer_paths(name, shape, family, tmp_path):
    from tools import file_tools, kanban_tools
    from tools.terminal_tool_result import finalize_foreground_result
    value, secret, preserve = END_OF_JSON_VALUES[name]
    raw = END_OF_JSON_SHAPES[shape](family + value)
    p = tmp_path / "probe.json"
    p.write_text(raw + "\n")
    proc = subprocess.run(["cat", str(p)], capture_output=True, text=True, check=True)
    shown = {
        "read": json.loads(file_tools.read_file_tool(str(p), task_id="t"))["content"].removeprefix("1|"),
        "search": json.loads(file_tools.search_tool("Authorization|x-api-key", path=str(p),
                                                    task_id="t"))["matches"][0]["content"],
        "terminal": json.loads(finalize_foreground_result(
            command=f"cat {p}", result={"output": proc.stdout, "returncode": proc.returncode},
            env=types.SimpleNamespace(cwd=str(tmp_path)), env_type="local", effective_task_id="t",
            task_id="t", session_id="t", session_key="t", workdir=str(tmp_path),
            command_cwd=str(tmp_path), approval_note=None))["output"].rstrip("\n"),
        "kanban": kanban_tools._redact(raw),
        "egress": R().redact_for_egress(raw),
    }
    for surface, out in shown.items():
        if preserve:
            assert out == raw, (surface, out)
            json.loads(out)
        else:
            assert secret not in out, (surface, out)


@pytest.mark.parametrize("family", ["Authorization: Bearer ", "x-api-key: "])
@pytest.mark.parametrize("value", ["$TOKEN", "${SERVICE_2026_TOKEN}", "{{  token  }}", "% s"])
def test_kanban_metadata_roundtrips_placeholder_header(family, value):
    from tools import kanban_tools
    meta = {"content": family + value}
    assert kanban_tools._redact_metadata(meta) == meta


# review6366 HF-11: an already-supported placeholder ends a NON-FINAL member/element of compact
# JSON (``separators=(",", ":")``: no whitespace after ``,``), so the next member sits in the
# same ``\S+`` run as the x-api-key slot. The exact source bytes survive on the real producer
# paths and the JSON still parses with every key; end-of-string, spaced and doubly-encoded
# forms are controls. Long compact lines (> the 128-character slot bound after the closing
# quote) behave like short ones for both header families.
def _five_real_paths(raw, tmp_path):
    from tools import file_tools, kanban_tools
    from tools.terminal_tool_result import finalize_foreground_result
    p = tmp_path / "probe.json"
    p.write_text(raw + "\n")
    proc = subprocess.run(["cat", str(p)], capture_output=True, text=True, check=True)
    return {
        "read": json.loads(file_tools.read_file_tool(str(p), task_id="t"))["content"].removeprefix("1|"),
        "search": json.loads(file_tools.search_tool("Authorization|x-api-key", path=str(p),
                                                    task_id="t"))["matches"][0]["content"],
        "terminal": json.loads(finalize_foreground_result(
            command=f"cat {p}", result={"output": proc.stdout, "returncode": proc.returncode},
            env=types.SimpleNamespace(cwd=str(tmp_path)), env_type="local", effective_task_id="t",
            task_id="t", session_id="t", session_key="t", workdir=str(tmp_path),
            command_cwd=str(tmp_path), approval_note=None))["output"].rstrip("\n"),
        "kanban": kanban_tools._redact(raw),
        "egress": R().redact_for_egress(raw),
    }


COMPACT_VALUES = ["$TOKEN", "${SERVICE_2026_TOKEN}", "%s" + BS + "n", "<token>", "{{ token }}"]
COMPACT_SHAPES = {
    "object_first": lambda h: {"content": h, "other": "PUBLIC"},
    "array_first": lambda h: [h, "PUBLIC"],
    "nested_first": lambda h: {"items": [{"content": h}, "PUBLIC"]},
    "end_control": lambda h: {"content": h},
    "typed_members": lambda h: {"content": h, "n": -1, "b": True, "z": None, "o": {"a": [1]}, "l": ["P"]},
    "nested_closers": lambda h: {"m": {"c": [{"t": h}]}, "other": "PUBLIC"},
    "long_line": lambda h: {"content": h, "pad": "P" * 300, "other": "PUBLIC"},
}
COMPACT_ENCODINGS = {
    "compact": lambda o: json.dumps(o, separators=(",", ":")),
    "double": lambda o: json.dumps(json.dumps(o, separators=(",", ":")), separators=(",", ":")),
    "spaced_control": json.dumps,
}


@pytest.mark.parametrize("encoding", sorted(COMPACT_ENCODINGS))
@pytest.mark.parametrize("shape", sorted(COMPACT_SHAPES))
@pytest.mark.parametrize("value", COMPACT_VALUES)
@pytest.mark.parametrize("family", ["Authorization: Bearer ", "x-api-key: "])
def test_hf11_compact_json_members_keep_bytes_and_keys(family, value, shape, encoding, tmp_path):
    obj = COMPACT_SHAPES[shape](family + value)
    raw = COMPACT_ENCODINGS[encoding](obj)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        decoded = json.loads(out)
        if encoding == "double":
            decoded = json.loads(decoded)
        assert decoded == obj, (surface, decoded)


# HF-11 protections: the compact boundary never exempts value material. A value-bearing slot
# in a non-final member keeps base masking; an actual-valued header in the NEXT member is
# still found and masked (the boundary resumes the scan there instead of swallowing it); a
# quote followed by anything but ``,`` + a JSON value at the same escape level is not a
# boundary.
COMPACT_PROTECTED = [
    *[(json.dumps({"content": "x-api-key: " + v, "other": "PUBLIC"}, separators=(",", ":")), secret)
      for v, secret in ((SYN, SYN[6:-4]),
                        ("${TOKEN:-SyntheticSecretPassword}", "SyntheticSecretPassword"),
                        ("<TOKEN_AbCdEfGhIjKlMnOp>", "AbCdEfGhIjKlMnOp"),
                        ("${TOKEN}AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp"),
                        ("$TOKEN" + BS + "AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp"))],
    *[(enc([first, nxt]), SYN[6:-4])
      for enc in (lambda o: json.dumps(o, separators=(",", ":")),
                  lambda o: json.dumps(json.dumps(o, separators=(",", ":"))))
      for first in ("x-api-key: $TOKEN", "x-api-key: {{ token }}", "Authorization: Bearer $TOKEN")
      for nxt in ("x-api-key: " + SYN, "Authorization: Bearer " + SYN, "X-Auth-Token: " + SYN)],
    (json.dumps({"a": "x-api-key: $TOKEN", "b": "x-api-key: " + SYN}, separators=(",", ":")), SYN[6:-4]),
    (json.dumps(["x-api-key: " + SYN, "x-api-key: $TOKEN"], separators=(",", ":")), SYN[6:-4]),
    ('curl -H "x-api-key: $TOKEN",AbCdEfGhIjKlMnOp', "AbCdEfGhIjKlMnOp"),
    ('curl -H "x-api-key: $TOKEN"AbCdEfGhIjKlMnOp', "AbCdEfGhIjKlMnOp"),
    ('x-api-key: $TOKEN",,"AbCdEfGhIjKlMnOp"', "AbCdEfGhIjKlMnOp"),
    # escape-level mismatch: ``\"`` closes, but the next string opens unescaped
    (f'x-api-key: $TOKEN{BS}","AbCdEfGhIjKlMnOp"', "AbCdEfGhIjKlMnOp"),
    (f'x-api-key: $TOKEN",{BS}"AbCdEfGhIjKlMnOp{BS}"', "AbCdEfGhIjKlMnOp"),
]


@pytest.mark.parametrize("raw,secret", COMPACT_PROTECTED)
def test_hf11_compact_boundary_keeps_value_and_adjacent_header_protection(raw, secret, tmp_path):
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert secret not in out, (surface, out)


@pytest.mark.parametrize("value", ["$TOKEN", "{{ token }}"])
def test_hf11_adjacent_actual_valued_header_masked_placeholder_kept(value, tmp_path):
    raw = json.dumps(["x-api-key: " + value, "x-api-key: " + SYN, "PUBLIC"], separators=(",", ":"))
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out.startswith(f'["x-api-key: {value}","x-api-key: '), (surface, out)
        assert SYN[6:-4] not in out, (surface, out)


@pytest.mark.parametrize("family", ["Authorization: Bearer ", "x-api-key: "])
def test_hf11_compact_metadata_comment_text_roundtrips(family):
    from tools import kanban_tools
    for value in COMPACT_VALUES:
        text = json.dumps({"content": family + value, "other": "PUBLIC"}, separators=(",", ":"))
        assert kanban_tools._redact(text) == text
        assert list(json.loads(kanban_tools._redact(text))) == ["content", "other"]


# review6471 HF-12: a quote, comma and JSON-looking token INSIDE one header string are data,
# not a member boundary -- equal escape counts on two interior quotes do not close the string.
# ``{"content":"x-api-key: $TOKEN\",\"<opaque>"}`` is ONE ``content`` string; its opaque tail
# stays masked like base on every real path, raw and JSON-encoded once/twice more. The paired
# true compact members (HF-11) keep their bytes.
HF12_SECRET = "OpaqueAbCdEfGhIjKlMnOp123456789"
HF12_VALUES = ["$TOKEN", "${TOKEN}", "<token>", "%s"]
HF12_TAILS = {
    "string": '","' + HF12_SECRET,
    "number": '",7' + HF12_SECRET,
    "brace": '",{' + HF12_SECRET + "}",
    "null": '",null,' + HF12_SECRET,
}


def _hf12_encode(obj, level):
    raw = json.dumps(obj, separators=(",", ":"))
    for _ in range(level):
        raw = json.dumps(raw)
    return raw


def _hf12_decode(raw, level):
    for _ in range(level + 1):
        raw = json.loads(raw)
    return raw


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("tail", sorted(HF12_TAILS))
@pytest.mark.parametrize("value", HF12_VALUES)
def test_hf12_escaped_quote_inside_one_string_stays_masked(value, tail, level, tmp_path):
    obj = {"content": "x-api-key: " + value + HF12_TAILS[tail]}
    raw = _hf12_encode(obj, level)
    assert _hf12_decode(raw, level) == obj  # one string holds the whole tail
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF12_SECRET not in out, (surface, out)
        assert HF12_SECRET[6:-4] not in out, (surface, out)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("shape", ["array", "object"])
@pytest.mark.parametrize("value", HF12_VALUES)
def test_hf12_paired_true_compact_members_keep_bytes(value, shape, level, tmp_path):
    h = "x-api-key: " + value
    obj = [h, "PUBLIC"] if shape == "array" else {"content": h, "other": "PUBLIC"}
    raw = _hf12_encode(obj, level)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        assert _hf12_decode(out, level) == obj, (surface, out)


# Escaped quotes earlier in the SAME string (a quoted word) do not open a string the header
# could close; a member boundary needs the string that is actually open at the header.
@pytest.mark.parametrize("level", [0, 1, 2])
def test_hf12_earlier_escaped_quotes_do_not_make_a_boundary(level, tmp_path):
    obj = {"content": 'echo "a" x-api-key: $TOKEN","' + HF12_SECRET + '"'}
    raw = _hf12_encode(obj, level)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF12_SECRET[6:-4] not in out, (surface, out)


def test_hf12_string_state_resets_per_line():
    from tools import kanban_tools
    member = json.dumps({"content": "x-api-key: $TOKEN", "other": "PUBLIC"}, separators=(",", ":"))
    text = 'it is 5" tall\n' + member
    assert R().redact_for_egress(text) == text
    assert kanban_tools._redact(text) == text


# review6556 HF-12 (extended) / HF-13: a data quote EARLIER in the same string
# (``5" tall x-api-key: …``) is not the opening of a nested string. Quotes one escape level
# deeper than a string's own delimiters delimit nested strings only when that string's decoded
# content is itself a JSON document. So the opaque tail after ``\",\"`` stays masked (HF-12),
# and the genuine closing quote still ends the header string, keeping the adjacent member
# byte-for-byte (HF-13), at every encoding depth.
HF13_PREFIXES = ['5" tall ', 'echo " ']


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("tail", sorted(HF12_TAILS))
@pytest.mark.parametrize("value", HF12_VALUES)
def test_hf12_data_quote_prefix_tail_stays_masked(value, tail, level, tmp_path):
    obj = {"content": '5" tall x-api-key: ' + value + HF12_TAILS[tail]}
    raw = _hf12_encode(obj, level)
    assert _hf12_decode(raw, level) == obj  # one string holds the whole tail
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF12_SECRET[6:-4] not in out, (surface, out)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("tail", sorted(HF12_TAILS))
def test_hf12_unbalanced_data_quote_prefix_tail_stays_masked(tail, level, tmp_path):
    obj = {"content": 'echo " x-api-key: $TOKEN' + HF12_TAILS[tail]}
    raw = _hf12_encode(obj, level)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF12_SECRET[6:-4] not in out, (surface, out)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("shape", ["array", "object"])
@pytest.mark.parametrize("value", HF12_VALUES)
@pytest.mark.parametrize("prefix", HF13_PREFIXES)
def test_hf13_data_quote_prefix_true_member_keeps_bytes(prefix, value, shape, level, tmp_path):
    h = prefix + "x-api-key: " + value
    obj = [h, "PUBLIC"] if shape == "array" else {"content": h, "other": "PUBLIC"}
    raw = _hf12_encode(obj, level)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        assert _hf12_decode(out, level) == obj, (surface, out)


# A member whose string value is an encoded JSON document: its inner members are real (kept),
# while a data quote inside the inner header string is still data (masked).
@pytest.mark.parametrize("value", ["$TOKEN", "<token>"])
def test_hf13_embedded_json_document_member(value, tmp_path):
    inner_ok = json.dumps({"h": "x-api-key: " + value, "o": "PUBLIC"}, separators=(",", ":"))
    raw = json.dumps({"cmd": inner_ok, "n": 1}, separators=(",", ":"))
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
    inner_bad = json.dumps({"h": '5" x-api-key: ' + value + '","' + HF12_SECRET}, separators=(",", ":"))
    raw = json.dumps({"cmd": inner_bad, "n": 1}, separators=(",", ":"))
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF12_SECRET[6:-4] not in out, (surface, out)


# review6640 HF-12 (extended) / HF-14: the member boundary needs a real JSON container.
# A ``,`` after a placeholder string is a member boundary only inside a well-formed JSON
# array/object open at that comma (checked at its own escape level). So quote/comma data in
# ONE single-quoted source string never splits it (HF-12), while a JSON literal inside source
# code, a shell command, a numbered or commented line keeps its members raw and inside one or
# two history-JSON wrappers (HF-14), without the surrounding text having to be JSON.
HF14_SECRET = "SyntheticBoundaryOpaque0123456789AbCdEf"


def _hf14_wrap(source, encoding):
    raw = source
    for _ in range(encoding):
        raw = json.dumps({"message": {"content": raw}}, separators=(",", ":"))
    return raw


def _hf14_unwrap(raw, encoding):
    for _ in range(encoding):
        raw = json.loads(raw)["message"]["content"]
    return raw


HF14_SOURCES = {
    "python": lambda h: 'headers = ["' + h + '","Accept: application/json"]',
    "javascript": lambda h: 'const headers = ["' + h + '","Accept: application/json"];',
    "shell": lambda h: "printf '%s\\n' '" + json.dumps({"content": h, "other": "PUBLIC"}, separators=(",", ":")) + "'",
    "numbered": lambda h: "1|" + json.dumps({"content": h, "other": "PUBLIC"}, separators=(",", ":")),
    "commented": lambda h: json.dumps({"content": h, "other": "PUBLIC"}, separators=(",", ":")) + " # public example",
    "pure_json": lambda h: json.dumps({"content": h, "other": "PUBLIC"}, separators=(",", ":")),
}


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("value", HF12_VALUES)
def test_hf12_single_quoted_source_data_quotes_stay_masked(value, encoding, tmp_path):
    text = '5" tall x-api-key: ' + value + '","' + HF14_SECRET
    source = "note = " + repr(text)
    raw = _hf14_wrap(source, encoding)
    assert _hf14_unwrap(raw, encoding) == source
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# Quote/comma data that only LOOKS like members (no well-formed container) stays masked too.
@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("source", [
    "note = 'a,\"x-api-key: $TOKEN\",\"" + HF14_SECRET + "\"'",
    json.dumps(["P", ':{"x-api-key: %s",null,' + HF14_SECRET], separators=(",", ":")),
    json.dumps(["P", '["x-api-key: $TOKEN","' + HF14_SECRET], separators=(",", ":")),
])
def test_hf12_member_lookalike_without_json_container_stays_masked(source, encoding, tmp_path):
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("value", HF12_VALUES)
@pytest.mark.parametrize("kind", sorted(HF14_SOURCES))
def test_hf14_json_literal_in_source_keeps_bytes(kind, value, encoding, tmp_path):
    source = HF14_SOURCES[kind]("x-api-key: " + value)
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        assert _hf14_unwrap(out, encoding) == source, (surface, out)


# review6751 HF-15: raw offsets map into a decoded context exactly as the decoder decodes. A
# ``\ud83d\ude00`` surrogate-pair escape (any supplementary character in ASCII-escaped history
# JSON) is ONE decoded character; counting it as two shifted the container lookup right, onto an
# unrelated later ``[]``/``{}``, which then "validated" a boundary inside one Python string and
# exposed the opaque suffix. The container must also OWN the comma (the decoder's member walk
# stops at it), so no other parseable container can vouch for it. BMP/ASCII prefixes and the
# non-container ``XX`` suffix are paired controls; ``12|`` is a numbered-line gutter.
HF15_CHARS = ["A", "é", "😀", "é😀A"]


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("suffix", ["[]", "{}", "XX"])
@pytest.mark.parametrize("gutter", ["", "     12|"])
@pytest.mark.parametrize("char", HF15_CHARS)
def test_hf15_unrelated_later_container_never_owns_the_boundary(char, gutter, suffix, encoding, tmp_path):
    fragment = '["x-api-key: %s","' + HF14_SECRET
    source = gutter + "note = " + repr(char * len(fragment) + fragment + suffix)
    raw = _hf14_wrap(source, encoding)
    assert _hf14_unwrap(raw, encoding) == source
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# The same mapping must keep legitimate Unicode-bearing source exact: a supplementary character
# before the JSON literal, inside the header string, or ``\u``-escaped inside the literal itself.
HF15_SOURCES = {
    "assignment": lambda c, h: "note = " + json.dumps(c, ensure_ascii=False) + "; rows = "
    + json.dumps([h, "PUBLIC"], separators=(",", ":")),
    "in_header_string": lambda c, h: "rows = " + json.dumps([c + " " + h, "PUBLIC"], ensure_ascii=False,
                                                            separators=(",", ":")),
    "escaped_in_literal": lambda c, h: "rows = " + json.dumps([c, h, "PUBLIC"], separators=(",", ":")),
}


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("char", HF15_CHARS)
@pytest.mark.parametrize("kind", sorted(HF15_SOURCES))
def test_hf15_unicode_source_keeps_bytes(kind, char, encoding, tmp_path):
    source = HF15_SOURCES[kind](char, "x-api-key: %s")
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        assert _hf14_unwrap(out, encoding) == source, (surface, out)


# An actual-valued header right after a kept placeholder behind Unicode is still masked.
@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("char", HF15_CHARS)
def test_hf15_unicode_adjacent_actual_valued_header_masked(char, encoding, tmp_path):
    source = "note = " + json.dumps(char, ensure_ascii=False) + "; rows = " + json.dumps(
        ["x-api-key: %s", "x-api-key: " + HF14_SECRET, "PUBLIC"], separators=(",", ":"))
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert "x-api-key: %s" in out, (surface, out)
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# review6751 HF-16: a data quote after ``:`` (or ``[``/``,``/``{"k":``) INSIDE a valid JSON
# string is not a nested string. The header string's real closing quote closes it -- and any
# string the scan opened deeper -- so the member boundary is kept and every byte survives.
# Paired protection: the same prefixes with an opaque tail in the SAME string stay masked.
HF16_PREFIXES = ["", ":", ':"', 'He said: "', '["', ',"', '{"k":"']
HF16_SHAPES = {
    "array": lambda h: [h, 0],
    "object": lambda h: {"content": h, "other": "PUBLIC"},
}


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("shape", sorted(HF16_SHAPES))
@pytest.mark.parametrize("value", HF12_VALUES + ["{{ token }}"])
@pytest.mark.parametrize("prefix", HF16_PREFIXES)
def test_hf16_data_quote_inside_header_string_keeps_bytes(prefix, value, shape, encoding, tmp_path):
    obj = HF16_SHAPES[shape](prefix + "x-api-key: " + value)
    source = json.dumps(obj, separators=(",", ":"))
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert out == raw, (surface, out)
        assert json.loads(_hf14_unwrap(out, encoding)) == obj, (surface, out)


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("value", ["$TOKEN", "%s"])
@pytest.mark.parametrize("prefix", HF16_PREFIXES)
def test_hf16_data_quote_prefix_opaque_tail_stays_masked(prefix, value, encoding, tmp_path):
    obj = [prefix + "x-api-key: " + value + '","' + HF14_SECRET, 0]
    raw = _hf14_wrap(json.dumps(obj, separators=(",", ":")), encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# review6751 HF-17: JSON numbers are validated lexically, never converted, so a public integer
# longer than Python's int-conversion limit (4300 digits) no longer makes a well-formed
# container look malformed. read_file/search_files shorten such long lines BEFORE redaction,
# so the redactor never sees the complete container there (producer behaviour, excluded by
# review6751 too); the three complete-output paths keep every byte. 4300 digits is the control.
HF17_NUMBERS = {
    "int4300": "1" * 4300,
    "int4301": "1" * 4301,
    "neg5000": "-" + "9" * 5000,
    "exp4301": "1" * 4301 + "e5",
    "frac5000": "0." + "0" * 5000 + "1",
}
HF17_SHAPES = {
    "after": lambda n: '["x-api-key: %s",' + n + "]",
    "before": lambda n: "[" + n + ',"x-api-key: %s","PUBLIC"]',
}


@pytest.mark.parametrize("encoding", [0, 1, 2])
@pytest.mark.parametrize("shape", sorted(HF17_SHAPES))
@pytest.mark.parametrize("number", sorted(HF17_NUMBERS))
def test_hf17_long_public_number_keeps_placeholder(number, shape, encoding, tmp_path):
    source = HF17_SHAPES[shape](HF17_NUMBERS[number])
    assert HF17_NUMBERS[number] in json.loads(source, parse_int=str, parse_float=str)
    raw = _hf14_wrap(source, encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        if surface in ("read", "search"):
            continue  # the producer shortened the line before redaction (see above)
        assert out == raw, (surface, out[:200])
        assert _hf14_unwrap(out, encoding) == source, surface


@pytest.mark.parametrize("encoding", [0, 1, 2])
def test_hf17_long_number_inside_one_string_opaque_tail_stays_masked(encoding, tmp_path):
    obj = {"content": "x-api-key: %s\",1" + "1" * 4301 + ',"' + HF14_SECRET}
    raw = _hf14_wrap(json.dumps(obj, separators=(",", ":")), encoding)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# review6810 HF-18: a newline ends every JSON string but not a container -- JSON allows newlines
# between members. A complete multi-line array/object holding a placeholder member and a compact
# adjacent public member keeps every byte on the complete-output paths (read_file with its
# ``N|`` gutters removed for comparison, terminal, Kanban, egress); search_files returns the
# matching line alone, without its container, so it is checked for protection only. Value-
# bearing slots, same-string tails and adjacent actual-valued headers stay masked on all five.
HF18_VALUES = HF12_VALUES + ["{{ token }}", "<YOUR_API_TOKEN>"]


def _hf18_split(compact):
    return compact[0] + "\n" + compact[1:-1] + "\n" + compact[-1]


HF18_SOURCES = {
    "array_split": lambda h: _hf18_split(json.dumps([h, 0], separators=(",", ":"))),
    "object_split": lambda h: _hf18_split(json.dumps({"v": h, "n": 0}, separators=(",", ":"))),
    "nested": lambda h: '{\n"h":["' + h + '","PUBLIC"],\n"o":{"k":[1,\n2]}\n}',
    "indented": lambda h: '{\n  "items": [\n    {"h":"' + h + '","n":-1},\n    "PUBLIC"\n  ]\n}',
    "crlf": lambda h: _hf18_split(json.dumps([h, 0], separators=(",", ":"))).replace("\n", "\r\n"),
    "python_literal": lambda h: 'headers = [\n    "' + h + '","Accept: application/json"\n]',
}


def _hf18_read(tmp_path, raw):
    from tools import file_tools
    p = tmp_path / "multiline.txt"
    p.write_text(raw + "\n", newline="")
    content = json.loads(file_tools.read_file_tool(str(p), task_id="t"))["content"]
    return re.sub(r"^\s*\d+\|", "", content, flags=re.MULTILINE)


def _hf18_complete_paths(raw, tmp_path):
    """Displayed text per complete-output path. ``terminal`` is fed ``cat`` output captured in
    text mode, so ``\r\n`` arrives as ``\n``; ``_hf18_expected`` accounts for that."""
    from tools import kanban_tools
    from tools.terminal_tool_result import finalize_foreground_result
    p = tmp_path / "probe.txt"
    p.write_text(raw + "\n", newline="")
    proc = subprocess.run(["cat", str(p)], capture_output=True, text=True, check=True)
    return {
        "read": _hf18_read(tmp_path, raw),
        "terminal": json.loads(finalize_foreground_result(
            command=f"cat {p}", result={"output": proc.stdout, "returncode": proc.returncode},
            env=types.SimpleNamespace(cwd=str(tmp_path)), env_type="local", effective_task_id="t",
            task_id="t", session_id="t", session_key="t", workdir=str(tmp_path),
            command_cwd=str(tmp_path), approval_note=None))["output"].rstrip("\n"),
        "kanban": kanban_tools._redact(raw),
        "egress": R().redact_for_egress(raw),
    }


def _hf18_expected(source, surface):
    return source.replace("\r\n", "\n") if surface == "terminal" else source


@pytest.mark.parametrize("value", HF18_VALUES)
@pytest.mark.parametrize("kind", sorted(HF18_SOURCES))
def test_hf18_multiline_json_keeps_bytes(kind, value, tmp_path):
    from tools import kanban_tools
    source = HF18_SOURCES[kind]("x-api-key: " + value)
    for surface, out in _hf18_complete_paths(source, tmp_path).items():
        assert out == _hf18_expected(source, surface), (surface, out)
        if kind != "python_literal":
            assert json.loads(out) == json.loads(source), surface
    # metadata (encoded) and one history-JSON wrapper stay exact too
    assert kanban_tools._redact_metadata({"content": source}) == {"content": source}
    wrapped = _hf14_wrap(source, 1)
    for surface, out in _five_real_paths(wrapped, tmp_path).items():
        assert out == wrapped, (surface, out)


HF18_PROTECTED = {
    "opaque": "x-api-key: " + HF14_SECRET,
    "default": "x-api-key: ${TOKEN:-" + HF14_SECRET + "}",
    "glued": "x-api-key: ${TOKEN}" + HF14_SECRET,
    "same_string_tail": 'x-api-key: %s","' + HF14_SECRET,
    "data_quote_tail": ':"x-api-key: $TOKEN","' + HF14_SECRET,
}


@pytest.mark.parametrize("shape", ["array_split", "object_split", "nested"])
@pytest.mark.parametrize("slot", sorted(HF18_PROTECTED))
def test_hf18_multiline_value_bearing_slots_stay_masked(slot, shape, tmp_path):
    from tools import kanban_tools
    obj_text = HF18_PROTECTED[slot]
    source = {
        "array_split": lambda: _hf18_split(json.dumps([obj_text, 0], separators=(",", ":"))),
        "object_split": lambda: _hf18_split(json.dumps({"v": obj_text, "n": 0}, separators=(",", ":"))),
        "nested": lambda: '{\n"h":[' + json.dumps(obj_text) + ',"PUBLIC"],\n"o":1\n}',
    }[shape]()
    json.loads(source)
    for surface, out in _five_real_paths(source, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)
    assert HF14_SECRET[6:-4] not in json.dumps(kanban_tools._redact_metadata({"content": source}))


# An earlier unclosed ``[``/``{`` in prose never vouches for a later line.
@pytest.mark.parametrize("source", [
    'See [the docs\nnote: "x-api-key: %s","' + HF14_SECRET + '"',
    '{\nnote = "x-api-key: $TOKEN","' + HF14_SECRET + '"\n',
    '[\n"x-api-key: %s","' + HF14_SECRET,
])
def test_hf18_multiline_lookalikes_stay_masked(source, tmp_path):
    for surface, out in _five_real_paths(source, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# An adjacent actual-valued header member of a multi-line container is masked on its own while
# the placeholder member keeps its bytes. (search_files shows the line without its container;
# there the placeholder slot gets base masking, which -- as in base, r11 and r12 -- runs to the
# next whitespace and so also covers the NEXT header's name; that inherited line-only behaviour
# is outside this container repair and is not asserted here.)
@pytest.mark.parametrize("next_header", ["x-api-key: ", "Authorization: Bearer ", "X-Auth-Token: "])
@pytest.mark.parametrize("value", ["%s", "{{ token }}"])
def test_hf18_multiline_adjacent_actual_valued_header_masked(value, next_header, tmp_path):
    source = '[\n"x-api-key: ' + value + '","' + next_header + HF14_SECRET + '",0\n]'
    for surface, out in _hf18_complete_paths(source, tmp_path).items():
        assert out.startswith('[\n"x-api-key: ' + value + '","'), (surface, out)
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# review6900 LG01 (HF-18 extent): the numbered read_file RESULT is itself serialized -- the
# tool's exact JSON response string, and that string inside a history record -- and then read,
# searched, printed, posted or forwarded again. Its gutters now sit inside a JSON string after
# escaped newlines. Composition through the actual producers keeps every byte: the real
# read_file_tool response (plain ``N|`` gutters) and the same response with space-padded
# ``     N|`` gutters, raw and in one history record, on all five real paths. (Unicode-digit
# gutters are not covered: no producer emits them.)
HF18C_SHAPES = {
    "array": lambda h: [h, 0],
    "object": lambda h: {"v": h, "n": 0},
}


def _hf18c_reader_response(obj, tmp_path, gutter):
    from tools import file_tools
    compact = json.dumps(obj, separators=(",", ":"))
    source = _hf18_split(compact)
    p = tmp_path / "source.json"
    p.write_text(source + "\n")
    response = file_tools.read_file_tool(str(p), task_id="t")
    numbered = json.loads(response)["content"]
    assert re.sub(r"^\d+\|", "", numbered, flags=re.MULTILINE) == source  # first read exact
    if gutter == "padded":
        envelope = json.loads(response)
        envelope["content"] = re.sub(r"^(\d+)\|", lambda m: f"{int(m[1]):>6}|", numbered,
                                     flags=re.MULTILINE)
        response = json.dumps(envelope, ensure_ascii=False)
    return source, response


def _hf18c_history(response):
    return json.dumps({"role": "tool", "content": response})


@pytest.mark.parametrize("wrapped", ["response", "history"])
@pytest.mark.parametrize("gutter", ["plain", "padded"])
@pytest.mark.parametrize("shape", sorted(HF18C_SHAPES))
@pytest.mark.parametrize("value", ["%s", "$TOKEN", "${TOKEN}", "{{ token }}", "<YOUR_API_TOKEN>"])
def test_hf18_serialized_numbered_reader_result_keeps_bytes(value, shape, gutter, wrapped, tmp_path):
    obj = HF18C_SHAPES[shape]("x-api-key: " + value)
    source, response = _hf18c_reader_response(obj, tmp_path, gutter)
    raw = response if wrapped == "response" else _hf18c_history(response)
    (tmp_path / "replay").mkdir()
    for surface, out in _five_real_paths(raw, tmp_path / "replay").items():
        assert out == raw, (surface, out)
        decoded = json.loads(out)
        if wrapped == "history":
            decoded = json.loads(decoded["content"])
        lines = re.sub(r"^ *\d+\|", "", decoded["content"], flags=re.MULTILINE)
        assert lines == source and json.loads(lines) == obj, surface


# Value-bearing slots, same-string tails and adjacent actual-valued headers in the same
# serialized numbered form stay masked on all five paths. The numbered text is built the way
# the reader renders it (the real reader would already mask these on its first pass).
HF18C_PROTECTED = {
    "opaque": ["x-api-key: " + HF14_SECRET, 0],
    "default": ["x-api-key: ${TOKEN:-" + HF14_SECRET + "}", 0],
    "glued": ["x-api-key: ${TOKEN}" + HF14_SECRET, 0],
    "same_string_tail": ['x-api-key: %s","' + HF14_SECRET, 0],
    "data_quote_tail": [':"x-api-key: $TOKEN","' + HF14_SECRET, 0],
    "adjacent_x_api_key": ["x-api-key: %s", "x-api-key: " + HF14_SECRET, 0],
    "adjacent_auth_token": ["x-api-key: %s", "X-Auth-Token: " + HF14_SECRET, 0],
    "adjacent_bearer": ["x-api-key: {{ token }}", "Authorization: Bearer " + HF14_SECRET, 0],
}


@pytest.mark.parametrize("wrapped", ["response", "history"])
@pytest.mark.parametrize("gutter", ["plain", "padded"])
@pytest.mark.parametrize("slot", sorted(HF18C_PROTECTED))
def test_hf18_serialized_numbered_value_bearing_slots_stay_masked(slot, gutter, wrapped, tmp_path):
    source = _hf18_split(json.dumps(HF18C_PROTECTED[slot], separators=(",", ":")))
    numbered = "\n".join((f"{i:>6}|" if gutter == "padded" else f"{i}|") + line
                         for i, line in enumerate(source.split("\n"), 1))
    response = json.dumps({"content": numbered, "total_lines": 3}, ensure_ascii=False)
    raw = response if wrapped == "response" else _hf18c_history(response)
    for surface, out in _five_real_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# review7078 HF7078-01: a PRETTY-printed JSON file (``json.dumps(..., indent=2)``) survives the
# first real read_file_tool call, but its numbered response carries escaped whitespace -- and a
# gutter after each escaped newline -- between the header string's closing quote, the member
# ``,`` and the next member (``\",\n3|  \"next\"``). Replaying that exact response (raw, or in
# one or two history records) through the five real paths keeps every byte and decodes back to
# the original JSON. The literal fixture is the reviewer's pretty-producer.py source.
HF7078_FIXTURE = json.dumps(["X-Api-Key: ${API_KEY}", "next"], indent=2)
HF7078_PLACEHOLDERS = ["${API_KEY}", "${TOKEN}", "%s"]
HF7078_SHAPES = {
    "array_string": lambda h: [h, "next"],
    "array_number": lambda h: [h, 0],
    "object_string": lambda h: {"h": h, "next": "PUBLIC"},
    "object_number": lambda h: {"h": h, "n": 7},
    "nested_first": lambda h: {"outer": {"headers": [h, "next"], "n": 1}, "z": None},
    "nested_last": lambda h: {"outer": [{"k": 1, "h": h}], "after": "PUBLIC"},
}
HF7078_INDENTS = {"i1": 1, "i2": 2, "i4": 4, "tab": "\t"}
HF7078_WRAPS = {
    "response": lambda s: s,
    "history": lambda s: json.dumps({"role": "tool", "content": s}),
    "history2": lambda s: json.dumps({"role": "tool",
                                      "content": json.dumps({"role": "tool", "content": s})}),
}


def _hf7078_search(p):
    from tools import file_tools
    return json.loads(file_tools.search_tool("X-Api-Key|X-Auth-Token|Authorization", path=str(p),
                                             task_id="t"))["matches"][0]["content"]


def _hf7078_search_shown_unredacted(raw, tmp_path):
    """What search_files itself shows for ``raw`` with redaction off: it shortens a long
    matching line (> 500 characters) before redaction, so that, not ``raw``, is exact there."""
    r = R()
    p = tmp_path / "plain.json"
    p.write_text(raw + "\n")
    enabled = r._REDACT_ENABLED
    r._REDACT_ENABLED = False
    try:
        return _hf7078_search(p)
    finally:
        r._REDACT_ENABLED = enabled


def _hf7078_paths(raw, tmp_path):
    """``_five_real_paths`` with a search pattern matching the ``X-Api-Key`` spelling."""
    from tools import file_tools, kanban_tools
    from tools.terminal_tool_result import finalize_foreground_result
    p = tmp_path / "probe.json"
    p.write_text(raw + "\n")
    proc = subprocess.run(["cat", str(p)], capture_output=True, text=True, check=True)
    return {
        "read": json.loads(file_tools.read_file_tool(str(p), task_id="t"))["content"].removeprefix("1|"),
        "search": _hf7078_search(p),
        "terminal": json.loads(finalize_foreground_result(
            command=f"cat {p}", result={"output": proc.stdout, "returncode": proc.returncode},
            env=types.SimpleNamespace(cwd=str(tmp_path)), env_type="local", effective_task_id="t",
            task_id="t", session_id="t", session_key="t", workdir=str(tmp_path),
            command_cwd=str(tmp_path), approval_note=None))["output"].rstrip("\n"),
        "kanban": kanban_tools._redact(raw),
        "egress": R().redact_for_egress(raw),
    }


def _hf7078_unwrap(raw, wrapped):
    for _ in range({"response": 0, "history": 1, "history2": 2}[wrapped]):
        raw = json.loads(raw)["content"]
    return raw


def _hf7078_replay_exact(source, wrapped, tmp_path):
    from tools import file_tools
    p = tmp_path / "pretty.json"
    p.write_text(source + "\n")
    response = file_tools.read_file_tool(str(p), task_id="t")
    assert re.sub(r"^\d+\|", "", json.loads(response)["content"], flags=re.MULTILINE) == source
    raw = HF7078_WRAPS[wrapped](response)
    (tmp_path / "replay").mkdir()
    shown_search = _hf7078_search_shown_unredacted(raw, tmp_path)
    for surface, out in _hf7078_paths(raw, tmp_path / "replay").items():
        if surface == "search" and len(raw) > 500:
            assert out == shown_search, (surface, out)  # producer-shortened line, unchanged
            continue
        assert out == raw, (surface, out)
        content = json.loads(_hf7078_unwrap(out, wrapped))["content"]
        assert re.sub(r"^\d+\|", "", content, flags=re.MULTILINE) == source, surface


@pytest.mark.parametrize("wrapped", sorted(HF7078_WRAPS))
def test_hf7078_reviewer_pretty_fixture_replays_exactly(wrapped, tmp_path):
    _hf7078_replay_exact(HF7078_FIXTURE, wrapped, tmp_path)


@pytest.mark.parametrize("wrapped", sorted(HF7078_WRAPS))
@pytest.mark.parametrize("indent", sorted(HF7078_INDENTS))
@pytest.mark.parametrize("placeholder", HF7078_PLACEHOLDERS)
@pytest.mark.parametrize("shape", sorted(HF7078_SHAPES))
def test_hf7078_pretty_reader_response_replays_exactly(shape, placeholder, indent, wrapped, tmp_path):
    obj = HF7078_SHAPES[shape]("X-Api-Key: " + placeholder)
    source = json.dumps(obj, indent=HF7078_INDENTS[indent])
    _hf7078_replay_exact(source, wrapped, tmp_path)
    assert json.loads(source) == obj


# Protection controls in the same pretty numbered-response form (built the way the reader
# renders it; the real reader would already mask these): actual values, value-bearing slots,
# same-string tails and adjacent actual-valued headers stay masked on all five paths.
HF7078_PROTECTED = {
    "opaque": lambda: ["X-Api-Key: " + HF14_SECRET, "next"],
    "default": lambda: ["X-Api-Key: ${API_KEY:-" + HF14_SECRET + "}", "next"],
    "glued": lambda: ["X-Api-Key: ${API_KEY}" + HF14_SECRET, "next"],
    "same_string_tail": lambda: ['X-Api-Key: ${API_KEY}","' + HF14_SECRET, "next"],
    "same_string_escaped_ws_tail": lambda: ['X-Api-Key: ${API_KEY}",\n\t"' + HF14_SECRET, "next"],
    "data_quote_tail": lambda: [':"X-Api-Key: %s","' + HF14_SECRET, 0],
    "adjacent_x_api_key": lambda: ["X-Api-Key: ${API_KEY}", "X-Api-Key: " + HF14_SECRET],
    "adjacent_auth_token": lambda: ["X-Api-Key: %s", "X-Auth-Token: " + HF14_SECRET, 0],
    "adjacent_bearer": lambda: {"h": "X-Api-Key: ${TOKEN}", "a": "Authorization: Bearer " + HF14_SECRET},
}


def _hf7078_numbered_response(source):
    numbered = "\n".join(f"{i}|{line}" for i, line in enumerate(source.split("\n"), 1))
    return json.dumps({"content": numbered, "total_lines": source.count("\n") + 1}, ensure_ascii=False)


@pytest.mark.parametrize("wrapped", ["response", "history"])
@pytest.mark.parametrize("indent", ["i2", "tab"])
@pytest.mark.parametrize("slot", sorted(HF7078_PROTECTED))
def test_hf7078_pretty_value_bearing_slots_stay_masked(slot, indent, wrapped, tmp_path):
    source = json.dumps(HF7078_PROTECTED[slot](), indent=HF7078_INDENTS[indent])
    raw = HF7078_WRAPS[wrapped](_hf7078_numbered_response(source))
    for surface, out in _hf7078_paths(raw, tmp_path).items():
        assert HF14_SECRET[6:-4] not in out, (surface, out)


# Vault precedence: a registered (fabricated) vault value never survives, even where it sits in
# a pretty-JSON placeholder slot that would otherwise be kept -- on the first real read and on
# every replay path of the numbered response.
@pytest.mark.parametrize("wrapped", ["response", "history"])
def test_hf7078_vault_value_in_pretty_placeholder_slot_never_survives(wrapped, tmp_path):
    from tools import file_tools
    r = R()
    vault = "${VAULTSYNTH_PRETTY_KEY}"
    source = json.dumps(["X-Api-Key: " + vault, "next"], indent=2)
    r.register_vault_redaction_value(vault)
    try:
        p = tmp_path / "vault.json"
        p.write_text(source + "\n")
        assert "VAULTSYNTH_PRETTY_KEY" not in file_tools.read_file_tool(str(p), task_id="t")
        raw = HF7078_WRAPS[wrapped](_hf7078_numbered_response(source))
        (tmp_path / "replay").mkdir()
        for surface, out in _hf7078_paths(raw, tmp_path / "replay").items():
            assert "VAULTSYNTH_PRETTY_KEY" not in out, (surface, out)
    finally:
        r.clear_vault_redaction_values()
