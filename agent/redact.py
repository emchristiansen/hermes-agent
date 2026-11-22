"""Regex-based secret redaction for logs and tool output.

Short tokens (< 18 chars) are fully masked; longer ones keep the first 6 and
last 4 characters for debuggability.
"""

import bisect
import copy
import json
import logging
import os
import re
import shlex
import threading
import unicodedata
from urllib.parse import unquote_plus

# Shared with agent/file_safety's read-block list so the two defenses can't
# drift: a blocked file_tools read that falls back to ``cat`` is still caught.
from agent.file_safety import _BLOCKED_PROJECT_ENV_BASENAMES as _ENV_FILE_BASENAMES
from agent.redact_command_segments import _command_segments
from agent.redact_connstr import _KV_CONNSTR_GATE_RE, _redact_kv_connection_strings
from agent.redact_control_split import mask_control_split_tokens
from agent.redact_discord import _redact_discord_tokens
from agent.redact_headers import (
    _COOKIE_HEADER_JSON_RE, _COOKIE_HEADER_JSON_UNTERMINATED_RE, _SECRET_HEADER_JSON_RE,
    _SECRET_HEADER_JSON_UNTERMINATED_RE, _SECRET_HEADER_NAMES, json_header_value_is_credential,
    redact_cookie_assignments, redact_header_credentials,
)
from agent.redact_pattern_shape import _pattern_structure
# Signed quantities and joined identifiers shaped like E.164 phone numbers are told apart there.
from agent.redact_quantities import non_phone_offsets
from agent.redact_spaced_cfg import redact_spaced_cfg_assignments
from agent.redact_zhipu import _ZHIPU_API_KEY_RE

logger = logging.getLogger(__name__)

# Browser and CDP results can be a page-controlled JSON tree. The walkers stop
# at this depth and replace the remainder, so a nested payload cannot raise
# RecursionError on the way to the model.
REDACT_WALK_MAX_DEPTH = 320
REDACT_WALK_DEPTH_MARKER = "[redacted: nesting limit]"

# ---------------------------------------------------------------------------
# Vault-value redaction registry (profile-scoped, bounded)
# ---------------------------------------------------------------------------
# Exact secret values that transited a server-side vault fill (browser_vault). Generic
# credential-shaped regexes cannot catch an arbitrary user password, so the fill path registers
# the exact bytes and every browser_* tool result (including browser_cdp Runtime.evaluate
# passthrough) is scrubbed against them before it can reach the model. Memory only: never
# persisted or logged. Keyed by profile home so a multiplex gateway never scrubs profile B's
# output with profile A's passwords (which would also confirm to B that the bytes exist), and
# bounded per profile: a fill-heavy session evicts its oldest entries rather than growing forever.
_VAULT_REDACTION_MAX_PER_PROFILE = 64
_VAULT_REDACTION_VALUES: dict = {}  # profile home → ordered {value: None}
_VAULT_REDACTION_LOCK = threading.Lock()


def _vault_scope() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def register_vault_redaction_value(value) -> None:
    """Register an exact vault secret value for model-facing redaction.

    Called by the vault fill path BEFORE the injection happens, so no later browser tool result
    can echo the value back into model context. Also registers the form a text input normalizes
    it to (CR/LF stripped), since that is what the page holds.
    """
    if not isinstance(value, str) or not value:
        return
    normalized = value.replace("\r", "").replace("\n", "")
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.setdefault(_vault_scope(), {})
        for v in (value, normalized):
            if v:
                bucket.pop(v, None)  # re-registering refreshes recency
                bucket[v] = None
        while len(bucket) > _VAULT_REDACTION_MAX_PER_PROFILE:
            del bucket[next(iter(bucket))]


def clear_vault_redaction_values() -> None:
    """Drop the current profile's registered values (profile teardown / explicit lock)."""
    with _VAULT_REDACTION_LOCK:
        _VAULT_REDACTION_VALUES.pop(_vault_scope(), None)


def redact_registered_vault_values(text: str) -> str:
    """Exact-substring scrub of every vault secret value registered for the current profile."""
    if not isinstance(text, str) or not text:
        return text
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.get(_vault_scope())
        values = sorted(bucket, key=len, reverse=True) if bucket else ()  # longest first: a substring never shadows its superstring
    for value in values:
        if value in text:
            text = text.replace(value, "«redacted-vault-secret»")
    return text

# Sensitive query-string param names (case-insensitive): opaque tokens / OAuth
# codes / pre-signed signatures with no vendor prefix.
# Ported from nearai/ironclaw#2529 — catches tokens whose values don't match any known vendor prefix regex
# (e.g. opaque tokens, short OAuth codes).
_SENSITIVE_QUERY_PARAMS = frozenset({
    "access_token", "refresh_token", "id_token", "token", "api_key", "apikey",
    "client_secret", "password", "auth", "jwt", "session", "secret", "key",
    "code", "signature", "x-amz-signature",
    "x-goog-signature", "sig",  # GCS V4 signed URLs, Azure SAS tokens
    "usertoken",  # Penpot MCP: the key can only travel as ``?userToken=`` (#133713)
})

# Snapshot at import time so runtime env mutations (e.g. an LLM-generated
# `export HERMES_REDACT_SECRETS=false`) cannot disable redaction mid-session.
# ON by default; `security.redact_secrets: false` bridges to this env var.
# ON by default — secure default per issue #17691. Users who need raw credential values in tool output (e.g.
# working on the redactor itself) can opt out via `security.redact_secrets: false` in config.yaml (bridged
# to this env var in hermes_cli/main.py, gateway/run.py, and cli.py) or `HERMES_REDACT_SECRETS=false` in
# ~/.hermes/.env. An opt-out warning is logged at gateway and CLI startup so operators see the downgrade —
# see `_log_redaction_status()` in gateway/run.py and cli.py.
_REDACT_ENABLED = os.getenv("HERMES_REDACT_SECRETS", "true").lower() in {"1", "true", "yes", "on"}

# Routed multiplex profiles: the import-time snapshot above is the LAUNCH profile's policy. A profile
# served under a HERMES_HOME override follows its own ``security.redact_secrets`` (its ``.env`` value
# first, like the standalone bridge in hermes_cli/main.py). The formatter never reads config or files
# for it (it runs under the logging handler's lock, per record): the values are recorded when that
# profile's .env / config is loaded (``note_env_policy`` / ``note_config_policy``) or when log routing
# is set up for it (``resolve_profile_policy``). Until then the profile redacts (the secure default).
_ENV_POLICY_BY_HOME: dict = {}     # home key -> raw HERMES_REDACT_SECRETS from its .env, or None
_CONFIG_POLICY_BY_HOME: dict = {}  # home key -> raw security.redact_secrets from its config, or None


def note_env_policy(home_key: str, raw) -> None:
    _ENV_POLICY_BY_HOME[home_key] = None if raw is None else str(raw)


def note_config_policy(home_key: str, raw) -> None:
    _CONFIG_POLICY_BY_HOME[home_key] = None if raw is None else str(raw)


def resolve_profile_policy(home) -> None:
    """Record ``home``'s redaction policy now (log-routing setup), so its records never wait for it."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_constants import hermes_home_key
    try:
        with bind_profile_scope(home, hydrate_secrets=False):
            key = hermes_home_key()
            from hermes_cli.config import load_config_readonly, load_env
            note_env_policy(key, load_env().get("HERMES_REDACT_SECRETS"))
            note_config_policy(key, (load_config_readonly().get("security") or {}).get("redact_secrets"))
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- unreadable now: stays unknown, which redacts
        logging.getLogger(__name__).debug("redaction policy for %s not read: %s", home, exc)


def _redact_enabled() -> bool:
    """Effective redaction switch for the active profile (launch snapshot when no override). Reads
    only in-memory state: the live secret scope and the recorded per-profile policy."""
    from hermes_constants import get_hermes_home_override, hermes_home_key
    if get_hermes_home_override() is None:
        return _REDACT_ENABLED
    home_key = hermes_home_key()
    from agent.secret_scope import current_secret_scope
    scope = current_secret_scope()
    raw = scope.get("HERMES_REDACT_SECRETS") if scope else None
    if raw is None and scope is None:
        # No live scope (the log listener thread formats routed records): the profile's own .env, as
        # its scope would have it. Not recorded yet: redact, since .env may opt the profile back in.
        if home_key not in _ENV_POLICY_BY_HOME:
            return True
        raw = _ENV_POLICY_BY_HOME[home_key]
    if raw is None:
        raw = _CONFIG_POLICY_BY_HOME.get(home_key)
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# Known API key prefixes -- match the prefix + contiguous token chars.
# Every pattern MUST start with a literal prefix: _PREFIX_SUBSTRINGS (the cheap
# pre-screen gate) is derived from these literals and must stay false-negative-free.
_PREFIX_PATTERNS = [
    # Some provider-issued ``sk-`` keys carry dot-delimited body segments (Alibaba
    # ``sk-sp-…``/``sk-ws-…``). Each unit is one body char optionally preceded by
    # a single dot, so the body ends on its last non-dot char (sentence punctuation
    # is never consumed) and can never span ``..``: the ``sk-pro...EFGH`` display
    # mask is left alone by a second redaction pass instead of collapsing to
    # ``***``. Kept free of nested unbounded repeats so the pattern passes the
    # same structural gate plugins must.
    r"sk-[A-Za-z0-9_-](?:\.?[A-Za-z0-9_-]){9,}",
    # Context7 API keys: a fixed vendor prefix (``ctx7sk-``) plus an opaque body.
    # The generic ``sk-`` family cannot catch it: the matcher's lookbehind
    # rejects a prefix preceded by a word char, and here the ``sk-`` sits inside
    # ``ctx7sk-`` - so a bare Context7 key passed through unmasked (#142576).
    r"ctx7sk-[A-Za-z0-9_-]{10,}",
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh token
    r"xapp-\d+-[A-Za-z0-9-]{10,}",      # Slack app-Level token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack bot/app/user tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"GOCSPX-[A-Za-z0-9_-]{8,}",        # Google OAuth client secret (pasted client_secret.json, #64351)
    r"ya29\.(?:[a-z]\.)?[A-Za-z0-9_-]{20,}",  # Google OAuth access token: user a0…, service-account c., STS d. (#64351)
    r"AQ\.[A-Za-z0-9_-]{40,}",           # Google Gemini authorization keys (newer "AQ." prefix; AIza pattern doesn't catch them) — #66920
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",          # Fal.ai
    r"fc-[A-Za-z0-9]{10,}",             # Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",      # BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",        # Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",        # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{10,})?",  # SendGrid API key (2- or 3-part)
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",             # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",         # DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",         # DigitalOcean OAuth
    # AgentMail API key: ``am_`` / ``am_org_`` + an opaque alphanumeric body. The body has no ``_``/``-``,
    # which is what separates it from ``am_example_identifier_123`` (#10983); public docs pin only the
    # prefix, so the charset stays broad and the length floor does the discriminating.
    r"am_(?:org_)?[A-Za-z0-9]{20,}",
    r"sk_[A-Za-z0-9_]{10,}",            # ElevenLabs TTS key (sk_ underscore, not sk- dash)
    r"tvly-(?:(?:dev|prod)-)?[A-Za-z0-9]{10,}",  # Tavily search API key (incl. tvly-dev-/tvly-prod-, #72925)
    r"exa_[A-Za-z0-9]{10,}",            # Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",            # Groq Cloud API key
    r"syt_[A-Za-z0-9]{10,}",            # Matrix access token
    r"retaindb_[A-Za-z0-9]{10,}",       # RetainDB API key
    r"hsk-[A-Za-z0-9]{10,}",            # Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",           # Mem0 Platform API key
    r"brv_[A-Za-z0-9]{10,}",            # ByteRover API key
    r"xai-[A-Za-z0-9]{30,}",            # xAI (Grok) API key
    r"ntn_[A-Za-z0-9]{10,}",            # Notion internal integration token
    r"fw-[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fw_[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fpk_[A-Za-z0-9]{30,}",            # Fireworks AI project key
    # GitLab token families (each keeps a full literal prefix for the pre-screen).
    # Ported from openclaw/openclaw#112954; follow-up invited in #4541.
    r"glpat-[A-Za-z0-9_\-]{10,}",       # GitLab personal access token
    r"gloas-[A-Za-z0-9_\-]{10,}",       # GitLab OAuth application secret
    r"gldt-[A-Za-z0-9_\-]{10,}",        # GitLab deploy token
    r"glrt-[A-Za-z0-9_.\-]{10,}",       # GitLab runner authentication token (routable tokens are dotted)
    r"glrtr-[A-Za-z0-9_.\-]{10,}",      # GitLab runner registration token (routable)
    r"glcbt-[A-Za-z0-9_\-]{10,}",       # GitLab CI/CD job token
    r"glptt-[A-Za-z0-9_\-]{10,}",       # GitLab pipeline trigger token
    r"glft-[A-Za-z0-9_\-]{10,}",        # GitLab feed token
    r"glimt-[A-Za-z0-9_\-]{10,}",       # GitLab incoming mail token
    r"glagent-[A-Za-z0-9_\-]{10,}",     # GitLab agent (KAS) token
    r"glsoat-[A-Za-z0-9_\-]{10,}",      # GitLab service-account access token
    r"glffct-[A-Za-z0-9_\-]{10,}",      # GitLab feature-flags client token
    r"glwt-[A-Za-z0-9_\-]{10,}",        # GitLab workspace token
    r"GR1348941[A-Za-z0-9_\-]{10,}",    # GitLab legacy runner registration token
    r"glsa_[A-Za-z0-9]{10,}(?:_[A-Fa-f0-9]{8})?",  # Grafana service-account token (+ hex checksum)
    r"glc_eyJ[A-Za-z0-9+/]{29,}={0,3}",  # Grafana Cloud token: base64 of a JSON object, so "eyJ"
    r"pk-lf-[A-Za-z0-9\-]{8,}",         # Langfuse public key (sk-lf- already covered by sk- pattern)
]

# ENV assignment: KEY=value where KEY carries a secret-like name. Uppercase keys
# tolerate spaces around "=" and allow the keyword embedded anywhere
# (``MYTOKEN=…``) — an all-caps key is almost never prose. Bare ``KEY``/``PASS``/
# ``PW`` suffixes are included; _key_has_secret_keyword rejects ``KEYBOARD=``.
# The regex is IGNORECASE so lowercase env names (``openai_key=…``) are caught here too. The secret name
# must sit at a word boundary (``_``-delimited or whole-word) so generic prose words (``password=``,
# ``token=``, ``KEYBOARD=``, ``PASSAGE=``) do not match — those are handled by the config/form/URL paths,
# and a bare ``password=…`` in a form body must not be swallowed greedily by ``\S+``. See #77484.
_SECRET_ENV_NAMES = r"(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PW|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2")
# Command-line credentials. ``--password=…``, ``--token …`` and ``docker login -p …`` reach
# the approval card and the chat egress verbatim because the ENV passes only look at
# ``NAME=value`` (issue #143982). The flag name is kept so the reader still knows what was
# passed; only the value is masked. ``-p`` is scoped to a container ``login`` because
# everywhere else it is a port or a path.
# Each quote style is matched on its own, so ``"Abc'def ghi"`` is consumed whole instead of
# falling back to ``\S+`` at the apostrophe and leaving the tail visible. A double-quoted
# value honours backslash escapes (``"a\"b c"``) and a single-quoted one the shell's
# ``'\''`` splice, and either may span a newline. The body is capped and possessive: an
# unterminated or over-long quote masks through the end of its line instead of only its
# first word, and never rescans the rest of the text from every flag.
_QUOTED_FLAG_VALUE = (r"\"(?:\\[\s\S]|[^\"\\]){1,512}+\"|'(?:'\\''|[^']){1,512}+'"
                      r"|[\"'][^\n]*+|\S+")
_CLI_SECRET_FLAG_RE = re.compile(
    r"(?P<flag>(?<![\w-])--?(?:password|passwd|passphrase|pass|token|api[-_]?key|apikey"
    r"|client[-_]?secret|access[-_]?key|account[-_]?key|auth[-_]?token|session[-_]?token|secret|bearer"
    r"|credential)s?)"
    r"(?P<sep>\s*=|\s+)"
    rf"(?P<value>{_QUOTED_FLAG_VALUE})",
    re.IGNORECASE,
)
_CLI_LOGIN_P_RE = re.compile(
    r"(?P<flag>\b(?:docker|podman|nerdctl)\s+login\b[^\n;|&]*?\s-p\s*)"
    rf"(?P<value>{_QUOTED_FLAG_VALUE})"
)
# Azure connection-string keys. They are ``Name=value`` inside a quoted string, so
# every ENV pass declines them: the name is mixed case and carries no underscore boundary.
# The key must open a connection-string segment (start, ``;``, ``,``, a URL ``?``/``&``, a quote or whitespace before
# it, no space around ``=``): ``this.accountKey = x`` / ``dt.PrimaryKey = new …`` in source
# read through read_file or a terminal stays byte-identical, or the agent cannot patch it.
# A spaced ``AccountKey = …`` (ini files, PowerShell) only counts when the value is a
# 16+ char base64 run carrying a digit, which an identifier like ``config.Key`` or
# ``StorageAccountKeyFromVault`` is not.
_CONNSTR_SECRET_KEY_RE = re.compile(
    r"(?P<flag>(?<![^;,?&'\"\s])(?:AccountKey|SharedAccessKey|(?P<sas>SharedAccessSignature))"
    r"(?:=|[ \t]*+=[ \t]*+(?=[A-Za-z+/]*[0-9])(?=[A-Za-z0-9+/]{16,}={0,2}(?![\w.(]))))"
    # A SAS value is itself a query string (``sv=…&sig=…``): it runs through the ``sig=``
    # parameter and stops there, or at a ``;`` / ``)`` / ``,`` / ``|`` / backtick delimiter.
    r"(?P<value>(?(sas)(?:(?!sig=)[^;\s'\"&),`|]++&)*+(?:sig=)?[^;\s'\"&),`|]*+|[^;\s'\"&]+))",
    re.IGNORECASE,
)

# Cheap pre-gates: every flag the regexes above can match contains one of these.
_CLI_SECRET_FLAG_GATE = re.compile(r"-{1,2}[A-Za-z]")
_CLI_LOGIN_GATE = re.compile(r"(?:docker|podman|nerdctl)\s+login\b", re.IGNORECASE)
_CONNSTR_GATE = re.compile(r"(?:AccountKey|SharedAccessKey|SharedAccessSignature)", re.IGNORECASE)

# Lowercase env names: only underscore-boundary forms (``openai_key=``) — NOT
# bare ``password=``/``token=``, which appear in prose, URLs, and form bodies.
# The lookbehind anchors each attempt to the start of an identifier run; without
# it re.sub retries the greedy prefix at every byte of a long opaque payload.
# See #77484.
_ENV_ASSIGN_LOWER_RE = re.compile(
    r"(?<![a-z0-9_])([a-z0-9_]+(?:_|^)(?:key|pass|pw|token|secret|password|passwd|credential|auth)(?=[^a-z0-9_]|$))\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

# Lowercase / dotted config-file keys (``spring.datasource.password=x``,
# line-start ``password=x``). Carve-outs vs prose/code/URLs: values stop at
# whitespace AND ``&`` (form bodies go pair-by-pair via _redact_form_body);
# _CFG_DOTTED_RE needs a NAMESPACED key; _CFG_ANCHORED_RE needs line start
# (optionally after ``export``). The ``://`` URL guard lives at the call site.
# The spaced ``key = value`` form is code-shaped too, so it has its own gate: agent/redact_spaced_cfg.py.
# The uppercase _ENV_ASSIGN_RE above never matched these, so config-file passwords leaked verbatim (issue
# #16413). These run only in a config-file context, NOT in prose, code, or URLs — three carve-outs preserved
# from the original design (#4367 + the documented web-URL passthrough below): 1. The value is bounded by
# ``[^\s&]`` (stops at whitespace AND ``&``) so form-urlencoded bodies are handled pair-by-pair (by
# _redact_form_body), not greedily swallowed. 2. _CFG_DOTTED_RE only matches when the key is NAMESPACED
# (contains a dot), which is unambiguously a config key — never a prose word. 3. _CFG_ANCHORED_RE matches a
# bare secret-word key only at line start (optionally after ``export``), so conversational ``I have
# password=foo`` mid-sentence is left alone.
_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
# Rendered line-number prefix: ``5|line`` (read_file), ``6:line`` (grep -n), ``7-line`` (grep -A/-B/-C
# context lines) and ``     8\tline`` (cat -n / nl: right-aligned number + TAB). Callers put the ONLY
# leading ``[ \t]*`` in front of it — stacking a second whitespace run around an optional gutter made
# the anchored passes quadratic on long indented lines (2s per 5k spaces).
_LINE_NUMBER_GUTTER = r"(?:[0-9]+(?:[|:\-]|\t)[ \t]*)?"
_CFG_VALUE = r"(['\"]?)([^\s&]+?)\2(?=[\s&]|$)"
# Linear pre-gate for the _CFG_*_RE subs: no secret keyword => neither can match.
_CFG_SECRET_WORD_RE = re.compile(_SECRET_CFG_NAMES, re.IGNORECASE)

# Programmatic env lookups (``os.getenv(...)``, ``process.env.X``, ``$ENV{X}``)
# as the VALUE of a KEY=... match name a variable; they are not a leaked secret.
_ENV_LOOKUP_VALUE_RE = re.compile(r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)")
# Namespaced key: the secret word may sit anywhere in a dotted path.
# NOTE(perf): possessive quantifiers (nested ``(?:[...]+\.)+`` backtracked
# exponentially); the ``*`` runs bordering {_SECRET_CFG_NAMES} must stay
# backtrackable (``app.api.key=``). The lookbehind anchors each attempt to a key
# run start so re.sub is not quadratic; the match set is unchanged.
_CFG_DOTTED_RE = re.compile(
    rf"(?<![A-Za-z0-9_.\-])"
    rf"([A-Za-z0-9_\-]++\.[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*+"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]++)"
    rf"={_CFG_VALUE}",
    re.IGNORECASE,
)
# Line-anchored bare key: ``password=…`` / ``export api_key=…`` at start of line.
# ``{_LINE_NUMBER_GUTTER}``: line-numbered dumps put the key behind a rendered gutter —
# ``read_file`` emits ``5|      ADS_API_TOKEN: …``, ``grep -n`` emits ``6:      ADS_API_TOKEN: …``
# and ``cat -n`` emits ``     7\tADS_API_TOKEN: …``. Anchored at ``^`` without it, none of those
# matched, so the rendered read of a secret-bearing file leaked what the raw text masked.
_CFG_ANCHORED_RE = re.compile(
    rf"(^[ \t]*{_LINE_NUMBER_GUTTER}(?:export[ \t]+)?[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*)={_CFG_VALUE}",
    re.IGNORECASE | re.MULTILINE,
)

# YAML / colon config: bare-key quoted scalars cannot match _JSON_FIELD_RE.
# Match escaped double quotes, doubled single quotes, or an unquoted token.
# An unterminated quote masks to end-of-line; no branch crosses LF or CRLF.
# NOTE(perf): possessive where the successor is disjoint; the leading class
# stays backtrackable (see _CFG_DOTTED_RE).
_YAML_CFG_NAMES = (
    r"(?:api[ _.\-]?key|token|secret|passwd|password|credential"
    r"|client[ _.\-]?key[ _.\-]?data|(?:private|signing|encryption|master|hmac)[ _.\-]?key)"
)
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*+{_LINE_NUMBER_GUTTER}[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*+)(:[ \t]*+)"
    r"(\"(?:\\[^\r\n]|[^\"\\\r\n])*\"|'(?:''|[^'\r\n])*'|['\"][^\r\n]*|[^\s&]++)",
    re.IGNORECASE | re.MULTILINE,
)

_YAML_CFG_WORD_RE = re.compile(_YAML_CFG_NAMES, re.IGNORECASE)

# Word-boundary validation for the key patterns above: their classes allow
# arbitrary affixes (``client_secret``, ``s3.secret-key``), which also matched
# prose CONTAINING a keyword (``Secretary:``, ``tokenizer:``). A keyword counts
# only at a word boundary: key edge, next to a non-letter, or a camelCase
# transition (``clientSecret``, ``APIToken``); trailing plural ``s`` is part of
# it. Concatenations match via explicit alternatives (``authtoken``, ``apikey``).
# The side effect: ordinary prose/document words that merely CONTAIN a keyword also matched — ``Secretary:
# J.Smith`` (secret), ``tokenizer: cl100k_base`` (token), ``author=Smith`` (auth) — mangling legitimate
# content on the surfaces that run these passes (browser snapshots, log lines, kanban summaries, CLI-echoed
# command output). Ported from nearai/ironclaw#6129, where the same substring false positive ("Secretary of
# the Treasury" matching the ``secret`` marker) scrubbed legitimate tool results from the replayed
# transcript and sent the model into a re-fetch loop. Common concatenated compounds keep matching via
# explicit alternatives (``authtoken`` ngrok, ``authkey`` tailscale, ``secretkey`` minio, ``apikey``).
# Embedded occurrences inside a larger word (``secretary``, ``tokenizer``, ``authored``, ``credentialing``)
# no longer match. ALL-CAPS keys keep the legacy embedded matching (``MYTOKEN=…``) — an all-caps key is
# almost never prose, the same rationale as _ENV_ASSIGN_RE.
_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|secret)[ _.\\-]?(?:key|token)"
    r"|token|secret|passwd|password|pass|pw|credential|auth|key",
    re.IGNORECASE,
)

# Key names that are credential-specific even when their values are short or
# human-readable. Bare ``token`` / ``key`` are intentionally absent: they also
# describe model limits, tensor names, and cache keys, so those assignments
# are gated on value shape (_looks_like_opaque_credential).
_STRONG_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|id|bearer)[ _.\\-]?(?:key|token)"
    r"|key[ _.\\-]?material|secret|passwd|password|pass|pw|credential|auth|bearer",
    re.IGNORECASE,
)
# Password-class keys mask any literal value; for other keys a value that starts like ``$HOME/...``,
# ``/usr/...`` or ``~/...`` references a variable or a path, not a credential, even under a strong key
# (``SSH_AUTH_SOCK=$HOME/.ssh/agent.sock``, ``DOCKER_AUTH_CONFIG=/home/u/.docker``).
_PASSWORD_KEY_RE = re.compile(r"passwd|password|pass|pw", re.IGNORECASE)
# Anchored on both ends: the whole value must be a ``$VAR``/``${VAR}`` reference, a ``~/``
# path, or an absolute path — not merely a string whose FIRST character is one of those.
# A 40-char AWS secret key starts with '/' ~1 in 64 times and argon2/bcrypt digests always
# start with '$'; an unanchored class let those secrets skip every check below.
# Further ``$VAR`` interpolations may appear anywhere in the path (``/run/user/$UID/ssh``,
# ``$XDG_RUNTIME_DIR/agent.$USER.sock``, ``$A:$B`` lists); crypt digests never parse as one
# because their ``$`` fields start with a digit or carry ``=``/``,``.
# A leading ``$(`` is a command substitution (``SSH_AUTH_SOCK=$(gpgconf --list-dirs
# agent-ssh-socket)``): the value token stops at whitespace, so only ``$(gpgconf`` is seen.
_SHELL_VAR_REF = r"\$(?:\{[A-Za-z_]\w*[^}]*\}|[A-Za-z_]\w*)"
_PATH_OR_VAR_VALUE_RE = re.compile(rf"^(?:{_SHELL_VAR_REF}|\$\(|~|/)(?:[\w./:-]|{_SHELL_VAR_REF})*$")
# ``$VAR`` / ``$(cmd`` are unambiguous references. A ``/``- or ``~``-led value is a path only
# while every segment reads like one: a 16+ char segment mixing case and digits with no ``.``
# (``/wJalrXUtnFEMIK7MDENG/bPxRf…``) is a secret that happens to start with a path character,
# whereas ``/home/u/.docker`` / ``~/.ssh/id_rsa`` / ``S.gpg-agent.ssh`` never clear that bar.
_OPAQUE_PATH_SEGMENT_RE = re.compile(r"(?=[^.]*[a-z])(?=[^.]*[A-Z])(?=[^.]*[0-9])[^.]{16,}")


def _is_word_start(s: str, i: int) -> bool:
    """True if position ``i`` in ``s`` begins a word (edge, non-letter before, camelCase/acronym boundary)."""
    if i == 0:
        return True
    prev, cur = s[i - 1], s[i]
    if not prev.isalpha() or (cur.isupper() and prev.islower()):  # clientSecret
        return True
    return cur.isupper() and prev.isupper() and i + 1 < len(s) and s[i + 1].islower()  # APIToken


def _is_word_end(s: str, j: int, *, allow_plural: bool = True) -> bool:
    """True if position ``j`` (exclusive end) in ``s`` ends a word; one trailing ``s`` is absorbed."""
    if j >= len(s):
        return True
    cur = s[j]
    if not cur.isalpha() or (cur.isupper() and s[j - 1].islower()):  # secretKey
        return True
    return allow_plural and cur in "sS" and _is_word_end(s, j + 1, allow_plural=False)


def _has_word_bounded_keyword(key: str, keyword_re: re.Pattern[str]) -> bool:
    """True if ``keyword_re`` matches ``key`` at a word boundary (see _KEY_KEYWORD_RE)."""
    return any(_is_word_start(key, m.start()) and _is_word_end(key, m.end()) for m in keyword_re.finditer(key))


_PASSWORD_WORD_RE = re.compile(r"passwd|password", re.IGNORECASE)


def _has_password_keyword(key: str) -> bool:
    """``password``/``passwd`` ending a word, wherever it starts: no prose word ends in them, so
    concatenated keys (``dbpassword``, ``rootpasswd``) are credentials, while ``passwordless``
    is not. Requiring a word START too passed ``dbpassword: <v>`` through in clear."""
    return any(_is_word_end(key, m.end()) for m in _PASSWORD_WORD_RE.finditer(key))


# Tool-defined env names that glue the keyword onto a prefix with no separator, so the
# word-boundary test can never see it: libpq's ``PGPASSWORD``, MySQL's ``MYSQL_PWD`` (bare
# ``PWD`` is the shell's working directory, so only this exact name), sshpass's ``SSHPASS``.
_GLUED_CREDENTIAL_ENV_RE = re.compile(r"PGPASSWORD|MYSQL_PWD|SSHPASS", re.IGNORECASE)


def _is_glued_credential_env(key: str) -> bool:
    return _GLUED_CREDENTIAL_ENV_RE.fullmatch(key.strip("\"'")) is not None


def _key_has_secret_keyword(key: str) -> bool:
    """Post-match key validator: ``API_KEY``/``DB_PW`` count, ``KEYBOARD``/``secretary`` do not."""
    return (_has_word_bounded_keyword(key, _KEY_KEYWORD_RE) or _has_password_keyword(key)
            or _is_glued_credential_env(key))


def _looks_like_opaque_credential(value: str) -> bool:
    """Credential-like shape test for ambiguous ``token``/``key`` values, so short
    technical scalars (``CPU``, ``local``) are not masked merely for their key name."""
    if value == "***" or value.startswith("«redacted:"):
        return True
    if len(value) >= 16 and re.fullmatch(r"[A-Fa-f0-9]+", value):
        return True
    if len(value) >= 20 and re.fullmatch(r"[A-Za-z0-9_./+=-]+", value):
        return True
    if len(value) < 12:
        return False
    return sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]")) >= 2


_CONFIG_LITERAL_VALUES = frozenset({"true", "false", "yes", "no", "on", "off", "null", "none"})


def _should_redact_assignment(key: str, value: str, *, check_keyword: bool) -> bool:
    """Shared gate for the ENV / JSON / YAML assignment passes: skip programmatic env
    lookups used as values, optionally require a word-bounded keyword in the key,
    then redact when the key is unambiguously credential-bearing or the value looks opaque."""
    # Programmatic env lookups reference variable *names*, not secret values — masking them corrupts code
    # snippets in prose/log contexts (issue #2852): ``KEY=os.getenv('X')``.
    # Same programmatic-env-lookup exception as _redact_env above (issue #2852): "apiKey": "os.getenv('X')"
    # is a code snippet, not a leaked secret value.
    # Same programmatic-env-lookup exception as _redact_env above (issue #2852): api_key: os.getenv('X') is
    # a code snippet, not a leaked secret value.
    if _ENV_LOOKUP_VALUE_RE.match(value):
        return False
    # An earlier pass already masked this value (``***`` or the ``«redacted:…»`` sentinel). Masking it
    # again only erases what the sentinel deliberately kept — the vendor label (``Digest ***`` →
    # ``***``, ``«redacted:ghp_…»`` → ``«redacted-secret»``). Same guard _redact_python_repr_fields uses.
    if value == "***" or value.startswith("«redacted"):
        return False
    # ``redact_secrets: true`` is configuration state, not a credential: masking a boolean/null
    # literal under a secret-named key corrupts the config view the agent copies back (#142574).
    if value.lower() in _CONFIG_LITERAL_VALUES:
        return False
    if check_keyword and not _key_has_secret_keyword(key):
        return False
    # A shell rc's ``SSH_AUTH_SOCK=$HOME/.ssh/agent.sock`` is configuration the agent must keep
    # readable; only password-class keys mask a path/variable reference.
    password_key = _has_word_bounded_keyword(key, _PASSWORD_KEY_RE) or _has_password_keyword(key)
    if _PATH_OR_VAR_VALUE_RE.match(value) and not password_key:
        # ``$VAR`` is an unambiguous reference. A bare ``/...`` or ``~...`` is not:
        # ``/home/u/.docker`` and ``/8f3kd9sKd0als...`` have the same shape, so every
        # segment has to look like a path (see _OPAQUE_PATH_SEGMENT_RE) before the
        # value is treated as configuration.
        if value[0] == "$" or not any(_OPAQUE_PATH_SEGMENT_RE.fullmatch(seg) for seg in value.split("/")):
            return False
    # ``signing_key`` / ``privateKey`` are credentials at any length, the same policy
    # is_secret_field_name applies to mappings, so JSON text and dicts agree.
    return (password_key or _has_word_bounded_keyword(key, _STRONG_KEY_KEYWORD_RE)
            or _is_glued_credential_env(key)
            or (_is_qualified_key_name(key) and not _CONFIG_PLACEHOLDER_VALUE_RE.fullmatch(value))
            or _looks_like_opaque_credential(value))


# JSON field patterns: "apiKey": "value", "token": "value", etc.
# ``key``-suffixed credential names (``aws_secret_access_key``, ``SECRET_KEY``,
# ``privateKey``, ``signing_key``) and kubeconfig's ``client-key-data`` (a base64 PEM).
# The bare word ``key`` stays out: it names cache keys and model limits. A qualifier
# word has to sit in front of it, which is what the alternation spells out.
_KEY_QUALIFIED_NAME = (
    r"[\w.\-]*(?:secret|private|signing|encryption|encrypt|master|hmac|symmetric|access)"
    r"[_.\-]?(?:key|Key|KEY)"
)
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|pgpassword|mysql_pwd|sshpass|access_token|refresh_token|auth_token|bearer"
    r"|secret_value|raw_secret|secret_input|key_material|client[_.\-]?key[_.\-]?data"
    rf"|{_KEY_QUALIFIED_NAME})"
)
_JSON_FIELD_RE = re.compile(rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"', re.IGNORECASE)
# The same field inside a JSON-encoded string (``{"output": "{\\"api_key\\": \\"…\\"}"}``): every tool result
# that wraps a config dump in JSON escapes the quotes, so the rule above never saw the value (#115104).
# The backreference keeps the key's and value's escape depth equal.
_JSON_FIELD_ESCAPED_RE = re.compile(rf'((\\+)"{_JSON_KEY_NAMES}\2")\s*:\s*\2"([^"\\]+)\2"', re.IGNORECASE)
# Any other identifier-shaped key (``"API_TOKEN"``, ``"DB_PASSWORD"``, ``"client_secret"``): the closed
# list above misses affixed names that the ENV and YAML passes already accept (#147240). The callback
# applies the YAML pass's word set at a word boundary, then the shared assignment gate.
# The value may carry JSON escapes (``"s3cret\\value"``): ``\\.`` consumes each escape pair whole.
_JSON_ANY_KEY_FIELD_RE = re.compile(r'("([A-Za-z0-9_.\-]{1,128}+)")\s*+:\s*+"((?:[^"\\\n]|\\.)++)"')
# The same field inside a JSON-encoded string (``{"output": "{\\"DB_PASSWORD\\": \\"…\\"}"}``),
# like _JSON_FIELD_ESCAPED_RE does for the closed key list: group 2 is the escape run.
_JSON_ANY_KEY_FIELD_ESCAPED_RE = re.compile(
    r'((\\++)"([A-Za-z0-9_.\-]{1,128}+)\2")\s*+:\s*+\2"([^"\\\n]++)\2"')
_JSON_KEY_NAMES_FULL_RE = re.compile(_JSON_KEY_NAMES, re.IGNORECASE)

# Python ``repr`` uses single-quoted mapping fields, so opaque credentials in
# tracebacks and pytest failure introspection bypass the double-quoted JSON rule
# above: ``{'BRAVE_API_KEY': 'opaque-value'}``. Capture identifier-shaped keys
# here, then apply the canonical high-confidence key policy in the callback.
_PYTHON_REPR_SECRET_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "auth_token",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "secret",
    "password",
    "passwd",
    "private_key",
    "credential",
    "credentials",
    "authorization",
    "bearer",
    "secret_value",
    "raw_secret",
    "secret_input",
    "key_material",
})
_PYTHON_REPR_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_CREDENTIAL",
    "_CREDENTIALS",
)
# Casefolded credential suffixes for mixed/camel-case key names
# (``UserPassword``, ``sessionToken``, ``clientApiKey``). Suffix-only so
# ``token_count`` / ``password_policy`` metadata keys never match. Widened per
# OpenHands/software-agent-sdk#4508.
_PYTHON_REPR_CREDENTIAL_SUFFIXES = (
    "apikey",
    "api_key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
)
_PYTHON_REPR_FIELD_RE = re.compile(
    r"'(?P<key>[A-Za-z_][A-Za-z0-9_]*)'(?P<sep>\s*:\s*)"
    r"(?:"
    r"(?P<single_prefix>[bB]?)'(?P<single_value>(?:\\.|[^'\\])+)'"
    r"|(?P<double_prefix>[bB]?)\"(?P<double_value>(?:\\.|[^\"\\])+)\""
    r")"
)

# Terminal/process output normally uses ``code_file=True`` to preserve source.
# Add repr masking only to high-confidence diagnostic lines: pytest assertion
# introspection (``E       ...``) and final Python exception lines.
_PYTEST_DIAGNOSTIC_LINE_RE = re.compile(r"^(?P<prefix>[ \t]*E[ \t]{2,})(?P<body>.*)$")
_PYTHON_EXCEPTION_LINE_RE = re.compile(
    r"^(?P<prefix>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*"
    r"(?:Error|Exception|Warning):[ \t]*)(?P<body>.*)$"
)

# Authorization / Proxy-Authorization, any scheme or bare credential; header
# name and scheme word preserved. The credential class excludes quotes: pulling
# a closing quote into the mask turns value corruption into SYNTAX corruption
# (unterminated quote → shell EOF / SyntaxError). A scheme word with no token after it
# (``"authorization: bearer "``, ``(Authorization: Basic)``) is not a credential: the optional
# scheme group backs off and would otherwise mask the scheme word itself (#142574).
_AUTH_SCHEME_WORD = r"(?:bearer|basic|bot|token|digest|negotiate|ntlm|oauth)(?=[`)\],.;:!?]*(?:[\s\"']|$))"
_AUTH_HEADER_RE = re.compile(
    rf"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?(?!{_AUTH_SCHEME_WORD})([^\s\"']+)", re.IGNORECASE)

# Gate for _AUTH_HEADER_RE: the header-name substring in ANY casing. A mixed-case
# name (aUtHoRiZaTiOn) contains neither the all-lower nor the all-upper run, so
# plain substring checks would never reach the IGNORECASE regex (issue #108807).
_AUTH_GATE_RE = re.compile(r"uthorization", re.IGNORECASE)


# Header placeholder classification (display-fidelity intake 10123). The secret-header masking
# itself lives in agent/redact_headers.py; this scanner reads each ``name: <\S+ run>`` slot the
# way the classification below was designed for and decides only whether that slot is wholly a
# value-free placeholder (see ``_redact_secret_headers``).
_SECRET_HEADER_RE = re.compile(rf"({_SECRET_HEADER_NAMES}\s*:\s*)(\S+)", re.IGNORECASE)
_SECRET_HEADER_NAME_RE = re.compile(_SECRET_HEADER_NAMES, re.IGNORECASE)

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
# is either the untouched original or exactly the base mask. Anything that can carry a
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


def _slot_is_credential_placeholder(m: re.Match, group: int, after_re: re.Pattern) -> bool:
    """True when header slot ``group`` of ``m`` begins ONE complete value-free placeholder
    (see above) that ends at a boundary ``after_re`` accepts (fullmatch of the rest of the
    run). Reads the original string past the slot (inner whitespace); never changes what is
    emitted."""
    start, end = m.span(group)
    if end - start > _PLACEHOLDER_MAX_SLOT:
        return False
    p = _slot_placeholder(m, group)
    if p is None:
        return False
    following = _FOLLOWING_RUN_RE.match(m.string, p.end()).group()
    return len(following) <= _PLACEHOLDER_MAX_SLOT and after_re.fullmatch(following) is not None


def _auth_header_sub(m: re.Match) -> str:
    # The Authorization slot stops at a quote, so only the boundary right after the
    # placeholder decides; whatever follows a real quote was never part of base's mask.
    if (m.end(3) - m.start(3) <= _PLACEHOLDER_MAX_SLOT
            and (p := _slot_placeholder(m, 3)) is not None
            and _AUTH_AFTER_RE.match(m.string, p.end()) is not None):
        return m.group(0)
    return m.group(1) + (m.group(2) or "") + _mask_token(m.group(3))


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


def _secret_header_keep_end(m: re.Match, scan: "_JsonStringScan") -> int | None:
    """Offset in ``m.string`` up to which the original x-api-key match is kept unchanged,
    or None for exactly base's mask of the whole slot. The slot ends either at the end of
    the run (``_SECRET_AFTER_RE``) or at a compact JSON member/element boundary
    (``_json_member_boundary_end``, review6366 HF-11 / review6471 HF-12 / review6640); ``scan``
    holds the JSON strings and containers open at the header (``_JsonStringScan``)."""
    if _slot_is_credential_placeholder(m, 2, _SECRET_AFTER_RE):
        return m.end()
    p = _slot_placeholder(m, 2)
    if p is None:
        return None
    c_end = _json_member_boundary_end(m.string, p.end(), scan)
    if (c_end is None or c_end - p.end() > _PLACEHOLDER_MAX_SLOT
            or min(c_end, m.end()) - m.start(2) > _PLACEHOLDER_MAX_SLOT):
        return None
    return c_end


# Inserted between an exempt header's name and its ``:`` so no agent/redact_headers.py pattern
# (all of which need ``NAME[ \t]*:``) sees that occurrence; removed again right after.
_PLACEHOLDER_HEADER_SHIELD = "\ue000"  # a private-use character


def _placeholder_header_name_ends(text: str) -> list[int]:
    """End offsets of the header names whose slot is kept verbatim (``_secret_header_keep_end``).
    A kept slot that ends at a compact JSON member/element boundary resumes the scan there, so the
    next member is classified on its own instead of being swallowed by the slot."""
    ends, pos = [], 0
    strings = _JsonStringScan(text)
    while (m := _SECRET_HEADER_RE.search(text, pos)) is not None:
        strings.depth(m.start())
        keep_end = _secret_header_keep_end(m, strings)
        if keep_end is None:
            pos = m.end()
        else:
            ends.append(_SECRET_HEADER_NAME_RE.match(text, m.start(1)).end())
            pos = min(keep_end, m.end())
    return ends


def _redact_secret_headers(text: str) -> str:
    """``redact_header_credentials`` (secret and cookie headers in every raw/curl/inline form),
    except that a secret-header slot which is wholly ONE value-free placeholder stays byte-for-byte.
    Text that already contains the shield character is masked exactly as without the exemption."""
    if _PLACEHOLDER_HEADER_SHIELD in text:
        return redact_header_credentials(text)
    ends = _placeholder_header_name_ends(text)
    if not ends:
        return redact_header_credentials(text)
    parts, last = [], 0
    for end in ends:
        parts.append(text[last:end])
        parts.append(_PLACEHOLDER_HEADER_SHIELD)
        last = end
    parts.append(text[last:])
    return redact_header_credentials("".join(parts)).replace(_PLACEHOLDER_HEADER_SHIELD, "")


# Telegram bot tokens: [bot]<digits>:<token>, token >= 30 chars. Start at an
# identifier boundary, never inside a digit run (quadratic retries on dumps).
# Consume only bounded escape spellings so JSON/repr/URL separators survive
# verbatim; their last letter/digit is not the token's logical left neighbor.
# An ANSI CSI control (colored output), raw or JSON/repr-escaped, is a separator too.
_TELEGRAM_RE = re.compile(
    r"(?P<boundary>\\[nrtbfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|"
    r"\\U[0-9a-fA-F]{8}|%[0-9a-fA-F]{2}|(?P<csi>(?:\x1b|\\x1b|\\u001[bB]|\\033)\[[0-?]*[ -/]*[@-~])|(?<![A-Za-z_\d]))"
    r"(?P<bot>bot)?(?P<id>\d{8,}):(?P<body>[-A-Za-z0-9_]{30,})"
)


def _redact_telegram_token(match: re.Match) -> str:
    boundary = match.group("boundary")
    # Encoded identifier characters are still identifiers, not separators.
    if len(boundary) > 2 and not match.group("csi"):
        value = int(boundary[1:] if boundary[0] == "%" else boundary[2:], 16)
        if value > 0x10FFFF:
            return match.group(0)
        char = chr(value)
        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_" or char.isdecimal():
            return match.group(0)
    return f"{boundary}{match.group('bot') or ''}{match.group('id')}:***"

_PRIVATE_KEY_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----")
# A base64-encoded PEM private key: kubeconfig's ``client-key-data`` and the ``*_b64``
# fields around it (``kubectl config view --raw``, cloud-init, CI secrets). The name is
# not reliable — kubeconfig writes ``client-key-data`` in YAML and JSON alike, and the
# terminal path deliberately skips the assignment passes on source-shaped output — so the
# rule keys on the payload: base64 that decodes to a ``BEGIN … PRIVATE KEY`` header.
_B64_PEM_RE = re.compile(r"LS0tLS1CRUdJTi[A-Za-z0-9+/=]{60,}")
# Private-key formats that are not PEM. All three reach the model on their own: PGP's
# armour carries a \` BLOCK\` suffix so the PEM pattern never matched it, PuTTY's .ppk is
# base64 under a \`Private-Lines:\` header (the usual SSH key format on Windows), and age
# writes a bech32 AGE-SECRET-KEY-1… string.
_PGP_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----"
)
# Line ends may be CRLF (Windows writes .ppk that way) or a JSON-escaped ``\\n``.
_PPK_EOL = r"(?:\r?\n|\\r\\n|\\n)"
# Body lines may carry read_file's ``  4|`` line-number gutter.
_PPK_BODY_GUTTER = r"[ \t]*+(?:[0-9]++(?:[|:]|\t)[ \t]*+)?"
_PUTTY_PRIVATE_LINES_RE = re.compile(
    rf"(?<![\w-])(Private-Lines:[ \t]*+[0-9]++[ \t]*+(?P<eol>{_PPK_EOL}))"
    rf"(?:{_PPK_BODY_GUTTER}[A-Za-z0-9+/=]{{16,}}+[ \t]*+{_PPK_EOL}?)++"
)
_AGE_SECRET_KEY_RE = re.compile(r"AGE-SECRET-KEY-1[0-9A-Z]{20,}")


def _is_b64_pem_key(blob: str) -> bool:
    """True when ``blob`` is base64 of a PEM private key block."""
    import base64
    import binascii

    padded = blob + "=" * (-len(blob) % 4)
    try:
        decoded = base64.b64decode(padded, validate=False)
    except (binascii.Error, ValueError):
        return False
    text = decoded[:64]
    return text.startswith(b"-----BEGIN") and b"PRIVATE KEY" in text


# Database connection strings: protocol://user:PASSWORD@host. The userinfo and
# password groups forbid whitespace so a match can never span a line break (a
# greedy ``[^@]+`` once ran to a decorator's ``@`` on the next code line).
# Database connection strings: protocol://user:PASSWORD@host Catches postgres, mysql, mongodb, redis, amqp
# URLs and redacts the password. A real DSN password never contains whitespace; without this bound the
# greedy [^@]+ would scan past the end of a code line to the next stray "@" (e.g. a Python decorator),
# swallowing intervening lines and corrupting tool OUTPUT for any source containing a postgresql:// f-string
# template. See issue #33801.
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?|amqps?)://[^:\s]*+:)((?:[^@\s:]++|:(?!//))++)(@)",
    re.IGNORECASE,
)

# Bare-token URL credential ``scheme://TOKEN@host`` (``git remote set-url
# https://PASSWORD@github.com/...``): unambiguously a secret, since round-trip
# URLs (OAuth callbacks, magic links) carry tokens in the QUERY STRING, never
# bare userinfo. ``user:pass@`` passes through (class forbids ``:``); DB schemes
# belong to _DB_CONNSTR_RE. 8+ char floor skips short usernames; the class
# forbids ``/`` so an ``@`` in a path/query (``?q=user@example.com``) never counts.
# The class also forbids ``"``: a double quote is not URL userinfo, and
# ``https://schema.org","@type`` is a JSON string boundary, not a credential
# (src#903). A comma stays legal userinfo.
# This is the ``git remote set-url origin https://PASSWORD@github.com/...`` shape from issue #6396 — a
# single opaque credential in the userinfo position with NO ``user:pass`` colon. The colon form's
# PASSWORD is masked separately by _redact_userinfo_passwords (issue #125664) — #34029's pass-through
# was about query-string round-trip tokens (magic links / OAuth callbacks / pre-signed URLs live in
# the query, never in userinfo), and a userinfo password is a credential, not a workflow token. DB
# schemes are handled by _DB_CONNSTR_RE above and excluded here. Guards against false positives:
_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"  # scheme
    r"([^\s:@/\"]{8,})"                             # bare token, 8+ chars; no quote
    r"(@[^\s]+)",                                   # @host...
    re.IGNORECASE,
)

# Incoming-webhook URLs. The URL IS the credential: anyone holding it can post into the
# workspace as the app, with no other secret. Web URLs are otherwise passed through on purpose
# (magic links, OAuth callbacks the agent must follow), so this stays vendor-shaped like
# _DB_CONNSTR_RE and _URL_BARE_TOKEN_RE: only the fixed host/path forms below match, and
# only the credential segment is masked, leaving the host and ids readable for diagnosis.
# Teams "Workflows" webhooks (the Power Automate replacement for Office 365 connectors) are
# logic.azure.com / api.powerplatform.com trigger URLs whose only secret is the ``sig`` query
# parameter; Slack workflow triggers live under hooks.slack.com/triggers/.
# Hermes stores one as ``incoming_webhook_url`` for Teams delivery and ships Slack and Discord
# adapters, so these land in config dumps, tool output and logs. Telegram's is already covered
# by _TELEGRAM_RE, which masks the bot token inside its API URL. ``http://`` is matched too: a
# mistyped scheme still carries the same secret in the path. Slack Workflow Builder URLs carry a
# numeric id before the secret (workflows/T/A/<id>/<secret>), so that form is tried first; the
# generic arm skips an all-digit segment followed by ``/`` so a second pass (RedactingFormatter
# redacts twice) leaves that public id readable.
# Every repeat in the Teams Workflows arm is capped: it is retried at each URL start, and the
# uncapped prefix/path runs overlapped, which made repeated URLs cubic (50 KB took ~48 s).
_WEBHOOK_HOST_END = r"\.?(?::\d+)?/"  # trailing-dot FQDN and explicit port reach the same host
_WEBHOOK_URL_RE = re.compile(
    r"(https?://hooks\.slack(?:-gov)?\.com" + _WEBHOOK_HOST_END
    + r"workflows/[A-Z0-9]+/[A-Z0-9]+/\d+/)([A-Za-z0-9]{8,})"
    r"|(https?://hooks\.slack(?:-gov)?\.com" + _WEBHOOK_HOST_END
    + r"(?:services|workflows|triggers)/[A-Z0-9]+/[A-Z0-9]+/)(?!\d+/)([A-Za-z0-9]{8,})"
    r"|(https?://(?:canary\.|ptb\.|www\.)?discord(?:app)?\.com" + _WEBHOOK_HOST_END
    + r"api/(?:v\d+/)?webhooks/\d+/)([\w-]{8,})"
    r"|(https?://[A-Za-z0-9.-]*\.?(?:webhook\.office\.com" + _WEBHOOK_HOST_END
    + r"webhookb2|outlook\.office\.com" + _WEBHOOK_HOST_END + r"webhook)/)([\w@/%+=-]{8,})"
    r"|(https?://[A-Za-z0-9.-]+\.(?:logic\.azure\.com|api\.powerplatform\.com)" + _WEBHOOK_HOST_END
    + r"(?:[^\s?#\"'<>/]{1,128}/){0,8}workflows/[^\s?#\"'<>]{1,512}?\?"
    r"(?:[^\s#\"'<>]{0,1024}?&)?sig=)([A-Za-z0-9_%-]{8,})",
    re.IGNORECASE,
)


# JWTs always start with "eyJ" (base64 "{"); 1-, 2- and 3-part forms.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}")

# Links ``hermes webapp`` prints that sign a browser in as the user: the gateway's browser credential
# after ``hermes-session=`` and the one-use ticket of a ``/webapp/launch`` link. Both values are
# URL-safe base64 (43 characters); the 20-character floor leaves placeholders and code (``${token}``).
_WEBAPP_LINK_CREDENTIAL_RE = re.compile(
    r"((?<![A-Za-z0-9_-])hermes-session=|/webapp/launch#(?:[^\s#]*?&)?ticket=)[A-Za-z0-9_-]{20,}")

# E.164 phone numbers, 7-15 digits; the lookahead rejects hex strings / identifiers. A ``+`` glued
# to a word or dot is a version suffix, not a phone: ``python-3.14.7+20260901-linux-x64`` (#142574).
# Signed quantities and joined identifiers of the same shape are told apart by agent.redact_quantities.non_phone_offsets.
_SIGNAL_PHONE_RE = re.compile(r"(?<![0-9A-Za-z_.])(\+[1-9]\d{6,14})(?![A-Za-z0-9])")  # ASCII: CJK-glued numbers still mask

# CDP-URL path: web URLs with a query string / with ``user:password@`` userinfo
# (DB protocols are covered by _DB_CONNSTR_RE).
_URL_WITH_QUERY_RE = re.compile(r"(https?|wss?|ftp)://([^\s/?#]+)([^\s?#]*)\?([^\s#]+)(#\S*)?")
_URL_USERINFO_RE = re.compile(r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@")

# Strict provider-egress URL redaction: delimiters stay in capture groups so the
# query/fragment layout is preserved byte-for-byte; the key is decoded
# separately for classification. Values stop at ``&``/``;`` (both valid). An apostrophe
# followed by a token character is part of the value (``?token=abc'tail``); one followed by
# anything else closes a quoted URL (``'…?token=abc')``, ``',``, ``'.strip()``) and stays.
_STRICT_URL_PARAM_RE = re.compile(r"([?#&;])([A-Za-z0-9_.~+%\-]+)=([^#&;\s\"'<>]*(?:'[A-Za-z0-9_~+%\-][^#&;\s\"'<>]*)*)")

# Userinfo in absolute and network-path (``//user:pass@host``) references; the
# authority stops at path/query/fragment delimiters. Anchored on the mandatory
# ``//`` — an optional-scheme prefix backtracked O(n²) on long alphanumeric runs
# (~55s per sub() on a 320KB compaction payload).
# No ``"``: ``"https://a.com","u:pw@y"`` is two JSON strings, not ``a.com","u`` userinfo.
_STRICT_URL_USERINFO_RE = re.compile(r"//[^/\s?#@\"]+@")
# The same authority span with the delimiter and userinfo captured, for _redact_userinfo_passwords.
_URL_USERINFO_PAIR_RE = re.compile(r"(//)([^/\s?#@\"]+)@")
# Scheme-less network path (``mount //user:pass@host/share``), masked under force only: ``//`` must
# start a token (comments are ``// note``, protocol-relative URLs carry no ``@``) and the password
# needs 8+ characters, so ``//user:note@x`` stays as written.
_NETWORK_PATH_USERINFO_RE = re.compile(r"(?<![^\s\"'=(`,\[])(//)([^/\s?#@:\"]+):([^/\s?#@\"]{8,})@")

# ``NAME=user:pass`` where NAME carries no secret keyword — an HTTP Basic auth pair is often stored
# this way (#125664). Shape-only on arbitrary names, so it runs force-only (persistence boundary,
# debug dumps, session export all pass force=True): ordinary prose/config keeps today's behaviour.
# Guards against non-secret colon values: the password needs an 8-char floor AND a letter (ports and
# digit-only tags fail it), the password group is a single token (no whitespace, quotes or ``/`` —
# so ``registry.example.com:5000/repo`` keeps its path, and an ISO timestamp's ``:00:00`` tail
# truncates the group to digits), and a version-shaped tail (``24.04-noble``, ``v1.2.3-alpine``)
# is rejected outright. The value is the WHOLE assignment (``user`` cannot contain ``:``/``@``/``/``),
# and the ``:(?!//)`` lookahead keeps ``scheme://`` values (``MY_URL=https://example.com``) out.
# ``user`` must also be at least 2 characters: a single-letter ``user:`` is a Windows drive
# (``SRC=C:\Users\...``), and masking the whole path as ``SRC=C:***`` at a force boundary is
# unrecoverable data loss — no real Basic pair has a one-letter user, and a secret-keyword NAME
# still masks via the ENV pass (#125680 review).
# ``@`` is not a password character here either: a ``user:pass@host``-shaped value belongs to the
# userinfo pass below, which knows the f-string ``{...}`` template exemption; excluding it lets that
# pass see the pure brace expression (``BASIC="user:{pw}@host"`` keeps its template verbatim).
_BASIC_PAIR_ASSIGN_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])([A-Za-z][A-Za-z0-9_.\-]{0,63})=([\"']?)"
    r"([A-Za-z0-9._~\-]{2,})(?::(?!//)([^\s'\"/:@]+))\2(?![^\s;,&|)\]}\"'`])"
)

# ``24.04-noble`` / ``v1.2.3-alpine`` image tags: digit-led dotted versions, optionally ``v``-prefixed
# — long and letter-bearing enough to pass the floor, yet never a credential.
_VERSION_TAG_RE = re.compile(r"^[vV]?\d+(\.\d+)+")
# ``nginx:stable-alpine`` / ``myrepo:latest-release``: lower-case words joined by ``-``/``.``
# are image tags and labels; a password carries a digit, a capital or a symbol.
_WORD_TAG_RE = re.compile(r"[a-z]++(?:[.\-][a-z]++)*+")

# ``CHECKSUM=sha256:1234…`` digest labels: the hex body is long and letter-bearing, but a hash
# algorithm name as the ``user`` half marks the pair as an algorithm:digest, not a credential
# (#125680 review). No real Basic pair has one of these as its username.
_HASH_ALGO_NAMES = frozenset({
    "md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sha3-224", "sha3-256",
    "sha3-384", "sha3-512", "blake2b", "blake2s", "ripemd160", "crc32", "crc32c",
})

# ``NAME=user:pass`` is only a Basic pair when NAME says it holds one: ``IMAGE=redis:7-alpine``,
# ``BRANCH=origin:feature2``, ``LS_COLORS=...:mh=00:...`` and ``Server=tcp:db,1433`` are colon
# values under names that never carry credentials (#149435 review).
_BASIC_PAIR_NAME_RE = re.compile(r"auth|basic|cred|login|proxy|userpass|userinfo", re.IGNORECASE)


def _is_basic_pair_password(password: str) -> bool:
    """8+ chars with a letter and not a version-shaped tail — a secret, not a tag or timestamp."""
    if _VERSION_TAG_RE.search(password) or _WORD_TAG_RE.fullmatch(password):
        return False
    return len(password) >= 8 and any(c.isalpha() for c in password)

# Form-urlencoded body: only when the ENTIRE text is a k=v&k=v string.
_FORM_BODY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$")


def _last_char_class(pattern: str) -> "str | None":
    """Text of the last ``[...]`` class in ``pattern`` (escape-aware), or None.

    For the built-in shape ``<literal prefix><body class><quantifier>`` this is the
    pattern's body class.
    """
    last = None
    i = 0
    while i < len(pattern):
        if pattern[i] == "\\":
            i += 2
            continue
        if pattern[i] == "[":
            j = i + 1
            if pattern[j:j + 1] == "^":
                j += 1
            if pattern[j:j + 1] == "]":  # a leading "]" is literal
                j += 1
            while j < len(pattern) and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            last = pattern[i:j + 1]
            i = j + 1
            continue
        i += 1
    return last


def _trailing_guard(pattern: str) -> str:
    """Lookahead ending ``pattern``: reject a following alphanumeric (word boundary), and ``_``/``-``
    only when the pattern's body class admits them. Never stricter than the shared guard, so a
    plugin pattern whose last class is a delimiter check (``(?=[\\s.,])``) still matches."""
    body_class = _last_char_class(pattern)
    try:  # a "[" inside a (?#...) comment is no class: fall back to the shared guard
        seps = "".join(c for c in "_-" if re.fullmatch(body_class, c)) if body_class else None
    except re.error:
        seps = None
    return r"(?![A-Za-z0-9_-])" if seps is None else f"(?![A-Za-z0-9{seps}])"


def _compile_prefix_matcher(patterns: list) -> re.Pattern[str]:
    # Each branch gets its own guard (_trailing_guard): a following alphanumeric always
    # blocks, ``_``/``-`` only when the body class admits them. The old shared
    # ``(?![A-Za-z0-9_-])`` failed ``ghp_…[A-Za-z0-9]{10,}`` at the body's end and at every
    # backtrack step, so a token followed by ``_``/``-`` was missed and shipped in cleartext
    # (#142852). A pattern without a ``[...]`` class falls back to the shared class.
    return re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(f"(?:{p}){_trailing_guard(p)}" for p in patterns) + r")")


def _compile_prefix_sub_matcher(patterns: list) -> re.Pattern[str]:
    """_PREFIX_RE for the masking pass, also admitting a key glued to an identifier by ``-``/``_``
    (``backup-sk-…``, ``x-ghp_…``), which _PREFIX_RE rejects so ``scikit-learn`` stays whole. One
    fixed-width lookbehind keeps main's single scan; _mask_prefix_match masks a glued match only
    when its body is opaque, so ``backup-sk-learn`` stays whole too (#147240)."""
    # Branches grouped by their literal first char (every pattern opens on one, see
    # register_redaction_patterns): admitting ``-``/``_`` before a key makes every separator a
    # candidate, and a flat 60-way alternation per candidate doubled the cost of a ``_ghp_`` flood.
    # Groups are disjoint by first char, so leftmost-first match order is unchanged.
    by_first: dict = {}
    for pattern in patterns:
        by_first.setdefault(pattern[0], []).append(f"(?:{pattern[1:]}){_trailing_guard(pattern)}")
    body = "|".join(f"{re.escape(c)}(?:{'|'.join(rest)})" for c, rest in by_first.items())
    return re.compile(r"(?<![A-Za-z0-9])(" + body + r")")


_PREFIX_RE = _compile_prefix_matcher(_PREFIX_PATTERNS)
_PREFIX_SUB_RE = _compile_prefix_sub_matcher(_PREFIX_PATTERNS)
# A real key body has a 16+ char run with no separator mixing two of lower/upper/digit;
# ``sk-prod-deployment-01`` and ``fc-feature-flags-v2`` do not.
_LONG_ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]{16,}+")


def _is_opaque_prefixed_token(token: str) -> bool:
    for run in _LONG_ALNUM_RUN_RE.finditer(token):
        run = run.group(0)
        if (run.isalpha() and (run.islower() or run.isupper())) or run.isdigit():
            continue
        return True
    return False


def _mask_prefix_match(m: "re.Match[str]", mask) -> str:
    glued = m.start() > 0 and m.string[m.start() - 1] in "-_"
    token = m.group(1)
    # Google OAuth prefixes are distinctive even when base64url separators break every long run.
    if glued and not token.startswith(("GOCSPX-", "ya29.")) and not _is_opaque_prefixed_token(token):
        return m.group(0)
    return mask(token)

# mask_secret strips EVERY control char (incl. \n/\t, C1, DEL, zero-width) so a
# masked secret never emits multiline or invisible bytes into display output.
_DISPLAY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f\x80-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064]")


def mask_secret(value: str, *, head: int = 4, tail: int = 4, floor: int = 12,
                placeholder: str = "***", empty: str = "") -> str:
    """Mask a secret for display (``hermes config`` / ``status`` / ``dump``):
    ``sk-p...7890``; shorter than ``floor`` (after control-byte stripping) →
    ``placeholder``; falsy → ``empty``."""
    value = _DISPLAY_CONTROL_RE.sub("", value) if value else value
    if not value:
        return empty
    return placeholder if len(value) < floor else f"{value[:head]}...{value[-tail:]}"


# Approval-prompt display integrity: the copy a human approves must show everything that will run.
# Code points that render as nothing or as a line break become visible escapes: every control,
# format, private-use, unassigned and surrogate code point (category C*, except tab/newline), line
# and paragraph separators (Zl/Zp), and blank-rendering marks/letters in other categories (variation
# selectors incl. U+E0100-E01EF, CGJ, Khmer/Mongolian selectors, Hangul fillers, Braille blank).
# Ordinary non-ASCII text is untouched. Padding that pushes the tail off screen or past a preview
# cut (space runs, whitespace-only lines) collapses to a marker; real indentation is kept.
_APPROVAL_NON_ASCII_RE = re.compile(r"[^\x20-\x7e\t\n]+")
_APPROVAL_BLANK_GLYPHS = frozenset(
    "\u034f\u115f\u1160\u17b4\u17b5\u180b\u180c\u180d\u180f\u2800\u3164\uffa0"
    + "".join(map(chr, range(0xFE00, 0xFE10))) + "".join(map(chr, range(0xE0100, 0xE01F0))))
_APPROVAL_SPACE = r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
# Mid-line runs (after visible text) of 21+, or line-leading runs past any sane indentation.
_APPROVAL_PAD_RE = re.compile(rf"(?<=\S){_APPROVAL_SPACE}{{21,}}|(?m:^){_APPROVAL_SPACE}{{65,}}")
# 3+ consecutive blank or whitespace-only lines; the match ends at a newline so the next line keeps its indent.
_APPROVAL_BLANK_RE = re.compile(rf"\n(?:{_APPROVAL_SPACE}*\n){{3,}}")
_APPROVAL_SAME_CHAR_RUN_RE = re.compile(r"(.)\1*", re.DOTALL)
# Prose mode (questions shown to a human, never executed) surfaces only what can hide or reorder
# text: controls, bidi embeddings/overrides/isolates, line/paragraph separators and padding. ZWJ,
# ZWNJ and variation selectors stay, because emoji and scripts such as Persian need them.
_PROSE_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _escape_approval_run(run: str, prose: bool = False) -> str:
    """Escape the invisible code points of a non-ASCII run; 4+ repeats of one show once with their count."""
    def one(m: re.Match) -> str:
        ch, n = m.group(1), len(m.group())
        cat = unicodedata.category(ch)
        if prose:
            hidden = cat == "Cc" or cat in ("Zl", "Zp") or ch in _PROSE_BIDI_CONTROLS
        else:
            hidden = ch in _APPROVAL_BLANK_GLYPHS or cat[0] == "C" or cat in ("Zl", "Zp")
        if not hidden:
            return m.group()
        esc = ch.encode("unicode_escape").decode("ascii")
        return f"⟨{esc} ×{n}⟩" if n >= 4 else esc * n
    return _APPROVAL_SAME_CHAR_RUN_RE.sub(one, run)


def sanitize_command_for_display(text: str | None, *, prose: bool = False) -> str:
    """Make hidden content in an approval-prompt string visible; clean text comes back unchanged.

    Display-only: the executed command, smart assessment and allowlist keys never see this copy.
    ``prose=True`` is the lighter pass for question text (see ``_PROSE_BIDI_CONTROLS``)."""
    if not text:
        return ""
    out = _APPROVAL_NON_ASCII_RE.sub(lambda m: _escape_approval_run(m.group(), prose), text)
    out = _APPROVAL_BLANK_RE.sub(lambda m: f"\n⟨+{m.group().count(chr(10)) - 1} blank lines⟩\n", out)
    return _APPROVAL_PAD_RE.sub(lambda m: f" ⟨+{len(m.group())} whitespace chars⟩ ", out)


def redact_approval_display(text: str | None, *, force: bool = False, prose: bool = False) -> str:
    """Secret redaction plus :func:`sanitize_command_for_display`, for every approval surface."""
    return sanitize_command_for_display(redact_sensitive_text(str(text or ""), force=force), prose=prose)


def _mask_token(token: str) -> str:
    """Mask a log token — 18-char floor, preserves 6 prefix / 4 suffix; empty → ``***``."""
    if not token:
        return "***"
    return mask_secret(token, head=6, tail=4, floor=18)


# Output of an earlier pass: masking it again would erase the kind it keeps visible. Only a
# value that IS a marker is skipped; ``***MyRealPass9`` or ``"«redacted:ghp_…» tail"`` still
# carries cleartext. An unquoted ``[REDACTED PRIVATE KEY]`` reaches here cut at its space.
_REDACTION_MARKER_RE = re.compile(r"\*{3}|«redacted(?::[^»\s]*|-secret)»|\[REDACTED(?: PRIVATE KEY\])?")


def _is_redaction_marker(match: "re.Match[str]", value: str) -> bool:
    if value == "[REDACTED":
        return match.string.startswith(" PRIVATE KEY]", match.end("value"))
    return _REDACTION_MARKER_RE.fullmatch(value) is not None


def _mask_flag_value(match: "re.Match[str]", mask=None) -> str:
    """Mask the value of a flag/connection-string key, keeping the flag and any quotes.

    ``mask`` is the caller's mask function: ``file_read`` passes the non-reusable sentinel,
    so a file read never hands back a head/tail value shaped like a usable key (#35519)."""
    value = match.group("value")
    stripped = value.strip("'\"")
    if not stripped or _is_redaction_marker(match, stripped.strip()):
        return match.group(0)
    # ``--password=$DB_PASS`` / ``--token=${TOKEN}`` name a variable; the reference is
    # not the credential and masking it would make the command unreplayable.
    if stripped[0] in "$<{%":
        return match.group(0)
    if stripped.isdigit():
        return match.group(0)  # ``--token 60`` is a timeout, not a credential
    if match.re is _CLI_SECRET_FLAG_RE:
        if "=" not in match.group("sep") and _is_flag_placeholder(value):
            return match.group(0)
        if _is_flag_path_reference(match.group("flag"), stripped):
            return match.group(0)
    masked = (mask or _mask_token)(stripped)
    start = match.start("value") - match.start(0)
    end = match.end("value") - match.start(0)
    return match.group(0)[:start] + masked + match.group(0)[end:]


# A space-separated flag is followed by whatever comes next, which in ``--help`` output and
# prose is a placeholder or a word (``--token TOKEN``, ``[--password PASSWORD]``, ``use
# --token to pass…``), the next flag (``psql --password -h db``) or a sub-option spec
# (``docker build --secret id=npmrc``). None of those is the credential.
# A metavar is upper case (``TOKEN``, ``DB_PASSWORD``), two short snake-case words
# (``db_name``) or a one-hump camelCase type name (cobra's ``stringArray``); a prose word
# is short and lower case. A passphrase is longer than that: a long lower-case word
# (``correcthorsebatterystaple``), three or more words (``correct_horse_battery_staple``,
# ``correctHorseBatteryStaple``) or anything carrying a digit or symbol is still masked.
_FLAG_PLACEHOLDER_RE = re.compile(
    r"(?:[A-Z][A-Z_]*|[a-z]{1,8}|[a-z]{1,8}_[a-z]{1,8}|[a-z]{1,8}[A-Z][a-z]{1,8})[\]).,:;]*"
    r"|[-\[(<].*|[a-z][\w-]*=.*"
    r"|\S*(?:xxx|XXX|\.\.\.)"  # a docs sample: ``--api-key sk-or-v1-xxx``
)


def _is_flag_placeholder(value: str) -> bool:
    """True for an unquoted space-separated flag value that is a word, not a credential."""
    return value[0] not in "'\"" and _FLAG_PLACEHOLDER_RE.fullmatch(value) is not None


def _is_flag_path_reference(flag: str, value: str) -> bool:
    """``--credentials ~/.config/gcloud/creds.json`` names a file, same rule as the ENV passes:
    only password-class flags mask a path, and only while no segment looks opaque."""
    if _PASSWORD_KEY_RE.search(flag) or not value.startswith(("/", "~", "./", "../")):
        return False
    return not any(_OPAQUE_PATH_SEGMENT_RE.fullmatch(seg) for seg in value.split("/"))


def _redact_cli_secrets(text: str, mask=None) -> str:
    """Mask credentials written as command-line flags and connection-string keys."""
    def sub(m: "re.Match[str]") -> str:
        return _mask_flag_value(m, mask)

    if "=" in text or " " in text:
        if _CLI_SECRET_FLAG_GATE.search(text):
            text = _CLI_SECRET_FLAG_RE.sub(sub, text)
        if _CLI_LOGIN_GATE.search(text):
            text = _CLI_LOGIN_P_RE.sub(sub, text)
    if _CONNSTR_GATE.search(text):
        text = _CONNSTR_SECRET_KEY_RE.sub(sub, text)
    return text


def _redact_private_key_blocks(text: str) -> str:
    """Whole private-key blocks (PEM, base64 PEM, PGP, PuTTY, age). Runs before the
    flag/connection-string passes: those stop a value at whitespace, so masking
    ``AccountKey=-----BEGIN`` first would break the PEM header and leave the body visible."""
    if "-----BEGIN" in text and "-----END" in text:  # no END, no match: skip the quadratic scan
        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    if "LS0tLS1CRUdJTi" in text:
        text = _B64_PEM_RE.sub(
            lambda m: "[REDACTED PRIVATE KEY]" if _is_b64_pem_key(m.group(0)) else m.group(0),
            text,
        )
    if "END PGP PRIVATE KEY BLOCK" in text:  # no END, no match: skip the quadratic scan
        text = _PGP_PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    if "Private-Lines:" in text:
        text = _PUTTY_PRIVATE_LINES_RE.sub(
            lambda m: m.group(1) + "[REDACTED PRIVATE KEY]" + m.group("eol"), text
        )
    if "AGE-SECRET-KEY-1" in text:
        text = _AGE_SECRET_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    return text


def _is_python_repr_secret_key(key: str) -> bool:
    """Return True for exact secret keys or credential-suffixed key names."""
    folded = key.casefold()
    if folded in _PYTHON_REPR_SECRET_KEYS:
        return True
    if key.isupper() and key.endswith(_PYTHON_REPR_ENV_SUFFIXES):
        return True
    # Mixed/camel-case keys ending in a credential word (``UserPassword``,
    # ``sessionToken``, ``clientApiKey``) — the exact-set and uppercase-suffix
    # rules above miss these. Suffix-only matching keeps metadata names like
    # ``TOKEN_COUNT`` / ``PASSWORD_POLICY`` / ``SECRET_NAME`` untouched.
    # Class widened per OpenHands/software-agent-sdk#4508 (their dict-entry
    # redaction was uppercase-only and leaked mixed-case keys).
    if folded.endswith(_PYTHON_REPR_CREDENTIAL_SUFFIXES):
        return True
    # Names ending in ``key`` where a qualifier says which kind of key it is:
    # ``aws_secret_access_key``, ``SECRET_KEY``, ``privateKey``, ``signing_key``,
    # ``encryption_key``, ``master_key``, ``hmac_key``. A bare ``key``, ``public_key``,
    # ``hotkey`` and ``key_id`` stay untouched — they name cache keys, limits and ids.
    return _is_qualified_key_name(key)


# A qualified key masks short values too, except a sample or switch that cannot be the
# credential: ``{"encryptKey": "true"}``, ``master_key: default``, ``"<your key>"``. Only a
# word-shaped bracket is a docs sample; ``<hunter2>`` / ``<Abc9!xyz>`` are still masked.
_CONFIG_PLACEHOLDER_VALUE_RE = re.compile(
    r"<[a-z]+(?:[ _-][a-z]+)+>|<(?:key|token|secret|password|value)>|true|false|yes|no|null|none|default",
    re.IGNORECASE)
_KEY_SECRET_QUALIFIERS = frozenset({
    "secret", "private", "signing", "encryption", "encrypt", "master", "hmac",
    "symmetric", "access",
})


def _key_name_words(key: str) -> list[str]:
    """Split a field name into lower-case words: ``aws_secret_access_key`` and
    ``awsSecretAccessKey`` both give ``[aws, secret, access, key]``."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def _is_qualified_key_name(key: str) -> bool:
    """True for a ``…key`` field name whose qualifier makes it a credential.

    ``access`` is in the qualifier set because AWS pairs an access-key id with the
    secret that signs with it, and the id alone still identifies an account; it is
    never a value safe to print in a chat approval card.
    """
    words = _key_name_words(key)
    if len(words) < 2 or words[-1] != "key":
        return False
    return any(w in _KEY_SECRET_QUALIFIERS for w in words[:-1])


def is_secret_field_name(key: object) -> bool:
    """True when a mapping field named ``key`` holds a credential — the repr-field policy, for callers
    that mask structured config by field instead of by text."""
    return isinstance(key, str) and _is_python_repr_secret_key(key)


def _redact_python_repr_fields(text: str) -> str:
    """Fully mask credential fields in Python mapping ``repr`` output."""
    def _sub(match: re.Match) -> str:
        key = match.group("key")
        if not _is_python_repr_secret_key(key):
            return match.group(0)

        single_value = match.group("single_value")
        if single_value is not None:
            prefix = match.group("single_prefix") or ""
            quote = "'"
            value = single_value
        else:
            prefix = match.group("double_prefix") or ""
            quote = '"'
            value = match.group("double_value")

        # Mapping repr can contain code-shaped fixture values too. Preserve
        # programmatic env lookups just like the ENV/JSON/YAML passes do.
        if _ENV_LOOKUP_VALUE_RE.match(value):
            return match.group(0)
        # An upstream pass (MCP probe header scrub, _mask_token) already masked this
        # value; re-masking would erase the scheme word it deliberately kept
        # (``'Authorization': 'Digest ***'`` → ``'***'``).
        if "***" in value or value.startswith("«redacted:"):
            return match.group(0)
        # Do not retain head/tail characters here: escaped repr atoms can cross
        # a slicing boundary and leave an unescaped quote behind. A full mask is
        # parseable for both str and bytes values and leaks no opaque bytes.
        return f"'{key}'{match.group('sep')}{prefix}{quote}***{quote}"

    return _PYTHON_REPR_FIELD_RE.sub(_sub, text)


def _redact_python_diagnostic_repr_fields(text: str) -> str:
    """Mask repr fields only on pytest/error lines in source-preserving output."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        ending = ""
        body_line = line
        if line.endswith("\r\n"):
            body_line, ending = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            body_line, ending = line[:-1], line[-1:]

        match = _PYTEST_DIAGNOSTIC_LINE_RE.match(body_line)
        if match is None:
            match = _PYTHON_EXCEPTION_LINE_RE.match(body_line)
        if match is not None:
            lines[index] = match.group("prefix") + _redact_python_repr_fields(match.group("body")) + ending
    return "".join(lines)


def _redact_query_string(query: str) -> str:
    """Replace values of sensitive ``k=v&k=v`` params with ``***``; others pass through."""
    if not query:
        return query
    return "&".join(
        f"{key}=***" if sep and key.lower() in _SENSITIVE_QUERY_PARAMS else pair
        for pair in query.split("&") for key, sep, _ in (pair.partition("="),)
    )


def _decode_url_param_name(name: str) -> str:
    decoded = name
    for _ in range(3):
        next_value = unquote_plus(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def _canonical_url_param_name(name: str) -> str:
    """Decode a URL parameter name (up to 3 unquote rounds) for case-insensitive matching."""
    folded = _decode_url_param_name(name).casefold()
    # Preserve policy names that are canonically hyphenated (for example
    # x-amz-signature) before accepting underscore-normalized aliases.
    return folded if folded in _SENSITIVE_QUERY_PARAMS else folded.replace("-", "_")


def _is_sensitive_url_param(name: str) -> bool:
    """A query-param name that holds a credential: the URL-only names above, plus the names the JSON-field
    rule masks (its closed list and the affixed-key policy: ``accessToken``, ``x-api-key``,
    ``private_token``, ``secret_key``, ``passwd``, ``api_key_value``), so both passes agree on what a
    secret is (#119852). Every ``redact_url_credentials=True`` caller shares this list, compaction
    summaries included, so a ``pageToken`` or ``X-Amz-Credential`` value is masked there too, as the
    JSON rule masks those fields."""
    if _canonical_url_param_name(name) in _SENSITIVE_QUERY_PARAMS:
        return True
    decoded = _decode_url_param_name(name).replace("-", "_")  # camelCase kept: _key_name_words splits on it
    # Same suffix drop as _json_affixed_key_secret: ``api_key_value`` / ``passwordValue`` hold the credential.
    return (bool(_JSON_KEY_NAMES_FULL_RE.fullmatch(decoded))
            or _is_python_repr_secret_key(_JSON_VALUE_SUFFIX_RE.sub("", decoded)))


def _redact_strict_url_credentials(text: str) -> str:
    """Strict egress-boundary redaction of URL credentials (absolute, relative and
    network references); preserves keys, separators, public params, hosts, paths."""
    text = _STRICT_URL_PARAM_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}=***"
        if _is_sensitive_url_param(m.group(2)) else m.group(0), text)
    return _STRICT_URL_USERINFO_RE.sub("//***:***@", text)


def redact_log_urls(text: str) -> str:
    """Mask credential-named query params and URL userinfo in log-bound text.

    A log file is a persistence boundary, not a live tool flow: magic-link /
    OAuth-callback URLs only need their non-secret parts to survive on disk, so
    the log path opts into the strict URL redactors that
    ``redact_sensitive_text`` deliberately leaves off for live flows. Without
    it, a remote MCP server configured as ``https://mcp.example.com/mcp?token=…``
    lands its API key on every log line that mentions the URL (issue #133713).
    """
    if not text or "://" not in text or not _redact_enabled():
        return text
    return _redact_strict_url_credentials(text)


def redact_cdp_url(value: object) -> str:
    """Mask secrets in a CDP/browser endpoint URL before it is logged.

    Unlike ``redact_sensitive_text`` (which passes web-URL query params and
    ``user:pass@`` through for OAuth callbacks / magic links), CDP discovery
    tokens are pure credentials, so this opts INTO both URL redactors.
    """
    text = redact_sensitive_text("" if value is None else str(value))
    if not text:
        return text
    text = _URL_WITH_QUERY_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}{m.group(3)}?{_redact_query_string(m.group(4))}{m.group(5) or ''}",
        text,
    )
    return _URL_USERINFO_RE.sub(lambda m: f"{m.group(1)}://{m.group(2)}:***@", text)


def _redact_form_body(text: str) -> str:
    """Redact sensitive values when the ENTIRE text is a clean ``k=v&k=v`` body."""
    if not text or "\n" in text or "&" not in text or not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())


def _mask_token_nonreusable(token: str) -> str:
    """Redact a prefix-matched credential to a NON-REUSABLE sentinel: no head/tail
    chars (an agent once wrote a truncated-looking mask back into a config file),
    only the vendor prefix label so the credential KIND stays visible.

    * cannot be mistaken for a usable-but-truncated key, so an agent that reads it from a config file and
    writes it back does NOT corrupt the stored credential into a dead 13-char string (issue #35519); and *
    still does not leak the secret material (no head/tail chars).
    """
    label = next((sub for sub in _PREFIX_SUBSTRINGS if token.startswith(sub)), "") if token else ""
    return f"«redacted:{label}…»" if label else "«redacted-secret»"


def _redact_json_string_value(m: re.Match, *, mask=_mask_token) -> str:
    """Mask a captured JSON string value without breaking its escapes."""
    encoded_value = m.group(2)
    try:
        value = json.loads(f'"{encoded_value}"')
    except json.JSONDecodeError:
        # The regex also accepts malformed escape pairs. Keep redaction
        # fail-safe while emitting a syntactically valid replacement value.
        value = encoded_value
    if not json_header_value_is_credential(m.group(1), value):
        return m.group(0)
    masked = json.dumps(mask(value), ensure_ascii=False)[1:-1]
    return m.group(1) + masked + m.group(3)


def _redact_unterminated_json_string_value(m: re.Match, *, mask=_mask_token) -> str:
    """Fail closed on a truncated JSON header value through end-of-line."""
    encoded_value = m.group(2)
    try:
        value = json.loads(f'"{encoded_value}"')
    except json.JSONDecodeError:
        value = encoded_value
    masked = json.dumps(mask(value), ensure_ascii=False)[1:-1]
    return m.group(1) + masked


def _assignment_sub(render, *, check_keyword: bool):
    """re.sub callback: keep the match unless the key/value pair (groups[0], groups[-1]) needs redaction."""
    def _sub(m):
        groups = m.groups()
        if not _should_redact_assignment(groups[0], groups[-1], check_keyword=check_keyword):
            return m.group(0)
        return render(groups)
    return _sub


_JSON_VALUE_SUFFIX_RE = re.compile(r"[_.\-]?(?:value|Value|VALUE)$")


def _json_affixed_key_secret(key: str, value: str) -> bool:
    """Whether ``"key": "value"`` holds a credential under an affixed secret-word key (#147240)."""
    if _JSON_KEY_NAMES_FULL_RE.fullmatch(key):  # the closed-list pass already decided this field
        return False
    if not (_has_word_bounded_keyword(key, _YAML_CFG_WORD_RE) or _has_password_keyword(key)):
        return False
    if _DSN_PW_PLACEHOLDER_RE.fullmatch(value) or not _should_redact_assignment(key, value, check_keyword=True):
        return False
    # A key that does not END in the credential word is metadata about one: ``passwordChanged``,
    # ``password_expires_at``, ``tokenUrl``, ``secret_id``, ``secret_arn``, ``PASSWORD_STORE_DIR``
    # carry timestamps, URLs and ids that look opaque, so the value shape cannot decide (#149435
    # review).
    # ``api_key_value`` / ``passwordValue`` hold the credential itself: drop that one suffix first.
    return _is_python_repr_secret_key(_JSON_VALUE_SUFFIX_RE.sub("", key))


def _redact_json_affixed_key(m: "re.Match[str]", mask) -> str:
    if not _json_affixed_key_secret(m.group(2), m.group(3)):
        return m.group(0)
    return f'{m.group(1)}: "{mask(m.group(3))}"'


def _redact_json_affixed_key_escaped(m: "re.Match[str]", mask) -> str:
    if not _json_affixed_key_secret(m.group(3), m.group(4)):
        return m.group(0)
    return f'{m.group(1)}: {m.group(2)}"{mask(m.group(4))}{m.group(2)}"'


def _redact_assignments(text: str, *, mask_nonreusable: bool = False) -> str:
    """ENV / config / JSON / YAML assignment passes (skipped for code files). Passes
    that would match ``token=``/``key=`` URL params skip ``://`` text (web-URL query
    params are intentionally passed through, see redact_sensitive_text).

    ``mask_nonreusable`` masks the assignments with the ``«redacted:…»`` sentinel that
    ``file_read=True`` already uses for prefix-matched credentials. Without it an agent
    that read a secret-bearing file would hold a head/tail mask shaped like a real but
    truncated key and could write it back as a dead credential (#35519)."""
    mask = _mask_token_nonreusable if mask_nonreusable else _mask_token
    if "=" in text:
        text = redact_cookie_assignments(text, mask)
        if "://" not in text and _CFG_SECRET_WORD_RE.search(text):  # before the ENV passes: their ``\S+``
            text = redact_spaced_cfg_assignments(text, mask)        # value stops at a quoted value's space
        _redact_env = _assignment_sub(lambda g: f"{g[0]}={g[1]}{mask(g[2])}{g[1]}", check_keyword=True)
        text = _ENV_ASSIGN_RE.sub(_redact_env, text)
        if "://" not in text:  # lowercase names would match URL params
            # Skip URLs — the query string may contain ``token=``/``key=`` params that are intentionally
            # passed through (see note near the bottom of this function; _redact_strict_url_credentials
            # handles the opt-in case). The uppercase regex above is all-caps-only, so it never matches URL
            # params; the lowercase one would (issue #77484).
            text = _ENV_ASSIGN_LOWER_RE.sub(_redact_env, text)
        # The keyword pre-gate is exact and matters: _CFG_DOTTED_RE backtracks
        # quadratically on long unbroken [A-Za-z0-9_.\-] runs.
        # Lowercase/dotted config keys (issue #16413). Skip URLs entirely — web-URL query params are
        # intentionally passed through (see note near the bottom of this function); _DB_CONNSTR_RE still
        # guards connection-string passwords. Extra gate: every _CFG_*_RE match requires a secret keyword in
        # the key, so a text without any secret keyword cannot match — skipping is exact. This matters
        # because _CFG_DOTTED_RE backtracks quadratically on long unbroken [A-Za-z0-9_.\-] runs (e.g.
        # base64/hex blobs in compaction payloads); the linear keyword scan prevents that pathological path
        # on secret-free text.
        if "://" not in text and _CFG_SECRET_WORD_RE.search(text):
            text = _CFG_DOTTED_RE.sub(_redact_env, text)
            text = _CFG_ANCHORED_RE.sub(_redact_env, text)
            text = redact_spaced_cfg_assignments(text, mask)

    if ":" in text and '"' in text:
        for pattern in (_SECRET_HEADER_JSON_RE, _COOKIE_HEADER_JSON_RE):
            text = pattern.sub(lambda m: _redact_json_string_value(m, mask=mask), text)
        for pattern in (_SECRET_HEADER_JSON_UNTERMINATED_RE, _COOKIE_HEADER_JSON_UNTERMINATED_RE):
            text = pattern.sub(lambda m: _redact_unterminated_json_string_value(m, mask=mask), text)
        redact_json = _assignment_sub(lambda g: f'{g[0]}: "{mask(g[1])}"', check_keyword=False)
        text = _JSON_FIELD_RE.sub(
            lambda m: m.group(0) if m.group(1)[1:-1].lower() == "apikey" else redact_json(m), text)
        if '\\"' in text:
            text = _JSON_FIELD_ESCAPED_RE.sub(
                _assignment_sub(lambda g: f'{g[0]}: {g[1]}"{mask(g[2])}{g[1]}"', check_keyword=False), text)
        text = _JSON_ANY_KEY_FIELD_RE.sub(lambda m: _redact_json_affixed_key(m, mask), text)
        if '\\"' in text:
            text = _JSON_ANY_KEY_FIELD_ESCAPED_RE.sub(lambda m: _redact_json_affixed_key_escaped(m, mask), text)

    # Python mapping repr fields ({'API_KEY': '…'}): single-quoted, so the JSON rule
    # above never sees them — the traceback / pytest-introspection leak shape.
    if ":" in text and "'" in text:
        text = _redact_python_repr_fields(text)

    # YAML after JSON; unwrap scalars before the shared assignment gate so
    # quoted programmatic env lookups retain the same exemption as bare ones.
    if ":" in text and "://" not in text:
        def _redact_yaml(m):
            key, sep, value = m.groups()
            quote = value[0] if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0] else ""
            inner = value[1:-1] if quote else value
            # An empty or docs-placeholder scalar (``""``, ``"<your key>"``, ``"${API_KEY}"``) is not
            # a credential; masking it hides what the config expects, as _redact_db already rules.
            if quote and (not inner or _DSN_PW_PLACEHOLDER_RE.fullmatch(inner)
                          or (inner.startswith("{{") and inner.endswith("}}"))):  # Helm/Jinja template
                return m.group(0)
            if not _should_redact_assignment(key, inner, check_keyword=True):
                return m.group(0)
            return f"{key}{sep}{quote}{mask(inner)}{quote}"
        text = _YAML_ASSIGN_RE.sub(_redact_yaml, text)
    return text


# A docs placeholder is a lower-case or ALL-CAPS name in angle brackets (``<password>``) or an
# upper-case env var reference (``${REDIS_PASSWORD}``); ``$ecretPassw0rd`` or ``<Zq8x…>`` is
# a real password that happens to start with the same character.
_DSN_PW_PLACEHOLDER_RE = re.compile(r"<(?:[a-z][a-z_ -]*|[A-Z][A-Z_ -]*)>|\$\{?[A-Z_][A-Z0-9_]*\}?|%[A-Z_][A-Z0-9_]*%")


# ``https://user:password@host`` in docs names the field instead of filling it in.
_USERINFO_PLACEHOLDER_WORDS = frozenset({"password", "passwd", "pass", "pw", "pwd", "secret", "token"})


def _is_userinfo_placeholder(password: str) -> bool:
    """A template (``{pw}``, ``{{ pw }}``), docs placeholder or field name, never a literal secret."""
    return ((password.startswith("{") and password.endswith("}"))
            or _DSN_PW_PLACEHOLDER_RE.fullmatch(password) is not None
            or password.casefold() in _USERINFO_PLACEHOLDER_WORDS)


def _redact_userinfo_passwords(text: str) -> str:
    """Mask the password half of ``scheme://user:pass@host`` userinfo (#125664).

    #34029's pass-through was about query-string round-trip tokens (magic links, OAuth callbacks,
    pre-signed URLs live in the query, never in userinfo); the userinfo password is a credential —
    the same reasoning #6396 used to mask the bare-token form. Keeps scheme, username and host
    visible. Templates and docs samples (``{pw}``, ``${TOKEN}``, ``password``) stay as written."""
    def _sub(m):
        user, sep, password = m.group(2).partition(":")
        # f-string / Jinja templates and docs samples (``user:{pw}@``, ``user:${TOKEN}@``,
        # ``user:password@``) in prose and logs as much as in code files.
        if not sep or not password or _is_userinfo_placeholder(password):
            return m.group(0)
        return f"{m.group(1)}{user}:***@"
    return _URL_USERINFO_PAIR_RE.sub(_sub, text)


def _redact_url_credentials(text: str, code_file: bool, *, file_read: bool = False) -> str:
    """DB connection-string passwords, incoming-webhook secrets, bare-token and ``user:pass`` URL
    userinfo (``://`` text only)."""
    def _redact_db(m):
        # code_file: a pure ``{...}`` password is an f-string template reference
        # (f"postgresql://{user}:{pass}@{host}"), not a literal credential.
        pw = m.group(2)
        if code_file and pw.startswith("{") and pw.endswith("}"):
            return m.group(0)
        # Password-only URLs in docs: ``redis://:<password>@`` / ``redis://:${REDIS_PASSWORD}@``.
        if m.group(1).endswith("://:") and _DSN_PW_PLACEHOLDER_RE.fullmatch(pw):
            return m.group(0)
        return f"{m.group(1)}***{m.group(3)}"
    def _redact_webhook(m):
        # file_read: the non-reusable sentinel, so an agent that reads a stored webhook out of a
        # config file cannot write a truncated-looking mask back over it (#35519).
        mask = _mask_token_nonreusable if file_read else _mask_token
        groups = m.groups()
        for prefix, secret in zip(groups[::2], groups[1::2]):
            if prefix:
                return f"{prefix}{mask(secret)}"
        return m.group(0)

    text = _DB_CONNSTR_RE.sub(_redact_db, text)
    text = _WEBHOOK_URL_RE.sub(_redact_webhook, text)
    text = _URL_BARE_TOKEN_RE.sub(lambda m: f"{m.group(1)}{_mask_token(m.group(2))}{m.group(3)}", text)
    return _redact_userinfo_passwords(text)


def _mask_network_path_userinfo(m) -> str:
    """``//user:pass@host`` without a scheme (#125664 residual); placeholders stay as written."""
    if _is_userinfo_placeholder(m.group(3)):
        return m.group(0)
    return f"{m.group(1)}{m.group(2)}:***@"


def _redact_phone(m):
    phone = m.group(1)
    keep = 2 if len(phone) <= 8 else 4
    return phone[:keep] + "****" + phone[-keep:]


def redact_sensitive_text(text: str, *, force: bool = False, code_file: bool = False,
                          file_read: bool = False, secret_file: bool = False,
                          redact_url_credentials: bool = False, keep_phone_numbers: bool = False) -> str:
    """Apply all redaction patterns to a block of text.

    Safe on any string. Enabled by default (``security.redact_secrets: false``
    disables); ``force=True`` is for safety boundaries that must never return
    raw secrets regardless.

    ``redact_url_credentials=True``: also redact credential-named query params
    and the entire URL userinfo (``***:***@``) — off by default because OAuth-callback /
    magic-link / pre-signed URLs must survive ordinary tool flows unchanged.
    ``code_file=True``: skip the ENV/JSON assignment passes for known source
    code (``MAX_TOKENS=***``, ``"apiKey": "test"`` fixtures). ``file_read=True``
    (implies code_file unless ``secret_file``): prefix-matched credentials become a
    non-reusable sentinel (``«redacted:ghp_…»``) instead of a head/tail mask an agent
    could write back into config.yaml as a dead credential.
    Every regex sits behind a cheap substring gate that its pattern requires,
    so the gates are never false-negative.

    Set code_file=True to also skip the Python-repr mapping pass (``{'API_KEY': '…'}``
    fixtures in source); pytest/exception diagnostic lines get a narrow pass in
    redact_terminal_output instead.

    Set file_read=True for file *content* returned to the agent (read_file / search_files / cat). The old
    mask looked like a real-but-truncated key, so an agent reading it from config.yaml and writing it back
    silently corrupted the stored credential into a dead 13-char value → 401 (issue #35519). The sentinel is
    syntactically invalid as a token, so it can't be mistaken for a usable key or written back as one.

    Set ``secret_file=True`` when the caller has already classified the SOURCE as secret-bearing
    (``_is_secret_file_arg``: command reads on the terminal side, resolved tool paths on the
    file side). It re-enables the ENV/JSON/YAML assignment passes that ``file_read`` would
    otherwise skip, so an opaque prefix-less credential assigned to a credential-shaped key is
    masked instead of passed through in cleartext (issue #110567). With ``file_read=True`` those
    assignments are masked with the non-reusable sentinel, so the #35519 write-back hazard stays
    closed. ``secret_file`` is authoritative: it wins over ``code_file``, so a caller cannot be
    fail-open by setting both. Files that are not secret-bearing (any source file, a project's own
    ``config.yaml``) keep the code_file behaviour: ``MAX_TOKENS: 100`` and ``"apiKey": "test"``
    fixtures are untouched.
    """
    if text is None:
        return None
    text = text if isinstance(text, str) else str(text)
    if not text:
        return text
    # Vault secrets are a hard model-egress boundary: scrubbed regardless of the redact_secrets preference.
    text = redact_registered_vault_values(text)
    if not (force or _redact_enabled()):
        return text
    # ``secret_file`` is authoritative: a caller that classified the source as secret-bearing must not
    # be silently fail-open because another flag (code_file, or file_read implying it) was also set.
    code_file = (code_file or file_read) and not secret_file

    # Control/zero-width chars can split a token body so _PREFIX_RE alone misses it.
    if _has_known_prefix_substring(text):
        _prefix_sub = _mask_token_nonreusable if file_read else _mask_token
        # Control/zero-width chars (\\n, \\r, ESC, U+200B, …) split a token body so _PREFIX_RE cannot match
        # across them — a secret smuggled as ``sk-abc\\x1bdef…`` leaks verbatim (issue #77484). Mask such
        # runs by first matching on a control-stripped copy, then re-masking the corresponding span in the
        # original (the stripped copy and the original are aligned 1:1 for non-control chars).
        text = mask_control_split_tokens(text, _prefix_sub, _PREFIX_RE)
        text = _PREFIX_SUB_RE.sub(lambda m: _mask_prefix_match(m, _prefix_sub), text)

    if "." in text:
        _mask = _mask_token_nonreusable if file_read else _mask_token
        text = _ZHIPU_API_KEY_RE.sub(lambda m: _mask(m.group(1)), text)
        text = _redact_discord_tokens(text, _mask)

    if not code_file:
        text = _redact_assignments(text, mask_nonreusable=file_read)

    # Flags and connection-string keys carry a credential in command text, so they are not
    # gated on code_file: the shape (``--password=`` / ``;AccountKey=``) is unambiguous, and
    # the surfaces that leaked them (approval cards, chat status) run this pass anyway.
    text = _redact_private_key_blocks(text)
    text = _redact_cli_secrets(text, _mask_token_nonreusable if file_read else None)

    if force and "=" in text:
        # ``NAME=user:pass`` under an innocuous name (#125664): shape-only, so force-only — every
        # force=True caller (persistence boundary, debug dumps, session export) is a safety boundary
        # that must not return raw secrets; ordinary flows keep today's pass-through.
        def _basic_pair(m):
            password = m.group(4)
            if password is None or not _BASIC_PAIR_NAME_RE.search(m.group(1)) \
                    or not _is_basic_pair_password(password):
                return m.group(0)
            # ``CHECKSUM=sha256:<hex>``: an algorithm:digest label, not a user:credential pair.
            if m.group(3).lower() in _HASH_ALGO_NAMES:
                return m.group(0)
            # A template or docs placeholder, not a literal credential — the rule
            # _redact_userinfo_passwords applies below.
            if _is_userinfo_placeholder(password):
                return m.group(0)
            return f"{m.group(1)}={m.group(2)}{m.group(3)}:***{m.group(2)}"
        text = _BASIC_PAIR_ASSIGN_RE.sub(_basic_pair, text)

    if _AUTH_GATE_RE.search(text):  # cheapest gate over every casing
        text = _AUTH_HEADER_RE.sub(_auth_header_sub, text)

    if ":" in text:
        text = _redact_secret_headers(text)
        text = _TELEGRAM_RE.sub(_redact_telegram_token, text)

    # Database connection string passwords. With code_file=True, a password group that is a pure ``{...}``
    # brace expression is an f-string template reference (e.g. f"postgresql://{user}:{pass}@{host}"), not a
    # literal credential — preserve it. Literal passwords are still redacted. The regex forbids whitespace
    # in the password group, so a single-line template's group(2) is exactly the brace expression. See issue
    # #33801.
    if "://" in text:
        text = _redact_url_credentials(text, code_file, file_read=file_read)
    if force and "//" in text:
        text = _NETWORK_PATH_USERINFO_RE.sub(_mask_network_path_userinfo, text)

    # Keyword/value connection strings (``Host=…;Password=…``, libpq ``host=… password=…``):
    # the forms the URL pass above never sees. Gate: a ``password=``/``pwd=`` key in any casing;
    # the per-line companion-key check in _redact_kv_connection_strings keeps prose out.
    if _KV_CONNSTR_GATE_RE.search(text):
        text = _redact_kv_connection_strings(text, code_file)

    if "eyJ" in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)

    # A Webapp sign-in link signs a browser in as the user, so its credential is masked whatever the
    # web-URL policy below.
    if "hermes-session=" in text or "/webapp/launch#" in text:
        text = _WEBAPP_LINK_CREDENTIAL_RE.sub(lambda m: m.group(1) + "***", text)

    if redact_url_credentials:  # opt-in; known credential shapes in URLs are caught above
        # NOTE: Web-URL redaction (query params + userinfo + HTTP access-log request targets) is
        # intentionally OFF. Many legitimate workflows pass opaque tokens through query strings — magic-link
        # checkouts, OAuth callbacks the agent is meant to follow, pre-signed share URLs — and
        # blanket-redacting param values by name breaks those skills mid-flow. DB connection-string
        # passwords are still caught by _DB_CONNSTR_RE. The ONE userinfo case still redacted is the
        # colon-less bare-token form ``scheme://TOKEN@host`` (#6396, handled by _URL_BARE_TOKEN_RE in the
        # ``://`` block above): a bare credential in userinfo is never a round-trip workflow token (those
        # live in the query string), so masking it can't break a skill. The ``user:pass@`` form is left to
        # pass through per #34029.
        text = _redact_strict_url_credentials(text)

    if "&" in text and "=" in text:
        text = _redact_form_body(text)

    if "+" in text and not keep_phone_numbers:  # opt-out: agent.redact_phone_policy
        kept = non_phone_offsets(text)
        text = _SIGNAL_PHONE_RE.sub(lambda m: m.group(0) if m.start() in kept else _redact_phone(m), text)

    return text


def redact_derived_text(text: object) -> str:
    """Strictly scrub model-visible text derived from stored conversation data.

    Derived summaries and search metadata can re-enter unrelated prompts, so
    the user's ordinary redaction opt-out must not expose secrets through them.
    """
    return redact_sensitive_text(
        "" if text is None else str(text),
        force=True,
        redact_url_credentials=True,
    )


# Commands whose stdout is an env-var dump: terminal redaction runs the
# ENV-assignment pass (code_file=False) for these so opaque tokens with no vendor
# prefix are masked; everything else uses code_file=True (``MAX_TOKENS=100``).
# Commands whose stdout is an environment-variable dump (KEY=value lines), NOT source code.
# ``MY_SERVICE_TOKEN=abc123randomstring``) are still masked. For all other commands, code_file=True is used
# to avoid mangling legitimate source/config dumps (``MAX_TOKENS=100``, ``"apiKey": "x"`` fixtures,
# ``postgresql://{user}`` f-string templates). See issue #43025.
_ENV_DUMP_COMMANDS = frozenset({"env", "printenv", "set", "export", "declare"})

# Commands that read file contents to stdout, plus the filter readers (``grep``/``awk``/``sed``)
# the model reaches for on config files. A secret-bearing target (``.env`` per AGENTS.md,
# a shell rc/profile, Hermes' own ``config.yaml`` where ``hermes mcp add --env`` writes
# tokens) is a credential dump, so the ENV/YAML assignment pass must run. Arbitrary
# ``config.yaml`` / source files stay on the code_file path (``MAX_TOKENS: 100``).
_FILE_READ_COMMANDS = frozenset({
    "cat", "head", "tail", "type", "bat", "less", "more", "nl",
    "zcat", "tac", "view", "batcat", "grep", "awk", "sed",
})
_SHELL_RC_BASENAMES = frozenset({
    ".bashrc", ".bash_profile", ".bash_login", ".profile",
    ".zshrc", ".zprofile", ".zlogin", ".zshenv",
})
# Filter readers take a PATTERN/program as their first positional; only the operands after
# it are files, so ``grep .bashrc app.py`` must not gate on the pattern.
_PATTERN_FIRST_COMMANDS = frozenset({"grep", "awk", "sed"})
_HERMES_HOME_PREFIXES = ("$HERMES_HOME/", "${HERMES_HOME}/")
# ``$HOME/.hermes/config.yaml`` keeps the ``.hermes`` segment, so stripping the prefix is
# enough to gate it; ``~/`` already survives the ``$``-bearing-path bail-out.
_HOME_PREFIXES = ("$HOME/", "${HOME}/")


def _is_under_hermes_home(path: str) -> bool:
    """True when an absolute ``config.yaml`` path sits under the active Hermes home or root.

    The default home's basename is an installation detail — ``.hermes`` on POSIX, ``hermes``
    under ``AppData/Local`` on Windows — and a resolved path never spells ``$HERMES_HOME``,
    so the literal-segment test in ``_is_secret_file_arg`` cannot see a native Windows path.
    Compare against the resolved homes instead. Only reached for a ``config.yaml`` basename,
    so the resolve cost stays off the per-token command scan.
    """
    from agent.file_safety import _hermes_dirs
    from tools.environments.sandbox_scope import sandbox_lexical_path
    # A hosted path names a file in the sandbox: judged as written (never empty), not resolved on this host.
    try:
        target = os.path.normcase(sandbox_lexical_path(path) or os.path.realpath(os.path.expanduser(path)))
    except (OSError, ValueError):
        return False
    for home in _hermes_dirs():
        try:
            base = os.path.normcase(os.path.realpath(str(home)))
        except (OSError, ValueError):
            continue
        if target == base or target.startswith(base + os.sep):
            return True
    return False


def _is_secret_file_arg(arg: str) -> bool:
    """``.env``-style or shell rc basename anywhere; ``config.yaml`` only under a
    ``.hermes`` directory, ``$HERMES_HOME``, or the resolved Hermes home (never arbitrary
    YAML). The resolved-home arm is what covers native Windows, where the home directory
    is ``%LOCALAPPDATA%\\hermes`` and carries no ``.hermes`` segment."""
    path = arg.strip("\"'").replace("\\", "/")
    hermes_home = False
    for prefix in _HERMES_HOME_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            hermes_home = True
            break
    for prefix in _HOME_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    if "$" in path:
        return False
    parts = [part.lower() for part in path.split("/") if part]
    if not parts:
        return False
    if parts[-1] in _ENV_FILE_BASENAMES or parts[-1] in _SHELL_RC_BASENAMES:
        return True
    # ``config.yaml`` plus the ``config.yaml.good.<stamp>`` / ``.corrupt.<stamp>`` copies Hermes
    # writes under ``backups/config/`` — same contents, same secrets.
    if parts[-1] != "config.yaml" and not parts[-1].startswith(("config.yaml.good.", "config.yaml.corrupt.")):
        return False
    return hermes_home or ".hermes" in parts[:-1] or _is_under_hermes_home(path)


def _command_reads_secret_file(command: str | None) -> bool:
    """True if ``command`` reads a secret-bearing file (see ``_is_secret_file_arg``) to
    stdout. Defense-in-depth, not a boundary: indirect reads (``sudo cat .env``, ``$(cat
    .env)``, unresolved variable paths) are not detected, matching ``is_env_dump_command``."""
    if not command or not isinstance(command, str):
        return False
    for seg in _command_segments(command):
        tokens = seg.split()  # not shlex: it mangles Windows paths (``C:\Users\...\.env``)
        if not tokens:
            continue
        reader = tokens[0].rsplit("/", 1)[-1].lower()
        if reader not in _FILE_READ_COMMANDS:
            continue
        positional = [arg for arg in tokens[1:] if not arg.startswith("-")]
        if reader in _PATTERN_FIRST_COMMANDS:
            positional = positional[1:]
        if any(_is_secret_file_arg(arg) for arg in positional):
            return True
    return False


def is_env_dump_command(command: str | None) -> bool:
    """True if any pipeline/sequence segment starts with an _ENV_DUMP_COMMANDS
    token. Conservative: unrecognized → False (callers fall back to code_file=True)."""
    if not command or not isinstance(command, str):
        return False
    for seg in _command_segments(command):
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens and tokens[0] in _ENV_DUMP_COMMANDS:
            return True
    return False


REDACTION_UNAVAILABLE = "[redaction-unavailable]"
# The opaque branch needs a 20-char floor (the floor the gateway/A2A sweeps always had): without it the
# English word "bearer" turns "the bearer of bad news" into "Bearer [redacted] bad news" on every chat
# reply. The bracket branch folds an already-masked residue ("Bearer [redacted-jwt]") to one marker.
# ASCII lookbehind (not Unicode ``\b``): a Unicode word neighbor can hide ``Bearer <opaque>``
# the same way prefix tokens used to hide behind ``\b``.
_BEARER_RESIDUE_RE = re.compile(
    r"(?<![A-Za-z0-9_])Bearer\s+(?:\[[^\]]+\]|[A-Za-z0-9._~+/-]{20,}=*)",
    re.IGNORECASE,
)


def redact_for_egress(text: str, *, urls: bool = False, keep_phone_numbers: bool = False) -> str:
    """The one scrub for text leaving the process for a remote reader (chat platforms, A2A peers,
    telemetry). ``redact_sensitive_text(force=True)`` — the only secret-pattern list — plus a bearer
    sweep, because a ``Bearer <opaque>`` value with no vendor prefix carries no shape the prefix
    matcher can key on. Fails CLOSED: if the redactor raises, the raw text is never returned.
    ``keep_phone_numbers`` skips only the E.164 pass (``redact_for_chat_egress``'s opt-in).

    ``urls=True`` also masks credential-named query params and URL userinfo — for archival sinks
    (memory providers, telemetry) where no one follows the URL. Chat delivery leaves it off so a magic
    link or OAuth URL the agent hands the user still works."""
    text = str(text or "")
    try:
        text = redact_sensitive_text(text, force=True, redact_url_credentials=urls,
                                     keep_phone_numbers=keep_phone_numbers)
    except Exception:
        return REDACTION_UNAVAILABLE
    if "earer" in text.lower():  # IGNORECASE regex; the gate must fold case too (#108807)
        text = _BEARER_RESIDUE_RE.sub("Bearer [redacted]", text)
    return text


def redact_terminal_output(output: str, command: str | None = None, *, force: bool = False) -> str:
    """Single redaction policy for ALL terminal-output surfaces: the ENV/YAML-assignment
    pass runs only when ``command`` is an env dump or reads a secret-bearing file (``.env``,
    shell rc, Hermes ``config.yaml``); otherwise code_file=True avoids false positives on
    source/config dumps."""
    if not output:
        return output
    code_file = not (is_env_dump_command(command) or _command_reads_secret_file(command))
    redacted = redact_sensitive_text(output, force=force, code_file=code_file)
    # Source-preserving output still gets the Python-repr pass on high-confidence
    # diagnostic lines (pytest ``E   `` introspection, final exception lines): that is
    # where {'BRAVE_API_KEY': '…'} leaks, not in source dumps.
    if code_file and (force or _redact_enabled()) and ":" in redacted and "'" in redacted:
        redacted = _redact_python_diagnostic_repr_fields(redacted)
    return redacted



# execute_code output is script stdout, not a command line, so redact_terminal_output's command
# classifier never sees an env dump there (#95509). A whole line of ``NAME=value`` (optionally
# ``export``-prefixed, no space around ``=``) is the ``.env`` / ``env`` shape; source assignments
# (``MAX_TOKENS = 100``, indented or embedded code) do not match and keep the code_file behaviour.
_ENV_DUMP_LINE_RE = re.compile(r"^((?:export[ \t]+)?)([A-Z][A-Z0-9_]*)=(\S.*)$", re.MULTILINE)


def _mask_env_dump_line(m: "re.Match[str]", mask_env) -> str:
    # The whole line is one assignment, so its value is the rest of the line: ``env`` prints
    # ``DB_PASSWORD=first second`` raw and a .env quotes it. _ENV_ASSIGN_RE's ``\S+`` would stop at
    # the first space and leave the remaining words visible.
    prefix, key, value = m.groups()
    quote = value[0] if len(value) > 1 and value[0] in "'\"" and value[-1] == value[0] else ""
    inner = value[1:-1] if quote else value
    if _should_redact_assignment(key, inner, check_keyword=True):
        return f"{prefix}{key}={quote}{_mask_token(inner)}{quote}"
    return _ENV_ASSIGN_RE.sub(mask_env, m.group(0))


def redact_code_output(output: str, *, force: bool = False) -> str:
    """Redaction for execute_code stdout/stderr: code_file=True, plus the ENV-assignment pass on
    lines shaped like an env dump so opaque prefix-less credentials are masked."""
    redacted = redact_sensitive_text(output, force=force, code_file=True)
    if not redacted or "=" not in redacted or not (force or _redact_enabled()):
        return redacted
    mask_env = _assignment_sub(lambda g: f"{g[0]}={g[1]}{_mask_token(g[2])}{g[1]}", check_keyword=True)
    return _ENV_DUMP_LINE_RE.sub(lambda m: _mask_env_dump_line(m, mask_env), redacted)

# --- Prefix pre-screen: derived from _PREFIX_PATTERNS so a new prefix can't
# silently break the gate (every match contains its pattern's literal prefix).

def _extract_literal_prefix(pattern: str) -> str:
    """Leading literal chars of a regex (up to the first metacharacter)."""
    meta = "[(\\.?*+|{^$"
    for i, ch in enumerate(pattern):
        if ch in meta:
            return pattern[:i]
    return pattern


def _has_top_level_alternation(pattern: str) -> bool:
    return _pattern_structure(pattern)[0]


def _has_nested_unbounded_repeat(pattern: str) -> bool:
    return _pattern_structure(pattern)[1]


_PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in _PREFIX_PATTERNS)


def _has_known_prefix_substring(text: str) -> bool:
    """Cheap pre-check before the expensive ``_PREFIX_RE``."""
    return any(p in text for p in _PREFIX_SUBSTRINGS)


# --- Plugin-registered redaction patterns -----------------------------------
# ADDITIVE-ONLY: a plugin can extend what gets masked but cannot weaken a
# built-in, so it can only over-redact. Keyed by registration source so plugin
# unload has a clean seam to drop ONE plugin's patterns.
# There is deliberately no public removal API — additive-only stands; unload is a host-owned lifecycle
# concern. See #64229.
_PLUGIN_PREFIX_PATTERNS: dict = {}
_registry_lock = threading.Lock()


def _plugin_patterns() -> list:
    """All plugin-registered patterns in registration order."""
    return [p for patterns in _PLUGIN_PREFIX_PATTERNS.values() for p in patterns]


def _rebuild_prefix_matcher() -> None:
    """Recompile the prefix alternation and pre-screen substrings; callers read the
    module globals at call time, so the swap propagates immediately."""
    global _PREFIX_RE, _PREFIX_SUB_RE, _PREFIX_SUBSTRINGS
    combined = _PREFIX_PATTERNS + _plugin_patterns()
    _PREFIX_RE = _compile_prefix_matcher(combined)
    _PREFIX_SUB_RE = _compile_prefix_sub_matcher(combined)
    _PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in combined)


# Structural validators for register_redaction_patterns, in check order:
# (predicate -> reject when True, warning message with (source, pattern) args).
_PATTERN_REJECT_RULES = (
    (_has_top_level_alternation,
     "%s: skipping redaction pattern %r — top-level alternation escapes the literal-prefix "
     "guarantee (in 'ab|.*' the prefix binds only the first branch); wrap alternation in "
     "a group after the prefix, e.g. 'ab(?:x|y)'"),
    (_has_nested_unbounded_repeat,
     "%s: skipping redaction pattern %r — nested unbounded quantifiers (e.g. '(a+)+') can "
     "backtrack catastrophically, and registered patterns run on every log line and tool output"),
    (lambda pattern: len(_extract_literal_prefix(pattern)) < 2,
     "%s: skipping redaction pattern %r — must start with at least 2 literal characters "
     "(needed for the pre-screen substring gate)"),
)


def register_redaction_patterns(patterns, source: str = "plugin") -> int:
    """Additively register credential-token regexes; returns the count accepted.

    Invalid entries (non-compiling, top-level alternation, nested unbounded
    quantifiers, < 2 literal prefix chars) and duplicates are warned/skipped,
    never raised — a broken plugin must not break startup.
    """
    accepted = []
    for pattern in patterns or []:
        if not isinstance(pattern, str) or not pattern.strip():
            logger.warning("%s: skipping empty/non-string redaction pattern", source)
            continue
        pattern = pattern.strip()
        try:
            re.compile(pattern)
        except re.error as exc:
            logger.warning("%s: skipping invalid redaction pattern %r (%s)", source, pattern, exc)
            continue
        rejected = next((message for reject, message in _PATTERN_REJECT_RULES if reject(pattern)), None)
        if rejected:
            logger.warning(rejected, source, pattern)
            continue
        if pattern in _PREFIX_PATTERNS or pattern in _plugin_patterns() or pattern in accepted:
            logger.debug("%s: redaction pattern %r already registered", source, pattern)
            continue
        accepted.append(pattern)

    if accepted:
        with _registry_lock:
            _PLUGIN_PREFIX_PATTERNS.setdefault(source, []).extend(accepted)
            _rebuild_prefix_matcher()
        logger.info("%s: registered %d redaction pattern(s)", source, len(accepted))
    return len(accepted)


def _reset_plugin_redaction_patterns() -> None:
    """Drop all plugin-registered patterns (tests/teardown only)."""
    with _registry_lock:
        _PLUGIN_PREFIX_PATTERNS.clear()
        _rebuild_prefix_matcher()


# Control chars + Unicode line separators (NEL, LS, PS) that log aggregators treat as breaks:
# interpolated text must not forge a log line.
_LOG_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")
# Same class minus "\n": the appended traceback/stack keeps its line structure.
_EXC_UNSAFE_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f\x85\u2028\u2029]")
# hermes_cli/logs.py _TS_RE (searched anywhere in a line) makes any stamped line a record.
# Zero-width context so overlapping stamps ("12:00:2026-07-09 12:00:00") are all broken.
_LOG_LINE_STAMP = re.compile(r"(?<=\d{4}-\d{2}-\d{2})[T ](?=\d{2}:\d{2}:\d{2})")


class RedactingFormatter(logging.Formatter):
    """Redact secrets and prevent message text from forging log records."""

    def format(self, record: logging.LogRecord) -> str:
        # Other handlers must receive the original message and exception cache.
        safe_record = copy.copy(record)
        safe_record.msg = _LOG_UNSAFE_CHARS.sub(" ", redact_sensitive_text(record.getMessage()))
        safe_record.args = None
        formatted = redact_log_urls(redact_sensitive_text(super().format(safe_record)))
        # aiohttp's access line carries the raw request target, and webhook platforms that cannot
        # send headers authenticate in the query string (BlueBubbles ``?password=``). An adapter's
        # own ``access_log=None`` does not reach the shared listener that serves multiplexed
        # profiles (#143550), so credential params are masked on every Hermes log handler. Its target
        # is relative (no ``://``), which redact_log_urls above skips.
        if record.name == "aiohttp.access":
            formatted = _redact_strict_url_credentials(formatted)
        first, *rest = _EXC_UNSAFE_CHARS.sub(" ", formatted).split("\n")
        # Traceback lines carry exception text; break their stamps so `hermes logs` keeps them
        # as continuation lines of this record.
        return "\n".join([first, *(_LOG_LINE_STAMP.sub("_", line) for line in rest)])

    def formatMessage(self, record: logging.LogRecord) -> str:
        from hermes_logging import display_record  # lazy: hermes_logging imports this module lazily
        return super().formatMessage(display_record(record))
