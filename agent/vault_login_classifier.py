"""Login / checkout form control classifier for vault autofill.

Python port (~170 LOC) of Merit-Systems/OpenInstinct's
``lib/manager/server/kernel-login-autofill.ts`` (MIT). Classifies visible
input controls on a page into login-autofill tokens. The vault fill path
uses the classification to select the single best current-password control
(the identifier is agent-visible metadata and is typed by the agent
itself via normal input tools).

Scoring:
- exact autocomplete-token match ................ 100
- type=password (not new/confirm/create/repeat) .. 90
- type=email / type=tel .......................... 85
- label/name regex heuristics .................. 70-75
Hard exclusions: autocomplete ``new-password`` / ``one-time-code``, and
label/name text matching ``(new|confirm|create|repeat)\\s*password``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import itertools

LOGIN_AUTOFILL_TOKENS = ("username", "email", "tel", "current-password")

# Payment / address autocomplete tokens (WHATWG) the checkout fill targets. ``cc-exp`` (combined
# MM/YY) is derived at fill time from exp_month + exp_year. Field-name/label heuristics below back
# up sites that omit autocomplete attributes.
PAYMENT_AUTOFILL_TOKENS = ("cc-number", "cc-name", "cc-exp", "cc-exp-month", "cc-exp-year", "cc-csc")
ADDRESS_AUTOFILL_TOKENS = ("address-line1", "address-line2", "address-level2", "address-level1",
                           "postal-code", "country-name", "country")
_CHECKOUT_HEURISTICS = (
    (re.compile(r"\b(?:card\s*number|cardnumber|ccnumber|cc\s*num|pan)\b"), "cc-number"),
    (re.compile(r"\b(?:name\s*on\s*card|cardholder|cc\s*name|ccname)\b"), "cc-name"),
    (re.compile(r"\b(?:cvc|cvv|csc|security\s*code|card\s*code)\b"), "cc-csc"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*month|exp\s*mm|ccmonth)\b"), "cc-exp-month"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*year|exp\s*yy(?:yy)?|ccyear)\b"), "cc-exp-year"),
    (re.compile(r"\b(?:exp(?:iry|iration)?(?:\s*date)?|mm\s*yy|valid\s*thru)\b"), "cc-exp"),
    (re.compile(r"\b(?:address\s*(?:line\s*)?2|apt|suite|unit)\b"), "address-line2"),
    (re.compile(r"\b(?:address(?:\s*line\s*1)?|street)\b"), "address-line1"),
    (re.compile(r"\b(?:city|town|locality)\b"), "address-level2"),
    (re.compile(r"\b(?:state|province|region|county)\b"), "address-level1"),
    (re.compile(r"\b(?:zip|postal|postcode)\b"), "postal-code"),
    (re.compile(r"\b(?:country)\b"), "country-name"),
)

_EXCLUDED_AUTOCOMPLETE = {"new-password", "one-time-code"}

_RE_EXCLUDED_PASSWORD = re.compile(r"\b(?:new|confirm|create|repeat)\s*password\b")
_RE_EMAIL = re.compile(r"\b(?:e[\s-]?mail|email address)\b")
_RE_TEL = re.compile(r"\b(?:phone|telephone|mobile)\b")
_RE_USERNAME = re.compile(
    r"\b(?:user\s*name|username|login|account|member|membership|mileageplus)\b"
)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True)
class LoginControl:
    """Descriptor of a visible input control, as inspected in the page."""

    autocomplete: str
    form_index: Optional[int]
    index: int
    label: str
    name: str
    type: str
    max_length: Optional[int] = None
    # Whether the control already holds a value (a boolean from the page; the value itself is never read).
    has_value: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoginControl:
        form_index = raw.get("formIndex", raw.get("form_index"))
        max_length = raw.get("maxLength", raw.get("max_length"))
        return cls(
            autocomplete=str(raw.get("autocomplete") or ""),
            form_index=int(form_index) if form_index is not None else None,
            index=int(raw.get("index") or 0),
            label=str(raw.get("label") or ""),
            name=str(raw.get("name") or ""),
            type=str(raw.get("type") or ""),
            max_length=int(max_length) if max_length is not None else None,
            has_value=bool(raw.get("hasValue", raw.get("has_value", False))),
        )


@dataclass(frozen=True)
class ClassifiedLoginControl:
    control: LoginControl
    score: int
    token: str


def classify_login_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control, or return None if it is not a login fill target."""
    autocomplete_tokens = [
        t for t in control.autocomplete.lower().split() if t
    ]
    if any(t in _EXCLUDED_AUTOCOMPLETE for t in autocomplete_tokens):
        return None

    for token in LOGIN_AUTOFILL_TOKENS:
        if token in autocomplete_tokens:
            return ClassifiedLoginControl(control, 100, token)

    searchable = _normalize_text(
        " ".join(part for part in (control.name, control.label) if part)
    )
    if _RE_EXCLUDED_PASSWORD.search(searchable):
        return None
    if control.type == "password":
        return ClassifiedLoginControl(control, 90, "current-password")
    if control.type == "email":
        return ClassifiedLoginControl(control, 85, "email")
    if control.type == "tel":
        return ClassifiedLoginControl(control, 85, "tel")
    if _RE_EMAIL.search(searchable):
        return ClassifiedLoginControl(control, 75, "email")
    if _RE_TEL.search(searchable):
        return ClassifiedLoginControl(control, 75, "tel")
    if _RE_USERNAME.search(searchable):
        return ClassifiedLoginControl(control, 70, "username")
    return None


_RE_OTP = re.compile(
    r"\b(?:one[\s-]?time|verification|security|auth(?:entication|enticator)?|2fa|two[\s-]?factor|mfa|totp|otp|"
    r"passcode|sms)\b.*\b(?:code|pin|token)\b|\b(?:otp|totp|2fa|mfa|verification\s*code|passcode)\b"
)


def classify_otp_controls(controls: list[LoginControl]) -> list[ClassifiedLoginControl]:
    """The controls that take a second-factor code. ``autocomplete=one-time-code`` is authoritative;
    otherwise a text/tel/number input whose name/label says code/OTP/2FA/verification. Some sites split
    the code into one input per digit (``maxlength=1`` boxes): they are returned in DOM order and the
    fill spreads the code across them."""
    out: list[ClassifiedLoginControl] = []
    for c in controls:
        tokens = c.autocomplete.lower().split()
        if "one-time-code" in tokens:
            out.append(ClassifiedLoginControl(c, 100, "one-time-code"))
            continue
        if c.type not in ("text", "tel", "number", "password", ""):
            continue
        if _RE_OTP.search(_normalize_text(" ".join(p for p in (c.name, c.label) if p))):
            out.append(ClassifiedLoginControl(c, 70, "one-time-code"))
    return out


def best_password_control(classified: list[ClassifiedLoginControl]) -> Optional[ClassifiedLoginControl]:
    """The ``current-password`` control a login fill targets: an actual ``type=password`` input (the fill
    script writes a password nowhere else, so an ``autocomplete=current-password`` text box is never a
    target), highest score, ties broken by DOM order. Inspection already drops disabled, read-only and
    invisible controls."""
    passwords = [c for c in classified if c.token == "current-password" and c.control.type == "password"]
    return min(passwords, key=lambda c: (-c.score, c.control.index)) if passwords else None


def select_password_fill(
    classified: list[ClassifiedLoginControl],
    password: str,
) -> list[dict[str, Any]]:
    """Select the single best current-password control to fill.

    The vault fill path is password-only: the identifier is agent-visible
    metadata and is typed by the agent via normal input tools. This picks
    the highest-scoring ``current-password`` control (ties broken by DOM
    order) and returns ``[{"index": int, "token": "current-password",
    "value": password}]`` or ``[]`` when no password field exists.
    """
    best_password = best_password_control(classified)
    if best_password is None or not password:
        return []
    return [
        {
            "index": best_password.control.index,
            "token": "current-password",
            "value": password,
        }
    ]


def classify_checkout_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control as a payment/address fill target (autocomplete token exact match 100,
    label/name heuristic 70), or None. Password/email inputs are never checkout targets."""
    tokens = [t for t in control.autocomplete.lower().split() if t]
    for token in PAYMENT_AUTOFILL_TOKENS + ADDRESS_AUTOFILL_TOKENS:
        if token in tokens:
            return ClassifiedLoginControl(control, 100, "country-name" if token == "country" else token)
    if control.type in ("password", "email"):
        return None
    searchable = _normalize_text(" ".join(part for part in (control.name, control.label) if part))
    for pattern, token in _CHECKOUT_HEURISTICS:
        if pattern.search(searchable):
            return ClassifiedLoginControl(control, 70, token)
    return None


def select_checkout_fills(classified: list[ClassifiedLoginControl], secret: dict[str, str],
                          field_tokens: dict[str, str]) -> list[dict[str, Any]]:
    """Map a payment/address secret payload onto the best control per autocomplete token.

    ``field_tokens`` is ``PAYMENT_FIELDS`` / ``ADDRESS_FIELDS`` (agent/vault_store.py). A combined
    ``cc-exp`` control gets ``MM/YY`` from exp_month + exp_year and then suppresses the separate
    month/year fills. Returns ``[{"index", "token", "value"}]``: one control per token, highest
    score then DOM order.
    """
    values: dict[str, str] = {tok: secret[f] for f, tok in field_tokens.items() if secret.get(f)}
    if "cc-exp-month" in values and "cc-exp-year" in values:
        values["cc-exp"] = f"{values['cc-exp-month'].zfill(2)}/{values['cc-exp-year'][-2:]}"
    fills: list[dict[str, Any]] = []
    for token, value in values.items():
        candidates = sorted((c for c in classified if c.token == token), key=lambda c: (-c.score, c.control.index))
        if candidates:
            fills.append({"index": candidates[0].control.index, "token": token, "value": value})
    if any(f["token"] == "cc-exp" for f in fills):
        fills = [f for f in fills if f["token"] not in ("cc-exp-month", "cc-exp-year")]
    return fills


# JS expression evaluated in the page to inspect candidate input controls.
# Ported from OpenInstinct's nativeLoginControlInspectionExpression.
# Inspection stamps every input with ``<nonce>:<index>`` under a per-inspection attribute; the fill
# script resolves targets by the stamp of ITS OWN inspection instead of re-querying by position, so
# neither a DOM reflow nor a second inspection in between can redirect the password into another field.
INSPECTION_STAMP_ATTR = "data-hermes-vault-slot"


def build_otp_fills(otp_controls: list[ClassifiedLoginControl], code: str) -> list[dict[str, Any]]:
    """One fill per box. Default: the single best-scoring code field takes the whole code.

    Per-digit entry only when the page unmistakably uses it: exactly len(code) OTP controls that are all
    ``maxlength=1``, all in the same form, and adjacent in DOM order (the classic N-box widget). Anything
    looser (several code-like inputs scattered over a page) gets ONE field, never a digit sprayed across
    unrelated inputs."""
    best = max(otp_controls, key=lambda c: c.score)
    boxes = sorted((c for c in otp_controls if c.control.max_length == 1), key=lambda c: c.control.index)
    if (len(boxes) == len(code)
            and len({b.control.form_index for b in boxes}) == 1
            and all(b.control.index - a.control.index == 1 for a, b in itertools.pairwise(boxes))):
        return [{"index": b.control.index, "token": "one-time-code", "value": ch} for b, ch in zip(boxes, code)]
    return [{"index": best.control.index, "token": "one-time-code", "value": code}]


_PLACEHOLDER = re.compile(r"__([A-Z][A-Z_]*)__")


def _render_js(template: str, **values: Any) -> str:
    """Substitute each ``__NAME__`` placeholder with ``values[name]`` as a JSON literal, in ONE pass over
    the template: substituted data is never scanned again, so a value that itself contains a placeholder
    (a password such as ``x__NONCE__y``) reaches the page byte for byte. ``json.dumps`` escapes quotes,
    backslashes and every non-ASCII character (including U+2028/U+2029), so each literal is a valid JS
    expression. A placeholder without a value, or a value without a placeholder, is a template defect."""
    names = set(_PLACEHOLDER.findall(template))
    if names != {k.upper() for k in values}:
        raise ValueError(f"JS template placeholders {sorted(names)} do not match values {sorted(values)}")
    literals = {k.upper(): json.dumps(v) for k, v in values.items()}
    return _PLACEHOLDER.sub(lambda m: literals[m.group(1)], template)


def build_inspection_js(nonce: str) -> str:
    return _render_js(_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE, nonce=nonce)


def build_clear_stamps_js(nonce: str) -> str:
    """Remove the slot stamps of the inspection ``nonce`` (and only those), in the document and open
    shadow roots. No values are read."""
    return _render_js(_CLEAR_STAMPS_JS_TEMPLATE, nonce=nonce)


_CLEAR_STAMPS_JS_TEMPLATE = """(() => {
  const prefix = __NONCE__ + ":";
  const roots = [];
  (function walk(root) { roots.push(root); for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot); })(document);
  let removed = 0;
  roots.forEach((r) => r.querySelectorAll("[data-hermes-vault-slot]").forEach((n) => {
    if (String(n.getAttribute("data-hermes-vault-slot")).startsWith(prefix)) { n.removeAttribute("data-hermes-vault-slot"); removed += 1; }
  }));
  return removed;
})()"""


_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE = """(() => {
  const nonce = __NONCE__;
  // Shadow-piercing collect: many logins (Home Assistant, custom-element SPAs) render their
  // form inside a shadow root, where document.querySelectorAll never sees it. Walk every
  // element (shadow hosts can be anything, not just inputs) and recurse into open shadow roots.
  // Forms are collected in the same walk: document.forms misses a <form> inside a shadow root, which
  // would leave its controls ungrouped (formIndex null) and merge split OTP boxes across forms.
  const elements = [];
  const forms = [];
  (function walk(root) {
    for (const e of root.querySelectorAll("*")) {
      if (e.tagName === "INPUT" || e.tagName === "SELECT") elements.push(e);
      else if (e.tagName === "FORM") forms.push(e);
      if (e.shadowRoot) walk(e.shadowRoot);
    }
  })(document);
  elements.forEach((element, index) => element.setAttribute("data-hermes-vault-slot", nonce + ":" + index));
  const out = elements.flatMap((element, index) => {
    if (element.disabled || element.readOnly) return [];
    if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(element.type)) return [];
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || element.getClientRects().length === 0) return [];
    const labels = element.labels ? Array.from(element.labels, (l) => l.textContent || "") : [];
    const ariaText = (element.getAttribute("aria-labelledby") || "")
      .split(/\\s+/).filter(Boolean)
      .map((id) => { const r = element.getRootNode(); const n = (r.getElementById ? r.getElementById(id) : null) || document.getElementById(id); return n ? (n.textContent || "") : ""; })
      .join(" ");
    const resolvedFormIndex = element.form ? forms.indexOf(element.form) : -1;
    return [{
      autocomplete: element.autocomplete || "",
      formIndex: resolvedFormIndex >= 0 ? resolvedFormIndex : null,
      index,
      maxLength: element.maxLength > 0 ? element.maxLength : null,
      label: [
        ...labels,
        element.getAttribute("aria-label") || "",
        ariaText,
        element.getAttribute("placeholder") || "",
        element.getAttribute("title") || "",
      ].join(" "),
      name: [element.name, element.id].join(" "),
      type: element.tagName === "SELECT" ? "select" : (element.type || ""),
      hasValue: String(element.value || "").length > 0,
    }];
  });
  return JSON.stringify(out);
})()"""


def build_fill_js(fills: list[dict[str, Any]], expected_origin: str, nonce: str = "", *,
                  require_empty: bool = False) -> str:
    """Build a JS expression that fills the selected controls and reports only a count. The
    returned expression never echoes the values back.

    ``expected_origin`` is asserted against ``window.location.origin`` synchronously inside the SAME
    evaluated script, immediately before any write. If the page navigated between inspection and fill
    (TOCTOU), the script writes nothing and returns ``{"refused": "origin_changed", "found": <actual>}``:
    proof scope equals mutation scope (#88706). Targets resolve by the ``<nonce>:<index>`` stamp of
    THIS inspection, in the document or any open shadow root; a ``current-password`` fill additionally
    requires ``type=password``; ``<select>`` controls (country, state, expiry month) match an option by
    value or visible text. No marker is left on filled controls so later model-driven DOM reads cannot
    address them deterministically.

    With ``require_empty`` a target control that already holds a value makes the script write nothing and
    return ``{"refused": "field_not_empty"}``, checked in the same evaluation before any write.
    ``el.focus()`` runs the page's focus listeners synchronously, so the target (still this inspection's
    connected, enabled password control on the expected origin) and, with ``require_empty``, its
    emptiness are checked again after it and immediately before the write; a change refuses
    (``target_changed`` / ``field_not_empty``) with nothing written. Every exit, including a refusal or an
    exception, removes this inspection's stamps (only those). Values are rendered by :func:`_render_js`,
    so they reach the page unchanged whatever they contain.
    """
    return _render_js(
        _FILL_JS_TEMPLATE,
        expected_origin=expected_origin,
        fills=[{"index": f["index"], "token": f.get("token", "current-password"), "value": f["value"]} for f in fills],
        nonce=nonce,
        require_empty=bool(require_empty),
    )


_FILL_JS_TEMPLATE = """(() => {
  const expectedOrigin = __EXPECTED_ORIGIN__;
  const fills = __FILLS__;
  const nonce = __NONCE__;
  const requireEmpty = __REQUIRE_EMPTY__;
  // The native accessors, so a value is read and written without page-defined element properties.
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  const current = (el) => String((el.tagName !== "SELECT" && setter && setter.get) ? setter.get.call(el) : (el.value || ""));
  const slotOf = (f) => find('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
  try {
    if (window.location.origin !== expectedOrigin) {
      return JSON.stringify({ refused: "origin_changed", found: window.location.origin });
    }
    if (requireEmpty && fills.some((f) => {
      const el = slotOf(f);
      return el && current(el).length > 0;
    })) {
      return JSON.stringify({ refused: "field_not_empty" });
    }
    let filled = 0;
    const norm = (t) => String(t || "").trim().toLowerCase();
    for (const f of fills) {
      const el = slotOf(f);
      if (!el || (f.token === "current-password" && el.type !== "password")) continue;
      try {
        if (el.tagName === "SELECT") {
          const want = norm(f.value);
          const opt = Array.from(el.options).find((o) => [o.value, o.textContent].some((t) => norm(t) === want || norm(t) === want.replace(/^0/, "")));
          if (opt) { el.value = opt.value; el.dispatchEvent(new Event("change", { bubbles: true, composed: true })); filled += 1; }
          continue;
        }
        el.focus();
        // Focus listeners have just run page code synchronously: re-establish the target and, with
        // requireEmpty, its emptiness; nothing runs between these checks and the write.
        const still = el.isConnected && window.location.origin === expectedOrigin && !el.disabled && !el.readOnly
          && el.getAttribute("data-hermes-vault-slot") === nonce + ":" + f.index
          && !(f.token === "current-password" && el.type !== "password");
        if (!still) {
          if (requireEmpty) return JSON.stringify({ refused: "target_changed" });
          continue;
        }
        if (requireEmpty && current(el).length > 0) return JSON.stringify({ refused: "field_not_empty" });
        // one-time-code split into single-character boxes: f.value is the slice for THIS box (see build_otp_fills)
        if (setter && setter.set) { setter.set.call(el, f.value); } else { el.value = f.value; }
        // composed: a component listening on its shadow host must see the value change too.
        el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, inputType: "insertText" }));
        el.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
        if (current(el).length > 0) filled += 1;
      } catch (e) { /* skip */ }
    }
    return JSON.stringify({ filled });
  } finally {
    find("[data-hermes-vault-slot]").forEach((n) => {
      if (String(n.getAttribute("data-hermes-vault-slot")).startsWith(nonce + ":")) n.removeAttribute("data-hermes-vault-slot");
    });
  }
  // Declared last (hoisted) so the origin check precedes every DOM lookup. Stamped controls may live
  // inside a shadow root (see the inspection script's walk): look in the document and every open
  // shadow root, walked afresh on each call since focus listeners may have changed the tree.
  function find(selector) {
    const roots = [];
    (function walk(root) {
      roots.push(root);
      for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
    })(document);
    return roots.flatMap((r) => Array.from(r.querySelectorAll(selector)));
  }
})()"""
