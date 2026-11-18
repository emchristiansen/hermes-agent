"""The redactor on the ``hermes vault fill`` diagnostic path: ordinary signed quantities and joined
identifiers stay exact, contact numbers are still masked, and a value a vault fill registered never
appears in redacted text."""
import pytest

from agent import redact

_CANARY = "canary-vault-value-6f1e2d"  # synthetic, registered as a fill would register a password


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    redact.clear_vault_redaction_values()
    yield
    redact.clear_vault_redaction_values()


@pytest.mark.parametrize("text", [
    "freed +1109262336 bytes on /home",
    "HOME+1109262336B/pool+2941231104/FREEING0",
    "noSupport/new09f71717+5323013 FULLread; build 1.2.3+7654321",
])
def test_ordinary_quantities_and_identifiers_stay_exact(text):
    assert redact.redact_sensitive_text(text) == text


def test_contact_number_is_still_masked():
    out = redact.redact_sensitive_text("Phone: +12025550123")
    assert "+12025550123" not in out


def test_registered_vault_value_never_survives_redaction():
    redact.register_vault_redaction_value(_CANARY)
    text = f"fill failed near {_CANARY}; freed +1109262336 bytes"
    out = redact.redact_sensitive_text(text)
    assert _CANARY not in out
    assert "+1109262336" in out
