"""The header path masks NO MORE than upstream (review22815 MERGE-FLAGS-1): differential negative controls.

Every expected string below is upstream's (ae2a81ccc00e) exact output for that input, taken from its own
agent.redact* modules (redact_header_credentials, then the public redact_credential_shapes).
Material glued after a placeholder (a quote, backslash, }} or opaque tail) keeps exactly upstream's
mask: the candidate's extra glued-slot pass, which removed suffixes upstream leaves in place, is gone,
and nothing replaces it. The placeholder exemption only ever keeps MORE than upstream: a slot that is
wholly a placeholder is shown as written where upstream masks it.
"""
import pytest

# (input, upstream redact_header_credentials, upstream redact_credential_shapes): ours is identical.
UPSTREAM_PARITY = [
    ('echo x-api-key: $TOKEN"AbCd', 'echo x-api-key: ***"AbCd', 'echo x-api-key: ***"AbCd'),
    ('echo x-api-key: $TOKEN\\AbCd', 'echo x-api-key: ***\\AbCd', 'echo x-api-key: ***\\AbCd'),
    ('echo x-api-key: {{ token }}AbCd', 'echo x-api-key: *** token }}AbCd', 'echo x-api-key: *** token }}AbCd'),
    ('x-api-key: ${API_KEY}AbCd', 'x-api-key: ***', 'x-api-key: ***'),
    ('x-api-key: "$TOKEN"AbCd', 'x-api-key: ***', 'x-api-key: ***'),
    ("curl -H 'x-api-key: $TOKEN'AbCd", "curl -H 'x-api-key: ***'AbCd", "curl -H 'x-api-key: ***'AbCd"),
    ('curl -H "x-api-key: {{ token }}AbCd" https://x', 'curl -H "x-api-key: ***" https://x', 'curl -H "x-api-key: ***" https://x'),
    ('x-api-key: {{token}}AbCd', 'x-api-key: ***', 'x-api-key: ***'),
    ('x-api-key: $TOKEN AbCd', 'x-api-key: ***', 'x-api-key: ***'),
    ('x-api-key: realsecretvalue123', 'x-api-key: ***', 'x-api-key: ***'),
    ('x-api-key: {{ token }}"AbCd x-auth-token: zz', 'x-api-key: ***', 'x-api-key: *** token }}"AbCd x-auth-token: ***'),
    ('x-api-key: <token>AbCd', 'x-api-key: ***', 'x-api-key: ***'),
    ('x-api-key: $TOKEN"AbCd\nAuthorization: Bearer abcdef123456', 'x-api-key: ***\nAuthorization: Bearer abcdef123456', 'x-api-key: ***\nAuthorization: Bearer ***'),
    ('"x-api-key": "$TOKEN\\"AbCd"', '"x-api-key": "$TOKEN\\"AbCd"', '"x-api-key": "***"'),
    ('X-Api-Key:${TOKEN}\\nAbCd', 'X-Api-Key:***', 'X-Api-Key:***'),
]

# (input, upstream headers, upstream public, ours headers, ours public): a wholly-placeholder slot
# that upstream masks on one path is kept as written there; nothing else differs.
PLACEHOLDER_KEPT = [
    ('aUtHoRiZaTiOn: Bearer $TOKEN', 'aUtHoRiZaTiOn: Bearer $TOKEN', 'aUtHoRiZaTiOn: Bearer ***', 'aUtHoRiZaTiOn: Bearer $TOKEN', 'aUtHoRiZaTiOn: Bearer $TOKEN'),
    ('Authorization: Bearer %s\\n', 'Authorization: Bearer %s\\n', 'Authorization: Bearer ***', 'Authorization: Bearer %s\\n', 'Authorization: Bearer %s\\n'),
    ('x-api-key: $TOKEN', 'x-api-key: ***', 'x-api-key: ***', 'x-api-key: $TOKEN', 'x-api-key: $TOKEN'),
    ('x-api-key: {{ token }}', 'x-api-key: ***', 'x-api-key: *** token }}', 'x-api-key: {{ token }}', 'x-api-key: *** token }}'),
    ('x-api-key: ${API_KEY}', 'x-api-key: ***', 'x-api-key: ***', 'x-api-key: ${API_KEY}', 'x-api-key: ${API_KEY}'),
]


@pytest.fixture(autouse=True)
def _redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("text,headers,public", UPSTREAM_PARITY)
def test_masking_is_exactly_upstreams(text, headers, public):
    from agent.redact import redact_credential_shapes
    from agent.redact_headers import redact_header_credentials

    assert redact_header_credentials(text) == headers
    assert redact_credential_shapes(text) == public


@pytest.mark.parametrize("text,up_headers,up_public,headers,public", PLACEHOLDER_KEPT)
def test_a_whole_placeholder_only_ever_keeps_more_than_upstream(text, up_headers, up_public, headers, public):
    from agent.redact import redact_credential_shapes
    from agent.redact_headers import redact_header_credentials

    assert redact_header_credentials(text) == headers
    assert redact_credential_shapes(text) == public
    # Each path either keeps the whole placeholder as written or masks exactly as upstream does,
    # never more; at least one keeps what upstream masks.
    assert headers in (text, up_headers) and public in (text, up_public)
    assert (headers, public) != (up_headers, up_public)
