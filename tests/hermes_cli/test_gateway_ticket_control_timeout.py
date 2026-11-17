"""An owner whose control socket takes the session-ticket request but answers nothing within the
mint's bound is still starting. ``connect_gateway`` reports it as ``starting`` (``control_timeout``),
the verdict the ensure gives a control timeout, so a Kanban worker treats it as a transient owner
state (nothing was submitted) rather than a failed connection."""

import pytest

from hermes_cli import gateway_client, gateway_runtime
from hermes_cli.gateway_owner_restart import _TRANSIENT_OWNER_STATES


def _ready(*_, **__):
    return gateway_runtime.GatewayDiscovery(
        "ready", gateway_runtime.GatewayEndpoint("p", "i", 1, 1, "http://127.0.0.1:1", "none", frozenset()))


@pytest.mark.asyncio
async def test_an_unanswered_session_ticket_reads_as_a_starting_owner(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", _ready)

    def unanswered(*_args, **_kwargs):
        raise TimeoutError("timed out")
    monkeypatch.setattr(gateway_client, "_session_ticket", unanswered)

    with pytest.raises(gateway_client.GatewayUnavailableError) as raised:
        async with gateway_client.connect_gateway():
            pass
    assert (raised.value.state, raised.value.reason_code) == ("starting", "control_timeout")
    assert raised.value.state in _TRANSIENT_OWNER_STATES
    assert isinstance(raised.value.__cause__, TimeoutError)
