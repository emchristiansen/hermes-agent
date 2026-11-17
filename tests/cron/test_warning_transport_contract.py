"""Real cron router/queue with local transport doubles; no external network."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from cron import delivery_queue, scheduler
from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    assert not thread.is_alive()
    loop.close()


@pytest.mark.parametrize("lane", ["native", "relay", "native-fallback", "relay-failure", "standalone", "standalone-async"])
@pytest.mark.parametrize("setting,for_failure", [(None, True), (False, True), (True, True), (True, False)])
def test_warning_policy_at_real_transport_boundary(tmp_path, monkeypatch, loop_thread, lane, setting, for_failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "delivery.db")
    config_text = "cron: {wrap_response: false}\n"
    if setting is not None:
        config_text += f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n"
    (tmp_path / "config.yaml").write_text(config_text)
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
    config.platforms[Platform.RELAY] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    sent = []
    fallbacks = []
    relay_recovered = []  # msg-retry-on-13f-7: the relay accepts once this is non-empty

    async def send(chat_id, content, metadata=None):
        sent.append(("discord", chat_id, content))
        if lane == "native-fallback":
            raise RuntimeError("isolated live failure")
        return {"success": True, "message_id": "native-id"}

    async def relay_send(platform, chat_id, content, metadata=None):
        sent.append((platform.value, chat_id, content))
        if lane == "relay-failure" and not relay_recovered:
            raise RuntimeError("isolated relay failure")
        return {"success": True, "message_id": "relay-id"}

    async def standalone(platform, pconfig, chat_id, content, **kwargs):
        fallbacks.append((platform.value, chat_id, content))
        return {"success": True, "message_id": "direct-id"}

    monkeypatch.setattr("gateway.outbound.send._send_to_platform", standalone)
    adapter = SimpleNamespace(send=send, send_for_platform=relay_send,
                              fronts_platform=lambda p: p == Platform.DISCORD)
    adapters = {} if lane.startswith("standalone") else {
        Platform.RELAY if lane.startswith("relay") else Platform.DISCORD: adapter}
    job = {"id": "fixture", "execution_id": "run", "deliver": "discord:fixture"}
    payload = "Requested warning quotation" if not for_failure else "Automatic failure notice"
    delivery_queue.enqueue("run", job, payload, for_failure=for_failure)

    def drain():
        assert scheduler.drain_delivery_queue(adapters, loop_thread) == 1

    if lane == "standalone-async":
        async def inside_loop():
            drain()
        asyncio.run(inside_loop())
    else:
        drain()
    muted = setting is True and for_failure
    assert sent == ([] if muted or lane.startswith("standalone") else [("discord", "fixture", payload)])
    assert fallbacks == ([] if muted or lane not in ("standalone", "standalone-async", "native-fallback") else [("discord", "fixture", payload)])
    # msg-retry-on-13f-6 (ea298f05, at least once): a refused send is retried, never terminal "failed".
    expected = "suppressed" if muted else "pending" if lane == "relay-failure" else "delivered"
    row = delivery_queue.get_status("run")
    assert row["status"] == expected
    if expected == "pending":
        assert row["content"] == payload and row["retry_after"] and "isolated relay failure" in row["error"]
    # Duplicate category and policy changes cannot overwrite durable SQL queue outcomes.
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: false}\n")
    delivery_queue.enqueue("run", job, payload, for_failure=not for_failure)
    assert scheduler.drain_delivery_queue(adapters, loop_thread) == 0
    assert delivery_queue.get_status("run")["status"] == expected
    if expected == "pending":
        # msg-retry-on-13f-7 (MR-TEST-11, ea298f05): the retry actually delivers; the duplicate attempt is
        # accepted and goes to the same target with the same body.
        relay_recovered.append(True)
        later = delivery_queue._now_seconds() + delivery_queue.RETRY_BACKOFF_MAX_SECONDS + 1
        monkeypatch.setattr(delivery_queue, "_now_seconds", lambda: later)
        assert scheduler.drain_delivery_queue(adapters, loop_thread) == 1
        assert delivery_queue.get_status("run")["status"] == "delivered"
        # The retry runs under the config written above (no wrap_response override), so the body may be wrapped.
        assert sent[0] == ("discord", "fixture", payload) and len(sent) == 2 and fallbacks == []
        assert sent[1][:2] == ("discord", "fixture") and payload in sent[1][2]
