"""Messages reaching the classic CLI while a turn runs steer into that turn (Eric, 2026-11-04).

"I want literally every message to always be a steering message." A spoken transcript is the user's
message: it takes the Enter path's busy routing. A process notice or subagent result that lands while the
turn runs reaches it at once, framed as Hermes automation's, never the user's (the gateway's busy pass
does the same for its admitted notices). Only what nothing running takes waits for the next turn, as it
did. Explicit choices stay what they were: ``/queue`` and Tab ask for the next turn, a barge-in is the
interrupt control whose transcript is its redirect, and stop/admin commands stay commands.

Real ``HermesCLI`` entry points (``_cmd_queue``, the Tab key binding, the voice transcribe handlers, the
turn's monitor loop and the notification drain); the running agent is the real steer seam
(``BufferedAgent``) and its boundary the canonical ``apply_pending_steer_to_tool_results``.
"""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from cli import HermesCLI, _VoiceInputMessage
from tests.gateway.test_prompt_queued_transfer import BufferedAgent
from tools.process_registry_notifications import format_process_notification

AUTOMATION_MARKER = "[DELIVERED MESSAGE — from Hermes automation"


class _Agent:
    def __init__(self, accepts=True):
        self.accepts, self.steered = accepts, []

    def steer(self, text):
        self.steered.append(text)
        return self.accepts

    def interrupt(self, *args):
        raise AssertionError("a busy message never interrupts the turn")


def _busy_cli(agent, **attrs):
    c = HermesCLI.__new__(HermesCLI)
    c._pending_input, c._interrupt_queue = queue.Queue(), queue.Queue()
    c._agent_running, c.agent, c.session_id = True, agent, None
    c._app, c._should_exit, c._attached_images = None, False, []
    for key, value in attrs.items():
        setattr(c, key, value)
    return c


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def _boundary(agent):
    messages = [{"role": "assistant", "tool_calls": []}, {"role": "tool", "content": "fixture result"}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    return messages[-1]["content"] if messages[-1]["role"] == "user" else None


# --- explicit next-turn choices stay next-turn ------------------------------------------------------

def test_queue_while_a_turn_runs_is_still_the_next_turn():
    agent = _Agent()
    c = _busy_cli(agent)

    c._cmd_queue("/queue also check the staging logs")

    assert agent.steered == []
    assert _drain(c._pending_input) == ["also check the staging logs"]


def _tab(c, text, monkeypatch):
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import BufferControl, Layout, Window
    from hermes_cli.cli_modern_layout import modern_input_rules

    c._tui_input_rule_height = lambda pos: 1
    monkeypatch.setattr(HermesCLI, "_inline_pastes", lambda self, buf: None)
    kb = KeyBindings()
    modern_input_rules(c, kb)
    tab = next(b for b in kb.bindings if b.keys == (Keys.ControlI,))
    buf = Buffer(document=None)
    buf.text = text
    app = Application(layout=Layout(Window(BufferControl(buffer=buf))))
    with set_app(app):
        assert tab.filter()
        tab.handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None)))


def test_tab_while_a_turn_runs_is_still_the_next_turn(monkeypatch):
    agent = _Agent()
    c = _busy_cli(agent)

    _tab(c, "and the staging logs", monkeypatch)

    assert agent.steered == []
    assert _drain(c._pending_input) == ["and the staging logs"]


@pytest.mark.parametrize("command", ["/stop", "/new"])
def test_explicit_controls_never_steer(command, monkeypatch):
    agent, seen = _Agent(), []
    c = _busy_cli(agent)
    monkeypatch.setattr(HermesCLI, "process_command", lambda self, cmd: seen.append(cmd) or True)

    class _Buf:
        text, reset = command, (lambda self, append_to_history=False: None)
    c._submit_editor_buffer(_Buf())

    assert seen == [command] and agent.steered == []


# --- a spoken message ------------------------------------------------------------------------------

def _voice_cli(agent, **attrs):
    c = _busy_cli(agent, **attrs)
    c._voice_lock, c._voice_mode, c._voice_tts, c._voice_continuous = threading.Lock(), False, False, False
    c._voice_processing, c._voice_tts_stop, c._voice_last_tts_text = False, None, ""
    c._voice_tts_done, c._voice_barge_capture = threading.Event(), threading.Event()
    c._voice_tts_done.set()
    c._voice_barge_phase, c.console = None, SimpleNamespace(width=80)
    return c


def _transcribed(transcript):
    def wrap(test):
        for decorator in reversed((
                patch("cli._cprint"), patch("cli.os.unlink"), patch("cli.os.path.isfile", return_value=True),
                patch("hermes_cli.config.load_config", return_value={"stt": {}}),
                patch("tools.voice_mode.transcribe_recording",
                      return_value={"success": True, "transcript": transcript}),
                patch("tools.voice_mode.play_beep"))):
            test = decorator(test)
        return test
    return wrap


def _recorded(agent, **attrs):
    recorder = MagicMock()
    recorder.stop.return_value = "/recording.wav"
    return _voice_cli(agent, _voice_recording=True, _voice_recorder=recorder, **attrs)


@_transcribed("check the logs too")
def test_a_spoken_message_while_a_turn_runs_steers_into_it(*_mocks):
    agent = _Agent()
    c = _recorded(agent)

    c._voice_stop_and_transcribe()

    assert agent.steered == ["check the logs too"]
    assert _drain(c._pending_input) == []


@_transcribed("check the logs too")
def test_a_spoken_message_the_turn_rejects_keeps_its_voice_form_for_the_next_turn(*_mocks):
    c = _recorded(_Agent(accepts=False))

    c._voice_stop_and_transcribe()

    [queued] = _drain(c._pending_input)
    assert isinstance(queued, _VoiceInputMessage) and queued.text == "check the logs too"


@_transcribed("check the logs")
def test_a_spoken_message_while_idle_starts_the_next_turn(*_mocks):
    c = _recorded(None, _agent_running=False)

    c._voice_stop_and_transcribe()

    [queued] = _drain(c._pending_input)
    assert isinstance(queued, _VoiceInputMessage) and queued.text == "check the logs"


def test_a_barge_in_stays_the_interrupt_controls_redirect(tmp_path, monkeypatch):
    agent = _Agent()
    c = _voice_cli(agent)
    c._voice_barge_capture.set()
    c._voice_barge_phase = "generation"
    wav = tmp_path / "barge.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr("tools.voice_mode.transcribe_recording",
                        lambda path, model=None: {"success": True, "transcript": "do it differently"})

    c._voice_submit_barge_utterance(str(wav))

    assert agent.steered == []
    [queued] = _drain(c._pending_input)
    assert isinstance(queued, _VoiceInputMessage) and queued.text == "do it differently"


# --- process notices and subagent results ---------------------------------------------------------

def _completion(sid="proc_1"):
    return {"type": "completion", "session_id": sid, "session_key": "cli-session", "command": "bash build.sh",
            "exit_code": 0, "completion_reason": "exited", "output": "build ok"}


def _delegation():
    return {"type": "async_delegation", "delegation_id": "d1", "session_key": "cli-session",
            "results": [{"status": "completed", "summary": "audit done", "goal": "audit"}]}


def _registry(monkeypatch, events):
    """The process registry, handing each event out once."""
    from tools.process_registry_notifications import format_process_notification as fmt
    pending = list(events)

    def drain_notifications(**_kw):
        out = [(event, fmt(event)) for event in pending]
        pending.clear()
        return out
    monkeypatch.setattr("tools.process_registry.process_registry", SimpleNamespace(
        drain_notifications=drain_notifications, completion_queue=queue.Queue(), is_completion_consumed=lambda sid: False))
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {}})


@pytest.mark.parametrize("event", [_completion(), _delegation()], ids=["process", "subagent"])
def test_a_notice_landing_while_a_turn_runs_reaches_that_turn_as_automations(monkeypatch, event):
    """Through the turn's own monitor loop: the notice is drained while the agent thread runs, and the
    turn takes it at its next boundary in the Hermes-automation frame."""
    _registry(monkeypatch, [event])
    agent, steered = BufferedAgent(), threading.Event()
    steer = agent.steer
    agent.steer = lambda *a, **k: (steer(*a, **k), steered.set())[0]
    c = _busy_cli(agent, session_id="cli-session", _voice_mode=False, _clarify_state=None, _clarify_freetext=None)
    monkeypatch.setattr(HermesCLI, "_invalidate", lambda self, min_interval=0: None)
    thread = threading.Thread(target=lambda: steered.wait(10), daemon=True)
    thread.start()

    assert c._chat_monitor_agent_thread(SimpleNamespace(), thread) is None
    content = _boundary(agent)

    assert steered.is_set() and _drain(c._pending_input) == []
    assert AUTOMATION_MARKER in content and "a direct message from the user" not in content
    assert format_process_notification(event).strip()[:40] in content


def test_a_notice_while_idle_is_the_next_turn_as_before(monkeypatch):
    _registry(monkeypatch, [_completion()])
    agent = BufferedAgent()
    c = _busy_cli(agent, session_id="cli-session", _agent_running=False)

    c._drain_process_notifications("cli-idle")

    assert agent._pending_steer is None
    assert len(_drain(c._pending_input)) == 1


def test_a_notice_no_running_agent_takes_is_the_next_turn(monkeypatch):
    """A surface agent without the sender-recording seam is never handed a notice as the user's text."""
    _registry(monkeypatch, [_delegation()])
    agent = _Agent()
    c = _busy_cli(agent, session_id="cli-session")

    c._drain_process_notifications("cli-busy")

    assert agent.steered == []
    [queued] = _drain(c._pending_input)
    assert queued.display_kind == "async_delegation_complete"


def test_a_notice_the_turn_never_read_becomes_the_next_turn_still_as_automations(monkeypatch):
    """Steered after the turn's last boundary: the finalizer hands it back as the next turn, framed."""
    _registry(monkeypatch, [_completion()])
    agent = BufferedAgent()
    c = _busy_cli(agent, session_id="cli-session")
    c._drain_process_notifications("cli-busy")

    leftover = agent._drain_leftover_steer()

    assert leftover.startswith(AUTOMATION_MARKER) and "build ok" in leftover
    assert agent._pending_steer is None
