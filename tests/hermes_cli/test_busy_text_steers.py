"""Messages reaching the classic CLI while a turn runs steer into that turn (Eric, 2026-11-04).

"I want literally every message to always be a steering message." A spoken transcript is the user's
message: it takes the Enter path's busy routing. Only what nothing running takes waits for the next turn,
as it did. Explicit choices stay what they were: ``/queue`` and Tab ask for the next turn, a barge-in is
the interrupt control whose transcript is its redirect, and stop/admin commands stay commands.

Process notices and subagent results are not the CLI's to steer: the classic CLI is gateway-backed, the
gateway admits them and its busy pass steers them; the CLI's busy monitor never reads the gateway event
stream its running turn pumps.

Real ``HermesCLI`` entry points (``_cmd_queue``, the Tab key binding, the voice transcribe handlers, the
turn's monitor loop); the running agent is a recording ``steer`` seam.
"""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI, _VoiceInputMessage


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


@_transcribed("check the logs too")
def test_on_a_gateway_session_a_refused_spoken_message_stays_the_next_voice_turn(*_mocks):
    """The classic CLI is gateway-backed: a refused steer never becomes a plain-text ``/queue`` admission."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    store = GatewaySessionStore.__new__(GatewaySessionStore)
    enqueued = []
    c = _recorded(_Agent(accepts=False), session_id="s1", _session_db=store)
    c._queue_enqueue = enqueued.append

    c._voice_stop_and_transcribe()

    assert enqueued == []
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


# --- the gateway-backed turn's event stream -------------------------------------------------------

def test_the_busy_monitor_never_takes_the_running_turns_gateway_frames(monkeypatch):
    """The turn's ``GatewayAgent`` pumps ``connection.events``: while it runs, the monitor loop leaves every
    frame there (``queue_admitted_turn`` is the idle tick's reader, never the busy loop's)."""
    from hermes_cli.gateway_connection import SessionState
    from hermes_cli.gateway_session_store import GatewaySessionStore
    store = GatewaySessionStore.__new__(GatewaySessionStore)
    states = {}
    store.connection = SimpleNamespace(events=queue.Queue(), state=lambda sid: states.setdefault(sid, SessionState(sid)))
    frame = {"method": "event", "params": {"session_id": "s1", "type": "message.delta", "payload": {"text": "hi"}}}
    store.connection.events.put(frame)
    c = _busy_cli(_Agent(), session_id="s1", _session_db=store, _voice_mode=False, _clarify_state=None,
                  _clarify_freetext=None)
    ticks, enough = [], threading.Event()

    def invalidate(self, min_interval=0):
        ticks.append(1)
        if len(ticks) >= 5:
            enough.set()
    monkeypatch.setattr(HermesCLI, "_invalidate", invalidate)
    thread = threading.Thread(target=lambda: enough.wait(30), daemon=True)
    thread.start()

    assert c._chat_monitor_agent_thread(SimpleNamespace(), thread) is None

    assert enough.is_set(), "the monitor loop idled through several polls"
    assert store.connection.events.get_nowait() is frame and store.connection.events.empty()


def test_a_delivered_steer_the_turn_never_read_becomes_the_next_turn_still_as_its_senders():
    """A ticketless delivered steer (a legacy wake steered by the gateway's busy handler) left after the
    turn's last boundary: the finalizer hands it back as the next turn, still framed as delivered."""
    from tests.gateway.test_prompt_queued_transfer import BufferedAgent
    agent = BufferedAgent()
    agent.steer("[Kanban task completed] t_11", delivered_by={})

    leftover = agent._drain_leftover_steer()

    assert leftover.startswith("[DELIVERED MESSAGE — from Hermes automation") and "t_11" in leftover
    assert agent._pending_steer is None
