import threading
import time

import pytest

from vrcai.config import ConfigError, ConfigStore
from vrcai.service import BridgeController, VRChatLLMBridge


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.01)
    assert predicate(), "condition was not reached before timeout"


class FakeAudio:
    device_name = "mock audio"
    current_rms = 0

    def __init__(self, cfg, callback, allowed, error):
        self.callback, self.allowed, self.error = callback, allowed, error
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def flush(self):
        pass


class FakeProvider:
    def __init__(self, cfg):
        self.closed = False
        self.calls = []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def chat(self, messages, on_token=None, cancelled=lambda: False):
        self.calls.append(messages)
        self.entered.set()
        self.release.wait(3)
        if on_token:
            on_token("你好。欢迎！")
        return "你好。欢迎！"

    def transcribe(self, pcm, rate):
        self.calls.append((pcm, rate))
        return "你好"

    def synthesize(self, text):
        self.calls.append(text)
        self.entered.set()
        self.release.wait(3)
        return b"\0" * 320, 32000

    def close(self):
        self.closed = True


class FakeOSC:
    def __init__(self, *args):
        self.messages = []

    def send_message(self, address, payload):
        self.messages.append((address, payload))


def make_bridge(cfg, **kwargs):
    return VRChatLLMBridge(
        cfg,
        audio_factory=FakeAudio,
        llm_factory=FakeProvider,
        stt_factory=FakeProvider,
        tts_factory=FakeProvider,
        osc_factory=FakeOSC,
        player=kwargs.pop("player", lambda *args: None),
        **kwargs,
    )


@pytest.fixture
def session(cfg):
    cfg["osc"]["send_interval"] = 0.01
    cfg["streaming"]["sentence_delay"] = 0
    bridge = make_bridge(cfg)
    thread = threading.Thread(target=bridge.start)
    thread.start()
    assert bridge.ready.wait(3)
    assert bridge.state == "running"
    yield bridge
    bridge.llm.release.set()
    bridge.tts.release.set()
    bridge.stop()
    thread.join(3)
    assert not thread.is_alive()
    assert bridge.audio.stopped
    assert all(p.closed for p in (bridge.llm, bridge.stt, bridge.tts))
    assert not any(t.is_alive() for t in bridge._threads)


def outputs(bridge):
    return [payload for path, payload in bridge.osc.messages if path == "/chatbox/input"]


def test_streaming_emits_ordered_osc_and_tts(session):
    session.cfg["streaming"]["enabled"] = True
    session.cfg["tts"]["minimax"]["api_key"] = "test-tts-secret"
    session.set_tts_enabled(True)
    session.submit_text("你好")
    wait_for(lambda: len(session.tts.calls) == 2)
    assert [p[0] for p in outputs(session)] == ["(AI喜多郁代) 你好。", "(AI喜多郁代) 欢迎！"]
    assert session.tts.calls == ["你好。", "欢迎！"]
    assert session.metrics["llm_first_token_ms"] is not None


def test_interrupt_discards_inflight_model_results_and_history(session):
    session.llm.release.clear()
    session.submit_text("hello")
    assert session.llm.entered.wait(1)
    session.interrupt()
    session.llm.release.set()
    wait_for(lambda: session.metrics["llm_ms"] is not None)
    assert not outputs(session)
    assert not session.conv.turns


def test_reset_command_bypasses_coherence_filter_and_cancels_old_turn(session):
    session.conv.record("old", "reply")
    session.llm.release.clear()
    session.submit_text("something")
    assert session.llm.entered.wait(1)
    session.submit_text("/重置")
    assert not session.conv.turns
    session.llm.release.set()
    wait_for(lambda: session.metrics["llm_ms"] is not None)
    assert not outputs(session)
    assert not session.conv.turns


def test_actual_capture_sample_rate_reaches_stt(session):
    session.cfg["audio"]["sample_rate"] = 22050
    session.audio.callback(b"pcm")
    wait_for(lambda: bool(session.stt.calls))
    assert session.stt.calls[0] == (b"pcm", 22050)


def test_disable_tts_discards_synthesis_already_in_progress(session):
    session.cfg["tts"]["minimax"]["api_key"] = "test-tts-secret"
    session.set_tts_enabled(True)
    played = []
    session._player = lambda *args: played.append(args)
    session.tts.release.clear()
    session.submit_text("你好")
    assert session.tts.entered.wait(1)
    session.set_tts_enabled(False)
    session.tts.release.set()
    wait_for(lambda: session.metrics["tts_ms"] is not None)
    assert not played
    assert session._tts.empty()


def test_capture_pauses_during_playback_and_ptt_has_expiring_lease(session):
    session._playing.set()
    assert not session._capture_allowed()
    session._playing.clear()
    session.cfg["chat"]["listen_mode"] = "push_to_talk"
    assert not session._capture_allowed()
    session.set_listening(True)
    assert session._capture_allowed()
    session._ptt_deadline = time.monotonic() - 1
    assert not session._capture_allowed()
    session.set_listening(False)


def test_queue_keeps_recent_inputs_and_drops_stale_ones(session):
    session.llm.release.clear()
    session.submit_text("block")
    assert session.llm.entered.wait(1)
    for i in range(8):
        session.submit_text("message " + str(i))
    assert session._inputs.qsize() == 3
    assert session.metrics["dropped_inputs"] == 5
    session.interrupt()
    session._inputs.put((time.monotonic() - 100, session._epoch, "text", "stale"))
    session.llm.release.set()
    wait_for(lambda: session.metrics["stale_inputs"] == 1)
    assert len(session.llm.calls) == 1


def test_controller_waits_for_old_session_before_restart(tmp_path, cfg):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(cfg)
    controller = BridgeController(store, bridge_factory=make_bridge)
    controller.start()
    bridge = controller.bridge
    assert bridge.ready.wait(2)
    bridge.llm.release.clear()
    bridge.submit_text("blocked model")
    assert bridge.llm.entered.wait(1)
    controller.stop()
    assert controller.status()["state"] == "stopping"
    with pytest.raises(ConfigError):
        controller.start()
    bridge.llm.release.set()
    controller.thread.join(3)
    assert controller.status()["state"] == "stopped"
    assert not outputs(bridge)
    controller.start()
    assert controller.bridge.ready.wait(2)
    assert controller.bridge is not bridge
    controller.stop(wait=3)
    assert not controller.thread.is_alive()


def test_dequeued_input_from_old_epoch_is_ignored_after_interrupt(session):
    old_epoch = session._epoch
    session.interrupt()
    session._inputs.put((time.monotonic(), old_epoch, "text", "old dequeued message"))
    session.submit_text("new message")
    wait_for(lambda: bool(session.llm.calls))
    assert len(session.llm.calls) == 1
    assert session.llm.calls[0][-1]["content"] == "new message"


def test_running_config_change_marks_restart_but_tts_toggle_is_live(tmp_path, cfg):
    cfg["tts"]["minimax"]["api_key"] = "test-tts-secret"
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(cfg)
    controller = BridgeController(store, bridge_factory=make_bridge)
    controller.start()
    assert controller.bridge.ready.wait(2)
    try:
        controller.update_config({"tts": {"enabled": True}})
        assert controller.bridge.status()["tts_enabled"]
        assert not controller.restart_required
        controller.update_config({"chat": {"temperature": 0}})
        assert controller.restart_required
        assert controller.bridge.cfg["chat"]["temperature"] == 0.7
    finally:
        controller.stop(wait=3)


def test_startup_failure_closes_partial_resources(cfg):
    def broken_audio(*args):
        raise OSError("audio cannot open")

    bridge = VRChatLLMBridge(
        cfg,
        audio_factory=broken_audio,
        llm_factory=FakeProvider,
        stt_factory=FakeProvider,
        tts_factory=FakeProvider,
        osc_factory=FakeOSC,
    )
    bridge.start()
    assert bridge.state == "error"
    assert bridge.done.is_set()
    assert not bridge._threads
    assert all(p.closed for p in (bridge.llm, bridge.stt, bridge.tts))


def test_real_osc_udp_packet_after_mock_dialogue(cfg):
    import socket
    from pythonosc.osc_message import OscMessage

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2)
    cfg["osc"]["vrc_port"] = receiver.getsockname()[1]
    bridge = VRChatLLMBridge(
        cfg,
        audio_factory=FakeAudio,
        llm_factory=FakeProvider,
        stt_factory=FakeProvider,
        tts_factory=FakeProvider,
        player=lambda *args: None,
    )
    thread = threading.Thread(target=bridge.start)
    thread.start()
    try:
        assert bridge.ready.wait(2)
        bridge.submit_text("hello")
        packets = []
        while not packets or packets[-1].address != "/chatbox/input":
            packets.append(OscMessage(receiver.recv(65535)))
        packet = packets[-1]
        assert packet.params == ["(AI喜多郁代) 你好。", True, False]
    finally:
        bridge.stop()
        thread.join(3)
        receiver.close()
    assert not thread.is_alive()
