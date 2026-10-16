import numpy as np
import pytest

from vrcai.audio import AudioCapture, VoiceSegmenter, resample
from vrcai.conversation import ConversationManager, SentenceSplitter, is_coherent


def test_short_noise_followed_by_long_silence_does_not_accumulate(cfg):
    clips = []
    vad = VoiceSegmenter(cfg["audio"], clips.append)
    vad.feed(np.full(vad.frame_size, 1000, np.int16))
    for _ in range(3000):
        vad.feed(np.zeros(vad.frame_size, np.int16))
    assert not clips
    assert not vad.frames
    assert len(vad.pre_roll) <= 10


def test_sustained_speech_is_not_learned_as_noise_and_has_hard_cap(cfg):
    clips = []
    cfg["audio"]["max_speech_duration"] = 2
    vad = VoiceSegmenter(cfg["audio"], clips.append)
    for _ in range(500):
        vad.feed(np.full(vad.frame_size, 1000, np.int16))
    assert len(clips) == 5
    assert all(len(c) <= 2 * 2 * vad.rate for c in clips)
    assert vad.noise_floor == 0


def test_vad_retains_word_onset_and_emits_after_pause(cfg):
    clips = []
    vad = VoiceSegmenter(cfg["audio"], clips.append)
    for _ in range(10):
        vad.feed(np.zeros(vad.frame_size, np.int16))
    for _ in range(12):
        vad.feed(np.full(vad.frame_size, 800, np.int16))
    for _ in range(35):
        vad.feed(np.zeros(vad.frame_size, np.int16))
    assert len(clips) == 1
    assert clips[0].startswith(b"\0" * (10 * vad.frame_size * 2))


def test_resampling_preserves_duration_and_filters_aliasing():
    source = (10000 * np.sin(2 * np.pi * 12000 * np.arange(4800) / 48000)).astype(np.int16)
    output = resample(source, 48000, 16000)
    assert len(output) == 1600
    assert np.sqrt(np.mean(output[100:-100].astype(float) ** 2)) < 100


def test_explicit_input_is_used_before_automatic_loopback(cfg, monkeypatch):
    from types import SimpleNamespace

    opened = []

    class FakePA:
        def get_device_info_by_index(self, index):
            assert index == 7
            return {"index": index, "name": "chosen input", "maxInputChannels": 1, "defaultSampleRate": 16000}

        def get_default_wasapi_loopback(self):
            pytest.fail("explicit selection must take priority")

        def open(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(stop_stream=lambda: None, close=lambda: None, is_active=lambda: True)

        def terminate(self):
            pass

    monkeypatch.setattr("vrcai.audio.audio_backend", lambda: SimpleNamespace(PyAudio=FakePA, paInt16=8))
    cfg["audio"]["device"] = 7
    capture = AudioCapture(cfg["audio"], lambda data: None)
    capture.start()
    capture.stop()
    assert opened[0]["input_device_index"] == 7
    assert not capture._thread.is_alive()


def test_audio_captured_while_muted_is_discarded_even_after_unmuting(cfg):
    import threading

    clips = []
    capture = AudioCapture(cfg["audio"], clips.append)
    capture._native_rate, capture._channels = 16000, 1
    size = capture.vad.frame_size
    for _ in range(12):
        capture._raw.put((np.full(size, 1000, np.int16).tobytes(), False))
    for _ in range(35):
        capture._raw.put((np.zeros(size, np.int16).tobytes(), True))
    worker = threading.Thread(target=capture._run)
    worker.start()
    try:
        # The worker consumes all packets; flush would expose any accepted speech.
        for _ in range(100):
            if capture._raw.empty():
                break
            threading.Event().wait(0.01)
        assert capture._raw.empty()
    finally:
        capture._stop.set()
        worker.join(2)
    capture.vad.flush()
    assert not clips


def test_ptt_release_flushes_valid_speech_even_without_new_device_packets(cfg):
    import threading

    clips = []
    capture = AudioCapture(cfg["audio"], clips.append)
    capture._native_rate, capture._channels = 16000, 1
    for _ in range(12):
        capture.vad.feed(np.full(capture.vad.frame_size, 1000, np.int16))
    capture.flush()
    worker = threading.Thread(target=capture._run)
    worker.start()
    try:
        for _ in range(100):
            if clips:
                break
            threading.Event().wait(0.01)
        assert len(clips) == 1
    finally:
        capture._stop.set()
        worker.join(2)


@pytest.mark.parametrize("text", ["你好", "为什么", "可以啊", "Hello there", "こんにちは", "谢谢大家"])
def test_short_and_multilingual_dialogue_is_allowed(text):
    assert is_coherent(text)


def test_sentence_splitter_processes_all_punctuation_in_text_order():
    sentences = []
    splitter = SentenceSplitter(["!", "?", "。"], sentences.append)
    splitter.feed("第一句？".replace("？", "?") + "第二句!第三句。尾巴")
    assert sentences == ["第一句?", "第二句!", "第三句。"]
    splitter.flush()
    assert sentences[-1] == "尾巴"
    assert not splitter.buffer


def test_long_unpunctuated_output_is_split_without_losing_characters():
    sentences = []
    splitter = SentenceSplitter(["。"], sentences.append, max_chars=10)
    text = "abcdefghijklmnopqrstuvwxy"
    splitter.feed(text)
    splitter.flush()
    assert "".join(sentences) == text
    assert max(map(len, sentences)) <= 10


def test_history_retains_complete_turns_and_does_not_reset_at_turn_ten(cfg):
    cfg["chat"]["max_history"] = 5
    conv = ConversationManager(cfg)
    for i in range(12):
        conv.record(str(i), "reply")
    messages = conv.build_messages("next")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "10"
    conv.reset()
    assert len(conv.build_messages("next")) == 2
