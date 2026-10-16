import base64
import io
import wave
from types import SimpleNamespace

from vrcai.providers import LLMClient, STTClient, TTSClient, safe_error


def test_qwen_uses_audio_chat_content_and_configurable_endpoint(cfg, monkeypatch):
    cfg["stt"]["provider"] = "dashscope_qwen_asr"
    cfg["stt"]["language"] = "zh"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="你好"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr("vrcai.providers.make_client", lambda *args: fake)
    stt = STTClient(cfg)
    assert stt.transcribe(b"\0" * 3200, 16000) == "你好"
    request = calls[0]
    assert request["model"] == "qwen3-asr-flash"
    assert request["extra_body"]["asr_options"]["language"] == "zh"
    content = request["messages"][0]["content"][0]
    assert content["type"] == "input_audio"
    encoded = content["input_audio"]["data"]
    assert encoded.startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600


def test_whisper_upload_is_in_memory_and_preserves_sample_rate(cfg, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return " hello "

    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    monkeypatch.setattr("vrcai.providers.make_client", lambda *args: fake)
    stt = STTClient(cfg)
    assert stt.transcribe(b"\0" * 2205 * 2, 22050) == "hello"
    assert calls[0]["file"][0] == "speech.wav"
    with wave.open(io.BytesIO(calls[0]["file"][1]), "rb") as wav:
        assert wav.getframerate() == 22050


def test_stream_ignores_usage_only_chunks_and_closes_on_cancellation(cfg, monkeypatch):
    class Stream:
        closed = False

        def __iter__(self):
            yield SimpleNamespace(choices=[])
            for text in ("one", "two", "three"):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

        def close(self):
            self.closed = True

    stream = Stream()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream)))
    monkeypatch.setattr("vrcai.providers.make_client", lambda *args: fake)
    llm = LLMClient(cfg)
    tokens = []
    result = llm.chat([], on_token=tokens.append, cancelled=lambda: len(tokens) == 1)
    assert result == "one" and tokens == ["one"]
    assert stream.closed


def test_minimax_uses_timeout_and_decodes_pcm(cfg, monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return {"base_resp": {"status_code": 0}, "data": {"audio": "00000100"}}

    client = TTSClient(cfg)
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: (calls.append((args, kwargs)), Response())[1])
    assert client.synthesize("hello") == (b"\0\0\1\0", 32000)
    assert calls[0][1]["timeout"] == (5, cfg["network"]["timeout"])
    assert calls[0][1]["json"]["audio_setting"]["format"] == "pcm"
    client.close()


def test_public_errors_never_echo_provider_exception_or_key():
    exc = RuntimeError("API failed: sk-very-private-secret https://server/?api_key=private")
    assert "secret" not in safe_error(exc)
    assert "private" not in safe_error(exc)
