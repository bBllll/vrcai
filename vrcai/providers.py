"""Provider adapters with bounded requests; credentials never enter public errors."""

from __future__ import annotations

import base64
import io
import os
import wave

import numpy as np
import requests


class ProviderError(RuntimeError):
    """A locally authored public message, never a raw remote response."""


def api_key(cfg, section, provider):
    value = cfg[section][provider].get("api_key", "")
    env = {
        ("stt", "groq_whisper"): ("GROQ_API_KEY",),
        ("stt", "openai_whisper"): ("OPENAI_API_KEY",),
        ("stt", "dashscope_qwen_asr"): ("DASHSCOPE_API_KEY",),
        ("llm", "openai"): ("OPENAI_API_KEY",),
        ("llm", "openai_compatible"): ("LLM_API_KEY", "DEEPSEEK_API_KEY"),
        ("tts", "minimax"): ("MINIMAX_API_KEY",),
    }
    return value or next((os.environ[e] for e in env.get((section, provider), ()) if os.environ.get(e)), "")


def check_credentials(cfg):
    missing = []
    for section in ("stt", "llm"):
        provider = cfg[section]["provider"]
        if provider not in ("funasr", "ollama") and not api_key(cfg, section, provider):
            missing.append(f"{section}.{provider}.api_key")
    if cfg["tts"]["enabled"] and not api_key(cfg, "tts", "minimax"):
        missing.append("tts.minimax.api_key")
    return missing


def safe_error(exc):
    if isinstance(exc, ProviderError):
        return str(exc)
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return "认证失败，请检查所选服务的 API Key 和权限"
    if status == 429:
        return "服务限流或额度不足，请稍后重试并检查账户额度"
    if "Timeout" in name or isinstance(exc, requests.Timeout):
        return "网络请求超时，请检查网络或服务地址"
    if "Connection" in name or isinstance(exc, requests.ConnectionError):
        return "无法连接服务，请检查网络和 Base URL"
    if status:
        return f"服务返回 HTTP {status}，请检查模型、地址及账户权限"
    if isinstance(exc, ImportError):
        return "缺少依赖，请按 README 安装所需组件"
    return "操作失败，请检查设备、模型配置和依赖（" + name + "）"


def wav_bytes(pcm, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def make_client(cfg, section, provider):
    from openai import OpenAI

    settings = cfg[section][provider]
    base_url = settings.get("base_url") or None
    if provider == "groq_whisper":
        base_url = "https://api.groq.com/openai/v1"
    return OpenAI(
        api_key=api_key(cfg, section, provider) or "ollama",
        base_url=base_url,
        timeout=cfg["network"]["timeout"],
        max_retries=cfg["network"]["max_retries"],
    )


class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        provider = cfg["llm"]["provider"]
        self.model = cfg["llm"][provider]["model"]
        self.client = make_client(cfg, "llm", provider)

    def chat(self, messages, on_token=None, cancelled=lambda: False):
        if cancelled():
            return ""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.cfg["chat"]["max_tokens"],
            temperature=self.cfg["chat"]["temperature"],
            stream=on_token is not None,
        )
        if on_token is None:
            return "" if cancelled() else (response.choices[0].message.content or "")
        parts = []
        try:
            for chunk in response:
                if cancelled():
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    parts.append(token)
                    on_token(token)
        finally:
            response.close()
        return "".join(parts)

    def close(self):
        self.client.close()


class STTClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = cfg["stt"]["provider"]
        self.settings = cfg["stt"][self.provider]
        self.client = None
        if self.provider == "funasr":
            from funasr import AutoModel

            self.model = AutoModel(model=self.settings["model"], device=self.settings["device"], disable_update=True)
        else:
            self.client = make_client(cfg, "stt", self.provider)

    def transcribe(self, pcm, sample_rate=16000):
        language = self.cfg["stt"]["language"]
        if self.provider == "funasr":
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
            samples = samples if sample_rate == 16000 else self._to_16k(samples, sample_rate)
            result = self.model.generate(input=samples, cache={}, use_itn=True, batch_size_s=30)
            return result[0].get("text", "").strip() if result else ""
        audio = wav_bytes(pcm, sample_rate)
        if self.provider == "dashscope_qwen_asr":
            uri = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
            options = {"enable_itn": True}
            if language:
                options["language"] = language
            result = self.client.chat.completions.create(
                model=self.settings["model"],
                stream=False,
                messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": uri}}]}],
                extra_body={"asr_options": options},
            )
            return (result.choices[0].message.content or "").strip()
        extra = {"language": language} if language else {}
        result = self.client.audio.transcriptions.create(
            model=self.settings["model"], file=("speech.wav", audio, "audio/wav"), response_format="text", **extra
        )
        return result.strip()

    @staticmethod
    def _to_16k(samples, rate):
        import math
        from scipy.signal import resample_poly

        factor = math.gcd(rate, 16000)
        return resample_poly(samples, 16000 // factor, rate // factor)

    def close(self):
        if self.client:
            self.client.close()


class TTSClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = requests.Session()

    def synthesize(self, text):
        settings = self.cfg["tts"]["minimax"]
        url = settings["base_url"].rstrip("/") + "/t2a_v2"
        with self.session.post(
            url,
            headers={"Authorization": "Bearer " + api_key(self.cfg, "tts", "minimax")},
            json={
                "model": settings["model"],
                "text": text,
                "stream": False,
                "voice_setting": {k: settings[k] for k in ("voice_id", "speed", "vol")},
                "audio_setting": {"sample_rate": 32000, "format": "pcm", "channel": 1},
            },
            timeout=(5, self.cfg["network"]["timeout"]),
        ) as response:
            response.raise_for_status()
            data = response.json()
        code = data.get("base_resp", {}).get("status_code", -1)
        if code != 0:
            # Provider status text may echo request data. Publish the code only.
            raise ProviderError(f"MiniMax 业务错误 {code}，请检查密钥、音色、模型和账户额度")
        audio = data.get("data", {}).get("audio", "")
        if not audio:
            raise ProviderError("MiniMax 返回空音频，请检查所选音色和模型")
        return bytes.fromhex(audio), 32000

    def close(self):
        self.session.close()
