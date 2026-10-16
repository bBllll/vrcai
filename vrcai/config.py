"""Validated configuration, secret redaction and atomic persistence."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "audio": {
        "device": None,
        "sample_rate": 16000,
        "chunk_duration": 0.02,
        "silence_duration": 0.7,
        "min_speech_duration": 0.16,
        "max_speech_duration": 15.0,
        "pre_roll_duration": 0.2,
        "energy_threshold": 150.0,
        "noise_ratio": 2.5,
    },
    "stt": {
        "provider": "groq_whisper",
        "language": "",
        "groq_whisper": {"api_key": "", "model": "whisper-large-v3"},
        "openai_whisper": {"api_key": "", "base_url": "", "model": "whisper-1"},
        "dashscope_qwen_asr": {
            "api_key": "",
            "model": "qwen3-asr-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "funasr": {
            "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "device": "cpu",
        },
    },
    "llm": {
        "provider": "openai_compatible",
        "openai": {"api_key": "", "base_url": "", "model": "gpt-4o-mini"},
        "openai_compatible": {"api_key": "", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
        "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
    },
    "tts": {
        "enabled": False,
        "output_device": None,
        "mute_capture": True,
        "minimax": {
            "api_key": "",
            "voice_id": "female-shaonv",
            "model": "speech-01-turbo",
            "base_url": "https://api.minimaxi.com/v1",
            "speed": 1.0,
            "vol": 1.0,
            "sample_rate": 32000,
        },
    },
    "osc": {"host": "127.0.0.1", "vrc_port": 9000, "max_chars": 144, "send_interval": 1.5},
    "chat": {
        "trigger_prefix": "",
        "reply_prefix": "(AI喜多郁代) ",
        "max_history": 20,
        "max_tokens": 300,
        "temperature": 0.7,
        "queue_size": 3,
        "max_input_age": 20.0,
        "listen_mode": "continuous",
    },
    "system_prompt": "你是喜多郁代，性格开朗活泼。你和朋友在 VRChat 里聊天，说中文。回复简洁自然，尽量在30字以内。",
    "streaming": {"enabled": False, "split_chars": ["。", "！", "？", "!", "?", "\n", "~"], "sentence_delay": 0.8},
    "network": {"timeout": 20.0, "max_retries": 0},
    "logging": {"level": "INFO"},
}


class ConfigError(ValueError):
    pass


def merge(base, updates, *, preserve_empty_keys=False):
    """Merge only known fields. Blank secrets mean keep; null means delete."""
    if not isinstance(updates, dict):
        raise ConfigError("配置必须是 JSON/YAML 对象")
    for key, value in updates.items():
        if key not in base:
            raise ConfigError(f"未知配置项: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} 必须是对象")
            merge(base[key], value, preserve_empty_keys=preserve_empty_keys)
        elif key == "api_key" and preserve_empty_keys:
            if value != "":
                base[key] = "" if value is None else value
        else:
            base[key] = copy.deepcopy(value)
    return base


def validate(cfg):
    def number(section, key, lo, hi, integer=False):
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{section}.{key} 必须是有限数字")
        if integer and not isinstance(value, int):
            raise ConfigError(f"{section}.{key} 必须是整数")
        if not lo <= value <= hi:
            raise ConfigError(f"{section}.{key} 范围为 {lo}～{hi}")

    def types(value, schema, path=""):
        for key, expected in schema.items():
            item, name = value[key], f"{path}{key}"
            if isinstance(expected, dict):
                types(item, expected, name + ".")
            elif isinstance(expected, bool) and not isinstance(item, bool):
                raise ConfigError(f"{name} 必须是布尔值")
            elif isinstance(expected, str) and (not isinstance(item, str) or len(item) > 20000):
                raise ConfigError(f"{name} 必须是长度不超过20000的字符串")

    types(cfg, DEFAULTS)
    for section, key, lo, hi, integer in [
        ("audio", "sample_rate", 8000, 48000, True),
        ("audio", "chunk_duration", 0.01, 0.1, False),
        ("audio", "silence_duration", 0.1, 3, False),
        ("audio", "min_speech_duration", 0.04, 2, False),
        ("audio", "max_speech_duration", 2, 30, False),
        ("audio", "pre_roll_duration", 0, 0.5, False),
        ("audio", "energy_threshold", 1, 10000, False),
        ("audio", "noise_ratio", 1.1, 10, False),
        ("chat", "max_history", 0, 200, True),
        ("chat", "max_tokens", 1, 8192, True),
        ("chat", "temperature", 0, 2, False),
        ("chat", "queue_size", 1, 10, True),
        ("chat", "max_input_age", 1, 120, False),
        ("osc", "vrc_port", 1, 65535, True),
        ("osc", "max_chars", 16, 144, True),
        ("osc", "send_interval", 1, 10, False),
        ("network", "timeout", 2, 120, False),
        ("network", "max_retries", 0, 2, True),
        ("streaming", "sentence_delay", 0, 10, False),
    ]:
        number(section, key, lo, hi, integer)
    for key in ("device",):
        value = cfg["audio"][key]
        if value is not None and (type(value) is not int or value < 0):
            raise ConfigError("audio.device 必须为空或非负整数")
    value = cfg["tts"]["output_device"]
    if value is not None and (type(value) is not int or value < 0):
        raise ConfigError("tts.output_device 必须为空或非负整数")
    for key, lo, hi in (("speed", 0.5, 2), ("vol", 0.1, 10)):
        value = cfg["tts"]["minimax"][key]
        if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
            raise ConfigError(f"tts.minimax.{key} 范围为 {lo}～{hi}")
    if cfg["tts"]["minimax"]["sample_rate"] != 32000:
        raise ConfigError("tts.minimax.sample_rate 当前必须为32000")
    if cfg["audio"]["min_speech_duration"] >= cfg["audio"]["max_speech_duration"]:
        raise ConfigError("最短语音时长必须小于最长片段时长")
    for section in ("stt", "llm"):
        if cfg[section]["provider"] not in DEFAULTS[section] or cfg[section]["provider"] in ("provider", "language"):
            raise ConfigError(f"不支持的 {section} 提供商")
        for name, provider in cfg[section].items():
            if not isinstance(provider, dict):
                continue
            if not provider.get("model", "").strip():
                raise ConfigError(f"{section}.{name}.model 不能为空")
    if cfg["chat"]["listen_mode"] not in ("continuous", "push_to_talk"):
        raise ConfigError("listen_mode 必须为 continuous 或 push_to_talk")
    if len(cfg["chat"]["reply_prefix"]) >= cfg["osc"]["max_chars"]:
        raise ConfigError("回复前缀必须短于聊天框长度限制")
    chars = cfg["streaming"]["split_chars"]
    if not isinstance(chars, list) or not chars or any(not isinstance(c, str) or len(c) != 1 for c in chars):
        raise ConfigError("split_chars 必须是非空的单字符列表")
    for section in ("stt", "llm", "tts"):
        for provider in cfg[section].values():
            if isinstance(provider, dict) and provider.get("base_url"):
                parsed = urlsplit(provider["base_url"])
                if (
                    parsed.scheme not in ("http", "https")
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ConfigError("Base URL 必须是有效的 HTTP(S) 地址，不能带凭据、查询参数或片段")
    if not cfg["osc"]["host"].strip() or len(cfg["osc"]["host"]) > 253:
        raise ConfigError("OSC 地址无效")
    if cfg["logging"]["level"] not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError("日志级别无效")
    return cfg


def redact(cfg):
    safe = copy.deepcopy(cfg)

    def walk(node):
        for key, value in list(node.items()):
            if key == "api_key":
                node[key] = ""
                node["key_configured"] = bool(value)
            elif isinstance(value, dict):
                walk(value)

    walk(safe)
    return safe


class ConfigStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else ROOT / "config.yaml"
        self._lock = threading.RLock()

    def load(self):
        with self._lock:
            cfg = copy.deepcopy(DEFAULTS)
            if self.path.exists():
                try:
                    with self.path.open(encoding="utf-8") as f:
                        merge(cfg, yaml.safe_load(f) or {})
                except yaml.YAMLError as exc:
                    raise ConfigError("config.yaml 格式错误，请检查缩进") from exc
            return validate(cfg)

    def update(self, updates):
        with self._lock:
            cfg = self.load()
            merge(cfg, updates, preserve_empty_keys=True)
            validate(cfg)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            return cfg


def load_config(path="config.yaml"):
    return ConfigStore(ROOT / path).load()
