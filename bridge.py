#!/usr/bin/env python3
"""
VRChat LLM Bridge — 系统音频捕获 + 语音识别 + LLM 对话

捕获系统扬声器输出(VRChat 中他人的语音),
通过 Whisper 转文字,送 LLM 生成回复,OSC 写回聊天框.

用法:
    pip install -r requirements.txt
    # 编辑 config.yaml 填入 API key
    python bridge.py
"""

import sys
import os
import time
import threading
import logging
import queue
import tempfile
import wave
from pathlib import Path
from collections import deque
import io
import base64
import requests as req_lib

import yaml
import numpy as np
import sounddevice as sd
from pythonosc import udp_client

# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vrchat-llm")

# 屏蔽 httpx / openai 的 HTTP 请求日志
for noisy in ("httpx", "openai", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ============================================================
# 配置
# ============================================================

def load_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(__file__).parent / path
    if not cfg_path.exists():
        log.error("找不到 %s,请先编辑 config.yaml", cfg_path)
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg["llm"]["provider"]
        self.model = ""
        self._init_client()

    def _init_client(self):
        from openai import OpenAI
        p = self.provider
        llm = self.cfg["llm"]

        if p == "openai":
            key = llm["openai"]["api_key"] or os.getenv("OPENAI_API_KEY")
            base = llm["openai"].get("base_url") or None
            self.model = llm["openai"]["model"]
            self.client = OpenAI(api_key=key, base_url=base or None)

        elif p == "openai_compatible":
            key = llm["openai_compatible"]["api_key"]
            base = llm["openai_compatible"]["base_url"]
            self.model = llm["openai_compatible"]["model"]
            self.client = OpenAI(api_key=key, base_url=base)

        elif p == "ollama":
            base = llm["ollama"]["base_url"]
            self.model = llm["ollama"]["model"]
            self.client = OpenAI(api_key="ollama", base_url=base)

        else:
            log.error("不支持的 provider: %s", p)
            sys.exit(1)

    def chat(self, messages: list, on_token=None) -> str:
        chat_cfg = self.cfg["chat"]
        max_tokens = chat_cfg.get("max_tokens", 300)
        temperature = chat_cfg.get("temperature", 0.9)

        stream = on_token is not None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            full = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    on_token(delta.content)
                    full.append(delta.content)
            return "".join(full)
        else:
            return response.choices[0].message.content

# ============================================================
# STT 客户端(支持多种语音识别引擎)
# ============================================================

class STTClient:
    """语音转文字,支持 OpenAI Whisper / Qwen-ASR / FunASR"""

    def __init__(self, cfg: dict):
        stt_cfg = cfg.get("stt", {})
        self.provider = stt_cfg.get("provider", "openai_whisper")

        if self.provider == "openai_whisper":
            from openai import OpenAI
            key = stt_cfg.get("openai_whisper", {}).get("api_key", "") or os.getenv("OPENAI_API_KEY")
            base = stt_cfg.get("openai_whisper", {}).get("base_url", None)
            self.client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)

        elif self.provider == "groq_whisper":
            from openai import OpenAI
            key = stt_cfg.get("groq_whisper", {}).get("api_key", "")
            self.client = OpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            )

        elif self.provider == "dashscope_qwen_asr":
            self.qwen_api_key = stt_cfg.get("dashscope_qwen_asr", {}).get("api_key", "")
            self.qwen_model = stt_cfg.get("dashscope_qwen_asr", {}).get("model", "qwen3-asr-flash")

        elif self.provider == "funasr":
            log.info("🔧 加载 FunASR 模型(首次运行会下载,约 1-2 GB)...")
            try:
                from funasr import AutoModel
            except ImportError:
                log.error("请先安装 funasr: pip install funasr")
                sys.exit(1)

            funasr_cfg = stt_cfg.get("funasr", {})
            model_name = funasr_cfg.get("model", "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
            device = funasr_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")
            self.funasr_model = AutoModel(
                model=model_name,
                vad_model="fsmn-vad" if "vad" not in model_name else None,
                punc_model="ct-punc" if "punc" not in model_name else None,
                device=device,
                disable_update=True,
            )
            log.info("   FunASR 模型加载完成,设备: %s", device)

        else:
            log.error("不支持的 STT provider: %s", self.provider)
            sys.exit(1)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """将 PCM 音频数据转文字."""
        if self.provider == "funasr":
            return self._transcribe_funasr(audio_data, sample_rate)
        elif self.provider == "dashscope_qwen_asr":
            return self._transcribe_qwen_asr(audio_data, sample_rate)
        else:
            return self._transcribe_openai(audio_data, sample_rate)

    def _transcribe_openai(self, audio_data: bytes, sample_rate: int) -> str:
        model = "whisper-large-v3" if self.provider == "groq_whisper" else "whisper-1"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
        try:
            with open(tmp_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="text",
                )
            return result.strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _transcribe_qwen_asr(self, audio_data: bytes, sample_rate: int) -> str:
        """Qwen-ASR (DashScope) — multipart 上传 WAV 文件."""
        import requests as req

        # 写临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

        try:
            with open(tmp_path, "rb") as f:
                resp = req.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.qwen_api_key}"},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"model": self.qwen_model},
                )
            if resp.status_code == 200:
                return resp.json().get("text", "").strip()
            else:
                log.error("Qwen ASR HTTP %d: %s", resp.status_code, resp.text[:200])
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _transcribe_funasr(self, audio_data: bytes, sample_rate: int) -> str:
        # 将 PCM bytes 转为 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = self.funasr_model.generate(
            input=audio_np,
            cache={},
            language="zh",
            use_itn=True,          # 逆文本正则化(数字/符号)
            batch_size_s=60,       # 60 秒一批
        )

        if result and len(result) > 0:
            text = result[0].get("text", "").strip()
            return text
        return ""

# ============================================================
# 音频捕获器(PyAudio WASAPI loopback)
# ============================================================

class AudioCapture:
    """捕获系统扬声器输出,用能量检测 VAD 切分语音段."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.1):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.chunk_duration = chunk_duration

        self._stream = None
        self._pa = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # VAD 状态(自适应阈值)
        self._noise_floor: float = 0.0
        self._noise_alpha: float = 0.02
        self._speech_margin: float = 3.0  # 高于噪底 RMS 差值才触发语音
        self._silence_frames: int = 0
        self._max_silence_frames: int = int(1.5 / chunk_duration)
        self._max_speech_frames: int = int(18.0 / chunk_duration)
        self._min_speech_frames: int = int(0.3 / chunk_duration)
        self._speech_buffer: list[bytes] = []
        self._speech_frames: int = 0
        self.current_rms: float = 0.0
        self.callback = None

    def list_devices(self):
        """列出 PyAudio 设备."""
        import pyaudio
        p = pyaudio.PyAudio()
        print("\n可用的 PyAudio 设备:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info['name']
            inp = info['maxInputChannels']
            outp = info['maxOutputChannels']
            sr = int(info['defaultSampleRate'])
            hostapi = p.get_host_api_info_by_index(info['hostApi'])['name']
            print(f"  [{i}] {name}")
            print(f"      输入:{inp}ch  输出:{outp}ch  采样率:{sr}  API:{hostapi}")
        p.terminate()
        print()

    def _find_wasapi_speaker(self, p) -> int | None:
        """查找 WASAPI 扬声器输出设备(用于 loopback)."""
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                api = p.get_host_api_info_by_index(info['hostApi'])
                if 'WASAPI' in api['name']:
                    name = info['name'].lower()
                    if 'speaker' in name or '扬声器' in name or '耳机' in name or 'headphone' in name:
                        return i
        # 回退:任意 WASAPI 输出
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                api = p.get_host_api_info_by_index(info['hostApi'])
                if 'WASAPI' in api['name']:
                    return i
        return None

    def _find_input_device(self, p) -> int | None:
        """查找立体声混音 / Virtual Cable(WASAPI 优先)"""
        best, best_score = None, -1
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] == 0:
                continue
            api = p.get_host_api_info_by_index(info['hostApi'])['name']
            name = info['name'].lower()

            # 基础优先级
            if 'stereo mix' in name or '立体声混音' in name:
                base = 100
            elif 'mixline' in name:
                base = 95
            elif 'loopback' in name:
                base = 80
            elif 'virtual desktop audio' in name:
                base = 70
            elif 'virtual cable' in name or 'virtual audio' in name:
                base = 60
            elif '主声音' in name:
                base = 40
            else:
                continue

            # WASAPI 加分(比 MME/DirectSound 更可靠)
            if 'WASAPI' in api:
                base += 10
            elif 'WDM-KS' in api:
                base += 5

            if base > best_score:
                best_score, best = base, i
        return best

    def start(self, device: int | None = None):
        import pyaudio
        self._pa = p = pyaudio.PyAudio()

        # 策略 A: WASAPI loopback(PyAudio 原生支持)
        wasapi_dev = self._find_wasapi_speaker(p)
        if wasapi_dev is not None:
            info = p.get_device_info_by_index(wasapi_dev)
            log.info("🎤 WASAPI Loopback: [%d] %s", wasapi_dev, info['name'])
            try:
                self._stream = p.open(
                    format=pyaudio.paInt16,
                    channels=info['maxOutputChannels'],
                    rate=int(info['defaultSampleRate']),
                    input=True,
                    input_device_index=wasapi_dev,
                    frames_per_buffer=int(info['defaultSampleRate'] * self.chunk_duration),
                    as_loopback=True,
                )
                # 重采样到目标采样率
                self._loopback_sr = int(info['defaultSampleRate'])
                self._loopback_channels = info['maxOutputChannels']
                log.info("✅ 已连接 (loopback, %dHz, %dch)", self._loopback_sr, self._loopback_channels)
                self._use_pyaudio = True
                self._stop.clear()
                self._thread = threading.Thread(target=self._read_loop_pyaudio, daemon=True)
                self._thread.start()
                return True
            except Exception as e:
                log.debug("WASAPI loopback 不支持,跳过: %s", e)

        # 策略 B: 传统输入设备
        input_dev = device if device is not None else self._find_input_device(p)
        if input_dev is not None:
            info = p.get_device_info_by_index(input_dev)
            native_sr = int(info['defaultSampleRate'])
            log.info("🎤 输入设备: [%d] %s (原生 %dHz)", input_dev, info['name'], native_sr)
            try:
                self._stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=native_sr,  # 用原生采样率,后面会重采样
                    input=True,
                    input_device_index=input_dev,
                    frames_per_buffer=int(native_sr * self.chunk_duration),
                )
                log.info("✅ 已连接 (%dHz → 重采样至 %dHz)", native_sr, self.sample_rate)
                self._use_pyaudio = True
                self._loopback_sr = native_sr
                self._loopback_channels = 1
                self._stop.clear()
                self._thread = threading.Thread(target=self._read_loop_pyaudio, daemon=True)
                self._thread.start()
                return True
            except Exception as e:
                log.warning("输入设备失败: %s", e)

        p.terminate()
        self._pa = None
        log.error("所有设备均无法打开")
        self._print_audio_help()
        return False

    def _read_loop_pyaudio(self):
        """后台线程 - 持续从 PyAudio 流读取音频数据"""
        while not self._stop.is_set():
            try:
                data = self._stream.read(self._stream._frames_per_buffer, exception_on_overflow=False)
            except Exception:
                time.sleep(0.1)
                continue

            # 转为 numpy 处理声道混合 + 重采样
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # 多声道 → 单声道
            if self._loopback_channels > 1:
                audio_np = audio_np.reshape(-1, self._loopback_channels).mean(axis=1)

            # 重采样到目标采样率(简单线性插值)
            if self._loopback_sr != self.sample_rate:
                ratio = self.sample_rate / self._loopback_sr
                in_len = len(audio_np)
                out_len = int(in_len * ratio)
                indices = np.linspace(0, in_len - 1, out_len)
                audio_np = np.interp(indices, np.arange(in_len), audio_np)

            # 拆成 chunk_size 的帧
            audio_i16 = audio_np.astype(np.int16)
            for i in range(0, len(audio_i16) - self.chunk_size + 1, self.chunk_size):
                chunk = audio_i16[i:i + self.chunk_size]
                self._process_audio_frame(chunk)

    def _process_audio_frame(self, indata: np.ndarray):
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        self.current_rms = rms

        if self._noise_floor == 0.0:
            self._noise_floor = rms
        else:
            self._noise_floor = (self._noise_alpha * rms + (1 - self._noise_alpha) * self._noise_floor)

        # 阈值 = 噪底 × 1.5,最低 +1.5
        threshold = max(self._noise_floor * 1.5, self._noise_floor + 1.5)
        audio_bytes = indata.tobytes()

        if rms > threshold:
            self._speech_buffer.append(audio_bytes)
            self._speech_frames += 1
            self._silence_frames = 0
            if self._speech_frames >= self._max_speech_frames:
                self._on_speech_end()
        else:
            self._silence_frames += 1
            if self._speech_frames > 0:
                self._speech_buffer.append(audio_bytes)
            if (self._silence_frames >= self._max_silence_frames
                    and self._speech_frames >= self._min_speech_frames):
                self._on_speech_end()

    def _on_speech_end(self):
        if not self._speech_buffer:
            return
        audio_data = b"".join(self._speech_buffer)
        duration = len(audio_data) / (2 * self.sample_rate)
        self._speech_buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        if duration < 0.3:
            return
        log.info("🎙 检测到语音 (%.1f 秒)", duration)
        if self.callback:
            self.callback(audio_data)

    def _print_audio_help(self):
        log.info("")
        log.info("🔧 音频捕获故障排除:")
        log.info("   1. 运行 python bridge.py --list-devices 查看 PyAudio 设备")
        log.info("   2. 在 config.yaml 的 audio.device 填入可用的输入设备编号")

    def stop(self):
        self._stop.set()
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa:
            self._pa.terminate()
            self._pa = None

    def set_callback(self, callback):
        self.callback = callback

# ============================================================
# 对话管理
# ============================================================

class ConversationManager:
    def __init__(self, cfg: dict):
        self.system_prompt = cfg.get("system_prompt", "").strip()
        self.max_history = cfg["chat"].get("max_history", 20)
        self.history: deque = deque(maxlen=self.max_history)

    def build_messages(self, user_text: str) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for entry in self.history:
            messages.append(entry)
        messages.append({"role": "user", "content": user_text})
        return messages

    def record(self, user_text: str, assistant_text: str):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

    def reset(self):
        self.history.clear()

# ============================================================
# 流式句子分割
# ============================================================

class SentenceSplitter:
    def __init__(self, split_chars: list, callback):
        self.split_chars = set(split_chars)
        self.callback = callback
        self.buffer = ""

    def feed(self, token: str):
        self.buffer += token
        for ch in self.split_chars:
            idx = self.buffer.find(ch)
            if idx != -1:
                sentence = self.buffer[: idx + len(ch)]
                self.buffer = self.buffer[idx + len(ch):]
                if sentence.strip():
                    self.callback(sentence.strip())
                return

    def flush(self):
        if self.buffer.strip():
            self.callback(self.buffer.strip())
            self.buffer = ""

# ============================================================
# MiniMax TTS 语音合成
# ============================================================

class TTSClient:
    """MiniMax 语音合成,输出为 WAV bytes。"""

    def __init__(self, cfg: dict):
        tts_cfg = cfg.get("tts", {})
        self.enabled = tts_cfg.get("enabled", False)
        self.output_device = tts_cfg.get("output_device", None)
        self.api_key = tts_cfg.get("minimax", {}).get("api_key", "")
        self.voice_id = tts_cfg.get("minimax", {}).get("voice_id", "female-shaonv")
        self.speed = tts_cfg.get("minimax", {}).get("speed", 1.0)
        self.vol = tts_cfg.get("minimax", {}).get("vol", 1.0)
        self.sample_rate = 32000

    def synthesize(self, text: str) -> tuple[bytes, int] | None:
        """合成语音,返回 (PCM bytes, sample_rate) 或 None。"""
        if not self.enabled or not self.api_key or not text.strip():
            return None
        try:
            resp = req_lib.post(
                "https://api.minimaxi.com/v1/t2a_v2",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "speech-01-turbo",
                    "text": text,
                    "stream": False,
                    "voice_setting": {
                        "voice_id": self.voice_id,
                        "speed": self.speed,
                        "vol": self.vol,
                    },
                    "audio_setting": {
                        "sample_rate": self.sample_rate,
                        "format": "pcm",
                    },
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                # 检查业务错误
                base = data.get("base_resp", {})
                if base.get("status_code") != 0:
                    log.warning("MiniMax TTS 错误: %s", base.get("status_msg", ""))
                    return None
                # MiniMax 返回 hex 编码的音频
                hex_audio = data.get("data", {}).get("audio", "")
                if hex_audio:
                    return bytes.fromhex(hex_audio), self.sample_rate
                log.warning("MiniMax TTS 返回空音频")
        except Exception as e:
            log.warning("TTS 合成失败: %s", e)
        return None

# ============================================================
# 主桥接
# ============================================================

class VRChatLLMBridge:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # OSC(仅发送)
        osc_cfg = cfg["osc"]
        self.osc = udp_client.SimpleUDPClient(osc_cfg["host"], osc_cfg["vrc_port"])

        # LLM
        self.llm = LLMClient(cfg)

        # STT
        self.stt = STTClient(cfg)

        # 对话
        self.conv = ConversationManager(cfg)

        # 流式
        streaming_cfg = cfg.get("streaming", {})
        self.streaming_enabled = streaming_cfg.get("enabled", True)
        self.split_chars = streaming_cfg.get("split_chars", [".", "！", "？", "!", "?", "\n", "~"])
        self.sentence_delay = streaming_cfg.get("sentence_delay", 0.8)

        # 音频捕获
        audio_cfg = cfg.get("audio", {})
        self.audio = AudioCapture(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            chunk_duration=audio_cfg.get("chunk_duration", 0.1),
        )
        self.audio.set_callback(self._on_speech)

        # 状态
        self._processing_lock = threading.Lock()
        self._processing = False
        self._stt_queue = queue.Queue()
        self._stt_stop = threading.Event()
        self._stt_thread: threading.Thread | None = None
        # 触发模式
        self.trigger_prefix = cfg["chat"].get("trigger_prefix", "")
        self.reply_prefix = cfg["chat"].get("reply_prefix", "")
        self._turn_count = 0

        # TTS
        self.tts = TTSClient(cfg)
        self._latest_tts_wav: bytes | None = None
        self._tts_queue = queue.Queue()
        self._tts_playing = False
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def _on_speech(self, audio_data: bytes):
        """收到一段语音,放入队列异步处理."""
        self._stt_queue.put(audio_data)

    def _stt_worker(self):
        """后台线程:从队列取音频,转文字,送 LLM."""
        while not self._stt_stop.is_set():
            try:
                audio_data = self._stt_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                import hashlib
                h = hashlib.md5(audio_data).hexdigest()[:8]
                log.debug("STT 音频 MD5=%s len=%d", h, len(audio_data))
                text = self.stt.transcribe(audio_data)
                if text:
                    log.info("📝 识别结果: %s", text)
                    self._on_text(text)
                else:
                    log.debug("STT 返回空文本")
            except Exception as e:
                log.error("STT 识别失败: %s", e)

    def _on_text(self, text: str):
        """收到识别文字,过滤不连贯语句后送 LLM."""
        if not text.strip():
            return

        # 触发前缀
        if self.trigger_prefix:
            if not text.startswith(self.trigger_prefix):
                log.debug("文字不以 '%s' 开头,跳过", self.trigger_prefix)
                return
            text = text[len(self.trigger_prefix):].strip()
            if not text:
                return

        # 过滤不连贯/噪音识别结果
        if not self._is_coherent(text):
            log.debug("🚫 过滤不连贯语句: %s", text)
            return

        if self._processing:
            log.debug("正在处理上一条,跳过")
            return

        t = threading.Thread(target=self._process_message, args=(text,), daemon=True)
        t.start()

    @staticmethod
    def _is_coherent(text: str) -> bool:
        """判断识别结果是否连贯(非噪音/乱码/幻觉)."""
        # 已知幻觉短语黑名单
        if any(b in text for b in ("明镜与点点", "不吝点赞", "订阅 转发", "谢谢大家")):
            return False
        # 必须含中文或日文
        import re
        if not re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text):
            return False
        # 去掉空格
        clean = text.replace(" ", "")
        if len(clean) < 4:
            return False

        # 检测过度重复:同一个 2-3 字片段出现 4 次以上
        for n in (2, 3):
            if len(clean) >= n * 4:
                chunks = [clean[i:i+n] for i in range(0, len(clean) - n + 1, n)]
                if len(set(chunks)) <= 2 and len(chunks) >= 4:
                    return False

        # 检测单字重复("嗯嗯嗯嗯嗯" / "有有有有有")
        for ch in set(clean):
            if clean.count(ch) > len(clean) * 0.7 and len(clean) > 4:
                return False

        return True

    def _process_message(self, user_text: str):
        with self._processing_lock:
            if self._processing:
                return
            self._processing = True

        try:
            if user_text.strip().lower() in ("/reset", "/重置"):
                self.conv.reset()
                self._turn_count = 0
                self._send_osc("(对话历史已清除 ✓)")
                return

            # 每 10 句话自动重置上下文
            self._turn_count += 1
            if self._turn_count >= 10:
                self.conv.reset()
                self._turn_count = 0
                log.info("🔄 已重置对话上下文")

            self._set_typing(True)

            messages = self.conv.build_messages(user_text)

            if self.streaming_enabled:
                self._stream_reply(messages, user_text)
            else:
                reply = self.llm.chat(messages)
                self.conv.record(user_text, reply)
                self._send_osc(reply)
                self._play_tts(reply)

        except Exception as e:
            log.exception("LLM 异常")
            self._send_osc(f"(出错: {e})")
        finally:
            self._set_typing(False)
            with self._processing_lock:
                self._processing = False

    def _stream_reply(self, messages: list, user_text: str):
        full_reply = []
        sentences_sent = []

        def on_sentence(s: str):
            sentences_sent.append(s)
            self._send_osc(s)
            time.sleep(self.sentence_delay)

        splitter = SentenceSplitter(self.split_chars, on_sentence)

        def on_token(t: str):
            full_reply.append(t)
            splitter.feed(t)

        reply = self.llm.chat(messages, on_token=on_token)
        splitter.flush()

        if not sentences_sent and reply.strip():
            self._send_osc(reply.strip())

        self.conv.record(user_text, "".join(full_reply))

    def _send_osc(self, text: str):
        if not text:
            return
        if self.reply_prefix:
            text = self.reply_prefix + text
        # VRChat 格式:\n → \v,第二个参数 True = 激活气泡
        text = text.replace("\n", "\v")
        log.info("📤 OSC: %s", text)
        self.osc.send_message("/chatbox/input", [text, True])

    def _set_typing(self, typing: bool):
        try:
            self.osc.send_message("/chatbox/typing", typing)
            log.info("⌨ OSC typing: %s", "ON" if typing else "OFF")
        except Exception:
            pass

    def _play_tts(self, text: str):
        """将 TTS 请求放入队列，逐句播放。"""
        if not self.tts.enabled:
            return
        # 按标点拆句
        import re
        sentences = re.split(r'[。！？!?~]', text)
        for s in sentences:
            s = s.strip()
            if s:
                self._tts_queue.put(s)

    def _tts_worker(self):
        """后台线程：逐句合成并播放 TTS。"""
        while True:
            text = self._tts_queue.get()  # 阻塞等待
            self._tts_playing = True
            result = self.tts.synthesize(text)
            if result:
                pcm, sr = result
                log.info("🔊 TTS: %s", text[:30])
                import io as _io
                wav_buf = _io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm)
                self._latest_tts_wav = wav_buf.getvalue()
                # 等待前端播放完毕（粗略估算 = 音频时长）
                duration = len(pcm) / (2 * sr)
                time.sleep(duration + 0.3)
            self._tts_playing = False

    def start(self):
        log.info("🤖 VRChat LLM Bridge (语音版) 启动中...")
        log.info("")

        # 启动 STT 线程
        self._stt_stop.clear()
        self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
        self._stt_thread.start()

        # 启动音频捕获
        device = self.cfg.get("audio", {}).get("device", None)
        if not self.audio.start(device):
            log.error("音频捕获启动失败")
            return

        log.info("")
        log.info("🔌 测试 OSC...")
        self.osc.send_message("/chatbox/input", ["(AI喜多郁代) 已连接 ✓", True])
        log.info("   OSC → VRChat OK")

        log.info("")
        log.info("✅ 桥接已就绪！")
        log.info("   🎤 监听语音 → DeepSeek → 聊天框气泡")
        log.info("   📊 每 5 秒输出音频电平")
        log.info("   ⏹ 按 Ctrl+C 退出")
        log.info("")
        if self.trigger_prefix:
            log.info("   ⚠ 触发前缀: '%s' — 只有以此开头的语音才会触发 AI", self.trigger_prefix)
        else:
            log.info("   💡 所有语音都会触发 AI 回复.如果太敏感,在 config.yaml 设置 trigger_prefix")
        print("-" * 50)

        # 定期输出音频电平(诊断用)
        def level_reporter():
            while not self._stt_stop.is_set():
                time.sleep(5)
                rms = self.audio.current_rms
                noise = self.audio._noise_floor
                threshold = max(noise * 1.5, noise + 1.5)
                bar = "█" * int(min(rms / max(threshold, 0.001), 30))
                if rms > threshold:
                    log.info("🔊 电平: %.4f  噪底: %.4f  阈值: %.4f  %s (语音)",
                             rms, noise, threshold, bar)
                else:
                    log.debug("🎧 电平: %.4f  噪底: %.4f  阈值: %.4f  %s",
                              rms, noise, threshold, bar)

        reporter_thread = threading.Thread(target=level_reporter, daemon=True)
        reporter_thread.start()

        try:
            while not self._stt_stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            log.info("正在退出...")
            self._send_osc("已断开连接,下次见~")
            time.sleep(0.3)
            self._stt_stop.set()
            self.audio.stop()

# ============================================================
# 入口
# ============================================================

def main():
    if "--list-devices" in sys.argv:
        AudioCapture().list_devices()
        return

    cfg = load_config()
    level = cfg.get("logging", {}).get("level", "INFO")
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))

    bridge = VRChatLLMBridge(cfg)
    bridge.start()

if __name__ == "__main__":
    main()
