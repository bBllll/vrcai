"""Windows loopback capture, bounded energy VAD and cancellable PCM playback."""

from __future__ import annotations

import math
import queue
import sys
import threading
from collections import deque

import numpy as np


def audio_backend():
    try:
        if sys.platform == "win32":
            import pyaudiowpatch as pa
        else:
            import pyaudio as pa
        return pa
    except ImportError as exc:
        raise RuntimeError("缺少音频依赖，请运行 python -m pip install -r requirements.txt") from exc


def list_devices():
    backend = audio_backend()
    pa = backend.PyAudio()
    try:
        return [
            {
                "id": i,
                "name": d["name"],
                "inputs": int(d["maxInputChannels"]),
                "outputs": int(d["maxOutputChannels"]),
                "sample_rate": int(d["defaultSampleRate"]),
                "loopback": bool(d.get("isLoopbackDevice", False)),
                "api": pa.get_host_api_info_by_index(d["hostApi"])["name"],
            }
            for i in range(pa.get_device_count())
            for d in [pa.get_device_info_by_index(i)]
        ]
    finally:
        pa.terminate()


def resample(data, source_rate, target_rate):
    if source_rate == target_rate:
        return data.astype(np.int16)
    from scipy.signal import resample_poly

    factor = math.gcd(int(source_rate), int(target_rate))
    result = resample_poly(data.astype(np.float32), target_rate // factor, source_rate // factor)
    return np.clip(result, -32768, 32767).astype(np.int16)


class VoiceSegmenter:
    """Noise floor learns only outside speech; all segments have a hard size cap."""

    def __init__(self, cfg, callback):
        self.cfg, self.callback = cfg, callback
        self.rate = cfg["sample_rate"]
        self.frame_size = int(self.rate * cfg["chunk_duration"])
        self.frame_seconds = self.frame_size / self.rate
        self.pre_roll = deque(maxlen=math.ceil(cfg["pre_roll_duration"] / self.frame_seconds))
        self.noise_floor = 0.0
        self.current_rms = 0.0
        self.threshold = cfg["energy_threshold"]
        self.reset()

    def reset(self):
        self.frames = []
        self.speech_seconds = 0.0
        self.silence_seconds = 0.0
        self.pre_roll.clear()

    def feed(self, frame):
        self.current_rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        self.threshold = max(self.cfg["energy_threshold"], self.noise_floor * self.cfg["noise_ratio"])
        speech = self.current_rms >= self.threshold
        if not self.frames and not speech:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * self.current_rms
            self.pre_roll.append(frame.tobytes())
            return
        if speech:
            if not self.frames:
                self.frames.extend(self.pre_roll)
                self.pre_roll.clear()
            self.speech_seconds += self.frame_seconds
            self.silence_seconds = 0.0
        else:
            self.silence_seconds += self.frame_seconds
        self.frames.append(frame.tobytes())
        if (
            self.silence_seconds + 1e-6 >= self.cfg["silence_duration"]
            or len(self.frames) * self.frame_seconds + 1e-6 >= self.cfg["max_speech_duration"]
        ):
            self.flush()

    def flush(self):
        clip = b"".join(self.frames) if self.speech_seconds + 1e-6 >= self.cfg["min_speech_duration"] else None
        self.reset()
        if clip:
            self.callback(clip)


class AudioCapture:
    def __init__(self, cfg, callback, should_capture=lambda: True, on_error=None):
        self.cfg, self.should_capture, self.on_error = cfg, should_capture, on_error
        self.vad = VoiceSegmenter(cfg, callback)
        self._stop = threading.Event()
        self._thread = self._stream = self._pa = None
        self._raw = queue.Queue(maxsize=50)
        self.device_name = ""
        self.dropped_frames = 0
        self._flush = threading.Event()

    @property
    def current_rms(self):
        return self.vad.current_rms

    def start(self):
        backend = audio_backend()
        self._pa = pa = backend.PyAudio()
        try:
            device = self.cfg["device"]
            if device is not None:
                info = pa.get_device_info_by_index(device)
                if info["maxInputChannels"] == 0:
                    if hasattr(pa, "get_wasapi_loopback_analogue_by_index"):
                        info = pa.get_wasapi_loopback_analogue_by_index(device)
                    else:
                        raise RuntimeError("请选择输入设备或 Loopback 设备")
            elif hasattr(pa, "get_default_wasapi_loopback"):
                info = pa.get_default_wasapi_loopback()
            else:
                raise RuntimeError("自动系统音频捕获需要 Windows；请手动选择输入设备")
            self.device_name = info["name"]
            self._native_rate = int(info["defaultSampleRate"])
            self._channels = int(info["maxInputChannels"])
            if self._channels < 1:
                raise RuntimeError("所选设备不能录音")
            self._stop.clear()
            self._stream = pa.open(
                format=backend.paInt16,
                channels=self._channels,
                rate=self._native_rate,
                input=True,
                input_device_index=int(info["index"]),
                frames_per_buffer=int(self._native_rate * self.cfg["chunk_duration"]),
                stream_callback=self._callback,
            )
            self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
            self._thread.start()
        except Exception:
            self.stop()
            raise

    def _callback(self, data, frame_count, time_info, status):
        if not self._stop.is_set():
            try:
                # Gate at capture time as well as consumption time: buffered playback
                # audio must never become fresh input after TTS finishes.
                self._raw.put_nowait((data, self.should_capture()))
            except queue.Full:
                self.dropped_frames += 1
        return (None, audio_backend().paContinue)

    def _run(self):
        pending = np.empty(0, dtype=np.int16)
        was_allowed = False
        try:
            while not self._stop.is_set():
                if self._flush.is_set():
                    self.vad.flush()
                    self._flush.clear()
                    pending = np.empty(0, dtype=np.int16)
                try:
                    data, captured_allowed = self._raw.get(timeout=0.1)
                except queue.Empty:
                    if self._stream and not self._stream.is_active():
                        raise RuntimeError("音频流已停止，请检查设备连接后重新启动")
                    if not self.should_capture():
                        if was_allowed:
                            self.vad.flush()
                        self.vad.reset()
                        pending = np.empty(0, dtype=np.int16)
                        was_allowed = False
                    elif self.vad.frames:
                        # Some WASAPI drivers stop callbacks while the output is silent.
                        for _ in range(math.ceil(0.1 / self.vad.frame_seconds)):
                            self.vad.feed(np.zeros(self.vad.frame_size, dtype=np.int16))
                    continue
                allowed = self.should_capture()
                if not allowed or not captured_allowed:
                    if was_allowed:
                        self.vad.flush()
                    else:
                        self.vad.reset()
                    pending = np.empty(0, dtype=np.int16)
                    was_allowed = False
                    continue
                was_allowed = True
                samples = np.frombuffer(data, dtype=np.int16)
                if self._channels > 1:
                    samples = samples.reshape(-1, self._channels).astype(np.float32).mean(axis=1)
                samples = resample(samples, self._native_rate, self.cfg["sample_rate"])
                pending = np.concatenate((pending, samples))
                size = self.vad.frame_size
                offset = 0
                while offset + size <= len(pending):
                    self.vad.feed(pending[offset : offset + size])
                    offset += size
                pending = pending[offset:]
        except Exception as exc:
            if not self._stop.is_set() and self.on_error:
                self.on_error(exc)

    def flush(self):
        self._flush.set()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        stream, self._stream = self._stream, None
        try:
            if stream:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        finally:
            if self._pa:
                self._pa.terminate()
                self._pa = None


def play_pcm(pcm, rate, device, cancelled):
    """Backend playback also works without a browser and supports virtual cables."""
    if cancelled():
        return
    backend = audio_backend()
    pa = backend.PyAudio()
    stream = None
    try:
        info = pa.get_device_info_by_index(device) if device is not None else pa.get_default_output_device_info()
        if info["maxOutputChannels"] < 1:
            raise RuntimeError("所选 TTS 设备不能播放音频")
        output_rate = int(info["defaultSampleRate"])
        channels = min(2, int(info["maxOutputChannels"]))
        samples = resample(np.frombuffer(pcm, dtype=np.int16), rate, output_rate)
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        raw = samples.tobytes()
        stream = pa.open(
            format=backend.paInt16,
            channels=channels,
            rate=output_rate,
            output=True,
            output_device_index=int(info["index"]),
        )
        size = max(1, int(output_rate * 0.02)) * channels * 2
        for offset in range(0, len(raw), size):
            if cancelled():
                break
            stream.write(raw[offset : offset + size])
    finally:
        if stream:
            try:
                stream.stop_stream()
            finally:
                stream.close()
        pa.terminate()
