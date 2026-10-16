"""Owned sessions, bounded queues, cancellation and shared UI control."""

from __future__ import annotations

import copy
import importlib.util
import logging
import queue
import threading
import time

from .audio import AudioCapture, play_pcm
from .config import ConfigError, merge, validate
from .conversation import ConversationManager, SentenceSplitter, is_coherent
from .providers import LLMClient, STTClient, TTSClient, api_key, check_credentials, safe_error

log = logging.getLogger("vrcai")


def drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def put_latest(q, item):
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass


class VRChatLLMBridge:
    def __init__(
        self,
        cfg,
        *,
        audio_factory=AudioCapture,
        llm_factory=LLMClient,
        stt_factory=STTClient,
        tts_factory=TTSClient,
        osc_factory=None,
        player=play_pcm,
    ):
        self.cfg = copy.deepcopy(cfg)
        self.conv = ConversationManager(cfg)
        self.stop_event, self.ready, self.done = threading.Event(), threading.Event(), threading.Event()
        self.push_to_talk = threading.Event()
        self._ptt_deadline = 0.0
        self._output_lock, self._status_lock = threading.RLock(), threading.Lock()
        self._epoch = self._tts_revision = 0
        self._tts_enabled = cfg["tts"]["enabled"]
        self._muted_until = 0.0
        self._playing = threading.Event()
        self._inputs = queue.Queue(maxsize=cfg["chat"]["queue_size"])
        self._tts = queue.Queue(maxsize=32)
        self._threads = []
        self._last_osc = 0.0
        self.state, self.error = "starting", ""
        self.metrics = {
            "stt_ms": None,
            "llm_first_token_ms": None,
            "llm_ms": None,
            "tts_ms": None,
            "dropped_inputs": 0,
            "stale_inputs": 0,
        }
        self.audio = self.llm = self.stt = self.tts = self.osc = None
        self._factories = audio_factory, llm_factory, stt_factory, tts_factory, osc_factory
        self._player = player

    def _set_state(self, state, error=None):
        with self._status_lock:
            self.state = state
            if error is not None:
                self.error = error

    def _metric(self, name, value):
        with self._status_lock:
            self.metrics[name] = value

    def _increment(self, name):
        with self._status_lock:
            self.metrics[name] += 1

    def _capture_allowed(self):
        if self.stop_event.is_set():
            return False
        if self.cfg["tts"]["mute_capture"] and (self._playing.is_set() or time.monotonic() < self._muted_until):
            return False
        if self.cfg["chat"]["listen_mode"] == "push_to_talk":
            return self.push_to_talk.is_set() and time.monotonic() < self._ptt_deadline
        return True

    def set_listening(self, active):
        if active:
            self._ptt_deadline = time.monotonic() + 1.5
            self.push_to_talk.set()
        else:
            if self.audio:
                self.audio.flush()
            self.push_to_talk.clear()

    def _cancelled(self, epoch):
        return self.stop_event.is_set() or epoch != self._epoch

    def start(self):
        audio_factory, llm_factory, stt_factory, tts_factory, osc_factory = self._factories
        resources = []
        try:
            missing = check_credentials(self.cfg)
            if missing:
                raise ConfigError("缺少配置: " + ", ".join(missing))
            for name, factory in (("llm", llm_factory), ("stt", stt_factory), ("tts", tts_factory)):
                if self.stop_event.is_set():
                    return
                resource = factory(self.cfg)
                setattr(self, name, resource)
                resources.append(resource)
            if self.stop_event.is_set():
                return
            if osc_factory is None:
                from pythonosc.udp_client import SimpleUDPClient

                osc_factory = SimpleUDPClient
            self.osc = osc_factory(self.cfg["osc"]["host"], self.cfg["osc"]["vrc_port"])
            self.audio = audio_factory(self.cfg["audio"], self._on_speech, self._capture_allowed, self._audio_error)
            self.audio.start()
            if self.stop_event.is_set():
                return
            for name, target in (("conversation", self._input_worker), ("tts", self._tts_worker)):
                thread = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(thread)
                thread.start()
            with self._status_lock:
                if self.stop_event.is_set():
                    return
                self.state = "running"
            self.ready.set()
            log.info("桥接已启动，输入设备: %s", self.audio.device_name)
            self.stop_event.wait()
        except Exception as exc:
            error = str(exc) if isinstance(exc, ConfigError) else safe_error(exc)
            self._set_state("error", error)
            log.error("启动/运行失败: %s", error)
        finally:
            self.stop_event.set()
            self.ready.set()
            if self.audio:
                try:
                    self.audio.stop()
                except Exception as exc:
                    log.warning("关闭音频设备: %s", safe_error(exc))
            for thread in self._threads:
                thread.join()
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception:
                    pass
            if self.osc and hasattr(self.osc, "_sock"):
                self.osc._sock.close()
            if self.state != "error":
                self._set_state("stopped")
            self.done.set()
            log.info("桥接会话已结束")

    def _audio_error(self, exc):
        error = safe_error(exc)
        with self._status_lock:
            self.state, self.error = "error", error
            self.stop_event.set()
        log.error("音频捕获失败: %s", error)

    def stop(self):
        self.stop_event.set()
        self.interrupt()
        if self.state != "error" and not self.done.is_set():
            self._set_state("stopping")

    def interrupt(self):
        with self._output_lock:
            self._epoch += 1
            self._tts_revision += 1
            drain(self._inputs)
            drain(self._tts)
            self._typing(False)

    def reset(self):
        with self._output_lock:
            self.interrupt()
            self.conv.reset()
        log.info("对话历史已清空")

    def set_tts_enabled(self, enabled):
        if enabled and not api_key(self.cfg, "tts", "minimax"):
            raise ConfigError("请先配置 MiniMax API Key，再重启桥接")
        with self._output_lock:
            self._tts_enabled = enabled
            if not enabled:
                self._tts_revision += 1
                drain(self._tts)

    def _on_speech(self, pcm):
        if not self.stop_event.is_set():
            self._enqueue("audio", pcm)

    def _enqueue(self, kind, value):
        with self._output_lock:
            if self.stop_event.is_set():
                return
            if put_latest(self._inputs, (time.monotonic(), self._epoch, kind, value)):
                self._increment("dropped_inputs")
                log.warning("输入队列已满，丢弃最旧片段以保持实时性")

    def submit_text(self, text):
        if self.state != "running" or self.stop_event.is_set():
            raise ConfigError("请先启动桥接")
        text = text.strip()
        if not text or len(text) > 2000:
            raise ConfigError("请输入1～2000字的消息")
        if text.lower() in ("/reset", "/重置"):
            self.reset()
            return
        self._enqueue("text", text)

    def _input_worker(self):
        while not self.stop_event.is_set():
            try:
                created, epoch, kind, value = self._inputs.get(timeout=0.1)
            except queue.Empty:
                continue
            if time.monotonic() - created > self.cfg["chat"]["max_input_age"]:
                self._increment("stale_inputs")
                continue
            if self._cancelled(epoch):
                continue
            try:
                if kind == "audio":
                    started = time.monotonic()
                    value = self.stt.transcribe(value, self.cfg["audio"]["sample_rate"])
                    self._metric("stt_ms", round((time.monotonic() - started) * 1000))
                    if self._cancelled(epoch):
                        continue
                    log.info("识别: %s", value)
                if value.strip().lower() in ("/reset", "/重置"):
                    self.reset()
                    continue
                prefix = self.cfg["chat"]["trigger_prefix"] if kind == "audio" else ""
                if prefix and not value.startswith(prefix):
                    continue
                text = value[len(prefix) :].strip()
                if is_coherent(text) and not self._cancelled(epoch):
                    self._reply(text, epoch)
            except Exception as exc:
                if not self._cancelled(epoch):
                    error = safe_error(exc)
                    with self._status_lock:
                        self.error = error
                    log.warning("识别/对话失败: %s", error)
            finally:
                with self._output_lock:
                    if epoch == self._epoch:
                        self._typing(False)

    def _reply(self, text, epoch):
        with self._output_lock:
            if self._cancelled(epoch):
                return
            self._typing(True)
        started, first = time.monotonic(), True
        splitter = SentenceSplitter(
            self.cfg["streaming"]["split_chars"],
            lambda s: self._emit(s, epoch),
            self.cfg["osc"]["max_chars"] - len(self.cfg["chat"]["reply_prefix"]),
        )

        def on_token(token):
            nonlocal first
            if first:
                self._metric("llm_first_token_ms", round((time.monotonic() - started) * 1000))
                first = False
            if not self._cancelled(epoch):
                splitter.feed(token)

        reply = self.llm.chat(
            self.conv.build_messages(text),
            on_token=on_token if self.cfg["streaming"]["enabled"] else None,
            cancelled=lambda: self._cancelled(epoch),
        )
        self._metric("llm_ms", round((time.monotonic() - started) * 1000))
        if self._cancelled(epoch):
            return
        if not self.cfg["streaming"]["enabled"]:
            self._metric("llm_first_token_ms", self.metrics["llm_ms"])
            splitter.feed(reply)
        splitter.flush()
        with self._output_lock:
            if reply and not self._cancelled(epoch):
                self.conv.record(text, reply)

    def _emit(self, sentence, epoch):
        delay = self.cfg["osc"]["send_interval"]
        if self.cfg["streaming"]["enabled"]:
            delay = max(delay, self.cfg["streaming"]["sentence_delay"])
        while not self._cancelled(epoch):
            remaining = delay - (time.monotonic() - self._last_osc)
            if remaining <= 0:
                break
            self.stop_event.wait(min(0.05, remaining))
        with self._output_lock:
            if self._cancelled(epoch):
                return
            text = (self.cfg["chat"]["reply_prefix"] + sentence).replace("\n", "\v")
            self.osc.send_message("/chatbox/input", [text, True, False])
            self._last_osc = time.monotonic()
            log.info("回复: %s", sentence)
            if self._tts_enabled and put_latest(self._tts, (epoch, self._tts_revision, sentence)):
                log.warning("语音队列已满，跳过最旧语音")

    def _typing(self, active):
        if self.osc:
            try:
                self.osc.send_message("/chatbox/typing", active)
            except OSError:
                log.warning("无法发送 OSC 输入状态")

    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                epoch, revision, text = self._tts.get(timeout=0.1)
            except queue.Empty:
                continue

            def cancelled():
                return self._cancelled(epoch) or revision != self._tts_revision or not self._tts_enabled

            if cancelled():
                continue
            try:
                started = time.monotonic()
                pcm, rate = self.tts.synthesize(text)
                self._metric("tts_ms", round((time.monotonic() - started) * 1000))
                if cancelled():
                    continue
                self._playing.set()
                self._player(pcm, rate, self.cfg["tts"]["output_device"], cancelled)
            except Exception as exc:
                if not cancelled():
                    log.warning("语音合成/播放失败: %s", safe_error(exc))
            finally:
                self._muted_until = time.monotonic() + 0.25
                self._playing.clear()

    def status(self):
        with self._status_lock:
            return {
                "state": self.state,
                "running": self.state == "running",
                "error": self.error,
                "metrics": dict(self.metrics),
                "input_queue": self._inputs.qsize(),
                "tts_queue": self._tts.qsize(),
                "tts_playing": self._playing.is_set(),
                "tts_enabled": self._tts_enabled,
                "audio_device": self.audio.device_name if self.audio else "",
                "audio_rms": round(self.audio.current_rms, 1) if self.audio else 0,
                "listen_mode": self.cfg["chat"]["listen_mode"],
            }


class BridgeController:
    def __init__(self, store, bridge_factory=VRChatLLMBridge):
        self.store, self.bridge_factory = store, bridge_factory
        self._lock = threading.RLock()
        self.bridge = self.thread = None
        self.restart_required = False

    def start(self):
        with self._lock:
            if self.thread and self.thread.is_alive():
                raise ConfigError("桥接正在启动、运行或停止，请等待当前会话结束")
            cfg = self.store.load()
            log.setLevel(cfg["logging"]["level"])
            missing = check_credentials(cfg)
            if missing:
                raise ConfigError("请先配置: " + ", ".join(missing))
            self.bridge = self.bridge_factory(cfg)
            self.thread = threading.Thread(target=self.bridge.start, name="bridge-session", daemon=True)
            self.restart_required = False
            self.thread.start()
        return self.status()

    def stop(self, wait=0):
        with self._lock:
            bridge, thread = self.bridge, self.thread
            if bridge:
                bridge.stop()
        if thread and wait:
            thread.join(timeout=wait)
        return self.status()

    def require_bridge(self):
        with self._lock:
            if not self.bridge or self.bridge.state != "running":
                raise ConfigError("请先启动桥接")
            return self.bridge

    def update_config(self, updates):
        with self._lock:
            old = self.store.load()
            validate(merge(copy.deepcopy(old), updates, preserve_empty_keys=True))
            enabled = updates.get("tts", {}).get("enabled") if isinstance(updates, dict) else None
            live = self.thread and self.thread.is_alive()
            if live and enabled is True and not api_key(self.bridge.cfg, "tts", "minimax"):
                raise ConfigError("当前会话尚未配置 TTS 密钥，请先停止桥接后保存设置")
            cfg = self.store.update(updates)
            if live:
                if enabled is not None:
                    self.bridge.set_tts_enabled(enabled)
                before, after = copy.deepcopy(old), copy.deepcopy(cfg)
                before["tts"]["enabled"] = after["tts"]["enabled"]
                if before != after:
                    self.restart_required = True
            return cfg

    def status(self):
        with self._lock:
            status = (
                self.bridge.status()
                if self.bridge
                else {"state": "stopped", "running": False, "error": "", "metrics": {}}
            )
            status["restart_required"] = self.restart_required
            return status

    def diagnostics(self):
        import sys

        cfg = self.store.load()
        checks = [{"name": "配置格式", "ok": True, "detail": "配置校验通过"}]
        for section in ("stt", "llm"):
            provider = cfg[section]["provider"]
            ready = provider in ("funasr", "ollama") or bool(api_key(cfg, section, provider))
            checks.append(
                {"name": section.upper(), "ok": ready, "detail": provider + (" 已配置" if ready else " 缺少 API Key")}
            )
        module = "pyaudiowpatch" if sys.platform == "win32" else "pyaudio"
        checks.append({"name": "音频依赖", "ok": importlib.util.find_spec(module) is not None, "detail": module})
        if cfg["stt"]["provider"] == "funasr":
            checks.append(
                {
                    "name": "本地识别依赖",
                    "ok": importlib.util.find_spec("funasr") is not None,
                    "detail": "FunASR（首次使用会下载模型）",
                }
            )
        checks.append(
            {
                "name": "TTS",
                "ok": not cfg["tts"]["enabled"] or bool(api_key(cfg, "tts", "minimax")),
                "detail": "已启用" if cfg["tts"]["enabled"] else "未启用（可选）",
            }
        )
        checks.append(
            {
                "name": "OSC 目标",
                "ok": True,
                "detail": f"{cfg['osc']['host']}:{cfg['osc']['vrc_port']}，UDP 不提供送达确认",
            }
        )
        return checks
