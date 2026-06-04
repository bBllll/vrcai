#!/usr/bin/env python3
"""VRChat LLM Bridge - Web GUI 后端"""
import sys, os, json, queue, time, threading, logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from bridge import VRChatLLMBridge, load_config

# 全局状态
_state = {"bridge": None, "thread": None}
log_queue = queue.Queue()

# ---- API Handler ----

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # 静音 HTTP 日志

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_GET(self):
        b = _state["bridge"]
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json({
                "running": b is not None and _state["thread"] is not None and _state["thread"].is_alive(),
                "audio_device": getattr(b.audio, '_stream', None) is not None if b else False,
            })
        elif path == "/api/tts-audio":
            b = _state["bridge"]
            if b and b._latest_tts_wav:
                wav = b._latest_tts_wav
                b._latest_tts_wav = None  # 消费掉
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(wav)
            else:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
        elif path == "/api/logs":
            logs = []
            while not log_queue.empty():
                try:
                    logs.append(log_queue.get_nowait())
                except queue.Empty:
                    break
            self._send_json({"logs": logs[-100:]})  # 最多 100 条
        elif path == "/api/config":
            cfg = load_config()
            # 隐藏敏感字段
            safe = {
                "system_prompt": cfg.get("system_prompt", ""),
                "reply_prefix": cfg.get("chat", {}).get("reply_prefix", ""),
                "stt_provider": cfg.get("stt", {}).get("provider", ""),
                "llm_provider": cfg.get("llm", {}).get("provider", ""),
                "llm_model": cfg.get("llm", {}).get(cfg.get("llm", {}).get("provider", ""), {}).get("model", ""),
                "temperature": cfg.get("chat", {}).get("temperature", 0.9),
                "max_tokens": cfg.get("chat", {}).get("max_tokens", 300),
                "audio_device": cfg.get("audio", {}).get("device", None),
                "tts_enabled": cfg.get("tts", {}).get("enabled", False),
                "tts_key": cfg.get("tts", {}).get("minimax", {}).get("api_key", ""),
                "tts_voice": cfg.get("tts", {}).get("minimax", {}).get("voice_id", ""),
                "tts_speed": cfg.get("tts", {}).get("minimax", {}).get("speed", 1.0),
                "tts_vol": cfg.get("tts", {}).get("minimax", {}).get("vol", 1.0),
                "stt_groq_key": cfg.get("stt", {}).get("groq_whisper", {}).get("api_key", ""),
                "stt_openai_key": cfg.get("stt", {}).get("openai_whisper", {}).get("api_key", ""),
                "stt_qwen_key": cfg.get("stt", {}).get("dashscope_qwen_asr", {}).get("api_key", ""),
                "llm_oc_key": cfg.get("llm", {}).get("openai_compatible", {}).get("api_key", ""),
                "llm_o_key": cfg.get("llm", {}).get("openai", {}).get("api_key", ""),
            }
            self._send_json(safe)
        elif path == "/" or path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
            self.wfile.write(html.encode())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/start":
            try:
                _state["bridge"] = VRChatLLMBridge(load_config())
                _state["thread"] = threading.Thread(target=_run_bridge, daemon=True)
                _state["thread"].start()
                time.sleep(0.5)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
        elif path == "/api/stop":
            b = _state["bridge"]
            if b:
                b._stt_stop.set()
                b.audio.stop()
            _state["bridge"] = None
            _state["thread"] = None
            self._send_json({"ok": True})
        elif path == "/api/config":
            try:
                body_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(body_len))
                cfg = load_config()
                _merge_config(cfg, body)
                import yaml
                with open(Path(__file__).parent / "config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def _merge_config(cfg, updates):
    """递归合并配置。"""
    for k, v in updates.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            _merge_config(cfg[k], v)
        else:
            cfg[k] = v

def _run_bridge():
    b = _state["bridge"]
    try:
        b.start()
    except Exception:
        pass

# ---- 日志桥接 ----

class BridgeLogHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = BridgeLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    for noisy in ("httpx", "openai", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# ---- 入口 ----

def main():
    setup_logging()
    port = 8765
    server = HTTPServer(("127.0.0.1", port), APIHandler)
    print(f"🌐 VRChat LLM Bridge Web GUI: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    main()
