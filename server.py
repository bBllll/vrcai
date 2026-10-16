#!/usr/bin/env python3
"""Loopback-only web control with same-origin session authentication."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import secrets
import socket
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from vrcai.config import ConfigError, ConfigStore, ROOT, redact
from vrcai.audio import list_devices
from vrcai.providers import api_key, safe_error
from vrcai.service import BridgeController


class LogBuffer(logging.Handler):
    def __init__(self, capacity=1000):
        super().__init__()
        self.entries = deque(maxlen=capacity)
        self.cursor = 0
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record):
        with self.lock:
            self.cursor += 1
            self.entries.append({"id": self.cursor, "text": self.format(record)[:2000]})

    def read(self, after):
        with self.lock:
            return {"logs": [e for e in self.entries if e["id"] > after][-200:], "cursor": self.cursor}


class BridgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, controller, logs=None):
        if address[0] != "127.0.0.1":
            raise ValueError("管理界面只能监听 127.0.0.1")
        super().__init__(address, APIHandler)
        self.controller = controller
        self.logs = logs or LogBuffer()
        self.token = secrets.token_urlsafe(32)
        port = self.server_address[1]
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {"http://" + h for h in self.hosts}


class APIHandler(BaseHTTPRequestHandler):
    server_version = "VRChatBridge"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def _headers(self, status, content_type, size):
        self.send_response(status)
        for key, value in {
            "Content-Type": content_type,
            "Content-Length": str(size),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        }.items():
            self.send_header(key, value)
        self.end_headers()

    def _send(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(raw))
        self.wfile.write(raw)

    def _guard(self, auth=True):
        if self.headers.get("Host", "") not in self.server.hosts:
            self._send({"error": "Host 不允许"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.origins:
            self._send({"error": "仅允许本机同源访问"}, 403)
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._send({"error": "不允许跨站访问"}, 403)
            return False
        token = self.headers.get("X-Session-Token", "")
        if auth and not hmac.compare_digest(token.encode("utf-8"), self.server.token.encode("utf-8")):
            self._send({"error": "会话已失效，请刷新页面"}, 403)
            return False
        return True

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ConfigError("不支持分块请求体")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ConfigError("无效的请求长度") from exc
        if not 0 < size <= 65536:
            raise ConfigError("请求体必须为1～65536字节")
        if self.headers.get_content_type() != "application/json":
            raise ConfigError("请使用 application/json")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ConfigError("请求体不完整")
        try:
            data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ConfigError("请求必须是有效的 JSON") from exc
        if not isinstance(data, dict):
            raise ConfigError("请求必须是 JSON 对象")
        return data

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self._guard(auth=path.startswith("/api/")):
            return
        try:
            controller = self.server.controller
            if path in ("/", "/app.js"):
                name = "index.html" if path == "/" else "app.js"
                content = (ROOT / name).read_text(encoding="utf-8-sig")
                if path == "/":
                    content = content.replace("__SESSION_TOKEN__", self.server.token)
                raw = content.encode("utf-8")
                self._headers(
                    200, "text/html; charset=utf-8" if path == "/" else "text/javascript; charset=utf-8", len(raw)
                )
                self.wfile.write(raw)
            elif path == "/api/status":
                self._send(controller.status())
            elif path == "/api/config":
                cfg = controller.store.load()
                safe = redact(cfg)
                for section in ("stt", "llm", "tts"):
                    for name, provider in safe[section].items():
                        if isinstance(provider, dict) and "api_key" in provider:
                            provider["key_configured"] = bool(api_key(cfg, section, name))
                self._send(safe)
            elif path == "/api/logs":
                after = max(0, int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0]))
                self._send(self.server.logs.read(after))
            elif path == "/api/devices":
                self._send({"devices": list_devices()})
            elif path == "/api/diagnostics":
                self._send({"checks": controller.diagnostics()})
            else:
                self._send({"error": "接口不存在"}, 404)
        except (ConfigError, ValueError) as exc:
            self._send({"error": str(exc) if isinstance(exc, ConfigError) else "请求参数无效"}, 400)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as exc:
            self._send({"error": safe_error(exc)}, 500)

    def do_POST(self):
        if not self._guard():
            return
        try:
            body = self._body()
            path = urlsplit(self.path).path
            controller = self.server.controller
            if path == "/api/start":
                result = controller.start()
            elif path == "/api/stop":
                result = controller.stop()
            elif path == "/api/config":
                cfg = controller.update_config(body)
                result = {"config": redact(cfg), "restart_required": controller.restart_required}
            elif path == "/api/message":
                if not isinstance(body.get("text"), str):
                    raise ConfigError("text 必须是字符串")
                controller.require_bridge().submit_text(body["text"])
                result = {}
            elif path in ("/api/interrupt", "/api/reset", "/api/listen"):
                bridge = controller.require_bridge()
                if path == "/api/interrupt":
                    bridge.interrupt()
                elif path == "/api/reset":
                    bridge.reset()
                else:
                    if type(body.get("active")) is not bool:
                        raise ConfigError("active 必须是布尔值")
                    bridge.set_listening(body["active"])
                result = {}
            elif path == "/api/test-osc":
                cfg = controller.store.load()
                from pythonosc.udp_client import SimpleUDPClient

                client = SimpleUDPClient(cfg["osc"]["host"], cfg["osc"]["vrc_port"])
                try:
                    client.send_message("/chatbox/input", ["VRChat AI: OSC test", True, False])
                finally:
                    client._sock.close()
                result = {"message": "测试消息已发送，请在 VRChat 内确认；UDP 不提供送达确认"}
            else:
                self._send({"error": "接口不存在"}, 404)
                return
            self._send({"ok": True, **result})
        except ConfigError as exc:
            self._send({"ok": False, "error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as exc:
            self._send({"ok": False, "error": safe_error(exc)}, 500)

    def do_OPTIONS(self):
        self._send({"error": "不允许跨来源请求"}, 403)


def main():
    parser = argparse.ArgumentParser(description="VRChat LLM Bridge Web GUI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config")
    parser.add_argument("--open", action="store_true", help="服务就绪后打开浏览器")
    args = parser.parse_args()
    logs = LogBuffer()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("vrcai").addHandler(logs)
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    controller = BridgeController(ConfigStore(args.config))
    try:
        server = BridgeHTTPServer(("127.0.0.1", args.port), controller, logs)
    except OSError:
        print("无法启动管理界面：端口可能已占用，请关闭旧窗口或使用 --port 8766")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"VRChat LLM Bridge: {url}")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.stop(wait=3)
        if controller.thread and controller.thread.is_alive():
            print("正在等待当前请求结束…")
            controller.thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
