#!/usr/bin/env python3
"""Optional tkinter window using the same controller and web settings."""

import logging
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk, scrolledtext

from server import BridgeHTTPServer, LogBuffer
from vrcai.config import ConfigError, ConfigStore
from vrcai.providers import safe_error
from vrcai.service import BridgeController


class LLMBridgeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VRChat AI · 对话桥接")
        self.root.geometry("850x650")
        self.root.minsize(650, 450)
        self.controller = BridgeController(ConfigStore())
        self.logs = LogBuffer()
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("vrcai").addHandler(self.logs)
        self.server = BridgeHTTPServer(("127.0.0.1", 0), self.controller, self.logs)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.cursor = 0
        self.closing = False
        self.ptt_job = None
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        self.start_button = ttk.Button(top, text="开启对话", command=lambda: self._action(self.controller.start))
        self.start_button.pack(side="left")
        ttk.Button(top, text="停止", command=lambda: self._action(self.controller.stop)).pack(side="left", padx=6)
        ttk.Button(
            top, text="立即打断", command=lambda: self._action(lambda: self.controller.require_bridge().interrupt())
        ).pack(side="left")
        ttk.Button(
            top, text="清空记忆", command=lambda: self._action(lambda: self.controller.require_bridge().reset())
        ).pack(side="left", padx=6)
        ttk.Button(top, text="设置 / 完整控制台", command=self.open_web).pack(side="right")
        self.status = ttk.Label(self.root, text="就绪", padding=12)
        self.status.pack(fill="x")
        self.tts_enabled = tk.BooleanVar(value=self.controller.store.load()["tts"]["enabled"])
        ttk.Checkbutton(self.root, text="语音回复", variable=self.tts_enabled, command=self.toggle_tts).pack(
            anchor="w", padx=12
        )
        self.ptt = ttk.Button(self.root, text="按住监听（需在设置中选择按键监听）")
        self.ptt.pack(anchor="w", padx=12, pady=8)
        self.ptt.bind("<ButtonPress-1>", self.begin_ptt)
        self.root.bind("<ButtonRelease-1>", self.end_ptt)
        self.root.bind("<FocusOut>", self.end_ptt)
        self.log_area = scrolledtext.ScrolledText(self.root, state="disabled", wrap="word", font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True, padx=12, pady=8)
        bottom = ttk.Frame(self.root, padding=12)
        bottom.pack(fill="x")
        self.message = ttk.Entry(bottom)
        self.message.pack(side="left", fill="x", expand=True)
        self.message.bind("<Return>", lambda event: self.send())
        ttk.Button(bottom, text="发送", command=self.send).pack(side="right", padx=8)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.poll)

    def _action(self, action):
        try:
            action()
        except Exception as exc:
            self.show_error(exc)

    def show_error(self, exc):
        message = str(exc) if isinstance(exc, ConfigError) else safe_error(exc)
        self.status.configure(text=message)
        logging.getLogger("vrcai").warning("界面操作失败: %s", message)

    def open_web(self):
        webbrowser.open(f"http://127.0.0.1:{self.server.server_address[1]}")

    def toggle_tts(self):
        self._action(lambda: self.controller.update_config({"tts": {"enabled": self.tts_enabled.get()}}))

    def send(self):
        text = self.message.get().strip()
        if text:
            try:
                self.controller.require_bridge().submit_text(text)
                self.message.delete(0, "end")
            except Exception as exc:
                self.show_error(exc)

    def begin_ptt(self, event=None):
        if self.ptt_job is None:
            self._action(lambda: self.controller.require_bridge().set_listening(True))
            self.ptt_job = self.root.after(400, self.ptt_heartbeat)

    def ptt_heartbeat(self):
        self.ptt_job = None
        self.begin_ptt()

    def end_ptt(self, event=None):
        if self.ptt_job is not None:
            self.root.after_cancel(self.ptt_job)
            self.ptt_job = None
            if self.controller.bridge:
                self.controller.bridge.set_listening(False)

    def poll(self):
        if self.closing:
            return
        state = self.controller.status()
        names = {
            "starting": "启动中",
            "running": "运行中",
            "stopping": "停止中，等待当前请求结束",
            "stopped": "已停止",
            "error": "出错",
        }
        text = names.get(state["state"], state["state"])
        if state.get("error"):
            text += " · " + state["error"]
        if state["restart_required"]:
            text += " · 设置已保存，请停止后重新开启"
        self.status.configure(text=text)
        self.start_button.configure(state="normal" if state["state"] in ("stopped", "error") else "disabled")
        if state.get("running"):
            self.tts_enabled.set(state["tts_enabled"])
        self.ptt.configure(
            state="normal" if state.get("running") and state.get("listen_mode") == "push_to_talk" else "disabled"
        )
        entries = self.logs.read(self.cursor)
        self.cursor = entries["cursor"]
        if entries["logs"]:
            self.log_area.configure(state="normal")
            for entry in entries["logs"]:
                self.log_area.insert("end", entry["text"] + "\n")
            if int(self.log_area.index("end-1c").split(".")[0]) > 1000:
                self.log_area.delete("1.0", "201.0")
            self.log_area.see("end")
            self.log_area.configure(state="disabled")
        self.root.after(200, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.end_ptt()
        self.controller.stop()
        self.status.configure(text="正在关闭，等待当前任务结束…")

        def finish():
            self.server.shutdown()
            self.server.server_close()
            if self.controller.thread:
                self.controller.thread.join()

        self.shutdown_thread = threading.Thread(target=finish, daemon=True)
        self.shutdown_thread.start()
        self.wait_close()

    def wait_close(self):
        if self.shutdown_thread.is_alive():
            self.root.after(100, self.wait_close)
        else:
            self.root.destroy()


def main():
    app = LLMBridgeGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
