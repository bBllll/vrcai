#!/usr/bin/env python3
"""
VRChat LLM Bridge — GUI 前端
"""
import sys
import os
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

import yaml
import logging

# 确保能找到 bridge 模块
sys.path.insert(0, str(Path(__file__).parent))

# ---- GUI 日志处理器 ----

class GuiLogHandler(logging.Handler):
    """将日志发送到 GUI 队列。"""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setLevel(logging.DEBUG)

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put((msg, ""))

# ---- 配置日志 ----

def setup_gui_logging(log_queue: queue.Queue):
    """配置日志系统，同时输出到 GUI 和终端。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # GUI handler
    gui_handler = GuiLogHandler(log_queue)
    gui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(gui_handler)

    # 屏蔽第三方库日志
    for noisy in ("httpx", "openai", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

from bridge import VRChatLLMBridge, load_config

# ============================================================
# GUI 应用
# ============================================================

class LLMBridgeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VRChat LLM Bridge")
        self.root.geometry("700x600")
        self.root.minsize(500, 400)

        self.cfg = load_config()
        self.bridge: VRChatLLMBridge | None = None
        self.bridge_thread: threading.Thread | None = None
        self.running = False
        self.log_queue = queue.Queue()

        # 配置日志到 GUI
        setup_gui_logging(self.log_queue)

        self._build_ui()
        self._poll_log()

    # ---- UI 构建 ----

    def _build_ui(self):
        # 顶部控制栏
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        self.btn_toggle = ttk.Button(top, text="▶ 开启对话", command=self._toggle)
        self.btn_toggle.pack(side="left", padx=(0, 10))

        self.lbl_status = ttk.Label(top, text="⚫ 已停止", foreground="gray")
        self.lbl_status.pack(side="left", padx=10)

        ttk.Button(top, text="⚙ 设置", command=self._open_settings).pack(side="right")

        # 日志区域
        log_frame = ttk.LabelFrame(self.root, text="实时日志", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self.log_area = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled",
                                                   font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True)

        # 底部状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom")

    # ---- 日志 ----

    def _log(self, text: str, tag: str = ""):
        self.log_queue.put((text, tag))

    def _poll_log(self):
        while True:
            try:
                text, tag = self.log_queue.get_nowait()
                self.log_area.configure(state="normal")
                if tag:
                    self.log_area.insert("end", text + "\n", tag)
                else:
                    self.log_area.insert("end", text + "\n")
                self.log_area.see("end")
                self.log_area.configure(state="disabled")
            except queue.Empty:
                break
        self.root.after(100, self._poll_log)

    # ---- 开关 ----

    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        try:
            self.cfg = load_config()
            self.bridge = VRChatLLMBridge(self.cfg)
            self._log("🔄 正在启动...")
            self.btn_toggle.configure(state="disabled")
            self.status_bar.configure(text="正在启动...")

            self.bridge_thread = threading.Thread(target=self._run_bridge, daemon=True)
            self.bridge_thread.start()

            # 等 2 秒看有没有报错
            self.root.after(2000, self._check_startup)
        except Exception as e:
            self._log(f"❌ 启动失败: {e}")
            messagebox.showerror("启动失败", str(e))

    def _check_startup(self):
        if self.running:
            return
        if self.bridge and not self.bridge.audio._stream:
            self._log("❌ 音频设备启动失败")
            self.btn_toggle.configure(text="▶ 开启对话", state="normal")
            self.lbl_status.configure(text="❌ 音频错误", foreground="red")
            self.status_bar.configure(text="音频设备启动失败 - 检查设备")
            messagebox.showerror("音频错误", "无法打开音频设备。\n请检查 config.yaml 中的 device 设置。")
        else:
            self._on_started()

    def _on_started(self):
        self.running = True
        self.btn_toggle.configure(text="■ 停止对话", state="normal")
        self.lbl_status.configure(text="🟢 运行中", foreground="green")
        self.status_bar.configure(text="正在监听语音...")

    def _run_bridge(self):
        try:
            self.bridge.start()
        except Exception as e:
            self._log(f"❌ 运行时错误: {e}")
        finally:
            self.running = False
            self.root.after(0, self._on_stopped)

    def _stop(self):
        self._log("⏹ 正在停止...")
        if self.bridge:
            self.bridge._stt_stop.set()
            self.bridge.audio.stop()
        self.running = False
        self._on_stopped()

    def _on_stopped(self):
        self.btn_toggle.configure(text="▶ 开启对话", state="normal")
        self.lbl_status.configure(text="⚫ 已停止", foreground="gray")
        self.status_bar.configure(text="已停止")

    # ---- 设置 ----

    def _open_settings(self):
        SettingsWindow(self.root, self.cfg, self._on_settings_saved)

    def _on_settings_saved(self, cfg: dict):
        self.cfg = cfg
        with open(Path(__file__).parent / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        self._log("⚙ 设置已保存")


# ============================================================
# 设置窗口
# ============================================================

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, cfg: dict, on_save):
        super().__init__(parent)
        self.title("设置")
        self.geometry("550x650")
        self.cfg = cfg
        self.on_save = on_save
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # 人设
        frm_persona = ttk.Frame(nb, padding=8)
        nb.add(frm_persona, text="人设")
        self._build_persona(frm_persona)

        # 语音识别
        frm_stt = ttk.Frame(nb, padding=8)
        nb.add(frm_stt, text="语音识别")
        self._build_stt(frm_stt)

        # 对话模型
        frm_llm = ttk.Frame(nb, padding=8)
        nb.add(frm_llm, text="对话模型")
        self._build_llm(frm_llm)

        # 保存按钮
        ttk.Button(self, text="保存并关闭", command=self._save).pack(pady=(0, 8))

    def _build_persona(self, parent):
        ttk.Label(parent, text="系统提示词(人设):").pack(anchor="w")
        self.txt_prompt = tk.Text(parent, height=12, wrap="word", font=("微软雅黑", 10))
        self.txt_prompt.pack(fill="both", expand=True, pady=4)
        prompt = self.cfg.get("system_prompt", "")
        self.txt_prompt.insert("1.0", prompt.strip())

        ttk.Label(parent, text="回复前缀:").pack(anchor="w", pady=(8, 0))
        self.var_prefix = tk.StringVar(value=self.cfg.get("chat", {}).get("reply_prefix", ""))
        ttk.Entry(parent, textvariable=self.var_prefix, width=30).pack(anchor="w")

    def _build_stt(self, parent):
        stt_cfg = self.cfg.get("stt", {})

        ttk.Label(parent, text="语音识别模型:").pack(anchor="w")
        self.var_stt = tk.StringVar(value=stt_cfg.get("provider", "groq_whisper"))
        cb = ttk.Combobox(parent, textvariable=self.var_stt,
                          values=["groq_whisper", "openai_whisper", "dashscope_qwen_asr"],
                          state="readonly", width=25)
        cb.pack(anchor="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._stt_changed())

        # Groq
        self.frm_groq = ttk.LabelFrame(parent, text="Groq Whisper", padding=4)
        self.frm_groq.pack(fill="x", pady=4)
        ttk.Label(self.frm_groq, text="API Key:").pack(anchor="w")
        self.var_groq_key = tk.StringVar(value=stt_cfg.get("groq_whisper", {}).get("api_key", ""))
        ttk.Entry(self.frm_groq, textvariable=self.var_groq_key, width=40, show="*").pack(anchor="w")

        # OpenAI
        self.frm_openai_stt = ttk.LabelFrame(parent, text="OpenAI Whisper", padding=4)
        self.frm_openai_stt.pack(fill="x", pady=4)
        ttk.Label(self.frm_openai_stt, text="API Key:").pack(anchor="w")
        self.var_openai_key = tk.StringVar(value=stt_cfg.get("openai_whisper", {}).get("api_key", ""))
        ttk.Entry(self.frm_openai_stt, textvariable=self.var_openai_key, width=40, show="*").pack(anchor="w")

        # Qwen ASR
        self.frm_qwen = ttk.LabelFrame(parent, text="Qwen ASR (DashScope)", padding=4)
        self.frm_qwen.pack(fill="x", pady=4)
        ttk.Label(self.frm_qwen, text="API Key:").pack(anchor="w")
        self.var_qwen_key = tk.StringVar(value=stt_cfg.get("dashscope_qwen_asr", {}).get("api_key", ""))
        ttk.Entry(self.frm_qwen, textvariable=self.var_qwen_key, width=40, show="*").pack(anchor="w")
        ttk.Label(self.frm_qwen, text="Model:").pack(anchor="w")
        self.var_qwen_model = tk.StringVar(value=stt_cfg.get("dashscope_qwen_asr", {}).get("model", "qwen3-asr-flash"))
        ttk.Entry(self.frm_qwen, textvariable=self.var_qwen_model, width=30).pack(anchor="w")

        self._stt_changed()

    def _stt_changed(self):
        prov = self.var_stt.get()
        for f in (self.frm_groq, self.frm_openai_stt, self.frm_qwen):
            f.pack_forget()
        if prov == "groq_whisper":
            self.frm_groq.pack(fill="x", pady=4)
        elif prov == "openai_whisper":
            self.frm_openai_stt.pack(fill="x", pady=4)
        elif prov == "dashscope_qwen_asr":
            self.frm_qwen.pack(fill="x", pady=4)

    def _build_llm(self, parent):
        llm_cfg = self.cfg.get("llm", {})

        ttk.Label(parent, text="对话模型:").pack(anchor="w")
        self.var_llm = tk.StringVar(value=llm_cfg.get("provider", "openai_compatible"))
        cb = ttk.Combobox(parent, textvariable=self.var_llm,
                          values=["openai_compatible", "openai", "ollama"],
                          state="readonly", width=25)
        cb.pack(anchor="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._llm_changed())

        # OpenAI compatible (DeepSeek)
        self.frm_llm_oc = ttk.LabelFrame(parent, text="OpenAI 兼容 (DeepSeek 等)", padding=4)
        self.frm_llm_oc.pack(fill="x", pady=4)
        ttk.Label(self.frm_llm_oc, text="API Key:").pack(anchor="w")
        self.var_llm_oc_key = tk.StringVar(value=llm_cfg.get("openai_compatible", {}).get("api_key", ""))
        ttk.Entry(self.frm_llm_oc, textvariable=self.var_llm_oc_key, width=40, show="*").pack(anchor="w")
        ttk.Label(self.frm_llm_oc, text="Base URL:").pack(anchor="w")
        self.var_llm_oc_url = tk.StringVar(value=llm_cfg.get("openai_compatible", {}).get("base_url", "https://api.deepseek.com/v1"))
        ttk.Entry(self.frm_llm_oc, textvariable=self.var_llm_oc_url, width=40).pack(anchor="w")
        ttk.Label(self.frm_llm_oc, text="Model:").pack(anchor="w")
        self.var_llm_oc_model = tk.StringVar(value=llm_cfg.get("openai_compatible", {}).get("model", "deepseek-chat"))
        ttk.Entry(self.frm_llm_oc, textvariable=self.var_llm_oc_model, width=30).pack(anchor="w")

        # OpenAI
        self.frm_llm_o = ttk.LabelFrame(parent, text="OpenAI", padding=4)
        self.frm_llm_o.pack(fill="x", pady=4)
        ttk.Label(self.frm_llm_o, text="API Key:").pack(anchor="w")
        self.var_llm_o_key = tk.StringVar(value=llm_cfg.get("openai", {}).get("api_key", ""))
        ttk.Entry(self.frm_llm_o, textvariable=self.var_llm_o_key, width=40, show="*").pack(anchor="w")
        ttk.Label(self.frm_llm_o, text="Model:").pack(anchor="w")
        self.var_llm_o_model = tk.StringVar(value=llm_cfg.get("openai", {}).get("model", "gpt-4o-mini"))
        ttk.Entry(self.frm_llm_o, textvariable=self.var_llm_o_model, width=30).pack(anchor="w")

        # Ollama
        self.frm_llm_ol = ttk.LabelFrame(parent, text="Ollama (本地)", padding=4)
        self.frm_llm_ol.pack(fill="x", pady=4)
        ttk.Label(self.frm_llm_ol, text="Base URL:").pack(anchor="w")
        self.var_llm_ol_url = tk.StringVar(value=llm_cfg.get("ollama", {}).get("base_url", "http://localhost:11434/v1"))
        ttk.Entry(self.frm_llm_ol, textvariable=self.var_llm_ol_url, width=40).pack(anchor="w")
        ttk.Label(self.frm_llm_ol, text="Model:").pack(anchor="w")
        self.var_llm_ol_model = tk.StringVar(value=llm_cfg.get("ollama", {}).get("model", "qwen2.5:7b"))
        ttk.Entry(self.frm_llm_ol, textvariable=self.var_llm_ol_model, width=30).pack(anchor="w")

        # 温度
        ttk.Label(parent, text="温度 (0.0 - 2.0):").pack(anchor="w", pady=(10, 0))
        self.var_temp = tk.DoubleVar(value=self.cfg.get("chat", {}).get("temperature", 0.9))
        self.lbl_temp = ttk.Label(parent, text=f"{self.var_temp.get():.1f}")
        self.lbl_temp.pack()
        temp_scale = ttk.Scale(parent, from_=0.0, to=2.0, variable=self.var_temp,
                               orient="horizontal", command=self._temp_changed)
        temp_scale.pack(fill="x")

        # Max tokens
        ttk.Label(parent, text="最大 Token:").pack(anchor="w", pady=(8, 0))
        self.var_max_tokens = tk.IntVar(value=self.cfg.get("chat", {}).get("max_tokens", 300))
        ttk.Spinbox(parent, from_=50, to=2000, textvariable=self.var_max_tokens, width=8).pack(anchor="w")

        self._llm_changed()

    def _temp_changed(self, val):
        self.lbl_temp.configure(text=f"{float(val):.1f}")

    def _llm_changed(self):
        prov = self.var_llm.get()
        for f in (self.frm_llm_oc, self.frm_llm_o, self.frm_llm_ol):
            f.pack_forget()
        if prov == "openai_compatible":
            self.frm_llm_oc.pack(fill="x", pady=4)
        elif prov == "openai":
            self.frm_llm_o.pack(fill="x", pady=4)
        elif prov == "ollama":
            self.frm_llm_ol.pack(fill="x", pady=4)

    def _save(self):
        # 人设
        self.cfg["system_prompt"] = self.txt_prompt.get("1.0", "end").strip()
        if "chat" not in self.cfg:
            self.cfg["chat"] = {}
        self.cfg["chat"]["reply_prefix"] = self.var_prefix.get()

        # STT
        if "stt" not in self.cfg:
            self.cfg["stt"] = {}
        self.cfg["stt"]["provider"] = self.var_stt.get()
        self.cfg["stt"]["groq_whisper"] = {"api_key": self.var_groq_key.get()}
        self.cfg["stt"]["openai_whisper"] = {"api_key": self.var_openai_key.get(), "base_url": ""}
        self.cfg["stt"]["dashscope_qwen_asr"] = {
            "api_key": self.var_qwen_key.get(),
            "model": self.var_qwen_model.get(),
        }

        # LLM
        if "llm" not in self.cfg:
            self.cfg["llm"] = {}
        self.cfg["llm"]["provider"] = self.var_llm.get()
        self.cfg["llm"]["openai_compatible"] = {
            "api_key": self.var_llm_oc_key.get(),
            "base_url": self.var_llm_oc_url.get(),
            "model": self.var_llm_oc_model.get(),
        }
        self.cfg["llm"]["openai"] = {
            "api_key": self.var_llm_o_key.get(),
            "model": self.var_llm_o_model.get(),
            "base_url": "",
        }
        self.cfg["llm"]["ollama"] = {
            "base_url": self.var_llm_ol_url.get(),
            "model": self.var_llm_ol_model.get(),
        }
        self.cfg["chat"]["temperature"] = self.var_temp.get()
        self.cfg["chat"]["max_tokens"] = self.var_max_tokens.get()

        self.on_save(self.cfg)
        self.destroy()


# ============================================================
# 入口
# ============================================================

def main():
    gui = LLMBridgeGUI()
    gui.root.mainloop()

if __name__ == "__main__":
    main()
