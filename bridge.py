#!/usr/bin/env python3
"""Command line entry point; server.py and gui.py share the same services."""

import argparse
import logging
import time

from vrcai.audio import AudioCapture, list_devices
from vrcai.config import ConfigStore, load_config
from vrcai.conversation import ConversationManager, SentenceSplitter
from vrcai.providers import LLMClient, STTClient, TTSClient
from vrcai.service import BridgeController, VRChatLLMBridge

__all__ = [
    "AudioCapture",
    "load_config",
    "ConversationManager",
    "SentenceSplitter",
    "LLMClient",
    "STTClient",
    "TTSClient",
    "VRChatLLMBridge",
]


def main():
    parser = argparse.ArgumentParser(description="VRChat LLM Bridge")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check", action="store_true", help="检查配置和依赖，不调用云端 API")
    parser.add_argument("--config", help="配置文件路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        if args.list_devices:
            for d in list_devices():
                print(f"[{d['id']}] {d['name']} | 输入 {d['inputs']} / 输出 {d['outputs']} | {d['api']}")
            return 0
        controller = BridgeController(ConfigStore(args.config))
        if args.check:
            checks = controller.diagnostics()
            for c in checks:
                print(f"{'OK' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
            return 0 if all(c["ok"] for c in checks) else 1
        controller.start()
        print("Ctrl+C 停止。按键监听模式请使用 Web 界面。")
        try:
            while controller.thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            controller.stop(wait=3)
            if controller.thread.is_alive():
                print("正在等待当前请求结束…")
                controller.thread.join()
        return 1 if controller.status()["state"] == "error" else 0
    except Exception as exc:
        from vrcai.config import ConfigError
        from vrcai.providers import safe_error

        print(str(exc) if isinstance(exc, ConfigError) else safe_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
