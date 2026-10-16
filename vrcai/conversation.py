"""Conversation turns and deterministic, length-bounded sentence splitting."""

import re
import threading
from collections import deque


class ConversationManager:
    def __init__(self, cfg):
        self.system_prompt = cfg["system_prompt"].strip()
        # Legacy max_history counts messages; retain complete user/assistant pairs.
        self.turns = deque(maxlen=cfg["chat"]["max_history"] // 2)
        self._lock = threading.Lock()

    def build_messages(self, user_text):
        with self._lock:
            messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
            for user, assistant in self.turns:
                messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
            return messages + [{"role": "user", "content": user_text}]

    def record(self, user, assistant):
        with self._lock:
            self.turns.append((user, assistant))

    def reset(self):
        with self._lock:
            self.turns.clear()


class SentenceSplitter:
    def __init__(self, split_chars, callback, max_chars=120):
        self.chars, self.callback, self.max_chars = set(split_chars), callback, max_chars
        self.buffer = ""

    def feed(self, token):
        self.buffer += token
        while self.buffer:
            end = next((i + 1 for i, ch in enumerate(self.buffer) if ch in self.chars), None)
            if end is None and len(self.buffer) < self.max_chars:
                break
            end = min(end or self.max_chars, self.max_chars)
            sentence, self.buffer = self.buffer[:end], self.buffer[end:]
            if sentence.strip():
                self.callback(sentence.strip())

    def flush(self):
        if self.buffer.strip():
            self.callback(self.buffer.strip())
        self.buffer = ""


def is_coherent(text):
    text = text.strip()
    if not text or not any(ch.isalnum() for ch in text):
        return False
    clean = re.sub(r"\W", "", text, flags=re.UNICODE)
    # Reject obvious long ASR loops; short greetings and all languages remain valid.
    return not (len(clean) >= 12 and re.fullmatch(r"(.{1,4})\1{3,}", clean))
