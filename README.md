# VRChat LLM Bridge

语音识别 → AI 对话 → VRChat 聊天框 + 语音合成播放。

## 功能

- 🎤 **语音识别**：捕获系统音频，支持 Groq Whisper / OpenAI Whisper / Qwen ASR
- 🤖 **AI 对话**：DeepSeek / OpenAI / Ollama 本地模型
- 💬 **聊天框输出**：通过 OSC 自动发送到 VRChat 聊天气泡
- 🔊 **语音合成**：MiniMax TTS 将 AI 回复转语音播放
- 🌐 **Web GUI**：白色简约风前端，开关控制 + 实时日志 + 可视化设置

## 快速开始

```bash
pip install -r requirements.txt
cp config.yaml.example config.yaml
# 编辑 config.yaml，填入 API Key
python server.py
# 浏览器打开 http://127.0.0.1:8765
```

或命令行模式：`python bridge.py`

## 配置说明

所有设置通过 Web GUI 的「设置」面板修改，或直接编辑 `config.yaml`。

| 模块 | 提供商 | 需要 |
|------|--------|------|
| 语音识别 | Groq Whisper（免费） | [Groq API Key](https://console.groq.com/keys) |
| 对话模型 | DeepSeek | [DeepSeek API Key](https://platform.deepseek.com) |
| 语音合成 | MiniMax | [MiniMax API Key](https://platform.minimaxi.com) |
| OSC | VRChat 内置 | Action Menu → Options → OSC → Enabled |

## 文件结构

```
vrchat_llm_bridge/
├── bridge.py           # 核心桥接逻辑
├── server.py           # Web GUI 后端
├── gui.py              # tkinter GUI（备用）
├── index.html          # Web GUI 前端
├── config.yaml.example # 配置模板
├── requirements.txt    # Python 依赖
└── 启动GUI.bat         # Windows 一键启动
```
