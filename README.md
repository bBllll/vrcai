# VRChat LLM Bridge

把 VRChat 中的语音转换成 AI 回复，通过 OSC 显示在聊天框，并可合成为语音播放。

支持 Groq Whisper、OpenAI Whisper、Qwen ASR、可选 FunASR；对话支持 DeepSeek 等 OpenAI 兼容服务、OpenAI 和 Ollama。主要面向 **Windows 10/11、Python 3.11～3.13（64 位）**。

```text
系统声音 / 指定输入 → 语音分段 → STT → 对话模型 → OSC 聊天框
                                             └→ MiniMax → 指定播放设备
```

## 0.2.0 改进

- 短噪声与长静音及时清理，保留语音开头；持续讲话按最长片段分割。
- 有界输入队列、过期丢弃、串行对话；停止/打断后丢弃旧请求结果。
- 流式与普通回复都支持 TTS；后端逐句播放，关闭网页后仍能继续工作。
- 支持手动输入/输出设备、自动系统声音回环、按住监听、立即打断和清空记忆。
- 设置完整回填、只保存改动、原子写入；语音开关立即生效，其余修改明确提示重启。
- 密钥不回传前端；本机接口有会话认证、来源/Host 校验，无开放 CORS。
- 配置检查、OSC 测试、阶段耗时、独立日志游标；自动化回归测试和跨平台 CI。

详细变更见 [CHANGELOG.md](CHANGELOG.md)。

![Web 控制台](docs/console.png)

## 安装与启动

Windows 推荐新建虚拟环境，避免与其他项目依赖混用。仓库路径可包含中文和空格。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py --open
```

也可以先双击 **安装依赖.bat**，安装成功后双击 **启动GUI.bat**。

默认管理页面：[http://127.0.0.1:8765](http://127.0.0.1:8765)。服务就绪后才会自动打开浏览器。首次运行不必手动复制配置，在“设置”中保存后会生成本地 `config.yaml`；也可先复制 [配置模板](config.yaml.example) 后编辑。

1. 在设置中选择 **语音识别服务**并填写对应密钥。
2. 选择 **对话服务**，填写密钥、模型和 Base URL。本地 Ollama 无需密钥，但须先安装、运行并下载所选模型。
3. 在 VRChat 的 Action Menu → Options → OSC 中启用 OSC。
4. 点击“刷新音频设备”，确认输入设备；留空则捕获 Windows 默认播放设备的声音。
5. 点击“检查配置”，再“开启对话”。页面会显示启动中、运行中或具体错误。
6. 首次排错可先发送一条文字，单独验证对话模型。文字测试会调用所选模型并消耗其额度。

其他入口：

```powershell
# 只检查配置和依赖，不联网调用模型
.\.venv\Scripts\python.exe bridge.py --check

# 列出音频设备（会访问声卡）
.\.venv\Scripts\python.exe bridge.py --list-devices

# 命令行连续监听
.\.venv\Scripts\python.exe bridge.py

# 备用 tkinter 窗口，使用同一套运行服务与网页设置
.\.venv\Scripts\python.exe gui.py

# 指定端口或独立配置文件
.\.venv\Scripts\python.exe server.py --port 8766 --config D:\path\config.yaml --open
```

命令行请使用 `chat.listen_mode: continuous`；按键监听在 Web 页面或 tkinter 窗口中控制。Web 空格监听只在页面获得焦点、且焦点不在输入框或按钮时生效；也可以直接按住“按住监听”按钮。切换窗口或松开按钮会停止监听，心跳丢失会在约1.5秒后自动停止。

## 语音输入与 VRChat 发声

**输入**：自动模式使用 PyAudioWPatch 的 WASAPI Loopback 捕获默认播放设备。手动指定设备优先于自动检测。可以选择声卡输入、Loopback 或虚拟音频设备；设备编号由当前系统枚举，插拔或更换设备后请刷新。

**输出**：MiniMax 返回的 PCM 在 Python 后端播放，可以指定输出设备。默认播放只会让本机听到声音，**不会自动成为 VRChat 麦克风输入**。让游戏内其他人听见，需要自行配置虚拟声卡，例如把 TTS 输出到虚拟线缆的播放端，再把 VRChat 麦克风设为对应的录音端。混入自己的麦克风可使用系统已有的音频混音工具。

默认在 AI 播放时暂停识别，并在播放结束后保留短暂缓冲，减少“听见自己再回复自己”的循环。需要同时听取他人发言时，可以分离输入/输出路由后关闭“AI 播放时暂停识别”。

捕获系统声音会同时包含游戏音效、音乐及其他应用声音。当前采用可调能量分段，**不具备说话人识别或可靠的音乐过滤**。多人/嘈杂场景建议使用按键监听、唤醒前缀或单独音频路由。

## 配置与密钥

设置面板回填所选服务的模型、地址、设备和数值。温度 `0` 是有效值。

- 密钥输入框留空：保留原密钥。
- 点“清除”再保存：删除该密钥；环境变量中的密钥仍会作为后备。
- 语音开关：在已配置密钥的运行会话中立即生效。
- 其他设置：保存到磁盘后，在下次开启对话时生效；运行中会显示重启提示。
- 原 `config.yaml` 自动补全新增默认项。未知字段或非法类型会报错，不会静默覆盖文件。
- 旧页面写入的 `tts.minimax.sample_rate: 32000` 继续兼容。

环境变量也可提供密钥：

| 服务 | 环境变量 |
| --- | --- |
| Groq STT | `GROQ_API_KEY` |
| OpenAI STT / LLM | `OPENAI_API_KEY` |
| Qwen ASR | `DASHSCOPE_API_KEY` |
| OpenAI 兼容 LLM | `LLM_API_KEY` 或 `DEEPSEEK_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |

配置文件中的非空密钥优先。密钥保存在本机配置中，未加密；不要把自己的配置文件上传、发给他人或提交到版本库。仓库已忽略实际配置和虚拟环境。管理界面只监听 `127.0.0.1`，每次启动产生新的会话令牌；服务重启后刷新页面即可。

语音会发送给你选择的识别服务，识别文字及有限上下文会发送给对话服务，启用 TTS 后回复文字会发送给 MiniMax。本项目不额外保存录音文件。运行日志和对话记忆只存在当前进程内存中；终端内容是否被保留取决于你的终端设置。

## 服务配置

| 模块 | 配置提示 |
| --- | --- |
| DeepSeek / 兼容接口 | 选择 `openai_compatible`；填服务提供的 Base URL 和模型名 |
| OpenAI | Base URL 留空使用 SDK 默认地址，也可填写兼容地址 |
| Ollama | 默认 `http://localhost:11434/v1`；模型名须与本机一致 |
| Groq Whisper | 默认 `whisper-large-v3`；可修改模型 |
| Qwen ASR | 使用音频 `chat/completions` 接口；默认北京旧域名，可按账号地域更改 Base URL |
| MiniMax | 支持 Base URL、模型、音色、语速和音量；PCM 单声道 32kHz |

Qwen 接入按[阿里云官方参考](https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference)使用 `input_audio` 和 WAV Data URL；不再把 Qwen3-ASR 发到 Whisper 的上传接口。地域、模型名和账户权限须匹配。Windows 回环使用 [PyAudioWPatch 官方支持方式](https://github.com/s0d3s/PyAudioWPatch)，不依赖标准 PyAudio 中不存在的 `as_loopback` 参数。

本地 FunASR 为可选安装，基础安装不会下载语音模型或 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local-asr.txt
```

随后选择 `funasr`，配置模型和设备。首次运行可能下载较大模型；GPU 用户应按 PyTorch 官方说明选择与显卡驱动匹配的安装版本。可选模型加载、下载和真实识别需要在目标机器验证。

## 实时性与停止行为

- 对话串行处理，默认最多保留3条待处理输入，超过时丢弃最旧片段；默认超过20秒的待处理输入不再回复。
- 上下文按完整问答保留，默认最多20条消息；不再每10轮突然清空。使用“清空记忆”或发送 `/reset`、`/重置` 主动清空。
- 断句按标点在文本中的顺序处理，一次到达多句时全部拆开；没有标点的长回复也会分段。
- 聊天框每条长度包含回复前缀，默认最多144字符；发送有间隔。OSC 使用 UDP，发送成功只表示本机发送完成，不能证明游戏已显示。
- “立即打断”清空待处理输入/语音并使旧结果失效，可继续接收新输入。
- “停止”立即禁止后续回复、取消待播放语音，等待正在执行的请求/音频工作结束后释放资源。停止中不能重复开启。
- 已发出的云端请求通常无法撤销计费；网络请求设置连接/读取超时。FunASR 的首次模型下载或本地推理无法保证立即中止。
- 耗时显示用于排查瓶颈：语音识别、模型首字、完整回复、语音合成。非流式模式的首字耗时等于完整回复耗时。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 安装失败 | 使用64位 Python 3.11～3.13；确认在项目目录安装；Windows 使用 PyAudioWPatch |
| 端口占用 | 关闭旧服务，或使用 `--port 8766` |
| 页面提示会话失效 | 服务已重启，刷新页面 |
| 没有录到声音 | 刷新设备；确认 VRChat 输出到被捕获的设备；检查系统音量 |
| 讲话被切断 / 背景音误触发 | 调整能量阈值、停顿时长、最短语音，或切换按键监听 |
| 改了模型没有变化 | 停止后重新开启，观察“需要重启”的提示 |
| 本机能听到，游戏内听不到 | 配置虚拟声卡路由，并在 VRChat 选对应麦克风 |
| 不显示聊天框 | 开启 VRChat OSC，检查目标端口、地址及游戏设置，用“测试聊天框”确认 |
| 认证失败 / 限流 | 检查所选服务密钥、账户额度、模型权限和地域 |
| 关闭网页仍有声音 | 播放由后端负责；关闭前先点“停止”，或在终端 Ctrl+C 结束服务 |

非 Windows 系统可手动选择输入设备，但自动系统声音回环不在支持范围。Linux 安装 PyAudio 可能需要先安装 PortAudio 开发包。

## 开发与验证

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m compileall -q vrcai bridge.py server.py gui.py
node --check app.js
```

测试使用模拟音频、模型和播放设备，以及真实本机 HTTP 服务器；不消耗 API 额度，不向游戏发送消息。CI 在 Ubuntu / Windows、Python 3.11 / 3.13 上运行。实际声卡、VRChat、云端 API 和可选 FunASR 仍需要用户按自己的设备与账户联调。

```text
vrcai/
  config.py        默认值、验证、密钥脱敏、原子保存
  audio.py         设备枚举、WASAPI、分段、重采样、播放
  providers.py     STT / LLM / MiniMax 服务适配
  conversation.py 问答历史、断句、基础文本过滤
  service.py       会话生命周期、队列、打断与状态
bridge.py          命令行入口
server.py          本机管理接口
index.html/app.js  网页控制台
gui.py             备用 tkinter 窗口
tests/             回归测试
```
