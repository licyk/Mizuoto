<div align="center">

# Mizuoto

<p align="center">
  <a href="https://github.com/licyk/Mizuoto/stargazers">
    <img src="https://img.shields.io/github/stars/licyk/Mizuoto?style=flat&logo=github&logoColor=silver&color=bluegreen&labelColor=grey" alt="Stars">
  </a>
  <a href="https://github.com/licyk/Mizuoto/issues">
    <img src="https://img.shields.io/github/issues/licyk/Mizuoto?style=flat&logo=github&logoColor=silver&color=bluegreen&labelColor=grey" alt="Issues">
  </a>
  <a href="https://github.com/licyk/Mizuoto/commits/main">
    <img src="https://flat.badgen.net/github/last-commit/licyk/Mizuoto/main?icon=github&color=green&label=last%20main%20commit" alt="Last main commit">
  </a>
  <a href="https://github.com/licyk/Mizuoto/actions/workflows/ci.yml">
    <img src="https://github.com/licyk/Mizuoto/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/licyk/Mizuoto/actions/workflows/release.yml">
    <img src="https://github.com/licyk/Mizuoto/actions/workflows/release.yml/badge.svg" alt="Release">
  </a>
  <a href="https://pypi.org/project/mizuoto/">
    <img src="https://img.shields.io/pypi/v/mizuoto?style=flat&logo=pypi&logoColor=silver&color=bluegreen&labelColor=grey" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/mizuoto/">
    <img src="https://img.shields.io/pypi/pyversions/mizuoto?style=flat&logo=python&logoColor=silver&color=bluegreen&labelColor=grey" alt="Python versions">
  </a>
  <a href="https://github.com/licyk/Mizuoto/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/licence-GPL--3.0--only-bluegreen?style=flat&labelColor=grey" alt="许可证：GPL-3.0-only">
  </a>
</p>

[English](README.md) · **简体中文**

</div>

一个平台汇集多款开源语音合成引擎——CosyVoice、F5-TTS、GPT-SoVITS、OmniVoice、Qwen3-TTS、VoxCPM、Fish Speech 和
IndexTTS——共用一个网页界面、一个命令行和一套 HTTP 接口：合成语音、克隆与设计音色、转换与编辑语音、由一组角色朗读整本书，
并用你自己的录音微调模型。

所有引擎都在 Mizuoto 内实现，运行在同一个运行时（PyTorch）与同一个 Python 环境中，输出与其上游一致。无论由哪个引擎处理，
请求、音色、参数、任务与结果的形式都相同。显存由所有引擎统一管理：放不下的模型部分加载，其余权重从内存流式加载。

- **官网：** <https://mizuoto.netlify.app>
- **文档：** <https://mizuoto.netlify.app/docs/>（中文与英文；源文件在 `site/`，`python scripts/dev.py site-dev` 可在本地预览）

## 功能

- **合成**：使用任一已安装模型，可选音色库中的音色、用于克隆的参考音频、预置说话人或描述出的音色；
  模型支持时可加风格指令与情感；边生成边播放（流式）；参数预设。
- **音色**：带文本的参考音频库（缺少的文本由语音识别补全）、设计出的音色、预置说话人，
  以及带有专属微调权重的音色；可导出与导入。
- **项目**：由一组角色逐句朗读整本书、剧本或对话，每个角色可用各自的模型（一段对话中可混用多个引擎）；
  修改过的句子会重新生成，保留历次录音，整个项目或单章可导出为一个文件并附带字幕。
- **工具**：音色转换（CosyVoice）与语音编辑（F5-TTS）。
- **训练**：从录音、文件、合成结果或常见格式的文件夹建立数据集（按静音切分、自动转写、检查），
  再建立实验，逐阶段运行各引擎自己的微调流程，实时查看损失曲线并自动合成试听；
  结果是音色的专属权重或一个新模型。
- **远程主机**：在另一台机器上运行引擎（在那里运行 `mizuoto agent`），音色、结果与界面仍在本机。
- **多种用法**：网页界面（Material Design 3，中英文）、`mizuoto` 命令行、REST 接口（含事件与流式的 WebSocket）、
  兼容 OpenAI 的 `/v1/audio/speech`，或在 Python 中调用（`mizuoto.core.context.open_services()`，或用
  `mizuoto.embed.MizuotoServer` 嵌入其他应用）。

## 引擎

| 引擎 | 模型 | 特点 | 训练 | 许可证 |
| --- | --- | --- | --- | --- |
| Qwen3-TTS | 1.7B / 0.6B Base、CustomVoice、VoiceDesign | 10 种语言，带指令的预置说话人，按描述生成音色 | 说话人微调（SFT） | Apache-2.0 |
| CosyVoice | 3（0.5B）、2（0.5B）、300M base / SFT / instruct | 输入输出均可流式，音色转换，指令控制 | LLM、LLM + flow | Apache-2.0 |
| F5-TTS | v1 Base、Base、E2-TTS | 流匹配，精确时长，语音编辑 | 全量微调 | 代码 MIT；权重 CC-BY-NC-4.0 |
| GPT-SoVITS | v2、v2Pro、v2ProPlus、v4、v5 Turbo | 少样本克隆，音色专属的 GPT 与 SoVITS 权重 | 音色权重 | MIT |
| OmniVoice | OmniVoice | 600 多种语言，按属性生成音色，行内标签 | 全量、LoRA | Apache-2.0 |
| VoxCPM | VoxCPM2、1.5、0.5B | 无离散分词；VoxCPM2：30 种语言、48 kHz，有无参考文本均可克隆，按描述生成音色，风格指令 | 全量、LoRA | Apache-2.0 |
| Fish Speech | S2 Pro | 富有表现力的克隆 | LoRA | Fish Audio Research License（需接受许可） |
| IndexTTS | 1.5、2、2.5 | 情感可由权重、参考音频或描述控制 | — | bilibili 模型使用许可（需接受许可） |

内置语音识别（Whisper large-v3-turbo）用于生成参考文本。需接受许可的引擎是单独的软件包
（`mizuoto-fishspeech`、`mizuoto-indextts`），会先展示许可证全文，接受后才能下载其模型；其合成结果附带
许可证要求的声明。F5-TTS 的预训练权重仅限非商业用途。

## 项目状态

Mizuoto **0.2.0** 是以此名称发布的第一个版本。

- 在 CPU 上（fp32、相同种子），每个引擎与每种训练方式的输出都与其上游一致（CosyVoice 在使用相同音色提示时一致）。
  对照对象是之前的引擎 worker 实现（标签 `v1-engine-workers`），VoxCPM 则直接对照上游自己的软件包；
  比对经过整个系统、逐采样进行。基于真实模型的测试套件在 CPU 上对所有引擎都通过。
- 在 NVIDIA 显卡上，所有自回归解码都以 CUDA 图运行，每步只等待设备一次，结果与即时执行（eager）相同；这些提速是在与其他任务
  共用的显卡上测得的。更全面的 GPU 测量（真实模型端到端、注意力后端、显存流式加载、测试套件）尚待完成。
- ROCm、Intel XPU 与 Apple MPS 在代码中已支持，并用模拟设备测试过，但尚未在这些硬件上验证。
- Docker 镜像已编写，但尚未完整构建并运行过。

## 安装

需要 Python 3.10–3.13 与 torch 2.8 及以上版本，并选用与硬件匹配的构建（CUDA、ROCm、XPU、Apple 芯片或 CPU）：
先从与本机匹配的索引安装 torch，再安装 Mizuoto。`mizuoto doctor` 会检查 torch 构建与硬件是否匹配，不匹配时给出
确切的安装命令。推荐 NVIDIA 显卡；所有引擎也能在 CPU 上运行，但较慢。

**从 PyPI：**

```bash
# 先装 torch，选与硬件匹配的构建（mizuoto doctor 会给出）：
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128    # NVIDIA，Linux 或 Windows（cu126、cu128、cu129、cu130、cu132：取驱动支持的版本）
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/rocm7.2  # AMD，Linux（较旧的 ROCm 用 rocm6.4、rocm7.1）
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/xpu      # Intel 显卡，Linux 或 Windows
pip install "torch>=2.8"                                                       # Apple 芯片的 macOS（MPS）
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu      # 仅 CPU
# 再一次装好 Mizuoto（需接受许可的引擎另加 mizuoto-fishspeech mizuoto-indextts）：
pip install mizuoto
mizuoto doctor
mizuoto webui                                   # http://127.0.0.1:7869
```

**从源码：**

```bash
git clone https://github.com/licyk/Mizuoto.git && cd Mizuoto
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .   # 需接受许可的引擎：-e packages/native/fishspeech -e packages/native/indextts
python scripts/dev.py web-install && python scripts/dev.py web    # 首次需构建界面（需要 bun）
.venv/bin/mizuoto webui
```

**使用 Docker**（模型存放在 `/data` 卷中）：

```bash
docker build -t mizuoto .        # torch 的 CUDA 12.8 构建；CPU 版加 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu，
                                 # 加 --build-arg GATED=1 则包含需接受许可的引擎
docker run --gpus all -p 7869:7869 -v mizuoto-data:/data -e MIZUOTO_SERVER__ACCESS_TOKEN=change-me mizuoto
```

一个镜像包含服务器、界面与全部引擎，以及 GPT-SoVITS 的日语与韩语前端。

## 快速开始

网页界面打开后进入“合成”页；dummy 引擎无需下载即可使用（它用音调“说话”）。“设置 › 计算”会检查 torch 与硬件是否匹配；
所选模型尚未安装时，“合成”按钮会变为下载该模型的按钮。命令行中的等价操作：

```bash
mizuoto model download qwen3tts/1.7b-custom-voice
mizuoto speak "很高兴认识你。" -m qwen3tts/1.7b-custom-voice --preset vivian --instruct "开心地说" -o vivian.wav
```

命令行速览：

```bash
# 无需模型：dummy 引擎用音调“说话”。
mizuoto speak "Hello there." -m dummy/default --preset alto -o hello.wav

# 语音识别会补全参考音频的文本。
mizuoto model download asr/whisper-large-v3-turbo

# 音色：设计一个，或从录音添加一个，然后克隆。
mizuoto model download qwen3tts/1.7b-voice-design qwen3tts/1.7b-base
mizuoto voice design Narrator -m qwen3tts/1.7b-voice-design -d "温暖低沉的男声，语速平缓"
mizuoto voice add Me --ref me.wav --consent "我自己的声音"
mizuoto speak "任意文本。" -m qwen3tts/1.7b-base --voice narrator -o narrator.wav

# 流式、音色转换与语音编辑。
mizuoto model download cosyvoice/cosyvoice2-0.5b
mizuoto speak "边生成边播放。" -m cosyvoice/cosyvoice2-0.5b --voice narrator --stream | ffplay -nodisp -
mizuoto convert speech.wav -m cosyvoice/cosyvoice2-0.5b --voice narrator -o converted.wav
mizuoto model download f5tts/v1-base
mizuoto edit speech.wav -m f5tts/v1-base -t "编辑后的完整文本。" --span 1.2:1.8 -o edited.wav

# 需接受许可的引擎（先安装 mizuoto-indextts）：先阅读许可证并接受，再下载。
mizuoto engine licence indextts && mizuoto engine accept-licence indextts
mizuoto model download indextts/v2
mizuoto speak "今天真开心！" -m indextts/v2 --ref me.wav --consent "我自己的声音" --emotion happy=0.8 -o happy.wav

# 长篇项目：带章节与对话的脚本、角色表，逐句生成并导出带字幕的音频。
mizuoto project new "夜行列车" -m qwen3tts/1.7b-custom-voice --preset ryan --script book.txt
mizuoto project cast 夜行列车 anna --preset serena
mizuoto project render 夜行列车
mizuoto project export 夜行列车 -o book.mp3 --srt book.srt

# 参数预设，以及任意 OpenAI 客户端访问 http://127.0.0.1:7869/v1
mizuoto preset save Calm -m cosyvoice/cosyvoice2-0.5b -p speed=0.9 --default
curl -s http://127.0.0.1:7869/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "cosyvoice/cosyvoice2-0.5b", "input": "你好！", "voice": "narrator"}' -o hello.mp3
```

`mizuoto --help` 列出全部命令；命令加 `--json` 输出与接口返回相同的内容。

## 训练

先建立一个音色的数据集，再建立实验逐阶段微调模型（网页界面的“训练”页功能相同，并实时显示损失曲线）：

```bash
mizuoto dataset create Narrator --lang zh --consent "我自己的声音"
mizuoto dataset add narrator recordings/          # 与音频同名的 .lab 或 .txt 即其文本
mizuoto dataset prepare narrator                  # 按静音切分、转写、统一音量
mizuoto dataset check narrator
mizuoto asset download gptsovits/s2d-v2pro        # v2Pro 的预训练判别器
mizuoto train methods gptsovits/v2pro-base        # 可用的训练方式及其设置
mizuoto train new narrator --base gptsovits/v2pro-base --dataset narrator --hp sovits_epochs=8
mizuoto train run narrator                        # 导出该音色专属的 GPT 与 SoVITS 权重
mizuoto train metrics narrator
```

修改设置或数据集后，读取它们的阶段会标记为“已过期”；`train run` 从第一个未完成的阶段继续。
微调实际上需要显卡；所有训练也可在 CPU 上做小规模试验。

## 配置与性能

设置保存在数据目录的 `settings.toml` 中（`mizuoto config path`；`mizuoto config show`），每项都可用环境变量
覆盖：`MIZUOTO_SERVER__PORT=8000`。常用设置：

| 设置 | 作用 |
| --- | --- |
| `engines.<id>.device`、`engines.<id>.dtype` | 引擎运行的设备（`auto`、`cuda:1`、`cpu`）与精度 |
| `downloads.source` | `huggingface`、`hf-mirror`、`modelscope` 或自定义地址 |
| `downloads.repository` | `mizuoto-model`（默认）：优先从 Mizuoto 的镜像下载，缺少的文件再从各模型自己的仓库下载；`official`：只用后者 |
| `compute.max_resident_models`、`compute.unload_after_minutes` | 同时保持加载的模型数，以及空闲模型多久后卸载（`mizuoto runtime` 列出已加载的模型） |
| `compute.vram_mode`、`compute.vram_limit_mb` | 模型如何使用显存：放不下的模型部分加载，其余权重从内存流式加载（`high`：只整体加载） |
| `compute.cuda_graphs` | 在 NVIDIA 显卡上以 CUDA 图运行解码步骤（默认开启；结果不变） |
| `compute.attention`、`compute.compile` | 可选的注意力后端（FlashAttention、xFormers、SageAttention），以及用 `torch.compile` 编译主要模块 |
| `training.memory` | 设为 `low` 或 `lowest` 可在较小的 GPU 上微调（梯度检查点等） |
| `server.access_token` | 服务对本机以外开放时应当设置 |
| `tts.micro_batch` | 排队中仅文本不同的请求合并为一批运行 |

引擎专属的加载选项（CosyVoice 的 `fp16`、IndexTTS 的情感模型、Qwen3-TTS 的 `fused_projections`…）在“模型 › 引擎 › 选项”中，
或 `engines.<id>.load_options`。`mizuoto doctor` 会检查 torch 与硬件是否匹配，并说明各引擎还缺少什么（需接受许可的引擎的
软件包、可选扩展，例如 GPT-SoVITS 的日语前端 `pip install 'mizuoto[gptsovits-ja]'`）。

## 在另一台机器上运行引擎

```bash
# 在有显卡的机器上（它有自己的引擎与模型）：
mizuoto agent --host 0.0.0.0                    # 会显示它生成的令牌；用 --ssl-certfile 启用 TLS
# 在本机：
mizuoto remote add gpu-box 192.168.1.20 --token <令牌>
mizuoto remote use qwen3tts gpu-box --device cuda:0
```

参考音频与音色缓存按内容传输并保存在 agent 上；流式播放与取消和本地一样可用。训练始终在本机进行。

## 文档

文档发布在 <https://mizuoto.netlify.app/docs/>，每页都有中文与英文版本（源文件在 `site/content/docs/`，`.mdx` 与 `.en.mdx`）：

- **开始**：[快速开始](https://mizuoto.netlify.app/docs/quick-start/)、[安装](https://mizuoto.netlify.app/docs/installation/)、[首次运行](https://mizuoto.netlify.app/docs/first-run/)
- 网页界面各页面的**使用教程**、**命令行**及其**配置**
- **进阶**：[引擎](https://mizuoto.netlify.app/docs/advanced-engines/)、[HTTP 接口](https://mizuoto.netlify.app/docs/advanced-api/)、[远程主机](https://mizuoto.netlify.app/docs/advanced-remote/)、[性能](https://mizuoto.netlify.app/docs/advanced-performance/)、[下载](https://mizuoto.netlify.app/docs/advanced-downloads/)、[Docker](https://mizuoto.netlify.app/docs/advanced-docker/)、[嵌入](https://mizuoto.netlify.app/docs/advanced-embed/)、[常见问题](https://mizuoto.netlify.app/docs/advanced-faq/)
- **引擎参考**：各引擎如何复现其上游（[概览](https://mizuoto.netlify.app/docs/engines/)）
- **开发文档**：[架构](https://mizuoto.netlify.app/docs/development-architecture/)、[开发约定](https://mizuoto.netlify.app/docs/development-conventions/)、[运行时](https://mizuoto.netlify.app/docs/development-runtime/)、[显存与内存](https://mizuoto.netlify.app/docs/development-memory/)、[加速](https://mizuoto.netlify.app/docs/development-acceleration/)、[接入引擎](https://mizuoto.netlify.app/docs/development-engine/)、[测试](https://mizuoto.netlify.app/docs/development-testing/)、[发布](https://mizuoto.netlify.app/docs/development-release/)

`python scripts/dev.py site` 构建站点，`site-dev` 以热重载方式预览（都需要 bun）。

## 开发

```bash
python scripts/dev.py check     # 与 CI 相同的检查：代码风格、类型、测试、前端测试、生成的接口类型
python scripts/dev.py dev       # 同时运行接口与前端开发服务器（http://localhost:5173）
python scripts/dev.py test-engines qwen3tts      # 用已安装的模型运行契约测试
```

[AGENTS.md](AGENTS.md) 是代码地图，也是贡献者与编码代理的工作规则；[开发约定](https://mizuoto.netlify.app/docs/development-conventions/)
一页有完整说明。

## 构建

```bash
python scripts/dev.py wheels           # mizuoto 与需接受许可的引擎的 wheel，输出到 dist/（需要 bun 和 uv）
python scripts/dev.py smoke            # 在新环境中先装 CPU 版 torch，再一次装好它们并试用
python scripts/mirror_models.py build <目录>   # 把已安装的模型整理为 mizuoto-model 镜像的布局
```

`python scripts/dev.py wheels host` 只构建 Mizuoto；`wheels native fishspeech indextts` 构建需接受许可的引擎，供自行使用。

## 许可证

Mizuoto 采用 GNU 通用公共许可证第 3 版（仅限该版本，见 [LICENSE](LICENSE)）。各引擎的权重保留各自的许可证（见上表；
`mizuoto engine licence <id>`），Fish Speech 与 IndexTTS 的移植按其模型的许可证分发（`mizuoto-fishspeech`、
`mizuoto-indextts`）；合成结果带有生成它的引擎的许可标记与声明。
