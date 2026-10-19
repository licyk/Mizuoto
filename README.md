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
    <img src="https://img.shields.io/badge/licence-GPL--3.0--only-bluegreen?style=flat&labelColor=grey" alt="Licence: GPL-3.0-only">
  </a>
</p>

**English** · [简体中文](README.zh-CN.md)

</div>

Open text-to-speech engines on one platform: CosyVoice, F5-TTS, GPT-SoVITS, OmniVoice, Qwen3-TTS, VoxCPM,
Fish Speech and IndexTTS. They share one web UI, one command line and one HTTP API. Speak, clone and design voices,
convert and edit speech, read whole books with a cast of voices, and fine-tune models on your own recordings.

Every engine is implemented in Mizuoto itself, on one runtime (PyTorch) in one Python environment, and gives the same
output as its upstream project. Requests, voices, parameters, jobs and results look the same whichever engine serves
them. GPU memory is managed for all of them together: a model that does not fit loads partly, with the rest of its
weights streamed from RAM.

- **Website:** <https://mizuoto.netlify.app>
- **Documentation:** <https://mizuoto.netlify.app/docs/> (English and Chinese; its sources are in `site/`, and
  `python scripts/dev.py site-dev` serves them locally)

## Features

- **Speak** with any installed model: a voice from your library, a clip to clone, a preset speaker or a described
  voice. Style instructions and emotions where the model has them, playback while the audio is being made, and
  parameter presets.
- **Voices**: a library of clips with transcripts (speech recognition fills them in), designed voices, preset
  speakers, and voices with their own fine-tuned weights. Export and import.
- **Projects**: books, scripts and dialogues read line by line by a cast of voices. Each speaker can use its own model,
  even a different engine in the same dialogue. Changed lines are rendered again, takes are kept, and a chapter or the
  whole project exports as one file with subtitles.
- **Tools**: voice conversion (CosyVoice) and speech editing (F5-TTS).
- **Train**: build datasets from recordings, files, outputs or folders in common layouts (cut at silences,
  transcribed, checked). Experiments then run each engine's own fine-tuning recipe stage by stage, with live loss
  curves and spoken previews. The result is a voice's own weights or a new model.
- **Remote hosts**: run engines on another machine (`mizuoto agent` there); voices, outputs and the UI stay here.
- **Run it your way**: the web UI (Material Design 3, English and Chinese), the `mizuoto` command line, the REST API
  with a WebSocket for events and streaming, an OpenAI-compatible `/v1/audio/speech`, or Python
  (`mizuoto.core.context.open_services()`, or `mizuoto.embed.MizuotoServer` inside another application).

## Engines

| Engine | Models | What it adds | Training | Licence |
| --- | --- | --- | --- | --- |
| Qwen3-TTS | 1.7B / 0.6B Base, CustomVoice, VoiceDesign | 10 languages, preset speakers with instructions, voices from a description | a speaker (SFT) | Apache-2.0 |
| CosyVoice | 3 (0.5B), 2 (0.5B), 300M base / SFT / instruct | streaming in and out, voice conversion, instructions | LLM, LLM + flow | Apache-2.0 |
| F5-TTS | v1 Base, Base, E2-TTS | flow matching, exact duration, speech editing | full fine-tune | MIT code; CC-BY-NC-4.0 weights |
| GPT-SoVITS | v2, v2Pro, v2ProPlus, v4, v5 Turbo | few-shot cloning, a voice's own GPT and SoVITS weights | a voice's weights | MIT |
| OmniVoice | OmniVoice | 600+ languages, voices from attributes, inline tags | full, LoRA | Apache-2.0 |
| VoxCPM | VoxCPM2, 1.5, 0.5B | tokenizer-free; VoxCPM2: 30 languages at 48 kHz, cloning with or without a transcript, voices from a description, style instructions | full, LoRA | Apache-2.0 |
| Fish Speech | S2 Pro | expressive cloning | LoRA | Fish Audio Research License (licence-gated) |
| IndexTTS | 1.5, 2, 2.5 | emotion from weights, a clip or a description | — | bilibili Model Use License (licence-gated) |

Speech recognition (Whisper large-v3-turbo) is built in for transcripts. The licence-gated engines are separate
packages (`mizuoto-fishspeech`, `mizuoto-indextts`). Each shows its licence, and its models download only after you
accept it. Results carry the licence's notices. F5-TTS's pretrained weights are for non-commercial use.

## Project status

Mizuoto **0.2.0** is the first release under this name.

- Every engine and every training method gives its upstream's output on the CPU (fp32, same seed). CosyVoice does so
  given the same voice prompt. The check is sample for sample, through the whole stack, against the previous
  engine-worker implementation (tag `v1-engine-workers`) or, for VoxCPM, upstream's own package. A test suite on real
  models passes for every engine on the CPU.
- On NVIDIA GPUs, every autoregressive decoder runs as CUDA graphs with one wait for the device per step, producing
  the same output as eager mode. These speed-ups were measured on a GPU shared with other work. Wider GPU measurements
  are still to come: real models end to end, attention backends, memory streaming and the test suite.
- ROCm, Intel XPU and Apple MPS are supported in code and tested with simulated devices, but not yet on such hardware.
- The Docker image is written but has not yet been built and run as a whole.

## Install

You need Python 3.10–3.13 and torch 2.8 or newer, in the build for your hardware (CUDA, ROCm, XPU, Apple silicon or
CPU). Install torch first, from the index that matches your machine, then Mizuoto. `mizuoto doctor` checks the build
against the hardware and prints the exact command when they do not match. An NVIDIA GPU is recommended; every engine
also runs on a CPU, slowly.

**From PyPI:**

```bash
# torch first, the build for your hardware (mizuoto doctor names it):
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128    # NVIDIA, Linux or Windows (cu126, cu128, cu129, cu130, cu132: what the driver supports)
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/rocm7.2  # AMD, Linux (rocm6.4, rocm7.1 for older ROCm)
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/xpu      # Intel GPUs, Linux or Windows
pip install "torch>=2.8"                                                       # macOS on Apple silicon (MPS)
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu      # CPU only
# then Mizuoto, in one install (add mizuoto-fishspeech mizuoto-indextts for the licence-gated engines):
pip install mizuoto
mizuoto doctor
mizuoto webui                                   # http://127.0.0.1:7869
```

**From a source checkout:**

```bash
git clone https://github.com/licyk/Mizuoto.git && cd Mizuoto
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .   # -e packages/native/fishspeech -e packages/native/indextts for the gated engines
python scripts/dev.py web-install && python scripts/dev.py web    # build the UI once (needs bun)
.venv/bin/mizuoto webui
```

**With Docker** (models go to the `/data` volume):

```bash
docker build -t mizuoto .        # torch's CUDA 12.8 build; --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu for CPU,
                                 # --build-arg GATED=1 to add the licence-gated engines
docker run --gpus all -p 7869:7869 -v mizuoto-data:/data -e MIZUOTO_SERVER__ACCESS_TOKEN=change-me mizuoto
```

One image holds the server, the UI and every engine, with GPT-SoVITS's Japanese and Korean front ends.

## Quick start

The web UI opens on **Speak**; the dummy engine works at once (it speaks in tones). Settings › Compute checks torch
against the hardware, and the Speak button turns into the download of a model that is not installed yet. The same from
the command line:

```bash
mizuoto model download qwen3tts/1.7b-custom-voice
mizuoto speak "Nice to meet you." -m qwen3tts/1.7b-custom-voice --preset vivian --instruct "cheerfully" -o vivian.wav
```

A tour of the command line:

```bash
# No model needed: the dummy engine speaks in tones.
mizuoto speak "Hello there." -m dummy/default --preset alto -o hello.wav

# Speech recognition fills in the transcripts of reference clips.
mizuoto model download asr/whisper-large-v3-turbo

# Voices: design one, or add one from a clip, then clone it.
mizuoto model download qwen3tts/1.7b-voice-design qwen3tts/1.7b-base
mizuoto voice design Narrator -m qwen3tts/1.7b-voice-design -d "A warm, low male voice, calm and slow"
mizuoto voice add Me --ref me.wav --consent "my own voice"
mizuoto speak "Any text at all." -m qwen3tts/1.7b-base --voice narrator -o narrator.wav

# Streaming, conversion and editing.
mizuoto model download cosyvoice/cosyvoice2-0.5b
mizuoto speak "Hear it as it is made." -m cosyvoice/cosyvoice2-0.5b --voice narrator --stream | ffplay -nodisp -
mizuoto convert speech.wav -m cosyvoice/cosyvoice2-0.5b --voice narrator -o converted.wav
mizuoto model download f5tts/v1-base
mizuoto edit speech.wav -m f5tts/v1-base -t "The whole text after the edit." --span 1.2:1.8 -o edited.wav

# Licence-gated engines (install mizuoto-indextts): read the licence, accept it, then download.
mizuoto engine licence indextts && mizuoto engine accept-licence indextts
mizuoto model download indextts/v2
mizuoto speak "What a day!" -m indextts/v2 --ref me.wav --consent "my own voice" --emotion happy=0.8 -o happy.wav

# A long-form project: a script with chapters and dialogue, a cast, rendered and exported with subtitles.
mizuoto project new "The Night Train" -m qwen3tts/1.7b-custom-voice --preset ryan --script book.txt
mizuoto project cast the-night-train anna --preset serena
mizuoto project render the-night-train
mizuoto project export the-night-train -o book.mp3 --srt book.srt

# Presets, and any OpenAI client against http://127.0.0.1:7869/v1
mizuoto preset save Calm -m cosyvoice/cosyvoice2-0.5b -p speed=0.9 --default
curl -s http://127.0.0.1:7869/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "cosyvoice/cosyvoice2-0.5b", "input": "Hello!", "voice": "narrator"}' -o hello.mp3
```

`mizuoto --help` lists every command; `--json` on a command prints what the API returns.

## Training

First a dataset of one voice's clips, then an experiment that fine-tunes a model on it, stage by stage. The web UI's
Train screen does the same, with live loss curves:

```bash
mizuoto dataset create Narrator --lang en --consent "my own voice"
mizuoto dataset add narrator recordings/          # a .lab or .txt beside a clip is its text
mizuoto dataset prepare narrator                  # cut at silences, transcribe, normalise
mizuoto dataset check narrator
mizuoto asset download gptsovits/s2d-v2pro        # v2Pro's pretrained discriminator
mizuoto train methods gptsovits/v2pro-base        # what can be trained, with which settings
mizuoto train new narrator --base gptsovits/v2pro-base --dataset narrator --hp sovits_epochs=8
mizuoto train run narrator                        # exports the voice's own GPT and SoVITS weights
mizuoto train metrics narrator
```

Changing a setting or the dataset marks every stage that depends on it as out of date. `train run` continues from the
first stage that is not done. In practice a fine-tune needs a GPU; every trainer also runs on a CPU for small trials.

## Configuration and performance

Settings live in `settings.toml` in the data folder (`mizuoto config path`; `mizuoto config show`). Any of them can
come from an environment variable instead, e.g. `MIZUOTO_SERVER__PORT=8000`. A few worth knowing:

| Setting | What it does |
| --- | --- |
| `engines.<id>.device`, `engines.<id>.dtype` | where an engine runs (`auto`, `cuda:1`, `cpu`) and its precision |
| `downloads.source` | `huggingface`, `hf-mirror`, `modelscope` or a custom endpoint |
| `downloads.repository` | `mizuoto-model` (default): Mizuoto's mirror first, then each model's own repository; `official`: only the latter |
| `compute.max_resident_models`, `compute.unload_after_minutes` | how many models stay loaded, and when an idle one is unloaded (`mizuoto runtime` lists them) |
| `compute.vram_mode`, `compute.vram_limit_mb` | how models use GPU memory: a model that does not fit loads partly, the rest of its weights streaming from RAM (`high`: whole or not at all) |
| `compute.cuda_graphs` | decoding steps as CUDA graphs on NVIDIA GPUs (on by default; same results) |
| `compute.attention`, `compute.compile` | optional attention backends (FlashAttention, xFormers, SageAttention) and `torch.compile` of the heavy modules |
| `training.memory` | `low` or `lowest` to fine-tune on a smaller GPU (gradient checkpointing and more) |
| `server.access_token` | set it whenever the server listens beyond this machine |
| `tts.micro_batch` | queued requests that differ only in their text run as one batch |

Engine-specific load options (CosyVoice's `fp16`, IndexTTS's emotion model, Qwen3-TTS's `fused_projections`…) are
under Models › Engines › Options, or `engines.<id>.load_options`. `mizuoto doctor` checks torch against the hardware
and says what each engine still needs: a licence-gated engine's package, or an optional extra such as GPT-SoVITS's
Japanese front end (`pip install 'mizuoto[gptsovits-ja]'`).

## Engines on another machine

```bash
# On the GPU machine (its own engines and models):
mizuoto agent --host 0.0.0.0                    # prints the token it made; TLS with --ssl-certfile
# Here:
mizuoto remote add gpu-box 192.168.1.20 --token <token>
mizuoto remote use qwen3tts gpu-box --device cuda:0
```

Reference clips and voice caches travel by content and are kept on the agent. Streaming and cancelling work as they
do locally; training stays on this machine.

## Documentation

The documentation is published at <https://mizuoto.netlify.app/docs/>, every page in Chinese and English (its sources: `site/content/docs/`,
`.mdx` and `.en.mdx`):

- **Getting started**: [quick start](https://mizuoto.netlify.app/docs/quick-start/), [installation](https://mizuoto.netlify.app/docs/installation/), [first run](https://mizuoto.netlify.app/docs/first-run/)
- **Guides** to every screen of the web UI, the **command line** and its **configuration**
- **Advanced**: [engines](https://mizuoto.netlify.app/docs/advanced-engines/), [HTTP API](https://mizuoto.netlify.app/docs/advanced-api/), [remote hosts](https://mizuoto.netlify.app/docs/advanced-remote/), [performance](https://mizuoto.netlify.app/docs/advanced-performance/), [downloads](https://mizuoto.netlify.app/docs/advanced-downloads/), [Docker](https://mizuoto.netlify.app/docs/advanced-docker/), [embedding](https://mizuoto.netlify.app/docs/advanced-embed/), [FAQ](https://mizuoto.netlify.app/docs/advanced-faq/)
- **Engine reference**: how each engine reproduces its upstream ([overview](https://mizuoto.netlify.app/docs/engines/))
- **Development**: [architecture](https://mizuoto.netlify.app/docs/development-architecture/), [conventions](https://mizuoto.netlify.app/docs/development-conventions/), [runtime](https://mizuoto.netlify.app/docs/development-runtime/), [memory](https://mizuoto.netlify.app/docs/development-memory/), [acceleration](https://mizuoto.netlify.app/docs/development-acceleration/), [adding an engine](https://mizuoto.netlify.app/docs/development-engine/), [testing](https://mizuoto.netlify.app/docs/development-testing/), [release](https://mizuoto.netlify.app/docs/development-release/)

`python scripts/dev.py site` builds the site and `site-dev` serves it with hot reload (both need bun).

## Development

```bash
python scripts/dev.py check     # what CI runs: lint, types, tests, web tests, generated API types
python scripts/dev.py dev       # the API with the web UI's dev server (http://localhost:5173)
python scripts/dev.py test-engines qwen3tts      # the contract suite against installed models
```

[AGENTS.md](AGENTS.md) is the map of the code and the working rules for contributors and coding agents; the
[conventions](https://mizuoto.netlify.app/docs/development-conventions/) page explains them in full.

## Build

```bash
python scripts/dev.py wheels           # mizuoto and the licence-gated engines' wheels into dist/ (needs bun and uv)
python scripts/dev.py smoke            # install them (CPU torch, then one install) in a new environment and try them
python scripts/mirror_models.py build <dir>   # the installed models laid out as the mizuoto-model mirror
```

`python scripts/dev.py wheels host` builds Mizuoto alone; `wheels native fishspeech indextts` builds the licence-gated
engines, for your own use.

## Licence

Mizuoto is licensed under the GNU General Public License v3.0 only ([LICENSE](LICENSE)). The engines' weights keep
their own licences (see the table above, or `mizuoto engine licence <id>`). The Fish Speech and IndexTTS ports are
distributed under their models' licences (`mizuoto-fishspeech`, `mizuoto-indextts`). Results carry the tags and notices
of the engine that made them.
