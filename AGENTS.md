# Mizuoto — notes for agents

The working record for anyone changing this code, human or agent: what is where, the rules the code keeps and why,
how to work here, and where unfinished work is tracked. It is short on purpose. The full design and reference
documentation is the documentation site, `site/content/docs/` (every page in Chinese, `<slug>.mdx`, and English,
`<slug>.en.mdx`). Read the page for the area you touch before changing it:

| Area | Site page (`site/content/docs/…`) |
| --- | --- |
| Architecture, layers, services, data folder | `development-architecture` |
| Code style, layer rules, native-code rules, exactness, commits | `development-conventions` |
| Design lineage (ComfyUI, InvokeAI, SD Model Hub, rvc-next, MD3) and the design decisions | `development-design` |
| The runtime, engine contract, request context | `development-runtime`, `development-engine` |
| GPU/RAM memory: phases, budget, streaming, OOM retry, estimates | `development-memory` |
| Attention backends, CUDA graphs, sync-free sampling, compile, CPU settings | `development-acceleration` |
| How speed is measured, and the measured record | `development-performance` |
| One page per engine: what it reproduces from upstream, deviations, training, parity | `engines`, `engines-<id>` |
| Training, stages, fingerprints, training memory | `development-training` |
| API, events, OpenAPI types; the web UI | `development-api`, `development-webui` |
| Tests, the contract suite, the parity harness, tolerance tiers | `development-testing` |
| Packaging, versions, wheels, Docker, CI | `development-release` |
| User-facing behaviour and settings | `guide-*`, `cli-*`, `advanced-*` |

Unfinished work (GPU checks, hardware verification, planned features, known issues) is **not** in the repository: it
lives in the experiment workspace's backlog, `~/code_workspace/experiment/mizuoto-dev/` (§9).

## 1. What Mizuoto is

A TTS platform over eight engines: CosyVoice, F5-TTS, GPT-SoVITS, OmniVoice, Qwen3-TTS, VoxCPM, Fish Speech and
IndexTTS, plus Whisper for transcripts. One FastAPI server, one Vue 3 web UI (Material Design 3) and one Typer
command line sit over one core, and every engine runs natively in the same Python process.

- Names: package `mizuoto`, command `mizuoto`, env prefix `MIZUOTO_`, data directory `mizuoto`, port **7869**,
  GPL-3.0-only. Repository: <https://github.com/licyk/Mizuoto>. Model mirror: `mizuoto-model` (Hugging Face
  `licyk/mizuoto-model`, ModelScope `licyks/mizuoto-model`).
- One distribution, `mizuoto`: the core, API, CLI, web UI, the runtime and the engines `qwen3tts`, `f5tts`,
  `omnivoice`, `cosyvoice`, `gptsovits`, `voxcpm`, `asr` (internal speech recognition) and `dummy` (no model; tests).
  torch (≥ 2.8) is a dependency whose build (CUDA, ROCm, XPU, MPS, CPU) the user installs first.
- The two licence-gated engines are separate distributions, because their licences cover the ports:
  `mizuoto-fishspeech` (`packages/native/fishspeech`) and `mizuoto-indextts` (`packages/native/indextts`). They
  register under the entry-point group `mizuoto.engines` and install into the same environment.
- No upstream code is vendored and no engine imports transformers. Each engine is a port that gives upstream's output
  sample for sample on the CPU (fp32, same seed). The reference is 0.1, the engine-worker release that ran upstream's
  code in one environment per engine (git tag `v1-engine-workers`). VoxCPM, which 0.1 never had, is checked against
  upstream's own package.
- Each distribution has its own version and is checked and released on its own: `mizuoto/version.py` (0.2.11; the
  web UI's `package.json` follows by hand) and `packages/native/<id>/mizuoto_<id>/version.py`. A push to main that
  changes a version file releases that distribution; tags are `v<version>` and `mizuoto-<id>-v<version>`. A gated
  package names the Mizuoto versions it works with (`mizuoto>=0.2.0,<0.3` in its `pyproject.toml`): raise the lower
  bound when it uses something newer (release Mizuoto first), and release it with a raised range when Mizuoto leaves
  the range (its `test_<id>_package.py` fails then). A gated engine's options are in its manifest in the core, so
  changing them needs a Mizuoto release too. Details: `development-release`.

## 2. Layout

```
mizuoto/
  core/         services; context.py builds them (build_services, open_services); errors.py; codes.py (error and exit
                codes, standard library only); process.py
    settings/ events/ db/ net/    skeleton (settings: models.py, service.py; unknown keys are ignored)
    compute/    device probe without torch (probe.py), torch_index.py (the right torch build), cpu.py (allocator and
                thread placement before torch loads), accel.py (optional attention packages from their metadata)
    jobs/       queue with named resources, persisted requests, retry; events.py + jsonl.py (job-worker events)
    inference/  the core's only door to model code: service.py (InferenceService), runtime.py (RuntimeService),
                engine.py (EngineHandle, EngineRequest, ModelRef), local.py (LocalEngine), tasks.py (task types,
                standard library only)
    engines/    manifests/*.toml + manifests/files/*.files.json (pinned files), registry, params, languages, licences
    models/     catalogue, shared assets, downloads (sources, resume, SHA-256), verify, import, detect
    audio/ text/ tts/ presets/ asr/ voices/ outputs/ datasets/ training/ projects/ remote/
  cli/          Typer: app.py registers everything; commands/ are plain functions. Thin.
  api/          create_app, routers/ (one per resource, /api/v1/...; openai.py at {prefix}/v1), security, socket.io
                events, OpenAPI, the built UI. Thin.
  embed.py      MizuotoServer: the API and UI inside another application
  runtime/      the native runtime: engine.py (Engine/LoadedEngine, registry), context.py (RequestContext, phases),
                device.py, handle.py (ModelHandle), units.py, stream.py (Streamer), policy.py (VramPolicy),
                manager.py (ModelManager), ops.py, loading.py, graphs.py (CUDA graphs), sampling.py, compile.py,
                accel/ (attention dispatch, backends, policy), lm/ (Qwen decoder, KV caches), convert/onnx.py, snapshot.py
  nn/           shared building blocks: dsp/ (bit-exact mel, fbank, resamplers, WAV/FLAC), flow/, transformer/,
                vocoders/, codecs/, ssl/, speaker/
  text/         BPE tokenizers (as transformers builds them), Whisper's vocabulary, wetext/WeTextProcessing normalisers
  training/     Trainer, StageContext, stage.py (python -m mizuoto.training.stage), memory.py, rng.py
  engines/      one package per engine (engine.py, train.py where it trains, its own modules and data)
  webui/        Vue 3 + Material 3 (src/; built into dist/, which ships in the wheel)
packages/native/  fishspeech/mizuoto_fishspeech, indextts/mizuoto_indextts (licence-gated engines), each with its
              version.py and tests/ (tiny random models, test_<id>_package.py; run with the root's tests, sharing
              tests/engines_native's and tests/core's helpers)
scripts/      dev.py (task runner), ci.py (what CI checks and Release releases for a change), model_manifests.py,
              build_wheels.py, smoke_wheels.py, generate_openapi.py, mirror_models.py / mirror_fetch.py /
              mirror_readme.py (the model mirror), parity/ (parity and speed harness, README there)
site/         the documentation site (fumadocs + TanStack Start, bun)
tests/        core/ api/ cli/ runtime/ text/ engines_native/ (tiny random models) engines/ (contract suite, deselected)
              test_architecture.py, test_ci.py
Dockerfile    one image: server, UI, every engine (GATED=1 adds the licence-gated ones)
.github/workflows/  ci.yml (checks by what a change touches), release.yml (per distribution), site.yml (GitHub Pages)
```

## 3. Rules the architecture keeps

Enforced by `tests/test_architecture.py`; the reasons are on `development-architecture` and `development-conventions`.

- `core`, `api` and `cli` never import torch, torchaudio, transformers, onnxruntime or a gated package. `core` reaches
  `runtime`, `nn`, `engines` and `training` only from `core/inference`, inside functions. Importing
  `mizuoto.core.context` loads neither torch, fastapi nor typer; `mizuoto.cli.app` neither torch nor fastapi.
- `core/codes.py`, `core/jobs/{events,jsonl}.py`, `core/remote/frames.py` and `core/inference/tasks.py` are standard
  library only. `runtime` imports from the core only `mizuoto.core.codes` and `mizuoto.core.inference.tasks`.
- Engines import `runtime`, `nn`, `text`, `training`, those two core modules and their own package, never another
  engine. The gated packages follow the same rule.
- **One operation, one core method**, taking and returning `Record` models. Routes and commands parse, call it and
  shape the output, so the API's JSON equals the CLI's `--json`.
- Errors are `core/errors.py` only, codes are `mizuoto.core.codes`; CLI hints ride in `detail["hint"]`.
- **Native code changes no process-wide state**: randomness only from `ctx.generator_for(device)` / `ctx.rng`; no
  `os.environ`, cwd, `sys.path`, `torch.backends` or `sdpa_kernel` changes, no `logging.basicConfig`; no draws while
  loading (`ops` layers skip initialisation). Only `runtime/accel/features.py` sets `compute.fast`'s flags and
  `training.deterministic`'s, called by the host and the training stage. The contract suite checks requests leave the
  process state alone (`mizuoto.snapshot`).
- Every attention computation in native code goes through `mizuoto.runtime.accel.attention(…, site=…)`; SDPA and its
  aten kernels are forbidden elsewhere. An autoregressive model's site sets `approx_ok=False`: quantised kernels
  (SageAttention) made them generate without end on a GPU.
- `pyproject.toml` is one distribution: no `mizuoto-*` dependency, torch a dependency.

## 4. The runtime in one page

Details: `development-runtime`, `development-memory`, `development-acceleration`, `development-engine`.

- **An engine** is a manifest (`core/engines/manifests/<id>.toml`: variants, capabilities, params, options, load
  options, catalogue with pinned files, shared `[[assets]]`, `[training.<method>]`) plus an `Engine` subclass
  (`runtime/engine.py`: `id`, `VERSION`, `OPTION_SCHEMA` equal to the manifest's options, `TRAINER`, `EXTRAS`,
  `DEVICES`, `detect_variant`, `load`) whose `load` returns a `LoadedEngine` (`describe`, `synthesize` as a generator
  of `AudioChunk`, optionally `prepare_voice`, `convert`, `edit`, `call`, `close`; its weight-holding `ModelHandle`s
  in `handles`). Built-in engines are in `BUILTIN`; others come from the `mizuoto.engines` entry points.
- **Using a model**: `RuntimeService.loaded(ModelRef, device, dtype, options, request)` gives an `EngineHandle` (a
  `LocalEngine` here, a `RemoteEngine` for `<host>/<device>`). Streams are generators: the engine runs only inside
  `next()`, with the request context bound and inference mode entered per step; callers close them to stop early.
  One call at a time per (engine, device) through the job resource `engine:<engine>:<device>`.
- **Loading runs outside inference mode**, and parameters stay as `nn.Module` makes them (no `requires_grad_(False)`):
  on the CPU, `F.linear` over a strided input takes another kernel when the weight requires grad or is an inference
  tensor, which breaks bit-exactness.
- **Memory** (ComfyUI-style): engines place components per phase (`ctx.use` / `ctx.run`) with a working-memory
  estimate sized by the request (not its worst case); the manager loads whole when free memory covers missing weights
  + working memory + `compute.vram_reserve_mb`, else other components give up memory and the request's own load
  partly, the rest streaming from RAM through forward hooks. No estimate refuses a load. A phase under `ctx.run` that
  runs out of memory runs again (at most twice) with more streamed and its generators put back: identical, slower.
  Partial placements are explained in the log and in `meta.memory.partial`. Host memory is registered in place
  (`cudaHostRegister`) only by `runtime/pinning.py`, which releases it with the storage. Only linear/conv weights
  stream; code that reads a unit's weights outside its forward must mark it `fixed`. New engines must pass
  `tests/engines_native/streaming.py` (every unit streamed, half, none: identical on a simulated device). Mark an
  autoregressive model's handle `kind="sequence"`: its streamed layers are never prefetched (a token's work hides no
  copy; prefetching only took room from resident weights), while models run once per input prefetch on side streams.
- **Speed without changing results**: on NVIDIA GPUs every autoregressive decoder runs its steps as CUDA graphs
  (`compute.cuda_graphs`, on by default, `runtime/graphs.py`) with one wait for the device per step
  (`runtime/sampling.py`'s sync-free draws). Graphs replay the eager kernels, so outputs are identical; they are used
  for a model wholly on its device (or a streamed `sequence` handle in pinned memory, `ModelHandle.graphable`: the
  graphs copy its weights; only Qwen3-TTS, where it was measured faster) and dropped when its weights move. Capture and replay them under
  `torch.cuda.device(graphs.device)` (a replay runs on the current device's stream: a model on `cuda:1` faults
  otherwise); none are captured under cudaMallocAsync (`graphs.usable`). Opt-ins that change bits are labelled in
  the output: `compute.compile`, non-`auto` attention backends, Qwen3-TTS's `fused_projections`, OmniVoice's
  `packed_guidance`, VoxCPM's `attention_span`, `compute.fast`.
- Devices: `cpu`, `cuda:N` (NVIDIA, and AMD under ROCm), `xpu:N`, `mps`. The host probes them without importing
  torch. CosyVoice 3's float64 F0 predictor stays on the CPU where the device lacks float64 (MPS, Intel Arc).

## 5. Exactness: the rule for every change

- A port gives its upstream's output sample for sample on the CPU in fp32 under the same seed. Each engine's page
  (`engines-<id>`) records what that took and every deliberate deviation. Read it before touching the engine.
- **Speed work keeps every result identical.** Measure with `scripts/parity/perf.py` / `train_perf.py`, which compare
  outputs with an earlier round. A faster path that changes bits ships only as an opt-in, named in the output's
  `applied`/`params`, measured against the exact path, never instead of it.
- Behaviour changes and bug fixes against upstream are separate changes, recorded on the engine's page.
- Exactness is checked by `tests/engines_native/` and `packages/native/*/tests/` (tiny random models, every engine and
  trainer, in CI), the contract suite (`scripts/dev.py test-engines`, real models through the whole stack) and the
  parity harness (`scripts/parity/`; tiers T0–T4 on `development-testing`).
- On a GPU, "identical" means identical to the code before on the same GPU, driver, torch and settings. A rewrite proven
  exact on the CPU is not proven on a GPU: SDPA's CUDA paths split their sums by the key length (so a shorter masked
  cache changes bits: VoxCPM's span is CPU-only), and under autocast in-place operations keep their buffer's dtype
  where the out-of-place ones autocast to float32 (Fish Speech's Snake). Check such a change on a GPU, against the code
  before (`development-performance`, G1, runs every step since round 1 that way).
- TorchScript functions are fused on CUDA after their first call per shape: their bits depend on the shapes the process
  has seen, and the fuser compiles on the current device (a training stage makes its device current for it). Only
  GPT-SoVITS's training gate stays scripted, as upstream's (its symbolic gradients differ from eager ones); VoxCPM's
  Snake runs eagerly. Do not add more.
- Compare CPU runs with the same thread count: CPU kernels split their sums by it.

## 6. Feature areas: where to look

- **Synthesis** (`core/tts`): `TtsService.plan` validates everything against the manifest before a job exists;
  `speak` runs a job per request (resources `engine:<engine>:<dev>`, `gpu:<dev>`), joins segments, normalises,
  encodes and records an `Output`. Streaming (`StreamSink`, WebSocket, chunked WAV), streamed text input (CosyVoice
  2/3), conversion and editing, the OpenAI-compatible endpoint, presets and micro-batching are here.
- **Models and assets** (`core/models`, `core/engines`): catalogue entries with pinned revisions, sizes and SHA-256
  (`scripts/model_manifests.py`); downloads with resume and source fallback (`downloads.source`, then the others);
  shared `_assets/`; licence gates (`engines.<id>.accepted_licence`); the `mizuoto-model` mirror
  (`downloads.repository`), built and checked by `scripts/mirror_*.py`.
- **Voices and ASR** (`core/voices`, `core/asr`): voice folders are the truth; voice caches keyed by model, revision,
  references and Mizuoto's version; transcripts cached by audio SHA-256.
- **Datasets and training** (`core/datasets`, `core/training`, `mizuoto/training`, each engine's `train.py`): stages
  declared in the manifest, fingerprints chained, each stage a job worker (`python -m mizuoto.training.stage`).
  Trainers reproduce upstream's losses and weights bit for bit on the CPU when seeded alike. Training memory presets
  (`training.memory`) are `development-memory`.
- **Projects**, **remote hosts** (`mizuoto agent`, `AGENT_PROTOCOL` 2), **embedding**: `core/projects`,
  `core/remote`, `embed.py`; user docs `guide-projects`, `advanced-remote`, `advanced-embed`.
- **API and web UI**: `scripts/dev.py typegen` writes `mizuoto/webui/src/api/schema.d.ts` from the OpenAPI schema
  (commit it; `typegen-check` fails when stale). Feature code imports only `@/ui`; the seed colour is `#4f6b3a`;
  `AppIcon` sets `fill: none` (Material buttons fill slotted icons). Every user-visible string is in both
  `i18n/en.ts` and `i18n/zh-CN.ts`; server English is shown through `tx` with `i18n/catalog.zh-CN.ts`.

## 7. Working here

```bash
python scripts/dev.py check          # all CI checks: lint, typecheck-py, test-py, test-web, typegen-check
python scripts/dev.py format         # ruff format and the safe fixes
python scripts/dev.py dev            # API server + Vite with hot reload (http://localhost:5173)
python scripts/dev.py test-engines qwen3tts      # contract suite (installed models; MIZUOTO_TEST_DEVICE=cpu|cuda:N,
                                                 # MIZUOTO_TEST_SETTINGS='{"compute": {"vram_limit_mb": 4096}}')
python scripts/dev.py model-manifests [engine]   # re-pin catalogue files after a manifest change
python scripts/dev.py typegen                    # the web UI's API types
python scripts/dev.py wheels && python scripts/dev.py smoke   # the distributions, tried in a new environment
python scripts/dev.py site-dev | site | test-site             # the documentation site
```

- Environment: an isolated `.venv` from Python 3.11: `uv venv .venv --python 3.11`, torch's CPU build
  (`uv pip install --python .venv/bin/python "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu`), then
  `uv pip install --python .venv/bin/python -e . -e packages/native/fishspeech -e packages/native/indextts pytest
  pytest-cov ruff ty tomli httpx`. Run everything with `.venv/bin/python`. ty resolves the gated packages through
  `tool.ty.environment.extra-paths`.
- A CLI command is a plain function in `cli/commands/`, registered in `cli/app.py:get_app()` and added to
  `EXPECTED_TREE` in `tests/cli/test_cli.py`. A setting goes in `core/settings/models.py` with its env override,
  the UI's Settings page (both languages) and `cli-config` on the site.
- **Commit and push only when the user asks.** Messages: `type(scope): summary` (`feat`, `fix`, `perf`, `docs`,
  `refactor`, `test`, `chore`), a body that says what changed and why. Stage with `git add -A` and check
  `git diff --cached --stat` before committing.
- **This machine** (the maintainer's development box): two RTX 3090s shared with another workload that usually
  leaves about 3 GB free on each. Never kill or slow it. Run tests with `CUDA_VISIBLE_DEVICES=` unless a GPU run is
  intended and the memory is free (`nvidia-smi`). The Bash tool's shell is zsh: an unquoted `$VAR` does not
  word-split, so use `bash -c` for loops over lists. Installed models and 0.1's engine environments are under
  `~/.local/share/kodama/` (the data folder's old name); point `MIZUOTO_DATA_DIR` or `MIZUOTO_TEST_DATA_DIR` there
  for real-model runs. The parity workspace is `~/code_workspace/experiment/kodama-refactor` (`MIZUOTO_PARITY_DIR`),
  whose reference host runs 0.1 (package `kodama`); `scripts/parity/stack.py` imports either name.

## 8. Conventions

- Python ≥ 3.10; ruff and ty (unpinned, `lint.select` written out); line length 180; every function annotated
  (tests and scripts exempt); pydantic v2 `Record` with `json_schema_serialization_defaults_required`; heavy imports
  (torch, numpy, engines) inside functions; no method named `list` on a class that annotates with `list[...]`.
- Comments and docstrings say what the code does and why, in plain sentences. Do not cite planning documents: there
  are none in the repository. Reasons belong in the code or on the site.
- Documentation: the site is the documentation. A user-visible change updates its `guide-*`/`cli-*`/`advanced-*`
  page; a design change updates its `development-*` or `engines-*` page; always both languages (a test checks every
  Chinese page has its English twin and both sidebars match: `site/content/docs/meta.json`, `meta.en.json`). Durable
  findings that a future change must not break go on the relevant site page, and in this file only if they are
  rules (§3–§5).
- `README.md` and `README.zh-CN.md` move together.

## 9. Continuing development

Outstanding work is tracked outside the repository, in `~/code_workspace/experiment/mizuoto-dev/`:

- `README.md` there: the index, with each item's place and the workspaces that hold its scripts and results;
- `backlog/`: one file per theme (GPU checks and optimisation, attention backends on a GPU, pending real-model runs,
  hardware and release checks, features and ideas, known issues), each a checklist;
- `archive/`: the design documents the repository used to carry (`docs/`, `hanaikada_docs/`), verbatim, for
  provenance: the refactor, VRAM and acceleration plans, the performance record, 0.1's design and upstream analyses.

To pick up work: read that README, take an item, read the site pages for its area, do it, then tick the item (with
date and commit) and move what was learned onto the site. A new idea or a found problem becomes an item in the right
backlog file, not a file in this repository.
