"""Settings models. Field docstrings become the descriptions in the API schema."""

import os
from typing import Any, Literal

from pydantic import Field

from mizuoto.core.record import Record

DEFAULT_PORT = 7869
"""Clear of SD Model Hub/Hanakura (7865), IIB (7866), Hanaikada (7867) and rvc-next (7868)."""


class ServerSettings(Record):
    host: str = "127.0.0.1"
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    strict_port: bool = False
    open_browser: bool = True
    access_token: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    ssl_certfile: str | None = None
    """A PEM certificate (chain): serve HTTPS. Browsers on other machines let a page record the
    microphone only over HTTPS."""
    ssl_keyfile: str | None = None
    """The certificate's private key, when ``ssl_certfile`` does not include it."""

    def ssl(self) -> dict[str, str]:
        """The TLS files in use (uvicorn's names), with ``~`` expanded; empty for plain HTTP."""
        if not self.ssl_certfile:
            return {}
        files = {"ssl_certfile": self.ssl_certfile, **({"ssl_keyfile": self.ssl_keyfile} if self.ssl_keyfile else {})}
        return {k: os.path.expanduser(v) for k, v in files.items()}


class PathSettings(Record):
    """Empty folders mean "inside the data directory"."""

    models_dir: str = ""
    voices_dir: str = ""
    outputs_dir: str = ""
    datasets_dir: str = ""
    experiments_dir: str = ""
    projects_dir: str = ""
    cache_dir: str = ""
    browse_roots: list[str] = Field(default_factory=list)
    """Server folders the UI may read. Empty: home, the data directory and every drive or volume."""


VramMode = Literal["auto", "high", "normal", "low", "none"]
AttentionBackend = Literal["auto", "fast", "sdpa", "math", "sdpa_flash", "sdpa_efficient", "sdpa_cudnn", "flash", "xformers", "sage", "sage3"]
"""``compute.attention``."""
TrainingAttention = Literal["auto", "fast", "sdpa", "math", "sdpa_flash", "sdpa_efficient", "sdpa_cudnn", "flash", "xformers"]
"""``training.attention``: the backends with a backward pass (SageAttention has none)."""


class ComputeSettings(Record):
    devices: list[str] = Field(default_factory=list)
    """Devices Mizuoto may use, such as ``cuda:0``. Empty: every device found."""
    default_device: str = "auto"
    """auto, cpu, cuda:N (NVIDIA, or AMD on ROCm), xpu:N (Intel) or mps."""
    unload_after_minutes: int = Field(default=20, ge=0)
    """Unload a model nobody used for this long; 0 keeps models loaded."""
    max_resident_models: int = Field(default=3, ge=1, le=32)
    """Models kept loaded at once (in memory, on their device or offloaded to RAM); the least recently used idle one
    is unloaded first."""
    vram_reserve_mb: int = Field(default=1024, ge=0)
    """GPU memory left free beyond what a request needs (for the system and other programs)."""
    vram_mode: VramMode = "auto"
    """How models use GPU memory. ``auto``/``normal``: whole when they fit, else partly, the rest of their weights
    streaming from RAM as layers need them (slower, but it runs). ``high``: whole or not at all. ``low``: weights take at
    most half the GPU's memory, leaving room for other models. ``none``: almost everything streams (least memory,
    slowest)."""
    vram_limit_mb: int = Field(default=0, ge=0)
    """Use at most this much of each GPU (0: all of it), e.g. when sharing it with other programs."""
    ram_reserve_mb: int | None = Field(default=None, ge=0)
    """RAM left free: below it host copies of weights are dropped and idle models unloaded. None: the larger of 2 GB
    and a tenth of the RAM."""
    host_copies: Literal["auto", "on", "off"] = "auto"
    """Keep a copy in RAM of the weights of models on a GPU, so moving them off it costs nothing (``auto``: while RAM
    allows)."""
    max_pinned_mb: int = Field(default=0, ge=0)
    """Page-locked RAM for weights that stream to a GPU (faster copies). 0: automatic (40 % of the RAM on Windows, up to
    90 % elsewhere)."""
    async_offload: int = Field(default=-1, ge=-1, le=8)
    """Side streams that copy streamed weights while the GPU computes (-1: 2 on NVIDIA and AMD, none elsewhere; 0: off)."""
    oom_retry: bool = True
    """When a step runs out of GPU memory, run it again with more of its weights streamed (the same result, slower)."""
    oom_fallback: bool = False
    """After that, allow an engine's cheaper way for the step (tiled decoding, split batches): it fits in less memory,
    and its output differs slightly."""
    kv_cache: Literal["dynamic", "static"] = "dynamic"
    """The language models' key/value caches (Qwen3-TTS, CosyVoice 2/3): ``dynamic`` grows them by concatenation, as
    the original implementations do; ``static`` writes into preallocated buffers (less copying and fragmentation on
    a GPU; may change results in the last bits there). Applies to models loaded after the change."""
    cuda_graphs: bool = True
    """Run language models' decoding steps as CUDA graphs on NVIDIA GPUs (Qwen3-TTS's talker and code predictor, Fish Speech's slow and fast transformers, CosyVoice 2/3's language model, VoxCPM's patches, GPT-SoVITS's and IndexTTS's GPT): each
    step's kernels launch at once instead of one by one, which makes decoding several times faster. The results are
    identical. A model whose weights partly stream from RAM decodes without them. Applies to models loaded after the
    change."""
    cuda_allocator: Literal["auto", "native", "async", "expandable"] = "auto"
    """torch's CUDA memory allocator (``PYTORCH_CUDA_ALLOC_CONF``): ``native`` (torch's caching allocator), ``async``
    (cudaMallocAsync, ComfyUI's default; CUDA graphs are not used under it, so decoding is slower), ``expandable``
    (expandable segments: less fragmentation as caches grow; Linux). ``auto`` is ``native`` and leaves a
    ``PYTORCH_CUDA_ALLOC_CONF`` of the environment alone. Applies when Mizuoto next starts."""
    cpu_allocator: Literal["auto", "tuned", "system"] = "auto"
    """The C allocator torch's CPU tensors come from: ``tuned`` (``auto`` on Linux with glibc) keeps large blocks in
    the heap, up to 512 MB of free memory for reuse and large tensors on huge pages, which spares the kernel mapping
    and zeroing memory at every layer (10–20 % faster on the CPU, identical results; the process may hold that much
    more memory); ``system`` keeps the C library's defaults. Applies when Mizuoto next starts."""
    cpu_threads: Literal["auto", "cores", "off"] = "auto"
    """Where torch's CPU threads run: ``cores`` (``auto`` on Linux) binds them to distinct physical cores, so two never
    share one (that made a process a third slower for its whole life); ``off`` leaves them to the scheduler (better
    when other busy programs share the cores). Identical results either way. Applies when Mizuoto next starts."""
    simulate_vram_mb: int = Field(default=0, ge=0)
    """For tests: the CPU plays a GPU of this size (weights placed and streamed as on a GPU). 0: off."""
    gpu_jobs: int = Field(default=1, ge=1, le=16)
    """GPU jobs that may run at once on one device."""
    inference_jobs: int = Field(default=1, ge=1, le=16)
    """Synthesis jobs (speak, voice conversion, speech editing) that run at once, whatever their engine or device; the
    others wait in the queue, in order. Requests run together as one batch (``tts.micro_batch``) count as one job."""
    attention: AttentionBackend = "auto"
    """How native engines compute attention. ``auto``: as each model's original implementation does, so results are
    identical to it. ``fast``: the fastest kernels this device offers (SageAttention, FlashAttention, xFormers when
    installed, else torch's own), which change results slightly. A backend's name: that one first, then ``fast``'s.
    A call a backend cannot take falls back to the next one (logged, and listed in the output). Applies to models
    loaded after the change."""
    attention_strict: bool = False
    """Fail a request instead of falling back when the chosen attention backend cannot compute part of it."""
    fast: list[Literal["tf32", "fp16_accumulation", "cudnn_benchmark", "sdpa_reduced_precision"]] = Field(default_factory=list)
    """GPU speed switches for the whole process, each changing results slightly (ComfyUI's ``--fast``): ``tf32``
    (TensorFloat-32 products on NVIDIA GPUs from Ampere on), ``fp16_accumulation`` (fp16 products sum in fp16),
    ``cudnn_benchmark`` (cuDNN tries kernels per shape), ``sdpa_reduced_precision`` (SDPA's math kernel sums fp16/bf16
    in their own precision). Outputs list those in effect. Applies when Mizuoto next starts. GPT-SoVITS runs slower
    under ``tf32`` or ``fp16_accumulation`` (its GPT 1.3–1.6×, measured on an RTX 3090)."""
    compile: bool = False
    """Compile the heavy modules of native engines with torch.compile (F5-TTS's DiT, CosyVoice's and GPT-SoVITS's
    flow estimators, IndexTTS's DiT and BigVGAN) for faster synthesis. The first synthesis after a model loads is
    slower while it compiles; without a working compiler the engine runs as before. Applies to models loaded after
    the change. On a GPU VoxCPM runs without its CUDA graphs while compiled, and slower (0.41–0.64× on an RTX 3090)."""


class DownloadSettings(Record):
    source: Literal["huggingface", "hf-mirror", "modelscope", "custom"] = "huggingface"
    """Where model files come from. A file a source lacks comes from the next one that has it."""
    endpoint: str = ""
    """The endpoint for ``custom``: a Hugging Face compatible server."""
    verify_checksums: bool = True
    concurrency: int = Field(default=2, ge=1, le=8)
    repository: Literal["mizuoto-model", "official"] = "mizuoto-model"
    """``mizuoto-model``: Mizuoto's mirror first (every file laid out as Mizuoto keeps it, ``<engine>/<model>/…``),
    at the chosen source; a file the mirror lacks comes from the official repositories. ``official``: each model's
    own repositories only."""
    mirror_repo: str = "licyk/mizuoto-model"
    """The mirror's repository on Hugging Face (also used through hf-mirror and a custom endpoint)."""
    mirror_repo_modelscope: str = "licyks/mizuoto-model"
    """The mirror's repository on ModelScope, used when ``source`` is ``modelscope``."""


OutputFormat = Literal["wav", "flac", "mp3", "ogg", "m4a"]


class TtsSettings(Record):
    default_model: str = ""
    output_format: OutputFormat = "wav"
    output_sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    """None: the model's own rate."""
    normalize: Literal["none", "peak", "loudness"] = "peak"
    auto_transcribe: bool = True
    """Transcribe a reference clip without a transcript (needs the asr engine)."""
    asr_model: str = "asr/whisper-large-v3-turbo"
    reference_policy: Literal["trim", "reject"] = "trim"
    """What to do with a reference clip outside a model's length window: cut it to fit (the default) or refuse it."""
    max_text_chars: int = Field(default=20000, ge=100)
    strict_params: bool = False
    """Refuse a parameter the model does not take. Off (the default): drop it with a warning, so a parameter one model lacks does not stop a request."""
    model_aliases: dict[str, str] = Field(default_factory=dict)
    """Other names for models, for the OpenAI-compatible endpoint: ``{"tts-1": "qwen3tts/1.7b-base"}``."""
    micro_batch: bool = True
    """Requests waiting for the same model that differ only in their text run as one batch, on engines that take one."""


class EngineSettings(Record):
    enabled: bool = True
    accepted_licence: str | None = None
    """The licence id and version accepted for a licence-gated engine, such as ``fish-audio-research@2026-03-07``."""
    device: str = "auto"
    dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    load_options: dict[str, Any] = Field(default_factory=dict)
    """Engine-specific load options (fp16, IndexTTS's emotion model…), checked against the engine's manifest."""
    idle_minutes: int | None = Field(default=None, ge=0)
    """Overrides ``compute.unload_after_minutes`` for this engine."""
    pinned_models: list[str] = Field(default_factory=list)
    """Models never unloaded to make room."""
    vram_mode: VramMode = "auto"
    """Overrides ``compute.vram_mode`` for this engine's models."""
    attention: AttentionBackend | Literal["inherit"] = "inherit"
    """Overrides ``compute.attention`` for this engine's models."""


class TrainingSettings(Record):
    devices: list[str] = Field(default_factory=list)
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    memory: Literal["standard", "low", "lowest"] = "standard"
    """How a fine-tune spends GPU memory: ``standard`` follows each engine's recipe exactly; ``low`` adds gradient
    checkpointing (the same results, about a third slower); ``lowest`` also what changes results slightly. A method's
    own hyperparameters (``gradient_checkpointing``) win."""
    attention: TrainingAttention = "auto"
    """How trainers compute attention: ``auto`` follows each recipe exactly; ``fast`` or a backend's name uses faster
    kernels with a backward pass (results change slightly)."""
    keep_checkpoints: int = Field(default=3, ge=1)
    previews: bool = True


class OutputSettings(Record):
    naming: str = "{date}-{voice}-{model}"
    keep_days: int = Field(default=0, ge=0)
    """Delete outputs older than this (favourites are kept); 0 keeps everything."""


AGENT_PORT = 7870


class RemoteHost(Record):
    """Another machine running ``mizuoto agent``: engines whose device is ``<name>/<device>`` run there."""

    address: str
    """``host:port`` (the port defaults to 7870)."""
    token: str = ""
    """The agent's token (``remote.agent.token`` on that machine)."""
    tls: bool = False
    """Connect with TLS (the agent serves with ``remote.agent.ssl_certfile``)."""
    ca_file: str = ""
    """A CA bundle, or the agent's own certificate when it is self-signed; empty: the system's CAs."""


class AgentSettings(Record):
    """This machine as a remote host for another Mizuoto (``mizuoto agent``)."""

    host: str = "127.0.0.1"
    """The address to listen on: 0.0.0.0 for every interface."""
    port: int = Field(default=AGENT_PORT, ge=1, le=65535)
    token: str | None = None
    """Required: hosts connect with it. ``mizuoto agent`` makes one when there is none."""
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


class RemoteSettings(Record):
    hosts: dict[str, RemoteHost] = Field(default_factory=dict)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    connect_timeout: float = Field(default=10.0, ge=1.0, le=120.0)
    heartbeat_timeout: float = Field(default=60.0, ge=10.0, le=3600.0)
    """A remote request whose agent stays silent this long fails."""


class Settings(Record):
    """Everything saved in ``settings.toml``."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    engines: dict[str, EngineSettings] = Field(default_factory=dict)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)


# Public views: the access token never leaves the server.


class ServerSettingsView(Record):
    host: str
    port: int
    strict_port: bool
    open_browser: bool
    access_token_configured: bool
    allowed_origins: list[str]
    ssl: bool


class RemoteHostView(Record):
    address: str
    token_configured: bool
    tls: bool
    ca_file: str


class AgentSettingsView(Record):
    host: str
    port: int
    token_configured: bool
    ssl: bool


class RemoteSettingsView(Record):
    hosts: dict[str, RemoteHostView]
    agent: AgentSettingsView
    connect_timeout: float
    heartbeat_timeout: float


class ResolvedPaths(Record):
    """The folders in use, with the data-directory defaults filled in."""

    models_dir: str
    voices_dir: str
    outputs_dir: str
    datasets_dir: str
    experiments_dir: str
    projects_dir: str
    cache_dir: str
    browse_roots: list[str]


class SettingsView(Record):
    """Settings as returned to clients, with the token replaced by a flag."""

    data_dir: str
    settings_file: str
    server: ServerSettingsView
    paths: PathSettings
    resolved_paths: ResolvedPaths
    compute: ComputeSettings
    downloads: DownloadSettings
    tts: TtsSettings
    engines: dict[str, EngineSettings]
    training: TrainingSettings
    outputs: OutputSettings
    remote: RemoteSettingsView
    env_overrides: list[str]
    pinned: list[str]
