"""GPT-SoVITS v2, v2Pro, v2ProPlus, v4 and v5 Turbo, native.

The port follows Mizuoto's v1 adapter over upstream's ``TTS_infer_pack.TTS`` step for step:
- the reference (``set_ref_audio``): the clip at 16 kHz (upstream's port of resampy's ``kaiser_best``), 3–10 s,
  0.3 s of silence at the model's rate appended, HuBERT's last hidden state, the SoVITS quantizer's first codes:
  the prompt's semantic tokens. Then the linear spectrogram at the model's rate (torchaudio's resampler) and, for
  v2Pro, the speaker-verification embedding of that audio at 16 kHz; each auxiliary clip adds a spectrogram (and
  an embedding). v4/v5 also take the mel of the last clip read at 32 kHz as the flow's prompt.
- the text (``text.TextPreprocessor``): the cut method, language segmentation, g2p and BERT features; the
  reference transcript gets a full stop when it ends without punctuation;
- ``TTS.run``: length buckets, the T2S model (``t2s.Text2SemanticDecoder``: batched, or row by row without
  ``parallel_infer`` or without a transcript), SoVITS (one decode for a whole batch when ``parallel_infer`` and
  speed 1) or, for v4/v5, the flow and the vocoder; ``audio_postprocess``: each fragment limited to a peak of 1,
  followed by ``fragment_interval`` seconds of silence counted at 32 kHz whatever the output rate, and the whole
  truncated to 16 bits (``(x × 32768).astype(int16)``);
- streamed: v2 decodes the tokens chunk by chunk (``decode_streaming``, overlap, SOLA); v4/v5 send each fragment.

Randomness: upstream seeds the global generators at each ``run`` (one per Mizuoto segment), then draws the T2S
sampling and the SoVITS noise (or the flow's) in that order. Here both come from the request's generator, reseeded
per segment, so a seeded request repeats upstream's audio. Preparing a reference draws nothing (HuBERT's layer
drop is not replayed: v1 prepared the reference before seeding for the same reason).

Not native: the CUDA-graph T2S (upstream's ``t2s_accel``, used on a GPU), ``set_seed``'s process-wide TF32 switch,
super-sampling (v3 only). Errors raise instead of upstream's second of 16 kHz silence. In streaming mode upstream
decodes only the first text of each group of ``batch_size``; here every text is spoken.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec

if TYPE_CHECKING:
    import numpy as np
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask, VoiceSpec
    from mizuoto.runtime.handle import ModelHandle
    from mizuoto.runtime.runtime import Runtime

VERSIONS = {"v2": "v2", "v2pro": "v2Pro", "v2proplus": "v2ProPlus", "v4": "v4", "v5turbo": "v5turbo"}
"""Mizuoto variant → upstream version."""

BASE_WEIGHTS = {
    "v2": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "gsv-v2final-pretrained/s2G2333k.pth"),
    "v2pro": ("s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
    "v2proplus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
    "v4": ("s1v3.ckpt", "gsv-v4-pretrained/s2Gv4.pth"),
    "v5turbo": ("s1v3.ckpt", "gsv-v5-pretrained/s2Gv5turbo.pth"),
}
"""The pretrained (GPT, SoVITS) pair of each variant, in the model's folder (upstream's layout)."""

VOCODERS = {"v4": "gsv-v4-pretrained/vocoder.pth", "v5turbo": "gsv-v5-pretrained/vocoder.pth"}

UPSTREAM_SAMPLING_RATE = 32000
"""``audio_postprocess`` makes the pause between fragments at this rate whatever the output rate."""

LANGUAGES = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue"}
"""Mizuoto language → upstream ``text_lang``: the mixed modes (Chinese with English…) as upstream's UIs default to."""

ENCODERS = "gptsovits/encoders"
G2PW = "gptsovits/g2pw"
NLTK = "gptsovits/nltk-data"
OPEN_JTALK = "gptsovits/open-jtalk"
SV_FILE = "sv/pretrained_eres2netv2w24s4ep4.ckpt"

HZ = 50
"""``TTS_Config.hz``: semantic tokens per second (``early_stop_num = hz × max_sec``)."""
CODES_PER_PHONE = 16
"""Semantic codes a phone is assumed to need at most, for the memory estimate (speech runs at ~4–5 a phone)."""
MUTE_TOKEN = 486
"""``TTS_Config.mute_tokens``: the token streaming cuts after (every version's is 486)."""
CHUNK_SPLIT_THRESHOLD = 0.0
"""``TTS.run``'s ``chunk_split_thershold``."""

OPTIONS = ("text_split_method", "batch_size", "batch_threshold", "split_bucket", "parallel_infer", "overlap_length", "min_chunk_length", "fixed_length_chunk")
PARAMS = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("repetition_penalty", "repetition_penalty"),
    ("speed", "speed_factor"),
    ("num_steps", "sample_steps"),
)


def detect(model_dir: Path) -> str | None:
    for variant, (gpt, sovits) in BASE_WEIGHTS.items():
        if (model_dir / gpt).is_file() and (model_dir / sovits).is_file():
            return variant
    return None


class GptSovitsEngine(Engine):
    id = "gptsovits"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto.engines.gptsovits.train:GptSovitsTrainer"
    EXTRAS = {"mizuoto[gptsovits-ja]": ("pyopenjtalk",), "mizuoto[gptsovits-ko]": ("mecab", "jamo")}
    OPTION_SCHEMA = {
        "text_split_method": OptionSpec("enum", "cut1", choices=("cut0", "cut1", "cut2", "cut3", "cut4", "cut5")),
        "batch_size": OptionSpec("int", 1, 1, 200),
        "batch_threshold": OptionSpec("float", 0.75, 0.0, 1.0),
        "split_bucket": OptionSpec("bool", True),
        "parallel_infer": OptionSpec("bool", True),
        "overlap_length": OptionSpec("int", 2, 0, 20),
        "min_chunk_length": OptionSpec("int", 16, 1, 100),
        "fixed_length_chunk": OptionSpec("bool", False),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        return detect(model_dir)

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or spec.variant not in VERSIONS:
            raise EngineError(f"{spec.model_id} is not a GPT-SoVITS base model ({spec.variant})", code=codes.INVALID_INPUT)
        loaded = LoadedGptSovits(spec, ctx, compile=runtime.options.compile)
        loaded.cuda_graphs = runtime.options.cuda_graphs
        return loaded


@dataclass
class _Reference:
    """``prompt_cache``'s audio part for one main clip and its auxiliary clips."""

    key: tuple[str, tuple[str, ...]]
    semantic: torch.Tensor
    specs: list[tuple[torch.Tensor, torch.Tensor | None]]
    """(spectrogram, 16 kHz audio for v2Pro's speaker verification) per clip, the main one first."""
    raw: tuple[torch.Tensor, int]
    """The last clip read (``prompt_cache["raw_audio"]``, ``raw_sr``): v4/v5's mel prompt."""
    sv: list[torch.Tensor] | None = None


@dataclass
class _Prompt:
    text: str
    lang: str
    phones: list[int]
    bert: torch.Tensor
    norm_text: str


@dataclass
class _Run:
    """One ``TTS.run`` call's inputs, as the v1 adapter builds them."""

    text: str
    text_lang: str
    prompt: _Prompt | None
    reference: _Reference
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    text_split_method: str = "cut1"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed: float = 1.0
    fragment_interval: float = 0.3
    parallel_infer: bool = True
    sample_steps: int | None = None
    streaming: bool = False
    return_fragment: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    fixed_length_chunk: bool = False
    applied: dict[str, Any] = field(default_factory=dict)


class LoadedGptSovits(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext, compile: bool = False) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.gptsovits.sv import load_sv
        from mizuoto.engines.gptsovits.text import TextPreprocessor, load_bert_model
        from mizuoto.nn.ssl.hubert import load_hubert
        from mizuoto.runtime.handle import ModelHandle

        folder = spec.model_dir
        assert folder is not None
        self.folder = folder
        self.variant = spec.variant
        self.version = VERSIONS[spec.variant]
        self.device = torch.device(spec.device)
        self.half = self.device.type == "cuda" and spec.dtype in ("auto", "fp16")
        self.precision = torch.float16 if self.half else torch.float32
        self.compile = compile
        """``compute.compile``: v4/v5's CFM estimator is compiled (each SoVITS model loaded, a voice's own too)."""
        self.assets = assets = self._assets(spec)
        gpt, sovits = BASE_WEIGHTS[spec.variant]
        self.base = (str(folder / gpt), str(folder / sovits))
        self.current: tuple[str, str] = ("", "")
        self.switches = 0
        encoders = assets[ENCODERS]

        ctx.progress("loading:encoders", None)
        # Built on the CPU in their dtype; the model manager places them.
        self.hubert: Any = load_hubert(encoders / "chinese-hubert-base").to(self.precision).eval()
        self.bert = load_bert_model(encoders, "cpu", self.precision).eval()
        # The Japanese user dictionary is compiled once, beside the dictionary it extends.
        self.text = TextPreprocessor(self.bert, self.device, self.precision, assets, cache_dir=assets[OPEN_JTALK] / ".mizuoto")
        self.sv = load_sv(encoders / SV_FILE).to(self.precision).eval() if self.version in ("v2Pro", "v2ProPlus") else None

        ctx.progress("loading:weights", None)
        self.t2s: Any
        self.sovits: Any
        """``sovits.SynthesizerTrn`` (v2 family) or ``sovits_v3.SynthesizerTrnV3`` (v4, v5)."""
        self.vocoder: Any = None
        self.t2s, self.max_sec, self.mute_similarity = self._load_t2s(self.base[0])
        self.sovits, self.ckpt = self._load_sovits(self.base[1])
        self.current = self.base
        if self.uses_vocoder:
            from mizuoto.engines.gptsovits.vocoder import load_vocoder

            self.vocoder = load_vocoder(folder / VOCODERS[spec.variant], "cpu", self.precision)
        self._handles: dict[str, ModelHandle] = {}
        for name, module in (("hubert", self.hubert), ("bert", self.bert), ("sv", self.sv), ("t2s", self.t2s), ("sovits", self.sovits), ("vocoder", self.vocoder)):
            if module is not None:
                self._handles[name] = ModelHandle(module, name=f"{spec.model_id}:{name}", load_device=spec.device)
        self.handles = list(self._handles.values())
        self._reference: _Reference | None = None
        self._prompt: _Prompt | None = None
        self._ctx: RequestContext = ctx
        """The running request's context (its phases place the components)."""

    @staticmethod
    def _assets(spec: LoadSpec) -> dict[str, Path]:
        assets = {k: Path(v) for k, v in (spec.options.get("assets") or {}).items()}
        missing = [a for a in (ENCODERS, G2PW, NLTK, OPEN_JTALK) if a not in assets]
        if missing:
            raise EngineError(f"GPT-SoVITS needs {', '.join(missing)}", {"assets": missing}, code=codes.ASSET_MISSING)
        return assets

    @property
    def uses_vocoder(self) -> bool:
        return self.version in ("v4", "v5turbo")

    @property
    def sample_rate(self) -> int:
        return 48000 if self.uses_vocoder else int(self.ckpt.sampling_rate)

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=self.sample_rate, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return 1024 * 2**20

    def _use(self, *names: str, memory: int = 256 * 2**20) -> Any:
        """A phase over the named components (those this model has)."""
        return self._ctx.use([self._handles[n] for n in names if n in self._handles], memory)

    def _phase(self, name: str, names: tuple[str, ...], fn: Any, memory: int = 256 * 2**20, fallback: tuple[str, Any] | None = None) -> Any:
        """``fn()`` as a phase over the named components, run again if it runs out of memory (``ctx.run``)."""
        return self._ctx.run(name, [self._handles[n] for n in names if n in self._handles], fn, memory, fallback=fallback)

    def _t2s_memory(self, rows: int, positions: int | None = None) -> int:
        """The GPT's caches (24 layers, ~48 KB a token a row in fp16; grown by concatenation, so one layer's copy more at
        a step) for ``positions`` positions, else for the most it generates (``MAX_STEPS``), and working memory."""
        from mizuoto.engines.gptsovits.t2s import MAX_STEPS

        per_token = 48 * 1024 * (2 if self.half else 4) // 2
        return rows * (positions or MAX_STEPS) * per_token * 25 // 24 + 256 * 2**20

    def _t2s_positions(self, batch: dict[str, Any], run: _Run) -> int:
        """A batch's longest row: its phones, the reference's codes and the codes its phones are likely to need
        (``CODES_PER_PHONE`` each, within ``MAX_STEPS``). A longer generation that outgrows the device runs again
        with more weights streamed (``ctx.run``), with the same result."""
        from mizuoto.engines.gptsovits.t2s import MAX_STEPS

        phones = max(int(p.shape[-1]) for p in batch["all_phones"])
        prompt = int(run.reference.semantic.shape[-1]) if run.prompt is not None else 0
        return phones + prompt + min(MAX_STEPS, CODES_PER_PHONE * phones + 64)

    # -- weights --------------------------------------------------------------------------

    def _load_t2s(self, path: str) -> tuple[Any, int, torch.Tensor]:
        """``init_t2s_weights``: the model, its ``max_sec`` and the mute token's cosine similarity to every token."""
        import torch.nn.functional as F

        from mizuoto.engines.gptsovits.checkpoints import load_checkpoint
        from mizuoto.engines.gptsovits.t2s import T2SConfig, Text2SemanticDecoder, t2s_state
        from mizuoto.runtime.loading import load_into

        ckpt = load_checkpoint(path)
        if not isinstance(ckpt, dict) or "weight" not in ckpt or "config" not in ckpt:
            raise EngineError(f"{Path(path).name} is not a GPT-SoVITS GPT checkpoint", {"path": path}, code=codes.INVALID_INPUT)
        config = T2SConfig.from_checkpoint(ckpt["config"])
        model = Text2SemanticDecoder(config)
        load_into(model, t2s_state(ckpt["weight"]), what="GPT-SoVITS GPT model")
        model = model.to(self.precision).eval()
        model.graphs_for = lambda: self._t2s_graphs(model)
        # on the device, as upstream computes it there (the model was on it)
        codebook = model.ar_audio_embedding.word_embeddings.weight.detach().to(self.device, copy=True)
        mute = codebook[MUTE_TOKEN].unsqueeze(0)
        return model, int(config.max_sec), F.cosine_similarity(mute.float(), codebook.float(), dim=-1)

    cuda_graphs = False
    _graphs: Any = None
    _graph_signature: tuple[int, ...] | None = None

    def _t2s_graphs(self, model: Any) -> Any:
        """The GPT's CUDA graphs (``compute.cuda_graphs``) while ``model`` is the current one, wholly on an NVIDIA GPU:
        new ones when its weights moved (or another GPT was loaded) since the last capture, none otherwise."""
        from mizuoto.runtime.graphs import Graphs, signature, usable

        handle = self._handles.get("t2s")
        if not usable(self.device, self.cuda_graphs) or model is not self.t2s or handle is None or not handle.loaded:
            return None
        where = signature([model])
        if where != self._graph_signature:
            self._graphs, self._graph_signature = Graphs(self.device), where
        return self._graphs

    def _load_sovits(self, path: str) -> tuple[Any, Any]:
        """``init_vits_weights``: the model and its checkpoint (versions, spectrogram settings)."""
        from mizuoto.engines.gptsovits.checkpoints import read_sovits

        ckpt = read_sovits(path)
        if ckpt.model_version != self.version:
            raise EngineError(
                f"These SoVITS weights are {ckpt.model_version}, not {self.version}",
                {"version": ckpt.model_version, "expected": self.version},
                code=codes.INVALID_INPUT,
            )
        if self.uses_vocoder:
            from mizuoto.engines.gptsovits import sovits_v3

            model = sovits_v3.load_synthesizer(path, self.version, lora_base=self.base[1] if ckpt.lora else None, device="cpu", dtype=self.precision)
            if self.compile:
                from mizuoto.runtime.compile import Compiled

                model.cfm.compiled_estimator = Compiled(model.cfm.estimator, f"{self.spec.model_id}:cfm estimator ({Path(path).name})")
        else:
            from mizuoto.engines.gptsovits.sovits import build_synthesizer

            model = build_synthesizer(ckpt).to(self.precision).eval()
        return model, ckpt

    def _replace(self, name: str, module: torch.nn.Module) -> None:
        self._handles[name].replace(module)  # placed by the next phase that uses it
        setattr(self, name, module)

    def _switch(self, weights: dict[str, Any] | None, ctx: RequestContext) -> None:
        """Load a voice's own weights, or the base pair back, when they are not the ones loaded."""
        gpt = str((weights or {}).get("gpt") or self.base[0])
        sovits = str((weights or {}).get("sovits") or self.base[1])
        if (gpt, sovits) == self.current:
            return
        for path in (gpt, sovits):
            if not Path(path).is_file():
                raise EngineError(f"The voice's weights are missing: {path}", {"path": path}, code=codes.INVALID_INPUT)
        ctx.progress("loading:voice weights", None)
        if sovits != self.current[1]:
            model, ckpt = self._load_sovits(sovits)
            self._replace("sovits", model)
            self.ckpt = ckpt
            self._reference = None  # the prompt codes and spectrograms depend on the SoVITS model
            self.current = (self.current[0], sovits)
        if gpt != self.current[0]:
            model, self.max_sec, self.mute_similarity = self._load_t2s(gpt)
            self._replace("t2s", model)
            self.current = (gpt, self.current[1])
        self.switches += 1

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        if op == "switch_weights":
            self._switch(payload.get("weights"), ctx)
            return {"weights": {"gpt": self.current[0], "sovits": self.current[1]}, "version": self.ckpt.model_version}
        return super().call(op, payload, ctx)

    # -- the reference --------------------------------------------------------------------

    def _spec(self, path: str) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, int]]:
        """``_get_ref_spec``: the spectrogram at the model's rate, v2Pro's 16 kHz audio, and the clip as read."""
        from mizuoto.engines.gptsovits.audio import load, resample
        from mizuoto.engines.gptsovits.sovits import reference_spec

        raw, raw_sr = load(path)
        raw = raw.to(self.device).float()
        audio = raw.mean(0).unsqueeze(0) if raw.shape[0] == 2 else raw
        params = self.ckpt.spec_params
        rate = int(params["sampling_rate"])
        if raw_sr != rate:
            audio = resample(audio, raw_sr, rate)
        spec, audio = reference_spec(audio, **params)
        spec = spec.to(self.precision)
        audio16 = resample(audio, rate, 16000).to(self.precision) if self.ckpt.is_v2pro else None
        return spec, audio16, (raw, raw_sr)

    def _semantic(self, path: str) -> torch.Tensor:
        """``_set_prompt_semantic``: the prompt's semantic tokens."""
        import torch

        from mizuoto.engines.gptsovits.audio import load, resample

        raw, sr = load(path)
        wav16k = resample(raw.to(self.device).float().mean(0, keepdim=True), sr, 16000, match_librosa=True)
        if wav16k.shape[-1] > 160000 or wav16k.shape[-1] < 48000:
            raise EngineError("The reference clip must last 3 to 10 seconds", {"seconds": round(wav16k.shape[-1] / 16000, 2)}, code=codes.INVALID_INPUT)
        rate = int(self.ckpt.spec_params["sampling_rate"])
        zero = torch.zeros(int(rate * 0.3), dtype=self.precision, device=self.device)
        wav16k = torch.cat([wav16k.squeeze(0).to(self.precision), zero])
        hidden = self.hubert(wav16k.unsqueeze(0))[-1].transpose(1, 2)
        return self.sovits.extract_latent(hidden)[0, 0].to(self.device)

    def _prepare_reference(self, main: str, aux: list[str]) -> _Reference:
        key = (main, tuple(aux))
        if self._reference is not None and self._reference.key == key:
            return self._reference
        semantic = self._phase("reference", ("hubert", "sovits"), lambda: self._semantic(main))
        spec, audio16, raw = self._spec(main)
        specs = [(spec, audio16)]
        for path in aux:
            if not path or not Path(path).is_file():
                continue
            s, a, raw = self._spec(path)
            specs.append((s, a))
        sv = None
        if self.ckpt.is_v2pro:
            sv = self._phase("speaker verification", ("sv",), lambda: [self._sv_embedding(a) for _, a in specs])
        self._reference = _Reference(key, semantic, specs, raw, sv)
        return self._reference

    def _sv_embedding(self, audio16: torch.Tensor | None) -> torch.Tensor:
        from mizuoto.engines.gptsovits.sv import sv_embedding

        assert audio16 is not None and self.sv is not None
        return sv_embedding(self.sv, audio16)

    def _prepare_prompt(self, text: str, lang: str) -> _Prompt:
        from mizuoto.engines.gptsovits.text.segmentation import SPLITS

        text = text.strip("\n")
        if text[-1] not in SPLITS:
            text += "。" if lang != "en" else "."
        if self._prompt is not None and (self._prompt.text, self._prompt.lang) == (text, lang):
            return self._prompt
        phones, bert, norm_text = self.text.segment_and_extract_feature_for_text(text, lang, self.version)
        self._prompt = _Prompt(text, lang, list(phones), bert, norm_text)
        return self._prompt

    # -- speech ---------------------------------------------------------------------------

    def _inputs(self, task: SynthesisTask, text: str, ctx: RequestContext) -> _Run:
        voice: VoiceSpec = task.voice
        if not voice.references:
            raise EngineError("GPT-SoVITS clones a voice: it needs a reference clip", code=codes.INVALID_INPUT)
        ref = voice.references[0]
        transcript = (ref.transcript or "").strip()
        if not transcript and self.uses_vocoder:
            raise EngineError("GPT-SoVITS v4 and v5 need the reference clip's transcript", {"capability": "clone_no_transcript"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        p, o = task.params, task.options
        streaming = bool(task.stream)
        text_lang = LANGUAGES.get(task.language or "", "auto")
        prompt_lang = LANGUAGES.get(ref.language or task.language or "", "auto")
        aux = [r.audio_path for r in voice.references[1:]]
        applied: dict[str, Any] = {
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "seed": int(p["seed"]) if p.get("seed") is not None else -1,
            "streaming_mode": streaming and not self.uses_vocoder,
            "return_fragment": streaming and self.uses_vocoder,
        }
        for name, target in PARAMS:
            if p.get(name) is not None:
                applied[target] = p[name]
        if p.get("segment_silence_ms") is not None:
            # Upstream makes the pause at 32 kHz whatever the output rate (0.3 s is 0.2 s at 48 kHz); asked for in seconds.
            applied["fragment_interval"] = float(p["segment_silence_ms"]) / 1000 * self.sample_rate / UPSTREAM_SAMPLING_RATE
        for name in OPTIONS:
            if o.get(name) is not None:
                applied[name] = o[name]
        ctx.applied.update(applied)
        ctx.applied["references"] = 1 + len(aux)
        reference = self._prepare_reference(ref.audio_path, aux)
        prompt = self._prepare_prompt(transcript, prompt_lang) if transcript else None
        run = _Run(text=text, text_lang=text_lang, prompt=prompt, reference=reference, applied=applied)
        for key, attr in (
            ("top_k", "top_k"),
            ("top_p", "top_p"),
            ("temperature", "temperature"),
            ("repetition_penalty", "repetition_penalty"),
            ("speed_factor", "speed"),
            ("fragment_interval", "fragment_interval"),
            ("sample_steps", "sample_steps"),
            ("streaming_mode", "streaming"),
            ("return_fragment", "return_fragment"),
            *((name, name) for name in OPTIONS),
        ):
            if key in applied:
                setattr(run, attr, applied[key])
        return run

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        from mizuoto.engines.gptsovits.text import EmptyTextError, FrontendError
        from mizuoto.engines.gptsovits.text.g2pk import KoreanMissing
        from mizuoto.engines.gptsovits.text.japanese import OpenJTalkMissing

        try:
            yield from self._synthesize(task, ctx)
        except (OpenJTalkMissing, KoreanMissing) as exc:
            extra = "gptsovits-ja" if isinstance(exc, OpenJTalkMissing) else "gptsovits-ko"
            raise EngineError(
                f"GPT-SoVITS's {'Japanese' if extra.endswith('ja') else 'Korean'} text front end is not installed",
                {"capability": "language", "hint": f"pip install 'mizuoto[{extra}]'"},
                code=codes.CAPABILITY_NOT_SUPPORTED,
            ) from exc
        except EmptyTextError as exc:
            raise EngineError("GPT-SoVITS found nothing to say in the text", code=codes.INVALID_INPUT) from exc
        except FrontendError as exc:
            raise EngineError(str(exc) or "GPT-SoVITS cannot read this text", code=codes.INVALID_INPUT) from exc

    def _synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np

        self._ctx = ctx
        self.text.bert_scope = lambda: self._use("bert")  # the transcript's BERT features too
        self._switch(task.voice.weights, ctx)
        ctx.applied["voice_weights"] = bool(task.voice.weights)
        rate = self.sample_rate
        texts = list(task.texts) if task.texts else [task.text]
        for index, text in enumerate(texts):
            segment = task.segment + index if task.texts else task.segment
            run = self._inputs(task, text, ctx)
            ctx.reseed(ctx.seed)  # upstream's set_seed at every run
            pending: np.ndarray | None = None
            for audio in self._run(run, ctx):
                ctx.check_cancel()
                samples = np.asarray(audio, dtype=np.float32).reshape(-1) / 32768.0
                if pending is not None:
                    yield AudioChunk(pending, rate, segment)
                pending = samples
            ctx.check_cancel()
            if pending is None:
                raise EngineError("GPT-SoVITS produced no speech for the text", {"text": text[:200]}, code=codes.ENGINE_ERROR)
            yield AudioChunk(pending, rate, segment, final=True)

    # -- TTS.run --------------------------------------------------------------------------

    def _run(self, run: _Run, ctx: RequestContext) -> Iterator[np.ndarray]:
        """``TTS.run`` after the reference and the transcript are prepared: 16-bit audio pieces."""
        streaming, return_fragment, split_bucket, parallel = run.streaming, run.return_fragment, run.split_bucket, run.parallel_infer
        if streaming and self.uses_vocoder:
            streaming, return_fragment = False, True
        if streaming:
            parallel = False
        if return_fragment and streaming:
            return_fragment = False
        if return_fragment or streaming:
            split_bucket = False
        if run.speed != 1.0 or (self.uses_vocoder and parallel):
            split_bucket = False
        prompt = run.prompt
        if not (return_fragment or streaming):
            data = self.text.preprocess(run.text, run.text_lang, run.text_split_method, self.version)
            if not data:
                return
            batches, order = self._to_batch(data, prompt, run.batch_size, run.batch_threshold, split_bucket)
            audio: list[list[torch.Tensor]] = []
            for batch in batches:
                audio.append(self._fragments(batch, run, parallel, ctx))
            yield self._postprocess(audio, order if split_bucket else None, run.fragment_interval)
            return
        texts = self.text.pre_seg_text(run.text, run.text_lang, run.text_split_method)
        groups = [texts[i : i + run.batch_size] for i in range(0, len(texts), run.batch_size)]
        for group in groups:
            items = []
            for piece in group:
                phones, bert, norm_text = self.text.segment_and_extract_feature_for_text(piece, run.text_lang, self.version)
                if phones is None:
                    continue
                items.append({"phones": phones, "bert_features": bert, "norm_text": norm_text})
            if not items:
                continue
            if streaming:
                for item in items:
                    (batch,), _ = self._to_batch([item], prompt, 1, run.batch_threshold, False)
                    yield from self._stream(batch, run, ctx)
                    yield self._silence(run.fragment_interval, self.sample_rate)
                continue
            (batch,), _ = self._to_batch(items, prompt, run.batch_size, run.batch_threshold, False)
            yield self._postprocess([self._fragments(batch, run, parallel, ctx)], None, run.fragment_interval)

    def _to_batch(self, data: list[dict[str, Any]], prompt: _Prompt | None, batch_size: int, threshold: float, split_bucket: bool) -> tuple[list[dict[str, Any]], list[list[int]]]:
        """``TTS.to_batch``: buckets of similar normalised-text lengths, each item joined to the transcript's features."""
        import numpy as np
        import torch

        index_and_len = [[i, len(item["norm_text"])] for i, item in enumerate(data)]
        batch_index_list: list[list[int]] = []
        if split_bucket:
            index_and_len.sort(key=lambda x: x[1])
            arr = np.array(index_and_len, dtype=np.int64)
            pos = 0
            while pos < arr.shape[0]:
                pos_end = min(pos + batch_size, arr.shape[0])
                while pos < pos_end:
                    lengths = arr[pos:pos_end, 1].astype(np.float32)
                    score = lengths[(pos_end - pos) // 2] / (lengths.mean() + 1e-8)
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index_list.append(arr[pos:pos_end, 0].tolist())
                        pos = pos_end
                        break
                    pos_end = pos_end - 1
        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)
        batches = []
        for index_list in batch_index_list:
            phones_list, phones_len, all_phones, all_len, all_bert = [], [], [], [], []
            bert_max = phones_max = 0
            for item in (data[i] for i in index_list):
                if prompt is not None:
                    bert = torch.cat([prompt.bert, item["bert_features"]], 1).to(dtype=self.precision, device=self.device)
                    phones_all = torch.LongTensor(prompt.phones + list(item["phones"])).to(self.device)
                else:
                    bert = item["bert_features"].to(dtype=self.precision, device=self.device)
                    phones_all = torch.LongTensor(list(item["phones"])).to(self.device)
                phones = torch.LongTensor(list(item["phones"])).to(self.device)
                bert_max = max(bert_max, bert.shape[-1])
                phones_max = max(phones_max, phones_all.shape[-1])
                phones_list.append(phones)
                phones_len.append(phones.shape[-1])
                all_phones.append(phones_all)
                all_len.append(phones_all.shape[-1])
                all_bert.append(bert)
            batches.append(
                {
                    "phones": phones_list,
                    "phones_len": torch.LongTensor(phones_len).to(self.device),
                    "all_phones": all_phones,
                    "all_phones_len": torch.LongTensor(all_len).to(self.device),
                    "all_bert_features": all_bert,
                    "max_len": max(bert_max, phones_max),
                }
            )
        return batches, batch_index_list

    def _sampling(self, run: _Run) -> Any:
        from mizuoto.engines.gptsovits.t2s import Sampling

        s = Sampling()
        s.top_k, s.top_p, s.temperature, s.repetition_penalty = int(run.top_k), float(run.top_p), float(run.temperature), float(run.repetition_penalty)
        s.early_stop_num = HZ * self.max_sec
        return s

    def _semantic_tokens(self, batch: dict[str, Any], run: _Run, parallel: bool, ctx: RequestContext) -> tuple[list[torch.Tensor], list[int]]:
        """``infer_panel``: ``infer_panel_batch_infer`` with ``parallel_infer`` and a prompt, else row by row."""
        generator = ctx.generator_for(self.device)
        s = self._sampling(run)
        ids, bert = batch["all_phones"], batch["all_bert_features"]
        prompts = run.reference.semantic.expand(len(ids), -1).to(self.device) if run.prompt is not None else None
        if parallel and prompts is not None:
            out = self.t2s.infer_batch(ids, batch["all_phones_len"], prompts, bert, s, generator, max_len=batch["max_len"], check=ctx.check_cancel)
            self._report_graphs(ctx)
            return out
        tokens, stops = [], []
        for i in range(len(ids)):
            y, idx = next(
                self.t2s.infer_naive(ids[i].unsqueeze(0), prompts[i].unsqueeze(0) if prompts is not None else None, bert[i].unsqueeze(0), s, generator, check=ctx.check_cancel)
            )
            tokens.append(y[0])
            stops.append(int(idx))
        self._report_graphs(ctx)
        return tokens, stops

    def _report_graphs(self, ctx: RequestContext) -> None:
        if self.t2s.graphed and not self.t2s._graphs_failed:
            ctx.applied["cuda_graphs"] = True

    def _fragments(self, batch: dict[str, Any], run: _Run, parallel: bool, ctx: RequestContext) -> list[torch.Tensor]:
        """One batch: semantic tokens, then audio per item."""

        serial = ("one text at a time", lambda: self._semantic_tokens(batch, run, False, ctx)) if parallel and len(batch["all_phones"]) > 1 else None
        tokens, stops = self._phase(
            "gpt", ("t2s",), lambda: self._semantic_tokens(batch, run, parallel, ctx), self._t2s_memory(len(batch["all_phones"]), self._t2s_positions(batch, run)), serial
        )
        ctx.check_cancel()
        apart = ("one text at a time", lambda: self._decode_fragments(tokens, stops, batch, run, False, ctx)) if parallel and len(stops) > 1 else None
        return self._phase("sovits", ("sovits", "vocoder"), lambda: self._decode_fragments(tokens, stops, batch, run, parallel, ctx), 1024 * 2**20, apart)

    def _decode_fragments(self, tokens: list[torch.Tensor], stops: list[int], batch: dict[str, Any], run: _Run, parallel: bool, ctx: RequestContext) -> list[torch.Tensor]:
        import torch

        generator = ctx.generator_for(self.device)
        phones = batch["phones"]
        if self.uses_vocoder:
            return self._vocoder_fragments(tokens, stops, phones, run, parallel, ctx)
        ref = run.reference
        specs = [spec.to(dtype=self.precision, device=self.device) for spec, _ in ref.specs]
        sv = ref.sv
        if run.speed == 1.0 and parallel:
            pieces = [t[-idx:] for t, idx in zip(tokens, stops, strict=True)]
            upsample = math.prod(self.sovits.upsample_rates)
            ends = [0]
            for piece in pieces:
                ends.append(ends[-1] + piece.shape[0] * 2 * upsample)
            codes = torch.cat(pieces).unsqueeze(0).unsqueeze(0).to(self.device)
            joined = torch.cat(phones).unsqueeze(0).to(self.device)
            audio = self.sovits.decode(codes, joined, specs, speed=run.speed, sv_emb=sv, generator=generator).detach()[0, 0, :]
            return [audio[ends[i - 1] : ends[i]] for i in range(1, len(ends))]
        out = []
        for i, idx in enumerate(stops):
            ctx.check_cancel()
            codes = tokens[i][-idx:].unsqueeze(0).unsqueeze(0)
            out.append(self.sovits.decode(codes, phones[i].unsqueeze(0).to(self.device), specs, speed=run.speed, sv_emb=sv, generator=generator).detach()[0, 0, :])
        return out

    def _flow_prompt(self, run: _Run) -> Any:
        import torch

        from mizuoto.engines.gptsovits import sovits_v3
        from mizuoto.engines.gptsovits.audio import resample

        assert run.prompt is not None
        raw, raw_sr = run.reference.raw
        audio = raw.to(self.device).float()
        if raw_sr != 32000:
            audio = resample(audio.mean(0).unsqueeze(0) if audio.shape[0] == 2 else audio, raw_sr, 32000)
        return sovits_v3.Prompt(run.reference.semantic, torch.LongTensor(run.prompt.phones), run.reference.specs[0][0], sovits_v3.reference_mel(audio))

    def _vocoder_fragments(self, tokens: list[torch.Tensor], stops: list[int], phones: list[torch.Tensor], run: _Run, parallel: bool, ctx: RequestContext) -> list[torch.Tensor]:
        from mizuoto.engines.gptsovits import sovits_v3

        generator = ctx.generator_for(self.device)
        prompt = self._flow_prompt(run)

        def step(_i: int, _n: int) -> None:
            ctx.check_cancel()

        if parallel:
            return sovits_v3.vocoder_synthesis_batched(
                self.sovits, self.vocoder, prompt, stops, tokens, phones, speed=run.speed, sample_steps=run.sample_steps, generator=generator, step=step
            )
        out = []
        for i, idx in enumerate(stops):
            codes = tokens[i][-idx:].unsqueeze(0).unsqueeze(0)
            out.append(
                sovits_v3.vocoder_synthesis(
                    self.sovits, self.vocoder, prompt, codes, phones[i].unsqueeze(0).to(self.device), speed=run.speed, sample_steps=run.sample_steps, generator=generator, step=step
                )
            )
        return out

    def _stream(self, batch: dict[str, Any], run: _Run, ctx: RequestContext) -> Iterator[np.ndarray]:
        """``TTS.run``'s streaming mode for one text: token chunks decoded with an overlap and joined by SOLA."""
        with self._use("t2s", "sovits", memory=self._t2s_memory(1) + 512 * 2**20):
            yield from self._stream_chunks(batch, run, ctx)

    def _stream_chunks(self, batch: dict[str, Any], run: _Run, ctx: RequestContext) -> Iterator[np.ndarray]:
        from mizuoto.engines.gptsovits.sovits_v3 import sola_algorithm

        generator = ctx.generator_for(self.device)
        ref = run.reference
        prompts = ref.semantic.expand(1, -1).to(self.device) if run.prompt is not None else None
        chunks = self.t2s.infer_naive(
            batch["all_phones"][0].unsqueeze(0),
            prompts,
            batch["all_bert_features"][0].unsqueeze(0),
            self._sampling(run),
            generator,
            streaming=True,
            chunk_length=int(run.min_chunk_length),
            mute_emb_sim_matrix=self.mute_similarity if not run.fixed_length_chunk else None,
            chunk_split_threshold=CHUNK_SPLIT_THRESHOLD,
            check=ctx.check_cancel,
        )
        phones = batch["phones"][0].unsqueeze(0).to(self.device)
        specs = [spec.to(dtype=self.precision, device=self.device) for spec, _ in ref.specs]
        frames_per_token = 2 if self.sovits.semantic_frame_rate == "25hz" else 1
        upsample = math.prod(self.sovits.upsample_rates) * (frames_per_token / run.speed)
        overlap_length = int(run.overlap_length)
        overlap_size = math.ceil(overlap_length * upsample)
        first = True
        last_audio = last_latent = None
        previous: list[torch.Tensor] = []
        import torch

        for tokens, is_final in chunks:
            if is_final:
                self._report_graphs(ctx)
            if tokens is None and last_audio is not None:
                yield self._postprocess([[last_audio[-overlap_size:]]], None, 0.0)
                break
            assert tokens is not None
            previous.append(tokens)
            joined = torch.cat(previous, dim=-1)
            overlap_len = overlap_length + (10 - tokens.shape[-1]) if not first and tokens.shape[-1] < 10 else overlap_length
            audio, latent, _ = self.sovits.decode_streaming(
                joined.unsqueeze(0),
                phones,
                specs,
                speed=run.speed,
                sv_emb=ref.sv,
                result_length=tokens.shape[-1] + overlap_len if not first else None,
                overlap_frames=last_latent[:, :, -overlap_len * frames_per_token :] if last_latent is not None else None,
                padding_length=0,
                generator=generator,
            )
            audio = audio.detach()[0, 0, :]
            if overlap_len > overlap_length:
                audio = audio[-int((overlap_length + tokens.shape[-1]) * upsample) :]
            piece = audio
            if first and not is_final:
                first = False
                piece = piece[:-overlap_size]
            elif first and is_final:
                first = False
            elif not first and not is_final:
                assert last_audio is not None
                piece = sola_algorithm([last_audio, piece], overlap_size)
                piece = piece[last_audio.shape[0] - overlap_size : -overlap_size]
            last_latent = latent
            last_audio = audio
            yield self._postprocess([[piece]], None, 0.0)

    # -- audio_postprocess ----------------------------------------------------------------

    def _silence(self, seconds: float, rate: int) -> np.ndarray:
        import numpy as np

        return np.zeros(int(rate * seconds), dtype=np.int16)

    def _postprocess(self, audio: list[list[torch.Tensor]], order: list[list[int]] | None, fragment_interval: float) -> np.ndarray:
        """``audio_postprocess`` (``speed_change`` is commented out upstream)."""
        import numpy as np
        import torch

        zero = torch.zeros(int(UPSTREAM_SAMPLING_RATE * fragment_interval), dtype=self.precision, device=self.device) if fragment_interval > 0 else None
        for i, batch in enumerate(audio):
            for j, fragment in enumerate(batch):
                peak = torch.abs(fragment).max()
                if peak > 1:
                    fragment = fragment / peak
                audio[i][j] = torch.cat([fragment, zero], dim=0) if zero is not None else fragment
        if order is not None:
            flat: list[Any] = [None] * sum(len(b) for b in order)
            for i, index_list in enumerate(order):
                for j, index in enumerate(index_list):
                    flat[index] = audio[i][j]
        else:
            flat = [fragment for batch in audio for fragment in batch]
        joined = torch.cat(flat, dim=0).cpu().numpy()
        return (joined * 32768).astype(np.int16)
