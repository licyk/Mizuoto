"""IndexTTS, native: IndexTTS-1.5, 2 and 2.5 — cloning; with 2 and 2.5 also emotion from a
clip, weights or text.

The request follows upstream's ``IndexTTS2.infer`` as Mizuoto's v1 adapter called it:
- the speaker clip is read as ``librosa.load`` reads it (22.05 kHz, soxr, at most 15 s); w2v-BERT's hidden state
  17 of its 16 kHz copy (normalised by ``wav2vec2bert_stats.pt``) conditions the GPT, its semantic codes the
  decoder's prompt, its mel the prompt mels and CAM++ the style; all of it is the voice's cache;
- the emotion is the clip's (or another clip's, mixed by ``emo_alpha``), or weights over eight emotions mixed
  into it, the matrix rows picked by CAM++ similarity;
- text is normalised, tokenised and split; each segment is beam-sampled by the GPT, re-read for its latents,
  turned into mels by the DiT (25 Euler steps, CFG 0.7) and into audio by BigVGAN; segments are joined with
  silence, the tail faded, and the result quantised to 16-bit as upstream returns it.
Every random draw comes from the request's generator, in upstream's order (GPT, then the DiT's noise).

IndexTTS-2.5 (``infer_v2_5.py``) conditions the GPT on the projected CAM++ embedding and a language embedding
instead of w2v-BERT's Conformer and perceiver, reads its text through ``text25`` (tiktoken), and turns codes
back into features with its own codec (``codec.EnhancedCodec``) instead of the GPT's latents; ``speed`` scales
the length (``duration_factor = 1 / speed``).

IndexTTS-1.5 (``infer.py``, ``LoadedIndexTts15``) has no emotion: the reference's log-mel conditions the GPT and
BigVGAN v1 (with its ECAPA speaker encoder) turns the GPT's latents into audio.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import StyleSpec, SynthesisTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

RATES = {"v1_5": 24000, "v2": 22050, "v2_5": 22050}
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
"""IndexTTS-2's emotion vector, in its order."""
ASSETS = {
    "v2": ("indextts/w2v-bert", "indextts/semantic-codec", "indextts/campplus", "indextts/bigvgan"),
    "v2_5": ("indextts/w2v-bert", "indextts/campplus", "indextts/bigvgan"),
}
MAX_REFERENCE_SECONDS = 15
W2V_LAYER = 17
"""The w2v-BERT hidden state IndexTTS-2 and 2.5 read (``hidden_states[17]``)."""
CACHE_FORMAT_VERSION = 2
"""2: IndexTTS-1.5's reference mel computed on the CPU (before, on the device: other bits on a GPU)."""
TAIL_FADE_MS = 20.0
CODES_PER_TOKEN = 24
"""Codes a text token is assumed to need at most, for the memory estimate (speech runs at ~5 a token)."""
PROMPT_POSITIONS = 64
"""The GPT prompt's positions besides the text (the conditioning latents and the start tokens)."""


def detect(model_dir: Path) -> str | None:
    if (model_dir / "codec.pth").is_file():
        return "v2_5"
    if (model_dir / "s2mel.pth").is_file():
        return "v2"
    if (model_dir / "bigvgan_generator.pth").is_file() and (model_dir / "gpt.pth").is_file():
        return "v1_5"
    return None


def guess_language(text: str) -> str:
    """IndexTTS-2.5's language for a request without one (or with one it lacks), from the text's script: kana → ja,
    Han → zh, letters only Spanish has → es, otherwise en. Upstream sends "zhen" there, which is no language token of
    the model (the tokenizer spells ``<|zhen|>`` out and ``lang_to_token`` falls back to ``common``); judged by T4 on
    the 2.5 clone set it made English 10× and Chinese 2–3× less intelligible than the language itself."""
    if any(0x3040 <= ord(ch) <= 0x30FF for ch in text):
        return "ja"
    if any(0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF for ch in text):
        return "zh"
    if any(ch in "ñÑ¿¡" for ch in text):
        return "es"
    return "en"


def norm_on(sd: dict[str, Any], device: Any) -> dict[str, Any]:
    """``sd`` with its weight-norm pairs (``weight_g``/``weight_v``, ``parametrizations.weight.original0/1``) on ``device``,
    so ``fold_weight_norm`` computes each norm there: upstream removes BigVGAN's weight norm after moving it to the
    GPU and computes the codec's and the CFM's there at every call, so on a GPU the GPU's reduction gives their bits."""
    pairs = ("weight_g", "weight_v", "original0", "original1")
    return {k: v.to(device) if k.rpartition(".")[2] in pairs else v for k, v in sd.items()}


class IndexTtsEngine(Engine):
    id = "indextts"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    EXTRAS = {"mizuoto-indextts[ja]": ("fugashi", "unidic_lite")}
    OPTION_SCHEMA = {
        "num_beams": OptionSpec("int", 3, 1, 10),
        "length_penalty": OptionSpec("float", 0.0, -2.0, 2.0),
        "max_text_tokens_per_segment": OptionSpec("int", 120, 20, 600),
        "use_random": OptionSpec("bool", False),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        return detect(model_dir)

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or detect(spec.model_dir) is None:
            raise EngineError(f"{spec.model_id} is not an IndexTTS model", code=codes.INVALID_INPUT)
        # compute.compile, or v1's use_torch_compile for this model: s2mel's DiT step and BigVGAN compiled
        compile = runtime.options.compile or bool(spec.options.get("use_torch_compile"))
        loaded: Any = LoadedIndexTts15(spec, ctx, compile=compile) if spec.variant == "v1_5" else LoadedIndexTts2(spec, ctx, compile=compile)
        loaded.gpt.graphs_for = _GptGraphs(loaded, runtime.options.cuda_graphs)
        return loaded


class _GptGraphs:
    """The GPT's CUDA graphs (``compute.cuda_graphs``) while it is wholly on an NVIDIA GPU: new ones when its weights
    moved since the last capture, none otherwise."""

    def __init__(self, loaded: Any, enabled: bool) -> None:
        self.loaded, self.enabled = loaded, enabled
        self.graphs: Any = None
        self.signature: tuple[int, ...] | None = None

    def __call__(self) -> Any:
        from mizuoto.runtime.graphs import Graphs, signature, usable

        loaded = self.loaded
        if not usable(loaded.device, self.enabled) or not loaded.h_gpt.loaded:
            return None
        where = signature([loaded.gpt])
        if where != self.signature:
            self.graphs, self.signature = Graphs(loaded.device), where
        return self.graphs


@dataclass
class SpeakerCond:
    """What upstream caches for its speaker clip."""

    spk_cond_emb: torch.Tensor
    prompt_condition: torch.Tensor
    ref_mel: torch.Tensor
    style: torch.Tensor
    emo_cond_emb: torch.Tensor
    """The clip's emotion embedding: upstream reads the clip again at 16 kHz for it (not the 22.05 kHz copy resampled)."""


SPEAKER_FIELDS = ("spk_cond_emb", "prompt_condition", "ref_mel", "style", "emo_cond_emb")


def fade_out_tail(wav: torch.Tensor, rate: int, fade_ms: float = TAIL_FADE_MS) -> torch.Tensor:
    import torch

    if wav.numel() == 0 or fade_ms <= 0:
        return wav
    n = min(int(rate * fade_ms / 1000.0), wav.shape[-1])
    if n <= 1:
        return wav
    ramp = 0.5 * (1.0 + torch.cos(torch.linspace(0.0, math.pi, n, device=wav.device)))
    out = wav.clone()
    out[..., -n:] = (out[..., -n:].to(torch.float32) * ramp).to(wav.dtype)
    return out


def load_audio(path: str, sr: int | None, max_seconds: float = MAX_REFERENCE_SECONDS) -> tuple[torch.Tensor, int]:
    """``_load_and_cut_audio``: ``librosa.load`` (22.05 kHz by default, soxr), mono, cut to ``max_seconds``."""
    import torch

    from mizuoto.nn.dsp.resample import soxr_resample
    from mizuoto.nn.dsp.wav import read_wav

    samples, rate = read_wav(path)
    target = sr or 22050
    if rate != target:
        samples = soxr_resample(samples, rate, target)
    audio = torch.tensor(samples).unsqueeze(0)
    limit = int(max_seconds * target)
    return (audio[:, :limit] if audio.shape[1] > limit else audio), target


def bigvgan_memory(samples: int, per_sample: int, fixed_mb: int = 256) -> int:
    """BigVGAN's activations for ``samples`` of output: ``per_sample`` bytes each and ``fixed_mb``, measured on an RTX
    3090 from about 4 to 30 s of speech (IndexTTS 2: 1,057 bytes; 1.5: 771 bytes and 256 MB, each segment decoded
    alone), with a fifth more."""
    return samples * per_sample + fixed_mb * 2**20


def gpt_memory(gpt: Any, beams: int, text_tokens: int, max_new_tokens: int, half: bool, codes: int | None = None) -> int:
    """The GPT's beam caches (each beam's keys and values per layer, twice: ``BeamCaches`` copies them into its other
    buffer each step) for the prompt and ``codes`` codes, else those its text is likely to need (``CODES_PER_TOKEN`` a
    token, within ``max_new_tokens``), in the dtype it computes in (fp16 under autocast), and working memory, which also
    covers the buffers' room to grow (``BeamCaches.ROOM`` positions each: about 190 MB for IndexTTS 2 in float32 with
    three beams). A generation that runs
    longer still runs: if its caches then outgrow the device, the phase runs again with more weights streamed."""
    layers, width = len(gpt.gpt.h), gpt.gpt.h[0].attn.dim
    new = codes if codes is not None else min(max_new_tokens, CODES_PER_TOKEN * text_tokens + 64)
    positions = PROMPT_POSITIONS + text_tokens + new
    return 2 * beams * 2 * layers * positions * width * (2 if half else 4) + 512 * 2**20


class LoadedIndexTts2(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext, compile: bool = False) -> None:
        super().__init__(spec)
        import torch
        import yaml

        from mizuoto.nn.codecs.repcodec import IGNORED as CODEC_IGNORED
        from mizuoto.nn.codecs.repcodec import RepCodec
        from mizuoto.nn.dsp.mel import matcha_basis
        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.speaker.campplus_torch import CAMPPlus
        from mizuoto.nn.ssl.w2vbert import W2VBert
        from mizuoto.nn.transformer.seedvc import CFM, DiTConfig, InterpolateRegulator
        from mizuoto.nn.vocoders.bigvgan import BigVGAN
        from mizuoto.runtime import ops
        from mizuoto.runtime.compile import compile_module
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import fold_weight_norm, load_into, load_state_dict, unwrap
        from mizuoto_indextts.codec import EnhancedCodec
        from mizuoto_indextts.gpt import GptConfig, UnifiedVoice
        from mizuoto_indextts.text import TextNormalizer, TextTokenizer
        from mizuoto_indextts.text25 import Frontend25

        folder = spec.model_dir
        assert folder is not None
        self.variant = spec.variant
        v25 = self.variant == "v2_5"
        assets = dict(spec.options.get("assets") or {})
        missing = [a for a in ASSETS[self.variant] if a not in assets]
        if missing:
            raise EngineError(f"IndexTTS {spec.variant} needs {', '.join(missing)}", {"assets": missing}, code=codes.ASSET_MISSING)
        self.device = torch.device(spec.device)
        cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        self.cfg = cfg

        w2v_dir = Path(assets["indextts/w2v-bert"])
        w2v_cfg = json.loads((w2v_dir / "config.json").read_text(encoding="utf-8"))
        semantic_dim = int(w2v_cfg["hidden_size"])
        ctx.progress("loading:gpt", None)
        gpt = UnifiedVoice(GptConfig.from_yaml(cfg["gpt"], "campplus" if v25 else "conformer", semantic_dim), ops.default)
        load_into(gpt, unwrap(load_state_dict(folder / cfg["gpt_checkpoint"]), "model"), what="IndexTTS GPT")
        self.gpt = gpt.eval()
        # as v1 chose: IndexTTS-2 in fp16 only when asked, 2.5 in bf16 on a GPU by default; float32 on the CPU
        cuda = self.device.type == "cuda"
        self.half: torch.dtype | None = None
        if not v25 and cuda and spec.dtype == "fp16":
            self.half = torch.float16
        elif v25 and cuda and spec.dtype in ("auto", "bf16"):
            self.half = torch.bfloat16
        if self.half is not None:
            self.gpt.to(dtype=self.half)

        ctx.check_cancel()
        ctx.progress("loading:w2v-bert", None)
        w2v = W2VBert(w2v_cfg, layers=W2V_LAYER)
        load_into(w2v, load_state_dict(w2v_dir / "model.safetensors"), ignore=W2VBert.ignored(w2v_cfg, W2V_LAYER), what="w2v-BERT")
        self.w2v = w2v.eval()
        stats = torch.load(str(folder / cfg["w2v_stat"]), map_location="cpu", weights_only=True)
        self.semantic_mean = stats["mean"]
        self.semantic_std = torch.sqrt(stats["var"])

        ctx.progress("loading:semantic codec", None)
        codec: Any
        if v25:
            codec = EnhancedCodec.from_config(cfg["semantic_codec"])
            load_into(codec, fold_weight_norm(norm_on(load_state_dict(folder / "codec.pth")["model"], spec.device)), what="IndexTTS-2.5 codec")
        else:
            codec = RepCodec.from_config(cfg["semantic_codec"])
            load_into(
                codec,
                fold_weight_norm(norm_on(load_state_dict(Path(assets["indextts/semantic-codec"]) / "model.safetensors"), spec.device)),
                ignore=CODEC_IGNORED,
                what="semantic codec",
            )
        self.semantic_codec: Any = codec.eval()

        ctx.check_cancel()
        ctx.progress("loading:s2mel", None)
        net = load_state_dict(folder / cfg["s2mel_checkpoint"])["net"]
        s2 = cfg["s2mel"]
        cfm = CFM(DiTConfig.from_s2mel(s2))
        load_into(cfm, fold_weight_norm(norm_on(_strip_module(net["cfm"]), spec.device)), what="s2mel CFM")
        lr = s2["length_regulator"]
        regulator = InterpolateRegulator(lr["channels"], lr["sampling_ratios"], lr["in_channels"], lr["content_codebook_size"])
        load_into(regulator, _strip_module(net["length_regulator"]), what="length regulator")
        self.gpt_layer: torch.nn.Module | None = None
        if not v25:  # 2.5's checkpoint carries one too, unused (upstream builds its s2mel without it)
            # upstream: Linear(1280, 256), Linear(256, 128), Linear(128, 1024) — the GPT's width to w2v-BERT's
            gpt_layer = torch.nn.Sequential(ops.default.Linear(gpt.c.model_dim, 256), ops.default.Linear(256, 128), ops.default.Linear(128, semantic_dim))
            load_into(gpt_layer, _strip_module(net["gpt_layer"]), what="gpt layer")
            self.gpt_layer = gpt_layer.eval()
        del net
        self.cfm = cfm.eval()
        self.regulator: Any = regulator.eval()

        ctx.progress("loading:campplus", None)
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        load_into(campplus, load_state_dict(Path(assets["indextts/campplus"]) / "campplus_cn_common.bin"), what="CAM++")
        self.campplus = campplus.eval()

        ctx.progress("loading:bigvgan", None)
        bv_dir = Path(assets["indextts/bigvgan"])
        bigvgan = BigVGAN(json.loads((bv_dir / "config.json").read_text(encoding="utf-8")))
        load_into(bigvgan, fold_weight_norm(norm_on(load_state_dict(bv_dir / "bigvgan_generator.pt")["generator"], spec.device)), what="BigVGAN")
        self.bigvgan = bigvgan.eval()
        self.cfm.compiled_estimator = compile_module(self.cfm.estimator, compile, f"{spec.model_id}:s2mel DiT")
        self.compiled_bigvgan = compile_module(self.bigvgan, compile, f"{spec.model_id}:bigvgan")

        glossary = None
        if (folder / "glossary.yaml").is_file():
            loaded = yaml.safe_load((folder / "glossary.yaml").read_text(encoding="utf-8"))
            glossary = loaded if isinstance(loaded, dict) else None
        normalizer = TextNormalizer(glossary)
        self.tokenizer: TextTokenizer | None = None
        self.frontend: Frontend25 | None = None
        if v25:
            self.frontend = Frontend25(folder, normalizer)
        else:
            self.tokenizer = TextTokenizer(folder / cfg["dataset"]["bpe_model"], normalizer)

        emo_num = list(cfg["emo_num"])
        self.emo_matrix = torch.split(torch.load(str(folder / cfg["emo_matrix"]), map_location="cpu", weights_only=True), emo_num)
        self.spk_matrix = torch.split(torch.load(str(folder / cfg["spk_matrix"]), map_location="cpu", weights_only=True), emo_num)
        self.emo_num = emo_num

        sp = s2["preprocess_params"]["spect_params"]
        fmax = sp.get("fmax", "None")
        self.mel_args: dict[str, Any] = {
            "n_fft": sp["n_fft"],
            "num_mels": sp["n_mels"],
            "sampling_rate": s2["preprocess_params"]["sr"],
            "hop_size": sp["hop_length"],
            "win_size": sp["win_length"],
            "fmin": sp.get("fmin", 0),
            "fmax": None if fmax == "None" else 8000,
        }
        self.mel_basis = matcha_basis(self.mel_args["sampling_rate"], sp["n_fft"], sp["n_mels"], self.mel_args["fmin"], self.mel_args["fmax"])
        self.resample_16k = Resample(22050, 16000)
        self.stop_mel_token = int(cfg["gpt"]["stop_mel_token"])
        self.qwen_emo: Any = None
        if spec.options.get("use_qwen_emo", True) and cfg.get("qwen_emo_path") and (folder / cfg["qwen_emo_path"]).is_dir():
            from mizuoto_indextts.qwen_emotion import QwenEmotion

            ctx.progress("loading:emotion model", None)
            self.qwen_emo = QwenEmotion(folder / cfg["qwen_emo_path"], self.device)
        name = spec.model_id
        # phases: w2v-BERT, CAM++ (and the codec, the regulator) condition on a voice that is not
        # cached or an emotion clip; the emotion model reads an emotion text; the GPT samples codes; the s2mel turns them
        # into a mel; BigVGAN speaks it
        self.h_gpt = ModelHandle(self.gpt, name=f"{name}:gpt", load_device=spec.device, kind="sequence")
        self.h_w2v = ModelHandle(self.w2v, name=f"{name}:w2v-bert", load_device=spec.device)
        self.h_codec = ModelHandle(self.semantic_codec, name=f"{name}:semantic codec", load_device=spec.device)
        self.h_s2mel = ModelHandle(torch.nn.ModuleList([m for m in (self.cfm, self.regulator, self.gpt_layer) if m is not None]), name=f"{name}:s2mel", load_device=spec.device)
        self.h_campplus = ModelHandle(self.campplus, name=f"{name}:campplus", load_device=spec.device)
        self.h_bigvgan = ModelHandle(self.bigvgan, name=f"{name}:bigvgan", load_device=spec.device)
        self.handles = [self.h_gpt, self.h_w2v, self.h_codec, self.h_s2mel, self.h_campplus, self.h_bigvgan]
        self.h_emotion: ModelHandle | None = None
        if self.qwen_emo is not None:
            self.h_emotion = ModelHandle(torch.nn.ModuleList(self.qwen_emo.modules()), name=f"{name}:emotion model", load_device=spec.device, kind="sequence")
            self.handles.append(self.h_emotion)

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=RATES[self.variant], vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        # the DiT's attention over prompt + output mels (up to ~3000 frames, batch 2, 8 heads, float32) dominates
        return 1024 * 2**20 + 2 * 8 * 3000 * 3000 * 4

    def _gpt_memory(self, sampling: Any, text_tokens: int, codes: int | None = None) -> int:
        return gpt_memory(self.gpt, int(sampling.num_beams), text_tokens, int(sampling.max_new_tokens), self.half is not None, codes)

    def _conditioning_memory(self) -> int:
        """w2v-BERT and CAM++ over a clip of up to ~30 s."""
        return 512 * 2**20

    # -- conditioning ---------------------------------------------------------------------------------

    def _emb(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """``get_emb``: w2v-BERT's hidden state 17, normalised."""
        import torch

        from mizuoto.nn.dsp.seamless import features

        feats, mask = features(audio_16k.numpy())
        feats_t, mask_t = torch.from_numpy(feats).to(self.device), torch.from_numpy(mask).to(self.device)
        feat = self.w2v.hidden_states(feats_t, mask_t, W2V_LAYER)
        return (feat - self.semantic_mean.to(feat.device)) / self.semantic_std.to(feat.device)

    def _speaker(self, path: str, ctx: RequestContext) -> SpeakerCond:
        audio, _ = load_audio(path, None)
        audio_16k = self.resample_16k(audio)
        handles = [self.h_w2v, self.h_campplus, self.h_s2mel] + ([] if self.variant == "v2_5" else [self.h_codec])
        return ctx.run("conditioning", handles, lambda: self._condition(path, audio, audio_16k, ctx), self._conditioning_memory())

    def _condition(self, path: str, audio: torch.Tensor, audio_16k: torch.Tensor, ctx: RequestContext) -> SpeakerCond:
        import torch

        from mizuoto.nn.dsp.kaldi import fbank
        from mizuoto.nn.dsp.mel import matcha_mel

        spk_cond_emb = self._emb(audio_16k)
        a = self.mel_args
        ref_mel = matcha_mel(audio.to(self.device).float(), a["n_fft"], a["num_mels"], a["sampling_rate"], a["hop_size"], a["win_size"], a["fmin"], a["fmax"], basis=self.mel_basis)
        lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)
        feat = fbank(audio_16k.to(self.device), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus(feat.unsqueeze(0))
        if self.variant == "v2_5":  # the features themselves, not their codes
            prompt_condition = self.regulator(spk_cond_emb, lengths)
        else:
            _, s_ref = self.semantic_codec.quantize(spk_cond_emb)
            prompt_condition = self.regulator(s_ref, lengths)
        return SpeakerCond(spk_cond_emb, prompt_condition, ref_mel, style, self._emotion_emb(path, ctx))

    def _emotion_emb(self, path: str, ctx: RequestContext) -> torch.Tensor:
        audio, _ = load_audio(path, 16000)
        return ctx.run("w2v-bert", [self.h_w2v], lambda: self._emb(audio), self._conditioning_memory())

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        import torch

        if not voice.references:
            return None
        with torch.inference_mode():
            cond = self._speaker(voice.references[0].audio_path, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "speaker.pt"
        data = {"format_version": CACHE_FORMAT_VERSION, **{k: getattr(cond, k).detach().cpu() for k in SPEAKER_FIELDS}}
        torch.save(data, str(path))
        return VoiceCache(files={"speaker": str(path)})

    def _cached_speaker(self, voice: VoiceSpec) -> SpeakerCond | None:
        import torch

        path = (voice.cache or {}).get("speaker")
        if not path or not Path(path).is_file():
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != CACHE_FORMAT_VERSION:
            return None
        return SpeakerCond(*(data[k].to(self.device) for k in SPEAKER_FIELDS))

    # -- speech ---------------------------------------------------------------------------------------

    def _emotion(self, style: StyleSpec, ctx: RequestContext) -> tuple[str | None, float, list[float] | None]:
        """(emotion clip, alpha, emotion weights) as ``infer`` resolves them before the speaker's own clip stands in."""
        import torch

        emotion = style.emotion
        if emotion is None:
            return None, 1.0, None
        if emotion.audio_path:
            alpha = float(emotion.strength) if emotion.strength is not None else 1.0
            ctx.applied["emo_alpha"] = alpha
            return emotion.audio_path, alpha, None
        if emotion.vector:
            unknown = sorted(set(emotion.vector) - set(EMOTIONS))
            if unknown:
                raise EngineError(f"Unknown emotions: {', '.join(unknown)}", {"emotions": list(EMOTIONS)}, code=codes.INVALID_INPUT)
            vector = [float(emotion.vector.get(name, 0.0)) for name in EMOTIONS]
            ctx.applied["emo_vector"] = vector
            alpha = 1.0
            if emotion.strength is not None:
                alpha = float(emotion.strength)
                ctx.applied["emo_alpha"] = alpha
            scale = max(0.0, min(1.0, alpha))
            if scale != 1.0:
                vector = [int(x * scale * 10000) / 10000 for x in vector]
            return None, 1.0, vector
        if emotion.text:
            if self.qwen_emo is None:
                raise EngineError("Emotion from text needs the emotion model (load option use_qwen_emo)", {"capability": "emotion_text"}, code=codes.CAPABILITY_NOT_SUPPORTED)
            ctx.applied.update({"use_emo_text": True, "emo_text": emotion.text})
            alpha = 1.0
            if emotion.strength is not None:
                alpha = float(emotion.strength)
                ctx.applied["emo_alpha"] = alpha
            with torch.inference_mode():
                scores = ctx.run(
                    "emotion model", [self.h_emotion] if self.h_emotion is not None else [], lambda: self.qwen_emo.inference(emotion.text, check=ctx.check_cancel), 512 * 2**20
                )
            vector = list(scores.values())
            scale = max(0.0, min(1.0, alpha))
            if scale != 1.0:
                vector = [int(x * scale * 10000) / 10000 for x in vector]
            return None, 1.0, vector
        return None, 1.0, None

    def _segment(
        self,
        text: torch.Tensor,
        lang: torch.Tensor | None,
        spk: SpeakerCond,
        speech_latent: torch.Tensor,
        emovec: torch.Tensor,
        sampling: Any,
        duration_factor: float,
        ctx: RequestContext,
    ) -> torch.Tensor:
        """One segment's waveform, scaled to 16-bit range (float, clamped), (1, samples)."""
        import torch

        from mizuoto_indextts.gpt import beam_sample

        g = ctx.generator_for(self.device)
        half = torch.autocast(self.device.type, dtype=self.half or torch.float16, enabled=self.half is not None)

        def sample() -> torch.Tensor:
            with half:
                conds = self.gpt._conds(speech_latent, emovec, 1)
                prompt, pad = self.gpt.prompt(conds, text, lang)
                return beam_sample(self.gpt, prompt, sampling, g, ctx.check_cancel, pad=pad)

        out = ctx.run("gpt", [self.h_gpt], sample, self._gpt_memory(sampling, int(text.shape[-1])))
        if self.gpt.graphed and not self.gpt.graphs_failed:
            ctx.applied["cuda_graphs"] = True
        if not bool((out[:, -1] == self.stop_mel_token).any()):
            ctx.warn(f"Generation stopped at max_new_tokens ({sampling.max_new_tokens}); the segment may be cut short")
        row = out[0]
        stops = (row == self.stop_mel_token).nonzero(as_tuple=False)
        code_len = int(stops[0, 0].item()) if stops.numel() > 0 else len(row)
        codes_t = out[:, :code_len]
        ctx.check_cancel()
        if self.variant == "v2_5":
            s_infer = ctx.run("semantic codec", [self.h_codec], lambda: self.semantic_codec.decode(codes_t))
            target_lengths = torch.LongTensor([int(s_infer.shape[1] * 1.72 * duration_factor)]).to(self.device)
        else:

            def latents() -> torch.Tensor:
                with half:
                    return self.gpt.latents(speech_latent, text, codes_t, emovec)

            latent = ctx.run("gpt", [self.h_gpt], latents, self._gpt_memory(sampling, int(text.shape[-1]), int(codes_t.shape[-1])))
            gpt_layer = self.gpt_layer
            assert gpt_layer is not None
            s_infer = ctx.run("s2mel", [self.h_s2mel, self.h_codec], lambda: self.semantic_codec.quantizer.vq2emb(codes_t.unsqueeze(1)).transpose(1, 2) + gpt_layer(latent.float()))
            target_lengths = (torch.LongTensor([code_len]).to(self.device) * 1.72).long()
        frames = int(target_lengths.max()) + spk.ref_mel.size(-1)

        def flow() -> torch.Tensor:
            cond = self.regulator(s_infer, target_lengths)
            cat_condition = torch.cat([spk.prompt_condition, cond], dim=1)
            return self.cfm.inference(
                cat_condition, torch.LongTensor([cat_condition.size(1)]).to(self.device), spk.ref_mel, spk.style, 25, g, inference_cfg_rate=0.7, check=ctx.check_cancel
            )

        mel = ctx.run("s2mel", [self.h_s2mel], flow, 2 * 8 * frames * frames * 4 + 512 * 2**20)
        mel = mel[:, :, spk.ref_mel.size(-1) :]
        ctx.check_cancel()
        from mizuoto.nn.dsp.tiles import tiled_1d

        vocoder = self.compiled_bigvgan or self.bigvgan
        tiled = ("in tiles of 256 frames", lambda: tiled_1d(vocoder, mel.float(), int(self.mel_args["hop_size"])).squeeze().unsqueeze(0).squeeze(1))
        wav = ctx.run("bigvgan", [self.h_bigvgan], lambda: vocoder(mel.float()).squeeze().unsqueeze(0).squeeze(1), bigvgan_memory(mel.shape[-1] * 256, 1272), fallback=tiled)
        return torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()

    def _texts(self, task: SynthesisTask, max_tokens: int, lang: str | None) -> list[torch.Tensor]:
        """The segments' token ids, (1, n) int32 each."""
        import torch
        from torch.nn import functional as F

        from mizuoto_indextts.text import split_segments

        if self.frontend is not None:
            assert lang is not None
            prefix = f"<|{lang.lower()}|> "
            text = self.frontend.prepare(task.text, lang)
            capacity = self.gpt.text_pos_embedding.emb.num_embeddings
            out = []
            for seg in self.frontend.split(text, max_tokens, prefix, capacity):
                ids = torch.IntTensor(self.frontend.encode(prefix + seg)).unsqueeze(0).to(self.device)
                out.append(F.pad(ids, (0, 1), value=1))
            return out
        assert self.tokenizer is not None
        tokens = self.tokenizer.tokenize(task.text)
        return [torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), dtype=torch.int32, device=self.device).unsqueeze(0) for sent in split_segments(tokens, max_tokens)]

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        from mizuoto_indextts.gpt import BeamSampling
        from mizuoto_indextts.text25 import lang_to_token

        voice = task.voice
        if not voice.references:
            raise EngineError("IndexTTS clones a voice: it needs a reference clip", code=codes.INVALID_INPUT)
        if task.style.instruct:
            raise EngineError("IndexTTS takes no instruction; describe the emotion instead", {"capability": "instruct"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        p, o = task.params, task.options
        recorded: dict[str, Any] = {}
        for name, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repetition_penalty", "repetition_penalty"),
            ("max_new_tokens", "max_mel_tokens"),
        ):
            if p.get(name) is not None:
                recorded[target] = p[name]
        for name in ("num_beams", "length_penalty"):
            if o.get(name) is not None:
                recorded[name] = o[name]
        sampling = BeamSampling(
            num_beams=int(recorded.get("num_beams", 3)),
            temperature=float(recorded.get("temperature", 0.8)),
            top_k=int(recorded.get("top_k", 30)),
            top_p=float(recorded.get("top_p", 0.8)),
            repetition_penalty=float(recorded.get("repetition_penalty", 10.0)),
            length_penalty=float(recorded.get("length_penalty", 0.0)),
            max_new_tokens=int(recorded.get("max_mel_tokens", 1500)),
        )
        ctx.applied.update(recorded)
        max_tokens = 120
        if o.get("max_text_tokens_per_segment") is not None:
            max_tokens = int(o["max_text_tokens_per_segment"])
            ctx.applied["max_text_tokens_per_segment"] = max_tokens
        emo_path, emo_alpha, emo_vector = self._emotion(task.style, ctx)
        silence_ms = 200
        if p.get("segment_silence_ms") is not None:
            silence_ms = int(p["segment_silence_ms"])
            ctx.applied["interval_silence"] = silence_ms
        use_random = False
        if o.get("use_random") is not None:
            use_random = bool(o["use_random"])
            ctx.applied["use_random"] = use_random
        lang: str | None = None
        duration_factor = 1.0
        if self.variant == "v2_5":
            lang = task.language if task.language in ("zh", "en", "ja", "es") else guess_language(task.text)
            ctx.applied["lang"] = lang
            if p.get("speed") is not None:
                duration_factor = 1.0 / float(p["speed"])
                ctx.applied["duration_factor"] = duration_factor
        rate = RATES[self.variant]

        with torch.inference_mode():
            spk = self._cached_speaker(voice) or self._speaker(voice.references[0].audio_path, ctx)
            ctx.check_cancel()
            if emo_path is None:  # the speaker's clip, at full strength
                emo_cond_emb, emo_alpha = spk.emo_cond_emb, 1.0
            else:
                emo_cond_emb = self._emotion_emb(emo_path, ctx)
            emovec_mat = weight = None
            if emo_vector is not None:
                weight = torch.tensor(emo_vector, device=self.device)
                if use_random:
                    index = [ctx.rng.randint(0, x - 1) for x in self.emo_num]
                else:
                    # on the style's device, as upstream moves its matrices to the model's (the matrices stay on the CPU here)
                    style = spk.style.float()
                    index = [int(torch.argmax(torch.nn.functional.cosine_similarity(style, m.to(style.device).float(), dim=1))) for m in self.spk_matrix]
                rows = torch.cat([m[i].unsqueeze(0) for i, m in zip(index, self.emo_matrix, strict=True)], 0).to(self.device)
                emovec_mat = torch.sum(weight.unsqueeze(1) * rows, 0).unsqueeze(0)
            lengths = torch.tensor([spk.spk_cond_emb.shape[-1]], device=self.device)
            emo_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=self.device)

            def conditioning() -> tuple[torch.Tensor, torch.Tensor]:
                with torch.autocast(self.device.type, dtype=self.half or torch.float16, enabled=self.half is not None):
                    emovec = self.gpt.merge_emovec(spk.spk_cond_emb, emo_cond_emb, lengths, emo_lengths, alpha=emo_alpha)
                    if emovec_mat is not None and weight is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight)) * emovec
                    latent = self.gpt.campplus_latent(spk.style) if self.variant == "v2_5" else self.gpt.get_conditioning(spk.spk_cond_emb, lengths)
                return emovec, latent

            emovec, speech_latent = ctx.run("gpt", [self.h_gpt], conditioning, self._gpt_memory(sampling, 0, 0))
            texts = self._texts(task, max_tokens, lang)
            if not texts:
                raise EngineError("IndexTTS produced no speech for the text", {"text": task.text[:200]}, code=codes.ENGINE_ERROR)
            lang_t = torch.LongTensor([lang_to_token(lang)]).to(self.device) if lang is not None else None
            silence = torch.zeros(1, int(rate * silence_ms / 1000.0))
            wavs = []
            for index, text in enumerate(texts):
                ctx.progress("generating", index / len(texts))
                wav = self._segment(text, lang_t, spk, speech_latent, emovec, sampling, duration_factor, ctx)
                wavs.append(wav)
                if task.stream:
                    yield AudioChunk((wav.float().numpy().reshape(-1) / 32768.0).astype(np.float32), rate, task.segment)
                    if silence_ms > 0:
                        yield AudioChunk((silence.numpy().reshape(-1) / 32768.0).astype(np.float32), rate, task.segment)
        ctx.check_cancel()
        if task.stream:
            yield AudioChunk(np.zeros(0, dtype=np.float32), rate, task.segment, final=True)
            return
        parts: list[torch.Tensor] = []
        for i, wav in enumerate(wavs):
            parts.append(wav)
            if silence_ms > 0 and i < len(wavs) - 1:
                parts.append(silence)
        out = fade_out_tail(torch.cat(parts, dim=1), rate)
        data = out.type(torch.int16).numpy().T
        yield AudioChunk((np.asarray(data, dtype=np.float32).reshape(-1) / 32768.0).astype(np.float32), rate, task.segment, final=True)


class LoadedIndexTts15(LoadedEngine):
    PHASED = True
    """IndexTTS-1.5 (``infer.py``): the reference's 100-band log-mel conditions the GPT and the vocoder; no emotion.

    As upstream: the clip read as torchaudio reads it and resampled to 24 kHz; the GPT runs without a KV cache unless
    it runs in fp16 (a CUDA device; v1's default there); long runs of the silent code are shortened; BigVGAN (with
    its ECAPA speaker encoder) turns the GPT's latents into audio; segments are joined without pauses.
    """

    def __init__(self, spec: LoadSpec, ctx: RequestContext, compile: bool = False) -> None:
        super().__init__(spec)
        import torch
        import yaml

        from mizuoto.nn.dsp.mel import MelSpectrogram
        from mizuoto.nn.vocoders.bigvgan import SpeakerBigVGAN
        from mizuoto.runtime import ops
        from mizuoto.runtime.compile import compile_module
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import fold_weight_norm, load_into, load_state_dict, unwrap
        from mizuoto_indextts.gpt import GptConfig, UnifiedVoice
        from mizuoto_indextts.text import TextNormalizer, TextTokenizer

        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        ctx.progress("loading:gpt", None)
        gpt = UnifiedVoice(GptConfig.from_yaml(cfg["gpt"], "mel"), ops.default)
        load_into(gpt, unwrap(load_state_dict(folder / cfg["gpt_checkpoint"]), "model"), what="IndexTTS-1.5 GPT")
        self.gpt = gpt.eval()
        self.half = self.device.type == "cuda" and spec.dtype in ("auto", "fp16")
        if self.half:
            self.gpt.half()
        ctx.check_cancel()
        ctx.progress("loading:bigvgan", None)
        bigvgan = SpeakerBigVGAN(cfg["bigvgan"])
        load_into(bigvgan, fold_weight_norm(norm_on(load_state_dict(folder / cfg["bigvgan_checkpoint"])["generator"], spec.device)), what="IndexTTS-1.5 BigVGAN")
        self.bigvgan = bigvgan.eval()
        self.compiled_bigvgan = compile_module(self.bigvgan, compile, f"{spec.model_id}:bigvgan")
        self.tokenizer = TextTokenizer(folder / cfg["dataset"]["bpe_model"], TextNormalizer())
        # upstream computes the reference's mel on the CPU and moves it to the device: a GPU's FFT rounds otherwise, and
        # the difference (~1e-4) changed the codes the GPT draws (no weights, so nothing to place)
        self.mel = MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, power=1, n_mels=100)
        self.stop_mel_token = int(cfg["gpt"]["stop_mel_token"])
        name = spec.model_id
        self.h_gpt = ModelHandle(self.gpt, name=f"{name}:gpt", load_device=spec.device, kind="sequence")
        self.h_bigvgan = ModelHandle(self.bigvgan, name=f"{name}:bigvgan", load_device=spec.device)
        self.handles = [self.h_gpt, self.h_bigvgan]

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=RATES["v1_5"], vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return 1024 * 2**20

    def _cond_mel(self, path: str, ctx: RequestContext) -> torch.Tensor:
        import torch

        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        samples, rate = read_wav(path)
        audio = torch.mean(torch.from_numpy(samples)[None], dim=0, keepdim=True)
        audio = Resample(rate, 24000)(audio)
        return torch.log(torch.clip(self.mel(audio), min=1e-7)).to(self.device)  # on the CPU, as upstream

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        import torch

        if not voice.references:
            return None
        with torch.inference_mode():
            mel = self._cond_mel(voice.references[0].audio_path, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "cond_mel.pt"
        torch.save({"format_version": CACHE_FORMAT_VERSION, "cond_mel": mel.detach().cpu()}, str(path))
        return VoiceCache(files={"cond_mel": str(path)})

    def _cached_mel(self, voice: VoiceSpec) -> torch.Tensor | None:
        import torch

        path = (voice.cache or {}).get("cond_mel")
        if not path or not Path(path).is_file():
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data["cond_mel"].to(self.device) if data.get("format_version") == CACHE_FORMAT_VERSION else None

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        from mizuoto_indextts.gpt import BeamSampling, beam_sample, remove_long_silence
        from mizuoto_indextts.text import split_segments

        voice = task.voice
        if not voice.references:
            raise EngineError("IndexTTS clones a voice: it needs a reference clip", code=codes.INVALID_INPUT)
        if task.style.instruct:
            raise EngineError("IndexTTS takes no instruction; describe the emotion instead", {"capability": "instruct"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        if task.style.emotion is not None:
            raise EngineError("IndexTTS-1.5 takes no emotion", {"capability": "emotion_vector"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        p, o = task.params, task.options
        recorded: dict[str, Any] = {}
        for name, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repetition_penalty", "repetition_penalty"),
            ("max_new_tokens", "max_mel_tokens"),
        ):
            if p.get(name) is not None:
                recorded[target] = p[name]
        for name in ("num_beams", "length_penalty"):
            if o.get(name) is not None:
                recorded[name] = o[name]
        sampling = BeamSampling(
            num_beams=int(recorded.get("num_beams", 3)),
            temperature=float(recorded.get("temperature", 1.0)),
            top_k=int(recorded.get("top_k", 30)),
            top_p=float(recorded.get("top_p", 0.8)),
            repetition_penalty=float(recorded.get("repetition_penalty", 10.0)),
            length_penalty=float(recorded.get("length_penalty", 0.0)),
            max_new_tokens=int(recorded.get("max_mel_tokens", 600)),
            kv_cache=self.half,
        )
        ctx.applied.update(recorded)
        max_tokens = 120
        if o.get("max_text_tokens_per_segment") is not None:
            max_tokens = int(o["max_text_tokens_per_segment"])
            ctx.applied["max_text_tokens_per_segment"] = max_tokens
        g = ctx.generator_for(self.device)
        half = torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half)
        wavs = []
        with torch.inference_mode():
            cond_mel = self._cached_mel(voice)
            if cond_mel is None:
                cond_mel = self._cond_mel(voice.references[0].audio_path, ctx)
            lengths = torch.tensor([cond_mel.shape[-1]], device=self.device)
            segments = split_segments(self.tokenizer.tokenize(task.text), max_tokens)
            if not segments:
                raise EngineError("IndexTTS produced no speech for the text", {"text": task.text[:200]}, code=codes.ENGINE_ERROR)
            for index, sent in enumerate(segments):
                ctx.progress("generating", index / len(segments))
                text = torch.tensor(self.tokenizer.convert_tokens_to_ids(sent), dtype=torch.int32, device=self.device).unsqueeze(0)
                beams, tokens = int(sampling.num_beams), int(text.shape[-1])

                def sample(text: torch.Tensor = text) -> tuple[torch.Tensor, torch.Tensor]:
                    with half:
                        latent_in = self.gpt.get_conditioning(cond_mel.transpose(1, 2), lengths)
                        prompt, pad = self.gpt.prompt(latent_in, text)
                        return latent_in, beam_sample(self.gpt, prompt, sampling, g, ctx.check_cancel, pad=pad)

                latent_in, out = ctx.run("gpt", [self.h_gpt], sample, gpt_memory(self.gpt, beams, tokens, int(sampling.max_new_tokens), bool(self.half)))
                if self.gpt.graphed and not self.gpt.graphs_failed:
                    ctx.applied["cuda_graphs"] = True
                if not bool((out[:, -1] == self.stop_mel_token).any()):
                    ctx.warn(f"Generation stopped at max_new_tokens ({sampling.max_new_tokens}); the segment may be cut short")
                codes_t, _ = remove_long_silence(out, self.stop_mel_token)
                ctx.check_cancel()

                def latents(text: torch.Tensor = text, latent_in: torch.Tensor = latent_in, codes_t: torch.Tensor = codes_t) -> torch.Tensor:
                    with half:
                        return self.gpt.latents(latent_in, text, codes_t, None)

                def vocode(latent: torch.Tensor) -> torch.Tensor:
                    with half:
                        return (self.compiled_bigvgan or self.bigvgan)(latent, cond_mel.transpose(1, 2)).squeeze(1)

                latent = ctx.run("gpt", [self.h_gpt], latents, gpt_memory(self.gpt, beams, tokens, int(sampling.max_new_tokens), bool(self.half), int(codes_t.shape[-1])))
                wav = ctx.run("bigvgan", [self.h_bigvgan], lambda latent=latent: vocode(latent), bigvgan_memory(latent.shape[1] * 1024, 928, 320))
                # Scaled in the vocoder's own dtype, as upstream: under half precision 32767 × wav rounds in float16 (-7.99976
                # becomes -8.0) before the truncation to int16; in float32 it truncates to -7.
                wavs.append(torch.clamp(32767 * wav, -32767.0, 32767.0).cpu())
        ctx.check_cancel()
        out_wav = fade_out_tail(torch.cat(wavs, dim=1), RATES["v1_5"])
        data = out_wav.type(torch.int16).numpy().T
        yield AudioChunk((np.asarray(data, dtype=np.float32).reshape(-1) / 32768.0).astype(np.float32), RATES["v1_5"], task.segment, final=True)


def _strip_module(state: dict[str, Any]) -> dict[str, Any]:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
