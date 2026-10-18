"""OmniVoice, native: cloning, design from attributes, its own voice.

The port follows Mizuoto's v1 adapter over upstream ``OmniVoice.generate`` (transformers 5.3) step for step,
so a seeded request on the CPU in fp32 gives the same audio:
- the voice prompt is ``create_voice_clone_prompt``: the reference levelled up to an RMS of 0.1, its
  silences shortened (pydub's arithmetic, ``audio.py``), cut to whole frames and encoded by the HiggsAudio
  tokenizer; it is cached as upstream's ``prompt.pt``;
- the target length comes from the rule-based duration estimate, scaled by ``speed`` or set by ``duration``;
- decoding is ``_generate_iterative``: the conditional and unconditional rows in one batch under a full
  boolean mask, ``num_step`` rounds that each unmask a scheduled number of the most confident codes, with
  classifier-free guidance, a per-codebook penalty and Gumbel noise on positions (and on classes when
  ``class_temperature`` is set) drawn from the request's generator;
- outputs longer than ``audio_chunk_threshold`` are generated in chunks of text, conditioned on the
  reference (or on the first chunk), and cross-faded;
- the audio is decoded, its long silences shortened, levelled back to the reference's RMS (or peak
  normalised to 0.5 without a reference), faded and padded.

FlashInfer and CUDA graphs (v1's load options) are not ported: SDPA serves every platform. In FlashInfer's place the
option ``packed_guidance`` packs each item's two sequences into one row (a fast mode: not bit-identical on a GPU).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    import numpy as np
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

SAMPLE_RATE = 24000
MAX_REFERENCE_SECONDS = 20.0
"""Longer references grow memory with the square of their length."""
PROMPT_FORMAT_VERSION = 1

ATTRIBUTE_ORDER = ("gender", "age", "pitch", "style", "accent", "dialect")
"""The design attributes, in the order OmniVoice's instructions list them."""

# Mizuoto's common parameter name → GenerationConfig field.
CONFIG_PARAMS = {"num_steps": "num_step", "guidance_scale": "guidance_scale", "temperature": "class_temperature"}
CONFIG_OPTIONS = (
    "t_shift",
    "layer_penalty_factor",
    "position_temperature",
    "denoise",
    "preprocess_prompt",
    "postprocess_output",
    "audio_chunk_duration",
    "audio_chunk_threshold",
    "packed_guidance",
)


@dataclass
class GenerationConfig:
    """Upstream's ``OmniVoiceGenerationConfig``."""

    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    packed_guidance: bool = False
    """Mizuoto's fast mode: each item's conditional and unconditional sequences in one row (``_layout``)."""
    pad_duration: float = 0.1
    fade_duration: float = 0.1


@dataclass
class VoicePrompt:
    """Upstream's ``VoiceClonePrompt``: the reference's codes (codebooks, frames), transcript and RMS."""

    tokens: torch.Tensor
    text: str
    rms: float

    def save(self, path: Path) -> None:
        import torch

        data = {"format_version": PROMPT_FORMAT_VERSION, "ref_audio_tokens": self.tokens.detach().cpu(), "ref_text": self.text, "ref_rms": float(self.rms)}
        torch.save(data, str(path))

    @classmethod
    def load(cls, path: Path) -> VoicePrompt:
        import torch

        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if data.get("format_version") != PROMPT_FORMAT_VERSION:
            raise EngineError(f"Unsupported voice prompt format {data.get('format_version')}", code=codes.INVALID_INPUT)
        return cls(data["ref_audio_tokens"], data["ref_text"], data["ref_rms"])


@dataclass
class _Item:
    text: str
    target_len: int
    lang: str | None
    instruct: str | None
    prompt: VoicePrompt | None
    speed: float | None


def instruct_from(attributes: dict[str, str]) -> str:
    """``{gender: female, pitch: low pitch}`` → ``female, low pitch`` (a dialect makes it Chinese, ``，``)."""
    values = [attributes[k] for k in ATTRIBUTE_ORDER if attributes.get(k)]
    values += [v for k, v in attributes.items() if k not in ATTRIBUTE_ORDER and v]
    chinese = any("一" <= c <= "鿿" for v in values for c in v)
    return ("，" if chinese else ", ").join(values)


class OmniVoiceEngine(Engine):
    id = "omnivoice"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto.engines.omnivoice.train:OmniVoiceTrainer"
    OPTION_SCHEMA = {
        "t_shift": OptionSpec("float", 0.1, 0.0, 1.0),
        "layer_penalty_factor": OptionSpec("float", 5.0, 0.0, 20.0),
        "position_temperature": OptionSpec("float", 5.0, 0.0, 20.0),
        "denoise": OptionSpec("bool", True),
        "preprocess_prompt": OptionSpec("bool", True),
        "postprocess_output": OptionSpec("bool", True),
        "audio_chunk_duration": OptionSpec("float", 15.0, 0.0, 60.0),
        "audio_chunk_threshold": OptionSpec("float", 30.0, 0.0, 600.0),
        "packed_guidance": OptionSpec("bool", False),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        try:
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return "v0" if config.get("model_type") == "omnivoice" else None

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or self.detect_variant(spec.model_dir) is None:
            raise EngineError(f"{spec.model_id} is not an OmniVoice model", code=codes.INVALID_INPUT)
        return LoadedOmniVoice(spec, ctx)


def _dtype(name: str, device: str) -> torch.dtype:
    """As v1: bf16 on a GPU that has it (else fp16), fp32 on the CPU; an explicit choice wins."""
    import torch

    explicit = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name in explicit:
        return explicit[name]
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def time_steps(num_step: int, t_shift: float) -> list[float]:
    import torch

    t = torch.linspace(0.0, 1.0, num_step + 1)
    return (t_shift * t / (1 + (t_shift - 1) * t)).tolist()


def schedule(target_len: int, codebooks: int, num_step: int, timesteps: list[float]) -> list[int]:
    """How many codes each step unmasks: the shifted time grid's share of them, the rest at the last step."""
    total = target_len * codebooks
    rem = total
    out = []
    for step in range(num_step):
        num = rem if step == num_step - 1 else min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), rem)
        out.append(int(num))
        rem -= int(num)
    return out


def gumbel(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    import torch

    scaled = logits / temperature
    u = torch.rand(scaled.shape, generator=generator, device=scaled.device, dtype=scaled.dtype)
    return scaled + -torch.log(-torch.log(u + 1e-10) + 1e-10)


def load_audio_tokenizer(folder: Path) -> tuple[Any, int]:
    """The model folder's HiggsAudio tokenizer (``audio_tokenizer/``), in eval mode and float32 as transformers loads it
    ("auto": the checkpoint's dtype), and the rate it takes."""
    from mizuoto.engines.omnivoice.audio_tokenizer import IGNORED, HiggsAudioTokenizer
    from mizuoto.nn.ssl.hubert import fold_pos_conv
    from mizuoto.runtime import ops
    from mizuoto.runtime.loading import load_into, load_state_dict

    tok_dir = folder / "audio_tokenizer"
    if not (tok_dir / "model.safetensors").is_file():
        raise EngineError(f"The audio tokenizer is missing from {folder}", code=codes.MODEL_MISSING)
    tok_config = json.loads((tok_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = HiggsAudioTokenizer(tok_config, ops.default)
    state = fold_pos_conv(load_state_dict(tok_dir / "model.safetensors"), "semantic_model.")
    load_into(tokenizer, state, ignore=[*IGNORED, *(f"semantic_model.{k}" for k in tokenizer.semantic_model.ignorable())], what="HiggsAudio tokenizer")
    del state
    preprocessor = tok_dir / "preprocessor_config.json"
    rate = int(json.loads(preprocessor.read_text(encoding="utf-8")).get("sampling_rate", SAMPLE_RATE)) if preprocessor.is_file() else SAMPLE_RATE
    return tokenizer.eval(), rate


class LoadedOmniVoice(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.omnivoice.model import OmniVoiceModel
        from mizuoto.runtime import ops
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict
        from mizuoto.text.tokenize.bpe import load_bpe

        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        self.dtype = _dtype(spec.dtype, spec.device)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        ctx.progress("loading:weights", None)
        model = OmniVoiceModel(config, ops.default)
        load_into(model, load_state_dict(folder / "model.safetensors"), what="OmniVoice model")
        model.to(dtype=self.dtype).eval()
        self.model = model

        ctx.check_cancel()
        ctx.progress("loading:audio tokenizer", None)
        tokenizer, self.sampling_rate = load_audio_tokenizer(folder)
        self.audio_tokenizer = tokenizer
        self.text_tokenizer = load_bpe(folder)
        # the LM runs every step over the whole sequence; the tokenizer encodes the voice prompt and decodes
        self.h_model = ModelHandle(model, name=f"{spec.model_id}:model", load_device=spec.device)
        self.h_tokenizer = ModelHandle(tokenizer, name=f"{spec.model_id}:audio tokenizer", load_device=spec.device)
        self.handles = [self.h_model, self.h_tokenizer]

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=self.sampling_rate, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return self._generation_memory(len(task.texts) if getattr(task, "texts", None) else 1, 1600)

    def _generation_memory(self, rows: int, positions: int) -> int:
        """Per batch row (conditional and unconditional): the logits and their log-softmax copies in float32, and the
        dense boolean attention mask, for ``positions`` positions."""
        per_row = positions * self.model.num_codebooks * self.model.vocab * 4 * 4 + positions * positions
        return 512 * 2**20 + 2 * rows * per_row

    def _tokenizer_memory(self, samples: int) -> int:
        """The audio tokenizer's activations for ``samples`` of audio (DAC's widest stages: ~64 float32 channels per sample)."""
        return samples * 4 * 64 + 256 * 2**20

    # -- voices -----------------------------------------------------------------------

    def _prompt(self, voice: VoiceSpec, preprocess: bool, ctx: RequestContext) -> VoicePrompt:
        """``create_voice_clone_prompt`` for the voice's first reference (cut to 20 s, as v1)."""
        import numpy as np
        import torch

        from mizuoto.engines.omnivoice.audio import remove_silence
        from mizuoto.engines.omnivoice.text import add_punctuation
        from mizuoto.nn.dsp.resample import resample
        from mizuoto.nn.dsp.wav import read_wav

        if not voice.references:
            raise EngineError("Cloning needs a reference clip", code=codes.INVALID_INPUT)
        ref = voice.references[0]
        transcript = (ref.transcript or "").strip()
        if not transcript:
            raise EngineError("OmniVoice needs the reference clip's transcript", {"capability": "clone_no_transcript"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        samples, sr = read_wav(ref.audio_path)
        wav = samples[: int(MAX_REFERENCE_SECONDS * sr)][np.newaxis, :]
        if sr != self.sampling_rate:
            wav = resample(torch.from_numpy(wav), sr, self.sampling_rate).numpy()
        rms = float(np.sqrt(np.mean(wav**2)))
        if 0 < rms < 0.1:
            wav = wav * 0.1 / rms
        if preprocess:
            wav = remove_silence(wav, self.sampling_rate, mid_sil=200, lead_sil=100, trail_sil=200)
            if wav.shape[-1] == 0:
                raise EngineError("The reference is empty after silence removal", {"hint": "set the preprocess_prompt option off"}, code=codes.INVALID_INPUT)
        hop = self.audio_tokenizer.hop_length
        clip = int(wav.shape[-1] % hop)
        wav = wav[:, :-clip] if clip > 0 else wav
        ctx.check_cancel()
        tokens = ctx.run(
            "audio tokenizer",
            [self.h_tokenizer],
            lambda: self.audio_tokenizer.encode(torch.from_numpy(wav).to(self.device).unsqueeze(0), layerdrop=ctx.generator_for("cpu")).squeeze(0),
            self._tokenizer_memory(wav.shape[-1]),
        )
        return VoicePrompt(tokens, add_punctuation(transcript) if preprocess else transcript, rms)

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        if not voice.references:
            return None
        prompt = self._prompt(voice, True, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "prompt.pt"
        prompt.save(path)
        return VoiceCache(files={"prompt": str(path)})

    @staticmethod
    def _cached_prompt(voice: VoiceSpec) -> VoicePrompt | None:
        path = (voice.cache or {}).get("prompt")
        if not path or not Path(path).is_file():
            return None
        return VoicePrompt.load(Path(path))

    # -- generation -------------------------------------------------------------------

    def _estimate(self, text: str, prompt_text: str | None, prompt_frames: int | None, speed: float = 1.0) -> int:
        from mizuoto.engines.omnivoice.duration import estimate_duration

        if prompt_frames is None or prompt_text is None or len(prompt_text) == 0:
            prompt_text, prompt_frames = "Nice to meet you.", 25
        est = estimate_duration(text, prompt_text, prompt_frames)
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _inputs(self, item: _Item, denoise: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """``_prepare_inference_inputs``: style, text, the reference's codes and the masked target, as (1, C, N)."""
        import torch

        from mizuoto.engines.omnivoice.text import combine_text, tokenize

        c = self.model.num_codebooks
        ref_tokens = item.prompt.tokens if item.prompt is not None else None
        style = "<|denoise|>" if denoise and ref_tokens is not None else ""
        style += f"<|lang_start|>{item.lang or 'None'}<|lang_end|>"
        style += f"<|instruct_start|>{item.instruct or 'None'}<|instruct_end|>"
        style_ids = torch.tensor([self.text_tokenizer.encode(style)], dtype=torch.long).repeat(c, 1).unsqueeze(0).to(self.device)
        wrapped = f"<|text_start|>{combine_text(item.text, item.prompt.text if item.prompt is not None else None)}<|text_end|>"
        text_ids = torch.tensor([tokenize(wrapped, self.text_tokenizer)], dtype=torch.long).repeat(c, 1).unsqueeze(0).to(self.device)
        target = torch.full((1, c, item.target_len), self.model.mask_id, dtype=torch.long, device=self.device)
        parts = [style_ids, text_ids]
        if ref_tokens is not None:
            parts.append(ref_tokens.unsqueeze(0).to(self.device))
        parts.append(target)
        ids = torch.cat(parts, dim=2)
        total = ids.shape[2]
        start = total - item.target_len - (ref_tokens.size(-1) if ref_tokens is not None else 0)
        audio_mask = torch.zeros(1, total, dtype=torch.bool, device=self.device)
        audio_mask[0, start:] = True
        return ids, audio_mask

    def _iterative(self, items: list[_Item], config: GenerationConfig, ctx: RequestContext, done: list[int], total_steps: int) -> list[torch.Tensor]:
        """``_generate_iterative``: the codes of each item, (codebooks, target_len)."""
        longest = max(it.target_len + (it.prompt.tokens.size(-1) if it.prompt is not None else 0) + 4 * len(it.text) for it in items)
        start = done[0]

        def iterate() -> list[torch.Tensor]:
            done[0] = start  # (progress counted again on a retry)
            return self._iterate(items, config, ctx, done, total_steps)

        if config.packed_guidance:  # one row per item, as long as both sequences
            memory = self._generation_memory(1, 2 * longest) * len(items) // 2
        else:
            memory = self._generation_memory(len(items), longest)
        return ctx.run("language model", [self.h_model], iterate, memory)

    def _layout(
        self, inputs: list[tuple[torch.Tensor, torch.Tensor]], targets: list[int], packed: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """The batch for every step: ids, audio mask, attention mask, positions (None: ``0…`` in every row) and, per
        item, where its target sits in the conditional and the unconditional sequence (row, first position).

        Upstream's layout: the conditional sequences in rows ``0…b−1``, the unconditional ones (the target alone) in
        rows ``b…2b−1``, each padded to the longest; a padding position attends only to itself. Packed (the
        ``packed_guidance`` option): an item's conditional and unconditional sequences side by side in one row, a
        block-diagonal mask between them and each one's positions from 0, so each computes what it computed in its own
        row, without the padding the unconditional rows carried: the same model, other rounding (the products and
        attention run over other shapes)."""
        import torch

        c, mask_id, device = self.model.num_codebooks, self.model.mask_id, self.device
        b = len(inputs)
        spans = [((ids.size(2), 0), (t, ids.size(2))) if packed else ((ids.size(2), 0), (t, 0)) for (ids, _), t in zip(inputs, targets, strict=True)]
        rows = b if packed else 2 * b
        width = max(u_at + u_len for _, (u_len, u_at) in spans) if packed else max(c_len for (c_len, _), _ in spans)
        batch_ids = torch.full((rows, c, width), mask_id, dtype=torch.long, device=device)
        batch_audio = torch.zeros((rows, width), dtype=torch.bool, device=device)
        batch_attn = torch.zeros((rows, 1, width, width), dtype=torch.bool, device=device)
        positions = torch.zeros((rows, width), dtype=torch.long, device=device) if packed else None
        places = []
        for i, ((ids, audio), ((c_len, _), (u_len, u_at))) in enumerate(zip(inputs, spans, strict=True)):
            u_row = i if packed else b + i
            batch_ids[i, :, :c_len] = ids
            batch_audio[i, :c_len] = audio
            batch_attn[i, :, :c_len, :c_len] = True
            batch_ids[u_row, :, u_at : u_at + u_len] = ids[..., -u_len:]
            batch_audio[u_row, u_at : u_at + u_len] = audio[..., -u_len:]
            batch_attn[u_row, :, u_at : u_at + u_len, u_at : u_at + u_len] = True
            if width > u_at + u_len:  # after the unconditional sequence (as upstream: the conditional rows' padding attends nowhere)
                diag = torch.arange(u_at + u_len, width, device=device)
                batch_attn[u_row, :, diag, diag] = True
            if positions is not None:
                positions[i, :c_len] = torch.arange(c_len, device=device)
                positions[i, u_at : u_at + u_len] = torch.arange(u_len, device=device)
                if width > u_at + u_len:
                    positions[i, u_at + u_len :] = torch.arange(u_at + u_len, width, device=device)
            places.append(((i, c_len - u_len), (u_row, u_at)))
        return batch_ids, batch_audio, batch_attn, positions, places

    def _iterate(self, items: list[_Item], config: GenerationConfig, ctx: RequestContext, done: list[int], total_steps: int) -> list[torch.Tensor]:
        import torch
        from torch.nn import functional as F

        model = self.model
        c, mask_id = model.num_codebooks, model.mask_id
        b = len(items)
        inputs = [self._inputs(item, config.denoise) for item in items]
        batch_ids, batch_audio, batch_attn, positions, places = self._layout(inputs, [it.target_len for it in items], config.packed_guidance)
        tokens = torch.full((b, c, max(it.target_len for it in items)), mask_id, dtype=torch.long, device=self.device)
        timesteps = time_steps(config.num_step, config.t_shift)
        schedules = [schedule(it.target_len, c, config.num_step, timesteps) for it in items]
        layer_ids = torch.arange(c, device=self.device).view(1, -1, 1)
        generator = ctx.generator_for(self.device)
        for step in range(config.num_step):
            ctx.check_cancel()
            logits = model(batch_ids, batch_audio, batch_attn, positions).to(torch.float32)
            for i in range(b):
                k = schedules[i][step]
                if k <= 0:
                    continue
                t_len = items[i].target_len
                (c_row, c_at), (u_row, u_at) = places[i]
                c_logits = logits[c_row : c_row + 1, :, c_at : c_at + t_len, :]
                u_logits = logits[u_row : u_row + 1, :, u_at : u_at + t_len, :]
                if config.guidance_scale != 0:
                    c_lp = F.log_softmax(c_logits, dim=-1)
                    u_lp = F.log_softmax(u_logits, dim=-1)
                    log_probs = torch.log_softmax(c_lp + config.guidance_scale * (c_lp - u_lp), dim=-1)
                else:
                    log_probs = F.log_softmax(c_logits, dim=-1)
                log_probs[..., mask_id] = -float("inf")
                if config.class_temperature > 0.0:
                    top = math.ceil(0.1 * log_probs.shape[-1])
                    val, ind = log_probs.topk(top, dim=-1)
                    filtered = torch.full_like(log_probs, float("-inf")).scatter_(-1, ind, val)
                    pred = gumbel(filtered, config.class_temperature, generator).argmax(dim=-1)
                else:
                    pred = log_probs.argmax(dim=-1)
                scores = log_probs.max(dim=-1)[0]
                scores = scores - (layer_ids * config.layer_penalty_factor)
                if config.position_temperature > 0.0:
                    scores = gumbel(scores, config.position_temperature, generator)
                sample = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(sample != mask_id, -float("inf"))
                _, top_idx = torch.topk(scores.flatten(), k)
                flat = sample.flatten()
                flat[top_idx] = pred.flatten()[top_idx]
                sample.copy_(flat.view_as(sample))
                tokens[i : i + 1, :, :t_len] = sample
                batch_ids[c_row : c_row + 1, :, c_at : c_at + t_len] = sample
                batch_ids[u_row : u_row + 1, :, u_at : u_at + t_len] = sample
            done[0] += 1
            ctx.progress("generating", done[0] / total_steps, f"step {done[0]}")
        return [tokens[i, :, : items[i].target_len] for i in range(b)]

    def _chunked(self, items: list[_Item], config: GenerationConfig, ctx: RequestContext, done: list[int]) -> list[list[torch.Tensor]]:
        """``_generate_chunked``: text cut to about ``audio_chunk_duration`` each, chunk ``n`` of every item in one batch."""
        from mizuoto.engines.omnivoice.text import chunk_text_punctuation

        rate = self.audio_tokenizer.frame_rate
        all_chunks = [chunk_text_punctuation(it.text, int(config.audio_chunk_duration * rate / (it.target_len / len(it.text))), min_chunk_len=3) for it in items]
        has_ref = [it.prompt is not None for it in items]
        if any(has_ref) and not all(has_ref):
            raise EngineError("Chunked generation needs every item with a reference or none", code=codes.INVALID_INPUT)
        rounds = max(len(c) for c in all_chunks)
        results: list[list[torch.Tensor]] = [[] for _ in items]
        total = config.num_step * rounds
        if all(has_ref):
            for ci in range(rounds):
                idx = [i for i in range(len(items)) if ci < len(all_chunks[i])]
                if idx:
                    self._run_chunk(items, idx, [all_chunks[i][ci] for i in idx], [items[i].prompt for i in idx], config, ctx, results, done, total)
        else:
            first = [i for i in range(len(items)) if all_chunks[i]]
            self._run_chunk(items, first, [all_chunks[i][0] for i in first], [None] * len(first), config, ctx, results, done, total)
            anchors = {i: VoicePrompt(results[i][0], all_chunks[i][0], 0.0) for i in first}
            for ci in range(1, rounds):
                idx = [i for i in range(len(items)) if ci < len(all_chunks[i])]
                if idx:
                    self._run_chunk(items, idx, [all_chunks[i][ci] for i in idx], [anchors[i] for i in idx], config, ctx, results, done, total)
        return results

    def _run_chunk(
        self,
        items: list[_Item],
        idx: list[int],
        texts: list[str],
        prompts: list[VoicePrompt | None],
        config: GenerationConfig,
        ctx: RequestContext,
        results: list[list[torch.Tensor]],
        done: list[int],
        total: int,
    ) -> None:
        sub = []
        for j, i in enumerate(idx):
            p = prompts[j]
            own = items[i].speed
            speed = own if own is not None else 1.0
            target = self._estimate(texts[j], p.text if p is not None else None, p.tokens.size(-1) if p is not None else None, speed=speed)
            sub.append(_Item(texts[j], target, items[i].lang, items[i].instruct, p, items[i].speed))
        for j, tokens in enumerate(self._iterative(sub, config, ctx, done, total)):
            results[idx[j]].append(tokens)

    def _decode(self, tokens: torch.Tensor | list[torch.Tensor], rms: float | None, config: GenerationConfig, ctx: RequestContext) -> np.ndarray:
        """``_decode_and_post_process``."""
        import numpy as np

        from mizuoto.engines.omnivoice.audio import cross_fade_chunks, fade_and_pad_audio, remove_silence

        def decode(t: torch.Tensor) -> np.ndarray:
            return self.audio_tokenizer.decode(t.to(self.device).unsqueeze(0))[0].cpu().numpy()

        frames = max(int(t.shape[-1]) for t in tokens) if isinstance(tokens, list) else int(tokens.shape[-1])
        audio = ctx.run(
            "audio tokenizer",
            [self.h_tokenizer],
            lambda: cross_fade_chunks([decode(t) for t in tokens], self.sampling_rate) if isinstance(tokens, list) else decode(tokens),
            self._tokenizer_memory(frames * self.audio_tokenizer.hop_length),
        )
        if config.postprocess_output:
            audio = remove_silence(audio, self.sampling_rate, mid_sil=500, lead_sil=100, trail_sil=100)
        if rms is not None and rms < 0.1:
            audio = audio * rms / 0.1
        elif rms is None:
            peak = np.abs(audio).max()
            if peak > 1e-6:
                audio = audio / peak * 0.5
        audio = fade_and_pad_audio(audio, pad_duration=config.pad_duration, fade_duration=config.fade_duration, sample_rate=self.sampling_rate)
        return audio.squeeze(0)

    @staticmethod
    def _config(task: SynthesisTask) -> GenerationConfig:
        values: dict[str, Any] = {}
        for name, target in CONFIG_PARAMS.items():
            if task.params.get(name) is not None:
                values[target] = task.params[name]
        for name in CONFIG_OPTIONS:
            if task.options.get(name) is not None:
                values[name] = task.options[name]
        return GenerationConfig(**values)

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np

        from mizuoto.engines.omnivoice.voice_design import ZH_RE, resolve_instruct, resolve_language

        texts = list(task.texts) if task.texts else [task.text]
        config = self._config(task)
        voice = task.voice
        prompt: VoicePrompt | None = None
        instruct: str | None = None
        if voice.references:
            prompt = self._cached_prompt(voice) or self._prompt(voice, config.preprocess_prompt, ctx)
            ctx.applied["voice"] = "clone"
        elif voice.attributes:
            instruct = instruct_from(voice.attributes)
            ctx.applied["instruct"] = instruct
        elif voice.description:
            raise EngineError("OmniVoice designs from attributes, not a free description", code=codes.CAPABILITY_NOT_SUPPORTED)
        lang = resolve_language(task.language) if task.language else None
        speed = float(task.params["speed"]) if task.params.get("speed") is not None else None
        duration = float(task.params["duration"]) if task.params.get("duration") is not None else None
        ctx.applied.update({f.name: getattr(config, f.name) for f in fields(config)})
        if speed is not None:
            ctx.applied["speed"] = speed
        if duration is not None:
            ctx.applied["duration"] = duration

        items = []
        for text in texts:
            try:
                resolved = resolve_instruct(instruct, use_zh=bool(text and ZH_RE.search(text)))
            except ValueError as exc:
                raise EngineError(str(exc), code=codes.INVALID_INPUT) from exc
            frames = prompt.tokens.size(-1) if prompt is not None else None
            target = self._estimate(text, prompt.text if prompt is not None else None, frames, speed=1.0 if duration is not None else (speed or 1.0))
            item_speed = speed
            if duration is not None:
                fixed = max(1, int(duration * self.audio_tokenizer.frame_rate))
                item_speed = target / fixed if fixed > 0 else 1.0
                target = fixed
            items.append(_Item(text, target, lang, resolved, prompt, item_speed if (speed is not None or duration is not None) else None))

        threshold = int(config.audio_chunk_threshold * self.audio_tokenizer.frame_rate)
        short = [i for i, it in enumerate(items) if it.target_len <= threshold]
        long = [i for i, it in enumerate(items) if it.target_len > threshold]
        results: list[Any] = [None] * len(items)
        done = [0]
        if short:
            for i, tokens in zip(short, self._iterative([items[i] for i in short], config, ctx, done, config.num_step), strict=True):
                results[i] = tokens
        if long:
            done[0] = 0
            for i, chunks in zip(long, self._chunked([items[i] for i in long], config, ctx, done), strict=True):
                results[i] = chunks
        ctx.check_cancel()
        for index, item in enumerate(items):
            audio = self._decode(results[index], item.prompt.rms if item.prompt is not None else None, config, ctx)
            samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
            segment = task.segment + index if task.texts else task.segment
            yield AudioChunk(samples=samples, sample_rate=self.sampling_rate, segment=segment, final=True)
