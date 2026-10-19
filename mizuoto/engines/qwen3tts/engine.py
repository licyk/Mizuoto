"""Qwen3-TTS, native: Base (cloning), CustomVoice (preset speakers), VoiceDesign.

The port follows Mizuoto's v1 adapter over upstream ``qwen_tts`` step for step, so a seeded request on
the CPU in fp32 gives the same audio:
- the prompt is built as ``Qwen3TTSForConditionalGeneration.generate`` builds it, with the same tensor
  shapes;
- the talker's loop is transformers' ``_sample``: processors in order, ``multinomial`` with the
  request's generator, EOS rows padded with EOS;
- each frame's 15 remaining codebooks come from the code predictor's own sampling loop;
- the codes are decoded in upstream's chunks, and the reference's share is cut after a joint decode
  when cloning in context.

A streamed request (one text) decodes the frames as they come: the codec's decoder is causal, so decoding each of
upstream's chunks frame by frame from its usual start gives its samples again (``codec.DecodeStream``), and the audio
is the non-streamed request's up to rounding.

The tokenizer is ``mizuoto.text``'s BPE with transformers' "Mistral regex" split, which upstream's
processor applies (its config names transformers 4.57.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAMES_PER_TOKEN = 16
"""Frames a text token is assumed to need at most, for the memory estimate (speech runs at ~3–4 a token)."""
STREAM_FIRST = 4
"""A streamed request's first audio comes after this many frames (0.32 s of speech)…"""
STREAM_EVERY = 12
"""…and then every this many (about a second)."""
PROMPT_POSITIONS = 32
"""The talker prompt's positions besides its texts and reference codes (role, special and codec embeddings)."""
LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "es": "spanish",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "ru": "russian",
}
GENERATE_PARAMS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")
HARD_DEFAULTS: dict[str, Any] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": 2048,
}


class Qwen3TtsEngine(Engine):
    id = "qwen3tts"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto.engines.qwen3tts.train:Qwen3Trainer"
    OPTION_SCHEMA = {
        "subtalker_temperature": OptionSpec("float", 0.9, 0.0, 2.0),
        "subtalker_top_k": OptionSpec("int", 50, 0, 1000),
        "subtalker_top_p": OptionSpec("float", 1.0, 0.0, 1.0),
        "subtalker_do_sample": OptionSpec("bool", True),
        "non_streaming_mode": OptionSpec("bool", None),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        from mizuoto.engines.qwen3tts.config import VARIANTS, read_config

        try:
            config = read_config(model_dir)
        except (OSError, ValueError):
            return None
        if config.get("tokenizer_type") != "qwen3_tts_tokenizer_12hz":
            return None
        return VARIANTS.get((config.get("tts_model_type", ""), config.get("tts_model_size", "")))

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None:
            raise EngineError("Qwen3-TTS needs a model folder", code=codes.MODEL_MISSING)
        variant = self.detect_variant(spec.model_dir)
        if variant is None:
            raise EngineError(f"{spec.model_dir.name} is not a Qwen3-TTS 12 Hz model", code=codes.INVALID_INPUT)
        if variant != spec.variant:
            raise EngineError(f"{spec.model_id} is a {variant} model, not {spec.variant}", {"detected": variant}, code=codes.INVALID_INPUT)
        return LoadedQwen3Tts(spec, runtime, ctx)


def _dtype(name: str, device: str) -> torch.dtype:
    """As v1: bf16 on a GPU that has it (else fp16), fp32 on the CPU; an explicit choice wins."""
    import torch

    explicit = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name in explicit:
        return explicit[name]
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _mistral_fix(model_dir: Path) -> bool:
    """Whether transformers (4.57.3, ``fix_mistral_regex=True``) swaps this tokenizer's split regex."""
    version = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("transformers_version") or ""
    try:
        parts = tuple(int(p) for p in version.split(".")[:3])
    except ValueError:
        return False
    return (4, 57, 2) < parts < (5, 0, 0)


class LoadedQwen3Tts(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> None:
        super().__init__(spec)
        import torch
        from torch import nn

        from mizuoto.engines.qwen3tts.codec import SpeechTokenizer
        from mizuoto.engines.qwen3tts.config import load_config
        from mizuoto.engines.qwen3tts.speaker import SpeakerEncoder
        from mizuoto.engines.qwen3tts.talker import Talker
        from mizuoto.runtime import ops
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict
        from mizuoto.text.tokenize.bpe import load_bpe

        assert spec.model_dir is not None
        folder = spec.model_dir
        self.config = load_config(folder)
        self.variant = spec.variant
        self.device = torch.device(spec.device)
        self.dtype = _dtype(spec.dtype, spec.device)
        self.kv_cache = runtime.options.kv_cache
        self.cuda_graphs = runtime.options.cuda_graphs
        self._graphs: Any = None
        self._decoders: dict[Any, Any] = {}
        self._graph_signature: tuple[int, ...] | None = None
        self._graphs_failed = False
        ctx.progress("loading:weights", None)

        model = nn.Module()
        self.talker = Talker(self.config.talker, ops.default)
        model.add_module("talker", self.talker)
        self.speaker_encoder: SpeakerEncoder | None = None
        if self.config.model_type == "base":
            self.speaker_encoder = SpeakerEncoder(self.config.speaker_encoder, ops.default)
            model.add_module("speaker_encoder", self.speaker_encoder)
        # Cast before loading (the parameters are still unwritten): the checkpoint is copied straight into the dtype the
        # model runs in, the same values as loading in float32 and casting after, without a float32 copy of 1.7B weights.
        model.to(dtype=self.dtype)
        state = load_state_dict(folder / "model.safetensors")
        load_into(model, state, what="Qwen3-TTS model")
        del state
        # Fast mode (load option fused_projections): fewer, larger products; not bit-identical on a GPU.
        self.fused_projections = bool(spec.options.get("fused_projections"))
        if self.fused_projections:
            self.talker.model.fuse()
            self.talker.code_predictor.model.fuse()
        model.eval()
        self.model = model

        ctx.check_cancel()
        ctx.progress("loading:speech tokenizer", None)
        codec_config = json.loads((folder / "speech_tokenizer" / "config.json").read_text(encoding="utf-8"))
        codec = SpeechTokenizer(codec_config, ops.default, with_encoder=self.config.model_type == "base")
        codec.to(dtype=self.dtype)  # cast before loading, as the model
        codec_state = load_state_dict(folder / "speech_tokenizer" / "model.safetensors")
        if codec.encoder is None:
            codec_state = {k: v for k, v in codec_state.items() if not k.startswith("encoder.")}
        load_into(codec, codec_state, what="Qwen3-TTS speech tokenizer")
        del codec_state
        codec.eval()
        self.codec = codec

        self.tokenizer = load_bpe(folder, mistral_fix=_mistral_fix(folder))
        # One handle per phase's part: the talker generates (its code predictor runs 15 times per
        # frame); the speaker encoder and the codec's encoder make a voice prompt; the codec's decoder speaks.
        self.h_talker = ModelHandle(self.talker, name=f"{spec.model_id}:talker", load_device=spec.device, kind="sequence")
        self.h_talker.mark(self.talker.code_predictor, "hot")
        self.h_decoder = ModelHandle(codec.decoder, name=f"{spec.model_id}:codec decoder", load_device=spec.device)
        self.h_prompt: list[ModelHandle] = []
        if self.speaker_encoder is not None:
            self.h_prompt.append(ModelHandle(self.speaker_encoder, name=f"{spec.model_id}:speaker encoder", load_device=spec.device))
        if codec.encoder is not None:
            self.h_prompt.append(ModelHandle(codec.encoder, name=f"{spec.model_id}:codec encoder", load_device=spec.device))
        self.handles = [self.h_talker, self.h_decoder, *self.h_prompt]

    # -- description ------------------------------------------------------------------

    def describe(self) -> Description:
        speakers = sorted(self.config.talker.speaker_ids) if self.variant.startswith("custom_voice") else []
        names = {v: k for k, v in LANGUAGE_NAMES.items()}
        languages = sorted(names[n] for n in self.config.languages if n in names)
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=SAMPLE_RATE, speakers=speakers, languages=languages, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        texts = list(task.texts) if getattr(task, "texts", None) else [task.text]
        longest = max(len(t) for t in texts)  # characters: at least as many as BPE tokens
        return self._generation_memory(len(texts), PROMPT_POSITIONS + longest + self._new_frames(longest, 8192))

    @staticmethod
    def _new_frames(text_tokens: int, max_new_tokens: int) -> int:
        """The frames a text of ``text_tokens`` tokens is likely to need at most: ``FRAMES_PER_TOKEN`` each (speech runs at
        ~3–4 frames a token), within ``max_new_tokens``. A generation that runs longer still runs: if its cache then
        outgrows the device, the phase runs again with more weights streamed (``ctx.run``), with the same result."""
        return min(max_new_tokens, FRAMES_PER_TOKEN * text_tokens + 64)

    def _generation_memory(self, rows: int, positions: int) -> int:
        """The talker's KV cache at its largest for ``rows`` rows of ``positions`` positions (prompt and new frames), and
        the working memory (the code predictor's cache, logits, the graphs' buffers: ~100 MB measured on a 3090)."""
        from mizuoto.runtime.lm.cache import peak_bytes

        c = self.config.talker.decoder
        position = 2 * c.num_kv_heads * c.head_dim * self._element_size()
        return peak_bytes(position, c.num_layers, positions, self.kv_cache, rows) + 256 * 2**20

    def _talker_positions(self, input_ids: list[Any], instruct_ids: list[Any] | None, ref_ids: list[Any] | None, prompt: dict[str, Any] | None, max_new_tokens: int) -> int:
        """The longest row's positions: its prompt (instruction, reference text and codes, text, special embeddings)
        and the frames its text is likely to need (``_new_frames``)."""
        longest = 0
        for index, ids in enumerate(input_ids):
            text = int(ids.shape[-1])
            positions = PROMPT_POSITIONS + text + self._new_frames(text, max_new_tokens)
            if instruct_ids is not None and instruct_ids[index] is not None:
                positions += int(instruct_ids[index].shape[-1])
            if ref_ids is not None and ref_ids[index] is not None:
                positions += int(ref_ids[index].shape[-1])
            if prompt is not None and prompt.get("ref_code") is not None and prompt["ref_code"][index] is not None:
                positions += int(prompt["ref_code"][index].shape[0])
            longest = max(longest, positions)
        return longest

    def _codec_memory(self, samples: int) -> int:
        """The codec's activations for ``samples`` of 24 kHz audio (decoded 300 frames at a time)."""
        return min(samples, 300 * 1920) * 4 * 256 + 256 * 2**20

    def _element_size(self) -> int:
        import torch

        return torch.empty((), dtype=self.dtype).element_size()

    # -- voices -----------------------------------------------------------------------

    def _prompt(self, voice: VoiceSpec, ctx: RequestContext) -> list[dict[str, Any]]:
        """Upstream ``create_voice_clone_prompt`` for the voice's first reference."""
        import numpy as np
        import torch

        from mizuoto.engines.qwen3tts.speaker import SPEAKER_RATE, speaker_mel
        from mizuoto.nn.dsp.resample import soxr_resample
        from mizuoto.nn.dsp.wav import read_wav

        if not voice.references:
            raise EngineError("A Base model needs a reference clip", code=codes.INVALID_INPUT)
        ref = voice.references[0]
        wav, sr = read_wav(ref.audio_path)
        transcript = (ref.transcript or "").strip() or None
        x_vector_only = transcript is None
        wav = wav.astype(np.float32)
        # speech codes, from the clip at the tokenizer's rate
        code_wav = wav if sr == self.codec.sample_rate else soxr_resample(wav.astype(np.float32), sr, self.codec.sample_rate).astype(np.float32)
        # speaker embedding, from the clip at 24 kHz
        spk_wav = wav if sr == SPEAKER_RATE else soxr_resample(wav.astype(np.float32), sr, SPEAKER_RATE)
        mels = speaker_mel(torch.from_numpy(spk_wav).unsqueeze(0))

        def encode() -> tuple[torch.Tensor, torch.Tensor]:
            assert self.speaker_encoder is not None
            audio = torch.from_numpy(code_wav).to(self.device).to(self.dtype)
            return self.codec.encode(audio), self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]

        ref_code, spk = ctx.run("voice prompt", self.h_prompt, encode, self._codec_memory(len(code_wav)))
        return [
            {
                "ref_code": None if x_vector_only else ref_code,
                "ref_spk_embedding": spk,
                "x_vector_only_mode": x_vector_only,
                "icl_mode": not x_vector_only,
                "ref_text": transcript,
            }
        ]

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        if self.variant != "base" or not voice.references:
            return None
        import torch

        items = self._prompt(voice, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "prompt.pt"
        torch.save(
            [
                {
                    "ref_code": None if it["ref_code"] is None else it["ref_code"].detach().cpu(),
                    "ref_spk_embedding": it["ref_spk_embedding"].detach().cpu(),
                    "x_vector_only_mode": bool(it["x_vector_only_mode"]),
                    "icl_mode": bool(it["icl_mode"]),
                    "ref_text": it["ref_text"],
                }
                for it in items
            ],
            path,
        )
        return VoiceCache(files={"prompt": str(path)}, meta={"x_vector_only": bool(items[0]["x_vector_only_mode"]), "model_size": self.config.model_size})

    def _cached_prompt(self, voice: VoiceSpec) -> list[dict[str, Any]] | None:
        path = (voice.cache or {}).get("prompt")
        if not path or not Path(path).is_file():
            return None
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
        return [
            {
                "ref_code": None if d["ref_code"] is None else d["ref_code"].to(self.device),
                "ref_spk_embedding": d["ref_spk_embedding"].to(self.device),
                "x_vector_only_mode": d["x_vector_only_mode"],
                "icl_mode": d["icl_mode"],
                "ref_text": d["ref_text"],
            }
            for d in data
        ]

    # -- parameters -------------------------------------------------------------------

    @staticmethod
    def _generate_kwargs(task: SynthesisTask) -> dict[str, Any]:
        kwargs: dict[str, Any] = {name: task.params[name] for name in GENERATE_PARAMS if task.params.get(name) is not None}
        if kwargs.get("temperature") == 0:
            kwargs.pop("temperature")
            kwargs["do_sample"] = False
        o = task.options
        for name in ("subtalker_temperature", "subtalker_top_k", "subtalker_top_p"):
            if o.get(name) is not None:
                kwargs[name] = o[name]
        if o.get("subtalker_do_sample") is not None:
            kwargs["subtalker_dosample"] = bool(o["subtalker_do_sample"])
        if kwargs.get("subtalker_temperature") == 0:
            kwargs.pop("subtalker_temperature")
            kwargs["subtalker_dosample"] = False
        if o.get("non_streaming_mode") is not None:
            kwargs["non_streaming_mode"] = bool(o["non_streaming_mode"])
        return kwargs

    def _merge(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Upstream ``_merge_generate_kwargs``: the request's values, else generation_config.json's, else the hard defaults."""
        merged = {k: v for k, v in kwargs.items() if k not in HARD_DEFAULTS}
        for name, default in HARD_DEFAULTS.items():
            value = kwargs.get(name)
            merged[name] = value if value is not None else self.config.generation.get(name, default)
        return merged

    # -- speech -----------------------------------------------------------------------

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np

        texts = list(task.texts) if task.texts else [task.text]
        language = LANGUAGE_NAMES.get(task.language or "", "auto")
        languages = [language] * len(texts)
        kwargs = self._generate_kwargs(task)
        merged = self._merge(kwargs)
        ctx.applied.update({k: v for k, v in merged.items() if k != "non_streaming_mode"})
        ctx.applied["language"] = language
        if self.fused_projections:
            ctx.applied["fused_projections"] = True
        if "non_streaming_mode" in kwargs:
            ctx.applied["non_streaming_mode"] = kwargs["non_streaming_mode"]
        ctx.progress("generating", None)
        self._validate_languages(languages)
        stream = task.stream and len(texts) == 1
        if self.variant == "base":
            prompt = self._cached_prompt(task.voice) or self._prompt(task.voice, ctx)
            ctx.applied["voice_prompt"] = "x_vector" if prompt[0]["x_vector_only_mode"] else "in_context"
            wavs = self._clone(texts, languages, prompt, kwargs.get("non_streaming_mode", False), merged, ctx, stream)
        elif self.variant.startswith("custom_voice"):
            if not task.voice.preset:
                raise EngineError("A CustomVoice model needs a preset speaker", code=codes.INVALID_INPUT)
            instruct = task.style.instruct if self.variant == "custom_voice" else None
            if instruct:
                ctx.applied["instruct"] = instruct
            wavs = self._custom_voice(texts, task.voice.preset, languages, instruct, kwargs.get("non_streaming_mode", True), merged, ctx, stream)
        else:
            description = (task.voice.description or "").strip()
            if not description:
                raise EngineError("A VoiceDesign model needs a voice description", code=codes.INVALID_INPUT)
            wavs = self._voice_design(texts, description, languages, kwargs.get("non_streaming_mode", True), merged, ctx, stream)
        if stream:  # pieces of one segment as they are decoded, the last one marked final
            pending = None
            for wav in wavs:
                samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
                if samples.size:
                    if pending is not None:
                        yield AudioChunk(samples=pending, sample_rate=SAMPLE_RATE, segment=task.segment)
                    pending = samples
            yield AudioChunk(samples=pending if pending is not None else np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)
            return
        ctx.check_cancel()
        for index, wav in enumerate(wavs):
            samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
            segment = task.segment + index if task.texts else task.segment
            yield AudioChunk(samples=samples, sample_rate=SAMPLE_RATE, segment=segment, final=True)

    def _validate_languages(self, languages: list[str]) -> None:
        supported = {x.lower() for x in self.config.languages}
        bad = [lang for lang in languages if lang is None or str(lang).lower() not in supported]
        if bad:
            raise EngineError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}", code=codes.INVALID_INPUT)

    def _tokenize(self, text: str) -> torch.Tensor:
        import torch

        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)

    @staticmethod
    def _assistant(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _ref(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    @staticmethod
    def _instruct(text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n"

    def _clone(
        self, texts: list[str], languages: list[str], items: list[dict[str, Any]], non_streaming: bool, merged: dict[str, Any], ctx: RequestContext, stream: bool = False
    ) -> Any:
        """The clips' audio (arrays), or with ``stream`` one text's audio as it is decoded (an iterator of arrays)."""
        if len(items) == 1 and len(texts) > 1:
            items = items * len(texts)
        input_ids = [self._tokenize(self._assistant(t)) for t in texts]
        ref_ids = [None if not it["ref_text"] else self._tokenize(self._ref(it["ref_text"])) for it in items]
        prompt = {
            "ref_code": [it["ref_code"] for it in items],
            "ref_spk_embedding": [it["ref_spk_embedding"] for it in items],
            "x_vector_only_mode": [it["x_vector_only_mode"] for it in items],
            "icl_mode": [it["icl_mode"] for it in items],
        }
        args = (None, ref_ids, prompt, languages, None, non_streaming, merged, ctx)
        if stream:
            return self._stream(input_ids, args, prompt["ref_code"][0])
        codes_list = self._generate(input_ids, *args)
        for_decode = []
        for i, c in enumerate(codes_list):
            ref_code = prompt["ref_code"][i]
            for_decode.append(c if ref_code is None else self._cat(ref_code, c))
        wavs = self._decode(for_decode, ctx)
        out = []
        for i, wav in enumerate(wavs):
            ref_code = prompt["ref_code"][i]
            if ref_code is not None:
                ref_len = int(ref_code.shape[0])
                total_len = int(for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                out.append(wav[cut:].cpu().numpy())
            else:
                out.append(wav.cpu().numpy())
        return out

    @staticmethod
    def _cat(ref_code: torch.Tensor, codes_: torch.Tensor) -> torch.Tensor:
        import torch

        return torch.cat([ref_code.to(codes_.device), codes_], dim=0)

    def _custom_voice(
        self,
        texts: list[str],
        speaker: str,
        languages: list[str],
        instruct: str | None,
        non_streaming: bool,
        merged: dict[str, Any],
        ctx: RequestContext,
        stream: bool = False,
    ) -> Any:
        speakers = [speaker] * len(texts)
        supported = {s.lower() for s in self.config.talker.speaker_ids}
        bad = [s for s in speakers if s and s.lower() not in supported]
        if bad:
            raise EngineError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}", code=codes.INVALID_INPUT)
        if self.config.model_size in "0b6":  # upstream: no instruct on the 0.6B CustomVoice model
            instruct = None
        instructs = [instruct] * len(texts) if instruct is not None else [""] * len(texts)
        input_ids = [self._tokenize(self._assistant(t)) for t in texts]
        instruct_ids = [None if not ins else self._tokenize(self._instruct(ins)) for ins in instructs]
        args = (instruct_ids, None, None, languages, speakers, non_streaming, merged, ctx)
        if stream:
            return self._stream(input_ids, args, None)
        return [w.cpu().numpy() for w in self._decode(self._generate(input_ids, *args), ctx)]

    def _voice_design(
        self, texts: list[str], description: str, languages: list[str], non_streaming: bool, merged: dict[str, Any], ctx: RequestContext, stream: bool = False
    ) -> Any:
        input_ids = [self._tokenize(self._assistant(t)) for t in texts]
        instruct_ids = [self._tokenize(self._instruct(description)) for _ in texts]
        args = (instruct_ids, None, None, languages, None, non_streaming, merged, ctx)
        if stream:
            return self._stream(input_ids, args, None)
        return [w.cpu().numpy() for w in self._decode(self._generate(input_ids, *args), ctx)]

    def _stream(self, input_ids: list[torch.Tensor], args: tuple[Any, ...], prefix: torch.Tensor | None) -> Iterator[Any]:
        """One text's audio as it is generated: the talker and the codec's decoder placed together, the frames decoded
        ``STREAM_FIRST`` and then every ``STREAM_EVERY`` at a time (``codec.DecodeStream``). ``prefix`` is the reference's
        codes when cloning in context, decoded with the rest but not returned."""
        from mizuoto.engines.qwen3tts.codec import CHUNK, LEFT_CONTEXT, DecodeStream

        merged, ctx = args[-2], args[-1]
        positions = self._talker_positions(input_ids, args[0], args[1], args[2], int(merged.get("max_new_tokens") or 8192))
        memory = self._generation_memory(1, positions) + self._codec_memory((CHUNK + LEFT_CONTEXT) * 1920)
        ctx.applied["stream"] = "frames"
        with ctx.use([self.h_talker, self.h_decoder], memory):
            decode = DecodeStream(self.codec.decoder, None if prefix is None else prefix.to(self.device))
            wanted = STREAM_FIRST
            for frame in self._generate_codes(input_ids, *args, frames=True):
                decode.push(frame[0])
                if decode.frames - decode.done >= wanted:
                    yield decode.audio().cpu().numpy()
                    wanted = STREAM_EVERY
            yield decode.audio().cpu().numpy()

    def _decode(self, codes_list: list[Any], ctx: RequestContext) -> list[Any]:
        memory = self._codec_memory(max(int(c.shape[0]) for c in codes_list) * 1920 * len(codes_list))
        return ctx.run("codec decoder", [self.h_decoder], lambda: self.codec.decode(codes_list, check=ctx.check_cancel), memory)

    # -- the model's generate -----------------------------------------------------------

    def _generate(self, input_ids: list[torch.Tensor], *args: Any) -> list[torch.Tensor]:
        instruct_ids, ref_ids, prompt, merged, ctx = args[0], args[1], args[2], args[-2], args[-1]
        positions = self._talker_positions(input_ids, instruct_ids, ref_ids, prompt, int(merged.get("max_new_tokens") or 8192))
        return ctx.run("talker", [self.h_talker], lambda: self._generate_codes(input_ids, *args), self._generation_memory(len(input_ids), positions))

    def _generate_codes(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[Any] | None,
        ref_ids: list[Any] | None,
        prompt: dict[str, Any] | None,
        languages: list[str],
        speakers: list[Any] | None,
        non_streaming: bool,
        merged: dict[str, Any],
        ctx: RequestContext,
        frames: bool = False,
    ) -> Any:
        """Upstream ``Qwen3TTSForConditionalGeneration.generate``: build each prompt, then run the talker. Returns each
        row's codes, or with ``frames`` an iterator of the frames as they are drawn ((rows, 16) each)."""
        import torch

        c = self.config.talker
        talker = self.talker
        dev = self.device
        embeds: list[list[torch.Tensor]] = [[] for _ in input_ids]
        spk_embeds = None
        if prompt is not None:
            spk_embeds = [e.to(dev).to(self.dtype) for e in prompt["ref_spk_embedding"]]
        if instruct_ids is not None:
            for index, ids in enumerate(instruct_ids):
                if ids is not None:
                    embeds[index].append(talker.text(ids))
        trailing: list[torch.Tensor] = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        tts_pad = tts_eos = None
        for index, (ids, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if spk_embeds is None:
                if not speaker:
                    speaker_embed = None
                else:
                    if speaker.lower() not in c.speaker_ids:
                        raise EngineError(f"Speaker {speaker} not implemented", code=codes.INVALID_INPUT)
                    speaker_embed = talker.codec(torch.tensor(c.speaker_ids[speaker.lower()], device=dev, dtype=ids.dtype))
            else:
                assert prompt is not None
                speaker_embed = spk_embeds[index] if (prompt["x_vector_only_mode"][index] or prompt["icl_mode"][index]) else None
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in c.language_ids:
                    raise EngineError(f"Language {language} not implemented", code=codes.INVALID_INPUT)
                language_id = c.language_ids[language.lower()]
            if language.lower() in ("chinese", "auto") and speaker and c.speaker_dialects.get(speaker.lower(), False) is not False:
                language_id = c.language_ids[str(c.speaker_dialects[speaker.lower()])]
            special = talker.text(torch.tensor([[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]], device=dev, dtype=ids.dtype))
            tts_bos, tts_eos, tts_pad = special.chunk(3, dim=1)
            if language_id is None:
                prefill = [[c.codec_nothink_id, c.codec_think_bos_id, c.codec_think_eos_id]]
            else:
                prefill = [[c.codec_think_id, c.codec_think_bos_id, language_id, c.codec_think_eos_id]]
            codec0 = talker.codec(torch.tensor(prefill, device=dev, dtype=ids.dtype))
            codec1 = talker.codec(torch.tensor([[c.codec_pad_id, c.codec_bos_id]], device=dev, dtype=ids.dtype))
            if speaker_embed is None:
                codec_in = torch.cat([codec0, codec1], dim=1)
            else:
                codec_in = torch.cat([codec0, speaker_embed.view(1, 1, -1), codec1], dim=1)
            role = talker.text(ids[:, :3])
            body = torch.cat((tts_pad.expand(-1, codec_in.shape[1] - 2, -1), tts_bos), dim=1) + codec_in[:, :-1]
            embed = torch.cat((role, body), dim=1)
            if prompt is not None and prompt["ref_code"] is not None and prompt["icl_mode"][index]:
                assert ref_ids is not None and ref_ids[index] is not None
                ref_id: torch.Tensor = ref_ids[index]
                icl, trail = self._icl_prompt(ids[:, 3:-5], ref_id[:, 3:-2], prompt["ref_code"][index].to(dev), tts_pad, tts_eos, non_streaming)
                embed = torch.cat([embed, icl], dim=1)
            else:
                embed = torch.cat([embed, talker.text(ids[:, 3:4]) + codec_in[:, -1:]], dim=1)
                if non_streaming:
                    embed = embed[:, :-1]
                    embed = torch.cat(
                        [
                            embed,
                            torch.cat((talker.text(ids[:, 3:-5]), tts_eos), dim=1)
                            + talker.codec(torch.tensor([[c.codec_pad_id] * (ids[:, 3:-5].shape[1] + 1)], device=dev, dtype=ids.dtype)),
                            tts_pad + talker.codec(torch.tensor([[c.codec_bos_id]], device=dev, dtype=ids.dtype)),
                        ],
                        dim=1,
                    )
                    trail = tts_pad
                else:
                    trail = torch.cat((talker.text(ids[:, 4:-5]), tts_eos), dim=1)
            embeds[index].append(embed)
            trailing.append(trail)
        assert tts_pad is not None
        joined = [torch.cat([e for e in parts if e is not None], dim=1) for parts in embeds]
        if frames:
            return self._frames(joined, trailing, tts_pad, merged, ctx)
        return self._talk(joined, trailing, tts_pad, merged, ctx)

    def _icl_prompt(
        self, text_id: torch.Tensor, ref_id: torch.Tensor, ref_code: torch.Tensor, tts_pad: torch.Tensor, tts_eos: torch.Tensor, non_streaming: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream ``generate_icl_prompt``: reference text + text, aligned with the reference's codes."""
        import torch

        c = self.config.talker
        talker = self.talker
        text_embed = talker.text(torch.cat([ref_id, text_id], dim=-1))
        text_embed = torch.cat([text_embed, tts_eos], dim=1)
        codec_embed = []
        for i in range(c.num_code_groups):
            if i == 0:
                codec_embed.append(talker.codec(ref_code[:, :1]))
            else:
                codec_embed.append(talker.code_predictor.model.codec_embedding[i - 1](ref_code[:, i : i + 1]))
        codec = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec = torch.cat([talker.codec(torch.tensor([[c.codec_bos_id]], device=self.device, dtype=text_id.dtype)), codec], dim=1)
        text_lens, codec_lens = text_embed.shape[1], codec.shape[1]
        if non_streaming:
            icl = text_embed + talker.codec(torch.tensor([[c.codec_pad_id] * text_lens], device=self.device, dtype=text_id.dtype))
            return torch.cat([icl, codec + tts_pad], dim=1), tts_pad
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec, text_embed[:, codec_lens:]
        text_embed = torch.cat([text_embed] + [tts_pad] * (codec_lens - text_lens), dim=1)
        return text_embed + codec, tts_pad

    def _talk(self, embeds: list[torch.Tensor], trailing: list[torch.Tensor], tts_pad: torch.Tensor, merged: dict[str, Any], ctx: RequestContext) -> list[torch.Tensor]:
        """The talker's generation, every frame of it: each row's codes up to its end of speech."""
        import torch

        frames = list(self._frames(embeds, trailing, tts_pad, merged, ctx))
        if not frames:
            return [torch.empty((0, self.config.talker.num_code_groups), dtype=torch.long, device=self.device) for _ in embeds]
        codes_ = torch.stack(frames, dim=1)
        first = codes_[:, :, 0]
        is_stop = first == self.config.talker.codec_eos_id
        stop_indices = torch.argmax(is_stop.int(), dim=1)
        lengths_out = torch.where(is_stop.any(dim=1), stop_indices, codes_.shape[1])
        return [codes_[i, : int(n)] for i, n in enumerate(lengths_out)]

    def _frames(self, embeds: list[torch.Tensor], trailing: list[torch.Tensor], tts_pad: torch.Tensor, merged: dict[str, Any], ctx: RequestContext) -> Iterator[torch.Tensor]:
        """The talker's generation (transformers' ``_sample`` with upstream's forward), batched with left padding: each
        frame's (rows, 16) codes as it is drawn (a finished row's are its end-of-speech code's)."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        from mizuoto.runtime.lm.cache import new_cache
        from mizuoto.runtime.lm.qwen import mask_for
        from mizuoto.runtime.sampling import INVALID_PROBABILITIES, HfSampling

        c = self.config.talker
        talker = self.talker
        dev = self.device
        lengths = torch.tensor([t.shape[1] for t in embeds])
        inputs = pad_sequence([t.squeeze(0).flip(dims=[0]) for t in embeds], batch_first=True, padding_value=0.0).flip(dims=[1])
        batch, max_len = inputs.shape[0], inputs.shape[1]
        num_pads = max_len - lengths
        padded = bool((num_pads > 0).any())
        attention = (torch.arange(max_len).expand(batch, -1) >= num_pads.unsqueeze(1)).long().to(dev)
        pad_vector = tts_pad.squeeze()
        trailing_rows = [t.squeeze(0) for t in trailing]
        trailing_lengths = [t.shape[0] for t in trailing_rows]
        trailing_t = pad_sequence(trailing_rows, batch_first=True, padding_value=0.0)
        pad_mask = torch.arange(max(trailing_lengths), device=trailing_t.device).expand(len(trailing_lengths), -1) >= torch.tensor(
            trailing_lengths, device=trailing_t.device
        ).unsqueeze(1)
        trailing_t[pad_mask] = pad_vector

        talker_sampling = HfSampling(
            do_sample=bool(merged["do_sample"]),
            temperature=float(merged["temperature"]),
            top_k=int(merged["top_k"]),
            top_p=float(merged["top_p"]),
            repetition_penalty=float(merged["repetition_penalty"]),
            min_new_tokens=2,
            eos_token_id=c.codec_eos_id,
            suppress_tokens=[i for i in range(c.vocab_size - 1024, c.vocab_size) if i != c.codec_eos_id],
        )
        sub_sampling = HfSampling(
            do_sample=bool(merged["subtalker_dosample"]),
            temperature=float(merged["subtalker_temperature"]),
            top_k=int(merged["subtalker_top_k"]),
            top_p=float(merged["subtalker_top_p"]),
        )
        max_new = int(merged["max_new_tokens"])

        # prefill: positions as upstream's get_rope_index (float cumsum; 1 where padded)
        positions = attention.float().cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 1)
        max_position = positions.max(-1, keepdim=True)[0]
        rope_deltas = (max_position + 1 - torch.sum(attention, dim=-1, keepdim=True)) - (1 - attention).sum(dim=-1).unsqueeze(1)
        cache = new_cache(self.kv_cache)
        logits, hidden = talker(inputs, positions, mask_for(attention, max_len, 0, padding="left", padded=padded), cache)
        decoder = self._decoder(batch, talker_sampling, sub_sampling)
        if decoder.graphs is not None:
            ctx.applied["cuda_graphs"] = True

        step = 0
        drawn = 0
        with decoder.run(ctx.generator_for(dev), logits[:, -1, :], hidden[:, -1:, :]):
            while True:
                decoder.sample(blocked=drawn < talker_sampling.min_new_tokens)
                drawn += 1
                unfinished, invalid = decoder.read_status()  # the frame's one wait for the device
                if invalid:
                    raise RuntimeError(INVALID_PROBABILITIES)
                if drawn >= max_new or not unfinished:
                    break
                ctx.check_cancel()
                if step % 25 == 0:
                    ctx.progress("generating", None, f"{step / FRAME_RATE:.1f} s")
                # the next forward: this frame's other codebooks, then the talker on their embeddings
                decoder.trail.copy_(trailing_t[:, step].unsqueeze(1) if step < trailing_t.shape[1] else tts_pad.expand_as(decoder.trail))
                decoder.predict()
                yield decoder.frame.clone()
                step += 1
                past = cache.length
                decoder.positions.copy_(rope_deltas + past)
                attention = torch.cat([attention, attention.new_ones((batch, 1))], dim=-1)
                decoder.talk(mask_for(attention, 1, past, padding="left", padded=padded), cache)
        if decoder.graphs is None and ctx.applied.pop("cuda_graphs", None):
            self._graphs_failed = True
        elif decoder.graphs is not None:
            self.h_talker.replayed(step)  # each step's graphs copied the streamed layers (none when wholly loaded)

    def _decoder(self, batch: int, talker_sampling: Any, sub_sampling: Any) -> Any:
        """The frame loop's decoder: with CUDA graphs (cached, up to two batch sizes or sampling settings) when they may
        be used, else a fresh eager one."""
        from mizuoto.engines.qwen3tts.decode import FrameDecoder
        from mizuoto.runtime.graphs import Graphs, signature, usable

        if not (usable(self.device, self.cuda_graphs) and not self._graphs_failed and self.h_talker.graphable()):
            return FrameDecoder(self.talker, self.config.talker, batch, talker_sampling, sub_sampling, self.kv_cache, None)
        where = signature([self.talker])
        if where != self._graph_signature:  # the weights moved since the graphs were captured
            self._decoders.clear()
            self._graph_signature = where
        key = (batch, talker_sampling.key(), sub_sampling.key())
        decoder = self._decoders.pop(key, None)
        if decoder is None:
            if self._graphs is None:
                self._graphs = Graphs(self.device)
            decoder = FrameDecoder(self.talker, self.config.talker, batch, talker_sampling, sub_sampling, self.kv_cache, self._graphs)
        self._decoders[key] = decoder
        while len(self._decoders) > 2:
            self._decoders.pop(next(iter(self._decoders)))
        return decoder

    def close(self) -> None:
        self._decoders.clear()
        self._graphs = None
        super().close()
