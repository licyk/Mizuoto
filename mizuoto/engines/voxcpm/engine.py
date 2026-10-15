"""VoxCPM, native: VoxCPM-0.5B (``v1``), VoxCPM1.5 (``v1_5``) and VoxCPM2 (``v2``).

What upstream's ``VoxCPM.generate`` / ``generate_streaming`` (``core.py``) does around the model, done the same way:
- The text's line breaks and runs of blanks become one space; with ``normalize`` the text is normalised (``text.py``).
- The reference clip is encoded as ``build_prompt_cache`` does. VoxCPM2 reads it with ``librosa.load`` (soxr) at
  16 kHz, the others with ``torchaudio.load`` + ``torchaudio.functional.resample`` at the VAE's rate. A continuation
  prompt is padded to whole patches on the left, a reference on the right.
- **Retry.** A non-streamed result with ``retry_badcase_ratio_threshold`` or more patches per text token is generated
  again, with the seed plus one, at most ``retry_badcase_max_times`` times. The patch cap is
  ``min(tokens × ratio + 10, max_len)``. Each attempt draws from a fresh generator seeded with its seed, which is
  where upstream's ``torch.manual_seed`` puts the process-wide one. Without a seed, a 32-bit seed is drawn from the
  request's random stream.
- **Streaming.** A streamed request generates once (no retry, as upstream) and yields a chunk per patch. VoxCPM2
  decodes through the VAE's stateful decoder; v1 re-decodes the last three patches and keeps the newest one.

Voice modes:
- VoxCPM2 clones from a reference without a transcript (the reference prefix). With a transcript it uses the
  reference *and* the continuation, as upstream's web demo does ("ultimate cloning"). The option
  ``reference_prefix`` off keeps only the continuation.
- VoxCPM2 designs a voice from ``(description)text`` and takes an instruction the same way; parentheses inside are
  removed, as the demo does.
- VoxCPM 0.5B/1.5 clone by continuation only (a transcript is needed). With no voice, every model speaks zero-shot.

``auto`` precision is upstream's: the config's ``bfloat16`` everywhere but MPS, which gets float32 as
``pick_runtime_dtype`` decides. The VAE always runs in float32.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    import numpy as np
    import torch

    from mizuoto.engines.voxcpm.model import VoxCpmModel
    from mizuoto.engines.voxcpm.text import TextNormalizer, VoxCpmTokenizer
    from mizuoto.engines.voxcpm.vae import AudioVAE
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

V2_LANGUAGES = [
    "ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he", "hi", "id", "it", "ja",
    "km", "ko", "lo", "ms", "no", "pl", "pt", "ru", "es", "sw", "sv", "tl", "th", "tr", "vi",
]  # fmt: skip
LANGUAGES = {"v1": ["zh", "en"], "v1_5": ["zh", "en"], "v2": V2_LANGUAGES}
STREAM_PREFIX = {1: 3, 2: 4}
"""``streaming_prefix_len``: the patches of context the decoder keeps (v1's default 3, v2's 4)."""
CACHE_FORMAT = 1


class VoxCpmEngine(Engine):
    id = "voxcpm"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto.engines.voxcpm.train:VoxCpmTrainer"
    OPTION_SCHEMA = {
        "min_len": OptionSpec("int", 2, 0, 100),
        "retry_badcase": OptionSpec("bool", True),
        "retry_badcase_max_times": OptionSpec("int", 3, 1, 10),
        "retry_badcase_ratio_threshold": OptionSpec("float", 6.0, 1.0, 20.0),
        "reference_prefix": OptionSpec("bool", True),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        from mizuoto.engines.voxcpm.config import architecture, read_config

        try:
            config = read_config(model_dir)
        except (OSError, ValueError):
            return None
        version = architecture(config)
        if version == 2:
            return "v2"
        if version == 1:
            rate = (config.get("audio_vae_config") or {}).get("sample_rate", 16000)
            return "v1_5" if rate == 44100 else "v1"
        return None

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None:
            raise EngineError("VoxCPM needs a model folder", code=codes.MODEL_MISSING)
        variant = self.detect_variant(spec.model_dir)
        if variant is None:
            raise EngineError(f"{spec.model_dir.name} is not a VoxCPM model", code=codes.INVALID_INPUT)
        if variant != spec.variant:
            raise EngineError(f"{spec.model_id} is a {variant} model, not {spec.variant}", {"detected": variant}, code=codes.INVALID_INPUT)
        return LoadedVoxCpm(spec, runtime, ctx)


def runtime_dtype(name: str, configured: str, device: str) -> torch.dtype:
    """An explicit choice, else upstream's: the config's dtype, float32 on MPS for a low-precision one."""
    import torch

    explicit = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name in explicit:
        return explicit[name]
    table = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
    if configured not in table:
        raise EngineError(f"Unsupported dtype {configured!r} in config.json", code=codes.INVALID_INPUT)
    dtype = table[configured]
    if device.split(":", 1)[0] == "mps" and dtype != torch.float32:
        return torch.float32
    return dtype


LORA_FILES = ("lora_weights.safetensors", "lora_weights.ckpt")


def read_lora(folder: Path) -> tuple[Any, Path] | None:
    """A LoRA fine-tune's config and weights file in ``folder`` (upstream's checkpoint layout), else None.

    Upstream's ``VoxCPM(lora_weights_path=…)`` reads ``lora_config.json``'s ``lora_config``; without that file it
    assumes LoRA on the language models and the DiT, rank 8, alpha 16."""
    import json

    from mizuoto.engines.voxcpm.model import LoraConfig

    weights = next((folder / name for name in LORA_FILES if (folder / name).is_file()), None)
    if weights is None:
        return None
    path = folder / "lora_config.json"
    if path.is_file():
        cfg = LoraConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))["lora_config"])
    else:
        cfg = LoraConfig(enable_lm=True, enable_dit=True, enable_proj=False)
    return cfg, weights


def load_lora_weights(model: Any, path: Path) -> None:
    """``load_lora_weights``: each ``lora_*`` tensor copied into its parameter (in the parameter's dtype)."""
    from mizuoto.runtime.loading import load_state_dict, unwrap

    state = unwrap(load_state_dict(path), "state_dict")
    params = dict(model.named_parameters())
    unknown = [k for k in state if k not in params]
    if unknown:
        raise EngineError(f"{path.name} has LoRA weights this model has no place for", {"unexpected": unknown[:20]}, code=codes.INVALID_INPUT)
    for key, value in state.items():
        params[key].data.copy_(value.to(params[key].device))


def control_text(text: str) -> str:
    """A description or instruction as the demo puts it in ``(…)``: its parentheses removed."""
    return re.sub(r"[()（）]", "", text or "").strip()


class LoadedVoxCpm(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.voxcpm.config import load_config
        from mizuoto.engines.voxcpm.model import VoxCpmModel, apply_lora
        from mizuoto.engines.voxcpm.text import VoxCpmTokenizer
        from mizuoto.engines.voxcpm.vae import AudioVAE
        from mizuoto.runtime.compile import compile_module
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import fold_weight_norm, load_into, load_state_dict, unwrap

        folder = spec.model_dir
        assert folder is not None
        self.config = load_config(folder)
        self.variant = spec.variant
        self.device = torch.device(spec.device)
        self.dtype = runtime_dtype(spec.dtype, self.config.dtype, spec.device)

        ctx.progress("loading:weights", None)
        weights = folder / "model.safetensors"
        if not weights.is_file():
            weights = folder / "pytorch_model.bin"
        if not weights.is_file():
            raise EngineError(f"{folder} has neither model.safetensors nor pytorch_model.bin", code=codes.MODEL_MISSING)
        model = VoxCpmModel(self.config)
        lora = read_lora(folder)
        if lora is not None:  # a LoRA fine-tune: the adapters go in before the cast, as upstream builds them
            apply_lora(model, lora[0])
        model = model.to(self.dtype)  # upstream casts before loading: the RoPE tables are rounded with it
        load_into(model, unwrap(load_state_dict(weights), "state_dict"), ignore=["*.lora_A", "*.lora_B"] if lora else (), what="VoxCPM model")
        if lora is not None:
            load_lora_weights(model, lora[1])
        self.lora = lora[0] if lora else None
        self.model: VoxCpmModel = model.eval()

        ctx.check_cancel()
        ctx.progress("loading:vae", None)
        vae_path = folder / "audiovae.safetensors"
        if not vae_path.is_file():
            vae_path = folder / "audiovae.pth"
        if not vae_path.is_file():
            raise EngineError(f"{folder} has no audiovae.safetensors or audiovae.pth", code=codes.MODEL_MISSING)
        vae = AudioVAE(self.config.vae)
        load_into(vae, fold_weight_norm(unwrap(load_state_dict(vae_path), "state_dict")), what="AudioVAE")
        self.vae: AudioVAE = vae.eval()

        self.tokenizer: VoxCpmTokenizer = VoxCpmTokenizer(folder)
        self._normalizer: TextNormalizer | None = None
        # compute.compile: the local DiT, run twice per Euler step of every patch
        self.compiled = compile_module(model.feat_decoder.estimator, runtime.options.compile, f"{spec.model_id}:dit")
        self.sample_rate = self.vae.out_sample_rate
        # the language models run once per patch; the local encoder and DiT many times per patch
        self.h_model = ModelHandle(model, name=f"{spec.model_id}:model", load_device=spec.device, kind="sequence")
        self.h_model.mark(model.feat_decoder, "hot").mark(model.feat_encoder, "hot")
        self.h_vae = ModelHandle(vae, name=f"{spec.model_id}:vae", load_device=spec.device)
        self.handles = [self.h_model, self.h_vae]
        from mizuoto.engines.voxcpm.model import PatchDecoder

        self.cuda_graphs = runtime.options.cuda_graphs
        self._graphs: Any = None
        self._graph_signature: tuple[int, ...] | None = None
        self.decoder = PatchDecoder(model, self._graphs_now)

    def _graphs_now(self) -> Any:
        """The model's CUDA graphs (``compute.cuda_graphs``) while it is wholly on an NVIDIA GPU: new ones when its
        weights moved since the last capture, none otherwise."""
        from mizuoto.runtime.graphs import Graphs, signature, usable

        if not usable(self.device, self.cuda_graphs) or not self.h_model.loaded:
            return None
        where = signature([self.model])
        if where != self._graph_signature:
            self._graphs, self._graph_signature = Graphs(self.device), where
        return self._graphs

    def close(self) -> None:
        self._graphs = None
        self.decoder.graphs, self.decoder.caches = None, None
        super().close()

    # -- description ------------------------------------------------------------------

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=self.sample_rate, languages=LANGUAGES[self.variant], vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        """The two static KV caches (upstream sizes them for ``max_length`` positions) and working memory."""
        return self._generation_memory()

    def _generation_memory(self) -> int:
        lm = self.config.lm
        per_position = 2 * (lm.num_hidden_layers + self.config.residual_lm_num_layers) * lm.num_key_value_heads * lm.head_dim * self._element_size()
        return per_position * self.config.max_length + 512 * 2**20

    def _vae_memory(self, samples: int) -> int:
        """The VAE's activations for ``samples`` of audio (its widest stages hold ~64 float32 channels per sample)."""
        return samples * 4 * 64 + 128 * 2**20

    def _element_size(self) -> int:
        import torch

        return torch.empty((), dtype=self.dtype).element_size()

    # -- voices -----------------------------------------------------------------------

    def _read(self, path: str) -> torch.Tensor:
        """A clip at the VAE's input rate, (1, samples), as upstream reads it."""
        import numpy as np
        import torch

        from mizuoto.nn.dsp.wav import read_wav

        samples, sr = read_wav(path)
        rate = self.vae.sample_rate
        if self.config.version == 2:  # librosa.load(sr=16000, mono=True)
            from mizuoto.nn.dsp.resample import soxr_resample

            y = samples.astype(np.float32)
            if sr != rate:
                y = soxr_resample(y, sr, rate).astype(np.float32)
            return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        audio = torch.from_numpy(np.ascontiguousarray(samples.astype(np.float32))).unsqueeze(0)  # torchaudio.load
        if sr != rate:
            from mizuoto.nn.dsp.resample import resample

            audio = resample(audio, sr, rate)
        return audio

    def _latents(self, voice: VoiceSpec, ctx: RequestContext) -> dict[str, Any]:
        """``build_prompt_cache``'s latents: ``prompt`` (left-padded, with its transcript) and, for VoxCPM2, ``ref``."""
        from mizuoto.engines.voxcpm.model import latent_patches

        ref = voice.references[0]
        transcript = (ref.transcript or "").strip()
        audio = self._read(ref.audio_path)

        def encode() -> dict[str, Any]:
            out: dict[str, Any] = {"prompt_text": transcript or None, "prompt": None, "ref": None}
            if self.config.version == 2:
                out["ref"] = latent_patches(self.vae, audio, self.config.patch_size, "right", self.device)
            if transcript:
                out["prompt"] = latent_patches(self.vae, audio, self.config.patch_size, "left", self.device)
            return out

        return ctx.run("vae", [self.h_vae], encode, self._vae_memory(audio.shape[-1]))

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        if not voice.references:
            return None
        import torch

        latents = self._latents(voice, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "prompt.pt"
        torch.save({"format": CACHE_FORMAT, **latents}, path)
        return VoiceCache(files={"prompt": str(path)}, meta={"format": CACHE_FORMAT, "transcript": latents["prompt_text"] is not None})

    def _cached(self, voice: VoiceSpec) -> dict[str, Any] | None:
        path = (voice.cache or {}).get("prompt")
        if not path or not Path(path).is_file():
            return None
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
        return data if data.get("format") == CACHE_FORMAT else None

    # -- speech -----------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        if self._normalizer is None:
            from mizuoto.engines.voxcpm.text import TextNormalizer

            self._normalizer = TextNormalizer()
        return self._normalizer.normalize(text)

    def _text(self, task: SynthesisTask, text: str) -> str:
        controls = []
        if self.config.version == 2:
            controls = [c for c in (control_text(task.voice.description or ""), control_text(task.style.instruct or "")) if c]
        if controls:
            text = f"({', '.join(controls)}){text}"
        return re.sub(r"\s+", " ", text.replace("\n", " "))

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        from mizuoto.engines.voxcpm.model import Generation, build_inputs

        p, o = task.params, task.options
        steps = int(p.get("num_steps") or 10)
        cfg = float(p["guidance_scale"]) if p.get("guidance_scale") is not None else 2.0
        normalize = bool(p.get("text_normalization") or False)
        max_len = int(p.get("max_new_tokens") or 4096)
        min_len = int(o["min_len"]) if o.get("min_len") is not None else 2
        retry = bool(o["retry_badcase"]) if o.get("retry_badcase") is not None else True
        retry_times = int(o.get("retry_badcase_max_times") or 3)
        ratio = float(o.get("retry_badcase_ratio_threshold") or 6.0)
        reference_prefix = bool(o["reference_prefix"]) if o.get("reference_prefix") is not None else True
        if task.stream and retry:
            retry = False  # upstream: no retry in streaming mode
        seed = ctx.seed if ctx.seed is not None else ctx.rng.getrandbits(32)

        text = self._text(task, task.text)
        if not text.strip():
            raise EngineError("Nothing to say: the text is empty", code=codes.INVALID_INPUT)
        if normalize:
            text = self._normalize(text)
        prompt_text, prompt_feat, ref_feat, mode = None, None, None, "zero_shot"
        if task.voice.references:
            latents = self._cached(task.voice) or self._latents(task.voice, ctx)
            if self.config.version == 1 and latents["prompt"] is None:
                raise EngineError(
                    "VoxCPM 0.5B and 1.5 clone by continuation: the reference clip's transcript is needed",
                    {"capability": "clone_no_transcript"},
                    code=codes.CAPABILITY_NOT_SUPPORTED,
                )
            if latents["prompt"] is not None:
                prompt_text, prompt_feat = latents["prompt_text"], latents["prompt"]
                mode = "continuation"
            if self.config.version == 2 and (latents["prompt"] is None or reference_prefix):
                ref_feat = latents["ref"]
                mode = "reference" if prompt_feat is None else "ref_continuation"
        ctx.applied.update(
            {
                "inference_timesteps": steps,
                "cfg_value": cfg,
                "normalize": normalize,
                "max_len": max_len,
                "min_len": min_len,
                "retry_badcase": retry,
                "retry_badcase_max_times": retry_times,
                "retry_badcase_ratio_threshold": ratio,
                "prompt_mode": mode,
                "dtype": str(self.dtype).removeprefix("torch."),
            }
        )
        cfgv = self.config
        tokens, text_mask, feat, audio_mask = build_inputs(self.tokenizer, text, prompt_text, prompt_feat, ref_feat, cfgv.patch_size, cfgv.vae.latent_dim)
        tokens, text_mask, audio_mask = tokens.unsqueeze(0).to(self.device), text_mask.unsqueeze(0).to(self.device), audio_mask.unsqueeze(0).to(self.device)
        feat = feat.unsqueeze(0).to(self.device).to(self.dtype)
        target_length = len(self.tokenizer(text))
        cap = min(int(target_length * ratio + 10), max_len)
        prefix_len = STREAM_PREFIX[cfgv.version]

        def generation(attempt_seed: int) -> Generation:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(attempt_seed)
            return Generation(
                self.model,
                tokens,
                text_mask,
                feat,
                audio_mask,
                min_len=min_len,
                max_len=cap,
                timesteps=steps,
                cfg_value=cfg,
                streaming=task.stream,
                prefix_len=prefix_len,
                generator=gen,
                cache_dtype=self.dtype,
                check=ctx.check_cancel,
                estimator=self.compiled,
                decoder=self.decoder,
            )

        ctx.progress("generating", None)
        if task.stream:
            ctx.applied["seed"] = seed
            # each patch is decoded as it comes: the language model and the VAE together
            with ctx.use([self.h_model, self.h_vae], self._generation_memory() + self._vae_memory(8 * cfgv.patch_size * self.vae.chunk_size)):
                yield from self._stream(generation(seed), task.segment, ctx)
            self._report_graphs(ctx)
            return

        def attempts_() -> tuple[Any, int, int]:
            """Upstream's retry loop over bad cases: (latent, context patches, the seed that gave it); each attempt
            draws from a generator of its own seed."""
            attempt_seed, attempts = seed, 0
            latent, context_len, used = None, 0, seed
            while attempts < retry_times:
                used = attempt_seed
                run = generation(attempt_seed).__iter__()
                try:
                    latent, generated, context_len = next(run)
                finally:
                    run.close()
                if retry and generated.shape[0] >= target_length * ratio:
                    ctx.log.info("VoxCPM: bad case (%.1f patches per token), retrying with seed %d", generated.shape[0] / max(target_length, 1), attempt_seed + 1)
                    attempts += 1
                    attempt_seed += 1
                    continue
                break
            return latent, context_len, used

        latent, context_len, used = ctx.run("language model", [self.h_model], attempts_, self._generation_memory())
        self._report_graphs(ctx)
        assert latent is not None
        ctx.applied["seed"] = used
        if used != seed:
            ctx.applied["retries"] = used - seed  # generated again this many times (upstream's last_successful_seed)
        ctx.check_cancel()
        ctx.progress("decoding", None)
        from mizuoto.nn.dsp.tiles import tiled_1d

        tiled = ("in tiles of 256 latent frames", lambda: tiled_1d(self.vae.decode, latent.to(torch.float32), self.vae.decode_chunk_size))
        audio = ctx.run("vae", [self.h_vae], lambda: self.vae.decode(latent.to(torch.float32)), self._vae_memory(latent.shape[-1] * self.vae.decode_chunk_size), fallback=tiled)
        if cfgv.version == 2:
            if context_len > 0:
                audio = audio[..., cfgv.patch_size * self.vae.decode_chunk_size * context_len :]
        elif int(audio_mask.sum().item()) > 0:
            audio = audio[..., cfgv.patch_size * self.vae.chunk_size * (prefix_len - 1) :]
        samples = audio.squeeze(1).cpu()[0].numpy().astype(np.float32)
        yield AudioChunk(samples=samples, sample_rate=self.sample_rate, segment=task.segment, final=True)

    def _report_graphs(self, ctx: RequestContext) -> None:
        if self.decoder.graphed and self.compiled is None and not self.decoder.failed:
            ctx.applied["cuda_graphs"] = True

    def _stream(self, generation: Any, segment: int, ctx: RequestContext) -> Iterator[AudioChunk]:
        import torch

        pending: np.ndarray | None = None
        patch_len = self.config.patch_size * self.vae.chunk_size

        def chunks() -> Iterator[np.ndarray]:
            if self.config.version == 2:
                with self.vae.streaming():
                    for latent, _, _ in generation:
                        yield self.vae.decode(latent.to(torch.float32)).squeeze(1).cpu()[0].numpy()
            else:
                for latent, _, _ in generation:
                    yield self.vae.decode(latent.to(torch.float32))[..., -patch_len:].squeeze(1).cpu()[0].numpy()

        for samples in chunks():
            if pending is not None:
                yield AudioChunk(samples=pending, sample_rate=self.sample_rate, segment=segment)
            pending = samples.astype("float32")
        if pending is not None:
            yield AudioChunk(samples=pending, sample_rate=self.sample_rate, segment=segment, final=True)
