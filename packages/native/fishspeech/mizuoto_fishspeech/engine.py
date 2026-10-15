"""Fish Speech S2 Pro, native: cloning from one or several clips, inline tags, dialogue.

The engine does what Mizuoto's v1 adapter did through upstream's ``TTSInferenceEngine`` and
``launch_thread_safe_queue``, without the daemon thread and its 32k-position caches:
- each reference is read, resampled to 44.1 kHz (``torchaudio.transforms.Resample``) and encoded by the codec;
  the codes are cached with the voice (upstream keeps them in memory by the clip's hash);
- the request's seed seeds the generator every draw comes from (upstream seeds the global RNG);
- text is generated in batches of speaker turns (``generate.py``), each batch decoded by the codec under
  autocast in the model's precision, as ``get_audio_segment``; streaming sends each batch's audio as it is done.

Upstream's ``normalize`` request field reaches nothing in its inference path; the option is accepted and has
no effect here either. ``repetition_penalty`` is likewise unused by S2's sampler.
"""

from __future__ import annotations

import itertools
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

SAMPLE_RATE = 44100
CODES_FORMAT_VERSION = 1
PROMPT_POSITIONS = 64
"""A prompt's positions besides its texts and reference codes (the conversation's role and special tokens)."""


class FishSpeechEngine(Engine):
    id = "fishspeech"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto_fishspeech.train:FishSpeechTrainer"
    OPTION_SCHEMA = {
        "chunk_length": OptionSpec("int", 200, 100, 1000),
        "normalize": OptionSpec("bool", True),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        if not ((model_dir / "codec.pth").is_file() and (model_dir / "config.json").is_file()):
            return None
        try:
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        from mizuoto_fishspeech.model import is_s2

        return "s2_pro" if is_s2(config) else None

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or self.detect_variant(spec.model_dir) is None:
            raise EngineError(f"{spec.model_id} is not a Fish Speech S2 model", code=codes.INVALID_INPUT)
        return LoadedFishSpeech(spec, ctx, runtime.options.cuda_graphs)


def _dtype(name: str, device: str) -> torch.dtype:
    """As v1: bf16 on a GPU (upstream's choice), fp32 on the CPU (bf16 is emulated there, far slower); an explicit choice wins."""
    import torch

    if name == "fp16":
        return torch.float16
    if name == "fp32" or (name == "auto" and not device.startswith("cuda")):
        return torch.float32
    return torch.bfloat16


def codec_config(folder: Path) -> Any:
    """The codec's architecture: S2's (``modded_dac_vq.yaml``; the checkpoint ships no configuration of its own)."""
    from mizuoto_fishspeech.codec import CodecConfig

    return CodecConfig()


def _weights(folder: Path) -> dict[str, Any]:
    """The language model's weights as ``BaseTransformer.from_pretrained`` finds them: sharded or single safetensors
    (S2's own), else ``model.pth`` (what upstream's ``save_pretrained`` writes: a fine-tuned model)."""
    from mizuoto.runtime.loading import load_state_dict, unwrap

    index = folder / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    elif (folder / "model.safetensors").is_file() or not (folder / "model.pth").is_file():
        shards = ["model.safetensors"]
    else:
        state = unwrap(load_state_dict(folder / "model.pth"), "state_dict")
        if state and next(iter(state)).startswith("model."):
            state = {k.replace("model.", ""): v for k, v in state.items()}
        return {k: v for k, v in state.items() if "audio_" not in k}
    state = {}
    for shard in shards:
        state.update(load_state_dict(folder / shard))
    return state


class LoadedFishSpeech(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext, cuda_graphs: bool = True) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.runtime import ops
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict, unwrap
        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto_fishspeech.codec import IGNORED, FishCodec, codec_state
        from mizuoto_fishspeech.model import DualARTransformer, ModelArgs, remap_keys

        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        self.dtype = _dtype(spec.dtype, spec.device)
        self.tokenizer = load_bpe(folder)
        config = ModelArgs.from_config(json.loads((folder / "config.json").read_text(encoding="utf-8")))
        # upstream takes the semantic range from the tokenizer (FishTokenizer) over the config's
        begin, end = self.tokenizer.token_to_id("<|semantic:0|>"), self.tokenizer.token_to_id("<|semantic:4095|>")
        config.semantic_begin_id, config.semantic_end_id = begin, end

        ctx.progress("loading:language model", None)
        model = DualARTransformer(config, ops.default).to(dtype=self.dtype)
        state = remap_keys(_weights(folder))
        load_into(model, state, what="Fish Speech language model")
        del state
        self.model = model.eval()

        ctx.check_cancel()
        ctx.progress("loading:codec", None)
        codec = FishCodec(codec_config(folder), ops.default)
        load_into(codec, codec_state(unwrap(load_state_dict(folder / "codec.pth"), "state_dict")), ignore=IGNORED, what="Fish Speech codec")
        self.codec = codec.to(dtype=self.dtype).eval()
        self.resamplers: dict[int, Any] = {}
        self._resample_cls = Resample
        # the slow model runs once per frame, its fast transformer 10 times per frame; the codec encodes references and
        # decodes each batch of frames
        self.h_model = ModelHandle(model, name=f"{spec.model_id}:language model", load_device=spec.device, kind="sequence")
        self.h_model.mark(model.fast_layers, "hot")
        self.h_codec = ModelHandle(codec, name=f"{spec.model_id}:codec", load_device=spec.device)
        self.handles = [self.h_model, self.h_codec]
        self.cuda_graphs = cuda_graphs
        self._decoder: Any = None
        self._graphs: Any = None
        self._graph_signature: tuple[int, ...] | None = None

    def _graphs_now(self) -> Any:
        """The language model's CUDA graphs (``compute.cuda_graphs``) while it is wholly on an NVIDIA GPU: new ones
        when its weights moved since the last capture, none otherwise."""
        from mizuoto.runtime.graphs import Graphs, signature, usable

        if not usable(self.device, self.cuda_graphs) or not self.h_model.loaded:
            return None
        where = signature([self.model])
        if where != self._graph_signature:
            self._graphs, self._graph_signature = Graphs(self.device), where
        return self._graphs

    def close(self) -> None:
        self._decoder = self._graphs = None
        super().close()

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=SAMPLE_RATE, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        from mizuoto_fishspeech.generate import Sampling

        return self._generation_memory(PROMPT_POSITIONS + 3 * len(task.text) + Sampling().max_new_tokens)

    def _generation_memory(self, positions: int) -> int:
        """The slow model's keys and values for ``positions`` positions (references, text, the frames so far and this
        batch's), grown by concatenation, one layer's repeated per head group at each step, and working memory."""
        from mizuoto.runtime.lm.cache import peak_bytes

        c = self.model.config
        element = self.model.embeddings.weight.element_size()
        positions = min(positions, c.max_seq_len)
        return 512 * 2**20 + peak_bytes(2 * c.n_local_heads * c.head_dim * element, c.n_layer, positions) + positions * 2 * c.n_head * c.head_dim * element

    def _codec_memory(self, samples: int) -> int:
        """The codec's activations for ``samples`` of 44.1 kHz audio (~96 float32 channels a sample at its widest)."""
        return samples * 4 * 96 + 256 * 2**20

    # -- references -------------------------------------------------------------------

    def _encode_audio(self, samples: Any, rate: int, ctx: RequestContext) -> torch.Tensor:
        """``encode_reference``: (codebooks, frames) of mono float32 samples at ``rate``."""
        import torch

        wav = torch.from_numpy(samples)[None]
        if rate != SAMPLE_RATE:
            resampler = self.resamplers.get(rate)
            if resampler is None:
                resampler = self.resamplers[rate] = self._resample_cls(rate, SAMPLE_RATE)
            wav = resampler(wav)
        audio = wav.squeeze().to(self.device)[None, None, :]
        lengths = torch.tensor([audio.shape[2]], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            return ctx.run("codec", [self.h_codec], lambda: self.codec.encode(audio.to(self.dtype), lengths)[0][0], self._codec_memory(audio.shape[2]))

    def _reference_codes(self, voice: VoiceSpec, ctx: RequestContext) -> tuple[list[torch.Tensor], list[str]]:
        import torch

        from mizuoto.nn.dsp.wav import read_wav

        cached = (voice.cache or {}).get("codes")
        if cached and Path(cached).is_file():
            data = torch.load(cached, map_location="cpu", weights_only=True)
            if data.get("format_version") == CODES_FORMAT_VERSION:
                return list(data["codes"]), list(data["texts"])
        prompt_codes, prompt_texts = [], []
        for ref in voice.references:
            transcript = (ref.transcript or "").strip()
            if not transcript:
                raise EngineError("Fish Speech needs each reference clip's transcript", {"capability": "clone_no_transcript"}, code=codes.CAPABILITY_NOT_SUPPORTED)
            ctx.check_cancel()
            samples, rate = read_wav(ref.audio_path)
            prompt_codes.append(self._encode_audio(samples, rate, ctx).cpu())
            prompt_texts.append(transcript)
        return prompt_codes, prompt_texts

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        import torch

        if not voice.references:
            return None
        prompt_codes, prompt_texts = self._reference_codes(voice, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "codes.pt"
        torch.save({"format_version": CODES_FORMAT_VERSION, "codes": [c.clone() for c in prompt_codes], "texts": prompt_texts}, str(path))
        return VoiceCache(files={"codes": str(path)})

    # -- speech -----------------------------------------------------------------------

    def _decode(self, frames: torch.Tensor, ctx: RequestContext) -> Any:
        """``get_audio_segment``: the codes through the codec under autocast in the model's precision."""
        import torch

        frames = frames.to(self.device)
        autocast = torch.autocast(self.device.type, dtype=self.dtype) if self.dtype != torch.float32 else torch.autocast(self.device.type, enabled=False)

        def decode() -> torch.Tensor:
            with autocast:
                return self.codec.from_indices(frames[None])[0].squeeze()

        with torch.inference_mode():
            audio = ctx.run("codec", [self.h_codec], decode, self._codec_memory(frames.shape[-1] * 2048))
        return audio.float().cpu().numpy().reshape(-1)

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        from mizuoto_fishspeech.generate import Sampling, generate_long

        if not task.voice.references:
            raise EngineError("Fish Speech clones a voice: it needs a reference clip", code=codes.INVALID_INPUT)
        prompt_codes, prompt_texts = self._reference_codes(task.voice, ctx)
        p, o = task.params, task.options
        sampling = Sampling()
        if p.get("temperature") is not None:
            sampling.temperature = float(p["temperature"])
        if p.get("top_p") is not None:
            sampling.top_p = float(p["top_p"])
        if p.get("max_new_tokens") is not None:
            sampling.max_new_tokens = int(p["max_new_tokens"])
        chunk_length = int(o.get("chunk_length") or 200)
        # what v1 recorded: the request fields it set (upstream's ServeTTSRequest), then the parameters and options given
        ctx.applied.update({"use_memory_cache": "on", "streaming": bool(task.stream)})
        for name in ("temperature", "top_p", "repetition_penalty", "max_new_tokens", "seed"):
            if p.get(name) is not None:
                ctx.applied[name] = float(p[name]) if name in ("temperature", "top_p", "repetition_penalty") else int(p[name])
        for name in ("chunk_length", "normalize"):
            if o.get(name) is not None:
                ctx.applied[name] = o[name]
        ctx.applied["references"] = len(prompt_codes)

        def progress(batch: int, batches: int, done: int, total: int) -> None:
            if done % 16 == 0:
                ctx.progress("generating", (batch + done / max(total, 1)) / batches)

        if self._decoder is None:
            from mizuoto_fishspeech.generate import IM_END, Decoder

            self._decoder = Decoder(self.model, self.tokenizer.token_to_id(IM_END), self._graphs_now)
        decoder = self._decoder
        segments = []
        with torch.inference_mode():
            batches = generate_long(
                self.model,
                self.tokenizer,
                task.text,
                prompt_texts,
                prompt_codes,
                sampling,
                chunk_length,
                ctx.generator_for(self.device),
                ctx.check_cancel,
                progress,
                decoder,
            )
            # every batch's prompt holds the references, the text so far and the frames before it: a batch's positions
            # stay below these, its new frames below max_new_tokens
            prompt = PROMPT_POSITIONS + sum(int(p.shape[-1]) for p in prompt_codes) + sum(len(self.tokenizer.encode(t)) for t in [*prompt_texts, task.text])
            done = 0
            for index in itertools.count():
                positions = prompt + 16 * index + done + sampling.max_new_tokens
                with ctx.use([self.h_model], self._generation_memory(positions)):  # each batch's frames (the generator resumes here)
                    frames = next(batches, None)
                if frames is None:
                    break
                done += int(frames.shape[-1])
                ctx.check_cancel()
                audio = np.clip(self._decode(frames, ctx), -1.0, 1.0)
                if task.stream:
                    yield AudioChunk(samples=audio.astype(np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=False)
                segments.append(audio)
        if decoder.graphed and not decoder.failed:
            ctx.applied["cuda_graphs"] = True
        ctx.check_cancel()
        if not segments:
            raise EngineError("Fish Speech produced no speech for the text", {"text": task.text[:200]}, code=codes.ENGINE_ERROR)
        if task.stream:
            yield AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)
        else:
            yield AudioChunk(samples=np.concatenate(segments).astype(np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        import numpy as np
        import torch

        from mizuoto.nn.dsp.wav import read_wav, write_wav

        if op == "codec_encode":
            try:
                samples, rate = read_wav(payload["audio_path"])
            except ValueError as exc:
                raise EngineError(f"codec_encode reads WAV files: {exc}", {"hint": "convert the clip to WAV first"}, code=codes.INVALID_INPUT) from exc
            tokens = self._encode_audio(samples, rate, ctx)
            out = Path(payload["output_path"])
            np.save(out, tokens.cpu().numpy())
            return {"codes": str(out), "shape": list(tokens.shape)}
        if op == "codec_decode":
            frames = torch.from_numpy(np.load(payload["codes_path"]))
            audio = self._decode(frames, ctx)
            write_wav(payload["output_path"], audio, SAMPLE_RATE)
            return {"audio": payload["output_path"], "samples": int(audio.shape[-1]), "sample_rate": SAMPLE_RATE}
        return super().call(op, payload, ctx)
