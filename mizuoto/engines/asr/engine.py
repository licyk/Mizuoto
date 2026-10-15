"""Speech recognition with Whisper, native.

It reproduces Mizuoto's v1 adapter, transformers 4.57.3's ``pipeline("automatic-speech-recognition")`` called with
``chunk_length_s=30, batch_size=4, return_timestamps=True`` and ``generate_kwargs={"task": "transcribe",
"language": …}``:
- the WAV read as soundfile reads it (channels averaged), resampled to 16 kHz by ``soxr.resample`` (its HQ default);
- without a language, ``detect_language`` on the first 30 s (one decoder step over the language tokens);
- the audio cut in 30 s windows overlapping by 5 s on each side (one window up to 30 s), each window's features as
  ``WhisperFeatureExtractor`` computes them, four windows per ``generate`` call;
- ``generate``: the prompt ``<|startoftranscript|><|lang|><|transcribe|>``, timestamps on, **beam search with 5
  beams** — the pipeline's default generation config (``num_beams=5``) wins over the model's, so v1's
  ``beam_size`` 1 decoded with 5 beams; a larger ``beam_size`` sets the beams; the windows' seek loop;
- the windows' tokens merged into text and timestamped segments by ``_decode_asr``.

``call("transcribe", {"path", "language", "options"})`` returns ``{text, language, segments, duration}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, Unsupported

if TYPE_CHECKING:
    import numpy as np
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import AudioChunk, SynthesisTask
    from mizuoto.runtime.runtime import Runtime

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
BATCH_SIZE = 4
"""The v1 pipeline's ``batch_size``: windows decoded together."""
PIPELINE_BEAMS = 5
"""``AutomaticSpeechRecognitionPipeline._default_generation_config.num_beams``."""


def detect(model_dir: Path) -> str | None:
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return "whisper" if isinstance(config, dict) and config.get("model_type") == "whisper" else None


def _dtype(name: str, device: str) -> torch.dtype:
    """As v1: bf16 on a GPU that has it (else fp16), fp32 on the CPU; an explicit choice wins."""
    import torch

    explicit = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name in explicit:
        return explicit[name]
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class AsrEngine(Engine):
    id = "asr"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    OPTION_SCHEMA = {
        "beam_size": OptionSpec("int", 1, 1, 10),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        return detect(model_dir)

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or detect(spec.model_dir) != "whisper":
            raise EngineError(f"{spec.model_id} is not a Whisper model", code=codes.INVALID_INPUT)
        return LoadedAsr(spec, ctx)


class LoadedAsr(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.asr.decoding import GenerationSettings
        from mizuoto.engines.asr.model import load_whisper
        from mizuoto.engines.asr.text import WhisperText
        from mizuoto.nn.dsp.mel import hf_whisper_filters
        from mizuoto.runtime.handle import ModelHandle

        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        self.dtype = _dtype(spec.dtype, spec.device)
        ctx.progress("loading:weights", None)
        try:
            self.generation = GenerationSettings.load(folder)
            self.model = load_whisper(folder, self.dtype)  # on the CPU; the model manager places it
        except (KeyError, ValueError) as e:
            raise EngineError(f"{spec.model_id} is not a usable Whisper model: {e}", code=codes.INVALID_INPUT) from e
        ctx.check_cancel()
        self.text = WhisperText(folder)
        preprocessor = folder / "preprocessor_config.json"
        pre = json.loads(preprocessor.read_text(encoding="utf-8")) if preprocessor.is_file() else {}
        self.n_samples = int(pre.get("n_samples", CHUNK_SECONDS * SAMPLE_RATE))
        self.filters = hf_whisper_filters(int(pre.get("feature_size", self.model.config.num_mel_bins)))
        # the decoder runs once per token and beam, the encoder once per window
        self.h_model = ModelHandle(self.model, name=f"{spec.model_id}:model", load_device=spec.device, kind="sequence")
        self.h_model.mark(self.model.decoder, "hot")
        self.handles = [self.h_model]

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=SAMPLE_RATE, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        # the encoder's activations for a batch of four windows and the beams' caches
        c = self.model.config
        element = self.model.encoder.conv1.weight.element_size()
        encoder = BATCH_SIZE * c.max_source_positions * max(c.encoder_ffn_dim, 4 * c.d_model) * element
        cross = BATCH_SIZE * 10 * c.decoder_layers * 2 * c.max_source_positions * c.d_model * element
        return encoder + cross + 256 * 2**20

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        raise Unsupported("Speech recognition does not synthesize")

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        if op != "transcribe":
            return super().call(op, payload, ctx)
        beam = int((payload.get("options") or {}).get("beam_size") or 1)
        samples = self._read(payload["path"])
        beams = PIPELINE_BEAMS if beam <= 1 else beam
        language = payload.get("language")
        return ctx.run(
            "transcribe",
            self.handles,
            lambda: self.transcribe(samples, language, beams, ctx),
            self.memory_required(None),
            fallback=("one 30 s window at a time", lambda: self.transcribe(samples, language, beams, ctx, batch_size=1)),
        )

    @staticmethod
    def _read(path: str) -> np.ndarray:
        import soxr

        from mizuoto.nn.dsp.wav import read_wav

        try:
            samples, rate = read_wav(path)
        except (OSError, ValueError, EOFError) as e:
            raise EngineError(f"Cannot read {Path(path).name}: {e}", {"path": path}, code=codes.INVALID_INPUT) from e
        if rate != SAMPLE_RATE:
            samples = soxr.resample(samples, rate, SAMPLE_RATE)
        return samples

    # -- the pipeline ---------------------------------------------------------------------

    def features(self, samples: np.ndarray) -> torch.Tensor:
        """``WhisperFeatureExtractor(samples, sampling_rate=16000, return_tensors="pt")``: padded or cut to 30 s."""
        import numpy as np
        import torch

        from mizuoto.nn.dsp.mel import hf_whisper_log_mel

        x = np.asarray(samples, dtype=np.float32)[: self.n_samples]
        padded = np.zeros((1, self.n_samples), dtype=np.float32)
        padded[0, : x.shape[0]] = x
        return hf_whisper_log_mel(torch.from_numpy(padded), self.filters)

    def detect_language(self, features: torch.Tensor) -> tuple[str, torch.Tensor]:
        """v1's ``_detect_language``: the language code and the encoder's output for these features."""
        from mizuoto.engines.asr.decoding import detect_language

        if not self.generation.lang_to_id:
            return "en", self.model.encoder(features.to(self.device, self.dtype))
        token, hidden = detect_language(self.model, features.to(self.device, self.dtype), self.generation)
        return self.text.token(int(token.reshape(-1)[0])).strip("<|>") or "en", hidden

    def transcribe(self, samples: np.ndarray, language: str | None, num_beams: int, ctx: RequestContext, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
        import torch

        from mizuoto.engines.asr.decoding import generate, prompt_tokens
        from mizuoto.engines.asr.text import chunk_bounds, decode_asr, language_id

        duration = len(samples) / SAMPLE_RATE
        first = self.features(samples[: self.n_samples])
        detected: torch.Tensor | None = None
        if not language:
            ctx.progress("detecting language", None)
            language, detected = self.detect_language(first)
        ctx.check_cancel()
        lang_id = None
        if self.generation.is_multilingual and self.generation.lang_to_id:
            try:
                lang_id = language_id(language, self.generation.lang_to_id)
            except ValueError as e:
                raise EngineError(str(e), {"language": language}, code=codes.INVALID_INPUT) from e
        prompt = prompt_tokens(self.generation, lang_id)

        chunks = chunk_bounds(len(samples), SAMPLE_RATE, CHUNK_SECONDS)
        outputs: list[tuple[list[int], tuple[float, float, float]]] = []
        for b in range(0, len(chunks), batch_size):
            batch = chunks[b : b + batch_size]
            ctx.progress("transcribing", b / len(chunks))
            features = [first if c.start == 0 and c.stop == min(len(samples), self.n_samples) else self.features(samples[c.start : c.stop]) for c in batch]
            stacked = torch.cat(features, dim=0).to(self.device, self.dtype)
            # the detection's encoder output is the generation's when the audio is one window (same input, same batch)
            reuse = detected if len(batch) == 1 and b == 0 else None
            tokens = generate(self.model, stacked, prompt, self.generation, num_beams, ctx.check_cancel, encoder_hidden=reuse)
            outputs += [(t, (c.stride[0] / SAMPLE_RATE, c.stride[1] / SAMPLE_RATE, c.stride[2] / SAMPLE_RATE)) for t, c in zip(tokens, batch)]
        ctx.progress("transcribing", 1.0)
        time_precision = CHUNK_SECONDS / self.model.config.max_source_positions
        text, segments = decode_asr(self.text, outputs, self.generation.timestamp_begin, time_precision)
        return {
            "text": text.strip(),
            "language": language,
            "segments": [
                {"start": float(s.timestamp[0] or 0.0), "end": float(s.timestamp[1] if s.timestamp[1] is not None else duration), "text": s.text.strip()} for s in segments
            ],
            "duration": duration,
        }
