"""The ``paraformer`` variant of the ``asr`` engine: FunASR's ``AutoModel(model=paraformer, vad_model=fsmn-vad,
punc_model=ct-punc).generate(input=path)``, as GPT-SoVITS's ``funasr_asr.py`` transcribes Chinese, ported (FunASR
1.4.21; ``inference_with_vad`` as it runs on the CPU):

1. the clip read as ``torchaudio.load`` reads it (channels averaged) and resampled to 16 kHz by
   ``torchaudio.transforms.Resample``;
2. FSMN-VAD's segments (``vad.detect``: 60 s chunks, the dynamic silence schedule);
3. each segment recognised alone (FunASR batches none on the CPU), shortest first, their texts put back in order
   (``text.sentence``). Each segment's features draw their dither from one generator seeded 0, in that order: upstream
   draws from torch's global generator, whose state depends on what its process did before; this is its output when that
   generator stands at seed 0;
4. the texts joined (``text.join_segments``) and punctuated by CT-Transformer (``punc.punctuate``): the transcript.

The segments come back with their VAD bounds and texts (before punctuation). The models compute in float32 on any device
(upstream's default); the features, the VAD's state machine and the text stay on the CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import Description, EngineError, LoadedEngine, LoadSpec

if TYPE_CHECKING:
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask

SAMPLE_RATE = 16000


def detect(model_dir: Path) -> bool:
    """A Paraformer folder as the catalogue lays it out: the recogniser, with its VAD in ``vad/`` and its punctuation
    model in ``punc/``."""
    config = model_dir / "config.yaml"
    try:
        head = config.read_text(encoding="utf-8")
    except OSError:
        return False
    return "model: Paraformer" in head and (model_dir / "vad" / "model.pt").is_file() and (model_dir / "punc" / "model.pt").is_file()


class LoadedParaformer(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.asr.paraformer import punc, vad
        from mizuoto.engines.asr.paraformer.model import Frontend, Paraformer, load_cmvn, load_config, load_tokens
        from mizuoto.runtime import ops
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict

        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        ctx.progress("loading:weights", None)
        try:
            config = load_config(folder)
            self.tokens = load_tokens(folder)
            self.model = Paraformer(config, len(self.tokens), ops.default)
            load_into(self.model, load_state_dict(folder / "model.pt"), ignore=Paraformer.ignorable(), what="Paraformer")
            self.frontend = Frontend(config.get("frontend_conf") or {}, load_cmvn(folder / "am.mvn"))
            ctx.check_cancel()
            self.vad, self.vad_frontend = vad.load(folder / "vad", ops.default)
            ctx.check_cancel()
            self.punc, self.vocabulary = punc.build(folder / "punc", ops.default)
            load_into(self.punc, load_state_dict(folder / "punc" / "model.pt"), what="CT-Transformer")
        except (KeyError, ValueError, OSError) as e:
            raise EngineError(f"{spec.model_id} is not a usable Paraformer model: {e}", code=codes.INVALID_INPUT) from e
        self.model.eval()
        self.punc.eval()
        # one phase each: the VAD over the clip, the recogniser per segment, the punctuation over the text
        self.h_vad = ModelHandle(self.vad, name=f"{spec.model_id}:vad", load_device=spec.device)
        self.h_model = ModelHandle(self.model, name=f"{spec.model_id}:paraformer", load_device=spec.device)
        self.h_punc = ModelHandle(self.punc, name=f"{spec.model_id}:punctuation", load_device=spec.device)
        self.handles = [self.h_vad, self.h_model, self.h_punc]

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=SAMPLE_RATE, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return 512 * 2**20

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Any:
        from mizuoto.runtime.engine import Unsupported

        raise Unsupported("Speech recognition does not synthesize")

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        if op != "transcribe":
            return super().call(op, payload, ctx)
        samples = self._read(payload["path"])
        return self.transcribe(samples, ctx)

    @staticmethod
    def _read(path: str) -> torch.Tensor:
        """``load_audio_text_image_video(path, fs=16000)``: ``torchaudio.load`` (16-bit PCM as ``/32768``, channels
        averaged) and ``torchaudio.transforms.Resample`` to 16 kHz."""
        import torch

        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        try:
            samples, rate = read_wav(path)
        except (OSError, ValueError, EOFError) as e:
            raise EngineError(f"Cannot read {Path(path).name}: {e}", {"path": path}, code=codes.INVALID_INPUT) from e
        x = torch.from_numpy(samples).float()
        if rate != SAMPLE_RATE:
            x = Resample(rate, SAMPLE_RATE)(x[None, :])[0, :]
        return x

    def transcribe(self, samples: torch.Tensor, ctx: RequestContext) -> dict[str, Any]:
        """FunASR's ``generate`` on 16 kHz float32 samples: ``{text, language, segments, duration}``."""
        import torch

        from mizuoto.engines.asr.paraformer import punc, text, vad

        duration = samples.shape[0] / SAMPLE_RATE
        with torch.inference_mode():
            segments = ctx.run("vad", [self.h_vad], lambda: vad.detect(self.vad, self.vad_frontend, samples), 64 * 2**20)
            order = sorted(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
            generator = torch.Generator().manual_seed(0)
            texts = [""] * len(segments)
            for i in order:
                ctx.check_cancel()
                beg, end = segments[i]
                piece = samples[int(beg * 16) : min(int(end * 16), samples.shape[0])]
                feats = self.frontend(piece, generator)
                ids = ctx.run("recognise", [self.h_model], lambda feats=feats: self.model.recognise(feats.to(self.device), generator), self.memory_required(None))
                texts[i] = text.sentence([self.tokens[t] for t in ids])
            if not " ".join(texts).strip():
                return {"text": "", "language": "zh", "segments": [], "duration": duration}
            joined = text.join_segments(texts)
            punctuated = ctx.run("punctuate", [self.h_punc], lambda: punc.punctuate(self.punc, self.vocabulary, joined)[0], 128 * 2**20)
        out = [{"start": round(b / 1000, 3), "end": round(e / 1000, 3), "text": t} for (b, e), t in zip(segments, texts, strict=True) if t]
        return {"text": punctuated, "language": "zh", "segments": out, "duration": duration}
