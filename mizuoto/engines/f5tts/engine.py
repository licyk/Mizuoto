"""F5-TTS, native: v1 Base, v0 Base (DiT), E2 Base (UNetT); cloning and speech editing.

The port follows Mizuoto's v1 adapter over upstream ``f5_tts``:
- the reference is trimmed and padded as ``preprocess_ref_audio_text`` does, levelled to ``target_rms`` and
  resampled with torchaudio's ``Resample``;
- the text is cut into upstream's chunks, which are generated in order and joined with its cross-fade;
- each chunk is ``CFM.sample``: seeded noise, the EPSS/sway time grid, Euler or midpoint steps with
  classifier-free guidance, then Vocos.

On a GPU the model runs in fp16 as upstream loads it (``load_checkpoint``), RoPE included. On the
CPU it runs in fp32, where the result is identical to v1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec

if TYPE_CHECKING:
    import numpy as np
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SpeechEditTask, SynthesisTask
    from mizuoto.runtime.runtime import Runtime

SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_MELS = 100
# upstream configs/<name>.yaml, model.arch (mel_spec is the same for all three)
ARCH: dict[str, dict[str, Any]] = {
    "v1_base": {"backbone": "DiT", "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": True, "conv_layers": 4, "pe_attn_head": None},
    "base": {"backbone": "DiT", "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1},
    "e2_base": {"backbone": "UNetT", "dim": 1024, "depth": 24, "heads": 16, "ff_mult": 4, "text_mask_padding": False, "pe_attn_head": 1},
}


class F5TtsEngine(Engine):
    id = "f5tts"
    DEVICES = DEVICE_KINDS
    VERSION = "1"
    TRAINER = "mizuoto.engines.f5tts.train:F5Trainer"
    OPTION_SCHEMA = {
        "sway_sampling_coef": OptionSpec("float", -1.0, -1.0, 1.0),
        "target_rms": OptionSpec("float", 0.1, 0.01, 1.0),
        "cross_fade_duration": OptionSpec("float", 0.15, 0.0, 1.0),
    }

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None:
            raise EngineError("F5-TTS needs a model folder", code=codes.MODEL_MISSING)
        if spec.variant not in ARCH:
            raise EngineError(f"Unknown F5-TTS variant {spec.variant}", code=codes.INVALID_INPUT)
        return LoadedF5Tts(spec, ctx, compile=runtime.options.compile)


def _float32() -> torch.dtype:
    import torch

    return torch.float32


def _dtype(device: str) -> torch.dtype:
    """upstream ``load_checkpoint``: fp16 on a CUDA GPU of compute capability 7 or more, else fp32."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        if torch.cuda.get_device_properties(index).major >= 7 and not torch.cuda.get_device_name(index).endswith("[ZLUDA]"):
            return torch.float16
    return torch.float32


def leading_silence(samples: np.ndarray, sr: int, threshold_db: float = -42.0, chunk_ms: int = 10) -> int:
    """Samples of leading silence, as pydub's ``detect_leading_silence`` (dBFS per 10 ms chunk)."""
    import numpy as np

    chunk = max(1, sr * chunk_ms // 1000)
    limit = 10 ** (threshold_db / 20)
    position = 0
    while position < len(samples):
        piece = samples[position : position + chunk]
        rms = float(np.sqrt(np.mean(np.square(piece.astype(np.float64))))) if len(piece) else 0.0
        if rms >= limit:
            break
        position += chunk
    return min(position, len(samples))


def edge_pad(sr: int) -> int:
    """Upstream's 50 ms of silence in samples: pydub makes it at 11025 Hz and converts it with
    ``audioop.ratecv``, whose frame count this reproduces (1198 at 24 kHz, not 1200)."""
    inrate, outrate = 11025, int(sr)
    g = math.gcd(inrate, outrate)
    inrate, outrate = inrate // g, outrate // g
    n, d, out = int(11025 * 0.05), -outrate, 0
    while True:
        while d < 0:
            if n == 0:
                return out
            n -= 1
            d += outrate
        while d >= 0:
            out += 1
            d -= inrate


def prepare_reference(samples: np.ndarray, sr: int) -> np.ndarray:
    """As upstream's ``preprocess_ref_audio_text``: trim the silent edges and add 50 ms of silence."""
    import numpy as np

    start = leading_silence(samples, sr)
    trimmed = samples[start:]
    end = leading_silence(trimmed[::-1], sr)
    trimmed = trimmed[: len(trimmed) - end] if end else trimmed
    return np.concatenate([trimmed, np.zeros(edge_pad(sr), dtype=np.float32)]).astype(np.float32)


class LoadedF5Tts(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext, compile: bool = False) -> None:
        super().__init__(spec)
        import torch
        import yaml
        from torch import nn

        from mizuoto.engines.f5tts.text import read_vocab
        from mizuoto.nn.dsp.mel import MelSpectrogram
        from mizuoto.nn.transformer.dit import DiT
        from mizuoto.nn.transformer.unett import UNetT
        from mizuoto.nn.vocoders.vocos import Vocos
        from mizuoto.runtime import ops
        from mizuoto.runtime.compile import compile_module
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict

        folder = spec.model_dir
        assert folder is not None
        checkpoints = sorted(folder.glob("*.safetensors")) + sorted(folder.glob("*.pt"))
        if not checkpoints:
            raise EngineError(f"{folder} has no checkpoint", code=codes.MODEL_MISSING)
        # a model's own vocabulary, else the one upstream ships (E2-TTS has none of its own)
        vocab_path = folder / "vocab.txt"
        if not vocab_path.is_file():
            vocab_path = Path(__file__).with_name("vocab.txt")
        vocoder_dir = folder / "vocos"
        if not vocoder_dir.is_dir():
            raise EngineError(f"The vocos vocoder is missing from {folder}", code=codes.MODEL_MISSING)
        self.device = torch.device(spec.device)
        self.variant = spec.variant
        self.dtype = _dtype(spec.device)
        self.vocab = read_vocab(vocab_path)
        self.ode_method = str(spec.options.get("ode_method") or "euler")

        ctx.progress("loading:vocoder", None)
        vocos_config = yaml.safe_load((vocoder_dir / "config.yaml").read_text(encoding="utf-8"))
        vocoder = Vocos(vocos_config, ops.default)
        load_into(vocoder, load_state_dict(vocoder_dir / "pytorch_model.bin"), ignore=Vocos.ignorable(), what="Vocos vocoder")
        self.vocoder = vocoder.eval()

        ctx.check_cancel()
        ctx.progress("loading:weights", None)
        arch = {k: v for k, v in ARCH[spec.variant].items() if k != "backbone"}
        backbone = DiT if ARCH[spec.variant]["backbone"] == "DiT" else UNetT
        self.transformer: DiT | UNetT = backbone(**arch, text_num_embeds=len(self.vocab), mel_dim=N_MELS, ops=ops.default)
        model = nn.Module()
        model.add_module("transformer", self.transformer)
        model.to(self.dtype)  # upstream casts before loading, so the checkpoint's fp32 values are rounded into it
        raw = load_state_dict(checkpoints[0])
        if checkpoints[0].suffix == ".pt" and "ema_model_state_dict" in raw:
            raw = raw["ema_model_state_dict"]
        state = {k.replace("ema_model.", ""): v for k, v in raw.items() if k not in ("initted", "step")}
        for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
            state.pop(key, None)
        load_into(model, state, what="F5-TTS model")
        del raw, state
        model.eval()
        self.model = model
        # compute.compile: the DiT/UNetT forward of every ODE step, its sequence length dynamic
        self.compiled = compile_module(self.transformer, compile, f"{spec.model_id}:transformer")
        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=HOP_LENGTH, n_mels=N_MELS, power=1, center=True, normalized=False, norm=None)
        # the transformer runs every ODE step over the whole sequence (with the mel for the reference); the vocoder decodes
        self.h_transformer = ModelHandle(model, name=f"{spec.model_id}:transformer", load_device=spec.device)
        self.h_mel = ModelHandle(self.mel, name=f"{spec.model_id}:mel", load_device=spec.device)
        self.h_vocoder = ModelHandle(vocoder, name=f"{spec.model_id}:vocoder", load_device=spec.device)
        self.handles = [self.h_transformer, self.h_vocoder, self.h_mel]

    def describe(self) -> Description:
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=SAMPLE_RATE, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return 1024 * 2**20

    def _sample_memory(self, frames: int) -> int:
        """One ODE step (conditional and unconditional together): the activations of a block and, where attention runs
        unfused (MPS, XPU), its score matrices."""
        heads = int(ARCH[self.variant].get("heads", 16))
        element = 2 if self.dtype != _float32() else 4
        return 2 * frames * 1024 * 4 * 16 + 2 * heads * frames * frames * element + 256 * 2**20

    def _vocoder_memory(self, frames: int) -> int:
        return frames * 512 * 4 * 8 + frames * HOP_LENGTH * 4 * 4 + 128 * 2**20

    # -- the model ----------------------------------------------------------------------

    def _mel(self, wave: torch.Tensor) -> torch.Tensor:
        """``MelSpec`` (vocos): (batch, samples) → (batch, mels, frames) log-mel."""
        return self.mel(wave).clamp(min=1e-5).log()

    def _text_ids(self, chars: list[list[str]]) -> torch.Tensor:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        tensors = [torch.tensor([self.vocab.get(c, 0) for c in t]) for t in chars]
        return pad_sequence(tensors, padding_value=-1, batch_first=True).to(self.device)

    def _sample(
        self, cond: torch.Tensor, text: torch.Tensor, duration: int, steps: int, cfg: float, sway: float | None, ctx: RequestContext, edit_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``CFM.sample`` for one item (batch of 1): the mel over ``duration`` frames, the reference frames kept."""
        memory = self._sample_memory(max(duration, int((text != -1).sum().item()) + 1, cond.shape[-1] // HOP_LENGTH + 1))
        return ctx.run("transformer", [self.h_transformer, self.h_mel], lambda: self._sample_in_phase(cond, text, duration, steps, cfg, sway, ctx, edit_mask), memory)

    def _sample_in_phase(
        self, cond: torch.Tensor, text: torch.Tensor, duration: int, steps: int, cfg: float, sway: float | None, ctx: RequestContext, edit_mask: torch.Tensor | None
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        from mizuoto.nn.flow.cfm import epss_timesteps, odeint_fixed
        from mizuoto.nn.flow.cfm import sway as sway_grid

        if cond.ndim == 2:
            cond = self._mel(cond).permute(0, 2, 1)
        cond = cond.to(self.dtype)
        batch, cond_seq_len = cond.shape[:2]
        lens = torch.full((batch,), cond_seq_len, device=self.device, dtype=torch.long)
        cond_mask = torch.arange(int(lens.amax()), device=self.device)[None, :] < lens[:, None]
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask
        dur = torch.full((batch,), duration, device=self.device, dtype=torch.long)
        dur = torch.maximum(torch.maximum((text != -1).sum(dim=-1), lens) + 1, dur).clamp(max=65536)
        max_duration = int(dur.amax())
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False).unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        text_embeds = self.transformer.text_condition(text, max_duration)
        transformer = self.compiled or self.transformer

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            if cfg < 1e-5:
                return transformer(x, step_cond, text_embeds, t, None, cfg_infer=False)
            pred_cfg = transformer(x, step_cond, text_embeds, t, None, cfg_infer=True)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg

        gen = ctx.generator_for(self.device)
        y0 = torch.randn(int(dur[0]), N_MELS, device=self.device, dtype=step_cond.dtype, generator=gen)[None]
        grid = epss_timesteps(steps, self.device, step_cond.dtype)
        if sway is not None:
            grid = sway_grid(grid, sway)

        def step(i: int, n: int) -> None:
            ctx.check_cancel()

        out = odeint_fixed(fn, y0, grid, self.ode_method, step)
        return torch.where(cond_mask, cond, out)

    def _decode(self, mel: torch.Tensor, ctx: RequestContext) -> torch.Tensor:
        from mizuoto.nn.dsp.tiles import tiled_1d

        tiled = ("in tiles of 256 frames", lambda: tiled_1d(self.vocoder.decode, mel, HOP_LENGTH))
        return ctx.run("vocoder", [self.h_vocoder], lambda: self.vocoder.decode(mel), self._vocoder_memory(mel.shape[-1]), fallback=tiled)

    # -- reference ----------------------------------------------------------------------

    def _reference(self, task: SynthesisTask, target_rms: float) -> tuple[torch.Tensor, float, str]:
        import torch

        from mizuoto.engines.f5tts.text import finish_ref_text
        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        if not task.voice.references:
            raise EngineError("F5-TTS clones a voice: it needs a reference clip", code=codes.INVALID_INPUT)
        ref = task.voice.references[0]
        if not (ref.transcript or "").strip():
            raise EngineError("F5-TTS needs the reference clip's transcript", {"capability": "clone_no_transcript"}, code=codes.CAPABILITY_NOT_SUPPORTED)
        samples, sr = read_wav(ref.audio_path)
        audio = torch.from_numpy(prepare_reference(samples, sr)).unsqueeze(0)
        rms = float(torch.sqrt(torch.mean(torch.square(audio))))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != SAMPLE_RATE:
            audio = Resample(sr, SAMPLE_RATE)(audio)
        return audio.to(self.device), rms, finish_ref_text(ref.transcript or "")

    def _generate(
        self,
        audio: torch.Tensor,
        rms: float,
        ref_text: str,
        gen_text: str,
        fix_duration: float | None,
        speed: float,
        steps: int,
        cfg: float,
        sway: float,
        target_rms: float,
        ctx: RequestContext,
    ) -> np.ndarray:
        """One chunk, as upstream's ``infer_batch_process`` does it."""
        import torch

        from mizuoto.engines.f5tts.text import convert_char_to_pinyin

        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        local_speed = 0.3 if len(gen_text.encode("utf-8")) < 10 else speed
        text = self._text_ids(convert_char_to_pinyin([ref_text + gen_text]))
        ref_audio_len = audio.shape[-1] // HOP_LENGTH
        if fix_duration is not None:
            duration = int(fix_duration * SAMPLE_RATE / HOP_LENGTH)
        else:
            duration = ref_audio_len + int(ref_audio_len / len(ref_text.encode("utf-8")) * len(gen_text.encode("utf-8")) / local_speed)
        generated = self._sample(audio, text, duration, steps, cfg, sway, ctx)
        generated = generated.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1)
        wave = self._decode(generated, ctx)
        if rms < target_rms:
            wave = wave * rms / target_rms
        return wave.squeeze().cpu().numpy()

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np

        from mizuoto.engines.f5tts.text import chunk_text

        p, o = task.params, task.options
        steps = int(p.get("num_steps") or 32)
        cfg = float(p["guidance_scale"]) if p.get("guidance_scale") is not None else 2.0
        speed = float(p.get("speed") or 1.0)
        sway = float(o["sway_sampling_coef"]) if o.get("sway_sampling_coef") is not None else -1.0
        target_rms = float(o.get("target_rms") or 0.1)
        fade = float(o["cross_fade_duration"]) if o.get("cross_fade_duration") is not None else 0.15
        audio, rms, ref_text = self._reference(task, target_rms)
        ref_seconds = audio.shape[-1] / SAMPLE_RATE
        ctx.applied.update({"nfe_step": steps, "cfg_strength": cfg, "speed": speed, "sway_sampling_coef": sway, "target_rms": target_rms, "cross_fade_duration": fade})
        texts = list(task.texts) if task.texts else [task.text]
        for index, text in enumerate(texts):
            segment = task.segment + index if task.texts else task.segment
            max_chars = int(len(ref_text.encode("utf-8")) / ref_seconds * (22 - ref_seconds) * speed)
            batches = chunk_text(text, max_chars=max(1, max_chars)) or [text]
            if p.get("duration") is not None:
                fix_total = ref_seconds + float(p["duration"])
                ctx.applied["fix_duration"] = fix_total
                if len(batches) > 1:
                    ref_sec = (audio.shape[-1] // HOP_LENGTH) * HOP_LENGTH / SAMPLE_RATE
                    weights = [len(c.encode("utf-8")) for c in batches]
                    allocatable = fix_total - ref_sec + fade * (len(batches) - 1)
                    durations: list[float | None] = [ref_sec + allocatable * w / sum(weights) for w in weights]
                else:
                    durations = [fix_total]
            else:
                durations = [None] * len(batches)
            fade_samples = int(fade * SAMPLE_RATE)
            tail = None
            for n, (chunk, fix) in enumerate(zip(batches, durations)):
                ctx.check_cancel()
                ctx.progress("generating", n / len(batches), f"chunk {n + 1} of {len(batches)}")
                wave = np.asarray(self._generate(audio, rms, ref_text, chunk, fix, speed, steps, cfg, sway, target_rms, ctx), dtype=np.float32).reshape(-1)
                if tail is not None:
                    overlap = min(fade_samples, len(tail), len(wave))
                    if overlap > 0:
                        mixed = tail[len(tail) - overlap :] * np.linspace(1, 0, overlap, dtype=np.float32) + wave[:overlap] * np.linspace(0, 1, overlap, dtype=np.float32)
                        yield AudioChunk(np.concatenate([tail[: len(tail) - overlap], mixed]).astype(np.float32), SAMPLE_RATE, segment)
                        wave = wave[overlap:]
                    else:
                        yield AudioChunk(tail, SAMPLE_RATE, segment)
                last = n == len(batches) - 1
                if last or fade_samples <= 0:
                    yield AudioChunk(np.clip(wave, -1.0, 1.0), SAMPLE_RATE, segment, final=last)
                    tail = None
                else:
                    keep = min(fade_samples, len(wave))
                    if len(wave) > keep:
                        yield AudioChunk(np.clip(wave[: len(wave) - keep], -1.0, 1.0), SAMPLE_RATE, segment)
                    tail = wave[len(wave) - keep :]

    # -- speech editing -----------------------------------------------------------------

    def edit(self, task: SpeechEditTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        """Regenerate spans of a recording to say the new transcript (``task.transcript`` is the full text after the edit)."""
        import numpy as np
        import torch
        import torch.nn.functional as F

        from mizuoto.engines.f5tts.text import convert_char_to_pinyin
        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        p, o = task.params, task.options
        steps = int(p.get("num_steps") or 32)
        cfg = float(p["guidance_scale"]) if p.get("guidance_scale") is not None else 2.0
        sway = float(o["sway_sampling_coef"]) if o.get("sway_sampling_coef") is not None else -1.0
        target_rms = float(o.get("target_rms") or 0.1)
        samples, sr = read_wav(task.audio_path)
        audio = torch.from_numpy(samples).unsqueeze(0)
        rms = float(torch.sqrt(torch.mean(torch.square(audio))))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != SAMPLE_RATE:
            audio = Resample(sr, SAMPLE_RATE)(audio)
        audio = audio.to(self.device)
        original = ctx.run("mel", [self.h_mel], lambda: self._mel(audio).permute(0, 2, 1))
        offset = 0
        cond = torch.zeros(1, 0, N_MELS, device=self.device)
        mask = torch.zeros(1, 0, dtype=torch.bool, device=self.device)
        for span in sorted(task.edits, key=lambda s: s.start):
            start_frame = round(span.start * SAMPLE_RATE / HOP_LENGTH)
            end_frame = round(span.end * SAMPLE_RATE / HOP_LENGTH)
            seconds = span.duration if span.duration else span.end - span.start
            frames = round(seconds * SAMPLE_RATE / HOP_LENGTH)
            cond = torch.cat((cond, original[:, offset:start_frame, :], torch.zeros(1, frames, N_MELS, device=self.device)), dim=1)
            mask = torch.cat(
                (mask, torch.ones(1, start_frame - offset, dtype=torch.bool, device=self.device), torch.zeros(1, frames, dtype=torch.bool, device=self.device)), dim=-1
            )
            offset = end_frame
        cond = torch.cat((cond, original[:, offset:, :]), dim=1)
        mask = F.pad(mask, (0, cond.shape[1] - mask.shape[-1]), value=True)
        ctx.check_cancel()
        ctx.applied.update({"nfe_step": steps, "cfg_strength": cfg, "sway_sampling_coef": sway})
        text = self._text_ids(convert_char_to_pinyin([task.transcript]))
        generated = self._sample(cond, text, cond.shape[1], steps, cfg, sway, ctx, edit_mask=mask)
        wave = self._decode(generated.to(torch.float32).permute(0, 2, 1), ctx)
        if rms < target_rms:
            wave = wave * rms / target_rms
        yield AudioChunk(np.clip(wave.squeeze().cpu().numpy().astype(np.float32), -1.0, 1.0), SAMPLE_RATE, final=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
