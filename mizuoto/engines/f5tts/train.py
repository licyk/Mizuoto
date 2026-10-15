"""F5-TTS fine-tuning, native: v1's trainer (upstream's ``finetune_cli`` recipe) on the
native modules, with no transformers, accelerate, datasets, wandb or ema_pytorch.

- ``prepare`` is ``prepare_csv_wavs``: each clip at 24 kHz as 16-bit WAV, its text in upstream's tokens
  (``convert_char_to_pinyin``), the durations and the base model's vocabulary (``raw.jsonl`` instead of the arrow file).
- ``train`` is upstream's ``Trainer`` as v1 ran it: ``CustomDataset`` mels, the frame-budget ``DynamicBatchSampler``
  shuffled with seed 666, ``CFM.forward``'s masked-span flow-matching loss with its classifier-free-guidance drops,
  dropout 0.1, fused AdamW, linear warm-up then linear decay, gradient clipping at 1, accelerate's gradient
  accumulation and mixed precision (CUDA only), and ema_pytorch's EMA, carried on from the base checkpoint's. It ends
  with ``ckpts/model_last.pt`` in upstream's layout.
- ``export`` keeps the EMA weights as ``model_<update>.safetensors`` (upstream's keys: ``ema_model.transformer.*``,
  ``initted``, ``step``) with the vocabulary and the vocoder: a model both v1 and the native engine load.

Seeded alike (the stage seeds Python's, NumPy's and torch's generators), a run draws what upstream draws, in its
order: the model's random initialisation (overwritten by the base weights), the data loader's worker seed, then per
batch the span fractions and starts, the noise, the time steps, two ``random()`` calls for the condition drops and
the dropout masks. Mels are computed on one thread, as upstream's data-loader workers compute them.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

if TYPE_CHECKING:
    import torch
    from torch import nn

SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_MELS = 100
MIN_SECONDS, MAX_SECONDS = 0.3, 30.0
"""Upstream's ``CustomDataset`` drops clips outside these."""
DROPOUT = 0.1
"""``DiT``/``UNetT``'s default, which upstream's configs keep."""
AUDIO_DROP, COND_DROP = 0.3, 0.2
"""``CFM``'s ``audio_drop_prob`` and ``cond_drop_prob``."""
FRAC_LENGTHS = (0.7, 1.0)
SHUFFLE_SEED = 666
"""``Trainer.train(resumable_with_seed=666)``, as v1 called it."""


class F5Trainer(Trainer):
    ENGINE = "f5tts"

    # -- stages ---------------------------------------------------------------------------------

    def stage_prepare(self, ctx: StageContext) -> dict[str, Any]:
        """Clips at 24 kHz with their text in upstream's tokens: ``raw.jsonl``, ``duration.json`` and ``vocab.txt``."""
        import torch

        from mizuoto.engines.f5tts.text import convert_char_to_pinyin
        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.resample import resample
        from mizuoto.nn.dsp.wav import write_wav

        _variant(ctx)
        data = ctx.workdir / "data"
        if data.exists():
            shutil.rmtree(data)
        wavs = data / "wavs"
        wavs.mkdir(parents=True)
        rows = [r for r in ctx.rows if r.text.strip()]
        records: list[dict[str, Any]] = []
        skipped = len(ctx.rows) - len(rows)
        for i, row in enumerate(rows):
            audio, sr = read_audio(row.audio)
            samples = torch.from_numpy(audio)
            if sr != SAMPLE_RATE:
                samples = resample(samples, sr, SAMPLE_RATE)
            duration = samples.shape[-1] / SAMPLE_RATE
            if not MIN_SECONDS <= duration <= MAX_SECONDS:
                skipped += 1
                continue
            target = wavs / f"{row.id}.wav"
            write_wav(target, samples.numpy(), SAMPLE_RATE)
            text = convert_char_to_pinyin([row.text.strip()], polyphone=True)[0]
            records.append({"audio_path": str(target), "text": text, "duration": duration})
            ctx.progress((i + 1) / len(rows))
        if not records:
            raise EngineError(f"No clip has text and lasts {MIN_SECONDS}–{MAX_SECONDS:.0f} s", code=codes.INVALID_INPUT)
        with (data / "raw.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        durations = [r["duration"] for r in records]
        (data / "duration.json").write_text(json.dumps({"duration": durations}), encoding="utf-8")
        shutil.copyfile(_vocab(Path(ctx.base_model["dir"])), data / "vocab.txt")
        if skipped:
            ctx.log(f"{skipped} clip(s) without text or outside {MIN_SECONDS}–{MAX_SECONDS:.0f} s are left out", "warning")
        return {"clips": len(records), "seconds": round(sum(durations), 1)}

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        """Upstream's ``Trainer`` from the base weights; losses as metrics."""
        import torch

        from mizuoto.engines.f5tts.text import read_vocab
        from mizuoto.training.memory import blocks, checkpoint_every, checkpoint_units, host_optimizer, preset, wants

        variant = _variant(ctx)
        data = ctx.workdir / "data"
        if not (data / "raw.jsonl").is_file():
            raise EngineError("Prepare the data first", code=codes.INVALID_INPUT)
        records = [json.loads(line) for line in (data / "raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        durations = json.loads((data / "duration.json").read_text(encoding="utf-8"))["duration"]
        vocab = read_vocab(data / "vocab.txt")
        ckpts = ctx.workdir / "ckpts"
        if ckpts.exists():
            shutil.rmtree(ckpts)
        ckpts.mkdir(parents=True)

        epochs = int(ctx.hp("epochs", 10))
        frames = int(ctx.hp("batch_frames", 3200))
        accumulation = int(ctx.hp("grad_accumulation", 1))
        learning_rate = float(ctx.hp("learning_rate", 1e-5))
        # v1's warm-up: 2% of the updates it expected (the frames over the budget), at least one.
        expected = epochs * max(1, math.ceil(sum(d * SAMPLE_RATE / HOP_LENGTH for d in durations) / frames))
        warmup = int(ctx.hp("warmup_updates", 0)) or max(1, expected // 50)
        device = torch.device(ctx.device)
        mixed = _mixed_precision(ctx)

        # Upstream builds the model (drawing its random initialisation), then the trainer, then loads the weights.
        model = _build(variant, len(vocab))
        model.to(device)
        optimizer = _adamw(model, learning_rate)
        batches = dynamic_batches(durations, frames, int(ctx.hp("max_samples", 64)))
        total = math.ceil(len(batches) / accumulation) * epochs
        scheduler = _scheduler(optimizer, warmup, total - warmup)
        ema = Ema(model, "cpu" if preset(ctx, "ema_device") == "cpu" and device.type != "cpu" else None)
        _load_base(Path(ctx.base_model["dir"]), model, ema)
        if wants(ctx, "optimizer_device", "cpu") and mixed != torch.float16:  # the lowest preset: AdamW's state in RAM
            optimizer = host_optimizer(list(model.parameters()), lambda ps: _adamw(ps, learning_rate))
            scheduler = _scheduler(optimizer, warmup, total - warmup)
            ctx.log("The optimizer's state is kept in RAM and stepped on the CPU")
        every = checkpoint_every(preset(ctx, "gradient_checkpointing"))
        if every:  # the dropout masks come from the global generators: replayed in the backward
            n = checkpoint_units(found := blocks(model), (), every)
            ctx.log(f"Gradient checkpointing: {n} of {len(found)} blocks")
        scaler = torch.amp.GradScaler("cuda") if mixed == torch.float16 else None

        mels = MelCache(records)
        update = 0
        ctx.progress(0.0, f"Epoch 1 of {epochs}")
        for epoch in range(epochs):
            model.train()
            if epoch < 2:
                # The data loader's worker seed: its persistent loaders draw one when each first iterates
                # (epoch 0 through accelerate's skip_first_batches copy, epoch 1 through the original).
                torch.empty((), dtype=torch.int64).random_()
            order = shuffled(batches, SHUFFLE_SEED + max(0, epoch - 1))
            for index, batch in enumerate(order):
                sync = (index + 1) % accumulation == 0 or index == len(order) - 1
                mel, lens = mels.batch(batch)
                text = token_ids([records[i]["text"] for i in batch], vocab)
                with torch.autocast(device.type, dtype=mixed, enabled=mixed is not None):
                    loss = cfm_loss(model.transformer, mel.permute(0, 2, 1).to(device), text.to(device), lens.to(device))
                loss = loss.float()
                scaled = loss / accumulation
                (scaler.scale(scaled) if scaler is not None else scaled).backward()
                if sync:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    skipped = False
                    if scaler is not None:
                        before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        skipped = scaler.get_scale() < before
                    else:
                        optimizer.step()
                    if not skipped:
                        scheduler.step()
                    optimizer.zero_grad()
                    ema.update(model)
                    update += 1
                ctx.metric(epoch + 1, update, {"loss": float(loss.item())}, float(scheduler.get_last_lr()[0]))
                if sync:
                    ctx.progress(min(1.0, update / total), f"Update {update} of {total}")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_model_state_dict": ema.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update": update,
        }
        torch.save(checkpoint, ckpts / "model_last.pt")
        return {"updates": update, "epochs": epochs}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """The EMA weights as a model folder of the base model's variant, and a preview spoken with it."""
        import torch
        from safetensors.torch import save_file

        variant = _variant(ctx)
        last = ctx.workdir / "ckpts" / "model_last.pt"
        if not last.is_file():
            raise EngineError("Training saved no checkpoint", code=codes.ENGINE_ERROR)
        checkpoint = torch.load(last, weights_only=True, map_location="cpu", mmap=True)
        update = int(checkpoint.get("update", 0))
        weights = {k: v.contiguous() for k, v in checkpoint["ema_model_state_dict"].items()}
        del checkpoint
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        save_file(weights, str(folder / f"model_{update}.safetensors"))
        del weights
        shutil.copyfile(ctx.workdir / "data" / "vocab.txt", folder / "vocab.txt")
        vocoder = Path(ctx.base_model["dir"]) / "vocos"
        if vocoder.is_dir():
            shutil.copytree(vocoder, folder / vocoder.name)
        reference = _reference(ctx.rows)
        if reference is not None:
            text = str(ctx.hp("preview_text", "") or "") or reference.text
            ctx.preview(text, _as_wav(reference, ctx.workdir / "data" / "preview"), "Preview", model_dir=folder)
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": variant, "title": ctx.name}], "update": update}


# -- the model ------------------------------------------------------------------------------------


def _variant(ctx: StageContext) -> str:
    from mizuoto.engines.f5tts.engine import ARCH

    variant = str(ctx.base_model.get("variant", ""))
    if variant not in ARCH:
        raise EngineError(f"Unknown F5-TTS variant {variant}", code=codes.INVALID_INPUT)
    return variant


def _vocab(model_dir: Path) -> Path:
    own = model_dir / "vocab.txt"
    return own if own.is_file() else Path(__file__).with_name("vocab.txt")


def _build(variant: str, vocab_size: int) -> nn.Module:
    """``CFM`` around the variant's backbone (only the backbone has weights), its random initialisation drawn as
    upstream draws it (layer by layer in its construction order), since the draws move the generators training uses."""
    from torch import nn

    from mizuoto.engines.f5tts.engine import ARCH
    from mizuoto.nn.transformer.dit import DiT
    from mizuoto.nn.transformer.unett import UNetT
    from mizuoto.runtime import ops

    spec = ARCH[variant]
    arch = {k: v for k, v in spec.items() if k != "backbone"}
    backbone = DiT if spec["backbone"] == "DiT" else UNetT
    transformer = backbone(**arch, text_num_embeds=vocab_size, mel_dim=N_MELS, ops=ops.default)
    for module in _construction_order(transformer):
        if isinstance(module, nn.Linear):
            nn.Linear.reset_parameters(module)
        elif isinstance(module, nn.Conv1d):
            nn.Conv1d.reset_parameters(module)
        elif isinstance(module, nn.Embedding):
            nn.Embedding.reset_parameters(module)
    model = nn.Module()
    model.add_module("transformer", transformer)
    return model


def _construction_order(transformer: nn.Module) -> Iterator[nn.Module]:
    """The layers in the order upstream's constructors make them: registration order, except that ``UNetT`` makes a
    layer's skip projection last (its ``ModuleList`` lists it first)."""
    from mizuoto.nn.transformer.unett import UNetT

    if not isinstance(transformer, UNetT):
        yield from transformer.modules()
        return
    for name, child in transformer.named_children():
        if name != "layers":
            yield from child.modules()
            continue
        for layer in child.children():
            parts = dict(layer.named_children())
            for key in ("1", "2", "3", "4", "0"):
                if parts.get(key) is not None:
                    yield from parts[key].modules()


def _adamw(model: nn.Module | list[Any], lr: float) -> torch.optim.Optimizer:
    """``AdamW(params, lr, fused=True)`` as upstream builds it (the loop kind where fused does not exist)."""
    import torch

    params = model if isinstance(model, list) else list(model.parameters())
    try:
        return torch.optim.AdamW(params, lr=lr, fused=True)
    except RuntimeError:
        return torch.optim.AdamW(params, lr=lr)


def _scheduler(optimizer: torch.optim.Optimizer, warmup: int, decay: int) -> Any:
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup])


def _mixed_precision(ctx: StageContext) -> torch.dtype | None:
    """accelerate's mixed precision as v1 asked for it: none off CUDA; bf16 for ``auto``."""
    import torch

    if not ctx.device.startswith("cuda"):
        return None
    return {"auto": torch.bfloat16, "bf16": torch.bfloat16, "fp16": torch.float16}.get(ctx.precision)


def _load_base(model_dir: Path, model: nn.Module, ema: Ema) -> None:
    """The base model's EMA weights into the EMA (its ``step`` carried on) and into the model, as upstream's
    ``load_checkpoint`` starts from a ``pretrained_*`` checkpoint."""
    from mizuoto.runtime.loading import load_into, load_state_dict

    base = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.pt"))
    if not base:
        raise EngineError(f"{model_dir} has no checkpoint", code=codes.MODEL_MISSING)
    raw = load_state_dict(base[0])
    if base[0].suffix == ".pt" and "ema_model_state_dict" in raw:
        raw = raw["ema_model_state_dict"]
    for key in ("ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"):
        raw.pop(key, None)
    state = {k.replace("ema_model.", ""): v for k, v in raw.items() if k not in ("initted", "update", "step")}
    load_into(model, state, what="F5-TTS model")
    ema.load(raw)


class Ema:
    """ema_pytorch's ``EMA(model, include_online_model=False)`` with its defaults (beta 0.9999, an update every 10
    steps after the first 100, warm-up power 2/3), which upstream's trainer keeps: the same state dict and updates."""

    BETA, UPDATE_AFTER, EVERY, INV_GAMMA, POWER, MIN_VALUE = 0.9999, 100, 10, 1.0, 2 / 3, 0.0

    def __init__(self, model: nn.Module, device: Any = None) -> None:
        """``device``: where the averaged copy lives (the model's own by default; the CPU keeps a GPU free of a second
        copy of the model, at the cost of a transfer every tenth step)."""
        import copy

        import torch

        self.model = copy.deepcopy(model)
        if device is not None:
            self.model.to(device)
        for p in self.model.parameters():
            p.detach_()
        device = next(self.model.parameters()).device
        self.initted = torch.tensor(False, device=device)
        self.step = torch.tensor(0, device=device)

    def load(self, state: dict[str, Any]) -> None:
        import torch

        own = {k.removeprefix("ema_model."): v for k, v in state.items() if k.startswith("ema_model.")}
        self.model.load_state_dict({k: v for k, v in own.items() if not k.startswith("mel_spec.")})
        if "initted" in state:
            self.initted.copy_(torch.as_tensor(state["initted"]))
        if "step" in state:
            self.step.copy_(torch.as_tensor(state["step"]))

    def state_dict(self) -> dict[str, Any]:
        return {"initted": self.initted, "step": self.step, **{f"ema_model.{k}": v for k, v in self.model.state_dict().items()}}

    def _pairs(self, online: nn.Module) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for (_, ma), (_, current) in zip(self.model.named_parameters(), online.named_parameters(), strict=True):
            if ma.is_floating_point() or ma.is_complex():
                yield ma.data, current.data
        for (_, ma), (_, current) in zip(self.model.named_buffers(), online.named_buffers(), strict=True):
            if ma.is_floating_point() or ma.is_complex():
                yield ma.data, current.data

    def _decay(self) -> float:
        epoch = (self.step - self.UPDATE_AFTER - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.INV_GAMMA) ** -self.POWER
        if epoch.item() <= 0:
            return 0.0
        return value.clamp(min=self.MIN_VALUE, max=self.BETA).item()

    def update(self, online: nn.Module) -> None:
        import torch

        step = int(self.step.item())
        self.step += 1
        if not self.initted.item():
            for ma, current in self._pairs(online):
                ma.copy_(current)
            self.initted.data.copy_(torch.tensor(True))
            return
        if step % self.EVERY:
            return
        if step <= self.UPDATE_AFTER:
            for ma, current in self._pairs(online):
                ma.copy_(current)
            return
        weight = 1.0 - self._decay()
        with torch.no_grad():
            for ma, current in self._pairs(online):
                ma.lerp_(current.to(ma.device), weight)


# -- data -----------------------------------------------------------------------------------------


def dynamic_batches(durations: list[float], frames: float, max_samples: int) -> list[list[int]]:
    """``DynamicBatchSampler``: clips sorted by length, packed into batches of at most ``frames`` mel frames and
    ``max_samples`` clips (a clip longer than the budget alone is dropped); the last batch kept."""
    indices = sorted(((i, d * SAMPLE_RATE / HOP_LENGTH) for i, d in enumerate(durations)), key=lambda e: e[1])
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_frames = 0.0
    for idx, frame_len in indices:
        if batch_frames + frame_len <= frames and (max_samples == 0 or len(batch) < max_samples):
            batch.append(idx)
            batch_frames += frame_len
        else:
            if batch:
                batches.append(batch)
            if frame_len <= frames:
                batch, batch_frames = [idx], frame_len
            else:
                batch, batch_frames = [], 0.0
    if batch:
        batches.append(batch)
    return batches


def shuffled(batches: list[list[int]], seed: int) -> list[list[int]]:
    """The sampler's epoch order: ``torch.randperm`` on its own generator seeded ``seed``."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]


def token_ids(texts: list[list[str]], vocab: dict[str, int]) -> torch.Tensor:
    """``list_str_to_idx``: unknown tokens are 0, batch padding -1."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    return pad_sequence([torch.tensor([vocab.get(c, 0) for c in t]) for t in texts], padding_value=-1, batch_first=True)


class MelCache:
    """``CustomDataset``'s vocos log-mels of the prepared clips, made once, and ``collate_fn``'s padded batches."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        from mizuoto.nn.dsp.mel import MelSpectrogram

        self.records = records
        self.mels: dict[int, torch.Tensor] = {}
        self.transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=HOP_LENGTH, n_mels=N_MELS, power=1, center=True, normalized=False, norm=None
        )

    def mel(self, index: int) -> torch.Tensor:
        import torch

        from mizuoto.nn.dsp.wav import read_wav

        if index not in self.mels:
            samples, _ = read_wav(self.records[index]["audio_path"])
            # Upstream computes them in data-loader workers, which run torch on one thread; more threads sum the
            # filterbank product in another order. The setting is restored at once.
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with torch.no_grad():
                    self.mels[index] = self.transform(torch.from_numpy(samples)[None]).clamp(min=1e-5).log().squeeze(0)
            finally:
                torch.set_num_threads(threads)
        return self.mels[index]

    def batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """(mels (batch, n_mels, frames) zero-padded, lengths)."""
        import torch
        import torch.nn.functional as F

        mels = [self.mel(i) for i in indices]
        lengths = torch.LongTensor([m.shape[-1] for m in mels])
        longest = lengths.amax()
        return torch.stack([F.pad(m, (0, int(longest - m.size(-1))), value=0) for m in mels]), lengths


# -- the loss -------------------------------------------------------------------------------------


def cfm_loss(transformer: Any, inp: torch.Tensor, text: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """``CFM.forward``: noise a random 70–100 % span of each mel towards it, predict the flow there from the rest
    (the audio condition dropped 30 % of the time, audio and text together 20 %), and average the squared error
    over the span."""
    import random

    import torch
    import torch.nn.functional as F

    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, inp.device
    mask = torch.arange(seq_len, device=device)[None, :] < lens[:, None]
    frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*FRAC_LENGTHS)
    lengths = (frac_lengths * lens).long()
    max_start = lens - lengths
    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths
    seq = torch.arange(int(lens.max().item()), device=start.device).long()
    rand_span_mask = (seq[None, :] >= start[:, None]) & (seq[None, :] < end[:, None])
    rand_span_mask &= mask
    x1 = inp
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    phi = (1 - t) * x0 + t * x1
    flow = x1 - x0
    cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
    drop_audio_cond = random.random() < AUDIO_DROP
    if random.random() < COND_DROP:
        drop_audio_cond = drop_text = True
    else:
        drop_text = False
    pred = backbone_forward(transformer, phi, cond, text, time, mask, drop_audio_cond, drop_text)
    loss = F.mse_loss(pred, flow, reduction="none")
    return loss[rand_span_mask].mean()


def backbone_forward(
    transformer: nn.Module, x: torch.Tensor, cond: torch.Tensor, text: torch.Tensor, time: torch.Tensor, mask: torch.Tensor, drop_audio_cond: bool, drop_text: bool
) -> torch.Tensor:
    """The backbone's training forward (no CFG packing, no text cache, dropout on), operation for operation as
    upstream's ``DiT.forward``/``UNetT.forward`` run it, on the native modules' weights."""
    from mizuoto.nn.transformer.unett import UNetT

    if isinstance(transformer, UNetT):
        return _unett_forward(transformer, x, cond, text, time, mask, drop_audio_cond, drop_text)
    return _dit_forward(transformer, x, cond, text, time, mask, drop_audio_cond, drop_text)


def _dit_text(module: Any, text: torch.Tensor, seq_len: torch.Tensor, drop_text: bool) -> torch.Tensor:
    """``TextEmbedding.forward`` with per-sample lengths (the training path): positions past each length stay zero."""
    import torch
    import torch.nn.functional as F

    text = text + 1
    seq_len = seq_len.to(device=text.device, dtype=torch.long)
    max_seq_len = int(seq_len.max().item())
    text = text[:, :max_seq_len]
    text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)
    valid = torch.arange(max_seq_len, device=text.device).unsqueeze(0) < seq_len.unsqueeze(1)
    text = text.masked_fill(~valid, 0)
    text_mask = text == 0
    if drop_text:
        text = torch.zeros_like(text)
    out = module.text_embed(text)
    out = out.masked_fill(~valid.unsqueeze(-1), 0.0)
    if module.extra_modeling:
        freqs = module.get_buffer("freqs_cis")[:max_seq_len, :]
        out = out + freqs.unsqueeze(0) * valid.unsqueeze(-1).to(freqs.dtype)
        if module.mask_padding:
            fill = text_mask.unsqueeze(-1).expand(-1, -1, out.size(-1))
            out = out.masked_fill(fill, 0.0)
            for block in module.text_blocks:
                out = block(out)
                out = out.masked_fill(fill, 0.0)
        else:
            out = module.text_blocks(out)
    return out


def _attention(attn: Any, x: torch.Tensor, mask: torch.Tensor | None, rope: torch.Tensor) -> torch.Tensor:
    """``Attention`` with ``AttnProcessor`` (no attention mask), its output dropout included."""
    import torch.nn.functional as F

    from mizuoto.nn.transformer.dit import SITE, apply_rotary
    from mizuoto.runtime.accel import attention

    b = x.shape[0]
    q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
    head_dim = k.shape[-1] // attn.heads
    q = q.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    k = k.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    v = v.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.pe_attn_head is not None:
        pn = attn.pe_attn_head
        q[:, :pn, :, :] = apply_rotary(q[:, :pn, :, :], rope, 1.0)
        k[:, :pn, :, :] = apply_rotary(k[:, :pn, :, :], rope, 1.0)
    else:
        q = apply_rotary(q, rope, 1.0)
        k = apply_rotary(k, rope, 1.0)
    out = attention(q, k, v, site=SITE)
    out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim).to(q.dtype)
    out = F.dropout(attn.to_out[0](out), DROPOUT, training=True)
    if mask is not None:
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
    return out


def _feed_forward(ff: Any, x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return ff.ff[2](F.dropout(ff.ff[0](x), DROPOUT, training=True))


def _dit_forward(dit: Any, x: torch.Tensor, cond: torch.Tensor, text: torch.Tensor, time: torch.Tensor, mask: torch.Tensor, drop_audio_cond: bool, drop_text: bool) -> torch.Tensor:
    import torch

    seq_len = x.shape[1]
    t = dit.time_embed(time)
    text_embed = _dit_text(dit.text_embed, text, mask.sum(dim=1), drop_text)
    x = dit.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, mask=mask)
    rope = dit.rotary_embed.forward_from_seq_len(seq_len)
    residual = x
    for block in dit.transformer_blocks:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * _attention(block.attn, norm, mask, rope)
        norm = block.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * _feed_forward(block.ff, norm)
    if dit.long_skip_connection is not None:
        x = dit.long_skip_connection(torch.cat((x, residual), dim=-1))
    return dit.proj_out(dit.norm_out(x, t))


def _unett_forward(
    unet: Any, x: torch.Tensor, cond: torch.Tensor, text: torch.Tensor, time: torch.Tensor, mask: torch.Tensor, drop_audio_cond: bool, drop_text: bool
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    seq_len = x.shape[1]
    t = unet.time_embed(time)
    x = unet.input_embed(x, cond, unet.text_embed(text, seq_len, drop_text=drop_text), drop_audio_cond=drop_audio_cond)
    x = torch.cat([t.unsqueeze(1), x], dim=1)
    mask = F.pad(mask, (1, 0), value=1)
    rope = unet.rotary_embed.forward_from_seq_len(seq_len + 1)
    skips = []
    for idx, layer in enumerate(unet.layers):
        parts = dict(layer.named_children())
        if idx + 1 <= unet.depth // 2:
            skips.append(x)
        else:
            x = parts["0"](torch.cat((x, skips.pop()), dim=-1))
        x = _attention(parts["2"], parts["1"](x), mask, rope) + x
        x = _feed_forward(parts["4"], parts["3"](x)) + x
    return unet.proj_out(unet.norm_out(x)[:, 1:, :])


# -- previews -------------------------------------------------------------------------------------


def _reference(rows: list[Row]) -> Row | None:
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 12.0]
    return min(fitting, key=lambda r: abs(r.duration - 6.0)) if fitting else None


def _as_wav(row: Row, folder: Path) -> Row:
    """The row with its clip as 16-bit WAV at its own rate (the engine reads WAV; a 16-bit FLAC converts exactly)."""
    from mizuoto.nn.dsp.flac import read_audio
    from mizuoto.nn.dsp.wav import write_wav

    with Path(row.audio).open("rb") as f:
        if f.read(4) == b"RIFF":
            return row
    samples, sr = read_audio(row.audio)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"reference-{row.id}.wav"
    write_wav(target, samples, sr)
    return replace(row, audio=str(target), sample_rate=sr)
