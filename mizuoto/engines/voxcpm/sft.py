"""VoxCPM fine-tuning's maths (upstream ``scripts/train_voxcpm_finetune.py`` and ``training/``), for ``train.py``.

What one optimiser step does, as upstream:
1. **Batches.** Each micro-batch comes from a shuffled pass over the clips: the loader draws its base seed and the
   sampler its permutation seed from torch's generator, the batch is cut whole (``drop_last``), its text padded with
   -100 and its audio with -100.0 (``HFVoxCPMDataset.collate_fn``).
2. **Packing** (``AudioFeatureProcessingPacker``). Each clip is encoded by the VAE (padded to whole patches on the
   right) and becomes ``[text…, 101, 0×patches, 102]``. Its latents sit under the audio positions, the loss covers the
   audio and the stop label is the last patch.
3. **Losses.** The model's training ``forward`` gives the CFM loss and the stop head's cross-entropy (both
   generations).
4. **Update.** Losses scaled by the accumulation, backward, gradient clipping (1e9 when off), AdamW and transformers'
   cosine schedule with warm-up.

Upstream's draws, in their order: its ``Accelerator`` seeds every generator with 42, the VAE and the model are built
with torch's default initialisation (and LoRA's ``A`` with Kaiming), then each loader pass draws two seeds and each
forward the CFM's ``cfg_mask``, ``noisy_mask``, the condition noise, ``r``/``t`` and ``z`` (and LoRA's dropout masks).
Here the same draws come from private generators seeded 42: ``cpu`` for the construction (``replay_construction``
makes its draws again; only LoRA's ``A`` values are kept, the checkpoint replaces the rest) and the loader, ``device``
for the forward's (the same generator on a CPU, as upstream's single default one).

The clips are read as ``datasets``' ``Audio`` feature reads them: soundfile's float64, mixed to mono,
``librosa.resample`` (soxr) to the VAE's rate, then float32 in the collator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.voxcpm.config import CfmConfig, VoxCpmConfig
from mizuoto.engines.voxcpm.model import AUDIO_START, FlowMatching, LoraConfig, LoRALinear, VoxCpmModel
from mizuoto.engines.voxcpm.vae import AudioVAE

AUDIO_END = 102
PAD = -100


# -- construction draws ----------------------------------------------------------------


def _layer_draws(module: nn.Module, g: torch.Generator) -> None:
    """The draws ``module``'s own ``reset_parameters`` makes (upstream's layers are torch's, initialised). A LoRA layer
    stands where its base ``nn.Linear`` was built, so it draws that layer's initialisation (its own ``A`` comes later)."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, LoRALinear)):
        weight = torch.empty(module.weight.shape)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=g)
        if module.bias is not None:
            torch.empty(module.bias.shape).uniform_(generator=g)
    elif isinstance(module, nn.Embedding):
        torch.empty(module.weight.shape).normal_(generator=g)


def _tree_draws(root: nn.Module, g: torch.Generator) -> None:
    """Every layer of ``root`` in construction order (pre-order: a module's own draws before its children's)."""
    for module in root.modules():
        if hasattr(module, "special_token") and isinstance(module.special_token, nn.Parameter):  # VoxCPMLocEnc: torch.randn
            torch.randn(module.special_token.shape, generator=g)
        _layer_draws(module, g)


def replay_construction(model: VoxCpmModel, vae: AudioVAE, lora_layers: list[LoRALinear], g: torch.Generator) -> None:
    """Make upstream's construction draws on torch's generator: the VAE (built first), the model, then LoRA's ``A``,
    whose Kaiming values are kept (in the order ``apply_lora`` made the layers, which is upstream's)."""
    _tree_draws(vae, g)
    _tree_draws(model, g)
    with torch.no_grad():
        for layer in lora_layers:
            if layer.lora_A is not None:
                values = torch.zeros(layer.lora_A.shape)
                nn.init.kaiming_uniform_(values, a=math.sqrt(5), generator=g)
                layer.lora_A.copy_(values)
                assert layer.lora_B is not None
                layer.lora_B.zero_()


# -- data ------------------------------------------------------------------------------


def read_clip(path: str, rate: int) -> Any:
    """A dataset clip as ``datasets.Audio(sampling_rate=rate)`` decodes it: float64, mono, soxr to ``rate``."""
    import numpy as np

    from mizuoto.nn.dsp.flac import read_audio
    from mizuoto.nn.dsp.resample import soxr_resample

    samples, sr = read_audio(path)  # dataset clips are mono 16-bit: exact in float32, so in float64
    y = np.asarray(samples, dtype=np.float64)
    if sr != rate:
        y = soxr_resample(y, sr, rate)
    return y


@dataclass
class Clip:
    text_ids: list[int]
    audio: Any
    """float64 samples at the VAE's rate."""


def collate(clips: list[Clip]) -> dict[str, Any]:
    """``HFVoxCPMDataset.collate_fn`` (no reference audio, no dataset ids): padded with -100."""
    text = [torch.tensor(c.text_ids, dtype=torch.int32) for c in clips]
    audio = [torch.tensor(c.audio, dtype=torch.float32) for c in clips]

    def pad(seqs: list[torch.Tensor], value: float) -> torch.Tensor:
        longest = max(s.shape[0] for s in seqs)
        return torch.stack([F.pad(s, (0, longest - s.shape[0]), value=value) if s.shape[0] < longest else s for s in seqs])

    return {"text_tokens": pad(text, PAD), "audio_tokens": pad(audio, float(PAD)), "count": len(clips)}


class Packer:
    """``AudioFeatureProcessingPacker`` for TTS samples without a reference or prompt flag."""

    def __init__(self, vae: AudioVAE, config: VoxCpmConfig) -> None:
        self.vae = vae
        self.patch_size = config.patch_size
        self.patch_len = vae.hop_length * config.patch_size
        self.feat_dim = config.feat_dim
        self.max_len = config.max_length

    @staticmethod
    def _unpad(tokens: torch.Tensor) -> torch.Tensor:
        positions = (tokens == PAD).nonzero(as_tuple=True)
        return tokens if positions[0].numel() == 0 else tokens[: int(positions[0][0])]

    def _feats(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.size(-1) % self.patch_len != 0:
            wav = F.pad(wav, (0, self.patch_len - wav.size(-1) % self.patch_len))
        with torch.no_grad():
            feat = self.vae.encode(wav).transpose(1, 2)
        if feat.size(1) % self.patch_size != 0:
            feat = F.pad(feat.transpose(1, 2), (0, self.patch_size - feat.size(1) % self.patch_size)).transpose(1, 2)
        b, t, c = feat.shape
        return feat.reshape(b, t // self.patch_size, self.patch_size, c)

    def _sample(self, audio: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, ...]:
        device = text.device
        text_info = torch.cat([text, torch.tensor([AUDIO_START], dtype=torch.int32, device=device)], dim=-1)
        text_length = text_info.shape[0]
        feat = self._feats(audio).squeeze(0)
        audio_length = feat.shape[0]
        text_info = torch.cat([text_info, torch.zeros(audio_length, dtype=torch.int32, device=device), torch.tensor([AUDIO_END], dtype=torch.int32, device=device)])
        pad_feat = torch.zeros((text_length, self.patch_size, feat.size(-1)), dtype=torch.float32, device=device)
        feat = torch.cat([pad_feat, feat, pad_feat[0:1, ...]], dim=0)
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length), torch.ones(1)]).type(torch.int32).to(device)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length), torch.zeros(1)]).type(torch.int32).to(device)
        loss_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length), torch.zeros(1)]).type(torch.int32).to(device)
        labels = torch.zeros(text_length + audio_length + 1).type(torch.int32).to(device)
        labels[-2] = 1
        return text_info, feat, text_mask, audio_mask, loss_mask, labels

    def __call__(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        parts = [self._sample(self._unpad(a).to(torch.float32), self._unpad(t)) for a, t in zip(audio_tokens, text_tokens)]
        max_len = min(self.max_len, max(p[0].shape[0] for p in parts))

        def pad(x: torch.Tensor) -> torch.Tensor:
            if x.size(0) >= max_len:
                return x[:max_len]
            return torch.cat([x, torch.zeros((max_len - x.size(0),) + x.shape[1:], dtype=x.dtype, device=x.device)], dim=0)

        names = ("text_tokens", "audio_feats", "text_mask", "audio_mask", "loss_mask", "labels")
        return {name: torch.stack([pad(p[i]) for p in parts], dim=0) for i, name in enumerate(names)}


class Loader:
    """The ``DataLoader(shuffle=True, drop_last=True)`` passes: each pass draws the loader's base seed and the sampler's
    seed from torch's generator (in that order), then a permutation on a generator seeded with the latter."""

    def __init__(self, clips: list[Clip], batch_size: int, generator: torch.Generator) -> None:
        self.generator = generator
        self.clips = clips
        self.batch_size = batch_size
        self.epoch = 0
        self._batches = self._pass()

    def _pass(self) -> Iterator[list[Clip]]:
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)  # _BaseDataLoaderIter._base_seed (no worker draws)
        seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())  # RandomSampler
        order = torch.randperm(len(self.clips), generator=torch.Generator().manual_seed(seed)).tolist()
        for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
            yield [self.clips[i] for i in order[start : start + self.batch_size]]

    def next(self) -> dict[str, Any]:
        try:
            return collate(next(self._batches))
        except StopIteration:
            self.epoch += 1
            self._batches = self._pass()
            return collate(next(self._batches))  # an empty pass raises, as upstream's loop does


def sample_lengths(clips: list[Clip], durations: list[float], fps: float, patch_size: int) -> list[int]:
    """``compute_sample_lengths``: text tokens + patches + 2 (the length ``max_batch_tokens`` filters on)."""
    out = []
    for clip, duration in zip(clips, durations):
        t_vae = math.ceil(float(duration) * fps)
        out.append(len(clip.text_ids) + math.ceil(t_vae / patch_size) + 2)
    return out


# -- the losses ------------------------------------------------------------------------


def sample_r_t(t_scheduler: str, x: torch.Tensor, ratio_r_neq_t: float, g: torch.Generator, mu: float = -0.4, sigma: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    if t_scheduler == "log-norm":
        s_r = torch.randn(batch_size, device=x.device, dtype=x.dtype, generator=g) * sigma + mu
        s_t = torch.randn(batch_size, device=x.device, dtype=x.dtype, generator=g) * sigma + mu
        r, t = torch.sigmoid(s_r), torch.sigmoid(s_t)
    elif t_scheduler == "uniform":
        r = torch.rand(batch_size, device=x.device, dtype=x.dtype, generator=g)
        t = torch.rand(batch_size, device=x.device, dtype=x.dtype, generator=g)
    else:
        raise ValueError(f"Unsupported t_scheduler: {t_scheduler}")
    mask = torch.rand(batch_size, device=x.device, dtype=x.dtype, generator=g) < ratio_r_neq_t
    r, t = torch.where(mask, torch.stack([torch.min(r, t), torch.max(r, t)], dim=0), torch.stack([t, t], dim=0))
    return r.squeeze(), t.squeeze()


def cfm_loss(
    cfm: FlowMatching, cfg: CfmConfig, x1: torch.Tensor, mu: torch.Tensor, cond: torch.Tensor, tgt_mask: torch.Tensor, progress: float, g: torch.Generator
) -> torch.Tensor:
    """``UnifiedCFM.compute_loss`` (no mean mode: none of the released models uses it). ``randn_like`` of these contiguous
    tensors draws as ``randn`` of their shape does."""
    if cfm.mean_mode:
        raise NotImplementedError("VoxCPM training in mean mode")
    b = x1.shape[0]
    if cfg.training_cfg_rate > 0:
        cfg_mask = torch.rand(b, device=x1.device, generator=g) > cfg.training_cfg_rate
        mu = mu * cfg_mask.view(-1, 1)
    low, high = cfg.noise_cond_prob_range
    noisy_mask = torch.rand(b, device=x1.device, generator=g) > (1.0 - (low + progress * (high - low)))
    cond = cond + noisy_mask.view(-1, 1, 1) * torch.randn(cond.shape, device=cond.device, dtype=cond.dtype, generator=g) * cfg.noise_cond_scale
    r, t = sample_r_t(cfg.t_scheduler, x1, 0.0, g)
    t_ = t.detach().clone()
    z = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=g)
    y = (1 - t_.view(-1, 1, 1)) * x1 + t_.view(-1, 1, 1) * z
    v = z - x1
    u_pred = cfm.estimator(y, mu, t, cond, t - r)
    losses = F.mse_loss(u_pred, v.detach(), reduction="none").mean(dim=1)
    weights = (1.0 / ((losses + 1e-3).pow(0.0)) * tgt_mask.squeeze(1)).detach()
    return (weights * losses).sum() / torch.clamp(torch.sum(tgt_mask), min=1.0)


def forward_losses(model: VoxCpmModel, batch: dict[str, torch.Tensor], dtype: torch.dtype, progress: float, g: torch.Generator) -> dict[str, torch.Tensor]:
    """The training ``forward`` of ``VoxCPMModel`` / ``VoxCPM2Model``: ``loss/diff`` and ``loss/stop``."""
    m = model
    device = next(m.parameters()).device
    text_tokens = batch["text_tokens"].to(device, dtype=torch.long)
    text_mask = batch["text_mask"].to(device, dtype=dtype)
    audio_feats = batch["audio_feats"].to(device, dtype=dtype)
    audio_mask = batch["audio_mask"].to(device, dtype=dtype)
    loss_mask = batch["loss_mask"].to(device, dtype=dtype)
    labels = batch["labels"].to(device, dtype=torch.long)
    b, t, p, d = audio_feats.shape
    feat_embed = m.enc_to_lm_proj(m.feat_encoder(audio_feats))
    text_embed = m.base_lm.embed_tokens(text_tokens) * m.scale_emb
    combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
    enc_outputs, _ = m.base_lm(combined_embed, is_causal=True)
    enc_outputs = enc_outputs.to(dtype)
    enc_outputs = m.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
    lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)
    if m.version == 2:
        residual_inputs = m.fusion_concat_proj(torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1))
    else:
        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
    residual_outputs, _ = m.residual_lm(residual_inputs, is_causal=True)
    residual_outputs = residual_outputs.to(dtype)
    residual_hidden = torch.cat((torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]), dim=1)
    if m.version == 2:
        dit_hidden = torch.cat((m.lm_to_dit_proj(lm_hidden), m.res_to_dit_proj(residual_hidden)), dim=-1)
    else:
        dit_hidden = m.lm_to_dit_proj(lm_hidden) + m.res_to_dit_proj(residual_hidden)
    dit_hidden = dit_hidden.reshape(b * t, -1)
    feat_gt = audio_feats.to(dtype).reshape(b * t, p, d)
    feat_cond = torch.cat((torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]), dim=1).to(dtype).reshape(b * t, p, d)
    loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, p).reshape(b * t, p, 1).to(dtype)
    diff_loss = cfm_loss(
        m.feat_decoder,
        m.config.cfm,
        feat_gt.transpose(1, 2).contiguous(),
        dit_hidden,
        feat_cond.transpose(1, 2).contiguous(),
        loss_seq_mask.transpose(1, 2).contiguous(),
        progress,
        g,
    )
    stop_logits = m.stop_head(m.stop_actn(m.stop_proj(lm_hidden)))
    stop_losses = F.cross_entropy(stop_logits.transpose(1, 2), labels, reduction="none")
    stop_loss = (stop_losses * loss_mask).sum() / torch.clamp(loss_mask.sum(), min=1.0)
    return {"loss/diff": diff_loss, "loss/stop": stop_loss}


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup: int, total: int, num_cycles: float = 0.5) -> torch.optim.lr_scheduler.LambdaLR:
    """transformers' ``get_cosine_schedule_with_warmup``."""

    def factor(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@dataclass
class StepResult:
    step: int
    losses: dict[str, float]
    lr: float
    grad_norm: float
    epoch: float


def train_steps(
    model: VoxCpmModel,
    loader: Loader,
    packer: Packer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    steps: int,
    grad_accum: int,
    max_grad_norm: float,
    dtype: torch.dtype,
    device: torch.device,
    clips: int,
    batch_size: int,
    autocast: Callable[[], Any],
    scaler: Any,
    on_step: Callable[[StepResult], None],
    check: Callable[[], None],
    generator: torch.Generator,
    lambdas: dict[str, float] | None = None,
) -> None:
    """Upstream's training loop (one process, no validation set)."""
    lambdas = lambdas or {"loss/diff": 1.0, "loss/stop": 1.0}
    grad_accum = max(int(grad_accum), 1)
    for step in range(steps):
        check()
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        for _ in range(grad_accum):
            processed = packer(loader.next(), device)
            with autocast():
                outputs = forward_losses(model, processed, dtype, step / max(1, steps), generator)
            total: Any = 0.0
            for key, value in outputs.items():
                total = total + value * lambdas.get(key, 1.0) / grad_accum
                losses[key] = value.detach()
            scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm if max_grad_norm > 0 else 1e9)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        on_step(
            StepResult(
                step=step,
                losses={k: float(v.item()) for k, v in losses.items()},
                lr=float(optimizer.param_groups[0]["lr"]),
                grad_norm=float(grad_norm),
                epoch=float(step * grad_accum * batch_size) / max(1, clips),
            )
        )


class PlainScaler:
    """Upstream's ``DummyScaler`` (no CUDA): scaling does nothing."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def trainable(model: VoxCpmModel, lora: LoraConfig | None) -> list[nn.Parameter]:
    """``from_local(training=True)``: with LoRA only its ``lora_*`` parameters train, else all of them."""
    out = []
    for name, param in model.named_parameters():
        param.requires_grad_(lora is None or "lora" in name)
        if param.requires_grad:
            out.append(param)
    return out
