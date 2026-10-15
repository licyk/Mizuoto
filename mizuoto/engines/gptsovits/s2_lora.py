"""GPT-SoVITS' SoVITS LoRA training for v4 and v5 Turbo (MIT; ``s2_train_v3_lora.py``), without peft.

Upstream wraps the CFM in ``peft.get_peft_model`` with ``LoraConfig(target_modules=["to_k", "to_q", "to_v",
"to_out.0"], r=rank, lora_alpha=rank, init_lora_weights=True)``: every attention projection of the DiT gets
``W x + B A x × α/r`` (here α = r: scaling 1), A kaiming-uniform and B zero, the rest of the CFM frozen. The text
encoder, the SSL projection and the quantizer are frozen too (``set_no_grad``); the style encoder, ``bridge``, ``wns1``
and ``linear_mel`` stay trainable. ``LoraLinear`` keeps peft's names (``base_layer``, ``lora_A.default``,
``lora_B.default``); the saved state gets peft's ``cfm.base_model.model.`` prefix, which ``sovits_v3.merge_lora`` reads.

The training forward is ``SynthesizerTrnV3.forward`` with ``CFM.forward`` (v4: the shortcut-model branch taken with
probability 0.3 from Python's generator, two extra estimator passes) or ``CFMV5.forward`` (v5), on the native modules,
the DiT with its dropout (0.1) on. The initialisation is drawn as upstream draws it: its first attempt to resume builds
the model and the LoRA, finds nothing and builds both again (the pretrained weights loaded in between).
"""

from __future__ import annotations

import math
import random
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mizuoto.engines.gptsovits import s2_modules as M
from mizuoto.engines.gptsovits.sovits_v3 import LORA_TARGETS, PEFT_PREFIX, SITE, V5_VERSIONS, SynthesizerTrnV3, mel_v4, norm_spec
from mizuoto.nn.transformer.dit import apply_rotary
from mizuoto.nn.transformer.vits import sequence_mask
from mizuoto.runtime.accel import MaskInfo, attention

if TYPE_CHECKING:
    from mizuoto.engines.gptsovits.train_s2 import BucketSampler, S2Data
    from mizuoto.training import StageContext

DIT_DROPOUT = 0.1
"""``DiT``'s default, which ``SynthesizerTrnV3`` keeps."""
SHORTCUT_PROBABILITY = 0.3
"""``CFM.forward``'s ``gailv``."""
V3_ARCH: dict[str, Any] = {}
"""Overrides of ``SynthesizerTrnV3``'s sizes (``bridge_channels``, ``wn_layers``, ``dit``): for tests only."""


def item_mel(audio: torch.Tensor) -> torch.Tensor:
    """``TextAudioSpeakerLoaderV4.get_audio``'s normalised 100-band mel (32 kHz, hop 320)."""
    return norm_spec(mel_v4(audio).squeeze(0))


def collate_v4(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """``TextAudioSpeakerCollateV4``: (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths)."""
    _, order = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
    max_ssl_len = int(2 * ((max(x[0].size(2) for x in batch) // 2) + 1))
    max_spec_len = int(2 * ((max(x[1].size(1) for x in batch) // 2) + 1))
    max_text_len = max(x[3].size(0) for x in batch)
    n = len(batch)
    ssl_lengths, spec_lengths, text_lengths, mel_lengths = (torch.LongTensor(n) for _ in range(4))
    spec_padded = torch.FloatTensor(n, batch[0][1].size(0), max_spec_len).zero_()
    mel_padded = torch.FloatTensor(n, batch[0][2].size(0), max_spec_len * 2).zero_()
    ssl_padded = torch.FloatTensor(n, batch[0][0].size(1), max_ssl_len).zero_()
    text_padded = torch.LongTensor(n, max_text_len).zero_()
    for i in range(n):
        row = batch[order[i]]
        ssl_padded[i, :, : row[0].size(2)] = row[0][0, :, :]
        ssl_lengths[i] = row[0].size(2)
        spec_padded[i, :, : row[1].size(1)] = row[1]
        spec_lengths[i] = row[1].size(1)
        mel_padded[i, :, : row[2].size(1)] = row[2]
        mel_lengths[i] = row[2].size(1)
        text_padded[i, : row[3].size(0)] = row[3]
        text_lengths[i] = row[3].size(0)
    return ssl_padded, spec_padded, mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, mel_lengths


# -- LoRA -----------------------------------------------------------------------------------------


class LoraLinear(nn.Module):
    """peft's ``lora.Linear`` (vanilla, no dropout, one adapter named ``default``): ``base(x) + B(A(x)) × scaling``."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        # peft makes both with nn.Linear's own initialisation, then redraws A and zeroes B.
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base_layer.in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, base_layer.out_features, bias=False)})
        self.scaling = alpha / rank
        a: Any = self.lora_A["default"]
        b: Any = self.lora_B["default"]
        with torch.no_grad():
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        base_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        return result + self.lora_B["default"](self.lora_A["default"](x)) * self.scaling


def add_lora(cfm: Any, rank: int) -> None:
    """``get_peft_model(cfm, LoraConfig(target_modules=LORA_TARGETS, r=rank, lora_alpha=rank))``: the targets in
    ``named_modules`` order, every other CFM parameter frozen."""
    cfm.requires_grad_(False)
    for block in cfm.estimator.transformer_blocks:
        attn: Any = block.attn
        attn.to_q = LoraLinear(attn.to_q, rank, rank)
        attn.to_k = LoraLinear(attn.to_k, rank, rank)
        attn.to_v = LoraLinear(attn.to_v, rank, rank)
        attn.to_out[0] = LoraLinear(attn.to_out[0], rank, rank)


# -- building ------------------------------------------------------------------------------------


@torch.no_grad()
def init_dit(dit: Any) -> None:
    """``DiT.__init__``'s draws in its order: time (and step) embeddings, the text blocks, the input embedding, the
    blocks (adaptive norm, q, k, v, out, feed-forward), the final norm and projection."""
    embeds = [dit.time_embed] + ([dit.d_embed] if dit.d_embed is not None else [])
    for embed in embeds:
        M._reset(embed.time_mlp[0], embed.time_mlp[2])
    if dit.text_embed.extra_modeling:
        for block in dit.text_embed.text_blocks:
            M._reset(block.dwconv)
            block.norm.weight.fill_(1.0)
            block.norm.bias.zero_()
            M._reset(block.pwconv1)
            block.grn.gamma.zero_()
            block.grn.beta.zero_()
            M._reset(block.pwconv2)
    M._reset(dit.input_embed.proj, dit.input_embed.conv_pos_embed.conv1d[0], dit.input_embed.conv_pos_embed.conv1d[2])
    for block in dit.transformer_blocks:
        attn = block.attn
        M._reset(block.attn_norm.linear, attn.to_q, attn.to_k, attn.to_v, attn.to_out[0], block.ff.ff[0][0], block.ff.ff[2])
    M._reset(dit.norm_out.linear, dit.proj_out)


@torch.no_grad()
def init_synthesizer(model: Any) -> None:
    """``SynthesizerTrnV3.__init__``'s draws: enc_p, ref_enc, ssl_proj, (the quantizer: none), bridge, wns1,
    linear_mel, the DiT."""
    M.init_text_encoder(model.enc_p)
    M.init_style_encoder(model.ref_enc)
    M._reset(model.ssl_proj)
    M.init_quantizer(model.quantizer)
    M._reset(model.bridge[0], model.wns1.pre)
    M._init_wn(model.wns1.enc)
    M._reset(model.wns1.proj, model.linear_mel)
    init_dit(model.cfm.estimator)


def build_synthesizer(version: str, hps: dict[str, Any], n_symbols: int = 732) -> SynthesizerTrnV3:
    """The training model from the config ``hps``: upstream's initialisation drawn, ``wns1`` weight-normed."""
    from mizuoto.engines.gptsovits.sovits_v3 import model_kwargs

    model: Any = SynthesizerTrnV3(version, n_symbols=n_symbols, **model_kwargs(hps), **V3_ARCH)
    M.add_codebook_state(model)
    init_synthesizer(model)
    M.weight_norm_wn(model.wns1.enc)
    return model


# -- training forwards ---------------------------------------------------------------------------


def _drop(x: torch.Tensor) -> torch.Tensor:
    return F.dropout(x, DIT_DROPOUT, True)


def _attention(attn: Any, x: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """``Attention`` + ``AttnProcessor`` (GPT-SoVITS): rotary on the projections, SDPA with the mask, dropout."""
    batch_size = x.shape[0]
    query = apply_rotary(attn.to_q(x), rope, 1.0)
    key = apply_rotary(attn.to_k(x), rope, 1.0)
    value = attn.to_v(x)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    attn_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
    out = attention(query, key, value, site=SITE, mask=attn_mask, mask_info=MaskInfo(keys=mask, padding="right"))
    out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
    out = _drop(attn.to_out[0](out))
    return out.masked_fill(~mask.unsqueeze(-1), 0.0)


def _block(block: Any, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(x, emb=t)
    x = x + gate_msa.unsqueeze(1) * _attention(block.attn, norm, mask, rope)
    norm = block.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff = block.ff.ff
    return x + gate_mlp.unsqueeze(1) * ff[2](_drop(ff[0](norm)))


def dit_forward(
    dit: Any, x0: torch.Tensor, cond0: torch.Tensor, x_lens: torch.Tensor, time: torch.Tensor, dt_base: torch.Tensor | None, text0: torch.Tensor, grad_ckpt: bool
) -> torch.Tensor:
    """``DiT.forward`` in training (no caches, no classifier-free drops): the velocity (batch, frames, mel)."""
    x = x0.transpose(2, 1)
    cond = cond0.transpose(2, 1)
    text = text0.transpose(2, 1)
    mask = sequence_mask(x_lens, max_length=x.size(1)).to(x.device)
    seq_len = x.shape[1]
    t = dit.time_embed(time)
    if dit.d_embed is not None:
        assert dt_base is not None
        t += dit.d_embed(dt_base)
    text_embed = dit.text_embed(text, seq_len, drop_text=False)
    x = dit.input_embed(x, cond, text_embed, drop_audio_cond=False)
    rope = dit.rotary_embed.forward_from_seq_len(seq_len)
    for block in dit.transformer_blocks:
        if grad_ckpt:
            # checkpoint keeps the generators' state for the recomputation (``preserve_rng_state``), as upstream's does
            x = checkpoint(partial(_block, block), x, t, mask, rope, use_reentrant=False)
        else:
            x = _block(block, x, t, mask, rope)
    x = dit.norm_out(x, t)
    return dit.proj_out(x)


def cfm_forward(cfm: Any, x1: torch.Tensor, x_lens: torch.Tensor, prompt_lens: torch.Tensor, mu: torch.Tensor, grad_ckpt: bool, rng: random.Random) -> torch.Tensor:
    """``CFM.forward`` (v4: the shortcut branch with probability 0.3) or ``CFMV5.forward`` (v5): the flow-matching loss."""
    dit = cfm.estimator
    b = x1.shape[0]
    t = torch.rand([b], device=mu.device, dtype=x1.dtype)
    x0 = torch.randn_like(x1, device=mu.device)
    vt = x1 - x0
    xt = x0 + t[:, None, None] * vt
    v5 = not cfm.use_step_embedding
    dt = None if v5 else torch.zeros_like(t, device=mu.device)
    prompt = torch.zeros_like(x1)
    for i in range(b):
        prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
        xt[i, :, : prompt_lens[i]] = 0
    if not v5 and rng.random() < SHORTCUT_PROBABILITY:
        base = torch.randint(2, 8, (t.shape[0],), device=mu.device)
        d = 1 / torch.pow(2, base)
        d_input = d.clone()
        d_input[d_input < 1e-2] = 0
        v_pred_1 = dit_forward(dit, xt, prompt, x_lens, t, d_input, mu, grad_ckpt).transpose(2, 1).detach()
        x_mid = xt + d[:, None, None] * v_pred_1
        v_pred_2 = dit_forward(dit, x_mid, prompt, x_lens, t + d, d_input, mu, grad_ckpt).transpose(2, 1).detach()
        vt = ((v_pred_1 + v_pred_2) / 2).detach()
        dt = 2 * d
    vt_pred = dit_forward(dit, xt, prompt, x_lens, t, dt, mu, grad_ckpt).transpose(2, 1)
    loss: Any = 0
    for i in range(b):
        loss += F.mse_loss(vt_pred[i, :, prompt_lens[i] : x_lens[i]], vt[i, :, prompt_lens[i] : x_lens[i]])
    return loss / b


def lora_forward(
    model: Any,
    ssl: torch.Tensor,
    y: torch.Tensor,
    mel: torch.Tensor,
    y_lengths: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
    grad_ckpt: bool,
    rng: random.Random,
) -> torch.Tensor:
    """``SynthesizerTrnV3.forward``: the style vector (dropout on), the frozen text encoder (eval: no dropout), the
    bridge and ``wns1`` (its condition detached, as ``models.Encoder`` does), a random prompt length, the CFM loss."""
    with torch.autocast(ssl.device.type, enabled=False):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        ge = M.style_encoder_forward(model.ref_enc, y[:, :704] * y_mask, y_mask)
        with torch.no_grad():
            quantized = model.quantizer.decode(model.quantizer.encode(model.ssl_proj(ssl), n_q=1))
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
            x = model.enc_p.features(quantized, y_lengths, text, text_lengths, ge)
    fea = model.bridge(x)
    fea = F.interpolate(fea, scale_factor=2, mode="nearest")
    wns1 = model.wns1
    fea_mask = torch.unsqueeze(sequence_mask(mel_lengths, fea.size(2)), 1).to(fea.dtype)
    h = wns1.pre(fea) * fea_mask
    h = M.wn_forward(wns1.enc, h, fea_mask, ge.detach())
    fea = wns1.proj(h) * fea_mask
    b = ssl.shape[0]
    prompt_len_max = mel_lengths * 2 / 3
    prompt_len = (torch.rand([b], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)
    minn = min(mel.shape[-1], fea.shape[-1])
    return cfm_forward(model.cfm, mel[:, :, :minn], mel_lengths, prompt_len, fea[:, :, :minn], grad_ckpt, rng)


# -- the loop ------------------------------------------------------------------------------------


def train_lora(ctx: StageContext, hps: dict[str, Any], data: S2Data, sampler: BucketSampler, device: torch.device, fp16: bool, save_dir: Path) -> dict[str, Any]:
    from mizuoto.core import codes
    from mizuoto.engines.gptsovits import train_s2 as T
    from mizuoto.engines.gptsovits.checkpoints import load_checkpoint

    version = str(hps["model"]["version"])
    train = hps["train"]
    rank = int(train["lora_rank"])
    base = load_checkpoint(train["pretrained_s2G"])["weight"]
    emb = base.get("enc_p.text_embedding.weight")
    n_symbols = int(emb.shape[0]) if emb is not None else 732
    # Upstream first tries to resume: it builds the model and its LoRA, finds no checkpoint, and builds both again.
    first = build_synthesizer(version, hps, n_symbols)
    add_lora(first.cfm, rank)
    del first
    net_g = build_synthesizer(version, hps, n_symbols)
    T._load(net_g, base, f"SoVITS {version}")
    del base
    _freeze(net_g)
    add_lora(net_g.cfm, rank)
    net_g.to(device)
    frozen = {name for name, p in net_g.named_parameters() if not p.requires_grad}

    betas = (float(train["betas"][0]), float(train["betas"][1]))
    optim_g = torch.optim.AdamW([p for p in net_g.parameters() if p.requires_grad], float(train["learning_rate"]), betas=betas, eps=float(train["eps"]))
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=float(train["lr_decay"]), last_epoch=-1)
    scheduler_g.step()  # once before training, as upstream (``for _ in range(epoch_str)``)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    grad_ckpt = bool(train["grad_ckpt"])
    epochs, save_every = int(train["epochs"]), int(train["save_every_epoch"])
    total = epochs * len(sampler)
    global_step = 0
    saved: list[str] = []
    rng = data.rng  # Python's generator after the loader's shuffle, as upstream's global one is
    ctx.progress(0.0, f"Epoch 1 of {epochs}")
    for epoch in range(1, epochs + 1):
        batches = T.epoch_batches(sampler, epoch)
        if epoch == 1:
            torch.empty((), dtype=torch.int64).random_()  # the data loader's base seed
        net_g.train()
        for indices in batches:
            ssl, spec, mel, _, spec_lengths, text, text_lengths, mel_lengths = (t.to(device) for t in collate_v4([data.item(i) for i in indices]))
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                loss = lora_forward(net_g, ssl, spec, mel, spec_lengths, text, text_lengths, mel_lengths, grad_ckpt, rng)
            optim_g.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim_g)
            scaler.step(optim_g)
            scaler.update()
            ctx.metric(epoch, global_step, {"cfm_loss": float(loss.item())}, float(optim_g.param_groups[0]["lr"]))
            global_step += 1
            ctx.progress(min(1.0, global_step / max(1, total)), f"Epoch {epoch} of {epochs}")
        if epoch % save_every == 0:
            path = save_dir / f"{hps['name']}_e{epoch}_s{global_step}_l{rank}.pth"
            T.save_weights(lora_state(net_g, frozen), path, hps, epoch, global_step, lora_rank=rank)
            saved.append(str(path))
            ctx.output(path, f"SoVITS LoRA epoch {epoch}", kind="checkpoint")
        scheduler_g.step()
    if not saved:
        from mizuoto.runtime.engine import EngineError

        raise EngineError("Training saved no weights (save_every_epoch is above the epochs)", code=codes.ENGINE_ERROR)
    return {"weights": saved[-1], "epochs": epochs, "steps": global_step, "files": saved, "lora_rank": rank}


def _freeze(model: Any) -> None:
    """``set_no_grad`` on the SSL projection, the quantizer and the text encoder (``freeze_quantizer``)."""
    for part in (model.ssl_proj, model.quantizer, model.enc_p):
        part.requires_grad_(False)


def lora_state(model: SynthesizerTrnV3, frozen: set[str]) -> dict[str, torch.Tensor]:
    """What upstream saves: the state dict without the frozen parameters (buffers stay), the CFM's keys under peft's
    ``cfm.base_model.model.`` prefix."""
    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key in frozen:
            continue
        out[PEFT_PREFIX + key[len("cfm.") :] if key.startswith("cfm.") else key] = value
    return out


__all__ = ["LORA_TARGETS", "V5_VERSIONS", "LoraLinear", "add_lora", "build_synthesizer", "lora_forward", "train_lora"]
