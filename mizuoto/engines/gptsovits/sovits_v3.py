"""GPT-SoVITS v4 and v5 SoVITS (MIT): ``SynthesizerTrnV3`` with its flow-matching mel generator, and the vocoder paths.

Ported from ``module/models.py`` (``SynthesizerTrnV3``, ``CFM``, ``Encoder``), ``module/models_v5.py`` (``CFMV5``,
``synthesize_v5_mel``), ``f5_tts/model`` (GPT-SoVITS's own copy of F5's DiT), ``module/mel_processing.py`` and
``TTS_infer_pack/TTS.py`` (``mel_fn_v4``, ``norm_spec``, ``using_vocoder_synthesis``,
``using_vocoder_synthesis_batched_infer``, ``sola_algorithm``). The parts v2 has too are v2's: the text encoder
(``TextEncoder.features``: no prior), the quantizer and the spectrogram from ``sovits``, ``WN`` from
``mizuoto.nn.flow.coupling``, the style encoder from ``mizuoto.nn.speaker.melstyle``. Parameter names are upstream's,
so its checkpoints load as they are.

How it differs from upstream, none of it in the numbers:
- the CFM's noise comes from an explicit ``torch.Generator`` (``torch.randn`` with the same shape, in the same order),
  so a generator seeded with *s* gives what upstream gives after ``torch.manual_seed(s)``;
- v5's static conditioning cache is always on (upstream's default; ``GSVV5_CFM_STATIC_CACHE=0`` turned it off);
- weight norm is folded at load (identical in float32), LoRA files are merged without peft (W += B·A·α/r);
- GPT-SoVITS's DiT differs from F5-TTS's (``mizuoto.nn.transformer.dit``): the text input is features, not ids, with
  positions up to 4096 and no padding mask; attention gets the boolean mask; the convolutional position embedding
  masks only around both convolutions; rotary embedding covers the first 64 channels of the projection (one head);
  a step-size embedding (``d_embed``, v4) and a static conditioning cache (v5). The identical parts are shared.

Tensor layouts follow upstream operation by operation (transposes, views): CPU kernels can differ by layout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.gptsovits.sovits import IGNORABLE, ResidualVectorQuantizer, TextEncoder, spectrogram
from mizuoto.nn.flow.coupling import WN
from mizuoto.nn.speaker.melstyle import MelStyleEncoder
from mizuoto.nn.transformer.dit import (
    AdaLayerNorm,
    AdaLayerNormFinal,
    ConvNeXtV2Block,
    FeedForward,
    Rope,
    RotaryEmbedding,
    TimestepEmbedding,
    apply_rotary,
    precompute_freqs_cis,
    rope_tables,
)
from mizuoto.nn.transformer.vits import sequence_mask
from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import MaskInfo, Site, attention

SITE = Site("gptsovits.dit")

if TYPE_CHECKING:
    from mizuoto.runtime.compile import Compiled

V5_VERSIONS = frozenset({"v5dev", "v5turbo"})
VERSIONS = frozenset({"v4"}) | V5_VERSIONS
V5_TEMPERATURE = 0.875
V5_ROLLING_TAIL_FRAMES = 32
V5_MAX_TARGET_CHUNK_FRAMES = 640
V5_REFERENCE_FRAMES = 500
V5_TOTAL_CHUNK_FRAMES = 1000

# TTS.init_vocoder's vocoder_configs for v4 and v5
T_REF = 500
T_CHUNK = 1000
OVERLAPPED_LEN = 12
UPSAMPLE_RATE = 480
MEL_RATE = 32000

SPEC_MIN = -12
SPEC_MAX = 2

StepCallback = Callable[[int, int], None]


def default_sample_steps(version: str) -> int:
    return 4 if version == "v5turbo" else 32


def default_cfg_rate(version: str) -> float:
    return 1.30 if version == "v5dev" else 0.0


# -- mel ----------------------------------------------------------------------------------------


def norm_spec(x: torch.Tensor) -> torch.Tensor:
    return (x - SPEC_MIN) / (SPEC_MAX - SPEC_MIN) * 2 - 1


def denorm_spec(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2 * (SPEC_MAX - SPEC_MIN) + SPEC_MIN


def mel_basis(
    sampling_rate: int, n_fft: int, num_mels: int, fmin: float, fmax: float | None, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> torch.Tensor:
    from mizuoto.nn.dsp.mel import librosa_mel

    return torch.from_numpy(librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).to(dtype=dtype, device=device)


def mel_spectrogram(y: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int, win_size: int, fmin: float, fmax: float | None) -> torch.Tensor:
    """``module.mel_processing.mel_spectrogram_torch`` (``center=False``): (batch, samples) → (batch, mels, frames) log-mel."""
    basis = mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, y.dtype, y.device)
    spec = torch.matmul(basis, spectrogram(y, n_fft, hop_size, win_size))
    return torch.log(torch.clamp(spec, min=1e-5) * 1)


def mel_v4(audio: torch.Tensor) -> torch.Tensor:
    """``TTS.mel_fn_v4``: 32 kHz audio (batch, samples) → (batch, 100, frames), 100 frames per second."""
    return mel_spectrogram(audio, n_fft=1280, num_mels=100, sampling_rate=MEL_RATE, hop_size=320, win_size=1280, fmin=0, fmax=None)


def reference_mel(audio_32k: torch.Tensor) -> torch.Tensor:
    """The reference's normalised mel for the CFM prompt: ``norm_spec(mel_fn_v4(ref_audio))``.

    ``audio_32k`` is the reference as ``torchaudio.load`` reads it (float32, (channels, samples)), averaged to mono
    when it has two channels and resampled to 32 kHz with ``torchaudio.transforms.Resample`` (``audio.resample``).
    """
    if audio_32k.shape[0] == 2:
        audio_32k = audio_32k.mean(0).unsqueeze(0)
    return norm_spec(mel_v4(audio_32k.float()))


# -- the WaveNet encoder after the bridge (module/models.py: Encoder) -----------------------------------


class WNEncoder(nn.Module):
    """``module.models.Encoder``: 1×1 in, ``WN``, 1×1 out, masked by length."""

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0, ops: Any = default_ops.default
    ) -> None:
        super().__init__()
        self.pre = ops.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, ops=ops)
        self.proj = ops.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        return self.proj(x) * x_mask


# -- GPT-SoVITS's DiT ---------------------------------------------------------------------------


class GsvTextEmbedding(nn.Module):
    """``f5_tts.model.backbones.dit.TextEmbedding`` (GPT-SoVITS): features + sinusoidal positions → ConvNeXtV2 blocks."""

    def __init__(self, text_dim: int, conv_layers: int = 0, conv_mult: int = 2, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.extra_modeling = conv_layers > 0
        if self.extra_modeling:
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult, ops=ops) for _ in range(conv_layers)])

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        batch = text.shape[0]
        if drop_text:
            text = torch.zeros_like(text)
        if self.extra_modeling:
            freqs_cis = self.get_buffer("freqs_cis")
            start = torch.zeros((batch,), dtype=torch.long, device=freqs_cis.device)
            scale = torch.ones_like(start, dtype=torch.float32)
            pos = start.unsqueeze(1) + (torch.arange(seq_len, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
            pos = torch.where(pos < self.precompute_max_pos, pos, self.precompute_max_pos - 1)
            text = text + freqs_cis[pos]
            text = self.text_blocks(text)
        return text


class GsvConvPositionEmbedding(nn.Module):
    """``f5_tts.model.modules.ConvPositionEmbedding`` (GPT-SoVITS): the mask applied before and after both convolutions."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16, ops: Any = default_ops.default) -> None:
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        m = mask[..., None] if mask is not None else None
        if m is not None:
            x = x.masked_fill(~m, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        if m is not None:
            out = out.masked_fill(~m, 0.0)
        return out


class GsvInputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.proj = ops.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = GsvConvPositionEmbedding(dim=out_dim, ops=ops)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_embed: torch.Tensor, drop_audio_cond: bool = False) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        return self.conv_pos_embed(x) + x


class GsvAttention(nn.Module):
    """``f5_tts.model.modules.Attention`` + ``AttnProcessor`` (GPT-SoVITS): rotary on the projection, SDPA with the mask."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_q = ops.Linear(dim, inner_dim)
        self.to_k = ops.Linear(dim, inner_dim)
        self.to_v = ops.Linear(dim, inner_dim)
        self.to_out = nn.ModuleList([ops.Linear(inner_dim, dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, rope: torch.Tensor | Rope | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        if rope is not None:
            # x-transformers' apply_rotary_pos_emb on (b, n, heads × dim_head): only the first ``rot_dim`` channels turn
            query = apply_rotary(query, rope, 1.0)
            key = apply_rotary(key, rope, 1.0)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        x = attention(query, key, value, site=SITE, mask=attn_mask, mask_info=MaskInfo(keys=mask, padding="right") if mask is not None else None)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class GsvDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 4, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, ops)
        self.attn = GsvAttention(dim, heads, dim_head, ops)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, ff_mult, ops)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None, rope: torch.Tensor | Rope | None = None) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        return x + gate_mlp.unsqueeze(1) * ff_output


class GsvDiT(nn.Module):
    """``f5_tts.model.backbones.dit.DiT`` as GPT-SoVITS has it (v4: with the step embedding; v5: without)."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4,
        mel_dim: int = 100,
        text_dim: int | None = None,
        conv_layers: int = 0,
        use_step_embedding: bool = True,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(dim, ops=ops)
        self.d_embed = TimestepEmbedding(dim, ops=ops) if use_step_embedding else None
        text_dim = mel_dim if text_dim is None else text_dim
        self.text_embed = GsvTextEmbedding(text_dim, conv_layers=conv_layers, ops=ops)
        self.input_embed = GsvInputEmbedding(mel_dim, text_dim, dim, ops)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList([GsvDiTBlock(dim, heads, dim_head, ff_mult, ops) for _ in range(depth)])
        self.norm_out = AdaLayerNormFinal(dim, ops)
        self.proj_out = ops.Linear(dim, mel_dim)

    def prepare_static_cache(self, cond0: torch.Tensor, x_lens: torch.Tensor, text0: torch.Tensor) -> dict[str, Any]:
        text = text0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        seq_len = cond.shape[1]
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        projection = self.input_embed.proj
        mel_dim = cond.shape[-1]
        static = F.linear(torch.cat((cond, text_embed), dim=-1), projection.weight[:, mel_dim:], projection.bias)
        negative_static = F.linear(torch.cat((torch.zeros_like(cond), text_embed), dim=-1), projection.weight[:, mel_dim:], projection.bias)
        return {
            "condition": static,
            "negative_condition": negative_static,
            "mask": sequence_mask(x_lens, max_length=seq_len).to(cond.device),
            "rope": rope_tables(self.rotary_embed.forward_from_seq_len(seq_len)),
        }

    def forward(
        self,
        x0: torch.Tensor,
        cond0: torch.Tensor,
        x_lens: torch.Tensor,
        time: torch.Tensor,
        dt_base_bootstrap: torch.Tensor | None = None,
        text0: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        text_cache: torch.Tensor | None = None,
        dt_cache: torch.Tensor | None = None,
        static_cache: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Inference (``infer=True``): the velocity, the text embedding (for the cache) and the step embedding."""
        assert text0 is not None
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = static_cache["mask"] if static_cache is not None else sequence_mask(x_lens, max_length=x.size(1)).to(x.device)
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        if self.d_embed is None:
            dt = None
        elif dt_cache is not None:
            dt = dt_cache
        else:
            assert dt_base_bootstrap is not None
            dt = self.d_embed(dt_base_bootstrap)
        if dt is not None:
            t += dt
        text_embed: torch.Tensor | None = None
        if static_cache is not None:
            static = static_cache["negative_condition" if drop_audio_cond else "condition"]
            x = F.linear(x, self.input_embed.proj.weight[:, : x.shape[-1]]) + static
            x = self.input_embed.conv_pos_embed(x, mask=mask) + x
        elif text_cache is not None:
            text_embed = text_cache
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        rope = static_cache["rope"] if static_cache is not None else rope_tables(self.rotary_embed.forward_from_seq_len(seq_len))
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output, text_embed, dt


# -- flow matching ------------------------------------------------------------------------------


class CFM(nn.Module):
    """``module.models.CFM`` (v4) and ``module.models_v5.CFMV5`` (``v5=True``): Euler steps from noise to mel.

    v5: no step embedding, noise × 0.875, the time grid ``j / n`` and a static conditioning cache.
    """

    def __init__(self, in_channels: int, dit: GsvDiT, v5: bool = False) -> None:
        super().__init__()
        self.estimator = dit
        self.in_channels = in_channels
        self.use_step_embedding = not v5
        self.cfg_drop_text = not v5
        self.use_static_cache = v5
        self.noise_temperature: float | None = V5_TEMPERATURE if v5 else None
        self.compiled_estimator: Compiled | None = None
        """``compute.compile``: the estimator compiled (``mizuoto.runtime.compile``), called instead of it."""

    @torch.inference_mode()
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0,
        generator: torch.Generator | None = None,
        step: StepCallback | None = None,
    ) -> torch.Tensor:
        """``mu`` (batch, frames, 512) features, ``prompt`` (batch, 100, prompt frames) normalised mel → (batch, 100, frames).

        The noise is one ``torch.randn([batch, 100, frames])`` from ``generator``; ``step(i, n)`` runs before each step.
        """
        n_timesteps, inference_cfg_rate = int(n_timesteps), float(inference_cfg_rate)
        b, t_len = mu.size(0), mu.size(1)
        if self.noise_temperature is not None:
            temperature = self.noise_temperature
        x = torch.randn([b, self.in_channels, t_len], device=mu.device, dtype=mu.dtype, generator=generator) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(2, 1)
        t = 0.0
        d = 1 / n_timesteps
        text_cache: torch.Tensor | None = None
        text_cfg_cache: torch.Tensor | None = None
        dt_cache: torch.Tensor | None = None
        static_cache = self.estimator.prepare_static_cache(prompt_x, x_lens, mu) if self.use_static_cache else None
        d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d if self.use_step_embedding else None
        estimator = self.compiled_estimator or self.estimator
        for j in range(n_timesteps):
            if step is not None:
                step(j, n_timesteps)
            t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * (t if self.use_step_embedding else j * d)
            v_pred, text_emb, dt = estimator(
                x, prompt_x, x_lens, t_tensor, dt_base_bootstrap=d_tensor, text0=mu, text_cache=text_cache, dt_cache=dt_cache, static_cache=static_cache
            )
            v_pred = v_pred.transpose(2, 1)
            text_cache = text_emb
            dt_cache = dt
            if inference_cfg_rate > 1e-5:
                neg, text_cfg_emb, _ = estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t_tensor,
                    dt_base_bootstrap=d_tensor,
                    text0=mu,
                    drop_audio_cond=True,
                    drop_text=self.cfg_drop_text,
                    text_cache=text_cfg_cache if self.cfg_drop_text else text_cache,
                    dt_cache=dt_cache,
                    static_cache=static_cache,
                )
                neg = neg.transpose(2, 1)
                text_cfg_cache = text_cfg_emb
                v_pred = v_pred + (v_pred - neg) * inference_cfg_rate
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0
        return x


# -- the synthesiser ----------------------------------------------------------------------------


class SynthesizerTrnV3(nn.Module):
    """``module.models.SynthesizerTrnV3`` for inference (v4, v5dev, v5turbo): semantic codes + phones → mel features."""

    def __init__(
        self,
        version: str = "v4",
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        gin_channels: int = 512,
        semantic_frame_rate: str = "25hz",
        n_symbols: int = 732,
        bridge_channels: int = 512,
        wn_layers: int = 8,
        dit: Mapping[str, Any] | None = None,
        ops: Any = default_ops.default,
    ) -> None:
        """Upstream's sizes are the defaults; ``bridge_channels``, ``wn_layers`` and ``dit`` (DiT sizes) exist for tests."""
        super().__init__()
        if version not in VERSIONS:
            raise ValueError(f"SynthesizerTrnV3 runs {sorted(VERSIONS)}, not {version!r}")
        self.version = version
        self.semantic_frame_rate = semantic_frame_rate
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, n_symbols=n_symbols, mrte_channels=gin_channels, ops=ops)
        self.ref_enc = MelStyleEncoder(704, style_vector_dim=gin_channels, ops=ops)
        ssl_dim = 768
        if semantic_frame_rate == "25hz":
            self.ssl_proj = ops.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = ops.Conv1d(ssl_dim, ssl_dim, 1, stride=1)
        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        inter_channels2 = bridge_channels
        self.bridge = nn.Sequential(ops.Conv1d(inter_channels, inter_channels2, 1, stride=1), nn.LeakyReLU())
        self.wns1 = WNEncoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, wn_layers, gin_channels=gin_channels, ops=ops)
        self.linear_mel = ops.Conv1d(inter_channels2, 100, 1, stride=1)  # training only; kept for the weights
        v5 = version in V5_VERSIONS
        arch = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": inter_channels2, "conv_layers": 4, **(dit or {})}
        self.cfm = CFM(100, GsvDiT(**arch, use_step_embedding=not v5, ops=ops), v5=v5)

    @torch.no_grad()
    def reference_embedding(self, refer: torch.Tensor) -> torch.Tensor:
        """``ge``: the speaker vector from the reference's linear spectrogram (batch, 1025, frames)."""
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        return self.ref_enc(refer[:, :704] * refer_mask, refer_mask)

    @torch.no_grad()
    def decode_encp(self, codes: torch.Tensor, text: torch.Tensor, refer: torch.Tensor, ge: torch.Tensor | None = None, speed: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Codes (1, 1, n) + phone ids (1, m) [+ ``ge``] → (features (1, 512, frames at 100 Hz), ``ge``)."""
        if ge is None:
            ge = self.reference_embedding(refer)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        if speed == 1:
            sizee = int(codes.size(2) * 4)
        else:
            sizee = int(codes.size(2) * 4 / speed) + 1
        y_lengths1 = torch.LongTensor([sizee]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        x = self.enc_p.features(quantized, y_lengths, text, text_lengths, ge, speed)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=2, mode="nearest")
        fea = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """HuBERT features (batch, 768, frames at 50 Hz) → codes (batch, 1, frames at 25 Hz)."""
        ssl = self.ssl_proj(x)
        return self.quantizer.encode(ssl).transpose(0, 1)


# -- loading ------------------------------------------------------------------------------------

LORA_TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
PEFT_PREFIX = "cfm.base_model.model."


def merge_lora(state: Mapping[str, torch.Tensor], lora: Mapping[str, torch.Tensor], rank: int, alpha: float | None = None) -> dict[str, torch.Tensor]:
    """A LoRA checkpoint (peft's names under ``cfm.base_model.model.``) laid over ``state`` and merged, as upstream's
    ``get_peft_model`` + ``load_state_dict`` + ``merge_and_unload``: W += (B @ A) × α/r, in float32 (α = r upstream).

    The checkpoint's other entries (the style encoder, ``wns1``…, trained with the LoRA) replace the base's.
    ``state`` is float32 (the model's parameters, as upstream loads the base into its float32 model).
    """
    out = dict(state)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora.items():
        name = "cfm." + key[len(PEFT_PREFIX) :] if key.startswith(PEFT_PREFIX) else key
        for part in ("A", "B"):
            marker = f".lora_{part}.default.weight"
            if name.endswith(marker):
                pairs.setdefault(name[: -len(marker)], {})[part] = value
                break
        else:
            name = name.replace(".base_layer.", ".")
            out[name] = value.float() if value.is_floating_point() else value
    scaling = (alpha if alpha is not None else rank) / rank
    for base, ab in pairs.items():
        if set(ab) != {"A", "B"} or not base.endswith(LORA_TARGETS):
            raise ValueError(f"Incomplete or unexpected LoRA pair for {base}: {sorted(ab)}")
        weight = out[f"{base}.weight"]
        a, b = ab["A"].float(), ab["B"].float()
        out[f"{base}.weight"] = weight + (b @ a) * scaling
    return out


def checkpoint_weights(path: str | Path) -> dict[str, Any]:
    """A SoVITS checkpoint (``.pth``, possibly with GPT-SoVITS's two-byte version tag): ``weight``, ``config``, ``lora_rank``…

    Read by ``checkpoints.load_checkpoint``: ``weights_only``, or for fine-tunes whose config is upstream's pickled
    ``utils.HParams``, a restricted unpickler; the config comes back as plain dicts.
    """
    from mizuoto.engines.gptsovits.checkpoints import _plain, load_checkpoint

    obj = load_checkpoint(path)
    if not isinstance(obj, Mapping) or "weight" not in obj:
        raise ValueError(f"{path} is not a GPT-SoVITS SoVITS checkpoint")
    return {k: (_plain(v) if k == "config" else v) for k, v in obj.items()}


def model_kwargs(config: Mapping[str, Any] | None) -> dict[str, Any]:
    model = dict((config or {}).get("model", {}))
    keys = ("inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers", "kernel_size", "gin_channels")
    return {k: model[k] for k in keys if k in model}


def load_synthesizer(
    path: str | Path,
    version: str,
    *,
    lora_base: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    ops: Any = default_ops.default,
    run_on: str | torch.device | None = None,
) -> SynthesizerTrnV3:
    """``TTS.init_vits_weights`` for v4/v5: the model from ``path`` in eval mode.

    A LoRA checkpoint (``lora_rank`` in it) needs ``lora_base``, the pretrained ``s2Gv4.pth``/``s2G<v5>.pth`` it was
    trained from. Weights are copied into float32 (upstream's model is float32; the files are float16), weight norm
    folded and LoRA merged in float32, then the model is cast to ``dtype``. With ``run_on``, the device it will run
    on, the weight norm is computed there in ``dtype``: upstream keeps it and computes it at every call on that device.
    """
    from mizuoto.runtime.loading import fold_weight_norm, load_into

    ckpt = checkpoint_weights(path)
    weights: Mapping[str, torch.Tensor] = ckpt["weight"]
    rank = ckpt.get("lora_rank")
    config = ckpt.get("config")
    if rank:
        if lora_base is None:
            raise ValueError(f"{path} is a LoRA of a SoVITS {version} model: its base checkpoint is needed")
        base = checkpoint_weights(lora_base)
        config = base.get("config") or config
        state = {k: v.float() if v.is_floating_point() else v for k, v in base["weight"].items()}
        state = merge_lora(state, weights, int(rank))
    else:
        state = {k: v.float() if v.is_floating_point() else v for k, v in weights.items()}
    state = fold_weight_norm(state, dtype=dtype if run_on is not None else None, device=run_on)
    n_symbols = int(state["enc_p.text_embedding.weight"].shape[0]) if "enc_p.text_embedding.weight" in state else 732
    model = SynthesizerTrnV3(version, n_symbols=n_symbols, ops=ops, **model_kwargs(config))
    load_into(model, state, ignore=IGNORABLE, what=f"SoVITS {version}")
    return model.to(device=device, dtype=dtype).eval()


# -- synthesis (TTS_infer_pack/TTS.py) ------------------------------------------------------------


@torch.inference_mode()
def synthesize_v5_mel(
    model: SynthesizerTrnV3,
    reference_features: torch.Tensor,
    target_features: torch.Tensor,
    reference_mel: torch.Tensor,
    sample_steps: int | None = None,
    cfg_rate: float | None = None,
    generator: torch.Generator | None = None,
    step: StepCallback | None = None,
) -> torch.Tensor:
    """``module.models_v5.synthesize_v5_mel``: chunks of at most 640 frames, each prompted by the reference's mel with
    its last 32 frames replaced by the previous chunk's (a rolling tail). Returns the normalised mel (1, 100, frames)."""
    steps = default_sample_steps(model.version) if sample_steps is None else int(sample_steps)
    cfg = (1.30 if model.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    reference_mel = reference_mel[..., :reference_frames][..., -V5_REFERENCE_FRAMES:]
    reference_features = reference_features[..., :reference_frames][..., -V5_REFERENCE_FRAMES:]
    reference_frames = reference_mel.shape[-1]
    chunk_frames = min(V5_TOTAL_CHUNK_FRAMES - reference_frames, V5_MAX_TARGET_CHUNK_FRAMES)
    original_mel = reference_mel
    original_features = reference_features
    rolling_mel = reference_mel
    rolling_features = reference_features
    results = []
    for index, start in enumerate(range(0, target_features.shape[-1], chunk_frames)):
        target = target_features[..., start : start + chunk_frames]
        if index == 0:
            prompt = original_mel
            features = original_features
        else:
            tail = min(V5_ROLLING_TAIL_FRAMES, reference_frames, rolling_mel.shape[-1], rolling_features.shape[-1])
            prefix = reference_frames - tail
            prompt = torch.cat((original_mel[..., :prefix], rolling_mel[..., -tail:]), dim=-1)
            features = torch.cat((original_features[..., :prefix], rolling_features[..., -tail:]), dim=-1)
        mu = torch.cat((features, target), dim=-1).transpose(2, 1)
        lengths = torch.full((mu.shape[0],), mu.shape[1], dtype=torch.long, device=mu.device)
        generated = model.cfm.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg, generator=generator, step=step)[..., prompt.shape[-1] :]
        results.append(generated)
        rolling_mel = generated[..., -reference_frames:].to(reference_mel.dtype)
        rolling_features = target[..., -reference_frames:]
    return torch.cat(results, dim=-1)


class Prompt:
    """What both synthesis routines take from the reference (upstream's ``prompt_cache``).

    - ``codes``: the reference's semantic codes (n,) (``prompt_semantic``, from ``extract_latent``);
    - ``phones``: the reference transcript's phone ids (m,);
    - ``refer_spec``: its linear spectrogram (1, 1025, frames) at 32 kHz (``spectrogram_torch``, filter 2048, hop 640);
    - ``mel``: ``reference_mel(audio at 32 kHz)``, (1, 100, frames).
    """

    def __init__(self, codes: torch.Tensor, phones: torch.Tensor | Sequence[int], refer_spec: torch.Tensor, mel: torch.Tensor) -> None:
        self.codes = codes
        self.phones = phones if isinstance(phones, torch.Tensor) else torch.LongTensor(list(phones))
        self.refer_spec = refer_spec
        self.mel = mel


def _prompt_features(model: SynthesizerTrnV3, prompt: Prompt, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_codes = prompt.codes.unsqueeze(0).unsqueeze(0).to(device)
    prompt_phones = prompt.phones.unsqueeze(0).to(device)
    refer = prompt.refer_spec.to(dtype=dtype, device=device)
    fea_ref, ge = model.decode_encp(prompt_codes, prompt_phones, refer)
    return fea_ref, ge, refer


def vocoder_synthesis(
    model: SynthesizerTrnV3,
    vocoder: nn.Module,
    prompt: Prompt,
    semantic_tokens: torch.Tensor,
    phones: torch.Tensor,
    speed: float = 1.0,
    sample_steps: int | None = None,
    cfg_rate: float | None = None,
    generator: torch.Generator | None = None,
    step: StepCallback | None = None,
) -> torch.Tensor:
    """``TTS.using_vocoder_synthesis``: one fragment's codes (1, 1, n) and phones (1, m) → 48 kHz audio (samples,).

    v4: the reference's last ≤ 500 frames prompt chunks of ``1000 − prompt`` frames, each next chunk prompted by the
    previous one's tail. v5: ``synthesize_v5_mel``. ``cfg_rate`` reaches only v5 (v4 runs without guidance upstream).
    """
    sample_steps = default_sample_steps(model.version) if sample_steps is None else int(sample_steps)
    cfg_rate = default_cfg_rate(model.version) if cfg_rate is None else float(cfg_rate)
    precision = next(model.parameters()).dtype
    device = next(model.parameters()).device
    fea_ref, ge, refer = _prompt_features(model, prompt, device, precision)
    mel2 = prompt.mel.to(device)
    if model.version in V5_VERSIONS:
        fea_todo, _ = model.decode_encp(semantic_tokens, phones, refer, ge, speed)
        mel = synthesize_v5_mel(model, fea_ref, fea_todo, mel2.to(precision), sample_steps, cfg_rate, generator=generator, step=step)
        with torch.inference_mode():
            return vocoder(denorm_spec(mel).to(precision))[0, 0]
    t_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :t_min]
    fea_ref = fea_ref[:, :, :t_min]
    if t_min > T_REF:
        mel2 = mel2[:, :, -T_REF:]
        fea_ref = fea_ref[:, :, -T_REF:]
        t_min = T_REF
    chunk_len = T_CHUNK - t_min
    mel2 = mel2.to(precision)
    fea_todo, ge = model.decode_encp(semantic_tokens, phones, refer, ge, speed)
    results = []
    idx = 0
    while True:
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break
        idx += chunk_len
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        cfm_res = model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0, generator=generator, step=step)
        cfm_res = cfm_res[:, :, mel2.shape[2] :]
        mel2 = cfm_res[:, :, -t_min:]
        fea_ref = fea_todo_chunk[:, :, -t_min:]
        results.append(cfm_res)
    cfm_res = denorm_spec(torch.cat(results, 2))
    with torch.inference_mode():
        return vocoder(cfm_res)[0][0]


def vocoder_synthesis_batched(
    model: SynthesizerTrnV3,
    vocoder: nn.Module,
    prompt: Prompt,
    idx_list: Sequence[int],
    semantic_tokens_list: Sequence[torch.Tensor],
    batch_phones: Sequence[torch.Tensor],
    speed: float = 1.0,
    sample_steps: int | None = None,
    cfg_rate: float | None = None,
    generator: torch.Generator | None = None,
    step: StepCallback | None = None,
) -> list[torch.Tensor]:
    """``TTS.using_vocoder_synthesis_batched_infer`` (``parallel_infer``): every fragment's features joined, cut into
    overlapping chunks of ``1000 − prompt`` frames, generated as one batch, vocoded at once, rejoined with
    ``sola_algorithm`` and split back per fragment. v5 runs each fragment through ``vocoder_synthesis``.

    ``semantic_tokens_list[i][-idx_list[i]:]`` are fragment *i*'s codes (as upstream slices them), ``batch_phones[i]``
    its phone ids (m,). Returns 48 kHz audio per fragment.
    """
    sample_steps = default_sample_steps(model.version) if sample_steps is None else int(sample_steps)
    cfg_rate = default_cfg_rate(model.version) if cfg_rate is None else float(cfg_rate)
    precision = next(model.parameters()).dtype
    device = next(model.parameters()).device
    if model.version in V5_VERSIONS:
        return [
            vocoder_synthesis(
                model,
                vocoder,
                prompt,
                semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0),
                batch_phones[i].unsqueeze(0).to(device),
                speed=speed,
                sample_steps=sample_steps,
                cfg_rate=cfg_rate,
                generator=generator,
                step=step,
            )
            for i, idx in enumerate(idx_list)
        ]
    fea_ref, ge, refer = _prompt_features(model, prompt, device, precision)
    mel2 = prompt.mel.to(device)
    t_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :t_min]
    fea_ref = fea_ref[:, :, :t_min]
    if t_min > T_REF:
        mel2 = mel2[:, :, -T_REF:]
        fea_ref = fea_ref[:, :, -T_REF:]
        t_min = T_REF
    chunk_len = T_CHUNK - t_min
    mel2 = mel2.to(precision)
    overlapped_len = OVERLAPPED_LEN
    feat_chunks = []
    feat_lens = []
    feat_list = []
    for i, idx in enumerate(idx_list):
        phones = batch_phones[i].unsqueeze(0).to(device)
        semantic_tokens = semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0)
        feat, _ = model.decode_encp(semantic_tokens, phones, refer, ge, speed)
        feat_list.append(feat)
        feat_lens.append(feat.shape[2])
    feats = torch.cat(feat_list, 2)
    feats_padded = F.pad(feats, (overlapped_len, 0), "constant", 0)
    pos = 0
    padding_len = 0
    while True:
        if pos == 0:
            chunk = feats_padded[:, :, pos : pos + chunk_len]
        else:
            pos = pos - overlapped_len
            chunk = feats_padded[:, :, pos : pos + chunk_len]
        pos += chunk_len
        if chunk.shape[-1] == 0:
            break
        padding_len = chunk_len - chunk.shape[2]
        if padding_len != 0:
            chunk = F.pad(chunk, (0, padding_len), "constant", 0)
        feat_chunks.append(chunk)
    feat_batch = torch.cat(feat_chunks, 0)
    bs = feat_batch.shape[0]
    fea_ref = fea_ref.repeat(bs, 1, 1)
    fea = torch.cat([fea_ref, feat_batch], 2).transpose(2, 1)
    pred_spec = model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0, generator=generator, step=step)
    pred_spec = pred_spec[:, :, -chunk_len:]
    dd = pred_spec.shape[1]
    pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(dd, -1).unsqueeze(0)
    pred_spec = denorm_spec(pred_spec)
    with torch.no_grad():
        audio = vocoder(pred_spec)[0][0]
    audio_fragments = []
    pos = 0
    while pos < audio.shape[-1]:
        audio_fragments.append(audio[pos : pos + chunk_len * UPSAMPLE_RATE])
        pos += chunk_len * UPSAMPLE_RATE
    audio = sola_algorithm(audio_fragments, overlapped_len * UPSAMPLE_RATE)
    if padding_len > 0:
        audio = audio[overlapped_len * UPSAMPLE_RATE : -padding_len * UPSAMPLE_RATE]
    else:
        audio = audio[overlapped_len * UPSAMPLE_RATE :]
    out = []
    for feat_len in feat_lens:
        out.append(audio[: feat_len * UPSAMPLE_RATE])
        audio = audio[feat_len * UPSAMPLE_RATE :]
    return out


def sola_algorithm(audio_fragments: list[torch.Tensor], overlap_len: int, search_len: int = 320) -> torch.Tensor:
    """``TTS.sola_algorithm``: join fragments whose ends overlap by ``overlap_len`` samples, each next one shifted to the
    best-correlated offset within ``search_len`` and cross-faded with a Hann window. ``audio_fragments`` is modified."""
    dtype = audio_fragments[0].dtype
    for i in range(len(audio_fragments) - 1):
        f1 = audio_fragments[i].float()
        f2 = audio_fragments[i + 1].float()
        w1 = f1[-overlap_len:]
        w2 = f2[: overlap_len + search_len]
        corr_norm = F.conv1d(w2.view(1, 1, -1), w1.view(1, 1, -1)).view(-1)
        corr_den = F.conv1d(w2.view(1, 1, -1) ** 2, torch.ones_like(w1).view(1, 1, -1)).view(-1) + 1e-8
        idx = (corr_norm / corr_den.sqrt()).argmax()
        audio_fragments[i] = f1[:-overlap_len]
        f2_ = f2[idx:]
        window = torch.hann_window(overlap_len * 2, device=f1.device, dtype=f1.dtype)
        f2_[:overlap_len] = window[:overlap_len] * f2_[:overlap_len] + window[overlap_len:] * f1[-overlap_len:]
        audio_fragments[i + 1] = f2_
    return torch.cat(audio_fragments, 0).to(dtype)
