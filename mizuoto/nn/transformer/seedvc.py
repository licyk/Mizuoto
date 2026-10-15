"""Seed-VC's diffusion transformer and its Euler CFM sampler, as IndexTTS-2's ``s2mel`` has them
(``s2mel/modules/{diffusion_transformer,flow_matching,wavenet}.py`` and gpt-fast's ``Transformer``; Seed-VC is
GPL-3.0).

- The transformer: RMSNorm made adaptive by the time embedding (``AdaptiveLayerNorm``), fused ``wqkv``, RoPE over
  adjacent pairs with a float32 table, SwiGLU, U-ViT skips (the first half of the layers feed the second half
  in reverse), and a full boolean padding mask.
- The head: a long skip with the noisy input, then a WaveNet conditioned on a second time embedding, a final
  adaptive LayerNorm and a 1×1 convolution to mels.
- ``CFM.inference``: ``randn`` noise from the request's generator, the prompt's mels kept, ``n_timesteps``
  Euler steps with classifier-free guidance as a batch of two (conditional, then everything zeroed).

Weight norm is folded at load (``fold_weight_norm``): identical in float32, where upstream runs this part.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import MaskInfo, Site, attention

SITE = Site("seedvc")

if TYPE_CHECKING:
    from mizuoto.runtime.compile import Compiled


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


@dataclass
class DiTConfig:
    in_channels: int = 80
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 13
    content_dim: int = 512
    content_codebook_size: int = 1024
    style_dim: int = 192
    style_condition: bool = True
    uvit_skip_connection: bool = True
    long_skip_connection: bool = True
    wavenet_hidden_dim: int = 512
    wavenet_num_layers: int = 8
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    block_size: int = 16384
    rope_base: float = 10000
    norm_eps: float = 1e-5

    @classmethod
    def from_s2mel(cls, s2mel: dict[str, Any]) -> DiTConfig:
        """From the ``s2mel`` section of IndexTTS-2's ``config.yaml``."""
        d, w = s2mel["DiT"], s2mel["wavenet"]
        if d.get("final_layer_type", "wavenet") != "wavenet" or d.get("time_as_token") or d.get("style_as_token") or d.get("is_causal"):
            raise ValueError("Only the configuration IndexTTS-2 ships is supported (WaveNet head, no time/style tokens, not causal)")
        return cls(
            in_channels=d["in_channels"],
            hidden_dim=d["hidden_dim"],
            num_heads=d["num_heads"],
            depth=d["depth"],
            content_dim=d["content_dim"],
            content_codebook_size=d["content_codebook_size"],
            style_dim=s2mel["style_encoder"]["dim"],
            style_condition=bool(d["style_condition"]),
            uvit_skip_connection=bool(d.get("uvit_skip_connection", False)),
            long_skip_connection=bool(d["long_skip_connection"]),
            wavenet_hidden_dim=w["hidden_dim"],
            wavenet_num_layers=w["num_layers"],
            wavenet_kernel_size=w["kernel_size"],
            wavenet_dilation_rate=w["dilation_rate"],
        )


# -- the transformer --------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        return (h * torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float, ops: Any) -> None:
        super().__init__()
        self.project_layer = ops.Linear(dim, 2 * dim)
        self.norm = RMSNorm(dim, eps)
        self.d_model = dim

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None) -> torch.Tensor:
        if embedding is None:
            return self.norm(x)
        weight, bias = torch.split(self.project_layer(embedding), split_size_or_sections=self.d_model, dim=-1)
        return weight * self.norm(x) + bias


def rope_table(seq_len: int, n_elem: int, base: float, dtype: torch.dtype) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    cis = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))
    return torch.stack([cis.real, cis.imag], dim=-1).to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = freqs_cis.view(1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack([xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1], xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int, ops: Any) -> None:
        super().__init__()
        self.wqkv = ops.Linear(dim, 3 * n_head * head_dim, bias=False)
        self.wo = ops.Linear(n_head * head_dim, dim, bias=False)
        self.n_head, self.head_dim = n_head, head_dim

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([size, size, size], dim=-1)
        q = apply_rotary_emb(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        k = apply_rotary_emb(k.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        k, v = k.repeat_interleave(1, dim=1), v.repeat_interleave(1, dim=1)
        y = attention(q, k, v, site=SITE, mask=mask, mask_info=MaskInfo(keys=mask[:, 0, 0], padding="right") if mask is not None else None)
        return self.wo(y.transpose(1, 2).contiguous().view(bsz, seqlen, size))


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate: int, ops: Any) -> None:
        super().__init__()
        self.w1 = ops.Linear(dim, intermediate, bias=False)
        self.w3 = ops.Linear(dim, intermediate, bias=False)
        self.w2 = ops.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, c: DiTConfig, ops: Any) -> None:
        super().__init__()
        dim = c.hidden_dim
        self.attention = Attention(dim, c.num_heads, dim // c.num_heads, ops)
        self.feed_forward = FeedForward(dim, find_multiple(int(2 * 4 * dim / 3), 256), ops)
        self.ffn_norm = AdaptiveLayerNorm(dim, c.norm_eps, ops)
        self.attention_norm = AdaptiveLayerNorm(dim, c.norm_eps, ops)
        self.uvit = c.uvit_skip_connection
        if self.uvit:
            self.skip_in_linear = ops.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        if self.uvit and skip is not None:
            x = self.skip_in_linear(torch.cat([x, skip], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h, c))


class Transformer(nn.Module):
    def __init__(self, c: DiTConfig, ops: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TransformerBlock(c, ops) for _ in range(c.depth))
        self.norm = AdaptiveLayerNorm(c.hidden_dim, c.norm_eps, ops)
        self.uvit = c.uvit_skip_connection
        self.emit = [i for i in range(c.depth) if i < c.depth // 2] if self.uvit else []
        self.receive = [i for i in range(c.depth) if i > c.depth // 2] if self.uvit else []
        self.c = c
        self._freqs: torch.Tensor | None = None

    def freqs_cis(self, n: int, like: torch.Tensor) -> torch.Tensor:
        # upstream builds the table for block_size positions in the model's dtype at setup (float32)
        if self._freqs is None or self._freqs.shape[0] < n or self._freqs.device != like.device:
            self._freqs = rope_table(max(n, 4096), self.c.hidden_dim // self.c.num_heads, self.c.rope_base, self.norm.project_layer.weight.dtype).to(like.device)
        return self._freqs[:n]

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        freqs_cis = self.freqs_cis(x.shape[1], x)
        skips: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            skip = skips.pop(-1) if i in self.receive else None
            x = layer(x, c, freqs_cis, mask, skip)
            if i in self.emit:
                skips.append(x)
        return self.norm(x, c)


# -- embeddings and the WaveNet head ----------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, ops: Any, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(ops.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), ops.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = 1000 * t[:, None].float() * self.get_buffer("freqs")[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, ops: Any) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = ops.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), ops.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class SConv1d(nn.Module):
    """encodec's ``SConv1d`` (not causal): reflect padding split around the kernel, then the convolution."""

    def __init__(self, cin: int, cout: int, kernel_size: int, ops: Any, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Module()
        self.conv.add_module("conv", ops.Conv1d(cin, cout, kernel_size, dilation=dilation))
        self.kernel_size = (kernel_size - 1) * dilation + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + total) / 1 + 1
        extra = (math.ceil(n_frames) - 1) + (self.kernel_size - total) - length
        right = total // 2
        left = total - right
        x = _pad_reflect(x, left, right + extra)
        return self.conv.get_submodule("conv")(x)


def _pad_reflect(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """encodec's ``pad1d`` in reflect mode: zeros first when the input is too short to reflect."""
    length = x.shape[-1]
    extra = 0
    if length <= max(left, right):
        extra = max(left, right) - length + 1
        x = F.pad(x, (0, extra))
    padded = F.pad(x, (left, right), "reflect")
    return padded[..., : padded.shape[-1] - extra]


class WaveNet(nn.Module):
    """VITS' ``WN`` with encodec convolutions (``s2mel/modules/wavenet.py``); dropout is inactive in inference."""

    def __init__(self, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int, ops: Any) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.cond_layer = SConv1d(gin_channels, 2 * hidden * n_layers, 1, ops)
        self.in_layers = nn.ModuleList(SConv1d(hidden, 2 * hidden, kernel_size, ops, dilation=dilation_rate**i) for i in range(n_layers))
        self.res_skip_layers = nn.ModuleList(SConv1d(hidden, 2 * hidden if i < n_layers - 1 else hidden, 1, ops) for i in range(n_layers))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * self.hidden : (i + 1) * 2 * self.hidden, :]
            t_act, s_act = torch.split(x_in + g_l, self.hidden, dim=1)
            acts = torch.tanh(t_act) * torch.sigmoid(s_act)
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, : self.hidden, :]) * x_mask
                output = output + res_skip[:, self.hidden :, :]
            else:
                output = output + res_skip
        return output * x_mask


# -- the estimator and the sampler ------------------------------------------------------------------


class DiT(nn.Module):
    def __init__(self, c: DiTConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.c = c
        self.transformer = Transformer(c, ops)
        self.x_embedder = ops.Linear(c.in_channels, c.hidden_dim)  # unused upstream as well; loaded for strictness
        self.cond_embedder = ops.Embedding(c.content_codebook_size, c.hidden_dim)  # unused
        self.cond_projection = ops.Linear(c.content_dim, c.hidden_dim)
        self.t_embedder = TimestepEmbedder(c.hidden_dim, ops)
        self.register_buffer("input_pos", torch.arange(c.block_size))
        self.t_embedder2 = TimestepEmbedder(c.wavenet_hidden_dim, ops)
        self.conv1 = ops.Linear(c.hidden_dim, c.wavenet_hidden_dim)
        self.conv2 = ops.Conv1d(c.wavenet_hidden_dim, c.in_channels, 1)
        self.wavenet = WaveNet(c.wavenet_hidden_dim, c.wavenet_kernel_size, c.wavenet_dilation_rate, c.wavenet_num_layers, c.wavenet_hidden_dim, ops)
        self.final_layer = FinalLayer(c.wavenet_hidden_dim, c.wavenet_hidden_dim, ops)
        self.res_projection = ops.Linear(c.hidden_dim, c.wavenet_hidden_dim)
        self.content_mask_embedder = ops.Embedding(1, c.hidden_dim)  # unused
        self.skip_linear = ops.Linear(c.hidden_dim + c.in_channels, c.hidden_dim)
        self.cond_x_merge_linear = ops.Linear(c.hidden_dim + c.in_channels * 2 + c.style_dim * int(c.style_condition), c.hidden_dim)

    def forward(self, x: torch.Tensor, prompt_x: torch.Tensor, x_lens: torch.Tensor, t: torch.Tensor, style: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x``, ``prompt_x``: (B, mels, T); ``cond``: (B, T, content); ``style``: (B, style); ``t``: (B,)."""
        _, _, length = x.size()
        t1 = self.t_embedder(t)
        cond = self.cond_projection(cond)
        x = x.transpose(1, 2)
        x_in = torch.cat([x, prompt_x.transpose(1, 2), cond], dim=-1)
        if self.c.style_condition:
            x_in = torch.cat([x_in, style[:, None, :].repeat(1, length, 1)], dim=-1)
        x_in = self.cond_x_merge_linear(x_in)
        x_mask = sequence_mask(x_lens, max_length=x_in.size(1)).to(x.device).unsqueeze(1)
        mask = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)
        x_res = self.transformer(x_in, t1.unsqueeze(1), mask)
        if self.c.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        h = self.conv1(x_res).transpose(1, 2)
        t2 = self.t_embedder2(t)
        h = self.wavenet(h, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(x_res)
        h = self.final_layer(h, t1).transpose(1, 2)
        return self.conv2(h)


class CFM(nn.Module):
    """``CFM`` with a DiT estimator; parameters under ``estimator.``."""

    def __init__(self, c: DiTConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.in_channels = c.in_channels
        self.estimator = DiT(c, ops)
        self.compiled_estimator: Compiled | None = None
        """``compute.compile``: the estimator compiled (``mizuoto.runtime.compile``), called instead of it."""

    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        n_timesteps: int,
        generator: torch.Generator,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
        check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        b, t = mu.size(0), mu.size(1)
        z = torch.randn([b, self.in_channels, t], device=mu.device, generator=generator) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate, check)

    def solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float,
        check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        t = t_span[0]
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        estimator = self.compiled_estimator or self.estimator
        for step in range(1, len(t_span)):
            if check is not None:
                check()
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                out = estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    x_lens,
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                dphi_dt, cfg_dphi_dt = out.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0
        return x


class InterpolateRegulator(nn.Module):
    """Seed-VC's length regulator for continuous content: project, stretch to ``ylens`` (nearest), convolve."""

    def __init__(self, channels: int, sampling_ratios: list[int], in_channels: int, codebook_size: int, ops: Any = default_ops.default, groups: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in sampling_ratios:
            layers += [ops.Conv1d(channels, channels, 3, 1, 1), ops.GroupNorm(groups, channels), nn.Mish()]
        layers.append(ops.Conv1d(channels, channels, 1, 1))
        self.interpolate = len(sampling_ratios) > 0
        self.model = nn.Sequential(*layers)
        self.embedding = ops.Embedding(codebook_size, channels)  # for discrete content; unused here
        self.mask_token = nn.Parameter(torch.empty(1, channels))  # unused
        self.content_in_proj = ops.Linear(in_channels, channels)

    def forward(self, x: torch.Tensor, ylens: torch.Tensor) -> torch.Tensor:
        """(B, T, in) content → (B, max(ylens), channels), zero past each length."""
        x = self.content_in_proj(x)
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=int(ylens.max()), mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, : x.size(2), :]
        return self.model(x).transpose(1, 2).contiguous() * mask
