"""VoxCPM's model (upstream ``model/voxcpm.py``, ``model/voxcpm2.py``, ``modules/locenc``, ``modules/locdit``,
``modules/layers/scalar_quantization_layer.py``), for inference.

Each step of the loop:
1. The local DiT, a flow-matching estimator, makes the next patch of VAE latents. Its inputs are the text-semantic
   LM's hidden state, the residual LM's and the previous patch. It runs ``UnifiedCFM``: Euler steps on upstream's sway
   grid, the first one zero (CFG-zero-star), then classifier-free guidance scaled by the optimised projection.
2. The local encoder embeds the new patch back into the LMs.
3. A stop head decides when to finish.

Every expression keeps upstream's shapes and dtypes, so a seeded run on the CPU gives upstream's latents bit for bit.
The CFM's noise is the loop's only draw. It comes from the generator the caller passes, where upstream draws from the
process-wide one it seeds per attempt.

Generation 1 (VoxCPM 0.5B, 1.5) adds the LM's and the residual LM's states. Generation 2 (VoxCPM2) concatenates them
through ``fusion_concat_proj``, feeds the DiT both (two conditioning tokens), has reference-audio tokens and streams
through the VAE's stateful decoder.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from mizuoto.engines.voxcpm.config import LmConfig, VoxCpmConfig
from mizuoto.engines.voxcpm.minicpm import MiniCPMModel, attend_span
from mizuoto.engines.voxcpm.vae import AudioVAE
from mizuoto.runtime import ops

log = logging.getLogger(__name__)

AUDIO_START = 101
REF_AUDIO_START = 103
REF_AUDIO_END = 104


PROJ_MODULES = {1: ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"], 2: ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]}
"""``LoRAConfig.target_proj_modules``' default per generation."""


@dataclass
class LoraConfig:
    """Upstream's ``LoRAConfig`` (``lora_config.json``'s ``lora_config``)."""

    enable_lm: bool = False
    enable_dit: bool = False
    enable_proj: bool = False
    r: int = 8
    alpha: float = 16
    dropout: float = 0.0
    target_modules_lm: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    target_modules_dit: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    target_proj_modules: list[str] | None = None
    """None: the generation's projections (``PROJ_MODULES``)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def proj_modules(self, version: int) -> list[str]:
        return list(self.target_proj_modules) if self.target_proj_modules is not None else list(PROJ_MODULES[version])

    def to_dict(self, version: int) -> dict[str, Any]:
        """As upstream's ``model_dump()``."""
        return {**asdict(self), "target_proj_modules": self.proj_modules(version)}


class LoRALinear(nn.Module):
    """Upstream's ``LoRALinear``: the base layer's own ``weight``/``bias`` (keys unchanged) plus ``lora_A``, ``lora_B`` and
    a non-persistent ``scaling`` buffer (``alpha / r``), which casts with the model as upstream's does."""

    scaling: torch.Tensor

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self._base_scaling = alpha / r if r > 0 else 0.0
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)
        self.weight = base.weight
        self.bias = base.bias
        self.lora_A: nn.Parameter | None = nn.Parameter(torch.zeros(r, self.in_features)) if r > 0 else None
        self.lora_B: nn.Parameter | None = nn.Parameter(torch.zeros(self.out_features, r)) if r > 0 else None
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.p = dropout
        self.generator: torch.Generator | None = None
        """Training sets it: the dropout's masks drawn from it as CPU ``F.dropout`` draws them (Bernoulli, 1 / (1 - p))."""
        self.streamed: torch.device | None = None
        """Training may set it (the lowest memory preset): the frozen weight rests in RAM and is copied to this device for
        each product (``mizuoto.training.memory.streamed_linear``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.streamed is not None:
            from mizuoto.training.memory import streamed_linear

            result = streamed_linear(x, self.weight, self.bias, self.streamed)
        else:
            result = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return result
        lora_out = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
        if self.training and self.p > 0 and self.generator is not None:
            noise = torch.empty_like(lora_out).bernoulli_(1 - self.p, generator=self.generator)
            noise.div_(1 - self.p)
            return result + lora_out * noise * self.scaling
        return result + self.dropout(lora_out) * self.scaling


def _lora_named(root: nn.Module, targets: list[str], cfg: LoraConfig) -> list[LoRALinear]:
    """``apply_lora_to_named_linear_modules``: every ``nn.Linear`` named one of ``targets``, in module order."""
    made = []
    for full_name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, short = full_name.rpartition(".")
        if short not in targets:
            continue
        parent = root.get_submodule(parent_name) if parent_name else root
        layer = LoRALinear(module, cfg.r, cfg.alpha, cfg.dropout)
        setattr(parent, short, layer)
        made.append(layer)
    return made


def apply_lora(model: VoxCpmModel, cfg: LoraConfig) -> list[LoRALinear]:
    """``_apply_lora``: the language models, then the DiT, then the projections; the layers made, in that order."""
    made: list[LoRALinear] = []
    if cfg.enable_lm:
        for lm in (model.base_lm, model.residual_lm):
            made += _lora_named(lm, cfg.target_modules_lm, cfg)
    if cfg.enable_dit:
        made += _lora_named(model.feat_decoder.estimator, cfg.target_modules_dit, cfg)
    if cfg.enable_proj:
        for name in cfg.proj_modules(model.version):
            module = getattr(model, name, None)
            if isinstance(module, nn.Linear):
                layer = LoRALinear(module, cfg.r, cfg.alpha, cfg.dropout)
                setattr(model, name, layer)
                made.append(layer)
    return made


class ScalarQuantization(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.in_proj = ops.default.Linear(in_dim, latent_dim)
        self.out_proj = ops.default.Linear(latent_dim, out_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.in_proj(hidden))
        if self.training:  # straight-through rounding
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()
        else:
            hidden = torch.round(hidden * self.scale) / self.scale
        return self.out_proj(hidden)


class LocalEncoder(nn.Module):
    """``VoxCPMLocEnc``: a patch → one embedding (the output at a learned CLS position)."""

    def __init__(self, config: LmConfig, input_dim: int) -> None:
        super().__init__()
        self.special_token = nn.Parameter(torch.empty(1, 1, 1, config.hidden_size))
        self.in_proj = ops.default.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = MiniCPMModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = x.shape
        x = self.in_proj(x)
        x = torch.cat([self.special_token.expand(b, t, 1, -1), x], dim=2)
        x = x.reshape(b * t, x.shape[2], x.shape[3])
        outputs, _ = self.encoder(x, is_causal=False)
        return outputs[:, 0, :].reshape(b, t, -1)


def sinusoidal(x: torch.Tensor, dim: int, scale: int = 1000) -> torch.Tensor:
    """``SinusoidalPosEmb``: computed in ``x``'s dtype, as upstream."""
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = ops.default.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = ops.default.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class LocalDiT(nn.Module):
    """``VoxCPMLocDiT`` (v1: ``mu + t`` as one token) and ``local_dit_v2`` (v2: ``mu``'s tokens, then ``t``)."""

    def __init__(self, config: LmConfig, in_channels: int, version: int) -> None:
        super().__init__()
        self.version = version
        self.hidden_size = config.hidden_size
        self.in_proj = ops.default.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = ops.default.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = ops.default.Linear(config.hidden_size, in_channels, bias=True)
        self.time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.delta_time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.decoder = MiniCPMModel(config)

    def forward(self, x: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x.transpose(1, 2).contiguous())
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)
        t = self.time_mlp(sinusoidal(t, self.hidden_size).to(x.dtype))
        dt = self.delta_time_mlp(sinusoidal(dt, self.hidden_size).to(x.dtype))
        t = t + dt
        if self.version == 2:
            mu = mu.view(x.size(0), -1, x.size(-1))
            x = torch.cat([mu, t.unsqueeze(1), cond, x], dim=1)
            skip = prefix + mu.size(1) + 1
        else:
            x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)
            skip = prefix + 1
        hidden, _ = self.decoder(x, is_causal=False)
        return self.out_proj(hidden[:, skip:, :]).transpose(1, 2).contiguous()


class FlowMatching(nn.Module):
    """``UnifiedCFM``'s inference (``forward`` + ``solve_euler``)."""

    def __init__(self, in_channels: int, estimator: LocalDiT, mean_mode: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.estimator = estimator

    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        generator: torch.Generator | None,
        cfg_value: float = 1.0,
        temperature: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
        estimator: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        b, _ = mu.shape
        z = torch.randn((b, self.in_channels, patch_size), device=mu.device, dtype=mu.dtype, generator=generator) * temperature
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, t_span, mu, cond, cfg_value, use_cfg_zero_star, estimator or self.estimator)

    @staticmethod
    def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def solve_euler(
        self, x: torch.Tensor, t_span: torch.Tensor, mu: torch.Tensor, cond: torch.Tensor, cfg_value: float, use_cfg_zero_star: bool, estimator: Callable[..., torch.Tensor]
    ) -> torch.Tensor:
        t, dt = t_span[0], t_span[0] - t_span[1]
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                b = x.size(0)
                x_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * b, self.in_channels, cond.size(2)], device=x.device, dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond
                out = estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(out, [b, b], dim=0)
                if use_cfg_zero_star:
                    st_star = self.optimized_scale(dphi_dt.view(b, -1), cfg_dphi_dt.view(b, -1)).view(b, *([1] * (dphi_dt.dim() - 1)))
                else:
                    st_star = 1.0
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
        return x


class VoxCpmModel(nn.Module):
    """The weights of ``model.safetensors`` / ``pytorch_model.bin`` (everything but the VAE), with their keys."""

    def __init__(self, config: VoxCpmConfig) -> None:
        super().__init__()
        self.config = config
        self.version = config.version
        self.patch_size = config.patch_size
        self.feat_dim = config.feat_dim
        lm = config.lm
        self.base_lm = MiniCPMModel(lm)
        self.residual_lm = MiniCPMModel(config.residual_lm())
        self.feat_encoder = LocalEncoder(config.block_lm(config.encoder), config.feat_dim)
        self.feat_decoder = FlowMatching(config.feat_dim, LocalDiT(config.block_lm(config.dit), config.feat_dim, config.version), mean_mode=config.dit_mean_mode)
        self.fsq_layer = ScalarQuantization(lm.hidden_size, lm.hidden_size, config.scalar_quantization_latent_dim, config.scalar_quantization_scale)
        self.enc_to_lm_proj = ops.default.Linear(config.encoder.hidden_dim, lm.hidden_size)
        self.lm_to_dit_proj = ops.default.Linear(lm.hidden_size, config.dit.hidden_dim)
        self.res_to_dit_proj = ops.default.Linear(lm.hidden_size, config.dit.hidden_dim)
        if self.version == 2:
            self.fusion_concat_proj = ops.default.Linear(lm.hidden_size * 2, lm.hidden_size)
        self.stop_proj = ops.default.Linear(lm.hidden_size, lm.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = ops.default.Linear(lm.hidden_size, 2, bias=False)

    @property
    def scale_emb(self) -> float:
        return self.config.lm.scale_emb if self.config.lm.use_mup else 1.0


def latent_patches(vae: AudioVAE, audio: torch.Tensor, patch_size: int, padding: str, device: torch.device) -> torch.Tensor:
    """A clip (1, T) at the VAE's rate → (patches, patch_size, latent_dim) on the CPU: padded to whole patches on the left
    (a continuation prompt) or the right (a reference), encoded, cut into patches, as upstream's ``_encode_wav``."""
    patch_len = patch_size * vae.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        audio = torch.nn.functional.pad(audio, (padding_size, 0) if padding == "left" else (0, padding_size))
    feat = vae.encode(audio.to(device)).cpu()
    return feat.view(vae.latent_dim, -1, patch_size).permute(1, 2, 0)


def build_inputs(
    tokens: Callable[[str], list[int]], target_text: str, prompt_text: str | None, prompt_feat: torch.Tensor | None, ref_feat: torch.Tensor | None, patch_size: int, latent_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sequence of ``_generate_with_prompt_cache``: (text tokens, text mask, audio patches, audio mask), unbatched.

    ``[ref_start ref… ref_end]`` (a reference), the text (the prompt's transcript first in a continuation) and
    ``audio_start``, then the prompt's patches.
    """
    continuation = prompt_feat is not None
    text = (prompt_text or "") + target_text if continuation else target_text
    text_token = torch.LongTensor(tokens(text))
    text_token = torch.cat([text_token, torch.tensor([AUDIO_START], dtype=torch.int32, device=text_token.device)], dim=-1)
    text_length = text_token.shape[0]
    text_pad_feat = torch.zeros((text_length, patch_size, latent_dim), dtype=torch.float32)
    token_parts, feat_parts, text_mask, audio_mask = [], [], [], []
    if ref_feat is not None:
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, patch_size, latent_dim), dtype=torch.float32)
        token_parts.append(
            torch.cat([torch.tensor([REF_AUDIO_START], dtype=torch.int32), torch.zeros(ref_len, dtype=torch.int32), torch.tensor([REF_AUDIO_END], dtype=torch.int32)])
        )
        feat_parts.append(torch.cat([z1, ref_feat, z1], dim=0))
        text_mask.append(torch.cat([torch.tensor([1], dtype=torch.int32), torch.zeros(ref_len, dtype=torch.int32), torch.tensor([1], dtype=torch.int32)]))
        audio_mask.append(torch.cat([torch.tensor([0], dtype=torch.int32), torch.ones(ref_len, dtype=torch.int32), torch.tensor([0], dtype=torch.int32)]))
    token_parts.append(text_token)
    feat_parts.append(text_pad_feat)
    text_mask.append(torch.ones(text_length, dtype=torch.int32))
    audio_mask.append(torch.zeros(text_length, dtype=torch.int32))
    if continuation:
        assert prompt_feat is not None
        audio_length = prompt_feat.size(0)
        token_parts.append(torch.zeros(audio_length, dtype=torch.int32))
        feat_parts.append(prompt_feat)
        text_mask.append(torch.zeros(audio_length, dtype=torch.int32))
        audio_mask.append(torch.ones(audio_length, dtype=torch.int32))
    elif ref_feat is None:  # zero-shot: upstream appends an empty prompt
        feat_parts.append(torch.empty((0, patch_size, latent_dim), dtype=torch.float32))
    return torch.cat(token_parts), torch.cat(text_mask), torch.cat(feat_parts, dim=0), torch.cat(audio_mask)


class PatchDecoder:
    """The decoding steps after the prefill, one patch each, over fixed tensors: eagerly, or on an NVIDIA GPU as CUDA
    graphs (``compute.cuda_graphs``; ``mizuoto.runtime.graphs``).

    A patch is the local DiT's Euler steps (``feat_decoder``, its noise drawn from the request's generator), the local
    encoder on the new patch, the stop head on the language model's last state, and both language models' step against
    their static caches. The language models step before the stop flag is read (upstream reads it first and stops
    before stepping): they draw nothing, and their caches are dropped when generation ends, so the patches are the same,
    and a patch costs one wait (the flag) instead of three. Captured, a patch is one graph per attention span
    (``attend_span``: the whole cache on CUDA, so one) and setting; the caches are then kept with the model, since the
    graphs hold their addresses.
    """

    def __init__(self, model: VoxCpmModel, graphs: Callable[[], Any] | None = None) -> None:
        self.m = model
        self._graphs_for = graphs or (lambda: None)
        self.graphs: Any = None
        self._captured: dict[tuple[Any, ...], Any] = {}
        self.failed = False
        self.graphed = False
        self.caches: tuple[Any, Any] | None = None
        self._ports: dict[str, torch.Tensor] = {}
        self.generator: torch.Generator | None = None

    def begin(self) -> None:
        """A generation: the graphs to use now (new ones when the weights moved), else eager."""
        graphs = None if self.failed else self._graphs_for()
        if graphs is not self.graphs:
            self.graphs, self._captured, self.caches, self._ports = graphs, {}, None, {}
        self.graphed = graphs is not None

    def new_caches(self, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
        """The language models' static caches: new ones eagerly (as upstream), the same ones while graphs are kept."""
        m = self.m
        if self.graphs is None:
            return m.base_lm.new_cache(m.config.max_length, device, dtype), m.residual_lm.new_cache(m.config.max_length, device, dtype)
        if self.caches is None:
            self.caches = (m.base_lm.new_cache(m.config.max_length, device, dtype), m.residual_lm.new_cache(m.config.max_length, device, dtype))
        return self.caches

    def start(self, lm_hidden: torch.Tensor, residual_hidden: torch.Tensor, prefix_feat_cond: torch.Tensor) -> None:
        if not self._ports:
            self._ports = {
                "lm_hidden": torch.zeros_like(lm_hidden),
                "residual_hidden": torch.zeros_like(residual_hidden),
                "cond": torch.zeros_like(prefix_feat_cond),
                "pred": torch.zeros_like(prefix_feat_cond),
                "stop": torch.zeros((), dtype=torch.long, device=lm_hidden.device),
                "base_pos": torch.zeros(1, dtype=torch.long, device=lm_hidden.device),
                "residual_pos": torch.zeros(1, dtype=torch.long, device=lm_hidden.device),
            }
        self._ports["lm_hidden"].copy_(lm_hidden)
        self._ports["residual_hidden"].copy_(residual_hidden)
        self._ports["cond"].copy_(prefix_feat_cond)

    @property
    def pred(self) -> torch.Tensor:
        return self._ports["pred"]

    def stop(self) -> int:
        """The stop head's choice on the patch's language model state: waits for the device."""
        return int(self._ports["stop"].cpu().item())

    def step(self, timesteps: int, cfg_value: float, generator: torch.Generator | None, estimator: Callable[..., torch.Tensor] | None) -> bool:
        """One patch: ``pred`` and ``stop``, and the language models' next states. False when a cache was full (the
        language models did not step: upstream raises there unless the patch stops)."""
        m = self.m
        base, residual = m.base_lm.kv_cache, m.residual_lm.kv_cache
        assert base is not None and residual is not None
        lm = base.current_length < base.max_length and residual.current_length < residual.max_length
        if lm:  # positions as upstream's ``step()`` counts them; the span follows from the length counted
            self._ports["base_pos"].fill_(base.step())
            self._ports["residual_pos"].fill_(residual.step())
        graphs = self.graphs if estimator is None and lm and generator is not None else None
        if graphs is None:
            self.generator = generator
            self._patch(timesteps, cfg_value, estimator, lm)
            return lm
        device = self._ports["base_pos"].device
        key = (attend_span(base.current_length, base.max_length, device), attend_span(residual.current_length, residual.max_length, device), timesteps, cfg_value)
        with torch.cuda.device(graphs.device), graphs.drawing(generator) as own:
            self.generator = own
            graph = self._captured.get(key)
            if graph is None:
                graph = self._capture(lambda: graphs.capture(lambda: self._patch(timesteps, cfg_value, None, True), list(self._ports.values())))
                if graph is not None:
                    while len(self._captured) >= 4:
                        self._captured.pop(next(iter(self._captured)))
                    self._captured[key] = graph
            self.generator = graphs.generator  # a fresh one, in the same state, after a failed capture
            if graph is None:
                self._patch(timesteps, cfg_value, None, True)
            else:
                graph.replay()
        return True

    def _patch(self, timesteps: int, cfg_value: float, estimator: Callable[..., torch.Tensor] | None, lm: bool) -> None:
        m, ports = self.m, self._ports
        lm_hidden, residual_hidden, cond = ports["lm_hidden"], ports["residual_hidden"], ports["cond"]
        if m.version == 2:
            dit_hidden = torch.cat((m.lm_to_dit_proj(lm_hidden), m.res_to_dit_proj(residual_hidden)), dim=-1)
        else:
            dit_hidden = m.lm_to_dit_proj(lm_hidden) + m.res_to_dit_proj(residual_hidden)
        p = cond.shape[1]
        pred_feat = m.feat_decoder(dit_hidden, timesteps, p, cond.transpose(1, 2).contiguous(), self.generator, cfg_value=cfg_value, estimator=estimator).transpose(1, 2)
        curr_embed = m.enc_to_lm_proj(m.feat_encoder(pred_feat.unsqueeze(1)))
        ports["pred"].copy_(pred_feat)
        ports["stop"].copy_(m.stop_head(m.stop_actn(m.stop_proj(lm_hidden))).argmax(dim=-1)[0])
        cond.copy_(pred_feat)
        if not lm:
            return
        new_lm = m.base_lm.forward_step(curr_embed[:, 0, :], ports["base_pos"]).clone()
        new_lm = m.fsq_layer(new_lm)
        if m.version == 2:
            residual_input = m.fusion_concat_proj(torch.cat((new_lm, curr_embed[:, 0, :]), dim=-1))
        else:
            residual_input = new_lm + curr_embed[:, 0, :]
        new_residual = m.residual_lm.forward_step(residual_input, ports["residual_pos"]).clone()
        lm_hidden.copy_(new_lm)
        residual_hidden.copy_(new_residual)

    def _capture(self, make: Callable[[], Any]) -> Any:
        from mizuoto.runtime.device import is_oom

        try:
            return make()
        except Exception as e:  # any failure to capture means running eagerly (the same computation)
            if is_oom(e):
                raise
            log.warning("VoxCPM: CUDA graphs failed (%s: %s); decoding eagerly", type(e).__name__, e)
            self.failed = True
            self._captured = {}
            return None


class Generation:
    """One ``_inference`` run: the prefill, then a patch per ``step()``. Iterate it for (latent, patches so far, context)."""

    def __init__(
        self,
        model: VoxCpmModel,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        *,
        min_len: int,
        max_len: int,
        timesteps: int,
        cfg_value: float,
        streaming: bool,
        prefix_len: int,
        generator: torch.Generator | None,
        cache_dtype: torch.dtype,
        check: Callable[[], None] | None = None,
        estimator: Callable[..., torch.Tensor] | None = None,
        decoder: PatchDecoder | None = None,
    ) -> None:
        self.m = model
        self.text, self.text_mask, self.feat, self.feat_mask = text, text_mask, feat, feat_mask
        self.min_len, self.max_len, self.timesteps, self.cfg_value = min_len, max_len, timesteps, cfg_value
        self.streaming, self.prefix_len, self.generator = streaming, prefix_len, generator
        self.cache_dtype = cache_dtype
        self.check = check
        self.estimator = estimator
        self.decoder = decoder

    def __iter__(self) -> Generator[tuple[torch.Tensor, Any, int], None, None]:
        m, feat, feat_mask, text_mask = self.m, self.feat, self.feat_mask, self.text_mask
        b, _, p, _ = feat.shape
        device = feat.device
        decoder = self.decoder or PatchDecoder(m)
        decoder.begin()
        m.base_lm.kv_cache, m.residual_lm.kv_cache = decoder.new_caches(device, self.cache_dtype)
        try:
            feat_embed = m.enc_to_lm_proj(m.feat_encoder(feat))
            text_embed = m.base_lm.embed_tokens(self.text) * m.scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
            prefix_feat_cond = feat[:, -1, ...]
            context_len = 0
            pred_feat_seq: list[torch.Tensor] = []
            if m.version == 2:
                if feat_mask[0, -1].item() == 1:
                    audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
                    context_len = min(self.prefix_len - 1, len(audio_indices))
                    pred_feat_seq = list(feat[:, audio_indices[-context_len:], :, :].split(1, dim=1))
            else:
                audio_patch_count = int(feat_mask.sum().item())
                if audio_patch_count > 0:
                    context_len = min(self.prefix_len - 1, audio_patch_count)
                    pred_feat_seq = list(feat[:, -context_len:, :, :].split(1, dim=1))
            enc_outputs, kv = m.base_lm(combined_embed, is_causal=True)
            m.base_lm.kv_cache.fill_caches(kv)
            del kv
            enc_outputs = m.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            lm_hidden = enc_outputs[:, -1, :]
            if m.version == 2:
                residual_inputs = m.fusion_concat_proj(torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1))
            else:
                residual_inputs = enc_outputs + feat_mask.unsqueeze(-1) * feat_embed
            residual_outputs, rkv = m.residual_lm(residual_inputs, is_causal=True)
            m.residual_lm.kv_cache.fill_caches(rkv)
            del rkv
            decoder.start(lm_hidden, residual_outputs[:, -1, :], prefix_feat_cond)
            for i in range(self.max_len):
                if self.check is not None:
                    self.check()
                stepped = decoder.step(self.timesteps, self.cfg_value, self.generator, self.estimator)
                pred_feat = decoder.pred.clone()
                pred_feat_seq.append(pred_feat.unsqueeze(1))
                if self.streaming:
                    if m.version == 2:
                        yield to_latent(pred_feat.unsqueeze(1)), pred_feat_seq, context_len
                        if len(pred_feat_seq) > self.prefix_len:
                            pred_feat_seq = pred_feat_seq[-self.prefix_len :]
                    else:
                        yield to_latent(torch.cat(pred_feat_seq[-self.prefix_len :], dim=1)), pred_feat_seq, context_len
                stop_flag = decoder.stop()  # the patch's one wait
                if i > self.min_len and stop_flag == 1:
                    break
                if not stepped:
                    raise ValueError("KV cache is full")  # where upstream's step raises
            if not self.streaming:
                seq = torch.cat(pred_feat_seq, dim=1)
                generated = seq[:, context_len:, :, :].squeeze(0).cpu() if m.version == 2 else seq.squeeze(0).cpu()
                yield to_latent(seq), generated, context_len
        finally:
            m.base_lm.kv_cache = None
            m.residual_lm.kv_cache = None


def to_latent(patches: torch.Tensor) -> torch.Tensor:
    """``rearrange(x, "b t p d -> b d (t p)")``."""
    b, t, p, d = patches.shape
    return patches.permute(0, 3, 1, 2).reshape(b, d, t * p)
