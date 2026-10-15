"""VoxCPM's ``config.json``, read as data: the defaults are upstream's pydantic models' (``model/voxcpm.py``,
``model/voxcpm2.py``, ``modules/minicpm4/config.py``, ``modules/audiovae/*.py``, ``modules/locdit/unified_cfm.py``).

Keys upstream's models do not declare are ignored, as pydantic ignores them: VoxCPM2's ``dit_config`` writes
``mean_mode``, which upstream never reads (its field is ``dit_mean_mode``, false), so the DiT runs without mean mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

ARCHITECTURES = {"voxcpm": 1, "voxcpm2": 2}
"""``config.json``'s ``architecture`` (missing: ``voxcpm``) → the model generation."""


@dataclass
class RopeScaling:
    type: str
    long_factor: list[float]
    short_factor: list[float]
    original_max_position_embeddings: int


@dataclass
class LmConfig:
    """``MiniCPM4Config``: one MiniCPM-4 decoder (the LMs, the local encoder and the local DiT are all one)."""

    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: RopeScaling
    vocab_size: int
    scale_emb: float
    dim_model_base: int
    scale_depth: float
    rope_theta: float
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_mup: bool = True
    kv_channels: int | None = None
    no_rope: bool = False

    @property
    def head_dim(self) -> int:
        return self.kv_channels if self.kv_channels else self.hidden_size // self.num_attention_heads


@dataclass
class BlockConfig:
    """``VoxCPMEncoderConfig`` / ``VoxCPMDitConfig``: the local encoder's or DiT's size."""

    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int | None = None
    dit_mean_mode: bool = False


@dataclass
class CfmConfig:
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


@dataclass
class VaeConfig:
    """``AudioVAEConfig`` (v1) and ``AudioVAEConfig`` of ``audio_vae_v2`` (v2: an output rate and rate conditioning)."""

    encoder_dim: int = 128
    encoder_rates: list[int] = field(default_factory=lambda: [2, 5, 8, 8])
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [8, 8, 5, 2])
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False
    out_sample_rate: int | None = None
    sr_bin_boundaries: list[int] | None = None
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False
    version: int = 1

    @property
    def output_rate(self) -> int:
        return self.out_sample_rate or self.sample_rate


VAE_V2_DEFAULTS = VaeConfig(decoder_dim=2048, decoder_rates=[8, 6, 5, 2, 2, 2], out_sample_rate=48000, sr_bin_boundaries=[20000, 30000, 40000], version=2)


@dataclass
class VoxCpmConfig:
    version: int
    lm: LmConfig
    patch_size: int
    feat_dim: int
    residual_lm_num_layers: int
    scalar_quantization_latent_dim: int
    scalar_quantization_scale: int
    encoder: BlockConfig
    dit: BlockConfig
    cfm: CfmConfig
    vae: VaeConfig
    max_length: int
    dtype: str
    residual_lm_no_rope: bool = False
    dit_mean_mode: bool = False

    def residual_lm(self) -> LmConfig:
        return replace(self.lm, num_hidden_layers=self.residual_lm_num_layers, vocab_size=0, no_rope=self.residual_lm_no_rope if self.version == 2 else self.lm.no_rope)

    def block_lm(self, block: BlockConfig) -> LmConfig:
        """The local encoder's or DiT's decoder: the LM's config with the block's sizes, no vocabulary."""
        return replace(
            self.lm,
            hidden_size=block.hidden_dim,
            intermediate_size=block.ffn_dim,
            num_attention_heads=block.num_heads,
            num_hidden_layers=block.num_layers,
            kv_channels=block.kv_channels,
            vocab_size=0,
        )


def _build(cls: Any, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def read_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def architecture(config: dict[str, Any]) -> int | None:
    """The generation a ``config.json`` describes (1 or 2), or None when it is not VoxCPM's."""
    if "lm_config" not in config or "dit_config" not in config:
        return None
    return ARCHITECTURES.get(str(config.get("architecture", "voxcpm")).lower())


def parse_config(data: dict[str, Any]) -> VoxCpmConfig:
    version = architecture(data)
    if version is None:
        raise ValueError("not a VoxCPM config.json")
    lm = dict(data["lm_config"])
    lm["rope_scaling"] = _build(RopeScaling, lm["rope_scaling"])
    dit_data = dict(data["dit_config"])
    cfm = _build(CfmConfig, dit_data.pop("cfm_config", {}) or {})
    defaults = {
        1: {"patch_size": 2, "residual_lm_num_layers": 6, "scalar_quantization_latent_dim": 256, "max_length": 4096},
        2: {"patch_size": 4, "residual_lm_num_layers": 8, "scalar_quantization_latent_dim": 512, "max_length": 8192},
    }[version]
    vae_data = data.get("audio_vae_config")
    base_vae = VaeConfig() if version == 1 else VAE_V2_DEFAULTS
    vae = replace(base_vae, **{k: v for k, v in (vae_data or {}).items() if k in {f.name for f in fields(VaeConfig)}})
    if version == 1:
        vae = replace(vae, out_sample_rate=None, sr_bin_boundaries=None)
    encoder = _build(BlockConfig, data.get("encoder_config") or {})
    dit = _build(BlockConfig, dit_data)
    if version == 1:  # v1's DiT config has no mean mode; the model's own field is read instead
        dit = replace(dit, dit_mean_mode=False)
    return VoxCpmConfig(
        version=version,
        lm=_build(LmConfig, lm),
        patch_size=int(data.get("patch_size", defaults["patch_size"])),
        feat_dim=int(data.get("feat_dim", 64)),
        residual_lm_num_layers=int(data.get("residual_lm_num_layers", defaults["residual_lm_num_layers"])),
        scalar_quantization_latent_dim=int(data.get("scalar_quantization_latent_dim", defaults["scalar_quantization_latent_dim"])),
        scalar_quantization_scale=int(data.get("scalar_quantization_scale", 9)),
        encoder=encoder,
        dit=dit,
        cfm=cfm,
        vae=vae,
        max_length=int(data.get("max_length", defaults["max_length"])),
        dtype=str(data.get("dtype", "bfloat16")),
        residual_lm_no_rope=bool(data.get("residual_lm_no_rope", False)) if version == 2 else False,
        dit_mean_mode=bool(data.get("dit_mean_mode", False)) if version == 1 else dit.dit_mean_mode,
    )


def load_config(model_dir: Path) -> VoxCpmConfig:
    return parse_config(read_config(model_dir))
