"""Drama 3's ``config.json`` (upstream ``fish_omni/config.py``): the backbone's text config, the codebook tower's and the
semantic band, checked as upstream checks them (flags its architecture fixes, geometry, the band inside the vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

MODEL_TYPE = "fish_omni"
TEXT_MODEL_TYPES = ("fish_omni_text", "qwen3_5_moe_text")
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"
FP8_QUANT_METHOD = "bsgemm-moe"
FP8_BLOCK = 128
"""The block edge of the experts' FP8 weights and their scales."""
FIXED_TEXT_FLAGS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", True, True),
    ("shared_expert_gate", True, True),
)
"""(key, default when absent, the value the architecture needs)."""
FIXED_TOWER_FLAGS = {
    "decoder_type": "codebook_mtp_shared",
    "rotary_interleaved": True,
    "attention_output_gate": True,
    "layernorm_zero_centered_gamma": True,
    "use_moe": False,
    "mtp_use_swa": False,
    "attention_qkv_bias": False,
    "attention_o_bias": False,
    "dropout": 0.0,
}
"""``attention_qk_norm`` is not among them: the tower always norms q and k (see ``tower.TowerAttention``)."""


class ConfigError(ValueError):
    """A config this port does not run (as upstream refuses it)."""


@dataclass(kw_only=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    max_position_embeddings: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool = True

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers or not set(self.layer_types) <= {"linear_attention", "full_attention"}:
            raise ConfigError(f"layer_types must name linear_attention / full_attention for each of the {self.num_hidden_layers} layers; got {self.layer_types!r}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ConfigError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ConfigError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ConfigError("num_experts_per_tok exceeds num_experts")
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ConfigError(f"rotary width must be positive and even; got {self.rotary_dim}")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextConfig:
        if data.get("model_type") not in TEXT_MODEL_TYPES:
            raise ConfigError(f"text_config.model_type={data.get('model_type')!r}; expected one of {TEXT_MODEL_TYPES}")
        if any(t != "moe" for t in data.get("mlp_layer_types") or ()):
            raise ConfigError("every text_config.mlp_layer_types must be moe")
        for key, default, required in FIXED_TEXT_FLAGS:
            if data.get(key, default) is not required:
                raise ConfigError(f"text_config.{key} must be {required}; got {data.get(key)!r}")
        if data.get("tie_word_embeddings", False):
            raise ConfigError("the backbone keeps a separate lm_head")
        rope = data.get("rope_parameters") or data
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ConfigError("only the default rope_type is supported")
        layers = data["num_hidden_layers"]
        layer_types = data.get("layer_types") or ["full_attention" if (i + 1) % data["full_attention_interval"] == 0 else "linear_attention" for i in range(layers)]
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            num_hidden_layers=layers,
            layer_types=list(layer_types),
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            rope_theta=rope["rope_theta"],
            # older configs keep the factor at the top level; 0.25 is the Hugging Face default
            partial_rotary_factor=rope.get("partial_rotary_factor", data.get("partial_rotary_factor", 0.25)),
            rms_norm_eps=data["rms_norm_eps"],
            max_position_embeddings=data["max_position_embeddings"],
            linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
            linear_key_head_dim=data["linear_key_head_dim"],
            linear_value_head_dim=data["linear_value_head_dim"],
            linear_num_key_heads=data["linear_num_key_heads"],
            linear_num_value_heads=data["linear_num_value_heads"],
            num_experts=data["num_experts"],
            num_experts_per_tok=data["num_experts_per_tok"],
            moe_intermediate_size=data["moe_intermediate_size"],
            shared_expert_intermediate_size=data["shared_expert_intermediate_size"],
            norm_topk_prob=data.get("norm_topk_prob", True),
        )


@dataclass(kw_only=True, frozen=True)
class TowerConfig:
    dim: int
    intermediate_size: int
    n_head: int
    n_local_heads: int
    head_dim: int
    norm_eps: float
    rope_base: float
    rotary_percent: float
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int

    def __post_init__(self) -> None:
        if self.n_head % self.n_local_heads or not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ConfigError("invalid tower attention geometry")
        if self.num_codebooks < 2 or not 0 <= self.vq_backbone_codebooks < self.num_codebooks:
            raise ConfigError("invalid codebook counts")
        if self.shared_enrichment_layers < 1:
            raise ConfigError("shared_enrichment_layers must be >= 1")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TowerConfig:
        for key, value in FIXED_TOWER_FLAGS.items():
            if data.get(key, value) != value:
                raise ConfigError(f"unsupported audio_decoder_config.{key}={data[key]!r}")
        if data.get("text_dim", data["dim"]) != data["dim"]:
            raise ConfigError("audio_decoder_config.text_dim must equal dim")
        if data.get("n_layer") != data["shared_enrichment_layers"]:
            raise ConfigError("n_layer must equal shared_enrichment_layers")
        placement = data.get("vq_feature_placement")
        if placement not in ("backbone_partial", "mtp_shared_pre") or (placement == "mtp_shared_pre" and data["vq_backbone_codebooks"] != 0):
            raise ConfigError(f"unsupported vq_feature_placement {placement!r}")
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(kw_only=True)
class Drama3Config:
    text: TextConfig
    tower: TowerConfig
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    fp8_experts: bool = False
    """The checkpoint's experts are block-FP8 (``bsgemm-moe``), with float32 scales per 128 × 128 block."""

    def __post_init__(self) -> None:
        start, end = self.semantic_start_token_id, self.semantic_end_token_id
        if not 0 <= start <= end < self.text.vocab_size:
            raise ConfigError("the semantic band must lie inside the text vocabulary")
        if start <= self.eos_token_id <= end:
            raise ConfigError("eos_token_id lies inside the semantic band")
        if self.tower.dim != self.text.hidden_size:
            raise ConfigError("tower and backbone hidden sizes must match")
        if self.tower.vocab_size != end - start + 1:
            raise ConfigError("tower vocabulary must match the semantic token band")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Drama3Config:
        if data.get("model_type") != MODEL_TYPE:
            raise ConfigError(f"model_type={data.get('model_type')!r}; expected {MODEL_TYPE!r}")
        if data.get("audio_encoder_config") is not None:
            raise ConfigError("audio_encoder_config is not supported")
        if data.get("speech_template_version") != SPEECH_TEMPLATE_VERSION:
            raise ConfigError(f"speech_template_version={data.get('speech_template_version')!r}; only {SPEECH_TEMPLATE_VERSION} is supported")
        quantization = data.get("quantization_config")
        if quantization is not None and (quantization.get("quant_method") != FP8_QUANT_METHOD or quantization.get("weight_block_size") != [FP8_BLOCK] * 2):
            raise ConfigError(f"only {FP8_QUANT_METHOD} quantization with weight_block_size [{FP8_BLOCK}, {FP8_BLOCK}] is supported")
        eos = data.get("eos_token_id", data["text_config"].get("eos_token_id"))
        if eos is None:
            raise ConfigError("config has no eos_token_id")
        return cls(
            text=TextConfig.from_dict(data["text_config"]),
            tower=TowerConfig.from_dict(data["audio_decoder_config"]),
            semantic_start_token_id=data["semantic_start_token_id"],
            semantic_end_token_id=data["semantic_end_token_id"],
            eos_token_id=eos,
            fp8_experts=quantization is not None,
        )


def is_drama3(data: dict[str, Any]) -> bool:
    """A Drama 3 checkpoint's ``config.json``."""
    return data.get("model_type") == MODEL_TYPE
