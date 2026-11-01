"""The backbone (upstream ``fish_omni/backbone.py``): Qwen3.5-MoE's hybrid text model and its ``lm_head``.

Upstream may spread the layers over several GPUs; here the model manager places them, whole or partly (each layer is two
units, ``memory_units``: its attention, then its MoE, whose experts stream by the row). The residual stream is float32
for a float16 model (it exceeds float16's range), the model's dtype otherwise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mizuoto.runtime import ops as default_ops
from mizuoto_fishspeech.drama3.config import TextConfig
from mizuoto_fishspeech.drama3.layers import FullAttention, GatedDeltaNet, LayerContext, ZeroCenteredRMSNorm
from mizuoto_fishspeech.drama3.moe import FP8, SparseMoeBlock, clamp_to_float16, dequantize_fp8_blocks

CHECKPOINT_PREFIX = "model.language_model."
LM_HEAD = "lm_head.weight"
TOWER_PREFIXES = ("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")
EXPERT_KEY = re.compile(r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)")
"""Per-expert checkpoint tensors (the released FP8 checkpoint), named after the backbone prefix became ``model.``."""
FUSED_EXPERT_KEY = re.compile(r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)")
"""Hugging Face's fused expert tensors (an unquantized checkpoint)."""


class DecoderLayer(nn.Module):
    def __init__(self, c: TextConfig, index: int, fp8: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.layer_type = c.layer_types[index]
        if self.layer_type == "full_attention":
            self.self_attn = FullAttention(c, ops)
        else:
            self.linear_attn = GatedDeltaNet(c, ops)
        self.mlp = SparseMoeBlock(c, fp8, ops)
        self.input_layernorm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_norm_eps)

    @property
    def mixer(self) -> FullAttention | GatedDeltaNet:
        return self.self_attn if self.layer_type == "full_attention" else self.linear_attn

    def forward(self, x: torch.Tensor, context: LayerContext, state: Any) -> torch.Tensor:
        """``x`` is the residual stream, possibly wider than the compute dtype."""
        compute_dtype = self.input_layernorm.weight.dtype
        x = x + self.mixer(self.input_layernorm(x).to(compute_dtype), context, state)
        return x + self.mlp(self.post_attention_layernorm(x).to(compute_dtype))


class TextModel(nn.Module):
    def __init__(self, c: TextConfig, fp8: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.embed_tokens = ops.Embedding(c.vocab_size, c.hidden_size)
        self.layers: Any = nn.ModuleList(DecoderLayer(c, i, fp8, ops) for i in range(c.num_hidden_layers))
        self.norm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_norm_eps)


class Backbone(nn.Module):
    """Qwen3.5-MoE's hybrid text model and its ``lm_head``; parameter names are upstream's ``FishOmniBackbone``'s."""

    def __init__(self, config: TextConfig, fp8: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = config
        self.model: Any = TextModel(config, fp8, ops)
        self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False)

    def memory_units(self) -> list[nn.Module]:
        """Per layer its attention, then its MoE (experts by the row), and the output head."""
        units: list[nn.Module] = []
        for layer in self.model.layers:
            units += [layer.mixer, layer.mlp]
        return [*units, self.lm_head]

    @property
    def dtype(self) -> torch.dtype:
        return self.model.norm.weight.dtype

    @property
    def residual_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == torch.float16 else self.dtype

    def new_states(self, capacity: int, device: torch.device) -> list[Any]:
        """One request's caches: a state per layer, ``capacity`` positions."""
        return [layer.mixer.new_state(capacity, self.dtype, device) for layer in self.model.layers]

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.embed_tokens
        return embeddings(input_ids.to(embeddings.weight.device))

    def forward(self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor, states: list[Any]) -> torch.Tensor:
        """The final-norm hidden states [B, T, H] of rows at the contiguous positions ``input_pos`` (position 0 starts a
        sequence; later ones continue the one in ``states``)."""
        hidden = inputs_embeds.to(self.residual_dtype)
        length = hidden.shape[1]
        start = int(input_pos[0])
        end = start + length
        rotary_dim = self.config.rotary_dim
        device = hidden.device
        exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        frequencies = 1.0 / (self.config.rope_theta ** (exponents / rotary_dim))
        positions = input_pos.to(device)
        angles = positions.float()[:, None] * frequencies[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        context = LayerContext(input_pos=positions, cos=angles.cos(), sin=angles.sin(), kv_len=end, continue_state=start > 0)
        for layer, state in zip(self.model.layers, states, strict=True):
            hidden = layer(hidden, context, state)
        return self.model.norm(hidden).to(self.dtype)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


# -- weights --------------------------------------------------------------------------------------


def checkpoint_tensors(folder: Path, skip: tuple[str, ...] = ()) -> Iterator[tuple[str, torch.Tensor]]:
    """(name, CPU tensor) of every safetensors tensor of the checkpoint not under ``skip``, shard by shard."""
    from safetensors import safe_open

    index = folder / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted({folder / name for name in json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()})
    else:
        shards = [folder / "model.safetensors"]
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 - a safetensors file is not iterable
                if not key.startswith(skip):
                    yield key, f.get_tensor(key)


def backbone_name(name: str) -> str:
    """A checkpoint key as the backbone's parameter name."""
    if name == LM_HEAD:
        return name
    if name.startswith(CHECKPOINT_PREFIX):
        return "model." + name[len(CHECKPOINT_PREFIX) :]
    if name.startswith("model."):
        return name
    raise ValueError(f"checkpoint key {name!r} is not a backbone weight")


@torch.no_grad()
def load_backbone(model: Backbone, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
    """Copy the checkpoint into ``model`` (upstream ``load_backbone_weights``), filling every slot exactly once: stacked
    experts per expert and per gate / up / down projection, everything else whole. FP8 experts loaded into a dense
    model are dequantized in float32, then cast."""
    targets: dict[str, torch.Tensor] = dict(model.named_parameters())
    expected: set[tuple[str, int, str]] = set()
    for name in targets:
        found = re.fullmatch(r"(model\.layers\.\d+\.mlp\.experts)\.(w13|w2)_(weight|scale_inv)", name)
        if found is None:
            expected.add((name, -1, "whole"))
            continue
        experts: Any = model.get_submodule(found.group(1))
        parts = ("gate", "up") if found.group(2) == "w13" else ("down",)
        for e in range(experts.num_experts):
            expected.update((name, e, part) for part in parts)
    loaded: set[tuple[str, int, str]] = set()
    pending: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}

    def fill(name: str, expert: int, part: str, target: torch.Tensor, tensor: torch.Tensor) -> None:
        slot = (name, expert, part)
        if slot not in expected:
            raise ValueError(f"no parameter slot {slot}")
        if slot in loaded:
            raise ValueError(f"parameter slot {slot} filled twice")
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(f"{slot}: checkpoint shape {tuple(tensor.shape)} != {tuple(target.shape)}")
        if target.dtype == FP8 and tensor.dtype != FP8:
            raise ValueError(f"{slot}: expected {FP8}, got {tensor.dtype}")
        if target.dtype == torch.float16 and tensor.dtype != torch.float16:
            tensor = clamp_to_float16(tensor)
        target.copy_(tensor)
        loaded.add(slot)

    for checkpoint_name, tensor in weights:
        name = backbone_name(checkpoint_name)
        expert_key = EXPERT_KEY.fullmatch(name)
        fused_key = FUSED_EXPERT_KEY.fullmatch(name)
        if expert_key is not None:
            prefix = f"model.layers.{expert_key['layer']}.mlp.experts"
            experts: Any = model.get_submodule(prefix)
            expert = int(expert_key["expert"])
            projection = expert_key["proj"]
            is_scale = expert_key["kind"] == "weight_scale_inv"
            if not is_scale and experts.fp8 and tensor.dtype != FP8:
                raise ValueError(f"{checkpoint_name}: expected an FP8 expert weight")
            if not experts.fp8 and (is_scale or tensor.dtype == FP8):
                key = (prefix, expert, projection)
                halves = pending.setdefault(key, {})
                if expert_key["kind"] in halves:
                    raise ValueError(f"{checkpoint_name} appears twice")
                halves[expert_key["kind"]] = tensor
                if len(halves) < 2:
                    continue
                del pending[key]
                target_dtype = experts.w13_weight.dtype
                tensor = dequantize_fp8_blocks(halves["weight"], halves["weight_scale_inv"], torch.float32)
                if target_dtype == torch.float16:
                    tensor = clamp_to_float16(tensor)
                tensor = tensor.to(target_dtype)
                is_scale = False
            suffix = "scale_inv" if is_scale else "weight"
            if projection == "down_proj":
                full = f"{prefix}.w2_{suffix}"
                fill(full, expert, "down", targets[full][expert], tensor)
            else:
                full = f"{prefix}.w13_{suffix}"
                rows = targets[full].shape[1] // 2
                gate = projection == "gate_proj"
                fill(full, expert, "gate" if gate else "up", targets[full][expert, slice(0, rows) if gate else slice(rows, 2 * rows)], tensor)
        elif fused_key is not None:
            prefix = f"model.layers.{fused_key['layer']}.mlp.experts"
            if fused_key["proj"] == "down_proj":
                full = f"{prefix}.w2_weight"
                for expert, matrix in enumerate(tensor):
                    fill(full, expert, "down", targets[full][expert], matrix)
            else:
                full = f"{prefix}.w13_weight"
                rows = targets[full].shape[1] // 2
                for expert, matrix in enumerate(tensor):
                    target = targets[full][expert]
                    fill(full, expert, "gate", target[:rows], matrix[:rows])
                    fill(full, expert, "up", target[rows:], matrix[rows:])
        else:
            if name not in targets:
                raise ValueError(f"checkpoint key {checkpoint_name!r} has no parameter {name!r}")
            fill(name, -1, "whole", targets[name], tensor)
    if pending:
        raise ValueError(f"{len(pending)} FP8 experts lack a weight or scale, e.g. {sorted(pending)[:4]}")
    missing = expected - loaded
    if missing:
        raise ValueError(f"{len(missing)} backbone parameter slots were not loaded, e.g. {sorted(missing)[:8]}")
