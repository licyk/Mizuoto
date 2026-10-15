"""CosyVoice's supervised speech tokenizers in torch, from their ONNX files (s3tokenizer 0.3.0, Apache-2.0).

- **v1** (CosyVoice 1, 50 Hz): a Whisper-style encoder (sinusoidal positions, 6 blocks) and a 4096-entry
  Euclidean codebook over L2-normalised frames.
- **v2** (CosyVoice 2, 25 Hz): blocks with rotary attention plus an FSMN memory (a depthwise convolution
  over the values), quantized by FSQ: 8 dimensions at 3 levels (6561 codes).
- **v3** (CosyVoice 3): v2's design with 12 blocks.

The weights come from the ONNX graph's initializers and constants (``mizuoto.runtime.convert.onnx``), named
as s3tokenizer names them. On CosyVoice's reference clips the codes equal onnxruntime's, code for code.
Input is Whisper's 128-bin log-mel at 16 kHz (``mizuoto.nn.dsp.mel.whisper_log_mel``), at most 30 s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

if TYPE_CHECKING:
    from mizuoto.runtime.convert.onnx import OnnxGraph

N_MELS = 128
N_STATE = 1280
N_HEAD = 20


def non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device).unsqueeze(0).expand(lengths.size(0), max_len)
    return ~(seq >= lengths.unsqueeze(-1))


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - mask.to(dtype)) * -1.0e10


class LayerNorm(nn.LayerNorm):
    """Computed in float32 (Whisper's)."""

    reset_parameters = default_ops.LayerNorm.reset_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type(x.dtype)


class Attention(nn.Module):
    def __init__(self, n_state: int, n_head: int, ops: Any, fsmn: bool = False, kernel_size: int = 31) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = ops.Linear(n_state, n_state)
        self.key = ops.Linear(n_state, n_state, bias=False)
        self.value = ops.Linear(n_state, n_state)
        self.out = ops.Linear(n_state, n_state)
        self.fsmn = fsmn
        if fsmn:
            self.fsmn_block = ops.Conv1d(n_state, n_state, kernel_size, stride=1, padding=0, groups=n_state, bias=False)
            self.left_padding = (kernel_size - 1) // 2
            self.right_padding = kernel_size - 1 - self.left_padding

    def _fsmn(self, v: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = v.size()
        inputs = v.view(b, t, -1)
        if mask_pad is not None and mask_pad.size(2) > 0:
            inputs = inputs * mask_pad
        x = F.pad(inputs.transpose(1, 2), (self.left_padding, self.right_padding), "constant", 0.0)
        x = self.fsmn_block(x).transpose(1, 2)
        x += inputs
        return x * mask_pad

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mask_pad: torch.Tensor | None = None, freqs: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        d = q.shape[-1]
        scale = (d // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)
        memory = None
        if self.fsmn:
            assert freqs is not None and mask_pad is not None
            q, k = _rotate(q, freqs), _rotate(k, freqs)
            memory = self._fsmn(v, mask_pad)
        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1) * scale
        qk = (q @ k) + mask
        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        out = self.out((w @ v).permute(0, 2, 1, 3).flatten(start_dim=2))
        return out + memory if memory is not None else out


def _rotate(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    d = x.shape[-1]
    rotated = torch.cat((-x[:, :, :, d // 2 :], x[:, :, :, : d // 2]), dim=-1)
    return x * cos + rotated * sin


class Block(nn.Module):
    def __init__(self, n_state: int, n_head: int, ops: Any, fsmn: bool) -> None:
        super().__init__()
        self.attn = Attention(n_state, n_head, ops, fsmn=fsmn)
        self.attn_ln = LayerNorm(n_state, eps=1e-5)
        self.mlp = nn.Sequential(ops.Linear(n_state, n_state * 4), nn.GELU(), ops.Linear(n_state * 4, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mask_pad: torch.Tensor | None = None, freqs: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask, mask_pad, freqs)
        return x + self.mlp(self.mlp_ln(x))


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    increment = np.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-increment * torch.arange(channels // 2))
    scaled = torch.arange(length)[:, np.newaxis] * inv[np.newaxis, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def rotary(dim: int = 64, end: int = 2048, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """s3tokenizer's ``precompute_freqs_cis`` as cos and sin: (end, dim) each."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float()
    cis = torch.polar(torch.ones_like(freqs), freqs)
    cis = torch.cat((cis, cis), dim=-1)
    real = torch.view_as_real(cis)
    return real[:, :, 0], real[:, :, 1]


class Encoder(nn.Module):
    def __init__(self, version: int, n_layer: int, stride: int, ops: Any, n_mels: int = N_MELS, n_state: int = N_STATE, n_head: int = N_HEAD) -> None:
        super().__init__()
        self.version = version
        self.stride = stride
        self.conv1 = ops.Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ops.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        if version == 1:
            self.register_buffer("positional_embedding", sinusoids(1500, n_state))
        else:
            cos, sin = rotary(n_state // n_head)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList(Block(n_state, n_head, ops, fsmn=version > 1) for _ in range(n_layer))

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.version == 1:
            mask = non_pad_mask(x_len).unsqueeze(1)
            x = F.gelu(self.conv1(x * mask))
            x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
            mask = non_pad_mask(x_len).unsqueeze(1)
            x = F.gelu(self.conv2(x * mask))
            x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
            mask = non_pad_mask(x_len).unsqueeze(1)
            x = x.permute(0, 2, 1)
            bias = mask_to_bias(mask, x.dtype)
            x = (x + self.get_buffer("positional_embedding")[: x.shape[1], :]).to(x.dtype)
            for block in self.blocks:
                x = block(x, bias.unsqueeze(1))
            return x, x_len
        t = x.shape[-1]
        mask = non_pad_mask(x_len, t).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        slen = (t + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = non_pad_mask(x_len, slen).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        # v2 divides by the first stride here, v3 by 2: both are 2
        slen = (slen + 2 - 1 * (3 - 1) - 1) // (self.stride if self.version == 2 else 2) + 1
        mask = non_pad_mask(x_len, slen).unsqueeze(1)
        x = x.permute(0, 2, 1)
        mask_pad = mask.transpose(1, 2)
        bias = mask_to_bias(mask, x.dtype)
        n = x.size(1)
        cos = self.get_buffer("cos")[:n].to(x.device).unsqueeze(0).unsqueeze(2).to(x.dtype)
        sin = self.get_buffer("sin")[:n].to(x.device).unsqueeze(0).unsqueeze(2).to(x.dtype)
        for block in self.blocks:
            x = block(x, bias.unsqueeze(1), mask_pad, (cos, sin))
        return x, x_len


class Codebook(nn.Module):
    """v1: the nearest of 4096 vectors to the L2-normalised frame."""

    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self.register_buffer("embed", torch.zeros(size, dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x.float(), p=2, dim=-1)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        embed = self.get_buffer("embed").t().to(flat.dtype)
        dist = -(flat.pow(2).sum(1, keepdim=True) - 2 * flat @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices.view(*shape[:-1])


class FSQ(nn.Module):
    """v2/v3: 8 dimensions, each tanh-squashed and rounded to 3 levels."""

    def __init__(self, dim: int, ops: Any, level: int = 3) -> None:
        super().__init__()
        self.project_down = ops.Linear(dim, 8)
        self.level = level

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = self.project_down(x.reshape(-1, shape[-1])).float()
        h = (h.tanh() * 0.9990000128746033).round() + 1
        powers = torch.pow(self.level, torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        return torch.sum(h * powers.unsqueeze(0), dim=-1).reshape(shape[0], shape[1]).int()


class _Quantizer(nn.Module):
    def __init__(self, codebook: nn.Module) -> None:
        super().__init__()
        self._codebook = codebook


class S3Tokenizer(nn.Module):
    """``version`` 1, 2 or 3; ``quantize(mel, mel_len)`` → (codes, lengths)."""

    def __init__(
        self,
        version: int,
        ops: Any = default_ops.default,
        n_mels: int = N_MELS,
        n_state: int = N_STATE,
        n_head: int = N_HEAD,
        n_layer: int | None = None,
        codebook_size: int = 4096,
    ) -> None:
        super().__init__()
        if version not in (1, 2, 3):
            raise ValueError(f"Unknown speech tokenizer version {version}")
        self.version = version
        n_layer = n_layer if n_layer is not None else 12 if version == 3 else 6
        self.encoder = Encoder(version, n_layer=n_layer, stride=1 if version == 1 else 2, ops=ops, n_mels=n_mels, n_state=n_state, n_head=n_head)
        self.quantizer = _Quantizer(Codebook(n_state, codebook_size) if version == 1 else FSQ(n_state, ops))

    @classmethod
    def from_state(cls, version: int, state: dict[str, torch.Tensor], ops: Any = default_ops.default) -> S3Tokenizer:
        """Sized by its weights (the heads are 64 channels wide, as in every released tokenizer)."""
        conv = state["encoder.conv1.weight"]
        n_layer = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("encoder.blocks."))
        embed = state.get("quantizer._codebook.embed")
        size = int(embed.shape[0]) if embed is not None else 4096
        return cls(version, ops, n_mels=int(conv.shape[1]), n_state=int(conv.shape[0]), n_head=int(conv.shape[0]) // 64, n_layer=n_layer, codebook_size=size)

    @property
    def token_rate(self) -> int:
        return 50 if self.version == 1 else 25

    def quantize(self, mel: torch.Tensor, mel_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(mel_len.max()) > 3000:
            raise ValueError("The speech tokenizer takes at most 30 s of audio")
        hidden, code_len = self.encoder(mel, mel_len)
        codebook: Any = self.quantizer._codebook
        return codebook.encode(hidden), code_len


def state_dict_from_onnx(graph: OnnxGraph, version: int) -> dict[str, torch.Tensor]:
    """s3tokenizer's ``onnx2torch`` (v1, v2) and ``onnx2torch_v3``, over ``mizuoto.runtime.convert.onnx``."""
    return _v3(graph) if version == 3 else _v1_v2(graph)


def _v1_v2(graph: OnnxGraph) -> dict[str, torch.Tensor]:
    named = {
        "onnx::Conv_1519": "encoder.conv1.weight",
        "encoders.conv1.weight": "encoder.conv1.weight",
        "onnx::Conv_2216": "encoder.conv1.weight",
        "onnx::Conv_1520": "encoder.conv1.bias",
        "encoders.conv1.bias": "encoder.conv1.bias",
        "onnx::Conv_2217": "encoder.conv1.bias",
        "onnx::Conv_1521": "encoder.conv2.weight",
        "encoders.conv2.weight": "encoder.conv2.weight",
        "onnx::Conv_2218": "encoder.conv2.weight",
        "onnx::Conv_1522": "encoder.conv2.bias",
        "encoders.conv2.bias": "encoder.conv2.bias",
        "onnx::Conv_2219": "encoder.conv2.bias",
        "encoders.positional_embedding": "encoder.positional_embedding",
        "quantizer.project_in.bias": "quantizer._codebook.project_down.bias",
        "onnx::MatMul_2536": "quantizer._codebook.project_down.weight",
    }
    weights: dict[str, torch.Tensor] = {}
    inits = graph.initializers
    for node in graph.nodes:
        for name in node.inputs:
            if name not in inits:
                continue
            if node.op_type == "LayerNormalization" and name not in named:
                ln = node.name.replace("/LayerNormalization", "")
                weights[ln + ".weight"] = torch.from_numpy(inits[node.inputs[1]].numpy())
                weights[ln + ".bias"] = torch.from_numpy(inits[node.inputs[2]].numpy())
                continue
            target = named.get(name, node.name)
            t = torch.from_numpy(inits[name].numpy())
            weights[target] = t if t.dim() > 2 or target == "encoder.positional_embedding" else t.t()
    out: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if "quantizer" in k:
            if k == "/quantizer/rq/model/layers.0/_codebook/Pow_1":
                out["quantizer._codebook.embed"] = v
            elif "project_down" in k:
                out[k] = v
        elif "positional_embedding" in k or "conv" in k:
            out[k] = v
        else:
            if "blocks" not in k:
                raise ValueError(f"Unexpected speech tokenizer weight {k}")
            new = (
                k[1:]
                .replace("/", ".")
                .replace("MatMul", "weight")
                .replace("Add_1", "bias")
                .replace("Mul", "weight")
                .replace("Add", "bias")
                .replace("mlp.mlp", "mlp")
                .replace("fsmn_block.Conv", "fsmn_block.weight")
            )
            out[f"encoder.{new}"] = v
    return out


def _v3(graph: OnnxGraph) -> dict[str, torch.Tensor]:
    consts = graph.constants()

    def get(name: str, transpose: bool = False) -> torch.Tensor | None:
        if name in graph.initializers:
            t = torch.from_numpy(graph.initializers[name].numpy())
        elif name in consts:
            t = torch.from_numpy(consts[name].numpy())
        else:
            return None
        return t.t() if transpose and t.ndim == 2 else t

    def bias(inputs: list[str]) -> torch.Tensor | None:
        for i in inputs:
            t = get(i)
            if t is not None:
                return t
        return None

    w: dict[str, torch.Tensor | None] = {}
    for node in graph.nodes:
        name, op, inputs = node.name, node.op_type, node.inputs
        if name == "/conv1/Conv":
            w["encoder.conv1.weight"] = get(inputs[1])
            if len(inputs) > 2:
                w["encoder.conv1.bias"] = get(inputs[2])
        elif name == "/conv2/Conv":
            w["encoder.conv2.weight"] = get(inputs[1])
            if len(inputs) > 2:
                w["encoder.conv2.bias"] = get(inputs[2])
        elif name.startswith("/blocks."):
            p = f"encoder.blocks.{name.split('/')[1].split('.')[1]}"
            for ln in ("attn_ln", "mlp_ln"):
                if f"{ln}/Mul" in name and op == "Mul":
                    w[f"{p}.{ln}.weight"] = get(inputs[1])
                elif f"{ln}/Add" in name and op == "Add":
                    t = bias(inputs)
                    if t is not None and t.numel() > 1:
                        w[f"{p}.{ln}.bias"] = t
            for part in ("query", "key", "value", "out"):
                if f"attn/{part}/MatMul" in name:
                    w[f"{p}.attn.{part}.weight"] = get(inputs[1], transpose=True)
                elif f"attn/{part}/Add" in name:
                    w[f"{p}.attn.{part}.bias"] = bias(inputs)
            for part in ("0", "2"):
                if f"mlp/mlp.{part}/MatMul" in name:
                    w[f"{p}.mlp.{part}.weight"] = get(inputs[1], transpose=True)
                elif f"mlp/mlp.{part}/Add" in name:
                    w[f"{p}.mlp.{part}.bias"] = bias(inputs)
    for init in graph.initializers:
        if "fsmn_block.weight" in init:
            w[f"encoder.{init}"] = get(init)
        if "quantizer.project_in.bias" in init:
            w["quantizer._codebook.project_down.bias"] = get(init)
    for node in graph.nodes:
        if "quantizer" in node.name and "MatMul" in node.op_type:
            w["quantizer._codebook.project_down.weight"] = get(node.inputs[1], transpose=True)
    return {k: v for k, v in w.items() if v is not None}
