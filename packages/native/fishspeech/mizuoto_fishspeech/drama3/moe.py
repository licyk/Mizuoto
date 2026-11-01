"""The backbone's mixture of experts (upstream ``fish_omni/moe.py``): top-k routed experts plus a sigmoid-gated shared
expert.

Each layer's routed experts are stacked, ``w13`` [E, 2I, H] (gate rows, then up rows) and ``w2`` [E, H, I]. The released
checkpoint stores them in block FP8 (``float8_e4m3fn`` with float32 scales per 128 × 128 block); they stay in FP8 here
and the experts a call uses are dequantized to the compute dtype as it runs, as upstream does wherever its FP8 kernel
does not run. The values are those of upstream's other path, which dequantizes them all at load, so either gives these
numbers; holding them in FP8 halves their memory (32 GB in all, against 64 GB in bfloat16 and 129 GB in float32).
Upstream's W8A8 kernel (Triton, SM89 and newer), which also quantizes the activations, is not ported: its results differ.

Upstream computes a call's experts in one batched matmul when it has at most 64 (token, expert) pairs (a decoding step)
and expert by expert otherwise (prefill); both are kept, since they give different bits. The experts stream by the row
(``mizuoto.runtime.rows``): when the layer's unit is not resident, a call copies only the experts its tokens use.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.rows import source
from mizuoto_fishspeech.drama3.config import FP8_BLOCK, TextConfig

FP8 = torch.float8_e4m3fn
GATHERED_MAX_ROWS = 64
"""Up to this many (token, expert) pairs run every expert in one batched matmul; more loop over the experts hit."""
FLOAT16_MAX = torch.finfo(torch.float16).max


def dequantize_fp8_blocks(weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Block-FP8 [..., out, in] with float32 scales [..., out/128, in/128] → ``dtype``."""
    *lead, rows, cols = weight.shape
    row_blocks, col_blocks = scale_inv.shape[-2:]
    blocks = weight.view(*lead, row_blocks, FP8_BLOCK, col_blocks, FP8_BLOCK)
    scaled = blocks.to(torch.float32) * scale_inv[..., :, None, :, None]
    return scaled.view(*lead, rows, cols).to(dtype)


def select_fp8(weight: torch.Tensor, index: torch.Tensor | int) -> torch.Tensor:
    """``weight[index]`` for a float8 tensor (advanced indexing has no float8 kernel)."""
    return weight.view(torch.uint8)[index].view(weight.dtype)


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.gate_proj = ops.Linear(hidden, intermediate, bias=False)
        self.up_proj = ops.Linear(hidden, intermediate, bias=False)
        self.down_proj = ops.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.Module):
    """A layer's routed experts, stacked: ``w13_weight`` [E, 2I, H] (gate then up), ``w2_weight`` [E, H, I], and with
    ``fp8`` their block scales ``w13_scale_inv`` [E, 2I/128, H/128], ``w2_scale_inv`` [E, H/128, I/128]. Rows stream
    (``runtime.rows``)."""

    STREAM_ROWS = True

    def __init__(self, experts: int, hidden: int, intermediate: int, fp8: bool) -> None:
        super().__init__()
        self.num_experts, self.hidden, self.intermediate, self.fp8 = experts, hidden, intermediate, fp8
        dtype = FP8 if fp8 else None
        if fp8 and (hidden % FP8_BLOCK or intermediate % FP8_BLOCK):
            raise ValueError(f"block-FP8 experts need hidden and intermediate sizes divisible by {FP8_BLOCK}; got {hidden} and {intermediate}")
        # float8 tensors cannot require gradients; dense ones are made as nn.Module makes parameters
        grad = not fp8
        self.w13_weight = nn.Parameter(torch.empty((experts, 2 * intermediate, hidden), dtype=dtype), requires_grad=grad)
        self.w2_weight = nn.Parameter(torch.empty((experts, hidden, intermediate), dtype=dtype), requires_grad=grad)
        if fp8:
            self.w13_scale_inv = nn.Parameter(torch.empty((experts, 2 * intermediate // FP8_BLOCK, hidden // FP8_BLOCK)), requires_grad=False)
            self.w2_scale_inv = nn.Parameter(torch.empty((experts, hidden // FP8_BLOCK, intermediate // FP8_BLOCK)), requires_grad=False)

    def expert_bytes(self) -> int:
        """One expert's rows of every stack."""
        return sum(p[0].numel() * p.element_size() for p in self.parameters())

    def row_reserve(self) -> int:
        """Streamed, a decoding step holds at most ``GATHERED_MAX_ROWS`` experts at once; prefill holds one at a time."""
        return min(GATHERED_MAX_ROWS, self.num_experts) * self.expert_bytes()

    def _names(self) -> tuple[str, ...]:
        return ("w13_weight", "w2_weight", "w13_scale_inv", "w2_scale_inv") if self.fp8 else ("w13_weight", "w2_weight")

    def _stacks(self, experts: list[int], device: torch.device) -> tuple[torch.Tensor, ...]:
        """The stacks of ``experts`` (in that order) on ``device``, from the streaming handle's host copies."""
        rows = source(self)
        assert rows is not None
        out = []
        for name in self._names():
            taken = [rows.take(self, name, e) for e in experts]
            if taken[0].dtype == FP8:  # stack the bytes: a float8 stack is a byte stack
                out.append(torch.stack([t.view(torch.uint8) for t in taken]).view(FP8))
            else:
                out.append(torch.stack(taken))
        return tuple(t.to(device) for t in out)

    def _weights(self, stacks: tuple[torch.Tensor, ...], index: torch.Tensor | int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate-up and down weights of the experts ``index`` selects in ``stacks``, in ``dtype``."""
        if not self.fp8:
            w13, w2 = stacks
            return w13[index].to(dtype), w2[index].to(dtype)
        w13, w2, s13, s2 = stacks
        return dequantize_fp8_blocks(select_fp8(w13, index), s13[index], dtype), dequantize_fp8_blocks(select_fp8(w2, index), s2[index], dtype)

    def _own(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, name) for name in self._names())

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """x [N, H], topk_ids / topk_weights [N, k] → the weighted sum of the experts' outputs [N, H]."""
        if topk_ids.numel() <= GATHERED_MAX_ROWS:
            return self.forward_gathered(x, topk_ids, topk_weights)
        return self.forward_grouped(x, topk_ids, topk_weights)

    def forward_gathered(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """One batched matmul over every (token, expert) pair."""
        tokens, top_k = topk_ids.shape
        flat = topk_ids.reshape(-1)
        if source(self) is None:
            stacks, index = self._own(), flat
        else:  # streamed: the experts used, in a stack of their own, indexed by their place in it
            used = sorted(set(flat.tolist()))
            stacks = self._stacks(used, x.device)
            remap = torch.zeros(self.num_experts, dtype=flat.dtype, device=flat.device)
            remap[torch.tensor(used, device=flat.device)] = torch.arange(len(used), dtype=flat.dtype, device=flat.device)
            index = remap[flat]
        w13, w2 = self._weights(stacks, index, x.dtype)
        rows = x.repeat_interleave(top_k, dim=0).unsqueeze(-1)  # [M, H, 1]
        gate, up = torch.bmm(w13, rows).squeeze(-1).chunk(2, dim=-1)
        hidden = (F.silu(gate) * up).unsqueeze(-1)
        out = torch.bmm(w2, hidden).squeeze(-1)  # [M, H]
        out = out.float() * topk_weights.reshape(-1, 1).float()
        return out.view(tokens, top_k, self.hidden).sum(dim=1).to(x.dtype)

    def forward_grouped(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """Expert by expert, each on its own tokens, summed into float32 in the experts' order."""
        tokens, top_k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        token_rows = order // top_k
        pair_weights = topk_weights.reshape(-1)[order]
        counts = torch.bincount(flat_ids, minlength=self.num_experts).tolist()
        out = torch.zeros(tokens, self.hidden, dtype=torch.float32, device=x.device)
        streamed = source(self) is not None
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            rows = token_rows[start : start + count]
            weights = pair_weights[start : start + count]
            start += count
            stacks, index = (self._stacks([expert], x.device), 0) if streamed else (self._own(), expert)
            w13, w2 = self._weights(stacks, index, x.dtype)
            gate, up = F.linear(x[rows], w13).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, w2)
            out.index_add_(0, rows, expert_out.float() * weights[:, None].float())
        return out.to(x.dtype)


class SparseMoeBlock(nn.Module):
    """Top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, c: TextConfig, fp8: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        hidden = c.hidden_size
        self.top_k = c.num_experts_per_tok
        self.norm_topk_prob = c.norm_topk_prob
        self.gate = ops.Linear(hidden, c.num_experts, bias=False)
        self.experts = Experts(c.num_experts, hidden, c.moe_intermediate_size, fp8)
        self.shared_expert = DenseMLP(hidden, c.shared_expert_intermediate_size, ops)
        self.shared_expert_gate = ops.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        probabilities = torch.softmax(self.gate(flat).float(), dim=-1)
        topk_weights, topk_ids = probabilities.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        out = self.experts(flat, topk_ids, topk_weights.to(flat.dtype))
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (out + shared_gate * self.shared_expert(flat)).view_as(x)


def clamp_to_float16(tensor: torch.Tensor) -> torch.Tensor:
    """Saturate what float16 cannot hold (upstream's ``clamp_to_float16``, for a float16 model)."""
    return tensor.clamp(-FLOAT16_MAX, FLOAT16_MAX) if bool((tensor.abs() > FLOAT16_MAX).any()) else tensor
