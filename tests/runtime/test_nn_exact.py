"""Shared layers rewritten for speed give what the code they replace gave, to the bit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mizuoto.nn.transformer.dit import RotaryEmbedding, apply_rotary, rope_tables  # noqa: E402


def _upstream_rotary(t: torch.Tensor, freqs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """x-transformers' ``apply_rotary_pos_emb`` as the port had it before (the reference)."""
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    x = t.reshape(*t.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    rotated = torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)
    t = (t * freqs.cos() * scale) + (rotated * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_with_precomputed_tables_matches_the_original(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g = torch.Generator().manual_seed(0)
    rotary = RotaryEmbedding(64)
    cases = [
        ((2, 16, 37, 64), 37),  # every head channel turns (F5-TTS's DiT)
        ((2, 37, 1024), 37),  # the first 64 of 1024 channels turn (CosyVoice 3, GPT-SoVITS v4/v5)
        ((1, 16, 30, 64), 31),  # a slice of the table (the last positions)
        ((3, 2, 5, 64), 5),
    ]
    for shape, table in cases:
        t = torch.randn(shape, generator=g).to(device, dtype)
        freqs = rotary.forward_from_seq_len(table).to(device)
        expected = _upstream_rotary(t, freqs)
        assert torch.equal(apply_rotary(t, rope_tables(freqs)), expected), shape
        assert torch.equal(apply_rotary(t, freqs), expected), shape
        assert torch.equal(apply_rotary(t, rope_tables(freqs), 0.5), _upstream_rotary(t, freqs, 0.5)), shape
