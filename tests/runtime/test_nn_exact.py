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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_with_precomputed_tables_matches_the_original(dtype):
    g = torch.Generator().manual_seed(0)
    rotary = RotaryEmbedding(64)
    cases = [
        ((2, 16, 37, 64), 37),  # every head channel turns (F5-TTS's DiT)
        ((2, 37, 1024), 37),  # the first 64 of 1024 channels turn (CosyVoice 3, GPT-SoVITS v4/v5)
        ((1, 16, 30, 64), 31),  # a slice of the table (the last positions)
        ((3, 2, 5, 64), 5),
    ]
    for shape, table in cases:
        t = torch.randn(shape, generator=g).to(dtype)
        freqs = rotary.forward_from_seq_len(table)
        expected = _upstream_rotary(t, freqs)
        assert torch.equal(apply_rotary(t, rope_tables(freqs)), expected), shape
        assert torch.equal(apply_rotary(t, freqs), expected), shape
        assert torch.equal(apply_rotary(t, rope_tables(freqs), 0.5), _upstream_rotary(t, freqs, 0.5)), shape


def test_gpt2s_in_place_gelu_matches_the_original():
    gpt = pytest.importorskip("mizuoto_indextts.gpt")
    import math

    def upstream(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((3, 1, 5120), (1, 50, 5120), (7,)):
            x = (torch.randn(shape, generator=g) * 4).to(dtype)
            assert torch.equal(gpt.gelu_new(x), upstream(x)), (dtype, shape)


def test_fish_codecs_in_place_snake_matches_the_original():
    codec = pytest.importorskip("mizuoto_fishspeech.codec")

    def upstream(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        return (x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)).reshape(shape)

    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((1, 64, 3001), (2, 16, 7)):
            x = (torch.randn(shape, generator=g) * 3).to(dtype)
            alpha = (torch.rand(1, shape[1], 1, generator=g) * 2).to(dtype)
            with torch.no_grad():
                assert torch.equal(codec.snake(x, alpha), upstream(x, alpha)), (dtype, shape)
            assert torch.equal(codec.snake(x, alpha), upstream(x, alpha)), (dtype, shape)
