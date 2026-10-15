"""GPT-SoVITS's chinese-roberta (``mizuoto.nn.transformer.bert``) on a tiny random model: its SDPA path (an attention
site of ``mizuoto.runtime.accel``) and its eager path agree. The engine's tiny tests stand a linear layer in for it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mizuoto.nn.transformer.bert import BertConfig, BertModel  # noqa: E402
from mizuoto.runtime import ops  # noqa: E402


def test_sdpa_and_eager_attention_agree_with_and_without_padding():
    config = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, max_position_embeddings=32)
    g = torch.Generator().manual_seed(0)
    sdpa = BertModel(config, attention="sdpa")
    ops.init_random(sdpa, g)
    eager = BertModel(config, attention="eager")
    eager.load_state_dict(sdpa.state_dict())
    ids = torch.randint(1, 50, (2, 9), generator=g)
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, 6:] = 0
    for attention_mask in (None, mask):
        a = sdpa.hidden_states(ids, attention_mask)[-1]
        b = eager.hidden_states(ids, attention_mask)[-1]
        assert torch.allclose(a[0], b[0], atol=1e-5)
        assert torch.allclose(a[1, :6], b[1, :6], atol=1e-5)
