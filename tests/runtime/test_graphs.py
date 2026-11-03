"""mizuoto.runtime.graphs on an NVIDIA GPU: what a failed capture leaves behind."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")


def test_a_failed_capture_leaves_the_default_generator_drawing_where_it_was():
    """A capture whose step reads a value back fails; the device's default generator, which every capture registers,
    then still draws (``torch.randn`` without a generator, dropout) and from the same state: ending the failed capture
    had left it in capture mode, where every draw raised."""
    from mizuoto.runtime.graphs import Graphs

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.ones(4, device=device)
    state = torch.cuda.get_rng_state(device)
    want = torch.randn(5, device=device)
    torch.cuda.set_rng_state(state, device)
    graphs = Graphs(device)
    with torch.cuda.device(device), pytest.raises(Exception):  # torch's error type differs by version
        graphs.capture(lambda: x.add_(1).sum().item(), [x])
    assert torch.equal(torch.cuda.get_rng_state(device), state)
    assert torch.equal(torch.randn(5, device=device), want)
    assert torch.nn.functional.dropout(torch.ones(64, device=device), 0.5).shape == (64,)
