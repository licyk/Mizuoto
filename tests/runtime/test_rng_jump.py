"""mizuoto.training.rng.advance: torch's CPU generator jumped ahead gives the state real draws give."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mizuoto.training import rng  # noqa: E402


@pytest.mark.parametrize(
    ("seed", "before", "draws"),
    [(5, 0, 1), (5, 0, 623), (5, 0, 624), (5, 0, 625), (5, 0, 1248), (7, 3, 1000), (7, 700, 12345), (11, 5, 1_000_017), (1234, 99, 20_000_003)],
)
def test_advance_equals_drawing(seed, before, draws):
    jumped, drawn = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    for g in (jumped, drawn):
        rng._draw(g, before)
        torch.randn(3, generator=g)  # a cached normal sample stays as it is
    rng.advance(jumped, draws)
    rng._draw(drawn, draws)
    assert torch.equal(jumped.get_state(), drawn.get_state())
    assert torch.equal(torch.rand(7, generator=jumped), torch.rand(7, generator=drawn))


def test_the_default_generator_and_nothing():
    torch.manual_seed(9)
    expected = torch.get_rng_state()
    rng.advance(torch.default_generator, 0)
    assert torch.equal(torch.get_rng_state(), expected)
    rng.advance(torch.default_generator, 100_000)
    jumped = torch.get_rng_state()
    torch.manual_seed(9)
    rng._draw(torch.default_generator, 100_000)
    assert torch.equal(jumped, torch.get_rng_state())
