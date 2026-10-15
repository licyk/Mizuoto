"""Fixed-grid ODE integration and the time grids flow-matching models sample on.

``odeint_fixed`` is ``torchdiffeq.odeint`` with a fixed-grid solver (``euler`` or ``midpoint``) on the grid
``t``: the same operations in the same order, so the trajectory matches. ``epss_timesteps`` and ``sway``
are F5-TTS's grids (EPSS for low step counts, sway sampling); CosyVoice and IndexTTS use the cosine
grid of ``cosine_timesteps``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Fn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

EPSS = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
}


def epss_timesteps(n: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """F5-TTS's Empirically Pruned Step Sampling grid for ``n`` steps, else ``linspace(0, 1, n + 1)``."""
    t = EPSS.get(n)
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return (1 / 32) * torch.tensor(t, device=device, dtype=dtype)


def sway(t: torch.Tensor, coef: float) -> torch.Tensor:
    """Sway sampling: more steps early (``coef`` < 0)."""
    return t + coef * (torch.cos(torch.pi / 2 * t) - 1 + t)


def odeint_fixed(fn: Fn, y0: torch.Tensor, t: torch.Tensor, method: str = "euler", step: Callable[[int, int], None] | None = None) -> torch.Tensor:
    """The state at ``t[-1]``, integrating ``dy/dt = fn(t, y)`` from ``y0`` at ``t[0]`` over the grid ``t``.

    ``step(i, n)`` is called before each step (progress, cancellation).
    """
    if method not in ("euler", "midpoint"):
        raise ValueError(f"Unknown ODE method: {method}")
    y = y0
    n = len(t) - 1
    for i, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
        if step is not None:
            step(i, n)
        dt = t1 - t0
        if method == "euler":
            dy = dt * fn(t0, y)
        else:
            half_dt = 0.5 * dt
            f0 = fn(t0, y)
            y_mid = y + f0 * half_dt
            dy = dt * fn(t0 + half_dt, y_mid)
        y = y + dy
    return y
