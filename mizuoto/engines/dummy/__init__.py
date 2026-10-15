"""The dummy engine, native: every capability, no model.

The engine the host's tests run on. It sounds as v1's dummy engine worker did for the same seed.
- **Randomness:** v1's worker seeded Python's global ``random``; this one draws from
  ``ctx.rng``, a ``random.Random`` seeded identically, so the streams match.
- **Imports:** numpy only, so the host's tests need no torch.
- **The ``crash`` failure option:** there is no process to crash, so it raises ``worker_failed``, which
  is what a crashed worker reports.
"""

from mizuoto.engines.dummy.engine import DummyEngine

__all__ = ["DummyEngine"]
