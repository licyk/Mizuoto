"""Fish Speech S2 Pro and Drama 3 as a native Mizuoto engine (a distribution of its own, since the licences cover the port:
the Fish Audio Research License, ``LICENSE.md``; for ``drama3``, ported from code under it, the Fish Audio Community
License, ``drama3/LICENSE.md``)."""

from mizuoto_fishspeech.engine import FishSpeechEngine
from mizuoto_fishspeech.version import VERSION

__all__ = ["VERSION", "FishSpeechEngine"]
