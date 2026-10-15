"""Audio signal processing, bit-exact with the libraries upstream engines call.

- ``mel``: librosa's and torchaudio's mel filterbanks, torchaudio's spectrogram and mel spectrogram,
  Whisper's log-mel and Matcha-TTS's mel.
- ``kaldi``: torchaudio's Kaldi-compatible ``fbank``.
- ``resample``: torchaudio's sinc resampling and soxr, as librosa calls it.

No module-level caches: every filterbank depends on all of its parameters, so a caller keeps its own,
keyed by all of them.
"""
