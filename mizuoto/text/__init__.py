"""Text for native engines: tokenizers, normalisation, g2p.

Torch-free. ``tokenize`` turns text into the ids each engine's model was trained on. They come from
the ``tokenizers``, ``sentencepiece`` and ``tiktoken`` libraries (wheels for every platform), never
from transformers.
"""
