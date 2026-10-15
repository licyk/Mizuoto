"""Whisper's multilingual tiktoken vocabulary (openai-whisper 20231117, MIT), as CosyVoice 1 encodes text.

``get_encoding(num_languages)`` is whisper's: the byte-pair ranks of ``assets/multilingual.tiktoken`` (shipped
with Mizuoto), then the special tokens in whisper's order (end of text, start of transcript, one per language,
the task tokens, 1501 timestamps). CosyVoice encodes with ``allowed_special="all"``.
"""

from __future__ import annotations

import base64
from functools import cache
from pathlib import Path
from typing import Any

LANGUAGES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
    "yue",
]
"""whisper's ``LANGUAGES`` keys, in its order (the order fixes the language tokens' ids)."""

PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@cache
def get_encoding(name: str = "multilingual", num_languages: int = 99) -> Any:
    import tiktoken

    vocab = Path(__file__).with_name("assets") / f"{name}.tiktoken"
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in vocab.read_text(encoding="utf-8").splitlines() if line)}
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = {}
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
    return tiktoken.Encoding(name=vocab.name, explicit_n_vocab=n_vocab, pat_str=PATTERN, mergeable_ranks=ranks, special_tokens=special_tokens)


class WhisperTokenizer:
    """``encode(text)`` with every special token allowed, as CosyVoice 1's frontend calls it."""

    def __init__(self, num_languages: int = 100) -> None:
        self.encoding = get_encoding("multilingual", num_languages)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")
