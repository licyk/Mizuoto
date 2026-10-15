"""Byte-level BPE tokenizers (Qwen2 family), from a model's files with the ``tokenizers`` library.

A model folder has either a ``tokenizer.json`` (used as it is), or ``vocab.json`` + ``merges.txt`` +
``tokenizer_config.json``. The latter become what transformers' ``Qwen2TokenizerFast`` builds from them
(``convert_slow_tokenizer.Qwen2Converter``): the same normaliser, pre-tokeniser, decoder and added
tokens, so the ids are the same. Used by Qwen3-TTS, Fish Speech, OmniVoice, CosyVoice 2/3 and
IndexTTS's emotion model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QWEN2_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
MISTRAL_FIX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
"""The split transformers 4.57.3–4.x substitutes when a tokenizer is loaded with ``fix_mistral_regex=True`` and
its model's ``config.json`` names a transformers version above 4.57.2 (``tokenization_utils_base``): Qwen3-TTS's
processor does both, so its text is split this way (no English contractions rule)."""


class BpeTokenizer:
    """Encode and decode text with a ``tokenizers.Tokenizer``; special tokens in the text are recognised."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        tid = self.tokenizer.token_to_id(token)
        if tid is None:
            raise KeyError(f"{token!r} is not in the vocabulary")
        return tid

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)


def load_bpe(folder: Path, mistral_fix: bool = False) -> BpeTokenizer:
    """The tokenizer of a model folder (``tokenizer.json``, else Qwen2 vocab + merges).

    ``mistral_fix`` splits text with ``MISTRAL_FIX_PATTERN``, as transformers does for a model loaded with
    ``fix_mistral_regex=True`` whose config names a transformers version in (4.57.2, 5.0).
    """
    from tokenizers import Regex, Tokenizer, pre_tokenizers

    folder = Path(folder)
    if (folder / "tokenizer.json").is_file():
        tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    else:
        tok = qwen2_from_files(folder / "vocab.json", folder / "merges.txt", folder / "tokenizer_config.json")
    if mistral_fix:
        tok.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore[assignment]
            [pre_tokenizers.Split(Regex(MISTRAL_FIX_PATTERN), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
        )
    return BpeTokenizer(tok)


def qwen2_from_files(vocab_path: Path, merges_path: Path, config_path: Path | None = None) -> Any:
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import BPE

    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    merges = read_merges(merges_path)
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="", fuse_unk=False, byte_fallback=False))
    tok.normalizer = normalizers.NFC()  # type: ignore[assignment]
    tok.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore[assignment]
        [
            pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()  # type: ignore[assignment]
    tok.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore[assignment]
    if config_path is not None and Path(config_path).is_file():
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        added = sorted(((int(i), t) for i, t in (config.get("added_tokens_decoder") or {}).items()), key=lambda x: x[0])
        for tid, t in added:
            token = AddedToken(
                t["content"],
                single_word=t.get("single_word", False),
                lstrip=t.get("lstrip", False),
                rstrip=t.get("rstrip", False),
                normalized=t.get("normalized", False),
                special=t.get("special", True),
            )
            if t.get("special", True):
                tok.add_special_tokens([token])
            else:
                tok.add_tokens([token])
            got = tok.token_to_id(t["content"])
            if got != tid:
                raise ValueError(f"Added token {t['content']!r} got id {got}, the model expects {tid}")
    return tok


def read_merges(path: Path) -> list[tuple[str, str]]:
    """``merges.txt`` as transformers reads it: the ``#version`` line skipped, blank lines ignored."""
    merges: list[tuple[str, str]] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        line = line.strip()
        if not line or (i == 0 and line.startswith("#version")):
            continue
        a, _, b = line.partition(" ")
        merges.append((a, b))
    return merges
