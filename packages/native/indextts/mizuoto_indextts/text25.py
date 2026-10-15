"""IndexTTS-2.5's text front end (index-tts ``infer_v2_5.py``, ``utils/tokenizer.py``, ``utils/ja_g2p.py``).

The text is mapped by the punctuation table, normalised (Chinese and English by ``text.TextNormalizer``; Japanese
and Spanish would go through NeMo, which upstream's environment lacks, so they pass unchanged), lower-cased
(upper-cased for Spanish), pronunciation annotations become special tokens, Japanese is spaced into words by
fugashi (the optional ``ja`` extra), and the text is split to the GPT's token budget. Tokens come from IndexTTS's
tiktoken vocabulary, each segment prefixed with ``<|lang|>`` and followed by the stop id.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from mizuoto_indextts.text import CHAR_REP_MAP, TextNormalizer, _replace_all

VOCAB = "multilingual_zh_ja_yue_char_del.tiktoken"
PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
"""Whisper's (GPT-2's) split."""

LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
]  # fmt: skip
"""``LANGUAGES``' keys, in order: their index is the language embedding's row."""
LANGUAGE_DICT = {lang: i for i, lang in enumerate(LANGUAGES)}
AUDIO_EVENTS = ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause"]
EMOTION_TOKENS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]
TTS_VOCAL_TOKENS = ["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H", *(f"TTS/SP{i:02d}" for i in range(1, 14))]
PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
SPLIT_PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")


def lang_to_token(lang: str) -> int:
    lang = lang.lower()
    return LANGUAGE_DICT[lang if lang in LANGUAGE_DICT else "common"]


def encoding(model_dir: Path, num_languages: int = 99) -> Any:
    """``get_encoding("multilingual_zh_ja_yue_char_del")``: the ranks of the model's file and upstream's special tokens."""
    import tiktoken

    path = Path(model_dir) / VOCAB
    with path.open(encoding="utf-8") as f:
        ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES[:num_languages]],
        *[f"<|{e}|>" for e in AUDIO_EVENTS],
        *[f"<|{e}|>" for e in EMOTION_TOKENS],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
        *[f"<|{t}|>" for t in TTS_VOCAL_TOKENS],
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = {}
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
    return tiktoken.Encoding(name=path.name, explicit_n_vocab=n_vocab, pat_str=PATTERN, mergeable_ranks=ranks, special_tokens=special_tokens)


def is_kana(s: str) -> bool:
    return bool(re.fullmatch(r"^[぀-ゟ]+$", s) or re.fullmatch(r"^[゠-ヿ]+$", s))


def apply_pronunciation_annotations(text: str) -> str:
    def replace(m: re.Match[str]) -> str:
        word, pronunciation = m.group(1), m.group(2).upper()
        if is_kana(pronunciation):
            return f" {pronunciation} "
        token = "SPECIAL_TOKEN_2" if re.search(r"[一-鿿]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pronunciation}<|{token}|>"

    return PRONUNCIATION_PATTERN.sub(replace, text)


class JapaneseSegmenter:
    """``JapaneseG2PProcessor(g2p_ratio=0)``: each run of text spaced into fugashi's words (no reading is substituted)."""

    def __init__(self) -> None:
        try:
            import fugashi
        except ImportError as exc:
            raise ImportError("Japanese text for IndexTTS-2.5 needs fugashi and unidic-lite (pip install 'mizuoto-indextts[ja]')") from exc
        self.tagger = fugashi.Tagger()

    def process(self, text: str) -> str:
        parts = re.split(r"( +)", text)
        return "".join(" ".join(t.surface for t in self.tagger(p)) if p.strip() else p for p in parts)


class Frontend25:
    def __init__(self, model_dir: Path, normalizer: TextNormalizer) -> None:
        self.encoding = encoding(model_dir)
        self.normalizer = normalizer
        self._ja: JapaneseSegmenter | None = None

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def prepare(self, text: str, lang: str, normalize: bool = True) -> str:
        text = _replace_all(text, CHAR_REP_MAP)
        code = lang.lower()
        if normalize and code in ("zh", "zhen", "en"):
            text = self.normalizer.normalize(text)
        if code in ("ja", "zh", "zhen", "en"):
            text = text.lower()
        if code == "es":
            text = text.upper()
        text = apply_pronunciation_annotations(text)
        if code == "ja":
            if self._ja is None:
                self._ja = JapaneseSegmenter()
            text = self._ja.process(text)
        return re.sub(r"<\|([^|]+)\|>", lambda m: f"<|{m.group(1).upper()}|>", text)

    def split(self, text: str, max_tokens: int, lang_prefix: str, capacity: int) -> list[str]:
        """``split_text_by_tokens``."""
        budget = max(1, min(max_tokens, capacity - 2) - len(self.encode(lang_prefix)))
        if len(self.encode(text)) <= budget:
            return [text]
        pieces: list[tuple[str, bool]] = []
        pos = 0
        for m in SPLIT_PROTECTED_PATTERN.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos : m.start()], False))
            pieces.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))
        chunks: list[str] = []
        for piece, atomic in pieces:
            if atomic:
                chunks.append(piece)
                continue
            for part in re.split(r"(?<=[，。！？、；：,\.!\?;:\n])", piece):
                if not part:
                    continue
                if len(self.encode(part)) <= budget:
                    chunks.append(part)
                    continue
                current = ""
                for ch in part:
                    if current and len(self.encode(current + ch)) > budget:
                        chunks.append(current)
                        current = ch
                    else:
                        current += ch
                if current:
                    chunks.append(current)
        segments: list[str] = []
        current = ""
        for chunk in chunks:
            if current and len(self.encode(current + chunk)) > budget:
                segments.append(current)
                current = chunk
            else:
                current += chunk
        if current:
            segments.append(current)
        return segments or [text]
