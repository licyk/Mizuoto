"""IndexTTS's text front end (index-tts ``utils/front.py``, ``utils/common.py``) for 1.5 and 2.

- ``tn_processor``: WeTextProcessing's ``Processor.normalize`` (tag, reorder the tokens, verbalise) with the FSTs
  this package ships (``assets/tn``), run by ``kaldifst`` (``mizuoto.text.normalize.wetext.TnProcessor``). Upstream runs these grammars through pynini on Linux
  and wetext's other FSTs on macOS and Windows; this is its Linux behaviour on every platform.
- ``TextNormalizer``: upstream's wrapper — pronunciation annotations, names, technical terms and pinyin with
  tones protected from the grammars, English contractions, the glossary, then its punctuation map.
- ``TextTokenizer``: CJK characters spaced and upper-cased, SentencePiece pieces, and ``split_segments``.

Names and pinyin found in a text are numbered in first-seen order (upstream numbers them in ``set`` order,
which varies between runs; the restored text is the same unless one name contains another).
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

ASSETS = Path(__file__).with_name("assets") / "tn"


def tn_processor(lang: str, folder: Path = ASSETS) -> Any:
    """One language's grammars as ``mizuoto.text``'s ``TnProcessor``: ``zh`` or ``en``."""
    from mizuoto.text.normalize.wetext import TnProcessor

    return TnProcessor(folder / f"{lang}_tn_tagger.fst", folder / f"{lang}_tn_verbalizer.fst", lang)


CHAR_REP_MAP = {
    "：": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": " ",
    "·": "-",
    "、": ",",
    "...": "…",
    ",,,": "…",
    "，，，": "…",
    "……": "…",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    ":": ",",
}
ZH_CHAR_REP_MAP = {"$": ".", **CHAR_REP_MAP}

PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
NAME_PATTERN = r"[一-鿿]+(?:[-·—][一-鿿]+){1,2}"
TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"
PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")


def _alpha(n: int) -> str:
    s = ""
    while True:
        s = chr(ord("a") + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            return s


def _replace_all(text: str, mapping: dict[str, str]) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in mapping))
    return pattern.sub(lambda m: mapping[m.group()], text)


class TextNormalizer:
    def __init__(self, glossary: dict[str, Any] | None = None, folder: Path = ASSETS) -> None:
        self.zh = tn_processor("zh", folder)
        self.en = tn_processor("en", folder)
        self.term_glossary: dict[str, Any] = dict(glossary or {})

    @staticmethod
    def use_chinese(s: str) -> bool:
        has_chinese = bool(re.search(r"[一-鿿]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = re.match(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$", s) is not None
        if has_chinese or not has_alpha or is_email:
            return True
        return bool(re.search(PINYIN_TONE_PATTERN, s, re.IGNORECASE))

    def apply_glossary_terms(self, text: str, lang: str) -> str:
        if not self.term_glossary:
            return text

        @lru_cache(maxsize=42)
        def pattern(term: str) -> re.Pattern[str]:
            return re.compile(re.escape(term), re.IGNORECASE)

        for term in sorted(self.term_glossary.keys(), key=len, reverse=True):
            value = self.term_glossary[term]
            replacement = value.get(lang, value.get(lang, term)) if isinstance(value, dict) else value
            text = pattern(term).sub(replacement, text)
        return text

    @staticmethod
    def _save(text: str, pattern: str, tag: str, flags: int = 0) -> tuple[str, list[str] | None]:
        found = re.findall(re.compile(pattern, flags), text)
        if not found:
            return text, None
        items = list(dict.fromkeys("".join(n) for n in found))
        for i, item in enumerate(items):
            text = text.replace(item, f"<{tag}_{chr(ord('a') + i)}>")
        return text, items

    @staticmethod
    def correct_pinyin(pinyin: str) -> str:
        if pinyin[0] not in "jqxJQX":
            return pinyin
        return re.sub(r"([jqx])[uü](n|e|an)*(\d)", r"\g<1>v\g<2>\g<3>", pinyin, flags=re.IGNORECASE).upper()

    @staticmethod
    def save_tech_terms(text: str) -> tuple[str, list[str] | None]:
        found = re.compile(TECH_TERM_PATTERN).findall(text)
        if not found:
            return text, None
        terms = sorted(set(found), key=len, reverse=True)
        for term in terms:
            text = text.replace(term, term.replace("-", "<H>"))
        return text, terms

    def normalize(self, text: str) -> str:
        placeholders: dict[str, str] = {}

        def protect(m: re.Match[str]) -> str:
            key = f"PRONPLACEHOLDER{_alpha(len(placeholders))}PRONPLACEHOLDER"
            placeholders[key] = m.group(0)
            return key

        text = PRONUNCIATION_PATTERN.sub(protect, text)
        if self.use_chinese(text):
            text = re.sub(ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            text = self.apply_glossary_terms(text, "zh")
            replaced, tech = self.save_tech_terms(text.rstrip())
            replaced, pinyin = self._save(replaced, PINYIN_TONE_PATTERN, "pinyin", re.IGNORECASE)
            replaced, names = self._save(replaced, NAME_PATTERN, "n", re.IGNORECASE)
            try:
                result = self.zh.normalize(replaced)
            except Exception:
                result = ""
            for i, name in enumerate(names or []):
                result = result.replace(f"<n_{chr(ord('a') + i)}>", name)
            for i, p in enumerate(pinyin or []):
                result = result.replace(f"<pinyin_{chr(ord('a') + i)}>", self.correct_pinyin(p))
            if tech:
                result = re.sub(r"\s*<H>\s*", "-", result)
            result = _replace_all(result, ZH_CHAR_REP_MAP)
        else:
            try:
                text = re.sub(ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                text = self.apply_glossary_terms(text, "en")
                replaced, tech = self.save_tech_terms(text)
                result = self.en.normalize(replaced)
                if tech:
                    result = re.sub(r"\s*<H>\s*", "-", result)
            except Exception:
                result = text
            result = _replace_all(result, CHAR_REP_MAP)
        for key, value in placeholders.items():
            result = result.replace(key, value)
        return result


CJK_RANGE_PATTERN = r"([ᄀ-ᇿ⺀-꓏ꡀ-힯豈-﫿︰-﹏･-ￜ\U00020000-\U0002FFFF])"


def tokenize_by_cjk_char(line: str) -> str:
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join(w.strip().upper() for w in chars if w.strip())


PUNCTUATION_MARKS = [".", "!", "?", "▁.", "▁?", "▁..."]


class TextTokenizer:
    def __init__(self, vocab_file: str | Path, normalizer: TextNormalizer | None) -> None:
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor(model_file=str(vocab_file))
        self.normalizer = normalizer

    @property
    def unk_token_id(self) -> int:
        return self.sp.unk_id()

    def tokenize(self, text: str) -> list[str]:
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp.Encode(text, out_type=str)
        if self.normalizer is not None:
            text = self.normalizer.normalize(text)
        return self.sp.Encode(tokenize_by_cjk_char(text), out_type=str)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.sp.PieceToId(t) for t in tokens]


def split_segments_by_token(tokens: list[str], split_tokens: list[str], max_tokens: int, quick_streaming_tokens: int = 0) -> list[list[str]]:
    """``TextTokenizer.split_segments_by_token``, its quirks included (the ``i += 1`` that skips nothing)."""
    if not tokens:
        return []
    segments: list[list[str]] = []
    current: list[str] = []
    length = 0
    for i in range(len(tokens)):
        token = tokens[i]
        current.append(token)
        length += 1
        if not ("," in split_tokens or "▁," in split_tokens) and ("," in current or "▁," in current):
            sub = split_segments_by_token(current, [",", "▁,"], max_tokens, quick_streaming_tokens)
        elif "-" not in split_tokens and "-" in current:
            sub = split_segments_by_token(current, ["-"], max_tokens, quick_streaming_tokens)
        elif length <= max_tokens:
            if token in split_tokens and length > 2:
                if i < len(tokens) - 1 and tokens[i + 1] in ["'", "▁'"]:
                    current.append(tokens[i + 1])
                segments.append(current)
                current = []
                length = 0
            continue
        else:
            sub = []
            for j in range(0, len(current), max_tokens):
                sub.append(current[j : j + max_tokens] if j + max_tokens < len(current) else current[j:])
            warnings.warn(f"The tokens length of segment exceeds limit: {max_tokens}, Tokens in segment: {current}.Maybe unexpected behavior", RuntimeWarning, stacklevel=2)
        segments.extend(sub)
        current = []
        length = 0
    if length > 0:
        segments.append(current)
    merged: list[list[str]] = []
    total = 0
    for segment in segments:
        total += len(segment)
        if len(segment) == 0:
            continue
        if not merged:
            merged.append(segment)
        elif len(merged[-1]) + len(segment) <= max_tokens and total > quick_streaming_tokens or len(merged[-1]) + len(segment) <= max_tokens / 2:
            merged[-1] = merged[-1] + segment
        else:
            merged.append(segment)
    return merged


def split_segments(tokens: list[str], max_tokens: int = 120, quick_streaming_tokens: int = 0) -> list[list[str]]:
    return split_segments_by_token(tokens, PUNCTUATION_MARKS, max_tokens, quick_streaming_tokens)
