"""Canonical language codes and their aliases.

Mizuoto speaks BCP-47-like codes (``zh``, ``en``, ``ja``, ``yue``, ``zh-sichuan``…) and ``auto``;
each adapter maps them to its engine's own names. Here they are normalised and checked.
"""

from __future__ import annotations

ALIASES: dict[str, str] = {
    "chinese": "zh",
    "mandarin": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "cmn": "zh",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "japanese": "ja",
    "jp": "ja",
    "korean": "ko",
    "kr": "ko",
    "cantonese": "yue",
    "zh-hk": "yue",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "nb": "no",
    "fil": "tl",
    "iw": "he",
    "in": "id",
}

NAMES: dict[str, str] = {
    "auto": "Detect",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "my": "Burmese",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "km": "Khmer",
    "lo": "Lao",
    "ms": "Malay",
    "no": "Norwegian",
    "pl": "Polish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def normalize(code: str | None) -> str | None:
    """A canonical code: lower case, ``_`` as ``-``, aliases resolved; None and empty stay None."""
    if code is None:
        return None
    text = code.strip().lower().replace("_", "-")
    if not text:
        return None
    return ALIASES.get(text, text)


def supports(languages: list[str], code: str | None) -> bool:
    """Whether a model's language list covers ``code`` (``*`` covers everything; ``zh-*`` covers ``zh-sichuan``)."""
    if code is None or code == "auto" or "*" in languages:
        return True
    if code in languages:
        return True
    return any(lang.endswith("-*") and (code == lang[:-2] or code.startswith(lang[:-1])) for lang in languages)


def detect(text: str) -> str | None:
    """A script-based guess for engines that need an explicit language: Kana → ja, Hangul → ko,
    Han → zh, Cyrillic → ru, Arabic → ar, Latin → en; None when nothing decides."""
    counts = {"ja": 0, "ko": 0, "zh": 0, "ru": 0, "ar": 0, "en": 0}
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            counts["ja"] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts["ko"] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            counts["zh"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["ru"] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts["ar"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["en"] += 1
    if counts["ja"]:
        return "ja"  # Japanese text mixes kana with kanji; any kana decides
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else None
