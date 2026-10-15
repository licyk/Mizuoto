"""Korean for GPT-SoVITS (upstream ``text/korean.py``, from MB-iSTFT-VITS-Korean; MIT, (c) 2024 RVC-Boss): Latin letters
spelt in Hangul, g2pk2's pronunciation rules (``g2pk``), then compatibility jamo with compound vowels divided, one phone
per jamo (``空`` for a space, ``停`` for anything outside the symbol table).

Needs the Korean extra (``python-mecab-ko`` and ``jamo``); without it ``KoreanG2P()`` raises ``KoreanMissing``.
"""

from __future__ import annotations

import re

from mizuoto.engines.gptsovits.text.g2pk import KoreanMissing, _jamo
from mizuoto.engines.gptsovits.text.symbols import SYMBOL_SET

__all__ = ["KoreanG2P", "KoreanMissing"]

_hangul_divided = [
    (re.compile(a), b)
    for a, b in [
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]


def _words(text: str) -> list[str]:
    return text.split()


_LATIN = _words("에이 비 시 디 이 에프 지 에이치 아이 제이 케이 엘 엠 엔 오 피 큐 아르 에스 티 유 브이 더블유 엑스 와이 제트")
_latin_to_hangul = [(re.compile(letter, re.IGNORECASE), hangul) for letter, hangul in zip("abcdefghijklmnopqrstuvwxyz", _LATIN, strict=True)]

POST_REP_MAP = {"：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": ".", "·": ",", "、": ",", "...": "…", " ": "空"}


def fix_g2pk2_error(text: str) -> str:
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
            new_text += text[i : i + 3] + " " + "ㄴ"
            i += 5
        else:
            new_text += text[i]
            i += 1
    return new_text + text[i:]


def latin_to_hangul(text: str) -> str:
    for pattern, replacement in _latin_to_hangul:
        text = re.sub(pattern, replacement, text)
    return text


def divide_hangul(text: str) -> str:
    jamo = _jamo()
    text = jamo.j2hcj(jamo.h2j(text))
    for pattern, replacement in _hangul_divided:
        text = re.sub(pattern, replacement, text)
    return text


def post_replace_ph(ph: str) -> str:
    ph = POST_REP_MAP.get(ph, ph)
    return ph if ph in SYMBOL_SET else "停"


class KoreanG2P:
    def __init__(self) -> None:
        from mizuoto.engines.gptsovits.text.g2pk import G2pK

        self.g2pk = G2pK()

    def g2p(self, text: str) -> list[str]:
        text = latin_to_hangul(text)
        text = self.g2pk(text)
        text = divide_hangul(text)
        text = fix_g2pk2_error(text)
        text = re.sub(r"([ㄱ-ㅣ])$", r"\1.", text)
        return [post_replace_ph(i) for i in text]
