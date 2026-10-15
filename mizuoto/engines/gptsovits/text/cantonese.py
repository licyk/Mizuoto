"""Cantonese for GPT-SoVITS (upstream ``text/cantonese.py``, from Naozumi's Bert-VITS2-Cantonese; MIT, (c) 2024 RVC-Boss):
the Mandarin normaliser, punctuation mapping, then ToJyutping's jyutping split into initials and toned finals, each
prefixed with ``Y`` so it cannot collide with a Mandarin phone.
"""

from __future__ import annotations

import re

from mizuoto.engines.gptsovits.text.symbols import PUNCTUATION


def _words(text: str) -> list[str]:
    return text.split()


INITIALS = [
    *_words("aa aai aak aap aat aau ai au ap at ak a p b e ts t dz d kw k gw g f h l m ng n s y w c z j ong on ou oi ok o uk ung"),
    *_words("sp spl spn sil"),
]

REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
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
}
_REP_RE = re.compile("|".join(re.escape(p) for p in REP_MAP))
_KEEP_RE = re.compile(r"[^一-龥" + "".join(PUNCTUATION) + r"]+")
_PUNCT_RUN = re.compile(r"^[{}]+$".format(re.escape("".join(PUNCTUATION))))
_PUNCT_SPLIT = re.compile(r"([{}])".format(re.escape("".join(PUNCTUATION))))
PUNCTUATION_SET = set(PUNCTUATION)


def replace_punctuation(text: str) -> str:
    replaced = _REP_RE.sub(lambda x: REP_MAP[x.group()], text)
    return _KEEP_RE.sub("", replaced)


def text_normalize(text: str) -> str:
    from mizuoto.engines.gptsovits.text.zh_normalization import normalize

    return "".join(replace_punctuation(sentence) for sentence in normalize(text))


def jyuping_to_initials_finals_tones(jyuping_syllables: list[str]) -> tuple[list[str], list[int]]:
    initials_finals: list[str] = []
    tones: list[int] = []
    word2ph: list[int] = []
    for syllable in jyuping_syllables:
        if syllable in PUNCTUATION or syllable == "_":
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)
        else:
            try:
                tone = int(syllable[-1])
                syllable_without_tone = syllable[:-1]
            except ValueError:
                tone = 0
                syllable_without_tone = syllable
            for initial in INITIALS:
                if syllable_without_tone.startswith(initial):
                    if syllable_without_tone.startswith("nga"):
                        initials_finals.extend([syllable_without_tone[:2], syllable_without_tone[2:] or syllable_without_tone[-1]])
                    else:
                        initials_finals.extend([initial, syllable_without_tone[len(initial) :] or initial[-1]])
                    tones.extend([-1, tone])
                    word2ph.append(2)
                    break
    assert len(initials_finals) == len(tones)
    phones = []
    for a, b in zip(initials_finals, tones, strict=True):
        todo = f"{a}{b}" if b not in [-1, 0] else a
        if todo not in PUNCTUATION_SET:
            todo = f"Y{todo}"
        phones.append(todo)
    return phones, word2ph


def get_jyutping(text: str) -> list[str]:
    import ToJyutping

    jyutping_array: list[str] = []
    for word, syllable in ToJyutping.get_jyutping_list(text):
        if _PUNCT_RUN.match(word):
            jyutping_array.extend(punct for punct in _PUNCT_SPLIT.split(word) if len(punct) > 0)
        else:
            if syllable is None:
                raise TypeError("expected string or bytes-like object, got 'NoneType'")  # upstream's re.search(…, None)
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable):
                raise ValueError(f"Failed to convert {word} to jyutping: {syllable}")
            jyutping_array.append(syllable)
    return jyutping_array


def g2p(text: str) -> tuple[list[str], list[int]]:
    return jyuping_to_initials_finals_tones(get_jyutping(text))
