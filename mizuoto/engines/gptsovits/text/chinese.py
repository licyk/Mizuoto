"""Mandarin for GPT-SoVITS v2+ (upstream ``text/chinese2.py``; MIT, (c) 2024 RVC-Boss).

Normalisation (``zh_normalization``), punctuation mapping, then per sentence: jieba's POS segmentation, upstream's
merges and tone sandhi, G2PW's pinyin (with upstream's corrections dictionary), erhua, and the opencpop-strict table
from pinyin to GPT-SoVITS's initials and finals. Upstream uses ``jieba_fast``'s global tokenizers; here the front end
owns a ``jieba.Tokenizer`` and a ``jieba.posseg.POSTokenizer`` (same dictionary and tables, identical cuts).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from mizuoto.engines.gptsovits.text.symbols import PUNCTUATION

if TYPE_CHECKING:
    from mizuoto.engines.gptsovits.text.g2pw import G2PW

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
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}
_REP_RE = re.compile("|".join(re.escape(p) for p in REP_MAP))
_KEEP_RE = re.compile(r"[^一-龥" + "".join(PUNCTUATION) + r"]+")
_SENTENCE_RE = re.compile(r"(?<=[{}])\s*".format("".join(PUNCTUATION)))
_CONSECUTIVE_RE = re.compile("([{0}])([{0}])+".format("".join(re.escape(p) for p in PUNCTUATION)))


def _words(text: str) -> list[str]:
    return text.split()


MUST_ERHUA = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
NOT_ERHUA = {
    *_words("虐儿 为儿 护儿 瞒儿 救儿 替儿 有儿 一儿 我儿 俺儿 妻儿 拐儿 聋儿 乞儿 患儿 幼儿 孤儿 婴儿 婴幼儿 连体儿 脑瘫儿 流浪儿 体弱儿 混血儿 蜜雪儿 舫儿"),
    *_words("祖儿 美儿 应采儿 可儿 侄儿 孙儿 侄孙儿 女儿 男儿 红孩儿 花儿 虫儿 马儿 鸟儿 猪儿 猫儿 狗儿 少儿"),
}

_V_REP = {"uei": "ui", "iou": "iu", "uen": "un"}
_PINYIN_REP = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
_SINGLE_REP = {"v": "yu", "e": "e", "i": "y", "u": "w"}


def replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced = _REP_RE.sub(lambda x: REP_MAP[x.group()], text)
    return _KEEP_RE.sub("", replaced)


def replace_consecutive_punctuation(text: str) -> str:
    return _CONSECUTIVE_RE.sub(r"\1", text)


def text_normalize(text: str) -> str:
    from mizuoto.engines.gptsovits.text.zh_normalization import normalize

    dest_text = "".join(replace_punctuation(sentence) for sentence in normalize(text))
    return replace_consecutive_punctuation(dest_text)


def merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> tuple[list[str], list[str]]:
    """``_merge_erhua``: 儿 after a syllable takes that syllable's tone (``er2``/``er5`` → ``er`` + tone)."""
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"
    if word not in MUST_ERHUA and (word in NOT_ERHUA or pos in {"a", "j", "nr"}):
        return initials, finals
    if len(finals) != len(word):
        return initials, finals
    new_initials: list[str] = []
    new_finals: list[str] = []
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn in {"er2", "er5"} and word[-2:] not in NOT_ERHUA and new_finals:
            phn = "er" + new_finals[-1][-1]
        new_initials.append(initials[i])
        new_finals.append(phn)
    return new_initials, new_finals


class ChineseG2P:
    """``chinese2.g2p`` with G2PW on (``is_g2pw = True`` upstream)."""

    def __init__(self, g2pw: G2PW, tokenizer: Any, pos_tokenizer: Any) -> None:
        from mizuoto.engines.gptsovits.text._data import read_text
        from mizuoto.engines.gptsovits.text.tone_sandhi import ToneSandhi

        self.g2pw = g2pw
        self.pos_tokenizer = pos_tokenizer
        self.tone_modifier = ToneSandhi(tokenizer)
        self.pinyin_to_symbol_map = {line.split("\t")[0]: line.strip().split("\t")[1] for line in read_text("opencpop-strict.txt").splitlines(keepends=True)}

    text_normalize = staticmethod(text_normalize)

    def g2p(self, text: str) -> tuple[list[str], list[int]]:
        sentences = [i for i in _SENTENCE_RE.split(text) if i.strip() != ""]
        return self._g2p(sentences)

    def _g2p(self, segments: list[str]) -> tuple[list[str], list[int]]:
        from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

        from mizuoto.engines.gptsovits.text.g2pw import correct_pronunciation

        phones_list: list[str] = []
        word2ph: list[int] = []
        processed_segments = [re.sub("[a-zA-Z]+", "", seg) for seg in segments]
        batch_inputs = [seg for seg in processed_segments if seg]
        g2pw_batch_results = self.g2pw(batch_inputs) if batch_inputs else []
        g2pw_batch_cursor = 0
        for seg in processed_segments:
            pinyins: list[Any] = []
            seg_cut = self.tone_modifier.pre_merge_for_modify([(p.word, p.flag) for p in self.pos_tokenizer.lcut(seg)])
            initials_by_word: list[list[str]] = []
            finals_by_word: list[list[str]] = []
            if seg:
                pinyins = g2pw_batch_results[g2pw_batch_cursor]
                g2pw_batch_cursor += 1
            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials: list[str] = []
                sub_finals: list[str] = []
                now_word_length = pre_word_length + len(word)
                if pos == "eng":
                    pre_word_length = now_word_length
                    continue
                word_pinyins = correct_pronunciation(word, pinyins[pre_word_length:now_word_length])
                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)
                pre_word_length = now_word_length
                sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                sub_initials, sub_finals = merge_erhua(sub_initials, sub_finals, word, pos)
                initials_by_word.append(sub_initials)
                finals_by_word.append(sub_finals)
            initials = sum(initials_by_word, [])
            finals = sum(finals_by_word, [])
            for c, v in zip(initials, finals, strict=False):
                raw_pinyin = c + v
                if c == v:
                    assert c in PUNCTUATION
                    phone = [c]
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]
                    pinyin = c + v_without_tone
                    assert tone in "12345"
                    if c:
                        if v_without_tone in _V_REP:
                            pinyin = c + _V_REP[v_without_tone]
                    elif pinyin in _PINYIN_REP:
                        pinyin = _PINYIN_REP[pinyin]
                    elif pinyin[0] in _SINGLE_REP:
                        pinyin = _SINGLE_REP[pinyin[0]] + pinyin[1:]
                    assert pinyin in self.pinyin_to_symbol_map, (pinyin, seg, raw_pinyin)
                    new_c, new_v = self.pinyin_to_symbol_map[pinyin].split(" ")
                    phone = [new_c, new_v + tone]
                    word2ph.append(len(phone))
                phones_list += phone
        return phones_list, word2ph
