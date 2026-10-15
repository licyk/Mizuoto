"""wetext's normaliser (wetext 0.0.4, Apache-2.0, Zhendong Peng), which CosyVoice uses for Chinese and English.

WeTextProcessing's grammars compiled to FSTs and run by ``kaldifst``: a tagger marks spans
(``cardinal { integer: "12" }``), the tokens are reordered into the verbaliser's expected field order,
and the verbaliser writes them out. Text without a digit is returned as it is (for normalisation, ``tn``).

The FSTs are files (``{lang}/{tn|itn}/tagger.fst`` and ``verbalizer.fst``, as ModelScope's
``pengzhendong/wetext`` lays them out); wetext downloads them at run time, Mizuoto ships them as an asset.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}
EN_TN_ORDERS = {
    "date": ["preserve_order", "text", "day", "month", "year"],
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj"],
}
ITN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value"],
    "money": ["currency", "value", "decimal"],
    "time": ["hour", "minute", "second", "noon"],
}


class _Token:
    def __init__(self, name: str) -> None:
        self.name = name
        self.order: list[str] = []
        self.members: dict[str, str] = {}

    def append(self, key: str, value: str) -> None:
        self.order.append(key)
        self.members[key] = value

    def string(self, orders: dict[str, list[str]]) -> str:
        out = self.name + " {"
        if self.name in orders and self.members.get("preserve_order") != "true":
            self.order = orders[self.name]
        for key in self.order:
            if key in self.members:
                out += f' {key}: "{self.members[key]}"'
        return out + " }"


class TokenParser:
    """wetext's ``TokenParser``: the tagger's output, its fields put in the verbaliser's order."""

    def __init__(self, lang: str, operator: str = "tn") -> None:
        if lang == "en":
            if operator != "tn":
                raise NotImplementedError("English inverse normalisation")
            self.orders = EN_TN_ORDERS
        elif lang == "zh":
            self.orders = TN_ORDERS if operator == "tn" else ITN_ORDERS
        else:
            raise ValueError(f"Unsupported language {lang!r}")

    def _load(self, text: str) -> None:
        if not text:
            raise ValueError("The tagger found nothing to verbalise")  # wetext asserts here
        self.index = 0
        self.text = text
        self.char = text[0]
        self.tokens: list[_Token] = []

    def _read(self) -> bool:
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def _ws(self) -> bool:
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self._read()
        return not_eos

    def _char(self, expected: str) -> bool:
        if self.char == expected:
            self._read()
            return True
        return False

    def _chars(self, expected: str) -> bool:
        ok = False
        for c in expected:
            ok |= self._char(c)
        return ok

    def _key(self) -> str:
        key = ""
        while self.char in string.ascii_letters + "_":
            key += self.char
            self._read()
        return key

    def _value(self) -> str:
        value = ""
        while self.char != '"':
            value += self.char
            escape = self.char == "\\"
            self._read()
            if escape:
                value += self.char
                self._read()
        return value

    def reorder(self, text: str) -> str:
        self._load(text)
        while self._ws():
            token = _Token(self._key())
            self._chars(" { ")
            while self._ws():
                if self.char == "}":
                    self._char("}")
                    break
                key = self._key()
                self._chars(': "')
                value = self._value()
                self._char('"')
                token.append(key, value)
            self.tokens.append(token)
        return " ".join(t.string(self.orders) for t in self.tokens).strip()


class WeTextNormalizer:
    """``wetext.Normalizer`` over FST files in ``folder``; ``lang`` is ``auto`` (Chinese when the text has Han characters), ``en`` or ``zh``.

    ``version`` is the wetext release whose steps are followed: ``0.0`` (CosyVoice's), or ``0.1`` (VoxCPM's, wetext
    0.1.2), which strips the text before tagging and after each step, and with ``remove_erhua`` also normalises text
    holding 儿 or 兒 without a digit. Each release ships its own FSTs; ``folder`` must hold the matching ones.
    """

    def __init__(self, folder: str | Path, lang: str = "auto", operator: str = "tn", remove_erhua: bool = False, enable_0_to_9: bool = False, version: str = "0.0") -> None:
        import kaldifst

        if lang not in ("auto", "en", "zh") or operator not in ("tn", "itn") or version not in ("0.0", "0.1"):
            raise ValueError(f"Unsupported normaliser {lang}/{operator} (wetext {version})")
        folder = Path(folder)
        self.lang = lang
        self.operator = operator
        self.version = version
        self.remove_erhua = remove_erhua
        taggers = {"en": "tagger.fst", "zh": "tagger.fst"}
        verbalizers = {"en": "verbalizer.fst", "zh": "verbalizer.fst"}
        if operator == "itn" and enable_0_to_9:
            taggers["zh"] = "tagger_enable_0_to_9.fst"
        if operator == "tn" and remove_erhua:
            verbalizers["zh"] = "verbalizer_remove_erhua.fst"
        self.taggers: dict[str, Any] = {}
        self.verbalizers: dict[str, Any] = {}
        for code in ("en", "zh"):
            if lang in ("auto", code):
                self.taggers[code] = kaldifst.TextNormalizer(str(folder / code / operator / taggers[code]))
                self.verbalizers[code] = kaldifst.TextNormalizer(str(folder / code / operator / verbalizers[code]))

    def normalize(self, text: str) -> str:
        if self.version == "0.1":
            return self._normalize_01(text)
        if self.operator == "itn" or re.search(r"\d", text):
            lang = self.lang
            if lang == "auto":
                lang = "zh" if any("一" <= ch <= "鿿" for ch in text) else "en"
            tagged = self.taggers[lang](text)
            return self.verbalizers[lang](TokenParser(lang, self.operator).reorder(tagged))
        return text

    def _normalize_01(self, text: str) -> str:
        """wetext 0.1.2's ``normalize`` (no contraction fixing, no pre- or post-processing FSTs: their defaults)."""
        text = text.strip()
        if self.operator == "tn":
            needed = bool(re.search(r"\d", text)) or (self.remove_erhua and bool(re.search(r"儿|兒", text)))
        else:
            needed = len(text) > 0
        if needed:
            lang = self.lang
            if lang == "auto":
                lang = "zh" if any("一" <= ch <= "鿿" for ch in text) or text.isdigit() else "en"
            if lang == "en" and self.operator == "itn":
                lang = "zh"
            tagged = self.taggers[lang](text).strip()
            text = self.verbalizers[lang](TokenParser(lang, self.operator).reorder(tagged)).strip()
        return text.strip()


class TnProcessor:
    """WeTextProcessing's ``Processor.normalize`` (the ``tn`` package, which IndexTTS runs on Linux) over one language's
    tagger and verbaliser FSTs: tag, reorder, verbalise — every text, digits or not (wetext skips text without one).

    ``lang`` picks the token order: ``zh`` (``tn``) or ``en`` (``en_tn``)."""

    def __init__(self, tagger: str | Path, verbalizer: str | Path, lang: str) -> None:
        import kaldifst

        self.tagger = kaldifst.TextNormalizer(str(tagger))
        self.verbalizer = kaldifst.TextNormalizer(str(verbalizer))
        self.parser = TokenParser(lang, "tn")

    def tag(self, text: str) -> str:
        return self.tagger(text) if text else ""

    def normalize(self, text: str) -> str:
        tagged = self.tag(text)
        if not tagged:
            return ""
        return self.verbalizer(self.parser.reorder(tagged))
