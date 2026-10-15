"""English text normalisation for GPT-SoVITS (upstream ``text/en_normalization/expend.py`` by Cosmo-klara; MIT, (c) 2024
RVC-Boss): ordinals, arithmetic, negative numbers, times, units, currencies, decimals, fractions, numbers (with
``inflect``), accents stripped, then only letters and ``'.,?!-`` kept and capitals split (``AI`` → ``A I``).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

MEASUREMENT_MAP = {
    "m": ["meter", "meters"],
    "km": ["kilometer", "kilometers"],
    "km/h": ["kilometer per hour", "kilometers per hour"],
    "ft": ["feet", "feet"],
    "L": ["liter", "liters"],
    "tbsp": ["tablespoon", "tablespoons"],
    "tsp": ["teaspoon", "teaspoons"],
    "h": ["hour", "hours"],
    "min": ["minute", "minutes"],
    "s": ["second", "seconds"],
    "°C": ["degree celsius", "degrees celsius"],
    "°F": ["degree fahrenheit", "degrees fahrenheit"],
}

_ordinal_number_re = re.compile(r"\b([0-9]+)\. ")
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_time_re = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")
_measurement_re = re.compile(r"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b")
_pounds_re_start = re.compile(r"£([0-9\.\,]*[0-9]+)")
_pounds_re_end = re.compile(r"([0-9\.\,]*[0-9]+)£")
_dollars_re_start = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_dollars_re_end = re.compile(r"([(0-9\.\,]*[0-9]+)\$")
_decimal_number_re = re.compile(r"([0-9]+\.\s*[0-9]+)")
_fraction_re = re.compile(r"([0-9]+/[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_NUM = r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))"
RE_ASMD = re.compile(_NUM + r"\s+([\+\-\×÷=])\s+" + _NUM)
ASMD_MAP = {"+": " plus ", "-": " minus ", "×": " times ", "÷": " divided by ", "=": " Equals "}
RE_INTEGER = re.compile(r"(?:^|\s+)(-)(\d+)")


def _remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def _expand_measurement(m: re.Match[str]) -> str:
    sign = m.group(3)
    ptr = 1
    num = int(m.group(1).replace(sign, "").replace(".", ""))
    decimal_part = m.group(2)
    if decimal_part is None and num == 1:
        ptr = 0
    return m.group(1).replace(sign, " " + MEASUREMENT_MAP[sign][ptr])


def _expand_money(m: re.Match[str], unit: tuple[str, str], sub: tuple[str, str]) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " " + unit[1]
    whole = int(parts[0]) if parts[0] else 0
    cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    whole_unit = unit[0] if whole == 1 else unit[1]
    sub_unit = sub[0] if cents == 1 else sub[1]
    if whole and cents:
        return f"{whole} {whole_unit} and {cents} {sub_unit}"
    if whole:
        return f"{whole} {whole_unit}"
    if cents:
        return f"{cents} {sub_unit}"
    return "zero " + unit[1]


def _expand_pounds(m: re.Match[str]) -> str:
    return _expand_money(m, ("pound", "pounds"), ("penny", "pence"))


def _expand_dollars(m: re.Match[str]) -> str:
    return _expand_money(m, ("dollar", "dollars"), ("cent", "cents"))


def _expand_decimal_number(m: re.Match[str]) -> str:
    parts = m.group(1).split(".")
    words = ["point" if char == "." else char for char in parts[1]]
    return parts[0] + " point " + " ".join(words)


def replace_asmd(match: re.Match[str]) -> str:
    return match.group(1) + ASMD_MAP[match.group(8)] + match.group(9)


def replace_negative_num(match: re.Match[str]) -> str:
    sign = "negative " if match.group(1) else ""
    return f"{sign}{match.group(2)}"


class EnglishNormalizer:
    """``expend.normalize`` with its own ``inflect`` engine."""

    def __init__(self) -> None:
        import inflect

        self.inflect: Any = inflect.engine()

    def _convert_ordinal(self, m: re.Match[str]) -> str:
        return self.inflect.ordinal(m.group(1)) + ", "

    def _expand_time(self, m: re.Match[str]) -> str:
        hours, minutes = map(int, m.group(1, 2))
        period = "a.m." if hours < 12 else "p.m."
        if hours > 12:
            hours -= 12
        hour_word = self.inflect.number_to_words(hours)
        minute_word = self.inflect.number_to_words(minutes) if minutes != 0 else ""
        if minutes == 0:
            return f"{hour_word} o'clock {period}"
        return f"{hour_word} {minute_word} {period}"

    def _expand_fraction(self, m: re.Match[str]) -> str:
        numerator, denominator = map(int, m.group(0).split("/"))
        numerator_part = self.inflect.number_to_words(numerator)
        if denominator == 2:
            denominator_part = "half" if numerator == 1 else "halves"
        elif denominator == 1:
            return f"{numerator_part}"
        else:
            denominator_part = self.inflect.ordinal(self.inflect.number_to_words(denominator))
            if numerator > 1:
                denominator_part += "s"
        return f"{numerator_part} {denominator_part}"

    def _expand_ordinal(self, m: re.Match[str]) -> str:
        return self.inflect.number_to_words(m.group(0))

    def _expand_number(self, m: re.Match[str]) -> str:
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return "two thousand"
            if num > 2000 and num < 2010:
                return "two thousand " + self.inflect.number_to_words(num % 100)
            if num % 100 == 0:
                return self.inflect.number_to_words(num // 100) + " hundred"
            return self.inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
        return self.inflect.number_to_words(num, andword="")

    def __call__(self, text: str) -> str:
        text = re.sub(_ordinal_number_re, self._convert_ordinal, text)
        while RE_ASMD.search(text):
            text = RE_ASMD.sub(replace_asmd, text)
        text = RE_INTEGER.sub(replace_negative_num, text)
        text = re.sub(_comma_number_re, _remove_commas, text)
        text = re.sub(_time_re, self._expand_time, text)
        text = re.sub(_measurement_re, _expand_measurement, text)
        text = re.sub(_pounds_re_start, _expand_pounds, text)
        text = re.sub(_pounds_re_end, _expand_pounds, text)
        text = re.sub(_dollars_re_start, _expand_dollars, text)
        text = re.sub(_dollars_re_end, _expand_dollars, text)
        text = re.sub(_decimal_number_re, _expand_decimal_number, text)
        text = re.sub(_fraction_re, self._expand_fraction, text)
        text = re.sub(_ordinal_re, self._expand_ordinal, text)
        text = re.sub(_number_re, self._expand_number, text)
        text = "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")
        text = re.sub("%", " percent", text)
        text = re.sub(r"[^ A-Za-z'.,?!\-]", "", text)
        text = re.sub(r"(?i)i\.e\.", "that is", text)
        text = re.sub(r"(?i)e\.g\.", "for example", text)
        return re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
