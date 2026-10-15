"""Chinese text normalisation: numbers, dates, times, phone numbers, units (GPT-SoVITS ``text/zh_normalization``).

Upstream's package is PaddleSpeech's ``zh_normalization`` (Apache-2.0, (c) 2021 PaddlePaddle Authors) with GPT-SoVITS's
changes (MIT, (c) 2024 RVC-Boss): arithmetic, powers, version numbers, ``~`` ranges. Ported unchanged in behaviour,
including its quirks (``replace_temperature`` reads the decimal group as the unit, so every unit but 摄氏度 reads 度;
a time range's second half reads 半 when the *first* minute is 30). The traditional → simplified table is ``data/t2s.txt``.
"""

from __future__ import annotations

import re
import string
from functools import cache

# -- char_convert ----------------------------------------------------------------------------------------------------


@cache
def _t2s() -> dict[str, str]:
    from mizuoto.engines.gptsovits.text._data import read_text

    traditional, simplified = read_text("t2s.txt").split("\n")[:2]
    return dict(zip(traditional, simplified, strict=True))


def traditional_to_simplified(text: str) -> str:
    table = _t2s()
    return "".join([table.get(item, item) for item in text])


# -- constants -------------------------------------------------------------------------------------------------------

F2H_ASCII_LETTERS = {ord(char) + 65248: ord(char) for char in string.ascii_letters}
F2H_DIGITS = {ord(char) + 65248: ord(char) for char in string.digits}
F2H_SPACE = {"\u3000": " "}
"""Upstream passes this to ``str.translate``, whose keys must be code points: it changes nothing, and neither does this."""

# -- num -------------------------------------------------------------------------------------------------------------

DIGITS = {str(i): tran for i, tran in enumerate("零一二三四五六七八九")}
UNITS = {1: "十", 2: "百", 3: "千", 4: "万", 8: "亿"}

COM_QUANTIFIERS = (
    "(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|"
    "砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|"
    "(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|"
    "袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|"
    "(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)"
)


def _get_value(value_string: str, use_zero: bool = True) -> list[str]:
    stripped = value_string.lstrip("0")
    if len(stripped) == 0:
        return []
    if len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS["0"], DIGITS[stripped]]
        return [DIGITS[stripped]]
    largest_unit = next(power for power in reversed(UNITS.keys()) if power < len(stripped))
    first_part = value_string[:-largest_unit]
    second_part = value_string[-largest_unit:]
    return [*_get_value(first_part), UNITS[largest_unit], *_get_value(second_part)]


def verbalize_cardinal(value_string: str) -> str:
    if not value_string:
        return ""
    value_string = value_string.lstrip("0")
    if len(value_string) == 0:
        return DIGITS["0"]
    result_symbols = _get_value(value_string)
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS["1"] and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    return "".join(result_symbols)


def verbalize_digit(value_string: str, alt_one: bool = False) -> str:
    result = "".join([DIGITS[digit] for digit in value_string])
    if alt_one:
        result = result.replace("一", "幺")
    return result


def num2str(value_string: str) -> str:
    integer_decimal = value_string.split(".")
    if len(integer_decimal) == 1:
        integer = integer_decimal[0]
        decimal = ""
    elif len(integer_decimal) == 2:
        integer, decimal = integer_decimal
    else:
        raise ValueError(f"The value string: '${value_string}' has more than one point in it.")
    result = verbalize_cardinal(integer)
    decimal = decimal.rstrip("0") + "0" if decimal.endswith("0") else decimal.rstrip("0")
    if decimal:
        result = result if result else "零"
        result += "点" + verbalize_digit(decimal)
    return result


RE_FRAC = re.compile(r"(-?)(\d+)/(\d+)")


def replace_frac(match: re.Match[str]) -> str:
    sign = "负" if match.group(1) else ""
    return f"{sign}{num2str(match.group(3))}分之{num2str(match.group(2))}"


RE_PERCENTAGE = re.compile(r"(-?)(\d+(\.\d+)?)%")


def replace_percentage(match: re.Match[str]) -> str:
    sign = "负" if match.group(1) else ""
    return f"{sign}百分之{num2str(match.group(2))}"


RE_INTEGER = re.compile(r"(-)(\d+)")


def replace_negative_num(match: re.Match[str]) -> str:
    sign = "负" if match.group(1) else ""
    return f"{sign}{num2str(match.group(2))}"


RE_DEFAULT_NUM = re.compile(r"\d{3}\d*")


def replace_default_num(match: re.Match[str]) -> str:
    return verbalize_digit(match.group(0), alt_one=True)


RE_ASMD = re.compile(
    r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([\+\-\×÷=])"
    r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))"
)
ASMD_MAP = {"+": "加", "-": "减", "×": "乘", "÷": "除", "=": "等于"}


def replace_asmd(match: re.Match[str]) -> str:
    return match.group(1) + ASMD_MAP[match.group(8)] + match.group(9)


RE_POWER = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+")
POWER_MAP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ˣ": "x", "ʸ": "y", "ⁿ": "n"}


def replace_power(match: re.Match[str]) -> str:
    return "的" + "".join(POWER_MAP[m] for m in match.group(0)) + "次方"


RE_DECIMAL_NUM = re.compile(r"(-?)((\d+)(\.\d+))|(\.(\d+))")
RE_POSITIVE_QUANTIFIERS = re.compile(r"(\d+)([多余几\+])?" + COM_QUANTIFIERS)
RE_NUMBER = re.compile(r"(-?)((\d+)(\.\d+)?)|(\.(\d+))")


def replace_positive_quantifier(match: re.Match[str]) -> str:
    number = match.group(1)
    match_2 = match.group(2)
    if match_2 == "+":
        match_2 = "多"
    match_2 = match_2 if match_2 else ""
    quantifiers = match.group(3)
    number = num2str(number)
    number = "两" if number == "二" else number
    return f"{number}{match_2}{quantifiers}"


def replace_number(match: re.Match[str]) -> str:
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal:
        return num2str(pure_decimal)
    return f"{'负' if sign else ''}{num2str(number)}"


RE_RANGE = re.compile(
    r"""
    (?<![\d\+\-\×÷=])
    ((-?)((\d+)(\.\d+)?))
    [-~]
    ((-?)((\d+)(\.\d+)?))
    (?![\d\+\-\×÷=])
    """,
    re.VERBOSE,
)


def replace_range(match: re.Match[str]) -> str:
    first = RE_NUMBER.sub(replace_number, match.group(1))
    second = RE_NUMBER.sub(replace_number, match.group(6))
    return f"{first}到{second}"


_UNITS_RE = r"(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)"
RE_TO_RANGE = re.compile(r"((-?)((\d+)(\.\d+)?)|(\.(\d+)))" + _UNITS_RE + r"[~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))" + _UNITS_RE)


def replace_to_range(match: re.Match[str]) -> str:
    return match.group(0).replace("~", "至")


RE_VERSION_NUM = re.compile(r"((\d+)(\.\d+)(\.\d+)?(\.\d+)+)")


def replace_version_num(match: re.Match[str]) -> str:
    return "".join("点" if c == "." else num2str(c) for c in match.group(1))


# -- chronology ------------------------------------------------------------------------------------------------------


def _time_num2str(num_string: str) -> str:
    result = num2str(num_string.lstrip("0"))
    if num_string.startswith("0"):
        result = DIGITS["0"] + result
    return result


RE_TIME = re.compile(r"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?")
RE_TIME_RANGE = re.compile(r"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?")


def replace_time(match: re.Match[str]) -> str:
    is_range = len(match.groups()) > 5
    hour, minute, second = match.group(1), match.group(2), match.group(4)
    result = f"{num2str(hour)}点"
    if minute.lstrip("0"):
        result += "半" if int(minute) == 30 else f"{_time_num2str(minute)}分"
    if second and second.lstrip("0"):
        result += f"{_time_num2str(second)}秒"
    if is_range:
        hour_2, minute_2, second_2 = match.group(6), match.group(7), match.group(9)
        result += "至"
        result += f"{num2str(hour_2)}点"
        if minute_2.lstrip("0"):
            result += "半" if int(minute) == 30 else f"{_time_num2str(minute_2)}分"
        if second_2 and second_2.lstrip("0"):
            result += f"{_time_num2str(second_2)}秒"
    return result


RE_DATE = re.compile(r"(\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?")


def replace_date(match: re.Match[str]) -> str:
    year, month, day = match.group(1), match.group(3), match.group(5)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}{match.group(9)}"
    return result


RE_DATE2 = re.compile(r"(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])")


def replace_date2(match: re.Match[str]) -> str:
    year, month, day = match.group(1), match.group(3), match.group(4)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}日"
    return result


# -- phonecode -------------------------------------------------------------------------------------------------------

RE_MOBILE_PHONE = re.compile(r"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)")
RE_TELEPHONE = re.compile(r"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)")
RE_NATIONAL_UNIFORM_NUMBER = re.compile(r"(400)(-)?\d{3}(-)?\d{4}")


def phone2str(phone_string: str, mobile: bool = True) -> str:
    parts = phone_string.strip("+").split() if mobile else phone_string.split("-")
    return "，".join([verbalize_digit(part, alt_one=True) for part in parts])


def replace_phone(match: re.Match[str]) -> str:
    return phone2str(match.group(0), mobile=False)


def replace_mobile(match: re.Match[str]) -> str:
    return phone2str(match.group(0))


# -- quantifier ------------------------------------------------------------------------------------------------------

RE_TEMPERATURE = re.compile(r"(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)")
MEASURE_DICT = {
    "cm2": "平方厘米",
    "cm²": "平方厘米",
    "cm3": "立方厘米",
    "cm³": "立方厘米",
    "cm": "厘米",
    "db": "分贝",
    "ds": "毫秒",
    "kg": "千克",
    "km": "千米",
    "m2": "平方米",
    "m²": "平方米",
    "m³": "立方米",
    "m3": "立方米",
    "ml": "毫升",
    "m": "米",
    "mm": "毫米",
    "s": "秒",
}


def replace_temperature(match: re.Match[str]) -> str:
    sign = "零下" if match.group(1) else ""
    temperature = num2str(match.group(2))
    unit = "摄氏度" if match.group(3) == "摄氏度" else "度"  # upstream reads group 3 (the decimals): always 度
    return f"{sign}{temperature}{unit}"


def replace_measure(sentence: str) -> str:
    for q_notation, word in MEASURE_DICT.items():
        if q_notation in sentence:
            sentence = sentence.replace(q_notation, word)
    return sentence


# -- text_normlization -----------------------------------------------------------------------------------------------

SENTENCE_SPLITOR = re.compile(r"([：、，；。？！,;?!][”’]?)")

_POST_REPLACE = [
    ("/", "每"),
    ("①", "一"),
    ("②", "二"),
    ("③", "三"),
    ("④", "四"),
    ("⑤", "五"),
    ("⑥", "六"),
    ("⑦", "七"),
    ("⑧", "八"),
    ("⑨", "九"),
    ("⑩", "十"),
    ("α", "阿尔法"),
    ("β", "贝塔"),
    ("γ", "伽玛"),
    ("Γ", "伽玛"),
    ("δ", "德尔塔"),
    ("Δ", "德尔塔"),
    ("ε", "艾普西龙"),
    ("ζ", "捷塔"),
    ("η", "依塔"),
    ("θ", "西塔"),
    ("Θ", "西塔"),
    ("ι", "艾欧塔"),
    ("κ", "喀帕"),
    ("λ", "拉姆达"),
    ("Λ", "拉姆达"),
    ("μ", "缪"),
    ("ν", "拗"),
    ("ξ", "克西"),
    ("Ξ", "克西"),
    ("ο", "欧米克伦"),
    ("π", "派"),
    ("Π", "派"),
    ("ρ", "肉"),
    ("ς", "西格玛"),
    ("Σ", "西格玛"),
    ("σ", "西格玛"),
    ("τ", "套"),
    ("υ", "宇普西龙"),
    ("φ", "服艾"),
    ("Φ", "服艾"),
    ("χ", "器"),
    ("ψ", "普赛"),
    ("Ψ", "普赛"),
    ("ω", "欧米伽"),
    ("Ω", "欧米伽"),
    ("+", "加"),
    ("-", "减"),
    ("×", "乘"),
    ("÷", "除"),
    ("=", "等"),
]


def _split(text: str, lang: str = "zh") -> list[str]:
    if lang == "zh":
        text = text.replace(" ", "")
        text = re.sub(r"[——《》【】<>{}()（）#&@“”^_|\\]", "", text)
    text = SENTENCE_SPLITOR.sub(r"\1\n", text)
    text = text.strip()
    return [sentence.strip() for sentence in re.split(r"\n+", text)]


def _post_replace(sentence: str) -> str:
    for old, new in _POST_REPLACE:
        sentence = sentence.replace(old, new)
    return re.sub(r"[-——《》【】<=>{}()（）#&@“”^_|\\]", "", sentence)


def normalize_sentence(sentence: str) -> str:
    sentence = traditional_to_simplified(sentence)
    sentence = sentence.translate(F2H_ASCII_LETTERS).translate(F2H_DIGITS)
    sentence = RE_DATE.sub(replace_date, sentence)
    sentence = RE_DATE2.sub(replace_date2, sentence)
    sentence = RE_TIME_RANGE.sub(replace_time, sentence)
    sentence = RE_TIME.sub(replace_time, sentence)
    sentence = RE_TO_RANGE.sub(replace_to_range, sentence)
    sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
    sentence = replace_measure(sentence)
    while RE_ASMD.search(sentence):
        sentence = RE_ASMD.sub(replace_asmd, sentence)
    sentence = RE_POWER.sub(replace_power, sentence)
    sentence = RE_FRAC.sub(replace_frac, sentence)
    sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
    sentence = RE_MOBILE_PHONE.sub(replace_mobile, sentence)
    sentence = RE_TELEPHONE.sub(replace_phone, sentence)
    sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, sentence)
    sentence = RE_RANGE.sub(replace_range, sentence)
    sentence = RE_INTEGER.sub(replace_negative_num, sentence)
    sentence = RE_VERSION_NUM.sub(replace_version_num, sentence)
    sentence = RE_DECIMAL_NUM.sub(replace_number, sentence)
    sentence = RE_POSITIVE_QUANTIFIERS.sub(replace_positive_quantifier, sentence)
    sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
    sentence = RE_NUMBER.sub(replace_number, sentence)
    return _post_replace(sentence)


def normalize(text: str) -> list[str]:
    """``TextNormalizer().normalize``: the text split into sentences, each normalised."""
    return [normalize_sentence(sent) for sent in _split(text)]
