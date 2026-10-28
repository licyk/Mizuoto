"""GPT-SoVITS's language segmentation (upstream ``text/LangSegmenter/langsegmenter.py``; MIT, (c) 2024 RVC-Boss), with the
library it calls, split-lang 2.1.1 (``LangSplitter``; MIT, DoodleBears), and fast-langdetect 1.0.1's detector (MIT).

Upstream points fast-langdetect's global detector at its model and builds a ``LangSplitter`` per call (whose
constructor calls ``logging.basicConfig``). Here a ``LangSegmenter`` owns its fastText model (``lid.176.bin``, the
``gptsovits/encoders`` asset, through ``fasttext-predict``, or ``fasttext-predict-v2`` on 3.14) and its two BudouX
parsers; the splitting rules are split-lang's, as GPT-SoVITS configures them (``merge_across_digit=False``,
punctuation and newlines merged).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# -- split_lang/config.py --------------------------------------------------------------------------------------------

ZH_JA_LANG_MAP = {"zh": "zh", "yue": "zh", "wuu": "zh", "zh-cn": "zh", "zh-tw": "x", "ja": "ja"}
NO_ZH_JA_LANG_MAP = {"de": "de", "fr": "fr", "en": "en", "hr": "en"}
DEFAULT_LANG = "x"
SPLIT_PUNCTUATION = r"""〜~,.;:!?，。！？；：、·([{<（【《〈「『“‘)]}>）】》〉」』”’"-_——#$%&……￥'*+<=>?@[\]^_`{|}~"""

GSV_LANG_MAP = {"zh": "zh", "yue": "zh", "wuu": "zh", "zh-cn": "zh", "zh-tw": "x", "ko": "ko", "ja": "ja", "en": "en"}
"""``LangSegmenter.DEFAULT_LANG_MAP``."""

# -- split_lang/split/utils.py ---------------------------------------------------------------------------------------

_hangul = re.compile(r"[가-힯]")
_kana = re.compile(r"[぀-ヿ々]")
_zh_ja = re.compile(r"[一-鿿぀-ヿ々]")


def contains_hangul(text: str) -> bool:
    return bool(_hangul.search(text))


def contains_ja_kana(text: str) -> bool:
    return bool(_kana.search(text))


def contains_zh_ja(text: str) -> bool:
    return bool(_zh_ja.search(text))


# -- fast_langdetect/infer.py ----------------------------------------------------------------------------------------


class LangDetector:
    """``fast_langdetect.LangDetector`` with the full model (``lid.176.bin``) at ``model_path``: input over 80 characters
    truncated, newlines as spaces, mostly-uppercase text lowercased."""

    def __init__(self, model_path: str | Path, max_input_length: int | None = 80) -> None:
        self.model = _load_fasttext(Path(model_path))
        self.max_input_length = max_input_length

    def detect(self, text: str, k: int = 1, threshold: float = 0.0) -> list[dict[str, Any]]:
        if "\n" in text:
            text = text.replace("\n", " ")
        if self.max_input_length is not None and len(text) > self.max_input_length:
            text = text[: self.max_input_length]
        if text.isupper() or (len(re.findall(r"[A-Z]", text)) > 0.8 * len(re.findall(r"[A-Za-z]", text)) and len(text) > 5):
            text = text.lower()
        labels, scores = self.model.predict(text, k=k, threshold=threshold)
        results = [{"lang": label.replace("__label__", ""), "score": min(float(score), 1.0)} for label, score in zip(labels, scores, strict=True)]
        return sorted(results, key=lambda x: x["score"], reverse=True)


def _load_fasttext(path: Path) -> Any:
    import fasttext

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return fasttext.load_model(str(path.resolve()))
    except Exception:
        # fastText cannot open some non-ASCII paths on Windows; fast-langdetect then loads a temporary copy
        import os
        import shutil
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            shutil.copy2(path, tmp)
            return fasttext.load_model(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)


# -- split_lang/model.py ---------------------------------------------------------------------------------------------


class LangSectionType(Enum):
    ZH_JA = "zh_ja"
    KO = "ko"
    PUNCTUATION = "punctuation"
    NEWLINE = "newline"
    DIGIT = "digit"
    OTHERS = "others"
    ALL = "all"


@dataclass
class SubString:
    lang: str
    text: str
    index: int
    length: int


@dataclass
class SubStringSection:
    lang_section_type: LangSectionType
    text: str
    substrings: list[SubString] = field(default_factory=list)


# -- split_lang/split/splitter.py ------------------------------------------------------------------------------------


class LangSplitter:
    """``LangSplitter(lang_map=GSV_LANG_MAP)`` with ``merge_across_digit = False``."""

    def __init__(self, detector: LangDetector, lang_map: dict[str, str] = GSV_LANG_MAP) -> None:
        import budoux

        self.detector = detector
        self.lang_map = lang_map
        self.default_lang = DEFAULT_LANG
        self.punctuation = SPLIT_PUNCTUATION
        self.not_merge_punctuation = ""
        self.zh_parser = budoux.load_default_simplified_chinese_parser()
        self.jp_parser = budoux.load_default_japanese_parser()

    # split_lang/detect_lang/detector.py
    def _detect_lang_combined(self, text: str, lang_section_type: LangSectionType) -> str:
        if lang_section_type is LangSectionType.ZH_JA and contains_ja_kana(text):
            return "ja"
        return str(self.detector.detect(text)[0]["lang"]).lower()

    def _possible_detection_list(self, text: str) -> list[str]:
        text = text.replace("\n", "").strip()
        return [str(item["lang"]) for item in self.detector.detect(text, k=5, threshold=0.01)]

    def split_by_lang(self, text: str) -> list[SubString]:
        sections = self._split(self.pre_split(text))
        sections = self._merge_substrings_across_punctuation_based_on_sections(sections)
        sections = self._merge_substrings_across_newline_based_on_sections(sections)
        for section in sections:
            if section.lang_section_type == LangSectionType.ZH_JA:
                section.substrings = self._special_merge_for_zh_ja(section.substrings)
        return [substr for section in sections for substr in section.substrings]

    def pre_split(self, text: str) -> list[SubStringSection]:
        text = text.strip()
        sections: list[SubStringSection] = []
        current_lang = LangSectionType.OTHERS
        current_text: list[str] = []

        def add_substring(lang_section_type: LangSectionType) -> None:
            if current_text:
                sections.append(SubStringSection(lang_section_type=lang_section_type, text="".join(current_text), substrings=[]))
                current_text.clear()

        for index, char in enumerate(text):
            if contains_zh_ja(char):
                if current_lang != LangSectionType.ZH_JA:
                    add_substring(current_lang)
                    current_lang = LangSectionType.ZH_JA
            elif contains_hangul(char):
                if current_lang != LangSectionType.KO:
                    add_substring(current_lang)
                    current_lang = LangSectionType.KO
            elif char.isdigit():
                if current_lang != LangSectionType.DIGIT:
                    add_substring(current_lang)
                    current_lang = LangSectionType.DIGIT
            elif char in self.punctuation:
                if not (char == "'" and index > 0 and text[index - 1].isspace() is False):
                    add_substring(current_lang)
                    current_lang = LangSectionType.PUNCTUATION
            elif char.isspace():
                if char in ["\n", "\r"]:
                    add_substring(current_lang)
                    current_lang = LangSectionType.NEWLINE
            elif current_lang != LangSectionType.OTHERS:
                add_substring(current_lang)
                current_lang = LangSectionType.OTHERS
            current_text.append(char)
        add_substring(current_lang)
        return sections

    def _split(self, pre_split_section: list[SubStringSection]) -> list[SubStringSection]:
        section_index = 0
        for section in pre_split_section:
            section_len = len(section.text)
            if section.lang_section_type is LangSectionType.PUNCTUATION:
                section.substrings.append(SubString(text=section.text, lang="punctuation", index=section_index, length=section_len))
            elif section.lang_section_type is LangSectionType.DIGIT:
                section.substrings.append(SubString(text=section.text, lang="digit", index=section_index, length=section_len))
            elif section.lang_section_type is LangSectionType.NEWLINE:
                section.substrings.append(SubString(text=section.text, lang="newline", index=section_index, length=section_len))
            else:
                if section.lang_section_type is LangSectionType.ZH_JA:
                    substrings_with_lang = self._init_substr_lang(self._parse_zh_ja(section.text), LangSectionType.ZH_JA, ZH_JA_LANG_MAP)
                elif section.lang_section_type is LangSectionType.KO:
                    substrings_with_lang = [SubString(text=section.text, lang="ko", index=section_index, length=section_len)]
                else:
                    substrings_with_lang = self._init_substr_lang(self._parse_without_zh_ja(section.text), LangSectionType.OTHERS)
                for substr in substrings_with_lang:
                    substr.index += section_index
                section.substrings = substrings_with_lang
            section_index += section_len
        return self._smart_merge_all(pre_split_section)

    def _smart_merge_all(self, pre_split_section: list[SubStringSection]) -> list[SubStringSection]:
        for section in pre_split_section:
            if section.lang_section_type is LangSectionType.PUNCTUATION or section.lang_section_type is LangSectionType.NEWLINE:
                continue
            section.substrings = self._smart_merge(section.substrings, section.lang_section_type)
        return pre_split_section

    def _parse_without_zh_ja(self, text: str) -> list[str]:
        words: list[str] = []
        exist_space = False
        chars: list[str] = []
        for char in text:
            if char.isspace() is False:
                if exist_space:
                    words.append("".join(chars))
                    chars.clear()
                    exist_space = False
                chars.append(char)
            else:
                exist_space = True
                chars.append(char)
        if len(chars) > 0:
            words.append("".join(chars))
        return words

    def _parse_zh_ja(self, text: str) -> list[str]:
        splitted_texts_zh_jp: list[str] = []
        for substring in self.jp_parser.parse(text):
            splitted_texts_zh_jp.extend(self.zh_parser.parse(substring))
        new_substrings = [splitted_texts_zh_jp[0]]
        for substring in splitted_texts_zh_jp[1:]:
            is_left_ja_kana = contains_ja_kana(new_substrings[-1])
            is_cur_ja_kana = contains_ja_kana(substring)
            is_both_same_lang = is_left_ja_kana == is_cur_ja_kana
            is_both_ja_kana = is_left_ja_kana is True and is_both_same_lang
            is_both_zh = is_left_ja_kana is False and is_both_same_lang
            if is_both_ja_kana or (is_both_zh and len(substring) == 1):
                new_substrings[-1] += substring
            else:
                new_substrings.append(substring)
        if len(new_substrings) >= 2 and len(new_substrings[0]) == 1:
            new_substrings[1] = new_substrings[0] + new_substrings[1]
            new_substrings = new_substrings[1:]
        return new_substrings

    def _smart_merge(self, substr_list: list[SubString], lang_section_type: LangSectionType) -> list[SubString]:
        # split-lang's ``_get_languages`` computes languages it never stores: it changes nothing and is left out
        lang_text_list = self._merge_substrings(substr_list)
        lang_text_list = self._merge_middle_substr_to_two_side(lang_text_list)
        lang_text_list = self._merge_substrings(lang_text_list)
        lang_text_list = self._merge_middle_substr_to_two_side(lang_text_list)
        lang_text_list = self._merge_substrings(lang_text_list)
        lang_text_list = self._fill_unknown_language(lang_text_list)
        lang_text_list = self._merge_side_substr_to_near(lang_text_list)
        lang_text_list = self._merge_substrings(lang_text_list)
        lang_text_list = self._merge_middle_substr_to_two_side(lang_text_list)
        return self._merge_substrings(lang_text_list)

    def _is_merge_middle_to_two_side(self, left: SubString, middle: SubString, right: SubString) -> bool:
        is_middle_contains_no_kana = not contains_ja_kana(middle.text)
        middle_lang_is_possible_the_same_as_left = left.lang in self._possible_detection_list(middle.text) and is_middle_contains_no_kana
        middle_lang_is_x = middle.lang == "x"
        is_middle_short_and_two_side_long = middle.length <= 4 and left.length + right.length >= 6 and is_middle_contains_no_kana
        return middle_lang_is_possible_the_same_as_left or middle_lang_is_x or is_middle_short_and_two_side_long

    def _is_cur_short_and_near_long(self, cur: SubString, near: SubString) -> bool:
        is_both_contains_no_kana = not contains_ja_kana(cur.text) and not contains_ja_kana(near.text)
        is_near_str_is_long_zh = near.length > cur.length and near.length >= 6 and near.lang == "zh"
        is_cur_short_and_near_is_zh_and_long = is_near_str_is_long_zh and is_both_contains_no_kana
        cur_lang_is_possible_the_same_as_near = near.lang in self._possible_detection_list(cur.text) and is_both_contains_no_kana
        return is_cur_short_and_near_is_zh_and_long or cur_lang_is_possible_the_same_as_near

    def _merge_middle_substr_to_two_side(self, substrings: list[SubString]) -> list[SubString]:
        substr_len = len(substrings)
        if substr_len <= 2:
            return substrings
        for index in range(substr_len - 2):
            left_block = substrings[index]
            middle_block = substrings[index + 1]
            right_block = substrings[index + 2]
            if (
                left_block.lang == right_block.lang
                and left_block.lang != "x"
                and left_block.lang != "newline"
                and self._is_merge_middle_to_two_side(left_block, middle_block, right_block)
            ):
                substrings[index + 1].lang = left_block.lang
        return substrings

    def _merge_side_substr_to_near(self, substrings: list[SubString]) -> list[SubString]:
        is_lang_x = substrings[0].lang == "x"
        is_cur_short_and_near_long = False
        is_possible_same_lang_with_near = False
        if len(substrings) >= 2:
            is_cur_short_and_near_long = self._is_cur_short_and_near_long(substrings[0], substrings[1])
            is_possible_same_lang_with_near = substrings[1].lang in self._possible_detection_list(substrings[0].text) and substrings[1].length <= 5
        if is_lang_x or is_cur_short_and_near_long or is_possible_same_lang_with_near:
            substrings[0].lang = self._get_nearest_lang_with_direction(substrings, 0, search_left=False)
        is_lang_x = substrings[-1].lang == "x"
        is_cur_short_and_near_long = False
        is_possible_same_lang_with_near = False
        if len(substrings) >= 2:
            is_cur_short_and_near_long = self._is_cur_short_and_near_long(substrings[-1], substrings[-2])
            is_possible_same_lang_with_near = substrings[-2].lang in self._possible_detection_list(substrings[-1].text) and substrings[-2].length <= 5
        if is_lang_x or is_cur_short_and_near_long or is_possible_same_lang_with_near:
            substrings[-1].lang = self._get_nearest_lang_with_direction(substrings, len(substrings) - 1, search_left=True)
        return substrings

    def _fill_unknown_language(self, substrings: list[SubString]) -> list[SubString]:
        for index, substr in enumerate(substrings):
            if substr.lang == "x":
                if index == 0:
                    substrings[index].lang = self._get_nearest_lang_with_direction(substrings, index, search_left=False)
                elif index == len(substrings) - 1:
                    substrings[index].lang = self._get_nearest_lang_with_direction(substrings, index, search_left=True)
                else:
                    substrings[index].lang = self._get_nearest_lang_with_direction(substrings, index, self._get_merge_direction(substrings, index))
        return substrings

    @staticmethod
    def _get_nearest_lang_with_direction(substrings: list[SubString], index: int, search_left: bool) -> str:
        if search_left:
            for i in range(1, len(substrings)):
                left_i_index = index - i
                if left_i_index >= 0 and substrings[left_i_index].lang != "x" and substrings[left_i_index].lang != "digit":
                    return substrings[left_i_index].lang
        else:
            for i in range(1, len(substrings)):
                right_i_index = index + i
                if right_i_index < len(substrings) and substrings[right_i_index].lang != "x" and substrings[right_i_index].lang != "digit":
                    return substrings[right_i_index].lang
        return substrings[index].lang

    @staticmethod
    def _get_merge_direction(substrings: list[SubString], index: int) -> bool:
        if index == 0:
            return False
        if index == len(substrings) - 1:
            return True
        return len(substrings[index - 1].text) >= len(substrings[index + 1].text)

    @staticmethod
    def _merge_substrings(substrings: list[SubString]) -> list[SubString]:
        smart_concat_result: list[SubString] = []
        lang = ""
        for block in substrings:
            cur_lang = block.lang
            if cur_lang != lang:
                smart_concat_result.append(block)
            else:
                smart_concat_result[-1].text += block.text
                smart_concat_result[-1].length += block.length
            lang = cur_lang
        return smart_concat_result

    def _special_merge_for_zh_ja(self, substrings: list[SubString]) -> list[SubString]:
        new_substrings: list[SubString] = []
        if len(substrings) == 1:
            return substrings
        lengths = {"zh": 0, "ja": 0, "x": 0, "digit": 0, "punctuation": 0, "newline": 0}
        index = 0
        while index < len(substrings):
            current_block = substrings[index]
            lengths[current_block.lang] += current_block.length
            if index == 0:
                if substrings[index + 1].lang in ["zh", "ja"] and substrings[index].lang in ["zh", "ja", "x"] and substrings[index].length * 10 < substrings[index + 1].length:
                    right_block = substrings[index + 1]
                    new_substrings.append(
                        SubString(
                            lang=right_block.lang,
                            text=current_block.text + right_block.text,
                            length=current_block.length + right_block.length,
                            index=current_block.index,
                        )
                    )
                    index += 1
                else:
                    new_substrings.append(substrings[index])
            elif index == len(substrings) - 1:
                left_block = new_substrings[-1]
                if left_block.lang in ["zh", "ja"] and current_block.lang in ["zh", "ja", "x"] and current_block.length * 10 < left_block.length:
                    new_substrings[-1].text += current_block.text
                    new_substrings[-1].length += current_block.length
                    index += 1
                else:
                    new_substrings.append(substrings[index])
            elif (
                new_substrings[-1].lang == substrings[index + 1].lang
                and new_substrings[-1].lang in ["zh", "ja"]
                and substrings[index].lang != "en"
                and substrings[index].length * 10 < new_substrings[-1].length + substrings[index + 1].length
            ):
                right_block = substrings[index + 1]
                new_substrings[-1].text += current_block.text + right_block.text
                new_substrings[-1].length += current_block.length + right_block.length
                index += 1
            else:
                new_substrings.append(substrings[index])
            index += 1
        if lengths["x"] > 0:
            max_lang = max(lengths, key=lambda k: lengths[k])
            for substr in new_substrings:
                if substr.lang == "x":
                    substr.lang = max_lang
        if lengths["ja"] >= lengths["zh"] * 10:
            for substr in new_substrings:
                if substr.lang == "zh":
                    substr.lang = "ja"
        return self._merge_substrings(new_substrings)

    @staticmethod
    def _reindex(sections: list[SubStringSection]) -> None:
        for section_index, section in enumerate(sections):
            for substr_index, substr in enumerate(section.substrings):
                if substr_index == 0:
                    if section_index != 0:
                        previous = sections[section_index - 1].substrings[-1]
                        substr.index = previous.index + previous.length
                else:
                    previous = section.substrings[substr_index - 1]
                    substr.index = previous.index + previous.length

    @staticmethod
    def _merge_same_type(new_sections: list[SubStringSection]) -> list[SubStringSection]:
        merged = [new_sections[0]]
        for section in new_sections[1:]:
            if merged[-1].lang_section_type == section.lang_section_type:
                merged[-1].text += section.text
                merged[-1].substrings.extend(section.substrings)
            else:
                merged.append(section)
        return merged

    @staticmethod
    def _merge_substrings_across(substrings: list[SubString], kind: str) -> list[SubString]:
        """``_merge_substrings_across_punctuation`` / ``_newline``: a ``kind`` substring joins the one before it."""
        new_substrings: list[SubString] = []
        last_lang = ""
        for substring in substrings:
            if new_substrings:
                if substring.lang == kind:
                    new_substrings[-1].text += substring.text
                    new_substrings[-1].length += substring.length
                elif substring.lang == last_lang or last_lang == "":
                    new_substrings[-1].text += substring.text
                    new_substrings[-1].length += substring.length
                    new_substrings[-1].lang = substring.lang if new_substrings[-1].lang == kind else new_substrings[-1].lang
                else:
                    new_substrings.append(substring)
            else:
                new_substrings.append(substring)
            last_lang = substring.lang
        return new_substrings

    def _merge_substrings_across_newline_based_on_sections(self, sections: list[SubStringSection]) -> list[SubStringSection]:
        new_sections = [sections[0]]
        for current_section in sections[1:]:
            if new_sections[-1].lang_section_type == LangSectionType.NEWLINE:
                new_sections[-1].lang_section_type = current_section.lang_section_type
                new_sections[-1].text += current_section.text
                new_sections[-1].substrings.extend(current_section.substrings)
                subs = new_sections[-1].substrings
                for index in range(1, len(subs)):
                    subs[index].index = subs[index - 1].index + subs[index - 1].length
            elif current_section.lang_section_type == LangSectionType.NEWLINE:
                new_sections[-1].text += current_section.text
                new_sections[-1].substrings.extend(current_section.substrings)
                subs = new_sections[-1].substrings
                subs[-1].index = subs[-2].index + subs[-2].length
            else:
                new_sections.append(current_section)
        merged = self._merge_same_type(new_sections)
        self._reindex(merged)
        for section in merged:
            section.substrings = self._merge_substrings_across(section.substrings, "newline")
        return merged

    def _merge_substrings_across_punctuation_based_on_sections(self, sections: list[SubStringSection]) -> list[SubStringSection]:
        new_sections = [sections[0]]
        for current_section in sections[1:]:
            one_of_section_is_punctuation = new_sections[-1].lang_section_type == LangSectionType.PUNCTUATION or current_section.lang_section_type == LangSectionType.PUNCTUATION
            if not one_of_section_is_punctuation:
                new_sections.append(current_section)
            elif new_sections[-1].lang_section_type == LangSectionType.PUNCTUATION and new_sections[-1].substrings[0].text not in self.not_merge_punctuation:
                new_sections[-1].text += current_section.text
                new_sections[-1].lang_section_type = current_section.lang_section_type
                new_sections[-1].substrings[-1].text += current_section.substrings[0].text
                new_sections[-1].substrings[-1].length += current_section.substrings[0].length
                new_sections[-1].substrings[-1].lang = current_section.substrings[0].lang
                new_sections[-1].substrings.extend(current_section.substrings[1:])
                subs = new_sections[-1].substrings
                for index in range(1, len(subs)):
                    subs[index].index = subs[index - 1].index + subs[index - 1].length
            elif current_section.lang_section_type == LangSectionType.PUNCTUATION and current_section.substrings[0].text not in self.not_merge_punctuation:
                new_sections[-1].text += current_section.text
                new_sections[-1].substrings[-1].text += current_section.substrings[0].text
                new_sections[-1].substrings[-1].length += current_section.substrings[0].length
                new_sections[-1].substrings.extend(current_section.substrings[1:])
                subs = new_sections[-1].substrings
                for index in range(1, len(subs)):
                    subs[index].index = subs[index - 1].index + subs[index - 1].length
            else:
                new_sections.append(current_section)
        merged = self._merge_same_type(new_sections)
        self._reindex(merged)
        for section in merged:
            section.substrings = self._merge_substrings_across(section.substrings, "punctuation")
        return merged

    def _init_substr_lang(self, texts: list[str], lang_section_type: LangSectionType, lang_map: dict[str, str] | None = None) -> list[SubString]:
        substrings: list[SubString] = []
        substring_index = 0
        lang_map = self.lang_map if lang_map is None else lang_map
        for text in texts:
            length = len(text)
            cur_lang = lang_map.get(self._detect_lang_combined(text, lang_section_type), self.default_lang)
            substrings.append(SubString(lang=cur_lang, text=text, length=length, index=substring_index))
            substring_index += length
        return substrings


# -- GPT_SoVITS/text/LangSegmenter/langsegmenter.py ------------------------------------------------------------------

_FULL_EN = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9\s -~ -⁯　-〿＀-￯]+$")
_CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DB5),
    (0x20000, 0x2A6DD),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
    (0x2EBF0, 0x2EE5D),
]
_CJK_EXTRA = re.compile(r"[0-9、-〜。！？.!?… /]+$")
_JA_RUN = re.compile(r"([ぁ-ゖ゙゚ァ-ヺー]+(?:[0-9、-〜。！？.!?… ]+[ぁ-ゖ゙゚ァ-ヺー]*)*)")
_KO_RUN = re.compile(r"([ᄀ-ᇿ㄰-㆏가-힯]+(?:[0-9、-〜。！？.!?… ]+[ᄀ-ᇿ㄰-㆏가-힯]*)*)")


def full_en(text: str) -> bool:
    return bool(_FULL_EN.match(text))


def full_cjk(text: str) -> str:
    out = ""
    for char in text:
        code_point = ord(char)
        if any(start <= code_point <= end for start, end in _CJK_RANGES) or _CJK_EXTRA.match(char):
            out += char
    return out


def split_jako(tag_lang: str, item: dict[str, str]) -> list[dict[str, str]]:
    pattern = _JA_RUN if tag_lang == "ja" else _KO_RUN
    lang_list: list[dict[str, str]] = []
    tag = 0
    for match in pattern.finditer(item["text"]):
        if match.start() > tag:
            lang_list.append({"lang": item["lang"], "text": item["text"][tag : match.start()]})
        tag = match.end()
        lang_list.append({"lang": tag_lang, "text": item["text"][match.start() : match.end()]})
    if tag < len(item["text"]):
        lang_list.append({"lang": item["lang"], "text": item["text"][tag : len(item["text"])]})
    return lang_list


def merge_lang(lang_list: list[dict[str, str]], item: dict[str, str]) -> list[dict[str, str]]:
    if lang_list and item["lang"] == lang_list[-1]["lang"]:
        lang_list[-1]["text"] += item["text"]
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter:
    """``LangSegmenter.getTexts(text, default_lang)`` → ``[{"lang": …, "text": …}]``; ``model_path`` is ``lid.176.bin``."""

    def __init__(self, model_path: str | Path) -> None:
        self.splitter = LangSplitter(LangDetector(model_path))

    def get_texts(self, text: str, default_lang: str = "") -> list[dict[str, str]]:
        substr = self.splitter.split_by_lang(text=text)
        lang_list: list[dict[str, str]] = []
        have_num = False
        for item in substr:
            dict_item = {"lang": item.lang, "text": item.text}
            if dict_item["lang"] == "digit":
                if default_lang != "":
                    dict_item["lang"] = default_lang
                else:
                    have_num = True
                lang_list = merge_lang(lang_list, dict_item)
                continue
            if full_en(dict_item["text"]):
                dict_item["lang"] = "en"
                lang_list = merge_lang(lang_list, dict_item)
                continue
            if default_lang != "":
                dict_item["lang"] = default_lang
                lang_list = merge_lang(lang_list, dict_item)
                continue
            ja_list: list[dict[str, str]] = []
            if dict_item["lang"] != "ja":
                ja_list = split_jako("ja", dict_item)
            if not ja_list:
                ja_list.append(dict_item)
            ko_list: list[dict[str, str]] = []
            temp_list: list[dict[str, str]] = []
            for ko_item in ja_list:
                if ko_item["lang"] != "ko":
                    ko_list = split_jako("ko", ko_item)
                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)
            if len(temp_list) == 1:
                if dict_item["lang"] == "x":
                    cjk_text = full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                lang_list = merge_lang(lang_list, dict_item)
                continue
            for temp_item in temp_list:
                if temp_item["lang"] == "x":
                    cjk_text = full_cjk(temp_item["text"])
                    lang_list = merge_lang(lang_list, {"lang": "zh", "text": cjk_text} if cjk_text else temp_item)
                else:
                    lang_list = merge_lang(lang_list, temp_item)
        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "digit":
                    if default_lang:
                        temp_item["lang"] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item["lang"] = temp_list[1]["lang"]
                    elif lang_list and i < len(temp_list) - 1:
                        stops = [",", ".", "!", "?", "，", "。", "！", "？"]
                        if lang_list[-1]["lang"] == temp_list[i + 1]["lang"]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif lang_list[-1]["text"][-1] in stops:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                        elif temp_list[i + 1]["text"][0] in stops or temp_item["text"][-1] in ["。", "."] or len(lang_list[-1]["text"]) >= len(temp_list[i + 1]["text"]):
                            temp_item["lang"] = lang_list[-1]["lang"]
                        else:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                    else:
                        temp_item["lang"] = "zh"
                lang_list = merge_lang(lang_list, temp_item)
        temp_list = lang_list
        lang_list = []
        for temp_item in temp_list:
            if temp_item["lang"] == "x":
                if lang_list:
                    temp_item["lang"] = lang_list[-1]["lang"]
                elif len(temp_list) > 1:
                    temp_item["lang"] = temp_list[1]["lang"]
                else:
                    temp_item["lang"] = "zh"
            lang_list = merge_lang(lang_list, temp_item)
        return lang_list
