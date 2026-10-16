"""GPT-SoVITS's native text front end (mizuoto.engines.gptsovits.text) against values upstream computes.

The expected values come from upstream's code in its own environment (GPT-SoVITS at the vendored commit, nltk 3.10.3,
g2p_en 2.1.0, inflect 7.5). The unit tests need no downloaded assets; the end-to-end ones use the ``gptsovits/*``
assets under ``$MIZUOTO_TEST_DATA_DIR/models/_assets`` (default ``~/.local/share/mizuoto``) and skip without them. The
whole corpus is compared in the refactor's experiment workspace (debug/gsv-text).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mizuoto.engines.gptsovits.text import segmentation
from mizuoto.engines.gptsovits.text.symbols import SYMBOLS, cleaned_text_to_sequence

if TYPE_CHECKING:
    from mizuoto.engines.gptsovits.text import TextFrontend

ASSETS = Path(os.environ.get("MIZUOTO_TEST_DATA_DIR") or Path.home() / ".local/share/mizuoto") / "models" / "_assets" / "gptsovits"


# -- symbols ---------------------------------------------------------------------------------------------------------


def test_symbol_table_is_upstreams() -> None:
    assert len(SYMBOLS) == 732
    assert hashlib.sha256(json.dumps(SYMBOLS).encode()).hexdigest() == "58e4e8a92c677b8ac9b500db981b43a5eae25f17a14eff48dc429f4deeaa667b"
    assert cleaned_text_to_sequence(["n", "i2", "h", "ao3", ".", "AY1", "UNK", "SP3"]) == [227, 167, 158, 119, 3, 22, 86, 79]


# -- segmentation ----------------------------------------------------------------------------------------------------


def test_cut_methods() -> None:
    text = "你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。今天天气不错。"
    assert segmentation.cut0(text) == text
    assert segmentation.cut0("…, ") == "\n"
    assert segmentation.cut1(text).split("\n") == ["你好，我是小明。你好，我是小红。", "你好，我是小刚。你好，我是小张。今天天气不错。"]  # four marks a piece, the tail joined
    assert segmentation.cut2(text) == "你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。今天天气不错。"
    assert segmentation.cut3(text).split("\n") == ["你好，我是小明", "你好，我是小红", "你好，我是小刚", "你好，我是小张", "今天天气不错"]
    assert segmentation.cut4("It is 3.5 m. Then. Done") == "It is 3.5 m\n Then\n Done"
    assert segmentation.cut5("It is 3.5 m, then; done!") == "It is 3.5 m,\n then;\n done!"


def test_pre_seg_text() -> None:
    assert segmentation.pre_seg_text("你好", "zh", "cut1") == ["。你好。"]
    assert segmentation.pre_seg_text("Hi", "en", "cut1") == [".Hi."]
    assert segmentation.pre_seg_text("\n\n", "zh", "cut1") == []
    assert segmentation.pre_seg_text("第一行\n第二行\n第三行", "zh", "cut0") == ["第一行第二行第三行。"]  # pieces under 5 characters merged
    with pytest.raises(segmentation.EmptyTextError):
        segmentation.pre_seg_text("...", "zh", "cut0")
    assert segmentation.pre_seg_text("！？", "zh", "cut1") == []  # full-width marks are not cut0-4's punctuation; \W+ drops the piece
    long = segmentation.pre_seg_text("这是一个没有标点的超长句子" * 45, "zh", "cut0")
    assert [len(p) for p in long] == [0, 585, 1]  # upstream's split_big_text: no mark to cut at
    assert segmentation.replace_consecutive_punctuation("你好!!!世界,.") == "你好!世界,"


# -- normalisation ---------------------------------------------------------------------------------------------------


def test_chinese_normalisation() -> None:
    from mizuoto.engines.gptsovits.text.zh_normalization import normalize

    expected = {
        "今天是2024年3月15日，星期五。": ["今天是二零二四年三月十五日，", "星期五。"],
        "现在是下午3点45分。": ["现在是下午三点四十五分。"],
        "增长率达到了35.6%。": ["增长率达到了百分之三十五点六。"],
        "他的手机号码是13812345678。": ["他的手机号码是幺三八幺二三四五六七八。"],
        "温度从-5°C升到了25°C。": ["温度从零下五度升到了二十五度。"],
        "我们需要100~200个志愿者。": ["我们需要一百到二百个志愿者。"],
        "这个数学题是3+5=8。": ["这个数学题是三加五等于八。"],
        "版本号是v2.1.3。": ["版本号是v二点一点三。"],
        "1/2的人认为可行。": ["二分之一的人认为可行。"],
        "這是一個繁體中文的句子。": ["这是一个繁体中文的句子。"],
    }
    for text, sentences in expected.items():
        assert normalize(text) == sentences, text


def test_english_normalisation() -> None:
    pytest.importorskip("inflect")
    from mizuoto.engines.gptsovits.text.en_normalization import EnglishNormalizer

    normalize = EnglishNormalizer()
    expected = {
        "I have $250 in my pocket.": "I have two hundred fifty dollars in my pocket.",
        "The meeting starts at 9:30 and ends at 13:45.": "The meeting starts at nine thirty a.m. and ends at one forty-five p.m..",
        "In 1999, the population was 6,000,000.": "In nineteen ninety-nine, the population was six million.",
        "1st, 2nd, 3rd, 22nd.": "first, second, third, twenty-second.",
        "It costs £7.14.": "It costs seven pounds and fourteen pence.",
        "He ran 5km in 25 min.": "He ran five kilometers in twenty-five min.",
        "3/4 of the total.": "three fourths of the total.",
        "It was 50% off.": "It was fifty percent off.",
        "The café served crème brûlée.": "The cafe served creme brulee.",
        "NASA and the FBI.": "N A S A and the F B I.",
    }
    for text, out in expected.items():
        assert normalize(text) == out, text


def test_tweet_tokenizer() -> None:
    pytest.importorskip("regex")
    from mizuoto.engines.gptsovits.text.en_tagger import tweet_tokenize

    assert tweet_tokenize("Hello, world! It's a well-known fact.") == ["Hello", ",", "world", "!", "It's", "a", "well-known", "fact", "."]
    assert tweet_tokenize("Call me at 555-123-4567 or +1 800 555 0199.") == ["Call", "me", "at", "555-123-4567", "or", "+1 800 555 0199", "."]
    assert tweet_tokenize("e.g. I'm here... OK?") == ["e", ".", "g", ".", "I'm", "here", "...", "OK", "?"]
    assert tweet_tokenize("rock 'n' roll, John's car & Mary's bike") == ["rock", "'", "n", "'", "roll", ",", "John's", "car", "&", "Mary's", "bike"]


# -- English ---------------------------------------------------------------------------------------------------------


def test_cmudict_and_gru() -> None:
    from mizuoto.engines.gptsovits.text.english import GraphemeGRU, read_cmudict

    cmu = read_cmudict()
    assert cmu["hello"] == [["HH", "AH0", "L", "OW1"]]
    assert cmu["chatgpt"] == [["CH", "AE1", "T", "JH", "IY1", "P", "IY1", "T", "IY1"]]  # engdict-hot.rep
    gru = GraphemeGRU()
    expected = {
        "mizuoto": ["M", "IY0", "Z", "W", "OW1", "T", "OW0"],
        "shinrin": ["SH", "IH1", "N", "R", "IH0", "N"],
        "zyxwvutsr": ["Z", "IH1", "K", "S", "V", "W", "EH2", "S", "T", "ER0", "Z"],
        "blorptastic": ["B", "L", "AO2", "R", "P", "T", "AE1", "S", "T", "IH0", "K"],
        "galumphed": ["G", "AE1", "L", "AH0", "M", "F", "T"],
        "tulgey": ["T", "AH1", "L", "G", "IY0"],
        "hetzner": ["HH", "EH1", "T", "S", "N", "ER0"],
    }
    for word, phones in expected.items():
        assert gru.predict(word) == phones, word


def _tagger_dir() -> Path:
    folder = ASSETS / "nltk-data" / "nltk_data" / "taggers" / "averaged_perceptron_tagger_eng"
    if not folder.is_dir():
        pytest.skip("the gptsovits/nltk-data asset is not downloaded")
    return folder


def test_pos_tagger() -> None:
    pytest.importorskip("regex")
    from mizuoto.engines.gptsovits.text.en_tagger import PerceptronTagger, tweet_tokenize

    tagger = PerceptronTagger(_tagger_dir())
    assert tagger.tag(tweet_tokenize("Hello, world! It's a well-known fact.")) == [
        ("Hello", "NNP"),
        (",", ","),
        ("world", "NN"),
        ("!", "."),
        ("It's", "VBZ"),
        ("a", "DT"),
        ("well-known", "JJ"),
        ("fact", "NN"),
        (".", "."),
    ]
    assert [t for _, t in tagger.tag(tweet_tokenize("rock 'n' roll, John's car & Mary's bike"))] == ["NN", "''", "JJ", "POS", "NN", ",", "NNP", "NN", "CC", "NNP", "NN"]


def test_english_g2p() -> None:
    pytest.importorskip("wordsegment")
    pytest.importorskip("inflect")
    from mizuoto.engines.gptsovits.text.english import EnglishG2P

    g2p = EnglishG2P(_tagger_dir())
    # homographs by POS ("read" as a verb in the present, "wind" as a noun and a verb), unknown words through the GRU
    assert cleaned_text_to_sequence(g2p.g2p(g2p.text_normalize("Please read the instructions carefully."))) == [
        73, 62, 58, 93, 74, 35, 26, 27, 12, 56, 64, 75, 80, 74, 13, 61, 76, 12, 64, 93, 61, 35, 74, 49, 12, 62, 57, 3,
    ]  # fmt: skip
    assert cleaned_text_to_sequence(g2p.g2p(g2p.text_normalize("The wind was too strong to wind the sail."))) == [
        27, 12, 91, 22, 64, 26, 91, 7, 93, 80, 88, 75, 80, 74, 16, 65, 80, 88, 91, 22, 64, 26, 27, 12, 75, 42, 62, 3,
    ]  # fmt: skip
    assert cleaned_text_to_sequence(g2p.g2p(g2p.text_normalize("The kodama whispered through the shinrin forest."))) == [
        27, 12, 61, 67, 26, 7, 63, 12, 91, 55, 75, 73, 38, 26, 81, 74, 88, 27, 12, 76, 55, 64, 7, 74, 22, 35, 64, 49, 16, 74, 12, 75, 80, 3,
    ]  # fmt: skip


# -- the whole front end (all assets) --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frontend(tmp_path_factory: pytest.TempPathFactory) -> TextFrontend:
    for module in ("torch", "tokenizers", "jieba", "opencc", "pypinyin", "budoux", "fasttext", "wordsegment", "inflect", "regex"):
        pytest.importorskip(module)
    if not all((ASSETS / name).is_dir() for name in ("encoders", "g2pw", "nltk-data", "open-jtalk")):
        pytest.skip("the gptsovits assets are not downloaded")
    from mizuoto.engines.gptsovits.text import TextFrontend

    assets = {f"gptsovits/{name}": ASSETS / name for name in ("encoders", "g2pw", "nltk-data", "open-jtalk")}
    return TextFrontend.from_assets(assets, cache_dir=tmp_path_factory.mktemp("gptsovits-text"))


UPSTREAM = [
    # (text after pre_seg_text, language, phone ids, normalised text)
    (
        "女儿和孙儿都很可爱，婴儿在睡觉。",
        "zh",
        [227, 298, 33, 151, 158, 131, 250, 285, 33, 151, 127, 240, 158, 141, 222, 132, 5, 105, 1, 318, 201, 33, 151, 319, 105, 251, 283, 221, 189, 3],
        "女儿和孙儿都很可爱,婴儿在睡觉.",
    ),
    (
        "不要说不，不会就学，不对就改。",
        "zh",
        [122, 256, 318, 120, 251, 290, 122, 258, 1, 122, 256, 158, 283, 221, 219, 317, 307, 1, 122, 256, 127, 283, 221, 219, 156, 104, 3],
        "不要说不,不会就学,不对就改.",
    ),
    ("他还没还钱。", "zh", [252, 97, 158, 103, 225, 136, 158, 271, 247, 177, 3], "他还没还钱."),
    ("。啊。", "zh", [3, 5, 101, 3], ".啊."),
    ("。^_^。", "zh", [3], "."),
    ("MyGO?你也喜欢まいご吗？", "auto", [63, 22, 60, 58, 68, 4, 227, 167, 318, 30, 317, 168, 158, 274, 225, 96, 323, 160, 156, 229, 225, 99, 4], "My G O?你也喜欢まいご吗?"),
    ("你食咗飯未呀？", "yue", [587, 480, 667, 533, 731, 597, 512, 390, 580, 481, 360, 356, 4], "你食咗饭未呀?"),
]


@pytest.mark.parametrize(("text", "language", "phones", "norm_text"), UPSTREAM, ids=[f"{lang}:{t[:8]}" for t, lang, _, _ in UPSTREAM])
def test_front_end_matches_upstream(frontend: TextFrontend, text: str, language: str, phones: list[int], norm_text: str) -> None:
    segment = frontend.segment(text, language)
    assert (segment.phones, segment.norm_text) == (phones, norm_text)


def test_japanese_matches_upstream(frontend: TextFrontend) -> None:
    pytest.importorskip("pyopenjtalk")
    segment = frontend.segment("こんにちは、今日はいい天気ですね。", "ja")
    assert segment.phones == [
        222,
        229,
        322,
        64,
        227,
        160,
        125,
        160,
        316,
        96,
        1,
        223,
        229,
        323,
        229,
        316,
        96,
        86,
        160,
        323,
        160,
        86,
        252,
        129,
        323,
        64,
        222,
        160,
        127,
        129,
        250,
        254,
        227,
        129,
        3,
    ]


def test_word2ph_spans_the_normalised_text(frontend: TextFrontend) -> None:
    segment = frontend.segment("今天是2024年3月15日，星期五。", "zh")
    (piece,) = segment.pieces
    assert piece.word2ph is not None and len(piece.word2ph) == len(piece.norm_text) and sum(piece.word2ph) == len(piece.ids)


def test_korean_rules() -> None:
    pytest.importorskip("jamo")
    pytest.importorskip("mecab")
    from mizuoto.engines.gptsovits.text.korean import KoreanG2P

    # 읽다 → 익따, 넓다 → 널따, 밟다 → 밥따 (g2pk2's rules, on jamo strings)
    assert KoreanG2P().g2p("읽다, 넓다, 밟다.") == [*"ㅇㅣㄱㄸㅏ", ",", "空", *"ㄴㅓㄹㄸㅏ", ",", "空", *"ㅂㅏㅂㄸㅏ", "."]
