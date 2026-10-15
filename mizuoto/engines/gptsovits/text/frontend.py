"""GPT-SoVITS's text front end: ``text/cleaner.py`` and ``TTS_infer_pack/TextPreprocessor.py`` (MIT, (c) 2024 RVC-Boss).

- ``TextFrontend``: text → phones. ``clean_text(text, language)`` is upstream's ``clean_text`` (version 2 symbols), and
  ``pieces(text, language)`` is ``get_phones_and_bert`` without the BERT features: the language segmentation for
  every ``language`` (``auto``, ``auto_yue``, ``all_*``, ``en`` and the mixed ``zh``/``ja``/``ko``/``yue``), each piece
  cleaned, and the retry with a leading ``.`` when there are fewer than 6 phones.
- ``TextPreprocessor``: upstream's class of that name (``preprocess``, ``pre_seg_text``,
  ``segment_and_extract_feature_for_text``…), the BERT features computed with ``mizuoto.nn.transformer.bert``.

Each language's converter is built on first use, as upstream imports its module on first use; Japanese needs
``pyopenjtalk`` and Korean the Korean extra (``python-mecab-ko``, ``jamo``).
"""

from __future__ import annotations

import contextlib
import re
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.engines.gptsovits.text import segmentation
from mizuoto.engines.gptsovits.text.symbols import SYMBOL_SET, cleaned_text_to_sequence

if TYPE_CHECKING:
    import torch

    from mizuoto.nn.transformer.bert import BertModel

ENCODERS = "gptsovits/encoders"
G2PW_ASSET = "gptsovits/g2pw"
NLTK = "gptsovits/nltk-data"
OPEN_JTALK = "gptsovits/open-jtalk"

LANGUAGES = ("zh", "ja", "en", "ko", "yue")
"""``clean_text``'s languages for every version but v1; any other is cleaned as English ``" "``."""

SPECIAL = [("￥", "zh", "SP2"), ("^", "zh", "SP3")]
"""``cleaner.special``: in Chinese, these become a pause symbol instead of a comma."""

MIN_PHONES = 6


class FrontendError(ValueError):
    pass


def find_one(folder: Path, *names: str) -> Path:
    """The first match of ``names`` (glob patterns) under ``folder``, ``folder`` itself if it is one."""
    for name in names:
        if folder.match(name):
            return folder
        hits = sorted(folder.rglob(name))
        if hits:
            return hits[0]
    raise FrontendError(f"{folder} has no {' or '.join(names)}")


@dataclass(frozen=True)
class FrontendPaths:
    """Where the front end's models and data are: from the four ``gptsovits/*`` assets (``from_assets``)."""

    g2pw_dir: Path
    """``G2PWModel/`` (``g2pW.onnx`` and its tables)."""
    bert_dir: Path
    """``chinese-roberta-wwm-ext-large/`` (G2PW's tokeniser, and the BERT features)."""
    langdetect_model: Path
    """fastText's ``lid.176.bin``."""
    tagger_dir: Path
    """nltk's ``taggers/averaged_perceptron_tagger_eng``."""
    open_jtalk_dir: Path
    """``open_jtalk_dic_utf_8-1.11``."""

    @classmethod
    def from_assets(cls, assets: Mapping[str, str | Path]) -> FrontendPaths:
        missing = [a for a in (ENCODERS, G2PW_ASSET, NLTK, OPEN_JTALK) if a not in assets]
        if missing:
            raise FrontendError(f"GPT-SoVITS's text front end needs the assets {', '.join(missing)}")
        encoders = Path(assets[ENCODERS])
        return cls(
            g2pw_dir=find_one(Path(assets[G2PW_ASSET]), "G2PWModel"),
            bert_dir=find_one(encoders, "chinese-roberta-wwm-ext-large"),
            langdetect_model=find_one(encoders, "lid.176.bin"),
            tagger_dir=find_one(Path(assets[NLTK]), "averaged_perceptron_tagger_eng"),
            open_jtalk_dir=find_one(Path(assets[OPEN_JTALK]), "open_jtalk_dic_utf_8-*"),
        )


@dataclass
class Piece:
    """One language run of a segment, cleaned: ``lang`` as upstream's language list has it (``all_zh`` stays ``all_zh``)."""

    lang: str
    text: str
    phones: list[str]
    """Phone symbols."""
    ids: list[int]
    word2ph: list[int] | None
    """Phones per character of ``norm_text`` (Chinese and Cantonese only)."""
    norm_text: str


@dataclass
class Segment:
    """``get_phones_and_bert``'s result without the features: the pieces of the last attempt (after the ``.`` retry)."""

    text: str
    pieces: list[Piece] = field(default_factory=list)
    retried: bool = False

    @property
    def phones(self) -> list[int]:
        return [i for p in self.pieces for i in p.ids]

    @property
    def norm_text(self) -> str:
        return "".join(p.norm_text for p in self.pieces)


class TextFrontend:
    """Text → GPT-SoVITS phones. ``cache_dir`` holds the compiled Japanese user dictionary (a temporary folder by default)."""

    def __init__(self, paths: FrontendPaths, *, device: str = "cpu", cache_dir: str | Path | None = None) -> None:
        self.paths = paths
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "mizuoto-gptsovits"
        self._lock = threading.RLock()
        self._langseg: Any = None
        self._g2p: dict[str, Any] = {}

    @classmethod
    def from_assets(cls, assets: Mapping[str, str | Path], **kwargs: Any) -> TextFrontend:
        return cls(FrontendPaths.from_assets(assets), **kwargs)

    # -- the language converters, built on first use -------------------------------------------------------------

    @property
    def langseg(self) -> Any:
        with self._lock:
            if self._langseg is None:
                from mizuoto.engines.gptsovits.text.langseg import LangSegmenter

                self._langseg = LangSegmenter(self.paths.langdetect_model)
            return self._langseg

    def converter(self, language: str) -> Any:
        """The converter of ``language`` (``zh``, ``en``, ``ja``, ``ko``, ``yue``), built on first use."""
        with self._lock:
            g2p = self._g2p.get(language)
            if g2p is None:
                g2p = self._g2p[language] = self._build(language)
            return g2p

    def _build(self, language: str) -> Any:
        if language == "zh":
            import jieba
            import jieba.posseg

            from mizuoto.engines.gptsovits.text.chinese import ChineseG2P
            from mizuoto.engines.gptsovits.text.g2pw import G2PW

            tokenizer = jieba.Tokenizer()
            tokenizer.tmp_dir = str(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            g2pw = G2PW(self.paths.g2pw_dir, self.paths.bert_dir / "tokenizer.json", device=self.device)
            return ChineseG2P(g2pw, tokenizer, jieba.posseg.POSTokenizer(tokenizer))
        if language == "en":
            from mizuoto.engines.gptsovits.text.english import EnglishG2P

            return EnglishG2P(self.paths.tagger_dir)
        if language == "ja":
            from mizuoto.engines.gptsovits.text.japanese import JapaneseG2P

            return JapaneseG2P(self.paths.open_jtalk_dir, self.cache_dir)
        if language == "ko":
            from mizuoto.engines.gptsovits.text.korean import KoreanG2P

            return KoreanG2P()
        if language == "yue":
            from mizuoto.engines.gptsovits.text import cantonese

            return cantonese
        raise FrontendError(f"No converter for {language!r}")

    # -- cleaner.py ----------------------------------------------------------------------------------------------

    def _normalize(self, g2p: Any, text: str) -> str:
        normalize = getattr(g2p, "text_normalize", None)
        return normalize(text) if normalize is not None else text

    def clean_text(self, text: str, language: str) -> tuple[list[str], list[int] | None, str]:
        """``cleaner.clean_text(text, language, version)`` for v2 and later: (phone symbols, word2ph, normalised text)."""
        if language not in LANGUAGES:
            language = "en"
            text = " "
        for special_s, special_l, target_symbol in SPECIAL:
            if special_s in text and language == special_l:
                return self._clean_special(text, language, special_s, target_symbol)
        g2p = self.converter(language)
        norm_text = self._normalize(g2p, text)
        word2ph: list[int] | None
        if language in ("zh", "yue"):
            phones, word2ph = g2p.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif language == "en":
            phones = g2p.g2p(norm_text)
            if len(phones) < 4:
                phones = [",", *phones]
            word2ph = None
        else:
            phones = g2p.g2p(norm_text)
            word2ph = None
        return ["UNK" if ph not in SYMBOL_SET else ph for ph in phones], word2ph, norm_text

    def _clean_special(self, text: str, language: str, special_s: str, target_symbol: str) -> tuple[list[str], list[int] | None, str]:
        text = text.replace(special_s, ",")
        g2p = self.converter(language)
        norm_text = g2p.text_normalize(text)
        phones, word2ph = g2p.g2p(norm_text)
        new_ph = []
        for ph in phones:
            assert ph in SYMBOL_SET
            new_ph.append(target_symbol if ph == "," else ph)
        return new_ph, word2ph, norm_text

    # -- TextPreprocessor.get_phones_and_bert, the text half -----------------------------------------------------------

    def language_runs(self, text: str, language: str) -> list[tuple[str, str]]:
        """(language, text) runs as ``get_phones_and_bert`` makes them (after collapsing double spaces)."""
        text = re.sub(r" {2,}", " ", text)
        langlist: list[str] = []
        textlist: list[str] = []
        if language == "en":
            return [("en", text)]
        if language in ("all_zh", "all_yue", "all_ja", "all_ko"):
            default = {"all_zh": "zh", "all_yue": "zh", "all_ja": "ja", "all_ko": "ko"}[language]
            for tmp in self.langseg.get_texts(text, default):
                if language == "all_yue" and tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language in ("auto", "auto_yue"):
            for tmp in self.langseg.get_texts(text):
                if language == "auto_yue" and tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in self.langseg.get_texts(text):
                if langlist and ((tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en")):
                    textlist[-1] += tmp["text"]
                    continue
                # CJK characters cannot be told apart: the user's language is taken
                langlist.append(tmp["lang"] if tmp["lang"] == "en" else language)
                textlist.append(tmp["text"])
        return list(zip(langlist, textlist, strict=True))

    def clean_piece(self, text: str, language: str) -> Piece:
        """``clean_text_inf``: ``all_`` dropped from the language, symbols to ids."""
        phones, word2ph, norm_text = self.clean_text(text, language.replace("all_", ""))
        return Piece(language, text, phones, cleaned_text_to_sequence(phones), word2ph, norm_text)

    def segment(self, text: str, language: str) -> Segment:
        """``get_phones_and_bert(text, language, version)`` without features, retried with ``"." + text`` under 6 phones."""
        with self._lock:
            pieces = [self.clean_piece(t, lang) for lang, t in self.language_runs(text, language)]
            result = Segment(text, pieces)
            if len(result.phones) < MIN_PHONES:
                text = "." + text
                pieces = [self.clean_piece(t, lang) for lang, t in self.language_runs(text, language)]
                result = Segment(text, pieces, retried=True)
            return result


def check_version(version: str) -> None:
    if version == "v1":
        raise FrontendError("GPT-SoVITS v1's symbol table is not supported")


class TextPreprocessor:
    """Upstream's ``TextPreprocessor``: ``bert_model`` is a ``BertModel`` (``load_bert_model``), or ``None`` to get phones only
    (``bert_features`` are then ``None``). ``assets`` maps the ``gptsovits/*`` asset ids to their folders."""

    def __init__(
        self,
        bert_model: BertModel | None,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        assets: Mapping[str, str | Path] | None = None,
        *,
        frontend: TextFrontend | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if frontend is None:
            if assets is None:
                raise FrontendError("TextPreprocessor needs the assets or a TextFrontend")
            frontend = TextFrontend.from_assets(assets, device="cpu", cache_dir=cache_dir)
        self.frontend = frontend
        self.bert_model = bert_model
        self.device = device
        self.dtype = dtype
        self.bert_lock = threading.RLock()
        self._tokenizer: Any = None
        self.bert_scope: Any = contextlib.nullcontext
        """A context manager factory around each BERT call: the engine places the model there (``ctx.use``)."""

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(self.frontend.paths.bert_dir / "tokenizer.json"))
        return self._tokenizer

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> list[dict[str, Any]]:
        text = self.replace_consecutive_punctuation(text)
        result = []
        for piece in self.pre_seg_text(text, lang, text_split_method):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(piece, lang, version)
            if phones is None or norm_text == "":
                continue
            result.append({"phones": phones, "bert_features": bert_features, "norm_text": norm_text})
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str) -> list[str]:
        return segmentation.pre_seg_text(text, lang, text_split_method)

    def segment_and_extract_feature_for_text(self, text: str, language: str, version: str = "v2") -> tuple[Any, Any, Any]:
        return self.get_phones_and_bert(text, language, version)

    def segments(self, text: str, language: str, version: str = "v2") -> Segment:
        """The text half of ``get_phones_and_bert``: per piece, its language, phone ids, word2ph and normalised text."""
        check_version(version)
        return self.frontend.segment(text, language)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False) -> tuple[list[int], Any, str]:
        check_version(version)
        with self.bert_lock:
            segment = (
                self.frontend.segment(text, language)
                if not final
                else Segment(text, [self.frontend.clean_piece(t, lang) for lang, t in self.frontend.language_runs(text, language)])
            )
            if self.bert_model is None:
                return segment.phones, None, segment.norm_text
            import torch

            bert_list = [self.get_bert_inf(p.ids, p.word2ph, p.norm_text, p.lang) for p in segment.pieces]
            bert = torch.cat(bert_list, dim=1)
            return segment.phones, bert, segment.norm_text

    def clean_text_inf(self, text: str, language: str, version: str = "v2") -> tuple[list[int], list[int] | None, str]:
        check_version(version)
        piece = self.frontend.clean_piece(text, language)
        return piece.ids, piece.word2ph, piece.norm_text

    def get_bert_feature(self, text: str, word2ph: list[int]) -> torch.Tensor:
        """chinese-roberta's ``hidden_states[-3]`` per character, repeated per phone: (1024, phones)."""
        import torch

        assert self.bert_model is not None
        enc = self.tokenizer.encode(text)
        device = self.device
        ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
        type_ids = torch.tensor([enc.type_ids], dtype=torch.long, device=device)
        mask = torch.tensor([enc.attention_mask], dtype=torch.long, device=device)
        with torch.no_grad(), self.bert_scope():
            res = self.bert_model.hidden_states(ids, attention_mask=mask, token_type_ids=type_ids)[-3][0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = torch.cat([res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))], dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones: list[int], word2ph: list[int] | None, norm_text: str, language: str) -> torch.Tensor:
        import torch

        if language.replace("all_", "") == "zh":
            assert word2ph is not None
            return self.get_bert_feature(norm_text, word2ph).to(self.device)
        return torch.zeros((1024, len(phones)), dtype=torch.float32).to(self.device)

    def filter_text(self, texts: list[str]) -> list[str]:
        return segmentation.filter_text(texts)

    def replace_consecutive_punctuation(self, text: str) -> str:
        return segmentation.replace_consecutive_punctuation(text)


def load_bert_model(encoders: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> BertModel:
    """chinese-roberta-wwm-ext-large from the ``gptsovits/encoders`` asset (or that folder itself), as upstream loads it
    (``AutoModelForMaskedLM``: SDPA attention, the masked-LM head unused)."""
    import torch

    from mizuoto.nn.transformer.bert import load_bert

    folder = find_one(Path(encoders), "chinese-roberta-wwm-ext-large")
    return load_bert(folder, attention="sdpa", device=device, dtype=dtype or torch.float32)
