"""English for GPT-SoVITS (upstream ``text/english.py``; MIT, (c) 2024 RVC-Boss) and the parts of g2p_en it uses
(``g2p_en/g2p.py``, Apache-2.0, Kyubyong Park and Jongseok Kim): the GRU that spells out unknown words, in numpy as
g2p_en runs it, with its weights (``checkpoint20.npz``) and homograph list shipped as package data.

Upstream's ``en_G2p`` subclasses g2p_en's ``G2p``, whose import downloads nltk data; here nothing of nltk is used
(``en_tagger``). Lookup order is upstream's: non-letters as they are, single letters from CMUdict (``A`` → EY1), the
homographs by POS tag, CMUdict (upstream's extended copy plus ``engdict-hot.rep``), the name dictionary for
title-case words, letter by letter up to three characters, possessive ``'s``, ``wordsegment`` compounds, then the GRU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from mizuoto.engines.gptsovits.text.symbols import PUNCTUATION, SYMBOL_SET

REP_MAP = {"[;:：，；]": ",", '["’]': "'", "。": ".", "！": "!", "？": "?"}
_REP_RE = re.compile("|".join(re.escape(p) for p in REP_MAP))
_CONSECUTIVE_RE = re.compile("([{0}\\s])([{0}])+".format("".join(re.escape(p) for p in PUNCTUATION)))

GRAPHEMES = ["<pad>", "<unk>", "</s>", *"abcdefghijklmnopqrstuvwxyz"]
PHONEMES = [
    "<pad>", "<unk>", "<s>", "</s>",
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]  # fmt: skip


def text_normalize_with(normalizer: Any, text: str) -> str:
    """``english.text_normalize``: punctuation mapped, then ``expend.normalize``, then runs of punctuation collapsed."""
    text = _REP_RE.sub(lambda x: REP_MAP[x.group()], text)
    text = normalizer(text)
    return _CONSECUTIVE_RE.sub(r"\1", text)


def read_cmudict() -> dict[str, list[list[str]]]:
    """``english.get_dict()``: ``cmudict.rep`` from line 57, ``cmudict-fast.rep`` for words it lacks, ``engdict-hot.rep`` over both."""
    from mizuoto.engines.gptsovits.text._data import read_text

    g2p_dict: dict[str, list[list[str]]] = {}
    for line_index, line in enumerate(read_text("cmudict.rep").splitlines(), start=1):
        if line_index >= 57:
            word_split = line.strip().split("  ")
            g2p_dict[word_split[0].lower()] = [word_split[1].split(" ")]
    for line in read_text("cmudict-fast.rep").splitlines():
        word_split = line.strip().split(" ")
        word = word_split[0].lower()
        if word not in g2p_dict:
            g2p_dict[word] = [word_split[1:]]
    for line in read_text("engdict-hot.rep").splitlines():
        word_split = line.strip().split(" ")
        g2p_dict[word_split[0].lower()] = [word_split[1:]]
    return g2p_dict


def read_homographs() -> dict[str, tuple[list[str], list[str], str]]:
    from mizuoto.engines.gptsovits.text._data import read_text

    homograph2features: dict[str, tuple[list[str], list[str], str]] = {}
    for line in read_text("g2p_en_homographs.en").splitlines():
        if line.startswith("#"):
            continue
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features


class GraphemeGRU:
    """g2p_en's seq2seq GRU (``G2p.predict``), numpy float32 as upstream."""

    def __init__(self) -> None:
        import io

        from mizuoto.engines.gptsovits.text._data import read_bytes

        v = np.load(io.BytesIO(read_bytes("g2p_en_checkpoint20.npz")))
        self.enc_emb, self.enc_w_ih, self.enc_w_hh, self.enc_b_ih, self.enc_b_hh = (v[k] for k in ("enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh"))
        self.dec_emb, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh = (v[k] for k in ("dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh"))
        self.fc_w, self.fc_b = v["fc_w"], v["fc_b"]
        self.g2idx = {g: idx for idx, g in enumerate(GRAPHEMES)}
        self.idx2p = dict(enumerate(PHONEMES))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _grucell(self, x: np.ndarray, h: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray, b_ih: np.ndarray, b_hh: np.ndarray) -> np.ndarray:
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh
        rz_ih, n_ih = rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :]
        rz_hh, n_hh = rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :]
        rz = self._sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)
        n = np.tanh(n_ih + r * n_hh)
        return (1 - z) * n + z * h

    def _gru(self, x: np.ndarray, steps: int, h0: np.ndarray) -> np.ndarray:
        h = h0
        outputs = np.zeros((x.shape[0], steps, self.enc_w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self._grucell(x[:, t, :], h, self.enc_w_ih, self.enc_w_hh, self.enc_b_ih, self.enc_b_hh)
            outputs[:, t, ::] = h
        return outputs

    def predict(self, word: str) -> list[str]:
        chars = [*word, "</s>"]
        x = np.take(self.enc_emb, np.expand_dims([self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars], 0), axis=0)
        enc = self._gru(x, len(word) + 1, np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        h = enc[:, -1, :]
        dec = np.take(self.dec_emb, [2], axis=0)  # <s>
        preds: list[int] = []
        for _ in range(20):
            h = self._grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = int(logits.argmax())
            if pred == 3:  # </s>
                break
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)
        return [self.idx2p.get(idx, "<unk>") for idx in preds]


class EnglishG2P:
    """``english.text_normalize`` and ``english.g2p``. ``tagger_dir`` is the nltk data's ``taggers/averaged_perceptron_tagger_eng``."""

    def __init__(self, tagger_dir: str | Path) -> None:
        import json

        import wordsegment

        from mizuoto.engines.gptsovits.text._data import read_text
        from mizuoto.engines.gptsovits.text.en_normalization import EnglishNormalizer
        from mizuoto.engines.gptsovits.text.en_tagger import PerceptronTagger

        self.normalizer = EnglishNormalizer()
        self.tagger = PerceptronTagger(tagger_dir)
        self.segmenter = wordsegment.Segmenter()
        self.segmenter.load()
        self.gru = GraphemeGRU()
        self.cmu = read_cmudict()
        self.namedict: dict[str, list[list[str]]] = json.loads(read_text("namedict.json"))
        for word in ("AE", "AI", "AR", "IOS", "HUD", "OS"):
            del self.cmu[word.lower()]
        self.homograph2features = read_homographs()
        self.homograph2features["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (["K", "AH0", "M", "P", "L", "EH1", "K", "S"], ["K", "AA1", "M", "P", "L", "EH0", "K", "S"], "JJ")

    def text_normalize(self, text: str) -> str:
        return text_normalize_with(self.normalizer, text)

    def __call__(self, text: str) -> list[str]:
        """``en_G2p.__call__``: phones, words separated by ``" "``."""
        from mizuoto.engines.gptsovits.text.en_tagger import tweet_tokenize

        prons: list[str] = []
        for o_word, pos in self.tagger.tag(tweet_tokenize(text)):
            word = o_word.lower()
            if re.search("[a-z]", word) is None:
                pron = [word]
            elif len(word) == 1:
                pron = ["EY1"] if o_word == "A" else self.cmu[word][0]
            elif word in self.homograph2features:
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1) or (len(pos) < len(pos1) and pos == pos1[: len(pos)]):
                    pron = pron1
                else:
                    pron = pron2
            else:
                pron = self.qryword(o_word)
            prons.extend(pron)
            prons.extend([" "])
        return prons[:-1]

    def qryword(self, o_word: str) -> list[str]:
        word = o_word.lower()
        if len(word) > 1 and word in self.cmu:
            return self.cmu[word][0]
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]
        if len(word) <= 3:
            phones: list[str] = []
            for w in word:
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            else:
                phones.extend(["Z"])
            return phones
        comps = self.segmenter.segment(word.lower())
        if len(comps) == 1:
            return self.gru.predict(word)
        return [phone for comp in comps for phone in self.qryword(comp)]

    def g2p(self, text: str) -> list[str]:
        """``english.g2p``: the phones without word breaks or g2p_en's specials, ``'`` as ``-``, unknowns dropped."""
        phone_list = self(text)
        phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]
        out: list[str] = []
        for ph in phones:
            if ph in SYMBOL_SET:
                out.append(ph)
            elif ph == "'":
                out.append("-")
            # upstream prints "ph not in symbols" and drops it
        return out
