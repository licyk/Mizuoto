"""Whisper's text side as transformers 4.57.3's ASR pipeline handles it (Apache-2.0).

- ``WhisperText``: the model's ``tokenizer.json`` through ``tokenizers``, decoding as ``WhisperTokenizerFast.decode``
  does (the byte-level decoder, ``clean_up_tokenization_spaces`` from ``tokenizer_config.json``, timestamp text
  removed); its special tokens are those ``tokenizer.json`` flags special (the same set as ``all_special_ids``).
- ``language_id``: ``_retrieve_init_tokens``'s ``language_to_id`` (a code, a ``<|code|>`` token or an English name).
- ``chunk_bounds``: the pipeline's ``chunk_iter`` (30 s windows, a sixth of that as stride on each side).
- ``decode_asr``: ``tokenization_whisper._decode_asr`` with ``return_timestamps=True``: the windows' tokens joined
  into text and timestamped chunks, overlaps resolved by ``_find_longest_common_sequence``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese", "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili", "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish",
    "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala",
    "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}  # fmt: skip
"""transformers' ``tokenization_whisper.LANGUAGES``."""

TO_LANGUAGE_CODE = {
    **{name: code for code, name in LANGUAGES.items()},
    "burmese": "my", "valencian": "ca", "flemish": "nl", "haitian": "ht", "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa",
    "moldavian": "ro", "moldovan": "ro", "sinhalese": "si", "castilian": "es", "mandarin": "zh",
}  # fmt: skip

TIMESTAMP_PATTERN = re.compile(r"<\|(\d+\.\d+)\|>")


def clean_up_tokenization(text: str) -> str:
    """``PreTrainedTokenizerBase.clean_up_tokenization``."""
    return (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )


class WhisperText:
    def __init__(self, folder: Path) -> None:
        from tokenizers import Tokenizer

        path = folder / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.special_ids = {int(t["id"]) for t in raw.get("added_tokens") or [] if t.get("special")}
        config_path = folder / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self.clean_up = bool(config.get("clean_up_tokenization_spaces", False))

    def token_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(int(token_id)) or ""

    def decode(self, ids: list[int]) -> str:
        """``WhisperTokenizerFast.decode(ids)`` with its defaults."""
        text = self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)
        if self.clean_up:
            text = clean_up_tokenization(text)
        return TIMESTAMP_PATTERN.sub("", text)


def language_id(language: str, lang_to_id: dict[str, int]) -> int:
    """``language_to_id``: the token id of ``language``; ``ValueError`` when the model has no such language.

    A regional code (``zh-cn``) falls back to its base language (transformers refuses it).
    """
    language = language.lower()
    if language in lang_to_id:
        token = language
    elif language in TO_LANGUAGE_CODE:
        token = f"<|{TO_LANGUAGE_CODE[language]}|>"
    elif language in TO_LANGUAGE_CODE.values():
        token = f"<|{language}|>"
    elif "-" in language and language.split("-", 1)[0] in TO_LANGUAGE_CODE.values():
        token = f"<|{language.split('-', 1)[0]}|>"
    else:
        raise ValueError(f"Unsupported language: {language}")
    if token not in lang_to_id:
        raise ValueError(f"{token} is not supported by this model")
    return lang_to_id[token]


# -- the pipeline's chunking and post-processing --------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    start: int
    """First sample."""
    stop: int
    """One past the last sample."""
    stride: tuple[int, int, int]
    """(samples, left stride, right stride) as ``chunk_iter`` reports it."""


def chunk_bounds(samples: int, rate: int = 16000, chunk_length_s: float = 30.0) -> list[Chunk]:
    """``AutomaticSpeechRecognitionPipeline.preprocess`` + ``chunk_iter`` with ``chunk_length_s`` and its default stride."""
    stride_length_s = chunk_length_s / 6
    chunk_len = int(round(chunk_length_s * rate))
    stride_left = stride_right = int(round(stride_length_s * rate))
    step = chunk_len - stride_left - stride_right
    out: list[Chunk] = []
    for start in range(0, samples, step):
        end = start + chunk_len
        length = min(end, samples) - start
        left = 0 if start == 0 else stride_left
        is_last = end >= samples
        right = 0 if is_last else stride_right
        if length > left:
            out.append(Chunk(start, start + length, (length, left, right)))
        if is_last:
            break
    return out


@dataclass
class Segment:
    text: str
    timestamp: tuple[float | None, float | None]


def find_longest_common_sequence(sequences: list[list[int]]) -> list[int]:
    """``tokenization_whisper._find_longest_common_sequence`` without token timestamps."""
    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence: list[int] = []
    for right_sequence in sequences[1:]:
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            eps = i / 10000.0
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])
            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])
            if len(left) != len(right):
                raise RuntimeError("Whisper's overlap resolution failed")
            matches = np.sum(left == right)
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)
        (left_start, left_stop, right_start, right_stop) = max_indices
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)
    total_sequence.extend(left_sequence)
    return total_sequence


def decode_asr(
    text: WhisperText, outputs: list[tuple[list[int], tuple[float, float, float]]], timestamp_begin: int, time_precision: float, segment_size: int = 1500
) -> tuple[str, list[Segment]]:
    """``_decode_asr(return_timestamps=True)``: ``outputs`` are each window's tokens with its stride in seconds."""
    chunks: list[Segment] = []
    start: float | None = None
    end: float | None = None
    time_offset = 0.0
    previous_tokens: list[list[int]] = []
    skip = False
    right_stride_start = None
    special = text.special_ids
    prompt_token_id = text.token_id("<|startofprev|>")
    decoder_start_token_id = text.token_id("<|startoftranscript|>")

    for token_ids, (chunk_len, stride_left, stride_right) in outputs:
        if token_ids and token_ids[0] == prompt_token_id:  # _strip_prompt
            token_ids = token_ids[token_ids.index(decoder_start_token_id) :] if decoder_start_token_id in token_ids else []
        last_timestamp = None
        first_timestamp: float = timestamp_begin
        cur_max_timestamp = 0.0
        prev_segments_len = 0.0
        penultimate_timestamp = 0.0

        time_offset -= stride_left
        right_stride_start = chunk_len - stride_right
        if stride_left:
            first_timestamp = stride_left / time_precision + timestamp_begin
        if stride_right:
            for token in reversed(token_ids):
                if token >= timestamp_begin:
                    if last_timestamp is not None and (token - timestamp_begin) * time_precision < right_stride_start:
                        break
                    last_timestamp = token

        current_tokens: list[int] = []
        for i, token in enumerate(token_ids):
            if token in special:
                pass
            elif token >= timestamp_begin:
                timestamp = float((token - timestamp_begin) * time_precision)
                if timestamp < cur_max_timestamp:
                    last_was_single_ending = i >= 2 and not (token_ids[i - 1] >= timestamp_begin and token_ids[i - 2] >= timestamp_begin)
                    if last_was_single_ending:
                        prev_segments_len += time_precision * segment_size
                    else:
                        cur_max_timestamp = penultimate_timestamp
                        prev_segments_len += penultimate_timestamp
                penultimate_timestamp = cur_max_timestamp
                cur_max_timestamp = timestamp
                time = (token - timestamp_begin) * time_precision + time_offset + prev_segments_len
                time = round(time, 2)
                if last_timestamp and token >= last_timestamp:
                    skip = True
                elif skip or (previous_tokens and token < first_timestamp):
                    skip = False
                elif start is None:
                    start = time
                elif time == start:
                    pass
                else:
                    end = time
                    previous_tokens.append(current_tokens)
                    chunks.append(Segment(text.decode(find_longest_common_sequence(previous_tokens)), (start, end)))
                    previous_tokens = []
                    current_tokens = []
                    start = end = None
            else:
                current_tokens.append(token)

        time_offset += chunk_len - stride_right
        if current_tokens:
            previous_tokens.append(current_tokens)
        elif not any(p for p in previous_tokens):
            start = end = None
            previous_tokens = []

    if previous_tokens:
        chunks.append(Segment(text.decode(find_longest_common_sequence(previous_tokens)), (start, end)))
    return "".join(c.text for c in chunks), chunks
