"""Drama 3's prompts and long-text chunks (upstream ``fish_omni/generate.py``'s ``build_drama3_prompt`` and
``fish_omni/chunking.py``).

A prompt is rows ``[token, code 0, …, code N-1]``: text rows ``[token, 0, …, 0]``, audio rows ``[code 0 + semantic
start, codes…]``. A system turn ("convert the provided text to speech", with ``[Reference i]`` blocks of transcript and
codes when there are references), the earlier chunks as finished user and assistant turns, then the user turn and an open
assistant voice turn. Each literal is tokenized on its own (joined strings tokenize differently at the seams).

Long text is cut into chunks of about 300 UTF-8 bytes with at most three speakers each (sentence ends, then clause
marks, then spaces), as Fish Audio's hosted server cuts it; the first chunk and the latest one go back into the next
prompt as history.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Any

import torch

from mizuoto_fishspeech.drama3.config import Drama3Config

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:")
"""Only the prompt builder writes these; other special tokens in a request's text are content."""
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER_ID = 15
"""The tokenizer has one token per speaker, ``<|speaker:0|>`` to ``<|speaker:15|>``."""
UNICODE_WHITE_SPACE = "".join(map(chr, (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)))
"""The Unicode White_Space property (``str.strip()`` without arguments also strips U+001C..U+001F)."""
MAX_PROMPT_ROWS = 8192
DEFAULT_MAX_NEW_TOKENS = 2048
"""Frames for a whole text generated in one pass (``single_chunk``) when a request sets no limit."""
CHUNK_BYTES = 300
"""A threshold, not a cap: a protected span longer than this stays whole."""
CHUNK_MAX_SPEAKERS = 3
CHUNK_MAX_NEW_TOKENS = 600
"""Frames per chunk when a request sets no limit (about 27.9 s)."""
HISTORY_RECENT_TURNS = 1
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
PROTECTED_SPAN = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)
"""Never split inside: special tokens, paired tags such as ``<whisper>…</whisper>``, ``[bracketed]`` cues."""


class PromptError(ValueError):
    """A request's text or references that upstream refuses (``InvalidInputError``)."""


class PromptTooLong(PromptError):
    """The prompt has more than ``MAX_PROMPT_ROWS`` rows."""


def validate_text(text: str, location: str) -> None:
    """Not empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise PromptError(f"{location}: text must not be empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise PromptError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise PromptError(f"{location}: speaker tags must be <|speaker:0|> to <|speaker:{MAX_SPEAKER_ID}|>; got {text!r}")


def with_speaker_tag(text: str) -> str:
    """``<|speaker:0|> `` before the text unless a tag already leads it (after white space)."""
    if SPEAKER_TAG.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_prompt(
    text: str,
    tokenizer: Any,
    config: Drama3Config,
    prompt_texts: Sequence[str] = (),
    prompt_codes: Sequence[torch.Tensor] = (),
    history: Sequence[tuple[str, torch.Tensor]] = (),
) -> torch.Tensor:
    """CPU int64 rows [rows, 1 + codebooks] ending in the open assistant voice turn."""
    num_codebooks = config.tower.num_codebooks
    codebook_size = config.tower.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_codes):
        raise PromptError("every reference needs both a transcript and codes")
    validate_text(text, "text")

    def check_codes(codes: torch.Tensor, location: str) -> None:
        if codes.dtype.is_floating_point or codes.dtype == torch.bool or codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise PromptError(f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; got {codes.dtype} {tuple(codes.shape)}")
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise PromptError(f"{location}: codes must be in [0, {codebook_size})")

    segments: list[str | torch.Tensor] = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if prompt_codes else TTS_INSTRUCTION]
    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_codes, strict=True), 1):
        location = f"reference {index}"
        validate_text(transcript, location)
        check_codes(codes, location)
        # the tag must follow "Text: " directly; leading white space moves behind it
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG.match(body)
        assert tag is not None
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_text(turn_text, location)
        check_codes(codes, location)
        segments += ["<|im_start|>user\n", with_speaker_tag(turn_text), MESSAGE_END, GENERATION_START, codes, MESSAGE_END]
    segments += ["<|im_start|>user\n", with_speaker_tag(text), MESSAGE_END, GENERATION_START]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment), dtype=torch.long)
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    out = torch.cat(rows)
    if len(out) > MAX_PROMPT_ROWS:
        raise PromptTooLong(f"prompt has {len(out)} rows; at most {MAX_PROMPT_ROWS}")
    return out


# -- chunks -------------------------------------------------------------------------------------------


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _protected_pieces(text: str) -> list[tuple[str, bool]]:
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last : match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def _breaks_at(text: str, index: int, marks: frozenset[str]) -> bool:
    if text[index] not in marks:
        return False
    # decimals such as 3.14 stay in one piece
    return not (text[index] == "." and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit())


def _break_at_marks(pieces: list[tuple[str, bool]], length: int, marks: frozenset[str]) -> list[tuple[str, bool]]:
    """Split free pieces longer than ``length`` bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if _breaks_at(text, index, marks):
                out.append((text[start : index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def _break_by_length(pieces: list[tuple[str, bool]], length: int) -> list[tuple[str, bool]]:
    """Split the free pieces still longer than ``length`` on character boundaries."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current, size = "", 0
        for char in text:
            if current and size + utf8_len(char) > length:
                out.append((current, False))
                current, size = "", 0
            current += char
            size += utf8_len(char)
        out.append((current, False))
    return out


def split_by_length(text: str, length: int = CHUNK_BYTES) -> list[str]:
    """Chunks of about ``length`` bytes, packed sentence by sentence (a sentence still too long is cut at clause marks,
    then spaces, then on raw length)."""
    if not text.strip(UNICODE_WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = _break_at_marks(_protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = _break_at_marks(pieces, length, marks)
    pieces = _break_by_length(pieces, length)
    chunks, current = [], ""
    for piece, _ in pieces:
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [chunk.strip(UNICODE_WHITE_SPACE) for chunk in chunks if chunk.strip(UNICODE_WHITE_SPACE)]


def parse_speaker_segments(text: str) -> list[tuple[str, str]]:
    """(speaker id, text) per tagged segment; text before the first tag is speaker 0's."""
    parts = SPEAKER_TAG.split(text)
    segments = [("0", parts[0])] if parts[0].strip(UNICODE_WHITE_SPACE) else []
    for speaker, body in zip(parts[1::2], parts[2::2], strict=True):
        if body.strip(UNICODE_WHITE_SPACE):
            segments.append((speaker, body))
    return segments


def split_text(text: str, max_bytes: int = CHUNK_BYTES, max_speakers: int = CHUNK_MAX_SPEAKERS) -> list[str]:
    """The text's chunks in order. Untagged text is split by length; tagged text into speaker segments (long ones cut
    further, each piece keeping its tag), regrouped: a chunk ends when the next segment would bring a speaker too many
    or pass ``max_bytes``."""
    if SPEAKER_TAG.search(text) is None:
        return split_by_length(text, max_bytes)
    segments = []
    for speaker, body in parse_speaker_segments(text):
        for piece in split_by_length(body, max_bytes):
            segments.append((speaker, f"<|speaker:{speaker}|> {piece}"))
    chunks: list[str] = []
    batch: list[str] = []
    speakers: list[str] = []
    size = 0
    for speaker, segment in segments:
        new_speaker = speaker not in speakers
        too_many = new_speaker and len(speakers) >= max_speakers
        too_long = bool(batch) and size + utf8_len(segment) > max_bytes
        if too_many or too_long:
            chunks.append("\n".join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append("\n".join(batch))
    return chunks


class History:
    """The context of the next chunk: the first chunk stays, plus the latest ``HISTORY_RECENT_TURNS``."""

    def __init__(self, recent_turns: int = HISTORY_RECENT_TURNS) -> None:
        self.pinned: tuple[str, torch.Tensor] | None = None
        self.recent: deque[tuple[str, torch.Tensor]] = deque(maxlen=recent_turns)

    def push(self, text: str, codes: torch.Tensor) -> None:
        if self.pinned is None:
            self.pinned = (text, codes)
        else:
            self.recent.append((text, codes))

    def candidates(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Histories to try, longest first: without the oldest recent turns, then without the pinned one, then none."""
        pinned = [self.pinned] if self.pinned is not None else []
        recent = list(self.recent)
        for start in range(len(recent) + 1):
            yield pinned + recent[start:]
        if pinned:
            yield []


def chunk_prompt(text: str, tokenizer: Any, config: Drama3Config, prompt_texts: Sequence[str], prompt_codes: Sequence[torch.Tensor], history: History) -> torch.Tensor:
    """The prompt of one chunk with as much history as ``MAX_PROMPT_ROWS`` allows (``build_drama3_chunk_prompt``)."""
    error: PromptTooLong | None = None
    for turns in history.candidates():
        try:
            return build_prompt(text, tokenizer, config, prompt_texts, prompt_codes, history=turns)
        except PromptTooLong as too_long:
            error = too_long
    assert error is not None
    raise error
