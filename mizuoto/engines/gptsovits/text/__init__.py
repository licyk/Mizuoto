"""GPT-SoVITS's text front end, native: text → phone ids, word2ph and normalised text,
as upstream's ``TextPreprocessor`` and ``text/`` package (MIT, (c) 2024 RVC-Boss) compute them.

``TextPreprocessor`` mirrors upstream's class; ``TextFrontend`` is its text half (``segment(text, language)`` →
``Segment`` of ``Piece``s); ``segmentation`` holds the cut methods; ``symbols`` the phone table.
"""

from mizuoto.engines.gptsovits.text.frontend import FrontendError, FrontendPaths, Piece, Segment, TextFrontend, TextPreprocessor, load_bert_model
from mizuoto.engines.gptsovits.text.segmentation import EmptyTextError, pre_seg_text, replace_consecutive_punctuation
from mizuoto.engines.gptsovits.text.symbols import SYMBOLS, cleaned_text_to_sequence

__all__ = [
    "SYMBOLS",
    "EmptyTextError",
    "FrontendError",
    "FrontendPaths",
    "Piece",
    "Segment",
    "TextFrontend",
    "TextPreprocessor",
    "cleaned_text_to_sequence",
    "load_bert_model",
    "pre_seg_text",
    "replace_consecutive_punctuation",
]
