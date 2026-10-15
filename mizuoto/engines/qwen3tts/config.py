"""Qwen3-TTS's configuration, read from the model folder's ``config.json`` (no transformers)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mizuoto.runtime.lm.qwen import QwenConfig

VARIANTS = {
    ("base", "1b7"): "base",
    ("base", "0b6"): "base",
    ("custom_voice", "1b7"): "custom_voice",
    ("custom_voice", "0b6"): "custom_voice_lite",
    ("voice_design", "1b7"): "voice_design",
    ("voice_design", "0b6"): "voice_design",
}


@dataclass
class TalkerConfig:
    decoder: QwenConfig
    predictor: QwenConfig
    vocab_size: int
    text_vocab_size: int
    text_hidden_size: int
    num_code_groups: int
    predictor_vocab_size: int
    codec_bos_id: int
    codec_eos_id: int
    codec_pad_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    language_ids: dict[str, int]
    speaker_ids: dict[str, int]
    speaker_dialects: dict[str, str | bool]


@dataclass
class Qwen3TtsConfig:
    model_type: str
    """``base``, ``custom_voice`` or ``voice_design``."""
    model_size: str
    tokenizer_type: str
    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int
    talker: TalkerConfig
    speaker_encoder: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str | None:
        return VARIANTS.get((self.model_type, self.model_size))

    @property
    def languages(self) -> list[str]:
        """As upstream's ``get_supported_languages``: ``auto`` and the language ids that are not dialects."""
        return ["auto", *[k for k in self.talker.language_ids if "dialect" not in k]]


def read_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def load_config(model_dir: Path) -> Qwen3TtsConfig:
    raw = read_config(model_dir)
    t = raw["talker_config"]
    p = t["code_predictor_config"]
    talker = TalkerConfig(
        decoder=QwenConfig(
            hidden_size=t["hidden_size"],
            intermediate_size=t["intermediate_size"],
            num_layers=t["num_hidden_layers"],
            num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"],
            head_dim=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
            rms_norm_eps=t["rms_norm_eps"],
            rope_theta=float(t["rope_theta"]),
            attention_bias=t.get("attention_bias", False),
            o_bias=t.get("attention_bias", False),
        ),
        predictor=QwenConfig(
            hidden_size=p["hidden_size"],
            intermediate_size=p["intermediate_size"],
            num_layers=p["num_hidden_layers"],
            num_heads=p["num_attention_heads"],
            num_kv_heads=p["num_key_value_heads"],
            head_dim=p.get("head_dim", p["hidden_size"] // p["num_attention_heads"]),
            rms_norm_eps=p["rms_norm_eps"],
            rope_theta=float(p["rope_theta"]),
            attention_bias=p.get("attention_bias", False),
            o_bias=p.get("attention_bias", False),
        ),
        vocab_size=t["vocab_size"],
        text_vocab_size=t["text_vocab_size"],
        text_hidden_size=t["text_hidden_size"],
        num_code_groups=t["num_code_groups"],
        predictor_vocab_size=p["vocab_size"],
        codec_bos_id=t["codec_bos_id"],
        codec_eos_id=t["codec_eos_token_id"],
        codec_pad_id=t["codec_pad_id"],
        codec_think_id=t["codec_think_id"],
        codec_nothink_id=t["codec_nothink_id"],
        codec_think_bos_id=t["codec_think_bos_id"],
        codec_think_eos_id=t["codec_think_eos_id"],
        language_ids=dict(t.get("codec_language_id") or {}),
        speaker_ids=dict(t.get("spk_id") or {}),
        speaker_dialects=dict(t.get("spk_is_dialect") or {}),
    )
    generation_path = Path(model_dir) / "generation_config.json"
    return Qwen3TtsConfig(
        model_type=raw["tts_model_type"],
        model_size=raw["tts_model_size"],
        tokenizer_type=raw.get("tokenizer_type", ""),
        tts_bos_token_id=raw["tts_bos_token_id"],
        tts_eos_token_id=raw["tts_eos_token_id"],
        tts_pad_token_id=raw["tts_pad_token_id"],
        talker=talker,
        speaker_encoder=dict(raw.get("speaker_encoder_config") or {}),
        generation=json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else {},
    )
