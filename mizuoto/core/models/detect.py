"""Telling which engine and variant a folder of model files belongs to, from its contents only.

Reads small configuration files; never imports an engine. Engines join as they are integrated
; the engine adapters' ``detect_variant`` agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QWEN3_VARIANTS = {
    ("base", "1b7"): "base",
    ("base", "0b6"): "base",
    ("custom_voice", "1b7"): "custom_voice",
    ("custom_voice", "0b6"): "custom_voice_lite",
    ("voice_design", "1b7"): "voice_design",
    ("voice_design", "0b6"): "voice_design",
}
"""(``tts_model_type``, ``tts_model_size``) → variant; the adapter's ``detect_variant`` agrees."""


def _json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def detect(folder: Path) -> tuple[str, str, str] | None:
    """``(engine, variant, title)`` of the model in ``folder``, or None."""
    config = _json(folder / "config.json")
    if config is not None:
        if config.get("tokenizer_type") == "qwen3_tts_tokenizer_12hz" and "tts_model_type" in config:
            variant = QWEN3_VARIANTS.get((config.get("tts_model_type", ""), config.get("tts_model_size", "")))
            if variant is not None:
                size = {"1b7": "1.7B", "0b6": "0.6B"}.get(config.get("tts_model_size", ""), "")
                return "qwen3tts", variant, f"Qwen3-TTS {size} {config['tts_model_type']}".strip()
        if config.get("model_type") == "whisper" and (folder / "preprocessor_config.json").is_file():
            return "asr", "whisper", f"Whisper ({folder.name})"
        if config.get("model_type") == "omnivoice":
            return "omnivoice", "v0", f"OmniVoice ({folder.name})"
        if "lm_config" in config and "dit_config" in config:
            # VoxCPM: ``architecture`` names the generation (missing: the first); 1.5 is the first at 44.1 kHz
            arch = str(config.get("architecture", "voxcpm")).lower()
            if arch == "voxcpm2":
                return "voxcpm", "v2", f"VoxCPM2 ({folder.name})"
            if arch == "voxcpm":
                if (config.get("audio_vae_config") or {}).get("sample_rate", 16000) == 44100:
                    return "voxcpm", "v1_5", f"VoxCPM 1.5 ({folder.name})"
                return "voxcpm", "v1", f"VoxCPM 0.5B ({folder.name})"
    # CosyVoice: the yaml names the generation; v1 folders differ by their speakers and their name.
    if (folder / "cosyvoice3.yaml").is_file():
        return "cosyvoice", "v3", f"Fun-CosyVoice3 ({folder.name})"
    if (folder / "cosyvoice2.yaml").is_file():
        return "cosyvoice", "v2", f"CosyVoice2 ({folder.name})"
    if (folder / "cosyvoice.yaml").is_file():
        if not (folder / "spk2info.pt").is_file():
            return "cosyvoice", "v1_base", f"CosyVoice-300M ({folder.name})"
        if "instruct" in folder.name.lower():
            return "cosyvoice", "v1_instruct", f"CosyVoice-300M-Instruct ({folder.name})"
        return "cosyvoice", "v1_sft", f"CosyVoice-300M-SFT ({folder.name})"
    # F5-TTS checkpoints carry no configuration: an import names the engine and the variant.
    return None


# -- GPT-SoVITS weights ------------------------------------------------------------------

SOVITS_HEADS = {
    b"00": "v1",
    b"01": "v2",
    b"02": "v3",
    b"03": "v3",
    b"04": "v4",
    b"05": "v2Pro",
    b"06": "v2ProPlus",
    b"07": "v5dev",
    b"08": "v5turbo",
}
"""The two bytes GPT-SoVITS writes in place of the zip magic of a fine-tuned SoVITS file (``process_ckpt.head2version``)."""

SOVITS_PRETRAINED_MD5 = {
    "dc3c97e17592963677a4a1681f30c653": "v2",
    "43797be674a37c1c83ee81081941ed0f": "v3",
    "6642b37f3dbb1f76882b69937c95a5f3": "v2",
    "4f26b9476d0c5033e04162c486074374": "v4",
    "c7e9fce2223f3db685cdfa1e6368728a": "v2Pro",
    "66b313e39455b57ab1b0bc0b239c9d0a": "v2ProPlus",
    "c4ce1f839a0d271e41963f9fb16ade76": "v5dev",
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": "v5turbo",
}
"""MD5 of the first 8 KiB of the pretrained SoVITS files (``process_ckpt.hash_pretrained_dict``)."""


def sovits_version(path: Path) -> str | None:
    """The GPT-SoVITS version of a SoVITS ``.pth``, as upstream's ``get_sovits_version_from_path_fast``
    tells it: the pretrained files by hash, fine-tunes by their header, older files by size."""
    import hashlib

    try:
        with open(path, "rb") as f:
            head = f.read(8192)
        size = path.stat().st_size
    except OSError:
        return None
    pretrained = SOVITS_PRETRAINED_MD5.get(hashlib.md5(head).hexdigest())
    if pretrained:
        return pretrained
    if head[:2] != b"PK":
        return SOVITS_HEADS.get(head[:2])
    # Old fine-tunes are plain torch zips: v1 is the smallest, v2 below 700 MB, v3 above.
    if size < 82978 * 1024:
        return "v1"
    return "v2" if size < 700 * 1024 * 1024 else "v3"


def weights_version(engine: str, files: dict[str, Path]) -> str | None:
    """The version of a voice's fine-tuned weights for ``engine``, from the files themselves."""
    if engine == "gptsovits" and "sovits" in files:
        return sovits_version(files["sovits"])
    return None
