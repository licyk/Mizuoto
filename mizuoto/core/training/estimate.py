"""What a fine-tune holds on its GPU, by memory preset. Standard library only.

Every method of every trainer was measured on an RTX 3090 (2026-10-18) with its recipe's default batch: the stages'
peak (``memory/peak_mb``, what torch allocated) under each preset, for each installed base model (``MEASURED``). An
estimate is the peak of the measured base model nearest in size, scaled by the size, with a tenth more for the
allocator's spare blocks and 512 MB for the CUDA context. Where a peak follows the batch (Fish Speech, GPT-SoVITS,
VoxCPM: ``BATCH``), each unit of the run's batch above or below the measured one adds or removes what one measured
unit added. Shown before a run and checked against the device; the stage reports what it really used.

Other engines and methods fall back to a rule of thumb: the weights as the trainer holds them, the gradients and
AdamW's two moments of what trains, activations as a fraction of the weights (a third with gradient checkpointing),
and what ``low`` and ``lowest`` take off the device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRESETS = ("standard", "low", "lowest")


@dataclass(frozen=True)
class Profile:
    weights: float = 1.0
    """The trained weights' bytes over the checkpoint's (2: float32 copies of a bfloat16 file)."""
    trained: float = 1.0
    """The share of the model folder the trainer loads (the rest: codecs it does not train with)."""
    activations: float = 0.5
    """Activations at the recipe's batch, over the weights."""
    ema: bool = False
    """It keeps an averaged copy of the model (F5-TTS)."""
    host_optimizer: bool = False
    """``lowest`` moves AdamW's state to RAM for its full fine-tunes."""
    streamed_lora: bool = False
    """``lowest`` streams LoRA's frozen weights from RAM."""
    frozen_dtype: bool = False
    """``low`` keeps LoRA's frozen weights in the autocast dtype."""
    checkpointing: bool = True
    """``low`` checkpoints its blocks."""


PROFILES: dict[str, Profile] = {
    "voxcpm": Profile(weights=2.0, trained=0.93, host_optimizer=False, streamed_lora=True, frozen_dtype=True),
    "qwen3tts": Profile(weights=1.0, trained=0.85, host_optimizer=True),
    "omnivoice": Profile(weights=1.0, trained=0.75, host_optimizer=True),
    "f5tts": Profile(weights=1.0, trained=0.96, ema=True, host_optimizer=True),
    "cosyvoice": Profile(weights=1.0, trained=0.5),
    "fishspeech": Profile(weights=1.0, trained=0.83, checkpointing=False),
    "gptsovits": Profile(weights=1.0, trained=0.3, checkpointing=False),
}


@dataclass(frozen=True)
class Measured:
    model_mb: float
    """The base model's size on disk."""
    peaks: dict[str, float]
    """Preset → the stages' peak, MB, at the recipe's default batch (``batch``, when the peak follows it)."""
    batch: float | None = None


MEASURED: dict[tuple[str, str], tuple[Measured, ...]] = {
    ("cosyvoice", "llm"): (
        Measured(2943, {"standard": 10319, "low": 9691, "lowest": 9691}),  # CosyVoice 2
        Measured(4233, {"standard": 10398, "low": 9693, "lowest": 9693}),  # CosyVoice 3 (its peak is not its folder's size)
    ),
    ("cosyvoice", "llm_flow"): (Measured(2943, {"standard": 10319, "low": 9691, "lowest": 9691}),),
    ("f5tts", "full"): (Measured(1338, {"standard": 8821, "low": 7395, "lowest": 5374}),),
    ("fishspeech", "lora"): (Measured(10498, {"standard": 15641, "low": 15641, "lowest": 15641}, batch=2),),
    ("gptsovits", "speaker"): (Measured(303, {"standard": 4676, "low": 4676, "lowest": 4670}, batch=4),),
    ("omnivoice", "full"): (Measured(3116, {"standard": 11724, "low": 11714, "lowest": 4713}),),
    ("omnivoice", "lora"): (Measured(3116, {"standard": 7301, "low": 2971, "lowest": 2766}),),
    ("qwen3tts", "speaker"): (Measured(4334, {"standard": 18276, "low": 18267, "lowest": 7356}),),
    ("voxcpm", "full"): (Measured(1535, {"standard": 16733, "low": 12719, "lowest": 12719}, batch=16),),
    ("voxcpm", "lora"): (
        Measured(1535, {"standard": 8661, "low": 3116, "lowest": 2970}, batch=16),  # 0.5B
        Measured(1863, {"standard": 10222, "low": 3301, "lowest": 3301}, batch=16),  # 1.5
        Measured(4731, {"standard": 14498, "low": 9094, "lowest": 9094}, batch=16),  # 2
    ),
}
"""``(engine, method)`` → its measurements (the stages' ``memory/peak_mb``, RTX 3090, 2026-10-18)."""

BATCH: dict[str, tuple[str, float, float]] = {
    "fishspeech": ("batch_size", 4, 3375),  # 12,267 MB at 1, 15,641 at 2; 4 does not fit 24 GB
    "gptsovits": ("batch_size", 4, 398),  # 3,881 MB at 2, 4,676 at 4
    "voxcpm": ("batch_size", 16, 308),  # 4,343 MB at 2, 8,661 at 16 (0.5B LoRA); 0 is upstream's 16
}
"""Engine → (the hyperparameter its peak follows, the recipe's default, MB a unit adds under ``standard``; under the
other presets in proportion to their peaks)."""


def _measured(engine: str, method: str, model_mb: float, hyperparams: dict[str, object] | None) -> dict[str, float] | None:
    found = MEASURED.get((engine, method))
    if not found or model_mb <= 0:
        return None
    m = min(found, key=lambda x: abs(math.log(model_mb / x.model_mb)))
    scale = model_mb / m.model_mb
    name, default, per_unit = BATCH.get(engine, ("", 0.0, 0.0))
    batch = float(default)
    given = hyperparams.get(name) if name and hyperparams else None
    if isinstance(given, (int, float)) and given > 0:  # 0: the recipe's own (VoxCPM's 16)
        batch = float(given)
    out = {}
    for preset in PRESETS:
        peak = m.peaks[preset] * scale
        if m.batch is not None and name:
            peak += (batch - m.batch) * per_unit * m.peaks[preset] / m.peaks["standard"]
        out[preset] = round(max(peak, 0.0) * 1.1 + 512, 0)
    return out


def estimate_mb(engine: str, method: str, model_mb: float, lora: bool, hyperparams: dict[str, object] | None = None) -> dict[str, float]:
    """MB on the device at the peak of a run, by preset; ``hyperparams``: the run's (default: the recipe's)."""
    measured = _measured(engine, method, model_mb, hyperparams)
    if measured is not None:
        return measured
    p = PROFILES.get(engine, Profile())
    weights = model_mb * p.trained * p.weights
    trains = weights * (0.02 if lora else 1.0)
    out: dict[str, float] = {}
    for preset in PRESETS:
        frozen = weights - trains if lora else 0.0
        resident = weights
        if lora and preset in ("low", "lowest") and p.frozen_dtype:
            resident -= frozen / 2
        if lora and preset == "lowest" and p.streamed_lora:
            resident -= frozen / 2 if p.frozen_dtype else frozen
        optimizer = 2 * trains if not (preset == "lowest" and p.host_optimizer and not lora) else 0.0
        activations = weights * p.activations
        if preset in ("low", "lowest") and p.checkpointing:
            activations /= 3
        ema = weights if p.ema and preset == "standard" else 0.0
        out[preset] = round(resident + trains + optimizer + activations + ema + 512, 0)
    return out
