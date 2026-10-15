"""What a fine-tune holds on its GPU, roughly, by memory preset. Standard library only.

From the base model's size on disk and how each engine trains it: the weights as the trainer holds them (float32
copies of a bfloat16 checkpoint double; Qwen3-TTS and Fish Speech train in bfloat16), the gradients and AdamW's two
moments of what trains (all of it for a full fine-tune; about 2 % for LoRA), and the activations (a fraction of the
weights, about a third of it with gradient checkpointing). ``low`` and ``lowest`` also count what their tools take off
the device: F5-TTS's averaged copy, LoRA's frozen weights in the autocast dtype or streamed from RAM, the optimizer's
state in RAM. Shown before a run and checked against the device; the stage reports what it really used
(``memory/peak_mb``).
"""

from __future__ import annotations

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


def estimate_mb(engine: str, method: str, model_mb: float, lora: bool) -> dict[str, float]:
    """MB on the device at the peak of a run, by preset."""
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
