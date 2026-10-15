"""What engines' licences allow for their outputs.

Outputs carry the ids of the restrictive licences that made them (``Output.licence_tags``). Fish
Speech's and IndexTTS's licences keep their outputs from becoming training data for other engines'
models; the datasets service (M6) asks ``training_refusal`` before it takes a clip.
"""

from __future__ import annotations

from mizuoto.core.engines.registry import EngineRegistry


def training_refusal(registry: EngineRegistry, licence_tags: list[str], target_engine: str) -> str | None:
    """Why audio with ``licence_tags`` may not train ``target_engine``'s models, or None when it may."""
    for engine_id in registry.ids():
        licence = registry.manifest(engine_id).licence
        if licence.id in licence_tags and not licence.trains_others and engine_id != target_engine:
            return f"{licence.name} does not allow its outputs to train other engines' models"
    return None
