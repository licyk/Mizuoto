"""Presets: named parameter sets for one model, or common parameters for any model."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from mizuoto.core.record import Record


class Preset(Record):
    id: str
    name: str
    engine: str | None = None
    model: str | None = None
    """The model it is for; None: any model (common parameters only)."""
    voice_id: str | None = None
    """The voice its default status is for (None: any voice of the model)."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Common parameters that are set, by their common names."""
    engine_options: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    """Applied when its model (and voice) is picked."""
    created_at: str
    updated_at: str


class PresetInput(Record):
    name: str
    model: str | None = None
    voice_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    engine_options: dict[str, Any] = Field(default_factory=dict)


class PresetPatch(Record):
    name: str | None = None
    params: dict[str, Any] | None = None
    engine_options: dict[str, Any] | None = None
