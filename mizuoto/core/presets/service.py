"""Named parameter sets. One default per model and voice."""

from __future__ import annotations

import json
from typing import Any

from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.db import Database
from mizuoto.core.engines.params import check_options, check_params
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import ConflictError, NotFoundError, ValidationError
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import PresetsChangedEvent
from mizuoto.core.models.service import ModelService
from mizuoto.core.presets.models import Preset, PresetInput, PresetPatch
from mizuoto.core.tts.models import CommonParams, SpeakRequest


class PresetService:
    def __init__(self, db: Database, events: EventBus, registry: EngineRegistry, models: ModelService) -> None:
        self._db = db
        self._events = events
        self._registry = registry
        self._models = models

    @staticmethod
    def _row(row: Any) -> Preset:
        return Preset(
            id=row["id"],
            name=row["name"],
            engine=row["engine"],
            model=row["model"],
            voice_id=row["voice_id"],
            params=json.loads(row["params"] or "{}"),
            engine_options=json.loads(row["engine_options"] or "{}"),
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_presets(self, model: str | None = None) -> list[Preset]:
        """Every preset, or those a model can use: its own and the common ones."""
        if model is None:
            rows = self._db.fetchall("SELECT * FROM presets ORDER BY model IS NOT NULL, model, name")
        else:
            rows = self._db.fetchall("SELECT * FROM presets WHERE model IS NULL OR model = ? ORDER BY model IS NOT NULL, name", (model,))
        return [self._row(r) for r in rows]

    def get(self, key: str) -> Preset:
        """By id, or by name when only one preset has it."""
        row = self._db.fetchone("SELECT * FROM presets WHERE id = ?", (key,))
        if row is not None:
            return self._row(row)
        rows = self._db.fetchall("SELECT * FROM presets WHERE name = ?", (key,))
        if len(rows) == 1:
            return self._row(rows[0])
        if rows:
            raise ConflictError(f"Several presets are named {key!r}; give its id", {"ids": [r["id"] for r in rows]})
        raise NotFoundError(f"No preset {key!r}", {"preset": key})

    def _checked(self, model: str | None, params: dict[str, Any], options: dict[str, Any]) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
        """The engine, and the set parameters and options, checked against the model when there is one."""
        unknown = sorted(set(params) - set(CommonParams.model_fields))
        if unknown:
            raise ValidationError(f"Unknown parameters: {', '.join(unknown)} (engine options go in engine_options)", {"unknown": unknown})
        try:
            values = CommonParams.model_validate(params).model_dump(exclude_none=True)
        except Exception as e:
            raise ValidationError(f"Invalid parameters: {e}") from None
        if model is None:
            if options:
                raise ValidationError("Engine options belong to one model; give the preset's model")
            return None, values, {}
        info = self._models.get(model)
        manifest = self._registry.manifest(info.engine)
        check_params(manifest, info.variant, values, strict=True)
        return info.engine, values, check_options(manifest, info.variant, options)

    def create(self, body: PresetInput) -> Preset:
        name = body.name.strip()
        if not name:
            raise ValidationError("A preset needs a name")
        engine, params, options = self._checked(body.model, body.params, body.engine_options)
        if self._db.fetchone("SELECT id FROM presets WHERE name = ? AND model IS ?", (name, body.model)):
            raise ConflictError(f"There is already a preset {name!r} for {body.model or 'every model'}", {"name": name})
        stamp = now_iso()
        preset = Preset(
            id=new_id("p"), name=name, engine=engine, model=body.model, voice_id=body.voice_id, params=params, engine_options=options, created_at=stamp, updated_at=stamp
        )
        self._db.execute(
            "INSERT INTO presets(id, name, engine, model, voice_id, params, engine_options, is_default, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (preset.id, name, engine, body.model, body.voice_id, json.dumps(params), json.dumps(options), stamp, stamp),
        )
        self._changed(preset.id)
        return preset

    def update(self, key: str, patch: PresetPatch) -> Preset:
        preset = self.get(key)
        name = patch.name.strip() if patch.name is not None else preset.name
        if not name:
            raise ValidationError("A preset needs a name")
        if name != preset.name and self._db.fetchone("SELECT id FROM presets WHERE name = ? AND model IS ? AND id != ?", (name, preset.model, preset.id)):
            raise ConflictError(f"There is already a preset {name!r} for {preset.model or 'every model'}", {"name": name})
        _, params, options = self._checked(
            preset.model, preset.params if patch.params is None else patch.params, preset.engine_options if patch.engine_options is None else patch.engine_options
        )
        self._db.execute(
            "UPDATE presets SET name = ?, params = ?, engine_options = ?, updated_at = ? WHERE id = ?", (name, json.dumps(params), json.dumps(options), now_iso(), preset.id)
        )
        self._changed(preset.id)
        return self.get(preset.id)

    def delete(self, key: str) -> None:
        preset = self.get(key)
        self._db.execute("DELETE FROM presets WHERE id = ?", (preset.id,))
        self._changed(preset.id)

    def set_default(self, key: str, default: bool = True) -> Preset:
        """Make it the one applied when its model (and voice) is picked, replacing the previous default."""
        preset = self.get(key)
        if default and preset.model is None:
            raise ValidationError("Only a model's own preset can be its default")
        with self._db.transaction() as conn:
            if default:
                conn.execute("UPDATE presets SET is_default = 0 WHERE model IS ? AND voice_id IS ?", (preset.model, preset.voice_id))
            conn.execute("UPDATE presets SET is_default = ?, updated_at = ? WHERE id = ?", (int(default), now_iso(), preset.id))
        self._changed(preset.id)
        return self.get(preset.id)

    def default_for(self, model: str, voice_id: str | None = None) -> Preset | None:
        """The default for this voice with the model, else the model's own default."""
        if voice_id:
            row = self._db.fetchone("SELECT * FROM presets WHERE is_default = 1 AND model = ? AND voice_id = ?", (model, voice_id))
            if row is not None:
                return self._row(row)
        row = self._db.fetchone("SELECT * FROM presets WHERE is_default = 1 AND model = ? AND voice_id IS NULL", (model,))
        return self._row(row) if row is not None else None

    def apply(self, key: str, req: SpeakRequest) -> SpeakRequest:
        """``req`` with the preset's values wherever the request sets none."""
        preset = self.get(key)
        if preset.model is not None and preset.model != req.model:
            raise ValidationError(f"The preset {preset.name!r} is for {preset.model}, not {req.model}", {"preset": preset.id, "model": preset.model})
        given = req.params.model_dump(exclude_none=True)
        params = CommonParams.model_validate({**preset.params, **given})
        options = {**preset.engine_options, **req.engine_options}
        return req.model_copy(update={"params": params, "engine_options": options})

    def _changed(self, preset_id: str) -> None:
        self._events.publish(PresetsChangedEvent(ids=[preset_id]))
