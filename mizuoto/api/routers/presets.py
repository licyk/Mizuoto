"""Named parameter sets for a model, or common parameters for any model."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.presets.models import Preset, PresetInput, PresetPatch
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/presets", tags=["presets"], responses=ERROR_RESPONSES)


class DefaultRequest(Record):
    default: bool = True


@router.get("", operation_id="list_presets")
def list_presets(services: ServicesDep, model: str | None = None) -> list[Preset]:
    """Every preset, or those a model can use (its own and the common ones)."""
    return services.presets.list_presets(model)


@router.post("", operation_id="create_preset")
def create_preset(services: ServicesDep, body: PresetInput) -> Preset:
    return services.presets.create(body)


@router.patch("/{preset_id}", operation_id="update_preset")
def update_preset(services: ServicesDep, preset_id: str, body: PresetPatch) -> Preset:
    return services.presets.update(preset_id, body)


@router.delete("/{preset_id}", operation_id="delete_preset")
def delete_preset(services: ServicesDep, preset_id: str) -> None:
    services.presets.delete(preset_id)


@router.post("/{preset_id}/default", operation_id="set_default_preset")
def set_default(services: ServicesDep, preset_id: str, body: DefaultRequest) -> Preset:
    """Make it (or stop it being) the preset applied when its model, and voice, is picked."""
    return services.presets.set_default(preset_id, body.default)
