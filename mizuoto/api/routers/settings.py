"""Settings and client state."""

import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Path

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.errors import ValidationError
from mizuoto.core.settings import SettingsView

router = APIRouter(prefix="/v1", tags=["settings"])

MAX_CLIENT_STATE_BYTES = 256 * 1024
ClientStateKey = Annotated[str, Path(pattern=r"^[A-Za-z0-9_.:-]{1,100}$")]


@router.get("/settings", operation_id="get_settings")
def get_settings(services: ServicesDep) -> SettingsView:
    return services.settings.view()


@router.patch("/settings", operation_id="update_settings", responses=ERROR_RESPONSES)
def update_settings(
    services: ServicesDep,
    patch: Annotated[dict[str, Any], Body(description="Partial settings, deep-merged. An access token set to null is cleared; an omitted one is kept.")],
) -> SettingsView:
    if "server" in patch and isinstance(patch["server"], dict) and "access_token" in patch["server"] and patch["server"]["access_token"] == "":
        raise ValidationError("An empty access token is not allowed; send null to clear it")
    services.settings.update(patch)  # publishes settings_changed and rescans on new paths (core/context.py)
    return services.settings.view()


@router.get("/client-state/{key}", operation_id="get_client_state")
def get_client_state(services: ServicesDep, key: ClientStateKey) -> Any:
    return services.db.get_client_state(key)


@router.put("/client-state/{key}", operation_id="set_client_state", responses=ERROR_RESPONSES)
def set_client_state(services: ServicesDep, key: ClientStateKey, value: Annotated[Any, Body()]) -> Any:
    if len(json.dumps(value)) > MAX_CLIENT_STATE_BYTES:
        raise ValidationError("Value too large")
    services.db.set_client_state(key, value)
    return value
