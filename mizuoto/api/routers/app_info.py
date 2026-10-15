"""App version, health and what the interface needs to know about this server."""

import socket

from fastapi import APIRouter, Request

from mizuoto.api.deps import ServicesDep, is_local
from mizuoto.core.inference.service import torch_version
from mizuoto.core.record import Record
from mizuoto.version import VERSION

router = APIRouter(prefix="/v1/app", tags=["app"])

FIRST_RUN_KEY = "first_run_done"


class AppVersion(Record):
    version: str


class Health(Record):
    status: str
    auth_required: bool


class AppMeta(Record):
    local: bool
    """Whether this request comes from the server's own machine: only then can it read server paths or reveal files."""
    host: str
    data_dir: str
    features: list[str]
    first_run: bool
    """No model is installed and the first-run guide was not dismissed."""
    default_model: str
    torch: str | None = None
    """The installed torch's version; None when it is missing (no engine can run: ``/runtime/doctor`` says what to install)."""


@router.get("/version", operation_id="get_app_version")
def get_version() -> AppVersion:
    return AppVersion(version=VERSION)


@router.get("/health", operation_id="get_health")
def get_health(services: ServicesDep) -> Health:
    return Health(status="ok", auth_required=bool(services.settings.settings.server.access_token))


@router.get("/meta", operation_id="get_app_meta")
def get_meta(request: Request, services: ServicesDep) -> AppMeta:
    local = is_local(request)
    installed = any(m.state == "installed" for m in services.models.list_models(installed=True))
    dismissed = bool(services.db.get_client_state(FIRST_RUN_KEY))
    return AppMeta(
        local=local,
        host=socket.gethostname(),
        data_dir=str(services.settings.data_dir) if local else "",
        features=["speak", "voices", "models", "jobs"],
        first_run=not installed and not dismissed,
        default_model=services.settings.settings.tts.default_model,
        torch=torch_version(),
    )
