"""Engines: what each offers, its licence and settings, and what it needs on this machine."""

from typing import Annotated, Any

from fastapi import APIRouter, Body

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.engines.models import EngineManifest, EngineStatus, LicenceText
from mizuoto.core.errors import ValidationError
from mizuoto.core.inference.models import DoctorReport
from mizuoto.core.jobs.models import Job
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/engines", tags=["engines"], responses=ERROR_RESPONSES)


class EngineDetail(Record):
    status: EngineStatus
    manifest: EngineManifest
    settings: dict[str, Any]
    """``engines.<id>`` as configured (device, dtype, load options…)."""


class LicenceAcceptance(Record):
    licence: str
    """The ``id@version`` shown to the user; it must be the current one."""


@router.get("", operation_id="list_engines")
def list_engines(services: ServicesDep) -> list[EngineStatus]:
    return services.engines.list_engines()


@router.get("/{engine_id}", operation_id="get_engine")
def get_engine(services: ServicesDep, engine_id: str) -> EngineDetail:
    es = services.settings.settings.engines.get(engine_id)
    return EngineDetail(
        status=services.engines.status(engine_id),
        manifest=services.engines.manifest(engine_id),
        settings=es.model_dump() if es else {},
    )


@router.get("/{engine_id}/licence", operation_id="get_engine_licence")
def licence(services: ServicesDep, engine_id: str) -> LicenceText:
    """The licence in full, as it must be read before it is accepted."""
    return services.engines.licence_text(engine_id)


@router.post("/{engine_id}/licence", operation_id="accept_engine_licence")
def accept_licence(services: ServicesDep, engine_id: str, body: LicenceAcceptance) -> EngineStatus:
    return services.engines.accept_licence(engine_id, body.licence)


@router.delete("/{engine_id}/licence", operation_id="revoke_engine_licence")
def revoke_licence(services: ServicesDep, engine_id: str) -> EngineStatus:
    return services.engines.revoke_licence(engine_id)


@router.patch("/{engine_id}", operation_id="configure_engine")
def configure(
    services: ServicesDep,
    engine_id: str,
    patch: Annotated[dict[str, Any], Body(description="Fields of engines.<id>: enabled, device, dtype, load_options, idle_minutes, pinned_models")],
) -> EngineStatus:
    return services.engines.configure(engine_id, patch)


@router.get("/{engine_id}/doctor", operation_id="engine_doctor")
def doctor(services: ServicesDep, engine_id: str) -> DoctorReport:
    """The doctor's report for this engine: torch against the hardware, the engine's packages and optional extras."""
    services.engines.manifest(engine_id)
    return services.runtime.doctor(engine_id)


class OpRequest(Record):
    model: str
    """The model to load for it."""
    payload: dict[str, Any] = {}


@router.post("/{engine_id}/ops/{op}", operation_id="run_engine_op", dependencies=[LocalDep])
def run_op(services: ServicesDep, engine_id: str, op: str, body: OpRequest) -> Job:
    """An operation the engine's manifest declares, as a job; its payload may name server files (local browsers only)."""
    if body.model.split("/", 1)[0] != engine_id:
        raise ValidationError(f"{body.model} is not a model of {engine_id}")
    return services.tts.engine_op(body.model, op, body.payload, origin="api")
