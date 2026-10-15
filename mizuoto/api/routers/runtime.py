"""The runtime: the models loaded in this process, unloading them, and the doctor's report."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.inference.models import DoctorReport, RuntimeStatus, UnloadRequest

router = APIRouter(prefix="/v1/runtime", tags=["runtime"], responses=ERROR_RESPONSES)


@router.get("", operation_id="get_runtime")
def get_runtime(services: ServicesDep) -> RuntimeStatus:
    """The loaded models (their bytes, device, last use) and the engines' state."""
    return services.runtime.status()


@router.post("/unload", operation_id="unload_models")
def unload(services: ServicesDep, body: UnloadRequest) -> RuntimeStatus:
    """Unload a model, or every model not in use."""
    return services.runtime.unload(body.model)


@router.get("/doctor", operation_id="get_doctor")
def doctor(services: ServicesDep) -> DoctorReport:
    """torch against the hardware (with the command that installs the right build), and each engine's availability and
    optional extras."""
    return services.runtime.doctor()
