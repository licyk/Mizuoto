"""Compute devices and memory."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.core.compute.models import ComputeInfo, ComputeUsage

router = APIRouter(prefix="/v1/compute", tags=["compute"])


@router.get("/info", operation_id="get_compute_info")
def get_info(services: ServicesDep, refresh: bool = False) -> ComputeInfo:
    return services.compute.info(refresh=refresh)


@router.get("/usage", operation_id="get_compute_usage")
def get_usage(services: ServicesDep) -> ComputeUsage:
    return services.compute.usage()


@router.post("/release", operation_id="release_compute")
def release(services: ServicesDep) -> ComputeUsage:
    """Free GPU memory: idle engines unload their models."""
    return services.compute.release()
