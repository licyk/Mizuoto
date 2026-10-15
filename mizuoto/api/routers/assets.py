"""Shared assets: files several models of an engine need, downloaded once."""

from fastapi import APIRouter

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.jobs.models import Job
from mizuoto.core.models.models import AssetInfo
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/assets", tags=["assets"], responses=ERROR_RESPONSES)


class AssetDownloadRequest(Record):
    ids: list[str]


@router.get("", operation_id="list_assets")
def list_assets(services: ServicesDep, engine: str | None = None) -> list[AssetInfo]:
    return services.models.list_assets(engine)


@router.post("/download", operation_id="download_assets")
def download(services: ServicesDep, body: AssetDownloadRequest) -> Job:
    return services.models.download_assets(body.ids, origin="api")


@router.delete("/{engine}/{name}", operation_id="delete_asset")
def delete(services: ServicesDep, engine: str, name: str) -> None:
    services.models.delete_asset(f"{engine}/{name}")
