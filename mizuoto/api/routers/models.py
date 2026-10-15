"""Engine models: the catalogue, downloads, verification, imports."""

from pathlib import Path

from fastapi import APIRouter, status

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.core.jobs.models import Job
from mizuoto.core.models.models import Detected, ModelInfo, VerifyResult
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/models", tags=["models"], responses=ERROR_RESPONSES)


class DownloadRequest(Record):
    ids: list[str]


class ImportRequest(Record):
    path: str
    """A folder on the server."""
    name: str | None = None
    move: bool = False


class PathRequest(Record):
    path: str


@router.get("", operation_id="list_models")
def list_models(services: ServicesDep, engine: str | None = None, installed: bool | None = None) -> list[ModelInfo]:
    return services.models.list_models(engine, installed)


@router.post("/download", operation_id="download_models")
def download(services: ServicesDep, body: DownloadRequest) -> Job:
    return services.models.download(body.ids, origin="api")


@router.post("/identify", operation_id="identify_model", dependencies=[LocalDep])
def identify(services: ServicesDep, body: PathRequest) -> Detected:
    """What the model folder at a server path is (local browsers only)."""
    return services.models.identify(services.audio.check_root(Path(body.path)))


@router.post("/import", operation_id="import_model", dependencies=[LocalDep])
def import_model(services: ServicesDep, body: ImportRequest) -> ModelInfo:
    """Copy (or move) a model folder from a server path (local browsers only)."""
    return services.models.import_model(services.audio.check_root(Path(body.path)), name=body.name, move=body.move)


@router.get("/{engine}/{name}", operation_id="get_model")
def get_model(services: ServicesDep, engine: str, name: str) -> ModelInfo:
    return services.models.get(f"{engine}/{name}")


@router.post("/{engine}/{name}/verify", operation_id="verify_model")
def verify(services: ServicesDep, engine: str, name: str) -> VerifyResult:
    return services.models.verify(f"{engine}/{name}")


@router.delete("/{engine}/{name}", operation_id="delete_model", status_code=status.HTTP_204_NO_CONTENT)
def delete(services: ServicesDep, engine: str, name: str) -> None:
    services.models.delete(f"{engine}/{name}")


@router.get("/{engine}/{name}/speakers", operation_id="get_model_speakers")
def speakers(services: ServicesDep, engine: str, name: str) -> list[str]:
    return services.tts.model_spec(f"{engine}/{name}").speakers
