"""Datasets: clips with transcripts for training."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.api.uploads import RAW_BODY, spool
from mizuoto.core.datasets.models import Dataset, DatasetPage, DatasetReport, DatasetRow, ImportLayout, PrepareRequest, RowIssue, RowPatch
from mizuoto.core.jobs.models import Job
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/datasets", tags=["datasets"], responses=ERROR_RESPONSES)


class DatasetInput(Record):
    name: str
    language: str | None = None
    consent: str = ""
    description: str = ""


class DatasetPatch(Record):
    name: str | None = None
    language: str | None = None
    consent: str | None = None
    description: str | None = None


class DatasetAdded(Record):
    dataset: Dataset
    warnings: list[str] = []


class DatasetImportRequest(Record):
    path: str
    """A folder on the server."""
    layout: ImportLayout = "auto"


class AudioIdsRequest(Record):
    audio_ids: list[str]
    texts: list[str | None] = []
    speaker: str = ""


class VoiceRequest(Record):
    voice: str


class OutputsRequest(Record):
    output_ids: list[str]
    speaker: str = ""


class RowIds(Record):
    ids: list[str]


@router.get("", operation_id="list_datasets")
def list_datasets(services: ServicesDep) -> list[Dataset]:
    return services.datasets.list_datasets()


@router.post("", operation_id="create_dataset")
def create(services: ServicesDep, body: DatasetInput) -> Dataset:
    return services.datasets.create(body.name, body.language, body.consent, body.description)


@router.get("/{dataset_id}", operation_id="get_dataset")
def get(services: ServicesDep, dataset_id: str) -> Dataset:
    return services.datasets.get(dataset_id)


@router.patch("/{dataset_id}", operation_id="update_dataset")
def update(services: ServicesDep, dataset_id: str, body: DatasetPatch) -> Dataset:
    return services.datasets.update(dataset_id, body.name, body.language, body.consent, body.description)


@router.delete("/{dataset_id}", operation_id="delete_dataset")
def delete(services: ServicesDep, dataset_id: str) -> None:
    services.datasets.delete(dataset_id)


@router.put("/{dataset_id}/files", operation_id="upload_dataset_file", openapi_extra=RAW_BODY)
async def upload(request: Request, services: ServicesDep, dataset_id: str, name: str, text: str = "", speaker: str = "") -> DatasetAdded:
    """One audio file as the request body; ``text`` is its transcript (empty: transcribed by Prepare)."""
    folder = services.settings.data_dir / "uploads" / "incoming"
    spooled = await spool(request, folder)
    named = spooled.with_name(f"{spooled.stem}-{Path(name).name}")
    spooled.rename(named)
    if text:
        named.with_suffix(".lab").write_text(text, encoding="utf-8")
    try:
        dataset, warnings = await run_in_threadpool(services.datasets.add_files, dataset_id, [named], speaker)
        return DatasetAdded(dataset=dataset, warnings=warnings)
    finally:
        named.unlink(missing_ok=True)
        named.with_suffix(".lab").unlink(missing_ok=True)


@router.post("/{dataset_id}/import", operation_id="import_dataset_folder", dependencies=[LocalDep])
def import_folder(services: ServicesDep, dataset_id: str, body: DatasetImportRequest) -> DatasetAdded:
    """A folder on the server, in a known layout (local browsers only)."""
    dataset, warnings = services.datasets.import_folder(dataset_id, services.audio.check_root(Path(body.path)), body.layout)
    return DatasetAdded(dataset=dataset, warnings=warnings)


@router.post("/{dataset_id}/audio", operation_id="add_dataset_audio")
def add_audio(services: ServicesDep, dataset_id: str, body: AudioIdsRequest) -> DatasetAdded:
    dataset, warnings = services.datasets.add_audio(dataset_id, body.audio_ids, body.texts, body.speaker)
    return DatasetAdded(dataset=dataset, warnings=warnings)


@router.post("/{dataset_id}/voice", operation_id="add_dataset_voice")
def add_voice(services: ServicesDep, dataset_id: str, body: VoiceRequest) -> Dataset:
    return services.datasets.add_voice(dataset_id, body.voice)


@router.post("/{dataset_id}/outputs", operation_id="add_dataset_outputs")
def add_outputs(services: ServicesDep, dataset_id: str, body: OutputsRequest) -> Dataset:
    return services.datasets.add_outputs(dataset_id, body.output_ids, body.speaker)


@router.post("/{dataset_id}/prepare", operation_id="prepare_dataset")
def prepare(services: ServicesDep, dataset_id: str, body: PrepareRequest) -> Job:
    return services.datasets.prepare(dataset_id, body, origin="api")


@router.get("/{dataset_id}/report", operation_id="check_dataset")
def report(services: ServicesDep, dataset_id: str) -> DatasetReport:
    return services.datasets.check(dataset_id)


@router.get("/{dataset_id}/rows", operation_id="list_dataset_rows")
def rows(services: ServicesDep, dataset_id: str, offset: int = 0, limit: int = 100, q: str | None = None, issue: RowIssue | None = None) -> DatasetPage:
    return services.datasets.rows(dataset_id, offset, min(limit, 1000), q, issue)


@router.patch("/{dataset_id}/rows/{row_id}", operation_id="update_dataset_row")
def update_row(services: ServicesDep, dataset_id: str, row_id: str, body: RowPatch) -> DatasetRow:
    return services.datasets.update_row(dataset_id, row_id, body)


@router.post("/{dataset_id}/rows/delete", operation_id="delete_dataset_rows")
def delete_rows(services: ServicesDep, dataset_id: str, body: RowIds) -> Dataset:
    return services.datasets.delete_rows(dataset_id, body.ids)


@router.get("/{dataset_id}/rows/{row_id}/audio", operation_id="get_dataset_row_audio", response_class=FileResponse)
def row_audio(services: ServicesDep, dataset_id: str, row_id: str) -> FileResponse:
    return FileResponse(services.datasets.audio_path(dataset_id, row_id), media_type="audio/flac")
