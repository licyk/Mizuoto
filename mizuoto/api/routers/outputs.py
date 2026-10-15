"""Every synthesized file and how it was made."""

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.api.files import audio_response, content_disposition
from mizuoto.core.audio.service import Peaks
from mizuoto.core.errors import NotFoundError
from mizuoto.core.files import reveal
from mizuoto.core.outputs.models import Output, OutputPage, OutputQuery
from mizuoto.core.record import Record
from mizuoto.core.tts.models import SpeakRequest

router = APIRouter(prefix="/v1/outputs", tags=["outputs"], responses=ERROR_RESPONSES)


class IdsRequest(Record):
    ids: list[str]


class OutputPatch(Record):
    name: str | None = None
    favourite: bool | None = None


@router.get("", operation_id="list_outputs")
def list_outputs(
    services: ServicesDep,
    voice_id: str | None = None,
    model: str | None = None,
    engine: str | None = None,
    kind: str | None = None,
    job_id: str | None = None,
    q: str | None = None,
    favourite: bool | None = None,
    cursor: str | None = None,
    limit: int = 50,
) -> OutputPage:
    query = OutputQuery.model_validate(
        {
            "voice_id": voice_id,
            "model": model,
            "engine": engine,
            "kind": kind,
            "job_id": job_id,
            "text": q,
            "favourite": favourite,
            "cursor": cursor,
            "limit": max(1, min(limit, 500)),
        }
    )
    return services.outputs.list_outputs(query)


@router.post("/delete", operation_id="delete_outputs")
def delete(services: ServicesDep, body: IdsRequest) -> list[str]:
    """Move outputs to the trash; returns the ids removed."""
    return services.outputs.delete(body.ids)


@router.post("/zip", operation_id="zip_outputs", response_class=StreamingResponse, responses={200: {"content": {"application/zip": {}}}})
def zip_outputs(services: ServicesDep, body: IdsRequest) -> StreamingResponse:
    return StreamingResponse(
        services.outputs.zip(body.ids), media_type="application/zip", headers={"Content-Disposition": content_disposition("attachment", "mizuoto-outputs.zip")}
    )


@router.get("/{output_id}", operation_id="get_output")
def get_output(services: ServicesDep, output_id: str) -> Output:
    return services.outputs.get(output_id)


@router.patch("/{output_id}", operation_id="update_output")
def update(services: ServicesDep, output_id: str, body: OutputPatch) -> Output:
    return services.outputs.update(output_id, body.name, body.favourite)


@router.get("/{output_id}/content", operation_id="get_output_content", response_class=FileResponse)
def content(services: ServicesDep, output_id: str, download: bool = False) -> FileResponse:
    return audio_response(services.outputs.path_of(output_id), download)


@router.get("/{output_id}/peaks", operation_id="get_output_peaks")
def peaks(services: ServicesDep, output_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.outputs.path_of(output_id), points)


@router.get("/{output_id}/rerun", operation_id="get_output_rerun")
def rerun(services: ServicesDep, output_id: str) -> SpeakRequest:
    """The request that made this output, to run it again (same seed when one was set)."""
    out = services.outputs.get(output_id)
    if not out.job_id:
        raise NotFoundError("This output has no request to run again", {"kind": "job", "id": output_id})
    request = {k: v for k, v in services.jobs.get(out.job_id).request.items() if not k.startswith("_")}
    return SpeakRequest.model_validate(request)


@router.post("/{output_id}/reveal", operation_id="reveal_output", status_code=204, dependencies=[LocalDep])
def reveal_output(services: ServicesDep, output_id: str) -> None:
    reveal(services.outputs.path_of(output_id))
