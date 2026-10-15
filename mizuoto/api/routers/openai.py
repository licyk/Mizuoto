"""The OpenAI-compatible speech endpoint, mounted at ``{prefix}/v1``.

Errors take OpenAI's shape (``{"error": {"message", "type", "code"}}``) so its clients show them.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response

from mizuoto.api.deps import ServicesDep
from mizuoto.api.streaming import streaming_response
from mizuoto.core.audio.io import decode
from mizuoto.core.errors import MizuotoError, ValidationError
from mizuoto.core.jobs.models import FINISHED_STATES
from mizuoto.core.record import Record
from mizuoto.core.tts.models import OPENAI_PCM_RATE, OpenAiSpeechRequest
from mizuoto.core.tts.streaming import pcm16

router = APIRouter(prefix="/v1", tags=["openai"])

MEDIA_TYPES = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/mp4", "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm"}
WAIT_SECONDS = 3600.0


class OpenAiModel(Record):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str


class OpenAiModelList(Record):
    object: str = "list"
    data: list[OpenAiModel]


def _openai_error(e: MizuotoError) -> JSONResponse:
    kind = "invalid_request_error" if e.http_status < 500 else "server_error"
    return JSONResponse(status_code=e.http_status, content={"error": {"message": e.message, "type": kind, "param": None, "code": e.code}})


@router.post("/audio/speech", operation_id="openai_speech", response_class=Response, responses={200: {"content": {t: {} for t in MEDIA_TYPES.values()}}})
def speech(services: ServicesDep, body: OpenAiSpeechRequest) -> Any:
    """Speak ``input`` and answer with the audio; ``stream_format: audio`` with ``wav`` or ``pcm`` streams it."""
    try:
        req = services.tts.from_openai(body)
        if body.stream_format == "sse":
            raise ValidationError("Server-sent events are not supported; use stream_format 'audio'")
        if body.stream_format == "audio" and body.response_format in ("wav", "pcm"):
            return streaming_response(services, req, body.response_format, OPENAI_PCM_RATE if body.response_format == "pcm" else None)
        job = services.tts.speak(req, origin="openai")
        job = services.jobs.wait(job.id, WAIT_SECONDS)
        if job.state not in FINISHED_STATES:
            raise MizuotoError("The synthesis took too long; it goes on as job " + job.id)
        if job.state != "completed" or not job.result or not job.result.get("output_id"):
            error = job.error
            raise MizuotoError(error.message if error else f"The synthesis ended {job.state}")
        path: Path = services.outputs.path_of(str(job.result["output_id"]))
    except MizuotoError as e:
        return _openai_error(e)
    headers = {"X-Mizuoto-Job": job.id, "X-Mizuoto-Model": req.model}
    if body.response_format == "pcm":
        samples, _ = decode(path, OPENAI_PCM_RATE)
        return Response(pcm16(samples), media_type=MEDIA_TYPES["pcm"], headers=headers)
    return FileResponse(path, media_type=MEDIA_TYPES[body.response_format], headers=headers)


@router.get("/models", operation_id="openai_models")
def models(services: ServicesDep) -> OpenAiModelList:
    """Installed speech models, and the aliases in ``tts.model_aliases``."""
    data = [OpenAiModel(id=m.id, owned_by=m.engine) for m in services.models.list_models(installed=True) if not services.engines.manifest(m.engine).internal]
    data += [OpenAiModel(id=alias, owned_by="mizuoto") for alias in services.settings.settings.tts.model_aliases]
    return OpenAiModelList(data=data)
