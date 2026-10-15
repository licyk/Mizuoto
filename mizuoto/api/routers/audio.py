"""Audio given to Mizuoto: uploads, server paths, waveforms, transcripts."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.api.files import audio_response
from mizuoto.api.uploads import RAW_BODY, read_chunks, spool
from mizuoto.core.asr.service import Transcript
from mizuoto.core.audio.service import AudioFile, BrowseListing, Peaks
from mizuoto.core.record import Record

router = APIRouter(prefix="/v1/audio", tags=["audio"], responses=ERROR_RESPONSES)


class PathsRequest(Record):
    paths: list[str]


class TranscribeRequest(Record):
    audio_id: str
    language: str | None = None
    model: str | None = None


@router.put("/files", operation_id="upload_audio", openapi_extra=RAW_BODY)
async def upload(request: Request, services: ServicesDep, name: str = "audio.wav") -> AudioFile:
    """The request body is the file itself (no multipart)."""
    spooled = await spool(request, services.settings.data_dir / "uploads" / "incoming")
    try:
        return await run_in_threadpool(services.audio.upload, name, read_chunks(spooled))
    finally:
        spooled.unlink(missing_ok=True)


@router.post("/files/paths", operation_id="register_audio_paths", dependencies=[LocalDep])
def register_paths(services: ServicesDep, body: PathsRequest) -> list[AudioFile]:
    """Files on the server's disk, inside ``paths.browse_roots`` (local browsers only)."""
    return [services.audio.register_path(services.audio.check_root(Path(p))) for p in body.paths]


@router.post("/files/from-output/{output_id}", operation_id="register_output_audio")
def from_output(services: ServicesDep, output_id: str) -> AudioFile:
    """An earlier output as an audio file (to use it as a reference)."""
    return services.audio.register_path(services.outputs.path_of(output_id), origin="output")


@router.get("/browse", operation_id="browse_server", dependencies=[LocalDep])
def browse(services: ServicesDep, path: str | None = None, dirs_only: bool = False) -> BrowseListing:
    """Folders and audio files on the server, inside ``paths.browse_roots`` (local browsers only)."""
    return services.audio.browse(path, dirs_only)


@router.get("/files/{audio_id}", operation_id="get_audio")
def get_audio(services: ServicesDep, audio_id: str) -> AudioFile:
    return services.audio.get(audio_id)


@router.get("/files/{audio_id}/content", operation_id="get_audio_content", response_class=FileResponse)
def get_content(services: ServicesDep, audio_id: str) -> FileResponse:
    return audio_response(services.audio.path_of(audio_id))


@router.get("/files/{audio_id}/peaks", operation_id="get_audio_peaks")
def get_peaks(services: ServicesDep, audio_id: str, points: int = 1024) -> Peaks:
    return services.audio.peaks(services.audio.path_of(audio_id), points)


@router.post("/transcribe", operation_id="transcribe_audio")
def transcribe(services: ServicesDep, body: TranscribeRequest) -> Transcript:
    """What is said in a clip (speech recognition; cached)."""
    return services.asr.transcribe(services.audio.get(body.audio_id), body.language, body.model)
