"""The voice library."""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from mizuoto.api.deps import LocalDep, ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.api.files import audio_response, content_disposition
from mizuoto.api.uploads import RAW_BODY, spool
from mizuoto.core.audio.service import Peaks
from mizuoto.core.jobs.models import Job
from mizuoto.core.record import Record
from mizuoto.core.voices.models import Voice, VoiceFit, VoiceSummary, WeightsUpload

router = APIRouter(prefix="/v1/voices", tags=["voices"], responses=ERROR_RESPONSES)


class CreateVoice(Record):
    name: str
    audio_ids: list[str]
    """Uploaded clips (``PUT /audio/files``)."""
    transcripts: list[str | None] = []
    language: str | None = None
    consent: str | None = None
    description: str = ""
    tags: list[str] = []
    transcribe: bool = True


class VoiceCreated(Record):
    voice: Voice
    warnings: list[str] = []


class FromOutput(Record):
    output_id: str
    name: str
    consent: str | None = None
    description: str = ""


class FromPreset(Record):
    model: str
    speaker: str
    name: str | None = None


class VoicePatch(Record):
    name: str | None = None
    language: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    consent: str | None = None


class AddReference(Record):
    audio_id: str
    transcript: str | None = None
    language: str | None = None
    primary: bool = False
    transcribe: bool = True


class ReferencePatch(Record):
    transcript: str | None = None
    primary: bool | None = None


@router.get("", operation_id="list_voices")
def list_voices(services: ServicesDep, q: str | None = None, kind: str | None = None, language: str | None = None) -> list[VoiceSummary]:
    return services.voices.list_voices(q, kind, language)


@router.post("", operation_id="create_voice")
def create(services: ServicesDep, body: CreateVoice) -> VoiceCreated:
    clips = [services.audio.path_of(a) for a in body.audio_ids]
    voice, warnings = services.voices.create_from_audio(
        body.name, clips, list(body.transcripts), body.language, body.consent, body.description, body.tags, transcribe=body.transcribe
    )
    return VoiceCreated(voice=voice, warnings=warnings)


@router.post("/from-output", operation_id="create_voice_from_output")
def from_output(services: ServicesDep, body: FromOutput) -> Voice:
    """Keep a synthesized clip as a voice (a designed voice when it came from a design)."""
    return services.voices.create_from_output(body.output_id, body.name, body.consent, body.description)


@router.post("/preset", operation_id="create_preset_voice")
def from_preset(services: ServicesDep, body: FromPreset) -> Voice:
    return services.voices.create_preset(body.model, body.speaker, body.name)


@router.put("/import", operation_id="import_voice", openapi_extra=RAW_BODY)
async def import_voice(request: Request, services: ServicesDep, name: str | None = None) -> Voice:
    """The request body is a zip written by ``GET /voices/{id}/export``."""
    spooled = await spool(request, services.settings.data_dir / "uploads" / "incoming")
    try:
        return await run_in_threadpool(services.voices.import_archive, spooled, name)
    finally:
        spooled.unlink(missing_ok=True)


@router.get("/{voice_id}", operation_id="get_voice")
def get_voice(services: ServicesDep, voice_id: str) -> Voice:
    return services.voices.get(voice_id)


@router.patch("/{voice_id}", operation_id="update_voice")
def update(services: ServicesDep, voice_id: str, body: VoicePatch) -> Voice:
    return services.voices.update(voice_id, body.name, body.language, body.description, body.tags, body.consent)


@router.delete("/{voice_id}", operation_id="delete_voice", status_code=status.HTTP_204_NO_CONTENT)
def delete(services: ServicesDep, voice_id: str) -> None:
    services.voices.delete(voice_id)


@router.post("/{voice_id}/references", operation_id="add_voice_reference")
def add_reference(services: ServicesDep, voice_id: str, body: AddReference) -> VoiceCreated:
    voice, warnings = services.voices.add_reference(voice_id, services.audio.path_of(body.audio_id), body.transcript, body.language, body.primary, body.transcribe)
    return VoiceCreated(voice=voice, warnings=warnings)


@router.patch("/{voice_id}/references/{sha}", operation_id="update_voice_reference")
def update_reference(services: ServicesDep, voice_id: str, sha: str, body: ReferencePatch) -> Voice:
    voice = services.voices.get(voice_id)
    if body.transcript is not None:
        voice = services.voices.set_transcript(voice_id, sha, body.transcript)
    if body.primary:
        voice = services.voices.set_primary(voice_id, sha)
    return voice


@router.delete("/{voice_id}/references/{sha}", operation_id="remove_voice_reference")
def remove_reference(services: ServicesDep, voice_id: str, sha: str) -> Voice:
    return services.voices.remove_reference(voice_id, sha)


@router.post("/{voice_id}/retranscribe", operation_id="retranscribe_voice")
def retranscribe(services: ServicesDep, voice_id: str, sha: str | None = None) -> Voice:
    return services.voices.retranscribe(voice_id, sha)


@router.get("/{voice_id}/references/{sha}/content", operation_id="get_voice_reference_content", response_class=FileResponse)
def reference_content(services: ServicesDep, voice_id: str, sha: str) -> FileResponse:
    voice = services.voices.get(voice_id)
    ref = services.voices.find_reference(voice, sha)
    return audio_response(services.voices.folder(voice) / ref.file)


@router.get("/{voice_id}/references/{sha}/peaks", operation_id="get_voice_reference_peaks")
def reference_peaks(services: ServicesDep, voice_id: str, sha: str, points: int = 1024) -> Peaks:
    voice = services.voices.get(voice_id)
    ref = services.voices.find_reference(voice, sha)
    return services.audio.peaks(services.voices.folder(voice) / ref.file, points)


@router.get("/{voice_id}/fit", operation_id="get_voice_fit")
def fit(services: ServicesDep, voice_id: str, model: str) -> VoiceFit:
    return services.voices.fit(voice_id, model)


@router.get("/{voice_id}/fits", operation_id="get_voice_fits")
def fits(services: ServicesDep, voice_id: str) -> list[VoiceFit]:
    """How every installed synthesis model can use the voice."""
    return services.voices.fits(voice_id)


@router.post("/{voice_id}/prepare", operation_id="prepare_voice")
def prepare(services: ServicesDep, voice_id: str, model: str) -> Job:
    return services.tts.prepare(voice_id, model, origin="api")


@router.get("/{voice_id}/export", operation_id="export_voice", response_class=FileResponse)
def export(services: ServicesDep, voice_id: str) -> FileResponse:
    voice = services.voices.get(voice_id)
    folder = Path(tempfile.mkdtemp(prefix="mizuoto-export-"))
    path = services.voices.export(voice_id, folder / f"{voice.slug}.zip")
    return FileResponse(
        path,
        media_type="application/zip",
        headers={"Content-Disposition": content_disposition("attachment", path.name)},
        background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True),
    )


class WeightsFromPaths(Record):
    engine: str
    files: dict[str, str]
    """Role → a file on the server (``gpt``, ``sovits`` for GPT-SoVITS)."""
    move: bool = False


@router.put("/{voice_id}/weights/{engine}/{role}", operation_id="upload_voice_weights", openapi_extra=RAW_BODY)
async def upload_weights(request: Request, services: ServicesDep, voice_id: str, engine: str, role: str, name: str | None = None) -> WeightsUpload:
    """One file of the voice's own fine-tuned weights for ``engine``; the weights are added once every file is in."""
    spooled = await spool(request, services.settings.data_dir / "uploads" / "incoming")
    try:
        return await run_in_threadpool(services.voices.upload_weights, voice_id, engine, role, spooled, name)
    finally:
        spooled.unlink(missing_ok=True)


@router.post("/{voice_id}/weights", operation_id="add_voice_weights", dependencies=[LocalDep])
def add_weights(services: ServicesDep, voice_id: str, body: WeightsFromPaths) -> Voice:
    """The voice's own fine-tuned weights from files on the server (local browsers only)."""
    files = {role: services.audio.check_root(Path(path)) for role, path in body.files.items()}
    return services.voices.add_weights(voice_id, body.engine, files, move=body.move)


@router.delete("/{voice_id}/weights/{engine}", operation_id="remove_voice_weights")
def remove_weights(services: ServicesDep, voice_id: str, engine: str) -> Voice:
    return services.voices.remove_weights(voice_id, engine)
