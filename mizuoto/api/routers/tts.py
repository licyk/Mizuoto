"""Speech synthesis: what a model takes, the plan of a request, running it, and hearing it as it is made."""

import asyncio
import contextlib
import json
from typing import Any, Literal

import pydantic
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mizuoto.api.deps import ServicesDep
from mizuoto.api.errors import ERROR_RESPONSES
from mizuoto.api.streaming import SETTLE_SECONDS, job_error_body, streaming_response
from mizuoto.core.context import Services
from mizuoto.core.engines.models import ModelParamSpecs
from mizuoto.core.errors import MizuotoError, OperationTimeoutError, ValidationError
from mizuoto.core.jobs.models import FINISHED_STATES, Job
from mizuoto.core.record import Record
from mizuoto.core.tts.models import ConvertRequest, EditRequest, SegmentPreview, SpeakRequest, SynthesisPlan
from mizuoto.core.tts.service import TextFeed
from mizuoto.core.tts.streaming import error_body, pcm_sink
from mizuoto.core.voices.models import VoiceFit

router = APIRouter(prefix="/v1/tts", tags=["tts"], responses=ERROR_RESPONSES)


class SegmentRequest(Record):
    text: str
    model: str | None = None
    segmentation: str = "engine"


@router.get("/models/{engine}/{name}/spec", operation_id="get_model_spec")
def model_spec(services: ServicesDep, engine: str, name: str) -> ModelParamSpecs:
    """Everything a form for this model needs: voice modes, languages, parameters, options."""
    return services.tts.model_spec(f"{engine}/{name}")


@router.get("/models/{engine}/{name}/voice-fits", operation_id="get_model_voice_fits")
def voice_fits(services: ServicesDep, engine: str, name: str) -> list[VoiceFit]:
    """How the model can use each library voice."""
    return services.voices.fits_for_model(f"{engine}/{name}")


@router.post("/plan", operation_id="plan_speech")
def plan(services: ServicesDep, req: SpeakRequest, stream: bool = False) -> SynthesisPlan:
    """Check a request and say what it will do, without running it."""
    return services.tts.plan(req, stream)


@router.post("/speak", operation_id="speak", response_model=Job)
def speak(services: ServicesDep, req: SpeakRequest, wait: bool = False, timeout: float = 600.0, stream: bool = False, stream_format: Literal["wav", "pcm"] = "wav") -> Any:
    """Queue a synthesis. With ``wait`` the answer is the finished job (``result.output_id``), or 504 after ``timeout`` seconds.

    With ``stream`` the answer is the audio as it is made: a WAV of unknown length or bare mono 16-bit
    PCM (``stream_format``), at the rate in ``X-Mizuoto-Sample-Rate``; the job (``X-Mizuoto-Job``) still
    stores the whole output.
    """
    if stream:
        if req.text_stream:
            raise ValidationError("Streamed text needs the /tts/stream WebSocket")
        return streaming_response(services, req, stream_format)
    job = services.tts.speak(req, origin="api")
    if not wait:
        return job
    job = services.jobs.wait(job.id, max(1.0, min(timeout, 3600.0)))
    if job.state not in FINISHED_STATES:
        raise OperationTimeoutError(f"The synthesis did not finish within {timeout:.0f} s; it goes on", {"stage": "speak", "job_id": job.id})
    return job


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """Speak and hear it as it is made.

    The client sends one JSON ``SpeakRequest``. The server answers ``{"type": "queued", "job_id"}``,
    then ``{"type": "start", "sample_rate", "granularity"}``, binary messages of mono 16-bit
    little-endian PCM at that rate with ``{"type": "segment", "index"}`` before each segment, and
    finally ``{"type": "end", "output_ids", "output_id"}`` or ``{"type": "error", "code", "message",
    "detail"}``. The client may send ``{"type": "cancel"}``. With ``text_stream`` (models with
    ``stream_text_input``) it sends the rest of the text as ``{"type": "text", "text"}`` messages and
    ``{"type": "flush"}`` after the last; leaving before the flush cancels the job.
    """
    services: Services = websocket.app.state.services
    await websocket.accept()
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def emit(kind: str, value: Any) -> None:  # on the job's thread
        with contextlib.suppress(RuntimeError):  # the loop has closed
            loop.call_soon_threadsafe(events.put_nowait, (kind, value))

    async def fail(body: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", **body})
            await websocket.close()

    try:
        first = await websocket.receive_text()
    except WebSocketDisconnect:
        return
    try:
        req = SpeakRequest.model_validate_json(first)
    except pydantic.ValidationError as e:
        await fail({"code": "invalid_input", "message": "The first message must be a speak request", "detail": {"errors": json.loads(e.json(include_url=False))}})
        return
    feed = TextFeed(max(0, services.settings.settings.tts.max_text_chars - len(req.text))) if req.text_stream else None
    try:
        job = await asyncio.to_thread(lambda: services.tts.speak(req, origin="api", stream=pcm_sink(emit, text=feed)))
    except MizuotoError as e:
        await fail(error_body(e))
        return
    await websocket.send_json({"type": "queued", "job_id": job.id})

    async def cancel() -> None:
        with contextlib.suppress(MizuotoError):
            await asyncio.to_thread(services.jobs.cancel, job.id)

    async def read() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            try:
                data = json.loads(message.get("text") or "")
            except ValueError:
                continue
            kind = data.get("type") if isinstance(data, dict) else None
            if kind == "cancel":
                await cancel()
            elif kind in ("text", "flush") and feed is not None:
                try:
                    if kind == "text":
                        feed.put(str(data.get("text") or ""))
                    else:
                        feed.close()
                except MizuotoError as e:
                    emit("error", error_body(e))
                    await cancel()

    reader = asyncio.create_task(read())
    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(events.get(), 0.5)
            except asyncio.TimeoutError:
                if reader.done():
                    break  # the client left
                current = await asyncio.to_thread(services.jobs.get, job.id)
                if current.state in FINISHED_STATES:
                    await asyncio.sleep(SETTLE_SECONDS)
                    if events.empty():
                        await websocket.send_json({"type": "error", "job_id": job.id, **job_error_body(current)})
                        break
                continue
            if kind == "audio":
                await websocket.send_bytes(value)
            elif kind == "start":
                await websocket.send_json({"type": "start", "job_id": job.id, **value})
            elif kind == "segment":
                await websocket.send_json({"type": "segment", "index": value})
            elif kind == "end":
                await websocket.send_json({"type": "end", "job_id": job.id, "output_ids": value, "output_id": value[0] if value else None})
                break
            elif kind == "error":
                await websocket.send_json({"type": "error", "job_id": job.id, **value})
                break
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()
        if feed is not None and not feed.closed:
            await cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


@router.post("/convert", operation_id="convert_voice")
def convert(services: ServicesDep, req: ConvertRequest) -> Job:
    """Say what a recording says in another voice (models with voice conversion)."""
    return services.tts.convert(req, origin="api")


@router.post("/edit", operation_id="edit_speech")
def edit(services: ServicesDep, req: EditRequest) -> Job:
    """Replace spans of a recording with new speech in the same voice (models with speech editing)."""
    return services.tts.edit(req, origin="api")


@router.post("/segment", operation_id="preview_segments")
def segment(services: ServicesDep, body: SegmentRequest) -> SegmentPreview:
    return services.tts.segment_preview(body.text, body.model, body.segmentation)
