"""Streamed synthesis over HTTP: the audio of a request as it is made (``mizuoto.core.tts.streaming``)."""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from fastapi.responses import StreamingResponse

from mizuoto.core.jobs.models import FINISHED_STATES, Job
from mizuoto.core.tts.models import SpeakRequest
from mizuoto.core.tts.streaming import pcm_sink, wav_header

if TYPE_CHECKING:
    from mizuoto.core.context import Services

SETTLE_SECONDS = 0.3
"""How long a stream waits for the sink's last events after its job has ended."""


def job_error_body(job: Job) -> dict[str, Any]:
    """Why a job that ended without its stream's ``end`` did so."""
    if job.error is not None:
        return job.error.model_dump(mode="json")
    if job.state == "cancelled":
        return {"code": "cancelled", "message": "Cancelled", "detail": {}}
    return {"code": "internal_error", "message": f"The job ended {job.state}", "detail": {}}


def streaming_response(services: Services, req: SpeakRequest, stream_format: str, sample_rate: int | None = None) -> StreamingResponse:
    """The audio of ``req`` as it is made: a WAV of unknown length or bare PCM16 (``stream_format``)."""
    planned = services.tts.plan(req, stream=True)
    rate = sample_rate or req.output.sample_rate or planned.model_sample_rate
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    job = services.tts.speak(req, origin="api", stream=pcm_sink(lambda kind, value: events.put((kind, value)), sample_rate=rate))

    def body() -> Iterator[bytes]:
        if stream_format == "wav":
            yield wav_header(rate)
        while True:
            try:
                kind, value = events.get(timeout=0.5)
            except queue.Empty:
                if services.jobs.get(job.id).state in FINISHED_STATES:
                    time.sleep(SETTLE_SECONDS)
                    if events.empty():
                        return
                continue
            if kind == "audio":
                yield value
            elif kind in ("end", "error"):
                return

    media = "audio/wav" if stream_format == "wav" else "audio/pcm"
    return StreamingResponse(body(), media_type=media, headers={"X-Mizuoto-Job": job.id, "X-Mizuoto-Sample-Rate": str(rate), "Cache-Control": "no-store"})
