"""Speech recognition for Mizuoto's own needs.

``AsrService.transcribe`` runs the ``asr`` engine (an internal engine that offers ``transcribe`` through
``call``). Results are cached by the audio's SHA-256 and the model, so a reference clip is transcribed once.
Every call goes through one lock: the loaded model is shared by transcription jobs and by synthesis jobs that
need a reference transcript.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from mizuoto.core.audio.service import AudioFile, AudioService
from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.db import Database
from mizuoto.core.engines import languages
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import EnvMissingError, MizuotoError, ModelMissingError
from mizuoto.core.inference.engine import EngineRequest, ModelRef
from mizuoto.core.inference.runtime import RuntimeService
from mizuoto.core.jobs.models import Job
from mizuoto.core.jobs.service import JobContext, JobService, JobSpec
from mizuoto.core.models.service import ModelService
from mizuoto.core.record import Record
from mizuoto.core.settings import SettingsService

ENGINE = "asr"


class TranscriptSegment(Record):
    start: float
    end: float
    text: str


class Transcript(Record):
    sha256: str
    model: str
    language: str | None = None
    text: str
    segments: list[TranscriptSegment] = []
    created_at: str


class AsrService:
    def __init__(
        self,
        settings: SettingsService,
        db: Database,
        registry: EngineRegistry,
        models: ModelService,
        runtime: RuntimeService,
        audio: AudioService,
        jobs: JobService | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._registry = registry
        self._models = models
        self._runtime = runtime
        self._audio = audio
        self._lock = threading.Lock()
        if jobs is not None:
            jobs.register("transcribe", self._transcribe_job)
        self._jobs = jobs

    def default_model(self) -> str:
        return self._settings.settings.tts.asr_model

    def unavailable_reason(self, model: str | None = None) -> str | None:
        """Why transcription cannot run now (None: it can)."""
        model = model or self.default_model()
        status = self._registry.status(ENGINE)
        if not status.available:
            return f"the speech recognition engine needs {', '.join(status.missing)} (mizuoto doctor)"
        try:
            self._models.require(model)
        except (ModelMissingError, MizuotoError):
            return f"the speech recognition model is not downloaded (mizuoto model download {model})"
        return None

    def available(self, model: str | None = None) -> bool:
        return self.unavailable_reason(model) is None

    def cached(self, sha256: str, model: str | None = None) -> Transcript | None:
        row = self._db.fetchone("SELECT * FROM transcripts WHERE sha256 = ? AND model = ?", (sha256, model or self.default_model()))
        return self._row(row) if row is not None else None

    def transcribe(self, audio: AudioFile | Path, language: str | None = None, model: str | None = None, ctx: JobContext | None = None) -> Transcript:
        """The transcript of a clip (cached). ``language`` None or ``auto``: Whisper decides."""
        record = audio if isinstance(audio, AudioFile) else self._audio.register_path(audio)
        model = model or self.default_model()
        language = languages.normalize(language)
        hit = self.cached(record.sha256, model)
        if hit is not None and (language in (None, "auto") or hit.language == language):
            return hit
        reason = self.unavailable_reason(model)
        if reason is not None:
            hint = "mizuoto doctor" if not self._registry.status(ENGINE).available else f"mizuoto model download {model}"
            raise EnvMissingError(f"Cannot transcribe: {reason}", {"engine": ENGINE, "hint": hint})
        info = self._models.require(model)
        wav = self._audio.canonical_wav(record)
        device = self._runtime.device_for(ENGINE)
        es = self._settings.settings.engines.get(ENGINE)
        cancel = ctx.cancel_event if ctx is not None else None
        with self._lock:
            eng = self._runtime.loaded(
                ModelRef(model, ENGINE, info.variant, info.path), device, es.dtype if es else "auto", dict(es.load_options) if es else {}, EngineRequest(cancel=cancel)
            )
            payload = {"path": str(wav), "language": None if language in (None, "auto") else language}
            result: dict[str, Any] = eng.call("transcribe", payload, EngineRequest(id=new_id("asr"), cancel=cancel))
        transcript = Transcript(
            sha256=record.sha256,
            model=model,
            language=languages.normalize(result.get("language")),
            text=str(result.get("text", "")).strip(),
            segments=[TranscriptSegment(**s) for s in result.get("segments") or []],
            created_at=now_iso(),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO transcripts(sha256, model, language, text, segments, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (transcript.sha256, model, transcript.language, transcript.text, json.dumps([s.model_dump() for s in transcript.segments]), transcript.created_at),
        )
        return transcript

    def forget(self, sha256: str) -> None:
        self._db.execute("DELETE FROM transcripts WHERE sha256 = ?", (sha256,))

    # -- as a job ------------------------------------------------------------------------

    def submit(self, path: Path, language: str | None = None, model: str | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        assert self._jobs is not None
        request = {"path": str(path), "language": language, "model": model}
        return self._jobs.run_now("transcribe", request, origin) if wait else self._jobs.submit("transcribe", request, origin)

    def _transcribe_job(self, request: dict[str, Any]) -> JobSpec:
        path = Path(request["path"])

        def run(ctx: JobContext) -> dict[str, Any]:
            ctx.message("Transcribing")
            return self.transcribe(path, request.get("language"), request.get("model"), ctx).model_dump(mode="json")

        return JobSpec(title=f"Transcribe {path.name}", run=run, resources=frozenset({"cpu"}))

    @staticmethod
    def _row(r: Any) -> Transcript:
        return Transcript(
            sha256=r["sha256"],
            model=r["model"],
            language=r["language"],
            text=r["text"],
            segments=[TranscriptSegment(**s) for s in json.loads(r["segments"] or "[]")],
            created_at=r["created_at"],
        )
