"""Datasets: folders of clips with their transcripts.

A dataset is ``<datasets>/<slug>/`` with ``dataset.json``, ``manifest.jsonl`` (one row per clip)
and ``audio/<row id>.flac`` (mono, at the clip's own rate, at most 48 kHz). The folders are the truth,
as with voices. Every trainer reads the same rows; ``dataset_prepare`` cuts long clips at their
silences, transcribes the clips without text and levels their loudness.
"""

from __future__ import annotations

import csv
import hashlib
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.audio.io import decode, encode
from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.datasets.models import Dataset, DatasetPage, DatasetReport, DatasetRow, DatasetStats, DurationBucket, ImportLayout, PrepareRequest, RowIssue, RowPatch
from mizuoto.core.engines import languages
from mizuoto.core.errors import NotFoundError, ValidationError
from mizuoto.core.files import trash
from mizuoto.core.jobs.models import Job, JobStep
from mizuoto.core.jobs.service import JobContext, JobService, JobSpec
from mizuoto.core.settings import SettingsService

if TYPE_CHECKING:
    import numpy as np

    from mizuoto.core.asr.service import AsrService
    from mizuoto.core.audio.service import AudioService
    from mizuoto.core.outputs.service import OutputService
    from mizuoto.core.voices.service import VoiceService

DATASET_FILE = "dataset.json"
MANIFEST = "manifest.jsonl"
AUDIO = "audio"
MAX_RATE = 48000
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
TEXT_SUFFIXES = (".lab", ".txt", ".normalized.txt")
TOO_SHORT = 1.0
TOO_LONG = 20.0


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w-]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-_")
    return slug[:48] or "dataset"


class DatasetService:
    def __init__(self, settings: SettingsService, audio: AudioService, outputs: OutputService, jobs: JobService | None = None, asr: AsrService | None = None) -> None:
        self._settings = settings
        self._audio = audio
        self._outputs = outputs
        self._jobs = jobs
        self._asr = asr
        self.voices: VoiceService | None = None
        """Set by ``build_services``."""
        self.events: Any = None
        """Set by ``build_services``: ``datasets_changed`` after every change."""
        self._lock = threading.RLock()
        if jobs is not None:
            jobs.register("dataset_prepare", self._prepare_job)

    def _changed(self, *ids: str) -> None:
        if self.events is not None:
            from mizuoto.core.events.models import DatasetsChangedEvent

            self.events.publish(DatasetsChangedEvent(ids=list(ids)))

    # -- folders --------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._settings.datasets_dir

    def folder(self, dataset: Dataset) -> Path:
        return self.root / dataset.slug

    def _read(self, folder: Path) -> Dataset | None:
        try:
            return Dataset.model_validate_json((folder / DATASET_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _rows(self, dataset: Dataset) -> list[DatasetRow]:
        path = self.folder(dataset) / MANIFEST
        if not path.is_file():
            return []
        return [DatasetRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _save(self, dataset: Dataset, rows: list[DatasetRow] | None = None) -> Dataset:
        """Write the rows (when given) and ``dataset.json`` with fresh statistics and fingerprint."""
        folder = self.folder(dataset)
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if rows is not None:
                text = "".join(r.model_dump_json() + "\n" for r in rows)
                tmp = folder / (MANIFEST + ".tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(folder / MANIFEST)
            current = self._rows(dataset)
            manifest = (folder / MANIFEST).read_bytes() if (folder / MANIFEST).is_file() else b""
            dataset.fingerprint = hashlib.sha256(manifest).hexdigest()[:20]
            dataset.stats = DatasetStats(
                rows=len(current),
                seconds=round(sum(r.duration for r in current if not r.excluded), 2),
                excluded=sum(1 for r in current if r.excluded),
                missing_text=sum(1 for r in current if not r.excluded and not r.text.strip()),
                speakers=sorted({r.speaker for r in current if r.speaker}),
            )
            dataset.updated_at = now_iso()
            tmp = folder / (DATASET_FILE + ".tmp")
            tmp.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(folder / DATASET_FILE)
        self._changed(dataset.id)
        return dataset

    # -- datasets ---------------------------------------------------------------------------

    def list_datasets(self) -> list[Dataset]:
        if not self.root.is_dir():
            return []
        found = [self._read(f) for f in sorted(self.root.iterdir()) if f.is_dir()]
        return sorted((d for d in found if d is not None), key=lambda d: d.updated_at, reverse=True)

    def get(self, key: str) -> Dataset:
        """By id or slug."""
        for d in self.list_datasets():
            if key in (d.id, d.slug):
                return d
        raise NotFoundError(f"No dataset {key!r}", {"kind": "dataset", "id": key, "hint": "mizuoto dataset list"})

    def create(self, name: str, language: str | None = None, consent: str = "", description: str = "") -> Dataset:
        if not name.strip():
            raise ValidationError("A dataset needs a name")
        slug = base = slugify(name)
        n = 2
        while (self.root / slug).exists():
            slug = f"{base}-{n}"
            n += 1
        stamp = now_iso()
        dataset = Dataset(
            id=new_id("d"),
            slug=slug,
            name=name.strip(),
            language=languages.normalize(language),
            consent=consent.strip(),
            description=description,
            created_at=stamp,
            updated_at=stamp,
        )
        (self.root / slug / AUDIO).mkdir(parents=True, exist_ok=True)
        return self._save(dataset, [])

    def update(self, key: str, name: str | None = None, language: str | None = None, consent: str | None = None, description: str | None = None) -> Dataset:
        dataset = self.get(key)
        if name is not None:
            if not name.strip():
                raise ValidationError("A dataset needs a name")
            dataset.name = name.strip()
        if language is not None:
            dataset.language = languages.normalize(language)
        if consent is not None:
            dataset.consent = consent.strip()
        if description is not None:
            dataset.description = description
        return self._save(dataset)

    def delete(self, key: str) -> None:
        dataset = self.get(key)
        trash(self.folder(dataset), self._settings.data_dir / "trash")
        self._changed(dataset.id)

    # -- rows -------------------------------------------------------------------------------

    def rows(self, key: str, offset: int = 0, limit: int = 100, q: str | None = None, issue: RowIssue | None = None) -> DatasetPage:
        """A page of rows: those whose text, source or speaker has ``q``, or that ``check`` lists under ``issue``."""
        rows = self._rows(self.get(key))
        if issue:
            flagged = set(getattr(self.check(key), issue))
            rows = [r for r in rows if r.id in flagged]
        if q:
            needle = q.lower()
            rows = [r for r in rows if needle in r.text.lower() or needle in r.source.lower() or needle in r.speaker.lower()]
        return DatasetPage(items=rows[offset : offset + max(1, limit)], total=len(rows))

    def audio_path(self, key: str, row_id: str) -> Path:
        dataset = self.get(key)
        row = next((r for r in self._rows(dataset) if r.id == row_id), None)
        if row is None:
            raise NotFoundError(f"No row {row_id} in {dataset.name}", {"row": row_id})
        return self.folder(dataset) / row.audio

    def update_row(self, key: str, row_id: str, patch: RowPatch) -> DatasetRow:
        dataset = self.get(key)
        rows = self._rows(dataset)
        for i, r in enumerate(rows):
            if r.id == row_id:
                changes = {k: v for k, v in patch.model_dump().items() if v is not None}
                if "language" in changes:
                    changes["language"] = languages.normalize(changes["language"])
                rows[i] = r.model_copy(update=changes)
                self._save(dataset, rows)
                return rows[i]
        raise NotFoundError(f"No row {row_id} in {dataset.name}", {"row": row_id})

    def delete_rows(self, key: str, row_ids: list[str]) -> Dataset:
        dataset = self.get(key)
        rows = self._rows(dataset)
        gone = [r for r in rows if r.id in set(row_ids)]
        for r in gone:
            (self.folder(dataset) / r.audio).unlink(missing_ok=True)
        return self._save(dataset, [r for r in rows if r.id not in set(row_ids)])

    # -- adding clips -------------------------------------------------------------------------

    def _store(self, dataset: Dataset, samples: np.ndarray, rate: int, **fields: Any) -> DatasetRow:
        from mizuoto.core.audio.io import resample

        if rate > MAX_RATE:
            samples, rate = resample(samples, rate, MAX_RATE), MAX_RATE
        row_id = new_id("r")
        rel = f"{AUDIO}/{row_id}.flac"
        encode(samples, rate, self.folder(dataset) / rel, "flac")
        return DatasetRow(id=row_id, audio=rel, duration=round(len(samples) / rate, 3), sample_rate=rate, **fields)

    @staticmethod
    def _sidecar_text(path: Path) -> str:
        for suffix in TEXT_SUFFIXES:
            side = path.with_suffix(suffix)
            if side.is_file():
                return side.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def _add(self, dataset: Dataset, items: list[dict[str, Any]], warnings: list[str]) -> Dataset:
        """``items``: ``{path, text?, speaker?, language?, source?, licence_tags?}``; unreadable files become warnings."""
        rows = self._rows(dataset)
        for item in items:
            path = Path(item["path"])
            try:
                samples, rate = decode(path)
            except ValidationError as e:
                warnings.append(e.message)
                continue
            if not len(samples):
                warnings.append(f"{path.name} is empty")
                continue
            rows.append(
                self._store(
                    dataset,
                    samples,
                    rate,
                    text=str(item.get("text") or "").strip(),
                    speaker=str(item.get("speaker") or ""),
                    language=languages.normalize(item.get("language")) or dataset.language,
                    source=str(item.get("source") or path.name),
                    licence_tags=list(item.get("licence_tags") or []),
                )
            )
        return self._save(dataset, rows)

    def add_files(self, key: str, paths: list[Path], speaker: str = "", language: str | None = None) -> tuple[Dataset, list[str]]:
        """Clips from files (a transcript beside one, ``<name>.lab`` or ``.txt``, is taken as its text)."""
        dataset = self.get(key)
        warnings: list[str] = []
        files: list[Path] = []
        for p in paths:
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES)
            elif p.is_file():
                files.append(p)
            else:
                warnings.append(f"No file {p}")
        items = [{"path": f, "text": self._sidecar_text(f), "speaker": speaker, "language": language} for f in files]
        return self._add(dataset, items, warnings), warnings

    def add_audio(self, key: str, audio_ids: list[str], texts: list[str | None] | None = None, speaker: str = "") -> tuple[Dataset, list[str]]:
        """Clips uploaded through ``PUT /audio/files``."""
        dataset = self.get(key)
        items = []
        for i, audio_id in enumerate(audio_ids):
            audio = self._audio.get(audio_id)
            text = (texts[i] if texts and i < len(texts) else None) or ""
            items.append({"path": self._audio.path_of(audio_id), "text": text, "speaker": speaker, "source": audio.name})
        warnings: list[str] = []
        return self._add(dataset, items, warnings), warnings

    def add_voice(self, key: str, voice_key: str) -> Dataset:
        """A library voice's reference clips, with their transcripts."""
        if self.voices is None:
            raise ValidationError("The voice library is not available")
        dataset = self.get(key)
        voice = self.voices.get(voice_key)
        items = [
            {"path": self.voices.folder(voice) / r.file, "text": r.transcript or "", "speaker": voice.slug, "language": r.language or voice.language, "source": f"voice:{voice.id}"}
            for r in voice.references
        ]
        if not items:
            raise ValidationError(f"{voice.name} has no reference clips")
        return self._add(dataset, items, [])

    def add_outputs(self, key: str, output_ids: list[str], speaker: str = "") -> Dataset:
        """Synthesized results, with their text and their licences' tags (some keep them from training other engines)."""
        dataset = self.get(key)
        items = []
        for output_id in output_ids:
            out = self._outputs.get(output_id)
            items.append({"path": out.path, "text": out.text or "", "speaker": speaker, "language": out.language, "source": f"output:{out.id}", "licence_tags": out.licence_tags})
        return self._add(dataset, items, [])

    def import_folder(self, key: str, folder: Path, layout: ImportLayout = "auto") -> tuple[Dataset, list[str]]:
        """A folder in a known layout: a GPT-SoVITS ``.list`` (``path|speaker|language|text``), an LJSpeech
        ``metadata.csv`` (``id|text|normalised``), a CSV (``audio_file|text``, as F5-TTS), or plain files."""
        if not folder.is_dir():
            raise NotFoundError(f"No folder {folder}", {"path": str(folder)})
        if layout == "auto":
            if any(folder.glob("*.list")):
                layout = "gptsovits_list"
            elif (folder / "metadata.csv").is_file() and (folder / "wavs").is_dir():
                layout = "ljspeech"
            elif any(folder.glob("*.csv")):
                layout = "csv"
            else:
                layout = "plain"
        if layout == "plain":
            return self.add_files(key, [folder])
        dataset = self.get(key)
        warnings: list[str] = []
        items: list[dict[str, Any]] = []
        if layout == "gptsovits_list":
            for list_file in sorted(folder.glob("*.list")):
                for line in list_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split("|")
                    if len(parts) < 4:
                        continue
                    path = Path(parts[0])
                    path = path if path.is_absolute() and path.exists() else folder / path.name
                    lang = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue", "all_zh": "zh", "all_ja": "ja", "all_ko": "ko", "all_yue": "yue"}.get(
                        parts[2].strip().lower(), parts[2]
                    )
                    items.append({"path": path, "speaker": parts[1], "language": lang, "text": "|".join(parts[3:])})
        elif layout == "ljspeech":
            for line in (folder / "metadata.csv").read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("|")
                if len(parts) >= 2:
                    items.append({"path": folder / "wavs" / f"{parts[0]}.wav", "text": parts[-1] or parts[1]})
        else:
            for csv_file in sorted(folder.glob("*.csv")):
                with open(csv_file, encoding="utf-8", errors="replace", newline="") as f:
                    sample = f.read(2048)
                    f.seek(0)
                    delimiter = "|" if sample.count("|") > sample.count(",") else ","
                    reader = csv.reader(f, delimiter=delimiter)
                    for parts in reader:
                        if len(parts) < 2 or parts[0].strip().lower() in ("audio_file", "audio", "path", "file"):
                            continue
                        path = Path(parts[0])
                        items.append({"path": path if path.is_absolute() else folder / path, "text": parts[1]})
        for item in items:
            if not Path(item["path"]).is_file():
                warnings.append(f"Listed but missing: {item['path']}")
        return self._add(dataset, [i for i in items if Path(i["path"]).is_file()], warnings), warnings

    # -- preparation --------------------------------------------------------------------------

    def prepare(self, key: str, req: PrepareRequest | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        if self._jobs is None:
            raise ValidationError("Preparing needs the job service")
        dataset = self.get(key)
        req = req or PrepareRequest()
        if req.transcribe and (self._asr is None or not self._asr.available()) and any(not r.text.strip() and not r.excluded for r in self._rows(dataset)):
            raise ValidationError("Transcribing needs speech recognition", {"hint": "mizuoto model download asr/whisper-large-v3-turbo"})
        request = {"dataset": dataset.id, **req.model_dump(mode="json")}
        return self._jobs.run_now("dataset_prepare", request, origin) if wait else self._jobs.submit("dataset_prepare", request, origin)

    def _prepare_job(self, request: dict[str, Any]) -> JobSpec:
        dataset = self.get(str(request["dataset"]))
        req = PrepareRequest.model_validate({k: v for k, v in request.items() if k != "dataset"})
        steps = [JobStep(id=s, title=s) for s, on in (("slice", req.slice), ("transcribe", req.transcribe), ("normalize", req.normalize)) if on]
        return JobSpec(title=f"Prepare {dataset.name}", run=lambda ctx: self._run_prepare(dataset.id, req, ctx), resources=frozenset({"cpu"}), steps=steps)

    def _run_prepare(self, dataset_id: str, req: PrepareRequest, ctx: JobContext) -> dict[str, Any]:
        from mizuoto.core.audio import dsp
        from mizuoto.core.audio.slicer import DEFAULTS, Slicer

        dataset = self.get(dataset_id)
        folder = self.folder(dataset)
        rows = self._rows(dataset)
        result: dict[str, Any] = {}
        if req.slice:
            ctx.step("slice", "running", 0.0)
            ctx.message("Cutting long clips at their silences")
            out: list[DatasetRow] = []
            cut = 0
            for n, row in enumerate(rows):
                ctx.check_cancel()
                if row.excluded or row.duration <= req.max_seconds:
                    out.append(row)
                    continue
                samples, rate = decode(folder / row.audio)
                slicer = Slicer(
                    rate,
                    threshold=req.threshold_db,
                    min_length=int(min(req.max_seconds, 4.0) * 1000),
                    min_interval=int(DEFAULTS["min_interval"]),
                    hop_size=int(DEFAULTS["hop_size"]),
                    max_sil_kept=int(DEFAULTS["max_sil_kept"]),
                )
                pieces = slicer.slice(samples)
                pieces = [p[i : i + int(req.max_seconds * rate)] for p in pieces for i in range(0, len(p), int(req.max_seconds * rate))]
                for piece in pieces:
                    if len(piece) < req.min_seconds * rate:
                        continue
                    out.append(self._store(dataset, piece, rate, text="", speaker=row.speaker, language=row.language, source=row.source, licence_tags=row.licence_tags))
                (folder / row.audio).unlink(missing_ok=True)
                cut += 1
                ctx.step("slice", "running", (n + 1) / len(rows))
            for row in out:
                if row.duration < req.min_seconds and not row.excluded:
                    row.excluded = True
            rows = out
            self._save(dataset, rows)
            ctx.step("slice", "done")
            result["cut"] = cut
        if req.transcribe:
            ctx.step("transcribe", "running", 0.0)
            todo = [r for r in rows if not r.excluded and not r.text.strip()]
            for n, row in enumerate(todo):
                ctx.check_cancel()
                ctx.message(f"Transcribing {n + 1} of {len(todo)}")
                assert self._asr is not None
                transcript = self._asr.transcribe(folder / row.audio, req.language or row.language or dataset.language, ctx=ctx)
                row.text = transcript.text
                row.language = row.language or transcript.language
                ctx.step("transcribe", "running", (n + 1) / max(1, len(todo)))
            self._save(dataset, rows)
            ctx.step("transcribe", "done")
            result["transcribed"] = len(todo)
        if req.normalize:
            ctx.step("normalize", "running", 0.0)
            ctx.message("Levelling loudness")
            for n, row in enumerate(rows):
                ctx.check_cancel()
                if row.excluded:
                    continue
                samples, rate = decode(folder / row.audio)
                encode(dsp.normalize(samples, "loudness"), rate, folder / row.audio, "flac")
                ctx.step("normalize", "running", (n + 1) / len(rows))
            self._save(dataset, rows)
            ctx.step("normalize", "done")
        dataset = self.get(dataset_id)
        return {**result, "rows": dataset.stats.rows, "seconds": dataset.stats.seconds}

    # -- checks -------------------------------------------------------------------------------

    def check(self, key: str) -> DatasetReport:
        import numpy as np

        dataset = self.get(key)
        rows = self._rows(dataset)
        active = [r for r in rows if not r.excluded]
        report = DatasetReport(dataset=dataset.id, rows=len(rows), usable=0, seconds=round(sum(r.duration for r in active), 2), histogram=[])
        edges = [0, 1, 2, 3, 5, 8, 12, 20, 30]
        counts = [0] * len(edges)
        for r in active:
            counts[max(i for i, e in enumerate(edges) if r.duration >= e)] += 1
            if r.duration < TOO_SHORT:
                report.too_short.append(r.id)
            if r.duration > TOO_LONG:
                report.too_long.append(r.id)
            if not r.text.strip():
                report.empty_text.append(r.id)
            report.sample_rates[str(r.sample_rate)] = report.sample_rates.get(str(r.sample_rate), 0) + 1
            report.speakers[r.speaker or "-"] = report.speakers.get(r.speaker or "-", 0) + 1
            report.languages[r.language or "-"] = report.languages.get(r.language or "-", 0) + 1
            for tag in r.licence_tags:
                if tag not in report.licence_tags:
                    report.licence_tags.append(tag)
            try:
                samples, _ = decode(self.folder(dataset) / r.audio)
                if len(samples) and float(np.mean(np.abs(samples) > 0.999)) > 0.001:
                    report.clipping.append(r.id)
            except ValidationError:
                report.warnings.append(f"{r.audio} cannot be read")
        report.histogram = [DurationBucket(seconds=e, count=c) for e, c in zip(edges, counts)]
        report.usable = sum(1 for r in active if r.text.strip() and TOO_SHORT <= r.duration <= TOO_LONG)
        if report.seconds < 60:
            report.warnings.append("Under a minute of audio: a fine-tune will learn little")
        if report.empty_text:
            report.warnings.append(f"{len(report.empty_text)} clip(s) have no text: prepare the dataset with transcription")
        if len(report.sample_rates) > 1:
            report.warnings.append("The clips have several sample rates; trainers resample them")
        return report

    def training_rows(self, key: str, target_engine: str, registry: Any) -> list[DatasetRow]:
        """The rows a trainer of ``target_engine`` gets: not excluded, with text; refused when a licence forbids it."""
        from mizuoto.core.engines.licences import training_refusal

        dataset = self.get(key)
        rows = [r for r in self._rows(dataset) if not r.excluded and r.text.strip()]
        tags = sorted({t for r in rows for t in r.licence_tags})
        reason = training_refusal(registry, tags, target_engine)
        if reason:
            raise ValidationError(f"{dataset.name} holds clips {reason}", {"reason": "licence", "licence_tags": tags})
        if not rows:
            raise ValidationError(f"{dataset.name} has no clip with text to train on", {"hint": f"mizuoto dataset prepare {dataset.slug}"})
        return rows
