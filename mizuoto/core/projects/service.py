"""Long-form projects: books, scripts and dialogues read by a cast of voices across engines.

A project is ``<projects>/<slug>/project.json``: the cast (speakers: a model, a voice or preset
speaker, a style and parameters each), the chapters and their lines (text, speaker, overrides) and,
per line, its takes. A take is an output of kind ``line`` made by an ordinary ``speak`` job, with the
fingerprint of the request it came from: a line whose current request differs (text, speaker, voice,
model, parameters) is ``stale``. ``project_render`` renders the lines that need it, a few speak jobs
at a time so that micro-batching can join look-alike lines; ``project_export`` joins the takes in
use with the project's pauses into one output of kind ``project``, with a cue per line (subtitles).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.errors import MizuotoError, NotFoundError, ValidationError
from mizuoto.core.files import slugify, trash
from mizuoto.core.jobs.models import Job, JobStep
from mizuoto.core.jobs.service import JobContext, JobService, JobSpec
from mizuoto.core.outputs.models import Output
from mizuoto.core.projects.models import (
    Chapter,
    ExportRequest,
    Line,
    Project,
    ProjectInput,
    ProjectStats,
    ProjectSummary,
    RenderRequest,
    ScriptImport,
    Speaker,
    Take,
)
from mizuoto.core.settings import SettingsService
from mizuoto.core.tts.models import CommonParams, OutputSpec, SpeakRequest, StyleInput, VoiceInput

if TYPE_CHECKING:
    from mizuoto.core.outputs.service import OutputService
    from mizuoto.core.tts.service import TtsService
    from mizuoto.core.voices.service import VoiceService

PROJECT_FILE = "project.json"
MAX_TAKES = 5
"""Takes kept per line; older ones (not in use) are deleted."""
WINDOW = 8
"""Speak jobs a render keeps queued at once: enough for micro-batching, few enough for the job list."""
TAKE_FORMAT = "flac"
NAME = re.compile(r"^[^\W\d_][\w.'’-]*(?: [\w.'’-]+){0,2}$")
"""At most three words of letters: "Dr. Watson", "林黛玉"; not "Note that" followed by a sentence."""
SPEAKER_PREFIX = re.compile(r"^\s*(?:\[([^\[\]]{1,32})\]|([^:：\s][^:：]{0,31}?))\s*[:：]\s*(.+)$")


class ProjectService:
    def __init__(self, settings: SettingsService, tts: TtsService, outputs: OutputService, jobs: JobService) -> None:
        self._settings = settings
        self._tts = tts
        self._outputs = outputs
        self._jobs = jobs
        self.voices: VoiceService | None = None
        """Set by ``build_services``: a library voice's revision is part of a line's fingerprint."""
        self.events: Any = None
        """Set by ``build_services``: ``projects_changed`` after every change."""
        self._lock = threading.RLock()
        self._rendering: dict[str, set[str]] = {}
        """Project id → the lines a running render has queued."""
        jobs.register("project_render", self._render_job)
        jobs.register("project_export", self._export_job)

    def _changed(self, *ids: str) -> None:
        if self.events is not None:
            from mizuoto.core.events.models import ProjectsChangedEvent

            self.events.publish(ProjectsChangedEvent(ids=list(ids)))

    # -- folders ------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._settings.projects_dir

    def _read(self, folder: Path) -> Project | None:
        try:
            project = Project.model_validate_json((folder / PROJECT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        project.slug = folder.name
        return project

    def _write(self, project: Project) -> None:
        folder = self.root / project.slug
        folder.mkdir(parents=True, exist_ok=True)
        project.updated_at = now_iso()
        stored = project.model_copy(deep=True)
        for chapter in stored.chapters:
            for line in chapter.lines:
                line.state = "missing"
        tmp = folder / (PROJECT_FILE + ".tmp")
        tmp.write_text(stored.model_dump_json(indent=2, exclude={"stats"}), encoding="utf-8")
        tmp.replace(folder / PROJECT_FILE)

    def _load(self, key: str) -> Project:
        folder = self.root / key
        if (folder / PROJECT_FILE).is_file():
            project = self._read(folder)
            if project is not None:
                return project
        if self.root.is_dir():
            for f in sorted(self.root.iterdir()):
                project = self._read(f) if f.is_dir() else None
                if project is not None and project.id == key:
                    return project
        raise NotFoundError(f"No project {key!r}", {"kind": "project", "id": key, "hint": "mizuoto project list"})

    # -- reading ------------------------------------------------------------------------------

    def list_projects(self) -> list[ProjectSummary]:
        if not self.root.is_dir():
            return []
        out: list[ProjectSummary] = []
        for folder in sorted(self.root.iterdir()):
            project = self._read(folder) if folder.is_dir() else None
            if project is None:
                continue
            self._annotate(project)
            out.append(
                ProjectSummary(
                    id=project.id,
                    slug=project.slug,
                    name=project.name,
                    description=project.description,
                    model=project.model,
                    speakers=len(project.speakers),
                    stats=project.stats,
                    updated_at=project.updated_at,
                )
            )
        return sorted(out, key=lambda p: p.updated_at, reverse=True)

    def get(self, key: str) -> Project:
        """By id or slug, with each line's state and the project's statistics."""
        return self._annotate(self._load(key))

    def _annotate(self, project: Project) -> Project:
        rendering = self._rendering.get(project.id, set())
        revisions: dict[str, str] = {}
        stats = ProjectStats(chapters=len(project.chapters))
        for chapter in project.chapters:
            for line in chapter.lines:
                stats.lines += 1
                take = self._take(line)
                if line.id in rendering:
                    line.state = "rendering"
                elif not line.text.strip():
                    line.state = "empty"
                elif line.error:
                    line.state = "failed"
                elif take is None:
                    line.state = "missing"
                else:
                    try:
                        current = self._fingerprint(self._request(project, line), revisions)
                    except MizuotoError:
                        current = None
                    line.state = "rendered" if current == take.fingerprint else "stale"
                if take is not None:
                    stats.seconds += take.duration
                if line.state in ("rendered", "stale", "missing", "failed"):
                    setattr(stats, line.state, getattr(stats, line.state) + 1)
        stats.seconds = round(stats.seconds, 2)
        project.stats = stats
        return project

    @staticmethod
    def _take(line: Line) -> Take | None:
        return next((t for t in line.takes if t.output_id == line.take), None)

    # -- editing ------------------------------------------------------------------------------

    def create(self, data: ProjectInput) -> Project:
        if not data.name.strip():
            raise ValidationError("A project needs a name")
        base = slugify(data.name, "project")[:48]
        slug, n = base, 2
        while (self.root / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        stamp = now_iso()
        project = Project(id=new_id("p"), slug=slug, name=data.name.strip(), created_at=stamp, updated_at=stamp)
        self._apply(project, data)
        with self._lock:
            self._write(project)
        self._changed(project.id)
        return self.get(project.id)

    def update(self, key: str, data: ProjectInput) -> Project:
        """Replace the editable part: settings, cast and script. Lines keep their takes by id."""
        with self._lock:
            project = self._load(key)
            self._apply(project, data)
            self._write(project)
        self._changed(project.id)
        return self.get(project.id)

    def _apply(self, project: Project, data: ProjectInput) -> None:
        if not data.name.strip():
            raise ValidationError("A project needs a name")
        speakers = self._check_speakers(data.speakers)
        known = {s.id for s in speakers}
        old_lines = {line.id: line for chapter in project.chapters for line in chapter.lines}
        seen_chapters: set[str] = set()
        seen_lines: set[str] = set()
        chapters: list[Chapter] = []
        for c in data.chapters:
            cid = c.id or new_id("c")
            if cid in seen_chapters:
                raise ValidationError(f"Two chapters have the id {cid}")
            seen_chapters.add(cid)
            lines: list[Line] = []
            for line in c.lines:
                lid = line.id or new_id("l")
                if lid in seen_lines:
                    raise ValidationError(f"Two lines have the id {lid}")
                seen_lines.add(lid)
                if line.speaker is not None and line.speaker not in known:
                    raise ValidationError(f"A line names the speaker {line.speaker!r}, who is not in the cast", {"line": lid, "speakers": sorted(known)})
                old = old_lines.get(lid)
                takes = old.takes if old else []
                take = line.take if line.take in {t.output_id for t in takes} else (old.take if old else None)
                lines.append(
                    Line(
                        id=lid,
                        text=line.text,
                        speaker=line.speaker,
                        instruct=line.instruct or None,
                        params=line.params,
                        pause_ms=line.pause_ms,
                        ends_paragraph=line.ends_paragraph,
                        takes=takes,
                        take=take,
                        error=old.error if old and old.text == line.text else None,
                    )
                )
            chapters.append(Chapter(id=cid, title=c.title, lines=lines))
        dropped = [t.output_id for lid, line in old_lines.items() if lid not in seen_lines for t in line.takes]
        project.name = data.name.strip()
        project.description = data.description
        project.model = data.model or None
        project.language = data.language or None
        project.speakers = speakers
        project.chapters = chapters
        for field in ("line_pause_ms", "paragraph_pause_ms", "chapter_pause_ms", "format", "sample_rate", "normalize"):
            setattr(project, field, getattr(data, field))
        if dropped:
            self._outputs.delete(dropped)

    @staticmethod
    def _check_speakers(speakers: list[Speaker]) -> list[Speaker]:
        out: list[Speaker] = []
        seen: set[str] = set()
        for s in speakers:
            sid = s.id.strip() or slugify(s.name, "speaker")
            if sid in seen:
                raise ValidationError(f"Two speakers have the id {sid}")
            if not s.name.strip():
                raise ValidationError("A speaker needs a name")
            if s.voice and s.preset:
                raise ValidationError(f"{s.name} has both a voice and a preset speaker: choose one")
            seen.add(sid)
            out.append(s.model_copy(update={"id": sid, "name": s.name.strip(), "voice": s.voice or None, "preset": s.preset or None, "model": s.model or None}))
        return out

    def editable(self, project: Project) -> ProjectInput:
        """The project as the editor sends it back."""
        return ProjectInput.model_validate(project.model_dump(include=set(ProjectInput.model_fields)))

    def delete(self, key: str) -> None:
        """Move the project to the trash and delete its takes."""
        with self._lock:
            project = self._load(key)
            takes = [t.output_id for c in project.chapters for line in c.lines for t in line.takes]
            trash(self.root / project.slug, self._settings.data_dir / "trash")
        self._outputs.delete(takes)
        self._changed(project.id)

    def use_take(self, key: str, line_id: str, output_id: str) -> Project:
        with self._lock:
            project = self._load(key)
            line = self._line(project, line_id)
            if output_id not in {t.output_id for t in line.takes}:
                raise NotFoundError(f"Line {line_id} has no take {output_id}", {"kind": "take", "id": output_id})
            line.take = output_id
            self._write(project)
        self._changed(project.id)
        return self.get(project.id)

    @staticmethod
    def _line(project: Project, line_id: str) -> Line:
        for chapter in project.chapters:
            for line in chapter.lines:
                if line.id == line_id:
                    return line
        raise NotFoundError(f"No line {line_id} in {project.name}", {"kind": "line", "id": line_id})

    # -- scripts ------------------------------------------------------------------------------

    def import_script(self, key: str, req: ScriptImport) -> Project:
        """Add chapters and lines from plain text (see ``ScriptImport``); new names join the cast."""
        from mizuoto.core.text.segment import sentences

        with self._lock:
            project = self._load(key)
            names = {s.name.lower(): s.id for s in project.speakers} | {s.id.lower(): s.id for s in project.speakers}
            raw = req.text.replace("\r\n", "\n").split("\n")
            prefixes: dict[str, int] = {}
            for text in raw:
                m = SPEAKER_PREFIX.match(text)
                if m and not text.lstrip().startswith("#"):
                    name = (m.group(1) or m.group(2)).strip()
                    prefixes[name.lower()] = prefixes.get(name.lower(), 0) + 1
            # A script with a recurring speaker is a dialogue: then any name-like prefix is a speaker too.
            dialogue = any(n >= 2 or name in names for name, n in prefixes.items())
            target = next((c for c in project.chapters if c.id == req.chapter), None) if req.chapter else None
            if req.chapter and target is None:
                raise NotFoundError(f"No chapter {req.chapter} in {project.name}", {"kind": "chapter", "id": req.chapter})
            chapters: list[Chapter] = []
            current = target
            for text in raw:
                stripped = text.strip()
                if not stripped:
                    if current is not None and current.lines:
                        current.lines[-1].ends_paragraph = True
                    continue
                if stripped.startswith("#"):
                    if target is None:  # into one chapter, headings are left out
                        current = Chapter(id=new_id("c"), title=stripped.lstrip("#").strip())
                        chapters.append(current)
                    continue
                speaker: str | None = None
                m = SPEAKER_PREFIX.match(stripped)
                if m:
                    name = (m.group(1) or m.group(2)).strip()
                    if name.lower() in names or prefixes.get(name.lower(), 0) >= 2 or (dialogue and NAME.match(name)):
                        if name.lower() not in names:
                            sid = slugify(name, "speaker")
                            while sid in {s.id for s in project.speakers}:
                                sid += "-2"
                            if not project.speakers:
                                project.speakers.append(Speaker(id="narrator", name="Narrator"))
                            project.speakers.append(Speaker(id=sid, name=name))
                            names[name.lower()] = sid
                        speaker = names[name.lower()]
                        stripped = m.group(3).strip()
                if current is None:
                    current = Chapter(id=new_id("c"), title=project.name if not project.chapters and not chapters else "")
                    chapters.append(current)
                pieces = sentences(stripped) if req.split == "sentence" else [stripped]
                for piece in pieces:
                    if piece.strip():
                        current.lines.append(Line(id=new_id("l"), text=piece.strip(), speaker=speaker))
            if current is not None and current.lines:
                current.lines[-1].ends_paragraph = True
            if req.replace and target is None:
                dropped = [t.output_id for c in project.chapters for line in c.lines for t in line.takes]
                project.chapters = chapters
                self._outputs.delete(dropped)
            elif target is None:
                project.chapters.extend(chapters)
            self._write(project)
        self._changed(project.id)
        return self.get(project.id)

    def script(self, key: str) -> str:
        """The project as a plain-text script ``import_script`` reads back."""
        project = self._load(key)
        names = {s.id: s.name for s in project.speakers}
        narrator = project.speakers[0].id if project.speakers else None
        out: list[str] = []
        for chapter in project.chapters:
            if chapter.title:
                out += [f"# {chapter.title}", ""]
            for line in chapter.lines:
                who = line.speaker if line.speaker and line.speaker != narrator else None
                out.append(f"{names.get(who, who)}: {line.text}" if who else line.text)
                if line.ends_paragraph:
                    out.append("")
            if out and out[-1]:
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    # -- requests -----------------------------------------------------------------------------

    def speaker_of(self, project: Project, line: Line) -> Speaker | None:
        if line.speaker:
            return next((s for s in project.speakers if s.id == line.speaker), None)
        return project.speakers[0] if project.speakers else None

    def _request(self, project: Project, line: Line) -> SpeakRequest:
        """The synthesis request of a line: its speaker's model, voice and settings, then its own."""
        speaker = self.speaker_of(project, line)
        model = (speaker.model if speaker else None) or project.model
        if not model:
            raise ValidationError(f"{speaker.name if speaker else 'The narrator'} has no model, and the project names none", {"line": line.id})
        params = (speaker.params if speaker else CommonParams()).model_dump(exclude_none=True)
        params.update(line.params.model_dump(exclude_none=True))
        instruct = line.instruct or (speaker.instruct if speaker else None)
        return SpeakRequest(
            model=model,
            text=line.text.strip(),
            language=(speaker.language if speaker else None) or project.language,
            voice=speaker.voice if speaker else None,
            voice_spec=VoiceInput(preset=speaker.preset) if speaker and speaker.preset else None,
            style=StyleInput(instruct=instruct) if instruct else None,
            params=CommonParams.model_validate(params),
            engine_options=dict(speaker.engine_options) if speaker else {},
            output=OutputSpec(format=TAKE_FORMAT, normalize=project.normalize, project=project.id),
        )

    def request_for(self, key: str, line_id: str) -> SpeakRequest:
        project = self._load(key)
        return self._request(project, self._line(project, line_id))

    def _fingerprint(self, req: SpeakRequest, revisions: dict[str, str]) -> str:
        data = req.model_dump(mode="json", exclude={"output": {"name", "project"}})
        if req.voice:
            if req.voice not in revisions:
                try:
                    revisions[req.voice] = self.voices.get(req.voice).updated_at if self.voices else ""
                except NotFoundError:
                    revisions[req.voice] = "missing"
            data["voice_revision"] = revisions[req.voice]
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    # -- rendering ----------------------------------------------------------------------------

    def render(self, key: str, req: RenderRequest | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        """Render the lines that have no current take (``force``: all of them) as speak jobs."""
        project = self.get(key)
        request = {"project": project.id, **(req or RenderRequest()).model_dump(mode="json")}
        if not self._todo(project, RenderRequest.model_validate(request)):
            raise ValidationError(f"Every line of {project.name} is rendered", {"hint": "force renders them again"})
        return self._jobs.run_now("project_render", request, origin) if wait else self._jobs.submit("project_render", request, origin)

    @staticmethod
    def _todo(project: Project, req: RenderRequest) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for chapter in project.chapters:
            if req.chapters is not None and chapter.id not in req.chapters:
                continue
            for line in chapter.lines:
                if req.lines is not None and line.id not in req.lines:
                    continue
                if line.state == "empty" or (line.state in ("rendered", "rendering") and not req.force):
                    continue
                out.append((chapter.id, line.id))
        return out

    def _render_job(self, request: dict[str, Any]) -> JobSpec:
        project = self.get(str(request["project"]))
        req = RenderRequest.model_validate({k: v for k, v in request.items() if k != "project"})
        todo = self._todo(project, req)
        titles = {c.id: c.title or f"Chapter {i + 1}" for i, c in enumerate(project.chapters)}
        steps = [JobStep(id=cid, title=titles[cid]) for cid in dict.fromkeys(c for c, _ in todo)]
        return JobSpec(
            title=f"Render {project.name} ({len(todo)} line{'s' if len(todo) != 1 else ''})",
            run=lambda ctx: self._run_render(project.id, todo, ctx),
            resources=frozenset({f"project:{project.id}"}),
            steps=steps,
        )

    def _run_render(self, project_id: str, todo: list[tuple[str, str]], ctx: JobContext) -> dict[str, Any]:
        chapter_of = dict((lid, cid) for cid, lid in todo)
        left = {cid: sum(1 for c, _ in todo if c == cid) for cid in chapter_of.values()}
        queue = deque(lid for _, lid in todo)
        inflight: deque[tuple[str, str, str]] = deque()
        """(line id, speak job id, fingerprint)"""
        scheduled = self._jobs.scheduling
        window = WINDOW if scheduled else 1
        done = failed = 0
        revisions: dict[str, str] = {}
        with self._lock:
            self._rendering[project_id] = set(queue)
        self._changed(project_id)

        def finish(lid: str, job: Job | None, fingerprint: str, error: str | None) -> None:
            nonlocal done, failed
            output_id = str(job.result.get("output_id")) if job and job.state == "completed" and job.result else None
            if output_id is None:
                failed += 1
                if error is None and job is not None:
                    error = job.error.message if job.error else f"The speak job was {job.state}"
            self._record(project_id, lid, output_id, fingerprint, error)
            done += 1
            cid = chapter_of[lid]
            left[cid] -= 1
            ctx.step(cid, "done" if left[cid] == 0 else "running", None)
            ctx.progress(done / len(todo))

        try:
            while queue or inflight:
                if ctx.cancelled:
                    for _, job_id, _ in inflight:
                        try:
                            self._jobs.cancel(job_id)
                        except MizuotoError:
                            pass
                    ctx.check_cancel()
                while queue and len(inflight) < window:
                    lid = queue.popleft()
                    project = self._load(project_id)
                    try:
                        line = self._line(project, lid)
                        speak = self._request(project, line)
                        speak.output.name = f"{project.slug}-{lid}"
                        fingerprint = self._fingerprint(speak, revisions)
                        ctx.step(chapter_of[lid], "running", None)
                        ctx.message(line.text[:80])
                        job = self._tts.speak(speak, wait=not scheduled, origin="project")
                    except MizuotoError as e:
                        finish(lid, None, "", e.message)
                        continue
                    if scheduled:
                        inflight.append((lid, job.id, fingerprint))
                    else:
                        finish(lid, job, fingerprint, None)
                if inflight:
                    lid, job_id, fingerprint = inflight[0]
                    job = self._jobs.wait(job_id, timeout=0.5)
                    if job.state in ("completed", "failed", "cancelled", "interrupted"):
                        inflight.popleft()
                        if job.state == "cancelled" and ctx.cancelled:
                            continue
                        finish(lid, job, fingerprint, None)
        finally:
            with self._lock:
                self._rendering.pop(project_id, None)
            self._changed(project_id)
        return {"project": project_id, "lines": len(todo), "failed": failed}

    def _record(self, project_id: str, line_id: str, output_id: str | None, fingerprint: str, error: str | None) -> None:
        """Keep a new take (and use it), or the error, on a line that may have been edited meanwhile."""
        drop: list[str] = []
        with self._lock:
            project = self._load(project_id)
            try:
                line = self._line(project, line_id)
            except NotFoundError:
                if output_id:
                    self._outputs.delete([output_id])
                return
            if output_id:
                out = self._outputs.get(output_id)
                line.takes.append(Take(output_id=output_id, fingerprint=fingerprint, duration=round(out.duration, 3), created_at=now_iso()))
                line.take = output_id
                line.error = None
                while len(line.takes) > MAX_TAKES:
                    old = next(t for t in line.takes if t.output_id != line.take)
                    line.takes.remove(old)
                    drop.append(old.output_id)
            else:
                line.error = error or "Failed"
            with self._lock:
                running = self._rendering.get(project_id)
                if running is not None:
                    running.discard(line_id)
            self._write(project)
        if drop:
            self._outputs.delete(drop)
        self._changed(project_id)

    # -- export -------------------------------------------------------------------------------

    def export(self, key: str, req: ExportRequest | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        """Join the takes in use into one output of kind ``project`` (a chapter, several, or all)."""
        project = self.get(key)
        req = req or ExportRequest()
        self._export_lines(project, req)
        request = {"project": project.id, **req.model_dump(mode="json")}
        return self._jobs.run_now("project_export", request, origin) if wait else self._jobs.submit("project_export", request, origin)

    def _export_lines(self, project: Project, req: ExportRequest) -> list[tuple[Chapter, list[Line]]]:
        if req.chapters is not None:
            unknown = set(req.chapters) - {c.id for c in project.chapters}
            if unknown:
                raise NotFoundError(f"No chapter {sorted(unknown)[0]} in {project.name}", {"kind": "chapter", "id": sorted(unknown)[0]})
        chosen = [c for c in project.chapters if req.chapters is None or c.id in req.chapters]
        not_ready = [line for c in chosen for line in c.lines if line.state not in ("rendered", "empty")]
        if not_ready and not req.allow_missing:
            raise ValidationError(
                f"{len(not_ready)} line(s) of {project.name} have no current take",
                {"lines": [line.id for line in not_ready[:50]], "hint": "render them first, or export with allow_missing"},
            )
        out = [(c, [line for line in c.lines if line.state == "rendered" or (req.allow_missing and self._take(line) is not None)]) for c in chosen]
        if not any(lines for _, lines in out):
            raise ValidationError(f"Nothing of {project.name} is rendered yet", {"hint": "mizuoto project render"})
        return out

    def _export_job(self, request: dict[str, Any]) -> JobSpec:
        project = self.get(str(request["project"]))
        req = ExportRequest.model_validate({k: v for k, v in request.items() if k != "project"})
        return JobSpec(title=f"Export {project.name}", run=lambda ctx: self._run_export(project.id, req, ctx), resources=frozenset({"cpu"}))

    def _run_export(self, project_id: str, req: ExportRequest, ctx: JobContext) -> dict[str, Any]:
        import numpy as np

        from mizuoto.core.audio import dsp
        from mizuoto.core.audio.io import decode, encode

        project = self.get(project_id)
        parts = self._export_lines(project, req)
        rate = project.sample_rate
        fmt = req.format or project.format
        pieces: list[np.ndarray] = []
        cues: list[dict[str, Any]] = []
        chapters: list[dict[str, Any]] = []
        tags: set[str] = set()
        position = 0
        total = sum(len(lines) for _, lines in parts)
        done = 0
        names = {s.id: s.name for s in project.speakers}
        for chapter, lines in parts:
            if not lines:
                continue
            if pieces:
                pieces.append(np.zeros(int(rate * project.chapter_pause_ms / 1000), np.float32))
                position += len(pieces[-1])
            chapters.append({"id": chapter.id, "title": chapter.title, "start": round(position / rate, 3)})
            for li, line in enumerate(lines):
                ctx.check_cancel()
                take = self._take(line)
                if take is None:
                    continue
                out = self._outputs.get(take.output_id)
                tags.update(out.licence_tags)
                audio, _ = decode(self._outputs.path_of(take.output_id), rate)
                speaker = self.speaker_of(project, line)
                cues.append(
                    {
                        "line": line.id,
                        "start": round(position / rate, 3),
                        "end": round((position + len(audio)) / rate, 3),
                        "text": line.text,
                        "speaker": speaker.name if speaker else None,
                    }
                )
                pieces.append(audio)
                position += len(audio)
                if li < len(lines) - 1:
                    pause = line.pause_ms if line.pause_ms is not None else (project.paragraph_pause_ms if line.ends_paragraph else project.line_pause_ms)
                    pieces.append(np.zeros(int(rate * pause / 1000), np.float32))
                    position += len(pieces[-1])
                done += 1
                ctx.progress(done / max(total, 1))
        audio = dsp.sanitize(np.concatenate(pieces) if pieces else np.zeros(0, np.float32))
        single = len(parts) == 1 and len(project.chapters) > 1
        title = f"{project.name} — {parts[0][0].title}" if single and parts[0][0].title else project.name
        path = self._outputs.new_path({"voice": project.slug, "model": "project", "engine": "project"}, fmt)
        encode(audio, rate, path, fmt)
        output = Output(
            id=self._outputs.new_id(),
            kind="project",
            name=title,
            path=str(path),
            format=fmt,
            sample_rate=rate,
            duration=len(audio) / rate,
            size=path.stat().st_size,
            job_id=ctx.job_id,
            text=None,
            meta={"project": project.id, "chapters": chapters, "cues": cues, "speakers": sorted({c["speaker"] for c in cues if c["speaker"]}), "names": names},
            licence_tags=sorted(tags),
            created_at=now_iso(),
        )
        self._outputs.add(output, ctx.job_id)
        return {"output_id": output.id, "output_ids": [output.id], "duration": round(output.duration, 2), "lines": len(cues)}

    def subtitles(self, output_id: str, fmt: str = "srt") -> str:
        """An exported project's cues as SubRip (``srt``) or WebVTT (``vtt``)."""
        out = self._outputs.get(output_id)
        cues = out.meta.get("cues") if out.kind == "project" else None
        if not cues:
            raise ValidationError(f"{out.name} is not an exported project", {"output": output_id})

        def stamp(t: float, sep: str) -> str:
            ms = int(round(t * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"

        if fmt == "vtt":
            body = [f"{stamp(c['start'], '.')} --> {stamp(c['end'], '.')}\n{(c['speaker'] + ': ') if c.get('speaker') else ''}{c['text']}" for c in cues]
            return "WEBVTT\n\n" + "\n\n".join(body) + "\n"
        body = [f"{i}\n{stamp(c['start'], ',')} --> {stamp(c['end'], ',')}\n{c['text']}" for i, c in enumerate(cues, 1)]
        return "\n\n".join(body) + "\n"
