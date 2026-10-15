"""The voice library.

A voice is a folder ``<voices>/<slug>/`` with ``voice.json`` and its reference clips in
``references/<sha256>.flac`` (mono, at the clip's own rate). The folders are the truth: the list is
read from them, so copying a voice folder in is enough to add it. Derived engine state (reference
prompts, speaker embeddings) lives under ``cache/<engine>/<model>/<key>/`` in the folder and can be
deleted at any time.

``fit(voice, model)`` says whether and how a model can use a voice; synthesis plans with it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.audio.io import decode, encode, sha256_file
from mizuoto.core.audio.service import AudioFile, AudioService
from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.engines import languages
from mizuoto.core.engines.models import VariantInfo
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import ConflictError, NotFoundError, ValidationError
from mizuoto.core.files import trash
from mizuoto.core.models.service import ModelService
from mizuoto.core.settings import SettingsService
from mizuoto.core.voices.models import Consent, Voice, VoiceDesign, VoiceFit, VoicePreset, VoiceReference, VoiceSummary, VoiceWeights, WeightsUpload

if TYPE_CHECKING:
    from mizuoto.core.asr.service import AsrService
    from mizuoto.core.events import EventBus
    from mizuoto.core.jobs.service import JobContext, JobService
    from mizuoto.core.outputs.service import OutputService
    from mizuoto.core.tts.service import TtsService

VOICE_FILE = "voice.json"
REFERENCES = "references"
WEIGHTS = "weights"


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w-]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-_")
    return slug[:48] or "voice"


def consent_from(value: Consent | str | None) -> Consent | None:
    if value is None or isinstance(value, Consent):
        return value
    return Consent(source=value, confirmed_at=now_iso()) if value.strip() else None


class VoiceService:
    def __init__(
        self,
        settings: SettingsService,
        registry: EngineRegistry,
        models: ModelService,
        audio: AudioService,
        outputs: OutputService,
        jobs: JobService,
        asr: AsrService | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._models = models
        self._audio = audio
        self._outputs = outputs
        self._jobs = jobs
        self._asr = asr
        self.tts: TtsService | None = None
        """Set by ``build_services`` (designing a voice speaks)."""
        self.events: EventBus | None = None
        """Set by ``build_services``: ``voices_changed`` after every change."""

    def _changed(self, *ids: str) -> None:
        if self.events is not None:
            from mizuoto.core.events.models import VoicesChangedEvent

            self.events.publish(VoicesChangedEvent(ids=list(ids)))

    @property
    def root(self) -> Path:
        return self._settings.voices_dir

    # -- storage --------------------------------------------------------------------------

    def folder(self, voice: Voice) -> Path:
        return self.root / voice.slug

    def _read(self, folder: Path) -> Voice | None:
        try:
            return Voice.model_validate_json((folder / VOICE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, voice: Voice) -> Voice:
        folder = self.folder(voice)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / (VOICE_FILE + ".tmp")
        tmp.write_text(voice.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, folder / VOICE_FILE)
        self._changed(voice.id)
        return voice

    def _all(self) -> list[Voice]:
        out = []
        for folder in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if folder.is_dir():
                voice = self._read(folder)
                if voice is not None:
                    if voice.slug != folder.name:
                        voice.slug = folder.name
                    out.append(voice)
        return out

    def list_voices(self, q: str | None = None, kind: str | None = None, language: str | None = None) -> list[VoiceSummary]:
        out = []
        needle = (q or "").strip().lower()
        for v in self._all():
            if kind and v.kind != kind:
                continue
            if language and v.language != languages.normalize(language):
                continue
            if needle and needle not in f"{v.name} {v.slug} {v.description} {' '.join(v.tags)}".lower():
                continue
            out.append(
                VoiceSummary(
                    id=v.id,
                    slug=v.slug,
                    name=v.name,
                    kind=v.kind,
                    language=v.language,
                    tags=v.tags,
                    references=len(v.references),
                    duration=round(sum(r.duration for r in v.references), 2),
                    preset_model=v.preset.model if v.preset else None,
                    primary=self.ordered_references(v)[0].sha256 if v.references else None,
                    description=v.description,
                    updated_at=v.updated_at,
                )
            )
        out.sort(key=lambda s: s.name.lower())
        return out

    def get(self, key: str) -> Voice:
        """A voice by id or slug."""
        folder = self.root / key
        if (folder / VOICE_FILE).is_file():
            voice = self._read(folder)
            if voice is not None:
                voice.slug = folder.name
                return voice
        for voice in self._all():
            if voice.id == key:
                return voice
        raise NotFoundError(f"Unknown voice: {key}", {"kind": "voice", "id": key, "hint": "mizuoto voice list"})

    def _new(self, name: str, kind: str, **fields: Any) -> Voice:
        if not name.strip():
            raise ValidationError("A voice needs a name")
        base = slugify(name)
        slug, n = base, 2
        while (self.root / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        now = now_iso()
        return Voice(id=new_id("v"), slug=slug, name=name.strip(), kind=kind, created_at=now, updated_at=now, **fields)  # ty: ignore[invalid-argument-type]

    # -- references -----------------------------------------------------------------------

    def _store_reference(self, voice: Voice, source: Path, transcript: str | None, language: str | None, transcript_source: str | None) -> VoiceReference:
        """Keep a clip in the voice's folder as mono FLAC at its own rate."""
        samples, sr = decode(source)
        refs = self.folder(voice) / REFERENCES
        refs.mkdir(parents=True, exist_ok=True)
        tmp = refs / f"incoming-{new_id()}.flac"
        encode(samples, sr, tmp, "flac")
        sha = sha256_file(tmp)
        target = refs / f"{sha}.flac"
        if target.exists():
            tmp.unlink()
        else:
            os.replace(tmp, target)
        text = transcript.strip() if transcript and transcript.strip() else None
        return VoiceReference(
            sha256=sha,
            file=f"{REFERENCES}/{sha}.flac",
            transcript=text,
            transcript_source=transcript_source if text else None,  # ty: ignore[invalid-argument-type]
            language=languages.normalize(language),
            duration=round(len(samples) / sr, 3) if sr else 0.0,
            sample_rate=sr,
        )

    def reference_audio(self, voice: Voice, ref: VoiceReference) -> AudioFile:
        return self._audio.register_path(self.folder(voice) / ref.file, origin="voice")

    def _transcribe_missing(self, voice: Voice, ctx: JobContext | None = None) -> list[str]:
        """Fill missing transcripts with the ASR engine when it is available; return warnings."""
        missing = [r for r in voice.references if not r.transcript]
        if not missing:
            return []
        if self._asr is None or not self._asr.available():
            reason = self._asr.unavailable_reason() if self._asr is not None else "speech recognition is not set up"
            return [f"{len(missing)} reference(s) have no transcript and {reason}"]
        for ref in missing:
            t = self._asr.transcribe(self.reference_audio(voice, ref), ref.language or voice.language, ctx=ctx)
            ref.transcript, ref.transcript_source = t.text, "asr"
            if not ref.language and t.language:
                ref.language = t.language
        return []

    def set_transcript(self, key: str, sha256: str, text: str, source: str = "user") -> Voice:
        voice = self.get(key)
        ref = self._ref(voice, sha256)
        ref.transcript = text.strip() or None
        ref.transcript_source = source if ref.transcript else None  # ty: ignore[invalid-assignment]
        voice.updated_at = now_iso()
        return self._write(voice)

    def find_reference(self, voice: Voice, sha256: str) -> VoiceReference:
        """A reference of the voice by its SHA-256 (a prefix will do)."""
        return self._ref(voice, sha256)

    @staticmethod
    def _ref(voice: Voice, sha256: str) -> VoiceReference:
        for r in voice.references:
            if r.sha256 == sha256 or r.sha256.startswith(sha256):
                return r
        raise NotFoundError(f"{voice.name} has no reference {sha256}", {"kind": "reference", "id": sha256})

    @staticmethod
    def _ensure_primary(voice: Voice) -> None:
        if voice.references and not any(r.primary for r in voice.references):
            voice.references[0].primary = True

    @staticmethod
    def ordered_references(voice: Voice) -> list[VoiceReference]:
        """The primary reference first, then the others in order."""
        return sorted(voice.references, key=lambda r: not r.primary)

    # -- creating -------------------------------------------------------------------------

    def create_from_audio(
        self,
        name: str,
        clips: list[Path],
        transcripts: list[str | None] | None = None,
        language: str | None = None,
        consent: Consent | str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        transcribe: bool = True,
        ctx: JobContext | None = None,
    ) -> tuple[Voice, list[str]]:
        """A ``reference`` voice from clips on this machine; missing transcripts come from ASR when it is set up."""
        if not clips:
            raise ValidationError("A voice needs at least one reference clip")
        texts = list(transcripts or [])
        if len(texts) > len(clips):
            raise ValidationError(f"{len(texts)} transcripts for {len(clips)} clips")
        texts += [None] * (len(clips) - len(texts))
        for clip in clips:
            if not clip.expanduser().is_file():
                raise NotFoundError(f"No such file: {clip}", {"kind": "file", "id": str(clip)})
        lang = languages.normalize(language)
        voice = self._new(name, "reference", language=lang, description=description, tags=list(tags or []), consent=consent_from(consent))
        try:
            for clip, text in zip(clips, texts):
                ref = self._store_reference(voice, clip.expanduser().resolve(), text, lang, "user")
                if all(r.sha256 != ref.sha256 for r in voice.references):
                    voice.references.append(ref)
            self._ensure_primary(voice)
            warnings = self._transcribe_missing(voice, ctx) if transcribe else []
            return self._write(voice), warnings
        except BaseException:
            shutil.rmtree(self.folder(voice), ignore_errors=True)
            raise

    def create_preset(self, model: str, speaker: str, name: str | None = None) -> Voice:
        """A voice that is one model's preset speaker."""
        info = self._models.get(model)
        variant = self._registry.variant(info.engine, info.variant)
        if "preset_speaker" not in variant.capabilities:
            raise ValidationError(f"{info.title} has no preset speakers", {"model": model})
        speakers = self._models.speakers(model)
        if speakers and speaker not in speakers:
            raise ValidationError(f"{info.title} has no speaker {speaker!r}", {"speakers": speakers})
        for v in self._all():
            if v.preset and v.preset.model == model and v.preset.speaker == speaker:
                raise ConflictError(f"{v.name} is already {speaker} of {model}", {"voice": v.id})
        voice = self._new(name or f"{speaker} ({info.title})", "preset", preset=VoicePreset(engine=info.engine, model=model, speaker=speaker))
        return self._write(voice)

    def create_from_output(self, output_id: str, name: str, consent: Consent | str | None = None, description: str = "") -> Voice:
        """Keep a synthesized clip as a voice: a ``designed`` voice when it came from a design, else a ``reference`` voice."""
        out = self._outputs.get(output_id)
        request: dict[str, Any] = {}
        if out.job_id:
            try:
                request = self._jobs.get(out.job_id).request
            except NotFoundError:
                request = {}
        spec = request.get("voice_spec") or {}
        designed = out.meta.get("voice_mode") == "design" and (spec.get("description") or spec.get("attributes"))
        kind = "designed" if designed else "reference"
        voice = self._new(name, kind, language=out.language, description=description or (spec.get("description") or ""), consent=consent_from(consent))
        if designed:
            seed = (request.get("params") or {}).get("seed")
            voice.design = VoiceDesign(
                model=out.model or "",
                description=spec.get("description"),
                attributes=spec.get("attributes"),
                sample_text=out.text or "",
                language=out.language,
                seed=seed,
            )
        try:
            ref = self._store_reference(voice, Path(out.path), out.text, out.language, "design" if designed else "user")
            ref.primary = True
            voice.references.append(ref)
            return self._write(voice)
        except BaseException:
            shutil.rmtree(self.folder(voice), ignore_errors=True)
            raise

    def create_from_design(
        self,
        name: str,
        model: str,
        sample_text: str,
        description: str | None = None,
        attributes: dict[str, str] | None = None,
        language: str | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[Voice, str]:
        """Design a voice with a design model, speaking ``sample_text``; the clip becomes the voice's
        reference, so every cloning model can use it. Returns the voice and the output id."""
        from mizuoto.core.tts.models import CommonParams, SpeakRequest, VoiceInput

        if self.tts is None:
            raise ValidationError("Designing needs the TTS service")
        if not (description or attributes):
            raise ValidationError("Describe the voice (a description or attributes)")
        p = CommonParams.model_validate({**(params or {}), **({"seed": seed} if seed is not None else {})})
        req = SpeakRequest(model=model, text=sample_text, language=language, voice_spec=VoiceInput(description=description, attributes=attributes), params=p)
        job = self.tts.speak(req, wait=True, origin="voice_design")
        if job.state != "completed" or not job.result:
            from mizuoto.core.errors import error_from_code

            error = job.error
            raise error_from_code(error.code, error.message, error.detail) if error else ValidationError(f"The design job ended {job.state}")
        output_id = job.result["output_id"]
        voice = self.create_from_output(output_id, name, description=description or "")
        return voice, output_id

    # -- changing -------------------------------------------------------------------------

    def update(
        self,
        key: str,
        name: str | None = None,
        language: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        consent: Consent | str | None = None,
    ) -> Voice:
        voice = self.get(key)
        if name is not None:
            if not name.strip():
                raise ValidationError("A voice needs a name")
            voice.name = name.strip()
        if language is not None:
            voice.language = languages.normalize(language)
        if description is not None:
            voice.description = description
        if tags is not None:
            voice.tags = list(tags)
        if consent is not None:
            voice.consent = consent_from(consent)
        voice.updated_at = now_iso()
        return self._write(voice)

    def delete(self, key: str) -> None:
        voice = self.get(key)
        trash(self.folder(voice), self._settings.data_dir / "trash")
        self._changed(voice.id)

    def add_reference(
        self, key: str, clip: Path, transcript: str | None = None, language: str | None = None, primary: bool = False, transcribe: bool = True
    ) -> tuple[Voice, list[str]]:
        voice = self.get(key)
        if voice.kind == "preset":
            raise ValidationError("A preset voice has no reference clips")
        ref = self._store_reference(voice, clip.expanduser().resolve(), transcript, language or voice.language, "user")
        if any(r.sha256 == ref.sha256 for r in voice.references):
            raise ConflictError(f"{voice.name} already has this clip", {"sha256": ref.sha256})
        voice.references.append(ref)
        if primary:
            for r in voice.references:
                r.primary = r.sha256 == ref.sha256
        self._ensure_primary(voice)
        warnings = self._transcribe_missing(voice) if transcribe else []
        voice.updated_at = now_iso()
        return self._write(voice), warnings

    def remove_reference(self, key: str, sha256: str) -> Voice:
        voice = self.get(key)
        ref = self._ref(voice, sha256)
        voice.references = [r for r in voice.references if r.sha256 != ref.sha256]
        (self.folder(voice) / ref.file).unlink(missing_ok=True)
        self._ensure_primary(voice)
        voice.updated_at = now_iso()
        return self._write(voice)

    # -- fine-tuned weights --------------------------------------------------------------

    def add_weights(self, key: str, engine: str, files: dict[str, Path], move: bool = False) -> Voice:
        """Give the voice its own fine-tuned weights for ``engine`` (replacing earlier ones): the files of
        every role the engine's manifest names, copied (or moved) into ``weights/<engine>/``."""
        from mizuoto.core.models.detect import weights_version

        voice = self.get(key)
        spec = self._registry.manifest(engine).weights
        if spec is None:
            raise ValidationError(f"{engine} takes no fine-tuned voice weights", {"engine": engine})
        missing = sorted(set(spec.roles) - set(files))
        unknown = sorted(set(files) - set(spec.roles))
        if missing or unknown:
            raise ValidationError(
                f"{engine} weights are {', '.join(f'{r} ({sfx})' for r, sfx in spec.roles.items())}", {"missing": missing, "unknown": unknown, "roles": spec.roles}
            )
        for role, path in files.items():
            if not path.is_file():
                raise NotFoundError(f"No file {path}", {"path": str(path)})
            if not path.name.lower().endswith(spec.roles[role]):
                raise ValidationError(f"The {role} weights should be a {spec.roles[role]} file, not {path.name}", {"role": role})
        version = weights_version(engine, files)
        target = self.folder(voice) / WEIGHTS / engine
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        stored: dict[str, str] = {}
        for role, path in files.items():
            dest = target / f"{role}{spec.roles[role]}"
            (shutil.move if move else shutil.copyfile)(str(path), str(dest))
            stored[role] = dest.relative_to(self.folder(voice)).as_posix()
        voice.weights[engine] = VoiceWeights(version=version, files=stored, names={r: p.name for r, p in files.items()}, added_at=now_iso())
        voice.updated_at = now_iso()
        return self._write(voice)

    def upload_weights(self, key: str, engine: str, role: str, path: Path, name: str | None = None) -> WeightsUpload:
        """One file of a weights pair, uploaded: kept until every role is in, then added (``add_weights``)."""
        voice = self.get(key)
        spec = self._registry.manifest(engine).weights
        if spec is None or role not in spec.roles:
            raise ValidationError(f"{engine} weights have no file {role!r}", {"roles": spec.roles if spec else {}})
        if name and not name.lower().endswith(spec.roles[role]):
            raise ValidationError(f"The {role} weights should be a {spec.roles[role]} file, not {name}", {"role": role})
        staging = self._settings.uploads_dir / WEIGHTS / voice.id / engine
        staging.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(staging / f"{role}{spec.roles[role]}"))
        (staging / f"{role}.name").write_text(name or f"{role}{spec.roles[role]}", encoding="utf-8")
        staged = sorted(r for r in spec.roles if (staging / f"{r}{spec.roles[r]}").is_file())
        missing = sorted(set(spec.roles) - set(staged))
        if missing:
            return WeightsUpload(engine=engine, staged=staged, missing=missing)
        files = {r: staging / f"{r}{spec.roles[r]}" for r in spec.roles}
        names = {r: (staging / f"{r}.name").read_text(encoding="utf-8") for r in spec.roles}
        voice = self.add_weights(key, engine, files, move=True)
        voice.weights[engine].names = names
        voice = self._write(voice)
        shutil.rmtree(staging, ignore_errors=True)
        return WeightsUpload(engine=engine, staged=staged, missing=[], voice=voice)

    def remove_weights(self, key: str, engine: str) -> Voice:
        voice = self.get(key)
        if engine not in voice.weights:
            raise NotFoundError(f"{voice.name} has no weights for {engine}", {"engine": engine})
        shutil.rmtree(self.folder(voice) / WEIGHTS / engine, ignore_errors=True)
        del voice.weights[engine]
        voice.updated_at = now_iso()
        return self._write(voice)

    def weight_paths(self, voice: Voice, engine: str) -> dict[str, str]:
        """Absolute paths of the voice's weights for ``engine``, by role."""
        return {role: str(self.folder(voice) / rel) for role, rel in voice.weights[engine].files.items()}

    def set_primary(self, key: str, sha256: str) -> Voice:
        voice = self.get(key)
        target = self._ref(voice, sha256)
        for r in voice.references:
            r.primary = r.sha256 == target.sha256
        voice.updated_at = now_iso()
        return self._write(voice)

    def retranscribe(self, key: str, sha256: str | None = None) -> Voice:
        """Transcribe the references again with the ASR engine (all, or one)."""
        voice = self.get(key)
        if self._asr is None:
            raise ValidationError("Speech recognition is not set up")
        reason = self._asr.unavailable_reason()
        if reason is not None:
            # Checked before the old transcripts are forgotten; the model names what to download.
            model = self._asr.default_model()
            raise ValidationError(f"Cannot transcribe: {reason}", {"reason": "asr_unavailable", "model": model})
        targets = [self._ref(voice, sha256)] if sha256 else voice.references
        for ref in targets:
            self._asr.forget(ref.sha256)
            ref.transcript, ref.transcript_source = None, None
        warnings = self._transcribe_missing(voice)
        if warnings:
            raise ValidationError(warnings[0])
        voice.updated_at = now_iso()
        return self._write(voice)

    # -- archives -------------------------------------------------------------------------

    def export(self, key: str, target: Path) -> Path:
        """A zip of the voice: ``voice.json`` and its reference clips (not its caches)."""
        voice = self.get(key)
        folder = self.folder(voice)
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(folder / VOICE_FILE, VOICE_FILE)
            for ref in voice.references:
                z.write(folder / ref.file, ref.file)
            for w in voice.weights.values():
                for rel in w.files.values():
                    z.write(folder / rel, rel)
        return target

    def import_archive(self, archive: Path, name: str | None = None) -> Voice:
        """Add a voice exported by ``export`` (a new id; a new slug when the name is taken)."""
        with zipfile.ZipFile(archive) as z:
            try:
                data = json.loads(z.read(VOICE_FILE).decode("utf-8"))
            except KeyError:
                raise ValidationError(f"{archive.name} is not a Mizuoto voice (no {VOICE_FILE})") from None
            original = Voice.model_validate(data)
            voice = self._new(name or original.name, original.kind)
            fields = original.model_dump(exclude={"id", "slug", "name", "created_at", "updated_at", "kind"})
            voice = voice.model_copy(update={k: getattr(original, k) for k in fields})
            folder = self.folder(voice)
            with tempfile.TemporaryDirectory(dir=self.root if self.root.is_dir() else None) as tmp:
                for ref in voice.references:
                    member = ref.file
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise ValidationError(f"{archive.name} has an unsafe path: {member}")
                    out = Path(tmp) / member
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(member))
                    if sha256_file(out) != ref.sha256:
                        raise ValidationError(f"{member} in {archive.name} does not match its checksum")
                for w in voice.weights.values():
                    for rel in w.files.values():
                        if rel.startswith("/") or ".." in Path(rel).parts or not rel.startswith(f"{WEIGHTS}/"):
                            raise ValidationError(f"{archive.name} has an unsafe path: {rel}")
                        out = Path(tmp) / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(rel) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
                folder.mkdir(parents=True, exist_ok=True)
                for part in (REFERENCES, WEIGHTS):
                    if (Path(tmp) / part).is_dir():
                        shutil.copytree(Path(tmp) / part, folder / part, dirs_exist_ok=True)
        return self._write(voice)

    # -- use by models --------------------------------------------------------------------

    def cache_dir(self, voice: Voice, engine: str, model: str) -> Path:
        return self.folder(voice) / "cache" / engine / model.split("/", 1)[-1]

    def fit(self, key: str, model: str) -> VoiceFit:
        """Whether ``model`` can use the voice, in which mode, and what must happen first."""
        voice = self.get(key)
        info = self._models.get(model)
        variant = self._registry.variant(info.engine, info.variant)
        return self.fit_voice(voice, model, variant)

    def fits(self, key: str) -> list[VoiceFit]:
        """How every installed synthesis model can use the voice."""
        voice = self.get(key)
        out = []
        for info in self._models.list_models(installed=True):
            if self._registry.manifest(info.engine).internal:
                continue
            out.append(self.fit_voice(voice, info.id, self._registry.variant(info.engine, info.variant)))
        return out

    def fits_for_model(self, model: str) -> list[VoiceFit]:
        """How a model can use each voice of the library (the voice picker's badges)."""
        info = self._models.get(model)
        variant = self._registry.variant(info.engine, info.variant)
        return [self.fit_voice(v, model, variant) for v in self._all()]

    def fit_voice(self, voice: Voice, model: str, variant: VariantInfo) -> VoiceFit:
        caps = set(variant.capabilities)
        fit = VoiceFit(voice_id=voice.id, model=model, usable=False)
        if voice.kind == "preset":
            assert voice.preset is not None
            if voice.preset.model != model:
                fit.reason = f"{voice.name} is a preset speaker of {voice.preset.model}"
            elif "preset_speaker" not in caps:
                fit.reason = "The model has no preset speakers"
            else:
                fit.usable, fit.mode = True, "preset"
            return fit
        engine = model.split("/", 1)[0]
        own = voice.weights.get(engine)
        if own is not None and "finetuned_voice" in caps:
            if variant.weights_version is None or own.version == variant.weights_version:
                fit.weights = True
            else:
                fit.warnings.append(f"Its own {engine} weights are {own.version}, which this model ({variant.weights_version}) cannot use; the clip is cloned instead")
        if not ("clone" in caps or "clone_no_transcript" in caps):
            if voice.design and ("voice_design_freeform" in caps or "voice_design" in caps) and (voice.design.description or voice.design.attributes):
                fit.usable, fit.mode = True, "design"
                if voice.design.model != model:
                    fit.warnings.append("The model designs the voice anew from its description; it will not sound the same")
                return fit
            fit.reason = "The model speaks only its own preset speakers" if "preset_speaker" in caps else "The model cannot clone a voice from a clip"
            return fit
        if not voice.references:
            fit.reason = f"{voice.name} has no reference clip"
            return fit
        limit = variant.reference.max_references if "multi_reference" in caps else 1
        used = self.ordered_references(voice)[:limit]
        policy = self._settings.settings.tts.reference_policy
        lims = variant.reference
        lose_transcript = False
        for ref in used:
            if lims.min_seconds is not None and ref.duration < lims.min_seconds:
                fit.reason = f"The reference is {ref.duration:.1f} s; the model needs at least {lims.min_seconds:g} s"
                return fit
            if lims.min_sample_rate is not None and ref.sample_rate < lims.min_sample_rate:
                fit.reason = f"The reference is {ref.sample_rate} Hz; the model needs at least {lims.min_sample_rate} Hz"
                return fit
            if lims.max_seconds is not None and ref.duration > lims.max_seconds:
                if policy != "trim":
                    fit.reason = f"The reference is {ref.duration:.1f} s; the model takes at most {lims.max_seconds:g} s (or set tts.reference_policy to trim)"
                    return fit
                if "trim" not in fit.needs:
                    fit.needs.append("trim")
                lose_transcript = True
        missing = lose_transcript or any(not r.transcript for r in used)
        if not missing:
            fit.usable, fit.mode = True, "clone" if "clone" in caps else "clone_no_transcript"
            return fit
        auto = self._settings.settings.tts.auto_transcribe and self._asr is not None and self._asr.available()
        if "clone" in caps and auto:
            fit.usable, fit.mode = True, "clone"
            fit.needs.append("transcribe")
        elif "clone_no_transcript" in caps:
            fit.usable, fit.mode = True, "clone_no_transcript"
            if "clone" in caps:
                fit.warnings.append(
                    "Without a transcript the model follows the voice less closely; download speech recognition (mizuoto model download asr/whisper-large-v3-turbo) or add the transcript"
                )
        else:
            fit.reason = "The model needs the reference's transcript" + (f"; {self._asr.unavailable_reason()}" if self._asr is not None else "")
        return fit
