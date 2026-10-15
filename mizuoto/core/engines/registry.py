"""The engines Mizuoto knows, from the manifests shipped with it.

The host learns everything it validates against from these files: no engine needs to be installed,
and no engine code is imported, to list engines, check a request or build a parameter form.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as PydanticValidationError

from mizuoto.core.engines.models import EngineManifest, EngineStatus, LicenceText, ModelEntry, ModelParamSpecs, VariantInfo, VoiceMode
from mizuoto.core.engines.params import param_fields
from mizuoto.core.errors import EnvMissingError, LicenceError, NotFoundError, ValidationError
from mizuoto.core.settings import SettingsService

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

if TYPE_CHECKING:
    from mizuoto.core.inference.service import InferenceService
    from mizuoto.core.remote.service import RemoteService

logger = logging.getLogger(__name__)

MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"

VOICE_MODE_CAPABILITY: dict[str, str] = {
    "clone": "clone",
    "clone_no_transcript": "clone_no_transcript",
    "preset": "preset_speaker",
    "design": "voice_design",
    "auto": "auto_voice",
    "trained": "finetuned_voice",
}


def load_manifest(path: Path) -> EngineManifest:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for vid, variant in (data.get("variants") or {}).items():
        if isinstance(variant, dict):
            variant.setdefault("id", vid)
    try:
        manifest = EngineManifest.model_validate(data)
    except PydanticValidationError as e:
        raise ValidationError(f"Invalid engine manifest {path.name}: {e.errors()[0]['msg']} at {'.'.join(str(p) for p in e.errors()[0]['loc'])}") from e
    for model in manifest.models:
        if not model.id.startswith(f"{manifest.id}/"):
            raise ValidationError(f"{path.name}: model {model.id} must start with '{manifest.id}/'")
        if model.variant not in manifest.variants:
            raise ValidationError(f"{path.name}: model {model.id} names unknown variant {model.variant}")
    for vid, variant in manifest.variants.items():
        if variant.id != vid:
            raise ValidationError(f"{path.name}: variant {vid} has id {variant.id}")
    return manifest


class EngineRegistry:
    def __init__(self, settings: SettingsService, manifests_dir: Path | None = None, extra: list[EngineManifest] | None = None) -> None:
        self._settings = settings
        self._manifests: dict[str, EngineManifest] = {}
        self.remote: RemoteService | None = None
        """Set by ``build_services``: an engine on a remote host is available when it is on the agent."""
        self.inference: InferenceService | None = None
        """Set by ``build_services``: which engines this process can run (their packages import)."""
        for path in sorted((manifests_dir or MANIFESTS_DIR).glob("*.toml")):
            m = load_manifest(path)
            self._manifests[m.id] = m
        for m in extra or []:
            self._manifests[m.id] = m

    # -- manifests -------------------------------------------------------------------

    def ids(self) -> list[str]:
        return sorted(self._manifests)

    def manifest(self, engine_id: str) -> EngineManifest:
        m = self._manifests.get(engine_id)
        if m is None:
            raise NotFoundError(f"Unknown engine: {engine_id}", {"kind": "engine", "id": engine_id, "engines": self.ids()})
        return m

    def add(self, manifest: EngineManifest) -> None:
        """Register a manifest at run time (tests, plugins)."""
        self._manifests[manifest.id] = manifest

    def variant(self, engine_id: str, variant_id: str) -> VariantInfo:
        m = self.manifest(engine_id)
        v = m.variants.get(variant_id)
        if v is None:
            raise NotFoundError(f"{m.title} has no variant {variant_id}", {"kind": "variant", "id": variant_id})
        return v

    def catalog_entry(self, model_id: str) -> tuple[EngineManifest, ModelEntry] | None:
        engine_id = model_id.split("/", 1)[0]
        m = self._manifests.get(engine_id)
        if m is None:
            return None
        for entry in m.models:
            if entry.id == model_id:
                return m, entry
        return None

    def catalog(self, engine_id: str | None = None) -> list[tuple[EngineManifest, ModelEntry]]:
        return [(m, e) for m in self._manifests.values() if engine_id in (None, m.id) for e in m.models]

    # -- state -----------------------------------------------------------------------

    def licence_accepted(self, engine_id: str) -> bool:
        m = self.manifest(engine_id)
        if not m.licence.gated:
            return True
        accepted = self._engine_settings(engine_id).accepted_licence
        return accepted == self.licence_key(m)

    @staticmethod
    def licence_key(manifest: EngineManifest) -> str:
        return f"{manifest.licence.id}@{manifest.licence.version}" if manifest.licence.version else manifest.licence.id

    def check_licence(self, engine_id: str) -> None:
        m = self.manifest(engine_id)
        if not self.licence_accepted(engine_id):
            raise LicenceError(
                f"{m.title} is licensed under {m.licence.name}, which must be accepted first",
                {"engine": engine_id, "licence": self.licence_key(m), "url": m.licence.url, "hint": f"Read it, then: mizuoto engine accept-licence {engine_id}"},
            )

    def licence_text(self, engine_id: str) -> LicenceText:
        m = self.manifest(engine_id)
        text = ""
        if m.licence.file:
            try:
                text = (MANIFESTS_DIR / "licences" / m.licence.file).read_text(encoding="utf-8")
            except OSError:
                text = ""
        return LicenceText(engine=engine_id, key=self.licence_key(m), name=m.licence.name, url=m.licence.url, text=text, accepted=self.licence_accepted(engine_id))

    def accept_licence(self, engine_id: str, key: str | None = None) -> EngineStatus:
        """Record that the user accepted the engine's licence. ``key`` (``id@version``), when given,
        must name the licence shown now, so a newer version is never accepted unseen."""
        m = self.manifest(engine_id)
        current = self.licence_key(m)
        if key is not None and key != current:
            raise ValidationError(f"The licence of {m.title} is now {current}; read it again", {"licence": current, "given": key})
        self._settings.update({"engines": {engine_id: {"accepted_licence": current}}})
        return self.status(engine_id)

    def revoke_licence(self, engine_id: str) -> EngineStatus:
        self.manifest(engine_id)
        self._settings.update({"engines": {engine_id: {"accepted_licence": None}}})
        return self.status(engine_id)

    def configure(self, engine_id: str, patch: dict[str, Any]) -> EngineStatus:
        """Change an engine's settings (``engines.<id>``): enabled, device, dtype, load options, idle minutes, pinned models.
        Load options are checked against the manifest. Its models load again with the new settings."""
        from mizuoto.core.engines.params import check_options
        from mizuoto.core.settings.models import EngineSettings

        m = self.manifest(engine_id)
        unknown = sorted(set(patch) - set(EngineSettings.model_fields) - {"accepted_licence"})
        if unknown or "accepted_licence" in patch:
            raise ValidationError(f"Unknown engine settings: {', '.join(unknown or ['accepted_licence'])}", {"fields": sorted(EngineSettings.model_fields)})
        if "load_options" in patch:
            # The options as given replace the stored ones: a removed option is cleared (None deletes a key).
            given = check_options(m, "", dict(patch["load_options"] or {}), schema=m.load_options)
            current = self._engine_settings(engine_id).load_options
            patch = {**patch, "load_options": {**{k: None for k in current if k not in given}, **given}}
        self._settings.update({"engines": {engine_id: patch}})
        return self.status(engine_id)

    def check_available(self, engine_id: str) -> None:
        """Refuse an engine whose packages do not import here, with what to install. An engine on a remote host is the
        agent's to refuse (when its session opens)."""
        if self.inference is None or (self.remote is not None and self.remote.host_for(engine_id) is not None):
            return
        missing = self.inference.missing(engine_id)
        if missing:
            title = self.manifest(engine_id).title
            raise EnvMissingError(f"{title} needs {', '.join(missing)}, which is not installed", {"engine": engine_id, "missing": missing, "hint": f"mizuoto doctor {engine_id}"})

    def enabled(self, engine_id: str) -> bool:
        return self._engine_settings(engine_id).enabled

    def check_enabled(self, engine_id: str) -> None:
        if not self.enabled(engine_id):
            m = self.manifest(engine_id)
            raise ValidationError(f"{m.title} is disabled", {"engine": engine_id, "hint": f"mizuoto engine enable {engine_id}"})

    def _engine_settings(self, engine_id: str):  # noqa: ANN202 - EngineSettings, imported lazily
        from mizuoto.core.settings.models import EngineSettings

        return self._settings.settings.engines.get(engine_id) or EngineSettings()

    def status(self, engine_id: str) -> EngineStatus:
        m = self.manifest(engine_id)
        missing = self.inference.missing(engine_id) if self.inference is not None else []
        host = self.remote.host_for(engine_id) if self.remote is not None else None
        if host is not None and self.remote is not None:
            missing = self.remote.engine_missing(host, engine_id)
        return EngineStatus(
            id=m.id,
            title=m.title,
            description=m.description,
            licence=m.licence,
            licence_accepted=self.licence_accepted(m.id),
            enabled=self.enabled(m.id),
            available=not missing,
            missing=missing,
            internal=m.internal,
            variants=sorted(m.variants),
            models=[e.id for e in m.models],
            devices=list(m.devices),
            weights=m.weights,
            training={k: v.variants for k, v in m.training.items() if v.enabled},
            load_options=dict(m.load_options),
            remote=host,
        )

    def list_engines(self) -> list[EngineStatus]:
        return [self.status(i) for i in self.ids()]

    # -- capabilities ----------------------------------------------------------------

    @staticmethod
    def voice_modes(variant: VariantInfo) -> list[VoiceMode]:
        caps = set(variant.capabilities)
        modes: list[VoiceMode] = []
        for mode, cap in VOICE_MODE_CAPABILITY.items():
            if cap in caps or (mode == "design" and "voice_design_freeform" in caps):
                modes.append(mode)  # ty: ignore[invalid-argument-type]
        return modes

    def param_specs(self, model_id: str, engine_id: str, variant_id: str, speakers: list[str] | None = None) -> ModelParamSpecs:
        m = self.manifest(engine_id)
        v = self.variant(engine_id, variant_id)
        granularity = v.stream_granularity or "emulated"
        return ModelParamSpecs(
            model=model_id,
            engine=engine_id,
            variant=variant_id,
            sample_rate=v.sample_rate,
            capabilities=list(v.capabilities),
            voice_modes=self.voice_modes(v),
            languages=list(v.languages or m.languages),
            params=param_fields(m, variant_id),
            reference=v.reference,
            design_attributes=v.design_attributes,
            inline_tags=list(v.inline_tags),
            emotions=list(v.emotions),
            speakers=list(speakers if speakers is not None else v.speakers),
            stream_granularity=granularity,
        )
