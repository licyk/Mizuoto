"""The model catalogue, what is installed, downloads, verification and imports.

Catalogue models come from the engine manifests, with their files pinned in
``manifests/files/*.files.json``. An installed model is a folder ``<models>/<engine>/<name>/``
holding its files and ``model.json``, which is written last: a folder without it is a partial
download. User models (imports, later fine-tunes) are folders with a ``model.json`` that is not in a
manifest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.clock import now_iso
from mizuoto.core.engines.models import AssetEntry, ModelEntry, ModelFiles
from mizuoto.core.engines.registry import MANIFESTS_DIR, EngineRegistry
from mizuoto.core.errors import AssetMissingError, ConflictError, ModelMissingError, NotFoundError, ValidationError
from mizuoto.core.jobs.models import Job, JobStep
from mizuoto.core.jobs.service import JobCancelled, JobContext, JobService, JobSpec
from mizuoto.core.models.archives import unpack
from mizuoto.core.models.detect import detect
from mizuoto.core.models.download import AnyEvent, Cancelled, FilesManifest, Progress, RemoteFile, fetch, sha256_of, sources
from mizuoto.core.models.models import AssetInfo, Detected, ModelInfo, ModelState, VerifyResult
from mizuoto.core.settings import SettingsService

MODEL_FILE = "model.json"
ASSET_FILE = "asset.json"
UNPACK_DIR = ".unpack"
"""Archives being unpacked, and a marker per archive listing what it became."""
FILES_DIR = MANIFESTS_DIR / "files"

if TYPE_CHECKING:
    from mizuoto.core.remote.service import RemoteService


def load_files_manifest(name: str) -> FilesManifest:
    path = FILES_DIR / name
    try:
        return FilesManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise NotFoundError(f"The files manifest {name} is missing", {"kind": "files_manifest", "id": name}) from e


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-.")
    return slug or "model"


class ModelService:
    def __init__(self, settings: SettingsService, registry: EngineRegistry, jobs: JobService | None = None) -> None:
        self._settings = settings
        self._registry = registry
        self._jobs = jobs
        self._downloading: set[str] = set()
        self._lock = threading.Lock()
        self.before_remove: Callable[[str], None] | None = None
        """Called with a model id before its folder goes (the runtime unloads it)."""
        self.http_transport: Any = None
        """Replaces the network in tests (an ``httpx.MockTransport``)."""
        self.events: Any = None
        """Set by ``build_services``: ``models_changed`` after a download, import, verification or deletion."""
        self.remote: RemoteService | None = None
        """Set by ``build_services``: models of an engine on a remote host are the agent's."""
        if jobs is not None:
            jobs.register("model_download", self._download_job)
            jobs.register("asset_download", self._asset_job)

    def _changed(self, *ids: str) -> None:
        if self.events is not None:
            from mizuoto.core.events.models import ModelsChangedEvent

            self.events.publish(ModelsChangedEvent(ids=list(ids)))

    # -- where things are -----------------------------------------------------------

    def folder(self, model_id: str) -> Path:
        engine, _, name = model_id.partition("/")
        return self._settings.models_dir / engine / name

    def _sidecar(self, model_id: str) -> dict | None:
        try:
            return json.loads((self.folder(model_id) / MODEL_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def files_of(self, entry: ModelEntry | AssetEntry) -> list[tuple[ModelFiles, FilesManifest]]:
        """Each repository a catalogue model is made of, with its pinned file list."""
        parts = [f for f in [entry.files, *entry.extra] if f is not None]
        out = []
        for files in parts:
            if not files.manifest:
                raise ValidationError(f"{entry.id} has no files manifest", {"model": entry.id, "hint": "python scripts/model_manifests.py"})
            out.append((files, load_files_manifest(files.manifest)))
        return out

    def _size_mb(self, entry: ModelEntry | AssetEntry) -> float | None:
        fallback = entry.size_mb if isinstance(entry, ModelEntry) else None
        try:
            total = sum(f.size for _, fm in self.files_of(entry) for f in fm.files)
        except (NotFoundError, ValidationError):
            return fallback
        return round(total / (1024 * 1024), 1) if total else fallback

    # -- listing --------------------------------------------------------------------

    def _state(self, model_id: str, sidecar: dict | None) -> ModelState:
        with self._lock:
            if model_id in self._downloading:
                return "downloading"
        return self._files_state(self.folder(model_id), sidecar)

    @staticmethod
    def _files_state(folder: Path, sidecar: dict | None) -> ModelState:
        if sidecar is None:
            return "partial" if folder.is_dir() and any(folder.iterdir()) else "missing"
        for f in sidecar.get("files") or []:
            path = folder / f["path"]
            try:
                if path.stat().st_size != f["size"]:
                    return "corrupt"
            except OSError:
                return "corrupt"
        if (sidecar.get("verified") or {}).get("ok") is False:
            return "corrupt"
        return "installed"

    def get(self, model_id: str) -> ModelInfo:
        found = self._registry.catalog_entry(model_id)
        if found is not None:
            manifest, entry = found
            if entry.builtin:
                state, path = "builtin", None
            else:
                sidecar = self._sidecar(model_id)
                state = self._state(model_id, sidecar)
                if state == "installed" and any(self.asset(a).state != "installed" for a in entry.assets):
                    state = "partial"  # its files are there, a shared asset it needs is not
                path = str(self.folder(model_id)) if state == "installed" else None
            host = self.remote.host_for(manifest.id) if self.remote is not None and not entry.builtin else None
            if host is not None and self.remote is not None:
                state, path = self.remote.model_state(host, model_id), None
            return ModelInfo(
                id=entry.id,
                engine=manifest.id,
                variant=entry.variant,
                title=entry.title,
                description=entry.description,
                state=state,
                path=path,
                size_mb=None if entry.builtin else self._size_mb(entry),
                vram_mb=entry.vram_mb,
                languages=entry.languages,
                recommended=entry.recommended,
                capabilities=self._capabilities(manifest.id, entry.variant),
                assets=list(entry.assets),
                remote=host,
            )
        sidecar = self._sidecar(model_id)
        if sidecar is None or "/" not in model_id:
            raise NotFoundError(f"Unknown model: {model_id}", {"kind": "model", "id": model_id, "hint": "mizuoto model list"})
        engine = model_id.split("/", 1)[0]
        self._registry.manifest(engine)
        return ModelInfo(
            id=model_id,
            engine=engine,
            variant=str(sidecar.get("variant", "")),
            title=str(sidecar.get("title") or model_id),
            description=str(sidecar.get("description", "")),
            state=self._state(model_id, sidecar),
            path=str(self.folder(model_id)),
            source="user",
            size_mb=sidecar.get("size_mb"),
            vram_mb=sidecar.get("vram_mb"),
            languages=sidecar.get("languages"),
            capabilities=self._capabilities(engine, str(sidecar.get("variant", ""))),
        )

    def _capabilities(self, engine: str, variant: str) -> list[str]:
        variant_info = self._registry.manifest(engine).variants.get(variant)
        return list(variant_info.capabilities) if variant_info is not None else []

    def list_models(self, engine: str | None = None, installed: bool | None = None) -> list[ModelInfo]:
        out = [self.get(e.id) for _, e in self._registry.catalog(engine)]
        known = {m.id for m in out}
        root = self._settings.models_dir
        for engine_dir in sorted(root.iterdir()) if root.is_dir() else []:
            if not engine_dir.is_dir() or (engine and engine_dir.name != engine) or engine_dir.name not in self._registry.ids():
                continue
            for folder in sorted(engine_dir.iterdir()):
                model_id = f"{engine_dir.name}/{folder.name}"
                if model_id not in known and (folder / MODEL_FILE).is_file():
                    out.append(self.get(model_id))
        if installed is not None:
            out = [m for m in out if (m.state in ("builtin", "installed")) == installed]
        return out

    # -- shared assets ----------------------------------------------------------------

    def _asset_entry(self, asset_id: str) -> AssetEntry:
        engine = asset_id.split("/", 1)[0]
        for a in self._registry.manifest(engine).assets if engine in self._registry.ids() else []:
            if a.id == asset_id:
                return a
        raise NotFoundError(f"Unknown asset: {asset_id}", {"kind": "asset", "id": asset_id, "hint": "mizuoto asset list"})

    def asset_folder(self, asset_id: str) -> Path:
        engine, _, name = asset_id.partition("/")
        return self._settings.assets_dir / engine / name

    def asset(self, asset_id: str) -> AssetInfo:
        entry = self._asset_entry(asset_id)
        engine = asset_id.split("/", 1)[0]
        try:
            sidecar = json.loads((self.asset_folder(asset_id) / ASSET_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sidecar = None
        with self._lock:
            busy = asset_id in self._downloading
        state: ModelState = "downloading" if busy else self._files_state(self.asset_folder(asset_id), sidecar)
        used_by = [m.id for _, m in self._registry.catalog(engine) if asset_id in m.assets]
        return AssetInfo(
            id=asset_id,
            engine=engine,
            title=entry.title,
            description=entry.description,
            state=state,
            path=str(self.asset_folder(asset_id)) if state == "installed" else None,
            size_mb=self._size_mb(entry),
            used_by=used_by,
        )

    def list_assets(self, engine: str | None = None) -> list[AssetInfo]:
        manifests = [self._registry.manifest(e) for e in self._registry.ids()]
        return [self.asset(a.id) for m in manifests if engine in (None, m.id) for a in m.assets]

    def asset_paths(self, model_id: str) -> dict[str, str]:
        """The folders of the assets a model needs, by asset id; ``AssetMissingError`` names the missing ones."""
        found = self._registry.catalog_entry(model_id)
        if found is not None:
            needed, title = found[1].assets, found[1].title
        else:
            # A user model (a fine-tune, an import) needs what the catalogue's models of its variant need.
            sidecar = self._sidecar(model_id) or {}
            engine, variant = model_id.split("/", 1)[0], str(sidecar.get("variant", ""))
            siblings = [e for _, e in self._registry.catalog(engine) if e.variant == variant] if engine in self._registry.ids() else []
            needed, title = (siblings[0].assets if siblings else []), str(sidecar.get("title") or model_id)
        if not needed:
            return {}
        infos = [self.asset(a) for a in needed]
        missing = [a for a in infos if a.state != "installed"]
        if missing:
            raise AssetMissingError(
                f"{title} needs {', '.join(a.title for a in missing)}",
                [a.id for a in missing],
                {"hint": f"mizuoto model download {model_id}"},
            )
        return {a.id: str(a.path) for a in infos}

    def load_options(self, model_id: str) -> dict[str, Any]:
        """What an engine needs to load the model besides its folder: its shared assets' folders."""
        assets = self.asset_paths(model_id)
        return {"assets": assets} if assets else {}

    def download_assets(self, asset_ids: list[str], *, wait: bool = False, origin: str | None = None) -> Job:
        """Queue an ``asset_download`` job."""
        if self._jobs is None:
            raise ValidationError("Downloads need the job service")
        for asset_id in asset_ids:
            self._registry.check_licence(asset_id.split("/", 1)[0])
            self.files_of(self._asset_entry(asset_id))
        request = {"assets": list(dict.fromkeys(asset_ids))}
        return self._jobs.run_now("asset_download", request, origin) if wait else self._jobs.submit("asset_download", request, origin)

    def _asset_job(self, request: dict[str, Any]) -> JobSpec:
        ids: list[str] = list(request["assets"])
        title = f"Download {ids[0]}" + (f" and {len(ids) - 1} more" if len(ids) > 1 else "")
        return JobSpec(title=title, run=lambda ctx: self._run_download([], ctx, ids), resources=frozenset({"io"}), steps=[JobStep(id=a, title=a) for a in ids])

    def _download_asset(self, client: Any, asset_id: str, settings: Any, headers: dict[str, str] | None, ctx: JobContext) -> dict[str, Any]:
        entry = self._asset_entry(asset_id)
        folder = self.asset_folder(asset_id)
        got = self._fetch_parts(client, folder, entry.title, self.files_of(entry), settings, headers, ctx)
        sidecar = {"id": asset_id, "title": entry.title, "revisions": got["revisions"], "files": got["files"], "installed_at": now_iso()}
        tmp = folder / (ASSET_FILE + ".tmp")
        tmp.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, folder / ASSET_FILE)
        return {"files": len(got["files"]), "downloaded": got["downloaded"], "bytes": got["bytes"]}

    def delete_asset(self, asset_id: str) -> None:
        self._asset_entry(asset_id)
        folder = self.asset_folder(asset_id)
        if folder.is_dir():
            shutil.rmtree(folder)
        self._changed(asset_id)

    def revision(self, model_id: str) -> str:
        """What identifies an installed model's files: its pinned revisions, else when it was installed."""
        sidecar = self._sidecar(model_id) or {}
        revisions = sidecar.get("revisions")
        return json.dumps(revisions, sort_keys=True) if revisions else str(sidecar.get("installed_at", ""))

    def require(self, model_id: str) -> ModelInfo:
        """The model, which must be usable now; ``ModelMissingError`` names what to download."""
        info = self.get(model_id)
        if info.remote is not None and info.state != "installed":
            if self.remote is not None:
                self.remote.info(info.remote)  # unreachable or refused says so
            raise ModelMissingError(
                f"{info.title} is not installed on {info.remote}",
                {"models": [{"id": info.id, "size_mb": info.size_mb, "state": info.state}], "host": info.remote, "hint": f"on {info.remote}: mizuoto model download {info.id}"},
            )
        if info.state not in ("builtin", "installed"):
            reason = {"corrupt": "is damaged", "partial": "is partly downloaded", "downloading": "is still downloading"}.get(info.state, "is not downloaded")
            raise ModelMissingError(
                f"{info.title} {reason}",
                {"models": [{"id": info.id, "size_mb": info.size_mb, "state": info.state}], "hint": f"mizuoto model download {info.id}"},
            )
        return info

    # -- downloads ------------------------------------------------------------------

    def download(self, model_ids: list[str], *, wait: bool = False, origin: str | None = None) -> Job:
        """Queue a ``model_download`` job for catalogue models (``wait``: run it here and return it finished)."""
        if self._jobs is None:
            raise ValidationError("Downloads need the job service")
        if not model_ids:
            raise ValidationError("Name at least one model")
        for model_id in model_ids:
            found = self._registry.catalog_entry(model_id)
            if found is None:
                raise NotFoundError(f"{model_id} is not in the catalogue", {"kind": "model", "id": model_id, "hint": "mizuoto model list"})
            if found[1].builtin:
                raise ValidationError(f"{model_id} is built in and needs no download")
            self._registry.check_licence(found[0].id)  # a gated licence covers its weights too
            self.files_of(found[1])
        request = {"models": list(dict.fromkeys(model_ids))}
        return self._jobs.run_now("model_download", request, origin) if wait else self._jobs.submit("model_download", request, origin)

    def _download_job(self, request: dict[str, Any]) -> JobSpec:
        ids: list[str] = list(request["models"])
        title = f"Download {ids[0]}" + (f" and {len(ids) - 1} more" if len(ids) > 1 else "")
        steps = [JobStep(id=m, title=m) for m in ids]
        return JobSpec(title=title, run=lambda ctx: self._run_download(ids, ctx), resources=frozenset({"io"}), steps=steps)

    def _client(self) -> Any:
        import httpx

        kwargs: dict[str, Any] = {"timeout": httpx.Timeout(30.0, read=120.0), "follow_redirects": True}
        if self.http_transport is not None:
            kwargs["transport"] = self.http_transport
        return httpx.Client(**kwargs)

    def _run_download(self, ids: list[str], ctx: JobContext, asset_ids: list[str] | None = None) -> dict[str, Any]:
        """Download models and assets: first the assets (named, and those the models need), then the models."""
        settings = self._settings.settings.downloads
        token = os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        plans: list[tuple[str, ModelEntry, list[tuple[ModelFiles, FilesManifest]]]] = []
        assets = list(asset_ids or [])
        for model_id in ids:
            found = self._registry.catalog_entry(model_id)
            assert found is not None
            plans.append((model_id, found[1], self.files_of(found[1])))
            assets += [a for a in found[1].assets if a not in assets]
        if ids:
            ctx.add_steps([JobStep(id=a, title=a) for a in assets])
        results: dict[str, Any] = {}
        asset_results: dict[str, Any] = {}
        with self._client() as client:
            for asset_id in assets:
                if self.asset(asset_id).state == "installed":
                    ctx.step(asset_id, "skipped")
                    asset_results[asset_id] = "installed"
                    continue
                with self._busy(asset_id, ctx):
                    asset_results[asset_id] = self._download_asset(client, asset_id, settings, headers, ctx)
            for model_id, entry, parts in plans:
                if self._sidecar(model_id) is not None and self._state(model_id, self._sidecar(model_id)) == "installed":
                    ctx.step(model_id, "skipped")
                    results[model_id] = "installed"
                    continue
                with self._busy(model_id, ctx):
                    results[model_id] = self._download_model(client, model_id, entry, parts, settings, headers, ctx)
        return {"models": results, "assets": asset_results}

    @contextmanager
    def _busy(self, item_id: str, ctx: JobContext) -> Generator[None, None, None]:
        """One model or asset of a download job: its step, and its ``downloading`` state meanwhile."""
        ctx.step(item_id, "running", 0.0)
        with self._lock:
            self._downloading.add(item_id)
        self._changed(item_id)
        try:
            yield
            ctx.step(item_id, "done")
        except BaseException:
            ctx.step(item_id, "cancelled" if ctx.cancelled else "failed")
            raise
        finally:
            with self._lock:
                self._downloading.discard(item_id)
            self._changed(item_id)

    def _download_model(
        self,
        client: Any,
        model_id: str,
        entry: ModelEntry,
        parts: list[tuple[ModelFiles, FilesManifest]],
        settings: Any,
        headers: dict[str, str] | None,
        ctx: JobContext,
    ) -> dict[str, Any]:
        folder = self.folder(model_id)
        got = self._fetch_parts(client, folder, entry.title, parts, settings, headers, ctx)
        sidecar = {
            "id": model_id,
            "engine": model_id.split("/", 1)[0],
            "variant": entry.variant,
            "title": entry.title,
            "source": "catalog",
            "revisions": got["revisions"],
            "downloaded_from": got["sources"],
            "files": got["files"],
            "installed_at": now_iso(),
        }
        self._write_sidecar(model_id, sidecar)
        return {"files": len(got["files"]), "downloaded": got["downloaded"], "bytes": got["bytes"], "sources": got["sources"]}

    def _fetch_parts(
        self,
        client: Any,
        folder: Path,
        title: str,
        parts: list[tuple[ModelFiles, FilesManifest]],
        settings: Any,
        headers: dict[str, str] | None,
        ctx: JobContext,
    ) -> dict[str, Any]:
        """Fetch the missing files of ``parts`` into ``folder`` (resumed, verified) and unpack the archives
        of ``unpack`` parts; what is there afterwards, for the sidecar."""
        folder.mkdir(parents=True, exist_ok=True)
        todo: list[tuple[list[Any], RemoteFile, Path]] = []
        listed: list[dict[str, Any]] = []
        revisions: dict[str, str] = {}
        archives: list[tuple[RemoteFile, Path, Path]] = []
        """(file, where it is downloaded, where it is unpacked)."""
        for files, fm in parts:
            srcs = sources(settings, files, fm.revision, self._mirror_prefix(folder))
            revisions[fm.repo] = fm.revision
            base = folder / files.into if files.into else folder
            for f in fm.files:
                rel = f.path[len(files.subdir.rstrip("/")) + 1 :] if files.subdir else f.path
                if files.unpack:
                    unpacked = self._unpacked(folder, f)
                    if unpacked is not None:
                        listed += unpacked
                        continue
                    dest = folder / UNPACK_DIR / rel
                    archives.append((f, dest, base))
                else:
                    dest = base / rel
                    listed.append({"path": dest.relative_to(folder).as_posix(), "size": f.size, "sha256": f.sha256})
                if dest.is_file() and dest.stat().st_size == f.size:
                    continue
                todo.append((srcs, f, dest))
        total = sum(f.size for _, f, _ in todo)
        ctx.message(f"Downloading {title}: {len(todo)} file(s)")
        used: dict[str, int] = {}

        def report(done: int, all_bytes: int) -> None:
            ctx.transfer(done, all_bytes)
            ctx.progress(done / all_bytes if all_bytes else None)

        progress = Progress(total=total, on_change=report)
        failed = threading.Event()
        stop = AnyEvent(ctx.cancel_event, failed)

        def one(item: tuple[list[Any], RemoteFile, Path]) -> None:
            srcs, f, dest = item
            if stop.is_set():
                return
            try:
                name = fetch(client, srcs, f, dest, progress, stop, verify=settings.verify_checksums, headers=headers)
            except BaseException:
                failed.set()
                raise
            used[name] = used.get(name, 0) + 1
            ctx.log(f"{f.path} ({f.size / 1e6:.1f} MB) from {name}")

        with ThreadPoolExecutor(max_workers=settings.concurrency, thread_name_prefix="download") as pool:
            futures = [pool.submit(one, item) for item in sorted(todo, key=lambda t: -t[1].size)]
            errors = [e for e in (f.exception() for f in futures) if e is not None]
        if ctx.cancelled:
            raise JobCancelled()
        real = [e for e in errors if not isinstance(e, Cancelled)]
        if real:
            raise real[0]
        for f, archive, base in archives:
            ctx.message(f"Unpacking {archive.name}")
            listed += self._unpack(folder, f, archive, base)
        if (folder / UNPACK_DIR).is_dir() and not any((folder / UNPACK_DIR).rglob("*.*")):
            shutil.rmtree(folder / UNPACK_DIR, ignore_errors=True)
        return {"files": listed, "revisions": revisions, "sources": used, "downloaded": len(todo), "bytes": total}

    def _mirror_prefix(self, folder: Path) -> str | None:
        """Where a model's or an asset's folder lies in Mizuoto's mirror: its path under the models folder."""
        try:
            return folder.resolve().relative_to(self._settings.models_dir.resolve()).as_posix()
        except ValueError:
            return None

    @staticmethod
    def _marker(folder: Path, f: RemoteFile) -> Path:
        return folder / UNPACK_DIR / f"{(f.sha256 or f.path.replace('/', '_'))[:32]}.json"

    def _unpacked(self, folder: Path, f: RemoteFile) -> list[dict[str, Any]] | None:
        """The files an archive was unpacked into earlier, when they are all still there."""
        try:
            entries = json.loads(self._marker(folder, f).read_text(encoding="utf-8"))["files"]
        except (OSError, ValueError, KeyError):
            return None
        for e in entries:
            path = folder / e["path"]
            if not path.is_file() or path.stat().st_size != e["size"]:
                return None
        return entries

    def _unpack(self, folder: Path, f: RemoteFile, archive: Path, base: Path) -> list[dict[str, Any]]:
        entries = [{"path": p.relative_to(folder.resolve()).as_posix(), "size": p.stat().st_size, "sha256": sha256_of(p)} for p in unpack(archive, base)]
        marker = self._marker(folder, f)
        marker.write_text(json.dumps({"archive": f.path, "files": entries}, indent=1), encoding="utf-8")
        archive.unlink(missing_ok=True)
        return entries

    def _write_sidecar(self, model_id: str, data: dict[str, Any]) -> None:
        path = self.folder(model_id) / MODEL_FILE
        tmp = path.with_name(MODEL_FILE + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    # -- verification, removal --------------------------------------------------------

    def verify(self, model_id: str) -> VerifyResult:
        """Hash every file of an installed model against what it was installed with."""
        info = self.get(model_id)
        if info.state == "builtin":
            return VerifyResult(model=model_id, ok=True, checked=0)
        sidecar = self._sidecar(model_id)
        if sidecar is None:
            raise ModelMissingError(f"{info.title} is not installed", {"models": [{"id": model_id}], "hint": f"mizuoto model download {model_id}"})
        folder = self.folder(model_id)
        bad: list[str] = []
        checked = 0
        for f in sidecar.get("files") or []:
            path = folder / f["path"]
            if not path.is_file() or path.stat().st_size != f["size"]:
                bad.append(f["path"])
                continue
            if f.get("sha256"):
                checked += 1
                if sha256_of(path) != f["sha256"]:
                    bad.append(f["path"])
        sidecar["verified"] = {"ok": not bad, "at": now_iso(), "bad": bad}
        self._write_sidecar(model_id, sidecar)
        self._changed(model_id)
        return VerifyResult(model=model_id, ok=not bad, checked=checked, bad=bad)

    def delete(self, model_id: str) -> None:
        """Remove a downloaded or imported model's folder (a catalogue model can be downloaded again)."""
        info = self.get(model_id)
        if info.state == "builtin":
            raise ValidationError(f"{model_id} is built in")
        with self._lock:
            if model_id in self._downloading:
                raise ConflictError(f"{model_id} is downloading", {"model": model_id})
        folder = self.folder(model_id)
        if not folder.exists():
            raise NotFoundError(f"{model_id} is not installed", {"kind": "model", "id": model_id})
        if self.before_remove is not None:
            self.before_remove(model_id)
        shutil.rmtree(folder)
        self._changed(model_id)

    # -- imports ----------------------------------------------------------------------

    def identify(self, path: Path) -> Detected:
        """What the model in ``path`` is, and which catalogue model it matches (by its files' sizes)."""
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise NotFoundError(f"No such folder: {path}", {"kind": "folder", "id": str(path)})
        found = detect(path)
        if found is None:
            raise ValidationError(f"{path} is not a model Mizuoto recognises", {"path": str(path), "engines": self._registry.ids()})
        engine, variant, title = found
        catalog_id = None
        for _, entry in self._registry.catalog(engine):
            if entry.variant == variant and not entry.builtin and self._matches(path, entry):
                catalog_id, title = entry.id, entry.title
                break
        return Detected(engine=engine, variant=variant, title=title, catalog_id=catalog_id)

    def _matches(self, path: Path, entry: ModelEntry) -> bool:
        try:
            parts = self.files_of(entry)
        except (NotFoundError, ValidationError):
            return False
        for files, fm in parts:
            base = path / files.into if files.into else path
            for f in fm.files:
                rel = f.path[len(files.subdir.rstrip("/")) + 1 :] if files.subdir else f.path
                target = base / rel
                if not target.is_file() or target.stat().st_size != f.size:
                    return False
        return True

    def speakers(self, model_id: str) -> list[str]:
        """A model's preset speakers: its own (a fine-tune's, in ``model.json``), else its variant's."""
        info = self.get(model_id)
        own = (self._sidecar(model_id) or {}).get("speakers") if info.state != "builtin" else None
        if isinstance(own, list) and own:
            return [str(s) for s in own]
        return list(self._registry.variant(info.engine, info.variant).speakers)

    def register_user_model(self, engine: str, variant: str, folder: Path, name: str, title: str, move: bool = True, extra: dict[str, Any] | None = None) -> ModelInfo:
        """A model a fine-tune made: its folder becomes ``<engine>/<name>`` (a name taken gets ``-2``, ``-3``…)."""
        self._registry.variant(engine, variant)
        base = slugify(name)
        model_id, n = f"{engine}/{base}", 2
        while self._registry.catalog_entry(model_id) is not None or self.folder(model_id).exists():
            model_id, n = f"{engine}/{base}-{n}", n + 1
        target = self.folder(model_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(folder), str(target))
        else:
            shutil.copytree(folder, target)
        paths = [p for p in sorted(target.rglob("*")) if p.is_file() and p.name != MODEL_FILE]
        files = [{"path": p.relative_to(target).as_posix(), "size": p.stat().st_size, "sha256": None} for p in paths]
        sidecar = {
            "id": model_id,
            "engine": engine,
            "variant": variant,
            "title": title,
            "source": "trained",
            "files": files,
            "size_mb": round(sum(p.stat().st_size for p in paths) / (1024 * 1024), 1),
            "installed_at": now_iso(),
            **(extra or {}),
        }
        self._write_sidecar(model_id, sidecar)
        self._changed(model_id)
        return self.get(model_id)

    def import_model(self, path: Path, name: str | None = None, move: bool = False, as_catalog: bool = True) -> ModelInfo:
        """Copy (or move) a model folder into Mizuoto's models folder.

        A folder whose files match a catalogue model becomes that model (its files are then
        verified like a download); any other recognised model becomes a user model
        ``<engine>/<name>``.
        """
        source = path.expanduser().resolve()
        detected = self.identify(source)
        if detected.catalog_id and as_catalog and name is None:
            model_id = detected.catalog_id
        else:
            model_id = f"{detected.engine}/{slugify(name or source.name)}"
            if self._registry.catalog_entry(model_id) is not None:
                raise ConflictError(f"{model_id} is a catalogue model's name; choose another", {"model": model_id})
        target = self.folder(model_id)
        if (target / MODEL_FILE).exists():
            raise ConflictError(f"{model_id} is already installed", {"model": model_id, "hint": f"mizuoto model delete {model_id}"})
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(source), str(target))
        else:
            shutil.copytree(source, target)
        files = [{"path": p.relative_to(target).as_posix(), "size": p.stat().st_size, "sha256": None} for p in sorted(target.rglob("*")) if p.is_file() and p.name != MODEL_FILE]
        if model_id == detected.catalog_id:
            found = self._registry.catalog_entry(model_id)
            assert found is not None
            pinned = {(f"{fs.into}/" if fs.into else "") + f.path: f.sha256 for fs, fm in self.files_of(found[1]) for f in fm.files}
            files = [{**f, "sha256": pinned.get(f["path"])} for f in files if f["path"] in pinned]
        sidecar = {
            "id": model_id,
            "engine": detected.engine,
            "variant": detected.variant,
            "title": detected.title if model_id == detected.catalog_id else f"{detected.title} ({target.name})",
            "source": "catalog" if model_id == detected.catalog_id else "import",
            "imported_from": str(source),
            "files": files,
            "size_mb": round(sum(int(f["size"] or 0) for f in files) / (1024 * 1024), 1),
            "installed_at": now_iso(),
        }
        self._write_sidecar(model_id, sidecar)
        if model_id == detected.catalog_id:
            result = self.verify(model_id)
            if not result.ok:
                raise ValidationError(f"{source} differs from {model_id}: {', '.join(result.bad[:5])}", {"bad": result.bad})
        self._changed(model_id)
        return self.get(model_id)
