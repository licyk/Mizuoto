"""Model downloads (sources, resume, verification), states, verify, delete and import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from mizuoto.core.engines.models import ModelEntry, ModelFiles
from mizuoto.core.errors import ConflictError, ModelMissingError, NotFoundError, ValidationError
from mizuoto.core.models import service as model_service
from mizuoto.core.models.download import sources
from mizuoto.core.settings.models import DownloadSettings

WEIGHTS = bytes(range(256)) * 400  # 100 KiB
CONFIG = b'{"hello": 1}'
FILES = {"model.bin": WEIGHTS, "sub/config.json": CONFIG}


class Hub:
    """A fake Hugging Face / ModelScope server, with switches for the failures under test."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.missing_on: set[str] = set()
        self.corrupt_on: set[str] = set()
        self.ignore_range = False
        self.drop_after: int | None = None
        """Drop the connection to ``model.bin`` after this many bytes, once (files download concurrently)."""
        self.mirror = False
        """Serve Mizuoto's mirror (``licyk/mizuoto-model``, ``licyks/mizuoto-model`` on ModelScope: the files under ``dummy/tiny/``)."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        name = request.url.path.split("/resolve/", 1)[1].split("/", 1)[1]
        if request.url.path.startswith(("/licyk/mizuoto-model/", "/models/licyks/mizuoto-model/")):
            if not self.mirror or not name.startswith("dummy/tiny/"):
                return httpx.Response(404)
            name = name.removeprefix("dummy/tiny/")
        if host in self.missing_on or name not in FILES:
            return httpx.Response(404)
        body = FILES[name]
        if host in self.corrupt_on:
            body = b"x" * len(body)
        rng = request.headers.get("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                return httpx.Response(416)
            return httpx.Response(206, content=body[start:], headers={"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
        if self.drop_after is not None and name == "model.bin":
            limit, self.drop_after = self.drop_after, None
            return httpx.Response(200, stream=_Broken(body[:limit]))
        return httpx.Response(200, content=body)


class _Broken(httpx.SyncByteStream):
    """Sends some bytes, then fails like a dropped connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.ReadError("connection reset")


@pytest.fixture
def hub(services, tmp_path, monkeypatch):
    """A catalogue model ``dummy/tiny`` whose files come from the fake hub."""
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "tiny.files.json").write_text(
        json.dumps(
            {
                "repo": "org/tiny",
                "revision": "abc123",
                "files": [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in FILES.items()],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_service, "FILES_DIR", files_dir)
    monkeypatch.setattr("mizuoto.core.models.download.ATTEMPTS", 2)
    monkeypatch.setattr("mizuoto.core.models.download.Cancelled", model_service.Cancelled)
    manifest = services.engines.manifest("dummy")
    entry = ModelEntry(id="dummy/tiny", variant="default", title="Tiny", files=ModelFiles(hf="org/tiny", modelscope="org/tiny", manifest="tiny.files.json"))
    services.engines.add(manifest.model_copy(update={"models": [*manifest.models, entry]}))
    fake = Hub()
    services.models.http_transport = httpx.MockTransport(fake.handler)
    # These tests are about the official sources; the mirror (the default) is tested on its own.
    services.settings.update({"downloads": {"repository": "official"}})
    # Retries wait 1 s, 2 s…; tests should not.
    monkeypatch.setattr("mizuoto.core.models.download.min", lambda a, b: 0.0, raising=False)
    return fake


def test_sources_order_and_urls():
    files = ModelFiles(hf="org/m", modelscope="org/ms", modelscope_revision="v1")
    names = [s.name for s in sources(DownloadSettings(source="modelscope"), files, "rev")]
    assert names == ["modelscope", "huggingface", "hf-mirror"]
    by_name = {s.name: s.url("a/b.bin") for s in sources(DownloadSettings(), files, "rev")}
    assert by_name["huggingface"] == "https://huggingface.co/org/m/resolve/rev/a/b.bin"
    assert by_name["hf-mirror"] == "https://hf-mirror.com/org/m/resolve/rev/a/b.bin"
    assert by_name["modelscope"] == "https://www.modelscope.cn/models/org/ms/resolve/v1/a/b.bin"
    custom = sources(DownloadSettings(source="custom", endpoint="https://hub.example/"), files, "rev")
    assert custom[0].name == "custom" and custom[0].url("x") == "https://hub.example/org/m/resolve/rev/x"
    assert [s.name for s in sources(DownloadSettings(), ModelFiles(hf="org/m"), "r")] == ["huggingface", "hf-mirror"]


def test_download_verify_delete(services, hub):
    assert services.models.get("dummy/tiny").state == "missing"
    with pytest.raises(ModelMissingError):
        services.models.require("dummy/tiny")
    from mizuoto.core.events.models import JobUpdatedEvent

    seen: list[int] = []
    services.events.subscribe(lambda e: seen.append(e.job.transfer.done_bytes) if isinstance(e, JobUpdatedEvent) and e.job.transfer else None)
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    assert seen and max(seen) <= len(WEIGHTS) + len(CONFIG)
    info = services.models.require("dummy/tiny")
    folder = Path(info.path or "")
    assert (folder / "model.bin").read_bytes() == WEIGHTS and (folder / "sub" / "config.json").read_bytes() == CONFIG
    sidecar = json.loads((folder / "model.json").read_text())
    assert sidecar["revisions"] == {"org/tiny": "abc123"} and sidecar["downloaded_from"] == {"huggingface": 2}
    assert all(r.url.host == "huggingface.co" and "/resolve/abc123/" in r.url.path for r in hub.requests)
    assert job.result["models"]["dummy/tiny"]["bytes"] == len(WEIGHTS) + len(CONFIG)
    assert services.models.verify("dummy/tiny").ok
    # A second download fetches nothing.
    again = services.models.download(["dummy/tiny"], wait=True)
    assert again.result["models"]["dummy/tiny"] == "installed"
    # Damage: a changed size makes it corrupt at once; a changed byte shows in verify.
    weights = folder / "model.bin"
    weights.write_bytes(WEIGHTS[:-1])
    assert services.models.get("dummy/tiny").state == "corrupt"
    weights.write_bytes(b"\x07" + WEIGHTS[1:])
    result = services.models.verify("dummy/tiny")
    assert not result.ok and result.bad == ["model.bin"] and services.models.get("dummy/tiny").state == "corrupt"
    services.models.delete("dummy/tiny")
    assert services.models.get("dummy/tiny").state == "missing"
    with pytest.raises(NotFoundError):
        services.models.delete("dummy/tiny")


def test_resume_from_a_part_file(services, hub):
    folder = services.models.folder("dummy/tiny")
    folder.mkdir(parents=True)
    (folder / "model.bin.part").write_bytes(WEIGHTS[:1000])
    assert services.models.get("dummy/tiny").state == "partial"
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    ranged = [r for r in hub.requests if r.url.path.endswith("model.bin")]
    assert ranged[0].headers["Range"] == "bytes=1000-"
    assert (folder / "model.bin").read_bytes() == WEIGHTS and not (folder / "model.bin.part").exists()


def test_server_ignoring_range_starts_again(services, hub):
    hub.ignore_range = True
    folder = services.models.folder("dummy/tiny")
    folder.mkdir(parents=True)
    (folder / "model.bin.part").write_bytes(b"junk" * 100)
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    assert (folder / "model.bin").read_bytes() == WEIGHTS


def test_dropped_connection_resumes(services, hub):
    hub.drop_after = 5000
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    weights = [r for r in hub.requests if r.url.path.endswith("model.bin")]
    assert len(weights) == 2 and weights[1].headers["Range"] == "bytes=5000-"


def test_bad_content_falls_back_to_the_next_source(services, hub):
    hub.corrupt_on = {"huggingface.co"}
    hub.missing_on = {"hf-mirror.com"}
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    sidecar = json.loads((services.models.folder("dummy/tiny") / "model.json").read_text())
    assert sidecar["downloaded_from"] == {"modelscope": 2}


def test_the_mirror_comes_first_and_the_official_repository_fills_in(services, hub):
    services.settings.update({"downloads": {"repository": "mizuoto-model"}})
    hub.mirror = True
    assert services.models.download(["dummy/tiny"], wait=True).state == "completed"
    sidecar = json.loads((services.models.folder("dummy/tiny") / "model.json").read_text())
    assert sidecar["downloaded_from"] == {"mizuoto-model": 2}
    assert {str(r.url) for r in hub.requests} >= {"https://huggingface.co/licyk/mizuoto-model/resolve/main/dummy/tiny/sub/config.json"}
    hub.mirror = False
    hub.requests.clear()
    folder = services.models.folder("dummy/tiny")
    for f in folder.rglob("*"):
        if f.is_file():
            f.unlink()
    assert services.models.download(["dummy/tiny"], wait=True).state == "completed"
    sidecar = json.loads((folder / "model.json").read_text())
    assert sidecar["downloaded_from"] == {"huggingface": 2}
    assert str(hub.requests[0].url).startswith("https://huggingface.co/licyk/mizuoto-model/")


def test_the_mirror_on_modelscope_has_its_own_repository(services, hub):
    services.settings.update({"downloads": {"repository": "mizuoto-model", "source": "modelscope"}})
    hub.mirror = True
    assert services.models.download(["dummy/tiny"], wait=True).state == "completed"
    sidecar = json.loads((services.models.folder("dummy/tiny") / "model.json").read_text())
    assert sidecar["downloaded_from"] == {"mizuoto-model": 2}
    assert "https://www.modelscope.cn/models/licyks/mizuoto-model/resolve/master/dummy/tiny/sub/config.json" in {str(r.url) for r in hub.requests}


def test_mirror_urls_follow_the_source():
    from mizuoto.core.models.download import mirror_source

    files = ModelFiles(hf="org/m", subdir="weights", into="vocoder")
    assert mirror_source(DownloadSettings(), files, "f5tts/v1").url("weights/a.pt") == "https://huggingface.co/licyk/mizuoto-model/resolve/main/f5tts/v1/vocoder/a.pt"
    ms = mirror_source(DownloadSettings(source="modelscope"), files, "_assets/x/y")
    assert ms.url("weights/b.pt") == "https://www.modelscope.cn/models/licyks/mizuoto-model/resolve/master/_assets/x/y/vocoder/b.pt"
    own = mirror_source(DownloadSettings(source="modelscope", mirror_repo="me/hf", mirror_repo_modelscope="me/ms"), files, "a/b")
    assert own.url("weights/b.pt") == "https://www.modelscope.cn/models/me/ms/resolve/master/a/b/vocoder/b.pt"
    hf_mirror = mirror_source(DownloadSettings(source="hf-mirror"), files, "a/b")
    assert hf_mirror.url("weights/b.pt") == "https://hf-mirror.com/licyk/mizuoto-model/resolve/main/a/b/vocoder/b.pt"
    custom = mirror_source(DownloadSettings(source="custom", endpoint="http://lan:8000/", mirror_repo="me/mirror"), ModelFiles(hf="o/m"), "a/b")
    assert custom.url("c.bin") == "http://lan:8000/me/mirror/resolve/main/a/b/c.bin"
    assert DownloadSettings().repository == "mizuoto-model"
    assert [s.name for s in sources(DownloadSettings(repository="mizuoto-model"), ModelFiles(hf="o/m"), "r", "a/b")][0] == "mizuoto-model"
    assert "mizuoto-model" not in [s.name for s in sources(DownloadSettings(repository="mizuoto-model"), ModelFiles(hf="o/m"), "r")]


def test_token_goes_to_hugging_face_only(services, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    hub.missing_on = {"huggingface.co"}
    assert services.models.download(["dummy/tiny"], wait=True).state == "completed"
    sent = {r.url.host: r.headers.get("Authorization") for r in hub.requests}
    assert sent["huggingface.co"] == "Bearer secret" and sent["hf-mirror.com"] is None


def test_download_fails_everywhere(services, hub):
    hub.missing_on = {"huggingface.co", "hf-mirror.com", "www.modelscope.cn"}
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "failed" and job.error is not None and job.error.code == "download_failed"
    assert job.error.detail["file"] in FILES and len(job.error.detail["errors"]) >= 3
    assert services.models.get("dummy/tiny").state in ("missing", "partial")


def test_download_refuses_unknown_and_builtin(services):
    with pytest.raises(NotFoundError):
        services.models.download(["dummy/nope"])
    with pytest.raises(ValidationError):
        services.models.download(["dummy/default"])


def _qwen_folder(root: Path, kind: str = "custom_voice", size: str = "0b6") -> Path:
    folder = root / "Qwen3-TTS-test"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_type": kind, "tts_model_size": size}))
    (folder / "model.safetensors").write_bytes(b"weights")
    return folder


def test_identify_and_import(services, tmp_path):
    folder = _qwen_folder(tmp_path / "elsewhere")
    found = services.models.identify(folder)
    assert (found.engine, found.variant, found.catalog_id) == ("qwen3tts", "custom_voice_lite", None)
    info = services.models.import_model(folder, name="My Qwen")
    assert info.id == "qwen3tts/my-qwen" and info.source == "user" and info.state == "installed" and info.variant == "custom_voice_lite"
    assert folder.exists()  # copied, not moved
    assert "qwen3tts/my-qwen" in [m.id for m in services.models.list_models("qwen3tts", installed=True)]
    with pytest.raises(ConflictError):
        services.models.import_model(folder, name="My Qwen")
    with pytest.raises(ConflictError):
        services.models.import_model(folder, name="1.7b-base")
    with pytest.raises(ValidationError):
        services.models.identify(tmp_path)
    moved = services.models.import_model(folder, name="moved", move=True)
    assert moved.state == "installed" and not folder.exists()


def test_detect_whisper(tmp_path):
    from mizuoto.core.models.detect import detect

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    (tmp_path / "preprocessor_config.json").write_text("{}")
    found = detect(tmp_path)
    assert found is not None and found[:2] == ("asr", "whisper")


@pytest.mark.parametrize(
    ("files", "name", "expected"),
    [
        ({"config.json": json.dumps({"model_type": "omnivoice"})}, "OmniVoice", ("omnivoice", "v0")),
        ({"cosyvoice3.yaml": ""}, "Fun-CosyVoice3-0.5B", ("cosyvoice", "v3")),
        ({"cosyvoice2.yaml": ""}, "CosyVoice2-0.5B", ("cosyvoice", "v2")),
        ({"cosyvoice.yaml": ""}, "CosyVoice-300M", ("cosyvoice", "v1_base")),
        ({"cosyvoice.yaml": "", "spk2info.pt": ""}, "CosyVoice-300M-SFT", ("cosyvoice", "v1_sft")),
        ({"cosyvoice.yaml": "", "spk2info.pt": ""}, "CosyVoice-300M-Instruct", ("cosyvoice", "v1_instruct")),
        ({"model_1250000.safetensors": "", "vocab.txt": ""}, "F5TTS_v1_Base", None),
        ({"config.json": json.dumps({"architecture": "voxcpm2", "lm_config": {}, "dit_config": {}})}, "VoxCPM2", ("voxcpm", "v2")),
        ({"config.json": json.dumps({"architecture": "voxcpm", "lm_config": {}, "dit_config": {}, "audio_vae_config": {"sample_rate": 44100}})}, "VoxCPM1.5", ("voxcpm", "v1_5")),
        ({"config.json": json.dumps({"lm_config": {}, "dit_config": {}})}, "VoxCPM-0.5B", ("voxcpm", "v1")),
    ],
)
def test_detect_engines(tmp_path, files, name, expected):
    from mizuoto.core.models.detect import detect

    folder = tmp_path / name
    folder.mkdir()
    for file, content in files.items():
        (folder / file).write_text(content)
    found = detect(folder)
    assert (found[:2] if found else None) == expected


def _zip(entries: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_assets_unpack_and_gate_their_models(services, hub, tmp_path, monkeypatch):
    from mizuoto.core.engines.models import AssetEntry
    from mizuoto.core.errors import AssetMissingError

    archive = _zip({"G2PWModel/config.py": b"x = 1", "G2PWModel/model.onnx": b"\x00" * 5000})
    monkeypatch.setitem(FILES, "data.zip", archive)
    (model_service.FILES_DIR / "shared.files.json").write_text(
        json.dumps({"repo": "org/shared", "revision": "r1", "files": [{"path": "data.zip", "size": len(archive), "sha256": hashlib.sha256(archive).hexdigest()}]})
    )
    manifest = services.engines.manifest("dummy")
    asset = AssetEntry(id="dummy/shared", title="Shared data", files=ModelFiles(hf="org/shared", manifest="shared.files.json", unpack=True, into="text"))
    tiny = next(m for m in manifest.models if m.id == "dummy/tiny").model_copy(update={"assets": ["dummy/shared"]})
    services.engines.add(manifest.model_copy(update={"models": [*[m for m in manifest.models if m.id != "dummy/tiny"], tiny], "assets": [asset]}))

    assert [a.id for a in services.models.list_assets("dummy")] == ["dummy/shared"]
    assert services.models.asset("dummy/shared").state == "missing" and services.models.asset("dummy/shared").used_by == ["dummy/tiny"]
    job = services.models.download(["dummy/tiny"], wait=True)
    assert job.state == "completed", job.error
    assert set(job.result["assets"]) == {"dummy/shared"} and services.models.get("dummy/tiny").state == "installed"
    folder = services.models.asset_folder("dummy/shared")
    assert (folder / "text" / "G2PWModel" / "config.py").read_bytes() == b"x = 1"
    assert not list((folder / ".unpack").glob("*.zip"))
    assert services.models.asset_paths("dummy/tiny") == {"dummy/shared": str(folder)}
    assert services.models.verify("dummy/tiny").ok
    # Nothing is fetched again; a missing asset makes its model partial until it is downloaded again.
    hub.requests.clear()
    assert services.models.download_assets(["dummy/shared"], wait=True).result["assets"]["dummy/shared"] == "installed" and not hub.requests
    services.models.delete_asset("dummy/shared")
    assert services.models.get("dummy/tiny").state == "partial"
    with pytest.raises(AssetMissingError):
        services.models.asset_paths("dummy/tiny")
    assert services.models.download_assets(["dummy/shared"], wait=True).state == "completed"
    assert services.models.get("dummy/tiny").state == "installed"


def test_unpack_refuses_escaping_members(tmp_path):
    from mizuoto.core.errors import DownloadError
    from mizuoto.core.models.archives import unpack

    bad = tmp_path / "bad.zip"
    bad.write_bytes(_zip({"../evil.txt": b"x"}))
    with pytest.raises(DownloadError):
        unpack(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
