import importlib.util

import pytest

from mizuoto.core.engines import languages
from mizuoto.core.engines.models import COMMON_PARAMS
from mizuoto.core.engines.params import check_options, check_params
from mizuoto.core.engines.registry import MANIFESTS_DIR, load_manifest
from mizuoto.core.errors import NotFoundError, ParameterError, ValidationError
from mizuoto.core.tts.models import CommonParams


def test_every_manifest_loads_and_is_consistent():
    paths = sorted(MANIFESTS_DIR.glob("*.toml"))
    assert paths
    for path in paths:
        m = load_manifest(path)
        assert m.id == path.stem, path.name
        assert set(m.params) <= set(COMMON_PARAMS), path.name
        for model in m.models:
            assert model.builtin or model.files is not None, model.id


def test_common_params_match_the_request_model():
    assert set(CommonParams.model_fields) == set(COMMON_PARAMS)


def test_bad_manifests_are_refused(tmp_path):
    (tmp_path / "x.toml").write_text(
        'id = "x"\ntitle = "X"\n[licence]\nid = "mit"\nname = "MIT"\n[variants.v]\nsample_rate = 1\n[[models]]\nid = "y/m"\nvariant = "v"\ntitle = "M"\nbuiltin = true\n'
    )
    with pytest.raises(ValidationError):
        load_manifest(tmp_path / "x.toml")


def test_check_params(services):
    m = services.engines.manifest("dummy")
    accepted, warnings = check_params(m, "default", {"speed": 1, "seed": 3.0, "temperature": None}, strict=True)
    assert accepted == {"speed": 1.0, "seed": 3} and warnings == []
    with pytest.raises(ParameterError) as e:
        check_params(m, "default", {"seed": 1.5}, strict=True)
    assert e.value.detail["fields"][0]["reason"] == "not_an_integer"
    with pytest.raises(ParameterError):
        check_params(m, "default", {"speed": True}, strict=True)
    accepted, warnings = check_params(m, "default", {"num_steps": 8}, strict=False)
    assert accepted == {} and warnings


def test_check_options(services):
    m = services.engines.manifest("dummy")
    assert check_options(m, "default", {"brightness": 0.5, "fail": "none"}) == {"brightness": 0.5, "fail": "none"}
    with pytest.raises(ParameterError):
        check_options(m, "default", {"brightness": 2})


def test_registry_lookups(services):
    reg = services.engines
    assert reg.ids() == ["asr", "cosyvoice", "dummy", "f5tts", "fishspeech", "gptsovits", "indextts", "omnivoice", "qwen3tts", "voxcpm"]
    # Every engine runs in this process: available where torch (and a gated engine's distribution) is installed.
    torch = importlib.util.find_spec("torch") is not None
    asr = reg.status("asr")
    assert reg.manifest("asr").internal and asr.available == torch and asr.missing == ([] if torch else ["torch"])
    with pytest.raises(NotFoundError):
        reg.manifest("nope")
    assert reg.status("dummy").available and reg.status("dummy").missing == []
    assert "preset" in reg.voice_modes(reg.variant("dummy", "default"))
    # The UI's form follows the spec, so every rule the synthesis checks a style against is in it.
    spec = reg.param_specs("cosyvoice/v2", "cosyvoice", "v2")
    assert spec.instruct_requires == ["clone", "clone_no_transcript"] and reg.param_specs("dummy/default", "dummy", "default").instruct_requires == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [("ZH", "zh"), ("zh_CN", "zh"), ("English", "en"), ("cantonese", "yue"), (" ", None), (None, None), ("pt-BR", "pt-br")],
)
def test_language_normalize(code, expected):
    assert languages.normalize(code) == expected


def test_language_support_and_detection():
    assert languages.supports(["*"], "xx")
    assert languages.supports(["zh-*"], "zh-sichuan") and languages.supports(["zh-*"], "zh")
    assert not languages.supports(["en"], "ja")
    assert languages.supports(["en"], "auto")
    assert languages.detect("今天天气不错") == "zh"
    assert languages.detect("こんにちは世界") == "ja"
    assert languages.detect("안녕하세요") == "ko"
    assert languages.detect("Hello") == "en"
    assert languages.detect("1234") is None


def check_options_mirror(engine_id: str) -> None:
    """``engine_id``'s ``OPTION_SCHEMA`` mirrors its manifest's options (engine modules import no torch). The gated
    packages' tests call it for their engine: their manifests ship in mizuoto, their code on its own."""
    from mizuoto.runtime.engine import get_engine

    m = load_manifest(MANIFESTS_DIR / f"{engine_id}.toml")
    schema = get_engine(m.id).OPTION_SCHEMA
    assert set(schema) == set(m.options), f"{m.id}: manifest options {sorted(m.options)} != OPTION_SCHEMA {sorted(schema)}"
    for name, spec in schema.items():
        o = m.options[name]
        assert (spec.type, spec.default, spec.min, spec.max) == (o.type, o.default, o.min, o.max), f"{m.id}.{name}"
        assert (list(spec.choices) if spec.choices else None) == o.choices, f"{m.id}.{name}"


def check_devices_mirror(engine_id: str) -> None:
    """``engine_id``'s manifest ``devices`` are the kinds its implementation declares (``Engine.DEVICES``); a trainer's
    ``DEVICES`` are among them."""
    from mizuoto.runtime.engine import get_engine
    from mizuoto.training.trainer import trainer_class, trainer_for

    m = load_manifest(MANIFESTS_DIR / f"{engine_id}.toml")
    engine = get_engine(m.id)
    assert set(engine.DEVICES) == set(m.devices), f"{m.id}: manifest devices {m.devices} != DEVICES {engine.DEVICES}"
    trainer = trainer_for(m.id)
    own = trainer_class(trainer).DEVICES if trainer else None
    assert own is None or set(own) <= set(m.devices), f"{m.id}: trainer DEVICES {own}"


def _builtin_manifests() -> list[str]:
    from mizuoto.core.inference.service import DISTRIBUTIONS

    return [p.stem for p in sorted(MANIFESTS_DIR.glob("*.toml")) if p.stem not in DISTRIBUTIONS]


def test_every_manifest_has_an_implementation():
    """A built-in engine is in BUILTIN; a licence-gated one comes in its distribution (its packages' tests check it)."""
    from mizuoto.core.inference.service import DISTRIBUTIONS
    from mizuoto.runtime.engine import BUILTIN

    for path in sorted(MANIFESTS_DIR.glob("*.toml")):
        assert path.stem in BUILTIN or path.stem in DISTRIBUTIONS, f"{path.stem} has no implementation"


@pytest.mark.parametrize("engine_id", _builtin_manifests())
def test_manifest_options_match_the_engines(engine_id):
    check_options_mirror(engine_id)


@pytest.mark.parametrize("engine_id", _builtin_manifests())
def test_manifest_devices_match_the_engines(engine_id):
    check_devices_mirror(engine_id)


def test_manifest_devices_are_known_kinds():
    """Every manifest's ``devices`` (and its training methods') are known kinds, the CPU among them; every engine but the
    dummy runs on every kind of device the runtime drives."""
    from mizuoto.runtime.engine import DEVICE_KINDS

    for path in sorted(MANIFESTS_DIR.glob("*.toml")):
        m = load_manifest(path)
        assert set(m.devices) <= set(DEVICE_KINDS) and "cpu" in m.devices, f"{m.id}: {m.devices}"
        for method_id, method in m.training.items():
            assert method.devices is None or set(method.devices) <= set(m.devices), f"{m.id}.{method_id}: {method.devices}"
    assert all(set(load_manifest(p).devices) == set(DEVICE_KINDS) for p in MANIFESTS_DIR.glob("*.toml") if p.stem != "dummy")


def test_catalogue_models_have_files_manifests():
    import json

    from mizuoto.core.engines.registry import MANIFESTS_DIR, load_manifest

    for path in sorted(MANIFESTS_DIR.glob("*.toml")):
        m = load_manifest(path)
        for model in m.models:
            for files in [f for f in [model.files, *model.extra] if f is not None]:
                assert files.manifest, f"{model.id} has no files manifest"
                data = json.loads((MANIFESTS_DIR / "files" / files.manifest).read_text())
                assert data["repo"] == files.hf and len(data["revision"]) == 40 and data["files"]
                assert all(f["size"] >= 0 and len(f["sha256"]) == 64 for f in data["files"])


def check_gated_distribution(engine_id: str) -> None:
    """A licence-gated engine's distribution, released on its own: its version in ``mizuoto_<id>/version.py`` (the
    file whose change releases it), and a ``mizuoto>=…,<…`` range that holds this Mizuoto's version, so a Mizuoto
    release past the range needs the package released too. Its packages' tests call this."""
    import re
    from pathlib import Path

    from mizuoto.core.inference.service import DISTRIBUTIONS, satisfies
    from mizuoto.version import VERSION

    package = Path(__file__).resolve().parents[2] / "packages" / "native" / engine_id
    pyproject = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert f'name = "{DISTRIBUTIONS[engine_id]}"' in pyproject
    assert f'version = {{ attr = "mizuoto_{engine_id}.version.VERSION" }}' in pyproject
    version = re.search(r'^VERSION = "([^"]+)"$', (package / f"mizuoto_{engine_id}" / "version.py").read_text(encoding="utf-8"), re.M)
    assert version and re.fullmatch(r"\d+\.\d+\.\d+", version.group(1)), f"{engine_id}: no VERSION in version.py"
    found = re.search(r'"mizuoto(>=[^",]+,<[^",]+)"', pyproject)
    assert found, f"{engine_id} does not depend on mizuoto>=…,<…"
    assert satisfies(VERSION, found.group(1)), f"{DISTRIBUTIONS[engine_id]} works with mizuoto{found.group(1)}, not this Mizuoto ({VERSION}): raise its range"


def test_version_ranges():
    from mizuoto.core.inference.service import satisfies

    assert satisfies("0.2.0", ">=0.2.0,<0.3") and satisfies("0.2.9", "<0.3,>=0.2.0")
    assert not satisfies("0.3.0", ">=0.2.0,<0.3") and not satisfies("0.1.9", ">=0.2.0,<0.3")
    assert satisfies("0.3", "==0.3.0") and satisfies("1.4.2", "~=1.4") and not satisfies("2.0", "~=1.4")
    assert satisfies("0.2.1rc1", ">=0.2.1") and satisfies("0.2.0", "")


def test_doctor_reports_a_gated_engine_that_does_not_go_with_this_mizuoto(services, monkeypatch):
    from mizuoto.core.inference import service

    inference = services.runtime.inference
    monkeypatch.setattr(inference, "implemented", lambda engine_id: True)
    monkeypatch.setattr(service, "distribution_version", lambda engine_id: "9.0.0")
    monkeypatch.setattr(service, "mizuoto_range", lambda engine_id: ">=9.0,<10")
    found = [f for f in services.runtime.doctor("indextts").findings if f.check == "engine"]
    assert found[0].level == "error" and "mizuoto-indextts 9.0.0 works with Mizuoto >=9.0,<10" in found[0].message
    assert found[0].fix == "pip install --upgrade mizuoto mizuoto-indextts"


def test_licence_training_rule(services):
    from mizuoto.core.engines.licences import training_refusal

    assert training_refusal(services.engines, ["fish-audio-research"], "gptsovits")
    assert training_refusal(services.engines, ["fish-audio-research"], "fishspeech") is None
    assert training_refusal(services.engines, ["bilibili-model-use"], "f5tts")
    assert training_refusal(services.engines, ["cc-by-nc-4.0"], "gptsovits") is None
    assert training_refusal(services.engines, [], "gptsovits") is None
    gated = {e: services.engines.manifest(e).licence for e in ("fishspeech", "indextts")}
    assert all(lic.gated and lic.file for lic in gated.values())
    text = services.engines.licence_text("fishspeech")
    assert "FISH AUDIO RESEARCH LICENSE" in text.text and text.key == "fish-audio-research@2026-03-07" and not text.accepted


def test_gated_downloads_need_the_licence(services):
    from mizuoto.core.errors import LicenceError

    with pytest.raises(LicenceError):
        services.models.download(["indextts/v2"])
    with pytest.raises(LicenceError):
        services.models.download_assets(["indextts/campplus"])


def test_load_options_are_replaced_and_checked(services):
    # use_torch_compile is for some variants only; at the engine's level it is allowed.
    services.engines.configure("indextts", {"load_options": {"use_torch_compile": True, "use_qwen_emo": False}})
    assert services.settings.settings.engines["indextts"].load_options == {"use_torch_compile": True, "use_qwen_emo": False}
    services.engines.configure("indextts", {"load_options": {"use_qwen_emo": True}})
    assert services.settings.settings.engines["indextts"].load_options == {"use_qwen_emo": True}
    services.engines.configure("indextts", {"load_options": {}})
    assert services.settings.settings.engines["indextts"].load_options == {}
    with pytest.raises(ParameterError):  # v1's accelerators are gone
        services.engines.configure("cosyvoice", {"load_options": {"load_trt": True}})


def test_parameters_without_effect_are_accepted_but_not_offered():
    """Fish Speech's ``repetition_penalty`` and ``normalize`` reach nothing (upstream's neither): requests that send them
    still validate, and the UI's form leaves them out."""
    from pathlib import Path

    import mizuoto.core.engines as engines_pkg
    from mizuoto.core.engines.params import check_options, check_params, param_fields
    from mizuoto.core.engines.registry import load_manifest

    manifest = load_manifest(Path(engines_pkg.__file__).parent / "manifests" / "fishspeech.toml")
    names = {f.name for f in param_fields(manifest, "s2_pro")}
    assert "temperature" in names and not {"repetition_penalty", "normalize"} & names
    params, _ = check_params(manifest, "s2_pro", {"repetition_penalty": 1.2}, strict=True)
    assert params["repetition_penalty"] == 1.2
    assert check_options(manifest, "s2_pro", {"normalize": False}) == {"normalize": False}
