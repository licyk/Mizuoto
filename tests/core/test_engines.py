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


def test_manifest_options_match_the_engines():
    """Each engine's ``OPTION_SCHEMA`` mirrors its manifest's options (engine modules import no torch)."""
    from mizuoto.core.inference.service import DISTRIBUTIONS
    from mizuoto.runtime.engine import engine_ids, get_engine

    implemented = engine_ids()
    for path in sorted(MANIFESTS_DIR.glob("*.toml")):
        m = load_manifest(path)
        if m.id not in implemented:
            assert m.id in DISTRIBUTIONS, f"{m.id} has no implementation"
            continue
        schema = get_engine(m.id).OPTION_SCHEMA
        assert set(schema) == set(m.options), f"{m.id}: manifest options {sorted(m.options)} != OPTION_SCHEMA {sorted(schema)}"
        for name, spec in schema.items():
            o = m.options[name]
            assert (spec.type, spec.default, spec.min, spec.max) == (o.type, o.default, o.min, o.max), f"{m.id}.{name}"
            assert (list(spec.choices) if spec.choices else None) == o.choices, f"{m.id}.{name}"


def test_manifest_devices_match_the_engines():
    """Each manifest's ``devices`` are the kinds its implementation declares (``Engine.DEVICES``), known kinds only;
    a training method's devices and a trainer's ``DEVICES`` are among them."""
    from mizuoto.core.inference.service import DISTRIBUTIONS
    from mizuoto.runtime.engine import DEVICE_KINDS, engine_ids, get_engine
    from mizuoto.training.trainer import trainer_class, trainer_for

    implemented = engine_ids()
    for path in sorted(MANIFESTS_DIR.glob("*.toml")):
        m = load_manifest(path)
        assert set(m.devices) <= set(DEVICE_KINDS) and "cpu" in m.devices, f"{m.id}: {m.devices}"
        for method_id, method in m.training.items():
            assert method.devices is None or set(method.devices) <= set(m.devices), f"{m.id}.{method_id}: {method.devices}"
        if m.id not in implemented:
            assert m.id in DISTRIBUTIONS, f"{m.id} has no implementation"
            continue
        engine = get_engine(m.id)
        assert set(engine.DEVICES) == set(m.devices), f"{m.id}: manifest devices {m.devices} != DEVICES {engine.DEVICES}"
        trainer = trainer_for(m.id)
        own = trainer_class(trainer).DEVICES if trainer else None
        assert own is None or set(own) <= set(m.devices), f"{m.id}: trainer DEVICES {own}"
    # every engine but the dummy runs on every kind of device the runtime drives
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


def test_gated_distributions_move_in_lockstep():
    import re
    from pathlib import Path

    from mizuoto.version import VERSION

    root = Path(__file__).resolve().parents[2]
    inits = sorted(root.glob("packages/native/*/mizuoto_*/__init__.py"))
    assert len(inits) == 2
    for init in inits:
        assert re.search(rf'^VERSION = "{re.escape(VERSION)}"$', init.read_text(), re.M), f"{init} is not at {VERSION}"
        assert '"mizuoto"' in (init.parent.parent / "pyproject.toml").read_text(), f"{init.parent.parent.name} does not depend on mizuoto"


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
