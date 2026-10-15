import pytest

from mizuoto.core.errors import ConflictError, NotFoundError, ParameterError, ValidationError
from mizuoto.core.presets.models import PresetInput, PresetPatch
from mizuoto.core.tts.models import CommonParams, SpeakRequest

MODEL = "dummy/default"


def test_presets_save_apply_and_default(services):
    p = services.presets
    warm = p.create(PresetInput(name="Warm", model=MODEL, params={"temperature": 0.7, "speed": 0.9}, engine_options={"brightness": 0.2}))
    common = p.create(PresetInput(name="Fast", params={"speed": 1.4}))
    assert warm.engine == "dummy" and warm.params == {"temperature": 0.7, "speed": 0.9}
    assert [x.name for x in p.list_presets(MODEL)] == ["Fast", "Warm"]
    assert [x.name for x in p.list_presets("qwen3tts/1.7b-base")] == ["Fast"]
    with pytest.raises(ConflictError):
        p.create(PresetInput(name="Warm", model=MODEL))
    with pytest.raises(ParameterError):
        p.create(PresetInput(name="Bad", model=MODEL, params={"top_k": 3}))
    with pytest.raises(ValidationError):
        p.create(PresetInput(name="Opts", engine_options={"brightness": 0.2}))
    with pytest.raises(ValidationError):
        p.create(PresetInput(name="Range", model=MODEL, params={"speed": 99}))
    with pytest.raises(ValidationError):
        p.create(PresetInput(name="Typo", params={"sped": 1.0}))

    req = p.apply("Warm", SpeakRequest(model=MODEL, text="Hi.", params=CommonParams(speed=1.1)))
    assert req.params.speed == 1.1 and req.params.temperature == 0.7 and req.engine_options == {"brightness": 0.2}
    with pytest.raises(ValidationError):
        p.apply(warm.id, SpeakRequest(model="qwen3tts/1.7b-base", text="Hi."))
    assert p.apply(common.id, SpeakRequest(model=MODEL, text="Hi.")).params.speed == 1.4

    assert p.default_for(MODEL) is None
    p.set_default(warm.id)
    second = p.create(PresetInput(name="Bright", model=MODEL, engine_options={"brightness": 0.9}))
    p.set_default(second.id)
    assert p.default_for(MODEL).id == second.id and not p.get(warm.id).is_default
    with pytest.raises(ValidationError):
        p.set_default(common.id)

    assert p.update("Warm", PresetPatch(name="Warmer", params={"temperature": 0.5})).params == {"temperature": 0.5}
    p.delete("Warmer")
    with pytest.raises(NotFoundError):
        p.get(warm.id)
