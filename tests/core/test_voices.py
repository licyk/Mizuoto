"""The voice library, fit, ASR transcripts and voice caches, against the dummy engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mizuoto.core.errors import CapabilityError, ConflictError, NotFoundError, ReferenceInvalidError, ValidationError
from mizuoto.core.tts.models import CommonParams, ReferenceInput, SpeakRequest, VoiceInput

MODEL = "dummy/default"


@pytest.fixture
def asr(services, monkeypatch):
    """Speech recognition answered by the dummy engine (its ``transcribe`` op), not the native Whisper."""
    import mizuoto.core.inference.service as inference
    import mizuoto.runtime.engine as engines

    monkeypatch.setitem(engines.BUILTIN, "asr", "mizuoto.engines.dummy:DummyEngine")
    monkeypatch.setattr(engines, "_instances", {})
    monkeypatch.setattr(inference, "TORCH_FREE", inference.TORCH_FREE | {"asr"})
    real = services.engines.manifest("asr")
    services.engines.add(real.model_copy(update={"devices": ["cpu"]}))
    folder = services.models.folder("asr/whisper-large-v3-turbo")
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps({"variant": "whisper", "files": []}))
    assert services.asr.available()
    return services.asr


def make_clip(tmp_path: Path, name: str, seconds: float, sr: int = 24000) -> Path:
    import numpy as np

    from mizuoto.core.audio.io import encode

    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    return encode(x, sr, tmp_path / "clips" / name, "wav")


def test_reference_voice_lifecycle(services, tmp_path):
    clip = make_clip(tmp_path, "a.wav", 2.0)
    voice, warnings = services.voices.create_from_audio("Ann Example", [clip], ["Hello there."], "en", consent="my own voice")
    assert voice.slug == "ann-example" and voice.kind == "reference" and voice.consent and voice.consent.source == "my own voice"
    ref = voice.references[0]
    assert ref.primary and ref.transcript == "Hello there." and ref.transcript_source == "user" and abs(ref.duration - 2.0) < 0.01
    assert (services.voices.folder(voice) / ref.file).is_file() and ref.file.endswith(".flac")
    assert warnings == []
    assert services.voices.get(voice.id).slug == "ann-example" and services.voices.get("ann-example").id == voice.id
    second, _ = services.voices.create_from_audio("Ann Example", [clip], ["Hi."])
    assert second.slug == "ann-example-2"
    assert [v.slug for v in services.voices.list_voices(q="ann")] == ["ann-example", "ann-example-2"]
    assert [v.slug for v in services.voices.list_voices(language="en")] == ["ann-example"]

    other = make_clip(tmp_path, "b.wav", 3.0)
    voice, _ = services.voices.add_reference(voice.id, other, "Second clip.", primary=True)
    assert len(voice.references) == 2 and services.voices.ordered_references(voice)[0].transcript == "Second clip."
    with pytest.raises(ConflictError):
        services.voices.add_reference(voice.id, other)
    voice = services.voices.set_primary(voice.id, ref.sha256[:10])
    assert services.voices.ordered_references(voice)[0].sha256 == ref.sha256
    voice = services.voices.set_transcript(voice.id, ref.sha256, "Corrected.")
    assert voice.references[0].transcript == "Corrected."
    voice = services.voices.remove_reference(voice.id, ref.sha256)
    assert len(voice.references) == 1 and voice.references[0].primary
    voice = services.voices.update(voice.id, name="Ann", tags=["calm"], language="zh")
    assert voice.name == "Ann" and voice.slug == "ann-example" and voice.tags == ["calm"] and voice.language == "zh"

    archive = services.voices.export(voice.id, tmp_path / "ann.zip")
    copy = services.voices.import_archive(archive, name="Ann copy")
    assert copy.id != voice.id and copy.slug == "ann-copy" and copy.references == voice.references
    services.voices.delete(copy.id)
    with pytest.raises(NotFoundError):
        services.voices.get("ann-copy")


def test_voice_needs_clips_and_name(services, tmp_path):
    with pytest.raises(ValidationError):
        services.voices.create_from_audio("x", [])
    with pytest.raises(NotFoundError):
        services.voices.create_from_audio("x", [tmp_path / "missing.wav"])
    with pytest.raises(ValidationError):
        services.voices.create_from_audio(" ", [make_clip(tmp_path, "a.wav", 1.0)])
    assert services.voices.list_voices() == []


def test_missing_transcript_without_asr_warns(services, tmp_path):
    voice, warnings = services.voices.create_from_audio("Bo", [make_clip(tmp_path, "a.wav", 1.5)])
    # Without its environment (worker) or its model (native), the speech recognition cannot run.
    assert voice.references[0].transcript is None and warnings and ("not installed" in warnings[0] or "not downloaded" in warnings[0])
    # The dummy clones without a transcript, so the voice is still usable there.
    fit = services.voices.fit(voice.id, MODEL)
    assert fit.usable and fit.mode == "clone_no_transcript"


def test_asr_fills_transcripts(services, asr, tmp_path):
    voice, warnings = services.voices.create_from_audio("Cy", [make_clip(tmp_path, "a.wav", 1.5)], language="en")
    assert warnings == []
    ref = voice.references[0]
    assert ref.transcript == "dummy transcript of 1.5 seconds" and ref.transcript_source == "asr"
    assert asr.cached(ref.sha256) is not None
    voice = services.voices.retranscribe(voice.id)
    assert voice.references[0].transcript == "dummy transcript of 1.5 seconds"
    job = asr.submit(make_clip(tmp_path, "b.wav", 2.5), wait=True)
    assert job.state == "completed" and job.result["text"] == "dummy transcript of 2.5 seconds"


def test_retranscribe_without_asr_names_the_model_and_keeps_the_transcripts(services, tmp_path):
    voice, _ = services.voices.create_from_audio("Rt", [make_clip(tmp_path, "a.wav", 1.5)], ["Kept."], language="en")
    with pytest.raises(ValidationError) as e:
        services.voices.retranscribe(voice.id)
    assert e.value.detail == {"reason": "asr_unavailable", "model": services.settings.settings.tts.asr_model}
    assert services.voices.get(voice.id).references[0].transcript == "Kept."


def test_speak_with_a_library_voice_and_its_cache(services, tmp_path):
    voice, _ = services.voices.create_from_audio("Di", [make_clip(tmp_path, "a.wav", 2.0)], ["Hello."])
    job = services.tts.speak(SpeakRequest(model=MODEL, text="One.", voice=voice.slug, params=CommonParams(seed=1)), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.voice_id == voice.id and out.meta["voice_mode"] == "clone" and Path(out.path).name.split("-", 2)[2].startswith("di-")
    caches = list((services.voices.folder(voice) / "cache" / "dummy" / "default").glob("*/cache.json"))
    assert len(caches) == 1 and json.loads(caches[0].read_text())["files"]["pitch"].endswith("pitch.txt")
    stamp = caches[0].stat().st_mtime_ns
    again = services.tts.speak(SpeakRequest(model=MODEL, text="Two.", voice=voice.id), wait=True)
    assert again.state == "completed" and caches[0].stat().st_mtime_ns == stamp
    # A changed transcript is a new cache.
    services.voices.set_transcript(voice.id, voice.references[0].sha256, "Changed.")
    services.tts.speak(SpeakRequest(model=MODEL, text="Three.", voice=voice.id), wait=True)
    assert len(list((services.voices.folder(voice) / "cache" / "dummy" / "default").glob("*/cache.json"))) == 2
    prepared = services.tts.prepare(voice.slug, MODEL, wait=True)
    assert prepared.state == "completed" and prepared.result["files"] == ["pitch"]
    with pytest.raises(ValidationError):
        services.tts.speak(SpeakRequest(model=MODEL, text="x", voice=voice.id, voice_spec=VoiceInput(preset="alto")))


def test_request_reference_transcribed_by_the_job(services, asr, tmp_path):
    clip = make_clip(tmp_path, "r.wav", 2.0)
    req = SpeakRequest(model=MODEL, text="Hi.", voice_spec=VoiceInput(references=[ReferenceInput(path=str(clip))]))
    plan = services.tts.plan(req)
    assert plan.voice_mode == "clone" and plan.references[0].transcribe and plan.references[0].transcript is None
    job = services.tts.speak(req, wait=True)
    assert job.state == "completed", job.error
    assert job.result["plan"]["references"][0]["transcript"] == "dummy transcript of 2.0 seconds"
    # Now cached: the next plan has the transcript at once.
    plan = services.tts.plan(req)
    assert not plan.references[0].transcribe and plan.references[0].transcript == "dummy transcript of 2.0 seconds"


def test_library_voice_transcribed_by_the_job_keeps_it(services, asr, tmp_path):
    voice, _ = services.voices.create_from_audio("Ed", [make_clip(tmp_path, "a.wav", 1.0)], transcribe=False)
    assert voice.references[0].transcript is None
    fit = services.voices.fit(voice.id, MODEL)
    assert fit.usable and fit.mode == "clone" and fit.needs == ["transcribe"]
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Hi.", voice=voice.id), wait=True)
    assert job.state == "completed", job.error
    ref = services.voices.get(voice.id).references[0]
    assert ref.transcript == "dummy transcript of 1.0 seconds" and ref.transcript_source == "asr"


def test_trim_policy(services, tmp_path):
    clip = make_clip(tmp_path, "long.wav", 70.0)  # the dummy takes at most 60 s
    req = SpeakRequest(model=MODEL, text="Hi.", voice_spec=VoiceInput(references=[ReferenceInput(path=str(clip), transcript="long")]))
    with pytest.raises(ReferenceInvalidError) as e:
        services.tts.plan(req)
    assert e.value.detail["reason"] == "too_long"
    voice, _ = services.voices.create_from_audio("Long", [clip], ["long"])
    fit = services.voices.fit(voice.id, MODEL)
    assert not fit.usable and "at most 60" in (fit.reason or "")
    services.settings.set_value("tts.reference_policy", "trim")
    plan = services.tts.plan(req)
    assert 58.0 <= plan.references[0].duration <= 60.0 and plan.references[0].transcript is None and plan.voice_mode == "clone_no_transcript"
    assert any("was cut" in w for w in plan.warnings)
    fit = services.voices.fit(voice.id, MODEL)
    assert fit.usable and "trim" in fit.needs


def test_preset_voice(services):
    voice = services.voices.create_preset(MODEL, "alto")
    assert voice.kind == "preset" and voice.preset is not None and voice.preset.speaker == "alto"
    with pytest.raises(ConflictError):
        services.voices.create_preset(MODEL, "alto")
    with pytest.raises(ValidationError):
        services.voices.create_preset(MODEL, "nobody")
    fit = services.voices.fit(voice.id, MODEL)
    assert fit.usable and fit.mode == "preset"
    plan = services.tts.plan(SpeakRequest(model=MODEL, text="Hi.", voice=voice.slug))
    assert plan.voice_mode == "preset" and plan.voice_id == voice.id
    assert not services.voices.fit(voice.id, "qwen3tts/1.7b-base").usable
    clone_only = services.engines.variant("dummy", "default").model_copy(update={"capabilities": ["clone"]})
    assert services.voices.fit_voice(voice, MODEL, clone_only).reason == "The model has no preset speakers"


def test_designed_voice(services):
    voice, output_id = services.voices.create_from_design("Deep", MODEL, "A sample sentence.", description="a deep voice", seed=3)
    assert voice.kind == "designed" and voice.design is not None and voice.design.description == "a deep voice" and voice.design.seed == 3
    ref = voice.references[0]
    assert ref.transcript == "A sample sentence." and ref.transcript_source == "design" and ref.primary
    assert services.outputs.get(output_id).meta["voice_mode"] == "design"
    # A cloning model uses the clip; a design-only model would design anew.
    assert services.voices.fit(voice.id, MODEL).mode == "clone"
    design_only = services.engines.variant("dummy", "default").model_copy(update={"capabilities": ["voice_design_freeform"]})
    fit = services.voices.fit_voice(voice, "dummy/other", design_only)
    assert fit.usable and fit.mode == "design" and fit.warnings
    with pytest.raises(ValidationError):
        services.voices.create_from_design("x", MODEL, "text")


def test_voice_from_output(services):
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Kept as a voice.", voice_spec=VoiceInput(preset="tenor")), wait=True)
    voice = services.voices.create_from_output(job.result["output_id"], "Tenor clip", consent="synthetic")
    assert voice.kind == "reference" and voice.references[0].transcript == "Kept as a voice."


def test_internal_engines_do_not_speak(services, asr):
    with pytest.raises(CapabilityError):
        services.tts.plan(SpeakRequest(model="asr/whisper-large-v3-turbo", text="Hi."))


def test_voice_weights(services, tmp_path):
    from mizuoto.core.models.detect import sovits_version

    voice, _ = services.voices.create_from_audio("Wei", [make_clip(tmp_path, "w.wav", 2.0)], ["Hello."])
    model_file = tmp_path / "tuned.bin"
    model_file.write_text("320")
    with pytest.raises(ValidationError):
        services.voices.add_weights(voice.id, "dummy", {"gpt": model_file})
    with pytest.raises(ValidationError):
        services.voices.add_weights(voice.id, "qwen3tts", {"model": model_file})
    v = services.voices.add_weights(voice.id, "dummy", {"model": model_file})
    assert v.weights["dummy"].files == {"model": "weights/dummy/model.bin"} and model_file.exists()
    fit = services.voices.fit(voice.id, MODEL)
    assert fit.usable and fit.weights and fit.mode == "clone"
    job = services.tts.speak(SpeakRequest(model=MODEL, text="Tuned.", voice=voice.id), wait=True)
    out = services.outputs.get(job.result["output_id"])
    assert out.params["voice_weights"] is True and out.params["pitch_hz"] == 320.0
    # Exported and imported with the voice; uploaded file by file.
    copy = services.voices.import_archive(services.voices.export(voice.id, tmp_path / "w.zip"), name="Wei copy")
    assert (services.voices.folder(copy) / copy.weights["dummy"].files["model"]).read_text() == "320"
    staged = tmp_path / "upload.part"
    staged.write_text("410")
    up = services.voices.upload_weights(copy.id, "dummy", "model", staged, name="other.bin")
    assert up.missing == [] and up.voice is not None and up.voice.weights["dummy"].names == {"model": "other.bin"}
    v = services.voices.remove_weights(voice.id, "dummy")
    assert v.weights == {} and not services.voices.fit(voice.id, MODEL).weights
    # GPT-SoVITS weights carry their version in their first bytes.
    pth = tmp_path / "x.pth"
    pth.write_bytes(b"05" + b"\0" * 100)
    assert sovits_version(pth) == "v2Pro"
