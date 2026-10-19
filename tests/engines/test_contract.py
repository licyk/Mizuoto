"""The engine contract suite, against real engines.

Runs through the whole stack (plan → job → runtime → engine) for every installed catalogue model of
every available engine, and checks what an integrated engine must do: load and describe as its
manifest says; speak in every voice mode it declares, audibly, at its rate, for about as long as the
text needs, saying the text (word error rate by the ASR engine, when installed); repeat itself
under a seed; split two segments; stream as its variant says; stop when cancelled; and leave no residue in the process.

    python scripts/dev.py test-engines [engine…]

uses the models of ``MIZUOTO_TEST_DATA_DIR`` (default: the normal data folder), writes nothing there,
and runs on ``MIZUOTO_TEST_DEVICE`` (default: auto). ``MIZUOTO_TEST_SETTINGS`` adds settings as JSON, e.g.
``{"compute": {"vram_limit_mb": 4096}}`` to run every model partly loaded on a smaller GPU played by a big one. Licence-gated engines run when
``MIZUOTO_TEST_ACCEPT_LICENCES=1`` says their licences are accepted (and their distributions are installed).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.engine, pytest.mark.weights]

TEXTS = {
    "zh": ("今天的天气很好，我们一起去公园散步吧。", "第一句话在这里。第二句话紧随其后。"),
    "en": ("The quick brown fox jumps over the lazy dog near the river bank.", "This is the first sentence. And here comes the second one."),
}
WER_BOUND = 0.35


def _data_dir() -> Path:
    from mizuoto.core.paths import default_data_dir

    value = os.environ.get("MIZUOTO_TEST_DATA_DIR")
    return Path(value).expanduser() if value else default_data_dir(os.environ)


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    from mizuoto.core.context import build_services

    real = _data_dir()
    device = os.environ.get("MIZUOTO_TEST_DEVICE", "auto")
    overrides: dict[str, Any] = {
        "paths": {"models_dir": str(real / "models")},
        "engines": {e: {"device": device} for e in ("qwen3tts", "asr", "omnivoice", "cosyvoice", "f5tts", "gptsovits", "fishspeech", "indextts", "voxcpm")},
        "tts": {"normalize": "none"},
    }
    extra = json.loads(os.environ.get("MIZUOTO_TEST_SETTINGS") or "{}")
    for section, values in extra.items():
        overrides[section] = {**overrides.get(section, {}), **values} if isinstance(values, dict) else values
    s = build_services(data_dir=tmp_path_factory.mktemp("contract"), environ={}, settings_overrides=overrides, start_background=True)
    if os.environ.get("MIZUOTO_TEST_ACCEPT_LICENCES") == "1":
        # Licence-gated engines, for whoever runs the suite and has accepted their licences: in this temporary folder only.
        for engine in s.engines.ids():
            if s.engines.manifest(engine).licence.gated:
                s.engines.accept_licence(engine)
    yield s
    s.close()


def installed_models() -> list[str]:
    from mizuoto.core.context import open_services

    real = _data_dir()
    wanted = set(filter(None, os.environ.get("MIZUOTO_TEST_ENGINES", "").split(",")))
    out = []
    with open_services(data_dir=real, environ={}) as s:
        for m in s.models.list_models(installed=True):
            manifest = s.engines.manifest(m.engine)
            if manifest.internal or m.engine == "dummy" or (wanted and m.engine not in wanted):
                continue
            if s.engines.status(m.engine).available:
                out.append(m.id)
    return out


MODELS = installed_models() if os.environ.get("MIZUOTO_CONTRACT") else []


def normalise_words(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    if lang in ("zh", "ja", "yue"):
        return [c for c in text if not c.isspace()]
    return text.split()


def wer(reference: str, hypothesis: str, lang: str) -> float:
    r, h = normalise_words(reference, lang), normalise_words(hypothesis, lang)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


@pytest.fixture(scope="module")
def reference(services, tmp_path_factory):
    """A speech clip and its transcript to clone: ``MIZUOTO_TEST_REFERENCE`` (+ ``_TEXT``), else one
    spoken by an installed preset or design model."""
    from mizuoto.core.tts.models import CommonParams, SpeakRequest, VoiceInput

    path = os.environ.get("MIZUOTO_TEST_REFERENCE")
    if path:
        return Path(path), os.environ.get("MIZUOTO_TEST_REFERENCE_TEXT")
    text = "Hello, this clip lets the model learn how my voice sounds when I speak."
    for model in MODELS:
        info = services.models.get(model)
        caps = set(services.engines.variant(info.engine, info.variant).capabilities)
        spec = services.tts.model_spec(model)
        voice = None
        if "preset_speaker" in caps and spec.speakers:
            voice = VoiceInput(preset=spec.speakers[0])
        elif "voice_design_freeform" in caps:
            voice = VoiceInput(description="A clear adult female voice, neutral and steady.")
        if voice is not None:
            job = services.tts.speak(SpeakRequest(model=model, text=text, language="en", voice_spec=voice, params=CommonParams(seed=1)), wait=True)
            assert job.state == "completed", job.error
            return Path(services.outputs.get(job.result["output_id"]).path), text
    pytest.skip("No reference clip: set MIZUOTO_TEST_REFERENCE, or install a preset or design model")


def voice_modes(services, model: str) -> list[str]:
    info = services.models.get(model)
    caps = set(services.engines.variant(info.engine, info.variant).capabilities)
    modes = []
    if "preset_speaker" in caps:
        modes.append("preset")
    if "voice_design_freeform" in caps:
        modes.append("design")
    if "clone" in caps:
        modes.append("clone")
    if "clone_no_transcript" in caps:
        modes.append("clone_no_transcript")
    if "auto_voice" in caps:
        modes.append("auto")
    return modes


def request(services, model: str, mode: str, text: str, lang: str, reference: tuple[Path, str | None] | None, seed: int | None = 7, max_new_tokens: int | None = None):
    from mizuoto.core.tts.models import CommonParams, ReferenceInput, SpeakRequest, VoiceInput

    spec = None
    if mode == "preset":
        spec = VoiceInput(preset=services.tts.model_spec(model).speakers[0])
    elif mode == "design":
        spec = VoiceInput(description="A calm, warm male voice with a low pitch.")
    elif mode in ("clone", "clone_no_transcript"):
        assert reference is not None
        clip, transcript = reference
        # The reference is the test's own clip (or one a model made).
        spec = VoiceInput(references=[ReferenceInput(path=str(clip), transcript=transcript if mode == "clone" else None)])
    params = CommonParams(seed=seed, max_new_tokens=max_new_tokens)
    return SpeakRequest(model=model, text=text, language=lang, voice_spec=spec, params=params, output={"format": "wav", "normalize": "none"})


def run(services, req) -> tuple[Any, Any]:
    from mizuoto.core.audio.io import decode

    # A clip without its transcript is transcribed when ASR is installed; to exercise cloning from
    # the clip alone, switch that off for the request.
    refs = req.voice_spec.references if req.voice_spec else []
    no_transcript = bool(refs) and not any(r.transcript for r in refs)
    if no_transcript:
        services.settings.update({"tts": {"auto_transcribe": False}})
    try:
        job = services.tts.speak(req, wait=True)
    finally:
        if no_transcript:
            services.settings.update({"tts": {"auto_transcribe": True}})
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    samples, sr = decode(Path(out.path))
    return out, (samples, sr, job)


@pytest.mark.parametrize("model", MODELS)
def test_load_and_describe(services, model):
    info = services.models.require(model)
    variant = services.engines.variant(info.engine, info.variant)
    from mizuoto.core.inference.engine import ModelRef

    handle = services.runtime.loaded(ModelRef(model, info.engine, info.variant, info.path), services.runtime.device_for(info.engine))
    description = handle.description
    assert description["sample_rate"] == variant.sample_rate
    expected = services.models.speakers(model)  # a fine-tuned model's own, else its variant's
    if expected:
        assert sorted(description["speakers"]) == sorted(expected)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_voice_mode_speaks_the_text(services, model, lang, reference):
    import numpy as np

    info = services.models.get(model)
    variant = services.engines.variant(info.engine, info.variant)
    text = TEXTS[lang][0]
    for mode in voice_modes(services, model):
        if mode.startswith("clone") and lang != "en":
            continue  # the reference is English; cross-lingual cloning is not part of the contract
        out, (samples, sr, job) = run(services, request(services, model, mode, text, lang, reference))
        assert sr == variant.sample_rate == out.sample_rate
        assert float(np.sqrt(np.mean(samples**2))) > 0.01, f"{mode}: silent"
        chars = len(text)
        assert 0.03 * chars < out.duration < 0.4 * chars + 3, f"{mode}: {out.duration:.1f} s for {chars} characters"
        assert job.result["plan"]["voice_mode"] == mode
        if services.asr.available():
            t = services.asr.transcribe(Path(out.path), lang)
            assert wer(text, t.text, lang) < WER_BOUND, f"{mode}: heard {t.text!r}"


@pytest.mark.parametrize("model", MODELS)
def test_seed_repeats_and_segments(services, model, reference):
    import numpy as np

    mode = voice_modes(services, model)[0]
    first = run(services, request(services, model, mode, TEXTS["en"][0], "en", reference, seed=11))[1][0]
    second = run(services, request(services, model, mode, TEXTS["en"][0], "en", reference, seed=11))[1][0]
    assert first.shape == second.shape and np.array_equal(first, second)
    # Two sentences that do not fit one segment together: Mizuoto splits them (or the engine, with long_text).
    from mizuoto.core.text.segment import DEFAULT_BUDGET_OTHER

    words = ["Every", "word", "of", "this", "sentence", "is", "here", "to", "make", "it", "long", "enough"]
    sentence = " ".join(words * (DEFAULT_BUDGET_OTHER * 3 // 5 // len(" ".join(words)) + 1))
    out, (_, _, job) = run(services, request(services, model, mode, f"{sentence}. {sentence}.", "en", reference))
    plan = job.result["plan"]
    assert plan["segmented_by"] == "engine" or len(plan["segments"]) == 2
    assert out.meta["segments"] == len(plan["segments"])


@pytest.mark.parametrize("model", MODELS)
def test_streaming_sends_the_audio_as_it_is_made(services, model, reference):
    """A streamed request: the sink hears the variant's granularity, gets the audio in pieces (more than one for one
    segment when the model streams tokens), and the stored output is those pieces joined."""
    import numpy as np

    from mizuoto.core.audio.io import decode
    from mizuoto.core.tts.service import StreamSink

    info = services.models.get(model)
    variant = services.engines.variant(info.engine, info.variant)
    if "stream_audio" not in variant.capabilities:
        pytest.skip("the model does not stream")
    mode = voice_modes(services, model)[0]
    started: list[tuple[int, str | None]] = []
    pieces: list[Any] = []
    sink = StreamSink()
    sink.start = lambda rate, granularity: started.append((rate, granularity))
    sink.chunk = lambda samples, rate, segment: pieces.append(np.asarray(samples, dtype=np.float32).copy())
    job = services.tts.speak(request(services, model, mode, TEXTS["en"][0], "en", reference), wait=True, stream=sink)
    assert job.state == "completed", job.error
    assert started == [(variant.sample_rate, variant.stream_granularity)]
    if variant.stream_granularity == "token":
        assert len(pieces) > 1
    samples, _ = decode(Path(services.outputs.get(job.result["output_id"]).path))
    joined = np.concatenate(pieces)
    assert joined.shape == samples.shape and np.allclose(joined, samples, atol=1e-4)


@pytest.mark.parametrize("model", MODELS)
def test_streamed_audio_starts_early(services, model, reference):
    """A model that streams tokens sends its first audio well before the request ends (within three quarters of it:
    CosyVoice waits for its first chunk of tokens and their lookahead). GPT-SoVITS v2 cuts its stream only at a pause
    by default, as upstream does, so a sentence without one would come whole: its fixed-length chunks show the
    streaming itself."""
    from mizuoto.core.tts.service import StreamSink

    info = services.models.get(model)
    variant = services.engines.variant(info.engine, info.variant)
    if "stream_audio" not in variant.capabilities or variant.stream_granularity != "token":
        pytest.skip("the model does not stream tokens")
    mode = voice_modes(services, model)[0]
    run(services, request(services, model, mode, TEXTS["en"][1], "en", reference))  # loaded and warm
    first: list[float] = []
    sink = StreamSink()
    sink.chunk = lambda samples, rate, segment: first.append(time.monotonic()) if not first else None
    started = time.monotonic()
    req = request(services, model, mode, " ".join([TEXTS["en"][0]] * 3), "en", reference)
    if "fixed_length_chunk" in services.engines.manifest(info.engine).options:
        req.engine_options = {"fixed_length_chunk": True}
    job = services.tts.speak(req, wait=True, stream=sink)
    ended = time.monotonic()
    assert job.state == "completed", job.error
    print(f"first audio {model}: {first[0] - started:.2f} of {ended - started:.2f} s")
    assert first[0] - started < 0.75 * (ended - started)


@pytest.mark.parametrize("model", MODELS)
def test_unloading_frees_the_device(services, model, reference):
    """After a request, unloading the model gives its device memory back (graphs, caches and weights alike)."""
    import gc

    import torch

    info = services.models.get(model)
    device = services.runtime.device_for(info.engine)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        pytest.skip("measured on CUDA devices")
    services.runtime.unload()
    gc.collect()
    baseline = torch.cuda.memory_allocated(device)
    run(services, request(services, model, voice_modes(services, model)[0], TEXTS["en"][1], "en", reference))
    loaded = torch.cuda.memory_allocated(device)
    services.runtime.unload_model(model)
    gc.collect()
    left = torch.cuda.memory_allocated(device) - baseline
    print(f"unload {model}: {(loaded - baseline) / 2**20:.0f} MB loaded, {left / 2**20:.1f} MB left")
    assert left < 64 * 2**20, f"{left / 2**20:.0f} MB still allocated after unloading {model}"


@pytest.mark.parametrize("model", MODELS)
def test_another_engine_in_between_changes_nothing(services, model, reference):
    """The same seeded request gives the same audio before and after another engine's model ran in the process."""
    import numpy as np

    info = services.models.get(model)
    other = next((m for m in MODELS if services.models.get(m).engine != info.engine), None)
    if other is None:
        pytest.skip("no other engine's model is installed")
    mode = voice_modes(services, model)[0]
    req = request(services, model, mode, TEXTS["en"][1], "en", reference)
    _, (before, _, _) = run(services, req)
    run(services, request(services, other, voice_modes(services, other)[0], TEXTS["en"][1], "en", reference))
    _, (after, _, _) = run(services, request(services, model, mode, TEXTS["en"][1], "en", reference))
    assert np.array_equal(before, after)


@pytest.mark.parametrize("model", MODELS)
def test_cancel_stops_the_engine(services, model, reference):
    from mizuoto.core.events.models import JobUpdatedEvent

    mode = voice_modes(services, model)[0]
    long_text = " ".join([TEXTS["en"][0]] * 12)  # long enough that a fast model on a GPU is still speaking when cancelled
    req = request(services, model, mode, long_text, "en", reference)
    req.segmentation = "engine"
    started = threading.Event()
    services.events.subscribe(lambda e: started.set() if isinstance(e, JobUpdatedEvent) and e.job.message == "Speaking" else None)
    job = services.tts.speak(req)
    assert started.wait(600), "the job never started speaking"
    time.sleep(0.5)
    cancelled_at = time.monotonic()
    services.jobs.cancel(job.id)
    while services.jobs.get(job.id).state not in ("cancelled", "failed", "completed"):
        time.sleep(0.05)
        # Measured 0.1-0.2 s for every model on an RTX 3090; a CPU's step can take seconds.
        assert time.monotonic() - cancelled_at < (2 if services.runtime.device_for(services.models.get(model).engine).startswith("cuda") else 30)
    assert services.jobs.get(job.id).state == "cancelled"
    print(f"cancel latency {model}: {time.monotonic() - cancelled_at:.2f} s")
    # The engine stopped at its next cancel check and still answers.
    info = services.models.get(model)
    engine = services.runtime.engine(info.engine, services.runtime.device_for(info.engine))
    assert engine.state in ("ready", "busy") and model in engine.loaded_models()
    # …and speaks again: nothing the cancelled request left half-done (an engine's cache) breaks the next one.
    out, (samples, sr, _job) = run(services, request(services, model, mode, TEXTS["en"][0], "en", reference))
    assert len(samples) / sr > 1.0


@pytest.mark.parametrize("model", MODELS)
def test_no_residue(services, model, reference):
    from mizuoto.core.inference.engine import SNAPSHOT_OP, ModelRef

    info = services.models.get(model)
    # Short prompts can keep a fine-tuned model talking up to its frame limit (Qwen3's narrator: 8192 frames, hours on
    # a CPU); where the engine takes max_new_tokens, a cap keeps the requests short. Residue does not depend on length.
    cap = 300 if "max_new_tokens" in services.engines.manifest(info.engine).params else None
    run(services, request(services, model, voice_modes(services, model)[0], "This request warms the engine up.", "en", reference, max_new_tokens=cap))
    handle = services.runtime.loaded(ModelRef(model, info.engine, info.variant, info.path), services.runtime.device_for(info.engine))
    before = handle.call(SNAPSHOT_OP)["snapshot"]
    for n, mode in enumerate(voice_modes(services, model) * 2):
        run(services, request(services, model, mode, f"Request number {n}.", "en", reference, seed=n, max_new_tokens=cap))
    after = handle.call(SNAPSHOT_OP)["snapshot"]
    assert after == before
