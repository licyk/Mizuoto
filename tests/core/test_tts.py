import time

import pytest

from mizuoto.core.audio.io import decode, probe
from mizuoto.core.errors import CapabilityError, LanguageError, ModelMissingError, NotFoundError, ParameterError, ReferenceInvalidError, ValidationError
from mizuoto.core.tts.models import CommonParams, EmotionInput, ReferenceInput, SpeakRequest, StyleInput, VoiceInput
from mizuoto.core.tts.service import StreamSink

MODEL = "dummy/default"


def req(**kw):
    kw.setdefault("model", MODEL)
    kw.setdefault("text", "Hello there.")
    return SpeakRequest(**kw)


def test_speak_stores_an_output(services):
    job = services.tts.speak(req(voice_spec=VoiceInput(preset="alto"), params=CommonParams(seed=3, speed=1.25)), wait=True, origin="test")
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.engine == "dummy" and out.model == MODEL and out.text == "Hello there."
    assert out.params["speed"] == 1.25 and out.params["seed"] == 3
    info = probe(__import__("pathlib").Path(out.path))
    assert info.sample_rate == 24000 and abs(info.duration - out.duration) < 0.05
    assert out.meta["timings"]["total"] > 0 and out.meta["voice_mode"] == "preset"
    assert job.origin == "test"


def test_same_seed_same_audio(services):
    a = services.tts.speak(req(params=CommonParams(seed=9, temperature=1.0)), wait=True)
    b = services.tts.speak(req(params=CommonParams(seed=9, temperature=1.0)), wait=True)
    xa, _ = decode(__import__("pathlib").Path(services.outputs.get(a.result["output_id"]).path))
    xb, _ = decode(__import__("pathlib").Path(services.outputs.get(b.result["output_id"]).path))
    assert (xa == xb).all()


def test_output_format_and_rate(services):
    job = services.tts.speak(req(output={"format": "flac", "sample_rate": 16000, "normalize": "loudness"}), wait=True)
    out = services.outputs.get(job.result["output_id"])
    assert out.format == "flac" and out.sample_rate == 16000 and out.path.endswith(".flac")


def test_mizuoto_segmentation_joins_with_pauses(services):
    text = "First sentence here. " * 30
    plan = services.tts.plan(req(text=text))
    assert plan.segmented_by == "mizuoto" and len(plan.segments) > 1
    job = services.tts.speak(req(text=text, params=CommonParams(segment_silence_ms=0)), wait=True)
    assert job.state == "completed"


def test_the_attention_report_adds_up_over_segments(services):
    pytest.importorskip("torch")
    services.settings.update({"compute": {"attention": "fast"}})
    text = "First sentence here. " * 30
    segments = len(services.tts.plan(req(text=text)).segments)
    job = services.tts.speak(req(text=text, engine_options={"attend": 2}), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.params["attention"] == {"policy": "fast", "used": {"sdpa": 2 * segments}, "fallbacks": {}}


def test_batch_gives_one_output_per_text(services):
    job = services.tts.speak(req(text="", texts=["one", "two words", "three more words"]), wait=True)
    assert job.state == "completed" and len(job.result["output_ids"]) == 3
    durations = [services.outputs.get(i).duration for i in job.result["output_ids"]]
    assert durations == sorted(durations)


def test_reference_clone_and_limits(services, wav_file):
    job = services.tts.speak(req(voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file), transcript="a tone")])), wait=True)
    assert job.state == "completed" and job.result["plan"]["voice_mode"] == "clone"
    plan = services.tts.plan(req(voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file))])))
    assert plan.voice_mode == "clone_no_transcript"
    assert plan.references[0].sample_rate == 22050 and abs(plan.references[0].duration - 2.0) < 0.05
    services.engines.manifest("dummy").variants["default"].reference.max_seconds = 1.0
    services.settings.set_value("tts.reference_policy", "reject")
    with pytest.raises(ReferenceInvalidError) as e:
        services.tts.plan(req(voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file))])))
    assert e.value.detail["reason"] == "too_long"


def test_validation_happens_before_any_job(services):
    cases = [
        (req(params=CommonParams(top_k=3), strict=True), ParameterError),
        (req(params=CommonParams(speed=50)), ParameterError),
        (req(engine_options={"nope": 1}), ParameterError),
        (req(engine_options={"fail": "maybe"}), ParameterError),
        (req(voice_spec=VoiceInput(attributes={"gender": "robot"})), ValidationError),
        (req(voice_spec=VoiceInput(preset="nobody")), ValidationError),
        (req(voice_spec=VoiceInput(preset="alto", description="calm")), ValidationError),
        (req(style=StyleInput(emotion=EmotionInput(vector={"bored": 1.0}))), ValidationError),
        (req(text="   "), ValidationError),
        (req(model="dummy/nothing"), NotFoundError),
    ]
    for request, error in cases:
        with pytest.raises(error):
            services.tts.speak(request, wait=True)
    assert services.jobs.list_jobs().items == []


def test_lenient_mode_drops_with_a_warning(services):
    plan = services.tts.plan(req(params=CommonParams(top_k=3), strict=False))
    assert "top_k" not in plan.params and any("top_k" in w for w in plan.warnings)
    # Lenient is the default; tts.strict_params refuses instead.
    assert any("top_k" in w for w in services.tts.plan(req(params=CommonParams(top_k=3))).warnings)
    services.settings.set_value("tts.strict_params", "true")
    with pytest.raises(ParameterError):
        services.tts.plan(req(params=CommonParams(top_k=3)))


def test_capabilities_are_checked(services):
    variant = services.engines.manifest("dummy").variants["default"]
    variant.capabilities = [c for c in variant.capabilities if c not in ("auto_voice", "instruct", "batch")]
    variant.languages = ["en", "zh-*"]
    with pytest.raises(CapabilityError):
        services.tts.plan(req())
    with pytest.raises(CapabilityError):
        services.tts.plan(req(voice_spec=VoiceInput(preset="alto"), style=StyleInput(instruct="whisper")))
    with pytest.raises(CapabilityError):
        services.tts.plan(req(voice_spec=VoiceInput(preset="alto"), text="", texts=["a"]))
    with pytest.raises(LanguageError):
        services.tts.plan(req(voice_spec=VoiceInput(preset="alto"), language="ja"))
    assert services.tts.plan(req(voice_spec=VoiceInput(preset="alto"), language="zh-sichuan")).language == "zh-sichuan"
    assert services.tts.plan(req(voice_spec=VoiceInput(preset="alto"), language="English")).language == "en"


def test_missing_model_names_the_download(services):
    services.engines.manifest("dummy").models[0].builtin = False
    with pytest.raises(ModelMissingError) as e:
        services.tts.plan(req())
    assert e.value.detail["models"][0]["id"] == MODEL and "mizuoto model download" in e.value.detail["hint"]


def test_licence_gate(services):
    licence = services.engines.manifest("dummy").licence
    licence.gated, licence.version = True, "2026-01-01"
    from mizuoto.core.errors import LicenceError

    with pytest.raises(LicenceError):
        services.tts.plan(req())
    services.settings.update({"engines": {"dummy": {"accepted_licence": "gpl-3.0@2026-01-01"}}})
    plan = services.tts.plan(req())
    assert plan.licence_tags == ["gpl-3.0"]


def test_engine_failures_fail_the_job(services):
    job = services.tts.speak(req(engine_options={"fail": "error"}), wait=True)
    assert job.state == "failed" and job.error.code == "engine_error"
    job = services.tts.speak(req(engine_options={"fail": "crash"}), wait=True)
    assert job.state == "failed" and job.error.code == "worker_failed"
    job = services.tts.speak(req(), wait=True)  # a fresh worker takes over
    assert job.state == "completed"


def test_out_of_memory_unloads_every_idle_model_from_the_gpus(services):
    released = []
    from mizuoto.core.compute.service import MemoryHolder

    services.compute.add_holder(MemoryHolder("gpu0", busy=lambda: None, cached=lambda: ["m"], release=lambda: released.append("gpu0"), device=lambda: "cuda:0"))
    services.compute.add_holder(MemoryHolder("gpu1", busy=lambda: None, cached=lambda: ["m"], release=lambda: released.append("gpu1"), device=lambda: "cuda:1"))
    job = services.tts.speak(req(engine_options={"fail": "oom"}), wait=True)
    assert job.state == "failed" and job.error.code == "compute_unavailable" and job.error.detail["reason"] == "out_of_memory"
    assert sorted(released) == ["gpu0", "gpu1"]  # every GPU, not only the failing device
    assert any("every model not in use was unloaded" in line for line in services.jobs.read_log(job.id).lines)


def test_streaming_sink_sees_every_chunk_in_order(services):
    events: list[tuple] = []
    sink = StreamSink(
        start=lambda sr, g: events.append(("start", sr, g)),
        chunk=lambda samples, sr, seg: events.append(("chunk", len(samples), seg)),
        end=lambda ids: events.append(("end", len(ids))),
    )
    text = "Sentence one is here. " * 20
    job = services.tts.speak(req(text=text, engine_options={"chunk_ms": 200}), wait=True, stream=sink)
    assert job.state == "completed"
    assert events[0] == ("start", 24000, "segment") and events[-1] == ("end", 1)
    segs = [e[2] for e in events if e[0] == "chunk"]
    assert segs == sorted(segs) and len(set(segs)) > 1


def test_cancel_a_queued_speak(make_services):
    services = make_services(start_background=True)
    job = services.tts.speak(req(text="word " * 200, engine_options={"chunk_ms": 50, "delay_ms": 50}))
    deadline = time.monotonic() + 15
    while services.jobs.get(job.id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    services.jobs.cancel(job.id)
    while services.jobs.get(job.id).state == "cancelling" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert services.jobs.get(job.id).state == "cancelled"
    assert services.outputs.list_outputs().items == []


def test_retry_runs_the_stored_request(services):
    job = services.tts.speak(req(engine_options={"fail": "error"}), wait=True)
    assert job.can_retry and job.request["model"] == MODEL
    assert services.jobs.retry(job.id).request == job.request


def test_model_spec_lists_what_the_ui_needs(services):
    spec = services.tts.model_spec(MODEL)
    names = {p.name for p in spec.params}
    assert {"temperature", "speed", "seed", "delay_ms", "brightness"} <= names
    assert "fail" not in names  # hidden options stay out of forms
    assert "preset" in spec.voice_modes and spec.speakers[:1] == ["alto"]


def test_a_field_the_request_does_not_have_is_ignored_and_said(services):
    """An unknown top-level field (``options`` for ``engine_options``) still validates, as every record does, and the
    plan's warnings name it, so a misspelt field does not silently give the defaults."""
    body = req().model_dump()
    body["options"] = {"fail": True}
    request = SpeakRequest.model_validate(body)
    assert request.ignored_fields == ["options"]
    plan = services.tts.plan(request)
    assert any("Ignored fields the request does not have: options" in w for w in plan.warnings)
    assert all("Ignored" not in w for w in services.tts.plan(req()).warnings)


def test_planning_never_refuses_on_estimated_gpu_memory(services):
    """A model always tries to load (ComfyUI's way): only a real allocation failure is an error. Planning does not ask
    for the devices' free memory, even with a full GPU and ``vram_mode = high``."""
    from mizuoto.core.compute.models import ComputeInfo, DeviceInfo

    gpu = DeviceInfo(id="cuda:0", kind="cuda", name="GPU", memory_total_mb=24000, memory_used_mb=23900)
    asked = []
    services.compute._prober = lambda: asked.append(1) or ComputeInfo(backend="CUDA 12.8", devices=[gpu, DeviceInfo(id="cpu", kind="cpu", name="CPU")], probed_with="test")
    services.settings.update({"compute": {"vram_mode": "high"}})
    services.compute.info(refresh=True)
    asked.clear()
    plan = services.tts.plan(req())
    assert plan.model == MODEL and not asked


def test_convert_and_edit(services, wav_file):
    from mizuoto.core.tts.models import ConvertRequest, EditRequest, EditSpanInput, SourceInput

    source = SourceInput(path=str(wav_file))
    job = services.tts.convert(ConvertRequest(model=MODEL, source=source, voice_spec=VoiceInput(references=[ReferenceInput(path=str(wav_file))])), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.kind == "conversion" and abs(out.duration - 2.0) < 0.2
    job = services.tts.edit(EditRequest(model=MODEL, source=source, transcript="A new sentence.", edits=[EditSpanInput(start=0.5, end=1.0, text="new")]), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.kind == "edit" and out.text == "A new sentence."
    with pytest.raises(ValidationError):
        services.tts.convert(ConvertRequest(model=MODEL, source=source))
    with pytest.raises(ValidationError):
        services.tts.edit(EditRequest(model=MODEL, source=source, transcript="x", edits=[EditSpanInput(start=1.0, end=0.5, text="y")]))


def test_streamed_text_reaches_the_engine(services):
    import threading

    from mizuoto.core.tts.service import TextFeed

    feed = TextFeed()
    chunks, ends = [], []
    sink = StreamSink(chunk=lambda samples, rate, segment: chunks.append(len(samples)), end=ends.append, text=feed)

    def write():
        for piece in ("Hello ", "there, ", "streamed ", "text."):
            time.sleep(0.05)
            feed.put(piece)
        feed.close()

    threading.Thread(target=write).start()
    job = services.tts.speak(req(text="Well, ", text_stream=True), wait=True, stream=sink)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.text == "Well, Hello there, streamed text." and out.params.get("seed") is None
    assert len(chunks) == 5 and ends == [[out.id]]
    with pytest.raises(ValidationError):
        services.tts.speak(req(text_stream=True))
    with pytest.raises(ValidationError):
        services.tts.plan(req(text_stream=True))
    with pytest.raises(ValidationError):
        TextFeed(max_chars=3).put("long")


def test_openai_requests_translate(services, wav_file):
    from mizuoto.core.tts.models import OpenAiSpeechRequest

    r = services.tts.from_openai(OpenAiSpeechRequest(model=MODEL, input="Hi.", voice="alloy", response_format="pcm", speed=1.5, instructions="Softly."))
    assert r.voice is None and r.voice_spec is None and r.params.speed == 1.5 and r.style.instruct == "Softly."
    assert r.output.format == "wav" and r.output.sample_rate == 24000 and r.strict is False
    assert services.tts.from_openai(OpenAiSpeechRequest(model=MODEL, input="Hi.", voice="tenor")).voice_spec.preset == "tenor"
    with pytest.raises(NotFoundError):
        services.tts.from_openai(OpenAiSpeechRequest(model=MODEL, input="Hi.", voice="nobody"))
    with pytest.raises(NotFoundError):
        services.tts.from_openai(OpenAiSpeechRequest(model="tts-1", input="Hi."))
    services.settings.update({"tts": {"model_aliases": {"tts-1": MODEL}}})
    assert services.tts.from_openai(OpenAiSpeechRequest(model="tts-1", input="Hi.")).model == MODEL


def test_a_consent_note_is_never_required(services, wav_file):
    """A clip or a library voice clones without a consent statement on every engine; the licence notice says what is allowed."""
    clip = VoiceInput(references=[ReferenceInput(path=str(wav_file), transcript="Hi.")])
    assert services.tts.plan(req(voice_spec=clip)).voice_mode == "clone"
    voice, _ = services.voices.create_from_audio("Nc", [wav_file], ["Hi."])
    assert services.tts.plan(req(voice=voice.id)).voice_mode == "clone"


def test_engine_ops(services):
    job = services.tts.engine_op(MODEL, "echo", {"x": 1}, wait=True)
    assert job.state == "completed" and job.result["echo"] == {"x": 1}
    with pytest.raises(CapabilityError):
        services.tts.engine_op(MODEL, "nope")


def _wait_running(services, job_id, timeout=30):
    end = time.monotonic() + timeout
    while services.jobs.get(job_id).state != "running":
        assert time.monotonic() < end
        time.sleep(0.05)


def test_queued_look_alikes_run_as_one_batch(make_services):
    services = make_services(start_background=True)
    tts = services.tts
    alto = VoiceInput(preset="alto")
    hold = tts.speak(SpeakRequest(model="dummy/default", text="Hold the worker.", voice_spec=alto, engine_options={"delay_ms": 1500}))
    _wait_running(services, hold.id)
    queued = [tts.speak(SpeakRequest(model="dummy/default", text=f"Line number {i}.", voice_spec=alto)) for i in range(3)]
    seeded = tts.speak(SpeakRequest(model="dummy/default", text="With a seed.", voice_spec=alto, params=CommonParams(seed=3)))
    done = [services.jobs.wait(j.id, 60) for j in queued]
    assert all(j.state == "completed" for j in done), [j.error for j in done]
    assert {j.result["batch"]["size"] for j in done} == {3} and len({j.result["batch"]["leader"] for j in done}) == 1
    texts = [services.outputs.get(j.result["output_id"]).text for j in done]
    assert texts == [f"Line number {i}." for i in range(3)]
    assert all(services.outputs.get(j.result["output_id"]).duration > 0.2 for j in done)
    assert "batch" not in services.jobs.wait(seeded.id, 60).result and "batch" not in services.jobs.wait(hold.id, 60).result


def test_a_failed_batch_sends_its_members_back_to_the_queue(make_services):
    services = make_services(start_background=True)
    tts = services.tts
    alto = VoiceInput(preset="alto")
    hold = tts.speak(SpeakRequest(model="dummy/default", text="Hold the worker.", voice_spec=alto, engine_options={"delay_ms": 1000}))
    _wait_running(services, hold.id)
    broken = [tts.speak(SpeakRequest(model="dummy/default", text=f"Line {i}.", voice_spec=alto, engine_options={"fail": "error"})) for i in range(3)]
    done = [services.jobs.wait(j.id, 60) for j in broken]
    assert all(j.state == "failed" and j.error is not None for j in done)
    assert services.settings.settings.tts.micro_batch


def test_synthesis_jobs_hold_the_inference_resource(services):
    """Speak, conversion and editing jobs count against compute.inference_jobs."""
    from mizuoto.core.jobs.service import INFERENCE

    assert INFERENCE in services.tts._speak_job(req(text="Hello.").model_dump(mode="json")).resources


def test_attention_reports_merge_their_counts():
    from mizuoto.core.tts.service import merge_attention

    one = {"policy": "flash", "used": {"flash": 4, "sdpa": 1}, "fallbacks": {"flash→sdpa": {"bool mask": 1}}}
    two = {"policy": "flash", "used": {"flash": 2, "sdpa": 3}, "fallbacks": {"flash→sdpa": {"bool mask": 2, "head dim 80": 1}}}
    assert merge_attention(None, None) == {}
    assert merge_attention(None, one) == {"attention": one}
    assert merge_attention(one, two) == {"attention": {"policy": "flash", "used": {"flash": 6, "sdpa": 4}, "fallbacks": {"flash→sdpa": {"bool mask": 3, "head dim 80": 1}}}}
