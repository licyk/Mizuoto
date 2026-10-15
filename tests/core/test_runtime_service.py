"""The runtime service: engines in this process, the models they hold and their handles, unloading, the reaper
and the doctor."""

import threading
import time
from contextlib import closing

import pytest

from mizuoto.core.errors import CancelledError, ComputeError, EngineError, NotFoundError, OperationTimeoutError, WorkerError
from mizuoto.core.inference.engine import SNAPSHOT_OP, EngineRequest, ModelRef
from mizuoto.core.inference.tasks import SynthesisTask, VoiceSpec

MODEL = "dummy/default"


def synth(sup, text="Hello there.", cancel=None, model=MODEL, timeout=None, **options):
    request = EngineRequest(cancel=cancel, timeout=timeout)
    stream = sup.handle(model).synthesize(SynthesisTask(text=text, voice=VoiceSpec(preset="alto"), options=options), request)
    with closing(stream):
        chunks = list(stream)
    return request, chunks


@pytest.fixture
def sup(services):
    s = services.runtime.engine("dummy", "cpu")
    s.load(MODEL, None, "default")
    return s


def test_load_and_synthesize(sup, services):
    assert sup.state == "ready"
    assert sup.loaded_models() == [MODEL]
    request, chunks = synth(sup, chunk_ms=100)
    assert {c.sample_rate for c in chunks} == {24000} and len(chunks) > 1
    assert all(c.samples.dtype.name == "float32" and c.samples.ndim == 1 for c in chunks)
    assert chunks[-1].final and request.timings["first_chunk"] is not None and request.timings["total"] >= request.timings["first_chunk"]
    status = services.runtime.status()
    [engine] = status.engines
    assert engine.engine == "dummy" and engine.state == "ready" and engine.remote is None
    [model] = status.models
    assert (model.model_id, model.engine, model.device, model.busy) == (MODEL, "dummy", "cpu", False) and model.last_used
    assert services.compute.usage().holders[-1].cached == [MODEL]


def test_errors_are_mapped(sup):
    with pytest.raises(EngineError) as e:
        synth(sup, fail="error")
    assert e.value.detail["reason"] == "requested"
    with pytest.raises(ComputeError) as e:
        synth(sup, fail="oom")
    assert e.value.detail["reason"] == "out_of_memory"
    with pytest.raises(EngineError) as e:
        synth(sup, fail="exception")
    assert "Traceback" in e.value.detail["traceback"]
    with pytest.raises(WorkerError):
        synth(sup, fail="crash")
    assert sup.state == "ready"  # every error leaves the engine usable
    synth(sup)


def test_handles_are_the_door(services):
    """``loaded`` loads once and gives the same model's handle again; a closed stream frees the engine."""
    handle = services.runtime.loaded(ModelRef(MODEL, "dummy", "default"), "cpu")
    assert handle.model_id == MODEL and handle.remote is None and handle.description["sample_rate"] == 24000
    again = services.runtime.loaded(ModelRef(MODEL, "dummy", "default"), "cpu")
    assert again.description is handle.description and services.runtime.description(MODEL) is handle.description
    engine = services.runtime.engine("dummy", "cpu")
    stream = handle.synthesize(SynthesisTask(text="a long text " * 10, voice=VoiceSpec(preset="alto"), options={"chunk_ms": 50}))
    next(stream)
    assert engine.busy
    stream.close()
    assert not engine.busy and engine.state == "ready"
    services.runtime.unload(MODEL)
    with pytest.raises(Exception, match="not loaded"):
        handle.call("transcribe", {"path": "x"})


def test_timeout(sup):
    with pytest.raises(OperationTimeoutError):
        synth(sup, text="a long text " * 30, timeout=0.3, chunk_ms=100, delay_ms=100)
    synth(sup)


def test_cancel_is_quick(sup):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(CancelledError):
        synth(sup, text="a long text " * 30, cancel=cancel, chunk_ms=100, delay_ms=100)
    assert time.monotonic() - started < 3
    assert sup.state == "ready"
    synth(sup)


def test_at_most_max_resident_models_stay_loaded(services, sup):
    services.settings.update({"compute": {"max_resident_models": 2}})
    sup.load("dummy/second", None, "default")
    assert sorted(sup.loaded_models()) == sorted([MODEL, "dummy/second"])
    synth(sup, model="dummy/second")  # the default model is now the least recently used
    sup.load("dummy/third", None, "default")
    assert sorted(sup.loaded_models()) == ["dummy/second", "dummy/third"]
    # a pinned model is never unloaded to make room
    services.settings.update({"engines": {"dummy": {"pinned_models": ["dummy/second"]}}})
    sup = services.runtime.engine("dummy", "cpu")  # the setting change unloaded the engine
    sup.load("dummy/second", None, "default")
    sup.load("dummy/third", None, "default")
    sup.load(MODEL, None, "default")
    assert sorted(sup.loaded_models()) == [MODEL, "dummy/second"]


def test_unload(services, sup):
    sup.load("dummy/second", None, "default")
    status = services.runtime.unload("dummy/second")
    assert [m.model_id for m in status.models] == [MODEL]
    with pytest.raises(NotFoundError):
        services.runtime.unload("dummy/second")
    assert services.runtime.unload().models == [] and sup.loaded_models() == []


def test_release_memory_and_shutdown(services, sup):
    services.compute.release()
    assert sup.loaded_models() == []
    sup.shutdown()
    assert sup.state == "stopped" and services.runtime.status().engines == []


def test_reaper_unloads_idle_models(services, sup):
    services.settings.update({"compute": {"unload_after_minutes": 1}})
    assert services.runtime.reap(now=time.time()) == []
    assert services.runtime.reap(now=time.time() + 120) == [MODEL]
    assert sup.loaded_models() == []
    services.settings.update({"compute": {"unload_after_minutes": 0, "max_resident_models": 1}})
    sup = services.runtime.engine("dummy", "cpu")
    sup.load(MODEL, None, "default")
    sup.make_room = None  # as if loaded beyond the limit
    sup.load("dummy/second", None, "default")
    assert services.runtime.reap() == [MODEL]


def test_engine_setting_change_unloads_its_models(services, sup):
    services.settings.update({"engines": {"dummy": {"load_options": {"x": 1}}}})
    assert sup.state == "stopped" and sup.loaded_models() == []
    fresh = services.runtime.engine("dummy", "cpu")
    assert fresh is not sup and fresh.options == {"x": 1}


def test_runtime_changed_events(make_services):
    services = make_services()
    seen = []
    services.events.subscribe(lambda e: seen.append(e) if getattr(e, "__event_name__", "") == "runtime_changed" else None)
    sup = services.runtime.engine("dummy", "cpu")
    sup.load(MODEL, None, "default")
    assert seen  # throttled to one every 0.5 s; the server's reaper thread sends the pending one
    time.sleep(0.6)
    services.runtime.unload(MODEL)
    assert seen[-1].runtime.models == [] and seen[-1].runtime.engines[0].engine == "dummy"


def test_no_residue_across_requests(sup):
    """The contract suite's snapshot: seeded requests leave the process state as it was."""
    handle = sup.handle(MODEL)

    def snapshot():
        return handle.call(SNAPSHOT_OP)["snapshot"]

    synth(sup)  # first use imports numpy, which adds its generator to the snapshot
    before = snapshot()
    assert set(before) >= {"cwd", "env", "rng", "torch_flags", "root_level", "warning_filters", "sys_path"}
    for n in range(5):
        task = SynthesisTask(text=f"Request {n}.", voice=VoiceSpec(preset="bass"), params={"seed": n, "temperature": 1.0})
        with closing(handle.synthesize(task)) as stream:
            list(stream)
    assert snapshot() == before


def test_transcribe_op(sup, wav_file):
    result = sup.handle(MODEL).call("transcribe", {"path": str(wav_file)})
    assert result["text"] == "dummy transcript of 2.0 seconds" and result["language"] == "en"


def test_doctor_reports_engines(services):
    report = services.runtime.doctor("dummy")
    assert [f.check for f in report.findings if f.engine == "dummy"] == ["engine"] and report.findings[-1].level == "ok"


def test_flavour_choice_and_install_command():
    from mizuoto.core.compute.models import ComputeInfo, DeviceInfo
    from mizuoto.core.compute.torch_index import choose_flavour, install_command

    cpu = DeviceInfo(id="cpu", kind="cpu", name="CPU")
    gpu = DeviceInfo(id="cuda:0", kind="cuda", name="RTX")
    assert choose_flavour(ComputeInfo(backend="CPU", devices=[cpu], probed_with="none")) == "cpu"
    assert choose_flavour(ComputeInfo(backend="CUDA 13.0", cuda_version="13.0", devices=[gpu, cpu], probed_with="nvidia-smi")) == "cu130"
    assert choose_flavour(ComputeInfo(backend="CUDA 12.9", cuda_version="12.9", devices=[gpu, cpu], probed_with="nvidia-smi")) == "cu129"
    assert choose_flavour(ComputeInfo(backend="CUDA 13.2", cuda_version="13.2", devices=[gpu, cpu], probed_with="nvidia-smi")) == "cu132"
    assert choose_flavour(ComputeInfo(backend="CUDA 12.7", cuda_version="12.7", devices=[gpu, cpu], probed_with="nvidia-smi")) == "cu126"
    assert choose_flavour(ComputeInfo(backend="CUDA 11.8", cuda_version="11.8", devices=[gpu, cpu], probed_with="nvidia-smi")) == "cpu"
    amd = DeviceInfo(id="cuda:0", kind="rocm", name="Radeon")
    assert choose_flavour(ComputeInfo(backend="ROCm", devices=[amd, cpu], probed_with="rocm-smi")) == "rocm7.2"
    mps = DeviceInfo(id="mps", kind="mps", name="Apple GPU")
    assert choose_flavour(ComputeInfo(backend="MPS", devices=[mps, cpu], probed_with="platform")) == "mps"
    command = install_command("cu128", python="python")
    assert command == 'python -m pip install --upgrade --force-reinstall "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128'
    assert "--index-url" not in install_command("mps", python="python")


def test_short_ram_unloads_idle_models_least_recently_used_first(services, sup, monkeypatch):
    """Below compute.ram_reserve_mb host copies go, then idle models (never the one loading)."""
    sup.load("dummy/second", None, "default")
    synth(sup, model="dummy/second")
    devices = services.runtime.inference.runtime.devices
    free = {"cpu": 10 * 2**20}
    monkeypatch.setattr(devices, "memory", lambda name: (free[name], 8 * 2**30))
    services.settings.update({"compute": {"ram_reserve_mb": 100, "max_resident_models": 8}})
    services.runtime.relieve_ram(keep="dummy/third")
    assert sup.loaded_models() == []
    sup.load(MODEL, None, "default")
    free["cpu"] = 2**30  # enough again: nothing goes
    services.runtime.relieve_ram()
    assert sup.loaded_models() == [MODEL]
    assert services.runtime.release_device("cpu") == []  # the dummy's models hold no device memory


def test_the_cuda_allocator_is_set_before_cuda_is_used(monkeypatch):
    import sys

    from mizuoto.core.inference.service import set_allocator

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    if "torch" in sys.modules:  # as in a fresh process: CUDA not used yet
        monkeypatch.setattr(sys.modules["torch"].cuda, "is_initialized", lambda: False)
    assert set_allocator("auto") is False and "PYTORCH_CUDA_ALLOC_CONF" not in __import__("os").environ
    assert set_allocator("expandable") is True
    assert __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert set_allocator("async") is False  # the environment's own choice stays


def test_attention_follows_the_settings_per_engine(services):
    rt = services.runtime
    assert rt.attention("dummy") == ("auto", False)
    services.settings.update({"compute": {"attention": "fast", "attention_strict": True}})
    assert rt.attention("dummy") == ("fast", True)
    services.settings.update({"engines": {"dummy": {"attention": "math"}, "qwen3tts": {"attention": "inherit"}}})
    assert rt.attention("dummy") == ("math", True)
    assert rt.attention("qwen3tts") == ("fast", True)  # inherit: compute.attention


def test_a_model_runs_with_the_attention_policy_it_loaded_with(services):
    pytest.importorskip("torch")
    sup = services.runtime.engine("dummy", "cpu")
    sup.load(MODEL, None, "default")
    request, _ = synth(sup, attend=3)
    assert "attention" not in request.applied  # auto: every site's own computation, nothing to report
    sup.unload(MODEL)
    services.settings.update({"compute": {"attention": "fast"}})
    sup.load(MODEL, None, "default")
    request, _ = synth(sup, attend=3)
    assert request.applied["attention"] == {"policy": "fast", "used": {"sdpa": 3}, "fallbacks": {}}
    [model] = services.runtime.status().models
    assert model.attention == "fast"


def test_a_backend_that_cannot_run_falls_back_or_fails_when_strict(services):
    pytest.importorskip("torch")
    services.settings.update({"compute": {"attention": "sdpa_flash"}})
    sup = services.runtime.engine("dummy", "cpu")
    sup.load(MODEL, None, "default")
    request, _ = synth(sup, attend=2)
    assert request.applied["attention"]["used"] == {"sdpa": 2}
    assert request.applied["attention"]["fallbacks"] == {"sdpa_flash→sdpa": {"CUDA and ROCm only": 2}}
    assert any("sdpa_flash→sdpa" in w for w in request.warnings)
    sup.unload(MODEL)
    services.settings.update({"compute": {"attention_strict": True}})
    sup.load(MODEL, None, "default")
    with pytest.raises(ComputeError) as e:
        synth(sup, attend=1)
    assert e.value.detail["backend"] == "sdpa_flash" and e.value.detail["reason"] == "CUDA and ROCm only"


def test_doctor_reports_the_attention_backends(services, monkeypatch):
    from mizuoto.core.compute import accel

    report = services.runtime.doctor("dummy")
    [finding] = [f for f in report.findings if f.check == "acceleration"]
    assert finding.level == "ok" and finding.message.startswith("attention auto; training auto")
    assert "xFormers not installed" in finding.message or "xFormers " in finding.message
    # The same as fields, for the web UI to word in its language.
    assert finding.data is not None and finding.data["attention"] == "auto" and not finding.data["strict"] and finding.data["training"] == "auto"
    assert [p["backend"] for p in finding.data["packages"]] == [p.backend for p in accel.PACKAGES] and finding.data["unusable"] == []
    services.settings.update({"compute": {"attention": "xformers"}})
    monkeypatch.setattr(accel, "installed_version", lambda p: None)
    [finding] = [f for f in services.runtime.doctor("dummy").findings if f.check == "acceleration"]
    assert finding.level == "warning" and "xformers is chosen but cannot run here" in finding.message
    assert finding.data is not None and finding.data["unusable"] == ["xformers"] and all(p["version"] is None for p in finding.data["packages"])


def test_install_commands_never_replace_torch():
    from mizuoto.core.compute.accel import install_command

    cmd = install_command("xformers", {"version": "2.8.0+cu126", "cuda": "12.6"}, python="py")
    assert cmd == 'py -m pip install "xformers==0.0.32.post2" --no-deps --index-url https://download.pytorch.org/whl/cu126'
    assert install_command("xformers", {"version": "2.11.0+cu128", "cuda": "12.8"}, python="py") == 'py -m pip install "xformers>=0.0.35" --no-deps'
    assert install_command("xformers", {"version": "2.8.0+cpu", "cuda": None}, python="py") is None
    assert install_command("sage", {"version": "2.8.0+rocm6.4", "hip": "6.4"}, python="py") is None
    assert "--no-deps" in (install_command("flash", {"version": "2.8.0+rocm6.4", "hip": "6.4"}, python="py") or "")
