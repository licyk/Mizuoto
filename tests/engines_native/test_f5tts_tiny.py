"""The native F5-TTS engine on a tiny random model: no weights needed."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("pypinyin")
pytest.importorskip("rjieba")
pytest.importorskip("yaml")

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.f5tts import F5TtsEngine  # noqa: E402
from mizuoto.engines.f5tts import engine as f5  # noqa: E402
from mizuoto.runtime.compile import Compiled  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EditSpan, LoadSpec, Reference, SpeechEditTask, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime, RuntimeOptions  # noqa: E402

VOCOS = {
    "feature_extractor": {
        "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
        "init_args": {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 100, "padding": "center"},
    },
    "backbone": {"class_path": "vocos.models.VocosBackbone", "init_args": {"input_channels": 100, "dim": 16, "intermediate_dim": 32, "num_layers": 1}},
    "head": {"class_path": "vocos.heads.ISTFTHead", "init_args": {"dim": 16, "n_fft": 1024, "hop_length": 256, "padding": "center"}},
}


def _random_state(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(t.shape, generator=g) * 0.02).to(t.dtype) if t.is_floating_point() else t for k, t in module.state_dict().items()}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, monkeypatch_module):
    import yaml

    from mizuoto.nn.transformer.dit import DiT
    from mizuoto.nn.vocoders.vocos import Vocos
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-f5")
    vocab = [" ", *"abcdefghijklmnopqrstuvwxyz.,'", "ni3", "hao3"]
    (root / "vocab.txt").write_text("".join(v + "\n" for v in vocab), encoding="utf-8")
    arch: dict = {"dim": 32, "depth": 2, "heads": 2, "ff_mult": 2, "text_dim": 16, "text_mask_padding": True, "conv_layers": 1, "pe_attn_head": None}
    monkeypatch_module.setitem(f5.ARCH, "v1_base", {"backbone": "DiT", **arch})
    dit = torch.nn.Module()
    dit.add_module("transformer", DiT(**arch, text_num_embeds=len(vocab), mel_dim=100, ops=ops.default))
    state = {f"ema_model.{k}": v for k, v in _random_state(dit, 1).items()}
    save_file(state, str(root / "model_1.safetensors"))
    (root / "vocos").mkdir()
    (root / "vocos" / "config.yaml").write_text(yaml.safe_dump(VOCOS), encoding="utf-8")
    torch.save(_random_state(Vocos(VOCOS, ops.default), 2), root / "vocos" / "pytorch_model.bin")
    return root


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    t = np.arange(24000) / 24000
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(samples.tobytes())
    return path


@pytest.fixture(scope="module")
def loaded(model_dir):
    devices = DeviceService()
    return F5TtsEngine().load(LoadSpec("f5tts/tiny", model_dir, "v1_base", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))


def speak(loaded, reference, seed=3, text="hello there, ni hao.", ctx=None, **params):
    ctx = ctx or RequestContext("r", seed=seed)
    task = SynthesisTask(text=text, voice=VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference.")]), params={"seed": seed, "num_steps": 4, **params})
    with torch.inference_mode():
        return list(loaded.synthesize(task, ctx)), ctx


def test_edge_pad_matches_pydub():
    # audioop.ratecv's frame counts for pydub's 50 ms of 11025 Hz silence
    assert [f5.edge_pad(sr) for sr in (16000, 22050, 24000, 44100, 48000)] == [799, 1101, 1198, 2201, 2395]


def test_clones_and_reports_its_parameters(loaded, reference):
    chunks, ctx = speak(loaded, reference)
    audio = np.concatenate([c.samples for c in chunks])
    assert chunks[-1].final and audio.size > 0 and np.isfinite(audio).all()
    assert ctx.applied["nfe_step"] == 4 and ctx.applied["cfg_strength"] == 2.0


def test_seed_repeats_and_duration_is_honoured(loaded, reference):
    a = speak(loaded, reference, seed=9)[0][0].samples
    b = speak(loaded, reference, seed=9)[0][0].samples
    assert (a == b).all()
    fixed = np.concatenate([c.samples for c in speak(loaded, reference, seed=9, duration=2.0)[0]])
    assert abs(fixed.size / 24000 - 2.0) < 0.1


def test_edit(loaded, reference):
    task = SpeechEditTask(audio_path=str(reference), transcript="a new text.", edits=[EditSpan(start=0.2, end=0.4, text="new")], params={"seed": 1, "num_steps": 4})
    with torch.inference_mode():
        chunks = list(loaded.edit(task, RequestContext("e", seed=1)))
    assert len(chunks) == 1 and chunks[0].samples.size > 0


def test_cancel(loaded, reference):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, reference, ctx=ctx)


def test_vocab_is_read_as_upstream_reads_it(tmp_path):
    from mizuoto.engines.f5tts.text import read_vocab

    (tmp_path / "v.txt").write_bytes(b" \r\na\r\nb")
    assert read_vocab(tmp_path / "v.txt") == {" ": 0, "a": 1, "": 2}  # the last line loses its last character, as upstream's
    json.dumps(read_vocab(Path(f5.__file__).with_name("vocab.txt")))


def _load(model_dir, compile):
    devices = DeviceService()
    runtime = Runtime(devices, ModelManager(devices), RuntimeOptions(compile=compile))
    return F5TtsEngine().load(LoadSpec("f5tts/tiny", model_dir, "v1_base", "cpu", options={}), runtime, RequestContext("load"))


def test_compute_compile_reaches_the_dit(model_dir, loaded, reference):
    assert loaded.compiled is None  # off by default: the eager code path
    on = _load(model_dir, True)
    assert isinstance(on.compiled, Compiled) and on.compiled.module is on.transformer
    calls = []

    def eager(*args, **kwargs):  # the compiled slot, run eagerly: no compilation in this test
        calls.append(1)
        return on.transformer(*args, **kwargs)

    on.compiled._compiled = eager
    a, b = speak(loaded, reference)[0], speak(on, reference)[0]
    assert len(calls) == 4  # one batched (conditional + unconditional) forward per ODE step
    assert all(np.array_equal(x.samples, y.samples) for x, y in zip(a, b))


@pytest.mark.skipif(sys.platform == "win32" or not (shutil.which("g++") or shutil.which("clang++")), reason="the CPU inductor needs a C++ compiler")
def test_compiled_dit_matches_eager(model_dir, loaded, reference):
    on = _load(model_dir, True)
    texts = ("hello there.", "hello there, a longer text to say now.")
    for text in texts:  # a new length reuses the dynamic graph
        a = np.concatenate([c.samples for c in speak(loaded, reference, text=text, num_steps=2)[0]])
        b = np.concatenate([c.samples for c in speak(on, reference, text=text, num_steps=2)[0]])
        assert a.shape == b.shape and np.abs(a - b).max() < 1e-4
    assert on.compiled.failure is None and on.compiled.calls == 2 * len(texts)


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, reference, monkeypatch):
    """On a simulated device, phase by phase, with units streamed: the same audio (and edit)."""
    from tests.engines_native.streaming import check_streamed

    def run(manager):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return np.concatenate([c.samples for c in speak(loaded, reference, ctx=ctx)[0]])

    def edit(manager):
        ctx = RequestContext("e", seed=1)
        ctx.manager = manager
        task = SpeechEditTask(audio_path=str(reference), transcript="a new text.", edits=[EditSpan(start=0.2, end=0.4, text="new")], params={"seed": 1, "num_steps": 4})
        with torch.inference_mode():
            return np.concatenate([c.samples for c in loaded.edit(task, ctx)])

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, edit, monkeypatch)


def test_a_phase_that_runs_out_of_memory_runs_again_identically(loaded, reference, monkeypatch):
    """The transformer runs out of memory once; the phase runs again with the request's draws put
    back: the same audio."""
    from tests.engines_native.streaming import check_oom_retry

    def run(manager, retries=None):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        audio = np.concatenate([c.samples for c in speak(loaded, reference, ctx=ctx)[0]])
        if retries is not None:
            retries.update(ctx.applied.get("oom_retries", {}))
        return audio

    assert check_oom_retry(loaded, run, loaded.transformer.transformer_blocks[1], monkeypatch) == {"transformer": 1}


def test_the_vocoder_falls_back_to_tiles_when_allowed(loaded, reference, monkeypatch):
    """The vocoder runs out of memory on every try; with compute.oom_fallback it decodes in tiles
    (one tile here: the tiny clip is short) and the request says so."""
    from mizuoto.runtime.policy import VramPolicy
    from tests.engines_native.streaming import simulated

    expected = np.concatenate([c.samples for c in speak(loaded, reference)[0]])
    original = loaded.vocoder.decode
    calls = {"n": 0}

    def flaky(mel):
        calls["n"] += 1
        if calls["n"] <= 3:  # the first try and both retries
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original(mel)

    with simulated(loaded, 1.0, monkeypatch) as manager:
        policy = VramPolicy(reserve_bytes=0, host_copies="on", oom_fallback=True)
        manager._policy = lambda: policy
        monkeypatch.setattr(loaded.vocoder, "decode", flaky)
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        got = np.concatenate([c.samples for c in speak(loaded, reference, ctx=ctx)[0]])
    assert np.array_equal(got, expected)
    assert ctx.applied["fallbacks"] == {"vocoder": "in tiles of 256 frames"} and ctx.applied["oom_retries"] == {"vocoder": 2}


def test_tiled_decoding_is_close_to_whole():
    from mizuoto.nn.dsp.tiles import tiled_1d

    gen = torch.Generator().manual_seed(0)
    conv = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 2, 8, stride=4, padding=2), torch.nn.Tanh(), torch.nn.Conv1d(2, 1, 5, padding=2))
    for p in conv.parameters():
        p.data = torch.randn(p.shape, generator=gen) * 0.3
    x = torch.randn(1, 4, 1000, generator=gen)
    with torch.no_grad():
        whole, tiled = conv(x), tiled_1d(conv, x, 4, tile=256, overlap=32)
    assert whole.shape == tiled.shape and float((whole - tiled).abs().max()) < 0.02 * float(whole.abs().max())
