"""The native F5-TTS trainer on a tiny random model: every stage through mizuoto.training,
then the exported model spoken by the native engine."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("pypinyin")
pytest.importorskip("rjieba")
pytest.importorskip("yaml")

import numpy as np  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.engines.f5tts import F5TtsEngine  # noqa: E402
from mizuoto.engines.f5tts import engine as f5  # noqa: E402
from mizuoto.engines.f5tts import train as f5train  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402

VOCOS = {
    "feature_extractor": {
        "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
        "init_args": {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 100, "padding": "center"},
    },
    "backbone": {"class_path": "vocos.models.VocosBackbone", "init_args": {"input_channels": 100, "dim": 16, "intermediate_dim": 32, "num_layers": 1}},
    "head": {"class_path": "vocos.heads.ISTFTHead", "init_args": {"dim": 16, "n_fft": 1024, "hop_length": 256, "padding": "center"}},
}
ARCH: dict[str, Any] = {"backbone": "DiT", "dim": 32, "depth": 2, "heads": 2, "ff_mult": 2, "text_dim": 16, "text_mask_padding": True, "conv_layers": 1, "pe_attn_head": None}
BASE_STEP = 1000


class Recorder(Emitter):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def of(self, kind):
        return [e for e in self.events if type(e).__name__ == kind]


def _random_state(module, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(t.shape, generator=g) * 0.02).to(t.dtype) if t.is_floating_point() else t for k, t in module.state_dict().items()}


def _wav(path: Path, seconds: float, rate: int, freq: float) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def _flac(path: Path, seconds: float, rate: int, freq: float) -> Path:
    av = pytest.importorskip("av")
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with av.open(str(path), "w") as c:
        stream = c.add_stream("flac", rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples[None], format="s16", layout="mono")
        frame.rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            c.mux(packet)
    return path


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def setup(tmp_path_factory, monkeypatch_module):
    import yaml

    from mizuoto.nn.transformer.dit import DiT
    from mizuoto.nn.vocoders.vocos import Vocos
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-f5-train")
    model = root / "model"
    model.mkdir()
    vocab = [" ", *"abcdefghijklmnopqrstuvwxyz.,'"]
    (model / "vocab.txt").write_text("".join(v + "\n" for v in vocab), encoding="utf-8")
    monkeypatch_module.setitem(f5.ARCH, "v1_base", ARCH)
    net = torch.nn.Module()
    net.add_module("transformer", DiT(**{k: v for k, v in ARCH.items() if k != "backbone"}, text_num_embeds=len(vocab), mel_dim=100, ops=ops.default))
    state = {f"ema_model.{k}": v for k, v in _random_state(net, 1).items()}
    save_file({**state, "initted": torch.tensor(True), "step": torch.tensor(BASE_STEP)}, str(model / "model_1000.safetensors"))
    (model / "vocos").mkdir()
    (model / "vocos" / "config.yaml").write_text(yaml.safe_dump(VOCOS), encoding="utf-8")
    torch.save(_random_state(Vocos(VOCOS, ops.default), 2), model / "vocos" / "pytorch_model.bin")
    clips = root / "clips"
    clips.mkdir()
    rows = [
        {"id": "a", "audio": str(_wav(clips / "a.wav", 1.0, 24000, 220)), "text": "a clip.", "duration": 1.0, "sample_rate": 24000},
        {"id": "b", "audio": str(_wav(clips / "b.wav", 1.5, 32000, 330)), "text": "another clip, longer.", "duration": 1.5, "sample_rate": 32000},
        {"id": "c", "audio": str(_flac(clips / "c.flac", 3.2, 32000, 260)), "text": "the longest clip of the three.", "duration": 3.2, "sample_rate": 32000},
        {"id": "d", "audio": str(_wav(clips / "d.wav", 1.0, 24000, 200)), "text": "  ", "duration": 1.0, "sample_rate": 24000},
    ]
    for r in rows:
        r.update(speaker="narrator", language="en")
    return {"root": root, "model": model, "rows": rows, "base_keys": set(state) | {"initted", "step"}}


def _request(setup, stage, results=None):
    return {
        "trainer": trainer_for("f5tts"),
        "stage": stage,
        "engine": "f5tts",
        "method": "full",
        "experiment": "tiny",
        "workdir": str(setup["root"] / "work"),
        "rows": setup["rows"],
        "base_model": {"id": "f5tts/tiny", "dir": str(setup["model"]), "variant": "v1_base"},
        "assets": {},
        "previews": True,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"epochs": 2, "learning_rate": 1e-3, "batch_frames": 320, "max_samples": 64, "grad_accumulation": 1, "warmup_updates": 0},
        "language": "en",
        "results": results or {},
    }


def _run(setup, stage):
    rec = Recorder()
    stage_main(_request(setup, stage), rec)
    (result,) = rec.of("ResultEvent")
    return result.data, rec


@pytest.fixture(scope="module")
def trained(setup):
    prepared, _ = _run(setup, "prepare")
    first, rec = _run(setup, "train")
    losses = [e.values["loss"] for e in rec.of("MetricEvent")]
    again, rec2 = _run(setup, "train")
    exported, rec3 = _run(setup, "export")
    return {
        "prepared": prepared,
        "train": first,
        "losses": losses,
        "metrics": rec.of("MetricEvent"),
        "again": [e.values["loss"] for e in rec2.of("MetricEvent")],
        "export": exported,
        "rec": rec3,
    }


def test_the_engine_names_its_trainer():
    assert trainer_for("f5tts") == "mizuoto.engines.f5tts.train:F5Trainer"
    assert f5train.F5Trainer.REQUIRES == ()


def test_prepare_resamples_and_tokenises(setup, trained):
    assert trained["prepared"] == {"clips": 3, "seconds": 5.7}
    data = setup["root"] / "work" / "data"
    with wave.open(str(data / "wavs" / "b.wav")) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 36000
    assert (data / "vocab.txt").read_text(encoding="utf-8") == (setup["model"] / "vocab.txt").read_text(encoding="utf-8")


def test_train_follows_the_frame_budget_and_repeats_under_its_seed(trained):
    # 94 + 141 frames fit 320, the 300-frame clip makes its own batch: 2 updates per epoch
    assert trained["train"] == {"updates": 4, "epochs": 2}
    assert [(m.epoch, m.step) for m in trained["metrics"]] == [(1, 1), (1, 2), (2, 3), (2, 4)]
    assert all(np.isfinite(trained["losses"]))
    assert trained["again"] == trained["losses"]
    assert trained["metrics"][0].lr == pytest.approx(1e-3)  # one warm-up update, then the decay


def test_export_keeps_upstreams_keys_and_the_ema_step(setup, trained):
    folder = Path(trained["export"]["artifacts"][0]["folder"])
    assert trained["export"]["update"] == 4 and trained["export"]["artifacts"][0]["variant"] == "v1_base"
    weights = load_file(str(folder / "model_4.safetensors"))
    assert set(weights) == setup["base_keys"]
    assert int(weights["step"]) == BASE_STEP + 4 and bool(weights["initted"])
    base = load_file(str(setup["model"] / "model_1000.safetensors"))
    # step 1000 is an EMA update (one every 10 steps): the averaged weights moved
    moved = [k for k in base if k.startswith("ema_model.") and not torch.equal(base[k], weights[k])]
    assert moved
    assert (folder / "vocab.txt").is_file() and (folder / "vocos" / "config.yaml").is_file()


def test_the_preview_and_the_exported_model_speak(setup, trained):
    (sample,) = [e for e in trained["rec"].of("OutputEvent") if e.kind == "sample"]
    assert Path(sample.path).is_file()
    folder = Path(trained["export"]["artifacts"][0]["folder"])
    devices = DeviceService()
    loaded = F5TtsEngine().load(LoadSpec("f5tts/tiny-trained", folder, "v1_base", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    task = SynthesisTask(text="hello.", voice=VoiceSpec(references=[Reference(audio_path=setup["rows"][0]["audio"], transcript="a clip.")]), params={"seed": 1, "num_steps": 4})
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, RequestContext("r", seed=1)))
    assert chunks and np.isfinite(np.concatenate([c.samples for c in chunks])).all()


def test_dynamic_batches_and_their_shuffle():
    # sorted by length, packed under the budget, a clip over it dropped
    durations = [1.0, 0.5, 40.0, 0.7, 0.4]  # frames 93.75, 46.9, 3750, 65.6, 37.5
    assert f5train.dynamic_batches(durations, 150, 64) == [[4, 1, 3], [0]]
    assert f5train.dynamic_batches(durations, 150, 2) == [[4, 1], [3], [0]]
    assert sorted(map(tuple, f5train.shuffled([[1], [2], [3]], 666))) == [(1,), (2,), (3,)]


def test_gradient_checkpointing_trains_identically(setup, tmp_path):
    from tests.engines_native.training_memory import check_checkpointing

    assert check_checkpointing(_request(setup, "prepare"), ["prepare", "train"], tmp_path) > 0


def test_the_optimizer_in_ram_trains_identically_on_the_cpu(setup, tmp_path):
    from tests.engines_native.training_memory import check_same_training

    assert check_same_training(_request(setup, "prepare"), ["prepare", "train"], tmp_path, {"optimizer_device": "cpu"}, "kept in RAM") > 0
