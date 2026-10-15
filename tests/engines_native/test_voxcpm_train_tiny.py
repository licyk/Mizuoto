"""The native VoxCPM trainer on tiny random models of both generations: every stage through mizuoto.training, full and
LoRA, then the exports spoken by the native engine.

With upstream's package and repository at hand (the VoxCPM reference environment; ``VOXCPM_UPSTREAM`` names the
checkout), upstream's own ``train_voxcpm_finetune.train`` runs on the same folders and clips: its losses, learning rates,
gradient norms and saved weights must equal the native trainer's.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

import numpy as np  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.engines.voxcpm import VoxCpmEngine  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, SynthesisTask  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime, RuntimeOptions  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402
from tests.engines_native.test_voxcpm_tiny import build_folder  # noqa: E402

UPSTREAM = Path(os.environ.get("VOXCPM_UPSTREAM", "~/code_workspace/experiment/voxcpm/upstream")).expanduser()
HYPERPARAMS = {"steps": 3, "learning_rate": 1e-3, "batch_size": 2, "grad_accumulation": 2, "warmup_steps": 1, "max_grad_norm": 0.5, "max_batch_tokens": 0}


class Recorder(Emitter):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def of(self, kind):
        return [e for e in self.events if type(e).__name__ == kind]


def _flac(path: Path, seconds: float, rate: int, freq: float) -> Path:
    av = pytest.importorskip("av")
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t) * 32767).astype("<i2")
    with av.open(str(path), "w") as c:
        stream = c.add_stream("flac", rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples[None], format="s16", layout="mono")
        frame.rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            c.mux(packet)
    return path


@pytest.fixture(scope="module", params=[1, 2], ids=["v1", "v2"])
def setup(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"voxcpm-train-v{request.param}")
    (root / "model").mkdir()
    model = build_folder(root / "model", request.param)
    clips = root / "clips"
    clips.mkdir()
    rows = [
        {"id": "a", "audio": str(_flac(clips / "a.flac", 0.6, 24000, 220)), "text": "hello.", "duration": 0.6, "sample_rate": 24000},
        {"id": "b", "audio": str(_flac(clips / "b.flac", 0.9, 32000, 330)), "text": "hello, 你好.", "duration": 0.9, "sample_rate": 32000},
        {"id": "c", "audio": str(_flac(clips / "c.flac", 0.5, 16000, 260)), "text": "a clip.", "duration": 0.5, "sample_rate": 16000},
        {"id": "d", "audio": str(_flac(clips / "d.flac", 0.7, 16000, 200)), "text": "  ", "duration": 0.7, "sample_rate": 16000},
    ]
    for r in rows:
        r.update(speaker="narrator", language="en")
    return {"root": root, "model": model, "rows": rows, "variant": "v1" if request.param == 1 else "v2"}


def _request(setup, stage, method):
    return {
        "trainer": trainer_for("voxcpm"),
        "stage": stage,
        "engine": "voxcpm",
        "method": method,
        "experiment": f"tiny-{method}",
        "workdir": str(setup["root"] / f"work-{method}"),
        "rows": setup["rows"],
        "base_model": {"id": "voxcpm/tiny", "dir": str(setup["model"]), "variant": setup["variant"]},
        "assets": {},
        "previews": True,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {**HYPERPARAMS, **({"lora_rank": 4, "lora_alpha": 8, "lora_proj": True} if method == "lora" else {})},
        "language": "en",
        "results": {},
    }


def _run(setup, stage, method):
    rec = Recorder()
    stage_main(_request(setup, stage, method), rec)
    (result,) = rec.of("ResultEvent")
    return result.data, rec


def _train(setup, method):
    prepared, _ = _run(setup, "prepare", method)
    trained, rec = _run(setup, "train", method)
    again, rec2 = _run(setup, "train", method)
    exported, rec3 = _run(setup, "export", method)
    return {"prepared": prepared, "train": trained, "metrics": rec.of("MetricEvent"), "again": rec2.of("MetricEvent"), "export": exported, "rec": rec3}


@pytest.fixture(scope="module")
def full(setup):
    return _train(setup, "full")


@pytest.fixture(scope="module")
def lora(setup):
    return _train(setup, "lora")


def test_the_engine_names_its_trainer():
    assert trainer_for("voxcpm") == "mizuoto.engines.voxcpm.train:VoxCpmTrainer"


def test_prepare_reads_the_clips_with_text(full):
    assert full["prepared"] == {"clips": 3, "dropped": 0}


def test_training_repeats_and_reports_upstreams_metrics(full):
    metrics = full["metrics"]
    assert [m.step for m in metrics] == [1, 2, 3]
    assert set(metrics[0].values) == {"loss/diff", "loss/stop", "grad_norm"}
    assert all(np.isfinite(v) for m in metrics for v in m.values.values())
    assert [m.values for m in full["again"]] == [m.values for m in metrics]
    assert metrics[0].lr == pytest.approx(1e-3)  # one warm-up step, then the cosine decay
    assert metrics[-1].lr == pytest.approx(0.0, abs=1e-12)


def test_full_export_loads_and_speaks(setup, full):
    folder = Path(full["export"]["artifacts"][0]["folder"])
    weights = load_file(str(folder / "model.safetensors"))
    base = load_file(str(setup["model"] / "model.safetensors"))
    assert set(weights) == set(base) and all(w.dtype == torch.float32 for w in weights.values())
    assert any(not torch.equal(base[k], weights[k]) for k in base)
    assert (folder / "audiovae.pth").is_file() and (folder / "tokenizer.json").is_file()
    (sample,) = [e for e in full["rec"].of("OutputEvent") if e.kind == "sample"]
    assert Path(sample.path).is_file()


def test_lora_trains_only_the_adapters_and_loads_as_upstream_does(setup, lora):
    folder = Path(lora["export"]["artifacts"][0]["folder"])
    adapters = load_file(str(folder / "lora_weights.safetensors"))
    assert adapters and all("lora_" in k for k in adapters)
    assert any(k.startswith("enc_to_lm_proj.") for k in adapters)  # lora_proj
    config = json.loads((folder / "lora_config.json").read_text(encoding="utf-8"))
    assert config["lora_config"]["r"] == 4 and config["base_model"] == "voxcpm/tiny"
    assert (folder / "model.safetensors").read_bytes() == (setup["model"] / "model.safetensors").read_bytes()
    devices = DeviceService()
    task = SynthesisTask(text="hello.", params={"seed": 1}, options={"retry_badcase": False})
    outs = []
    for path in (setup["model"], folder):
        loaded = VoxCpmEngine().load(LoadSpec("voxcpm/x", path, setup["variant"], "cpu"), Runtime(devices, ModelManager(devices), RuntimeOptions()), RequestContext("l"))
        assert (getattr(loaded, "lora", None) is not None) == (path == folder)
        with torch.inference_mode():
            outs.append(np.concatenate([c.samples for c in loaded.synthesize(task, RequestContext("r", seed=1))]))
    assert not np.array_equal(outs[0], outs[1])


# -- against upstream's training script ----------------------------------------------------


def _upstream_train(setup, method, out: Path):
    pytest.importorskip("voxcpm")
    script = UPSTREAM / "scripts" / "train_voxcpm_finetune.py"
    if not script.is_file():
        pytest.skip(f"upstream's repository is not at {UPSTREAM}")
    sys.path.insert(0, str(script.parent))
    import train_voxcpm_finetune as T  # ty: ignore[unresolved-import]

    manifest = out / "train.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps({"audio": r["audio"], "text": r["text"]}) + "\n" for r in setup["rows"] if r["text"].strip()), encoding="utf-8")
    metrics = []
    original = T.TrainingTracker.log_metrics

    def log_metrics(self, values, split="train"):
        metrics.append({"step": self.step + 1, **values})
        return original(self, values, split=split)

    T.TrainingTracker.log_metrics = log_metrics
    hp = HYPERPARAMS
    args = {
        "pretrained_path": str(setup["model"]),
        "train_manifest": str(manifest),
        "sample_rate": 16000,
        "batch_size": hp["batch_size"],
        "grad_accum_steps": hp["grad_accumulation"],
        "num_workers": 0,
        "num_iters": hp["steps"],
        "max_steps": hp["steps"],
        "log_interval": 1,
        "valid_interval": 10**9,
        "save_interval": 10**9,
        "learning_rate": hp["learning_rate"],
        "warmup_steps": hp["warmup_steps"],
        "max_batch_tokens": 0,
        "max_grad_norm": hp["max_grad_norm"],
        "save_path": str(out / "ckpt"),
        "tensorboard": str(out / "tb"),
    }
    if method == "lora":
        args["lora"] = {"enable_lm": True, "enable_dit": True, "enable_proj": True, "r": 4, "alpha": 8, "dropout": 0.0}
    try:
        T.train(**args)
    finally:
        T.TrainingTracker.log_metrics = original
    return metrics, out / "ckpt" / f"step_{hp['steps']:07d}"


@pytest.mark.parametrize("method", ["full", "lora"])
def test_matches_upstreams_training(setup, method, request):
    result = request.getfixturevalue(method)
    expected, checkpoint = _upstream_train(setup, method, setup["root"] / f"upstream-{method}")
    got = result["metrics"]
    assert [m["step"] for m in expected] == [m.step for m in got]
    for e, g in zip(expected, got):
        assert {k: e[k] for k in ("loss/diff", "loss/stop", "grad_norm")} == g.values, e["step"]
        assert e["lr"] == g.lr
    name = "lora_weights.safetensors" if method == "lora" else "model.safetensors"
    upstream = load_file(str(checkpoint / name))
    native = load_file(str(Path(result["export"]["artifacts"][0]["folder"]) / name))
    assert set(upstream) == set(native)
    assert all(torch.equal(upstream[k], native[k]) for k in upstream)


@pytest.mark.parametrize("method", ["full", "lora"])
def test_gradient_checkpointing_trains_identically(setup, method, tmp_path):
    """With every block checkpointed (and LoRA's dropout drawing), the same losses, gradient
    norms and saved weights as without."""
    from safetensors.torch import load_file

    def train(work, **hp):
        request = _request(setup, "prepare", method)
        request["workdir"] = str(work)
        request["hyperparams"] = {**request["hyperparams"], **({"lora_dropout": 0.1} if method == "lora" else {}), **hp}
        stage_main(request, Recorder())
        rec = Recorder()
        stage_main({**request, "stage": "train"}, rec)
        name = "lora_weights.safetensors" if method == "lora" else "model.safetensors"
        return [m.values for m in rec.of("MetricEvent")], load_file(str(work / "out" / name)), rec

    plain, weights, _ = train(tmp_path / "plain")
    checkpointed, weights2, rec = train(tmp_path / "ckpt", gradient_checkpointing="blocks")
    assert any("Gradient checkpointing" in e.message for e in rec.of("LogEvent"))
    assert checkpointed == plain
    assert weights.keys() == weights2.keys() and all(torch.equal(weights[k], weights2[k]) for k in weights)


def test_a_streamed_frozen_base_trains_identically(setup, tmp_path):
    """The lowest preset's streamed LoRA base (here asked for on the CPU): the same losses and adapters."""
    from tests.engines_native.training_memory import check_same_training

    request = _request(setup, "prepare", "lora")
    request["hyperparams"]["lora_dropout"] = 0.1
    assert check_same_training(request, ["prepare", "train"], tmp_path, {"frozen_device": "cpu"}, "stream from RAM") > 0
