"""Datasets and training, end to end against the dummy engine and its trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mizuoto.core.datasets.models import PrepareRequest, RowPatch
from mizuoto.core.errors import CapabilityError, ValidationError
from mizuoto.core.training.models import ExperimentInput, ExperimentPatch, RunRequest

MODEL = "dummy/default"


def tone(path: Path, seconds: float, sr: int = 24000, gap: float = 0.0) -> Path:
    import numpy as np

    from mizuoto.core.audio.io import encode

    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    if gap:
        mid = len(x) // 2
        x[mid : mid + int(gap * sr)] = 0.0
    return encode(x, sr, path, "wav")


@pytest.fixture
def dataset(services, tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for i in range(4):
        tone(folder / f"c{i}.wav", 2.0 + i)
        (folder / f"c{i}.lab").write_text(f"Clip number {i}.")
    d = services.datasets.create("Tones", language="en", consent="made by the test")
    d, warnings = services.datasets.add_files(d.id, [folder])
    assert not warnings and d.stats.rows == 4 and d.stats.missing_text == 0
    return d


def test_dataset_rows_prepare_and_check(services, dataset, tmp_path):
    ds = services.datasets
    long = tone(tmp_path / "long.wav", 30.0, gap=1.0)
    d, _ = ds.add_files(dataset.id, [long])
    assert d.stats.rows == 5
    page = ds.rows(dataset.id)
    row = next(r for r in page.items if r.source == "long.wav")
    assert row.text == "" and abs(row.duration - 30.0) < 0.05
    before = d.fingerprint
    job = ds.prepare(dataset.id, PrepareRequest(transcribe=False, max_seconds=12.0), wait=True)
    assert job.state == "completed", job.error
    d = ds.get(dataset.id)
    pieces = [r for r in ds.rows(d.id).items if r.source == "long.wav"]
    assert len(pieces) >= 3 and all(r.duration <= 12.01 for r in pieces) and d.fingerprint != before
    edited = ds.update_row(d.id, pieces[0].id, RowPatch(text="A long tone.", excluded=False))
    assert edited.text == "A long tone."
    report = ds.check(d.id)
    assert report.rows == d.stats.rows and report.empty_text and report.histogram
    ds.delete_rows(d.id, [r.id for r in pieces[1:]])
    assert ds.get(d.id).stats.rows == 5
    with pytest.raises(ValidationError):
        ds.create(" ")


def test_import_layouts(services, tmp_path):
    folder = tmp_path / "gsv"
    folder.mkdir()
    tone(folder / "a.wav", 2.0)
    (folder / "x.list").write_text(f"{folder / 'a.wav'}|spk|EN|Hello there.\n")
    d = services.datasets.create("List")
    d, warnings = services.datasets.import_folder(d.id, folder)
    rows = services.datasets.rows(d.id).items
    assert not warnings and rows[0].text == "Hello there." and rows[0].speaker == "spk"


def test_training_runs_stages_and_exports_a_voice(services, dataset):
    tr = services.training
    with pytest.raises(CapabilityError):
        tr.create(ExperimentInput(name="x", base_model=MODEL, dataset=dataset.id, method="nope"))
    exp = tr.create(ExperimentInput(name="tones", base_model=MODEL, dataset=dataset.id, method="speaker", hyperparams={"epochs": 2}))
    assert [s.status for s in exp.stages] == ["pending"] * 3 and exp.hyperparams == {"epochs": 2}
    seen = []
    services.events.subscribe(lambda e: seen.append(e) if type(e).__name__ == "TrainMetricsEvent" else None)
    job = tr.run(exp.key, wait=True)
    assert job.state == "completed", job.error
    exp = tr.get(exp.key)
    assert [s.status for s in exp.stages] == ["done"] * 3
    assert exp.artifacts and exp.artifacts[0].kind == "voice"
    voice = services.voices.get(exp.artifacts[0].id)
    assert voice.weights["dummy"].files["model"] and len(voice.references) >= 1
    assert services.voices.fit(voice.id, MODEL).weights
    points = tr.metrics(exp.key)
    assert len(points) == 2 * 4 and points[-1].values["loss"] < points[0].values["loss"] and seen
    # Changing what a stage uses makes it and the ones after it stale; prepare does not use epochs.
    exp = tr.update(exp.key, ExperimentPatch(hyperparams={"epochs": 3}))
    assert [s.status for s in exp.stages] == ["done", "stale", "stale"]
    with pytest.raises(ValidationError):
        tr.run(exp.key, RunRequest(stages=["export"]))
    job = tr.run(exp.key, wait=True)
    assert job.state == "completed" and job.result["stages"].keys() == {"train", "export"}
    # Running a stage again makes the finished ones after it stale, even with the same inputs.
    assert tr.run(exp.key, RunRequest(stages=["train"]), wait=True).state == "completed"
    assert len([p for p in tr.metrics(exp.key) if p.stage == "train"]) == 3 * 4  # this run's points only
    assert [s.status for s in tr.get(exp.key).stages] == ["done", "done", "stale"]
    assert tr.run(exp.key, wait=True).state == "completed"
    # Editing the dataset makes everything stale.
    row = services.datasets.rows(dataset.id).items[0]
    services.datasets.update_row(dataset.id, row.id, RowPatch(text="Changed."))
    assert [s.status for s in tr.get(exp.key).stages] == ["stale"] * 3


def test_training_failure_and_model_export(services, dataset):
    tr = services.training
    exp = tr.create(ExperimentInput(name="broken", base_model=MODEL, dataset=dataset.id, method="speaker", hyperparams={"fail": "yes"}))
    job = tr.run(exp.key, wait=True)
    assert job.state == "failed" and "on purpose" in job.error.message
    exp = tr.get(exp.key)
    assert [s.status for s in exp.stages] == ["done", "failed", "pending"]
    full = tr.create(ExperimentInput(name="whole", base_model=MODEL, dataset=dataset.id, method="full"))
    job = tr.run(full.key, wait=True)
    assert job.state == "completed", job.error
    model_id = tr.get(full.key).artifacts[0].id
    info = services.models.get(model_id)
    assert info.state == "installed" and info.engine == "dummy" and model_id.startswith("dummy/whole")
    tr.delete(full.key)
    assert all(e.name != "whole" for e in tr.list_experiments())


def test_training_refuses_restricted_outputs(services, dataset):
    rows = services.datasets.rows(dataset.id).items
    folder = services.datasets.folder(services.datasets.get(dataset.id))
    lines = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
    lines[0]["licence_tags"] = ["fish-audio-research"]
    (folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert rows
    with pytest.raises(ValidationError):
        services.training.create(ExperimentInput(name="nope", base_model=MODEL, dataset=dataset.id, method="speaker"))


def test_methods_say_why_not(services, dataset):
    tr = services.training
    methods = {m.id: m for m in tr.methods(MODEL)}
    assert methods["speaker"].available and set(methods["speaker"].method.hyperparams) == {"epochs", "step_ms", "fail"}
    speaker = services.engines.manifest("dummy").training["speaker"]
    speaker.devices = ["cuda"]
    try:
        reason = {m.id: m for m in tr.methods(MODEL)}["speaker"].reason
        assert reason and "CUDA" in reason
        with pytest.raises(CapabilityError):
            tr.create(ExperimentInput(name="gpu-only", base_model=MODEL, dataset=dataset.id, method="speaker"))
        # Without a method, the first one that can train the model is taken.
        assert tr.create(ExperimentInput(name="auto", base_model=MODEL, dataset=dataset.id)).method == "full"
        # A trainer's own DEVICES win over its manifest method's.
        assert tr.trainer("dummy") == "mizuoto.engines.dummy.train:DummyTrainer"
        from mizuoto.engines.dummy.train import DummyTrainer

        DummyTrainer.DEVICES = ("cuda", "cpu")
        try:
            assert {m.id: m for m in tr.methods(MODEL)}["speaker"].available
        finally:
            DummyTrainer.DEVICES = None
    finally:
        speaker.devices = None


def test_memory_estimates_shrink_with_the_presets():
    """Rough, but ordered: each preset needs less, and LoRA less than a full fine-tune."""
    from mizuoto.core.training.estimate import estimate_mb

    full = estimate_mb("voxcpm", "full", 4580, lora=False)
    lora = estimate_mb("voxcpm", "lora", 4580, lora=True)
    assert full["standard"] > full["low"] >= full["lowest"] > 0
    assert lora["standard"] > lora["low"] >= lora["lowest"]
    assert lora["standard"] < full["standard"]
    f5 = estimate_mb("f5tts", "full", 1286, lora=False)
    assert f5["standard"] - f5["low"] > 1286 * 0.9  # the averaged copy leaves the device
    assert estimate_mb("unknown", "full", 100, lora=False)["standard"] > 100


def test_memory_estimates_cover_the_measured_peaks_and_follow_the_batch():
    """The calibration: every measured peak is under its estimate; Fish Speech's default batch (4) does not fit a
    24 GB card, batch 2 does (measured: 15,641 MB); GPT-SoVITS's estimate follows its batch."""
    from mizuoto.core.training.estimate import MEASURED, estimate_mb

    for (engine, method), found in MEASURED.items():
        for m in found:
            est = estimate_mb(engine, method, m.model_mb, "lora" in method)
            assert all(est[p] >= m.peaks[p] for p in m.peaks), (engine, method)
    fish = estimate_mb("fishspeech", "lora", 10498, lora=True)
    assert fish["standard"] > 24 * 1024 > estimate_mb("fishspeech", "lora", 10498, lora=True, hyperparams={"batch_size": 2})["standard"] > 15641
    small = estimate_mb("gptsovits", "speaker", 303, lora=False, hyperparams={"batch_size": 2})["standard"]
    assert 3881 < small < estimate_mb("gptsovits", "speaker", 303, lora=False)["standard"]
    assert estimate_mb("voxcpm", "lora", 1535, lora=True, hyperparams={"batch_size": 0}) == estimate_mb("voxcpm", "lora", 1535, lora=True)


def test_the_warning_before_a_run_names_a_batch_that_fits(services, monkeypatch):
    """No preset helps Fish Speech's LoRA at its recipe's batch (4) on a 24 GB card: the warning names the largest
    batch the estimate fits (3: 2 measured 15.6 GB, 4 ran out of memory)."""
    from types import SimpleNamespace

    tr = services.training
    method = SimpleNamespace(hyperparams={"batch_size": SimpleNamespace(default=4)})
    monkeypatch.setattr(tr._models, "get", lambda _id: SimpleNamespace(size_mb=10498))
    device = SimpleNamespace(id="cuda:0", memory_total_mb=24576, memory_used_mb=1100)
    monkeypatch.setattr(tr._runtime.compute, "info", lambda refresh=False: SimpleNamespace(devices=[device]))
    monkeypatch.setattr(tr, "method", lambda exp, engine: (None, method))
    lines: list[str] = []
    exp = SimpleNamespace(engine="fishspeech", method="lora", base_model="fishspeech/s2-pro", hyperparams={})
    tr._preflight(exp, "cuda:0", SimpleNamespace(log=lines.append))
    assert len(lines) == 1 and "lower the batch_size hyperparameter to 3" in lines[0]
    lines.clear()
    tr._preflight(SimpleNamespace(**{**vars(exp), "hyperparams": {"batch_size": 2}}), "cuda:0", SimpleNamespace(log=lines.append))
    assert lines == []


def test_qwen3tts_speaker_training_refuses_the_0_6b_base_before_a_run(tmp_path):
    """Upstream's speaker recipe needs text embeddings as wide as the talker: the 0.6B Base (2048 against 1024) is
    refused with the reason before a run, the 1.7B's widths pass; other methods and unreadable configs are left to the
    stage."""
    import json

    from mizuoto.engines.qwen3tts.train import Qwen3Trainer

    def model(text: int, width: int):
        folder = tmp_path / f"m{text}-{width}"
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(json.dumps({"talker_config": {"text_hidden_size": text, "hidden_size": width}}))
        return folder

    reason = Qwen3Trainer.refuses("speaker", model(2048, 1024))
    assert reason is not None and "2048" in reason and "1.7B" in reason
    assert Qwen3Trainer.refuses("speaker", model(2048, 2048)) is None
    assert Qwen3Trainer.refuses("other", model(2048, 1024)) is None
    assert Qwen3Trainer.refuses("speaker", tmp_path) is None
