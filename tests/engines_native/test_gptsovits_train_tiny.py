"""The native GPT-SoVITS trainer on tiny random models: every stage through ``mizuoto.training`` in seconds.

Parity with v1 on the real v2Pro model (the prepared files, the GPT's losses step for step and its exported weights,
identical on the CPU) is checked by the experiment's ``debug/train-gptsovits`` drivers and scripts/parity/train.py; this
covers the code paths in CI: clips as FLAC and WAV at 32, 24 and 16 kHz, upstream's file layout (``2-name2text.txt``,
``4-cnhubert``, ``5-wav32k``, ``7-sv_cn``, ``6-name2semantic.tsv``), the GPT's loop with its bucket sampler, ScaledAdam,
the locked schedule, DPO, the half weights as upstream saves them, SoVITS training (``train_s2`` on its tiny model, two batches),
the export and its preview, and the exported voice speaking natively. The text front end is a stand-in (the real one
needs the G2PW, NLTK and dictionary assets).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mizuoto.core.audio.io import encode  # noqa: E402
from mizuoto.core.jobs.events import LogEvent, MetricEvent, OutputEvent, ResultEvent  # noqa: E402
from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.engines.gptsovits import engine as engine_module  # noqa: E402
from mizuoto.engines.gptsovits import s2_modules, train_s2  # noqa: E402
from mizuoto.engines.gptsovits import sv as sv_module  # noqa: E402
from mizuoto.engines.gptsovits import text as text_module  # noqa: E402
from mizuoto.engines.gptsovits.checkpoints import load_checkpoint  # noqa: E402
from mizuoto.engines.gptsovits.text.symbols import SYMBOLS  # noqa: E402
from mizuoto.engines.gptsovits.train_s1 import BucketSampler, Text2SemanticDataset, WarmupCosineLRSchedule, top_k_accuracy  # noqa: E402
from mizuoto.nn.dsp.wav import read_wav, write_wav  # noqa: E402
from mizuoto.runtime import ops  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EngineError, LoadSpec, Reference, SynthesisTask, VoiceSpec, get_engine  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402
from tests.engines_native.test_gptsovits_s2_tiny import _tiny_config as tiny_s2_config  # noqa: E402
from tests.engines_native.test_gptsovits_s2_tiny import _write as write_s2  # noqa: E402
from tests.engines_native.test_gptsovits_tiny import FakeText, _bert, _hubert, _t2s  # noqa: E402
from tests.engines_native.test_gptsovits_tiny import write_sovits as _write_sovits  # noqa: E402

PHONES = [s for s in SYMBOLS if s.isalpha() and s.isascii()]
SV_CONFIG: dict = {"num_blocks": (1, 1, 1, 1), "m_channels": 8}


class FakeFrontend:
    """``TextFrontend.clean_text``'s interface: a phone per letter, no word2ph (as English)."""

    @classmethod
    def from_assets(cls, assets: Any, **_: Any) -> FakeFrontend:
        return cls()

    def clean_text(self, text: str, language: str) -> tuple[list[str], None, str]:
        if "fail" in text:
            raise ValueError("cannot read this")
        return [PHONES[ord(c) % len(PHONES)] for c in text if c.isalnum()], None, text


@pytest.fixture(scope="module")
def patched():
    patch = pytest.MonkeyPatch()
    patch.setattr(text_module, "TextFrontend", FakeFrontend)
    patch.setattr(text_module, "TextPreprocessor", FakeText)
    patch.setattr(text_module, "load_bert_model", lambda encoders, device, dtype=None: _bert())
    patch.setattr(sv_module, "load_sv", lambda path, **kw: _real_load_sv(path, **SV_CONFIG))
    # SoVITS training on its tiny model, two batches an epoch (tests/engines_native/test_gptsovits_s2_tiny.py)
    patch.setattr(train_s2, "s2_config", tiny_s2_config)
    patch.setattr(train_s2, "epoch_batches", lambda sampler, epoch: sampler.batches(epoch)[:2])
    yield
    patch.undo()


_real_load_sv = sv_module.load_sv


@pytest.fixture(scope="module")
def files(tmp_path_factory, patched) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("tiny-gptsovits-train")
    models = {}
    for variant, tag in (("v2", None), ("v2pro", b"05")):
        gpt, sovits = engine_module.BASE_WEIGHTS[variant]
        folder = root / f"{variant}-base"
        _t2s(folder / gpt, 1)
        (folder / sovits).parent.mkdir(parents=True, exist_ok=True)
        if variant == "v2":  # the base SoVITS training starts from: train_s2's tiny model
            config = tiny_s2_config("v2")
            config["model"]["version"] = "v2"
            torch.manual_seed(0)
            write_s2(folder / sovits, s2_modules.build_synthesizer(config).state_dict(), config)
        else:
            _write_sovits(folder / sovits, "v2Pro", tag, seed=1)
        models[variant] = folder
    assets = {}
    for asset in (engine_module.ENCODERS, engine_module.G2PW, engine_module.NLTK, engine_module.OPEN_JTALK):
        assets[asset] = root / "_assets" / asset
        assets[asset].mkdir(parents=True)
    assets["gptsovits/s2d-v2"] = root / "_assets" / "s2d-v2"
    write_s2(assets["gptsovits/s2d-v2"] / "gsv-v2final-pretrained" / "s2D2333k.pth", s2_modules.build_discriminator("v2").state_dict())
    encoders = assets[engine_module.ENCODERS]
    _hubert(encoders / "chinese-hubert-base")
    sv = sv_module.ERes2NetV2(baseWidth=24, scale=4, expansion=4, **SV_CONFIG)
    ops.init_random(sv, torch.Generator().manual_seed(2), scale=0.1)
    for name, buf in sv.named_buffers():
        if name.endswith("running_var"):
            buf.fill_(1.0)
    (encoders / "sv").mkdir()
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in sv.state_dict().items()}, encoders / engine_module.SV_FILE)
    clips = root / "clips"
    clips.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, (rate, seconds, kind) in enumerate([(32000, 1.2, "flac"), (24000, 3.5, "flac"), (16000, 1.5, "wav"), (32000, 1.0, "flac")]):
        t = np.arange(int(rate * seconds)) / rate
        samples = (0.3 * np.sin(2 * np.pi * (150 + 50 * i) * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
        path = clips / f"c{i}.{kind}"
        if kind == "flac":
            encode(samples, rate, path, "flac")
        else:
            write_wav(path, samples, rate)
        text = f"Clip number {i}, said slowly." if i < 3 else "This one will fail."
        rows.append({"id": f"r{i}", "audio": str(path), "text": text, "speaker": "s", "language": "en", "duration": seconds, "sample_rate": rate})
    rows.append({"id": "r9", "audio": str(path), "text": "Elle parle français.", "speaker": "s", "language": "fr", "duration": 1.0, "sample_rate": 32000})
    return {"models": models, "assets": {k: str(v) for k, v in assets.items()}, "rows": rows}


class Recorder(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def result(self) -> dict:
        return self.of(ResultEvent)[0].data


def run(stage: str, workdir: Path, files: dict, variant: str = "v2", **hp: Any) -> Recorder:
    request = {
        "trainer": trainer_for("gptsovits"),
        "stage": stage,
        "engine": "gptsovits",
        "method": "speaker",
        "experiment": "tiny",
        "workdir": str(workdir),
        "rows": files["rows"],
        "base_model": {"id": f"gptsovits/{variant}-base", "dir": str(files["models"][variant]), "variant": variant},
        "assets": files["assets"],
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"sovits_epochs": 1, "gpt_epochs": 2, "batch_size": 4, "dpo": False, **hp},
        "results": {},
        "language": "en",
    }
    rec = Recorder()
    stage_main(request, rec)
    return rec


@pytest.fixture(scope="module")
def trained(tmp_path_factory, files) -> tuple[Path, dict[str, Recorder]]:
    workdir = tmp_path_factory.mktemp("work")
    return workdir, {stage: run(stage, workdir, files) for stage in ("text", "speech", "semantic", "sovits", "gpt", "export")}


def test_the_engine_names_its_trainer():
    assert trainer_for("gptsovits") == "mizuoto.engines.gptsovits.train:GptSovitsTrainer"


def test_text_writes_upstreams_lists(trained):
    workdir, recs = trained
    exp = workdir / "exp"
    assert recs["text"].result == {"clips": 3, "left_out": 2}
    assert any(e.message.startswith("c3.flac is left out: ValueError") for e in recs["text"].of(LogEvent))
    inputs = (exp / "inputs.list").read_text(encoding="utf-8").splitlines()
    assert len(inputs) == 4 and inputs[0].endswith("|s|en|Clip number 0, said slowly.")
    lines = (exp / "2-name2text.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[0] for line in lines] == ["c0.flac", "c1.flac", "c2.wav"]
    name, phones, word2ph, norm = lines[0].split("\t")
    assert phones.split(" ") == FakeFrontend().clean_text(norm, "en")[0] and word2ph == "None"
    assert not any((exp / "3-bert").iterdir())  # BERT features are for Chinese only


def test_speech_writes_features_and_32_khz_audio(trained):
    workdir, recs = trained
    exp = workdir / "exp"
    assert recs["speech"].result == {"clips": 4}
    for name, seconds in (("c0.flac", 1.2), ("c1.flac", 3.5), ("c2.wav", 1.5)):
        samples, rate = read_wav(exp / "5-wav32k" / name)  # a WAV under the clip's own name, as upstream
        assert rate == 32000 and abs(len(samples) - 32000 * seconds) <= 1
        peak = np.abs(samples).max() * 32768
        assert 0.95 * 0.5 * 32768 < peak < 32768  # scaled to 0.95 × 0.5 of full scale plus half of itself
        ssl = torch.load(exp / "4-cnhubert" / f"{name}.pt", weights_only=True)
        assert ssl.shape == (1, 768, int(seconds * 50) - 1) and ssl.dtype == torch.float32
    assert not (exp / "7-sv_cn").exists()  # v2 has no speaker embeddings
    with wave.open(str(exp / "5-wav32k" / "c0.flac")) as w:
        assert (w.getnchannels(), w.getsampwidth()) == (1, 2)


def test_speech_embeds_speakers_for_v2pro(tmp_path, files):
    run("text", tmp_path, files, "v2pro")
    assert run("speech", tmp_path, files, "v2pro").result == {"clips": 4}
    emb = torch.load(tmp_path / "exp" / "7-sv_cn" / "c1.flac.pt", weights_only=True)
    assert emb.shape == (1, 8 * 8 * 4 * 10) and torch.isfinite(emb).all()
    run("semantic", tmp_path, files, "v2pro")
    assert (tmp_path / "exp" / "6-name2semantic.tsv").read_text(encoding="utf-8").startswith("item_name\tsemantic_audio\n")


def test_semantic_writes_the_base_codes(trained):
    workdir, recs = trained
    lines = (workdir / "exp" / "6-name2semantic.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "item_name\tsemantic_audio" and recs["semantic"].result == {"clips": 4}
    name, codes = lines[1].split("\t")
    assert name == "c0.flac" and len(codes.split(" ")) == 29 and all(0 <= int(c) < 1024 for c in codes.split(" "))


def test_sovits_hands_its_weights_on(trained):
    workdir, recs = trained
    result = recs["sovits"].result
    assert result["weights"] == "tiny_e1_s2.pth" and result["steps"] == 2 and result["epochs"] == 1
    assert Path(result["path"]) == workdir / "weights" / "sovits" / "tiny_e1_s2.pth" and result["files"] == [result["path"]]
    metrics = recs["sovits"].of(MetricEvent)
    assert [m.step for m in metrics] == [0, 1] and all(np.isfinite(m.values["loss_mel"]) for m in metrics)


def test_sovits_needs_the_pretrained_discriminator(tmp_path, trained, files):
    workdir, _ = trained
    (tmp_path / "exp").symlink_to(workdir / "exp")
    without = {**files, "assets": {k: v for k, v in files["assets"].items() if k != "gptsovits/s2d-v2"}}
    with pytest.raises(EngineError, match="s2d-v2"):
        run("sovits", tmp_path, without)


def test_gpt_trains_as_upstream(trained, files):
    workdir, recs = trained
    metrics = recs["gpt"].of(MetricEvent)
    # 3 clips repeated 33 times, batches of 2 (4 halved in full precision): 50 batches an epoch
    assert len(metrics) == 100 and [m.step for m in metrics] == list(range(1, 101)) and {m.epoch for m in metrics} == {1, 2}
    assert all(set(m.values) == {"total_loss", "top_3_acc"} and np.isfinite(m.values["total_loss"]) for m in metrics)
    # the schedule: ScaledAdam's own 0.01 at its first step (batch index 4), 0.002 ever after
    assert [m.lr for m in metrics[:6]] == [1e-05] * 4 + [0.01, 0.01] and {m.lr for m in metrics[9:]} == {0.002}
    result = recs["gpt"].result
    assert result["weights"] == "tiny-e2.ckpt" and result["steps"] == 100
    ckpt = load_checkpoint(result["path"])
    base = load_checkpoint(files["models"]["v2"] / engine_module.BASE_WEIGHTS["v2"][0])
    assert list(ckpt) == ["weight", "config", "info"] and ckpt["info"] == "GPT-e2"
    assert list(ckpt["weight"]) == list(base["weight"]) and {v.dtype for v in ckpt["weight"].values()} == {torch.float16}
    assert ckpt["config"]["train"]["batch_size"] == 2 and ckpt["config"]["model"]["hidden_dim"] == 32 and ckpt["config"]["data"]["max_sec"] == 54
    moved = [k for k in base["weight"] if not torch.equal(ckpt["weight"][k], base["weight"][k].half())]
    assert "model.ar_predict_layer.weight" in moved and "model.bert_proj.weight" not in moved  # zero BERT features: no gradient
    # the same stage again: the same losses
    again = run("gpt", workdir, files).of(MetricEvent)
    assert [m.values for m in again] == [m.values for m in metrics]


def test_dpo_trains_on_half_batches(tmp_path, trained, files):
    workdir, _ = trained
    (tmp_path / "exp").symlink_to(workdir / "exp")
    rec = run("gpt", tmp_path, files, gpt_epochs=1, batch_size=8, dpo=True)
    metrics = rec.of(MetricEvent)
    assert len(metrics) == 50 and all(np.isfinite(m.values["total_loss"]) for m in metrics)  # batches of 4 // 2


def test_export_hands_over_the_pair_with_a_preview(trained):
    workdir, recs = trained
    result = recs["export"].result
    (artifact,) = result["artifacts"]
    assert artifact["kind"] == "voice_weights" and set(artifact["files"]) == {"gpt", "sovits"}
    assert Path(artifact["files"]["gpt"]) == workdir / "export" / "tiny.ckpt" and Path(artifact["files"]["sovits"]).is_file()
    assert result["gpt"] == "tiny-e2.ckpt" and result["sovits"] == "tiny_e1_s2.pth"
    samples = [e for e in recs["export"].of(OutputEvent) if e.kind == "sample"]
    assert len(samples) == 1 and Path(samples[0].path).is_file(), [e.message for e in recs["export"].of(LogEvent)]


def test_the_exported_voice_speaks_natively(trained, files):
    workdir, _ = trained
    devices = DeviceService()
    loaded = get_engine("gptsovits").load(
        LoadSpec("gptsovits/tiny", files["models"]["v2"], "v2", "cpu", options={"assets": files["assets"]}), Runtime(devices, ModelManager(devices)), RequestContext("load")
    )
    ref = workdir / "preview" / "reference-r1.wav"
    try:

        def speak(weights: dict | None) -> np.ndarray:
            task = SynthesisTask(
                text="Hello there.", language="en", voice=VoiceSpec(references=[Reference(str(ref), "Clip number one.", "en")], weights=weights), params={"seed": 3}
            )
            with torch.inference_mode():
                return np.concatenate([c.samples for c in loaded.synthesize(task, RequestContext("s", seed=3))])

        base = speak(None)
        own = speak({"gpt": str(workdir / "export" / "tiny.ckpt"), "sovits": str(workdir / "export" / "tiny.pth")})
        assert base.size > 0 and own.size > 0 and not np.array_equal(base, own)
    finally:
        loaded.close()


def test_the_sampler_and_schedule_are_upstreams(tmp_path):
    exp = tmp_path
    names = [f"n{i}" for i in range(6)]  # n5 says 2 phones in 4 s: under 3 phones a second, left out
    phones = [10 + 5 * i for i in range(5)] + [2]
    (exp / "2-name2text.txt").write_text("".join(f"{n}\t{' '.join(['a'] * p)}\tNone\tt\n" for n, p in zip(names, phones)), encoding="utf-8")
    (exp / "6-name2semantic.tsv").write_text(
        "item_name\tsemantic_audio\n" + "".join(f"{n}\t{' '.join(['1'] * (20 + 30 * i))}\n" for i, n in enumerate(names[:5])) + f"n5\t{' '.join(['1'] * 100)}\n", encoding="utf-8"
    )
    data = Text2SemanticDataset(exp / "2-name2text.txt", exp / "6-name2semantic.tsv")
    assert data.clips == 5 and len(data) == 100 and data.skipped["phone_rate"] == 1  # repeated max(2, 100 // 5) times
    sampler = BucketSampler(data, batch_size=3)
    order = sampler.indices(0)
    assert sorted(order) == list(range(100)) and order == sampler.indices(0) and order != sampler.indices(1)
    batch = data.collate([data[i] for i in (0, 4)])
    assert batch["phoneme_ids"].shape == (2, 30) and batch["semantic_ids"].shape == (2, 140) and batch["semantic_ids"][0, 20:].eq(1024).all()
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    sched = WarmupCosineLRSchedule(opt, 1e-5, 0.01, 1e-4, 2000, 40000)
    assert sched.get_last_lr() == [1e-5]
    sched.step()
    assert sched.get_last_lr() == [0.01] and opt.param_groups[0]["lr"] == 0.002
    logits = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]], [[4.0], [3.0], [2.0], [1.0], [0.0]]])  # (batch, classes, time)
    assert top_k_accuracy(logits, torch.tensor([[2], [3]]), ignore_index=4) == 0.5
    assert top_k_accuracy(logits, torch.tensor([[4], [4]]), ignore_index=4) == 0.0
