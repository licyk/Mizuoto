"""The native Fish Speech LoRA trainer (packages/native/fishspeech) on a tiny random S2 model: every stage through
``mizuoto.training`` in seconds.

Parity with v1's trainer on S2 Pro is checked by scripts/parity/train.py and the seeded drivers; this covers the code
path in CI: clips encoded by the codec in batches, the dataset's samples, the LoRA steps, the merge saved as upstream's
``save_pretrained`` saves a model, the preview, and the exported model loading in the native engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
pytest.importorskip("mizuoto_fishspeech")

from safetensors.torch import load_file  # noqa: E402
from test_fishspeech_tiny import TINY_CODEC, build_model  # noqa: E402

from mizuoto.core.audio.io import encode  # noqa: E402
from mizuoto.core.jobs.events import MetricEvent, OutputEvent, ResultEvent  # noqa: E402
from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec, get_engine  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402
from mizuoto_fishspeech import engine as engine_module  # noqa: E402
from mizuoto_fishspeech.finetune import LORA_CONFIGS, Draws, Group, Samples, Sentence, collate, encode_sentence  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def tiny_codec():
    patch = pytest.MonkeyPatch()
    patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    return build_model(tmp_path_factory.mktemp("tiny-fishspeech"))


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> list[dict]:
    """Three clips as a dataset keeps them: FLAC at their own rate."""
    folder = tmp_path_factory.mktemp("clips")
    rng = np.random.default_rng(0)
    out = []
    for i, (rate, seconds) in enumerate([(16000, 0.6), (32000, 0.4), (44100, 0.5)]):
        t = np.arange(int(rate * seconds)) / rate
        samples = (0.3 * np.sin(2 * np.pi * (150 + 50 * i) * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
        path = folder / f"c{i}.flac"
        encode(samples, rate, path, "flac")
        out.append(
            {
                "id": f"r{i}",
                "audio": str(path),
                "text": f"Clip {i}, <noise> said {{slowly}}.",
                "speaker": "the narrator",
                "language": "en",
                "duration": seconds,
                "sample_rate": rate,
            }
        )
    return out


class Recorder(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def run(stage: str, workdir: Path, base_dir: Path, rows: list[dict], **hp) -> Recorder:
    request = {
        "trainer": trainer_for("fishspeech"),
        "stage": stage,
        "engine": "fishspeech",
        "method": "lora",
        "experiment": "tiny",
        "workdir": str(workdir),
        "rows": rows,
        "base_model": {"id": "fishspeech/tiny", "dir": str(base_dir), "variant": "s2_pro"},
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"steps": 3, "batch_size": 2, "max_length": 256, "learning_rate": 0.05, "preview_text": "Hi.", **hp},
        "language": "en",
    }
    rec = Recorder()
    stage_main(request, rec)
    return rec


@pytest.fixture(scope="module")
def trained(tmp_path_factory, model_dir, rows) -> tuple[Path, dict[str, Recorder]]:
    workdir = tmp_path_factory.mktemp("work")
    return workdir, {stage: run(stage, workdir, model_dir, rows) for stage in ("tokens", "train", "export")}


def test_the_engine_names_its_trainer():
    assert trainer_for("fishspeech") == "mizuoto_fishspeech.train:FishSpeechTrainer"


def test_tokens_groups_the_clips_by_speaker(trained):
    workdir, recs = trained
    assert recs["tokens"].of(ResultEvent)[0].data == {"clips": 3, "seconds": 1.5, "speakers": 1}
    (group,) = json.loads((workdir / "tokens" / "groups.json").read_text(encoding="utf-8"))
    assert group["name"] == "the_narrator" and [s["id"] for s in group["sentences"]] == ["r0", "r1", "r2"]
    assert group["sentences"][0]["texts"] == ["Clip 0, said ."]  # build_dataset's cleaning
    for s, seconds in zip(group["sentences"], (0.6, 0.4, 0.5)):
        codes = np.load(workdir / "tokens" / s["codes"])
        assert codes.shape == (3, -(-round(seconds * 44100) // (4 * 32)))  # 1 semantic + 2 residual codebooks, a frame per 128 samples
        assert codes.min() >= 0 and codes[0].max() < 16 and codes[1:].max() < 8


def test_a_sentence_interleaves_text_and_codes(model_dir):
    from mizuoto.text.tokenize.bpe import load_bpe

    tokenizer = load_bpe(model_dir)
    begin = tokenizer.token_to_id("<|semantic:0|>")
    codes = np.array([[1, 2, 3], [4, 5, 6], [7, 0, 1]])
    tokens, labels = encode_sentence("Hi", codes, tokenizer, begin)
    assert tokens.shape == labels.shape == (4, tokens.shape[1])
    voice = tokenizer.token_to_id("<|voice|>")
    at = int((tokens[0] == voice).nonzero()[0, 0])
    # the voice marker, then the frames: semantic ids in row 0, every codebook in rows 1-3; labels are shifted by one
    assert tokens[0, at + 1 : at + 4].tolist() == [begin + 1, begin + 2, begin + 3] and tokens[1:, at + 1 : at + 4].tolist() == codes.tolist()
    assert labels[0, at : at + 3].tolist() == [begin + 1, begin + 2, begin + 3] and labels[1:, at : at + 3].tolist() == codes.tolist()
    assert labels[0, at + 3] == tokenizer.token_to_id("<|im_end|>") and (labels[0, : at - 1] == -100).all()
    assert (labels[1:, -1] == 0).all() and (tokens[1:, :at] == 0).all()


def test_samples_concatenate_clips_and_the_collator_truncates_and_pads(model_dir):
    from mizuoto.text.tokenize.bpe import load_bpe

    tokenizer = load_bpe(model_dir)
    begin = tokenizer.token_to_id("<|semantic:0|>")
    group = Group("s", [Sentence(["One."], np.ones((3, 4), dtype=np.int64)), Sentence(["Two."], np.ones((3, 6), dtype=np.int64))])
    whole = next(iter(Samples([group], tokenizer, 1000, begin, base_seed=7)))
    one = encode_sentence("One.", group.sentences[0].codes, tokenizer, begin)[0]
    assert whole["tokens"].shape[1] == one.shape[1] + encode_sentence("Two.", group.sentences[1].codes, tokenizer, begin)[0].shape[1]
    # max_length 40 → windows of 2 clips (max_length // 20), so a window may start anywhere
    short = Samples([group], tokenizer, 40, begin, base_seed=7)
    batch = collate([short.augment(), {"tokens": one, "labels": one}], 40, pad_id=0)
    assert batch["inputs"].shape[0] == 2 and batch["inputs"].shape[2] == 40
    padded = batch["attention_masks"][1]
    assert not padded[: one.shape[1]].any() and padded[one.shape[1] :].all()
    assert (batch["labels"][1][:, one.shape[1] :] == -100).all()


def test_draws_advance_the_generator_as_initialisation_does():
    torch.manual_seed(3)
    torch.nn.Embedding(70, 33)
    torch.nn.Linear(70, 33)
    torch.empty(5).normal_()
    expected = torch.get_rng_state()
    torch.manual_seed(3)
    draws = Draws()
    draws.normal(70 * 33)
    draws.linear(70, 33, bias=True)
    draws.normal(5)
    draws.flush()
    assert torch.equal(torch.get_rng_state(), expected)


def test_train_reports_every_step_and_repeats_under_its_seed(trained, tmp_path, model_dir, rows):
    workdir, recs = trained
    metrics = recs["train"].of(MetricEvent)
    assert [m.step for m in metrics] == [1, 2, 3]
    assert all(set(m.values) == {"loss", "base_loss", "semantic_loss", "top_5_accuracy"} for m in metrics)
    for m in metrics:
        assert m.values["loss"] == pytest.approx(m.values["base_loss"] + m.values["semantic_loss"], rel=1e-5)
    assert [m.lr for m in metrics] == pytest.approx([0.0, 0.005, 0.01])  # the warm-up: the first step's rate is 0
    assert metrics[2].values["loss"] < metrics[0].values["loss"]
    (tmp_path / "tokens").symlink_to(workdir / "tokens")
    again = run("train", tmp_path, model_dir, rows).of(MetricEvent)
    assert [m.values for m in again] == [m.values for m in metrics]


def test_only_the_lora_weights_train(trained, model_dir):
    workdir, _ = trained
    lora = load_file(str(workdir / "lora" / "lora.safetensors"))
    # r_8_alpha_16: both transformers' attention and MLP, the three embeddings, the fast output
    assert {k.rsplit(".", 1)[0] for k in lora} >= {
        "embeddings",
        "codebook_embeddings",
        "fast_embeddings",
        "fast_output",
        "layers.0.attention.wqkv",
        "fast_layers.1.feed_forward.w2",
    }
    assert all(v.dtype == torch.float32 for v in lora.values())
    assert lora["layers.0.attention.wqkv.lora_B"].abs().sum() > 0  # moved from zero
    assert LORA_CONFIGS["r_8_alpha_16"].r == lora["layers.0.attention.wqkv.lora_A"].shape[0]


def test_export_is_upstreams_saved_model(trained, model_dir):
    workdir, recs = trained
    (artifact,) = recs["export"].of(ResultEvent)[0].data["artifacts"]
    assert artifact == {"kind": "model", "folder": str(workdir / "export"), "variant": "s2_pro", "title": "tiny"}
    folder = workdir / "export"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "dual_ar" and config["scale_codebook_embeddings"] and config["norm_fastlayer_input"]
    # the semantic range from the tokenizer (the tiny config says 0): 256 byte tokens, then 8 special ones
    assert config["num_codebooks"] == 3 and (config["semantic_begin_id"], config["semantic_end_id"]) == (264, 264 + 4095)
    state = torch.load(folder / "model.pth", weights_only=True)
    base = load_file(str(model_dir / "model.safetensors"))
    assert len(state) == len(base) and all(v.dtype == torch.bfloat16 for v in state.values())
    assert next(iter(state)) == "embeddings.weight" and "fast_output.weight" in state
    # merged: the LoRA layers moved, the norms did not
    assert not torch.equal(state["layers.0.attention.wqkv.weight"], base["text_model.model.layers.0.attention.wqkv.weight"].to(torch.bfloat16))
    assert torch.equal(state["norm.weight"], base["text_model.model.norm.weight"].to(torch.bfloat16))
    assert {"tokenizer.json", "codec.pth"} <= {p.name for p in folder.iterdir()}
    samples = [e for e in recs["export"].of(OutputEvent) if e.kind == "sample"]
    assert len(samples) == 1 and Path(samples[0].path).is_file()


def test_the_exported_model_speaks_natively(trained, rows):
    workdir, _ = trained
    folder = workdir / "export"
    engine = get_engine("fishspeech")
    assert engine.detect_variant(folder) == "s2_pro"
    devices = DeviceService()
    loaded = engine.load(LoadSpec("fishspeech/tiny-lora", folder, "s2_pro", "cpu"), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    try:
        ref = workdir / "preview" / "reference-r0.wav"
        task = SynthesisTask(text="Hello.", language="en", voice=VoiceSpec(references=[Reference(str(ref), "Clip zero.")]), params={"seed": 1, "max_new_tokens": 8})
        with torch.inference_mode():
            (chunk,) = list(loaded.synthesize(task, RequestContext("s", seed=1)))
        assert chunk.sample_rate == 44100 and chunk.samples.size and np.isfinite(chunk.samples).all()
    finally:
        loaded.close()


def test_the_fast_lora_config_trains_only_the_fast_transformer(tmp_path, trained, model_dir, rows):
    workdir, _ = trained
    (tmp_path / "tokens").symlink_to(workdir / "tokens")
    run("train", tmp_path, model_dir, rows, lora="r_32_alpha_16_fast", steps=1)
    lora = load_file(str(tmp_path / "lora" / "lora.safetensors"))
    assert lora and all(k.startswith(("fast_", "fast_layers.")) for k in lora)


def test_lora_layers_built_in_bfloat16_equal_the_float32_copies_cast():
    """bf16-true (a GPU's default): building the LoRA layers' frozen copies in bfloat16 gives the values the float32
    copies have once the model is cast, without the float32 copies."""
    from mizuoto_fishspeech.finetune import LoraConfig, LoraLinear

    gen = torch.Generator().manual_seed(0)
    base = torch.nn.Linear(16, 8).to(torch.bfloat16)
    with torch.no_grad():
        base.weight.copy_(torch.randn(8, 16, generator=gen))
        base.bias.copy_(torch.randn(8, generator=gen))
    config = LoraConfig(r=4, alpha=8, dropout=0.0, targets=("attention",))
    a, b = torch.randn(4, 16, generator=gen), torch.zeros(8, 4)
    cast = LoraLinear(base, config, a.clone(), b.clone()).to(torch.bfloat16)
    direct = LoraLinear(base, config, a.clone(), b.clone(), torch.bfloat16).to(torch.bfloat16)
    for (n, p), (m, q) in zip(cast.named_parameters(), direct.named_parameters(), strict=True):
        assert n == m and p.dtype == q.dtype == torch.bfloat16 and torch.equal(p, q)
