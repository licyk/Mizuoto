"""The native OmniVoice trainer on a tiny random model: every stage of both methods
through mizuoto.training, then the exported models spoken by the native engine."""

from __future__ import annotations

import json
import tarfile
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

import numpy as np  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.engines.omnivoice import OmniVoiceEngine  # noqa: E402
from mizuoto.engines.omnivoice import train as ovtrain  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_end|>", "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>", "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]
TOKENIZER = {
    "acoustic_model_config": {"encoder_hidden_size": 2, "downsampling_ratios": [8, 5, 4, 2, 3], "decoder_hidden_size": 32, "upsampling_ratios": [8, 5, 4, 2, 3], "hidden_size": 8},
    "semantic_model_config": {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "conv_dim": [8] * 7,
        "num_conv_pos_embeddings": 4,
        "num_conv_pos_embedding_groups": 2,
    },
    "block_dilations": [1, 1],
    "channel_ratios": [1, 1],
    "strides": [1, 1],
    "kernel_size": 3,
    "unit_kernel_size": 3,
    "codebook_size": 16,
    "codebook_dim": 4,
    "target_bandwidths": [0.2],
    "sample_rate": 24000,
    "semantic_sample_rate": 16000,
    "downsample_factor": 320,
}
STEPS = 3


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
    out = {}
    for name, t in module.state_dict().items():
        if name.endswith("alpha"):
            out[name] = torch.ones_like(t)
        elif t.is_floating_point():
            out[name] = (torch.randn(t.shape, generator=g) * 0.2).to(t.dtype)
        else:
            out[name] = t
    return out


def _tone(seconds: float, rate: int, freq: float) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 32767 / 1.5).astype("<i2")


def _wav(path: Path, seconds: float, rate: int, freq: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(_tone(seconds, rate, freq).tobytes())
    return path


def _flac(path: Path, seconds: float, rate: int, freq: float) -> Path:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        stream = c.add_stream("flac", rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(_tone(seconds, rate, freq)[None], format="s16", layout="mono")
        frame.rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            c.mux(packet)
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from tokenizers.pre_tokenizers import ByteLevel

    from mizuoto.engines.omnivoice.audio_tokenizer import HiggsAudioTokenizer
    from mizuoto.engines.omnivoice.model import OmniVoiceModel
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-omnivoice-train")
    model = root / "model"
    model.mkdir()
    vocab = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    (model / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (model / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(len(vocab) + i): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for i, s in enumerate(SPECIALS)}
    (model / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added, "pad_token": "<|endoftext|>", "eos_token": "<|im_end|>"}), encoding="utf-8")
    (model / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    llm = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "vocab_size": len(vocab) + len(SPECIALS),
    }
    config = {"model_type": "omnivoice", "audio_vocab_size": 17, "audio_mask_id": 16, "num_audio_codebook": 2, "audio_codebook_weights": [2, 1], "llm_config": llm}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    base = _random_state(OmniVoiceModel(config, ops.default), 1)
    save_file(base, str(model / "model.safetensors"))
    (model / "audio_tokenizer").mkdir()
    (model / "audio_tokenizer" / "config.json").write_text(json.dumps(TOKENIZER), encoding="utf-8")
    (model / "audio_tokenizer" / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 24000}), encoding="utf-8")
    state = _random_state(HiggsAudioTokenizer(TOKENIZER, ops.default), 2)
    w = state.pop("semantic_model.encoder.pos_conv_embed.conv.weight")
    state["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = torch.linalg.vector_norm(w, dim=(0, 1), keepdim=True)
    state["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = w
    save_file(state, str(model / "audio_tokenizer" / "model.safetensors"))

    clips = root / "clips"
    clips.mkdir()
    rows = [
        {"id": "a", "audio": str(_wav(clips / "a.wav", 2.4, 24000, 220)), "text": "a clip.", "duration": 2.4, "sample_rate": 24000},
        {"id": "b", "audio": str(_wav(clips / "b.wav", 2.8, 32000, 330)), "text": "another clip, longer.", "duration": 2.8, "sample_rate": 32000},
        {"id": "c", "audio": str(_flac(clips / "c.flac", 3.5, 16000, 260)), "text": "the longest clip of the three.", "duration": 3.5, "sample_rate": 16000},
        {"id": "d", "audio": str(_wav(clips / "d.wav", 2.4, 24000, 200)), "text": "  ", "duration": 2.4, "sample_rate": 24000},
    ]
    for r in rows:
        r.update(speaker="narrator", language="en")
    return {"root": root, "model": model, "rows": rows, "base": base}


def _request(setup, method, stage):
    hyper = {"steps": STEPS, "learning_rate": 1e-3, "batch_tokens": 256, "grad_accumulation": 1, "workers": 2, "preview_text": ""}
    if method == "lora":
        hyper["lora_rank"] = 4
    return {
        "trainer": trainer_for("omnivoice"),
        "stage": stage,
        "engine": "omnivoice",
        "method": method,
        "experiment": f"tiny-{method}",
        "workdir": str(setup["root"] / f"work-{method}"),
        "rows": setup["rows"],
        "base_model": {"id": "omnivoice/tiny", "dir": str(setup["model"]), "variant": "v0"},
        "assets": {},
        "previews": True,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": hyper,
        "language": "en",
        "results": {},
    }


def _run(setup, method, stage):
    rec = Recorder()
    stage_main(_request(setup, method, stage), rec)
    (result,) = rec.of("ResultEvent")
    return result.data, rec


def _train(setup, method):
    tokens, _ = _run(setup, method, "tokens")
    first, rec = _run(setup, method, "train")
    again, rec2 = _run(setup, method, "train")
    exported, rec3 = _run(setup, method, "export")
    return {
        "tokens": tokens,
        "train": first,
        "metrics": rec.of("MetricEvent"),
        "again": [e.values["loss"] for e in rec2.of("MetricEvent")],
        "export": exported,
        "rec": rec3,
        "work": setup["root"] / f"work-{method}",
    }


@pytest.fixture(scope="module")
def lora(setup):
    return _train(setup, "lora")


@pytest.fixture(scope="module")
def full(setup):
    return _train(setup, "full")


def _speak(folder: Path, reference: str):
    devices = DeviceService()
    loaded = OmniVoiceEngine().load(LoadSpec("omnivoice/tiny-trained", folder, "v0", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    voice = VoiceSpec(references=[Reference(audio_path=reference, transcript="a clip.")])
    task = SynthesisTask(text="hello.", language="en", voice=voice, params={"seed": 1, "num_steps": 4}, options={"postprocess_output": False})
    with torch.inference_mode():
        return list(loaded.synthesize(task, RequestContext("r", seed=1)))


def test_the_engine_names_its_trainer():
    assert trainer_for("omnivoice") == "mizuoto.engines.omnivoice.train:OmniVoiceTrainer"
    assert ovtrain.OmniVoiceTrainer.REQUIRES == ()


def test_tokens_are_shards_of_codes(setup, lora):
    assert lora["tokens"] == {"clips": 3, "skipped": 0}
    train = lora["work"] / "tokens" / "train"
    (line,) = (train / "data.lst").read_text(encoding="utf-8").splitlines()
    tar, jsonl, count, seconds = line.split()
    assert count == "3" and float(seconds) == pytest.approx(8.7, abs=0.01)
    labels = [json.loads(x) for x in Path(jsonl).read_text(encoding="utf-8").splitlines()]
    with tarfile.open(tar) as t:
        names = [m.name for m in t]
    # shuffled with seed 42, as extract_audio_tokens' reader does
    assert names == [f"{label['id']}.npy" for label in labels] and sorted(names) == ["a.npy", "b.npy", "c.npy"]
    (sample,) = [s for s in ovtrain.read_shard(ovtrain.Shard(tar, jsonl)) if s["label"]["id"] == "b"]
    # 2.8 s at 24 kHz, padded to whole 960-sample frames: 70 frames of 2 codebooks
    assert sample["audio_tokens"].shape == (2, 70) and sample["audio_tokens"].dtype == torch.int16
    assert sample["label"]["language_id"] == "en" and sample["label"]["num_tokens"] == 70


@pytest.mark.parametrize("method", ["lora", "full"])
def test_train_repeats_under_its_seed(method, request):
    trained = request.getfixturevalue(method)
    assert trained["train"] == {"steps": STEPS, "checkpoint": f"checkpoint-{STEPS}"}
    losses = [m.values["loss"] for m in trained["metrics"]]
    assert [m.step for m in trained["metrics"]] == [1, 2, 3] and all(np.isfinite(losses))
    assert trained["again"] == losses
    # one warm-up step (1 % of 3, rounded up), then half a cosine
    assert [m.lr for m in trained["metrics"]] == pytest.approx([1e-3, 5e-4, 0.0])


def test_lora_checkpoint_and_merge(setup, lora):
    checkpoint = lora["work"] / "out" / f"checkpoint-{STEPS}"
    adapter = load_file(str(checkpoint / "adapter_model.safetensors"))
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 4 and config["lora_alpha"] == 8 and config["peft_type"] == "LORA"
    assert "base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight" in adapter
    assert adapter["base_model.model.llm.layers.1.mlp.down_proj.lora_B.weight"].shape == (32, 4)
    assert {"base_model.model.audio_embeddings.weight", "base_model.model.audio_heads.weight"} <= set(adapter)
    assert len(adapter) == 2 * 7 * 2 + 2
    folder = Path(lora["export"]["artifacts"][0]["folder"])
    merged = load_file(str(folder / "model.safetensors"))
    base = setup["base"]
    assert set(merged) == set(base)
    key = "llm.layers.0.self_attn.q_proj.weight"
    a = adapter["base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight"]
    b = adapter["base_model.model.llm.layers.0.self_attn.q_proj.lora_B.weight"]
    assert torch.equal(merged[key], base[key] + (b @ a) * 2.0) and not torch.equal(merged[key], base[key])
    assert torch.equal(merged["audio_heads.weight"], adapter["base_model.model.audio_heads.weight"])
    # untargeted weights are the base model's
    assert torch.equal(merged["llm.norm.weight"], base["llm.norm.weight"]) and torch.equal(merged["llm.embed_tokens.weight"], base["llm.embed_tokens.weight"])


def test_full_checkpoint_has_upstreams_keys(setup, full):
    checkpoint = full["work"] / "out" / f"checkpoint-{STEPS}"
    weights = load_file(str(checkpoint / "model.safetensors"))
    assert set(weights) == set(setup["base"]) and weights["codebook_layer_offsets"].dtype == torch.int64
    assert not torch.equal(weights["llm.embed_tokens.weight"], setup["base"]["llm.embed_tokens.weight"])
    folder = Path(full["export"]["artifacts"][0]["folder"])
    assert (folder / "model.safetensors").read_bytes() == (checkpoint / "model.safetensors").read_bytes()


@pytest.mark.parametrize("method", ["lora", "full"])
def test_the_preview_and_the_export_speak(setup, method, request):
    trained = request.getfixturevalue(method)
    (sample,) = [e for e in trained["rec"].of("OutputEvent") if e.kind == "sample"]
    assert Path(sample.path).is_file()
    folder = Path(trained["export"]["artifacts"][0]["folder"])
    for name in ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.jinja", "audio_tokenizer/model.safetensors"):
        assert (folder / name).is_file()
    chunks = _speak(folder, setup["rows"][0]["audio"])
    assert chunks and np.isfinite(np.concatenate([c.samples for c in chunks])).all()


def test_pipeline_parts():
    import random

    # webdataset's shuffle: nothing comes out before the buffer holds `initial`, then random picks
    out = list(ovtrain.shuffle_buffer(iter(range(10)), random.Random(0), size=4, initial=3))
    assert sorted(out) == list(range(10)) and out != list(range(10))
    recipe = ovtrain.Recipe(method="full", steps=1, learning_rate=1e-5, batch_tokens=300, accumulation=1, workers=1)
    samples = [{"length": n} for n in (60, 61, 400, 2500, 62, 30)]
    # buckets of 97.5 tokens; a bucket goes once one more of its longest passes 300 tokens; too long or short dropped
    assert [[s["length"] for s in g] for g in ovtrain.length_groups(iter(samples), recipe)] == [[400], [60, 61, 62]]
    batch = ovtrain.collate(
        [
            {"input_ids": torch.ones(2, 3, dtype=torch.long), "labels": torch.ones(2, 3, dtype=torch.long), "audio_mask": torch.ones(3, dtype=torch.bool), "length": 3},
            {"input_ids": torch.ones(2, 1, dtype=torch.long), "labels": torch.ones(2, 1, dtype=torch.long), "audio_mask": torch.ones(1, dtype=torch.bool), "length": 1},
        ],
        pad=7,
    )
    assert batch["input_ids"][1, :, 1:].eq(7).all() and batch["labels"][1, :, 1:].eq(-100).all()
    assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert batch["attention_mask"].shape == (2, 1, 3, 3) and batch["attention_mask"][1, 0].tolist() == [[True, False, False]] * 3


@pytest.mark.parametrize("method", ["full", "lora"])
def test_gradient_checkpointing_trains_identically(setup, method, tmp_path):
    from tests.engines_native.training_memory import check_checkpointing

    request = _request(setup, method, "tokens")
    if method == "lora":
        request["hyperparams"]["lora_dropout"] = 0.1  # the masks come from a private generator: replayed
    assert check_checkpointing(request, ["tokens", "train"], tmp_path) > 0


def test_the_optimizer_in_ram_trains_identically_on_the_cpu(setup, tmp_path):
    from tests.engines_native.training_memory import check_same_training

    assert check_same_training(_request(setup, "full", "tokens"), ["tokens", "train"], tmp_path, {"optimizer_device": "cpu"}, "kept in RAM") > 0
