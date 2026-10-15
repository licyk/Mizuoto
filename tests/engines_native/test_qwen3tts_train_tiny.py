"""The native Qwen3-TTS trainer on a tiny random Base model: every stage through ``mizuoto.training`` in seconds.

Parity with v1's trainer on the real model is checked by scripts/parity/train.py; this covers the code path in CI:
FLAC and WAV clips resampled to 24 kHz and encoded, the SFT loop with batches, padding and gradient accumulation, the
checkpoint in upstream's layout, the preview, and the exported CustomVoice model speaking its new preset speaker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from mizuoto.core.audio.io import encode  # noqa: E402
from mizuoto.core.jobs.events import MetricEvent, OutputEvent, ResultEvent  # noqa: E402
from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.engines.qwen3tts import Qwen3TtsEngine  # noqa: E402
from mizuoto.nn.dsp.wav import read_wav, write_wav  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import trainer_for  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>"]


def _byte_vocab() -> dict[str, int]:
    from tokenizers.pre_tokenizers import ByteLevel

    return {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}


def _random_state(module, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in module.state_dict().items():
        if name.endswith(("cluster_usage", "initialized")):
            out[name] = torch.ones_like(t)
        elif name.endswith("embed_sum"):
            out[name] = torch.randn(t.shape, generator=g)
        else:
            out[name] = (torch.randn(t.shape, generator=g) * 0.1).to(t.dtype) if t.is_floating_point() else t
    return out


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory) -> Path:
    from mizuoto.engines.qwen3tts.codec import SpeechTokenizer
    from mizuoto.engines.qwen3tts.config import load_config
    from mizuoto.engines.qwen3tts.speaker import SpeakerEncoder
    from mizuoto.engines.qwen3tts.talker import Talker
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-qwen3tts-base")
    vocab = _byte_vocab()
    base = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(base + i): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for i, s in enumerate(SPECIALS)}
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
    small = {"rms_norm_eps": 1e-6, "rope_theta": 1000000}
    talker = {
        **small,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 3072,
        "text_vocab_size": base + len(SPECIALS),
        "text_hidden_size": 32,
        "num_code_groups": 16,
        "codec_bos_id": 2149,
        "codec_eos_token_id": 2150,
        "codec_pad_id": 2148,
        "codec_think_id": 2154,
        "codec_nothink_id": 2155,
        "codec_think_bos_id": 2156,
        "codec_think_eos_id": 2157,
        "codec_language_id": {"english": 2050, "chinese": 2055},
        "spk_id": {},
        "spk_is_dialect": {},
        "code_predictor_config": {
            **small,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 2048,
        },
    }
    config = {
        "tts_model_type": "base",
        "tts_model_size": "1b7",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_bos_token_id": base + 4,
        "tts_eos_token_id": base + 5,
        "tts_pad_token_id": base + 3,
        "transformers_version": "4.57.3",
        "speaker_encoder_config": {"enc_dim": 32, "sample_rate": 24000, "enc_channels": [16, 16, 16, 16, 48], "enc_attention_channels": 8, "enc_se_channels": 8},
        "talker_config": talker,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"max_new_tokens": 8}), encoding="utf-8")
    (root / "model.json").write_text("{}", encoding="utf-8")  # Mizuoto's sidecar: not copied into the checkpoint
    cfg = load_config(root)
    model = torch.nn.Module()
    model.talker = Talker(cfg.talker, ops.default)
    model.speaker_encoder = SpeakerEncoder(cfg.speaker_encoder, ops.default)
    save_file({k: v.to(torch.bfloat16) for k, v in _random_state(model, 1).items()}, str(root / "model.safetensors"))
    attention = {"num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 1}
    codec_config = {
        "encoder_valid_num_quantizers": 16,
        "encode_downsample_rate": 1920,
        "decode_upsample_rate": 1920,
        "output_sample_rate": 24000,
        "encoder_config": {
            **attention,
            "_frame_rate": 12.5,
            "audio_channels": 1,
            "codebook_size": 2048,
            "compress": 2,
            "dilation_growth_rate": 2,
            "head_dim": 8,
            "hidden_size": 16,
            "intermediate_size": 32,
            "kernel_size": 7,
            "last_kernel_size": 3,
            "norm_eps": 1e-5,
            "num_filters": 4,
            "num_quantizers": 16,
            "num_residual_layers": 1,
            "num_semantic_quantizers": 1,
            "residual_kernel_size": 3,
            "rope_theta": 10000.0,
            "sampling_rate": 24000,
            "upsampling_ratios": [8, 6, 5, 4],
            "vector_quantization_hidden_dimension": 8,
        },
        "decoder_config": {
            **attention,
            "latent_dim": 16,
            "codebook_dim": 8,
            "codebook_size": 2048,
            "decoder_dim": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "head_dim": 4,
            "num_quantizers": 16,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "sliding_window": 72,
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
        },
    }
    (root / "speech_tokenizer").mkdir()
    (root / "speech_tokenizer" / "config.json").write_text(json.dumps(codec_config), encoding="utf-8")
    codec = SpeechTokenizer(codec_config, ops.default, with_encoder=True)
    save_file(_random_state(codec, 2), str(root / "speech_tokenizer" / "model.safetensors"))
    return root


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> list[dict]:
    """Three clips as a dataset keeps them: FLAC at their own rate (and one WAV)."""
    folder = tmp_path_factory.mktemp("clips")
    rng = np.random.default_rng(0)
    out = []
    for i, (rate, seconds, kind) in enumerate([(16000, 1.2, "flac"), (24000, 0.9, "flac"), (22050, 1.5, "wav")]):
        t = np.arange(int(rate * seconds)) / rate
        samples = (0.3 * np.sin(2 * np.pi * (150 + 50 * i) * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
        path = folder / f"c{i}.{kind}"
        if kind == "flac":
            encode(samples, rate, path, "flac")
        else:
            write_wav(path, samples, rate)
        out.append({"id": f"r{i}", "audio": str(path), "text": f"Clip number {i}, said slowly.", "speaker": "s", "language": "en", "duration": seconds, "sample_rate": rate})
    return out


class Recorder(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def run(stage: str, workdir: Path, base_dir: Path, rows: list[dict], previews: bool = True) -> Recorder:
    request = {
        "trainer": trainer_for("qwen3tts"),
        "stage": stage,
        "engine": "qwen3tts",
        "method": "speaker",
        "experiment": "tiny",
        "workdir": str(workdir),
        "rows": rows,
        "base_model": {"id": "qwen3tts/tiny", "dir": str(base_dir), "variant": "base"},
        "previews": previews,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"epochs": 2, "batch_size": 2, "grad_accumulation": 2, "speaker": "Tiny Voice"},
        "language": "en",
    }
    rec = Recorder()
    stage_main(request, rec)
    return rec


@pytest.fixture(scope="module")
def trained(tmp_path_factory, base_dir, rows) -> tuple[Path, dict[str, Recorder]]:
    workdir = tmp_path_factory.mktemp("work")
    return workdir, {stage: run(stage, workdir, base_dir, rows) for stage in ("prepare", "train", "export")}


def test_the_engine_names_its_trainer():
    assert trainer_for("qwen3tts") == "mizuoto.engines.qwen3tts.train:Qwen3Trainer"


def test_prepare_writes_24_khz_clips_and_their_codes(trained):
    workdir, recs = trained
    assert recs["prepare"].of(ResultEvent)[0].data["clips"] == 3
    lines = [json.loads(line) for line in (workdir / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and len({line["ref_audio"] for line in lines}) == 1
    for line in lines:
        wav, sr = read_wav(line["audio"])
        assert sr == 24000
        codes = np.asarray(line["audio_codes"])
        assert codes.shape == (-(-len(wav) // 1920), 16) and codes.min() >= 0 and codes.max() < 2048


def test_train_reports_every_step_and_repeats_under_its_seed(trained, tmp_path, base_dir, rows):
    workdir, recs = trained
    metrics = recs["train"].of(MetricEvent)
    assert [m.step for m in metrics] == [1, 2, 3, 4] and [m.epoch for m in metrics] == [1, 1, 2, 2]
    assert all(set(m.values) == {"loss", "talker", "sub_talker"} and np.isfinite(m.values["loss"]) for m in metrics)
    for m in metrics:
        assert m.values["loss"] == pytest.approx(m.values["talker"] + 0.3 * m.values["sub_talker"], rel=1e-5)
    # the same stage again, from the same prepared data: the same losses
    (tmp_path / "data").symlink_to(workdir / "data")
    again = run("train", tmp_path, base_dir, rows).of(MetricEvent)
    assert [m.values for m in again] == [m.values for m in metrics]


def test_the_checkpoint_is_a_custom_voice_model_in_upstreams_layout(trained, base_dir):
    workdir, _ = trained
    checkpoint = workdir / "checkpoint"
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["tts_model_type"] == "custom_voice"
    assert config["talker_config"]["spk_id"] == {"tiny_voice": 3000} and config["talker_config"]["spk_is_dialect"] == {"tiny_voice": False}
    assert not (checkpoint / "model.json").exists() and (checkpoint / "speech_tokenizer" / "model.safetensors").is_file()
    state = load_file(str(checkpoint / "model.safetensors"))
    base = load_file(str(base_dir / "model.safetensors"))
    assert set(state) == {k for k in base if not k.startswith("speaker_encoder.")}
    assert all(v.dtype == torch.float32 for v in state.values())
    # trained weights moved; the text projection (unused by upstream's loss) did not
    assert not torch.equal(state["talker.codec_head.weight"], base["talker.codec_head.weight"].float())
    assert torch.equal(state["talker.text_projection.linear_fc1.weight"], base["talker.text_projection.linear_fc1.weight"].float())
    assert state["talker.model.codec_embedding.weight"][3000].abs().sum() > 0


def test_export_hands_over_the_model_with_a_preview(trained):
    workdir, recs = trained
    result = recs["export"].of(ResultEvent)[0].data
    (artifact,) = result["artifacts"]
    assert artifact["kind"] == "model" and artifact["variant"] == "custom_voice" and artifact["speakers"] == ["tiny_voice"]
    assert Path(artifact["folder"]) == workdir / "export"
    samples = [e for e in recs["export"].of(OutputEvent) if e.kind == "sample"]
    assert len(samples) == 1 and Path(samples[0].path).is_file()


def test_the_exported_model_speaks_its_new_speaker_natively(trained):
    workdir, _ = trained
    folder = workdir / "export"
    engine = Qwen3TtsEngine()
    assert engine.detect_variant(folder) == "custom_voice"
    devices = DeviceService()
    loaded = engine.load(LoadSpec("qwen3tts/tiny-voice", folder, "custom_voice", "cpu"), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    try:
        assert loaded.describe().speakers == ["tiny_voice"]
        task = SynthesisTask(text="Hello there.", language="en", voice=VoiceSpec(preset="tiny_voice"), params={"seed": 1})
        with torch.inference_mode():
            chunks = list(loaded.synthesize(task, RequestContext("s", seed=1)))
        assert len(chunks) == 1 and chunks[0].sample_rate == 24000 and chunks[0].samples.shape[0] % 1920 == 0
    finally:
        loaded.close()


def test_base_clone_runs_identically_with_its_weights_placed_by_phase_and_streamed(base_dir, rows, monkeypatch, tmp_path):
    """The voice prompt (codec encoder + speaker encoder), the talker and the decoder, each in
    its phase on a simulated device, units streamed: the same audio and the same cached prompt."""
    from mizuoto.runtime.engine import Reference
    from tests.engines_native.streaming import check_streamed

    devices = DeviceService()
    loaded = Qwen3TtsEngine().load(LoadSpec("qwen3tts/tiny-base", base_dir, "base", "cpu"), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    wav = rows[2]["audio"]
    voice = VoiceSpec(references=[Reference(audio_path=wav, transcript=rows[2]["text"])])

    def run(manager):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        task = SynthesisTask(text="Hello there.", language="en", voice=voice, params={"seed": 3, "max_new_tokens": 6})
        with torch.inference_mode():
            return np.concatenate([c.samples for c in loaded.synthesize(task, ctx)])

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0

    def prepared(manager):
        ctx = RequestContext("p", seed=3)
        ctx.manager = manager
        with torch.inference_mode():
            cache = loaded.prepare_voice(voice, tmp_path / f"c{id(ctx)}", ctx)
        assert cache is not None
        data = torch.load(cache.files["prompt"], weights_only=True)[0]
        return data["ref_code"], data["ref_spk_embedding"]

    check_streamed(loaded, prepared, monkeypatch)


def test_gradient_checkpointing_trains_identically(base_dir, rows, tmp_path):
    from tests.engines_native.training_memory import check_checkpointing

    request = {
        "trainer": trainer_for("qwen3tts"),
        "engine": "qwen3tts",
        "method": "speaker",
        "experiment": "tiny",
        "rows": rows,
        "base_model": {"id": "qwen3tts/tiny", "dir": str(base_dir), "variant": "base"},
        "previews": False,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"epochs": 2, "batch_size": 2, "grad_accumulation": 2, "speaker": "Tiny Voice"},
        "language": "en",
    }
    assert check_checkpointing(request, ["prepare", "train"], tmp_path) > 0


def test_the_optimizer_in_ram_trains_identically_on_the_cpu(base_dir, rows, tmp_path):
    from tests.engines_native.training_memory import check_same_training

    request = {
        "trainer": trainer_for("qwen3tts"),
        "engine": "qwen3tts",
        "method": "speaker",
        "experiment": "tiny",
        "rows": rows,
        "base_model": {"id": "qwen3tts/tiny", "dir": str(base_dir), "variant": "base"},
        "previews": False,
        "device": "cpu",
        "precision": "auto",
        "hyperparams": {"epochs": 2, "batch_size": 2, "grad_accumulation": 2, "speaker": "Tiny Voice"},
        "language": "en",
    }
    assert check_same_training(request, ["prepare", "train"], tmp_path, {"optimizer_device": "cpu"}, "kept in RAM") > 0
