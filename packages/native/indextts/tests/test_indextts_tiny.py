"""The native IndexTTS engine (packages/native/indextts) on tiny random models.

Real-model parity is checked by scripts/parity (1.5, 2 and 2.5 on the CPU against v1); this covers the code paths
in CI: loading every checkpoint, the text front ends (SentencePiece; tiktoken for 2.5), conditioning, beam
sampling with and without a KV cache, the semantic codecs, the DiT and both BigVGANs, emotion weights, voice
caches, streaming, seeds and cancellation.
"""

from __future__ import annotations

import base64
import json
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentencepiece")
pytest.importorskip("tiktoken")
pytest.importorskip("kaldifst")
pytest.importorskip("mizuoto_indextts")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EmotionSpec, EngineError, LoadSpec, Reference, StyleSpec, SynthesisTask, VoiceSpec, get_engine  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime, RuntimeOptions  # noqa: E402
from mizuoto_indextts import engine as engine_module  # noqa: E402

W2V: dict[str, Any] = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 48,
    "feature_projection_input_dim": 160,
    "layer_norm_eps": 1e-5,
    "conv_depthwise_kernel_size": 31,
    "left_max_position_embeddings": 64,
    "right_max_position_embeddings": 8,
    "position_embeddings_type": "relative_key",
    "add_adapter": False,
    "use_intermediate_ffn_before_adapter": False,
}
CODEC = {"codebook_size": 8192, "hidden_size": 32, "codebook_dim": 4, "vocos_dim": 16, "vocos_intermediate_dim": 32, "vocos_num_layers": 1}
CONDITION = {"output_size": 32, "linear_units": 48, "attention_heads": 2, "num_blocks": 1, "input_layer": "conv2d2", "perceiver_mult": 2}
S2MEL: dict[str, Any] = {
    "preprocess_params": {"sr": 22050, "spect_params": {"n_fft": 1024, "win_length": 1024, "hop_length": 256, "n_mels": 80, "fmin": 0, "fmax": "None"}},
    "dit_type": "DiT",
    "style_encoder": {"dim": 192},
    "length_regulator": {
        "channels": 32,
        "is_discrete": False,
        "in_channels": 32,
        "content_codebook_size": 16,
        "sampling_ratios": [1, 1],
        "vector_quantize": False,
        "n_codebooks": 1,
    },
    "DiT": {
        "hidden_dim": 32,
        "num_heads": 2,
        "depth": 3,
        "in_channels": 80,
        "style_condition": True,
        "final_layer_type": "wavenet",
        "content_dim": 32,
        "content_codebook_size": 16,
        "is_causal": False,
        "long_skip_connection": True,
        "time_as_token": False,
        "style_as_token": False,
        "uvit_skip_connection": True,
    },
    "wavenet": {"hidden_dim": 32, "num_layers": 2, "kernel_size": 5, "dilation_rate": 1, "style_condition": True},
}
BIGVGAN = {
    "resblock": "1",
    "upsample_rates": [8, 8, 4],
    "upsample_kernel_sizes": [16, 16, 8],
    "upsample_initial_channel": 32,
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3]],
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "activation": "snakebeta",
    "snake_logscale": True,
    "num_mels": 80,
}
EMO_NUM = [1, 2, 1, 1, 1, 1, 1, 2]


def _random_state(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in module.state_dict().items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "running_var":
            out[name] = torch.ones_like(t)
        elif leaf in ("running_mean", "num_batches_tracked"):
            out[name] = torch.zeros_like(t)
        elif t.is_floating_point():
            out[name] = (torch.randn(t.shape, generator=g) * 0.05).to(t.dtype)
        else:
            out[name] = t
    return out


def _sentencepiece(path: Path) -> int:
    import sentencepiece as spm

    corpus = path.parent / "corpus.txt"
    corpus.write_text("\n".join(["HELLO THERE THIS IS A TEST", "THE QUICK BROWN FOX", "你 好 世 界", "A B C D E F G", "ONE TWO THREE . , ! ?"] * 20), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(path.with_suffix("")), vocab_size=60, character_coverage=1.0, model_type="bpe", bos_id=0, eos_id=1, unk_id=2, pad_id=-1
    )
    return spm.SentencePieceProcessor(model_file=str(path)).GetPieceSize()


def _assets(root: Path) -> dict[str, str]:
    from mizuoto.nn.codecs.repcodec import RepCodec
    from mizuoto.nn.speaker.campplus_torch import CAMPPlus
    from mizuoto.nn.ssl.w2vbert import W2VBert
    from mizuoto.nn.vocoders.bigvgan import BigVGAN

    w2v = root / "w2v-bert"
    w2v.mkdir()
    (w2v / "config.json").write_text(json.dumps(W2V), encoding="utf-8")
    save_file(_random_state(W2VBert(W2V), 1), str(w2v / "model.safetensors"))
    codec = root / "semantic-codec"
    codec.mkdir()
    save_file(_random_state(RepCodec(**CODEC), 2), str(codec / "model.safetensors"))
    campplus = root / "campplus"
    campplus.mkdir()
    torch.save(_random_state(CAMPPlus(), 3), str(campplus / "campplus_cn_common.bin"))
    bigvgan = root / "bigvgan"
    bigvgan.mkdir()
    (bigvgan / "config.json").write_text(json.dumps(BIGVGAN), encoding="utf-8")
    torch.save({"generator": _random_state(BigVGAN(BIGVGAN), 4)}, str(bigvgan / "bigvgan_generator.pt"))
    return {"indextts/w2v-bert": str(w2v), "indextts/semantic-codec": str(codec), "indextts/campplus": str(campplus), "indextts/bigvgan": str(bigvgan)}


def _gpt_yaml(text_tokens: int, mel_mode: bool = False) -> dict:
    gpt = {
        "model_dim": 64,
        "max_mel_tokens": 120,
        "max_text_tokens": 120,
        "heads": 2,
        "use_mel_codes_as_input": True,
        "mel_length_compression": 1024,
        "layers": 2,
        "number_text_tokens": text_tokens,
        "number_mel_codes": 8194,
        "start_mel_token": 8192,
        "stop_mel_token": 8193,
        "start_text_token": 0,
        "stop_text_token": 1,
        "condition_type": "conformer_perceiver",
        "condition_num_latent": 4,
        "condition_module": CONDITION,
    }
    if not mel_mode:
        gpt["emo_condition_module"] = CONDITION
    return gpt


def _v2_dir(root: Path, v25: bool) -> Path:
    from mizuoto.nn.transformer.seedvc import CFM, DiTConfig, InterpolateRegulator
    from mizuoto_indextts.codec import EnhancedCodec
    from mizuoto_indextts.gpt import GptConfig, UnifiedVoice

    folder = root / ("v2.5" if v25 else "v2")
    folder.mkdir()
    if v25:
        # a byte-level tiktoken vocabulary: every byte its own rank
        lines = [f"{base64.b64encode(bytes([i])).decode()} {i}" for i in range(256)]
        (folder / "multilingual_zh_ja_yue_char_del.tiktoken").write_text("\n".join(lines) + "\n", encoding="utf-8")
        text_tokens = 256 + 2000  # the special tokens follow the 256 ranks
    else:
        text_tokens = _sentencepiece(folder / "bpe.model")
    cfg = {
        "dataset": {"bpe_model": "bpe.model"},
        "gpt": _gpt_yaml(text_tokens),
        "semantic_codec": CODEC,
        "s2mel": S2MEL,
        "gpt_checkpoint": "gpt.pth",
        "w2v_stat": "wav2vec2bert_stats.pt",
        "s2mel_checkpoint": "s2mel.pth",
        "emo_matrix": "feat2.pt",
        "spk_matrix": "feat1.pt",
        "emo_num": EMO_NUM,
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    gpt = UnifiedVoice(GptConfig.from_yaml(cfg["gpt"], "campplus" if v25 else "conformer", W2V["hidden_size"]))
    torch.save(_random_state(gpt, 5), str(folder / "gpt.pth"))
    torch.save({"mean": torch.zeros(32), "var": torch.ones(32)}, str(folder / "wav2vec2bert_stats.pt"))
    lr = S2MEL["length_regulator"]
    net = {
        "cfm": _random_state(CFM(DiTConfig.from_s2mel(S2MEL)), 6),
        "length_regulator": _random_state(InterpolateRegulator(lr["channels"], lr["sampling_ratios"], lr["in_channels"], 16), 7),
    }
    layer = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.Linear(256, 128), torch.nn.Linear(128, 32))
    net["gpt_layer"] = _random_state(layer, 8)
    torch.save({"net": net}, str(folder / "s2mel.pth"))
    torch.save(torch.randn(sum(EMO_NUM), 64, generator=torch.Generator().manual_seed(9)), str(folder / "feat2.pt"))
    torch.save(torch.randn(sum(EMO_NUM), 192, generator=torch.Generator().manual_seed(10)), str(folder / "feat1.pt"))
    if v25:
        torch.save({"model": _random_state(EnhancedCodec(**CODEC), 11)}, str(folder / "codec.pth"))
    return folder


def _v15_dir(root: Path) -> Path:
    from mizuoto.nn.vocoders.bigvgan import SpeakerBigVGAN
    from mizuoto_indextts.gpt import GptConfig, UnifiedVoice

    folder = root / "v1.5"
    folder.mkdir()
    text_tokens = _sentencepiece(folder / "bpe.model")
    vocoder = {**BIGVGAN, "num_mels": 100, "gpt_dim": 64, "speaker_embedding_dim": 32, "cond_d_vector_in_each_upsampling_layer": True, "feat_upsample": False}
    vocoder["upsample_rates"], vocoder["upsample_kernel_sizes"] = [8, 8, 4, 4], [16, 16, 8, 8]
    cfg = {
        "dataset": {"bpe_model": "bpe.model"},
        "gpt": _gpt_yaml(text_tokens, mel_mode=True),
        "bigvgan": vocoder,
        "gpt_checkpoint": "gpt.pth",
        "bigvgan_checkpoint": "bigvgan_generator.pth",
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save({"model": _random_state(UnifiedVoice(GptConfig.from_yaml(cfg["gpt"], "mel")), 12)}, str(folder / "gpt.pth"))
    torch.save({"generator": _random_state(SpeakerBigVGAN(vocoder), 13)}, str(folder / "bigvgan_generator.pth"))
    return folder


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Iterator[dict]:
    root = tmp_path_factory.mktemp("tiny-indextts")
    patch = pytest.MonkeyPatch()
    patch.setattr(engine_module, "W2V_LAYER", 2)
    assets = _assets(root)
    engine = get_engine("indextts")
    devices = DeviceService()
    runtime = Runtime(devices, ModelManager(devices))
    loaded = {}
    try:
        for variant, folder in (("v2", _v2_dir(root, False)), ("v2_5", _v2_dir(root, True)), ("v1_5", _v15_dir(root))):
            assert engine.detect_variant(folder) == variant
            options = {"assets": assets, "use_qwen_emo": False}
            loaded[variant] = engine.load(LoadSpec(f"indextts/tiny-{variant}", folder, variant, "cpu", options=options), runtime, RequestContext("load"))
        yield loaded
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    rate = 16000
    t = np.arange(int(rate * 1.2)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 180 * t) * (t > 0.1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def speak(engine, reference, seed=3, text="Hello there.", style=None, stream=False, ctx=None, voice=None, **params):
    ctx = ctx or RequestContext("r", seed=seed)
    voice = voice or VoiceSpec(references=[Reference(audio_path=str(reference), transcript="")])
    task = SynthesisTask(text=text, language="en", voice=voice, style=style or StyleSpec(), params={"seed": seed, "max_new_tokens": 6, **params}, options={}, stream=stream)
    with torch.inference_mode():
        return list(engine.synthesize(task, ctx)), ctx


@pytest.mark.parametrize("variant", ["v2", "v2_5", "v1_5"])
def test_every_generation_speaks_and_repeats_under_a_seed(models, reference, variant):
    engine = models[variant]
    (a,), _ = speak(engine, reference, seed=5)
    (b,), _ = speak(engine, reference, seed=5)
    assert a.final and a.samples.size and np.isfinite(a.samples).all()
    assert a.sample_rate == engine_module.RATES[variant]
    assert np.array_equal(a.samples, b.samples)


def test_emotion_weights_and_the_voice_cache(models, reference, tmp_path):
    engine = models["v2"]
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="")])
    with torch.inference_mode():
        cache = engine.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    assert cache is not None
    cached = VoiceSpec(references=voice.references, cache=cache.files)
    assert np.array_equal(speak(engine, reference, voice=cached)[0][0].samples, speak(engine, reference)[0][0].samples)
    style = StyleSpec(emotion=EmotionSpec(vector={"happy": 0.6}, strength=0.5))
    (chunk,), ctx = speak(engine, reference, style=style)
    assert chunk.samples.size and ctx.applied["emo_vector"][0] == 0.6 and ctx.applied["emo_alpha"] == 0.5
    with pytest.raises(EngineError):
        speak(engine, reference, style=StyleSpec(emotion=EmotionSpec(text="happy")))  # no emotion model loaded


def test_streaming_and_speed(models, reference):
    chunks, _ = speak(models["v2"], reference, text="One. Two.", stream=True)
    assert chunks[-1].final and chunks[-1].samples.size == 0 and sum(c.samples.size for c in chunks) > 0
    (slow,), ctx = speak(models["v2_5"], reference, speed=0.5)
    assert ctx.applied["duration_factor"] == 2.0 and ctx.applied["lang"] == "en" and slow.samples.size


def test_cancel(models, reference):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(models["v2"], reference, ctx=ctx)


def _spy(compiled, calls: list[str], name: str) -> None:
    """Run a ``Compiled``'s module eagerly through its compiled slot, counting the calls (no compilation)."""

    def eager(*args, **kwargs):
        calls.append(name)
        return compiled.module(*args, **kwargs)

    compiled._compiled = eager


@pytest.mark.parametrize(("variant", "how"), [("v2", "setting"), ("v2", "use_torch_compile"), ("v1_5", "setting")])
def test_compute_compile_reaches_s2mel_and_bigvgan(models, reference, variant, how):
    from dataclasses import replace

    from mizuoto.runtime.compile import Compiled

    eager = models[variant]
    assert eager.compiled_bigvgan is None and (variant == "v1_5" or eager.cfm.compiled_estimator is None)  # off by default
    devices = DeviceService()
    spec = replace(eager.spec, options={**eager.spec.options, "use_torch_compile": True}) if how == "use_torch_compile" else eager.spec
    runtime = Runtime(devices, ModelManager(devices), RuntimeOptions(compile=how == "setting"))
    on: Any = get_engine("indextts").load(spec, runtime, RequestContext("load"))
    calls: list[str] = []
    assert isinstance(on.compiled_bigvgan, Compiled)
    _spy(on.compiled_bigvgan, calls, "bigvgan")
    if variant != "v1_5":
        assert isinstance(on.cfm.compiled_estimator, Compiled)
        _spy(on.cfm.compiled_estimator, calls, "dit")
    (a,), _ = speak(eager, reference)
    (b,), _ = speak(on, reference)
    assert np.array_equal(a.samples, b.samples)
    assert calls.count("bigvgan") == 1 and calls.count("dit") == (0 if variant == "v1_5" else 25)  # 25 Euler steps


@pytest.mark.parametrize("variant", ["v2", "v2_5", "v1_5"])
def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(models, reference, variant, monkeypatch, tmp_path):
    """Conditioning (w2v-BERT, CAM++, codec), GPT, s2mel and BigVGAN each in its phase on a
    simulated device, units streamed: the same audio, cached or not, with emotion weights, streamed and slowed."""
    from tests.engines_native.streaming import check_streamed

    engine = models[variant]

    def run(manager, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return np.concatenate([c.samples for c in speak(engine, reference, ctx=ctx, **kw)[0]])

    streamed = check_streamed(engine, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(engine, lambda m: run(m, text="One. Two.", stream=True), monkeypatch)
    if variant == "v1_5":
        return
    check_streamed(engine, lambda m: run(m, style=StyleSpec(emotion=EmotionSpec(vector={"happy": 0.6}, strength=0.5))), monkeypatch)
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="")])

    def cached(manager):
        ctx = RequestContext("p")
        ctx.manager = manager
        with torch.inference_mode():
            cache = engine.prepare_voice(voice, tmp_path / f"c{id(ctx)}", ctx)
        return run(manager, voice=VoiceSpec(references=voice.references, cache=cache.files))

    check_streamed(engine, cached, monkeypatch)


def test_a_cached_voice_never_places_the_conditioning_models(models, reference, monkeypatch, tmp_path):
    """With the voice cached, w2v-BERT and CAM++ are not needed, so they are never put on the
    device; the GPT, the codec's quantiser (v2), the s2mel and BigVGAN are."""
    from tests.engines_native.streaming import simulated

    engine = models["v2"]
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="")])
    with torch.inference_mode():
        cache = engine.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    cached = VoiceSpec(references=voice.references, cache=cache.files)
    placed: set[str] = set()
    with simulated(engine, 1.0, monkeypatch) as manager:
        original = manager.load

        def load(handles, memory_required=0):
            placed.update(h.name.rsplit(":", 1)[1] for h in handles)
            return original(handles, memory_required)

        monkeypatch.setattr(manager, "load", load)
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        speak(engine, reference, voice=cached, ctx=ctx)
    assert placed == {"gpt", "semantic codec", "s2mel", "bigvgan"}


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(tmp_path_factory, reference):
    """compute.cuda_graphs (mizuoto.runtime.graphs): the GPT's beam steps as graphs around each layer's attention replay
    the eager kernels: the same audio for every generation, fresh loads (the requests that capture) and later ones."""
    from mizuoto.runtime.runtime import RuntimeOptions

    root = tmp_path_factory.mktemp("tiny-indextts-gpu")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "W2V_LAYER", 2)
        assets = _assets(root)
        folders = {"v2": _v2_dir(root, False), "v2_5": _v2_dir(root, True), "v1_5": _v15_dir(root)}
        engine = get_engine("indextts")

        def load(variant: str, graphs: bool):
            devices = DeviceService()
            loaded = engine.load(
                LoadSpec(f"indextts/tiny-{variant}", folders[variant], variant, "cuda", options={"assets": assets, "use_qwen_emo": False}),
                Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
                RequestContext("load"),
            )
            for h in loaded.handles:
                h.load()
            return loaded

        for variant in folders:
            for _ in range(2):
                eager, graphed = load(variant, False), load(variant, True)
                for seed in (3, 4, 3):
                    (a,), _ = speak(eager, reference, seed=seed, ctx=RequestContext("r", seed=seed, device="cuda"), max_new_tokens=40)
                    (b,), ctx = speak(graphed, reference, seed=seed, ctx=RequestContext("r", seed=seed, device="cuda"), max_new_tokens=40)
                    assert ctx.applied.get("cuda_graphs"), variant
                    assert a.samples.shape == b.samples.shape and (a.samples == b.samples).all(), (variant, seed)
                if variant != "v1_5":  # an emotion vector picks its rows by the style on the GPU
                    style = StyleSpec(emotion=EmotionSpec(vector={"happy": 0.6}, strength=0.5))
                    (a,), _ = speak(eager, reference, style=style, ctx=RequestContext("r", seed=3, device="cuda"), max_new_tokens=40)
                    (b,), _ = speak(graphed, reference, style=style, ctx=RequestContext("r", seed=3, device="cuda"), max_new_tokens=40)
                    assert a.samples.size and (a.samples == b.samples).all(), variant


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(tmp_path_factory, reference, monkeypatch, caplog):
    """A step that cannot be captured (here: one that reads a value back) is decoded eagerly: the same audio."""
    from mizuoto.runtime.runtime import RuntimeOptions
    from mizuoto_indextts import gpt

    root = tmp_path_factory.mktemp("tiny-indextts-fallback")
    monkeypatch.setattr(engine_module, "W2V_LAYER", 2)
    assets = _assets(root)
    folder = _v2_dir(root, False)

    def run(graphs: bool) -> np.ndarray:
        devices = DeviceService()
        loaded = get_engine("indextts").load(
            LoadSpec("indextts/tiny-v2", folder, "v2", "cuda", options={"assets": assets, "use_qwen_emo": False}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
        for h in loaded.handles:
            h.load()
        (chunk,), _ = speak(loaded, reference, seed=3, ctx=RequestContext("r", seed=3, device="cuda"), max_new_tokens=30)
        return chunk.samples

    want = run(False)
    attention_in = gpt.GPT2Block.attention_in
    monkeypatch.setattr(gpt.GPT2Block, "attention_in", lambda self, x: (x.sum().item(), attention_in(self, x))[1])
    got = run(True)
    assert "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()


def test_the_gpt_estimate_follows_the_text_and_the_dtype(models):
    """The beam caches are sized by the segment's text (``CODES_PER_TOKEN`` a token, within ``max_new_tokens``) or the
    codes already made, in the dtype the GPT computes in."""
    from mizuoto_indextts.engine import CODES_PER_TOKEN, PROMPT_POSITIONS, gpt_memory

    gpt = models["v2"].gpt
    layers, width, base = len(gpt.gpt.h), gpt.gpt.h[0].attn.dim, 512 * 2**20

    def cache(positions: int, beams: int = 3, element: int = 2) -> int:
        return 2 * beams * 2 * layers * positions * width * element

    assert gpt_memory(gpt, 3, 10, 1500, True) == cache(PROMPT_POSITIONS + 10 + CODES_PER_TOKEN * 10 + 64) + base
    assert gpt_memory(gpt, 3, 100, 1500, True) == cache(PROMPT_POSITIONS + 100 + 1500) + base  # capped
    assert gpt_memory(gpt, 3, 10, 1500, False) == cache(PROMPT_POSITIONS + 10 + CODES_PER_TOKEN * 10 + 64, element=4) + base
    assert gpt_memory(gpt, 1, 10, 1500, True, codes=7) == cache(PROMPT_POSITIONS + 17, beams=1) + base
