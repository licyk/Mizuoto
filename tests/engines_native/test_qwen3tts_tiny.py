"""The native Qwen3-TTS engine on a tiny random model: no weights needed.

Real-model parity is checked by scripts/parity (identical audio to v1 on the CPU); this keeps the code
path covered in CI: loading from a model folder, prompts, the talker and code-predictor loops, the
codec, seeds and cancellation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.qwen3tts import Qwen3TtsEngine  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, StyleSpec, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>"]


def _byte_vocab() -> dict[str, int]:
    from tokenizers.pre_tokenizers import ByteLevel

    return {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}


def _random_state(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in module.state_dict().items():
        if name.endswith("cluster_usage"):
            out[name] = torch.ones_like(t)
        else:
            out[name] = (torch.randn(t.shape, generator=g) * 0.05).to(t.dtype) if t.is_floating_point() else t
    return out


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    from mizuoto.engines.qwen3tts.codec import SpeechTokenizer
    from mizuoto.engines.qwen3tts.config import load_config
    from mizuoto.engines.qwen3tts.talker import Talker
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-qwen3tts")
    vocab = _byte_vocab()
    base = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(base + i): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for i, s in enumerate(SPECIALS)}
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
    talker = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
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
        "codec_language_id": {"english": 2050, "chinese": 2055, "sichuan_dialect": 2062},
        "spk_id": {"alto": 3000, "sichuan": 3001},
        "spk_is_dialect": {"alto": False, "sichuan": "sichuan_dialect"},
        "code_predictor_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "vocab_size": 2048,
        },
    }
    config = {
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_bos_token_id": base + 4,
        "tts_eos_token_id": base + 5,
        "tts_pad_token_id": base + 3,
        "transformers_version": "4.57.3",
        "talker_config": talker,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"max_new_tokens": 12}), encoding="utf-8")
    cfg = load_config(root)
    model = torch.nn.Module()
    model.talker = Talker(cfg.talker, ops.default)
    save_file(_random_state(model, 1), str(root / "model.safetensors"))
    codec_config = {
        "encoder_valid_num_quantizers": 16,
        "encode_downsample_rate": 1920,
        "decode_upsample_rate": 1920,
        "output_sample_rate": 24000,
        "decoder_config": {
            "latent_dim": 16,
            "codebook_dim": 8,
            "codebook_size": 2048,
            "decoder_dim": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
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
    codec = SpeechTokenizer(codec_config, ops.default, with_encoder=False)
    save_file(_random_state(codec, 2), str(root / "speech_tokenizer" / "model.safetensors"))
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    engine = Qwen3TtsEngine()
    assert engine.detect_variant(model_dir) == "custom_voice"
    devices = DeviceService()
    runtime = Runtime(devices, ModelManager(devices))
    return engine.load(LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cpu"), runtime, RequestContext("load"))


def speak(loaded, seed: int | None = 3, speaker: str = "alto", ctx: RequestContext | None = None):
    ctx = ctx or RequestContext("r", seed=seed)
    task = SynthesisTask(text="Hello there.", language="en", voice=VoiceSpec(preset=speaker), style=StyleSpec(instruct="calm"), params={"seed": seed})
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, ctx))
    return chunks, ctx


def test_a_streamed_request_gives_the_audio_piece_by_piece(loaded):
    """task.stream: the frames are decoded as they come (codec.DecodeStream), in several pieces, the last final, and
    together they are the request's audio without streaming."""
    want = speak(loaded)[0][0].samples
    task = SynthesisTask(text="Hello there.", language="en", voice=VoiceSpec(preset="alto"), style=StyleSpec(instruct="calm"), params={"seed": 3}, stream=True)
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, RequestContext("s", seed=3)))
    assert len(chunks) >= 2 and chunks[-1].final and not any(c.final for c in chunks[:-1])
    got = np.concatenate([c.samples for c in chunks])
    assert got.shape == want.shape and np.allclose(got, want, atol=1e-6)


def test_the_decode_stream_gives_the_chunked_decode_frame_by_frame(loaded):
    """DecodeStream over upstream's chunk grid (300 frames, 25 of context): the samples chunked_decode gives, frame by
    frame and across chunk boundaries, without the prefix's (the reference's codes when cloning in context)."""
    from mizuoto.engines.qwen3tts.codec import DecodeStream

    decoder = loaded.codec.decoder
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, decoder.config["codebook_size"], (700, 16), generator=g)
    prefix = 37
    with torch.inference_mode():
        want = decoder.chunked_decode(codes.T.unsqueeze(0))[0, 0, prefix * 1920 :].float()
        stream = DecodeStream(decoder, codes[:prefix])
        pieces = []
        for n, frame in enumerate(codes[prefix:], 1):
            stream.push(frame)
            if n % 13 == 0:
                pieces.append(stream.audio())
        pieces.append(stream.audio())
    got = torch.cat(pieces)
    assert got.shape == want.shape and torch.allclose(got, want, atol=1e-6)


def test_describe_lists_the_speakers(loaded):
    d = loaded.describe()
    assert d.sample_rate == 24000 and d.speakers == ["alto", "sichuan"] and "en" in d.languages


def test_speaks_frames_of_1920_samples_and_reports_its_parameters(loaded):
    chunks, ctx = speak(loaded)
    assert len(chunks) == 1 and chunks[0].sample_rate == 24000
    n = chunks[0].samples.shape[0]
    assert n % 1920 == 0 and 0 < n <= 12 * 1920
    assert ctx.applied["max_new_tokens"] == 12 and ctx.applied["language"] == "english" and ctx.applied["instruct"] == "calm"


def test_a_seed_repeats_and_another_differs(loaded):
    a = speak(loaded, seed=5)[0][0].samples
    b = speak(loaded, seed=5)[0][0].samples
    c = speak(loaded, seed=6)[0][0].samples
    assert (a == b).all()
    assert a.shape != c.shape or not (a == c).all()


def test_dialect_speakers_and_batches(loaded):
    task = SynthesisTask(text="", texts=["One.", "Two, longer."], language="zh", voice=VoiceSpec(preset="sichuan"), params={"seed": 1})
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, RequestContext("b", seed=1)))
    assert [c.segment for c in chunks] == [0, 1]


def test_cancel_stops_generation(loaded):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, ctx=ctx)


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, monkeypatch):
    """On a simulated device, phase by phase, with units streamed: the same audio."""
    from tests.engines_native.streaming import check_streamed

    def run(manager):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return speak(loaded, ctx=ctx)[0][0].samples

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0


def test_a_phase_that_runs_out_of_memory_runs_again_identically(loaded, monkeypatch):
    """A talker layer runs out of memory once; the phase runs again with the request's sampling
    draws put back: the same audio."""
    from tests.engines_native.streaming import check_oom_retry

    def run(manager, retries=None):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        audio = speak(loaded, ctx=ctx)[0][0].samples
        if retries is not None:
            retries.update(ctx.applied.get("oom_retries", {}))
        return audio

    assert check_oom_retry(loaded, run, loaded.talker.model.layers[0], monkeypatch) == {"talker": 1}


def test_the_static_kv_cache_gives_the_same_audio_on_the_cpu(model_dir, loaded):
    """compute.kv_cache = static (preallocated buffers): identical here on the CPU."""
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()
    static = Qwen3TtsEngine().load(
        LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cpu"), Runtime(devices, ModelManager(devices), RuntimeOptions(kv_cache="static")), RequestContext("load")
    )
    assert static.kv_cache == "static"  # ty: ignore[unresolved-attribute]
    assert (speak(static)[0][0].samples == speak(loaded)[0][0].samples).all()


def test_the_talker_estimate_follows_the_text_not_max_new_tokens(loaded):
    """The talker's working memory is sized by the request's text, capped by ``max_new_tokens``; on the 1.7B's sizes a
    sentence sets aside under 400 MB, where the worst case for 8192 frames, doubled, was 2.4 GB."""
    from types import SimpleNamespace

    from mizuoto.engines.qwen3tts.engine import FRAMES_PER_TOKEN, PROMPT_POSITIONS

    def ids(n: int) -> torch.Tensor:
        return torch.zeros(1, n, dtype=torch.long)

    short = loaded._talker_positions([ids(40)], None, None, None, 8192)
    assert short == PROMPT_POSITIONS + 40 + FRAMES_PER_TOKEN * 40 + 64
    assert loaded._talker_positions([ids(400)], None, None, None, 100) == PROMPT_POSITIONS + 400 + 100
    prompt = {"ref_code": [torch.zeros(90, 16)]}
    assert loaded._talker_positions([ids(40)], [ids(12)], [ids(30)], prompt, 8192) == short + 12 + 30 + 90
    assert loaded._talker_positions([ids(40), ids(10)], None, None, None, 8192) == short  # the longest row
    base = 256 * 2**20
    assert loaded._generation_memory(8, short) - base == 8 * (loaded._generation_memory(1, short) - base)

    real = object.__new__(type(loaded))  # the 1.7B's talker sizes, in bf16
    real.config = SimpleNamespace(talker=SimpleNamespace(decoder=SimpleNamespace(num_layers=28, num_kv_heads=8, head_dim=128)))
    real.dtype, real.kv_cache = torch.bfloat16, "dynamic"
    assert real._generation_memory(1, short) < 400 * 2**20


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(model_dir):
    """compute.cuda_graphs (mizuoto.runtime.graphs): the captured frame replays the eager kernels, so the audio is the
    same, request after request (the graphs are reused), batched with padding too, and seeds still repeat."""
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()

    def load(graphs: bool):
        loaded = Qwen3TtsEngine().load(
            LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cuda"), Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)), RequestContext("load")
        )
        for h in loaded.handles:
            h.load()
        return loaded

    def run(loaded, seed: int):
        ctx = RequestContext("r", seed=seed, device="cuda")
        one = speak(loaded, seed=seed, ctx=ctx)[0][0].samples
        task = SynthesisTask(text="", texts=["One.", "Two, a longer one."], language="zh", voice=VoiceSpec(preset="sichuan"), params={"seed": seed})
        batch_ctx = RequestContext("b", seed=seed, device="cuda")
        with torch.inference_mode():
            batch = [c.samples for c in loaded.synthesize(task, batch_ctx)]
        return [one, *batch], ctx.applied.get("cuda_graphs"), batch_ctx.applied.get("cuda_graphs")

    eager, graphed = load(False), load(True)
    for seed in (3, 4, 3):
        want, *_ = run(eager, seed)
        got, single, batched = run(graphed, seed)
        assert single and batched
        assert all(a.shape == b.shape and (a == b).all() for a, b in zip(want, got, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_over_a_streamed_talker_decode_exactly_as_eagerly(model_dir, monkeypatch):
    """A talker partly on the GPU, its other layers streaming from page-locked memory, still decodes with CUDA graphs
    (ModelHandle.graphable): each captured step copies the streamed layers into the graph's memory before reading them,
    so the audio equals the wholly loaded eager decoder's, request after request, though the copies are
    dropped between requests and the memory they leave filled with NaNs. A capture that reused a copy made before it
    would read those."""
    from mizuoto.engines.qwen3tts.engine import LoadedQwen3Tts
    from mizuoto.runtime import units
    from mizuoto.runtime.runtime import RuntimeOptions
    from mizuoto.runtime.stream import PinBudget, Streamer

    monkeypatch.setattr(units, "UNIT_MIN_BYTES", 0)  # the tiny talker in units at all
    monkeypatch.setattr(units, "TINY_BYTES", 1)
    devices = DeviceService()

    def load(graphs: bool, streamed: bool):
        loaded = Qwen3TtsEngine().load(
            LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cuda"), Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)), RequestContext("load")
        )
        assert isinstance(loaded, LoadedQwen3Tts)
        streamer = Streamer("cuda", pin=PinBudget(1 << 30)) if streamed else None
        for h in loaded.handles:
            if h is loaded.h_talker and streamer is not None:
                h.load(h.minimum_bytes(1), streamer)
            else:
                h.load()
        return loaded, streamer

    eager, _ = load(False, False)
    graphed, streamer = load(True, True)
    assert streamer is not None and not graphed.h_talker.loaded and graphed.h_talker.graphable()
    assert sum(not u.resident for u in graphed.h_talker.units) > 1
    for seed in (3, 4, 3):
        want = speak(eager, seed=seed, ctx=RequestContext("r", seed=seed, device="cuda"))[0][0].samples
        ctx = RequestContext("r", seed=seed, device="cuda")
        before = streamer.copied_bytes
        got = speak(graphed, seed=seed, ctx=ctx)[0][0].samples
        assert ctx.applied.get("cuda_graphs")
        assert want.shape == got.shape and (want == got).all()
        assert streamer.copied_bytes - before >= graphed.h_talker.streamed_bytes  # the replays' copies are counted
        # drop the copies and fill the memory they leave with NaNs, held through the next request: a graph that reads a
        # copy made outside it (in the eager run before its capture, on the graphs' stream) reads these
        streamer.release_all(graphed.h_talker)
        host = [graphed.h_talker._host[(id(m), n)] for u in graphed.h_talker.units if not u.resident for m, n in u.slots]
        with torch.cuda.stream(graphed._graphs.stream):
            spoiled = [torch.full(t.shape, float("nan"), dtype=t.dtype, device="cuda") for _ in range(4) for t in host]  # noqa: F841
        torch.cuda.synchronize()


def test_fused_projections_compute_the_same_maths():
    """QwenDecoder.fuse (fast mode): q/k/v and gate/up as one product each give the unfused decoder's output (not
    necessarily bit for bit: a wider product may sum in another order)."""
    from mizuoto.runtime.lm.qwen import QwenConfig, QwenDecoder

    c = QwenConfig(hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8)
    torch.manual_seed(0)
    plain = QwenDecoder(c)
    for t in plain.parameters():
        t.data.normal_(0, 0.1)
    fused = QwenDecoder(c)
    fused.load_state_dict(plain.state_dict())
    fused.fuse()
    names = set(fused.state_dict())
    assert "layers.0.self_attn.qkv_proj.weight" in names and "layers.0.mlp.gate_up_proj.weight" in names
    assert not any(n.endswith(("q_proj.weight", "gate_proj.weight")) for n in names)
    x = torch.randn(2, 5, 32)
    positions = torch.arange(5).expand(2, -1)
    for grad in (True, False):  # the separate and the merged q/k paths
        with torch.set_grad_enabled(grad):
            assert torch.allclose(plain(x, positions), fused(x, positions), atol=1e-5)


@pytest.fixture(scope="module")
def fused(model_dir):
    devices = DeviceService()
    return Qwen3TtsEngine().load(
        LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cpu", options={"fused_projections": True}), Runtime(devices, ModelManager(devices)), RequestContext("load")
    )


def test_fast_mode_speaks_and_says_so(fused):
    chunks, ctx = speak(fused)
    assert ctx.applied["fused_projections"] is True and chunks[0].samples.shape[0] % 1920 == 0
    assert speak(fused, seed=5)[0][0].samples.shape == speak(fused, seed=5)[0][0].samples.shape


def test_fast_mode_streams_its_fused_layers_identically(fused, monkeypatch):
    """The fused projections are ordinary linear layers to the VRAM manager: placed and streamed, the same audio."""
    from tests.engines_native.streaming import check_streamed

    def run(manager):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return speak(fused, ctx=ctx)[0][0].samples

    streamed = check_streamed(fused, run, monkeypatch)
    assert streamed[0] > 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_fast_mode_with_cuda_graphs_decodes_as_its_eager_path_on_a_gpu(model_dir):
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()

    def run(graphs: bool):
        loaded = Qwen3TtsEngine().load(
            LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cuda", options={"fused_projections": True}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
        for h in loaded.handles:
            h.load()
        ctx = RequestContext("r", seed=3, device="cuda")
        return speak(loaded, ctx=ctx)[0][0].samples, ctx.applied

    (eager, _), (graphed, applied) = run(False), run(True)
    assert applied["fused_projections"] and applied["cuda_graphs"]
    assert eager.shape == graphed.shape and (eager == graphed).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(model_dir, monkeypatch, caplog):
    """A step that cannot be captured (it reads a value back) is decoded eagerly from a fresh generator in the same
    state: the same audio as without graphs."""
    from mizuoto.engines.qwen3tts import decode
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()

    def run(graphs: bool):
        loaded = Qwen3TtsEngine().load(
            LoadSpec("qwen3tts/tiny", model_dir, "custom_voice", "cuda"), Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)), RequestContext("load")
        )
        for h in loaded.handles:
            h.load()
        return speak(loaded, ctx=RequestContext("r", seed=3, device="cuda"))[0][0].samples

    want = run(False)
    sample = decode.FrameDecoder._sample
    monkeypatch.setattr(decode.FrameDecoder, "_sample", lambda self, blocked: (sample(self, blocked), self.status.tolist()))
    got = run(True)
    assert "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()
