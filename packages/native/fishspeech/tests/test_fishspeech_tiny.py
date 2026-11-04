"""The native Fish Speech engine (packages/native/fishspeech) on a tiny random model.

Real-model parity is checked by scripts/parity (S2 Pro on the CPU against v1); this covers the code path in
CI: loading both checkpoints, the conversation prompt, dual-AR decoding with repetition-aware sampling, the
codec both ways, voice caches, speaker-turn batches, streaming, seeds and cancellation.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
pytest.importorskip("mizuoto_fishspeech")

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EngineError, LoadSpec, Reference, SynthesisTask, VoiceSpec, get_engine  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto_fishspeech import engine as engine_module  # noqa: E402
from mizuoto_fishspeech.codec import CodecConfig, TransformerConfig  # noqa: E402
from mizuoto_fishspeech.generate import batches_of, group_turns, logits_to_probs, split_turns  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|voice|>", *(f"<|speaker:{i}|>" for i in range(4))]
SEMANTIC = [f"<|semantic:{i}|>" for i in range(4096)]
TINY_CODEC = CodecConfig(
    encoder_dim=16,
    encoder_rates=[4, 8],
    encoder_transformer_layers=[0, 1],
    latent_dim=32,
    decoder_dim=32,
    decoder_rates=[8, 4],
    n_codebooks=2,
    codebook_size=8,
    codebook_dim=4,
    semantic_codebook_size=16,
    quantizer_transformer=TransformerConfig(n_layer=1, n_head=1, dim=64, intermediate_size=96, window_size=8),
)


def _random_state(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in module.state_dict().items():
        if name.endswith("alpha") or name.endswith("weight_g"):
            out[name] = torch.ones_like(t)
        elif t.is_floating_point():
            out[name] = (torch.randn(t.shape, generator=g) * 0.2).to(t.dtype)
        else:
            out[name] = t
    return out


def _tokenizer(root: Path) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([*SPECIALS, *SEMANTIC])
    tok.save(str(root / "tokenizer.json"))
    return tok.get_vocab_size(with_added_tokens=True)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    return build_model(tmp_path_factory.mktemp("tiny-fishspeech"))


def build_model(root: Path) -> Path:
    """A tiny random S2 model in ``root`` (its codec is ``TINY_CODEC``; the engine must be given it)."""
    from mizuoto.runtime import ops
    from mizuoto_fishspeech.codec import FishCodec
    from mizuoto_fishspeech.model import DualARTransformer, ModelArgs

    vocab = _tokenizer(root)
    text = {"n_layer": 2, "n_head": 2, "n_local_heads": 1, "head_dim": 16, "dim": 32, "intermediate_size": 64, "rope_base": 10000, "norm_eps": 1e-6, "max_seq_len": 4096}
    config = {
        "model_type": "fish_qwen3_omni",
        "semantic_start_token_id": 0,
        "semantic_end_token_id": 0,
        "text_config": {**text, "vocab_size": vocab, "attention_qk_norm": True, "tie_word_embeddings": True},
        "audio_decoder_config": {"n_layer": 2, "n_head": 2, "n_local_heads": 1, "head_dim": 16, "dim": 32, "intermediate_size": 64, "num_codebooks": 3, "vocab_size": 16},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = DualARTransformer(ModelArgs.from_config(config), ops.default)
    state = _random_state(model, 1)
    # stored as the S2 checkpoint: the slow model under text_model.model, the fast one under audio_decoder
    stored = {}
    for k, v in state.items():
        if k.startswith("fast_"):
            stored["audio_decoder." + k[len("fast_") :]] = v
        elif k.startswith("codebook_embeddings."):
            stored["audio_decoder." + k] = v
        else:
            stored["text_model.model." + k] = v
    save_file(stored, str(root / "model.safetensors"))
    codec = FishCodec(TINY_CODEC, ops.default)
    codec_state = _random_state(codec, 2)
    # an older upstream saved its RoPE tables and masks too; they are ignored
    codec_state["quantizer.pre_module.freqs_cis"] = torch.zeros(4, 2)
    torch.save({"state_dict": {f"generator.{k}": v for k, v in codec_state.items()}, "discriminator.x": torch.zeros(1)}, str(root / "codec.pth"))
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    engine = get_engine("fishspeech")
    assert engine.detect_variant(model_dir) == "s2_pro"
    devices = DeviceService()
    patch = pytest.MonkeyPatch()
    patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
    try:
        yield engine.load(LoadSpec("fishspeech/tiny", model_dir, "s2_pro", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    rate = 16000  # resampled to the codec's 44.1 kHz
    t = np.arange(rate // 2) / rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def voice_of(reference: Path, **kw) -> VoiceSpec:
    return VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")], **kw)


def speak(loaded, voice, seed=3, text="Hello there.", ctx=None, stream=False, **options):
    ctx = ctx or RequestContext("r", seed=seed)
    task = SynthesisTask(text=text, language="en", voice=voice, params={"seed": seed, "max_new_tokens": 8}, options=options, stream=stream)
    with torch.inference_mode():
        return list(loaded.synthesize(task, ctx)), ctx


def test_turns_and_batches():
    text = "<|speaker:0|>Hi there.<|speaker:1|>Hello!<|speaker:0|>How are you?"
    assert split_turns(text) == ["<|speaker:0|>Hi there.", "<|speaker:1|>Hello!", "<|speaker:0|>How are you?"]
    assert split_turns("No tags here.") == []
    assert batches_of("No tags here.", 200) == ["No tags here."]
    assert group_turns(["a" * 10, "b" * 10, "c" * 10], max_speakers=5, max_bytes=25) == ["a" * 10 + "\n" + "b" * 10, "c" * 10]
    assert len(group_turns([f"<|speaker:{i % 2}|>x" for i in range(7)], max_speakers=5, max_bytes=1000)) == 2


def test_sampling_keeps_the_top_token_and_the_top_k():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0] + [0.0] * 60)
    probs = logits_to_probs(logits, torch.tensor(1.0), torch.tensor(0.01), 30)
    assert probs[0] == 1.0 and probs[1:].sum() == 0  # top-p below the first token's share keeps the first only
    probs = logits_to_probs(logits, torch.tensor(1.0), torch.tensor(1.0), 30)
    assert int((probs > 0).sum()) == 30


def test_clone_repeats_under_a_seed(loaded, reference):
    (a,), ctx = speak(loaded, voice_of(reference), seed=5)
    (b,), _ = speak(loaded, voice_of(reference), seed=5)
    (c,), _ = speak(loaded, voice_of(reference), seed=6)
    assert a.final and a.sample_rate == 44100 and a.samples.size and np.isfinite(a.samples).all()
    assert (a.samples == b.samples).all() and not (a.samples.shape == c.samples.shape and (a.samples == c.samples).all())
    # what v1 recorded: upstream's request fields, the parameters given and the number of references
    assert ctx.applied == {"use_memory_cache": "on", "streaming": False, "max_new_tokens": 8, "seed": 5, "references": 1}


def test_the_voice_cache_holds_the_codes(loaded, reference, tmp_path):
    voice = voice_of(reference)
    with torch.inference_mode():
        cache = loaded.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    assert cache is not None and Path(cache.files["codes"]).is_file()
    cached = voice_of(reference, cache=cache.files)
    assert (speak(loaded, cached)[0][0].samples == speak(loaded, voice)[0][0].samples).all()


def test_dialogue_streams_one_chunk_per_batch(loaded, reference):
    text = "<|speaker:0|>" + "a" * 60 + "<|speaker:1|>" + "b" * 60
    chunks, _ = speak(loaded, voice_of(reference), text=text, stream=True, chunk_length=100)
    assert [c.final for c in chunks] == [False, False, True] and chunks[-1].samples.size == 0
    (whole,), _ = speak(loaded, voice_of(reference), text=text, chunk_length=100)
    assert np.array_equal(whole.samples, np.concatenate([c.samples for c in chunks]))


def test_clone_needs_a_transcript(loaded, reference):
    with pytest.raises(EngineError):
        speak(loaded, VoiceSpec(references=[Reference(audio_path=str(reference), transcript="")]))
    with pytest.raises(EngineError):
        speak(loaded, VoiceSpec())


def test_codec_operations(loaded, reference, tmp_path):
    with torch.inference_mode():
        encoded = loaded.call("codec_encode", {"audio_path": str(reference), "output_path": str(tmp_path / "codes.npy")}, RequestContext("e"))
        assert encoded["shape"][0] == 3
        decoded = loaded.call("codec_decode", {"codes_path": encoded["codes"], "output_path": str(tmp_path / "out.wav")}, RequestContext("d"))
    assert decoded["sample_rate"] == 44100 and decoded["samples"] == encoded["shape"][1] * 4 * 32


def test_cancel(loaded, reference):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, voice_of(reference), ctx=ctx)


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, reference, monkeypatch, tmp_path):
    """The codec (references, each batch's decoding) and the language model each in its phase on a
    simulated device, units streamed: the same audio, also for a dialogue in batches and the codec operations."""
    from tests.engines_native.streaming import check_streamed

    def run(manager, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return np.concatenate([c.samples for c in speak(loaded, voice_of(reference), ctx=ctx, **kw)[0]])

    def codec(manager):
        ctx = RequestContext("e")
        ctx.manager = manager
        with torch.inference_mode():
            encoded = loaded.call("codec_encode", {"audio_path": str(reference), "output_path": str(tmp_path / f"c{id(ctx)}.npy")}, ctx)
            loaded.call("codec_decode", {"codes_path": encoded["codes"], "output_path": str(tmp_path / f"o{id(ctx)}.wav")}, ctx)
        return np.load(encoded["codes"]), (tmp_path / f"o{id(ctx)}.wav").read_bytes()

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    text = "<|speaker:0|>" + "a" * 60 + "<|speaker:1|>" + "b" * 60
    check_streamed(loaded, lambda m: run(m, text=text, stream=True, chunk_length=100), monkeypatch)
    check_streamed(loaded, lambda m: (lambda r: (r[0], np.frombuffer(r[1], dtype=np.uint8)))(codec(m)), monkeypatch)


def test_the_codec_falls_back_to_tiles_when_allowed(loaded, reference, monkeypatch):
    """The codec's decoding runs out of memory on every try; with compute.oom_fallback its decoder runs in tiles (one
    here: the tiny clip is short), the same audio, and the request says so."""
    from mizuoto.runtime.policy import VramPolicy
    from tests.engines_native.streaming import simulated

    expected = np.concatenate([c.samples for c in speak(loaded, voice_of(reference))[0]])

    def out_of_memory(indices):
        raise torch.OutOfMemoryError("CUDA out of memory")

    with simulated(loaded, 1.0, monkeypatch) as manager:
        policy = VramPolicy(reserve_bytes=0, host_copies="on", oom_fallback=True)
        manager._policy = lambda: policy
        monkeypatch.setattr(loaded.codec, "from_indices", out_of_memory)
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        got = np.concatenate([c.samples for c in speak(loaded, voice_of(reference), ctx=ctx)[0]])
    assert np.array_equal(got, expected)
    assert ctx.applied["fallbacks"] == {"codec": "in tiles of 256 latent frames"} and ctx.applied["oom_retries"] == {"codec": 2}


def test_the_codec_decoder_in_tiles_gives_the_whole(loaded):
    """Over many tiles, each decoded after the frames its first sample depends on (the causal decoder's bound: 21
    for the tiny codec's rates), the tiles' samples are the whole's to float rounding; with 16
    frames they are not."""
    from mizuoto.nn.dsp.tiles import context_tiled_1d

    decoder = loaded.codec.decoder
    assert decoder.context_frames() == 21  # 6 + (2 + 78/8) + (2/8 + 78/32) + 6/32, rounded up
    latent = torch.randn(1, TINY_CODEC.latent_dim, 700, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        whole = decoder(latent)
        tiled = context_tiled_1d(decoder, latent, 32, tile=256, before=decoder.context_frames())
        short = context_tiled_1d(decoder, latent, 32, tile=256, before=16)
    assert tiled.shape == whole.shape and float((tiled - whole).abs().max()) < 1e-4
    assert float((short - whole).abs().max()) > 0.1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(model_dir, reference):
    """compute.cuda_graphs (mizuoto.runtime.graphs): the slow step as graphs around its attention calls and the frame's
    draws and fast transformer as one graph replay the eager kernels: the same audio for fresh loads (the request that
    captures), later requests, several batches of a dialogue and other seeds."""
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()

    def load(graphs: bool):
        engine = get_engine("fishspeech")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
            loaded = engine.load(
                LoadSpec("fishspeech/tiny", model_dir, "s2_pro", "cuda", options={}),
                Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
                RequestContext("load"),
            )
        for h in loaded.handles:
            h.load()
        return loaded

    def run(loaded, seed: int, text: str):
        ctx = RequestContext("r", seed=seed, device="cuda")
        task = SynthesisTask(text=text, language="en", voice=voice_of(reference), params={"seed": seed, "max_new_tokens": 24}, options={"chunk_length": 100})
        with torch.inference_mode():
            return [c.samples for c in loaded.synthesize(task, ctx)], ctx.applied.get("cuda_graphs")

    dialogue = "<|speaker:0|>" + "a" * 60 + "<|speaker:1|>" + "b" * 60
    for _ in range(3):
        eager, graphed = load(False), load(True)
        for seed, text in ((3, "Hello there."), (4, dialogue), (3, "Hello there.")):
            want, _ = run(eager, seed, text)
            got, used = run(graphed, seed, text)
            assert used
            assert all(a.shape == b.shape and (a == b).all() for a, b in zip(want, got, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(model_dir, reference, monkeypatch, caplog):
    """A step that cannot be captured (here: one that reads a value back, which a capture forbids) is decoded eagerly,
    drawing from a fresh generator in the same state: the same audio as without graphs."""
    from mizuoto.runtime.runtime import RuntimeOptions
    from mizuoto_fishspeech import generate

    devices = DeviceService()

    def load(graphs: bool):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
            loaded = get_engine("fishspeech").load(
                LoadSpec("fishspeech/tiny", model_dir, "s2_pro", "cuda", options={}),
                Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
                RequestContext("load"),
            )
        for h in loaded.handles:
            h.load()
        return loaded

    def run(loaded):
        ctx = RequestContext("r", seed=3, device="cuda")
        task = SynthesisTask(text="Hello there.", language="en", voice=voice_of(reference), params={"seed": 3, "max_new_tokens": 24})
        with torch.inference_mode():
            return [c.samples for c in loaded.synthesize(task, ctx)][0], ctx.applied.get("cuda_graphs")

    want, _ = run(load(False))
    tail = generate.Decoder._tail
    monkeypatch.setattr(generate.Decoder, "_tail", lambda self: (tail(self), self.ended.item()))
    got, used = run(load(True))
    assert not used and "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()


def _load_with(model_dir, device: str, **options):
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
        loaded = get_engine("fishspeech").load(
            LoadSpec("fishspeech/tiny", model_dir, "s2_pro", device, options={}), Runtime(devices, ModelManager(devices), RuntimeOptions(**options)), RequestContext("load")
        )
    for h in loaded.handles:
        h.load()
    return loaded


COMPILER = pytest.mark.skipif(sys.platform == "win32" or not (shutil.which("g++") or shutil.which("clang++")), reason="the CPU inductor needs a C++ compiler")


@COMPILER
def test_the_compiled_pieces_compute_the_eager_step(loaded):
    """compute.compile (mizuoto_fishspeech.fused): the slow step, the fast transformer and the draws' probabilities
    from compiled pieces are the eager ones to within float32 rounding (fusing is not bit-identical)."""
    from mizuoto_fishspeech.fused import Fused
    from mizuoto_fishspeech.generate import IM_END, TOP_K, Decoder, Sampling, logits_to_probs

    model = loaded.model
    c = model.config
    fused = Fused("tiny")
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 16, (1, 1 + c.num_codebooks, 7), generator=g, dtype=torch.int)
    with torch.inference_mode():
        steps = {}
        for name, pieces in (("eager", None), ("compiled", fused)):
            decoder = Decoder(model, loaded.tokenizer.token_to_id(IM_END), None, lambda pieces=pieces: pieces)
            decoder.start(Sampling(), torch.Generator().manual_seed(1), lambda: None)
            decoder.fused = pieces
            caches = model.new_caches()
            model.forward_generate(prompt, torch.arange(7), 7, caches, False)
            decoder.cur.copy_(prompt[:, :, -1:])
            decoder.pos.fill_(7)
            decoder._slow(caches, 8)
            fast = [model.forward_generate_fast(decoder.hidden, 0, decoder.fast_caches)] if pieces is None else [fused.fast_forward(model, decoder.hidden, 0, decoder.fast_caches)]
            h = model.fast_embeddings(torch.tensor([3]))
            fast.append(model.forward_generate_fast(h, 1, decoder.fast_caches) if pieces is None else fused.fast_forward(model, h, 1, decoder.fast_caches))
            steps[name] = (decoder.logits.clone(), decoder.hidden.clone(), *fast)
        for want, got in zip(steps["eager"], steps["compiled"], strict=True):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        logits = steps["eager"][0]
        t, p = torch.tensor(0.8), torch.tensor(0.9)
        torch.testing.assert_close(fused.run("probs", logits, t, p), logits_to_probs(logits[0, -1], t, p, TOP_K), rtol=1e-4, atol=1e-6)
    assert fused.failure is None and fused.calls > 0


@COMPILER
def test_compute_compile_decodes_with_the_compiled_pieces(model_dir, reference):
    (chunk,), ctx = speak(_load_with(model_dir, "cpu", compile=True), voice_of(reference), seed=5)
    assert ctx.applied["compile"] == "decoder"
    assert chunk.samples.size and np.isfinite(chunk.samples).all()


def test_a_compiler_error_decodes_eagerly(model_dir, reference, monkeypatch, caplog):
    """Any compiler failure makes every piece run eagerly from then on, with one warning: the eager audio."""
    from mizuoto_fishspeech import fused

    def broken(*args, **kwargs):
        raise RuntimeError("no compiler here")

    monkeypatch.setattr(torch, "compile", lambda fn, dynamic=None: broken)
    (want,), _ = speak(_load_with(model_dir, "cpu"), voice_of(reference), seed=5)
    (got,), ctx = speak(_load_with(model_dir, "cpu", compile=True), voice_of(reference), seed=5)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == "mizuoto_fishspeech.fused"]
    assert len(warnings) == 1 and "torch.compile of fishspeech/tiny:decoder failed" in warnings[0] and "no compiler here" in warnings[0]
    assert "compile" not in ctx.applied and fused.PIECES
    assert want.samples.shape == got.samples.shape and (want.samples == got.samples).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_compiled_steps_replay_exactly_as_graphs_on_a_gpu(model_dir, reference):
    """compute.compile with compute.cuda_graphs: the graphs capture the compiled pieces' kernels and replay them, so
    the audio is the compiled pieces' run without graphs, request after request."""

    def run(loaded, seed: int):
        ctx = RequestContext("r", seed=seed, device="cuda")
        task = SynthesisTask(text="Hello there.", language="en", voice=voice_of(reference), params={"seed": seed, "max_new_tokens": 24})
        with torch.inference_mode():
            return [c.samples for c in loaded.synthesize(task, ctx)][0], ctx.applied

    plain, graphed = _load_with(model_dir, "cuda", compile=True, cuda_graphs=False), _load_with(model_dir, "cuda", compile=True)
    for seed in (3, 4, 3):
        want, applied = run(plain, seed)
        got, used = run(graphed, seed)
        assert applied.get("compile") == used.get("compile") == "decoder" and used.get("cuda_graphs") and not applied.get("cuda_graphs")
        assert want.shape == got.shape and (want == got).all()
