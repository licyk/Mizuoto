"""The native VoxCPM engine on tiny random models of both generations: no weights needed.

With upstream's ``voxcpm`` package importable (the VoxCPM reference environment, not CI), the same tiny folders are
also loaded by upstream and every request must give identical audio.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.voxcpm import VoxCpmEngine  # noqa: E402
from mizuoto.engines.voxcpm.config import parse_config  # noqa: E402
from mizuoto.engines.voxcpm.model import VoxCpmModel  # noqa: E402
from mizuoto.engines.voxcpm.vae import AudioVAE  # noqa: E402
from mizuoto.runtime import ops  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, StyleSpec, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime, RuntimeOptions  # noqa: E402

LM = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "max_position_embeddings": 512,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "vocab_size": 400,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4,
    "use_mup": False,
}
CFM = {"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "log-norm", "inference_cfg_rate": 2.0}


def config(version: int) -> dict:
    head_dim = 16 if version == 1 else 24  # v2: kv_channels
    lm = {
        **LM,
        "rope_scaling": {
            "type": "longrope",
            "long_factor": [1.0] * (head_dim // 2),
            "short_factor": [1.0 + i / 10 for i in range(head_dim // 2)],
            "original_max_position_embeddings": 512,
        },
    }
    block = {"hidden_dim": 32, "ffn_dim": 64, "num_heads": 2, "num_layers": 1}
    out = {
        "patch_size": 2,
        "feat_dim": 8,
        "scalar_quantization_latent_dim": 16,
        "scalar_quantization_scale": 9,
        "residual_lm_num_layers": 1,
        "lm_config": lm,
        "encoder_config": dict(block),
        "dit_config": {**block, "cfm_config": CFM},
        "max_length": 256,
        "device": "cpu",
        "dtype": "float32",
    }
    if version == 1:
        out["architecture"] = "voxcpm"
        out["audio_vae_config"] = {"encoder_dim": 2, "encoder_rates": [4, 5, 8], "latent_dim": 8, "decoder_dim": 16, "decoder_rates": [8, 5, 4], "sample_rate": 16000}
    else:
        lm["kv_channels"] = 24  # heads × kv_channels ≠ hidden: the projections' own width
        out["architecture"] = "voxcpm2"
        out["residual_lm_no_rope"] = True
        out["encoder_config"]["kv_channels"] = 24
        out["dit_config"]["kv_channels"] = 24
        out["audio_vae_config"] = {
            "encoder_dim": 2,
            "encoder_rates": [4, 5, 8],
            "latent_dim": 8,
            "decoder_dim": 16,
            "decoder_rates": [8, 5, 4, 3],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        }
    return out


def tokenizer_files(root: Path) -> None:
    """A Llama-style BPE (byte fallback, ``▁``): bytes, letters and a few Han characters, ``你好`` merged."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for piece in ["▁", *"abcdefghijklmnopqrstuvwxyz.,!?'", "你", "好", "世", "界", "▁h", "▁he", "▁hel", "▁hell", "▁hello", "你好"]:
        vocab[piece] = len(vocab)
    merges = ["▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o", "你 好"]
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for c, i in (("<unk>", 0), ("<s>", 1), ("</s>", 2))
        ],
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    (root / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    added = {
        str(i): {"content": c, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True} for c, i in (("<unk>", 0), ("<s>", 1), ("</s>", 2))
    }
    added["398"] = {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
    config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "legacy": True,
        "tokenizer_class": "LlamaTokenizer",
        "added_tokens_decoder": added,
    }
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")


def _random(shape: torch.Size, g: torch.Generator, scale: float = 0.2) -> torch.Tensor:
    return torch.randn(shape, generator=g) * scale


def build_folder(root: Path, version: int) -> Path:
    data = config(version)
    (root / "config.json").write_text(json.dumps(data), encoding="utf-8")
    tokenizer_files(root)
    parsed = parse_config(data)
    g = torch.Generator().manual_seed(10 + version)
    model = VoxCpmModel(parsed)
    state = {k: (_random(t.shape, g) if t.is_floating_point() else t) for k, t in model.state_dict().items()}
    for k in list(state):
        if k.endswith("norm.weight") or k.endswith("layernorm.weight"):
            state[k] = 1.0 + _random(state[k].shape, g, 0.05)
    save_file(state, str(root / "model.safetensors"))
    vae = AudioVAE(parsed.vae)
    vae_state: dict[str, torch.Tensor] = {}
    for k, t in vae.state_dict().items():
        if k.endswith(".weight") and t.dim() == 3:  # a convolution: stored weight-normed, as upstream saves it
            base = k[: -len(".weight")]
            vae_state[f"{base}.weight_v"] = _random(t.shape, g, 0.3)
            vae_state[f"{base}.weight_g"] = 0.5 + torch.rand((t.shape[0], 1, 1), generator=g)
        elif k.endswith("alpha"):
            vae_state[k] = 0.5 + torch.rand(t.shape, generator=g)
        elif k.endswith("scale_embed.weight"):
            vae_state[k] = 1.0 + _random(t.shape, g, 0.05)
        else:
            vae_state[k] = _random(t.shape, g, 0.05) if t.is_floating_point() else t
    torch.save({"state_dict": vae_state}, root / "audiovae.pth")
    return root


@pytest.fixture(scope="module", params=[1, 2], ids=["v1", "v2"])
def folder(request, tmp_path_factory) -> Path:
    return build_folder(tmp_path_factory.mktemp(f"voxcpm-v{request.param}"), request.param)


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    t = np.arange(12000) / 24000  # 0.5 s at 24 kHz: both readers resample it
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(samples.tobytes())
    return path


def load(folder: Path, device: str = "cpu", **options):
    engine = VoxCpmEngine()
    variant = engine.detect_variant(folder)
    devices = DeviceService()
    return engine.load(
        LoadSpec(f"voxcpm/{folder.name}", folder, variant or "", device, options=options), Runtime(devices, ModelManager(devices), RuntimeOptions()), RequestContext("load")
    )


@pytest.fixture(scope="module")
def loaded(folder):
    return load(folder)


def speak(loaded, text="hello, 你好世界.", seed=3, voice=None, style=None, stream=False, ctx=None, **options):
    params = {k: options.pop(k) for k in ("num_steps", "guidance_scale", "text_normalization", "max_new_tokens") if k in options}
    task = SynthesisTask(text=text, voice=voice or VoiceSpec(), style=style or StyleSpec(), params={"seed": seed, **params}, options=options, stream=stream)
    ctx = ctx or RequestContext("r", seed=seed)
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, ctx))
    return chunks, ctx


def audio(chunks) -> np.ndarray:
    return np.concatenate([c.samples for c in chunks])


def clone(reference, transcript="hello."):
    return VoiceSpec(references=[Reference(audio_path=str(reference), transcript=transcript)])


# -- files ----------------------------------------------------------------------------


def test_variants_and_rates(folder, loaded):
    variant = VoxCpmEngine().detect_variant(folder)
    assert variant == ("v1" if "v1" in folder.name else "v2")
    from mizuoto.core.models.detect import detect

    found = detect(folder)
    assert found is not None and found[:2] == ("voxcpm", variant)
    assert loaded.describe().sample_rate == (16000 if variant == "v1" else 48000)


def test_tokenizer_splits_han_words(loaded):
    tokens = loaded.tokenizer.tokenize("hello 你好")
    assert "你好" not in tokens and tokens[-2:] == ["你", "好"]
    assert loaded.tokenizer("hello") == [loaded.tokenizer.tokenizer.token_to_id("▁hello")]
    assert loaded.tokenizer("<|im_end|>")  # the config's special token, appended to the vocabulary


def test_normaliser_writes_out_numbers():
    from mizuoto.engines.voxcpm.text import TextNormalizer

    n = TextNormalizer()
    assert n.normalize("I have 2 cats.") == "I have two cats ."  # wetext's English grammar spaces the stop, as upstream
    assert n.normalize("我有2只猫。") == "我有两只猫。"


# -- speech ---------------------------------------------------------------------------


def test_zero_shot_repeats_with_its_seed(loaded):
    a, ctx = speak(loaded, seed=5)
    b, _ = speak(loaded, seed=5)
    c, _ = speak(loaded, seed=6)
    assert a[-1].final and np.isfinite(audio(a)).all() and audio(a).size > 0
    assert (audio(a) == audio(b)).all() and not np.array_equal(audio(a), audio(c))
    # a random model rarely stops by itself, so it may count as a bad case and be generated again
    assert ctx.applied["prompt_mode"] == "zero_shot" and ctx.applied["seed"] == 5 + ctx.applied.get("retries", 0) and ctx.applied["inference_timesteps"] == 10


def test_unseeded_requests_draw_a_seed(loaded):
    _, ctx = speak(loaded, seed=None, ctx=RequestContext("r"))
    assert isinstance(ctx.applied["seed"], int)


def test_clone_modes(loaded, reference):
    v2 = loaded.config.version == 2
    _, ctx = speak(loaded, voice=clone(reference))
    assert ctx.applied["prompt_mode"] == ("ref_continuation" if v2 else "continuation")
    if v2:
        _, ctx = speak(loaded, voice=clone(reference, None))
        assert ctx.applied["prompt_mode"] == "reference"
        _, ctx = speak(loaded, voice=clone(reference), reference_prefix=False)
        assert ctx.applied["prompt_mode"] == "continuation"
    else:
        from mizuoto.runtime.engine import EngineError

        with pytest.raises(EngineError):
            speak(loaded, voice=clone(reference, None))


def test_cached_voice_speaks_as_the_clip(loaded, reference, tmp_path):
    voice = clone(reference)
    plain = audio(speak(loaded, voice=voice)[0])
    cache = loaded.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    voice.cache = cache.files
    assert (audio(speak(loaded, voice=voice)[0]) == plain).all()


def test_design_and_instruction_prefix_the_text(loaded, reference):
    if loaded.config.version != 2:
        pytest.skip("VoxCPM2 only")
    assert loaded._text(SynthesisTask(text="hi", voice=VoiceSpec(description="(a calm) voice")), "hi") == "(a calm voice)hi"
    task = SynthesisTask(text="hi", voice=clone(reference), style=StyleSpec(instruct="fast"))
    assert loaded._text(task, "hi  there\nnow") == "(fast)hi there now"
    designed = audio(speak(loaded, voice=VoiceSpec(description="a calm voice"))[0])
    assert designed.size > 0


def test_retry_regenerates_with_the_next_seed(loaded):
    _, ctx = speak(loaded, text="hi", seed=4, retry_badcase_ratio_threshold=1.0, retry_badcase_max_times=2)
    assert ctx.applied["seed"] == 5 and ctx.applied["retries"] == 1


def test_streaming_yields_patches(loaded, reference):
    chunks, ctx = speak(loaded, voice=clone(reference), stream=True)
    assert len(chunks) > 1 and chunks[-1].final and not any(c.final for c in chunks[:-1])
    assert ctx.applied["retry_badcase"] is False
    per_patch = loaded.config.patch_size * (loaded.vae.decode_chunk_size if loaded.config.version == 2 else loaded.vae.chunk_size)
    assert {len(c.samples) for c in chunks} == {per_patch}


def test_cancel_stops_generation(loaded):
    ctx = RequestContext("c", seed=1)
    ctx.cancel()
    with pytest.raises(Cancelled):
        speak(loaded, text="hello hello hello hello", ctx=ctx)


# -- against upstream (VoxCPM reference environment only) -----------------------------


@pytest.fixture(scope="module")
def upstream(folder):
    voxcpm = pytest.importorskip("voxcpm")
    return voxcpm.VoxCPM(str(folder), enable_denoiser=False, optimize=False, device="cpu")


def test_matches_upstream(loaded, upstream, reference):
    v2 = loaded.config.version == 2
    cases = [
        ({"text": "hello, 你好世界."}, {"text": "hello, 你好世界."}),
        (
            {"text": "hello.", "prompt_wav_path": str(reference), "prompt_text": "hello."},
            {"text": "hello.", "voice": clone(reference), **({"reference_prefix": False} if v2 else {})},
        ),
    ]
    if v2:
        cases.append(({"text": "hi there.", "reference_wav_path": str(reference)}, {"text": "hi there.", "voice": clone(reference, None)}))
        cases.append(
            (
                {"text": "(calm)hi.", "reference_wav_path": str(reference), "prompt_wav_path": str(reference), "prompt_text": "hello."},
                {"text": "hi.", "voice": clone(reference), "style": StyleSpec(instruct="calm")},
            )
        )
    for up_kwargs, native_kwargs in cases:
        expected = upstream.generate(**up_kwargs, seed=7)
        got = audio(speak(loaded, seed=7, **native_kwargs)[0])
        assert np.array_equal(expected, got), up_kwargs
        streamed = np.concatenate(list(upstream.generate_streaming(**up_kwargs, seed=7)))
        assert np.array_equal(streamed, audio(speak(loaded, seed=7, stream=True, **native_kwargs)[0])), up_kwargs
    forced = upstream.generate(text="hi", seed=4, retry_badcase_ratio_threshold=1.0, retry_badcase_max_times=2)
    assert np.array_equal(forced, audio(speak(loaded, text="hi", seed=4, retry_badcase_ratio_threshold=1.0, retry_badcase_max_times=2)[0]))


# -- memory -------------------------------------------------------------------------------


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, reference, monkeypatch, tmp_path):
    """On a simulated device, phase by phase, with units streamed: the same audio."""
    from tests.engines_native.streaming import check_streamed

    def run(manager, stream=False, voice=None):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return audio(speak(loaded, voice=voice, stream=stream, ctx=ctx)[0])

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, lambda m: run(m, stream=True), monkeypatch)
    check_streamed(loaded, lambda m: run(m, voice=clone(reference)), monkeypatch)

    def prepared(manager):
        ctx = RequestContext("p", seed=3)
        ctx.manager = manager
        with torch.inference_mode():
            cache = loaded.prepare_voice(clone(reference), tmp_path / f"cache-{id(ctx)}", ctx)
        data = torch.load(cache.files["prompt"], weights_only=True)
        return tuple(t for t in (data["prompt"], data["ref"]) if t is not None)

    check_streamed(loaded, prepared, monkeypatch)


def test_a_phase_that_runs_out_of_memory_runs_again_identically(loaded, monkeypatch):
    """A language-model layer runs out of memory once; the phase runs again: the same audio."""
    from tests.engines_native.streaming import check_oom_retry

    def run(manager, retries=None):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        out = audio(speak(loaded, ctx=ctx)[0])
        if retries is not None:
            retries.update(ctx.applied.get("oom_retries", {}))
        return out

    assert check_oom_retry(loaded, run, loaded.model.base_lm.layers[0], monkeypatch) == {"language model": 1}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_steps_attend_over_the_filled_blocks_of_the_cache_as_over_all_of_it(dtype, monkeypatch):
    """On the CPU a step reads the static cache only up to the next ``ATTEND_BLOCK`` multiple past the
    current length (mask kept). Every step's output equals upstream's attention over the whole cache, across blocks."""
    from mizuoto.engines.voxcpm import minicpm
    from mizuoto.engines.voxcpm.config import LmConfig, RopeScaling

    head_dim = 16
    rope = RopeScaling("longrope", [1.0] * (head_dim // 2), [1.0 + i / 10 for i in range(head_dim // 2)], 512)
    c = LmConfig(64, 128, 4096, 4, 2, 2, 1e-5, rope, 0, 12, 256, 1.4, 10000, kv_channels=head_dim)
    model = minicpm.MiniCPMModel(c)
    ops.init_random(model, torch.Generator().manual_seed(0), scale=0.2)
    model = model.to(dtype).eval()
    g = torch.Generator().manual_seed(1)
    prefill = torch.randn(1, 5, 64, generator=g).to(dtype)
    steps = torch.randn(300, 1, 64, generator=g).to(dtype)

    def run(block: int) -> list[torch.Tensor]:
        monkeypatch.setattr(minicpm, "ATTEND_BLOCK", block)
        with torch.inference_mode():
            _, caches = model(prefill)
            model.kv_cache = model.new_cache(1024, "cpu", dtype)
            model.kv_cache.fill_caches(caches)
            return [model.forward_step(x, torch.tensor([model.kv_cache.step()])) for x in steps]

    whole = run(1 << 30)
    lengths: list[int] = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, *args, **kwargs):
        lengths.append(k.shape[-2])
        return sdpa(q, k, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    spans = run(64)
    assert all(torch.equal(a, b) for a, b in zip(whole, spans, strict=True))
    assert set(lengths) >= {64, 128, 192, 256, 320} and 1024 not in lengths  # five blocks crossed, never the whole cache


def test_steps_attend_over_the_whole_cache_off_the_cpu():
    """Elsewhere a step attends over the whole cache, as upstream: on CUDA no shorter cut gives the same bits."""
    from mizuoto.engines.voxcpm.minicpm import ATTEND_BLOCK, attend_span

    assert attend_span(5, 8192, torch.device("cpu")) == ATTEND_BLOCK
    assert attend_span(ATTEND_BLOCK + 1, 8192, torch.device("cpu")) == 2 * ATTEND_BLOCK
    assert attend_span(8100, 8192, torch.device("cpu")) == 8192
    for device in ("cuda", "cuda:1", "mps", "xpu"):
        assert attend_span(5, 8192, torch.device(device)) == 8192


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(folder, reference, monkeypatch):
    """compute.cuda_graphs (mizuoto.runtime.graphs): a patch (the DiT's Euler steps with their noise, the encoder, the
    stop head, both language models' steps) as one graph (on CUDA a step attends over the whole cache, so one span)
    replays the eager kernels: the same audio for fresh loads, later requests and streaming."""

    def gpu(graphs: bool):
        engine = VoxCpmEngine()
        devices = DeviceService()
        loaded = engine.load(
            LoadSpec(f"voxcpm/{folder.name}", folder, engine.detect_variant(folder) or "", "cuda", options={}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
        for h in loaded.handles:
            h.load()
        return loaded

    def requests(loaded):
        out = []
        for seed, kw in ((3, {}), (4, {"voice": clone(reference)}), (5, {"stream": True}), (3, {})):
            chunks, ctx = speak(loaded, seed=seed, ctx=RequestContext("r", seed=seed, device="cuda"), max_new_tokens=40, **kw)
            out.append((audio(chunks), ctx.applied.get("cuda_graphs")))
        return out

    # The first VoxCPM request in a process on CUDA decodes its (identical) latents with other bits than later ones,
    # graphs or not (cuDNN's first call; the code before graphs does it too): warm up before comparing.
    requests(gpu(False))
    for _ in range(2):
        for (want, _), (got, used) in zip(requests(gpu(False)), requests(gpu(True)), strict=True):
            assert used
            assert want.shape == got.shape and (want == got).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(folder, monkeypatch, caplog):
    """A patch that cannot be captured (here: one that reads a value back) is decoded eagerly, drawing from a fresh
    generator in the same state: the same audio as without graphs."""
    from mizuoto.engines.voxcpm import model

    def gpu(graphs: bool):
        engine = VoxCpmEngine()
        devices = DeviceService()
        loaded = engine.load(
            LoadSpec(f"voxcpm/{folder.name}", folder, engine.detect_variant(folder) or "", "cuda", options={}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
        for h in loaded.handles:
            h.load()
        return audio(speak(loaded, seed=3, ctx=RequestContext("r", seed=3, device="cuda"), max_new_tokens=20)[0])

    gpu(False)  # the first request in a process decodes with other bits (see above)
    want = gpu(False)
    patch = model.PatchDecoder._patch
    monkeypatch.setattr(model.PatchDecoder, "_patch", lambda self, *a: (patch(self, *a), self._ports["stop"].item()))
    got = gpu(True)
    assert "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()


def test_the_attention_span_option_is_exact_on_the_cpu_and_listed(folder, loaded):
    """``attention_span`` (a fast mode for GPUs) changes nothing on the CPU, which already attends over the filled part
    of the cache; outputs list it."""
    spanned = load(folder, attention_span=True)
    (a, _), (b, ctx) = speak(loaded), speak(spanned)
    assert np.array_equal(audio(a), audio(b)) and ctx.applied["attention_span"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_attention_span_option_on_a_gpu_attends_over_the_filled_cache(folder):
    """On a GPU the option makes each step attend over the filled blocks of its cache, as on the CPU (by default the
    whole cache, exact): the audio still has its length and repeats under a seed."""
    spanned = load(folder, "cuda", attention_span=True)
    for h in spanned.handles:
        h.load()
    assert spanned.model.base_lm.span_everywhere and spanned.model.residual_lm.span_everywhere
    (a, _), (b, _) = speak(spanned, ctx=RequestContext("r", seed=3, device="cuda")), speak(spanned, ctx=RequestContext("r", seed=3, device="cuda"))
    assert audio(a).size and np.array_equal(audio(a), audio(b))
