"""Drama 3, the Fish Speech engine's second variant, on a tiny random model (FP8 experts, every layer kind, the tower).

Parity with upstream (its modules run on the CPU, frames and hidden states bit for bit) is checked by the parity
harness, on a tiny model and on the real one; this covers the code path in CI: upstream's chunking and prompt rules,
loading the checkpoint as released (per-expert FP8 tensors with block scales), generation with and without references,
both samplers, seeds, streaming, voice caches, cancellation, and the experts streamed by the row on a simulated device.
"""

from __future__ import annotations

import json
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
from mizuoto.runtime.graphs import Graphs  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto_fishspeech import engine as engine_module  # noqa: E402
from mizuoto_fishspeech.codec import CodecConfig, TransformerConfig  # noqa: E402
from mizuoto_fishspeech.drama3.prompt import History, PromptError, build_prompt, parse_speaker_segments, split_by_length, split_text, utf8_len  # noqa: E402

VOCAB = 16
"""The tower's vocabulary: the semantic band, and here every codebook's size."""
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", *(f"<|speaker:{i}|>" for i in range(16))]
TINY_CODEC = CodecConfig(
    encoder_dim=16,
    encoder_rates=[4, 8],
    encoder_transformer_layers=[0, 1],
    latent_dim=32,
    decoder_dim=32,
    decoder_rates=[8, 4],
    n_codebooks=2,
    codebook_size=VOCAB,
    codebook_dim=4,
    semantic_codebook_size=VOCAB,
    quantizer_transformer=TransformerConfig(n_layer=1, n_head=1, dim=64, intermediate_size=96, window_size=8),
)


# -- upstream's chunking and prompt rules (its tests' cases) ---------------------------------------------------------


def test_split_by_length_prefers_sentence_ends_then_clauses_spaces_and_length():
    assert split_by_length("  Hello there.  ", 300) == ["  Hello there.  "]
    assert split_by_length(" \n ", 300) == []
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = split_by_length(text, 100)
    assert len(chunks) > 1 and all(utf8_len(c) <= 100 and c.endswith(".") for c in chunks) and " ".join(chunks) == text
    text = "今天天气很好，我们去公园散步。" * 20
    chunks = split_by_length(text, 100)
    assert all(utf8_len(c) <= 100 and c.endswith("。") for c in chunks) and "".join(chunks) == text
    clauses = "，".join(["这是一个很长的分句"] * 20)
    chunks = split_by_length(clauses, 60)
    assert all(c.endswith("，") for c in chunks[:-1]) and "".join(chunks) == clauses
    words = " ".join(["word"] * 100)
    chunks = split_by_length(words, 50)
    assert all(utf8_len(c) <= 50 for c in chunks) and " ".join(chunks) == words
    unbroken = "字" * 100
    chunks = split_by_length(unbroken, 50)
    assert all(utf8_len(c) <= 50 for c in chunks) and "".join(chunks) == unbroken


def test_decimals_and_protected_spans_stay_whole():
    whisper = "<whisper>" + "secret words. " * 10 + "</whisper>"
    chunks = split_by_length("Pi is 3.14159 roughly. " * 10 + whisper + " [laughing] done.", 60)
    assert whisper in chunks and utf8_len(whisper) > 60
    assert not any(c.endswith("3.") for c in chunks) and any("[laughing]" in c for c in chunks)


def test_speaker_chunks():
    assert parse_speaker_segments("intro <|speaker:1|>Hi<|speaker:2|> Yo") == [("0", "intro "), ("1", "Hi"), ("2", " Yo")]
    text = "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2<|speaker:2|>C1<|speaker:3|>D1"
    assert split_text(text, max_bytes=300, max_speakers=3) == ["<|speaker:0|> A1\n<|speaker:1|> B1\n<|speaker:0|> A2\n<|speaker:2|> C1", "<|speaker:3|> D1"]
    long = "This sentence is long enough. " * 10
    chunks = split_text(f"<|speaker:0|>{long}<|speaker:0|>Short.", 100)
    assert len(chunks) > 2 and all(s.startswith("<|speaker:0|> ") for c in chunks for s in c.split("\n")) and chunks[-1].endswith("Short.")
    assert split_text("Hello.") == ["Hello."] and split_text("  ") == []


def test_history_pins_the_first_chunk_and_keeps_the_latest():
    history = History(recent_turns=1)
    assert list(history.candidates()) == [[]]
    for index in range(3):
        history.push(f"chunk {index}", torch.full((2, 1), index))
    assert [[text for text, _ in turns] for turns in history.candidates()] == [["chunk 0", "chunk 2"], ["chunk 0"], []]


# -- a tiny model -------------------------------------------------------------------------------------------------


def _tokenizer(root: Path) -> tuple[int, int | None, int | None]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([*SPECIALS, *(f"<|semantic:{i}|>" for i in range(VOCAB))])
    tok.save(str(root / "tokenizer.json"))
    return tok.get_vocab_size(with_added_tokens=True), tok.token_to_id("<|semantic:0|>"), tok.token_to_id("<|im_end|>")


def _config(vocab: int, start: int | None, eos: int | None) -> dict:
    text = {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": vocab,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 64,
    }
    tower = {
        "dim": 128,
        "intermediate_size": 192,
        "n_head": 4,
        "n_local_heads": 2,
        "head_dim": 32,
        "norm_eps": 1e-6,
        "rope_base": 10000000.0,
        "rotary_percent": 0.25,
        "vocab_size": VOCAB,
        "num_codebooks": 3,
        "n_layer": 2,
        "shared_enrichment_layers": 2,
        "vq_backbone_codebooks": 1,
        "vq_feature_placement": "backbone_partial",
    }
    return {
        "model_type": "fish_omni",
        "text_config": text,
        "audio_decoder_config": tower,
        "semantic_start_token_id": start,
        "semantic_end_token_id": (start or 0) + VOCAB - 1,
        "eos_token_id": eos,
        "speech_template_version": "reference-bracket-v1",
        "quantization_config": {"quant_method": "bsgemm-moe", "weight_block_size": [128, 128]},
    }


def build_model(root: Path) -> Path:
    """A tiny random Drama 3 in ``root``, its tensors named and stored as the released checkpoint's (per-expert FP8)."""
    from mizuoto.runtime import ops
    from mizuoto_fishspeech.codec import FishCodec
    from mizuoto_fishspeech.drama3.backbone import Backbone
    from mizuoto_fishspeech.drama3.config import Drama3Config
    from mizuoto_fishspeech.drama3.tower import CHECKPOINT_PREFIXES, CodebookTower

    data = _config(*_tokenizer(root))
    (root / "config.json").write_text(json.dumps(data), encoding="utf-8")
    config = Drama3Config.from_dict(data)
    g = torch.Generator().manual_seed(1)
    stored: dict[str, torch.Tensor] = {}
    for name, p in Backbone(config.text, True, ops.default).named_parameters():
        if p.dtype == torch.float8_e4m3fn:
            value = torch.randn(p.shape, generator=g).to(p.dtype)
        elif "scale_inv" in name:
            value = torch.rand(p.shape, generator=g) * 0.02 + 0.02
        elif "norm" in name or name.endswith(("A_log", "dt_bias")):
            value = torch.randn(p.shape, generator=g) * 0.2
        else:
            value = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        if ".experts." in name:
            prefix, stack = name.rsplit(".", 1)
            prefix = "model.language_model." + prefix[len("model.") :]
            kind = "weight_scale_inv" if stack.endswith("scale_inv") else "weight"
            for e in range(value.shape[0]):
                if stack.startswith("w13"):
                    half = value.shape[1] // 2
                    stored[f"{prefix}.{e}.gate_proj.{kind}"] = value[e, :half].clone()
                    stored[f"{prefix}.{e}.up_proj.{kind}"] = value[e, half:].clone()
                else:
                    stored[f"{prefix}.{e}.down_proj.{kind}"] = value[e].clone()
        else:
            if name == "lm_head.weight":  # audio over the end of the turn: random weights would end it at once
                start, end = data["semantic_start_token_id"], data["semantic_end_token_id"]
                value[start : end + 1] *= 10
                value[data["eos_token_id"]] = 0
            stored[name if name == "lm_head.weight" else "model.language_model." + name[len("model.") :]] = value.to(torch.bfloat16)
    reverse = {new: old for old, new in CHECKPOINT_PREFIXES.items()}
    for name, p in CodebookTower(config, ops.default).named_parameters():
        value = torch.randn(p.shape, generator=g) * (0.2 if "norm" in name else 1.0 if "embedding" in name else 1 / p.shape[-1] ** 0.5)
        prefix = next(x for x in reverse if name.startswith(x))
        stored[reverse[prefix] + name[len(prefix) :]] = value.to(torch.bfloat16)
    keys = sorted(stored)
    shards = {"model-00001-of-00002.safetensors": keys[: len(keys) // 2], "model-00002-of-00002.safetensors": keys[len(keys) // 2 :]}
    for shard, names in shards.items():
        save_file({k: stored[k].contiguous() for k in names}, str(root / shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: s for s, names in shards.items() for k in names}}), encoding="utf-8")
    codec = FishCodec(TINY_CODEC, ops.default)
    ops.init_random(codec, torch.Generator().manual_seed(2))
    torch.save({"state_dict": {f"generator.{k}": (torch.ones_like(v) if k.endswith(("alpha", "weight_g")) else v) for k, v in codec.state_dict().items()}}, str(root / "codec.pth"))
    return root


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    return build_model(tmp_path_factory.mktemp("tiny-drama3"))


@pytest.fixture(scope="module")
def loaded(model_dir):
    engine = get_engine("fishspeech")
    assert engine.detect_variant(model_dir) == "drama3"
    devices = DeviceService()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
        yield engine.load(LoadSpec("fishspeech/tiny-drama3", model_dir, "drama3", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    rate = 16000
    t = np.arange(rate // 2) / rate
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes())
    return path


def voice_of(reference: Path, **kw) -> VoiceSpec:
    return VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")], **kw)


def speak(loaded, voice, seed=3, text="Hello there.", ctx=None, stream=False, params=None, **options):
    ctx = ctx or RequestContext("r", seed=seed)
    task = SynthesisTask(text=text, language="en", voice=voice, params={"seed": seed, "max_new_tokens": 8, **(params or {})}, options=options, stream=stream)
    with torch.inference_mode():
        return list(loaded.synthesize(task, ctx)), ctx


def test_the_checkpoint_loads_as_released(loaded):
    experts = loaded.backbone.model.layers[0].mlp.experts
    assert experts.w13_weight.dtype == torch.float8_e4m3fn and experts.w13_scale_inv.dtype == torch.float32
    assert loaded.backbone.model.norm.weight.dtype == torch.float32  # the CPU computes in float32
    assert [layer.layer_type for layer in loaded.backbone.model.layers] == ["linear_attention"] * 3 + ["full_attention"]
    assert all(m.__class__.__name__ != "WN" or "folded" in m._parameters for m in loaded.codec.modules())  # weight norm folded


def test_the_prompt(loaded):
    tok, config = loaded.tokenizer, loaded.config
    rows = build_prompt("Hi.", tok, config)
    assert rows.shape[1] == 4 and (rows[:, 1:] == 0).all()
    assert (
        tok.decode(rows[:, 0].tolist())
        == "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n<|im_start|>user\n<|speaker:0|> Hi.<|im_end|>\n<|im_start|>assistant\n<|voice|>"
    )
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    rows = build_prompt("Hi.", tok, config, [" spoken"], [codes])
    audio = rows[rows[:, 0] >= config.semantic_start_token_id]
    assert audio.tolist() == [[config.semantic_start_token_id + 1, 1, 3, 5], [config.semantic_start_token_id + 2, 2, 4, 6]]
    assert "[Reference 1]\nText: <|speaker:0|>  spoken\nSpeech: " in tok.decode(rows[:, 0].tolist())
    for bad in ("", "<|im_end|>", "<|speaker:16|> hi", "<|speaker:01|> hi"):
        with pytest.raises(PromptError):
            build_prompt(bad, tok, config)


def test_clone_and_no_reference_repeat_under_a_seed(loaded, reference):
    (a,), ctx = speak(loaded, voice_of(reference), seed=5)
    (b,), _ = speak(loaded, voice_of(reference), seed=5)
    (c,), _ = speak(loaded, voice_of(reference), seed=6)
    assert a.final and a.sample_rate == 44100 and a.samples.size and np.isfinite(a.samples).all()
    assert (a.samples == b.samples).all() and not (a.samples.shape == c.samples.shape and (a.samples == c.samples).all())
    assert ctx.applied == {"streaming": False, "references": 1, "sampler": "default", "temperature": 0.7, "top_p": 0.7, "top_k": 30, "max_new_tokens": 8, "seed": 5}
    (d,), _ = speak(loaded, VoiceSpec(), seed=5)  # no reference: a voice of its own
    (e,), _ = speak(loaded, VoiceSpec(), seed=5)
    assert d.samples.size and (d.samples == e.samples).all()


def test_the_reference_sampler_and_bad_settings(loaded, reference):
    (a,), ctx = speak(loaded, voice_of(reference), sampler="reference_ras", epsilon_cutoff=0.01, params={"top_k": 0})
    assert a.samples.size and ctx.applied["sampler"] == "reference_ras" and ctx.applied["epsilon_cutoff"] == 0.01
    (b,), ctx = speak(loaded, voice_of(reference), sampler="reference_argmax")
    assert b.samples.size and ctx.applied["epsilon_cutoff"] == 0.001
    for options, params in (({"epsilon_cutoff": 0.01}, {}), ({}, {"top_k": 0}), ({"sampler": "greedy"}, {})):
        with pytest.raises(EngineError):
            speak(loaded, voice_of(reference), params=params, **options)


def test_long_text_streams_one_chunk_per_piece_and_single_chunk_is_one(loaded, reference):
    text = " ".join(f"Sentence number {i} goes on for a while, with commas and clauses." for i in range(6))
    chunks, _ = speak(loaded, voice_of(reference), text=text, stream=True)
    assert len(chunks) == len(split_text(text)) + 1 and [c.final for c in chunks][-1] and chunks[-1].samples.size == 0
    (whole,), _ = speak(loaded, voice_of(reference), text=text)
    assert np.array_equal(whole.samples, np.concatenate([c.samples for c in chunks]))
    (one,), ctx = speak(loaded, voice_of(reference), text=text, single_chunk=True)
    assert ctx.applied["single_chunk"] and one.samples.size


def test_the_voice_cache_holds_the_codes(loaded, reference, tmp_path):
    voice = voice_of(reference)
    with torch.inference_mode():
        cache = loaded.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    assert cache is not None and Path(cache.files["codes"]).is_file()
    assert (speak(loaded, voice_of(reference, cache=cache.files))[0][0].samples == speak(loaded, voice)[0][0].samples).all()


def test_cancel(loaded, reference):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, voice_of(reference), ctx=ctx)


def test_runs_identically_with_its_weights_placed_by_phase_and_its_experts_streamed_by_the_row(loaded, reference, monkeypatch):
    """The backbone, tower and codec each in its phase on a simulated device: streamed, the backbone's experts copy only
    the rows a call reads (prefill expert by expert, a step's experts at once): the same audio."""
    from mizuoto.runtime import rows
    from tests.engines_native.streaming import check_streamed

    taken: list[int] = []
    original = loaded.h_backbone.take_row
    monkeypatch.setattr(loaded.h_backbone, "take_row", lambda m, n, r: taken.append(r) or original(m, n, r))

    def run(manager, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return np.concatenate([c.samples for c in speak(loaded, voice_of(reference), ctx=ctx, **kw)[0]])

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0 and taken
    assert all(rows.source(layer.mlp.experts) is None for layer in loaded.backbone.model.layers)  # whole again after
    text = " ".join(f"Sentence number {i} goes on for a while, with commas and clauses." for i in range(4))
    check_streamed(loaded, lambda m: run(m, text=text, stream=True), monkeypatch)


# -- CUDA graphs ------------------------------------------------------------------------------------------------------


class ReplayedGraphs(Graphs):
    """``mizuoto.runtime.graphs.Graphs``' contract on the CPU, replayed as a graph replays. A capture runs the stretch once
    to warm up and once more as the capture (which on a GPU only records), putting back its state and the generator
    after each. A replay runs the stretch again on the names its capture read (``namespace``, the decoder's ``Stretches``'),
    then copies what it made into the tensors its capture made, which the eager calls and later stretches hold: as a
    graph, which reads and writes fixed addresses. A stretch that rebinds a name an earlier one bound reads its own
    warm-up's tensor here, as on a GPU."""

    def __init__(self) -> None:  # no stream: nothing runs on a GPU
        self.device = torch.device("cpu")
        self.generator = torch.Generator()
        self.captures = 0
        self.namespace: dict = {}

    def pool(self):
        return None

    def capture(self, fn, state=(), pool=None):
        saved = self.generator.get_state()
        ns = self.namespace

        def run() -> None:
            before = [t.clone() for t in state]
            fn()
            for t, b in zip(state, before, strict=True):
                t.copy_(b)
            self.generator.set_state(saved)

        run()  # the warm-up
        read = dict(ns)
        run()  # the capture
        made = {k: v for k, v in ns.items() if read.get(k) is not v}
        _poison(made)  # a capture only records: what it made holds garbage until a replay
        self.captures += 1

        def replay() -> None:
            now = dict(ns)
            ns.clear()
            ns.update(read)
            fn()
            for k, v in made.items():
                _copy_into(v, ns[k])
            ns.clear()
            ns.update(now)

        return type("Replayed", (), {"replay": staticmethod(replay)})()


def _poison(value) -> None:
    if isinstance(value, torch.Tensor):  # not integers: a capture's names also hold fixed ones (a context's positions)
        if value.is_floating_point():
            value.fill_(float("nan"))
        elif value.dtype == torch.bool:
            value.fill_(False)
    elif isinstance(value, dict | list | tuple):
        for v in value.values() if isinstance(value, dict) else value:
            _poison(v)
    elif hasattr(value, "__dict__"):  # a LayerContext
        _poison(list(vars(value).values()))


def _copy_into(target, value) -> None:
    if isinstance(target, torch.Tensor):
        target.copy_(value)
    elif isinstance(target, (list, tuple)):
        for t, v in zip(target, value, strict=True):
            _copy_into(t, v)
    else:  # a LayerContext
        for name, t in vars(target).items():
            if isinstance(t, torch.Tensor):
                t.copy_(getattr(value, name))


def test_the_step_as_graph_stretches_decodes_exactly_as_eagerly(loaded, reference, monkeypatch):
    """The graphed step (``drama3.steps``: the tower's and the backbone's stretches around their attention calls, the
    fixed tensors, the window of recent main tokens, the Gated DeltaNet states updated in place) gives the eager frames
    for both samplers, several chunks and requests after the first; the tower's graphs are captured again when the
    sampling settings change."""
    from mizuoto_fishspeech.drama3 import steps
    from mizuoto_fishspeech.drama3.generate import Decoder

    graphs = ReplayedGraphs()
    stretches = steps.Stretches.__init__

    def stretches_with_names(self, graphs, parts, calls, namespace) -> None:
        graphs.namespace = namespace
        stretches(self, graphs, parts, calls, namespace)

    monkeypatch.setattr(steps.Stretches, "__init__", stretches_with_names)
    text = " ".join(f"Sentence number {i} goes on for a while, with commas and clauses." for i in range(3))
    cases = [
        ({}, {}, "Hello there."),
        ({}, {}, text),
        ({"sampler": "reference_ras", "epsilon_cutoff": 0.01}, {"top_k": 0}, "Hello there."),
        ({"sampler": "reference_argmax"}, {"temperature": 0.9}, "Hello there."),
        ({}, {}, "Hello there."),
    ]
    want = []
    for options, params, words in cases:
        (chunk,), _ = speak(loaded, voice_of(reference), seed=4, text=words, params={"max_new_tokens": 24, **params}, **options)
        want.append(chunk.samples)
    (unvoiced,), _ = speak(loaded, VoiceSpec(), seed=4, params={"max_new_tokens": 24})
    try:
        loaded._decoder = Decoder(loaded.backbone, loaded.tower, loaded.config, lambda: graphs)
        captures = []
        for (options, params, words), expected in zip(cases, want, strict=True):
            (chunk,), ctx = speak(loaded, voice_of(reference), seed=4, text=words, params={"max_new_tokens": 24, **params}, **options)
            assert ctx.applied["cuda_graphs"] and chunk.samples.shape == expected.shape and (chunk.samples == expected).all()
            captures.append(graphs.captures)
        # the backbone's two stretches and the tower's five, once; the tower's again for each change of settings
        assert captures == [7, 7, 12, 17, 22]
        (chunk,), _ = speak(loaded, VoiceSpec(), seed=4, params={"max_new_tokens": 24})  # no reference: a short prompt
        assert chunk.samples.shape == unvoiced.samples.shape and (chunk.samples == unvoiced.samples).all()
    finally:
        loaded._decoder = None


def _cuda_loaded(model_dir, graphs: bool, dtype: str = "auto"):
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "codec_config", lambda folder: TINY_CODEC)
        loaded = get_engine("fishspeech").load(
            LoadSpec("fishspeech/tiny-drama3", model_dir, "drama3", "cuda", dtype=dtype, options={}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
    for h in loaded.handles:
        h.load()
    return loaded


def _cuda_speak(loaded, reference, seed: int, text: str, **options):
    ctx = RequestContext("r", seed=seed, device="cuda")
    params = {"seed": seed, "max_new_tokens": 24, **options.pop("params", {})}
    task = SynthesisTask(text=text, language="en", voice=voice_of(reference), params=params, options=options)
    with torch.inference_mode():
        return [c.samples for c in loaded.synthesize(task, ctx)], ctx.applied.get("cuda_graphs")


NVIDIA = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")


@NVIDIA
@pytest.mark.parametrize("dtype", ["auto", "fp16"])
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(model_dir, reference, dtype):
    """compute.cuda_graphs: the tower's and the backbone's graphs around their attention calls replay the eager kernels:
    the same audio for fresh loads (the request that captures), later requests, several chunks, both samplers (the
    tower's graphs captured again) and other seeds."""
    text = " ".join(f"Sentence number {i} goes on for a while, with commas and clauses." for i in range(3))
    cases = [
        (3, "Hello there.", {}),
        (4, text, {}),
        (3, "Hello there.", {"sampler": "reference_ras"}),
        (5, "Hello there.", {"params": {"temperature": 0.9}}),
        (3, "Hello there.", {}),
    ]
    for _ in range(2):
        eager, graphed = _cuda_loaded(model_dir, False, dtype), _cuda_loaded(model_dir, True, dtype)
        for seed, words, options in cases:
            want, used = _cuda_speak(eager, reference, seed, words, **dict(options))
            assert not used
            got, used = _cuda_speak(graphed, reference, seed, words, **dict(options))
            assert used
            assert len(want) == len(got) and all(a.shape == b.shape and (a == b).all() for a, b in zip(want, got, strict=True))


@NVIDIA
def test_a_failed_capture_falls_back_to_the_same_audio_on_a_gpu(model_dir, reference, monkeypatch, caplog):
    """A step that cannot be captured (here: one that reads a value back, which a capture forbids) is decoded eagerly,
    drawing from a fresh generator in the same state: the same audio as without graphs."""
    from mizuoto_fishspeech.drama3.tower import CodebookTower

    want, _ = _cuda_speak(_cuda_loaded(model_dir, False), reference, 3, "Hello there.")
    first_code = CodebookTower.first_code
    monkeypatch.setattr(CodebookTower, "first_code", lambda self, main: (main.item(), first_code(self, main))[1])
    got, used = _cuda_speak(_cuda_loaded(model_dir, True), reference, 3, "Hello there.")
    assert not used and "CUDA graphs failed" in caplog.text
    assert len(want) == len(got) and all(a.shape == b.shape and (a == b).all() for a, b in zip(want, got, strict=True))
