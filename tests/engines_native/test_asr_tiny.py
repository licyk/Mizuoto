"""The native speech recognition engine (mizuoto.engines.asr) on a tiny random Whisper.

Real-model parity is checked against v1 (transformers' pipeline) on the corpus; this covers the code paths in CI:
features, language detection, the prompt, beam search and greedy decoding, Whisper's timestamp rules and seek loop,
the pipeline's 30 s windows and their merge into text and timestamped segments, progress and cancellation.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
pytest.importorskip("soxr")

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.asr import decoding  # noqa: E402
from mizuoto.engines.asr import engine as engine_module  # noqa: E402
from mizuoto.engines.asr.model import WhisperConfig, WhisperModel, sinusoids  # noqa: E402
from mizuoto.engines.asr.text import WhisperText, chunk_bounds, clean_up_tokenization, decode_asr, find_longest_common_sequence, language_id  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.engine import EngineError, LoadSpec, SynthesisTask, Unsupported, get_engine  # noqa: E402

LANGS = ["en", "zh", "ja"]
SPECIALS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    *(f"<|{lang}|>" for lang in LANGS),
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
]
TIMESTAMPS = [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
MAX_LENGTH = 24


def _tokenizer(root: Path) -> dict[str, int | None]:
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    tok.add_tokens([AddedToken(t, normalized=True) for t in TIMESTAMPS])
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({"clean_up_tokenization_spaces": True}))
    return {t: tok.token_to_id(t) for t in [*SPECIALS, TIMESTAMPS[0]]}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("whisper")
    ids = {k: int(v) for k, v in _tokenizer(root).items() if v is not None}
    vocab = ids[TIMESTAMPS[0]] + len(TIMESTAMPS)
    eos, sot = ids["<|endoftext|>"], ids["<|startoftranscript|>"]
    config = {
        "model_type": "whisper",
        "d_model": 32,
        "encoder_layers": 1,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 64,
        "decoder_layers": 2,
        "decoder_attention_heads": 2,
        "decoder_ffn_dim": 64,
        "num_mel_bins": 80,
        "max_source_positions": 1500,
        "max_target_positions": MAX_LENGTH,
        "vocab_size": vocab,
        "pad_token_id": eos,
        "bos_token_id": eos,
        "eos_token_id": eos,
        "decoder_start_token_id": sot,
    }
    (root / "config.json").write_text(json.dumps(config))
    generation = {
        "decoder_start_token_id": sot,
        "eos_token_id": eos,
        "pad_token_id": eos,
        "bos_token_id": eos,
        "no_timestamps_token_id": ids["<|notimestamps|>"],
        "suppress_tokens": [1, 2, ids["<|translate|>"], ids["<|startofprev|>"]],
        "begin_suppress_tokens": [220, eos],
        "max_initial_timestamp_index": 50,
        "max_length": MAX_LENGTH,
        "lang_to_id": {f"<|{lang}|>": ids[f"<|{lang}|>"] for lang in LANGS},
        "task_to_id": {"transcribe": ids["<|transcribe|>"], "translate": ids["<|translate|>"]},
        "is_multilingual": True,
    }
    (root / "generation_config.json").write_text(json.dumps(generation))
    (root / "preprocessor_config.json").write_text(json.dumps({"feature_size": 80, "n_samples": 480000, "sampling_rate": 16000}))
    model = WhisperModel(WhisperConfig.from_dict(config), ops.default)
    ops.init_random(model, torch.Generator().manual_seed(0), scale=0.3)
    with torch.no_grad():
        model.encoder.embed_positions.weight.copy_(sinusoids(1500, 32))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(root / "model.safetensors"))
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    spec = LoadSpec(model_id="asr/tiny", model_dir=model_dir, variant="whisper", device="cpu", dtype="fp32")
    return get_engine("asr").load(spec, None, RequestContext("load"))  # ty: ignore[invalid-argument-type]


def _wav(path: Path, seconds: float, rate: int = 24000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    x = 0.3 * np.sin(2 * np.pi * 180 * t * (1 + 0.3 * np.sin(t))) + 0.05 * rng.standard_normal(t.shape)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _ctx(log: list | None = None) -> RequestContext:
    return RequestContext("r", progress=(lambda stage, value, message: log.append((stage, value))) if log is not None else None)


def test_engine_contract(model_dir, loaded):
    engine = get_engine("asr")
    assert engine.detect_variant(model_dir) == "whisper"
    assert engine.detect_variant(model_dir.parent) is None
    assert set(engine.OPTION_SCHEMA) == {"beam_size"}
    assert loaded.describe().sample_rate == 16000
    with pytest.raises(Unsupported):
        list(loaded.synthesize(SynthesisTask(text="x"), _ctx()))
    with pytest.raises(Unsupported):
        loaded.call("speak", {}, _ctx())


def test_features_are_30_second_windows(loaded):
    short = loaded.features(np.zeros(16000, dtype=np.float32))
    long = loaded.features(np.ones(16000 * 40, dtype=np.float32) * 0.1)
    assert short.shape == long.shape == (1, 80, 3000)
    assert short.dtype == torch.float32


def test_transcribe_returns_v1s_shape_and_repeats(loaded, tmp_path):
    path = _wav(tmp_path / "a.wav", 3.0)
    log: list = []
    with torch.inference_mode():
        first = loaded.call("transcribe", {"path": str(path), "language": "en"}, _ctx(log))
        again = loaded.call("transcribe", {"path": str(path), "language": "en"}, _ctx())
    assert set(first) == {"text", "language", "segments", "duration"}
    assert first == again
    assert first["language"] == "en"
    assert first["duration"] == pytest.approx(3.0)
    for s in first["segments"]:
        assert set(s) == {"start", "end", "text"} and isinstance(s["start"], float) and isinstance(s["end"], float)
    assert ("transcribing", 1.0) in log


def test_language_detection_is_one_decoder_step(loaded, tmp_path):
    path = _wav(tmp_path / "b.wav", 2.0, seed=1)
    log: list = []
    with torch.inference_mode():
        result = loaded.call("transcribe", {"path": str(path), "language": None}, _ctx(log))
        features = loaded.features(loaded._read(str(path)))
        hidden = loaded.model.encoder(features)
        ids = torch.tensor([[loaded.generation.decoder_start_token_id]])
        logits = loaded.model.decoder(ids, hidden, decoding.DecoderCache(2))[:, -1]
    lang_ids = loaded.generation.lang_to_id
    best = max(lang_ids, key=lambda token: float(logits[0, lang_ids[token]]))
    assert result["language"] == best.strip("<|>")
    assert ("detecting language", None) in log


def test_languages_map_as_transformers(loaded):
    lang_to_id = loaded.generation.lang_to_id
    assert language_id("en", lang_to_id) == language_id("<|en|>", lang_to_id) == language_id("English", lang_to_id)
    assert language_id("mandarin", lang_to_id) == language_id("zh-cn", lang_to_id) == lang_to_id["<|zh|>"]
    with pytest.raises(ValueError):
        language_id("xx", lang_to_id)
    with pytest.raises(ValueError):
        language_id("fr", lang_to_id)  # a Whisper language this model lacks


def test_unknown_language_is_invalid_input(loaded, tmp_path):
    path = _wav(tmp_path / "c.wav", 1.0)
    with pytest.raises(EngineError):
        loaded.call("transcribe", {"path": str(path), "language": "klingon"}, _ctx())


def test_long_audio_goes_in_overlapping_windows_four_at_a_time(loaded, tmp_path, monkeypatch):
    path = _wav(tmp_path / "long.wav", 100.0, rate=16000, seed=2)
    calls: list[tuple[int, int]] = []
    real = decoding.generate

    def spy(model, features, prompt, gen, num_beams, step, encoder_hidden=None):
        calls.append((features.shape[0], num_beams))
        return real(model, features, prompt, gen, num_beams, step, encoder_hidden)

    monkeypatch.setattr(decoding, "generate", spy)
    with torch.inference_mode():
        result = loaded.call("transcribe", {"path": str(path), "language": "zh"}, _ctx())
    assert [b for b, _ in calls] == [4, 1]  # five windows: 0-30, 20-50, 40-70, 60-90, 80-100 s
    assert all(n == 5 for _, n in calls)  # beam_size 1 is the pipeline's five beams, as in v1
    assert result["duration"] == pytest.approx(100.0)
    assert [(c.start, c.stop, c.stride) for c in chunk_bounds(16000 * 100)][1] == (320000, 800000, (480000, 80000, 80000))


def test_beam_size_option_sets_the_beams(loaded, tmp_path, monkeypatch):
    path = _wav(tmp_path / "d.wav", 2.0)
    seen: list[int] = []
    real = decoding.generate
    monkeypatch.setattr(decoding, "generate", lambda m, f, p, g, n, s, encoder_hidden=None: seen.append(n) or real(m, f, p, g, n, s, encoder_hidden))
    with torch.inference_mode():
        loaded.call("transcribe", {"path": str(path), "language": "en", "options": {"beam_size": 3}}, _ctx())
        loaded.call("transcribe", {"path": str(path), "language": "en", "options": {"beam_size": 1}}, _ctx())
    assert seen == [3, 5]


def test_greedy_and_beam_search_decode(loaded):
    gen = loaded.generation
    prompt = decoding.prompt_tokens(gen, gen.lang_to_id["<|en|>"])
    features = loaded.features(np.random.default_rng(3).standard_normal(16000).astype(np.float32) * 0.1)
    with torch.inference_mode():
        greedy = decoding.generate(loaded.model, features, prompt, gen, 1, lambda: None)
        beams = decoding.generate(loaded.model, features, prompt, gen, 4, lambda: None)
    for tokens in (greedy[0], beams[0]):
        assert tokens and tokens[0] >= gen.timestamp_begin  # the first token is a timestamp
        assert gen.eos_token_id not in tokens


def test_beam_search_without_the_shared_cross_cache_shortcut(loaded, monkeypatch):
    gen = loaded.generation
    prompt = decoding.prompt_tokens(gen, gen.lang_to_id["<|ja|>"])
    features = torch.cat([loaded.features(np.random.default_rng(s).standard_normal(16000 * 2).astype(np.float32) * 0.1) for s in (4, 5)])
    with torch.inference_mode():
        fast = decoding.generate(loaded.model, features, prompt, gen, 3, lambda: None)
        monkeypatch.setattr(decoding, "_beams_share", lambda *a: False)
        slow = decoding.generate(loaded.model, features, prompt, gen, 3, lambda: None)
    assert fast == slow


def test_timestamp_rules(loaded):
    gen = loaded.generation
    tb, eos = gen.timestamp_begin, gen.eos_token_id
    vocab = loaded.model.config.vocab_size
    proc = decoding.Processors(gen, vocab, 3, torch.device("cpu"))
    prompt = [gen.decoder_start_token_id, gen.lang_to_id["<|en|>"], gen.task_to_id["transcribe"]]
    scores = torch.zeros(1, vocab)
    scores[0, eos] = 20.0  # more likely than all timestamps together: text stays allowed
    first = proc(torch.tensor([prompt]), scores.clone())
    allowed = torch.isfinite(first[0]).nonzero().flatten().tolist()
    assert allowed == list(range(tb, tb + 51))  # only timestamps up to max_initial_timestamp_index
    after_text = proc(torch.tensor([[*prompt, tb, 65, tb + 10]]), scores.clone())  # one timestamp after text: no text
    assert not torch.isfinite(after_text[0, :eos]).any() and torch.isfinite(after_text[0, eos])
    assert not torch.isfinite(after_text[0, tb : tb + 10]).any() and torch.isfinite(after_text[0, tb + 10])
    after_pair = proc(torch.tensor([[*prompt, tb, 65, tb + 10, tb + 10]]), scores.clone())  # a pair: no timestamp
    assert not torch.isfinite(after_pair[0, tb:]).any()
    assert not torch.isfinite(after_pair[0, gen.no_timestamps_token_id])
    for t in gen.suppress_tokens:
        assert not torch.isfinite(after_pair[0, t])


def test_seek_loop_decodes_again_after_a_timestamp_pair(loaded, monkeypatch):
    gen = loaded.generation
    tb, eos = gen.timestamp_begin, gen.eos_token_id
    prompt = decoding.prompt_tokens(gen, gen.lang_to_id["<|en|>"])
    scripted = [[tb, 65, 66, tb + 50, tb + 50, 67, eos, eos], [tb, 68, tb + 25, eos]]
    inputs: list[torch.Tensor] = []
    real_encoder = loaded.model.encoder.forward

    def encoder(features):
        inputs.append(features.clone())
        return real_encoder(features)

    def fake_beam_search(model, hidden, input_ids, gen_, processors, num_beams, step):
        return torch.tensor([prompt + scripted.pop(0)])

    monkeypatch.setattr(loaded.model.encoder, "forward", encoder)
    monkeypatch.setattr(decoding, "beam_search", fake_beam_search)
    features = loaded.features(np.random.default_rng(6).standard_normal(16000 * 5).astype(np.float32) * 0.1)
    tokens = decoding.generate(loaded.model, features, prompt, gen, 5, lambda: None)
    # the first pass keeps everything up to the pair, drops " C" and seeks to 1.00 s (100 frames); the second
    # pass decodes from there, the rest padded with zeros
    assert tokens == [[tb, 65, 66, tb + 50, tb + 50, tb, 68, tb + 25]]
    assert len(inputs) == 2
    assert torch.equal(inputs[1][0, :, : 3000 - 100], features[0, :, 100:])
    assert torch.equal(inputs[1][0, :, 3000 - 100 :], torch.zeros(80, 100))


def test_decode_asr_text_and_segments(loaded):
    text: WhisperText = loaded.text
    tb = loaded.generation.timestamp_begin
    hello = text.tokenizer.encode("Hello", add_special_tokens=False).ids
    world = text.tokenizer.encode(" world .", add_special_tokens=False).ids
    # one window: two segments
    out, segments = decode_asr(text, [([tb, *hello, tb + 50, tb + 50, *world, tb + 100], (3.0, 0.0, 0.0))], tb, 0.02)
    assert out == "Hello world."
    assert [(s.text, s.timestamp) for s in segments] == [("Hello", (0.0, 1.0)), (" world.", (1.0, 2.0))]
    # a seek pass restarts the timestamps; the pipeline adds the earlier passes' length
    out, segments = decode_asr(text, [([tb, *hello, tb + 50, tb + 50, tb, *world, tb + 25], (3.0, 0.0, 0.0))], tb, 0.02)
    assert [s.timestamp for s in segments] == [(0.0, 1.0), (1.0, 1.5)]
    # no closing timestamp: the text is kept, the end left to the caller (the clip's duration)
    out, segments = decode_asr(text, [([tb, *hello], (3.0, 0.0, 0.0))], tb, 0.02)
    assert [(s.text, s.timestamp) for s in segments] == [("Hello", (0.0, None))]


def test_overlapping_windows_merge(loaded):
    assert find_longest_common_sequence([[1, 2, 3, 4], [3, 4, 6]]) == [1, 2, 3, 4, 6]
    text: WhisperText = loaded.text
    tb = loaded.generation.timestamp_begin
    a_to_h = text.tokenizer.encode("abcdefgh", add_special_tokens=False).ids
    # window 1 (0-30 s, 5 s right stride) hears "abcdef", its closing timestamp inside the stride; window 2 (from
    # 20 s, 5 s left stride) hears "efgh" again: the overlap "ef" is resolved, the segment ends at 20 + 12 s
    windows = [([tb, *a_to_h[:6], tb + 1400], (30.0, 0.0, 5.0)), ([tb, *a_to_h[4:], tb + 600], (20.0, 5.0, 0.0))]
    out, segments = decode_asr(text, windows, tb, 0.02)
    assert out == "abcdefgh"
    assert [s.timestamp for s in segments] == [(0.0, 32.0)]


def test_text_clean_up_matches_transformers():
    assert clean_up_tokenization("Hi , it 's here . Yes !") == "Hi, it's here. Yes!"


def test_cancel_stops_decoding(loaded, tmp_path):
    path = _wav(tmp_path / "e.wav", 2.0)
    ctx = RequestContext("r", progress=lambda stage, value, message: ctx.cancel() if stage == "transcribing" else None)
    with pytest.raises(Cancelled), torch.inference_mode():
        loaded.call("transcribe", {"path": str(path), "language": "en"}, ctx)


def test_engine_module_reproduces_v1_dtype_choice():
    assert engine_module._dtype("auto", "cpu") == torch.float32
    assert engine_module._dtype("fp16", "cpu") == torch.float16


def test_transcribes_identically_with_its_weights_streamed(loaded, tmp_path, monkeypatch):
    """On a simulated device, units streamed: the same transcript (and every window's tokens),
    the language given or detected."""
    from tests.engines_native.streaming import check_streamed

    path = _wav(tmp_path / "s.wav", 3.0)

    def run(manager, language="en"):
        ctx = _ctx()
        ctx.manager = manager
        with torch.inference_mode():
            result = loaded.call("transcribe", {"path": str(path), "language": language}, ctx)
        return np.frombuffer(repr(result).encode(), dtype=np.uint8)

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, lambda m: run(m, None), monkeypatch)
