"""The native OmniVoice engine on a tiny random model: no weights needed.

Real-model parity is checked by scripts/parity (identical audio to v1 on the CPU); this keeps the code path
covered in CI: loading, the HiggsAudio tokenizer (HuBERT, DAC, quantizer), voice prompts and their cache,
iterative decoding, the chunked path, seeds and cancellation.
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

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.omnivoice import OmniVoiceEngine  # noqa: E402
from mizuoto.engines.omnivoice.engine import VoicePrompt  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_end|>", "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>", "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]
QUIET = {"postprocess_output": False}


def _byte_vocab() -> dict[str, int]:
    from tokenizers.pre_tokenizers import ByteLevel

    return {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}


def _random_state(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
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


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    from mizuoto.engines.omnivoice.audio_tokenizer import HiggsAudioTokenizer
    from mizuoto.engines.omnivoice.model import OmniVoiceModel
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-omnivoice")
    vocab = _byte_vocab()
    base = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(base + i): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for i, s in enumerate(SPECIALS)}
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
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
        "vocab_size": base + len(SPECIALS),
    }
    config = {"model_type": "omnivoice", "audio_vocab_size": 17, "audio_mask_id": 16, "num_audio_codebook": 2, "llm_config": llm}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(_random_state(OmniVoiceModel(config, ops.default), 1), str(root / "model.safetensors"))

    tok = {
        "acoustic_model_config": {
            "encoder_hidden_size": 2,
            "downsampling_ratios": [8, 5, 4, 2, 3],
            "decoder_hidden_size": 32,
            "upsampling_ratios": [8, 5, 4, 2, 3],
            "hidden_size": 8,
        },
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
    (root / "audio_tokenizer").mkdir()
    (root / "audio_tokenizer" / "config.json").write_text(json.dumps(tok), encoding="utf-8")
    (root / "audio_tokenizer" / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 24000}), encoding="utf-8")
    tokenizer = HiggsAudioTokenizer(tok, ops.default)
    assert tokenizer.num_quantizers == 2
    state = _random_state(tokenizer, 2)
    # stored as transformers does: weight norm on the positional convolution (dim 2), training-only parts kept
    w = state.pop("semantic_model.encoder.pos_conv_embed.conv.weight")
    state["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = torch.linalg.vector_norm(w, dim=(0, 1), keepdim=True)
    state["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = w
    state["fc1.weight"] = torch.zeros(16, 24)
    save_file(state, str(root / "audio_tokenizer" / "model.safetensors"))
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    engine = OmniVoiceEngine()
    assert engine.detect_variant(model_dir) == "v0"
    devices = DeviceService()
    return engine.load(LoadSpec("omnivoice/tiny", model_dir, "v0", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    rate = 16000  # resampled to the model's 24 kHz
    t = np.arange(rate * 2) / rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.3) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def speak(loaded, voice=None, seed=3, text="Hello there.", ctx=None, options=QUIET, **params):
    ctx = ctx or RequestContext("r", seed=seed)
    task = SynthesisTask(text=text, language="en", voice=voice or VoiceSpec(), params={"seed": seed, "num_steps": 4, **params}, options=dict(options))
    with torch.inference_mode():
        return list(loaded.synthesize(task, ctx)), ctx


def test_its_own_voice_repeats_under_a_seed(loaded):
    (a,), ctx = speak(loaded, seed=5)
    (b,), _ = speak(loaded, seed=5)
    (c,), _ = speak(loaded, seed=6)
    assert a.final and a.sample_rate == 24000 and np.isfinite(a.samples).all()
    assert (a.samples == b.samples).all() and not (a.samples.shape == c.samples.shape and (a.samples == c.samples).all())
    assert ctx.applied["num_step"] == 4 and ctx.applied["guidance_scale"] == 2.0


def test_design_resolves_the_attributes(loaded):
    (chunk,), ctx = speak(loaded, voice=VoiceSpec(attributes={"gender": "female", "dialect": "四川话"}), text="你好。")
    assert ctx.applied["instruct"] == "female，四川话" and chunk.samples.size


def test_clone_through_the_cached_prompt(loaded, reference, tmp_path):
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    with torch.inference_mode():
        cache = loaded.prepare_voice(voice, tmp_path / "cache", RequestContext("p"))
    assert cache is not None
    prompt = VoicePrompt.load(Path(cache.files["prompt"]))
    assert prompt.tokens.shape[0] == 2 and prompt.text == "a reference." and prompt.rms > 0
    with torch.inference_mode():
        again = loaded._prompt(voice, True, RequestContext("p"))
    assert torch.equal(again.tokens, prompt.tokens)
    # without the cache the prompt is made in the request, whose HuBERT layer-drop draws come first (as upstream's)
    cached = VoiceSpec(references=voice.references, cache=cache.files)
    assert (speak(loaded, voice=cached)[0][0].samples == speak(loaded, voice=cached)[0][0].samples).all()
    assert speak(loaded, voice=voice)[0][0].samples.size


def test_duration_and_postprocessing(loaded):
    (chunk,), _ = speak(loaded, duration=1.0)
    # 1 s of frames (25 a second, 960 samples each) and 0.1 s of padding at each end
    assert chunk.samples.size == 25 * 960 + 2 * 2400
    (processed,), _ = speak(loaded, options={})
    assert np.isfinite(processed.samples).all()


def test_long_text_goes_in_chunks(loaded):
    text = "One sentence here. Another one follows. And a third, to be sure."
    (chunk,), _ = speak(loaded, text=text, options={**QUIET, "audio_chunk_threshold": 0.5, "audio_chunk_duration": 0.5})
    assert chunk.samples.size > 0
    texts = SynthesisTask(text="", texts=["One.", "Two, longer."], language="en", voice=VoiceSpec(), params={"seed": 1, "num_steps": 2}, options=dict(QUIET))
    with torch.inference_mode():
        assert [c.segment for c in loaded.synthesize(texts, RequestContext("b", seed=1))] == [0, 1]


def test_cancel(loaded):
    ctx = RequestContext("c", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, ctx=ctx)


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, reference, monkeypatch):
    """On a simulated device, phase by phase, with units streamed: the same audio (its own voice,
    a clone whose prompt is made in the request, long text in chunks)."""
    from tests.engines_native.streaming import check_streamed

    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def run(manager, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return np.concatenate([c.samples for c in speak(loaded, ctx=ctx, **kw)[0]])

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, lambda m: run(m, voice=voice), monkeypatch)
    long = "One sentence here. Another one follows. And a third, to be sure."
    check_streamed(loaded, lambda m: run(m, text=long, options={**QUIET, "audio_chunk_threshold": 0.5, "audio_chunk_duration": 0.5}), monkeypatch)


def test_packed_guidance_computes_each_sequence_as_its_own_row(loaded, reference):
    """``packed_guidance``: an item's conditional and unconditional sequences side by side in one row, under a
    block-diagonal mask with each one's positions from 0, give the logits upstream's two padded rows give (to rounding),
    several items of different lengths together; a request lists the option."""
    from mizuoto.engines.omnivoice.engine import _Item

    prompt = loaded._prompt(VoiceSpec(references=[Reference(audio_path=str(reference), transcript="Hello.")]), True, RequestContext("p"))
    items = [_Item("Hello there.", 9, "en", None, prompt, None), _Item("A longer line of text to say.", 14, "en", None, None, None)]
    inputs = [loaded._inputs(item, True) for item in items]
    targets = [it.target_len for it in items]
    with torch.inference_mode():
        logits = {}
        for packed in (False, True):
            ids, audio, attn, positions, places = loaded._layout(inputs, targets, packed)
            out = loaded.model(ids, audio, attn, positions).float()
            logits[packed] = [[out[row, :, at : at + t] for row, at in place] for place, t in zip(places, targets, strict=True)]
    for a, b in zip(logits[False], logits[True], strict=True):
        for x, y in zip(a, b, strict=True):
            assert x.shape == y.shape and torch.allclose(x, y, atol=1e-4, rtol=1e-4)
    (chunk,), ctx = speak(loaded, options={**QUIET, "packed_guidance": True})
    assert chunk.samples.size and np.isfinite(chunk.samples).all() and ctx.applied["packed_guidance"] is True
