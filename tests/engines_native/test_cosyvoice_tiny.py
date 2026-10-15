"""The native CosyVoice engine on a tiny random CosyVoice 2-shaped model: no weights needed.

Real-model parity is checked by scripts/parity and the experiment scripts (identical to upstream given the same
voice prompt); this keeps the code path covered in CI: the yaml, the tokenizers, the frontend features, the
language model, the flow, HiFT, voice prompts and their cache, cross-lingual cloning, instructions, preset
speakers, streamed audio and text, voice conversion, seeds and cancellation.
"""

from __future__ import annotations

import json
import queue
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
for _module in ("tokenizers", "safetensors", "yaml", "inflect", "regex"):
    pytest.importorskip(_module)

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.runtime.context import Cancelled, RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, StyleSpec, SynthesisTask, VoiceConversionTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime, RuntimeOptions  # noqa: E402

YAML = """
sample_rate: 24000
llm_input_size: 32
spk_embed_dim: 192
chunk_size: 25
token_mel_ratio: 2
llm: !new:cosyvoice.llm.llm.Qwen2LM
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_input_size>
    speech_token_size: 6561
flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithXvec
    input_size: 16
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    vocab_size: 6561
    input_frame_rate: 25
    token_mel_ratio: !ref <token_mel_ratio>
    pre_lookahead_len: 3
    encoder: !new:cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder
        output_size: 16
        attention_heads: 2
        linear_units: 32
        num_blocks: 1
        input_size: 16
        static_chunk_size: !ref <chunk_size>
    decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        in_channels: 240
        cfm_params: !new:omegaconf.DictConfig
            content:
                inference_cfg_rate: 0.7
        estimator: !new:cosyvoice.flow.decoder.CausalConditionalDecoder
            in_channels: 320
            out_channels: 80
            channels: [16]
            attention_head_dim: 8
            n_blocks: 1
            num_mid_blocks: 1
            num_heads: 2
            static_chunk_size: !ref <chunk_size> * <token_mel_ratio>
hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
    in_channels: 80
    base_channels: 16
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    upsample_rates: [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params:
        n_fft: 16
        hop_len: 4
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor
        in_channels: 80
        cond_channels: 16
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False
"""


def _byte_vocab() -> dict[str, int]:
    from tokenizers.pre_tokenizers import ByteLevel

    return {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}


def _random_state(module: torch.nn.Module, seed: int, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in module.state_dict().items():
        if not t.is_floating_point():
            out[name] = t
        elif name.endswith(("running_var", "alpha")):
            out[name] = torch.ones_like(t)
        elif name.endswith("running_mean"):
            out[name] = torch.zeros_like(t)
        else:
            out[name] = (torch.randn(t.shape, generator=g) * scale).to(t.dtype)
    return out


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    from mizuoto.engines.cosyvoice.config import read_config
    from mizuoto.engines.cosyvoice.engine import hift_kwargs
    from mizuoto.engines.cosyvoice.flow import Flow
    from mizuoto.engines.cosyvoice.llm import QwenLM
    from mizuoto.nn.codecs.s3tokenizer import S3Tokenizer
    from mizuoto.nn.speaker.campplus import CAMPPlus
    from mizuoto.nn.vocoders.hift import HiFTGenerator
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("tiny-cosyvoice2")
    (root / "cosyvoice2.yaml").write_text(YAML, encoding="utf-8")
    blank = root / "CosyVoice-BlankEN"
    blank.mkdir()
    vocab = _byte_vocab()
    (blank / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (blank / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(len(vocab)): {"content": "<|endoftext|>", "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False}}
    (blank / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
    qwen = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 300,
    }
    (blank / "config.json").write_text(json.dumps(qwen), encoding="utf-8")
    config = read_config(root)
    llm = _random_state(QwenLM(2, blank, 6561, ops.default), 1)
    # random logits would rarely end and could pick the fill or another stop token (which upstream's streamed-text
    # path refuses): make EOS likely and the other special tokens impossible
    llm["llm_decoder.bias"] = torch.zeros(6564)
    llm["llm_decoder.bias"][6561] = 6.0
    llm["llm_decoder.bias"][6562:] = -1e4
    torch.save(llm, root / "llm.pt")
    torch.save(_random_state(Flow(2, config["flow"], ops.default), 2), root / "flow.pt")
    torch.save(_random_state(HiFTGenerator(**hift_kwargs(config["hift"]), ops=ops.default), 3, scale=0.05), root / "hift.pt")
    save_file(_random_state(S3Tokenizer(2, ops.default, n_state=128, n_head=2, n_layer=1), 4), str(root / "speech_tokenizer_v2.safetensors"))
    save_file({k: v for k, v in _random_state(CAMPPlus(ops=ops.default), 5).items() if not k.endswith("num_batches_tracked")}, str(root / "campplus.safetensors"))
    torch.save({"calm": {"embedding": torch.randn(1, 192, generator=torch.Generator().manual_seed(6))}}, root / "spk2info.pt")
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    from mizuoto.engines.cosyvoice import CosyVoiceEngine

    engine = CosyVoiceEngine()
    assert engine.detect_variant(model_dir) == "v2"
    devices = DeviceService()
    return engine.load(LoadSpec("cosyvoice/tiny", model_dir, "v2", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    rate = 22050  # resampled to 16 and 24 kHz
    t = np.arange(int(rate * 1.5)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 180 * t) * (t > 0.2) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def speak(loaded, voice, text="hello there", seed=3, ctx=None, **kw):
    ctx = ctx or RequestContext("r", seed=seed)
    with torch.inference_mode():
        chunks = list(loaded.synthesize(SynthesisTask(text=text, language="en", voice=voice, params={"seed": seed}, **kw), ctx))
    return chunks, ctx


def joined(chunks) -> np.ndarray:
    return np.concatenate([c.samples for c in chunks])


def test_describe_lists_the_preset_speakers(loaded):
    d = loaded.describe()
    assert d.sample_rate == 24000 and d.speakers == []  # a v2 model's spk2info speakers are not offered as presets


def test_clone_through_the_cached_prompt_repeats_under_a_seed(loaded, reference, tmp_path):
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    with torch.inference_mode():
        cache = loaded.prepare_voice(voice, tmp_path / "voice", RequestContext("p"))
    assert cache is not None
    cached = VoiceSpec(references=voice.references, cache=cache.files)
    a, ctx = speak(loaded, cached)
    b, _ = speak(loaded, cached)
    direct, _ = speak(loaded, voice)
    assert a[-1].final and np.isfinite(joined(a)).all() and joined(a).size > 0
    assert (joined(a) == joined(b)).all() and (joined(a) == joined(direct)).all()  # the prompt has no randomness
    assert ctx.applied["cached_speaker"] is True


def test_cross_lingual_instruct_and_preset(loaded, reference):
    no_text = VoiceSpec(references=[Reference(audio_path=str(reference), transcript=None)])
    assert joined(speak(loaded, no_text)[0]).size > 0
    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    chunks, ctx = speak(loaded, clip, style=StyleSpec(instruct="speak slowly"))
    assert joined(chunks).size > 0 and ctx.applied["instruct"] == "speak slowly"
    assert joined(speak(loaded, VoiceSpec(preset="calm"))[0]).size > 0


def test_streamed_audio_and_text(loaded, reference):
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    chunks, _ = speak(loaded, voice, text="a somewhat longer sentence to say", stream=True)
    assert chunks[-1].final and all(np.isfinite(c.samples).all() for c in chunks)
    q: queue.Queue[tuple[str, bool]] = queue.Queue()
    for piece, final in (("hello ", False), ("there.", True)):
        q.put((piece, final))
    ctx = RequestContext("t", seed=1, text_input=q)
    chunks, ctx = speak(loaded, voice, text="", ctx=ctx, text_stream=True)
    assert ctx.applied["text_stream"] is True and chunks[-1].final


def test_voice_conversion(loaded, reference):
    task = VoiceConversionTask(source_path=str(reference), voice=VoiceSpec(references=[Reference(audio_path=str(reference))]), params={"speed": 1.2})
    with torch.inference_mode():
        chunks = list(loaded.convert(task, RequestContext("c", seed=1)))
    assert len(chunks) == 1 and chunks[0].final and chunks[0].samples.size > 0


def test_cancel(loaded, reference):
    ctx = RequestContext("x", seed=1)
    threading.Timer(0.0, ctx.cancel).start()
    with pytest.raises(Cancelled):
        speak(loaded, VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")]), ctx=ctx)


def test_compute_compile_reaches_the_flow_estimator(model_dir, loaded, reference):
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from mizuoto.runtime.compile import Compiled

    assert loaded.flow.decoder.compiled_estimator is None  # off by default: the eager code path
    devices = DeviceService()
    runtime = Runtime(devices, ModelManager(devices), RuntimeOptions(compile=True))
    on: Any = CosyVoiceEngine().load(LoadSpec("cosyvoice/tiny", model_dir, "v2", "cpu", options={}), runtime, RequestContext("load"))
    compiled = on.flow.decoder.compiled_estimator
    assert isinstance(compiled, Compiled) and compiled.module is on.flow.decoder.estimator
    calls = []

    def eager(*args, **kwargs):  # the compiled slot, run eagerly: no compilation in this test
        calls.append(1)
        return compiled.module(*args, **kwargs)

    compiled._compiled = eager
    voice = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    a, b = joined(speak(loaded, voice)[0]), joined(speak(on, voice)[0])
    assert len(calls) == 10 and np.array_equal(a, b)  # ten Euler steps


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, reference, monkeypatch):
    """On a simulated device, phase by phase (frontend, language model, flow and vocoder), with
    units streamed: the same audio, streamed or not, cloned, by instruction, from a preset and converted."""
    from tests.engines_native.streaming import check_streamed

    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def run(manager, voice=clip, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return joined(speak(loaded, voice, ctx=ctx, **kw)[0])

    def convert(manager):
        ctx = RequestContext("c", seed=1)
        ctx.manager = manager
        task = VoiceConversionTask(source_path=str(reference), voice=VoiceSpec(references=[Reference(audio_path=str(reference))]), params={"speed": 1.2})
        with torch.inference_mode():
            return joined(list(loaded.convert(task, ctx)))

    streamed = check_streamed(loaded, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, lambda m: run(m, text="a somewhat longer sentence to say", stream=True), monkeypatch)
    check_streamed(loaded, lambda m: run(m, style=StyleSpec(instruct="speak slowly")), monkeypatch)
    check_streamed(loaded, lambda m: run(m, voice=VoiceSpec(preset="calm")), monkeypatch)
    check_streamed(loaded, convert, monkeypatch)


@pytest.mark.parametrize("version", [1, 3])
def test_v1_and_v3_run_identically_placed_by_phase_and_streamed(version, reference, monkeypatch, tmp_path):
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from tests.engines_native.streaming import check_streamed
    from tests.engines_native.test_cosyvoice_train_tiny import _tiny

    folder = tmp_path / f"v{version}"
    folder.mkdir()
    model_dir = _tiny(folder, version)
    variant = CosyVoiceEngine().detect_variant(model_dir)
    devices = DeviceService()
    model = CosyVoiceEngine().load(
        LoadSpec(f"cosyvoice/tiny{version}", model_dir, variant or "", "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load")
    )
    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def run(manager, **kw):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        return joined(speak(model, clip, ctx=ctx, **kw)[0])

    streamed = check_streamed(model, run, monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(model, lambda m: run(m, text="a somewhat longer sentence to say", stream=True), monkeypatch)


def test_a_phase_that_runs_out_of_memory_runs_again_identically(loaded, reference, monkeypatch):
    """The flow runs out of memory once (after the language model sampled its tokens); the flow
    phase runs again with the request's draws put back: the same audio."""
    from tests.engines_native.streaming import check_oom_retry

    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def run(manager, retries=None):
        ctx = RequestContext("r", seed=3)
        ctx.manager = manager
        out = joined(speak(loaded, clip, ctx=ctx)[0])
        if retries is not None:
            retries.update(ctx.applied.get("oom_retries", {}))
        return out

    assert check_oom_retry(loaded, run, loaded.flow.decoder.estimator, monkeypatch) == {"flow": 1}


def test_the_static_kv_cache_gives_the_same_audio_on_the_cpu(model_dir, loaded, reference):
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()
    static = CosyVoiceEngine().load(
        LoadSpec("cosyvoice/tiny", model_dir, "v2", "cpu", options={}), Runtime(devices, ModelManager(devices), RuntimeOptions(kv_cache="static")), RequestContext("load")
    )
    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])
    assert np.array_equal(joined(speak(static, clip)[0]), joined(speak(loaded, clip)[0]))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
@pytest.mark.parametrize("version", [2, 3])
def test_cuda_decoding_draws_the_tokens_upstreams_sampler_draws(version, model_dir, reference, monkeypatch, tmp_path):
    """On CUDA the language model's draw waits once per token (``nucleus_device``) and, with compute.cuda_graphs, runs
    as graphs (mizuoto.runtime.graphs): the same audio as upstream's sampling loop on the same device, for fresh loads
    (the requests that capture), later requests, streamed audio and streamed text."""
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from mizuoto.engines.cosyvoice.llm import _Steps
    from mizuoto.runtime.runtime import RuntimeOptions
    from tests.engines_native.test_cosyvoice_train_tiny import _tiny

    folder = model_dir
    if version == 3:
        (tmp_path / "v3").mkdir()
        folder = _tiny(tmp_path / "v3", 3)
    devices = DeviceService()

    def load(graphs: bool):
        model = CosyVoiceEngine().load(
            LoadSpec(f"cosyvoice/tiny{version}", folder, CosyVoiceEngine().detect_variant(folder) or "", "cuda", options={}),
            Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
            RequestContext("load"),
        )
        for h in model.handles:
            h.load()
        return model

    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def requests(model):
        out = []
        for seed, kw in ((3, {}), (4, {"text": "a somewhat longer sentence to say", "stream": True}), (3, {})):
            chunks, ctx = speak(model, clip, seed=seed, ctx=RequestContext("r", seed=seed, device="cuda"), **kw)
            out.append((joined(chunks), ctx.applied.get("cuda_graphs")))
        q: queue.Queue[tuple[str, bool]] = queue.Queue()
        for piece, final in (("hello ", False), ("there, and some more.", True)):
            q.put((piece, final))
        try:  # a tiny random model may draw a token streamed text forbids: then every way must fail alike
            chunks, ctx = speak(model, clip, text="", ctx=RequestContext("t", seed=5, device="cuda", text_input=q), text_stream=True)
            out.append((joined(chunks), ctx.applied.get("cuda_graphs")))
        except ValueError as e:
            out.append((np.array([hash(str(e))]), True))
        return out

    for _ in range(2):
        with monkeypatch.context() as patch:  # upstream's loop, as Mizuoto ran it before
            patch.setattr(_Steps, "sample", lambda self, logp, decoded, generator, ignore_eos: self.lm._sample(logp, decoded, generator, ignore_eos))
            upstream = requests(load(False))
        device = requests(load(False))
        graphed = requests(load(True))
        for (want, _), (eager, _), (got, used) in zip(upstream, device, graphed, strict=True):
            assert used
            assert want.shape == eager.shape == got.shape and (want == eager).all() and (want == got).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(model_dir, reference, monkeypatch, caplog):
    """A draw that cannot be captured (here: one that reads a value back) is made eagerly, from a fresh generator in the
    same state: the same audio as without graphs."""
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from mizuoto.engines.cosyvoice.llm import _Steps
    from mizuoto.runtime.runtime import RuntimeOptions

    devices = DeviceService()
    clip = VoiceSpec(references=[Reference(audio_path=str(reference), transcript="a reference")])

    def run(graphs: bool):
        model = CosyVoiceEngine().load(
            LoadSpec("cosyvoice/tiny", model_dir, "v2", "cuda", options={}), Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)), RequestContext("load")
        )
        for h in model.handles:
            h.load()
        return joined(speak(model, clip, ctx=RequestContext("r", seed=3, device="cuda"))[0])

    want = run(False)
    pick = _Steps._pick
    monkeypatch.setattr(_Steps, "_pick", lambda self, logp, generator, ignore_eos: (top := pick(self, logp, generator, ignore_eos), top.item())[0])
    got = run(True)
    assert "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()
