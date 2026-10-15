"""GPT-SoVITS v4/v5 SoVITS and the 48 kHz vocoder on tiny random models: no weights needed.

Real-model parity (identical to upstream sample for sample on the CPU) is checked by the experiment scripts in
``debug/gsv-v4v5``; this covers the code paths: ``decode_encp`` with speeds, the v4 CFM (step embedding) and v5's
(static cache, rolling prompt), both vocoder routines, seeds, the step callback and LoRA merging.
"""

from __future__ import annotations

import shutil
import sys
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from mizuoto.engines.gptsovits import sovits_v3 as sv  # noqa: E402
from mizuoto.engines.gptsovits.vocoder import Generator  # noqa: E402
from mizuoto.runtime.compile import Compiled  # noqa: E402
from mizuoto.runtime.engine import LoadSpec  # noqa: E402
from mizuoto.runtime.ops import init_random  # noqa: E402

DIT = {"dim": 32, "depth": 2, "heads": 2, "dim_head": 16, "ff_mult": 2, "conv_layers": 1}


def _model(version: str) -> sv.SynthesizerTrnV3:
    g = torch.Generator().manual_seed(0)
    model = sv.SynthesizerTrnV3(
        version, inter_channels=16, hidden_channels=16, filter_channels=32, n_heads=2, n_layers=2, gin_channels=16, bridge_channels=16, wn_layers=2, dit=DIT
    )
    init_random(model, g, scale=0.05)
    with torch.no_grad():
        for layer in model.quantizer.vq.layers:
            layer.get_submodule("_codebook").get_buffer("embed").copy_(torch.randn(1024, 768, generator=g))
    return model.eval()


def _vocoder() -> Generator:
    voc = Generator(100, "1", [3], [[1, 3, 5]], [10, 6, 2, 2, 2], 32, [20, 12, 4, 4, 4], is_bias=True)
    init_random(voc, torch.Generator().manual_seed(1), scale=0.05)
    return voc.eval()


def _prompt(seconds: float = 2.0) -> sv.Prompt:
    g = torch.Generator().manual_seed(2)
    audio = torch.rand(1, int(32000 * seconds), generator=g) * 0.2 - 0.1
    frames = int(seconds * 32000 / 640) + 1
    return sv.Prompt(
        torch.randint(0, 1024, (int(seconds * 25) + 7,), generator=g),
        torch.randint(0, 732, (12,), generator=g).tolist(),
        torch.rand(1, 1025, frames, generator=g),
        sv.reference_mel(audio),
    )


@pytest.fixture(scope="module", params=["v4", "v5turbo"])
def parts(request):
    return request.param, _model(request.param), _vocoder(), _prompt()


def _speak(model, voc, prompt, codes, phones, seed, **kw):
    return sv.vocoder_synthesis(model, voc, prompt, codes, phones, generator=torch.Generator().manual_seed(seed), **kw)


def test_mel_and_spec_helpers() -> None:
    audio = torch.rand(1, 32000) - 0.5
    mel = sv.reference_mel(audio)
    assert mel.shape == (1, 100, 100)
    assert torch.allclose(sv.denorm_spec(sv.norm_spec(mel)), mel, atol=1e-5)
    stereo = torch.stack([audio[0], audio[0]])
    assert torch.equal(sv.reference_mel(stereo), mel)


def test_extract_latent_round_trips_codebook(parts) -> None:
    _, model, _, _ = parts
    codes = model.extract_latent(torch.randn(1, 768, 40, generator=torch.Generator().manual_seed(3)))
    assert codes.shape == (1, 1, 20)
    assert int(codes.max()) < 1024


@pytest.mark.parametrize("speed", [1.0, 1.25, 0.8])
def test_vocoder_synthesis(parts, speed) -> None:
    version, model, voc, prompt = parts
    g = torch.Generator().manual_seed(4)
    codes = torch.randint(0, 1024, (1, 1, 30), generator=g)
    phones = torch.randint(0, 732, (1, 10), generator=g)
    steps = []
    kw: dict[str, Any] = {"sample_steps": 2} if version == "v4" else {}
    a = _speak(model, voc, prompt, codes, phones, 7, speed=speed, step=lambda i, n: steps.append((i, n)), **kw)
    frames = 2 * (int(30 * 2 / speed) + 1) if speed != 1 else 120  # the text encoder interpolates, then 2× upsampling
    assert a.ndim == 1 and a.shape[0] == frames * sv.UPSAMPLE_RATE
    assert torch.isfinite(a).all() and float(a.abs().max()) > 0
    assert steps and steps[0] == (0, 2 if version == "v4" else 4)
    assert torch.equal(a, _speak(model, voc, prompt, codes, phones, 7, speed=speed, **kw))
    assert not torch.equal(a, _speak(model, voc, prompt, codes, phones, 8, speed=speed, **kw))


def test_long_target_is_chunked(parts) -> None:
    """More frames than one CFM chunk: v4's 1000 − prompt frames, v5's rolling 640-frame chunks."""
    version, model, voc, prompt = parts
    g = torch.Generator().manual_seed(5)
    codes = torch.randint(0, 1024, (1, 1, 260), generator=g)  # 1040 frames
    phones = torch.randint(0, 732, (1, 30), generator=g)
    calls = []
    a = _speak(model, voc, prompt, codes, phones, 9, sample_steps=1, step=lambda i, n: calls.append(i))
    assert a.shape[0] == 1040 * sv.UPSAMPLE_RATE and torch.isfinite(a).all()
    assert len(calls) == 2


def test_batched_synthesis(parts) -> None:
    version, model, voc, prompt = parts
    g = torch.Generator().manual_seed(6)
    frags = [(torch.randint(0, 1024, (n,), generator=g), torch.randint(0, 732, (m,), generator=g)) for n, m in ((40, 12), (90, 20), (25, 8))]
    tokens = [torch.cat([torch.zeros(5, dtype=torch.long), c]) for c, _ in frags]
    idx = [c.shape[0] for c, _ in frags]
    phones = [p for _, p in frags]
    kw: dict[str, Any] = {"sample_steps": 2} if version == "v4" else {}
    out = sv.vocoder_synthesis_batched(model, voc, prompt, idx, tokens, phones, generator=torch.Generator().manual_seed(3), **kw)
    assert [o.shape[0] for o in out] == [n * 4 * sv.UPSAMPLE_RATE for n in idx]
    assert all(torch.isfinite(o).all() for o in out)
    again = sv.vocoder_synthesis_batched(model, voc, prompt, idx, tokens, phones, generator=torch.Generator().manual_seed(3), **kw)
    assert all(torch.equal(x, y) for x, y in zip(out, again, strict=True))


def test_v5_has_no_step_embedding() -> None:
    assert _model("v5turbo").cfm.estimator.d_embed is None
    assert _model("v4").cfm.estimator.d_embed is not None
    with pytest.raises(ValueError):
        sv.SynthesizerTrnV3("v2")


def test_sola_joins_fragments() -> None:
    g = torch.Generator().manual_seed(7)
    frags = [torch.randn(4000, generator=g) for _ in range(3)]
    out = sv.sola_algorithm([f.clone() for f in frags], 480, search_len=32)
    assert torch.isfinite(out).all()
    assert 3 * 4000 - 2 * (480 + 32) <= out.shape[0] <= 3 * 4000 - 2 * 480


def test_merge_lora() -> None:
    g = torch.Generator().manual_seed(8)
    base = {"cfm.estimator.transformer_blocks.0.attn.to_q.weight": torch.randn(6, 5, generator=g), "ref_enc.fc.fc.bias": torch.zeros(3)}
    a, b = torch.randn(2, 5, generator=g), torch.randn(6, 2, generator=g)
    lora = {
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_A.default.weight": a.half(),
        "cfm.base_model.model.estimator.transformer_blocks.0.attn.to_q.lora_B.default.weight": b.half(),
        "cfm.base_model.model.estimator.rotary_embed.inv_freq": torch.ones(4).half(),
        "ref_enc.fc.fc.bias": torch.ones(3).half(),
    }
    out = sv.merge_lora(base, lora, rank=2)
    key = "cfm.estimator.transformer_blocks.0.attn.to_q.weight"
    assert torch.equal(out[key], base[key] + (b.half().float() @ a.half().float()) * 1.0)
    assert torch.equal(out["ref_enc.fc.fc.bias"], torch.ones(3))
    assert out["cfm.estimator.rotary_embed.inv_freq"].dtype == torch.float32
    with pytest.raises(ValueError):
        sv.merge_lora(base, {k: v for k, v in lora.items() if "lora_B" not in k}, rank=2)


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1024, (1, 1, 30), generator=g), torch.randint(0, 732, (1, 10), generator=g)


@pytest.mark.parametrize("compile", [False, True])
def test_the_engine_compiles_the_estimator_of_each_sovits_it_loads(monkeypatch, compile) -> None:
    from mizuoto.engines.gptsovits import checkpoints
    from mizuoto.engines.gptsovits import engine as gsv

    model = _model("v4")
    monkeypatch.setattr(checkpoints, "read_sovits", lambda path: SimpleNamespace(model_version="v4", lora=False))
    monkeypatch.setattr(sv, "load_synthesizer", lambda *args, **kwargs: model)
    loaded = object.__new__(gsv.LoadedGptSovits)  # only what _load_sovits reads
    loaded.spec = LoadSpec("gptsovits/tiny", None, "v4", "cpu")
    loaded.version, loaded.compile, loaded.base = "v4", compile, ("s1.ckpt", "s2Gv4.pth")
    loaded.device, loaded.precision = torch.device("cpu"), torch.float32
    assert loaded._load_sovits("voice.pth")[0] is model
    compiled = model.cfm.compiled_estimator
    assert compiled is None if not compile else isinstance(compiled, Compiled) and compiled.module is model.cfm.estimator


@pytest.mark.parametrize("version", ["v4", "v5turbo"])
def test_the_cfm_calls_its_compiled_estimator(version) -> None:
    eager, voc, prompt = _model(version), _vocoder(), _prompt()
    on = _model(version)
    on.cfm.compiled_estimator = compiled = Compiled(on.cfm.estimator, "tiny cfm")
    calls = []

    def run(*args, **kwargs):  # the compiled slot, run eagerly: no compilation in this test
        calls.append(1)
        return compiled.module(*args, **kwargs)

    compiled._compiled = run
    codes, phones = _inputs(4)
    kw: dict[str, Any] = {"sample_steps": 2} if version == "v4" else {}
    assert torch.equal(_speak(eager, voc, prompt, codes, phones, 7, **kw), _speak(on, voc, prompt, codes, phones, 7, **kw))
    assert len(calls) == (2 if version == "v4" else 4)


@pytest.mark.skipif(sys.platform == "win32" or not (shutil.which("g++") or shutil.which("clang++")), reason="the CPU inductor needs a C++ compiler")
def test_the_compiled_cfm_matches_eager() -> None:
    eager, voc, prompt = _model("v4"), _vocoder(), _prompt()
    on = _model("v4")
    on.cfm.compiled_estimator = compiled = Compiled(on.cfm.estimator, "tiny cfm")
    for seed, frames in ((4, 30), (5, 41)):  # a new length reuses the dynamic graphs
        g = torch.Generator().manual_seed(seed)
        codes, phones = torch.randint(0, 1024, (1, 1, frames), generator=g), torch.randint(0, 732, (1, 10), generator=g)
        a = _speak(eager, voc, prompt, codes, phones, 7, sample_steps=2)
        b = _speak(on, voc, prompt, codes, phones, 7, sample_steps=2)
        assert a.shape == b.shape and float((a - b).abs().max()) < 1e-4
    assert compiled.failure is None and compiled.calls == 4
