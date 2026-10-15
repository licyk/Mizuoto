"""GPT-SoVITS' SoVITS decoder (v2, v2Pro) and SV model on tiny random weights.

Real-model parity (identical audio to upstream for v2, v2Pro and v2ProPlus on the CPU) is checked by the
experiment's ``debug/gsv-sovits/parity.py``; this covers the code paths in CI: checkpoints as upstream writes
them (a plain zip and a version-tagged one, float16, weight norm), the version rules, prompt codes, the reference
spectrogram, decoding (several references, speed), streaming chunks and seeds.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mizuoto.engines.gptsovits import sovits as S  # noqa: E402
from mizuoto.engines.gptsovits import sv as SV  # noqa: E402
from mizuoto.engines.gptsovits.checkpoints import read_sovits, sovits_version  # noqa: E402
from mizuoto.runtime import ops  # noqa: E402

# MRTE's sizes are fixed upstream (192 → 512 → 192), so hidden_channels is 192 and v2's gin 512.
MODEL = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 64,
    "n_heads": 2,
    "n_layers": 2,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "resblock": "1",
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
    "upsample_rates": [4, 4],
    "upsample_initial_channel": 32,
    "upsample_kernel_sizes": [8, 8],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "semantic_frame_rate": "25hz",
    "freeze_quantizer": True,
}
DATA = {"sampling_rate": 32000, "filter_length": 2048, "hop_length": 640, "win_length": 2048, "n_speakers": 300}
WEIGHT_NORMED = (".ups.", ".resblocks.", ".enc.in_layers.", ".enc.res_skip_layers.", ".enc.cond_layer")


def _hps(version: str) -> dict:
    return {"train": {"segment_size": 20480}, "data": dict(DATA), "model": {**MODEL, "gin_channels": 512 if version == "v2" else 64, "version": version}}


def _upstream_weights(model: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    """Random weights in the file's form: float16, upstream's weight norm (``weight_g``/``weight_v``), an ``enc_q`` key."""
    g = torch.Generator().manual_seed(seed)
    ops.init_random(model, g, scale=0.05)
    for name, buf in model.named_buffers():
        if name.endswith("_codebook.embed"):
            buf.copy_(torch.randn(buf.shape, generator=g))
    out: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.endswith(".weight") and v.dim() == 3 and any(p in k for p in WEIGHT_NORMED):
            base = k[: -len(".weight")]
            out[f"{base}.weight_g"] = v.flatten(1).norm(dim=1).view(-1, 1, 1).half()
            out[f"{base}.weight_v"] = v.half()
        else:
            out[k] = v.half() if v.is_floating_point() else v
    out["quantizer.vq.layers.0._codebook.inited"] = torch.tensor([1.0])
    out["enc_q.proj.weight"] = torch.zeros(4, 4, 1, dtype=torch.float16)  # training-only, ignored
    return out


def _write(path: Path, version: str, tag: bytes | None, seed: int) -> Path:
    model = S.SynthesizerTrn.from_hps({**_hps(version), "model": {**_hps(version)["model"]}})
    bio = io.BytesIO()
    torch.save({"weight": _upstream_weights(model, seed), "config": _hps(version), "info": "test"}, bio)
    data = bio.getvalue()
    path.write_bytes(tag + data[2:] if tag else data)
    return path


@pytest.fixture(scope="module", params=["v2", "v2Pro"])
def loaded(request, tmp_path_factory) -> tuple[S.SynthesizerTrn, object]:
    version = request.param
    folder = tmp_path_factory.mktemp(version)
    path = _write(folder / "s2G.pth", version, b"05" if version == "v2Pro" else None, seed=1)
    model, ckpt = S.load_synthesizer(path)
    return model, ckpt


def test_versions(loaded) -> None:
    model, ckpt = loaded
    if ckpt.path.read_bytes()[:2] == b"05":
        assert sovits_version(ckpt.path) == ("v2", "v2Pro", False)
        assert ckpt.model_version == "v2Pro" and model.is_v2pro
    else:
        # a small untagged file is "v1" by size; the 732-symbol embedding makes it v2, as init_vits_weights does
        assert sovits_version(ckpt.path) == ("v1", "v1", False)
        assert ckpt.model_version == "v2" and ckpt.version == "v2" and not model.is_v2pro
    assert ckpt.hps["model"]["semantic_frame_rate"] == "25hz"
    assert ckpt.spec_params == {"filter_length": 2048, "hop_length": 640, "win_length": 2048, "sampling_rate": 32000}
    assert not any(p.dtype != torch.float32 for p in model.parameters())


def _inputs(model: S.SynthesizerTrn, ckpt) -> tuple:
    g = torch.Generator().manual_seed(0)
    audio = (torch.randn(1, 32000, generator=g) * 0.2).clamp(-0.9, 0.9)
    spec, normed = S.reference_spec(audio, **ckpt.spec_params)
    assert spec.shape[1] == 1025 and torch.equal(normed, audio)
    loud = S.normalize_reference(audio * 10)
    assert loud.abs().max() <= 5.0 and torch.allclose(loud, audio * 10 / 2)
    codes = model.extract_latent(torch.randn(1, 768, 40, generator=g))
    assert codes.shape == (1, 1, 20) and codes.dtype == torch.long
    semantic = torch.randint(0, 1024, (1, 1, 30), generator=g)
    phones = torch.randint(0, 732, (1, 12), generator=g)
    sv = torch.randn(1, 20480, generator=g) if model.is_v2pro else None
    return semantic, phones, spec, sv


def test_decode_seeds(loaded) -> None:
    model, ckpt = loaded
    semantic, phones, spec, sv = _inputs(model, ckpt)
    sv_list = [sv] if sv is not None else None
    with torch.inference_mode():
        a = model.decode(semantic, phones, [spec], sv_emb=sv_list, generator=torch.Generator().manual_seed(5))
        b = model.decode(semantic, phones, [spec], sv_emb=sv_list, generator=torch.Generator().manual_seed(5))
        c = model.decode(semantic, phones, [spec], sv_emb=sv_list, generator=torch.Generator().manual_seed(6))
    assert a.shape == (1, 1, 30 * model.hop_samples)
    assert torch.isfinite(a).all() and a.abs().max() <= 1
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_decode_references_and_speed(loaded) -> None:
    model, ckpt = loaded
    semantic, phones, spec, sv = _inputs(model, ckpt)
    spec2 = spec[:, :, : spec.shape[2] // 2].contiguous()
    svs = [sv, sv * 0.5] if sv is not None else None
    with torch.inference_mode():
        ge = model.global_embedding([spec, spec2], svs)
        one = model.global_embedding([spec], svs[:1] if svs else None)
        assert ge.shape == (1, model.gin_channels, 1) and not torch.equal(ge, one)
        a = model.decode(semantic, phones, [spec, spec2], sv_emb=svs, generator=torch.Generator().manual_seed(1))
        b = model.decode(semantic, phones, None, ge=ge, generator=torch.Generator().manual_seed(1))  # type: ignore[arg-type]
        assert torch.equal(a, b)
        fast = model.decode(semantic, phones, [spec], sv_emb=svs[:1] if svs else None, speed=1.5, generator=torch.Generator().manual_seed(1))
    assert fast.shape[-1] == (int(60 / 1.5) + 1) * model.hop_samples // 2
    assert torch.isfinite(fast).all()


def test_decode_streaming(loaded) -> None:
    model, ckpt = loaded
    semantic, phones, spec, sv = _inputs(model, ckpt)
    svs = [sv] if sv is not None else None
    g = torch.Generator().manual_seed(3)
    last = None
    with torch.inference_mode():
        for i, end in enumerate((10, 20, 30)):
            n = 10
            overlap = 2
            audio, latent, mask = model.decode_streaming(
                semantic[:, :, :end],
                phones,
                [spec],
                sv_emb=svs,
                result_length=n + overlap if i else None,
                overlap_frames=last[:, :, -overlap * 2 :] if last is not None else None,
                padding_length=0,
                generator=g,
            )
            frames = 2 * end if i == 0 else 2 * (n + overlap)
            assert latent.shape == (1, 192, frames) and mask.shape == (1, 1, frames)
            assert audio.shape[-1] == frames * model.hop_samples // 2 and torch.isfinite(audio).all()
            last = latent


def test_sv_tiny(tmp_path: Path) -> None:
    config: dict = {"num_blocks": (1, 1, 1, 1), "m_channels": 8}
    model = SV.ERes2NetV2(baseWidth=24, scale=4, expansion=4, **config)
    ops.init_random(model, torch.Generator().manual_seed(2), scale=0.1)
    for name, buf in model.named_buffers():
        if name.endswith("running_var"):
            buf.fill_(1.0)
    path = tmp_path / "sv.ckpt"
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)
    loaded = SV.load_sv(path, **config)
    wav = torch.randn(2, 16000, generator=torch.Generator().manual_seed(4)) * 0.1
    feats = SV.sv_features(wav)
    assert feats.shape == (2, 98, 80)
    emb = SV.sv_embedding(loaded, wav)
    assert emb.shape == (2, 8 * 8 * 4 * 10) and torch.isfinite(emb).all()
    assert torch.equal(emb, SV.sv_embedding(loaded, wav))
    assert loaded(feats).shape == (2, 192)


def test_restricted_unpickler(tmp_path: Path) -> None:
    """Fine-tunes pickle upstream's ``utils.HParams``; it is read as a dict and nothing else is allowed."""
    import pickle
    import sys
    import types

    from mizuoto.engines.gptsovits.checkpoints import load_checkpoint

    utils = types.ModuleType("utils")

    class HParams:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update({k: HParams(**v) if isinstance(v, dict) else v for k, v in kw.items()})

    HParams.__module__ = "utils"
    HParams.__qualname__ = "HParams"
    setattr(utils, "HParams", HParams)  # noqa: B010
    sys.modules["utils"] = utils
    try:
        path = tmp_path / "ft.pth"
        torch.save({"weight": {"a": torch.ones(2, dtype=torch.float16)}, "config": HParams(model={"version": "v2"}, data={"x": 1})}, path)
        data = path.read_bytes()
        path.write_bytes(b"05" + data[2:])
        bad = tmp_path / "bad.pth"
        torch.save({"weight": {}, "config": types.SimpleNamespace(x=1)}, bad)
    finally:
        del sys.modules["utils"]
    got = load_checkpoint(path)
    assert got["config"] == {"model": {"version": "v2"}, "data": {"x": 1}}
    assert torch.equal(got["weight"]["a"], torch.ones(2, dtype=torch.float16))
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(bad)


def test_read_sovits_rejects_unknown_tag(tmp_path: Path) -> None:
    from mizuoto.runtime.engine import EngineError

    path = tmp_path / "x.pth"
    path.write_bytes(b"zz" + b"\0" * 100)
    with pytest.raises(EngineError):
        read_sovits(path)
