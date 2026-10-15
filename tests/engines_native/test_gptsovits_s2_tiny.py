"""GPT-SoVITS SoVITS training (``train_s2``: ``s2_train`` for v2/v2Pro, the LoRA ``s2_train_v3_lora`` for v4) on tiny
models and a fake prepared folder.

Parity with upstream (identical losses step for step and identical saved weights, seeded alike on the CPU) is checked
by the experiment's ``debug/train-sovits`` drivers on the real models; this covers the code paths in CI: the loader's
listing and buckets, a step or two of each loop, the saved files (half precision, version tag, config, LoRA keys) and
loading them back with the inference loaders.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from mizuoto.engines.gptsovits import s2_lora as L  # noqa: E402
from mizuoto.engines.gptsovits import s2_modules as M  # noqa: E402
from mizuoto.engines.gptsovits import sovits as S  # noqa: E402
from mizuoto.engines.gptsovits import sovits_v3 as SV  # noqa: E402
from mizuoto.engines.gptsovits import train_s2 as T  # noqa: E402
from mizuoto.engines.gptsovits.checkpoints import read_sovits  # noqa: E402
from mizuoto.nn.dsp.wav import write_wav  # noqa: E402

RATE = 32000
CLIPS = {"a.wav": 0.7, "b.wav": 0.9, "c.wav": 0.8}
PHONES = "AH0 B K AA1 T ."
TINY_MODEL = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 64,
    "n_heads": 2,
    "n_layers": 2,
    "kernel_size": 3,
    "resblock": "1",
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
    "upsample_rates": [10, 8, 8],
    "upsample_initial_channel": 32,
    "upsample_kernel_sizes": [20, 16, 16],
}
V3_ARCH: dict[str, Any] = {"bridge_channels": 32, "wn_layers": 2, "dit": {"dim": 32, "depth": 2, "heads": 2, "dim_head": 16, "ff_mult": 2, "text_dim": 32, "conv_layers": 1}}


_REAL_CONFIG = T.s2_config


def _tiny_config(version: str) -> dict[str, Any]:
    config = _REAL_CONFIG(version)
    config["model"].update(TINY_MODEL)
    config["model"]["gin_channels"] = {"v2": 512, "v2Pro": 64, "v4": 64}[version]
    config["train"]["segment_size"] = 640 * 4
    return config


class Ctx:
    device = "cpu"
    precision = "auto"
    name = "tiny"

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.outputs: list[str] = []

    def metric(self, epoch: int, step: int, values: dict[str, float], lr: float | None = None) -> None:
        self.metrics.append({"epoch": epoch, "step": step, "lr": lr, **values})

    def progress(self, value: float | None, message: str | None = None) -> None:
        pass

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append(f"{level}: {message}")

    def output(self, path: Path | str, label: str, kind: str = "file") -> None:
        self.outputs.append(str(path))


def _prepared(folder: Path, v2pro: bool) -> Path:
    g = torch.Generator().manual_seed(7)
    for sub in ("4-cnhubert", "5-wav32k", "7-sv_cn"):
        (folder / sub).mkdir(parents=True)
    lines = []
    for name, seconds in CLIPS.items():
        samples = (torch.rand(int(RATE * seconds), generator=g) - 0.5) * 0.4
        write_wav(folder / "5-wav32k" / name, samples.numpy(), RATE)
        frames = int(RATE * seconds) // 640
        # one feature frame short for a clip: the loader pads it, as upstream's does
        torch.save(torch.randn(1, 768, frames - (name == "b.wav"), generator=g), folder / "4-cnhubert" / f"{name}.pt")
        if v2pro:
            torch.save(torch.randn(1, 20480, generator=g), folder / "7-sv_cn" / f"{name}.pt")
        lines.append(f"{name}\t{PHONES}\tNone\ttext")
    lines.append("broken line")
    (folder / "2-name2text.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def _write(path: Path, state: dict[str, torch.Tensor], config: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"weight": {k: v.half() if v.is_floating_point() else v for k, v in state.items()}, "info": "base"}
    if config is not None:
        payload["config"] = config
    bio = io.BytesIO()
    torch.save(payload, bio)
    path.write_bytes(bio.getvalue())
    return path


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(T, "s2_config", _tiny_config)
    monkeypatch.setattr(T, "epoch_batches", lambda sampler, epoch: sampler.batches(epoch)[:2])
    monkeypatch.setattr(L, "V3_ARCH", V3_ARCH)
    return monkeypatch


def test_listing_and_buckets(tmp_path) -> None:
    folder = _prepared(tmp_path / "exp", v2pro=False)
    data = T.S2Data(folder, "v2", RATE, 2048, 640, 2048).scan()
    assert len(data.clips) == 99  # three clips, repeated 33 times
    assert {c.name for c in data.clips} == set(CLIPS)
    assert data.lengths[0] == (folder / "5-wav32k" / data.clips[0].name).stat().st_size // 1280
    sampler = T.BucketSampler(data.lengths, 2, T.BOUNDARIES_V2)
    assert len(sampler) == 50
    batches = sampler.batches(1)
    assert len(batches) == 50 and batches == sampler.batches(1) and batches != sampler.batches(2)
    seen = [i for b in batches for i in b]
    assert len(seen) == 100 and set(seen) == set(range(99))  # each bucket padded to whole batches by repeating items
    item = data.item(0)
    assert item[0].shape[-1] == item[1].shape[-1]  # the short feature padded to the spectrogram
    ssl, ssl_len, spec, spec_len, wav, wav_len, text, text_len = T.collate_v2([data.item(0), data.item(1)], v2pro=False)
    assert spec.shape[0] == 2 and spec_len[0] >= spec_len[1] and text.dtype == torch.long


@pytest.mark.parametrize("version", ["v2", "v2Pro"])
def test_train_v2_family(tmp_path, tiny, version) -> None:
    v2pro = version == "v2Pro"
    folder = _prepared(tmp_path / "exp", v2pro=v2pro)
    config = _tiny_config(version)
    config["model"]["version"] = version
    torch.manual_seed(0)
    base_state = M.build_synthesizer(config).state_dict()
    base = _write(tmp_path / "base" / "s2G.pth", base_state, config)
    assets: dict[str, str] = {}
    if v2pro:
        _write(tmp_path / "s2d" / version / "s2Dv2Pro.pth", M.build_discriminator(version).state_dict())
        assets[T.DISCRIMINATORS[version]] = str(tmp_path / "s2d")
    torch.manual_seed(1234)
    ctx = Ctx()
    result = T.train_sovits(cast(Any, ctx), folder, version, {"sovits_epochs": 1, "batch_size": 4}, base, assets)

    assert result["steps"] == 2 and result["epochs"] == 1
    assert [m["step"] for m in ctx.metrics] == [0, 1]
    assert set(ctx.metrics[0]) == {"epoch", "step", "lr", "loss_disc", "loss_gen", "loss_fm", "loss_mel", "kl_ssl", "loss_kl"}
    assert all(torch.isfinite(torch.tensor(m["loss_mel"])) for m in ctx.metrics)
    assert ctx.metrics[0]["kl_ssl"] == 0.0 and ctx.metrics[0]["lr"] == pytest.approx(1e-4 * 0.999875)
    assert any("discriminator" in line for line in ctx.logs) != v2pro  # v2 has no pretrained discriminator here

    path = Path(result["weights"])
    assert path.name == "tiny_e1_s2.pth" and path.parent == folder / "weights" and ctx.outputs == [str(path)]
    assert path.read_bytes()[:2] == (b"05" if v2pro else b"PK")
    ckpt = read_sovits(path)
    assert ckpt.model_version == version and ckpt.hps["train"]["batch_size"] == 2 and ckpt.hps["model"]["version"] == version
    assert all(v.dtype == torch.float16 for v in ckpt.weight.values() if v.is_floating_point())
    assert not any(k.startswith("enc_q.") for k in ckpt.weight)
    assert list(ckpt.weight) == [k for k in base_state if not k.startswith("enc_q.")]  # upstream's order
    # trained: the decoder moved; frozen: the SSL projection and the codebook did not
    assert not torch.equal(ckpt.weight["dec.conv_pre.weight"], base_state["dec.conv_pre.weight"].half())
    assert torch.equal(ckpt.weight["ssl_proj.weight"], base_state["ssl_proj.weight"].half())
    assert torch.equal(ckpt.weight["quantizer.vq.layers.0._codebook.embed"], base_state["quantizer.vq.layers.0._codebook.embed"].half())

    model, loaded = S.load_synthesizer(path)
    assert loaded.model_version == version
    codes = torch.randint(0, 1024, (1, 1, 10))
    refer = torch.rand(1, 1025, 30)
    sv = torch.randn(1, 20480) if v2pro else None
    audio = model.decode(codes, torch.randint(0, 732, (1, 6)), refer, sv_emb=sv, generator=torch.Generator().manual_seed(0))
    assert audio.shape == (1, 1, 20 * 640) and torch.isfinite(audio).all()


def test_train_v4_lora(tmp_path, tiny) -> None:
    folder = _prepared(tmp_path / "exp", v2pro=False)
    config = _tiny_config("v4")
    config["model"]["version"] = "v4"
    torch.manual_seed(0)
    model = L.build_synthesizer("v4", config)
    base_state = model.state_dict()
    base = _write(tmp_path / "base" / "s2Gv4.pth", base_state, config)
    torch.manual_seed(1234)
    ctx = Ctx()
    result = T.train_sovits(cast(Any, ctx), folder, "v4", {"sovits_epochs": 1, "batch_size": 2, "lora_rank": 4}, base, {})

    assert result["steps"] == 2 and result["lora_rank"] == 4
    assert [set(m) for m in ctx.metrics] == [{"epoch", "step", "lr", "cfm_loss"}] * 2
    path = Path(result["weights"])
    assert path.name == "tiny_e1_s2_l4.pth" and path.read_bytes()[:2] == b"04"
    ckpt = SV.checkpoint_weights(path)
    assert ckpt["lora_rank"] == 4 and ckpt["info"] == "1epoch_2iteration"
    weight = ckpt["weight"]
    lora = [k for k in weight if ".lora_" in k]
    assert len(lora) == 2 * 4 * 2  # A and B for q, k, v and out in each of the two blocks
    assert all(k.startswith(SV.PEFT_PREFIX) for k in lora)
    assert not any(k.startswith(("enc_p.", "ssl_proj.")) or ".base_layer." in k for k in weight)  # frozen: not saved
    assert any(k.startswith("wns1.") for k in weight) and "quantizer.vq.layers.0._codebook.embed" in weight
    b = weight[f"{SV.PEFT_PREFIX}estimator.transformer_blocks.0.attn.to_q.lora_B.default.weight"]
    assert b.abs().sum() > 0  # B starts at zero; two steps moved it

    arch = V3_ARCH
    orig = SV.SynthesizerTrnV3

    def tiny_model(version: str, **kw: Any) -> SV.SynthesizerTrnV3:
        return orig(version, **{**kw, **arch})

    tiny.setattr(SV, "SynthesizerTrnV3", tiny_model)
    merged = SV.load_synthesizer(path, "v4", lora_base=base)
    fea, ge = merged.decode_encp(torch.randint(0, 1024, (1, 1, 8)), torch.randint(0, 732, (1, 5)), torch.rand(1, 1025, 20))
    assert fea.shape[:2] == (1, 32) and torch.isfinite(fea).all()
