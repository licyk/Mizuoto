"""Devices beyond NVIDIA, simulated on the CPU: where float64 modules run, keeping
one on the CPU while its model moves, ``torch.compile`` off on MPS, and the torch device probe.

The ``meta`` device stands in for a GPU without float64 (MPS): moving to it changes a tensor's device without needing
the hardware.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mizuoto.runtime.device import DeviceService, float64_home, module_device, pin_to_cpu, probe_devices  # noqa: E402


def test_float64_runs_where_the_device_has_it():
    assert float64_home("mps", False) == "cpu"
    assert float64_home("xpu:0", False) == "cpu"  # Intel Arc: no fp64
    assert float64_home("xpu:1", True) == "xpu:1"  # a data-centre GPU
    assert float64_home("cuda:0", True) == "cuda:0"
    assert float64_home("cpu", False) == "cpu"
    devices = DeviceService()
    assert devices.float64_device("cpu") == "cpu"
    assert devices.float64_device("mps") == "cpu"  # MPS has no float64 (no torch.mps call needed to know)
    assert devices.supports("mps", "fp64") is False and devices.supports("cuda:0", "fp64") is True


def test_a_pinned_module_stays_on_the_cpu_in_float64():
    model = torch.nn.ModuleList([torch.nn.Linear(4, 4), pin_to_cpu(torch.nn.Linear(4, 1).double())])
    model.to("meta")  # as the model manager's load moves a handle to its device
    assert model[0].weight.device.type == "meta"
    assert module_device(model[1]).type == "cpu" and model[1].weight.dtype == torch.float64
    model.float()
    model.half()
    assert model[1].weight.dtype == torch.float64 and module_device(model[1]).type == "cpu"
    assert set(model.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias"}  # nothing renamed
    x = torch.randn(2, 4, dtype=torch.float64)
    assert torch.equal(model[1](x), x @ model[1].weight.T + model[1].bias)


def test_cosyvoice3_keeps_its_f0_predictor_on_the_cpu_where_float64_is_missing(tmp_path):
    """CosyVoice 3's F0 predictor computes in float64 (as upstream); loaded for a device without float64 it stays on
    the CPU while the rest of the model moves, and the vocoder feeds it there."""
    for module in ("tokenizers", "safetensors", "yaml", "inflect", "regex"):
        pytest.importorskip(module)
    from mizuoto.engines.cosyvoice import CosyVoiceEngine
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import LoadSpec
    from mizuoto.runtime.manager import ModelManager
    from mizuoto.runtime.runtime import Runtime
    from tests.engines_native.test_cosyvoice_train_tiny import _tiny

    (tmp_path / "v3").mkdir()
    model_dir = _tiny(tmp_path / "v3", 3)
    devices = DeviceService()
    asked: list[str] = []
    devices.float64_device = lambda name: asked.append(name) or "cpu"  # ty: ignore[invalid-assignment] - as on MPS
    loaded = CosyVoiceEngine().load(LoadSpec("cosyvoice/tiny", model_dir, "v3", "meta", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    hift = loaded.hift  # ty: ignore[unresolved-attribute]
    assert asked == ["meta"]
    for handle in loaded.handles:
        handle.load()
    assert hift.conv_post.weight.device.type == "meta"
    assert {(p.device.type, p.dtype) for p in hift.f0_predictor.parameters()} == {("cpu", torch.float64)}
    assert module_device(hift.f0_predictor).type == "cpu"

    # On the CPU (float64 there) nothing is pinned: the same code path as before.
    cpu = CosyVoiceEngine().load(LoadSpec("cosyvoice/tiny", model_dir, "v3", "cpu", options={}), Runtime(DeviceService(), ModelManager(DeviceService())), RequestContext("load"))
    mel = torch.randn(1, 80, 12)
    gen = torch.Generator().manual_seed(0)
    reference, _ = cpu.hift.inference(mel, gen, torch.Generator().manual_seed(1))  # ty: ignore[unresolved-attribute]
    pin_to_cpu(cpu.hift.f0_predictor)  # ty: ignore[unresolved-attribute]
    again, _ = cpu.hift.inference(mel, torch.Generator().manual_seed(0), torch.Generator().manual_seed(1))  # ty: ignore[unresolved-attribute]
    assert torch.equal(reference, again)


def test_compile_is_off_on_mps(caplog, monkeypatch):
    """Simulated: the CPU counted as an eager-only device, as MPS is."""
    from mizuoto.runtime import compile as compile_module

    monkeypatch.setattr(compile_module, "EAGER_DEVICES", frozenset({"cpu"}))
    module = torch.nn.Linear(3, 2)
    compiled = compile_module.Compiled(module, "tiny linear")
    called = []
    compiled._compiled = lambda *a, **k: called.append(1)
    x = torch.randn(2, 3)
    with caplog.at_level(logging.INFO, logger="mizuoto.runtime.compile"):
        assert torch.equal(compiled(x), module(x)) and torch.equal(compiled(x), module(x))
    assert called == [] and compiled.failure == "torch.compile is off on CPU"
    assert [r.levelno for r in caplog.records if r.name == "mizuoto.runtime.compile"] == [logging.INFO]


def test_probe_devices_reports_what_torch_sees():
    report = probe_devices(initialise=False)
    if report is None:  # a GPU torch has not initialised: this process does not create a context for it
        assert torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())
        return
    assert report["version"] == torch.__version__
    for d in report["devices"]:
        assert d["kind"] in ("cuda", "rocm", "xpu", "mps") and d["id"].split(":")[0] in ("cuda", "xpu", "mps")


def test_the_child_process_probe_prints_json():
    """``python -m mizuoto.runtime.device``, as the core's probe runs it (GPUs hidden: the result is the CPU's view)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    root = str(Path(__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-m", "mizuoto.runtime.device"], capture_output=True, text=True, timeout=180, env=env, cwd=root, check=True)
    report = json.loads(out.stdout)
    assert report["version"] == torch.__version__
    assert all(d["kind"] in ("mps", "xpu") for d in report["devices"])  # no CUDA or ROCm device while they are hidden


def test_the_core_probe_runs_the_child_from_anywhere(monkeypatch, tmp_path):
    """``probe_torch_subprocess`` as ``probe()`` calls it, from another working directory (GPUs hidden)."""
    from mizuoto.core.compute.probe import probe_torch_subprocess

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.chdir(tmp_path)
    report = probe_torch_subprocess()
    assert report is not None and report["version"] == torch.__version__
    assert all(d["kind"] in ("mps", "xpu") for d in report["devices"])
