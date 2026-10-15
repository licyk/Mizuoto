"""The device probe (``mizuoto.core.compute.probe``) and device choice on each kind of machine, with the command-line
tools and torch mocked: NVIDIA, AMD (rocm-smi, amd-smi), Intel (XPU through torch), Apple (MPS) and the CPU.

No such hardware is needed (none of it but the CPU is used here): each probe is fed the output its tool or torch
gives on that hardware.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mizuoto.core.compute import probe as probe_module
from mizuoto.core.compute.models import ComputeInfo, DeviceInfo
from mizuoto.core.compute.probe import (
    info_from_torch,
    installed_torch_build,
    parse_amd_smi,
    parse_rocm_smi,
    probe,
    torch_may_find_gpu,
)
from mizuoto.core.compute.service import ComputeService, runs_on
from mizuoto.core.compute.torch_index import choose_flavour

ALL = ["cuda", "rocm", "xpu", "mps", "cpu"]

ROCM_SMI = """WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

{"card0": {"Card Series": "Radeon RX 7900 XTX", "Card Model": "0x744c", "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
 "Card SKU": "D70701", "GFX Version": "gfx1100", "VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "1073741824"},
 "card1": {"Card series": "Instinct MI210", "VRAM Total Memory (B)": "68702699520", "VRAM Total Used Memory (B)": "0"},
 "system": {"Driver version": "6.8.5"}}
"""

AMD_SMI_STATIC = [
    {"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X", "vendor_id": "0x1002"}, "vram": {"type": "HBM", "size": {"value": 196592, "unit": "MB"}}},
    {"gpu": 1, "asic": {"market_name": "AMD Instinct MI300X"}, "vram": {"size": "196592 MB"}},
]
AMD_SMI_METRIC = {"gpu_data": [{"gpu": 0, "mem_usage": {"total_vram": {"value": 196592, "unit": "MB"}, "used_vram": {"value": 2048, "unit": "MB"}}}]}


def tools(monkeypatch, found: dict[str, str], outputs: dict[str, str | None], *, apple: bool = False, build: dict | None = None, child: dict | None = None) -> list[str]:
    """Mock the command-line tools ``found`` (name → path), what each prints (by the program's name), the platform,
    the installed torch's build and the child-process torch probe. Returns the programs run."""
    ran: list[str] = []

    def run(cmd: list[str], timeout: float = 10.0) -> str | None:
        ran.append(Path(cmd[0]).name)
        return outputs.get(Path(cmd[0]).name)

    def child_probe(python: str | None = None) -> dict | None:
        ran.append("torch-child")
        return child

    monkeypatch.setattr(probe_module.shutil, "which", lambda name: found.get(name))
    monkeypatch.setattr(probe_module, "find_nvidia_smi", lambda: found.get("nvidia-smi"))
    monkeypatch.setattr(probe_module, "_run", run)
    monkeypatch.setattr(probe_module, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(probe_module, "installed_torch_build", lambda: build)
    monkeypatch.setattr(probe_module, "probe_torch_subprocess", child_probe)
    monkeypatch.setattr(probe_module, "rocm_version", lambda: "6.4")
    monkeypatch.setattr(
        probe_module,
        "probe_apple",
        lambda: ComputeInfo(backend="MPS", devices=[DeviceInfo(id="mps", kind="mps", name="Apple M2"), probe_module.cpu_device()], probed_with="platform"),
    )
    return ran


def service(info: ComputeInfo, settings=None) -> ComputeService:
    settings = settings or SimpleNamespace(settings=SimpleNamespace(compute=SimpleNamespace(devices=[], default_device="auto")))
    return ComputeService(settings, SimpleNamespace(publish=lambda event: None), prober=lambda: info)  # ty: ignore[invalid-argument-type]


# -- the probes ----------------------------------------------------------------------------------


def test_nvidia_smi_comes_first(monkeypatch):
    smi = "0, NVIDIA GeForce RTX 3090, 24576, 1000, 580.65\n"
    ran = tools(monkeypatch, {"nvidia-smi": "/usr/bin/nvidia-smi", "rocm-smi": "/usr/bin/rocm-smi"}, {"nvidia-smi": smi + "| CUDA Version: 13.0 |"})
    info = probe()
    assert info.backend == "CUDA 13.0" and info.probed_with == "nvidia-smi" and "rocm-smi" not in ran
    assert [(d.id, d.kind) for d in info.devices] == [("cuda:0", "cuda"), ("cpu", "cpu")]


def test_rocm_smi_without_nvidia_smi(monkeypatch):
    ran = tools(monkeypatch, {"rocm-smi": "/opt/rocm/bin/rocm-smi"}, {"rocm-smi": ROCM_SMI})
    info = probe()
    assert info.backend == "ROCm 6.4" and info.probed_with == "rocm-smi" and info.driver == "6.8.5" and info.cuda_version is None
    gpus = [d for d in info.devices if d.kind != "cpu"]
    assert [(d.id, d.kind, d.name) for d in gpus] == [("cuda:0", "rocm", "Radeon RX 7900 XTX"), ("cuda:1", "rocm", "Instinct MI210")]
    assert gpus[0].memory_total_mb == 25753026560 / 2**20 and gpus[0].memory_used_mb == 1024.0
    assert "torch-child" not in ran
    assert choose_flavour(info) == "rocm7.2"


def test_amd_smi_when_rocm_smi_is_missing(monkeypatch):
    outputs = {"amd-smi": None}
    tools(monkeypatch, {"amd-smi": "/usr/bin/amd-smi"}, outputs)
    monkeypatch.setattr(
        probe_module, "_run", lambda cmd, timeout=10.0: json.dumps(AMD_SMI_STATIC if cmd[1] == "static" else AMD_SMI_METRIC) if Path(cmd[0]).name == "amd-smi" else None
    )
    info = probe()
    assert info.probed_with == "amd-smi" and info.backend == "ROCm 6.4"
    gpus = [d for d in info.devices if d.kind != "cpu"]
    assert [(d.id, d.name, d.memory_total_mb, d.memory_used_mb) for d in gpus] == [
        ("cuda:0", "AMD Instinct MI300X", 196592.0, 2048.0),
        ("cuda:1", "AMD Instinct MI300X", 196592.0, None),  # no usage reported: unknown, so the room check skips it
    ]


def test_tool_parsers_tolerate_other_layouts():
    assert parse_rocm_smi(None) == ([], None) and parse_rocm_smi({"system": {}}) == ([], None)
    devices, _ = parse_rocm_smi({"card0": {"Device Name": "AMD Radeon PRO W7900"}})
    assert devices[0].name == "AMD Radeon PRO W7900" and devices[0].memory_total_mb is None
    # amd-smi's older layout: a bare list, sizes as bare numbers (MB) or in GB
    old = parse_amd_smi([{"gpu": 0, "asic": {"market_name": "MI250"}, "vram": {"size": 65536}}], [{"gpu": 0, "mem_usage": {"total_vram": "64 GB", "used_vram": "1024 MB"}}])
    assert [(d.memory_total_mb, d.memory_used_mb) for d in old] == [(65536.0, 1024.0)]
    assert parse_amd_smi(None, None) == []


def test_torch_in_this_process_reports_xpu(monkeypatch):
    ran = tools(monkeypatch, {}, {}, build={"version": "2.9.0+xpu", "cuda": None, "hip": None, "xpu": "20250101"})
    report = {
        "version": "2.9.0+xpu",
        "cuda": None,
        "hip": None,
        "xpu": "20250101",
        "devices": [{"id": "xpu:0", "kind": "xpu", "name": "Intel(R) Arc(TM) B580 Graphics", "total_mb": 12216.0, "free_mb": 11000.0}],
    }
    info = probe(lambda: report)
    assert info.backend == "XPU" and info.probed_with == "torch" and "torch-child" not in ran
    assert [(d.id, d.kind, d.memory_total_mb, d.memory_used_mb) for d in info.devices] == [("xpu:0", "xpu", 12216.0, 1216.0), ("cpu", "cpu", None, None)]
    assert choose_flavour(info) == "xpu"


def test_torch_in_this_process_that_sees_no_gpu_means_the_cpu(monkeypatch):
    ran = tools(monkeypatch, {}, {}, apple=True, build={"version": "2.9.0", "cuda": None, "hip": None, "xpu": None})
    info = probe(lambda: {"version": "2.9.0", "devices": []})
    assert info.backend == "CPU" and info.probed_with == "torch" and ran == []


def test_torch_in_a_child_process_only_for_a_build_that_can_find_a_gpu(monkeypatch):
    xpu = {"version": "2.9.0+xpu", "cuda": None, "hip": None, "xpu": "20250101"}
    child = {"version": "2.9.0+xpu", "devices": [{"id": "xpu:0", "kind": "xpu", "name": "Intel(R) Data Center GPU Max 1100", "total_mb": 49152.0, "free_mb": None}]}
    ran = tools(monkeypatch, {}, {}, build=xpu, child=child)
    info = probe(lambda: None)  # torch is not loaded here yet
    assert ran == ["torch-child"] and info.backend == "XPU" and info.devices[0].memory_used_mb is None
    ran = tools(monkeypatch, {}, {}, build={"version": "2.9.0+cpu", "cuda": None, "hip": None, "xpu": None}, child=child)
    assert probe().backend == "CPU" and ran == []  # a CPU build never pays for a torch import
    ran = tools(monkeypatch, {}, {}, build=None, child=child)
    assert probe().probed_with == "none" and ran == []  # nor does a machine without torch


def test_apple_silicon_is_mps_without_torch(monkeypatch):
    ran = tools(monkeypatch, {}, {}, apple=True, build={"version": "2.9.0", "cuda": None, "hip": None, "xpu": None})
    info = probe()
    assert info.backend == "MPS" and [d.kind for d in info.devices] == ["mps", "cpu"] and ran == []
    assert choose_flavour(info) == "mps"


def test_when_torch_could_find_a_gpu(monkeypatch, tmp_path):
    assert torch_may_find_gpu(None) is False
    assert torch_may_find_gpu({"version": "2.9.0+cpu", "cuda": None, "hip": None, "xpu": None}) is False
    assert torch_may_find_gpu({"version": "2.9.0+xpu", "cuda": None, "hip": None, "xpu": None}, "win32") is True
    cuda = {"version": "2.9.0+cu128", "cuda": "12.8", "hip": None, "xpu": None}
    hip = {"version": "2.9.0+rocm6.4", "cuda": None, "hip": "6.4.43482", "xpu": None}
    assert torch_may_find_gpu(cuda, "win32") is False and torch_may_find_gpu(cuda, "darwin") is False  # nvidia-smi comes with the driver there
    assert torch_may_find_gpu(hip, "win32") is True
    present: set[str] = set()
    monkeypatch.setattr(probe_module.Path, "exists", lambda self: str(self) in present)
    assert torch_may_find_gpu(cuda, "linux") is False and torch_may_find_gpu(hip, "linux") is False
    present.update({"/dev/nvidia0", "/dev/kfd"})
    assert torch_may_find_gpu(cuda, "linux") is True and torch_may_find_gpu(hip, "linux") is True


def test_the_installed_torch_build_is_read_without_importing_torch(monkeypatch, tmp_path):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n", encoding="utf-8")
    (package / "version.py").write_text(
        "from typing import Optional\n__all__ = ['__version__']\n__version__ = '2.9.1+rocm6.4'\ndebug = False\ncuda: Optional[str] = None\nhip: Optional[str] = '6.4.43482-0f2d60242'\nxpu: Optional[str] = None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe_module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(package / "__init__.py")) if name == "torch" else None)
    assert installed_torch_build() == {"version": "2.9.1+rocm6.4", "cuda": None, "hip": "6.4.43482-0f2d60242", "xpu": None}
    monkeypatch.setattr(probe_module.importlib.util, "find_spec", lambda name: None)
    assert installed_torch_build() is None


def test_backend_names_from_torch():
    def info(kind: str, **build) -> ComputeInfo | None:
        return info_from_torch(
            {
                **build,
                "devices": [{"id": "mps" if kind == "mps" else f"{'xpu' if kind == 'xpu' else 'cuda'}:0", "kind": kind, "name": "GPU", "total_mb": 8192.0, "free_mb": 4096.0}],
            }
        )

    cuda = info("cuda", cuda="12.8")
    assert cuda is not None and cuda.backend == "CUDA 12.8" and cuda.cuda_version == "12.8" and cuda.devices[0].memory_used_mb == 4096.0
    rocm = info("rocm", hip="6.4.43482-0f2d60242")
    assert rocm is not None and rocm.backend == "ROCm 6.4" and rocm.cuda_version is None and rocm.devices[0].id == "cuda:0"
    assert (info("xpu") or cuda).backend == "XPU" and (info("mps") or cuda).backend == "MPS"
    assert info_from_torch(None) is None and info_from_torch({"devices": []}) is None


def test_compute_service_asks_torch_through_its_hook(monkeypatch):
    tools(monkeypatch, {}, {})
    settings = SimpleNamespace(settings=SimpleNamespace(compute=SimpleNamespace(devices=[], default_device="auto")))
    compute = ComputeService(settings, SimpleNamespace(publish=lambda event: None))  # ty: ignore[invalid-argument-type]
    assert compute.info().backend == "CPU"
    compute.torch_probe = lambda: {"devices": [{"id": "mps", "kind": "mps", "name": "Apple GPU", "total_mb": None, "free_mb": None}]}
    assert compute.info(refresh=True).backend == "MPS"


def test_services_wire_the_torch_hook_to_the_runtime(services):
    """Before the runtime is built (torch not loaded by Mizuoto), the hook says nothing: the probe asks a child process."""
    assert services.compute.torch_probe is not None and services.inference.started is False
    assert services.compute.torch_probe() is None


# -- choosing a device ---------------------------------------------------------------------------

CPU = DeviceInfo(id="cpu", kind="cpu", name="CPU")


@pytest.mark.parametrize(
    ("gpu", "backend"),
    [
        (DeviceInfo(id="cuda:0", kind="cuda", name="RTX 4090", memory_total_mb=24564, memory_used_mb=500), "CUDA 12.8"),
        (DeviceInfo(id="cuda:0", kind="rocm", name="Radeon RX 7900 XTX", memory_total_mb=24560, memory_used_mb=500), "ROCm 6.4"),
        (DeviceInfo(id="xpu:0", kind="xpu", name="Intel Arc B580", memory_total_mb=12216, memory_used_mb=None), "XPU"),
        (DeviceInfo(id="mps", kind="mps", name="Apple M2", memory_total_mb=16384), "MPS"),
    ],
    ids=["cuda", "rocm", "xpu", "mps"],
)
def test_auto_picks_the_gpu_of_every_kind(gpu, backend):
    compute = service(ComputeInfo(backend=backend, devices=[gpu, CPU], probed_with="test"))
    assert compute.resolve_device(ALL) == gpu.id
    assert compute.resolve_device(ALL, gpu.id) == gpu.id
    assert compute.resolve_device(["cpu"]) == "cpu"  # an engine without the GPU's kind runs on the CPU


def test_cuda_in_a_device_list_covers_rocm():
    """torch's ROCm build names AMD GPUs ``cuda:N``: an engine that runs on ``cuda`` runs there too."""
    amd = DeviceInfo(id="cuda:0", kind="rocm", name="Radeon")
    assert runs_on("rocm", ["cuda", "cpu"]) and runs_on("rocm", ["rocm"]) and not runs_on("cuda", ["rocm", "cpu"])
    assert not runs_on("xpu", ["cuda", "cpu"]) and not runs_on("mps", ["cuda", "rocm", "cpu"])
    compute = service(ComputeInfo(backend="ROCm", devices=[amd, CPU], probed_with="test"))
    assert compute.resolve_device(["cuda", "cpu"]) == "cuda:0" and compute.resolve_device(["cuda", "cpu"], "cuda:0") == "cuda:0"
    from mizuoto.core.errors import DeviceError

    mps = service(ComputeInfo(backend="MPS", devices=[DeviceInfo(id="mps", kind="mps", name="Apple"), CPU], probed_with="test"))
    with pytest.raises(DeviceError):
        mps.resolve_device(["cuda", "cpu"], "mps")


def test_out_of_memory_unloads_every_idle_model_on_every_gpu():
    """As ComfyUI's ``unload_all_models`` after an out-of-memory error: every GPU, not only the one that failed; models
    on the CPU and models in use stay."""
    from mizuoto.core.compute.service import MemoryHolder

    compute = service(ComputeInfo(backend="CUDA 12.8", devices=[CPU], probed_with="test"))
    released: list[str] = []
    for name, device, busy in (("a", "cuda:0", None), ("b", "cuda:1", None), ("c", "cpu", None), ("d", "cuda:1", "speaking"), ("e", None, None)):
        compute.add_holder(MemoryHolder(name, busy=lambda b=busy: b, cached=lambda: ["m"], release=lambda n=name: released.append(n), device=lambda d=device: d))
    compute.release_after_oom("job 1 (speak)", "cuda:0")
    assert sorted(released) == ["a", "b", "e"]


def test_probing_this_machine_loads_no_torch():
    """The real probe, in a fresh interpreter: whatever it finds, it never imports torch into the host."""
    import subprocess

    code = "import sys; from mizuoto.core.compute.probe import probe; info = probe(); print(info.backend); assert 'torch' not in sys.modules"
    root = str(Path(__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, cwd=root, check=False)
    assert out.returncode == 0, out.stderr
