"""``compute.fast``: set once for the process by the host, listed in outputs. Each
test runs in a child process: the switches are process-wide."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _child(code: str) -> dict:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_features_set_torchs_switches():
    pytest.importorskip("torch")
    got = _child(
        "import json, torch; from mizuoto.runtime.accel.features import apply; done = apply(['tf32', 'cudnn_benchmark', 'sdpa_reduced_precision']);"
        "print(json.dumps({'done': done, 'tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn': torch.backends.cudnn.allow_tf32, 'bench': torch.backends.cudnn.benchmark}))"
    )
    assert got == {"done": ["tf32", "cudnn_benchmark", "sdpa_reduced_precision"], "tf32": True, "cudnn": True, "bench": True}


def test_the_host_applies_them_when_its_runtime_starts_and_outputs_list_them(tmp_path):
    pytest.importorskip("torch")
    code = f"""
import json, pathlib, torch
from mizuoto.core.context import build_services
from mizuoto.core.tts.models import SpeakRequest
s = build_services(data_dir=pathlib.Path({str(tmp_path)!r}), environ={{}})
s.settings.update({{"compute": {{"fast": ["cudnn_benchmark"]}}}})
before = torch.backends.cudnn.benchmark
job = s.tts.speak(SpeakRequest(model="dummy/default", text="Hello there."), wait=True)
out = s.outputs.get(job.result["output_id"])
print(json.dumps({{"before": before, "after": torch.backends.cudnn.benchmark, "fast": out.params.get("fast")}}))
s.close()
"""
    assert _child(code) == {"before": False, "after": True, "fast": ["cudnn_benchmark"]}
