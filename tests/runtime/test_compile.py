"""``compute.compile``: off by default, read at each load, and an eager fallback."""

from __future__ import annotations

import logging

import pytest

from mizuoto.core.inference.service import InferenceService
from mizuoto.core.settings import SettingsService
from mizuoto.core.settings.models import ComputeSettings
from mizuoto.runtime.runtime import RuntimeOptions


def test_compile_is_off_by_default():
    assert ComputeSettings().compile is False
    assert RuntimeOptions().compile is False


def test_the_setting_reaches_the_runtime_for_the_next_load(tmp_path):
    settings = SettingsService(data_dir=tmp_path, environ={})
    inference = InferenceService(settings)
    assert inference.runtime.options.compile is False
    settings.update({"compute": {"compile": True}})
    assert inference.runtime.options.compile is True
    settings.update({"compute": {"compile": False}})
    assert inference.runtime.options.compile is False


def _broken(error: Exception):
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        raise error

    return run, calls


def test_a_failed_compile_runs_eagerly_with_one_warning(caplog):
    torch = pytest.importorskip("torch")
    from mizuoto.runtime.compile import Compiled, compile_module

    module = torch.nn.Linear(3, 2)
    assert compile_module(module, False, "off") is None
    compiled = compile_module(module, True, "tiny linear")
    assert isinstance(compiled, Compiled)
    compiled._compiled, calls = _broken(RuntimeError("InductorError: no working C++ compiler\ndetails"))
    x = torch.randn(4, 3)
    with caplog.at_level(logging.WARNING, logger="mizuoto.runtime.compile"):
        outs = [compiled(x), compiled(x)]
    assert all(torch.equal(out, module(x)) for out in outs)
    assert len(calls) == 1 and compiled.calls == 0
    assert compiled.failure == "RuntimeError: InductorError: no working C++ compiler"
    assert [r.levelno for r in caplog.records if r.name == "mizuoto.runtime.compile"] == [logging.WARNING]
    # the module tree is unchanged: the wrapper is no submodule
    assert list(dict(module.named_modules())) == [""]


def test_running_out_of_memory_is_not_a_compile_failure():
    torch = pytest.importorskip("torch")
    from mizuoto.runtime.compile import Compiled

    compiled = Compiled(torch.nn.Linear(3, 2), "tiny linear")
    compiled._compiled, _ = _broken(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    with pytest.raises(torch.OutOfMemoryError):
        compiled(torch.randn(1, 3))
    assert compiled.failure is None


def test_the_chunk_mask_traces_without_a_graph_break():
    """CosyVoice's U-Net estimator builds its attention mask with wenet's ``add_optional_chunk_mask``."""
    torch = pytest.importorskip("torch")
    from mizuoto.nn.transformer.wenet import add_optional_chunk_mask

    traced = torch.compile(add_optional_chunk_mask, backend="eager", fullgraph=True)
    xs = torch.zeros(2, 6, 4)
    masks = torch.tensor([[[True] * 4 + [False] * 2], [[False] * 6]])
    for chunk in (0, 2):
        assert torch.equal(traced(xs, masks.clone(), chunk), add_optional_chunk_mask(xs, masks.clone(), chunk))
