"""Device memory for native trainers.

A stage is its own process and owns its device while it runs; these tools let a trainer fit a smaller one:

- ``checkpoint_units(units, generators)``: gradient checkpointing over the blocks a trainer names (decoder layers, DiT
  blocks): each block keeps only its input during the forward and runs again in the backward. The recomputation draws
  exactly what the forward drew, from torch's global generators (``torch.utils.checkpoint`` restores them) *and from
  the trainer's own* (``generators``: LoRA dropout, a CFM's noise), whose states are put back for the recomputation and
  restored after, so the gradients equal those without checkpointing.
- ``preset(ctx)``: ``training.memory`` (``standard``: the upstream recipe exactly; ``low``: checkpointing and the other
  tools that keep results identical; ``lowest``: also those that change them), each overridden by its hyperparameter.
- ``peak_mb(device)`` and ``start_peak(device)``: the most the step allocated, reported as the metric ``memory/peak_mb``.
- ``estimate(...)``: what a run needs on the device (weights, gradients, optimizer state, activations), shown before
  it starts.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mizuoto.training.context import StageContext

MB = 2**20

PRESETS: dict[str, dict[str, Any]] = {
    "standard": {"gradient_checkpointing": "off"},
    "low": {"gradient_checkpointing": "blocks", "ema_device": "cpu", "frozen_dtype": "autocast"},
    "lowest": {"gradient_checkpointing": "blocks", "ema_device": "cpu", "frozen_dtype": "autocast", "optimizer_device": "cpu", "frozen_device": "cpu"},
}
"""``training.memory`` → the settings it makes when the run leaves them unset:

- ``gradient_checkpointing`` (a hyperparameter of the methods that have it): blocks run again in the backward;
- ``ema_device``: F5-TTS keeps its averaged copy of the model in RAM;
- ``frozen_dtype``: LoRA keeps the frozen weights of linear layers in the autocast dtype (identical products under
  autocast, half their memory; CUDA);
- ``optimizer_device`` (``lowest``): the optimizer's state in RAM, stepped on the CPU (the update's sums differ in
  their last bits from the GPU's); full fine-tunes without a gradient scaler;
- ``frozen_device`` (``lowest``): LoRA's frozen linear weights in RAM, streamed to the device for each product
  (``streamed_linear``)."""


def wants(ctx: StageContext, name: str, value: str) -> bool:
    """Whether ``name`` (a preset's setting) is ``value`` for this run. A setting that moves things off the device does
    nothing on a CPU unless the run asks for it itself (tests)."""
    if preset(ctx, name) != value:
        return False
    return ctx.device != "cpu" or name in ctx.hyperparams


def preset(ctx: StageContext, name: str, default: Any = None) -> Any:
    """Hyperparameter ``name``: the run's own value, else the memory preset's, else ``default``."""
    value = ctx.hyperparams.get(name)
    if value is not None and value not in ("", "auto"):
        return value
    chosen = PRESETS.get(str(ctx.request.get("memory") or "standard"), {})
    return chosen.get(name, default)


def checkpoint_every(value: Any) -> int:
    """``gradient_checkpointing`` → every how many blocks one is checkpointed (0: none): ``off``, ``blocks``, ``every:k``."""
    text = str(value or "off").strip().lower()
    if text in ("off", "false", "0", "none"):
        return 0
    if text in ("blocks", "on", "true", "1"):
        return 1
    if text.startswith("every:"):
        return max(1, int(text.split(":", 1)[1]))
    raise ValueError(f"gradient_checkpointing is off, blocks or every:k, not {value!r}")


class _Replay:
    """The states of some generators when a checkpointed block first ran, put back while it runs again."""

    def __init__(self, generators: Sequence[Any]) -> None:
        self.generators = list(generators)

    def snapshot(self) -> list[Any]:
        return [g.get_state() for g in self.generators]

    def restore(self, states: list[Any]) -> None:
        for g, s in zip(self.generators, states, strict=True):
            g.set_state(s)


def checkpoint_units(units: Iterable[Any], generators: Sequence[Any] = (), every: int = 1) -> int:
    """Wrap every ``every``-th of ``units`` (modules) in gradient checkpointing; returns how many. Their forward keeps
    only its inputs and runs again in the backward, with the global generators and ``generators`` as they were."""
    replay = _Replay(generators)
    wrapped = 0
    for index, module in enumerate(units):
        if every <= 0 or index % every:
            continue
        _wrap(module, replay)
        wrapped += 1
    return wrapped


def _wrap(module: Any, replay: _Replay) -> None:
    import torch
    from torch.utils.checkpoint import checkpoint

    original = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        first = replay.snapshot()
        calls = [0]

        def run(*inner: Any) -> Any:
            calls[0] += 1
            if calls[0] == 1:  # the forward
                return original(*inner, **kwargs)
            now = replay.snapshot()  # the recomputation in the backward: the forward's draws again
            replay.restore(first)
            try:
                return original(*inner, **kwargs)
            finally:
                replay.restore(now)

        return checkpoint(run, *args, use_reentrant=False, preserve_rng_state=True)

    module.forward = forward


def blocks(model: Any) -> list[Any]:
    """A model's repeated blocks: the children of its outermost ``ModuleList``s and ``Sequential``s that hold
    parameters (decoder layers, DiT blocks, conformer layers), in order."""
    from torch import nn

    found: list[Any] = []

    def visit(m: Any) -> None:
        for child in m.children():
            if isinstance(child, nn.ModuleList | nn.Sequential):
                found.extend(c for c in child.children() if any(True for _ in c.parameters()))
            else:
                visit(child)

    visit(model)
    return found


def frozen_in(model: Any, dtype: Any, kinds: tuple[type, ...] = ()) -> int:
    """Store the frozen weights and biases of ``model``'s linear layers (``nn.Linear`` and ``kinds``) in ``dtype``, the
    autocast dtype they run in: autocast casts them to it at every product anyway (round to nearest even, as here), so
    the products are the same, and the weights take half the memory, with no cast per step. Returns the bytes saved."""
    from torch import nn

    saved = 0
    for m in model.modules():
        if not isinstance(m, (nn.Linear, *kinds)):
            continue
        for name in ("weight", "bias"):
            p = getattr(m, name, None)
            if p is None or p.requires_grad or not p.is_floating_point() or p.dtype == dtype:
                continue
            before = p.numel() * p.element_size()
            p.data = p.data.to(dtype)
            saved += before - p.numel() * p.element_size()
    return saved


def start_peak(device: str) -> None:
    """Start measuring the most the device's allocator holds (CUDA, XPU; nothing elsewhere)."""
    import torch

    kind = device.split(":", 1)[0]
    backend = torch.cuda if kind == "cuda" else getattr(torch, "xpu", None) if kind == "xpu" else None
    if backend is not None and backend.is_available():
        backend.reset_peak_memory_stats(torch.device(device))


def peak_mb(device: str) -> float | None:
    """The most allocated on the device since ``start_peak``, in MB (None where it cannot tell)."""
    import torch

    kind = device.split(":", 1)[0]
    backend = torch.cuda if kind == "cuda" else getattr(torch, "xpu", None) if kind == "xpu" else None
    if backend is None or not backend.is_available():
        return None
    return round(backend.max_memory_allocated(torch.device(device)) / MB, 1)


@dataclass
class Estimate:
    """What a training run holds on its device at its peak, by part (bytes)."""

    weights: int
    gradients: int
    optimizer: int
    activations: int
    other: int = 0

    @property
    def total(self) -> int:
        return self.weights + self.gradients + self.optimizer + self.activations + self.other

    def as_mb(self) -> dict[str, float]:
        return {k: round(v / MB, 1) for k, v in {**vars(self), "total": self.total}.items()}


def estimate(
    params: int,
    trainable: int,
    *,
    weight_bytes: int = 4,
    master_bytes: int = 0,
    optimizer_bytes: int = 8,
    activation_bytes_per_token: int = 0,
    tokens: int = 0,
    checkpointing: int = 0,
    blocks: int = 1,
    other: int = 0,
) -> Estimate:
    """A run's peak on its device: every weight at ``weight_bytes``; the trainable ones' gradients (at their weights'
    size), fp32 master copies (``master_bytes``) and optimizer state (AdamW: 8 bytes); the activations of ``tokens``
    tokens a micro-batch, of which checkpointing every ``checkpointing``-th of ``blocks`` keeps about one block's worth
    plus the checkpointed inputs."""
    activations = activation_bytes_per_token * tokens
    if checkpointing and blocks > 1:
        per_block = activations // blocks
        kept = -(-blocks // checkpointing)  # checkpointed blocks keep their input (about an eighth of their activations)
        activations = (blocks - kept) * per_block + per_block + kept * per_block // 8
    return Estimate(
        weights=params * weight_bytes,
        gradients=trainable * weight_bytes,
        optimizer=trainable * (optimizer_bytes + master_bytes),
        activations=activations,
        other=other,
    )


Units = Callable[[Any], list[Any]]
"""A trainer's way to its model's checkpointable blocks."""


def host_optimizer(params: list[Any], make: Callable[[list[Any]], Any]) -> Any:
    """An optimizer whose state and float32 copies of the parameters live in RAM (``optimizer_device: cpu``, the
    ``lowest`` preset): ``make(host_params)`` builds it over the copies, and its ``step`` copies the gradients to the
    host, steps there and copies the parameters back. The device holds only the parameters and their gradients;
    AdamW's two moments (8 bytes a parameter) and the step's sums stay off it, so the update's last bits may differ from
    a GPU's. It is the optimizer ``make`` returns (schedulers take it); the clipping runs on the device's gradients
    before ``step``. Not with a gradient scaler (its unscaling reads the optimizer's own gradients).

    The copies are views of one float32 buffer and the gradients of another (one allocation each, not one per tensor).
    On a CUDA device a step moves the gradients and the parameters of each other dtype (bf16, fp16) through one
    page-locked buffer in that dtype, converting on the host; float32 parameters move straight between the device and
    the float32 buffers, which are then page-locked. The copies are queued without a wait each and waited for once.
    Every value is what a per-tensor ``.to`` gives. Most of a step is AdamW's own arithmetic on the CPU (Qwen3-TTS
    1.7B: about 4.5 of 5.5 s), which only a fused optimizer, with other results, would shorten."""
    import torch

    device_params = [p for p in params if p.requires_grad]
    pin = any(p.device.type == "cuda" for p in device_params) and torch.cuda.is_available()
    sizes = [p.numel() for p in device_params]
    # a float32 parameter's gradient lands in its gradient view and the parameter goes back from its copy (no staging):
    # the float32 buffers need page-locking only then
    other = {p.dtype for p in device_params if p.dtype != torch.float32}
    pin_flat = pin and any(p.dtype == torch.float32 for p in device_params)
    host_flat = torch.empty(sum(sizes), dtype=torch.float32, pin_memory=pin_flat)
    grad_flat = torch.empty(sum(sizes), dtype=torch.float32, pin_memory=pin_flat)
    staging = {dt: torch.empty(sum(n for p, n in zip(device_params, sizes, strict=True) if p.dtype == dt), dtype=dt, pin_memory=pin) for dt in other}
    host_params, grads, stages, offset = [], [], [], 0
    seen = dict.fromkeys(other, 0)
    for p, n in zip(device_params, sizes, strict=True):
        host = host_flat[offset : offset + n].view(p.shape)
        host.copy_(p.detach())
        host_params.append(torch.nn.Parameter(host, requires_grad=True))
        grads.append(grad_flat[offset : offset + n].view(p.shape))
        stages.append(staging[p.dtype][seen[p.dtype] : seen[p.dtype] + n].view(p.shape) if p.dtype in other else None)
        offset = offset + n
        if p.dtype in other:
            seen[p.dtype] += n
    optimizer = make(host_params)
    inner_step, inner_zero = optimizer.step, optimizer.zero_grad
    devices = {p.device for p in device_params if p.device.type == "cuda"}

    def wait() -> None:
        for device in devices:
            torch.cuda.current_stream(device).synchronize()

    def step(_self: Any, closure: Any = None) -> Any:
        with torch.no_grad():
            wait()  # the last step's parameters have left the host buffers
            for d, s, g in zip(device_params, stages, grads, strict=True):
                if d.grad is not None:
                    (g if s is None else s).copy_(d.grad, non_blocking=pin)
            wait()
            for d, h, s, g in zip(device_params, host_params, stages, grads, strict=True):
                h.grad = None if d.grad is None else g if s is None else g.copy_(s)
            out = inner_step(closure) if closure is not None else inner_step()
            for d, h, s in zip(device_params, host_params, stages, strict=True):
                d.copy_(h if s is None else s.copy_(h), non_blocking=pin)
        return out

    def zero_grad(_self: Any, set_to_none: bool = True) -> None:
        for p in device_params:
            p.grad = None
        inner_zero(set_to_none=set_to_none)

    # bound methods, as the schedulers expect to wrap (``optimizer.step.__func__``)
    optimizer.step = types.MethodType(step, optimizer)
    optimizer.zero_grad = types.MethodType(zero_grad, optimizer)
    return optimizer


def stream_frozen(layers: Iterable[Any], device: Any) -> int:
    """Move the frozen ``weight``/``bias`` of LoRA layers (with a ``streamed`` attribute: VoxCPM's ``LoRALinear``) to
    page-locked RAM and have them stream for each product (``streamed_linear``). Returns the bytes taken off the device."""
    import torch

    moved = 0
    pin = torch.device(device).type == "cuda" and torch.cuda.is_available()
    for layer in layers:
        for name in ("weight", "bias"):
            p = getattr(layer, name, None)
            if p is None or p.requires_grad:
                continue
            host = p.data.to("cpu", copy=True)
            moved += host.numel() * host.element_size() if p.device.type != "cpu" else 0
            p.data = host.pin_memory() if pin else host
        layer.streamed = torch.device(device)
    return moved


def streamed_linear(x: Any, weight: Any, bias: Any, device: Any) -> Any:
    """``F.linear(x, weight, bias)`` for a frozen weight kept in RAM: copied to ``device`` for the product and again
    for the backward's (``x``'s gradient needs it), never kept for the backward (ComfyUI's ``OffloadCheckpointFunction``,
    for one layer). The same products as the resident weight's."""
    return _StreamedLinear.apply(x, weight, bias, device)


def _streamed_linear_class() -> Any:
    import torch
    import torch.nn.functional as F

    class StreamedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, weight: Any, bias: Any, device: Any) -> Any:
            w = weight.to(device, non_blocking=True)
            b = None if bias is None else bias.to(device, non_blocking=True)
            ctx.weight, ctx.device = weight, device
            return F.linear(x, w, b)

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            (grad,) = grads
            w = ctx.weight.to(ctx.device, non_blocking=True)
            # what autograd computes for F.linear's input: grad @ weight, over the flattened leading dimensions
            return grad.matmul(w), None, None, None

    return StreamedLinear


class _Lazy:
    def __init__(self) -> None:
        self._cls: Any = None

    def apply(self, *args: Any) -> Any:
        if self._cls is None:
            self._cls = _streamed_linear_class()
        return self._cls.apply(*args)


_StreamedLinear = _Lazy()
