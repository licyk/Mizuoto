"""What the settings chose, made into a chain of backends, resolved per call with fallbacks.

``AccelPolicy(name)``: ``fast`` (the fastest backends the device offers, then SDPA) or a backend's name (it first,
then ``fast``'s chain); every chain ends with the site's reference. ``auto`` is no policy at all: nothing is bound and
every site computes its reference. A policy resolves each call once per ``(site, Call)``:

- a site that is not swappable keeps its reference; one with ``approx_ok=False`` refuses approximate (quantised) backends;
- each backend in the chain is skipped when it cannot run on the device (``Registry.unavailable``), lacks a backward
  pass the call needs, lacks dropout, or its constraints refuse the call; the first that takes it computes it;
- a backend that raises (anything but running out of memory or a cancel) is marked broken for that call and the next
  one computes it; out of memory propagates to the VRAM retry.

``bind(policy)`` makes it the context's for a request step or a stage; ``Run`` counts what each request used and why
it fell back (the output's ``applied.attention``). Fallbacks are logged once per site, backend and reason. With
``strict`` a named backend that does not take a call is an error (``compute_unavailable``) instead.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import Counter
from collections.abc import Generator
from typing import Any

import torch

from mizuoto.core import codes
from mizuoto.runtime.accel.dispatch import POLICY, TRACE_POLICY, MaskInfo, Site
from mizuoto.runtime.accel.registry import REGISTRY, Backend, Call, Registry, describe
from mizuoto.runtime.engine import EngineError

log = logging.getLogger("mizuoto.runtime.accel")

AUTO = "auto"
FAST = "fast"

HALF_CHAINS = {"rocm": ("flash", "xformers", "sdpa")}
"""``fast`` for fp16/bf16 calls, by device kind (NVIDIA: ``PADDED_CHAIN``); fp32 calls and other devices use SDPA
(its kernels are torch's best there)."""
PADDED_CHAIN = ("flash", "sdpa")
"""``fast`` on NVIDIA GPUs for fp16/bf16 calls whose keys are padded (a cache preallocated to its longest, a padded
batch: ``MaskInfo`` with ``keys`` padded on one side), which FlashAttention skips; every other call keeps SDPA.
Measured on RTX 3090s (2026-10-18, every engine): FlashAttention made VoxCPM's decoding 1.39–1.65× faster, and was
slower than SDPA for calls with nothing to skip (single-query steps 0.13–0.5×, IndexTTS 1.5 0.82× overall); xFormers
was slower where it differed (GPT-SoVITS 0.62×, VoxCPM 0.76×); SageAttention made autoregressive models run on
without end (Qwen3-TTS 112× the audio) and sped nothing else up. Each stays available by name."""

_warned: set[tuple[str, str, str]] = set()
_warned_lock = threading.Lock()


class AccelError(EngineError):
    """A named backend could not compute a call under ``compute.attention_strict``."""

    code = codes.COMPUTE_UNAVAILABLE


class Resolution:
    """How one ``(site, Call)`` is computed: ``backend`` (None: the reference) and the backends skipped before it."""

    __slots__ = ("backend", "skipped")

    def __init__(self, backend: Backend | None, skipped: list[tuple[str, str]]) -> None:
        self.backend = backend
        self.skipped = skipped


class AccelPolicy:
    def __init__(self, name: str, *, strict: bool = False, registry: Registry | None = None) -> None:
        from mizuoto.runtime.accel import backends

        backends.register_all()
        self.name = name
        self.strict = strict
        self.registry = registry or REGISTRY
        self._cache: dict[tuple[str, Call], Resolution] = {}
        self._lock = threading.Lock()
        self.trace_run = Run(self)
        """The run compiled modules trace with (``tracing``): one object, so their graphs are not traced again for each
        request; what it counts is not reported."""

    def __repr__(self) -> str:
        return f"AccelPolicy({self.name!r}, strict={self.strict})"

    def chain(self, call: Call) -> tuple[str, ...]:
        half = call.dtype in (torch.float16, torch.bfloat16)
        if half and call.device == "cuda":
            padded = {"prefix", "suffix"} & set(call.structured.split("+"))
            fast: tuple[str, ...] = PADDED_CHAIN if padded else ("sdpa",)
        else:
            fast = HALF_CHAINS.get(call.device, ("sdpa",)) if half else ("sdpa",)
        if self.name == FAST:
            return fast
        return (self.name, *(n for n in fast if n != self.name))

    def resolve(self, site: Site, call: Call, tensors: Any, exclude: frozenset[str] = frozenset()) -> Resolution:
        skipped: list[tuple[str, str]] = []
        device = tensors[0].device
        for name in self.chain(call):
            if name in exclude:
                skipped.append((name, "failed"))
                continue
            backend = self.registry.get(name)
            reason = "not installed" if backend is None else self.registry.unavailable(name, device)
            if backend is not None and reason is None:
                reason = self._refuses(backend, site, call, tensors)
            if reason is None:
                return Resolution(backend, skipped)
            skipped.append((name, reason))
        return Resolution(None, skipped)

    def _refuses(self, backend: Backend, site: Site, call: Call, tensors: Any) -> str | None:
        if call.grad and not backend.backward:
            return "no backward"
        if call.dropout and not backend.dropout:
            return "dropout"
        if not site.approx_ok and backend.approximate:
            return "the site refuses quantised attention"
        reason = backend.constraints.check(call, self.registry.capability(tensors[0].device))
        if reason is None and backend.accepts is not None:
            reason = backend.accepts(call, tensors)
        return reason

    def resolution(self, site: Site, call: Call, tensors: Any) -> Resolution:
        key = (site.name, call)
        found = self._cache.get(key)
        if found is None:
            found = self.resolve(site, call, tensors)
            with self._lock:
                self._cache[key] = found
            self._warn(site, found)
        if self.strict:
            self._refuse_fallback(site, found)
        return found

    def broken(self, site: Site, call: Call, tensors: Any, failed: Resolution, error: BaseException) -> Resolution:
        """``failed.backend`` raised on this call: never use it for it again, take the next."""
        assert failed.backend is not None
        name = failed.backend.name
        _warn_once(site.name, name, f"failed: {type(error).__name__}: {error}")
        exclude = frozenset({name, *(n for n, reason in failed.skipped if reason == "failed")})
        found = self.resolve(site, call, tensors, exclude)
        with self._lock:
            self._cache[(site.name, call)] = found
        return found

    def _warn(self, site: Site, found: Resolution) -> None:
        for name, reason in found.skipped:
            if reason != "not installed" or name == self.name:
                _warn_once(site.name, name, reason)

    def _refuse_fallback(self, site: Site, found: Resolution) -> None:
        """Strict mode: a named backend that does not compute a swappable site's call is an error, every time."""
        if site.swappable and self.name != FAST and (found.backend is None or found.backend.name != self.name):
            reason = dict(found.skipped).get(self.name, "not chosen")
            raise AccelError(
                f"Attention backend {self.name} cannot compute {site.name}: {reason}",
                {"backend": self.name, "site": site.name, "reason": reason, "hint": "install the backend, choose another one, or turn compute.attention_strict off"},
            )

    def run(self) -> Run:
        return Run(self)


class Run:
    """A policy as one request or stage uses it, with what it used and why it fell back."""

    transparent = False
    """Computes every site's reference (tests that only watch the calls): sites keep their reference's inputs."""

    def __init__(self, policy: AccelPolicy) -> None:
        self.policy = policy
        self.used: Counter[str] = Counter()
        self.fallbacks: dict[str, Counter[str]] = {}

    def attend(
        self,
        site: Site,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        causal: bool,
        scale: float | None,
        enable_gqa: bool,
        dropout_p: float,
        mask_info: MaskInfo | None,
    ) -> torch.Tensor:
        if not site.swappable:
            self.used["reference"] += 1
            return site.reference(q, k, v, mask, causal, scale, enable_gqa, dropout_p)
        call = describe(q, k, v, mask, causal, scale, dropout_p, mask_info)
        tensors = (q, k, v, mask)
        found = self.policy.resolution(site, call, tensors)
        while True:
            self._count(found)
            backend = found.backend
            if backend is None:
                return site.reference(q, k, v, mask, causal, scale, enable_gqa, dropout_p)
            try:
                return backend.fn(q, k, v, mask, causal, scale, enable_gqa, dropout_p, mask_info)
            except Exception as e:
                from mizuoto.runtime.device import is_oom

                if is_oom(e):
                    raise
                if self.policy.strict and backend.name == self.policy.name:
                    raise AccelError(
                        f"Attention backend {backend.name} failed on {site.name}: {type(e).__name__}: {e}",
                        {"backend": backend.name, "site": site.name, "reason": f"{type(e).__name__}: {e}"},
                    ) from e
                self.used[backend.name] -= 1
                found = self.policy.broken(site, call, tensors, found, e)

    def _count(self, found: Resolution) -> None:
        chosen = found.backend.name if found.backend is not None else "reference"
        self.used[chosen] += 1
        for name, reason in found.skipped:
            if reason == "not installed" and name != self.policy.name:
                continue
            self.fallbacks.setdefault(f"{name}→{chosen}", Counter())[reason] += 1

    def report(self) -> dict[str, Any]:
        return {
            "policy": self.policy.name,
            "used": {k: n for k, n in self.used.items() if n},
            "fallbacks": {k: dict(c) for k, c in self.fallbacks.items()},
        }

    def warnings(self) -> list[str]:
        out = []
        for key, reasons in self.fallbacks.items():
            out.append(f"attention {key}: " + ", ".join(f"{reason} ({n} calls)" for reason, n in reasons.items()))
        return out


def _warn_once(site: str, backend: str, reason: str) -> None:
    key = (site, backend, reason)
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    log.warning("Attention backend %s is not used for %s: %s", backend, site, reason)


_policies: dict[tuple[str, bool], AccelPolicy] = {}
_policies_lock = threading.Lock()


def policy_for(name: str | None, strict: bool = False) -> AccelPolicy | None:
    """The process's policy for a setting's value (None for ``auto``: every site's reference); one per value, so
    resolutions are shared by the models that use it."""
    if not name or name == AUTO:
        return None
    with _policies_lock:
        found = _policies.get((name, strict))
        if found is None:
            found = _policies[(name, strict)] = AccelPolicy(name, strict=strict)
        return found


def current() -> Run | None:
    return POLICY.get()


@contextlib.contextmanager
def bind(run: Run | None) -> Generator[Run | None, None, None]:
    """Make ``run`` the policy of this context (None: every site's reference)."""
    token = POLICY.set(run)
    try:
        yield run
    finally:
        POLICY.reset(token)


_trace_lock = threading.Lock()


@contextlib.contextmanager
def tracing(run: Run | None) -> Generator[None, None, None]:
    """For a ``Compiled`` module: torch.compile cannot read the context variable, so its graph sees this policy (one
    compiled call at a time while a policy other than ``auto`` is in use)."""
    if run is None:
        yield
        return
    with _trace_lock:
        TRACE_POLICY[0] = run
        try:
            yield
        finally:
            TRACE_POLICY[0] = None
