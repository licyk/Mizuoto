"""``compute.fast``: torch's process-wide speed switches, after ComfyUI's ``--fast``.

The one place in native code allowed to set them (``tests/test_architecture.py``: ``PROCESS_FLAGS``): the host calls
``apply`` once, when its runtime is built (``InferenceService.runtime``), and a training stage before its trainer
loads. Every one of them changes results (lower-precision sums), so none is on by default; outputs list them
(``applied.fast``).
"""

from __future__ import annotations

import logging

log = logging.getLogger("mizuoto.runtime.accel")

FEATURES = ("tf32", "fp16_accumulation", "cudnn_benchmark", "sdpa_reduced_precision")


def deterministic() -> None:
    """Deterministic kernels for this process (``training.deterministic``): torch picks the deterministic version of
    every operation that has one and warns where none exists; cuDNN stops benchmarking (it would choose by timing).
    cuBLAS also needs ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA starts, which the host puts in the stage's environment."""
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    log.info("training.deterministic: deterministic kernels")


def apply(features: list[str]) -> list[str]:
    """Turn ``features`` on for this process; returns those this torch has."""
    import torch

    done: list[str] = []
    for feature in features:
        if feature == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif feature == "fp16_accumulation":
            if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                log.warning("compute.fast: fp16_accumulation needs torch 2.7 or later")
                continue
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        elif feature == "cudnn_benchmark":
            torch.backends.cudnn.benchmark = True
        elif feature == "sdpa_reduced_precision":
            if not hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp"):
                log.warning("compute.fast: sdpa_reduced_precision needs torch 2.5 or later")
                continue
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        else:
            log.warning("compute.fast: unknown feature %s", feature)
            continue
        done.append(feature)
    if done:
        log.info("compute.fast: %s", ", ".join(done))
    return done
