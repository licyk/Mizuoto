"""GPT-SoVITS v4/v5's 48 kHz vocoder: ``module/models.py``'s HiFi-GAN ``Generator`` (MIT) as ``TTS.init_vocoder`` builds it,
which is ``mizuoto.nn.vocoders.hifigan.Generator`` (v2's decoder) without the speaker condition.

100 mel bins at 100 frames per second → 48 kHz (×480: upsample rates 10, 6, 2, 2, 2), ``ResBlock1`` with kernels
3/7/11 and dilations 1/3/5, a biased output convolution and ``tanh``. Parameter names are upstream's
(``conv_pre``, ``ups.N``, ``resblocks.N.convs1/2.M``, ``conv_post``).

The released ``vocoder.pth`` files hold plain weights (upstream removes weight norm before loading them); a
state dict with weight norm is folded at load (``mizuoto.runtime.loading.fold_weight_norm``), as
``remove_weight_norm`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mizuoto.nn.vocoders.hifigan import Generator
from mizuoto.runtime import ops as default_ops

SAMPLE_RATE = 48000
UPSAMPLE_RATE = 480

# TTS.init_vocoder for {"v4"} | V5_VERSIONS
V4_CONFIG: dict[str, Any] = {
    "initial_channel": 100,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 6, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [20, 12, 4, 4, 4],
    "gin_channels": 0,
    "is_bias": True,
}


def build_vocoder(ops: Any = default_ops.default) -> Generator:
    """The v4/v5 vocoder's architecture, without weights."""
    return Generator(**V4_CONFIG, ops=ops)


def load_vocoder(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> Generator:
    """``gsv-v4-pretrained/vocoder.pth`` or ``gsv-v5-pretrained/vocoder.pth``, in eval mode on ``device``.

    The weights are copied into float32 parameters first (as upstream's ``load_state_dict`` into its float32
    model), then cast to ``dtype``.
    """
    from mizuoto.runtime.loading import fold_weight_norm, load_into, load_state_dict, unwrap

    state = unwrap(load_state_dict(path), "weight", "generator", "state_dict")
    state = fold_weight_norm({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model = build_vocoder()
    load_into(model, state, what="GPT-SoVITS vocoder")
    return model.to(device=device, dtype=dtype).eval()
