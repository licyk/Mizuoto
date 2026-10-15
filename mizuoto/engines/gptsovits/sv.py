"""The speaker-verification embedding v2Pro and v2ProPlus decode with (``sv.py``; ``eres2net/ERes2NetV2.py``,
``fusion.py``, ``pooling_layers.py``: 3D-Speaker, Apache-2.0).

16 kHz mono audio → Kaldi fbank (80 bins, dither 0; ``mizuoto.nn.dsp.kaldi`` is torchaudio's, which upstream's
``eres2net/kaldi.py`` equals but for a filterbank cache) → ERes2NetV2 (``baseWidth`` 24, ``scale`` 4, ``expansion`` 4)
→ ``forward3``: the fused stage-3/4 map (batch, 2048, 10, frames) flattened over channels and frequency and averaged
over time, (batch, 20480).

Parameter names are upstream's; the weights are ``pretrained_eres2netv2w24s4ep4.ckpt`` (the ``gptsovits/encoders``
asset's ``sv/``). ``seg_1`` (the 192-d embedding head) is built so the file loads strictly; ``forward3`` skips it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

SV_FILE = "sv/pretrained_eres2netv2w24s4ep4.ckpt"
"""The SV weights, relative to the encoders asset (upstream: ``GPT_SoVITS/pretrained_models/sv/…``)."""
SAMPLE_RATE = 16000


def relu20(x: torch.Tensor) -> torch.Tensor:
    """3D-Speaker's ``ReLU``: ``Hardtanh(0, 20)``."""
    return F.hardtanh(x, 0.0, 20.0)


class AFF(nn.Module):
    def __init__(self, channels: int = 64, r: int = 4, ops: Any = default_ops.default) -> None:
        super().__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            ops.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            ops.BatchNorm2d(inter_channels),
            nn.SiLU(),
            ops.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            ops.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, ds_y: torch.Tensor) -> torch.Tensor:
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        return torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)


class BasicBlockERes2NetV2(nn.Module):
    """The block of stages 1–2 (``fuse=False``: running sums between splits) and 3–4 (``fuse=True``: AFF)."""

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, baseWidth: int = 26, scale: int = 2, expansion: int = 2, fuse: bool = False, ops: Any = default_ops.default
    ) -> None:
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = ops.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = ops.BatchNorm2d(width * scale)
        self.nums = scale
        self.width = width
        self.convs = nn.ModuleList(ops.Conv2d(width, width, kernel_size=3, padding=1, bias=False) for _ in range(scale))
        self.bns = nn.ModuleList(ops.BatchNorm2d(width) for _ in range(scale))
        self.fuse = fuse
        if fuse:
            self.fuse_models = nn.ModuleList(AFF(channels=width, r=4, ops=ops) for _ in range(scale - 1))
        self.conv3 = ops.Conv2d(width * scale, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = ops.BatchNorm2d(planes * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(ops.Conv2d(in_planes, expansion * planes, kernel_size=1, stride=stride, bias=False), ops.BatchNorm2d(expansion * planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu20(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i in range(self.nums):
            if i > 0:
                sp = self.fuse_models[i - 1](sp, spx[i]) if self.fuse else sp + spx[i]
            sp = self.convs[i](sp)
            sp = relu20(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return relu20(out)


class ERes2NetV2(nn.Module):
    def __init__(
        self,
        num_blocks: tuple[int, int, int, int] = (3, 4, 6, 3),
        m_channels: int = 64,
        feat_dim: int = 80,
        embedding_size: int = 192,
        baseWidth: int = 26,
        scale: int = 2,
        expansion: int = 2,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.baseWidth = baseWidth
        self.scale = scale
        self.expansion = expansion
        stats_dim = int(feat_dim / 8) * m_channels * 8
        self.conv1 = ops.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ops.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], 1, False, ops)
        self.layer2 = self._make_layer(m_channels * 2, num_blocks[1], 2, False, ops)
        self.layer3 = self._make_layer(m_channels * 4, num_blocks[2], 2, True, ops)
        self.layer4 = self._make_layer(m_channels * 8, num_blocks[3], 2, True, ops)
        self.layer3_ds = ops.Conv2d(m_channels * 4 * expansion, m_channels * 8 * expansion, kernel_size=3, padding=1, stride=2, bias=False)
        self.fuse34 = AFF(channels=m_channels * 8 * expansion, r=4, ops=ops)
        self.seg_1 = ops.Linear(stats_dim * expansion * 2, embedding_size)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, fuse: bool, ops: Any) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlockERes2NetV2(self.in_planes, planes, s, baseWidth=self.baseWidth, scale=self.scale, expansion=self.expansion, fuse=fuse, ops=ops))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, T, F) => (B, 1, F, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        return self.fuse34(out4, out3_ds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The 192-d speaker embedding (TSTP pooling, ``seg_1``)."""
        out = self._trunk(x)
        mean = out.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(out, dim=-1) + 1e-8).flatten(start_dim=1)
        return self.seg_1(torch.cat((mean, std), 1))

    def forward3(self, x: torch.Tensor) -> torch.Tensor:
        """Fbank (batch, frames, 80) → (batch, 20480): what ``SV.compute_embedding3`` returns."""
        return self._trunk(x).flatten(start_dim=1, end_dim=2).mean(-1)


def build_sv(ops: Any = default_ops.default) -> ERes2NetV2:
    """``ERes2NetV2(baseWidth=24, scale=4, expansion=4)``, as ``sv.SV`` builds it."""
    return ERes2NetV2(baseWidth=24, scale=4, expansion=4, ops=ops)


def load_sv(path: str | Path, ops: Any = default_ops.default, **config: Any) -> ERes2NetV2:
    """The SV model from ``pretrained_eres2netv2w24s4ep4.ckpt`` (float16 in the file; float32 here), in eval mode.
    ``config`` overrides ``ERes2NetV2``'s arguments (tests build tiny ones)."""
    from mizuoto.runtime.loading import load_into, load_state_dict

    args: dict[str, Any] = {"baseWidth": 24, "scale": 4, "expansion": 4, **config}
    model = ERes2NetV2(**args, ops=ops)
    state = {k: v.float() if v.is_floating_point() else v for k, v in load_state_dict(path).items()}
    load_into(model, state, what="speaker-verification model")
    return model.eval()


def sv_features(wav: torch.Tensor) -> torch.Tensor:
    """16 kHz audio (batch, samples) → fbank (batch, frames, 80), each row as ``Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)``."""
    from mizuoto.nn.dsp.kaldi import fbank

    return torch.stack([fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=SAMPLE_RATE, dither=0) for w in wav])


@torch.no_grad()
def sv_embedding(model: ERes2NetV2, wav: torch.Tensor) -> torch.Tensor:
    """``SV.compute_embedding3``: 16 kHz audio (batch, samples), in the model's dtype → (batch, 20480)."""
    return model.forward3(sv_features(wav))
