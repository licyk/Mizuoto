"""Qwen3-TTS's speaker encoder: ECAPA-TDNN as Qwen2.5-Omni ships it (upstream ``Qwen3TTSSpeakerEncoder``).

No BatchNorm; every convolution pads "same" by reflection. The input is a 128-bin log-mel at 24 kHz
(``speaker_mel``), and the output a 2048-wide embedding that takes the speaker slot of the talker's
prompt.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.dsp.mel import matcha_basis, matcha_mel
from mizuoto.runtime import ops as default_ops


def _conv(ops: Any, cin: int, cout: int, kernel: int, dilation: int = 1) -> nn.Module:
    return ops.Conv1d(cin, cout, kernel, dilation=dilation, padding="same", padding_mode="reflect")


class TimeDelayNetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv = _conv(ops, cin, cout, kernel, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, scale: int = 8, kernel: int = 3, dilation: int = 1, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TimeDelayNetBlock(cin // scale, cout // scale, kernel, dilation, ops) for _ in range(scale - 1)])
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        part_out: torch.Tensor | None = None
        for i, part in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                part_out = part
            elif i == 1:
                part_out = self.blocks[i - 1](part)
            else:
                assert part_out is not None
                part_out = self.blocks[i - 1](part + part_out)
            outputs.append(part_out)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, cin: int, se: int, cout: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv1 = _conv(ops, cin, se, 1)
        self.conv2 = _conv(ops, se, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=2, keepdim=True)
        m = torch.sigmoid(self.conv2(F.relu(self.conv1(m))))
        return x * m


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, scale: int, se: int, kernel: int, dilation: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(cin, cout, 1, 1, ops)
        self.res2net_block = Res2NetBlock(cout, cout, scale, kernel, dilation, ops)
        self.tdnn2 = TimeDelayNetBlock(cout, cout, 1, 1, ops)
        self.se_block = SqueezeExcitationBlock(cout, se, cout, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x)))) + x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1, ops)
        self.conv = _conv(ops, attention_channels, channels, 1)

    def _stats(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (m * x).sum(2)
        std = torch.sqrt((m * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device) * length
        mask = (torch.arange(length, device=x.device, dtype=lengths.dtype).expand(len(lengths), length) < lengths.unsqueeze(1)).to(dtype=x.dtype).unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._stats(x, mask / total)
        attention = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, length), std.unsqueeze(2).repeat(1, 1, length)], dim=1)
        attention = self.conv(torch.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean, std = self._stats(x, attention)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class SpeakerEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        mel_dim = config.get("mel_dim", 128)
        channels = config.get("enc_channels", [512, 512, 512, 512, 1536])
        kernels = config.get("enc_kernel_sizes", [5, 3, 3, 3, 1])
        dilations = config.get("enc_dilations", [1, 2, 3, 4, 1])
        scale = config.get("enc_res2net_scale", 8)
        se = config.get("enc_se_channels", 128)
        self.blocks = nn.ModuleList([TimeDelayNetBlock(mel_dim, channels[0], kernels[0], dilations[0], ops)])
        for i in range(1, len(channels) - 1):
            self.blocks.append(SqueezeExcitationRes2NetBlock(channels[i - 1], channels[i], scale, se, kernels[i], dilations[i], ops))
        self.mfa = TimeDelayNetBlock(channels[-1], channels[-1], kernels[-1], dilations[-1], ops)
        self.asp = AttentiveStatisticsPooling(channels[-1], config.get("enc_attention_channels", 128), ops)
        self.fc = _conv(ops, channels[-1] * 2, config.get("enc_dim", 2048), 1)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """(batch, frames, 128) log-mels → (batch, enc_dim)."""
        x = mels.transpose(1, 2)
        outs = []
        for layer in self.blocks:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        return self.fc(x).squeeze(-1)


SPEAKER_RATE = 24000


def speaker_mel(audio: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    """Upstream's ``mel_spectrogram`` for the speaker encoder: (1, n) audio at 24 kHz → (1, frames, 128)."""
    if basis is None:
        basis = matcha_basis(SPEAKER_RATE, 1024, 128, 0, 12000)
    return matcha_mel(audio, 1024, 128, SPEAKER_RATE, 256, 1024, 0, 12000, center=False, basis=basis).transpose(1, 2)
