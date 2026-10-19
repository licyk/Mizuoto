"""HiFT, CosyVoice's vocoder: HiFi-GAN with a neural source filter and an inverse STFT (Apache-2.0, Alibaba).

Mel frames → a waveform. An F0 predictor gives the pitch; a sine source at the harmonics of that pitch
(plus noise in unvoiced frames) is analysed by an STFT and fed into every upsampling stage; the last
stage predicts a magnitude and phase spectrum that an inverse STFT turns into audio.

Three flavours, as CosyVoice ships them:
- **v1** (22.05 kHz): ``SineGen`` with a random initial phase per harmonic.
- **v2** (24 kHz): ``SineGen2``, the phase integrated at the frame rate and interpolated.
- **v3** (``causal=True``): causal convolutions throughout, the F0 predictor in float64, and fixed noise:
  upstream draws its initial phases and its noise once, when the model is built, right after the flow's
  seeded noise. ``CausalNoise`` replays those draws on a private generator (``cosyvoice3_noise``).

Random numbers: v1 and v2 draw from the request's generators in upstream's order (the initial phases, the
noise, then the source's unused noise), so a seeded request repeats upstream's audio. Parameter names are
upstream's; weight norm is folded when loading (``fold``), in float64 for v3's F0 predictor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


def hann_window(n: int) -> torch.Tensor:
    """``scipy.signal.get_window("hann", n, fftbins=True)`` in float32 (computed in float64, as scipy does)."""
    fac = np.linspace(-np.pi, np.pi, n + 1)
    w = np.zeros(n + 1)
    for k, a in enumerate((0.5, 0.5)):
        w += a * np.cos(k * fac)
    return torch.from_numpy(w[:n].astype(np.float32))


def randn_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """``torch.randn_like(x)`` from ``generator``: same strides as ``x``, so the draws land where upstream's do."""
    return torch.empty_like(x).normal_(generator=generator)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


class CausalConv1d(nn.Conv1d):
    """cosyvoice.transformer.convolution's: zero context on the left (or right)."""

    reset_parameters = default_ops.Conv1d.reset_parameters

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = True, causal_type: str = "left") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, bias=bias)
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_type = causal_type

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        if cache is None or cache.size(2) == 0:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding).to(x)
        x = torch.concat([cache, x], dim=2) if self.causal_type == "left" else torch.concat([x, cache], dim=2)
        return super().forward(x)


class CausalConv1dDownSample(nn.Conv1d):
    reset_parameters = default_ops.Conv1d.reset_parameters

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0)
        self.causal_padding = stride - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class CausalConv1dUpsample(nn.Conv1d):
    reset_parameters = default_ops.Conv1d.reset_parameters

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, 1, padding=0)
        self.causal_padding = kernel_size - 1
        self.stride_up = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        x = F.interpolate(x, scale_factor=self.stride_up, mode="nearest")
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int], causal: bool, ops: Any) -> None:
        super().__init__()
        if causal:
            self.convs1 = nn.ModuleList(CausalConv1d(channels, channels, kernel_size, 1, dilation=d, causal_type="left") for d in dilations)
            self.convs2 = nn.ModuleList(CausalConv1d(channels, channels, kernel_size, 1, dilation=1, causal_type="left") for _ in dilations)
        else:
            self.convs1 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilations)
            self.convs2 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in dilations)
        self.activations1 = nn.ModuleList(Snake(channels) for _ in dilations)
        self.activations2 = nn.ModuleList(Snake(channels) for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations1, self.activations2, strict=True):
            x = c2(a2(c1(a1(x)))) + x
        return x


class F0Predictor(nn.Module):
    """``ConvRNNF0Predictor`` (``causal``: ``CausalConvRNNF0Predictor``): five convolutions with ELU, a linear head, ``abs``."""

    def __init__(self, in_channels: int = 80, cond_channels: int = 512, causal: bool = False, ops: Any = default_ops.default) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(5):
            cin = in_channels if i == 0 else cond_channels
            if causal:
                layers.append(CausalConv1d(cin, cond_channels, 4 if i == 0 else 3, causal_type="right" if i == 0 else "left"))
            else:
                layers.append(ops.Conv1d(cin, cond_channels, kernel_size=3, padding=1))
            layers.append(nn.ELU())
        self.condnet = nn.Sequential(*layers)
        self.classifier = ops.Linear(cond_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1))


class CausalNoise:
    """v3's fixed noise: initial phases (1, 9) and a long noise sequence, drawn as upstream draws them at build time.

    The sequence (upstream: 300 s at 24 kHz × 9 harmonics, 259 MB) is generated only as far as a request needs.
    """

    def __init__(self, generator: torch.Generator, harmonics: int = 9) -> None:
        self.harmonics = harmonics
        self.rand_ini = torch.rand(1, harmonics, generator=generator)
        self.rand_ini[:, 0] = 0
        self._gen = generator
        self._noise = torch.zeros(1, 0, harmonics)

    def noise(self, length: int) -> torch.Tensor:
        if length > 300 * 24000:
            raise ValueError("CosyVoice 3's vocoder noise covers 300 s")
        have = self._noise.shape[1]
        if length > have:
            more = max(length - have, 24000 * 10)
            more = min(more, 300 * 24000 - have)
            self._noise = torch.cat([self._noise, torch.rand(1, more, self.harmonics, generator=self._gen)], dim=1)
        return self._noise[:, :length]


class SourceModule(nn.Module):
    """``SourceModuleHnNSF`` with its sine generator (``SineGen`` for v1, ``SineGen2`` otherwise)."""

    def __init__(
        self, sampling_rate: int, upsample_scale: int, harmonic_num: int, sine_amp: float, noise_std: float, voiced_threshold: float, sinegen: str, causal: bool, ops: Any
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.sinegen = sinegen
        self.causal = causal
        self.causal_noise: CausalNoise | None = None
        self.l_linear = ops.Linear(harmonic_num + 1, 1)

    def _uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _sinegen1(self, f0: torch.Tensor, cpu: torch.Generator, gen: torch.Generator) -> torch.Tensor:
        f0 = f0.transpose(1, 2)
        f_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            f_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta = 2 * np.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        low, high = torch.tensor(-np.pi), torch.tensor(np.pi)
        phase = (low + torch.rand((f0.size(0), self.harmonic_num + 1, 1), generator=cpu, dtype=low.dtype) * (high - low)).to(f_mat.device)
        phase[:, 0, :] = 0
        sines = self.sine_amp * torch.sin(theta + phase)
        uv = self._uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * randn_like(sines, gen)
        return (sines * uv + noise).transpose(1, 2)

    def _sinegen2(self, f0: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        rad = (fn / self.sampling_rate) % 1
        if self.causal:
            assert self.causal_noise is not None
            rad[:, 0, :] = rad[:, 0, :] + self.causal_noise.rand_ini.to(rad.device)
        else:
            rand_ini = torch.rand(fn.shape[0], fn.shape[2], generator=gen, device=fn.device)
            rand_ini[:, 0] = 0
            rad[:, 0, :] = rad[:, 0, :] + rand_ini
        scale = np.int64(self.upsample_scale)
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=float(1 / scale), mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad, dim=1) * 2 * np.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=float(scale), mode="nearest" if self.causal else "linear").transpose(1, 2)
        sines = torch.sin(phase) * self.sine_amp
        uv = self._uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if self.causal:
            assert self.causal_noise is not None
            noise = noise_amp * self.causal_noise.noise(sines.shape[1]).to(sines.device)
        else:
            noise = noise_amp * randn_like(sines, gen)
        return sines * uv + noise

    def forward(self, x: torch.Tensor, cpu: torch.Generator, gen: torch.Generator) -> torch.Tensor:
        with torch.no_grad():
            sines = self._sinegen1(x, cpu, gen) if self.sinegen == "1" else self._sinegen2(x, gen)
        merged = torch.tanh(self.l_linear(sines))
        if not self.causal:
            uv = self._uv(x)
            randn_like(uv, gen)  # upstream's unused noise, drawn all the same
        return merged


class HiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        istft_params: Mapping[str, int] | None = None,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        source_resblock_kernel_sizes: list[int] | None = None,
        source_resblock_dilation_sizes: list[list[int]] | None = None,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        causal: bool = False,
        conv_pre_look_right: int = 4,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        upsample_rates = upsample_rates or [8, 8]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16]
        self.istft_params = dict(istft_params or {"n_fft": 16, "hop_len": 4})
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5]] * 3
        source_resblock_kernel_sizes = source_resblock_kernel_sizes or [7, 11]
        source_resblock_dilation_sizes = source_resblock_dilation_sizes or [[1, 3, 5]] * 2
        self.sampling_rate = sampling_rate
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.causal = causal
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates
        self.conv_pre_look_right = conv_pre_look_right
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        self.upsample_scale = int(np.prod(upsample_rates) * hop)
        self.m_source = SourceModule(
            sampling_rate, self.upsample_scale, nb_harmonics, nsf_alpha, nsf_sigma, nsf_voiced_threshold, "1" if sampling_rate == 22050 else "2", causal, ops
        )
        if causal:
            self.conv_pre: nn.Module = CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type="right")
            self.ups = nn.ModuleList(
                CausalConv1dUpsample(base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u)
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
            )
        else:
            self.conv_pre = ops.Conv1d(in_channels, base_channels, 7, 1, padding=3)
            self.ups = nn.ModuleList(
                ops.ConvTranspose1d(base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2)
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
            )
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        cum = np.cumprod([1] + upsample_rates[::-1][:-1])
        for i, (u, k, d) in enumerate(zip(cum[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes, strict=True)):
            u = int(u)
            out = base_channels // (2 ** (i + 1))
            if causal:
                self.source_downs.append(CausalConv1d(n_fft + 2, out, 1, 1, causal_type="left") if u == 1 else CausalConv1dDownSample(n_fft + 2, out, u * 2, u))
            else:
                self.source_downs.append(ops.Conv1d(n_fft + 2, out, 1, 1) if u == 1 else ops.Conv1d(n_fft + 2, out, u * 2, u, padding=(u // 2)))
            self.source_resblocks.append(ResBlock(out, k, d, causal, ops))
        self.resblocks = nn.ModuleList()
        ch = base_channels
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock(ch, k, d, causal, ops))
        self.conv_post = CausalConv1d(ch, n_fft + 2, 7, 1, causal_type="left") if causal else ops.Conv1d(ch, n_fft + 2, 7, 1, padding=3)
        self.register_buffer("stft_window", hann_window(n_fft), persistent=False)
        self.f0_predictor = F0Predictor(in_channels, base_channels, causal=causal, ops=ops)

    @staticmethod
    def ignorable() -> list[str]:
        return []

    def fold(self, state: Mapping[str, torch.Tensor], device: torch.device | str | None = None, f0_device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """The checkpoint with weight norm folded (``generator.`` prefixes dropped); v3's F0 predictor in float64.

        Upstream keeps HiFT's weight norm and computes it at every call where the model runs, so each norm is computed on
        ``device`` (the F0 predictor's on ``f0_device``, by default the same), as there (``fold_weight_norm``).
        """
        from mizuoto.runtime.loading import fold_weight_norm

        f0_device = device if f0_device is None else f0_device
        state = {k.replace("generator.", ""): v for k, v in state.items()}
        if not self.causal:
            return fold_weight_norm(state, device=device)
        f0 = {k: v for k, v in state.items() if k.startswith("f0_predictor.")}
        rest = fold_weight_norm({k: v for k, v in state.items() if k not in f0}, device=device)
        rest.update(fold_weight_norm({k: v.double() for k, v in f0.items()}, device=f0_device))
        return rest

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        spec = torch.view_as_real(torch.stft(x, n_fft, hop, n_fft, window=self.get_buffer("stft_window").to(x.device), return_complex=True))
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        n_fft, hop = self.istft_params["n_fft"], self.istft_params["hop_len"]
        magnitude = torch.clip(magnitude, max=1e2)
        real, img = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        return torch.istft(torch.complex(real, img), n_fft, hop, n_fft, window=self.get_buffer("stft_window").to(magnitude.device))

    def decode(self, x: torch.Tensor, s: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        real, imag = self._stft(s.squeeze(1))
        if self.causal and not finalize:
            x = self.conv_pre(x[:, :, : -self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right :])
            cut = int(np.prod(self.upsample_rates) * self.conv_pre_look_right)
            real, imag = real[:, :, :-cut], imag[:, :, :-cut]
        else:
            x = self.conv_pre(x)
        s_stft = torch.cat([real, imag], dim=1)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, self.lrelu_slope))
            if i == self.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + self.source_resblocks[i](self.source_downs[i](s_stft))
            xs: torch.Tensor | None = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            assert xs is not None
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        half = self.istft_params["n_fft"] // 2 + 1
        x = self._istft(torch.exp(x[:, :half, :]), torch.sin(x[:, half:, :]))
        if self.causal and not finalize:
            x = x[:, : -int(np.prod(self.upsample_rates) * self.istft_params["hop_len"])]
        return torch.clamp(x, -self.audio_limit, self.audio_limit)

    def inference(
        self, speech_feat: torch.Tensor, cpu: torch.Generator, gen: torch.Generator, finalize: bool = True, cache_source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, 80, frames) mel → (batch, samples) and the source (for streaming caches)."""
        if self.causal:
            first = self.f0_predictor.condnet[0]
            assert isinstance(first, CausalConv1d)
            # float64 where the F0 predictor is: the CPU when the device has no float64 (MPS; ``pin_to_cpu``)
            home = first.weight.device
            if finalize:
                f0 = self.f0_predictor(speech_feat.to(home).double()).to(speech_feat.dtype).to(speech_feat.device)
            else:
                x = speech_feat.to(home).double()
                h = first(x[:, :, : -first.causal_padding], x[:, :, -first.causal_padding :])
                for layer in list(self.f0_predictor.condnet)[1:]:
                    h = layer(h)
                f0 = torch.abs(self.f0_predictor.classifier(h.transpose(1, 2)).squeeze(-1)).to(speech_feat.dtype).to(speech_feat.device)
        else:
            f0 = self.f0_predictor(speech_feat)
        s = F.interpolate(f0[:, None], scale_factor=float(self.upsample_scale), mode="nearest").transpose(1, 2)
        s = self.m_source(s, cpu, gen).transpose(1, 2)
        if cache_source is not None and cache_source.shape[2] != 0:
            s[:, :, : cache_source.shape[2]] = cache_source
        if self.causal and not finalize:
            first = self.f0_predictor.condnet[0]
            assert isinstance(first, CausalConv1d)
            return self.decode(speech_feat[:, :, : -first.causal_padding], s, finalize=False), s
        return self.decode(speech_feat, s, finalize), s


def cosyvoice3_noise(
    vocab_size: int = 6561, input_size: int = 80, spk_embed_dim: int = 192, output_size: int = 80, f0_in: int = 80, f0_cond: int = 512, harmonics: int = 9
) -> CausalNoise:
    """CosyVoice 3's vocoder noise, replaying the draws upstream makes while building the model.

    ``cosyvoice3.yaml`` builds the flow, whose ``CausalConditionalCFM`` seeds the global generator with 0
    and draws its noise (1×80×15000 normals); then the flow's token embedding (``vocab_size`` × ``input_size``,
    normal) and speaker projection, the vocoder's causal F0 predictor (five convolutions and a linear head,
    PyTorch's default initialisation), and then the sine generator's initial phases and noise.
    """
    from torch.nn import init

    g = torch.Generator().manual_seed(0)
    torch.randn([1, 80, 50 * 300], generator=g)

    def linear_like(out_f: int, in_f: int, extra: tuple[int, ...] = ()) -> None:
        w = torch.empty(out_f, in_f, *extra)
        init.kaiming_uniform_(w, a=math.sqrt(5), generator=g)
        fan_in = in_f * (int(np.prod(extra)) if extra else 1)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(torch.empty(out_f), -bound, bound, generator=g)

    init.normal_(torch.empty(vocab_size, input_size), generator=g)
    linear_like(output_size, spk_embed_dim)
    linear_like(f0_cond, f0_in, (4,))
    for _ in range(4):
        linear_like(f0_cond, f0_cond, (3,))
    linear_like(1, f0_cond)
    return CausalNoise(g, harmonics)
