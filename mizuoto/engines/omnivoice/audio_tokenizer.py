"""HiggsAudio v2's audio tokenizer, as transformers 5.3 computes it (``models/higgs_audio_v2_tokenizer``, Apache-2.0).

24 kHz audio ↔ 8 codebooks of 1024 codes at 25 frames a second. Encoding concatenates DAC's acoustic
encoder with a semantic branch (HuBERT at 16 kHz: the mean of every hidden state, every other frame,
then a small convolutional encoder), projects them (``fc``) and quantizes them residually with
Euclidean codebooks. Decoding sums the codebooks' vectors, projects them to the acoustic space (``fc2``)
and runs DAC's decoder (odd strides padded, no final tanh).

The semantic decoder and ``fc1`` are training-only and are not loaded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.codecs.dac import DacDecoder, DacEncoder
from mizuoto.nn.ssl.hubert import HubertConfig, HubertModel
from mizuoto.runtime import ops as default_ops

IGNORED = ["decoder_semantic.*", "fc1.*", "quantizer.quantizers.*.codebook.inited", "quantizer.quantizers.*.codebook.cluster_size", "quantizer.quantizers.*.codebook.embed_avg"]


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.conv1 = ops.Conv1d(channels, channels, kernel, stride=1, padding=((kernel - 1) // 2) * dilation, dilation=dilation, bias=False)
        self.conv2 = ops.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.elu(self.conv1(F.elu(x))))


class SemanticEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel: int, dilations: list[int], ops: Any) -> None:
        super().__init__()
        self.res_units = nn.ModuleList(ResidualUnit(in_channels, kernel, d, ops) for d in dilations)
        k = 3 if stride == 1 else 2 * stride
        self.conv = ops.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride, padding=(k - 1) // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.res_units:
            x = unit(x)
        return self.conv(x)


class SemanticEncoder(nn.Module):
    def __init__(self, c: Mapping[str, Any], hidden: int, ops: Any) -> None:
        super().__init__()
        k = c["kernel_size"]
        self.conv = ops.Conv1d(hidden, hidden, k, 1, k // 2, bias=False)
        blocks = []
        channels = hidden
        for stride, ratio in zip(c["strides"], c["channel_ratios"], strict=True):
            out = int(hidden * ratio)
            blocks.append(SemanticEncoderBlock(channels, out, stride, c["unit_kernel_size"], c["block_dilations"], ops))
            channels = out
        self.conv_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class Codebook(nn.Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.register_buffer("embed", torch.empty(size, dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape((-1, shape[-1]))
        embed = self.get_buffer("embed").t()
        dist = -(flat.pow(2).sum(1, keepdim=True) - 2 * flat @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices.view(*shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        embed = self.get_buffer("embed")
        return F.embedding(indices.to(embed.device), embed)


class VectorQuantizer(nn.Module):
    def __init__(self, hidden: int, size: int, dim: int, ops: Any) -> None:
        super().__init__()
        self.codebook = Codebook(size, dim)
        self.project_in = ops.Linear(hidden, dim)
        self.project_out = ops.Linear(dim, hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(self.project_in(x.permute(0, 2, 1)))

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.project_out(self.codebook.decode(indices)).permute(0, 2, 1)


class ResidualQuantizer(nn.Module):
    def __init__(self, n: int, hidden: int, size: int, dim: int, ops: Any) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(VectorQuantizer(hidden, size, dim, ops) for _ in range(n))

    def stages(self) -> list[VectorQuantizer]:
        return [q for q in self.quantizers if isinstance(q, VectorQuantizer)]


class HiggsAudioTokenizer(nn.Module):
    def __init__(self, config: Mapping[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        ac = config["acoustic_model_config"]
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.semantic_sample_rate = int(config.get("semantic_sample_rate", 16000))
        self.hop_length = int(math.prod(ac["downsampling_ratios"]))
        self.frame_rate = math.ceil(self.sample_rate / self.hop_length)
        self.pad = self.hop_length // 2
        semantic = HubertConfig.from_dict(config["semantic_model_config"])
        hidden = ac["hidden_size"] + semantic.hidden_size
        nbits = math.ceil(math.log2(config["codebook_size"]))
        self.bandwidths = list(config["target_bandwidths"])
        self.num_quantizers = int(1000 * self.bandwidths[-1] // (self.frame_rate * nbits))
        self.semantic_downsample = int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / config.get("downsample_factor", 320))
        self.acoustic_encoder = DacEncoder(ac["encoder_hidden_size"], ac["downsampling_ratios"], ac["hidden_size"], ops)
        self.acoustic_decoder = DacDecoder(ac["hidden_size"], ac["decoder_hidden_size"], ac["upsampling_ratios"], odd_output_padding=True, tanh=False, ops=ops)
        self.encoder_semantic = SemanticEncoder(config, semantic.hidden_size, ops)
        self.semantic_model = HubertModel(semantic, ops)
        self.fc = ops.Linear(hidden, hidden)
        self.fc2 = ops.Linear(hidden, ac["hidden_size"])
        self.quantizer = ResidualQuantizer(self.num_quantizers, hidden, config["codebook_size"], config["codebook_dim"], ops)

    def _semantic_features(self, wav: torch.Tensor, layerdrop: torch.Generator | None) -> torch.Tensor:
        from mizuoto.nn.dsp.resample import resample

        if self.sample_rate != self.semantic_sample_rate:
            wav = resample(wav, self.sample_rate, self.semantic_sample_rate)
        x = F.pad(wav[:, 0, :], (160, 160))
        with torch.no_grad():
            states = self.semantic_model(x, layerdrop)
        features = torch.stack([h.to(x.device) for h in states], dim=1).mean(dim=1)
        if self.semantic_downsample > 1:
            features = features[:, :: self.semantic_downsample, :]
        return features

    def encode(self, wav: torch.Tensor, layerdrop: torch.Generator | None = None) -> torch.Tensor:
        """(batch, 1, samples) at 24 kHz → (batch, codebooks, frames). ``layerdrop``: see ``HubertModel``."""
        if wav.shape[1] != 1:
            raise ValueError(f"Audio must be mono, but got {wav.shape[1]} channels")
        semantic = self.encoder_semantic(self._semantic_features(wav, layerdrop).detach().transpose(1, 2))
        if self.acoustic_encoder.output_length(wav.shape[2]) != semantic.shape[2]:
            acoustic = self.acoustic_encoder(F.pad(wav, (self.pad, self.pad)))
        else:
            acoustic = self.acoustic_encoder(wav)
        embeddings = torch.cat([acoustic.to(semantic.device), semantic], dim=1)
        embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        residual = embeddings
        codes = []
        for quantizer in self.quantizer.stages()[: self.num_quantizers]:
            indices = quantizer.encode(residual)
            residual = residual - quantizer.decode(indices)
            codes.append(indices)
        return torch.stack(codes).transpose(0, 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """(batch, codebooks, frames) → (batch, 1, samples)."""
        codes = codes.transpose(0, 1)
        quantized = torch.tensor(0.0, device=codes.device)
        for quantizer, indices in zip(self.quantizer.stages(), codes, strict=False):
            quantized = quantized + quantizer.decode(indices).to(codes.device)
        return self.acoustic_decoder(self.fc2(quantized.transpose(1, 2)).transpose(1, 2))
