"""Qwen3-TTS's 12 Hz speech tokenizer (upstream ``tokenizer_12hz``): Mimi-style encoder, causal decoder.

- **Decoder** (every mode): RVQ lookup (1 semantic + 15 acoustic codebooks) → causal conv → an 8-layer
  transformer with a 72-frame sliding window → 2× (transposed conv + ConvNeXt) → 4 SnakeBeta blocks
  → 24 kHz. 1920 samples per frame. Upstream decodes in chunks of 300 frames with 25 of left context.
- **Encoder** (cloning with a transcript): transformers' ``MimiModel`` encode path. SEANet convolutions,
  an 8-layer transformer, a stride-2 downsample to 12.5 Hz, then Euclidean RVQ, keeping 16 of the
  32 codebooks.

Parameter names are those of ``speech_tokenizer/model.safetensors``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Masked, Site, attention
from mizuoto.runtime.lm.qwen import RMSNorm, Rotary, apply_rope, hf_sdpa, mask_for

MIMI = Site("qwen3tts.mimi")
CHUNK = 300
"""Upstream decodes the codes in chunks of this many frames…"""
LEFT_CONTEXT = 25
"""…each decoded with this many frames before it and their samples dropped."""

# -- shared ----------------------------------------------------------------------------------------


class CausalConvNet(nn.Module):
    """``Qwen3TTSTokenizerV2CausalConvNet``: left padding of kernel − stride, right padding to whole frames."""

    def __init__(self, cin: int, cout: int, kernel: int, dilation: int = 1, stride: int = 1, groups: int = 1, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv = ops.Conv1d(cin, cout, kernel, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel - 1) * dilation + 1
        self.padding = self.kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        x = F.pad(x, (self.padding, ideal - length), mode="constant", value=0)
        return self.conv(x).contiguous()


class CausalTransConvNet(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv = ops.ConvTranspose1d(cin, cout, kernel, stride=stride)
        self.right_pad = kernel - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.right_pad > 0:
            x = x[..., : x.shape[-1] - self.right_pad]
        return x.contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.dwconv = CausalConvNet(dim, dim, 7, groups=dim, ops=ops)
        self.norm = ops.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ops.Linear(dim, 4 * dim)
        self.pwconv2 = ops.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x).permute(0, 2, 1)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 2, 1)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(channels))
        self.beta = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


# -- decoder ---------------------------------------------------------------------------------------


class LayerScale(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class DecoderAttention(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.head_dim = c["head_dim"]
        self.n_rep = c["num_attention_heads"] // c["num_key_value_heads"]
        self.scale = self.head_dim**-0.5
        bias = c.get("attention_bias", False)
        self.q_proj = ops.Linear(c["hidden_size"], c["num_attention_heads"] * self.head_dim, bias=bias)
        self.k_proj = ops.Linear(c["hidden_size"], c["num_key_value_heads"] * self.head_dim, bias=bias)
        self.v_proj = ops.Linear(c["hidden_size"], c["num_key_value_heads"] * self.head_dim, bias=bias)
        self.o_proj = ops.Linear(c["num_attention_heads"] * self.head_dim, c["hidden_size"], bias=bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Masked | None) -> torch.Tensor:
        shape = (*x.shape[:-1], -1, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = hf_sdpa(q, k, v, mask, self.scale, self.n_rep)
        return self.o_proj(out.reshape(*x.shape[:-1], -1).contiguous())


class DecoderMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.gate_proj = ops.Linear(hidden, intermediate, bias=False)
        self.up_proj = ops.Linear(hidden, intermediate, bias=False)
        self.down_proj = ops.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderTransformerLayer(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(c, ops)
        self.mlp = DecoderMlp(c["hidden_size"], c["intermediate_size"], ops)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.self_attn_layer_scale = LayerScale(c["hidden_size"])
        self.mlp_layer_scale = LayerScale(c["hidden_size"])

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Masked | None) -> torch.Tensor:
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class DecoderTransformer(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.window = c.get("sliding_window")
        self.layers = nn.ModuleList([DecoderTransformerLayer(c, ops) for _ in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.rotary_emb = Rotary(c["head_dim"], float(c["rope_theta"]))
        self.input_proj = ops.Linear(c["latent_dim"], c["hidden_size"])
        self.output_proj = ops.Linear(c["hidden_size"], c["latent_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        length = x.shape[1]
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        mask = mask_for(None, length, 0, sliding_window=self.window)
        if mask is not None:
            mask = Masked(mask.dense.to(x.device).expand(x.shape[0], -1, -1, -1), mask.info)
        cos, sin = self.rotary_emb(x, positions)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.output_proj(self.norm(x))


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.empty(size))
        self.embedding_sum = nn.Parameter(torch.empty(size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self._codebook = EuclideanCodebook(dim, size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._codebook.decode(codes).transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, n_q: int, dim: int, size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dim, size) for _ in range(n_q)])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_q: int, dimension: int, input_dimension: int, output_dimension: int, bins: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.input_proj = ops.Conv1d(input_dimension, dimension, 1, bias=False)
        self.output_proj = ops.Conv1d(dimension, output_dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(n_q, dimension, bins)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.vq.decode(codes.transpose(0, 1)))


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, n_q: int, n_q_semantic: int, dimension: int, input_dimension: int, output_dimension: int, bins: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(n_q_semantic, dimension, input_dimension, output_dimension, bins, ops)
        self.rvq_rest = ResidualVectorQuantizer(n_q - n_q_semantic, dimension, input_dimension, output_dimension, bins, ops)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class DecoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConvNet(dim, dim, 7, dilation=dilation, ops=ops)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConvNet(dim, dim, 1, ops=ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act2(self.conv1(self.act1(x)))) + x


class DecoderBlock(nn.Module):
    def __init__(self, c: dict[str, Any], index: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        in_dim = c["decoder_dim"] // 2**index
        out_dim = c["decoder_dim"] // 2 ** (index + 1)
        rate = c["upsample_rates"][index]
        block: list[nn.Module] = [SnakeBeta(in_dim), CausalTransConvNet(in_dim, out_dim, 2 * rate, rate, ops)]
        block += [DecoderResidualUnit(out_dim, d, ops) for d in (1, 3, 9)]
        self.block = nn.ModuleList(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block:
            x = b(x)
        return x


class Decoder(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        self.num_quantizers = c["num_quantizers"]
        self.total_upsample = int(math.prod(c["upsample_rates"] + c["upsampling_ratios"]))
        self.pre_transformer = DecoderTransformer(c, ops)
        self.quantizer = SplitResidualVectorQuantizer(c["num_quantizers"], 1, c["codebook_dim"] // 2, c["codebook_dim"], c["codebook_dim"], c["codebook_size"], ops)
        self.pre_conv = CausalConvNet(c["codebook_dim"], c["latent_dim"], 3, ops=ops)
        self.upsample = nn.ModuleList(
            [nn.ModuleList([CausalTransConvNet(c["latent_dim"], c["latent_dim"], f, f, ops), ConvNeXtBlock(c["latent_dim"], ops)]) for f in c["upsampling_ratios"]]
        )
        decoder: list[nn.Module] = [CausalConvNet(c["latent_dim"], c["decoder_dim"], 7, ops=ops)]
        decoder += [DecoderBlock(c, i, ops) for i in range(len(c["upsample_rates"]))]
        out_dim = c["decoder_dim"] // 2 ** len(c["upsample_rates"])
        decoder += [SnakeBeta(out_dim), CausalConvNet(out_dim, 1, 7, ops=ops)]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """(batch, quantizers, frames) codes → (batch, 1, samples) audio in [-1, 1]."""
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(hidden).permute(0, 2, 1)
        for blocks in self.upsample:
            for b in blocks.children():
                hidden = b(hidden)
        wav = hidden
        for b in self.decoder:
            wav = b(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes: torch.Tensor, chunk_size: int = CHUNK, left_context_size: int = LEFT_CONTEXT, check: Any = None) -> torch.Tensor:
        wavs = []
        start = 0
        while start < codes.shape[-1]:
            if check is not None:
                check()
            end = min(start + chunk_size, codes.shape[-1])
            context = left_context_size if start - left_context_size > 0 else start
            wav = self(codes[..., start - context : end])
            wavs.append(wav[..., context * self.total_upsample :])
            start = end
        return torch.cat(wavs, dim=-1)


class DecodeStream:
    """``Decoder.chunked_decode`` as the frames arrive, for streamed audio.

    The decoder is causal throughout (causal convolutions, transposed convolutions trimmed on the right, a causal
    sliding-window transformer), so within one of upstream's chunks the samples of a frame depend only on the codes from
    the chunk's left context up to that frame. Decoding the chunk's frames so far, from the same start, therefore gives
    the samples ``chunked_decode`` gives for them, up to rounding (another length can take other kernels): each
    ``audio`` decodes the current chunk's prefix again and returns the samples not returned before. The redundant work
    is the decoder's: about ``CHUNK / (2 · every)`` decodes of a chunk instead of one.

    ``prefix`` is codes decoded with the rest but not returned (the reference's, when cloning in context; upstream cuts
    their share after the joint decode)."""

    def __init__(self, decoder: Decoder, prefix: torch.Tensor | None = None) -> None:
        self.decoder = decoder
        self.codes: list[torch.Tensor] = [] if prefix is None else list(prefix.unbind(0))
        self.done = len(self.codes)
        """Frames whose samples were returned (or are the prefix's)."""

    @property
    def frames(self) -> int:
        return len(self.codes)

    def push(self, frame: torch.Tensor) -> None:
        """One frame's (16,) codes."""
        self.codes.append(frame)

    def audio(self) -> torch.Tensor:
        """The (samples,) float32 audio of the frames pushed since the last call."""
        n = len(self.codes)
        out = []
        m = self.done
        while m < n:
            start = (m // CHUNK) * CHUNK
            end = min(start + CHUNK, n)
            context = LEFT_CONTEXT if start - LEFT_CONTEXT > 0 else start
            codes = torch.stack(self.codes[start - context : end], dim=1).unsqueeze(0).clamp(min=0)
            wav = self.decoder(codes)[0, 0]
            out.append(wav[(m - start + context) * self.decoder.total_upsample :])
            m = end
        self.done = n
        return torch.cat(out).to(torch.float32) if out else torch.zeros(0, dtype=torch.float32)


# -- encoder (transformers' MimiModel encode path) -------------------------------------------------


class MimiConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1, bias: bool = True, pad_mode: str = "constant", ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv = ops.Conv1d(cin, cout, kernel, stride, dilation=dilation, bias=bias)
        self.pad_mode = pad_mode
        self.stride = stride
        self.kernel_size = (kernel - 1) * dilation + 1
        self.padding_total = self.kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        # transformers computes this with int64 tensors; the result is the same integer
        n_frames = math.ceil((length - self.kernel_size + self.padding_total) / self.stride + 1) - 1
        ideal = n_frames * self.stride + self.kernel_size - self.padding_total
        extra = ideal - length
        mode = "constant" if self.pad_mode == "constant" else self.pad_mode
        x = F.pad(x, (self.padding_total, extra), mode)
        return self.conv(x)


class MimiResnetBlock(nn.Module):
    def __init__(self, dim: int, compress: int, kernel: int, dilations: list[int], ops: Any = default_ops.default) -> None:
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleList(
            [nn.ELU(), MimiConv1d(dim, hidden, kernel, dilation=dilations[0], ops=ops), nn.ELU(), MimiConv1d(hidden, dim, 1, dilation=dilations[1], ops=ops)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.block:
            h = layer(h)
        return x + h


class MimiEncoder(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        layers: list[nn.Module] = [MimiConv1d(c["audio_channels"], c["num_filters"], c["kernel_size"], ops=ops)]
        scaling = 1
        for ratio in reversed(c["upsampling_ratios"]):
            current = scaling * c["num_filters"]
            for j in range(c["num_residual_layers"]):
                layers.append(MimiResnetBlock(current, c["compress"], c["residual_kernel_size"], [c["dilation_growth_rate"] ** j, 1], ops))
            layers.append(nn.ELU())
            layers.append(MimiConv1d(current, current * 2, ratio * 2, stride=ratio, ops=ops))
            scaling *= 2
        layers.append(nn.ELU())
        layers.append(MimiConv1d(scaling * c["num_filters"], c["hidden_size"], c["last_kernel_size"], ops=ops))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MimiAttention(nn.Module):
    """``MimiSdpaAttention``: KV heads expanded, SDPA without a mask is causal over the whole clip."""

    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        bias = c.get("attention_bias", False)
        self.q_proj = ops.Linear(c["hidden_size"], self.heads * self.head_dim, bias=bias)
        self.k_proj = ops.Linear(c["hidden_size"], self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = ops.Linear(c["hidden_size"], self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = ops.Linear(self.heads * self.head_dim, c["hidden_size"], bias=bias)
        self.rotary_emb = Rotary(self.head_dim, float(c["rope_theta"]))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from mizuoto.runtime.lm.qwen import repeat_kv

        bsz, q_len, _ = x.size()
        q = self.q_proj(x).view(bsz, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, positions)
        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.heads // self.kv_heads), repeat_kv(v, self.heads // self.kv_heads)
        out = attention(q, k, v, site=MIMI, causal=q_len > 1)
        return self.o_proj(out.transpose(1, 2).contiguous().view(bsz, q_len, -1))


class MimiMLP(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.fc1 = ops.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.fc2 = ops.Linear(c["intermediate_size"], c["hidden_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MimiTransformerLayer(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.self_attn = MimiAttention(c, ops)
        self.mlp = MimiMLP(c, ops)
        self.input_layernorm = ops.LayerNorm(c["hidden_size"], eps=c["norm_eps"])
        self.post_attention_layernorm = ops.LayerNorm(c["hidden_size"], eps=c["norm_eps"])
        self.self_attn_layer_scale = LayerScale(c["hidden_size"])
        self.mlp_layer_scale = LayerScale(c["hidden_size"])

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), positions))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class MimiTransformer(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MimiTransformerLayer(c, ops) for _ in range(c["num_hidden_layers"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, positions)
        return x


class MimiCodebook(nn.Module):
    def __init__(self, dim: int, size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("initialized", torch.empty(1))
        self.register_buffer("cluster_usage", torch.empty(size))
        self.register_buffer("embed_sum", torch.empty(size, dim))

    @property
    def embed(self) -> torch.Tensor:
        return self.get_buffer("embed_sum") / self.get_buffer("cluster_usage").clamp(min=self.epsilon)[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape((-1, shape[-1]))
        dists = torch.cdist(flat[None].float(), self.embed[None].float(), p=2)[0]
        return dists.argmin(dim=-1).view(*shape[:-1])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.embed)


class MimiVQ(nn.Module):
    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self.codebook = MimiCodebook(dim, size)


class MimiRVQ(nn.Module):
    def __init__(self, n_q: int, hidden: int, dim: int, size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MimiVQ(dim, size) for _ in range(n_q)])
        self.input_proj = ops.Conv1d(hidden, dim, 1, bias=False)
        self.output_proj = ops.Conv1d(dim, hidden, 1, bias=False)

    def encode(self, embeddings: torch.Tensor, n_q: int) -> torch.Tensor:
        residual = self.input_proj(embeddings)
        indices = []
        for layer in self.layers[:n_q]:
            book = layer.get_submodule("codebook")
            assert isinstance(book, MimiCodebook)
            idx = book.encode(residual.permute(0, 2, 1))
            residual = residual - book.decode(idx).permute(0, 2, 1)
            indices.append(idx)
        return torch.stack(indices)


class MimiSplitRVQ(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        n_sem = c["num_semantic_quantizers"]
        dim, size, hidden = c["vector_quantization_hidden_dimension"], c["codebook_size"], c["hidden_size"]
        self.n_sem = n_sem
        self.semantic_residual_vector_quantizer = MimiRVQ(n_sem, hidden, dim, size, ops)
        self.acoustic_residual_vector_quantizer = MimiRVQ(c["num_quantizers"] - n_sem, hidden, dim, size, ops)

    def encode(self, embeddings: torch.Tensor, n_q: int) -> torch.Tensor:
        codes = self.semantic_residual_vector_quantizer.encode(embeddings, self.n_sem)
        if n_q > self.n_sem:
            codes = torch.cat([codes, self.acoustic_residual_vector_quantizer.encode(embeddings, n_q - self.n_sem)], dim=0)
        return codes


class Encoder(nn.Module):
    """``encoder.*`` of the speech tokenizer: the parts of ``MimiModel`` that encode."""

    def __init__(self, c: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.encoder = MimiEncoder(c, ops)
        self.encoder_transformer = MimiTransformer(c, ops)
        encodec_rate = c["sampling_rate"] / math.prod(c["upsampling_ratios"])
        self.downsample = MimiConv1d(c["hidden_size"], c["hidden_size"], 2 * int(encodec_rate / c["_frame_rate"]), stride=2, bias=False, pad_mode="replicate", ops=ops)
        self.quantizer = MimiSplitRVQ(c, ops)

    def encode(self, audio: torch.Tensor, n_q: int) -> torch.Tensor:
        """(batch, 1, samples) → (batch, n_q, frames) codes."""
        x = self.encoder(audio)
        x = self.encoder_transformer(x.transpose(1, 2)).transpose(1, 2)
        x = self.downsample(x)
        return self.quantizer.encode(x, n_q).transpose(0, 1)


class SpeechTokenizer(nn.Module):
    def __init__(self, config: dict[str, Any], ops: Any = default_ops.default, with_encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.valid_quantizers = config["encoder_valid_num_quantizers"]
        self.downsample_rate = config["encode_downsample_rate"]
        self.upsample_rate = config["decode_upsample_rate"]
        self.sample_rate = config["output_sample_rate"]
        self.encoder = Encoder(config["encoder_config"], ops) if with_encoder else None
        self.decoder = Decoder(config["decoder_config"], ops)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """One clip, (samples,) at 24 kHz → (frames, 16) codes, as upstream's ``encode`` with no padding."""
        assert self.encoder is not None
        # upstream encodes all 32 codebooks and keeps 16; the residual chain makes the first 16 the same
        codes = self.encoder.encode(audio[None, None, :], self.valid_quantizers)
        frames = -(-audio.shape[-1] // self.downsample_rate)
        return codes[0, :, :frames].transpose(0, 1)

    def decode(self, codes: list[torch.Tensor], check: Any = None) -> list[torch.Tensor]:
        """(frames, 16) code tensors → float32 audio tensors, batched with -1 padding as upstream decodes."""
        from torch.nn.utils.rnn import pad_sequence

        padded = pad_sequence(codes, batch_first=True, padding_value=-1)
        lengths = (padded[..., 0] > -1).sum(1) * self.upsample_rate
        padded = torch.clamp(padded, min=0)
        audio = self.decoder.chunked_decode(padded.transpose(1, 2), check=check).squeeze(1)
        return [a[:n].to(torch.float32) for a, n in zip(audio, lengths)]
