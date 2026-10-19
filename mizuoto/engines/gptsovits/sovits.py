"""GPT-SoVITS' SoVITS decoder for v1/v2/v2Pro/v2ProPlus (``module/models.py``: ``SynthesizerTrn``; MIT), inference only.

- ``extract_latent``: HuBERT features (768 × frames at 50 Hz) → ``ssl_proj`` (stride 2: 25 Hz) → the residual
  quantizer's codes, the prompt's semantic tokens.
- ``decode`` / ``decode_streaming``: semantic tokens + phone ids + reference spectrogram(s) → waveform. The
  reference's global embedding ``ge`` comes from the mel style encoder over the spectrogram's first 704 bins
  (v1: all of them); v2Pro adds the speaker-verification embedding (``sv_emb``: 20480 → gin) and a PReLU, and its
  text encoder takes ``ge_to512(ge)``. Several references are averaged. The text encoder (SSL encoder, text
  encoder, MRTE cross-attention, a second encoder) gives the prior; ``z_p = m_p + ε·exp(logs_p)·noise_scale``;
  the coupling flow runs in reverse; HiFi-GAN decodes.
- ``spectrogram`` / ``reference_spec``: ``mel_processing.spectrogram_torch`` as ``TTS._get_ref_spec`` calls it.

The only random draw is ε, ``torch.randn_like(m_p)`` upstream: here ``empty_like(m_p).normal_(generator=…)``, the
same op on the same layout, so a generator seeded with *s* gives upstream's audio under ``torch.manual_seed(s)``.

Not ported here: the posterior encoder ``enc_q`` (training; its weights are ignored at load), the training ``forward``
and ``infer``. The v4/v5 models (``SynthesizerTrnV3``: CFM and a vocoder) are in ``sovits_v3``, which shares the
quantizer, the text encoder and ``IGNORABLE``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.gptsovits.checkpoints import V2_FAMILY, V2PRO, SovitsCheckpoint, read_sovits
from mizuoto.nn.flow.coupling import ResidualCouplingBlock
from mizuoto.nn.speaker.melstyle import MelStyleEncoder
from mizuoto.nn.transformer.vits import Encoder, MultiHeadAttention, sequence_mask
from mizuoto.nn.vocoders.hifigan import Generator
from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.engine import EngineError

SSL_DIM = 768
N_SYMBOLS = {"v1": 322, "v2": 732}
"""``len(text.symbols.symbols)`` and ``len(text.symbols2.symbols)``."""

IGNORABLE = ("enc_q.*", "quantizer.vq.layers.*._codebook.inited", "quantizer.vq.layers.*._codebook.cluster_size", "quantizer.vq.layers.*._codebook.embed_avg")
"""Checkpoint keys inference does not use: the posterior encoder (training only; fine-tunes leave it out) and the codebooks' EMA state."""


# -- the residual vector quantizer (module/quantize.py, module/core_vq.py; inference paths) -------------------------


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.register_buffer("embed", torch.empty(codebook_size, dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # einops "... d -> (...) d"
        embed = self.get_buffer("embed").t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_ind, self.get_buffer("embed"))


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self._codebook = EuclideanCodebook(dim, codebook_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x.permute(0, 2, 1))  # "b d n -> b n d"

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self._codebook.decode(embed_ind).permute(0, 2, 1)  # "b n d -> b d n"


class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, num_quantizers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VectorQuantization(dim, codebook_size) for _ in range(num_quantizers))


class ResidualVectorQuantizer(nn.Module):
    """``quantizer.vq.layers.<i>._codebook.embed``. (batch, dim, frames) ⇄ (n_q, batch, frames) codes."""

    def __init__(self, dimension: int = 256, n_q: int = 8, bins: int = 1024) -> None:
        super().__init__()
        self.n_q = n_q
        self.vq = ResidualVectorQuantization(dimension, bins, n_q)

    def encode(self, x: torch.Tensor, n_q: int | None = None, st: int = 0) -> torch.Tensor:
        n_q = n_q or self.n_q
        residual = x
        all_indices = []
        layers: list[Any] = list(self.vq.layers)
        for layer in layers[st:n_q]:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            all_indices.append(indices)
        return torch.stack(all_indices)

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        layers: list[Any] = list(self.vq.layers)
        quantized_out = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            quantized_out = quantized_out + layers[st + i].decode(indices)
        return quantized_out


# -- the text encoder (models.py: TextEncoder; mrte_model.py: MRTE) ------------------------------------------------


class MRTE(nn.Module):
    def __init__(self, content_enc_channels: int = 192, hidden_size: int = 512, out_channels: int = 192, n_heads: int = 4, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads, window_size=None, ops=ops)
        self.c_pre = ops.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = ops.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = ops.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc: torch.Tensor, ssl_mask: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor, ge: torch.Tensor | None) -> torch.Tensor:
        ge_: torch.Tensor | int = 0 if ge is None else ge
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask, attn_mask) + ssl_enc + ge_
        return self.c_post(x * ssl_mask)


class TextEncoder(nn.Module):
    """``models.TextEncoder``: SSL features + phone ids [+ ``ge``] → features, and for ``SynthesizerTrn`` the prior.

    ``SynthesizerTrnV3`` (v4, v5) takes the features only (``features``): its ``proj`` is loaded but never run.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        n_symbols: int,
        mrte_channels: int = 512,
        ops: Any = default_ops.default,
    ) -> None:
        """``mrte_channels`` exists for tests: upstream builds ``MRTE()``, 192 → 512 → 192, the sizes of its models."""
        super().__init__()
        self.out_channels = out_channels
        self.ssl_proj = ops.Conv1d(SSL_DIM, hidden_channels, 1)
        self.encoder_ssl = Encoder(hidden_channels, filter_channels, n_heads, n_layers // 2, kernel_size, ops=ops)
        self.encoder_text = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, ops=ops)
        self.text_embedding = ops.Embedding(n_symbols, hidden_channels)
        self.mrte = MRTE(hidden_channels, mrte_channels, hidden_channels, ops=ops)
        self.encoder2 = Encoder(hidden_channels, filter_channels, n_heads, n_layers // 2, kernel_size, ops=ops)
        self.proj = ops.Conv1d(hidden_channels, out_channels * 2, 1)

    def encode(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ge: torch.Tensor | None,
        result_length: int | None = None,
        overlap_frames: torch.Tensor | None = None,
        padding_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The features and their mask before the speed change."""
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)
        text_mask = torch.unsqueeze(sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
        text_h = self.text_embedding(text).transpose(1, 2)
        text_h = self.encoder_text(text_h * text_mask, text_mask)
        y = self.mrte(y, y_mask, text_h, text_mask, ge)
        if padding_length is not None and padding_length != 0:
            y = y[:, :, :-padding_length]
            y_mask = y_mask[:, :, :-padding_length]
        y = self.encoder2(y * y_mask, y_mask)
        if result_length is not None:
            y = y[:, :, -result_length:]
            y_mask = y_mask[:, :, -result_length:]
        if overlap_frames is not None:
            overlap_len = overlap_frames.shape[-1]
            # upstream caches this per length in a module-level dict (models.WINDOW); built per call here
            window = torch.sin(torch.arange(overlap_len * 2, device=y.device) * torch.pi / (overlap_len * 2))
            y[:, :, :overlap_len] = window[:overlap_len].view(1, 1, -1) * y[:, :, :overlap_len] + window[overlap_len:].view(1, 1, -1) * overlap_frames
        return y, y_mask

    def features(self, y: torch.Tensor, y_lengths: torch.Tensor, text: torch.Tensor, text_lengths: torch.Tensor, ge: torch.Tensor, speed: float = 1) -> torch.Tensor:
        """``SynthesizerTrnV3``'s use (upstream's first return value; the prior's ``m``/``logs`` are not needed there)."""
        y, _ = self.encode(y, y_lengths, text, text_lengths, ge)
        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
        return y

    def forward(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ge: torch.Tensor | None,
        speed: float = 1,
        result_length: int | None = None,
        overlap_frames: torch.Tensor | None = None,
        padding_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, y_mask = self.encode(y, y_lengths, text, text_lengths, ge, result_length, overlap_frames, padding_length)
        y_, y_mask_ = y, y_mask
        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, y_, y_mask_


# -- the synthesizer ----------------------------------------------------------------------------------------------


class SynthesizerTrn(nn.Module):
    """``SynthesizerTrn`` for inference. Build it with ``from_hps`` (a checkpoint's ``config``) or ``load_synthesizer``."""

    def __init__(
        self,
        spec_channels: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        resblock: str,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        gin_channels: int = 0,
        semantic_frame_rate: str = "25hz",
        version: str = "v2",
        n_symbols: int | None = None,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        if version not in V2_FAMILY:
            raise EngineError(f"SoVITS {version} is not a SynthesizerTrn model", {"version": version})
        if semantic_frame_rate not in ("25hz", "50hz"):
            raise EngineError(f"Unknown semantic frame rate {semantic_frame_rate!r}", {"semantic_frame_rate": semantic_frame_rate})
        self.version = version
        self.inter_channels = inter_channels
        self.gin_channels = gin_channels
        self.upsample_rates = list(upsample_rates)
        self.semantic_frame_rate = semantic_frame_rate
        symbols = N_SYMBOLS["v1" if version == "v1" else "v2"] if n_symbols is None else n_symbols
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, symbols, ops=ops)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            ops=ops,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels, ops=ops)
        self.ref_enc = MelStyleEncoder(spec_channels if version == "v1" else 704, style_vector_dim=gin_channels, ops=ops)
        if semantic_frame_rate == "25hz":
            self.ssl_proj = ops.Conv1d(SSL_DIM, SSL_DIM, 2, stride=2)
        else:
            self.ssl_proj = ops.Conv1d(SSL_DIM, SSL_DIM, 1, stride=1)
        self.quantizer = ResidualVectorQuantizer(dimension=SSL_DIM, n_q=1, bins=1024)
        self.is_v2pro = version in V2PRO
        if self.is_v2pro:
            self.sv_emb = ops.Linear(20480, gin_channels)
            self.ge_to512 = ops.Linear(gin_channels, 512)
            self.prelu = nn.PReLU(num_parameters=gin_channels)

    @classmethod
    def from_hps(cls, hps: dict[str, Any], n_symbols: int | None = None, ops: Any = default_ops.default) -> SynthesizerTrn:
        """As ``TTS.init_vits_weights``: ``SynthesizerTrn(filter_length // 2 + 1, …, **hps["model"])``."""
        m = hps["model"]
        return cls(
            int(hps["data"]["filter_length"]) // 2 + 1,
            int(m["inter_channels"]),
            int(m["hidden_channels"]),
            int(m["filter_channels"]),
            int(m["n_heads"]),
            int(m["n_layers"]),
            int(m["kernel_size"]),
            str(m["resblock"]),
            [int(k) for k in m["resblock_kernel_sizes"]],
            [[int(d) for d in ds] for ds in m["resblock_dilation_sizes"]],
            [int(u) for u in m["upsample_rates"]],
            int(m["upsample_initial_channel"]),
            [int(k) for k in m["upsample_kernel_sizes"]],
            gin_channels=int(m.get("gin_channels", 0)),
            semantic_frame_rate=str(m.get("semantic_frame_rate", "25hz")),
            version=str(m.get("version", "v2")),
            n_symbols=n_symbols,
            ops=ops,
        )

    @property
    def hop_samples(self) -> int:
        """Output samples per semantic token (``prod(upsample_rates)`` × 2 at 25 Hz)."""
        rate = 1
        for u in self.upsample_rates:
            rate *= u
        return rate * (2 if self.semantic_frame_rate == "25hz" else 1)

    # -- prompt codes --

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """HuBERT's last hidden state, (batch, 768, frames) → codes (batch, n_q, frames / 2). ``codes[0, 0]`` is the prompt."""
        ssl = self.ssl_proj(x)
        return self.quantizer.encode(ssl).transpose(0, 1)

    # -- the reference --

    def reference_embedding(self, refer: torch.Tensor, sv_emb: torch.Tensor | None = None) -> torch.Tensor:
        """``decode``'s ``get_ge``: one spectrogram (1, freq, frames) [+ its SV embedding (1, 20480) for v2Pro] → (1, gin, 1)."""
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        if self.version == "v1":
            ge = self.ref_enc(refer * refer_mask, refer_mask)
        else:
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        if self.is_v2pro:
            if sv_emb is None:
                raise EngineError(f"SoVITS {self.version} needs the reference's speaker-verification embedding", {"version": self.version})
            ge += self.sv_emb(sv_emb).unsqueeze(-1)
            ge = self.prelu(ge)
        return ge

    def global_embedding(self, refer: torch.Tensor | Sequence[torch.Tensor], sv_emb: torch.Tensor | Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        """``ge`` for one reference, or the mean over a list (``sv_emb`` then a list too, one per reference)."""
        if isinstance(refer, (list, tuple)):
            ges = [self.reference_embedding(r, sv_emb[i] if self.is_v2pro and sv_emb is not None else None) for i, r in enumerate(refer)]
            return torch.stack(ges, 0).mean(0)
        assert isinstance(refer, torch.Tensor)
        return self.reference_embedding(refer, sv_emb if isinstance(sv_emb, torch.Tensor) or sv_emb is None else sv_emb[0])

    # -- decoding --

    def _prior(
        self,
        codes: torch.Tensor,
        text: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        result_length: int | None = None,
        overlap_frames: torch.Tensor | None = None,
        padding_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        return self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
            result_length=result_length,
            overlap_frames=overlap_frames,
            padding_length=padding_length,
        )

    def _synthesize(self, m_p: torch.Tensor, logs_p: torch.Tensor, y_mask: torch.Tensor, ge: torch.Tensor, noise_scale: float, generator: torch.Generator | None) -> torch.Tensor:
        noise = torch.empty_like(m_p).normal_(generator=generator)  # torch.randn_like(m_p) upstream
        z_p = m_p + noise * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        return self.dec((z * y_mask)[:, :, :], g=ge)

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        text: torch.Tensor,
        refer: torch.Tensor | Sequence[torch.Tensor],
        noise_scale: float = 0.5,
        speed: float = 1,
        sv_emb: torch.Tensor | Sequence[torch.Tensor] | None = None,
        *,
        generator: torch.Generator | None = None,
        ge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Semantic codes (n_q, 1, frames) + phone ids (1, phones) + reference spectrogram(s) → audio (1, 1, samples).

        ``generator`` draws the prior's noise (``None``: the default generator, as upstream). ``ge`` skips the
        reference encoders when the caller has it already (``global_embedding``).
        """
        if ge is None:
            ge = self.global_embedding(refer, sv_emb)
        _, m_p, logs_p, y_mask, _, _ = self._prior(codes, text, ge, speed)
        return self._synthesize(m_p, logs_p, y_mask, ge, noise_scale, generator)

    @torch.no_grad()
    def decode_streaming(
        self,
        codes: torch.Tensor,
        text: torch.Tensor,
        refer: torch.Tensor | Sequence[torch.Tensor],
        noise_scale: float = 0.5,
        speed: float = 1,
        sv_emb: torch.Tensor | Sequence[torch.Tensor] | None = None,
        result_length: int | None = None,
        overlap_frames: torch.Tensor | None = None,
        padding_length: int | None = None,
        *,
        generator: torch.Generator | None = None,
        ge: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One chunk of ``TTS.run``'s streaming mode → (audio, latent, latent mask); the latent's tail is the next
        chunk's ``overlap_frames``. ``result_length`` and ``padding_length`` count tokens (doubled at 25 Hz)."""
        if ge is None:
            ge = self.global_embedding(refer, sv_emb)
        if self.semantic_frame_rate == "25hz":
            result_length = 2 * result_length if result_length is not None else None
            padding_length = 2 * padding_length if padding_length is not None else None
        _, m_p, logs_p, y_mask, y_, y_mask_ = self._prior(codes, text, ge, speed, result_length, overlap_frames, padding_length)
        return self._synthesize(m_p, logs_p, y_mask, ge, noise_scale, generator), y_, y_mask_


# -- the reference spectrogram (module/mel_processing.py) -----------------------------------------------------------


def spectrogram(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int) -> torch.Tensor:
    """``spectrogram_torch(y, n_fft, sr, hop, win, center=False)``: reflect-padded linear magnitude, (batch, n_fft/2+1, frames)."""
    window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.view_as_real(
        torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=window, center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    )
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-8)


def normalize_reference(audio: torch.Tensor) -> torch.Tensor:
    """``_get_ref_spec``'s level check: a clip peaking above 1 is divided by min(2, peak)."""
    maxx = audio.abs().max()
    if maxx > 1:
        audio = audio / min(2, maxx)  # type: ignore[call-overload]
    return audio


def reference_spec(audio: torch.Tensor, filter_length: int, hop_length: int, win_length: int, **_: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``TTS._get_ref_spec`` from mono audio (1, samples) already at the model's sampling rate (float32) →
    (spectrogram (1, filter_length/2+1, frames), the level-checked audio). For v2Pro, the SV embedding is taken from
    that audio resampled to 16 kHz (``torchaudio.transforms.Resample`` upstream). ``**ckpt.spec_params`` fits."""
    audio = normalize_reference(audio)
    return spectrogram(audio, filter_length, hop_length, win_length), audio


# -- loading --------------------------------------------------------------------------------------------------------


def synthesizer_state(weight: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """A checkpoint's ``weight`` in float32 with weight norm folded (upstream's per-forward ``_weight_norm`` of the
    float32 parameters it copies the float16 file into: the same weights). Upstream keeps the weight norm and computes
    it at every call on the model's device in its dtype (``.half()`` on a GPU), so with a ``device`` the norm is computed
    there in ``dtype``, as it would be."""
    from mizuoto.runtime.loading import fold_weight_norm

    return fold_weight_norm({k: v.float() if v.is_floating_point() else v for k, v in weight.items()}, dtype=dtype if device is not None else None, device=device)


def build_synthesizer(ckpt: SovitsCheckpoint, ops: Any = default_ops.default, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> SynthesizerTrn:
    """The model for a checkpoint, weights loaded (float32, CPU, eval). Strict but for ``IGNORABLE``. ``dtype`` and
    ``device`` are where it will run (``synthesizer_state``)."""
    from mizuoto.runtime.loading import load_into

    if ckpt.model_version not in V2_FAMILY:
        raise EngineError(f"SoVITS {ckpt.model_version} is not a SynthesizerTrn model", {"version": ckpt.model_version})
    if ckpt.lora:
        raise EngineError(f"{ckpt.path.name} is a LoRA fine-tune, which only v3+ models use", {"path": str(ckpt.path)})
    emb = ckpt.weight.get("enc_p.text_embedding.weight")
    model = SynthesizerTrn.from_hps(ckpt.hps, n_symbols=int(emb.shape[0]) if emb is not None else None, ops=ops)
    load_into(model, synthesizer_state(ckpt.weight, dtype, device), ignore=IGNORABLE, what=f"SoVITS {ckpt.model_version} model")
    return model.eval()


def load_synthesizer(path: str | Path, ops: Any = default_ops.default) -> tuple[SynthesizerTrn, SovitsCheckpoint]:
    """``init_vits_weights`` for v1/v2/v2Pro/v2ProPlus: the model and its checkpoint (versions, hps, spectrogram settings)."""
    ckpt = read_sovits(path)
    return build_synthesizer(ckpt, ops), ckpt
