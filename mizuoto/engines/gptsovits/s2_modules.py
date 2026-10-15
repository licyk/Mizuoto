"""GPT-SoVITS' SoVITS training parts for the v2 family (MIT; ``s2_train.py``), around the native inference modules.

What inference leaves out and ``s2_train`` needs, operation for operation as upstream has it:

- **weight norm as parameters** (``weight_g``/``weight_v``, the weight recomputed before every forward, as
  ``torch.nn.utils.weight_norm`` does) on the layers upstream normalises; inference folds it at load;
- the **posterior encoder** ``enc_q`` (``models.PosteriorEncoder``) and the **multi-period discriminator**
  (``models.MultiPeriodDiscriminator``: the scale discriminator and five or seven period ones);
- the **training forwards** with dropout where upstream has it: the text encoder's transformers (``p_dropout``), the
  style encoder (0.1, every version), and ``SynthesizerTrn.forward`` (the frozen quantizer in eval mode, the
  posterior, the flow, ``rand_slice_segments``, the decoder);
- the **losses** (``module/losses.py``) and the mel features (``module/mel_processing.py``);
- the **initialisation** upstream's constructors draw (``init_synthesizer``, ``init_discriminator``): the pretrained
  weights overwrite it, but the draws move the generator dropout, the posterior's noise and the slices come from,
  so they are made in upstream's order, layer by layer, on the default generator.

Parameter names are upstream's, so a state dict saves and loads as upstream's does.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.gptsovits.sovits import SynthesizerTrn
from mizuoto.nn.flow.coupling import WN
from mizuoto.nn.transformer.vits import sequence_mask
from mizuoto.nn.vocoders.hifigan import ResBlock1, ResBlock2, get_padding
from mizuoto.runtime import ops as default_ops

LRELU_SLOPE = 0.1
STYLE_DROPOUT = 0.1
"""``MelStyleEncoder``'s dropout: upstream never passes another."""
PERIODS = {"v2Pro": (2, 3, 5, 7, 11, 17, 23), "v2ProPlus": (2, 3, 5, 7, 11, 17, 23)}
DEFAULT_PERIODS = (2, 3, 5, 7, 11)


# -- weight norm --------------------------------------------------------------------------------


class _WeightNorm:
    """``torch.nn.utils.weight_norm``'s forward pre-hook: ``weight = _weight_norm(weight_v, weight_g, dim)``."""

    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim

    def compute(self, module: nn.Module) -> torch.Tensor:
        return torch._weight_norm(getattr(module, self.name + "_v"), getattr(module, self.name + "_g"), self.dim)

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute(module))


def weight_norm(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    """Reparametrise ``module.weight`` as upstream's ``weight_norm`` does: ``g = ‖w‖`` (all but ``dim``), ``v = w``."""
    weight = module._parameters.pop(name)
    assert weight is not None
    hook = _WeightNorm(name, dim)
    module.register_parameter(name + "_g", nn.Parameter(torch.norm_except_dim(weight, 2, dim).data))
    module.register_parameter(name + "_v", nn.Parameter(weight.data))
    setattr(module, name, hook.compute(module))
    module.register_forward_pre_hook(hook)
    return module


def _wn_layers(wn: Any) -> Iterator[nn.Module]:
    if hasattr(wn, "cond_layer"):
        yield wn.cond_layer
    yield from wn.in_layers
    yield from wn.res_skip_layers


def weight_norm_generator(dec: Any) -> None:
    for up in dec.ups:
        weight_norm(up)
    for block in dec.resblocks:
        if isinstance(block, ResBlock1):
            for conv in [*block.convs1, *block.convs2]:
                weight_norm(conv)
        else:
            assert isinstance(block, ResBlock2)
            for conv in block.convs:
                weight_norm(conv)


def weight_norm_wn(wn: Any) -> None:
    for layer in _wn_layers(wn):
        weight_norm(layer)


# -- the posterior encoder and the discriminators ------------------------------------------------


class PosteriorEncoder(nn.Module):
    """``models.PosteriorEncoder``: the linear spectrogram → ``z`` (sampled), its mean and log-scale."""

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0, ops: Any = default_ops.default
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre = ops.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, ops=ops)
        self.proj = ops.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if g is not None:
            g = g.detach()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = wn_forward(self.enc, x, x_mask, g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.period = period
        pad = (get_padding(kernel_size, 1), 0)
        channels = (1, 32, 128, 512, 1024)
        self.convs = nn.ModuleList(ops.Conv2d(channels[i], channels[i + 1], (kernel_size, 1), (stride, 1), padding=pad) for i in range(4))
        self.convs.append(ops.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=pad))
        self.conv_post = ops.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                ops.Conv1d(1, 16, 15, 1, padding=7),
                ops.Conv1d(16, 64, 41, 4, groups=4, padding=20),
                ops.Conv1d(64, 256, 41, 4, groups=16, padding=20),
                ops.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
                ops.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
                ops.Conv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = ops.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """``models.MultiPeriodDiscriminator``: v2Pro and v2ProPlus add the periods 17 and 23. Weight norm throughout."""

    def __init__(self, version: str, ops: Any = default_ops.default) -> None:
        super().__init__()
        periods = PERIODS.get(version, DEFAULT_PERIODS)
        self.discriminators = nn.ModuleList([DiscriminatorS(ops), *(DiscriminatorP(p, ops=ops) for p in periods)])

    def apply_weight_norm(self) -> None:
        for d in self.discriminators:
            for conv in [*d.get_submodule("convs").children(), d.get_submodule("conv_post")]:
                weight_norm(conv)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# -- building, with upstream's initialisation draws ---------------------------------------------


def _reset(*layers: Any) -> None:
    """The layer's own ``reset_parameters`` (kaiming-uniform weight, uniform bias; normal for an embedding), which
    the ``ops.default`` layers skip."""
    for layer in layers:
        for cls in (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d):
            if isinstance(layer, cls):
                cls.reset_parameters(layer)  # type: ignore[arg-type]
                break
        else:
            raise TypeError(f"No initialisation for {type(layer).__name__}")


def _init_weights_draw(layer: Any) -> None:
    """``commons.init_weights`` after ``weight_norm``: ``m.weight.data.normal_(0, 0.01)`` lands on the weight computed
    from ``g`` and ``v``, which the next forward recomputes. Only the draw remains."""
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    torch.empty(weight.shape, dtype=weight.dtype, device=weight.device).normal_(0.0, 0.01)


@torch.no_grad()
def _init_attention(attn: Any) -> None:
    _reset(attn.conv_q, attn.conv_k, attn.conv_v, attn.conv_o)
    if attn.window_size is not None:
        std = attn.k_channels**-0.5
        attn.emb_rel_k.copy_(torch.randn(attn.emb_rel_k.shape) * std)
        attn.emb_rel_v.copy_(torch.randn(attn.emb_rel_v.shape) * std)
    for conv in (attn.conv_q, attn.conv_k, attn.conv_v):
        nn.init.xavier_uniform_(conv.weight)


@torch.no_grad()
def _init_encoder(encoder: Any) -> None:
    for i in range(encoder.n_layers):
        _init_attention(encoder.attn_layers[i])
        for norm in (encoder.norm_layers_1[i], encoder.norm_layers_2[i]):
            norm.gamma.fill_(1.0)
            norm.beta.zero_()
        _reset(encoder.ffn_layers[i].conv_1, encoder.ffn_layers[i].conv_2)


@torch.no_grad()
def _init_wn(wn: Any) -> None:
    """``modules.WN``: the condition layer, then each layer's input and residual/skip convolutions in turn."""
    if hasattr(wn, "cond_layer"):
        _reset(wn.cond_layer)
    for in_layer, res_skip in zip(wn.in_layers, wn.res_skip_layers, strict=True):
        _reset(in_layer, res_skip)


@torch.no_grad()
def init_text_encoder(enc_p: Any) -> None:
    _reset(enc_p.ssl_proj)
    _init_encoder(enc_p.encoder_ssl)
    _init_encoder(enc_p.encoder_text)
    _reset(enc_p.text_embedding)
    _init_attention(enc_p.mrte.cross_attention)
    _reset(enc_p.mrte.c_pre, enc_p.mrte.text_pre, enc_p.mrte.c_post)
    _init_encoder(enc_p.encoder2)
    _reset(enc_p.proj)


@torch.no_grad()
def init_style_encoder(ref_enc: Any) -> None:
    _reset(ref_enc.spectral[0].fc, ref_enc.spectral[3].fc)
    for glu in ref_enc.temporal:
        _reset(glu.conv1.conv)
    attn = ref_enc.slf_attn
    _reset(attn.w_qs, attn.w_ks, attn.w_vs, attn.fc, ref_enc.fc.fc)


@torch.no_grad()
def init_generator(dec: Any) -> None:
    """``models.Generator``: its layers, then ``init_weights`` on the residual blocks' and the upsamplers' weights."""
    _reset(dec.conv_pre, *dec.ups)
    for block in dec.resblocks:
        groups = [block.convs1, block.convs2] if isinstance(block, ResBlock1) else [block.convs]
        for convs in groups:
            _reset(*convs)
            for conv in convs:
                _init_weights_draw(conv)
    _reset(dec.conv_post)
    for up in dec.ups:
        _init_weights_draw(up)
    if hasattr(dec, "cond"):
        _reset(dec.cond)


@torch.no_grad()
def init_flow(flow: Any) -> None:
    for layer in flow.flows[::2]:
        _reset(layer.pre)
        _init_wn(layer.enc)
        _reset(layer.post)
        layer.post.weight.zero_()
        layer.post.bias.zero_()


@torch.no_grad()
def init_quantizer(quantizer: Any) -> None:
    """``EuclideanCodebook(kmeans_init=True)``: zeros, ``inited`` false (the pretrained state sets them); no draws."""
    for layer in quantizer.vq.layers:
        book = layer._codebook
        book.embed.zero_()
        book.inited.zero_()
        book.cluster_size.zero_()
        book.embed_avg.zero_()


def add_codebook_state(model: Any) -> None:
    """The codebooks' EMA state (``inited``, ``cluster_size``, ``embed_avg``) beside ``embed``, as upstream registers it,
    so the state dict has upstream's keys (inference keeps only ``embed``)."""
    for layer in model.quantizer.vq.layers:
        book = layer._codebook
        embed = book._buffers.pop("embed")
        assert embed is not None
        book.register_buffer("inited", torch.zeros(1))
        book.register_buffer("cluster_size", torch.zeros(embed.shape[0]))
        book.register_buffer("embed", embed)
        book.register_buffer("embed_avg", torch.zeros_like(embed))


@torch.no_grad()
def init_synthesizer(model: Any) -> None:
    """Upstream ``SynthesizerTrn.__init__``'s draws, in its order (enc_p, dec, enc_q, flow, ref_enc, ssl_proj, the
    quantizer, v2Pro's speaker layers), into the native modules."""
    init_text_encoder(model.enc_p)
    init_generator(model.dec)
    enc_q = model.enc_q
    _reset(enc_q.pre)
    _init_wn(enc_q.enc)
    _reset(enc_q.proj)
    init_flow(model.flow)
    init_style_encoder(model.ref_enc)
    _reset(model.ssl_proj)
    init_quantizer(model.quantizer)
    if model.is_v2pro:
        _reset(model.sv_emb, model.ge_to512)
        model.prelu.weight.fill_(0.25)


@torch.no_grad()
def init_discriminator(mpd: Any) -> None:
    for d in mpd.discriminators:
        _reset(*d.convs, d.conv_post)


def build_synthesizer(hps: dict[str, Any], n_symbols: int | None = None) -> SynthesizerTrn:
    """The training model: the native ``SynthesizerTrn`` with ``enc_q``, upstream's weight norm and its
    initialisation (drawn from the default generator, in upstream's order)."""
    model: Any = SynthesizerTrn.from_hps(hps, n_symbols=n_symbols)
    m = hps["model"]
    spec_channels = int(hps["data"]["filter_length"]) // 2 + 1
    inter, hidden, gin = int(m["inter_channels"]), int(m["hidden_channels"]), int(m.get("gin_channels", 0))
    model.add_module("enc_q", PosteriorEncoder(spec_channels, inter, hidden, 5, 1, 16, gin_channels=gin))
    add_codebook_state(model)
    init_synthesizer(model)
    weight_norm_generator(model.dec)
    weight_norm_wn(model.enc_q.enc)
    for layer in model.flow.flows[::2]:
        weight_norm_wn(layer.enc)
    return model


def build_discriminator(version: str) -> MultiPeriodDiscriminator:
    mpd = MultiPeriodDiscriminator(version)
    init_discriminator(mpd)
    mpd.apply_weight_norm()
    return mpd


# -- training forwards --------------------------------------------------------------------------


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    """``commons.fused_add_tanh_sigmoid_multiply``, scripted as upstream's is: TorchScript differentiates it
    symbolically, and its gradients differ from eager autograd's in the last bits."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def wn_forward(wn: Any, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
    """``modules.WN.forward`` (dropout 0 in every SoVITS use) through the scripted gate."""
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([wn.hidden_channels])
    if g is not None:
        g = wn.cond_layer(g)
    for i in range(wn.n_layers):
        x_in = wn.in_layers[i](x)
        if g is not None:
            cond_offset = i * 2 * wn.hidden_channels
            g_l = g[:, cond_offset : cond_offset + 2 * wn.hidden_channels, :]
        else:
            g_l = torch.zeros_like(x_in)
        acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
        res_skip_acts = wn.res_skip_layers[i](acts)
        if i < wn.n_layers - 1:
            x = (x + res_skip_acts[:, : wn.hidden_channels, :]) * x_mask
            output = output + res_skip_acts[:, wn.hidden_channels :, :]
        else:
            output = output + res_skip_acts
    return output * x_mask


def flow_forward(flow: Any, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
    """``ResidualCouplingBlock.forward`` (not reversed): mean-only couplings, each followed by a channel flip."""
    for layer in flow.flows[::2]:
        x0, x1 = torch.split(x, [layer.half_channels] * 2, 1)
        h = layer.pre(x0) * x_mask
        h = wn_forward(layer.enc, h, x_mask, g)
        m = layer.post(h) * x_mask
        logs = torch.zeros_like(m)
        x1 = m + x1 * torch.exp(logs) * x_mask
        x = torch.flip(torch.cat([x0, x1], 1), [1])
    return x


def _drop(x: torch.Tensor, p: float) -> torch.Tensor:
    """``nn.Dropout(p)`` in training mode (no draw at p = 0, as upstream's)."""
    return F.dropout(x, p, True) if p > 0 else x


def attention_forward(attn: Any, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None, p: float) -> torch.Tensor:
    """``attentions.MultiHeadAttention.forward`` with dropout on the attention weights."""
    query, key, value = attn.conv_q(x), attn.conv_k(c), attn.conv_v(c)
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, attn.n_heads, attn.k_channels, t_t).transpose(2, 3)
    key = key.view(b, attn.n_heads, attn.k_channels, t_s).transpose(2, 3)
    value = value.view(b, attn.n_heads, attn.k_channels, t_s).transpose(2, 3)
    scores = torch.matmul(query / math.sqrt(attn.k_channels), key.transpose(-2, -1))
    if attn.window_size is not None:
        key_relative_embeddings = attn._get_relative_embeddings(attn.emb_rel_k, t_s)
        rel_logits = torch.matmul(query / math.sqrt(attn.k_channels), key_relative_embeddings.unsqueeze(0).transpose(-2, -1))
        scores = scores + attn._relative_position_to_absolute_position(rel_logits)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e4)
    p_attn = _drop(F.softmax(scores, dim=-1), p)
    output = torch.matmul(p_attn, value)
    if attn.window_size is not None:
        relative_weights = attn._absolute_position_to_relative_position(p_attn)
        value_relative_embeddings = attn._get_relative_embeddings(attn.emb_rel_v, t_s)
        output = output + torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0))
    output = output.transpose(2, 3).contiguous().view(b, d, t_t)
    return attn.conv_o(output)


def ffn_forward(ffn: Any, x: torch.Tensor, x_mask: torch.Tensor, p: float) -> torch.Tensor:
    x = ffn.conv_1(ffn.padding(x * x_mask))
    x = _drop(torch.relu(x), p)
    x = ffn.conv_2(ffn.padding(x * x_mask))
    return x * x_mask


def encoder_forward(encoder: Any, x: torch.Tensor, x_mask: torch.Tensor, p: float) -> torch.Tensor:
    """``attentions.Encoder.forward`` with its dropouts (after attention, inside the FFN, after the FFN)."""
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    norms_1, norms_2 = encoder.norm_layers_1, encoder.norm_layers_2
    for i in range(encoder.n_layers):
        y = _drop(attention_forward(encoder.attn_layers[i], x, x, attn_mask, p), p)
        x = norms_1[i](x + y)
        y = _drop(ffn_forward(encoder.ffn_layers[i], x, x_mask, p), p)
        x = norms_2[i](x + y)
    return x * x_mask


def text_encoder_forward(
    enc_p: Any, y: torch.Tensor, y_lengths: torch.Tensor, text: torch.Tensor, text_lengths: torch.Tensor, ge: torch.Tensor, p: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``models.TextEncoder.forward`` (speed 1) in training: (features, m_p, logs_p, mask)."""
    y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
    y = enc_p.ssl_proj(y * y_mask) * y_mask
    y = encoder_forward(enc_p.encoder_ssl, y * y_mask, y_mask, p)
    text_mask = torch.unsqueeze(sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
    text_h = enc_p.text_embedding(text).transpose(1, 2)
    text_h = encoder_forward(enc_p.encoder_text, text_h * text_mask, text_mask, p)
    y = enc_p.mrte(y, y_mask, text_h, text_mask, ge)  # its attention has no dropout upstream
    y = encoder_forward(enc_p.encoder2, y * y_mask, y_mask, p)
    stats = enc_p.proj(y) * y_mask
    m, logs = torch.split(stats, enc_p.out_channels, dim=1)
    return y, m, logs, y_mask


def style_encoder_forward(enc: Any, x: torch.Tensor, mask: torch.Tensor, p: float = STYLE_DROPOUT) -> torch.Tensor:
    """``modules.MelStyleEncoder.forward`` in training: dropout after both spectral layers, in both gated
    convolutions, on the attention weights and after the attention's projection."""
    x = x.transpose(1, 2)
    pad = (mask.int() == 0).squeeze(1)
    max_len = x.shape[1]
    slf_attn_mask = pad.unsqueeze(1).expand(-1, max_len, -1)
    x = _drop(enc.spectral[1](enc.spectral[0](x)), p)
    x = _drop(enc.spectral[4](enc.spectral[3](x)), p)
    x = x.transpose(1, 2)
    for glu in enc.temporal:
        residual = x
        h = glu.conv1(x)
        x1, x2 = torch.split(h, split_size_or_sections=glu.out_channels, dim=1)
        x = residual + _drop(x1 * torch.sigmoid(x2), p)
    x = x.transpose(1, 2)
    x = x.masked_fill(pad.unsqueeze(-1), 0)
    attn = enc.slf_attn
    d_k, d_v, n_head = attn.d_k, attn.d_v, attn.n_head
    sz_b, len_x, _ = x.size()
    residual = x
    q = attn.w_qs(x).view(sz_b, len_x, n_head, d_k)
    k = attn.w_ks(x).view(sz_b, len_x, n_head, d_k)
    v = attn.w_vs(x).view(sz_b, len_x, n_head, d_v)
    q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)
    k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)
    v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)
    weights = torch.bmm(q, k.transpose(1, 2)) / attn.temperature
    weights = weights.masked_fill(slf_attn_mask.repeat(n_head, 1, 1), -float("inf"))
    weights = _drop(F.softmax(weights, dim=2), p)
    output = torch.bmm(weights, v)
    output = output.view(n_head, sz_b, len_x, d_v)
    output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)
    x = _drop(attn.fc(output), p) + residual
    x = enc.fc(x)
    return enc.temporal_avg_pool(x, mask=pad).unsqueeze(-1)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        ret[i] = x[i, :, idx_str : idx_str + segment_size]
    return ret


def rand_slice_segments(x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``commons.rand_slice_segments``: one ``torch.rand([batch])`` on the CPU picks each start."""
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([x.size(0)]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size), ids_str


def synthesizer_forward(
    model: Any,
    ssl: torch.Tensor,
    y: torch.Tensor,
    y_lengths: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    sv_emb: torch.Tensor | None,
    segment_size: int,
    p_dropout: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """``SynthesizerTrn.forward`` (frozen quantizer): (audio slice, kl_ssl, slice starts, mask, (z, z_p, m_p, logs_p, m_q, logs_q))."""
    y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
    if model.version == "v1":
        ge = style_encoder_forward(model.ref_enc, y * y_mask, y_mask)
    else:
        ge = style_encoder_forward(model.ref_enc, y[:, :704] * y_mask, y_mask)
    cond = ge
    if model.is_v2pro:
        assert sv_emb is not None
        ge = ge + model.sv_emb(sv_emb).unsqueeze(-1)
        ge = model.prelu(ge)
        cond = model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
    with torch.autocast(ssl.device.type, enabled=False), torch.no_grad():
        # The quantizer is in eval mode (frozen): no straight-through, no commitment loss, so nothing reaches ssl_proj.
        quantized = model.quantizer.decode(model.quantizer.encode(model.ssl_proj(ssl), n_q=1))
    kl_ssl = torch.zeros((), device=ssl.device)
    if model.semantic_frame_rate == "25hz":
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
    _, m_p, logs_p, y_mask = text_encoder_forward(model.enc_p, quantized, y_lengths, text, text_lengths, cond, p_dropout)
    z, m_q, logs_q, y_mask = model.enc_q(y, y_lengths, g=ge)
    z_p = flow_forward(model.flow, z, y_mask, g=ge)
    z_slice, ids_slice = rand_slice_segments(z, y_lengths, segment_size)
    o = model.dec(z_slice, g=ge)
    return o, kl_ssl, ids_slice, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)


# -- losses (module/losses.py) ------------------------------------------------------------------


def feature_loss(fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]) -> torch.Tensor:
    loss: Any = 0
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(real: list[torch.Tensor], fake: list[torch.Tensor]) -> torch.Tensor:
    loss: Any = 0
    for dr, dg in zip(real, fake, strict=True):
        loss += torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
    return loss


def generator_loss(fake: list[torch.Tensor]) -> torch.Tensor:
    loss: Any = 0
    for dg in fake:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss


def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
    z_p, logs_q, m_p, logs_p, z_mask = z_p.float(), logs_q.float(), m_p.float(), logs_p.float(), z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


# -- mel features (module/mel_processing.py) ----------------------------------------------------


def spec_to_mel(spec: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int, fmin: float, fmax: float | None) -> torch.Tensor:
    from mizuoto.nn.dsp.mel import librosa_mel

    basis = torch.from_numpy(librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).to(dtype=spec.dtype, device=spec.device)
    return torch.log(torch.clamp(torch.matmul(basis, spec), min=1e-5) * 1)
