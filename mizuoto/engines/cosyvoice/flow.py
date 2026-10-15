"""CosyVoice's flow models: speech tokens + prompt → mel (upstream ``flow/``, Apache-2.0, Alibaba).

- **v1** ``MaskedDiffWithXvec``: Conformer encoder, interpolating length regulator, U-Net estimator; noise
  drawn per request.
- **v2** ``CausalMaskedDiffWithXvec``: upsampling Conformer encoder (2 mel frames a token), causal U-Net.
- **v3** ``CausalMaskedDiffWithDiT``: a pre-lookahead convolution and CosyVoice 3's DiT.

v2 and v3 start from fixed noise: upstream seeds its global generator with 0 when it builds the model and
keeps the first 1×80×15000 normal draws, so the same tokens always give the same mel. The Euler solver
runs the conditional and unconditional halves as one batch of two with classifier-free guidance (0.7) on
a cosine time grid of 10 steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.transformer.wenet import make_pad_mask
from mizuoto.runtime import ops as default_ops

if TYPE_CHECKING:
    from mizuoto.runtime.compile import Compiled


def fixed_noise() -> torch.Tensor:
    """``CausalConditionalCFM.rand_noise``: ``set_all_random_seed(0)`` then ``torch.randn([1, 80, 15000])``."""
    return torch.randn([1, 80, 50 * 300], generator=torch.Generator().manual_seed(0))


class CFM(nn.Module):
    """``ConditionalCFM`` / ``CausalConditionalCFM``: the estimator and the Euler solver."""

    def __init__(self, estimator: nn.Module, inference_cfg_rate: float = 0.7, causal: bool = False) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.rand_noise = fixed_noise() if causal else None
        self.compiled_estimator: Compiled | None = None
        """``compute.compile``: the estimator compiled (``mizuoto.runtime.compile``), called instead of it."""

    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        spks: torch.Tensor,
        cond: torch.Tensor,
        generator: torch.Generator | None = None,
        streaming: bool = False,
        prompt_len: int = 0,
        cache: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        new_cache = None
        if self.rand_noise is not None:
            z = self.rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        else:
            assert generator is not None
            z = torch.empty_like(mu).normal_(generator=generator).to(mu.device).to(mu.dtype) * temperature
            cache_size = cache.shape[2] if cache is not None else 0
            if cache is not None and cache_size != 0:
                z[:, :, :cache_size] = cache[:, :, :, 0]
                mu[:, :, :cache_size] = cache[:, :, :, 1]
            z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
            mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
            new_cache = torch.stack([z_cache, mu_cache], dim=-1)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond, streaming), new_cache

    def solve_euler(self, x: torch.Tensor, t_span: torch.Tensor, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor, streaming: bool) -> torch.Tensor:
        t, dt = t_span[0], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []
        n = x.size(2)
        x_in = torch.zeros([2, 80, n], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, n], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, n], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, n], device=x.device, dtype=spks.dtype)
        estimator = self.compiled_estimator or self.estimator
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming)
            dphi, cfg_dphi = torch.split(dphi, [x.size(0), x.size(0)], dim=0)
            dphi = (1.0 + self.inference_cfg_rate) * dphi - self.inference_cfg_rate * cfg_dphi
            x = x + dt * dphi
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1].float()


class InterpolateRegulator(nn.Module):
    def __init__(self, channels: int, sampling_ratios: int, ops: Any) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(sampling_ratios):
            layers += [ops.Conv1d(channels, channels, 3, 1, 1), ops.GroupNorm(1, channels), nn.Mish()]
        layers.append(ops.Conv1d(channels, channels, 1, 1))
        self.model = nn.Sequential(*layers)

    def inference(self, x1: torch.Tensor, x2: torch.Tensor, mel_len1: int, mel_len2: int, input_frame_rate: int = 50) -> torch.Tensor:
        edge = int(20 / input_frame_rate * 22050 / 256)
        if x2.shape[1] > 40:
            head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=edge, mode="linear")
            mid = F.interpolate(x2[:, 20:-20].transpose(1, 2).contiguous(), size=mel_len2 - edge * 2, mode="linear")
            tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=edge, mode="linear")
            x2 = torch.concat([head, mid, tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode="linear")
        if x1.shape[1] != 0:
            x = torch.concat([F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode="linear"), x2], dim=2)
        else:
            x = x2
        return self.model(x).transpose(1, 2).contiguous()


class Flow(nn.Module):
    """The three flow models; ``version`` 1, 2 or 3, sized by the model's yaml (``config["flow"]``)."""

    def __init__(self, version: int, cfg: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        from mizuoto.nn.flow.unet1d import ConditionalDecoder
        from mizuoto.nn.transformer.wenet import Encoder, PreLookaheadLayer, UpsampleConformerEncoder

        self.version = version
        self.output_size = int(cfg.get("output_size", 80))
        self.input_frame_rate = int(cfg.get("input_frame_rate", 50 if version == 1 else 25))
        self.input_embedding = ops.Embedding(int(cfg["vocab_size"]), int(cfg["input_size"]))
        self.spk_embed_affine_layer = ops.Linear(int(cfg.get("spk_embed_dim", 192)), self.output_size)
        dec = cfg["decoder"]
        est = dec["estimator"]
        cfg_rate = float(dec.get("cfm_params", {}).get("content", {}).get("inference_cfg_rate", 0.7))
        if version in (2, 3):
            self.token_mel_ratio = int(cfg.get("token_mel_ratio", 2))
            self.pre_lookahead_len = int(cfg.get("pre_lookahead_len", 3))
        if version in (1, 2):
            enc = cfg["encoder"]
            size = int(enc["output_size"])
            if version == 1:
                self.encoder: nn.Module = Encoder(
                    int(enc.get("input_size", cfg["input_size"])),
                    size,
                    int(enc["attention_heads"]),
                    int(enc["linear_units"]),
                    int(enc["num_blocks"]),
                    kind="conformer",
                    input_layer=enc.get("input_layer", "linear"),
                    static_chunk_size=int(enc.get("static_chunk_size", 0)),
                    ops=ops,
                )
                reg = cfg["length_regulator"]
                self.length_regulator = InterpolateRegulator(int(reg["channels"]), len(reg["sampling_ratios"]), ops)
            else:
                self.encoder = UpsampleConformerEncoder(
                    int(enc.get("input_size", cfg["input_size"])),
                    size,
                    int(enc["attention_heads"]),
                    int(enc["linear_units"]),
                    int(enc["num_blocks"]),
                    static_chunk_size=int(enc.get("static_chunk_size", 25)),
                    ops=ops,
                )
            self.encoder_proj = ops.Linear(size, self.output_size)
            unet = ConditionalDecoder(
                int(est["in_channels"]),
                int(est["out_channels"]),
                tuple(est["channels"]),
                int(est.get("attention_head_dim", 64)),
                int(est.get("n_blocks", 4)),
                int(est.get("num_mid_blocks", 12)),
                int(est.get("num_heads", 8)),
                causal=version == 2,
                static_chunk_size=int(est.get("static_chunk_size", 50)),
                ops=ops,
            )
            self.decoder = CFM(unet, cfg_rate, causal=version == 2)
        elif version == 3:
            from mizuoto.engines.cosyvoice.dit import DiT

            pre = cfg["pre_lookahead_layer"]
            self.pre_lookahead_layer = PreLookaheadLayer(int(pre["in_channels"]), int(pre["channels"]), int(pre.get("pre_lookahead_len", 3)), ops)
            dit = DiT(
                dim=int(est["dim"]),
                depth=int(est["depth"]),
                heads=int(est["heads"]),
                dim_head=int(est["dim_head"]),
                ff_mult=float(est.get("ff_mult", 2)),
                mel_dim=int(est.get("mel_dim", 80)),
                mu_dim=int(est.get("mu_dim", 80)),
                spk_dim=int(est.get("spk_dim", 80)),
                static_chunk_size=int(est.get("static_chunk_size", 50)),
                ops=ops,
            )
            self.decoder = CFM(dit, cfg_rate, causal=True)
        else:
            raise ValueError(f"Unknown CosyVoice version {version}")

    def _embed(self, token: torch.Tensor, token_len: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.spk_embed_affine_layer(F.normalize(embedding, dim=1))
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        return self.input_embedding(torch.clamp(token, min=0)) * mask, embedding

    def inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        generator: torch.Generator | None = None,
        streaming: bool = False,
        finalize: bool = True,
        flow_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mel (1, 80, frames) for ``token`` after the prompt; v1 also returns its streaming cache."""
        assert token.shape[0] == 1
        device = token.device
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token_len = torch.tensor([token_len1], dtype=torch.int32, device=device) + torch.tensor([token_len2], dtype=torch.int32, device=device)
        token = torch.concat([prompt_token, token], dim=1)
        x, embedding = self._embed(token, token_len, embedding)
        mel_len1 = prompt_feat.shape[1]
        cache = None
        if self.version == 1:
            h, _ = self.encoder(x, token_len)
            h = self.encoder_proj(h)
            mel_len2 = int(token_len2 / self.input_frame_rate * 22050 / 256)
            h = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)
        elif self.version == 2:
            if finalize:
                h, _ = self.encoder(x, token_len, streaming=streaming)
            else:
                x, context = x[:, : -self.pre_lookahead_len], x[:, -self.pre_lookahead_len :]
                h, _ = self.encoder(x, token_len, context=context, streaming=streaming)
            mel_len2 = h.shape[1] - mel_len1
            h = self.encoder_proj(h)
        else:
            if finalize:
                h = self.pre_lookahead_layer(x)
            else:
                h = self.pre_lookahead_layer(x[:, : -self.pre_lookahead_len], context=x[:, -self.pre_lookahead_len :])
            h = h.repeat_interleave(self.token_mel_ratio, dim=1)
            mel_len2 = h.shape[1] - mel_len1
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            n_timesteps=10,
            spks=embedding,
            cond=conds,
            generator=generator,
            streaming=streaming,
            prompt_len=mel_len1,
            cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), cache
