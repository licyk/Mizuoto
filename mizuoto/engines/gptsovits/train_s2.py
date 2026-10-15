"""GPT-SoVITS' SoVITS training (stage ``sovits``), native: upstream's ``s2_train.py`` for v2, v2Pro and v2ProPlus and
``s2_train_v3_lora.py`` for v4 and v5 Turbo, as v1 ran them (the web UI's config, one process, no DDP).

``train_sovits(ctx, workdir, version, hyperparams, base, assets)`` reads the experiment folder the preparation
stages wrote (``2-name2text.txt``, ``4-cnhubert/``, ``5-wav32k/``, and ``7-sv_cn/`` for v2Pro) and:

- **data** as ``TextAudioSpeakerLoader`` (or ``…V4``) builds it: the clips every folder has, repeated to at least 100,
  shuffled by ``random.seed(1234)``, kept between 0.6 and 54 s; batched by ``DistributedBucketSampler`` (length buckets,
  shuffled by a generator seeded with the epoch) and padded by the collate function, sorted by length;
- **v2 family**: the generator and the multi-period discriminator from the base weights and the pretrained
  discriminator; per batch the discriminator's step (least squares) and the generator's (adversarial + feature
  matching + 45 × mel L1 + KL, the frozen quantizer's ``kl_ssl`` being 0); AdamW (0.8, 0.99), the text encoder's parts
  at ``text_low_lr_rate`` × the rate, ExponentialLR stepped once before training and after each epoch;
- **v4, v5 Turbo**: the CFM's attention projections as a LoRA (``s2_lora``), the rest frozen; the flow-matching loss;
- fp32 on a CPU, autocast fp16 with a gradient scaler on CUDA (``training.precision`` auto or fp16), upstream's
  halved batch in full precision;
- every ``save_every_epoch`` epochs (default: the last) the weights as ``savee`` writes them: half precision, without
  ``enc_q`` (the LoRA's trained tensors only), ``config``, ``info`` and, for v2Pro/v2ProPlus and LoRAs, the version tag
  in the first two bytes.

Seeded alike, a run draws what upstream draws, in its order: the model's initialisation (overwritten by the base
weights), the data loader's base seed, then per step the dropouts, the posterior's noise and the slice starts (v2) or
the time steps, the noise, the prompt lengths and the shortcut draw (v4). Upstream lists the clips from a ``set`` of
names, whose order depends on the process's string hashing; here they are sorted before the shuffle.

Not written: upstream's resumable ``G_*.pth``/``D_*.pth`` (optimiser states, ~1.7 GB), which nothing here resumes from.
"""

from __future__ import annotations

import copy
import io
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError

if TYPE_CHECKING:
    import torch
    from torch import nn

    from mizuoto.training import StageContext

V2_FAMILY = ("v2", "v2Pro", "v2ProPlus")
LORA_VERSIONS = ("v4", "v5turbo")
V2PRO = ("v2Pro", "v2ProPlus")
DISCRIMINATORS = {"v2": "gptsovits/s2d-v2", "v2Pro": "gptsovits/s2d-v2pro", "v2ProPlus": "gptsovits/s2d-v2proplus"}
VERSION_TAGS = {"v3": b"03", "v4": b"04", "v2Pro": b"05", "v2ProPlus": b"06", "v5dev": b"07", "v5turbo": b"08"}
"""``process_ckpt.model_version2byte``."""

BOUNDARIES_V2 = (32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)
BOUNDARIES_V3 = (32, 300, 400, 500, 600, 700, 800, 900, 1000)
MIN_ITEMS = 100
"""The loaders repeat a smaller dataset up to this many items."""
LOADER_SEED = 1234
"""``random.seed(1234)`` before the loaders' shuffle."""

S2_CONFIG: dict[str, Any] = {
    "train": {
        "log_interval": 100,
        "eval_interval": 500,
        "seed": 1234,
        "epochs": 100,
        "learning_rate": 0.0001,
        "betas": [0.8, 0.99],
        "eps": 1e-09,
        "batch_size": 32,
        "fp16_run": True,
        "lr_decay": 0.999875,
        "segment_size": 20480,
        "init_lr_ratio": 1,
        "warmup_epochs": 0,
        "c_mel": 45,
        "c_kl": 1.0,
        "text_low_lr_rate": 0.4,
        "grad_ckpt": False,
    },
    "data": {
        "max_wav_value": 32768.0,
        "sampling_rate": 32000,
        "filter_length": 2048,
        "hop_length": 640,
        "win_length": 2048,
        "n_mel_channels": 128,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "add_blank": True,
        "n_speakers": 300,
        "cleaned_text": True,
    },
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 8, 2, 2],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 512,
        "semantic_frame_rate": "25hz",
        "freeze_quantizer": True,
    },
    "s2_ckpt_dir": "logs/s2/big2k1",
    "content_module": "cnhubert",
}
"""Upstream's ``configs/s2.json`` (v2, and v4/v5 as the web UI trains them)."""
CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "v2Pro": {"p_dropout": 0.0, "gin_channels": 1024},
    "v2ProPlus": {"p_dropout": 0.0, "gin_channels": 1024, "upsample_initial_channel": 768, "upsample_kernel_sizes": [20, 16, 8, 2, 2]},
}
"""``configs/s2v2Pro.json`` and ``configs/s2v2ProPlus.json``: their model section's differences from ``s2.json``."""


def s2_config(version: str) -> dict[str, Any]:
    """The config upstream's web UI starts from for ``version`` (a copy)."""
    config = copy.deepcopy(S2_CONFIG)
    config["model"].update(copy.deepcopy(CONFIG_OVERRIDES.get(version, {})))
    return config


# -- data ---------------------------------------------------------------------------------------


@dataclass
class Clip:
    name: str
    phones: list[int]


@dataclass
class S2Data:
    """``TextAudioSpeakerLoader`` (``V4`` for LoRA versions): the item list, the lengths and the items themselves."""

    folder: Path
    version: str
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    clips: list[Clip] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    _cache: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    @property
    def v2pro(self) -> bool:
        return self.version in V2PRO

    @property
    def lora(self) -> bool:
        return self.version in LORA_VERSIONS

    def scan(self) -> S2Data:
        from mizuoto.engines.gptsovits.text.symbols import cleaned_text_to_sequence

        folder = self.folder
        listing = folder / "2-name2text.txt"
        needed = [listing, folder / "4-cnhubert", folder / "5-wav32k", *([folder / "7-sv_cn"] if self.v2pro else [])]
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            raise EngineError(f"The SoVITS stage needs the prepared features: {', '.join(missing)} missing in {folder}", code=codes.INVALID_INPUT)
        phonemes: dict[str, str] = {}
        for line in listing.read_text(encoding="utf-8").strip("\n").split("\n"):
            parts = line.split("\t")
            if len(parts) == 4:
                phonemes[parts[0]] = parts[1]
        names = set(phonemes) & {n[:-3] for n in os.listdir(folder / "4-cnhubert")} & set(os.listdir(folder / "5-wav32k"))
        if self.v2pro:
            names &= {n[:-3] for n in os.listdir(folder / "7-sv_cn")}
        if not names:
            raise EngineError(f"No clip has its text, features and audio in {folder}", code=codes.INVALID_INPUT)
        items = sorted(names)  # upstream: list(set(...)), in the process's hash order
        if len(items) < MIN_ITEMS:
            items = items * max(2, int(MIN_ITEMS / len(items)))
        # The loader's own seed; upstream reseeds the global generator, which v4's shortcut draw then continues.
        self.rng = random.Random(LOADER_SEED)
        self.rng.shuffle(items)
        for name in items:
            try:
                phones = cleaned_text_to_sequence(phonemes[name].split(" "))
            except KeyError:
                continue
            size = (folder / "5-wav32k" / name).stat().st_size
            duration = size / self.sampling_rate / 2
            if duration == 0:
                continue
            if 54 > duration > 0.6:
                self.clips.append(Clip(name, phones))
                self.lengths.append(size // (2 * self.hop_length))
        if len(self.clips) <= 1:
            raise EngineError("SoVITS training needs clips between 0.6 and 54 seconds", code=codes.INVALID_INPUT)
        return self

    def item(self, index: int) -> tuple[torch.Tensor, ...]:
        """``get_audio_text_speaker_pair``: (ssl, spec, wav, text[, sv_emb]) for v2; (ssl, spec, mel, text) for v4/v5."""
        import torch

        clip = self.clips[index]
        if clip.name not in self._cache:
            # Upstream's data-loader workers run torch on one thread; the spectrograms are made the same way.
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                self._cache[clip.name] = self._load(clip.name)
            finally:
                torch.set_num_threads(threads)
        text = torch.FloatTensor(clip.phones)
        return (*self._cache[clip.name][:3], text, *self._cache[clip.name][3:])

    def _load(self, name: str) -> tuple[torch.Tensor, ...]:
        import torch
        import torch.nn.functional as F

        from mizuoto.engines.gptsovits.sovits import spectrogram
        from mizuoto.nn.dsp.wav import read_wav

        samples, rate = read_wav(self.folder / "5-wav32k" / name)
        if rate != self.sampling_rate:
            raise EngineError(f"{name} is at {rate} Hz, not {self.sampling_rate}: prepare the speech features again", code=codes.INVALID_INPUT)
        audio = torch.from_numpy(samples).unsqueeze(0)
        spec = spectrogram(audio, self.filter_length, self.hop_length, self.win_length).squeeze(0)
        ssl = torch.load(self.folder / "4-cnhubert" / f"{name}.pt", map_location="cpu", weights_only=True)
        if ssl.shape[-1] != spec.shape[-1]:
            ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
        if self.lora:
            from mizuoto.engines.gptsovits.s2_lora import item_mel

            return ssl, spec, item_mel(audio)
        if self.v2pro:
            sv = torch.load(self.folder / "7-sv_cn" / f"{name}.pt", map_location="cpu", weights_only=True)
            return ssl, spec, audio, sv
        return ssl, spec, audio


class BucketSampler:
    """``DistributedBucketSampler`` with one replica: length buckets, shuffled per epoch by a generator seeded with it."""

    def __init__(self, lengths: list[int], batch_size: int, boundaries: tuple[int, ...]) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(lengths):
            b = self._bisect(length)
            if b != -1:
                self.buckets[b].append(i)
        for i in range(len(self.buckets) - 1, -1, -1):
            if not self.buckets[i]:
                self.buckets.pop(i)
                self.boundaries.pop(i + 1)
        self.num_samples_per_bucket = [len(b) + (batch_size - len(b) % batch_size) % batch_size for b in self.buckets]
        self.num_samples = sum(self.num_samples_per_bucket)
        if not self.buckets:
            raise EngineError("No clip fits SoVITS training's length buckets", code=codes.INVALID_INPUT)

    def _bisect(self, x: int, lo: int = 0, hi: int | None = None) -> int:
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            if x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            return self._bisect(x, mid + 1, hi)
        return -1

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def batches(self, epoch: int) -> list[list[int]]:
        import torch

        g = torch.Generator()
        g.manual_seed(epoch)
        indices = [torch.randperm(len(bucket), generator=g).tolist() for bucket in self.buckets]
        batches = []
        for bucket, ids, total in zip(self.buckets, indices, self.num_samples_per_bucket, strict=True):
            rem = total - len(bucket)
            ids = ids + ids * (rem // len(bucket)) + ids[: rem % len(bucket)]
            for j in range(len(ids) // self.batch_size):
                batches.append([bucket[i] for i in ids[j * self.batch_size : (j + 1) * self.batch_size]])
        order = torch.randperm(len(batches), generator=g).tolist()
        return [batches[i] for i in order]


def collate_v2(batch: list[tuple[torch.Tensor, ...]], v2pro: bool) -> tuple[torch.Tensor, ...]:
    """``TextAudioSpeakerCollate``: zero-padded, longest spectrogram first."""
    import torch

    _, order = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
    max_ssl_len = int(2 * ((max(x[0].size(2) for x in batch) // 2) + 1))
    max_spec_len = int(2 * ((max(x[1].size(1) for x in batch) // 2) + 1))
    max_wav_len = max(x[2].size(1) for x in batch)
    max_text_len = max(x[3].size(0) for x in batch)
    n = len(batch)
    ssl_lengths, spec_lengths, wav_lengths, text_lengths = (torch.LongTensor(n) for _ in range(4))
    spec_padded = torch.FloatTensor(n, batch[0][1].size(0), max_spec_len).zero_()
    wav_padded = torch.FloatTensor(n, 1, max_wav_len).zero_()
    ssl_padded = torch.FloatTensor(n, batch[0][0].size(1), max_ssl_len).zero_()
    text_padded = torch.LongTensor(n, max_text_len).zero_()
    sv_embs = torch.FloatTensor(n, 20480) if v2pro else None
    for i in range(n):
        row = batch[order[i]]
        ssl_padded[i, :, : row[0].size(2)] = row[0][0, :, :]
        ssl_lengths[i] = row[0].size(2)
        spec_padded[i, :, : row[1].size(1)] = row[1]
        spec_lengths[i] = row[1].size(1)
        wav_padded[i, :, : row[2].size(1)] = row[2]
        wav_lengths[i] = row[2].size(1)
        text_padded[i, : row[3].size(0)] = row[3]
        text_lengths[i] = row[3].size(0)
        if sv_embs is not None:
            sv_embs[i] = row[4].reshape(-1)
    out = (ssl_padded, ssl_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, text_padded, text_lengths)
    return (*out, sv_embs) if sv_embs is not None else out


def epoch_batches(sampler: BucketSampler, epoch: int) -> list[list[int]]:
    """One epoch's batches (a seam for tests and comparisons, which shorten epochs)."""
    return sampler.batches(epoch)


# -- the stage ----------------------------------------------------------------------------------


def train_sovits(
    ctx: StageContext,
    workdir: Path,
    version: str,
    hyperparams: dict[str, Any],
    base: Path,
    assets: dict[str, Any],
    *,
    save_dir: Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Train SoVITS from the base weights ``base`` (the pretrained ``s2G*.pth``) on the prepared folder ``workdir``.

    ``hyperparams``: ``sovits_epochs`` (8), ``batch_size`` (4, halved in full precision as upstream's web UI does),
    ``text_low_lr_rate`` (0.4), ``lora_rank`` (32), ``grad_ckpt`` (false), ``save_every_epoch`` (the last epoch).
    The weights go to ``save_dir`` (default ``workdir / "weights"``) as ``<name>_e<epoch>_s<step>.pth`` (``_l<rank>``
    for a LoRA). Returns ``{"weights": path, "epochs", "steps", "files"}``.
    """
    import torch

    if version not in V2_FAMILY and version not in LORA_VERSIONS:
        raise EngineError(f"SoVITS training runs {', '.join(V2_FAMILY + LORA_VERSIONS)}, not {version}", code=codes.INVALID_INPUT)
    if not Path(base).is_file():
        raise EngineError(f"The base SoVITS weights are missing: {base}", code=codes.MODEL_MISSING)
    device = torch.device(ctx.device)
    fp16 = device.type == "cuda" and ctx.precision in ("auto", "fp16")
    batch = int(hyperparams.get("batch_size") or 4)
    batch = batch if fp16 else max(1, batch // 2)
    epochs = int(hyperparams.get("sovits_epochs") or 8)
    save_every = int(hyperparams.get("save_every_epoch") or epochs)
    name = name or ctx.name
    save_dir = Path(save_dir) if save_dir is not None else Path(workdir) / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    hps = s2_config(version)
    hps["model"]["version"] = version
    train = hps["train"]
    train.update(
        {
            "fp16_run": fp16,
            "batch_size": batch,
            "epochs": epochs,
            "text_low_lr_rate": float(hyperparams.get("text_low_lr_rate") or 0.4),
            "pretrained_s2G": str(base),
            "pretrained_s2D": "",
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": save_every,
            "gpu_numbers": "0" if device.type == "cuda" else "",
            "grad_ckpt": bool(hyperparams.get("grad_ckpt", False)),
            "lora_rank": int(hyperparams.get("lora_rank") or 32),
        }
    )
    if version in V2_FAMILY:
        disc = _discriminator(assets, version)
        if disc is None:
            ctx.log(f"No pretrained discriminator ({DISCRIMINATORS[version]}): it starts from random weights", "warning")
        else:
            train["pretrained_s2D"] = str(disc)
    hps["data"]["exp_dir"] = hps["s2_ckpt_dir"] = str(workdir)
    hps.update({"save_weight_dir": str(save_dir), "name": name, "version": version, "pretrain": None, "resume_step": None})

    data_hps = hps["data"]
    data = S2Data(Path(workdir), version, int(data_hps["sampling_rate"]), int(data_hps["filter_length"]), int(data_hps["hop_length"]), int(data_hps["win_length"])).scan()
    sampler = BucketSampler(data.lengths, batch, BOUNDARIES_V3 if version in LORA_VERSIONS else BOUNDARIES_V2)
    ctx.log(f"SoVITS {version}: {len(data.clips)} items, {len(sampler)} steps per epoch, batch {batch}, {epochs} epoch(s), {'fp16' if fp16 else 'fp32'}")
    if version in LORA_VERSIONS:
        from mizuoto.engines.gptsovits.s2_lora import train_lora

        return train_lora(ctx, hps, data, sampler, device, fp16, save_dir)
    return _train_v2(ctx, hps, data, sampler, device, fp16, save_dir)


def _discriminator(assets: dict[str, Any], version: str) -> Path | None:
    folder = assets.get(DISCRIMINATORS[version])
    if not folder:
        return None
    folder = Path(folder)
    if folder.is_file():
        return folder
    found = sorted(folder.rglob("s2D*.pth"))
    return found[0] if found else None


def _train_v2(ctx: StageContext, hps: dict[str, Any], data: S2Data, sampler: BucketSampler, device: torch.device, fp16: bool, save_dir: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from mizuoto.engines.gptsovits import s2_modules as M
    from mizuoto.engines.gptsovits.checkpoints import load_checkpoint
    from mizuoto.engines.gptsovits.sovits_v3 import mel_spectrogram

    version = hps["model"]["version"]
    train, data_hps = hps["train"], hps["data"]
    segment = int(train["segment_size"]) // int(data_hps["hop_length"])
    p_dropout = float(hps["model"]["p_dropout"])

    # Upstream builds the generator, then the discriminator (each drawing its initialisation), then loads the
    # pretrained weights over them (strict=False).
    base = load_checkpoint(train["pretrained_s2G"])["weight"]
    emb = base.get("enc_p.text_embedding.weight")
    net_g = M.build_synthesizer(hps, n_symbols=int(emb.shape[0]) if emb is not None else None)
    net_d = M.build_discriminator(version)
    _load(net_g, base, "SoVITS")
    order = list(base)
    del base
    if train["pretrained_s2D"]:
        _load(net_d, load_checkpoint(train["pretrained_s2D"])["weight"], "discriminator")
    net_g.to(device)
    net_d.to(device)

    lr, ratio = float(train["learning_rate"]), float(train["text_low_lr_rate"])
    enc_p = net_g.enc_p
    text_parts = [enc_p.text_embedding, enc_p.encoder_text, enc_p.mrte]
    low = {id(p) for part in text_parts for p in part.parameters()}
    base_params = [p for p in net_g.parameters() if id(p) not in low and p.requires_grad]
    betas = (float(train["betas"][0]), float(train["betas"][1]))
    optim_g = torch.optim.AdamW(
        [{"params": base_params, "lr": lr}, *({"params": part.parameters(), "lr": lr * ratio} for part in text_parts)], lr, betas=betas, eps=float(train["eps"])
    )
    optim_d = torch.optim.AdamW(net_d.parameters(), lr, betas=betas, eps=float(train["eps"]))
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=float(train["lr_decay"]), last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=float(train["lr_decay"]), last_epoch=-1)
    scheduler_g.step()  # upstream steps both once before training (``for _ in range(epoch_str)``); torch warns about it
    scheduler_d.step()
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    c_mel, c_kl = float(train["c_mel"]), float(train["c_kl"])
    mel_args = (int(data_hps["filter_length"]), int(data_hps["n_mel_channels"]), int(data_hps["sampling_rate"]))
    fmin, fmax = float(data_hps["mel_fmin"]), data_hps["mel_fmax"]
    hop, win = int(data_hps["hop_length"]), int(data_hps["win_length"])
    epochs, save_every = int(train["epochs"]), int(train["save_every_epoch"])
    total = epochs * len(sampler)
    global_step = 0
    saved: list[str] = []
    autocast = torch.autocast(device.type, dtype=torch.float16, enabled=fp16)
    no_autocast = torch.autocast(device.type, enabled=False)
    ctx.progress(0.0, f"Epoch 1 of {epochs}")
    for epoch in range(1, epochs + 1):
        batches = epoch_batches(sampler, epoch)
        if epoch == 1:
            # The data loader's base seed: its iterator draws one when it is made (persistent workers: once).
            torch.empty((), dtype=torch.int64).random_()
        net_g.train()
        net_d.train()
        for indices in batches:
            items = collate_v2([data.item(i) for i in indices], data.v2pro)
            ssl, _, spec, spec_lengths, y, _, text, text_lengths = (t.to(device) for t in items[:8])
            sv_emb = items[8].to(device) if data.v2pro else None
            with autocast:
                y_hat, kl_ssl, ids_slice, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = M.synthesizer_forward(
                    net_g, ssl, spec, spec_lengths, text, text_lengths, sv_emb, segment, p_dropout
                )
                mel = M.spec_to_mel(spec, *mel_args, fmin, fmax)
                y_mel = M.slice_segments(mel, ids_slice, segment)
                y_hat_mel = mel_spectrogram(y_hat.squeeze(1), mel_args[0], mel_args[1], mel_args[2], hop, win, fmin, fmax)
                y = M.slice_segments(y, ids_slice * hop, int(train["segment_size"]))
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with no_autocast:
                    loss_disc = M.discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            scaler.step(optim_d)
            with autocast:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with no_autocast:
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel
                    loss_kl = M.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl
                    loss_fm = M.feature_loss(fmap_r, fmap_g)
                    loss_gen = M.generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            scaler.step(optim_g)
            scaler.update()
            values = {"loss_disc": loss_disc, "loss_gen": loss_gen, "loss_fm": loss_fm, "loss_mel": loss_mel, "kl_ssl": kl_ssl, "loss_kl": loss_kl}
            ctx.metric(epoch, global_step, {k: float(v.item()) for k, v in values.items()}, float(optim_g.param_groups[0]["lr"]))
            global_step += 1
            ctx.progress(min(1.0, global_step / max(1, total)), f"Epoch {epoch} of {epochs}")
        if epoch % save_every == 0:
            path = save_dir / f"{hps['name']}_e{epoch}_s{global_step}.pth"
            save_weights(net_g.state_dict(), path, hps, epoch, global_step, order=order)
            saved.append(str(path))
            ctx.output(path, f"SoVITS epoch {epoch}", kind="checkpoint")
        scheduler_g.step()
        scheduler_d.step()
    if not saved:
        raise EngineError("Training saved no weights (save_every_epoch is above the epochs)", code=codes.ENGINE_ERROR)
    return {"weights": saved[-1], "epochs": epochs, "steps": global_step, "files": saved}


def _load(model: nn.Module, state: dict[str, torch.Tensor], what: str) -> None:
    """``load_state_dict(..., strict=False)`` as upstream; here every key must fit (the published files have them all)."""
    from mizuoto.runtime.loading import load_into

    load_into(model, state, what=what)


def save_weights(state: dict[str, torch.Tensor], path: Path, hps: dict[str, Any], epoch: int, steps: int, *, order: list[str] | None = None, lora_rank: int | None = None) -> None:
    """``process_ckpt.savee``: every tensor but ``enc_q``'s in half precision, the config and ``"<e>epoch_<s>iteration"``;
    v2Pro/v2ProPlus and LoRA files carry the version tag over the zip's first two bytes (``my_save2``).

    ``config`` is a plain dict (upstream pickles its ``utils.HParams``); both load the same way.
    """
    import torch

    keys = [k for k in (order or []) if k in state] + [k for k in state if k not in set(order or [])]
    weight = OrderedDict((k, state[k].detach().half().cpu()) for k in keys if "enc_q" not in k)
    opt: OrderedDict[str, Any] = OrderedDict(weight=weight, config=copy.deepcopy(hps), info=f"{epoch}epoch_{steps}iteration")
    version = str(hps["model"]["version"])
    if lora_rank:
        opt["lora_rank"] = lora_rank
    bio = io.BytesIO()
    torch.save(opt, bio)
    raw = bio.getvalue()
    if lora_rank or version in V2PRO:
        raw = VERSION_TAGS[version] + raw[2:]
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(raw)
    tmp.replace(path)
