"""The maths of Qwen3-TTS's speaker fine-tuning (upstream ``finetuning/dataset.py`` and ``sft_12hz.py``), for ``train.py``.

Everything keeps upstream's shapes and order of operations, so a step on the CPU in fp32 gives v1's loss and gradients:
the prompt's embeddings are summed in upstream's order, the talker and the code predictor run as transformers runs them
in training (a fresh cache on each forward, which hands attention contiguous copies of the keys and values), and both
losses are transformers' ``ForCausalLMLoss``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from mizuoto.core import codes
from mizuoto.engines.qwen3tts.codec import SpeechTokenizer
from mizuoto.engines.qwen3tts.config import Qwen3TtsConfig
from mizuoto.engines.qwen3tts.speaker import SpeakerEncoder, speaker_mel
from mizuoto.engines.qwen3tts.talker import Talker
from mizuoto.runtime import ops
from mizuoto.runtime.engine import EngineError
from mizuoto.runtime.lm.cache import DynamicCache
from mizuoto.runtime.lm.qwen import mask_for
from mizuoto.runtime.loading import load_into, load_state_dict

SAMPLE_RATE = 24000
SPEAKER_SLOT = 6
"""The prompt position the speaker embedding takes."""


def load_codec(model_dir: Path, device: str) -> SpeechTokenizer:
    """The speech tokenizer with its encoder, in fp32 (as v1's ``Qwen3TTSTokenizer.from_pretrained`` loads it)."""
    codec_config = json.loads((model_dir / "speech_tokenizer" / "config.json").read_text(encoding="utf-8"))
    codec = SpeechTokenizer(codec_config, ops.default, with_encoder=True)
    load_into(codec, load_state_dict(model_dir / "speech_tokenizer" / "model.safetensors"), what="Qwen3-TTS speech tokenizer")
    return codec.float().to(device).eval()


def encode_batch(codec: SpeechTokenizer, clips: list[Any], device: str) -> list[torch.Tensor]:
    """``Qwen3TTSTokenizer.encode`` of several 24 kHz clips: zero-padded to the longest, encoded together, each clip's
    codes cut to ``ceil(samples / 1920)`` frames → (frames, 16) each."""
    assert codec.encoder is not None
    longest = max(len(c) for c in clips)
    batch = torch.zeros((len(clips), longest), dtype=torch.float32)
    for i, clip in enumerate(clips):
        batch[i, : len(clip)] = torch.as_tensor(clip, dtype=torch.float32)
    encoded = codec.encoder.encode(batch.to(device)[:, None, :], codec.valid_quantizers)
    return [code[:, : -(-len(clip) // codec.downsample_rate)].transpose(0, 1) for clip, code in zip(clips, encoded)]


def load_model(model_dir: Path, config: Qwen3TtsConfig, dtype: torch.dtype, device: str) -> nn.Module:
    """Upstream ``Qwen3TTSForConditionalGeneration``'s trained parts in its parameter order: the talker, then the speaker
    encoder (which only gives the speaker embedding and gets no gradient)."""
    model = nn.Module()
    model.add_module("talker", Talker(config.talker, ops.default))
    model.add_module("speaker_encoder", SpeakerEncoder(config.speaker_encoder, ops.default))
    state = load_state_dict(model_dir / "model.safetensors")
    load_into(model, state, what="Qwen3-TTS model")
    del state
    return model.to(dtype=dtype).to(device)


def dataset_item(entry: dict[str, Any], tokenizer: Any) -> dict[str, torch.Tensor]:
    """``TTSDataset.__getitem__``: the text's ids without the trailing assistant turn, the codes, the reference's mel."""
    from mizuoto.nn.dsp.wav import read_wav

    text = f"<|im_start|>assistant\n{entry['text']}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    ref = entry["ref_audio"]
    wav, sr = read_wav(ref[0] if isinstance(ref, list) else ref)
    if sr != SAMPLE_RATE:
        raise EngineError(f"The reference must be at 24 kHz, not {sr} Hz", code=codes.INVALID_INPUT)
    with torch.inference_mode():
        ref_mel = speaker_mel(torch.from_numpy(wav).unsqueeze(0))
    return {"text_ids": text_ids[:, :-5], "audio_codes": torch.tensor(entry["audio_codes"], dtype=torch.long), "ref_mel": ref_mel}


def collate(batch: list[dict[str, torch.Tensor]], config: Qwen3TtsConfig) -> dict[str, torch.Tensor]:
    """``TTSDataset.collate_fn``: the text and codec channels of each prompt, their masks and the first codebook's labels."""
    t_cfg = config.talker
    lengths = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
    b, t = len(batch), max(lengths) + 8
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
    for i, data in enumerate(batch):
        text_ids = data["text_ids"]
        audio_codecs = data["audio_codes"]
        audio_codec_0 = audio_codecs[:, 0]
        n_text = text_ids.shape[1]
        n_codec = audio_codec_0.shape[0]
        # text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8 : 8 + n_text - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + n_text - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + n_text - 2 : 8 + n_text + n_codec, 0] = config.tts_pad_token_id
        text_embedding_mask[i, : 8 + n_text + n_codec] = True
        # codec channel
        input_ids[i, 3:8, 1] = torch.tensor([t_cfg.codec_nothink_id, t_cfg.codec_think_bos_id, t_cfg.codec_think_eos_id, 0, t_cfg.codec_pad_id])
        input_ids[i, 8 : 8 + n_text - 3, 1] = t_cfg.codec_pad_id
        input_ids[i, 8 + n_text - 3, 1] = t_cfg.codec_pad_id
        input_ids[i, 8 + n_text - 2, 1] = t_cfg.codec_bos_id
        input_ids[i, 8 + n_text - 1 : 8 + n_text - 1 + n_codec, 1] = audio_codec_0
        input_ids[i, 8 + n_text - 1 + n_codec, 1] = t_cfg.codec_eos_id
        codec_0_labels[i, 8 + n_text - 1 : 8 + n_text - 1 + n_codec] = audio_codec_0
        codec_0_labels[i, 8 + n_text - 1 + n_codec] = t_cfg.codec_eos_id
        codec_ids[i, 8 + n_text - 1 : 8 + n_text - 1 + n_codec, :] = audio_codecs
        codec_embedding_mask[i, 3 : 8 + n_text + n_codec] = True
        codec_embedding_mask[i, SPEAKER_SLOT] = False
        codec_mask[i, 8 + n_text - 1 : 8 + n_text - 1 + n_codec] = True
        attention_mask[i, : 8 + n_text + n_codec] = True
    return {
        "input_ids": input_ids,
        "ref_mels": torch.cat([d["ref_mel"] for d in batch], dim=0),
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }


class Indices(Dataset[int]):
    """The dataset's indices: the loader draws the order and batches them; the items are made once, beforehand (upstream's
    ``__getitem__`` makes the same item every epoch)."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class FreshCache(DynamicCache):
    """The cache a forward in training makes (the configs' ``use_cache``): transformers' first update concatenates onto
    an empty tensor, so attention gets contiguous copies of the keys and values."""

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat([key], dim=-2), torch.cat([value], dim=-2)


def step(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward of ``sft_12hz.py``: (loss, the talker's loss, the code predictor's loss, the speaker embedding)."""
    talker = model.get_submodule("talker")
    assert isinstance(talker, Talker)
    dtype = talker.codec_head.weight.dtype
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    codec_mask = batch["codec_mask"]
    attention_mask = batch["attention_mask"]
    speaker_embedding = model.get_submodule("speaker_encoder")(batch["ref_mels"].to(dtype)).detach()

    # upstream embeds the text without the text projection here (its width is the talker's in the 1.7B model)
    input_text_embedding = talker.model.text_embedding(input_ids[:, :, 0]) * batch["text_embedding_mask"]
    input_codec_embedding = talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"]
    input_codec_embedding[:, SPEAKER_SLOT, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        codec_i_embedding = talker.code_predictor.model.codec_embedding[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    # the talker over every position but the last; positions as upstream's get_rope_index (float cumsum, 1 where padded)
    embeds = input_embeddings[:, :-1, :]
    mask = attention_mask[:, :-1]
    positions = mask.float().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    logits, hidden = talker(embeds, positions, mask_for(mask, embeds.shape[1], 0), FreshCache())
    talker_loss = causal_lm_loss(logits, batch["codec_0_labels"][:, 1:])

    # the code predictor on each frame: the talker's hidden state, then the frame's codebooks 0–14
    sub_loss = sub_talker_loss(talker, codec_ids[codec_mask], hidden[codec_mask[:, :-1]])
    return talker_loss + 0.3 * sub_loss, talker_loss, sub_loss, speaker_embedding


def sub_talker_loss(talker: Talker, codec_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Upstream ``forward_sub_talker_finetune`` and the code predictor's ``forward_finetune``."""
    cp = talker.code_predictor
    groups = codec_ids.shape[1]
    embeds = [hidden.unsqueeze(1)]
    for i in range(groups - 1):
        if i == 0:
            embeds.append(talker.model.codec_embedding(codec_ids[:, :1]))
        else:
            embeds.append(cp.model.codec_embedding[i - 1](codec_ids[:, i : i + 1]))
    x = cp.small_to_mtp_projection(torch.cat(embeds, dim=1))
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    out = cp.model(x, positions, None, FreshCache())
    logits = torch.stack([cp.lm_head[i - 1](out[:, i]) for i in range(1, groups)], dim=1)
    return causal_lm_loss(logits, codec_ids[:, 1:])


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """transformers' ``ForCausalLMLoss``: logits in float32, the labels shifted left once more (-100 appended), the mean
    cross-entropy over the labels that are not -100."""
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), shift.view(-1).to(logits.device), ignore_index=-100)
