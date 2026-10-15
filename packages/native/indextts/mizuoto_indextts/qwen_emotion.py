"""IndexTTS-2's ``QwenEmotion``: a fine-tuned Qwen3-0.6B (``qwen0.6bemo4-merge``) that turns a description into the
eight emotion weights.

As upstream runs it (``infer_v2.QwenEmotion``, transformers 4.52): float16 weights, the model's own chat template
(``System: 文本情感分类<|endoftext|>\\nHuman: …<|endoftext|>\\nAssistant:``), greedy decoding with a KV cache
until ``<|endoftext|>``, then the answer parsed as JSON (or ``key: value`` pairs), "sad" swapped for
"melancholic" when the text names melancholy, and the scores clamped to [0, 1.2] (all zero: calm).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

PROMPT = "文本情感分类"
CN_TO_EN = {"高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid", "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm"}
ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
MELANCHOLIC_WORDS = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"}
MAX_SCORE, MIN_SCORE = 1.2, 0.0
THINK_END = 151668


def chat_prompt(text: str) -> str:
    """``apply_chat_template`` with the model's ``chat_template.jinja`` (a system and a user message)."""
    return "System: " + PROMPT + "<|endoftext|>" + "\n" + "Human: " + text + "<|endoftext|>" + "\nAssistant:"


class QwenEmotion:
    def __init__(self, folder: Path, device: torch.device) -> None:
        from mizuoto.runtime import ops
        from mizuoto.runtime.lm.qwen import QwenConfig, QwenDecoder
        from mizuoto.runtime.loading import load_into, load_state_dict, strip_prefix
        from mizuoto.text.tokenize.bpe import load_bpe

        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if cfg.get("model_type") != "qwen3":
            raise ValueError(f"The emotion model is a {cfg.get('model_type')}, not Qwen3")
        c = QwenConfig(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            qk_norm=True,
            sdpa_gqa=False,  # transformers 4.52 repeats the KV heads
        )
        self.decoder = QwenDecoder(c, ops.default)
        self.embed = ops.default.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        state = load_state_dict(folder / "model.safetensors")
        load_into(self.decoder, strip_prefix(state, "model.", only=True), ignore=["embed_tokens.*"], what="emotion model")
        load_into(self.embed, {"weight": state["model.embed_tokens.weight"]}, what="emotion model embeddings")
        self.tied = bool(cfg.get("tie_word_embeddings", True))
        self.lm_head = None if self.tied else state["lm_head.weight"]
        # float16 on the CPU; the model manager places it
        self.decoder.to(dtype=torch.float16).eval()
        self.embed.to(dtype=torch.float16).eval()
        self.eos = int(cfg.get("eos_token_id", 151643))
        self.tokenizer = load_bpe(folder)
        self.device = device

    def modules(self) -> list[torch.nn.Module]:
        return [self.decoder, self.embed]

    def generate(self, text: str, max_new_tokens: int = 32768, check: Any = None) -> list[int]:
        from mizuoto.runtime.lm.cache import DynamicCache

        ids = torch.tensor([self.tokenizer.encode(chat_prompt(text))], device=self.device)
        cache = DynamicCache()
        out: list[int] = []
        x = ids
        past = 0
        for _ in range(max_new_tokens):
            if check is not None:
                check()
            positions = torch.arange(past, past + x.shape[1], device=self.device)[None]
            hidden = self.decoder(self.embed(x), positions, None, cache)
            past += x.shape[1]
            weight = self.embed.weight if self.lm_head is None else self.lm_head.to(self.device, torch.float16)
            logits = torch.nn.functional.linear(hidden[:, -1:, :], weight)[:, -1, :].to(copy=True, dtype=torch.float32)
            token = int(torch.argmax(logits, dim=-1).item())
            out.append(token)
            if token == self.eos:
                break
            x = torch.tensor([[token]], device=self.device)
        return out

    @staticmethod
    def _label(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if value in CN_TO_EN:
            return value
        lower = value.lower()
        for cn, en in CN_TO_EN.items():
            if lower == en:
                return cn
        return None

    def normalize_content(self, content: Any) -> dict[str, Any]:
        normalized = dict(content) if isinstance(content, dict) else {}
        detected = self._label(content) if isinstance(content, str) else None
        if detected is None:
            for alias in ("emotion", "emotion_label", "label", "情感", "情绪"):
                detected = self._label(normalized.get(alias))
                if detected is not None:
                    break
        if detected is not None and all(key not in normalized for key in ORDER):
            normalized[detected] = 1.0
        for cn in ORDER:
            found = self._label(normalized.get(cn))
            if found is not None:
                normalized[cn] = 1.0 if found == cn else 0.0
                if found != cn:
                    normalized[found] = 1.0
        return normalized

    @staticmethod
    def clamp_score(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The emotion model returned a non-numeric score {value!r}; retry the request") from exc
        return max(MIN_SCORE, min(MAX_SCORE, value))

    def convert(self, content: Any) -> dict[str, float]:
        content = self.normalize_content(content)
        scores = {CN_TO_EN[cn]: self.clamp_score(content.get(cn, 0.0)) for cn in ORDER}
        if all(v <= 0.0 for v in scores.values()):
            scores["calm"] = 1.0
        return scores

    def inference(self, text: str, check: Any = None) -> dict[str, float]:
        out = self.generate(text, check=check)
        try:
            index = len(out) - out[::-1].index(THINK_END)
        except ValueError:
            index = 0
        answer = self.tokenizer.decode(out[index:], skip_special_tokens=True)
        try:
            content: Any = json.loads(answer)
        except json.decoder.JSONDecodeError:
            content = {m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', answer)}
        if any(word in text.lower() for word in MELANCHOLIC_WORDS):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
        return self.convert(content)
