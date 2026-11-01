"""Drama 3 as the ``drama3`` variant of the Fish Speech engine.

What upstream's ``TTSInferenceEngine`` and ``ModelManager`` do for a Drama 3 checkpoint, without the daemon thread:
- references go through S2's codec (Drama 3 ships the same ``codec.pth``) as for S2 Pro, their codes cached with the
  voice; without a reference Drama 3 speaks in a voice of its own;
- the request's seed seeds the generator every draw comes from (upstream seeds the global RNG);
- text is generated chunk by chunk (``generate.generate_long``), each chunk's codes decoded by the codec under autocast in
  the model's precision, the codec's weight norm folded in float32 before its cast (upstream's ``load_codec_model``);
  streaming sends each chunk's audio as it is done, and the whole is the chunks' audio joined (no clipping).

Upstream runs Drama 3 only on NVIDIA GPUs, in bfloat16 (float16 before SM80). The model's code runs anywhere, so here
it runs on every device, in float32 on the CPU (``_dtype``, as S2 Pro): the reference the port is checked against on the
CPU is upstream's modules run there in float32.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import AudioChunk, EngineError, LoadSpec
from mizuoto_fishspeech.engine import SAMPLE_RATE, LoadedFishSpeech, _dtype

if TYPE_CHECKING:
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask
    from mizuoto_fishspeech.drama3.tower import Fallback

SAMPLERS: dict[str, Fallback | None] = {"default": None, "reference_argmax": "argmax", "reference_ras": "ras"}


class LoadedDrama3(LoadedFishSpeech):
    """S2 Pro's codec, references and operations (``LoadedFishSpeech``) with Drama 3's backbone and tower."""

    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext) -> None:
        import torch

        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.runtime import ops
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into, load_state_dict, unwrap
        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto_fishspeech.codec import IGNORED, FishCodec, codec_state
        from mizuoto_fishspeech.drama3.backbone import TOWER_PREFIXES, Backbone, checkpoint_tensors, load_backbone
        from mizuoto_fishspeech.drama3.config import Drama3Config
        from mizuoto_fishspeech.drama3.tower import CodebookTower, load_tower
        from mizuoto_fishspeech.engine import codec_config

        super(LoadedFishSpeech, self).__init__(spec)
        folder = spec.model_dir
        assert folder is not None
        self.device = torch.device(spec.device)
        self.dtype = _dtype(spec.dtype, spec.device)
        self.config = Drama3Config.from_dict(json.loads((folder / "config.json").read_text(encoding="utf-8")))
        self.tokenizer = load_bpe(folder)

        ctx.progress("loading:backbone", None)
        backbone = Backbone(self.config.text, self.config.fp8_experts, ops.default)
        # dense weights in the model's dtype; FP8 experts (and their float32 scales) stay as stored
        for name, p in backbone.named_parameters():
            if ".experts." not in name or not self.config.fp8_experts:
                p.data = p.data.to(self.dtype)
        load_backbone(backbone, checkpoint_tensors(folder, skip=TOWER_PREFIXES))
        self.backbone = backbone.eval()

        ctx.check_cancel()
        ctx.progress("loading:tower", None)
        tower = CodebookTower(self.config, ops.default).to(dtype=self.dtype)
        load_tower(tower, checkpoint_tensors(folder, skip=("model.", "lm_head.")))
        self.tower = tower.eval()

        ctx.check_cancel()
        ctx.progress("loading:codec", None)
        codec = FishCodec(codec_config(folder), ops.default)
        load_into(codec, codec_state(unwrap(load_state_dict(folder / "codec.pth"), "state_dict")), ignore=IGNORED, what="Fish Speech codec")
        self.codec = codec.fold_weight_norm().to(dtype=self.dtype).eval()
        self.resamplers: dict[int, Any] = {}
        self._resample_cls = Resample

        # the backbone runs once per frame: its attention layers and output head first, then whole layers' experts as
        # memory allows; the rest of the experts stream by the row (only those a token is routed to)
        self.h_backbone = ModelHandle(backbone, name=f"{spec.model_id}:backbone", load_device=spec.device, kind="sequence")
        for layer in backbone.model.layers:
            self.h_backbone.mark(layer.mixer, "hot")
        self.h_backbone.mark(backbone.lm_head, "hot")
        self.h_tower = ModelHandle(tower, name=f"{spec.model_id}:tower", load_device=spec.device, kind="sequence")
        self.h_codec = ModelHandle(codec, name=f"{spec.model_id}:codec", load_device=spec.device)
        self.handles = [self.h_backbone, self.h_tower, self.h_codec]
        self.cuda_graphs = False
        self._decoder = None
        self._graphs = None
        self._graph_signature = None

    # -- memory -------------------------------------------------------------------------

    def memory_required(self, task: Any) -> int:
        from mizuoto_fishspeech.drama3.generate import capacity_for

        return self._generation_memory(capacity_for(self.config))

    def _generation_memory(self, positions: int) -> int:
        """A request's caches for ``positions`` rows (the backbone's softmax layers' keys and values, every DeltaNet
        layer's state, the tower's keys and values) and working memory: a prefill block's activations, the logits, and
        the experts a step dequantizes."""
        t, w = self.config.text, self.config.tower
        element = self.tower.codebook_embeddings.weight.element_size()
        full = sum(1 for kind in t.layer_types if kind == "full_attention")
        linear = t.num_hidden_layers - full
        attention = full * 2 * t.num_key_value_heads * t.head_dim * positions * element
        conv_dim = 2 * t.linear_num_key_heads * t.linear_key_head_dim + t.linear_num_value_heads * t.linear_value_head_dim
        delta = linear * (conv_dim * t.linear_conv_kernel_dim * element + t.linear_num_value_heads * t.linear_key_head_dim * t.linear_value_head_dim * 4)
        tower = (w.shared_enrichment_layers + w.num_codebooks - 1) * 2 * w.n_local_heads * w.head_dim * positions * element
        experts = self.backbone.model.layers[0].mlp.experts
        dequantized = 64 * (3 * experts.intermediate * experts.hidden) * max(element, 4)
        return 768 * 2**20 + attention + delta + tower + t.vocab_size * 4 * 4 + dequantized

    # -- speech -------------------------------------------------------------------------

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np
        import torch

        from mizuoto_fishspeech.drama3.generate import NonFiniteLogits, Sampling, capacity_for, generate_long
        from mizuoto_fishspeech.drama3.prompt import PromptError

        prompt_codes, prompt_texts = self._reference_codes(task.voice, ctx) if task.voice.references else ([], [])
        p, o = task.params, task.options
        sampler = str(o.get("sampler") or "default")
        if sampler not in SAMPLERS:
            raise EngineError(f"Unknown sampler {sampler!r}", {"choices": sorted(SAMPLERS)}, code=codes.INVALID_INPUT)
        sampling = Sampling(fallback=SAMPLERS[sampler])
        if p.get("temperature") is not None:
            sampling.temperature = float(p["temperature"])
        if p.get("top_p") is not None:
            sampling.top_p = float(p["top_p"])
        if p.get("top_k") is not None:
            sampling.top_k = int(p["top_k"])
        if p.get("max_new_tokens") is not None:
            sampling.max_new_tokens = int(p["max_new_tokens"])
        if o.get("epsilon_cutoff") is not None:
            sampling.epsilon_cutoff = float(o["epsilon_cutoff"])
        single_chunk = bool(o.get("single_chunk") or False)
        ctx.applied.update({"streaming": bool(task.stream), "references": len(prompt_codes), "sampler": sampler})
        ctx.applied.update({"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k, "max_new_tokens": sampling.max_new_tokens})
        if p.get("seed") is not None:
            ctx.applied["seed"] = int(p["seed"])
        if single_chunk:
            ctx.applied["single_chunk"] = True
        if sampling.fallback is not None:
            ctx.applied["epsilon_cutoff"] = sampling.epsilon_cutoff if sampling.epsilon_cutoff is not None else 0.001

        def progress(chunk: int, chunks: int, done: int, total: int) -> None:
            if done % 16 == 0:
                ctx.progress("generating", (chunk + done / max(total, 1)) / chunks)

        memory = self._generation_memory(capacity_for(self.config))
        segments = []
        with torch.inference_mode():
            pieces = generate_long(
                self.backbone,
                self.tower,
                self.config,
                self.tokenizer,
                task.text,
                prompt_texts,
                prompt_codes,
                sampling,
                single_chunk,
                ctx.generator_for(self.device),
                ctx.check_cancel,
                progress,
                around=lambda run: ctx.run("generating", [self.h_backbone, self.h_tower], run, memory),
            )
            try:
                for _chunk, frames in pieces:
                    ctx.check_cancel()
                    audio = self._decode(frames, ctx)
                    if task.stream:
                        yield AudioChunk(samples=audio.astype(np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=False)
                    segments.append(audio)
            except PromptError as e:
                raise EngineError(str(e), code=codes.INVALID_INPUT) from e
            except NonFiniteLogits as e:
                raise EngineError(str(e), code=codes.ENGINE_ERROR) from e
        ctx.check_cancel()
        if not segments:
            raise EngineError("Drama 3 produced no speech for the text", {"text": task.text[:200]}, code=codes.ENGINE_ERROR)
        if task.stream:
            yield AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)
        else:
            yield AudioChunk(samples=np.concatenate(segments).astype(np.float32), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)

    def _graphs_now(self) -> Any:
        return None
