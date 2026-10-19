"""CosyVoice 1, 2 and 3, native: preset speakers, instructions, cloning,
cross-lingual cloning, CosyVoice 2/3's instructions with a clip, voice conversion, streamed audio and text.

The port follows Mizuoto's v1 adapter over upstream ``CosyVoice``/``CosyVoice2``/``CosyVoice3`` step for step:
- the frontend: wetext normalisation and splitting (``frontend.py``); speech tokens from the 16 kHz clip
  (``nn.codecs.s3tokenizer``, the ONNX file's weights), the CAM++ embedding (``nn.speaker.campplus``) and
  the 24 kHz prompt mel (Matcha's); CosyVoice 2/3 keep exactly two mel frames per prompt token;
- the language model, the flow and HiFT as ``llm.py``, ``flow.py`` and ``nn.vocoders.hift`` port them;
- ``CosyVoiceModel.tts``: the tokens (CosyVoice 3 drops runs of more than five silent tokens), then the mel
  and the waveform; when streaming, upstream's chunking (v1: 100 tokens with a 20-token overlap and Hamming
  cross-fades; v2/v3: 25 tokens plus the prompt's padding, doubling up to 100, three tokens of lookahead).

Upstream's language model runs on a thread beside the vocoder; here it runs in the request's thread and
the chunks are made as its tokens arrive. Without streaming, the random numbers are drawn in upstream's
order (language model, flow noise for v1, vocoder noise), so a seeded request repeats upstream's audio.
Streamed, upstream's v1 and v2 draw vocoder noise concurrently with the language model (their output
depends on timing); here the vocoder draws from its own generator, so the tokens are the unstreamed ones.
TensorRT and vLLM (v1 load options) are not available natively; ``fp16`` is CUDA autocast as upstream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    import torch

    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SynthesisTask, VoiceConversionTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

V3_SYSTEM_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
CROSS_LINGUAL_TAGS = {"zh": "<|zh|>", "en": "<|en|>", "ja": "<|jp|>", "yue": "<|yue|>", "ko": "<|ko|>"}
"""CosyVoice 1's language tags for a clip without its transcript."""
SILENT_TOKENS = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
"""CosyVoice 3's FSQ silence and breath tokens."""
WETEXT_ASSET = "cosyvoice/wetext"


def detect(model_dir: Path) -> str | None:
    if (model_dir / "cosyvoice3.yaml").is_file():
        return "v3"
    if (model_dir / "cosyvoice2.yaml").is_file():
        return "v2"
    if (model_dir / "cosyvoice.yaml").is_file():
        if not (model_dir / "spk2info.pt").is_file():
            return "v1_base"
        return "v1_instruct" if "instruct" in model_dir.name.lower() else "v1_sft"
    return None


class CosyVoiceEngine(Engine):
    id = "cosyvoice"
    DEVICES = DEVICE_KINDS  # CosyVoice 3's float64 F0 predictor runs on the CPU where the device has no float64
    VERSION = "1"
    OPTION_SCHEMA = {"system_prompt": OptionSpec("str", V3_SYSTEM_PROMPT)}
    TRAINER = "mizuoto.engines.cosyvoice.train:CosyVoiceTrainer"

    def detect_variant(self, model_dir: Path) -> str | None:
        return detect(model_dir)

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        if spec.model_dir is None or detect(spec.model_dir) is None:
            raise EngineError(f"{spec.model_id} is not a CosyVoice model", code=codes.INVALID_INPUT)
        loaded = LoadedCosyVoice(spec, ctx, compile=runtime.options.compile, float64_device=runtime.devices.float64_device(spec.device))
        for m in loaded.llm.modules():
            if hasattr(type(m), "kv_cache"):
                m.kv_cache = runtime.options.kv_cache
        loaded.cuda_graphs = runtime.options.cuda_graphs
        if hasattr(loaded.llm, "graphs_for"):
            loaded.llm.graphs_for = loaded._graphs_now
        return loaded


CONVERTER_VERSION = 1
"""Bumped when an ONNX conversion changes, so cached conversions are made again."""


def hift_kwargs(node: dict[str, Any]) -> dict[str, Any]:
    """``HiFTGenerator`` arguments from the yaml's ``hift`` node."""
    from mizuoto.engines.cosyvoice.config import kwargs

    out = kwargs(
        node,
        "in_channels",
        "base_channels",
        "nb_harmonics",
        "sampling_rate",
        "nsf_alpha",
        "nsf_sigma",
        "nsf_voiced_threshold",
        "upsample_rates",
        "upsample_kernel_sizes",
        "istft_params",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "source_resblock_kernel_sizes",
        "source_resblock_dilation_sizes",
        "lrelu_slope",
        "audio_limit",
        "conv_pre_look_right",
    )
    out["causal"] = "Causal" in str(node.get("_class", ""))
    return out


@dataclass
class _Session:
    """``CosyVoiceModel``'s per-request dictionaries."""

    hift_cache: dict[str, Any] | None = None
    mel_overlap: Any = None
    flow_cache: Any = None


class LoadedCosyVoice(LoadedEngine):
    PHASED = True

    def __init__(self, spec: LoadSpec, ctx: RequestContext, compile: bool = False, float64_device: str | None = None) -> None:
        super().__init__(spec)
        import torch

        from mizuoto.engines.cosyvoice.config import read_config
        from mizuoto.engines.cosyvoice.flow import Flow
        from mizuoto.engines.cosyvoice.frontend import TextFrontend
        from mizuoto.engines.cosyvoice.llm import QwenLM, TransformerLM
        from mizuoto.engines.cosyvoice.tokenizer import text_tokenizer
        from mizuoto.nn.codecs.s3tokenizer import S3Tokenizer
        from mizuoto.nn.codecs.s3tokenizer import state_dict_from_onnx as tokenizer_state
        from mizuoto.nn.dsp.mel import matcha_basis
        from mizuoto.nn.speaker.campplus import CAMPPlus
        from mizuoto.nn.speaker.campplus import state_dict_from_onnx as campplus_state
        from mizuoto.nn.vocoders.hift import HiFTGenerator, cosyvoice3_noise
        from mizuoto.runtime import ops
        from mizuoto.runtime.compile import compile_module
        from mizuoto.runtime.device import pin_to_cpu
        from mizuoto.runtime.handle import ModelHandle
        from mizuoto.runtime.loading import load_into

        folder = spec.model_dir
        assert folder is not None
        self.variant = spec.variant
        self.version = 3 if spec.variant == "v3" else 2 if spec.variant == "v2" else 1
        v = self.version
        self.device = torch.device(spec.device)
        self.fp16 = bool(spec.options.get("fp16")) and self.device.type == "cuda"
        self.config = config = read_config(folder)
        self.sample_rate = int(config["sample_rate"])

        ctx.progress("loading:text", None)
        self.tokenizer = text_tokenizer(folder, v)
        fst = (spec.options.get("assets") or {}).get(WETEXT_ASSET)
        self.frontend = TextFrontend(self.tokenizer, Path(fst) if fst else None)
        self.spk2info: dict[str, Any] = torch.load(str(folder / "spk2info.pt"), map_location="cpu", weights_only=True) if (folder / "spk2info.pt").is_file() else {}

        def checkpoint(name: str) -> dict[str, torch.Tensor]:
            return torch.load(str(folder / name), map_location="cpu", weights_only=True, mmap=True)

        ctx.check_cancel()
        ctx.progress("loading:language model", None)
        llm_cfg = config["llm"]
        self.llm: Any = TransformerLM(llm_cfg, ops.default) if v == 1 else QwenLM(v, folder / "CosyVoice-BlankEN", int(llm_cfg.get("speech_token_size", 6561)), ops.default)
        load_into(self.llm, checkpoint("llm.pt"), ignore=self.llm.ignorable(), what="CosyVoice language model")
        self.llm.eval()

        ctx.check_cancel()
        ctx.progress("loading:flow", None)
        self.flow = Flow(v, config["flow"], ops.default)
        load_into(self.flow, checkpoint("flow.pt"), what="CosyVoice flow")
        self.flow.eval()
        # compute.compile: the flow decoder's estimator (U-Net or DiT), run twice per Euler step, its length dynamic
        self.flow.decoder.compiled_estimator = compile_module(self.flow.decoder.estimator, compile, f"{spec.model_id}:flow estimator")

        ctx.check_cancel()
        ctx.progress("loading:vocoder", None)
        hift_cfg = config["hift"]
        self.hift = HiFTGenerator(**hift_kwargs(hift_cfg), ops=ops.default)
        if self.hift.causal:
            f0 = hift_cfg.get("f0_predictor", {})
            flow_cfg = config["flow"]
            self.hift.f0_predictor.double()
            self.hift.m_source.causal_noise = cosyvoice3_noise(
                int(flow_cfg["vocab_size"]),
                int(flow_cfg["input_size"]),
                int(flow_cfg.get("spk_embed_dim", 192)),
                int(flow_cfg.get("output_size", 80)),
                int(f0.get("in_channels", 80)),
                int(f0.get("cond_channels", 512)),
                int(hift_cfg.get("nb_harmonics", 8)) + 1,
            )
        f0_device = "cpu" if self.hift.causal and float64_device == "cpu" else spec.device
        load_into(self.hift, self.hift.fold(checkpoint("hift.pt"), device=spec.device, f0_device=f0_device), what="CosyVoice vocoder")
        self.hift.eval()
        if self.hift.causal and float64_device == "cpu" and self.device.type != "cpu":
            pin_to_cpu(self.hift.f0_predictor)  # float64 on a device without it (MPS): the F0 predictor stays on the CPU

        ctx.check_cancel()
        ctx.progress("loading:speech tokenizer", None)
        tokenizer_sd = self._onnx_weights(folder, f"speech_tokenizer_v{v}.onnx", lambda g: tokenizer_state(g, v))
        self.speech_tokenizer = S3Tokenizer.from_state(v, tokenizer_sd, ops.default)
        load_into(self.speech_tokenizer, tokenizer_sd, what="speech tokenizer")
        self.speech_tokenizer.eval()
        self.campplus = CAMPPlus(ops=ops.default)
        load_into(self.campplus, self._onnx_weights(folder, "campplus.onnx", campplus_state), ignore=["*.num_batches_tracked"], what="CAM++")
        # On the CPU wherever the model runs: upstream runs campplus.onnx with onnxruntime's CPU provider only, and a
        # GPU's convolutions give the embedding (and so the whole voice) other bits.
        self.campplus.eval()

        feat = config["feat_extractor"]
        self.feat_params: dict[str, Any] = {k: feat[k] for k in ("n_fft", "num_mels", "sampling_rate", "hop_size", "win_size", "fmin", "fmax", "center")}
        self.feat_basis = matcha_basis(feat["sampling_rate"], feat["n_fft"], feat["num_mels"], feat["fmin"], feat["fmax"])
        # Built on the CPU; the model manager places them.
        # phases: the frontend makes a voice prompt (uncached voices, conversion), the language
        # model makes speech tokens, the flow and the vocoder speak them (together with the LM when streaming)
        self.h_llm = ModelHandle(self.llm, name=f"{spec.model_id}:language model", load_device=spec.device, kind="sequence")
        self.h_flow = ModelHandle(torch.nn.ModuleList([self.flow, self.hift]), name=f"{spec.model_id}:flow and vocoder", load_device=spec.device)
        self.h_flow.mark(self.hift.f0_predictor, "fixed")  # HiFT reads its first layer's device to find where it runs
        self.h_frontend = ModelHandle(self.speech_tokenizer, name=f"{spec.model_id}:frontend", load_device=spec.device)
        self.handles = [self.h_llm, self.h_flow, self.h_frontend]
        self.cuda_graphs = False
        self._graphs: Any = None
        self._graph_signature: tuple[int, ...] | None = None

    def _graphs_now(self) -> Any:
        """The language model's CUDA graphs (``compute.cuda_graphs``; CosyVoice 2/3) while it is wholly on an NVIDIA GPU:
        new ones when its weights moved since the last capture, none otherwise."""
        from mizuoto.runtime.graphs import Graphs, signature, usable

        if not usable(self.device, self.cuda_graphs) or not self.h_llm.loaded:
            return None
        where = signature([self.llm])
        if where != self._graph_signature:
            self._graphs, self._graph_signature = Graphs(self.device), where
        return self._graphs

    def close(self) -> None:
        self._graphs = None
        if hasattr(self.llm, "steps"):
            self.llm.steps.graphs = None
            self.llm.steps._reset()
        super().close()

    @staticmethod
    def _onnx_weights(folder: Path, name: str, convert: Any) -> dict[str, torch.Tensor]:
        """A model shipped as ONNX, as torch weights: ``<name>.safetensors`` beside it when there is one, else the
        conversion cached in ``<model>/.mizuoto/`` (keyed by the file's checksum or size and time), else converted now."""
        import hashlib
        import json
        import os

        from safetensors.torch import save_file

        from mizuoto.runtime.convert.onnx import read_onnx
        from mizuoto.runtime.loading import load_state_dict

        given = folder / (Path(name).stem + ".safetensors")
        if given.is_file():
            return load_state_dict(given)
        source = folder / name
        identity = ""
        try:
            listed = json.loads((folder / "model.json").read_text(encoding="utf-8")).get("files") or []
            identity = next((f.get("sha256") or "" for f in listed if f.get("path") == name), "")
        except (OSError, ValueError):
            pass
        if not identity:
            st = source.stat()
            identity = f"{st.st_size}:{st.st_mtime_ns}"
        key = hashlib.sha256(f"{name}:{identity}:{CONVERTER_VERSION}".encode()).hexdigest()[:20]
        cached = folder / ".mizuoto" / f"converted-{key}.safetensors"
        if cached.is_file():
            return load_state_dict(cached)
        graph = read_onnx(source)
        try:
            state = convert(graph)
        finally:
            graph.close()
        try:
            cached.parent.mkdir(exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
            os.replace(tmp, cached)
        except OSError:
            pass  # a read-only model folder: convert on every load
        return state

    # -- description ------------------------------------------------------------------

    def describe(self) -> Description:
        speakers = sorted(self.spk2info) if self.variant in ("v1_sft", "v1_instruct") else []
        size = sum(h.size_bytes for h in self.handles) / 2**20
        return Description(sample_rate=self.sample_rate, speakers=speakers, vram_mb=round(size, 1) if self.device.type != "cpu" else 0.0)

    def memory_required(self, task: Any) -> int:
        return 1536 * 2**20

    def _llm_memory(self, text_tokens: int) -> int:
        """The language model's cache for up to 20 speech tokens per text token (upstream's ``max_token_text_ratio``) after a
        prompt of up to 300 positions, grown by concatenation; one layer's keys and values repeated per head group
        (transformers 4.51's SDPA, ``sdpa_gqa=False``); and working memory."""
        from mizuoto.runtime.lm.cache import peak_bytes

        positions = 300 + 21 * max(text_tokens, 1)
        lm = self.llm.llm
        keys = [m for n, m in lm.named_modules() if n.endswith(("k_proj", "linear_k"))]
        element = next(lm.parameters()).element_size()
        width, hidden = int(keys[0].out_features), int(keys[0].in_features)
        repeated = 2 * hidden * positions * element
        return peak_bytes(2 * width * element, len(keys), positions, getattr(lm, "kv_cache", "dynamic")) + repeated + 512 * 2**20

    def _flow_memory(self, tokens: int) -> int:
        """The flow's activations for ``tokens`` speech tokens (conditional and unconditional together) and HiFT's."""
        frames = (tokens + 300) * 2
        samples = frames * 480
        return 2 * frames * 1024 * 4 * 16 + samples * 4 * 64 + 256 * 2**20

    def _frontend_memory(self) -> int:
        return 30 * 16000 * 4 * 64 + 256 * 2**20

    # -- frontend: audio --------------------------------------------------------------

    def _load_wav(self, path: str, rate: int) -> torch.Tensor:
        """``load_wav``: mono (channels averaged), resampled with torchaudio's ``Resample``; (1, samples)."""
        import torch

        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        samples, sr = read_wav(path)
        speech = torch.from_numpy(samples).unsqueeze(0)
        if sr != rate:
            if sr < 16000:
                raise EngineError(f"The clip's sample rate ({sr} Hz) must be at least 16000 Hz", code=codes.INVALID_INPUT)
            speech = Resample(sr, rate)(speech)
        return speech

    def _speech_token(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        from mizuoto.nn.dsp.mel import whisper_log_mel

        speech = self._load_wav(path, 16000)
        if speech.shape[1] / 16000 > 30:
            raise EngineError("CosyVoice extracts speech tokens from at most 30 s of audio", code=codes.INVALID_INPUT)
        feat = whisper_log_mel(speech, n_mels=128).to(self.device)
        codes_, lens = self.speech_tokenizer.quantize(feat, torch.tensor([feat.shape[2]], dtype=torch.int32, device=self.device))
        tokens = codes_[0, : int(lens[0])].flatten().tolist()
        token = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        return token, torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device)

    def _spk_embedding(self, path: str) -> torch.Tensor:
        import torch

        from mizuoto.nn.dsp.kaldi import fbank

        feat = fbank(self._load_wav(path, 16000), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus(feat.unsqueeze(dim=0)).flatten().tolist()
        return torch.tensor([embedding]).to(self.device)

    def _speech_feat(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        from mizuoto.nn.dsp.mel import matcha_mel

        speech = self._load_wav(path, 24000)
        feat = matcha_mel(speech, basis=self.feat_basis, **self.feat_params).squeeze(dim=0).transpose(0, 1).to(self.device).unsqueeze(dim=0)
        return feat, torch.tensor([feat.shape[1]], dtype=torch.int32, device=self.device)

    def _text_token(self, text: str) -> torch.Tensor:
        import torch

        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.int32, device=self.device)

    def _zero_shot(self, prompt_text: str, path: str, ctx: RequestContext) -> dict[str, Any]:
        """``frontend_zero_shot`` without the text to say (the voice's part)."""
        prompt = self._text_token(prompt_text)

        def frontend() -> tuple[torch.Tensor, ...]:
            feat, feat_len = self._speech_feat(path)
            token, token_len = self._speech_token(path)
            if self.sample_rate == 24000:
                n = min(int(feat.shape[1] / 2), token.shape[1])
                feat, feat_len[:] = feat[:, : 2 * n], 2 * n
                token, token_len[:] = token[:, :n], n
            return feat, feat_len, token, token_len, self._spk_embedding(path)

        feat, feat_len, token, token_len, embedding = ctx.run("frontend", [self.h_frontend], frontend, self._frontend_memory())
        return {
            "prompt_text": prompt,
            "prompt_text_len": torch_len(prompt),
            "llm_prompt_speech_token": token,
            "llm_prompt_speech_token_len": token_len,
            "flow_prompt_speech_token": token,
            "flow_prompt_speech_token_len": token_len,
            "prompt_speech_feat": feat,
            "prompt_speech_feat_len": feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }

    # -- voices -----------------------------------------------------------------------

    def _system(self, options: dict[str, Any]) -> str:
        if self.version != 3:
            return ""
        return f"{options.get('system_prompt') or V3_SYSTEM_PROMPT}{END_OF_PROMPT}"

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        import torch

        if not voice.references:
            return None
        ref = voice.references[0]
        prompt_text = (ref.transcript or "").strip()
        if prompt_text:
            prompt_text = self.frontend.normalize(self._system({}) + prompt_text, split=False)
        model_input = self._zero_shot(prompt_text, ref.audio_path, ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "speaker.pt"
        torch.save({k: v.detach().cpu() for k, v in model_input.items()}, path)
        return VoiceCache(files={"speaker": str(path)})

    def _cached(self, voice: VoiceSpec) -> dict[str, Any] | None:
        import torch

        path = (voice.cache or {}).get("speaker")
        if not path or not Path(path).is_file():
            return None
        return {k: v.to(self.device) for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}

    # -- the model --------------------------------------------------------------------

    def _tokens(self, model_input: dict[str, Any], text: Any, generator: torch.Generator, ctx: RequestContext) -> Iterator[int]:
        """``llm_job``: the language model's tokens (CosyVoice 3: runs of more than five silent tokens dropped)."""
        import torch

        empty = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
        prompt_text = model_input.get("prompt_text", empty)
        prompt_speech = model_input.get("llm_prompt_speech_token", empty)
        silent = 0
        if self.version == 1:
            embedding = model_input.get("llm_embedding", torch.zeros(0, 192, device=self.device))
            tokens = self.llm.inference(text, prompt_text, prompt_speech, embedding, generator, ctx.check_cancel)
        elif isinstance(text, torch.Tensor):
            tokens = self.llm.inference(text, prompt_text, prompt_speech, generator, ctx.check_cancel)
        else:
            tokens = self.llm.inference_bistream(text, prompt_text, prompt_speech, generator, ctx.check_cancel)
        while True:
            with self._autocast():
                token = next(tokens, None)
            if token is None:
                steps = getattr(self.llm, "steps", None)
                if steps is not None and steps.graphed and not steps.failed:
                    ctx.applied["cuda_graphs"] = True
                return
            if self.version == 3 and token in SILENT_TOKENS:
                silent += 1
                if silent > 5:
                    continue
            else:
                silent = 0
            yield token

    def _autocast(self) -> Any:
        """``torch.cuda.amp.autocast(fp16)`` as upstream wraps the language model and the flow (CUDA only)."""
        import torch

        return torch.autocast("cuda", enabled=self.fp16) if self.device.type == "cuda" else _null()

    def _token2wav(
        self,
        session: _Session,
        tokens: list[int],
        model_input: dict[str, Any],
        generator: torch.Generator,
        cpu: torch.Generator,
        finalize: bool,
        token_offset: int = 0,
        stream: bool = False,
        speed: float = 1.0,
    ) -> torch.Tensor:
        import torch
        from torch.nn import functional as F

        token = torch.tensor(tokens).unsqueeze(dim=0).to(self.device, dtype=torch.int32)
        empty_token = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
        prompt_token = model_input.get("flow_prompt_speech_token", empty_token)
        prompt_feat = model_input.get("prompt_speech_feat", torch.zeros(1, 0, 80, device=self.device))
        embedding = model_input.get("flow_embedding", torch.zeros(0, 192, device=self.device))
        autocast = self._autocast()
        if self.version == 1:
            with autocast:
                mel, session.flow_cache = self.flow.inference(token, prompt_token, prompt_feat, embedding, generator=generator, flow_cache=session.flow_cache)
            if session.mel_overlap is not None and session.mel_overlap.shape[2] != 0:
                mel = fade_in_out(mel, session.mel_overlap, MEL_WINDOW)
            if session.hift_cache is not None:
                mel = torch.concat([session.hift_cache["mel"], mel], dim=2)
                cache_source = session.hift_cache["source"]
            else:
                cache_source = torch.zeros(1, 1, 0, device=self.device)
            if not finalize:
                session.mel_overlap = mel[:, :, -V1_MEL_OVERLAP:]
                mel = mel[:, :, :-V1_MEL_OVERLAP]
                speech, source = self.hift.inference(mel, cpu, generator, cache_source=cache_source)
                if session.hift_cache is not None:
                    speech = fade_in_out(speech, session.hift_cache["speech"], V1_SPEECH_WINDOW)
                session.hift_cache = {"mel": mel[:, :, -20:], "source": source[:, :, -V1_SOURCE_CACHE:], "speech": speech[:, -V1_SOURCE_CACHE:]}
                return speech[:, :-V1_SOURCE_CACHE]
            if speed != 1.0:
                mel = F.interpolate(mel, size=int(mel.shape[2] / speed), mode="linear")
            speech, _ = self.hift.inference(mel, cpu, generator, cache_source=cache_source)
            if session.hift_cache is not None:
                speech = fade_in_out(speech, session.hift_cache["speech"], V1_SPEECH_WINDOW)
            return speech
        with autocast:
            mel, _ = self.flow.inference(token, prompt_token, prompt_feat, embedding, streaming=stream, finalize=finalize)
        mel = mel[:, :, token_offset * self.flow.token_mel_ratio :]
        if self.version == 2:
            if session.hift_cache is not None:
                mel = torch.concat([session.hift_cache["mel"], mel], dim=2)
                cache_source = session.hift_cache["source"]
            else:
                cache_source = torch.zeros(1, 1, 0, device=self.device)
            if not finalize:
                speech, source = self.hift.inference(mel, cpu, generator, cache_source=cache_source)
                if session.hift_cache is not None:
                    speech = fade_in_out(speech, session.hift_cache["speech"], V2_SPEECH_WINDOW)
                session.hift_cache = {"mel": mel[:, :, -8:], "source": source[:, :, -V2_SOURCE_CACHE:], "speech": speech[:, -V2_SOURCE_CACHE:]}
                return speech[:, :-V2_SOURCE_CACHE]
            if speed != 1.0:
                mel = F.interpolate(mel, size=int(mel.shape[2] / speed), mode="linear")
            speech, _ = self.hift.inference(mel, cpu, generator, cache_source=cache_source)
            if session.hift_cache is not None:
                speech = fade_in_out(speech, session.hift_cache["speech"], V2_SPEECH_WINDOW)
            return speech
        with autocast:
            if session.hift_cache is not None:
                mel = torch.concat([session.hift_cache["mel"], mel], dim=2)
                session.hift_cache["mel"] = mel
            else:
                session.hift_cache = {"mel": mel, "speech_offset": 0}
            if speed != 1.0:
                mel = F.interpolate(mel, size=int(mel.shape[2] / speed), mode="linear")
            speech, _ = self.hift.inference(mel, cpu, generator, finalize=finalize)
            speech = speech[:, session.hift_cache["speech_offset"] :]
            session.hift_cache["speech_offset"] += speech.shape[1]
        return speech

    def _tts(self, model_input: dict[str, Any], text: Any, ctx: RequestContext, stream: bool, speed: float, source_tokens: list[int] | None = None) -> Iterator[torch.Tensor]:
        """``CosyVoiceModel.tts``, synchronously: the waveform pieces (1, samples) of one text."""
        import torch

        generator = ctx.generator_for(self.device)
        cpu = ctx.generator_for("cpu")
        session = _Session()
        if stream and self.version < 3:
            # Upstream draws the vocoder's (and v1's flow) noise on another thread than the language model, so its
            # streamed audio depends on timing. Here the vocoder gets its own generators (the seed + 1): the tokens
            # stay those of the same request unstreamed, and the audio repeats under a seed.
            seed = ctx.seed if ctx.seed is not None else int(torch.randint(0, 2**62, (1,), generator=torch.Generator().manual_seed(generator.initial_seed())).item())
            vocoder = torch.Generator(device=self.device).manual_seed(seed + 1)
            vocoder_cpu = vocoder if self.device.type == "cpu" else torch.Generator().manual_seed(seed + 1)
        else:
            vocoder, vocoder_cpu = generator, cpu
        token_iter: Iterator[int] = iter(source_tokens) if source_tokens is not None else self._tokens(model_input, text, generator, ctx)
        tokens: list[int] = []
        text_tokens = int(text.shape[-1]) if isinstance(text, torch.Tensor) else 64
        if not stream:
            if source_tokens is not None or isinstance(text, torch.Tensor):

                def lm() -> list[int]:
                    return list(source_tokens) if source_tokens is not None else list(self._tokens(model_input, text, generator, ctx))

                tokens = ctx.run("language model", [] if source_tokens is not None else [self.h_llm], lm, self._llm_memory(text_tokens))
            else:  # text as it comes: read once, so not run again
                with ctx.use([self.h_llm], self._llm_memory(text_tokens)):
                    tokens = list(token_iter)
            ctx.check_cancel()
            yield ctx.run(
                "flow", [self.h_flow], lambda: self._token2wav(_Session(), tokens, model_input, generator, cpu, finalize=True, speed=speed), self._flow_memory(len(tokens))
            )
            return
        with ctx.use([self.h_llm, self.h_flow], self._llm_memory(text_tokens) + self._flow_memory(200)):
            yield from self._streamed(session, token_iter, tokens, model_input, vocoder, vocoder_cpu, ctx)

    def _streamed(
        self,
        session: _Session,
        token_iter: Iterator[int],
        tokens: list[int],
        model_input: dict[str, Any],
        vocoder: torch.Generator,
        vocoder_cpu: torch.Generator,
        ctx: RequestContext,
    ) -> Iterator[torch.Tensor]:
        """``tts(stream=True)``: the speech of the tokens so far, as they come."""
        import numpy as np

        if self.version == 1:
            hop, overlap = 2 * 50, 20
            for t in token_iter:
                tokens.append(t)
                if len(tokens) >= hop + overlap:
                    yield self._token2wav(session, tokens[: hop + overlap], model_input, vocoder, vocoder_cpu, finalize=False)
                    tokens = tokens[hop:]
                    hop = min(4 * 50, int(hop * 1))
            ctx.check_cancel()
            yield self._token2wav(session, tokens, model_input, vocoder, vocoder_cpu, finalize=True)
            return
        hop_len, max_hop, lookahead = 25, 100, self.flow.pre_lookahead_len
        prompt_tokens = model_input.get("flow_prompt_speech_token")
        n_prompt = prompt_tokens.shape[1] if prompt_tokens is not None else 0
        prompt_pad = int(np.ceil(n_prompt / hop_len) * hop_len - n_prompt)
        offset = 0
        for t in token_iter:
            tokens.append(t)
            this_hop = hop_len + prompt_pad if offset == 0 else hop_len
            if len(tokens) - offset >= this_hop + lookahead:
                yield self._token2wav(session, tokens[: offset + this_hop + lookahead], model_input, vocoder, vocoder_cpu, finalize=False, token_offset=offset, stream=True)
                offset += this_hop
                hop_len = min(max_hop, hop_len * 2)
        ctx.check_cancel()
        yield self._token2wav(session, tokens, model_input, vocoder, vocoder_cpu, finalize=True, token_offset=offset)

    # -- requests ---------------------------------------------------------------------

    def _model_input(self, task: SynthesisTask, piece: str, cached: dict[str, Any] | None, normalize: bool, ctx: RequestContext) -> dict[str, Any]:
        """The frontend call for the task's voice mode, as Mizuoto's v1 adapter makes it."""
        voice = task.voice
        instruct = (task.style.instruct or "").strip()
        system = self._system(task.options)
        text = self._text_token(piece)
        if voice.preset:
            if voice.preset not in self.spk2info:
                raise EngineError(f"Unknown speaker {voice.preset!r}", {"speakers": sorted(self.spk2info)}, code=codes.INVALID_INPUT)
            embedding = self.spk2info[voice.preset]["embedding"].to(self.device)
            mi = {"text": text, "llm_embedding": embedding, "flow_embedding": embedding}
            if instruct:
                if self.version != 1:
                    raise EngineError("Instructions with a preset speaker need CosyVoice 1 Instruct", code=codes.CAPABILITY_NOT_SUPPORTED)
                del mi["llm_embedding"]
                mi["prompt_text"] = self._text_token(self.frontend.normalize(instruct + END_OF_PROMPT, split=False, text_frontend=normalize))
            return mi
        if not voice.references:
            raise EngineError("CosyVoice needs a voice: a reference clip or a preset speaker", code=codes.INVALID_INPUT)
        ref = voice.references[0]
        if instruct:
            if self.version == 1:
                raise EngineError("CosyVoice 1 takes instructions with a preset speaker only", code=codes.CAPABILITY_NOT_SUPPORTED)
            prefix = f"{system[: -len(END_OF_PROMPT)]} " if system else ""
            mi = self._zero_shot(f"{prefix}{instruct}{END_OF_PROMPT}", ref.audio_path, ctx)
            del mi["llm_prompt_speech_token"], mi["llm_prompt_speech_token_len"]
            return {**mi, "text": text}
        transcript = (ref.transcript or "").strip()
        if transcript:
            mi = dict(cached) if cached is not None else self._zero_shot(self.frontend.normalize(system + transcript, split=False, text_frontend=normalize), ref.audio_path, ctx)
            return {**mi, "text": text}
        mi = dict(cached) if cached is not None else self._zero_shot("", ref.audio_path, ctx)
        for key in ("prompt_text", "prompt_text_len", "llm_prompt_speech_token", "llm_prompt_speech_token_len"):
            mi.pop(key, None)
        return {**mi, "text": text}

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        import numpy as np

        stream = task.stream
        speed = float(task.params.get("speed") or 1.0)
        if stream and speed != 1.0:
            raise EngineError("Speed changes need a non-streaming request", code=codes.INVALID_INPUT)
        normalize = task.params.get("text_normalization")
        normalize = True if normalize is None else bool(normalize)
        ctx.applied.update({"speed": speed, "text_frontend": normalize, "stream": stream})
        if task.style.instruct:
            ctx.applied["instruct"] = task.style.instruct
        voice = task.voice
        cached = self._cached(voice) if voice.references else None
        ctx.applied["cached_speaker"] = cached is not None
        texts: list[Any] = list(task.texts) if task.texts else [task.text]
        system = self._system(task.options)
        for index, text in enumerate(texts):
            segment = task.segment + index if task.texts else task.segment
            if task.text_stream:
                if self.version == 1:
                    raise EngineError("CosyVoice 1 cannot take streamed text", {"capability": "stream_text_input"}, code=codes.CAPABILITY_NOT_SUPPORTED)
                if not voice.preset and not task.style.instruct and voice.references and not (voice.references[0].transcript or "").strip():
                    raise EngineError("Streamed text needs the reference clip's transcript", {"capability": "stream_text_input"}, code=codes.CAPABILITY_NOT_SUPPORTED)
                ctx.applied["text_stream"] = True
                yield from self._stream_text(task, text, cached, normalize, ctx, segment)
                continue
            if voice.references and not voice.preset and not task.style.instruct and not (voice.references[0].transcript or "").strip():
                tag = CROSS_LINGUAL_TAGS.get(task.language or "", "") if self.version == 1 else ""
                text = system + tag + text
            pending = None
            for piece in self.frontend.normalize(text, split=True, text_frontend=normalize):
                model_input = self._model_input(task, piece, cached, normalize, ctx)
                for speech in self._tts(model_input, model_input["text"], ctx, stream, speed):
                    samples = np.clip(np.asarray(speech.squeeze(0).float().cpu().numpy(), dtype=np.float32).reshape(-1), -1.0, 1.0)
                    if pending is not None:
                        yield AudioChunk(pending, self.sample_rate, segment)
                    pending = samples
            ctx.check_cancel()
            yield AudioChunk(pending if pending is not None else np.zeros(0, dtype=np.float32), self.sample_rate, segment, final=True)

    def _stream_text(self, task: SynthesisTask, first: str, cached: dict[str, Any] | None, normalize: bool, ctx: RequestContext, segment: int) -> Iterator[AudioChunk]:
        """Upstream's bistream path: the language model reads text tokens as they come, unnormalised."""
        import numpy as np

        def pieces() -> Iterator[torch.Tensor]:
            if first:
                for tok in self.tokenizer.encode(first):
                    yield self._text_token_ids([tok])
            for piece in ctx.text_pieces():
                for tok in self.tokenizer.encode(piece):
                    yield self._text_token_ids([tok])

        mi = self._model_input(task, "", cached, normalize, ctx)
        mi.pop("text", None)
        pending = None
        for speech in self._tts(mi, pieces(), ctx, task.stream, 1.0):
            samples = np.clip(np.asarray(speech.squeeze(0).float().cpu().numpy(), dtype=np.float32).reshape(-1), -1.0, 1.0)
            if pending is not None:
                yield AudioChunk(pending, self.sample_rate, segment)
            pending = samples
        ctx.check_cancel()
        yield AudioChunk(pending if pending is not None else np.zeros(0, dtype=np.float32), self.sample_rate, segment, final=True)

    def _text_token_ids(self, ids: list[int]) -> torch.Tensor:
        import torch

        return torch.tensor([ids], dtype=torch.int32, device=self.device)

    def convert(self, task: VoiceConversionTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        """``inference_vc``: say what the source recording says in the voice of the reference clip."""
        import numpy as np

        if not task.voice.references:
            raise EngineError("Voice conversion needs a reference clip of the target voice", code=codes.INVALID_INPUT)
        speed = float(task.params.get("speed") or 1.0)
        ctx.applied["speed"] = speed
        prompt = task.voice.references[0].audio_path
        prompt_feat, _ = self._speech_feat(prompt)
        prompt_token, embedding, source_token = ctx.run(
            "frontend",
            [self.h_frontend],
            lambda: (self._speech_token(prompt)[0], self._spk_embedding(prompt), self._speech_token(task.source_path)[0]),
            self._frontend_memory(),
        )
        model_input = {"flow_prompt_speech_token": prompt_token, "prompt_speech_feat": prompt_feat, "flow_embedding": embedding}
        outputs = list(self._tts(model_input, None, ctx, False, speed, source_tokens=source_token.flatten().tolist()))
        ctx.check_cancel()
        for n, speech in enumerate(outputs):
            samples = np.clip(np.asarray(speech.squeeze(0).float().cpu().numpy(), dtype=np.float32).reshape(-1), -1.0, 1.0)
            yield AudioChunk(samples, self.sample_rate, final=n == len(outputs) - 1)


# v1's streaming windows (``CosyVoiceModel.__init__``)
V1_MEL_OVERLAP = int(20 / 50 * 22050 / 256)
V1_SOURCE_CACHE = 20 * 256
V2_SOURCE_CACHE = 8 * 480


def _hamming(n: int) -> Any:
    import numpy as np

    return np.hamming(n)


MEL_WINDOW = _hamming(2 * V1_MEL_OVERLAP)
V1_SPEECH_WINDOW = _hamming(2 * V1_SOURCE_CACHE)
V2_SPEECH_WINDOW = _hamming(2 * V2_SOURCE_CACHE)


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: Any) -> torch.Tensor:
    """``fade_in_out``: the overlap mixed with a Hamming window, computed in float64 as upstream's numpy window makes it."""
    import torch

    device = fade_in.device
    fade_in, fade_out = fade_in.cpu().clone(), fade_out.cpu()
    n = int(window.shape[0] / 2)
    w = torch.from_numpy(window)
    fade_in[..., :n] = fade_in[..., :n].double() * w[:n] + fade_out[..., -n:].double() * w[n:]
    return fade_in.to(device)


def torch_len(t: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.tensor([t.shape[1]], dtype=torch.int32, device=t.device)


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
