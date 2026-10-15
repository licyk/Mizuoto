"""The GPT-SoVITS engine on tiny random models: no weights or assets needed.

Real-model parity (all parity cases identical to v1 sample for sample on the CPU) is checked through the stack
(``scripts/parity``); the decoders and the text front end have their own tests. This covers the engine's own
code: loading from upstream's folder layout, reference preparation, ``TTS.run``'s batching, both T2S paths,
streaming, the cut-free text path without a transcript, seeds, post-processing and switching weights. The text
front end is replaced by a stand-in (the real one needs the G2PW, NLTK and dictionary assets).
"""

from __future__ import annotations

import contextlib
import json
import wave
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mizuoto.engines.gptsovits import engine as engine_module  # noqa: E402
from mizuoto.engines.gptsovits import text as text_module  # noqa: E402
from mizuoto.engines.gptsovits.t2s import T2SConfig, Text2SemanticDecoder  # noqa: E402
from mizuoto.nn.ssl.hubert import HubertConfig, HubertModel  # noqa: E402
from mizuoto.runtime import ops  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EngineError, LoadSpec, Reference, SynthesisTask, VoiceSpec, get_engine  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from tests.engines_native.test_gptsovits_sovits_tiny import _write as write_sovits  # noqa: E402

T2S_MODEL = {"hidden_dim": 32, "embedding_dim": 32, "head": 2, "n_layer": 2, "vocab_size": 1025, "phoneme_vocab_size": 732, "EOS": 1024}
HUBERT = {"hidden_size": 768, "num_hidden_layers": 1, "num_attention_heads": 12, "intermediate_size": 64, "conv_dim": [32] * 7}


class FakeText:
    """``TextPreprocessor``'s interface: a piece per sentence, a phone per character, zero BERT features (from a BERT
    stand-in, called inside ``bert_scope`` as the real front end calls it)."""

    def __init__(self, bert: Any, device: Any, dtype: Any = None, assets: Any = None, **_: Any) -> None:
        self.calls: list[tuple[str, str]] = []
        self.bert = bert
        self.bert_scope: Any = contextlib.nullcontext

    def pre_seg_text(self, text: str, lang: str, method: str) -> list[str]:
        pieces = [p.strip() for p in text.replace("。", "。\n").replace(". ", ".\n").splitlines()]
        return [p for p in pieces if p]

    def segment_and_extract_feature_for_text(self, text: str, lang: str, version: str = "v2") -> tuple[list[int], Any, str]:
        self.calls.append((text, lang))
        phones = [1 + ord(c) % 700 for c in text if not c.isspace()]
        with torch.no_grad(), self.bert_scope():
            weight = self.bert.weight
            features = self.bert(torch.ones(len(phones), 4, device=weight.device, dtype=weight.dtype)).T.cpu()
        return phones, features, text

    def preprocess(self, text: str, lang: str, method: str, version: str = "v2") -> list[dict[str, Any]]:
        out = []
        for piece in self.pre_seg_text(text, lang, method):
            phones, bert, norm = self.segment_and_extract_feature_for_text(piece, lang, version)
            out.append({"phones": phones, "bert_features": bert, "norm_text": norm})
        return out


def _bert() -> Any:
    """A BERT stand-in with weights (zeros), so a call outside its phase fails on a strict simulated device."""
    bert = torch.nn.Linear(4, 1024, bias=False)
    with torch.no_grad():
        bert.weight.zero_()
    return bert


def _t2s(path: Path, seed: int) -> Path:
    model = Text2SemanticDecoder(T2SConfig.from_checkpoint({"model": T2S_MODEL, "data": {"max_sec": 2}}))
    ops.init_random(model, torch.Generator().manual_seed(seed), scale=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = {f"model.{k}": v for k, v in model.state_dict().items() if k != "ar_text_position.pe" and k != "ar_audio_position.pe"}
    torch.save({"config": {"model": T2S_MODEL, "data": {"max_sec": 2}}, "weight": weight}, str(path))
    return path


def _hubert(folder: Path) -> None:
    folder.mkdir(parents=True)
    model = HubertModel(HubertConfig.from_dict(HUBERT))
    ops.init_random(model, torch.Generator().manual_seed(3), scale=0.05)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(json.dumps(HUBERT), encoding="utf-8")


def _wav(path: Path, seconds: float, rate: int = 24000) -> Path:
    t = np.arange(int(rate * seconds)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 20000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("tiny-gptsovits")
    gpt, sovits = engine_module.BASE_WEIGHTS["v2"]
    model = root / "v2-base"
    _t2s(model / gpt, 1)
    write_sovits(model / sovits, "v2", None, seed=1)
    assets = {}
    for asset in (engine_module.ENCODERS, engine_module.G2PW, engine_module.NLTK, engine_module.OPEN_JTALK):
        assets[asset] = root / "_assets" / asset
        assets[asset].mkdir(parents=True)
    _hubert(assets[engine_module.ENCODERS] / "chinese-hubert-base")
    weights = root / "weights"
    weights.mkdir()
    return {
        "model": model,
        "root": root,
        "ref": _wav(root / "ref.wav", 4.0),
        "short": _wav(root / "short.wav", 1.0),
        "gpt2": _t2s(weights / "voice.ckpt", 2),
        "sovits2": write_sovits(weights / "voice.pth", "v2", None, seed=2),
        "pro": write_sovits(weights / "pro.pth", "v2Pro", b"05", seed=3),
        **{f"asset:{k}": v for k, v in assets.items()},
    }


@pytest.fixture(scope="module")
def loaded(files):
    engine = get_engine("gptsovits")
    assert engine.detect_variant(files["model"]) == "v2"
    patch = pytest.MonkeyPatch()
    patch.setattr(text_module, "TextPreprocessor", FakeText)
    patch.setattr(text_module, "load_bert_model", lambda encoders, device, dtype=None: _bert())
    assets = {k.split(":", 1)[1]: str(v) for k, v in files.items() if k.startswith("asset:")}
    devices = DeviceService()
    try:
        yield engine.load(LoadSpec("gptsovits/tiny", files["model"], "v2", "cpu", options={"assets": assets}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    finally:
        patch.undo()


def _task(files: dict[str, Path], text: str, *, transcript: str | None = "参考的文本", refs: int = 1, **kw: Any) -> SynthesisTask:
    references = [Reference(str(files["ref"]), transcript, "zh")] + [Reference(str(files["ref"]), None, "zh")] * (refs - 1)
    return SynthesisTask(text=text, language="zh", voice=VoiceSpec(references=references), **kw)


def _speak(loaded, task: SynthesisTask, seed: int | None = 7) -> tuple[np.ndarray, list[Any], RequestContext]:
    ctx = RequestContext("r", seed=seed)
    with torch.inference_mode():
        chunks = list(loaded.synthesize(task, ctx))
    return np.concatenate([c.samples for c in chunks]), chunks, ctx


def test_load_and_describe(loaded) -> None:
    d = loaded.describe()
    assert d.sample_rate == 32000
    assert {h.name.rsplit(":", 1)[1] for h in loaded.handles} == {"hubert", "bert", "t2s", "sovits"}


def test_speak_repeats_under_a_seed(loaded, files) -> None:
    task = _task(files, "今天天气很好。我们去公园。", params={"seed": 7})
    a, chunks, ctx = _speak(loaded, task)
    b, _, _ = _speak(loaded, task)
    c, _, _ = _speak(loaded, task, seed=8)
    assert len(chunks) == 1 and chunks[0].final and chunks[0].sample_rate == 32000
    assert a.size > 0 and np.abs(a).max() <= 1.0
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # 16-bit audio, each fragment followed by 0.3 s of silence at 32 kHz
    assert np.array_equal(a * 32768, np.round(a * 32768)) and not a[-9600:].any()
    assert ctx.applied["text_lang"] == "zh" and ctx.applied["seed"] == 7 and ctx.applied["references"] == 1


def test_options_and_paths(loaded, files) -> None:
    text = "第一句话。第二句话稍微长一点。第三句。"
    batched, _, _ = _speak(loaded, _task(files, text, options={"batch_size": 3}))
    serial, _, ctx = _speak(loaded, _task(files, text, options={"batch_size": 3, "parallel_infer": False}))
    assert batched.size > 0 and serial.size > 0
    assert ctx.applied["parallel_infer"] is False and ctx.applied["batch_size"] == 3
    slow, _, _ = _speak(loaded, _task(files, text, params={"speed": 0.8}))
    pause, _, ctx = _speak(loaded, _task(files, text, params={"segment_silence_ms": 0}))
    assert slow.size > 0 and ctx.applied["fragment_interval"] == 0.0
    no_transcript, _, _ = _speak(loaded, _task(files, text, transcript=None))
    two_refs, _, ctx = _speak(loaded, _task(files, text, refs=2))
    assert no_transcript.size > 0 and two_refs.size > 0 and ctx.applied["references"] == 2


def test_stream(loaded, files) -> None:
    whole, _, _ = _speak(loaded, _task(files, "今天天气很好。"))
    audio, chunks, ctx = _speak(loaded, _task(files, "今天天气很好。", stream=True, options={"min_chunk_length": 4}))
    assert ctx.applied["streaming_mode"] is True and len(chunks) > 2 and chunks[-1].final
    assert not chunks[-1].samples.any()  # the pause after the text
    assert audio.size > 0 and np.abs(audio).max() <= 1.0
    assert not np.array_equal(audio, whole)  # decoded chunk by chunk, row by row


def test_reference_limits(loaded, files) -> None:
    task = SynthesisTask(text="你好。", language="zh", voice=VoiceSpec(references=[Reference(str(files["short"]), "你好", "zh")]))
    with pytest.raises(EngineError, match="3 to 10 seconds"):
        _speak(loaded, task)
    with pytest.raises(EngineError, match="reference clip"):
        _speak(loaded, SynthesisTask(text="你好。", language="zh"))


def test_switch_weights(loaded, files) -> None:
    task = _task(files, "今天天气很好。")
    base, _, _ = _speak(loaded, task)
    voice = _task(files, "今天天气很好。")
    voice.voice.weights = {"gpt": str(files["gpt2"]), "sovits": str(files["sovits2"])}
    own, _, ctx = _speak(loaded, voice)
    assert ctx.applied["voice_weights"] is True and not np.array_equal(base, own)
    assert loaded.current == (str(files["gpt2"]), str(files["sovits2"]))
    with pytest.raises(EngineError, match="v2Pro, not v2"):
        loaded.call("switch_weights", {"weights": {"sovits": str(files["pro"])}}, RequestContext("w"))
    back, _, _ = _speak(loaded, task)
    assert np.array_equal(base, back) and loaded.current == loaded.base


def test_front_end_errors(loaded, files, monkeypatch) -> None:
    from mizuoto.engines.gptsovits.text.japanese import OpenJTalkMissing
    from mizuoto.engines.gptsovits.text.segmentation import EmptyTextError

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise OpenJTalkMissing("pyopenjtalk")

    def empty(*args: Any, **kwargs: Any) -> Any:
        raise EmptyTextError("请输入有效文本")

    monkeypatch.setattr(loaded.text, "preprocess", missing)
    with pytest.raises(EngineError, match="Japanese") as info:
        _speak(loaded, _task(files, "こんにちは。"))
    assert info.value.detail["hint"] == "pip install 'mizuoto[gptsovits-ja]'"
    monkeypatch.setattr(loaded.text, "preprocess", empty)
    with pytest.raises(EngineError, match="nothing to say"):
        _speak(loaded, _task(files, "。"))


def test_runs_identically_with_its_weights_placed_by_phase_and_streamed(loaded, files, monkeypatch) -> None:
    """Reference (HuBERT and SoVITS), text (BERT), GPT and SoVITS each in its phase on a simulated
    device, units streamed: the same audio, batched, serial, streamed and with a voice's own weights switched in."""
    from tests.engines_native.streaming import check_streamed

    text = "第一句话。第二句话稍微长一点。第三句。"

    def run(manager, task):
        loaded._reference = loaded._prompt = None  # made again in each run: their phases are part of what is checked
        loaded.text.bert_scope = contextlib.nullcontext  # as loaded: the first request sets it
        ctx = RequestContext("r", seed=7)
        ctx.manager = manager
        with torch.inference_mode():
            return np.concatenate([c.samples for c in loaded.synthesize(task, ctx)])

    streamed = check_streamed(loaded, lambda m: run(m, _task(files, text, options={"batch_size": 3})), monkeypatch)
    assert streamed[0] > 0 and streamed[-1] == 0
    check_streamed(loaded, lambda m: run(m, _task(files, text, options={"batch_size": 3, "parallel_infer": False})), monkeypatch)
    check_streamed(loaded, lambda m: run(m, _task(files, "今天天气很好。", stream=True, options={"min_chunk_length": 4})), monkeypatch)
    voice = _task(files, "今天天气很好。")
    voice.voice.weights = {"gpt": str(files["gpt2"]), "sovits": str(files["sovits2"])}
    check_streamed(loaded, lambda m: run(m, voice), monkeypatch)
    _speak(loaded, _task(files, "今天天气很好。"))  # back to the base weights for the tests after


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_cuda_graphs_decode_exactly_as_eagerly_on_a_gpu(files) -> None:
    """compute.cuda_graphs (mizuoto.runtime.graphs): the GPT's one-token steps as graphs around its attention calls replay
    the eager kernels: the same audio, batched (rows dropping as they finish), row by row and streamed, for fresh loads
    and later requests."""
    from mizuoto.runtime.runtime import RuntimeOptions

    def load(graphs: bool):
        engine = get_engine("gptsovits")
        assets = {k.split(":", 1)[1]: str(v) for k, v in files.items() if k.startswith("asset:")}
        devices = DeviceService()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(text_module, "TextPreprocessor", FakeText)
            patch.setattr(text_module, "load_bert_model", lambda encoders, device, dtype=None: _bert())
            loaded = engine.load(
                LoadSpec("gptsovits/tiny", files["model"], "v2", "cuda", options={"assets": assets}),
                Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
                RequestContext("load"),
            )
        for h in loaded.handles:
            h.load()
        return loaded

    text = "第一句话。第二句话稍微长一点。第三句。"
    tasks = [
        _task(files, text, options={"batch_size": 3}),
        _task(files, text, options={"batch_size": 3, "parallel_infer": False}),
        # (streamed chunks this tiny model cuts too short leave upstream's SOLA an empty piece on a GPU: not these)
        _task(files, "今天天气很好。", stream=True, options={"min_chunk_length": 8}),
        _task(files, "今天天气很好，我们一起去公园散步吧。", stream=True, options={"min_chunk_length": 4}),
        _task(files, text, options={"batch_size": 3}),
    ]

    def run(loaded):
        out = []
        for task in tasks:
            ctx = RequestContext("r", seed=7, device="cuda")
            with torch.inference_mode():
                chunks = list(loaded.synthesize(task, ctx))
            out.append((np.concatenate([c.samples for c in chunks]), ctx.applied.get("cuda_graphs")))
        return out

    for _ in range(2):
        for (want, _), (got, used) in zip(run(load(False)), run(load(True)), strict=True):
            assert used
            assert want.shape == got.shape and (want == got).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA graphs need an NVIDIA GPU")
def test_a_failed_capture_falls_back_to_the_same_audio(files, monkeypatch, caplog) -> None:
    """A step that cannot be captured (here: one that reads a value back) is decoded eagerly: the same audio."""
    from mizuoto.engines.gptsovits import t2s
    from mizuoto.runtime.runtime import RuntimeOptions

    def run(graphs: bool) -> np.ndarray:
        assets = {k.split(":", 1)[1]: str(v) for k, v in files.items() if k.startswith("asset:")}
        devices = DeviceService()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(text_module, "TextPreprocessor", FakeText)
            patch.setattr(text_module, "load_bert_model", lambda encoders, device, dtype=None: _bert())
            loaded = get_engine("gptsovits").load(
                LoadSpec("gptsovits/tiny", files["model"], "v2", "cuda", options={"assets": assets}),
                Runtime(devices, ModelManager(devices), RuntimeOptions(cuda_graphs=graphs)),
                RequestContext("load"),
            )
        for h in loaded.handles:
            h.load()
        with torch.inference_mode():
            chunks = list(loaded.synthesize(_task(files, "第一句话。第二句话稍微长一点。", options={"batch_size": 2}), RequestContext("r", seed=7, device="cuda")))
        return np.concatenate([c.samples for c in chunks])

    want = run(False)
    qkv = t2s.EncoderLayer.step_qkv
    monkeypatch.setattr(t2s.EncoderLayer, "step_qkv", lambda self, x: (x.sum().item(), qkv(self, x))[1])
    got = run(True)
    assert "CUDA graphs failed" in caplog.text
    assert want.shape == got.shape and (want == got).all()
