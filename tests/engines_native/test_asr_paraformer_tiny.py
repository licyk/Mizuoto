"""The asr engine's Paraformer variant (FunASR's Paraformer + FSMN-VAD + CT-Transformer) on tiny random models.

Real-model parity is checked against FunASR 1.4.21 on a corpus of clips (engines-asr on the site: every transcript, VAD
segment and intermediate tensor identical); this covers the code paths in CI: the folder's detection, reading and
resampling, the VAD's chunks, recognition per segment with its dither and layer-drop draws, the text, punctuation,
placement by phase and streaming.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, get_engine  # noqa: E402

HAN = list(
    "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实"
)
WORDS = ["hello", "world", "mizu@@", "oto", "a", "b", "c"]

VAD_CONFIG = """frontend: WavFrontendOnline
frontend_conf:
    fs: 16000
    window: hamming
    n_mels: 80
    frame_length: 25
    frame_shift: 10
    dither: 0.0
    lfr_m: 5
    lfr_n: 1

model: FsmnVADStreaming
model_conf:
    sample_rate: 16000
    detect_mode: 1
    snr_mode: 0
    max_end_silence_time: 800
    max_start_silence_time: 3000
    do_start_point_detection: True
    do_end_point_detection: True
    window_size_ms: 200
    sil_to_speech_time_thres: 150
    speech_to_sil_time_thres: 150
    speech_2_noise_ratio: 1.0
    do_extend: 1
    lookback_time_start_point: 200
    lookahead_time_end_point: 100
    max_single_segment_time: 60000
    snr_thres: -100.0
    noise_frame_num_used_for_snr: 100
    decibel_thres: -100.0
    speech_noise_thres: 0.6
    fe_prior_thres: 0.0001
    silence_pdf_num: 1
    sil_pdf_ids: [0]
    speech_noise_thresh_low: -0.1
    speech_noise_thresh_high: 0.3
    output_frame_probs: False
    frame_in_ms: 10
    frame_length_ms: 25

encoder: FSMN
encoder_conf:
    input_dim: 400
    input_affine_dim: 140
    fsmn_layers: 4
    linear_dim: 250
    proj_dim: 128
    lorder: 20
    rorder: 0
    lstride: 1
    rstride: 0
    output_affine_dim: 140
    output_dim: 248
"""

ASR_CONFIG = """model: Paraformer
encoder: SANMEncoder
encoder_conf:
    output_size: 16
    attention_heads: 2
    linear_units: 32
    num_blocks: 3
    kernel_size: 5
decoder: ParaformerSANMDecoder
decoder_conf:
    attention_heads: 2
    linear_units: 32
    num_blocks: 2
    att_layer_num: 2
    kernel_size: 5
predictor: CifPredictorV2
predictor_conf:
    idim: 16
    threshold: 1.0
    l_order: 1
    r_order: 1
    tail_threshold: 0.45
frontend: WavFrontend
frontend_conf:
    fs: 16000
    window: hamming
    n_mels: 80
    frame_length: 25
    frame_shift: 10
    lfr_m: 7
    lfr_n: 6
input_size: 560
"""

PUNC_CONFIG = """model: CTTransformer
model_conf:
    ignore_id: 0
    embed_unit: 16
    att_unit: 16
    dropout_rate: 0.1
    punc_list:
        - <unk>
        - _
        - ，
        - 。
        - ？
        - 、
    sentence_end_id: 3
encoder: SANMEncoder
encoder_conf:
    input_size: 16
    output_size: 16
    attention_heads: 2
    linear_units: 32
    num_blocks: 2
    kernel_size: 5
    input_layer: pe
    padding_idx: 0
tokenizer: CharTokenizer
tokenizer_conf:
  unk_symbol: <unk>
"""


def _mvn(path: Path, dim: int, seed: int) -> None:
    g = np.random.default_rng(seed)
    shift = " ".join(f"{v:.6f}" for v in -8 - g.random(dim))
    scale = " ".join(f"{v:.6f}" for v in 0.2 + 0.1 * g.random(dim))
    path.write_text(
        f"<Nnet>\n<Splice> {dim} {dim}\n[ 0 ]\n<AddShift> {dim} {dim}\n<LearnRateCoef> 0 [ {shift} ]\n<Rescale> {dim} {dim}\n<LearnRateCoef> 0 [ {scale} ]\n</Nnet>\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    from mizuoto.engines.asr.paraformer import punc, vad
    from mizuoto.engines.asr.paraformer.model import Paraformer, load_config
    from mizuoto.runtime import ops

    root = tmp_path_factory.mktemp("paraformer")
    (root / "vad").mkdir()
    (root / "punc").mkdir()
    tokens = ["<blank>", "<s>", "</s>", *WORDS, *HAN, "<unk>"]
    (root / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    (root / "config.yaml").write_text(ASR_CONFIG, encoding="utf-8")
    _mvn(root / "am.mvn", 560, 1)
    asr = ops.init_random(Paraformer(load_config(root), len(tokens)), torch.Generator().manual_seed(1), scale=0.2)
    torch.save(asr.state_dict(), root / "model.pt")

    (root / "vad" / "config.yaml").write_text(VAD_CONFIG, encoding="utf-8")
    _mvn(root / "vad" / "am.mvn", 400, 2)
    detector = vad.Vad(load_config(root / "vad"))
    ops.init_random(detector, torch.Generator().manual_seed(2))
    with torch.no_grad():  # speech everywhere: the silence class (0) improbable, so the VAD finds segments
        detector.encoder.out_linear2.linear.bias[0] = -10.0
    torch.save(detector.state_dict(), root / "vad" / "model.pt")

    (root / "punc" / "config.yaml").write_text(PUNC_CONFIG, encoding="utf-8")
    (root / "punc" / "tokens.json").write_text(json.dumps(["<blank>", "<s>", "</s>", *WORDS, *HAN, "<unk>"], ensure_ascii=False), encoding="utf-8")
    model, _ = punc.build(root / "punc")
    ops.init_random(model, torch.Generator().manual_seed(3), scale=0.5)
    torch.save(model.state_dict(), root / "punc" / "model.pt")
    return root


@pytest.fixture(scope="module")
def loaded(model_dir):
    from mizuoto.runtime.device import DeviceService
    from mizuoto.runtime.manager import ModelManager
    from mizuoto.runtime.runtime import Runtime

    engine = get_engine("asr")
    assert engine.detect_variant(model_dir) == "paraformer"
    devices = DeviceService()
    return engine.load(LoadSpec("asr/paraformer-tiny", model_dir, "paraformer", "cpu"), Runtime(devices, ModelManager(devices)), RequestContext("load"))


def _wav(path: Path, seconds: float, rate: int = 22050, seed: int = 0) -> Path:
    g = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1.3 * t)) + 0.05 * g.standard_normal(t.size)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def transcribe(loaded, path: Path, ctx: RequestContext | None = None) -> dict:
    return loaded.call("transcribe", {"path": str(path), "language": "zh"}, ctx or RequestContext("t"))


def test_a_clip_is_transcribed_the_same_every_time(loaded, tmp_path):
    path = _wav(tmp_path / "a.wav", 2.5)
    first = transcribe(loaded, path)
    assert first["language"] == "zh" and first["duration"] == pytest.approx(2.5, abs=1e-3)
    assert isinstance(first["text"], str) and first["text"] and first["segments"]
    assert all(0 <= s["start"] < s["end"] <= 2.5 + 1e-6 and s["text"] for s in first["segments"])
    assert transcribe(loaded, path) == first  # the dither's generator is seeded per transcription


def test_long_audio_goes_through_the_vads_chunks(loaded, tmp_path):
    """Over 60 s the VAD runs in 60 s chunks with its state carried across, and the segments stay in order."""
    out = transcribe(loaded, _wav(tmp_path / "long.wav", 61.0, rate=16000, seed=4))
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts) and out["segments"][-1]["end"] <= 61.0 + 1e-6 and out["text"]


def test_recognition_draws_upstreams_layer_drop_numbers(loaded):
    """Upstream's layer stacks draw one uniform number per layer at every forward: the encoder's (1 + its other
    layers) always, the decoder's (its layers + 1) when a token fired."""
    feats = torch.randn(40, 560, generator=torch.Generator().manual_seed(5))
    g = torch.Generator().manual_seed(7)
    with torch.inference_mode():
        ids = loaded.model.recognise(feats, g)
    assert ids  # tokens fired: the decoder ran
    expected = torch.Generator().manual_seed(7)
    stacks = [1, len(loaded.model.encoder.encoders), len(loaded.model.decoder.decoders), 1]
    for n in stacks:
        torch.empty(n).uniform_(generator=expected)
    assert torch.equal(g.get_state(), expected.get_state())


def test_runs_identically_placed_by_phase_and_streamed(loaded, tmp_path, monkeypatch):
    from tests.engines_native.streaming import check_streamed

    path = _wav(tmp_path / "s.wav", 3.0, seed=6)

    def run(manager):
        ctx = RequestContext("r")
        ctx.manager = manager
        out = transcribe(loaded, path, ctx)
        return np.frombuffer(out["text"].encode("utf-32-le"), dtype=np.uint32).copy()

    assert check_streamed(loaded, run, monkeypatch)[0] > 0


def test_dither_needs_a_generator():
    from mizuoto.nn.dsp import kaldi

    with pytest.raises(ValueError, match="generator"):
        kaldi.fbank(torch.zeros(1, 1600), dither=1.0)
    a = kaldi.fbank(torch.ones(1, 1600), dither=1.0, generator=torch.Generator().manual_seed(0), energy_floor=0.0)
    b = kaldi.fbank(torch.ones(1, 1600), dither=1.0, generator=torch.Generator().manual_seed(0), energy_floor=0.0)
    assert torch.equal(a, b)
