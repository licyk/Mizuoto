"""The native CosyVoice trainer on tiny random CosyVoice 1, 2 and 3 models: every stage of both methods (``llm``:
features, llm, export; ``llm_flow``: features, llm, flow, export) through ``mizuoto.training`` in seconds.

Parity with upstream's training on the real models (losses and checkpoints identical step for step on the CPU) is
checked by the experiment's drivers and scripts/parity/train.py; this covers the code path in CI: FLAC and WAV clips,
the native speech tokens and CAM++ embeddings, the recipe's filter, the data loader's workers, both language models
and the three flows with their losses, gradient accumulation, the checkpoints in upstream's layout, the preview, and
the exported model loaded and speaking in the native engine.
"""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
for _module in ("tokenizers", "safetensors", "yaml", "inflect", "regex"):
    pytest.importorskip(_module)

from safetensors.torch import save_file  # noqa: E402

from mizuoto.core.audio.io import encode  # noqa: E402
from mizuoto.core.jobs.events import LogEvent, MetricEvent, OutputEvent, ResultEvent  # noqa: E402
from mizuoto.core.jobs.jsonl import Emitter  # noqa: E402
from mizuoto.runtime.context import RequestContext  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.runtime import Runtime  # noqa: E402
from mizuoto.training.stage import stage_main  # noqa: E402
from mizuoto.training.trainer import missing_requirements, trainer_for  # noqa: E402
from tests.engines_native.test_cosyvoice_tiny import YAML as YAML_V2  # noqa: E402
from tests.engines_native.test_cosyvoice_tiny import _byte_vocab, _random_state  # noqa: E402

YAML_V1 = """
sample_rate: 22050
llm_input_size: 16
spk_embed_dim: 192
llm: !new:cosyvoice.llm.llm.TransformerLM
    text_encoder_input_size: 16
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_input_size>
    text_token_size: 51866
    speech_token_size: 4096
    spk_embed_dim: !ref <spk_embed_dim>
    text_encoder: !new:cosyvoice.transformer.encoder.ConformerEncoder
        input_size: 16
        output_size: 16
        attention_heads: 2
        linear_units: 32
        num_blocks: 1
        input_layer: 'linear'
        static_chunk_size: 1
    llm: !new:cosyvoice.transformer.encoder.TransformerEncoder
        input_size: !ref <llm_input_size>
        output_size: !ref <llm_input_size>
        attention_heads: 2
        linear_units: 32
        num_blocks: 1
        input_layer: 'linear_legacy'
        static_chunk_size: 1
flow: !new:cosyvoice.flow.flow.MaskedDiffWithXvec
    input_size: 16
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    vocab_size: 4096
    input_frame_rate: 50
    encoder: !new:cosyvoice.transformer.encoder.ConformerEncoder
        input_size: 16
        output_size: 16
        attention_heads: 2
        linear_units: 32
        num_blocks: 1
        input_layer: 'linear'
    length_regulator: !new:cosyvoice.flow.length_regulator.InterpolateRegulator
        channels: 80
        sampling_ratios: [1, 1]
    decoder: !new:cosyvoice.flow.flow_matching.ConditionalCFM
        in_channels: 240
        cfm_params: !new:omegaconf.DictConfig
            content:
                inference_cfg_rate: 0.7
        estimator: !new:cosyvoice.flow.decoder.ConditionalDecoder
            in_channels: 320
            out_channels: 80
            channels: [16, 16]
            attention_head_dim: 8
            n_blocks: 1
            num_mid_blocks: 1
            num_heads: 2
hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
    in_channels: 80
    base_channels: 16
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    upsample_rates: [8, 8]
    upsample_kernel_sizes: [16, 16]
    istft_params:
        n_fft: 16
        hop_len: 4
    source_resblock_kernel_sizes: [7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5]]
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor
        in_channels: 80
        cond_channels: 16
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1024
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 256
    win_size: 1024
    fmin: 0
    fmax: 8000
    center: False
"""

YAML_V3 = """
sample_rate: 24000
llm_input_size: 32
spk_embed_dim: 192
chunk_size: 25
token_mel_ratio: 2
llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_input_size>
    speech_token_size: 6561
flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
    input_size: 80
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    vocab_size: 6561
    input_frame_rate: 25
    token_mel_ratio: !ref <token_mel_ratio>
    pre_lookahead_len: 3
    pre_lookahead_layer: !new:cosyvoice.transformer.upsample_encoder.PreLookaheadLayer
        in_channels: 80
        channels: 16
        pre_lookahead_len: 3
    decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        in_channels: 240
        cfm_params: !new:omegaconf.DictConfig
            content:
                inference_cfg_rate: 0.7
        estimator: !new:cosyvoice.flow.DiT.dit.DiT
            dim: 32
            depth: 1
            heads: 2
            dim_head: 16
            ff_mult: 2
            mel_dim: 80
            mu_dim: 80
            spk_dim: 80
            out_channels: 80
            static_chunk_size: !ref <chunk_size> * <token_mel_ratio>
hift: !new:cosyvoice.hifigan.generator.CausalHiFTGenerator
    in_channels: 80
    base_channels: 16
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    upsample_rates: [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params:
        n_fft: 16
        hop_len: 4
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    conv_pre_look_right: 4
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor
        in_channels: 80
        cond_channels: 16
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: null
    center: False
"""

VARIANTS = {1: "v1_base", 2: "v2", 3: "v3"}
END_OF_TEXT = 151643
"""Qwen2's: CosyVoice 3's specials follow it, so ``<|endofprompt|>`` is 151646, the id its language model looks for."""


def _tiny(root: Path, version: int) -> Path:
    """A tiny random CosyVoice ``version`` model folder (the v2 one as ``test_cosyvoice_tiny`` builds it)."""
    from mizuoto.engines.cosyvoice.config import read_config
    from mizuoto.engines.cosyvoice.engine import hift_kwargs
    from mizuoto.engines.cosyvoice.flow import Flow
    from mizuoto.engines.cosyvoice.llm import QwenLM, TransformerLM
    from mizuoto.nn.codecs.s3tokenizer import S3Tokenizer
    from mizuoto.nn.speaker.campplus import CAMPPlus
    from mizuoto.nn.vocoders.hift import HiFTGenerator
    from mizuoto.runtime import ops

    name = {1: "cosyvoice.yaml", 2: "cosyvoice2.yaml", 3: "cosyvoice3.yaml"}[version]
    (root / name).write_text({1: YAML_V1, 2: YAML_V2, 3: YAML_V3}[version], encoding="utf-8")
    if version > 1:
        blank = root / "CosyVoice-BlankEN"
        blank.mkdir()
        vocab = _byte_vocab()
        if version == 3:
            vocab.update({f"<unused{i}>": i for i in range(len(vocab), END_OF_TEXT)})
        (blank / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (blank / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        added = {str(len(vocab)): {"content": "<|endoftext|>", "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False}}
        (blank / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
        qwen = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "vocab_size": 151936 if version == 3 else 300,
        }
        (blank / "config.json").write_text(json.dumps(qwen), encoding="utf-8")
    config = read_config(root)
    if version == 1:
        llm = _random_state(TransformerLM(config["llm"], ops.default), 1)
        llm["llm_decoder.bias"] = torch.zeros(4097)
        llm["llm_decoder.bias"][4096] = 6.0  # end of speech likely: short previews
    else:
        module = QwenLM(version, root / "CosyVoice-BlankEN", 6561, ops.default)
        llm = _random_state(module, 1)
        # upstream saves the tied output projection too (the token embedding)
        llm["llm.model.lm_head.weight"] = llm["llm.model.model.embed_tokens.weight"]
        if version == 2:
            llm["llm_decoder.bias"] = torch.zeros(6564)
            llm["llm_decoder.bias"][6561] = 6.0
            llm["llm_decoder.bias"][6562:] = -1e4
    torch.save(llm, root / "llm.pt")
    torch.save(_random_state(Flow(version, config["flow"], ops.default), 2), root / "flow.pt")
    hift = HiFTGenerator(**hift_kwargs(config["hift"]), ops=ops.default)
    torch.save(_random_state(hift, 3, scale=0.05), root / "hift.pt")
    save_file(_random_state(S3Tokenizer(version, ops.default, n_state=128, n_head=2, n_layer=1), 4), str(root / f"speech_tokenizer_v{version}.safetensors"))
    save_file({k: v for k, v in _random_state(CAMPPlus(ops=ops.default), 5).items() if not k.endswith("num_batches_tracked")}, str(root / "campplus.safetensors"))
    (root / "model.json").write_text("{}", encoding="utf-8")  # Mizuoto's sidecar: not exported
    return root


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> list[dict]:
    """Three clips as a dataset keeps them (FLAC at their own rate, one WAV), one long enough to be the preview's voice."""
    folder = tmp_path_factory.mktemp("clips")
    rng = np.random.default_rng(0)
    out = []
    for i, (rate, seconds, kind) in enumerate([(16000, 1.2, "flac"), (22050, 3.2, "wav"), (24000, 1.4, "flac")]):
        t = np.arange(int(rate * seconds)) / rate
        samples = (0.3 * np.sin(2 * np.pi * (150 + 50 * i) * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
        path = folder / f"c{i}.{kind}"
        if kind == "flac":
            encode(samples, rate, path, "flac")
        else:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((samples * 32767).astype("<i2").tobytes())
        out.append({"id": f"r{i}", "audio": str(path), "text": f"Clip number {i}, said slowly.", "speaker": "s", "language": "en", "duration": seconds, "sample_rate": rate})
    return out


class Recorder(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def run(stage: str, workdir: Path, model_dir: Path, version: int, rows: list[dict], **hp) -> Recorder:
    request = {
        "trainer": trainer_for("cosyvoice"),
        "stage": stage,
        "engine": "cosyvoice",
        "method": "llm_flow",
        "experiment": "tiny",
        "workdir": str(workdir),
        "rows": rows,
        "base_model": {"id": "cosyvoice/tiny", "dir": str(model_dir), "variant": VARIANTS[version]},
        "previews": True,
        "device": "cpu",
        "precision": "auto",
        "keep_checkpoints": 1,
        "hyperparams": {"epochs": 2, "grad_accumulation": 2, "workers": 2, **hp},
        "language": "en",
    }
    rec = Recorder()
    stage_main(request, rec)
    return rec


@pytest.fixture(scope="module", params=[1, 2, 3], ids=["v1", "v2", "v3"])
def trained(request, tmp_path_factory, rows):
    """Both methods: ``llm`` (features, llm, export) in one folder and ``llm_flow`` (the same features, llm, flow,
    export) in another."""
    version = request.param
    model_dir = _tiny(tmp_path_factory.mktemp(f"tiny-cosyvoice{version}"), version)
    llm_only = tmp_path_factory.mktemp(f"llm-v{version}")
    recs = {stage: run(stage, llm_only, model_dir, version, rows) for stage in ("features", "llm", "export")}
    both = tmp_path_factory.mktemp(f"llm-flow-v{version}")
    shutil.copytree(llm_only / "data", both / "data")
    shutil.copytree(llm_only / "exp", both / "exp")
    recs["flow"] = run("flow", both, model_dir, version, rows)
    recs["export-both"] = run("export", both, model_dir, version, rows)
    return version, model_dir, llm_only, both, recs


def test_the_engine_names_its_trainer_and_needs_nothing_more():
    assert trainer_for("cosyvoice") == "mizuoto.engines.cosyvoice.train:CosyVoiceTrainer"
    assert missing_requirements(str(trainer_for("cosyvoice"))) == []


def test_features_tokenise_and_embed_every_clip(trained):
    version, _, workdir, _, recs = trained
    assert recs["features"].of(ResultEvent)[0].data == {"clips": 3, "shards": 1}
    items = [json.loads(line) for line in (workdir / "data" / "utts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [i["utt"] for i in items] == ["r0", "r1", "r2"]
    for item in items:
        assert len(item["utt_embedding"]) == 192 and np.isfinite(item["utt_embedding"]).all()
        assert len(item["speech_token"]) > 0 and max(item["speech_token"]) < (4096 if version == 1 else 6561)
        assert ("instruct" in item) == (version == 3)
        with wave.open(item["wav"], "rb") as w:
            assert w.getsampwidth() == 2 and w.getnchannels() == 1


def test_training_reports_every_batch_and_saves_upstreams_checkpoint(trained):
    version, model_dir, llm_only, both, recs = trained
    for part, folder in (("llm", llm_only), ("flow", both)):
        metrics = recs[part].of(MetricEvent)
        # two workers each read the one shard: two batches an epoch, one update every two
        assert [m.step for m in metrics] == [1, 2, 3, 4] and [m.epoch for m in metrics] == [1, 1, 2, 2]
        assert all(set(m.values) == ({"loss", "acc"} if part == "llm" else {"loss"}) and np.isfinite(m.values["loss"]) for m in metrics)
        assert any("validation" in e.message for e in recs[part].of(LogEvent))
        assert recs[part].of(ResultEvent)[0].data == {"epochs": 2, "checkpoint": "epoch_1_whole.pt"}
        assert [p.name for p in (folder / "exp" / part).iterdir()] == ["epoch_1_whole.pt"]  # keep_checkpoints = 1
        state = torch.load(folder / "exp" / part / "epoch_1_whole.pt", weights_only=True)
        base = torch.load(model_dir / f"{part}.pt", weights_only=True)
        assert set(state) == set(base) | {"epoch", "step"} and (state["epoch"], state["step"]) == (1, 2)
        if part == "llm" and version > 1:
            assert state["llm.model.lm_head.weight"].data_ptr() == state["llm.model.model.embed_tokens.weight"].data_ptr()
        assert any(not torch.equal(state[k], base[k]) for k in base)


def test_a_stage_repeats_under_its_seed(trained, tmp_path, rows):
    version, model_dir, llm_only, _, recs = trained
    (tmp_path / "data").symlink_to(llm_only / "data")
    again = run("llm", tmp_path, model_dir, version, rows).of(MetricEvent)
    assert [m.values for m in again] == [m.values for m in recs["llm"].of(MetricEvent)]


def test_no_workers_and_accumulation_across_epochs(trained, tmp_path, rows):
    """Without workers the pipeline draws from the stage's own generator; a leftover accumulation carries on."""
    version, model_dir, llm_only, _, _ = trained
    (tmp_path / "data").symlink_to(llm_only / "data")
    metrics = run("flow", tmp_path, model_dir, version, rows, workers=0, grad_accumulation=3).of(MetricEvent)
    assert [m.step for m in metrics] == [1, 2] and all(np.isfinite(m.values["loss"]) for m in metrics)
    assert torch.load(tmp_path / "exp" / "flow" / "epoch_1_whole.pt", weights_only=True)["step"] == 0


def test_export_replaces_the_trained_parts_and_speaks(trained, rows):
    version, model_dir, llm_only, both, recs = trained
    for rec, folder, parts in ((recs["export"], llm_only, ["llm"]), (recs["export-both"], both, ["llm", "flow"])):
        result = rec.of(ResultEvent)[0].data
        export = folder / "export"
        assert result["trained"] == parts
        assert result["artifacts"] == [{"kind": "model", "folder": str(export), "variant": VARIANTS[version], "title": "tiny"}]
        assert not (export / "model.json").exists()
        for part in ("llm", "flow"):
            state = torch.load(export / f"{part}.pt", weights_only=True)
            trained_state = torch.load(folder / "exp" / part / "epoch_1_whole.pt", weights_only=True) if part in parts else None
            assert "epoch" not in state and "step" not in state
            if trained_state is not None:
                assert all(torch.equal(state[k], trained_state[k]) for k in state)
            else:
                assert (export / f"{part}.pt").samefile(model_dir / f"{part}.pt")
        samples = [e for e in rec.of(OutputEvent) if e.kind == "sample"]
        assert len(samples) == 1 and Path(samples[0].path).is_file()
    # the export loads in the native engine and clones a voice
    from mizuoto.engines.cosyvoice import CosyVoiceEngine

    engine = CosyVoiceEngine()
    export = both / "export"
    assert engine.detect_variant(export) == VARIANTS[version]
    devices = DeviceService()
    loaded = engine.load(LoadSpec("cosyvoice/tiny-trained", export, VARIANTS[version], "cpu", options={}), Runtime(devices, ModelManager(devices)), RequestContext("load"))
    wav = both / "data" / "wavs" / "r1.wav"
    voice = VoiceSpec(references=[Reference(audio_path=str(wav), transcript=rows[1]["text"])])
    with torch.inference_mode():
        chunks = list(loaded.synthesize(SynthesisTask(text="hello there", language="en", voice=voice, params={"seed": 3}), RequestContext("r", seed=3)))
    audio = np.concatenate([c.samples for c in chunks])
    assert chunks[-1].final and audio.size > 0 and np.isfinite(audio).all()
    loaded.close()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_gradient_checkpointing_trains_identically(version, rows, tmp_path):
    from tests.engines_native.training_memory import check_checkpointing

    model_dir = _tiny(tmp_path / "model", version) if (tmp_path / "model").mkdir() is None else None
    assert model_dir is not None
    request = {
        "trainer": trainer_for("cosyvoice"),
        "engine": "cosyvoice",
        "method": "llm_flow",
        "experiment": "tiny",
        "rows": rows,
        "base_model": {"id": "cosyvoice/tiny", "dir": str(model_dir), "variant": VARIANTS[version]},
        "previews": False,
        "device": "cpu",
        "precision": "auto",
        "keep_checkpoints": 1,
        "hyperparams": {"epochs": 2, "grad_accumulation": 2, "workers": 2},
        "language": "en",
    }
    assert check_checkpointing(request, ["features", "llm", "flow"], tmp_path / "runs") > 0
