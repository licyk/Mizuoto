"""GPT-SoVITS speaker fine-tuning, native: upstream's recipe as v1's trainer (``kodama_engine_gptsovits.trainer``) runs it,
stage by stage, on the native modules. Each stage writes what upstream's script writes, where it writes it, so the two
can be compared file by file: the work folder's ``exp/`` is upstream's ``logs/<exp>``.

- ``text`` is ``prepare_datasets/1-get-text.py``: ``inputs.list`` (``path|speaker|language|text``), then per clip
  ``clean_text`` (version-2 symbols) on the text with ``%`` → ``-`` and ``￥`` → ``,``, and for Chinese the BERT features
  (chinese-roberta's third-last hidden state per character, repeated per phone) as ``3-bert/<name>.pt``; the lines
  ``name\\tphones\\tword2ph\\tnorm_text`` in ``2-name2text.txt``. ``<name>`` is the clip's file name, as upstream's.
- ``speech`` is ``2-get-hubert-wav32k.py`` (and ``2-get-sv.py`` for v2Pro/v2ProPlus): each clip at 32 kHz as the
  ``ffmpeg`` program decodes and resamples it (``mizuoto.nn.dsp.swresample``), skipped above a peak of 2.2, scaled to
  0.95 × 0.5 of full scale plus half of itself and written as 16-bit ``5-wav32k/<name>`` (a WAV under the clip's own
  name); the same scaled to 1145.14 and resampled to 16 kHz (librosa's soxr) for HuBERT, whose last hidden state is
  ``4-cnhubert/<name>.pt`` (1, 768, frames); the speaker embedding of the 32 kHz file at 16 kHz is ``7-sv_cn/<name>.pt``
  (1, 20480).
- ``semantic`` is ``3-get-semantic.py``: the base SoVITS's first codebook codes of each HuBERT file, in
  ``6-name2semantic.tsv`` under its header.
- ``sovits`` trains SoVITS (``train_s2.train_sovits``: ``s2_train`` or the v4/v5 LoRA ``s2_train_v3_lora``).
- ``gpt`` is ``s1_train.py`` (``train_s1``), half weights in ``weights/gpt/<experiment>-e<N>.ckpt``.
- ``export`` hands the trained pair to the host as the voice's weights and speaks a preview with them.

Not as upstream: a clip the text front end cannot handle is left out with its error in the log (upstream prints the
traceback and goes on); a clip with several channels is averaged (ffmpeg's ``-ac 1`` mixes by its own matrix; Mizuoto's
datasets are mono).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.engines.gptsovits.engine import BASE_WEIGHTS, ENCODERS, G2PW, NLTK, OPEN_JTALK, SV_FILE, VERSIONS
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

DISCRIMINATORS = {"v2": "gptsovits/s2d-v2", "v2pro": "gptsovits/s2d-v2pro", "v2proplus": "gptsovits/s2d-v2proplus"}
"""The pretrained discriminator each v2-family variant's SoVITS training starts from (a training asset)."""
LANGUAGES = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue"}
"""Mizuoto language → what upstream's ``.list`` takes."""
SAMPLE_RATE = 32000
"""``2-get-hubert-wav32k``'s rate."""
MAX_PEAK = 2.2
"""Clips peaking above this are skipped (``tmp_max > 2.2``)."""
MAXX, ALPHA = 0.95, 0.5


class GptSovitsTrainer(Trainer):
    ENGINE = "gptsovits"

    # -- helpers --------------------------------------------------------------------------------

    @staticmethod
    def _setup(ctx: StageContext) -> tuple[str, str, Path, dict[str, Path]]:
        """(variant, upstream version, model folder, assets), checked."""
        variant = str(ctx.base_model.get("variant", ""))
        if variant not in VERSIONS:
            raise EngineError(f"{ctx.base_model.get('id')} is not a GPT-SoVITS base model", code=codes.INVALID_INPUT)
        assets = {k: Path(v) for k, v in ctx.assets.items()}
        missing = [a for a in (ENCODERS, G2PW, NLTK, OPEN_JTALK) if a not in assets]
        if missing:
            raise EngineError(f"GPT-SoVITS needs {', '.join(missing)}", {"assets": missing}, code=codes.ASSET_MISSING)
        return variant, VERSIONS[variant], Path(ctx.base_model["dir"]), assets

    @staticmethod
    def _exp(ctx: StageContext) -> Path:
        exp = ctx.workdir / "exp"
        exp.mkdir(parents=True, exist_ok=True)
        return exp

    @staticmethod
    def _half(ctx: StageContext) -> bool:
        """Upstream's ``is_half``: on a GPU unless full precision is asked for."""
        return ctx.device.startswith("cuda") and ctx.precision in ("auto", "fp16")

    @staticmethod
    def _clear(*paths: Path) -> None:
        for p in paths:
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    @staticmethod
    def _lines(exp: Path) -> list[tuple[str, str]]:
        """``inputs.list``'s clips: (audio path, name), the name as upstream takes it (``basename(clean_path(path))``)."""
        path = exp / "inputs.list"
        if not path.is_file():
            raise EngineError("There are no prepared inputs; run the text stage", code=codes.INVALID_INPUT)
        out = []
        for line in path.read_text(encoding="utf-8").strip("\n").split("\n"):
            parts = line.split("|")
            if len(parts) == 4:
                audio = _clean_path(parts[0])
                out.append((audio, os.path.basename(audio)))
        return out

    def _batch(self, ctx: StageContext) -> int:
        batch = int(ctx.hp("batch_size", 4))
        # Upstream's web UI halves the batch when it trains in full precision.
        return batch if self._half(ctx) else max(1, batch // 2)

    @staticmethod
    def _newest(folder: Path, pattern: str) -> Path:
        found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
        if not found:
            raise EngineError(f"Training wrote no weights to {folder}", code=codes.ENGINE_ERROR)
        return found[-1]

    @staticmethod
    def _trained(ctx: StageContext, stage: str, folder: Path, pattern: str) -> Path:
        """The weights a training stage made: the file it named in its folder, the path it reported, else the newest
        file in its folder. The name comes first: the reported path is absolute, and stale once the experiments
        folder moves."""
        result = ctx.results.get(stage) or {}
        for candidate in (folder / str(result["weights"]) if isinstance(result.get("weights"), str) else None, result.get("path")):
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        if not folder.is_dir():
            raise EngineError(f"There are no trained {stage} weights; run the {stage} stage", code=codes.INVALID_INPUT)
        return GptSovitsTrainer._newest(folder, pattern)

    # -- stages ---------------------------------------------------------------------------------

    def stage_text(self, ctx: StageContext) -> dict[str, Any]:
        """Phones of every clip, and BERT features of the Chinese ones (1-get-text)."""
        import torch

        from mizuoto.engines.gptsovits.text import TextFrontend, TextPreprocessor, load_bert_model

        _, _, _, assets = self._setup(ctx)
        exp = self._exp(ctx)
        lines, skipped = [], 0
        for row in ctx.rows:
            language = LANGUAGES.get((row.language or ctx.request.get("language") or "").split("-")[0].lower())
            text = " ".join(row.text.replace("|", " ").split())
            if language is None or not text:
                skipped += 1
                continue
            lines.append(f"{row.audio}|{row.speaker or ctx.name}|{language}|{text}")
        if not lines:
            raise EngineError("No clip has text in a language GPT-SoVITS trains (zh, en, ja, ko, yue)", code=codes.INVALID_INPUT)
        if skipped:
            ctx.log(f"{skipped} clip(s) without text or in another language are left out", "warning")
        (exp / "inputs.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
        bert_dir = exp / "3-bert"
        self._clear(exp / "2-name2text.txt", bert_dir)
        bert_dir.mkdir(parents=True)
        frontend = TextFrontend.from_assets(assets, device="cpu", cache_dir=assets[OPEN_JTALK] / ".mizuoto")
        half = self._half(ctx)
        text_pre: Any = None
        if any(line.split("|")[2] == "zh" for line in lines):
            bert = load_bert_model(assets[ENCODERS], ctx.device, torch.float16 if half else torch.float32).eval()
            text_pre = TextPreprocessor(bert, ctx.device, frontend=frontend)
        out: list[str] = []
        errors: list[str] = []
        for i, line in enumerate(lines):
            audio, _, language, text = line.split("|")
            name = os.path.basename(_clean_path(audio))
            ctx.progress(i / len(lines))
            try:
                phones, word2ph, norm_text = frontend.clean_text(text.replace("%", "-").replace("￥", ","), language)
                path_bert = bert_dir / f"{name}.pt"
                if language == "zh" and not path_bert.exists():
                    assert word2ph is not None
                    feature = text_pre.get_bert_feature(norm_text, word2ph)
                    assert feature.shape[-1] == len(phones)
                    torch.save(feature, str(path_bert))
            except Exception as e:  # upstream prints the traceback and goes on with the next clip
                errors.append(f"{name}: {type(e).__name__}: {e}")
                ctx.log(f"{name} is left out: {type(e).__name__}: {e}", "warning")
                continue
            out.append(f"{name}\t{' '.join(phones)}\t{word2ph}\t{norm_text}")
        if not out:
            raise EngineError("The text front end handled no clip" + (f" ({errors[0]})" if errors else ""), {"errors": errors[:5]}, code=codes.ENGINE_ERROR)
        (exp / "2-name2text.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
        return {"clips": len(out), "left_out": len(lines) - len(out) + skipped}

    def stage_speech(self, ctx: StageContext) -> dict[str, Any]:
        """HuBERT features and 32 kHz audio (2-get-hubert-wav32k), and speaker embeddings for v2Pro (2-get-sv)."""
        import numpy as np
        import torch

        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.resample import soxr_resample
        from mizuoto.nn.dsp.swresample import resample as ffmpeg_resample
        from mizuoto.nn.ssl.hubert import load_hubert

        _, version, _, assets = self._setup(ctx)
        exp = self._exp(ctx)
        lines = self._lines(exp)
        hubert_dir, wav_dir, sv_dir = exp / "4-cnhubert", exp / "5-wav32k", exp / "7-sv_cn"
        self._clear(hubert_dir, wav_dir, sv_dir)
        hubert_dir.mkdir()
        wav_dir.mkdir()
        half = self._half(ctx)
        ctx.progress(0.0, "HuBERT features")
        hubert: Any = load_hubert(assets[ENCODERS] / "chinese-hubert-base").to(ctx.device, torch.float16 if half else torch.float32).eval()

        def features(name: str, path: str, dtype: torch.dtype) -> bool | None:
            """``name2go``: True when written, False for NaN features, None when skipped."""
            if (hubert_dir / f"{name}.pt").exists():
                return None
            audio, rate = read_audio(path)
            if rate != SAMPLE_RATE:
                audio = ffmpeg_resample(audio, rate, SAMPLE_RATE)  # load_audio's ffmpeg -ar 32000
            peak = np.abs(audio).max()
            if peak > MAX_PEAK:
                ctx.log(f"{name} is left out: its peak is {peak:.2f}", "warning")
                return None
            audio32 = (audio / peak * (MAXX * ALPHA * 32768)) + ((1 - ALPHA) * 32768) * audio
            audio32b = (audio / peak * (MAXX * ALPHA * 1145.14)) + ((1 - ALPHA) * 1145.14) * audio
            wav16 = torch.from_numpy(soxr_resample(audio32b, SAMPLE_RATE, 16000)).to(ctx.device, dtype)
            with torch.inference_mode():
                ssl = hubert(wav16.unsqueeze(0))[-1].transpose(1, 2).cpu()
            if torch.isnan(ssl.float()).any():
                return False
            _write_pcm16(wav_dir / name, audio32.astype("int16"), SAMPLE_RATE)
            torch.save(ssl.clone(), str(hubert_dir / f"{name}.pt"))
            return True

        nan_fails = []
        for i, (path, name) in enumerate(lines):
            ctx.progress(0.5 * i / len(lines) if "Pro" in version else i / len(lines))
            if features(name, path, torch.float16 if half else torch.float32) is False:
                nan_fails.append((name, path))
        if nan_fails and half:  # upstream retries those clips in full precision
            hubert = hubert.float()
            for name, path in nan_fails:
                features(name, path, torch.float32)
        count = len(list(hubert_dir.glob("*.pt")))
        if not count:
            raise EngineError("No speech features were made; see the log", code=codes.ENGINE_ERROR)
        hubert = _gone()  # off the device before the speaker model comes
        _empty_cache(ctx.device)
        if "Pro" in version:
            ctx.progress(0.5, "Speaker embeddings")
            self._speaker_embeddings(ctx, lines, assets, wav_dir, sv_dir, half)
        return {"clips": count}

    @staticmethod
    def _speaker_embeddings(ctx: StageContext, lines: list[tuple[str, str]], assets: dict[str, Path], wav_dir: Path, sv_dir: Path, half: bool) -> None:
        """``2-get-sv``: the 32 kHz file as torchaudio reads it, resampled to 16 kHz, ERes2NetV2's ``forward3``."""
        import torch

        from mizuoto.engines.gptsovits.sv import load_sv, sv_embedding
        from mizuoto.engines.gptsovits.text.frontend import find_one
        from mizuoto.nn.dsp.resample import Resample
        from mizuoto.nn.dsp.wav import read_wav

        sv_dir.mkdir()
        path = assets[ENCODERS] / SV_FILE
        dtype = torch.float16 if half else torch.float32
        model = load_sv(path if path.is_file() else find_one(assets[ENCODERS], "pretrained_eres2netv2w24s4ep4.ckpt")).to(ctx.device, dtype)
        resample = Resample(SAMPLE_RATE, 16000).to(ctx.device)
        for i, (_, name) in enumerate(lines):
            ctx.progress(0.5 + 0.5 * i / len(lines))
            target = sv_dir / f"{name}.pt"
            if target.exists() or not (wav_dir / name).is_file():
                continue
            samples, rate = read_wav(wav_dir / name)
            assert rate == SAMPLE_RATE
            with torch.inference_mode():
                wav16 = resample(torch.from_numpy(samples).unsqueeze(0).to(ctx.device)).to(dtype)
                emb = sv_embedding(model, wav16).cpu()
            torch.save(emb.clone(), str(target))

    def stage_semantic(self, ctx: StageContext) -> dict[str, Any]:
        """The base SoVITS's semantic tokens of every clip (3-get-semantic)."""
        import torch

        variant, version, model_dir, _ = self._setup(ctx)
        exp = self._exp(ctx)
        lines = self._lines(exp)
        self._clear(exp / "6-name2semantic.tsv")
        half = self._half(ctx)
        dtype = torch.float16 if half else torch.float32
        model = _semantic_model(model_dir / BASE_WEIGHTS[variant][1], version, ctx.device, dtype)
        out = []
        for i, (_, name) in enumerate(lines):
            ctx.progress(i / len(lines))
            path = exp / "4-cnhubert" / f"{name}.pt"
            if not path.exists():
                continue
            ssl = torch.load(str(path), map_location="cpu", weights_only=True).to(ctx.device, dtype)
            with torch.inference_mode():
                codes_ = model.extract_latent(ssl)
            out.append(f"{name}\t{' '.join(str(c) for c in codes_[0, 0, :].tolist())}")
        if not out:
            raise EngineError("No semantic tokens were made; see the log", code=codes.ENGINE_ERROR)
        (exp / "6-name2semantic.tsv").write_text("\n".join(["item_name\tsemantic_audio", *out]) + "\n", encoding="utf-8")
        return {"clips": len(out)}

    def stage_sovits(self, ctx: StageContext) -> dict[str, Any]:
        """SoVITS training (s2_train, or s2_train_v3_lora for v4 and v5), from the base model's weights."""
        from mizuoto.engines.gptsovits.train_s2 import train_sovits

        variant, version, model_dir, assets = self._setup(ctx)
        exp = self._exp(ctx)
        if not (exp / "2-name2text.txt").is_file() or not (exp / "4-cnhubert").is_dir():
            raise EngineError("There are no prepared clips; run the text and speech stages", code=codes.INVALID_INPUT)
        disc = DISCRIMINATORS.get(variant)
        if disc is not None and disc not in assets:
            # upstream starts from a random discriminator without it; v1 refused, and so does this
            raise EngineError(f"Training {version} needs {disc}", {"assets": [disc], "hint": f"mizuoto asset download {disc}"}, code=codes.ASSET_MISSING)
        out = ctx.workdir / "weights" / "sovits"  # v1's SoVITS_weights folder
        self._clear(out)
        result = dict(train_sovits(ctx, exp, version, dict(ctx.hyperparams), model_dir / BASE_WEIGHTS[variant][1], dict(assets), save_dir=out, name=ctx.name))
        weights = Path(result.pop("weights"))
        if not weights.is_file():
            raise EngineError(f"SoVITS training wrote no weights ({weights})", code=codes.ENGINE_ERROR)
        return {**_jsonable(result), "weights": weights.name, "path": str(weights), "epochs": int(ctx.hp("sovits_epochs", 8))}

    def stage_gpt(self, ctx: StageContext) -> dict[str, Any]:
        """GPT training (s1_train) from the base model's weights."""
        from mizuoto.engines.gptsovits.train_s1 import S1Run, train_gpt

        variant, version, model_dir, _ = self._setup(ctx)
        exp = self._exp(ctx)
        for f in ("2-name2text.txt", "6-name2semantic.tsv"):
            if not (exp / f).is_file():
                raise EngineError(f"{f} is missing; run the earlier stages", code=codes.INVALID_INPUT)
        epochs = int(ctx.hp("gpt_epochs", 15))
        out = ctx.workdir / "weights" / "gpt"
        self._clear(out)
        run = S1Run(
            exp_dir=exp,
            weights_dir=out,
            output_dir=exp / f"logs_s1_{version}",
            pretrained=model_dir / BASE_WEIGHTS[variant][0],
            exp_name=ctx.name,
            epochs=epochs,
            batch_size=self._batch(ctx),
            save_every_n_epoch=epochs,
            dpo=bool(ctx.hp("dpo", False)),
            precision="16-mixed" if self._half(ctx) else "32",
            device=ctx.device,
        )

        def on_step(epoch: int, step: int, batches: int, values: dict[str, float], lr: float) -> None:
            ctx.metric(epoch, step, values, lr)
            ctx.progress(min(1.0, step / max(1, batches * epochs)), f"Epoch {epoch} of {epochs}")

        ctx.progress(0.0, f"Epoch 1 of {epochs}")
        result = train_gpt(run, on_step=on_step)
        weights = Path(result["weights"][-1])
        ctx.log(f"GPT: {result['clips']} clips ({result['samples']} samples), batches of {result['batch_size']}, {result['steps']} steps")
        return {"weights": weights.name, "path": str(weights), "epochs": epochs, "steps": result["steps"]}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """The trained SoVITS and GPT weights as the voice's own pair, and a preview spoken with them."""
        sovits = self._trained(ctx, "sovits", ctx.workdir / "weights" / "sovits", "*.pth")
        gpt = self._trained(ctx, "gpt", ctx.workdir / "weights" / "gpt", "*.ckpt")
        folder = ctx.workdir / "export"
        self._clear(folder)
        folder.mkdir(parents=True)
        files = {"gpt": folder / f"{ctx.name}.ckpt", "sovits": folder / f"{ctx.name}.pth"}
        shutil.copyfile(gpt, files["gpt"])
        shutil.copyfile(sovits, files["sovits"])
        reference = _reference(ctx.rows)
        if reference is not None:
            text = str(ctx.hp("preview_text", "") or "") or reference.text
            ctx.preview(text, reference, "Preview", weights={k: str(v) for k, v in files.items()})
        return {"artifacts": [{"kind": "voice_weights", "files": {k: str(v) for k, v in files.items()}}], "gpt": gpt.name, "sovits": sovits.name}


# -- helpers ------------------------------------------------------------------------------------


def _clean_path(path: str) -> str:
    """``tools/my_utils.clean_path``."""
    if path.endswith(("\\", "/")):
        return _clean_path(path[:-1])
    return path.replace("/", os.sep).replace("\\", os.sep).strip(" '\n\"‪")


def _write_pcm16(path: Path, samples: Any, rate: int) -> None:
    """``scipy.io.wavfile.write(path, rate, int16 array)``: a plain 16-bit PCM WAV."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(samples.astype("<i2").tobytes())


def _semantic_model(path: Path, version: str, device: str, dtype: Any) -> Any:
    """The base SoVITS, for ``extract_latent``: only its SSL projection and quantizer go to the device; the rest (most
    of it, 730 MB for v4/v5) stays in RAM in its dtype."""
    if version in ("v4", "v5turbo"):
        from mizuoto.engines.gptsovits.sovits_v3 import load_synthesizer

        model = load_synthesizer(path, version, device="cpu", dtype=dtype)
    else:
        from mizuoto.engines.gptsovits.sovits import load_synthesizer as load_v2

        model = load_v2(path)[0].to(dtype).eval()
    model.ssl_proj.to(device)
    model.quantizer.to(device)
    return model


def _gone() -> Any:
    """What stands in for a model let go (its last reference dropped)."""
    return None


def _empty_cache(device: str) -> None:
    """Hand a GPU's cached memory back (the next model of the stage needs it)."""
    import gc

    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """What can travel in a stage result: plain numbers, strings (paths as strings), booleans and short lists of them."""

    def plain(v: Any) -> Any:
        return str(v) if isinstance(v, os.PathLike) else v

    out: dict[str, Any] = {}
    for k, v in result.items():
        v = plain(v)
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 20 and all(isinstance(plain(x), (int, float, str, bool)) for x in v):
            out[k] = [plain(x) for x in v]
    return out


def _reference(rows: list[Row]) -> Row | None:
    """A clip for the preview: with text, 3–10 s (upstream's reference limits), nearest 6 s."""
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 10.0]
    return min(fitting, key=lambda r: abs(r.duration - 6.0)) if fitting else None
