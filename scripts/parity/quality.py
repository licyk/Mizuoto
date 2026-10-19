"""T4, the statistical tier: a corpus synthesised by two implementations in their production dtype and device, scored,
and the two compared against the tier's thresholds (site page ``development-testing``).

    <python> scripts/parity/quality.py run --impl native|reference|upstream [--device cuda:0] [--name NAME] [--only SET]
        [--seeds 3] [--texts N] [--voice V] [--language zh] [--unset-language] [--setting key=value] [--option key=value]
        [--suite corpus/t4.json]
    <python> scripts/parity/quality.py score [--name NAME]... [--device cuda:0] [--only SET]
    <python> scripts/parity/quality.py compare [--a reference] [--b native] [--only SET] [--report out.md]

``run`` synthesises each set of the suite (``corpus/t4.json``): every voice reads every text, once per seed (1, 2, 3),
through a whole stack (``stack.py``'s ``build`` and ``run_case``), with the engine's own dtype and settings. ``<python>``
decides which implementation runs, as for ``stack.py``: this checkout's interpreter for ``native``; the reference host for
``reference`` (the sets whose ``reference`` is ``v1``, with ``MIZUOTO_PARITY_DATA`` pointing at the data folder the set
names, ``data-t28`` for CosyVoice); an environment with the upstream package for ``upstream`` (VoxCPM). Clips go to
``<workspace>/quality/<name>/<set>/<voice>-<text>-s<seed>.wav`` with a ``.json`` record beside each; a clip already there is
kept, so a run that stops resumes where it stopped. The default name is ``native`` for the native side and ``reference``
for both reference kinds. ``--setting`` (``compute.attention=fast``) or ``--option`` (a request option, OmniVoice's
``packed_guidance=true``) runs with an opt-in, under its own ``--name``, to be compared with the default run as ``a``.

``score`` (this checkout's interpreter) measures each clip once, into ``<clip>.score.json``:

- the transcript by Whisper large-v3-turbo (Mizuoto's ``asr`` engine, the clip's language given) and the error rate against
  the text: words for English, characters for Chinese (punctuation and case dropped, traditional characters mapped to
  simplified ones with OpenCC's ``TSCharacters.txt``, fetched once into the workspace);
- the CAM++ speaker embedding (CosyVoice's ``campplus.onnx`` through ``mizuoto.nn.speaker.campplus``) and, for cloned
  voices, its cosine to the voice's reference clip;
- UTMOS (UTMOS22 strong, SpeechMOS's port through ``torch.hub``; its one torchaudio call, a resample, is given 16 kHz audio
  and served by ``mizuoto.nn.dsp.resample``); skipped with a note when it cannot be loaded;
- the duration, the integrated loudness (ITU-R BS.1770-4: K-weighting applied in the frequency domain with the filters'
  exact responses, 400 ms blocks, absolute and relative gates) and the median F0 of voiced frames (YIN on 16 kHz audio,
  60–500 Hz, threshold 0.15) with the voiced fraction.

``compare`` judges side ``b`` against side ``a`` per set, over the clips both have. A clip whose error rate is above 50 %
failed (a runaway, the wrong language, silence); its error rate is unbounded, so one would decide every mean. Failures
are compared on their own, with Fisher's exact test: ``b`` fails the set only when it fails more often and p < 0.05 (one
bad draw in a sampled model is chance; a targeted run with more seeds, ``--voice``, ``--language``, ``--seeds 10``, shows
whether failures are more frequent). The other measures judge each side's clips that did not fail:

- error rate: ``b``'s mean ≤ ``a``'s + 0.5 pp, and ``b``'s worst clip ≤ ``a``'s worst + 5 pp;
- speaker similarity: mean ≥ ``a``'s − 0.01 (cloned voices: to the reference clip; preset and designed voices: to the
  centroid of ``a``'s other clips of that voice: both sides' clip of a text and seed against the centroid without ``a``'s);
- duration: the ratio of the means within ±5 %;
- UTMOS: mean ≥ ``a``'s − 0.05;
- loudness and F0: for every voice, ``b``'s median within ``a``'s interquartile range of that voice's clips;

and reports, for each paired measure (by voice and text, seeds averaged; pairs neither side failed), the two-sided
Wilcoxon signed-rank p-value (normal approximation, ties corrected), which says whether a difference is systematic, not
whether it matters. A threshold exceeded fails the set only when that test (the error rate's for the worst clip) says
the difference is systematic, p < 0.05; otherwise the verdict passes and names the check as within the sampling noise.
Loudness and F0 have no paired test: outside the spread always fails.

The workspace is ``MIZUOTO_PARITY_DIR`` (default ``~/code_workspace/experiment/kodama-refactor``).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack import WORKSPACE, build, parse_settings, real_data_dir, refs, run_case

SUITE = Path(__file__).resolve().parent / "corpus" / "t4.json"
SEEDS = (1, 2, 3)
OPENCC_TS = "https://raw.githubusercontent.com/BYVoid/OpenCC/ver.1.1.9/data/dictionary/TSCharacters.txt"
UTMOS_HUB = "tarepan/SpeechMOS:v1.2.0"
FAILED_PP = 50.0
"""A clip whose error rate is above this (%) failed: a runaway, a wrong language, silence."""
THRESHOLDS = {"error_pp": 0.5, "worst_pp": 5.0, "similarity": 0.01, "duration": 0.05, "utmos": 0.05}


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_voices(suite: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("voices") or suite["voices"][spec["mode"]]


def clips(suite: dict[str, Any], spec: dict[str, Any], seeds: tuple[int, ...], texts: int = 0) -> list[tuple[str, str, int]]:
    """(voice, text id, seed) for every clip of a set, voice by voice; ``texts`` > 0: the first that many texts of each
    language (a quick run)."""
    chosen = list(suite["texts"])
    if texts > 0:
        by: dict[str, list[str]] = {}
        for t in chosen:
            by.setdefault(suite["texts"][t][0], []).append(t)
        chosen = [t for ts in by.values() for t in ts[:texts]]
    return [(v, t, s) for v in set_voices(suite, spec) for t in chosen for s in seeds]


def stem(voice: str, text: str, seed: int) -> str:
    return f"{voice}-{text}-s{seed}"


def default_name(impl: str) -> str:
    return "native" if impl == "native" else "reference"


def runs_here(spec: dict[str, Any], impl: str) -> bool:
    if impl == "native":
        return True
    return spec["reference"] == ("v1" if impl == "reference" else "upstream")


# -- run ------------------------------------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    name = args.name or default_name(args.impl)
    table = refs()
    seeds = tuple(range(1, args.seeds + 1))
    chosen = {k: v for k, v in suite["sets"].items() if (not args.only or k in args.only) and runs_here(v, args.impl)}
    if not chosen:
        print(f"no set runs with --impl {args.impl}" + (f" among {args.only}" if args.only else ""))
        return 1
    tmp = Path(tempfile.mkdtemp(prefix=f"t4-{args.impl}-"))
    settings = parse_settings(args.setting or [])
    options = dict(_value(pair) for pair in args.option or [])
    s = build(args.device, tmp, args.impl, settings)
    failures = 0
    try:
        for key, spec in chosen.items():
            folder = WORKSPACE / "quality" / name / key
            folder.mkdir(parents=True, exist_ok=True)
            todo = [
                c
                for c in clips(suite, spec, seeds, args.texts)
                if (not args.voice or c[0] in args.voice) and (not args.language or suite["texts"][c[1]][0] == args.language) and not (folder / f"{stem(*c)}.wav").exists()
            ]
            print(f"[{name}] {key} ({spec['model']}): {len(todo)} clips to make", flush=True)
            started = time.perf_counter()
            for n, (voice, text_id, seed) in enumerate(todo, 1):
                language, text = suite["texts"][text_id]
                request = {"model": spec["model"], "text": text, "language": language, "params": {"seed": seed}, "voice_spec": set_voices(suite, spec)[voice]}
                if args.unset_language:  # the engine's default for a request without one; scored in the text's language
                    del request["language"]
                if options:
                    request["engine_options"] = options
                case = {"id": f"t4-{key}-{stem(voice, text_id, seed)}", "request": request}
                record: dict[str, Any]
                try:
                    path, record = run_case(s, case, table)
                except Exception as exc:  # recorded, so one bad clip does not stop the corpus
                    path = None
                    record = {"case": case, "job_state": "exception", "error": {"message": repr(exc), "traceback": traceback.format_exc()}}
                record.update(
                    {
                        "impl": args.impl,
                        "settings": settings,
                        "options": options,
                        "device": args.device,
                        "set": key,
                        "voice": voice,
                        "text_id": text_id,
                        "seed": seed,
                        "language": language,
                        "text": text,
                    }
                )
                if path is None:
                    failures += 1
                    print(f"    {stem(voice, text_id, seed)}: {record['job_state']}: {(record.get('error') or {}).get('message')}", flush=True)
                    (folder / f"{stem(voice, text_id, seed)}.failed.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                    continue
                (folder / f"{stem(voice, text_id, seed)}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                shutil.copyfile(path, folder / f"{stem(voice, text_id, seed)}.wav")  # last: its presence means done
                if n % 20 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)} in {time.perf_counter() - started:.0f} s", flush=True)
    finally:
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


def _value(pair: str) -> tuple[str, Any]:
    """``key=value``, the value read as JSON when it parses (``true``, ``8``), else as a string."""
    key, _, raw = pair.partition("=")
    try:
        return key.strip(), json.loads(raw)
    except json.JSONDecodeError:
        return key.strip(), raw


# -- measures -------------------------------------------------------------------------------------------------------


def read_audio(path: Path) -> tuple[Any, int]:
    """Mono float32 samples and their rate (WAV or FLAC)."""
    import numpy as np

    from mizuoto.core.audio.io import decode

    samples, rate = decode(path)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1 if samples.shape[-1] <= 8 else 0)
    return samples, rate


def resampled(samples: Any, rate: int, target: int) -> Any:
    import torch

    from mizuoto.nn.dsp.resample import resample

    x = torch.from_numpy(samples).float().unsqueeze(0)
    return (x if rate == target else resample(x, rate, target))[0].numpy()


def loudness(samples: Any, rate: int) -> float | None:
    """Integrated loudness, LUFS (ITU-R BS.1770-4, mono). The two K-weighting biquads (pyloudnorm's coefficients for any
    rate) are applied through their exact frequency response on a zero-padded FFT: the filters' tails are milliseconds
    long, so this is the causal filtering up to rounding."""
    import numpy as np

    if len(samples) < int(0.4 * rate):
        return None

    def biquad(kind: str, gain_db: float, q: float, fc: float) -> tuple[Any, Any]:
        a_ = 10 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * fc / rate
        alpha, cos = math.sin(w0) / (2.0 * q), math.cos(w0)
        if kind == "shelf":
            b = [a_ * ((a_ + 1) + (a_ - 1) * cos + 2 * math.sqrt(a_) * alpha), -2 * a_ * ((a_ - 1) + (a_ + 1) * cos), a_ * ((a_ + 1) + (a_ - 1) * cos - 2 * math.sqrt(a_) * alpha)]
            a = [(a_ + 1) - (a_ - 1) * cos + 2 * math.sqrt(a_) * alpha, 2 * ((a_ - 1) - (a_ + 1) * cos), (a_ + 1) - (a_ - 1) * cos - 2 * math.sqrt(a_) * alpha]
        else:
            b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
            a = [1 + alpha, -2 * cos, 1 - alpha]
        return np.array(b), np.array(a)

    n = len(samples) + rate  # a second of padding: no wrap-around
    w = np.exp(-1j * np.pi * np.arange(n // 2 + 1) / (n / 2))
    response = np.ones(n // 2 + 1, dtype=complex)
    for b, a in (biquad("shelf", 4.0, 1 / math.sqrt(2), 1500.0), biquad("pass", 0.0, 0.5, 38.0)):
        response *= (b[0] + b[1] * w + b[2] * w**2) / (a[0] + a[1] * w + a[2] * w**2)
    y = np.fft.irfft(np.fft.rfft(samples.astype(np.float64), n) * response, n)[: len(samples)]
    block, step = int(0.4 * rate), int(0.1 * rate)
    z = np.array([np.mean(y[i : i + block] ** 2) for i in range(0, len(y) - block + 1, step)])
    z = z[z > 0]
    if not len(z):
        return None
    lk = -0.691 + 10 * np.log10(z)
    z = z[lk > -70.0]
    if not len(z):
        return None
    gate = -0.691 + 10 * np.log10(z.mean()) - 10.0
    z = z[-0.691 + 10 * np.log10(z) > gate]
    return float(-0.691 + 10 * np.log10(z.mean()))


def pitch(samples16: Any) -> tuple[float | None, float]:
    """Median F0 (Hz) of the voiced frames and the voiced fraction: YIN on 16 kHz audio, 32 ms windows every 10 ms,
    60–500 Hz, threshold 0.15, frames below 3 % of the loudest frame's RMS unvoiced."""
    import numpy as np

    rate, win, hop = 16000, 512, 160
    tmin, tmax = rate // 500, rate // 60
    span = win + tmax
    x = samples16.astype(np.float64)
    if len(x) < span:
        return None, 0.0
    starts = np.arange(0, len(x) - span + 1, hop)
    frames = np.stack([x[i : i + span] for i in starts])
    n = 1 << int(math.ceil(math.log2(2 * span)))
    head = np.zeros_like(frames)
    head[:, :win] = frames[:, :win]
    r = np.fft.irfft(np.conj(np.fft.rfft(head, n)) * np.fft.rfft(frames, n), n)[:, : tmax + 1]
    sq = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames**2, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    e0 = sq[:, win][:, None]
    et = sq[:, taus + win] - sq[:, taus]
    d = e0 + et - 2 * r
    d[:, 0] = 0
    cmnd = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.where(cum > 0, cum, 1)
    rms = np.sqrt(e0[:, 0] / win)
    loud = rms > 0.03 * rms.max() if rms.max() > 0 else np.zeros(len(rms), bool)
    f0 = []
    for i in range(len(frames)):
        if not loud[i]:
            continue
        below = np.nonzero(cmnd[i, tmin:] < 0.15)[0]
        if not len(below):
            continue
        t = tmin + below[0]
        while t + 1 <= tmax and cmnd[i, t + 1] < cmnd[i, t]:
            t += 1
        if 0 < t < tmax:  # parabolic interpolation around the minimum
            a, b, c = cmnd[i, t - 1], cmnd[i, t], cmnd[i, t + 1]
            shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            f0.append(rate / (t + shift))
        else:
            f0.append(rate / t)
    if not f0:
        return None, 0.0
    return float(np.median(f0)), len(f0) / len(frames)


def normalise(text: str, language: str, ts: dict[str, str]) -> list[str]:
    """Words (English) or characters (Chinese), lower-cased, apostrophes and other punctuation dropped, traditional
    characters simplified."""
    import unicodedata

    text = "".join(ts.get(ch, ch) for ch in unicodedata.normalize("NFKC", text).lower()).replace("'", "").replace("’", "")
    kept = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    if language == "zh":
        return [ch for ch in kept if not ch.isspace()]
    return kept.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def opencc_ts() -> dict[str, str]:
    """Traditional → simplified characters (OpenCC's table, Apache-2.0), fetched once into the workspace."""
    path = WORKSPACE / "quality" / "TSCharacters.txt"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OPENCC_TS, timeout=60) as r:
            path.write_bytes(r.read())
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and len(parts[0]) == 1:
            table[parts[0]] = parts[1].split()[0]
    return table


class Scorer:
    """The judges, loaded once: Whisper (through a stack's ASR service), CAM++ and UTMOS."""

    def __init__(self, device: str) -> None:
        import torch

        self.device = device
        self.tmp = Path(tempfile.mkdtemp(prefix="t4-score-"))
        self.services = build(device, self.tmp, "native")
        self.ts = opencc_ts()
        self.campplus = self._campplus().to(device)
        self.utmos, self.utmos_note = self._utmos()
        self.ref_embeddings: dict[str, Any] = {}
        torch.set_grad_enabled(False)

    def close(self) -> None:
        self.services.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _campplus(self) -> Any:
        from mizuoto.engines.cosyvoice.engine import LoadedCosyVoice
        from mizuoto.nn.speaker.campplus import CAMPPlus, state_dict_from_onnx
        from mizuoto.runtime import ops
        from mizuoto.runtime.loading import load_into

        model = CAMPPlus(ops=ops.default)
        folder = real_data_dir() / "models" / "cosyvoice" / "cosyvoice2-0.5b"
        load_into(model, LoadedCosyVoice._onnx_weights(folder, "campplus.onnx", state_dict_from_onnx), ignore=["*.num_batches_tracked"], what="CAM++")
        return model.eval()

    def _utmos(self) -> tuple[Any, str | None]:
        import importlib.machinery
        import types

        import torch

        from mizuoto.nn.dsp.resample import resample

        def resample16(wave: Any, orig_freq: int, new_freq: int) -> Any:
            return wave if orig_freq == new_freq else resample(wave, orig_freq, new_freq)

        stub = types.ModuleType("torchaudio")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)  # torch.hub checks the dependency with find_spec
        setattr(stub, "functional", types.SimpleNamespace(resample=resample16))  # noqa: B010 (a module's attribute)
        saved = sys.modules.get("torchaudio")
        sys.modules["torchaudio"] = stub
        try:
            model = torch.hub.load(UTMOS_HUB, "utmos22_strong", trust_repo=True, verbose=False)
            return model.to(self.device).eval(), None
        except Exception as e:
            return None, f"UTMOS unavailable: {e!r}"
        finally:
            if saved is None:
                sys.modules.pop("torchaudio", None)
            else:
                sys.modules["torchaudio"] = saved

    def embedding(self, samples16: Any) -> Any:
        import torch

        from mizuoto.nn.dsp.kaldi import fbank

        feat = fbank(torch.from_numpy(samples16).float().unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        e = self.campplus(feat.unsqueeze(0).to(self.device)).flatten().float().cpu()
        return e / e.norm()

    def ref_embedding(self, ref_id: str) -> Any:
        if ref_id not in self.ref_embeddings:
            entry = refs()[ref_id]
            samples, rate = read_audio(WORKSPACE / "corpus" / "refs" / entry["file"])
            self.ref_embeddings[ref_id] = self.embedding(resampled(samples, rate, 16000))
        return self.ref_embeddings[ref_id]

    def score(self, wav: Path, record: dict[str, Any], voice_spec: dict[str, Any]) -> dict[str, Any]:
        import torch

        samples, rate = read_audio(wav)
        s16 = resampled(samples, rate, 16000)
        language = record["language"]
        transcript = self.services.asr.transcribe(wav, language=language).text
        want, got = normalise(record["text"], language, self.ts), normalise(transcript, language, self.ts)
        out: dict[str, Any] = {
            "duration": len(samples) / rate,
            "transcript": transcript,
            "unit": "char" if language == "zh" else "word",
            "errors": edit_distance(want, got),
            "length": len(want),
            "error_rate": edit_distance(want, got) / max(1, len(want)),
            "lufs": loudness(samples, rate),
        }
        out["f0_median"], out["voiced"] = pitch(s16)
        e = self.embedding(s16)
        out["embedding"] = [round(float(v), 6) for v in e]
        references = voice_spec.get("references") or []
        if references:
            out["similarity_ref"] = float(e @ self.ref_embedding(references[0]["ref"]))
        if self.utmos is not None:
            out["utmos"] = float(self.utmos(torch.from_numpy(s16).float().unsqueeze(0).to(self.device), 16000).flatten()[0])
        return out


def cmd_score(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    names = args.name or ["reference", "native"]
    scorer = Scorer(args.device)
    if scorer.utmos_note:
        print(scorer.utmos_note, flush=True)
    try:
        for name in names:
            for key, spec in suite["sets"].items():
                folder = WORKSPACE / "quality" / name / key
                if (args.only and key not in args.only) or not folder.is_dir():
                    continue
                voices = set_voices(suite, spec)
                todo = [w for w in sorted(folder.glob("*.wav")) if not w.with_suffix(".score.json").exists()]
                print(f"[score {name}] {key}: {len(todo)} clips", flush=True)
                for wav in todo:
                    record = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
                    out = scorer.score(wav, record, voices[record["voice"]])
                    wav.with_suffix(".score.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    finally:
        scorer.close()
    return 0


# -- compare --------------------------------------------------------------------------------------------------------


def wilcoxon(a: list[float], b: list[float]) -> float | None:
    """Two-sided p-value of the Wilcoxon signed-rank test on the pairs' differences (zero differences dropped; the
    normal approximation with the tie correction)."""
    d = [y - x for x, y in zip(a, b, strict=True) if y != x]
    n = len(d)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    w = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return None
    z = (w - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value of Fisher's exact test: ``k1`` of ``n1`` against ``k2`` of ``n2`` (the tables no likelier than
    the one seen)."""
    total, k = n1 + n2, k1 + k2

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)

    seen = prob(k1)
    return min(1.0, sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= seen * (1 + 1e-9)))


def quartiles(values: list[float]) -> tuple[float, float, float]:
    v = sorted(values)

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    return q(0.25), q(0.5), q(0.75)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_scores(name: str, key: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    folder = WORKSPACE / "quality" / name / key
    out = {}
    for path in folder.glob("*.score.json") if folder.is_dir() else []:
        record = json.loads(path.with_name(path.name.replace(".score.json", ".json")).read_text(encoding="utf-8"))
        out[(record["voice"], record["text_id"], record["seed"])] = json.loads(path.read_text(encoding="utf-8"))
    return out


def judge(spec: dict[str, Any], a: dict[tuple[str, str, int], dict[str, Any]], b: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    keys = sorted(set(a) & set(b))
    voices = sorted({k[0] for k in keys})
    res: dict[str, Any] = {"clips": len(keys), "missing_a": len(set(b) - set(a)), "missing_b": len(set(a) - set(b)), "checks": {}, "p": {}}
    if not keys:
        return res

    def sim(side: dict[tuple[str, str, int], dict[str, Any]], key: tuple[str, str, int]) -> float:
        if "similarity_ref" in side[key]:
            return side[key]["similarity_ref"]
        pool = [np.array(a[k]["embedding"]) for k in keys if k[0] == key[0] and k != key and k not in failed_a]
        centroid = np.mean(pool, axis=0)
        return float(np.array(side[key]["embedding"]) @ centroid / np.linalg.norm(centroid))

    # A failed clip (a runaway, the wrong language, silence) has an unbounded error rate, so a single one would decide
    # every mean: failures are counted and compared on their own, and the measures judge each side's other clips.
    failed_a = {k for k in keys if a[k]["error_rate"] * 100 > FAILED_PP}
    failed_b = {k for k in keys if b[k]["error_rate"] * 100 > FAILED_PP}
    ka, kb = [k for k in keys if k not in failed_a], [k for k in keys if k not in failed_b]
    both = [k for k in keys if k not in failed_a and k not in failed_b]
    res["failures"] = (len(failed_a), len(failed_b))
    res["p"]["failures"] = fisher(len(failed_a), len(keys), len(failed_b), len(keys))
    measures: dict[str, Any] = {
        "error_rate": lambda side, k: side[k]["error_rate"] * 100,
        "similarity": sim,
        "duration": lambda side, k: side[k]["duration"],
    }
    if all("utmos" in a[k] and "utmos" in b[k] for k in keys):
        measures["utmos"] = lambda side, k: side[k]["utmos"]
    per = {m: ([f(a, k) for k in ka], [f(b, k) for k in kb]) for m, f in measures.items()}
    res["means"] = {m: (mean(x), mean(y)) for m, (x, y) in per.items()}
    ea, eb = per["error_rate"]
    res["worst"] = (max(ea, default=0.0), max(eb, default=0.0))
    checks = res["checks"]
    checks["failures"] = len(failed_b) <= len(failed_a) or res["p"]["failures"] >= 0.05
    checks["error_rate"] = mean(eb) <= mean(ea) + THRESHOLDS["error_pp"]
    checks["worst"] = res["worst"][1] <= res["worst"][0] + THRESHOLDS["worst_pp"]
    checks["similarity"] = mean(per["similarity"][1]) >= mean(per["similarity"][0]) - THRESHOLDS["similarity"]
    ratio = mean(per["duration"][1]) / mean(per["duration"][0])
    res["duration_ratio"] = ratio
    checks["duration"] = abs(ratio - 1) <= THRESHOLDS["duration"]
    if "utmos" in per:
        checks["utmos"] = mean(per["utmos"][1]) >= mean(per["utmos"][0]) - THRESHOLDS["utmos"]
    spread: dict[str, dict[str, Any]] = {}
    for measure in ("lufs", "f0_median"):
        ok = True
        for v in voices:
            xa = [a[k][measure] for k in ka if k[0] == v and a[k].get(measure) is not None]
            xb = [b[k][measure] for k in kb if k[0] == v and b[k].get(measure) is not None]
            if len(xa) < 4 or not xb:
                continue
            q1, med, q3 = quartiles(xa)
            mb = quartiles(xb)[1]
            spread.setdefault(measure, {})[v] = (q1, med, q3, mb)
            ok = ok and q1 <= mb <= q3
        checks[measure] = ok
    res["spread"] = spread
    # paired by (voice, text), seeds averaged, over the clips neither side failed
    pairs = sorted({(k[0], k[1]) for k in both})
    for m, f in measures.items():
        by: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for k in both:
            by.setdefault((k[0], k[1]), []).append((f(a, k), f(b, k)))
        res["p"][m] = wilcoxon([mean([p[0] for p in by[q]]) for q in pairs], [mean([p[1] for p in by[q]]) for q in pairs])
    # a threshold exceeded fails the set only when the difference is systematic: the measure's paired test (the error
    # rate's for the worst clip, Fisher's for the failures) at p < 0.05; otherwise it is within the sampling noise
    test = {"worst": "error_rate", "lufs": None, "f0_median": None}
    res["noise"] = [c for c, ok in checks.items() if not ok and test.get(c, c) is not None and (res["p"].get(test.get(c, c)) is None or res["p"][test.get(c, c)] >= 0.05)]
    res["pass"] = all(ok or c in res["noise"] for c, ok in checks.items())
    return res


def cmd_compare(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    rows = []
    results = {}
    for key, spec in suite["sets"].items():
        if args.only and key not in args.only:
            continue
        a, b = load_scores(args.a, key), load_scores(args.b, key)
        if not a or not b:
            rows.append(f"| {key} | {spec['model']} | {len(a)} / {len(b)} clips scored | | | | | | | | — |")
            continue
        r = judge(spec, a, b)
        results[key] = r
        m = r["means"]

        def p(name: str, r: dict[str, Any] = r) -> str:
            v = r["p"].get(name)
            return "" if v is None else f" (p {v:.2g})"

        def mark(name: str, r: dict[str, Any] = r) -> str:
            return "" if r["checks"].get(name, True) else " ✗"

        utmos = f"{m['utmos'][0]:.2f} → {m['utmos'][1]:.2f}{p('utmos')}{mark('utmos')}" if "utmos" in m else "—"
        failed = [c for c, ok in r["checks"].items() if not ok and c not in r["noise"]]
        verdict = "**pass**" if not r["noise"] else "**pass** (within noise: " + ", ".join(r["noise"]) + ")"
        rows.append(
            f"| {key} | {spec['model']} | {r['clips']} | {m['error_rate'][0]:.2f} → {m['error_rate'][1]:.2f}{p('error_rate')}{mark('error_rate')}"
            f" | {r['worst'][0]:.1f} / {r['worst'][1]:.1f}{mark('worst')} | {r['failures'][0]} / {r['failures'][1]}{p('failures')}{mark('failures')}"
            f" | {m['similarity'][0]:.3f} → {m['similarity'][1]:.3f}{p('similarity')}{mark('similarity')}"
            f" | {r['duration_ratio']:.3f}{p('duration')}{mark('duration')} | {utmos}"
            f" | {'ok' if r['checks'].get('lufs', True) and r['checks'].get('f0_median', True) else 'outside: ' + ', '.join(c for c in ('lufs', 'f0_median') if not r['checks'].get(c, True))}"
            f" | {verdict if r['pass'] else 'fail: ' + ', '.join(failed)} |"
        )
    head = (
        f"## T4: {args.b} against {args.a}\n\n"
        "| Set | Model | Clips | Error rate % (CER zh, WER en) | Worst % | Failed clips | Speaker similarity | Duration ratio | UTMOS | Loudness, F0 | Verdict |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    text = head + "\n".join(rows) + "\n"
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        Path(args.report).with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0 if all(r["pass"] for r in results.values()) else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--impl", required=True, choices=["reference", "native", "upstream"])
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--name")
    run.add_argument("--only", action="append")
    run.add_argument("--seeds", type=int, default=len(SEEDS))
    run.add_argument("--texts", type=int, default=0, help="Only the first N texts of each language (a quick run)")
    run.add_argument("--option", action="append", help="An engine option for every request, key=value (an opt-in that is a request option); repeatable")
    run.add_argument("--voice", action="append", help="Only these voices of the set (a targeted run, e.g. more seeds for one voice)")
    run.add_argument("--language", help="Only the texts in this language")
    run.add_argument("--unset-language", action="store_true", help="Send no language (the engine's default), scoring each clip in its text's language")
    run.add_argument("--setting", action="append", help="A setting for the whole run, key=value (an opt-in judged against the default); repeatable")
    run.add_argument("--suite", default=str(SUITE))
    score = sub.add_parser("score")
    score.add_argument("--name", action="append")
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--only", action="append")
    score.add_argument("--suite", default=str(SUITE))
    compare = sub.add_parser("compare")
    compare.add_argument("--a", default="reference")
    compare.add_argument("--b", default="native")
    compare.add_argument("--only", action="append")
    compare.add_argument("--report")
    compare.add_argument("--suite", default=str(SUITE))
    args = parser.parse_args(argv)
    return {"run": cmd_run, "score": cmd_score, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
