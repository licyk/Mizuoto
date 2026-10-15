"""VoxCPM: the native engine against upstream's ``voxcpm`` package, in one process.

    <ref-env>/bin/python scripts/parity/voxcpm_modules.py <model_dir> [--case NAME]... [--dtype auto|fp32] [--threads 8]

The interpreter must have both upstream VoxCPM and this checkout (the VoxCPM reference environment). Each case
runs upstream's ``VoxCPM.generate`` (or ``generate_streaming``) and the native engine's ``synthesize`` on the same
text, clip and seed, and compares the audio sample for sample. Components are compared first: the tokenizer, the
VAE's latents of the clip, and its decode.

``--dtype fp32`` runs both in float32: upstream reads the dtype from ``config.json``, so its side loads a copy of the
folder whose config says ``float32`` (the weights linked, not copied).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get("MIZUOTO_PARITY_DIR", "~/code_workspace/experiment/kodama-refactor")).expanduser() / "corpus"

CASES: dict[str, dict[str, Any]] = {
    "zero-en": {"text": "Hello, this is a short test of the speech model.", "seed": 7},
    "zero-zh": {"text": "今天的天气很好，我们一起去公园散步吧。", "seed": 11},
    "clone-en": {"text": "This is the first sentence. And here comes the second one.", "ref": "en-narrator", "seed": 7},
    "clone-zh": {"text": "欢迎使用语音合成系统，希望你喜欢这个声音。", "ref": "zh-serena", "seed": 5},
    "ref-only": {"text": "A cloned voice without any transcript at all.", "ref": "en-narrator", "transcript": False, "seed": 3, "v2": True},
    "continuation-only": {"text": "Continuation without the reference prefix.", "ref": "en-narrator", "reference_prefix": False, "seed": 3, "v2": True},
    "design": {"text": "Hello, welcome to the voice design demo.", "control": "A young woman, gentle and sweet voice", "seed": 42, "v2": True},
    "instruct-clone": {
        "text": "This is a cloned voice with style control.",
        "ref": "en-narrator",
        "transcript": False,
        "control": "slightly faster, cheerful tone",
        "seed": 9,
        "v2": True,
    },
    "normalize": {"text": "It costs $19.99 on March 15, 2024, at 3:30 PM.", "normalize": True, "seed": 4},
    "retry": {"text": "Short.", "seed": 1, "ratio": 1.5},
    "stream": {"text": "Streaming text to speech is easy with this model.", "seed": 8, "stream": True},
    "stream-clone": {"text": "Streaming a cloned voice, patch by patch.", "ref": "en-narrator", "seed": 8, "stream": True},
}


def refs() -> dict[str, Any]:
    return json.loads((CORPUS / "refs.json").read_text(encoding="utf-8"))


def canonical_wav(name: str, entry: dict[str, Any], folder: Path) -> Path:
    """The clip as the stack gives it to engines: mono 16-bit WAV at its own rate (``AudioService.canonical_wav``)."""
    from mizuoto.core.audio.io import decode, encode

    target = folder / f"{name}.wav"
    if not target.is_file():
        samples, sr = decode(CORPUS / "refs" / entry["file"])
        encode(samples, sr, target, "wav")
    return target


def fp32_copy(model_dir: Path) -> Path:
    """A folder whose config.json says float32, its other files linked."""
    target = Path(tempfile.mkdtemp(prefix="voxcpm-fp32-"))
    for item in model_dir.iterdir():
        if item.name == "config.json":
            config = json.loads(item.read_text(encoding="utf-8"))
            config["dtype"] = "float32"
            (target / item.name).write_text(json.dumps(config), encoding="utf-8")
        elif item.is_file():
            (target / item.name).symlink_to(item)
    return target


def compare(name: str, a: Any, b: Any) -> dict[str, Any]:
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    row: dict[str, Any] = {"what": name, "lengths": (len(a), len(b))}
    if len(a) == len(b):
        diff = np.abs(a - b)
        row["identical"] = bool(diff.max(initial=0.0) == 0.0)
        row["max_abs"] = float(diff.max(initial=0.0))
    else:
        row["identical"] = False
    return row


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir")
    parser.add_argument("--case", action="append")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from voxcpm import VoxCPM

    from mizuoto.engines.voxcpm.engine import VoxCpmEngine
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import LoadSpec, Reference, StyleSpec, SynthesisTask, VoiceSpec

    model_dir = Path(args.model_dir).resolve()
    up_dir = fp32_copy(model_dir) if args.dtype == "fp32" else model_dir
    engine = VoxCpmEngine()
    variant = engine.detect_variant(model_dir)
    assert variant is not None
    version = 2 if variant == "v2" else 1

    class _Options:
        compile = False

    class _Runtime:
        options = _Options()

    ctx = RequestContext("load", device="cpu")
    native = engine.load(LoadSpec(model_id=f"voxcpm/{model_dir.name}", model_dir=model_dir, variant=variant, device="cpu", dtype=args.dtype), _Runtime(), ctx)  # ty: ignore[invalid-argument-type]
    upstream = VoxCPM(str(up_dir), enable_denoiser=False, optimize=False, device="cpu")
    rows: list[dict[str, Any]] = []
    table = refs()

    # -- components ----------------------------------------------------------------------
    from voxcpm.utils.text_normalize import TextNormalizer as UpstreamNormalizer

    from mizuoto.engines.voxcpm.model import latent_patches
    from mizuoto.engines.voxcpm.text import TextNormalizer

    corpus_dir = ROOT / "scripts" / "parity" / "corpus"
    texts = [
        line for name in ("tokenizer.txt", "normalize.txt") for line in (corpus_dir / name).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("//")
    ]
    up_tokens = [upstream.tts_model.text_tokenizer(t) for t in texts]
    nat_tokens = [native.tokenizer(t) for t in texts]  # ty: ignore[unresolved-attribute]
    rows.append({"what": f"tokenizer ({len(texts)} texts)", "identical": up_tokens == nat_tokens, "lengths": (len(up_tokens), sum(a == b for a, b in zip(up_tokens, nat_tokens)))})
    up_norm, nat_norm = UpstreamNormalizer(), TextNormalizer()
    same = [up_norm.normalize(t) == nat_norm.normalize(t) for t in texts]
    rows.append({"what": f"normaliser ({len(texts)} texts)", "identical": all(same), "lengths": (len(same), sum(same))})

    wav_dir = Path(tempfile.mkdtemp(prefix="voxcpm-refs-"))
    clip = canonical_wav("en-narrator", table["en-narrator"], wav_dir)
    with torch.inference_mode():
        audio = native._read(str(clip))  # ty: ignore[unresolved-attribute]
        lat_native = latent_patches(native.vae, audio, native.config.patch_size, "left", torch.device("cpu"))  # ty: ignore[unresolved-attribute]
        cache = upstream.tts_model.build_prompt_cache(prompt_text="x", prompt_wav_path=str(clip))
        rows.append(compare("vae-encode (prompt latents)", cache["audio_feat"], lat_native))
        lat = lat_native.permute(2, 0, 1).reshape(1, native.config.vae.latent_dim, -1)  # ty: ignore[unresolved-attribute]
        rows.append(compare("vae-decode", upstream.tts_model.audio_vae.decode(lat.clone()), native.vae.decode(lat.clone())))  # ty: ignore[unresolved-attribute]

    # -- whole requests ------------------------------------------------------------------
    for name, case in CASES.items():
        if args.case and name not in args.case:
            continue
        if case.get("v2") and version != 2:
            continue
        ref = table[case["ref"]] if case.get("ref") else None
        path = str(canonical_wav(case["ref"], ref, wav_dir)) if ref else None
        transcript = (ref.get("transcript") or "").strip() if ref and case.get("transcript", True) else None
        control = case.get("control")
        text = f"({control}){case['text']}" if control and version == 2 else case["text"]
        kwargs: dict[str, Any] = {"text": text, "seed": case["seed"], "normalize": bool(case.get("normalize"))}
        if case.get("ratio"):
            kwargs["retry_badcase_ratio_threshold"] = case["ratio"]
        if path and transcript:
            kwargs.update(prompt_wav_path=path, prompt_text=transcript)
        if path and version == 2 and (not transcript or case.get("reference_prefix", True)):
            kwargs["reference_wav_path"] = path
        started = time.perf_counter()
        if case.get("stream"):
            up_chunks = [c for c in upstream.generate_streaming(**kwargs)]
            up_audio = np.concatenate(up_chunks)
        else:
            up_chunks = None
            up_audio = upstream.generate(**kwargs)
        up_time = time.perf_counter() - started
        task = SynthesisTask(
            text=case["text"],
            voice=VoiceSpec(references=[Reference(audio_path=path, transcript=transcript)] if path else []),
            style=StyleSpec(instruct=control if (control and path) else None),
            params={"seed": case["seed"], "text_normalization": bool(case.get("normalize"))},
            options={k: v for k, v in (("retry_badcase_ratio_threshold", case.get("ratio")), ("reference_prefix", case.get("reference_prefix"))) if v is not None},
            stream=bool(case.get("stream")),
        )
        if control and not path:
            task.voice.description = control
        started = time.perf_counter()
        rctx = RequestContext(name, device="cpu", seed=case["seed"])
        with torch.inference_mode():
            chunks = [c.samples for c in native.synthesize(task, rctx)]
        nat_time = time.perf_counter() - started
        row = compare(name, up_audio, np.concatenate(chunks))
        row["times"] = (round(up_time, 1), round(nat_time, 1))
        row["applied"] = {k: rctx.applied.get(k) for k in ("prompt_mode", "seed", "retries")}
        if up_chunks is not None:
            row["chunks_equal"] = [len(c) for c in up_chunks] == [len(c) for c in chunks]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    lines = ["| what | identical | max abs | lengths | times up/native (s) | notes |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        notes = ", ".join(f"{k}={v}" for k, v in {**r.get("applied", {}), **({"chunks_equal": r["chunks_equal"]} if "chunks_equal" in r else {})}.items() if v is not None)
        lines.append(f"| {r['what']} | {'yes' if r['identical'] else 'no'} | {r.get('max_abs', '—')} | {r['lengths'][0]} / {r['lengths'][1]} | {r.get('times', '—')} | {notes} |")
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).write_text(f"# VoxCPM modules: {model_dir.name}, {args.dtype}\n\n{text}\n", encoding="utf-8")
    shutil.rmtree(wav_dir, ignore_errors=True)
    if up_dir != model_dir:
        shutil.rmtree(up_dir, ignore_errors=True)
    return 0 if all(r["identical"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
