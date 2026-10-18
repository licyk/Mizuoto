# Parity harness

The tools that check the native engines against the 0.1 engine-worker implementation (tag `v1-engine-workers`) and
upstream packages; tiers and tolerances are on the documentation site's testing page
([development-testing](../../site/content/docs/development-testing.en.mdx)). The code is here; the data
(environments, clips, fixtures, reports) lives in the workspace, `MIZUOTO_PARITY_DIR`, which defaults to
`~/code_workspace/experiment/kodama-refactor`.

## The workspace

```
reference/          git worktree at tag v1-engine-workers, vendor trees copied in; the engine envs in
                    ~/.local/share/kodama/envs load v1's worker package and engine adapters from here
reference-host/     venv: the worktree's kodama (0.1 kept the old name), i.e. the original host that drives those engine envs
native-env/         venv: this checkout's mizuoto + torch 2.8.0+cu128 (the reference envs' torch); this
                    checkout's .venv runs the native side as well (stack.py --impl native)
envs/cosyvoice-t28  CosyVoice's frozen package set on torch 2.8 (openai-whisper's triton<3 overridden)
data-t28/           a data folder that links the real models and envs, with envs/cosyvoice → cosyvoice-t28
freeze/<env>.txt    uv pip freeze of every reference env, as frozen when the reference was set up
corpus/refs/        reference clips (refs.py), corpus/refs.json their transcripts
fixtures/<engine>/<case>/<name>.wav|.json   one run of one case (stack.py)
reports/            comparison reports (compare.py)
logs/
```

## Commands

```bash
W=~/code_workspace/experiment/kodama-refactor
python scripts/parity/refs.py                                   # clips → $W/corpus
CUDA_VISIBLE_DEVICES= $W/reference-host/bin/python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl reference
CUDA_VISIBLE_DEVICES= $W/reference-host/bin/python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl reference --name reference-repeat
MIZUOTO_PARITY_DATA=$W/data-t28 CUDA_VISIBLE_DEVICES= $W/reference-host/bin/python scripts/parity/stack.py run scripts/parity/cases/cosyvoice.jsonl --impl reference --name reference-t28
CUDA_VISIBLE_DEVICES= $W/native-env/bin/python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl native
$W/native-env/bin/python scripts/parity/compare.py qwen3tts --a reference --b native --report $W/reports/qwen3tts-T3.md
CUDA_VISIBLE_DEVICES= $W/native-env/bin/python scripts/parity/stack.py run scripts/parity/cases/f5tts.jsonl --impl native --only clone-en --name native-compile --setting compute.compile=true --repeat 2
<env python> scripts/parity/check_rng.py --device cuda:0         # explicit generators == seeded default stream
```

- `cases/<engine>.jsonl` holds one JSON object per line. `request` is a `SpeakRequest`
  (`kind: convert|edit` for the other two requests). Voices name corpus clips as `{"ref": "<id>"}` (and
  `"transcript": null` to drop the transcript); a source clip is `{"source_ref": "<id>"}`.
  `settings` holds per-case settings, restored after the case.
  `stream: true` runs the case with a stream sink and records the streamed chunks' lengths (compared as
  behaviour). `voices` names library voices to copy into the run's data folder (without their caches), from the
  workspace's `voices/` or else the real data folder: `narrator-tuned` is the narrator with GPT-SoVITS v2Pro
  weights perturbed by 2 % seeded noise, saved as upstream saves fine-tunes (the SoVITS file tagged `05`).
- **Self-consistency first.** The reference is compared with a repeat of itself (`reference-repeat`)
  before anything is compared with it. A case that does not repeat is not usable for T3.

## Training

```bash
W=~/code_workspace/experiment/kodama-refactor
CUDA_VISIBLE_DEVICES= $W/reference-host/bin/python scripts/parity/train.py run qwen3tts --impl reference --model qwen3tts/1.7b-base --method speaker --hp epochs=1 --name narrator
CUDA_VISIBLE_DEVICES= $W/native-env/bin/python scripts/parity/train.py run qwen3tts --impl native --model qwen3tts/1.7b-base --method speaker --hp epochs=1 --name narrator
$W/native-env/bin/python scripts/parity/train.py compare qwen3tts narrator
```

`train.py` trains on `$W/train-data/datasets/narrator` (11 clips, the dataset M6's trainers were checked on) in a
temporary data folder and keeps `$W/train-fixtures/<engine>/<name>/<impl>/`: the experiment, its metrics, previews,
the job's log and the exported voice or model (`artifacts/`), for loading by the other implementation. M6's v1 runs
(experiment records and metrics) are in `$W/train-data/m6-experiments/`.


## VoxCPM (no 0.1 implementation)

Mizuoto 0.1 never had VoxCPM, so its reference is upstream's own package,
in its own workspace `~/code_workspace/experiment/voxcpm`:

```
upstream/   OpenBMB/VoxCPM at cce58a5         models/   the three model repositories (hf download)
ref-env/    venv: upstream editable + this checkout, torch 2.8.0 (native-env's build), upstream's lock for
            transformers 5.3.0, tokenizers 0.22.2, wetext 0.1.2, inflect 7.5.0
data/       a Mizuoto data folder: voxcpm/2, voxcpm/1.5, voxcpm/0.5b and asr/whisper-large-v3-turbo, imported
reports/    logs/
```

```bash
V=~/code_workspace/experiment/voxcpm W=~/code_workspace/experiment/kodama-refactor
# components and whole requests in one process (tokenizer, VAE, every voice mode, retry, streaming)
CUDA_VISIBLE_DEVICES= $V/ref-env/bin/python scripts/parity/voxcpm_modules.py $V/models/VoxCPM2 [--dtype fp32]
# through the stack: --impl upstream swaps the engine for voxcpm_upstream.py (upstream's VoxCPM.generate)
export MIZUOTO_PARITY_DATA=$V/data CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$W/native-env/bin/python scripts/parity/stack.py run scripts/parity/cases/voxcpm.jsonl --impl native
$V/ref-env/bin/python scripts/parity/stack.py run scripts/parity/cases/voxcpm.jsonl --impl upstream
$W/native-env/bin/python scripts/parity/compare.py voxcpm --a upstream --b native
# the tiny-model tests against upstream's modules
CUDA_VISIBLE_DEVICES= $V/ref-env/bin/python -m pytest tests/engines_native/test_voxcpm_tiny.py -p no:cacheprovider
```

The tokenizer and normaliser are checked by `voxcpm_modules.py` on `scripts/parity/corpus/tokenizer.txt` and `normalize.txt` (one text per line; `//` comments).

## Memory

The memory work changes where weights live, not what is computed: every case must stay identical to the earlier
native runs.

```bash
W=~/code_workspace/experiment/kodama-refactor
export CUDA_VISIBLE_DEVICES=
# default settings, and a simulated device (the CPU plays a GPU: weights placed phase by phase, units streamed,
# everything off the device parked as meta tensors so a component used outside its phase fails). Its size less the
# default 1 GB reserve must hold the fixed parts (embeddings, norms) in fp32: 1536 MB for F5-TTS and GPT-SoVITS, 2048
# for CosyVoice, OmniVoice and VoxCPM, 2560 for IndexTTS, 3072 for Qwen3-TTS, 4096 for Fish Speech; below that the
# manager places the minimum anyway (a real GPU would then run out of memory; a simulated one never does)
python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl native --name final-plain
python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl native --name final-sim --setting compute.simulate_vram_mb=3072
python scripts/parity/compare.py qwen3tts --a native --b final-sim
# memory and speed per case (one process each): peak, load, offload and reload, RTF
python scripts/parity/memory.py run scripts/parity/cases/qwen3tts.jsonl --only base-clone-en --name cpu
```

On a free GPU (check `nvidia-smi --query-gpu=index,memory.used --format=csv` first; never uncapped on a shared one):

```bash
unset CUDA_VISIBLE_DEVICES
# a smaller card played by a big one: the allocator capped, the runtime told (compute.vram_limit_mb)
python scripts/parity/memory.py run scripts/parity/cases/qwen3tts.jsonl --device cuda:0 --vram-limit-mb 4096 --name cuda-4g
# streamed against whole on the GPU (E1): the same seeds, compute.vram_limit_mb forcing partial loads
python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl native --device cuda:0 --name cuda-whole
python scripts/parity/stack.py run scripts/parity/cases/qwen3tts.jsonl --impl native --device cuda:0 --name cuda-4g --setting compute.vram_limit_mb=4096
python scripts/parity/compare.py qwen3tts --a cuda-whole --b cuda-4g --tolerance 0
```

## Speed

Optimisations must not change a result: each round is timed and checked against the previous one.

```bash
P=~/code_workspace/experiment/perf          # MIZUOTO_PERF_DIR; bench.sh and train-bench.sh there run the whole set
export CUDA_VISIBLE_DEVICES=
# three runs per case (the first loads and prepares the voice), the time per phase, the audio against the baseline's
python scripts/parity/perf.py run scripts/parity/cases/f5tts.jsonl --only clone-en --name p1 --reference p0-baseline
python scripts/parity/perf.py run scripts/parity/cases/f5tts.jsonl --only clone-en --name p1-prof --profile  # + cProfile
python scripts/parity/perf.py show p1                                   # warm time and phases per case
python scripts/parity/perf.py compare --a p0-baseline --b p1 --report $P/p0-vs-p1.md
# a fine-tune with a parity fixture's recipe: stage times, seconds per step, metrics and weights against the baseline's
python scripts/parity/train_perf.py run f5tts --fixture narrator --name t1 --no-previews --reference t0-baseline
python scripts/parity/train_perf.py compare --a t0-baseline --b t1
```

Run each round from a worktree frozen at the code it measures (`PYTHONPATH=<worktree>…`, as `bench.sh` does): an
edit in the checkout during a run would mix two versions. The machine's other work slows single runs by up to 10 %;
`compare` takes each case's fastest warm run, and a change of a few percent needs runs interleaved with the code it is
compared with.

## Statistical tier (T4)

On a GPU the two implementations cannot be compared sample for sample (their kernels differ, so the first different
logit sends a draw elsewhere), so the statistical tier compares what they make over a corpus:
`corpus/t4.json`'s sets (a model and a voice mode each), every voice reading its 20 texts (10 English, 10 Chinese) with
seeds 1–3, 180 clips a set and side. `quality.py`'s docstring says what is measured and the thresholds; the site's
testing page records the results.

```bash
W=~/code_workspace/experiment/kodama-refactor V=~/code_workspace/experiment/voxcpm M=$PWD
cd /tmp   # the reference host must import kodama, not this checkout's mizuoto from the working directory
# native (every set) and the references: 0.1's host for the v1 sets (CosyVoice from $W/data-t28), upstream for VoxCPM
MIZUOTO_PARITY_DATA=~/.local/share/kodama $M/.venv/bin/python $M/scripts/parity/quality.py run --impl native --device cuda:0 --only qwen3tts-base-clone
env -u PYTHONPATH MIZUOTO_PARITY_DATA=~/.local/share/kodama $W/reference-host/bin/python $M/scripts/parity/quality.py run --impl reference --device cuda:1 --only qwen3tts-base-clone
PYTHONPATH=$M MIZUOTO_PARITY_DATA=$V/data $V/ref-env/bin/python $M/scripts/parity/quality.py run --impl upstream --device cuda:1 --only voxcpm2-clone
# the judges (Whisper large-v3-turbo, CAM++, UTMOS22 strong, loudness, F0), then the verdict per set
MIZUOTO_PARITY_DATA=~/.local/share/kodama $M/.venv/bin/python $M/scripts/parity/quality.py score --device cuda:0
$M/.venv/bin/python $M/scripts/parity/quality.py compare --a reference --b native --report $W/reports/t4.md
```

`--texts 1 --seeds 1` makes a quick run (six clips a set). A run resumes where it stopped; `perf/t4/run.sh` in the
experiment workspace runs one side set by set with a guard for the GPU's other workload; the guard cannot tell this
machine's processes apart (nvidia-smi gives host PIDs), so score on a GPU nothing is generating on.
