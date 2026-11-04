"""FSMN-VAD (FunASR's ``FsmnVADStreaming``): the voice activity detector that cuts a file into speech segments before
Paraformer recognises them. Ported from FunASR 1.4.21 (MIT, Copyright (c) 2022 Alibaba Group).

What file inference does there, and so here (``detect``):

- The waveform is cut into 60 s chunks (``chunk_size`` 60000 ms; a file is final, so there is one chunk more than whole
  chunks fit: the last holds the remainder, possibly nothing). Every piece of state is carried from chunk to chunk.
- ``WavFrontendOnline`` (``OnlineFrontend``): Kaldi's filterbank (80 bins, 25/10 ms, Hamming, no dither) of each
  chunk with the samples the previous one left over in front of it, low frame rate (5 frames stacked every frame) whose
  last 4 frames wait for the next chunk (the final one pads with its last frame), then the CMVN of ``am.mvn``. It also
  hands the detector the exact samples behind the frames it emits (``aligned_waveforms``).
- ``FSMN`` (``Fsmn``): two affine layers, four memory blocks (a depthwise convolution over the 19 frames before each
  one, its state carried across chunks), two affine layers and a softmax over 248 classes; class 0 is silence.
- The state machine: a frame is speech when ``1 - p(silence) >= p(silence) + speech_noise_thres`` and its energy
  (``10·log10(Σ x² + 1e-6)`` over its 400 samples, in float32 with numpy as upstream computes it) passes the thresholds;
  a 200 ms window turns frame decisions into speech starts (15 speech frames of 20) and ends (15 silent of 20); a
  segment starts 400 ms (window and look-back) before its confirmed start and ends after the configured trailing
  silence, less 110 ms of look-ahead, or after 60 s of speech.
- The dynamic silence schedule that 1.4.21 applies by default (``DEFAULT_SILENCE_SCHEDULE``): before each chunk the
  trailing silence that ends a segment is set by how long speech has run without a segment being emitted (2000 ms for
  the first chunk; 200 ms when a segment has been open since the previous chunk), and ``speech_noise_thres`` becomes 0.5
  (the config's 0.6 is never used by file inference).

Left out, as file inference never reaches them: training, export, the streaming output (``[start, -1]`` events), frame
probabilities, the sample buffers the state machine keeps only to print warnings (``data_buf``, ``data_buf_all``:
their lengths always cover the frames asked of them, so only ``data_buf_start_frame`` is kept), and the dropping of
cached scores and decibels (frames are indexed from the start of the file here instead of from the last drop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mizuoto.engines.asr.paraformer.model import apply_cmvn, load_cmvn, load_config
from mizuoto.nn.dsp import kaldi
from mizuoto.runtime import ops as default_ops

CHUNK_MS = 60000
"""``chunk_size``'s default: file inference feeds the frontend and the detector 60 s at a time."""

DEFAULT_SILENCE_SCHEDULE: tuple[tuple[float, int], ...] = (
    (10000, 2000),
    (20000, 1000),
    (30000, 800),
    (40000, 600),
    (50000, 400),
    (60000, 200),
    (float("inf"), 100),
)
"""(speech accumulated without a segment ending, in ms; the trailing silence that ends a segment, in ms)."""

# VadStateMachine
START_NOT_DETECTED, IN_SPEECH, END_DETECTED = 1, 2, 3
# FrameState
FRAME_INVALID, FRAME_SPEECH, FRAME_SIL = -1, 1, 0
# AudioChangeState
SPEECH2SPEECH, SPEECH2SIL, SIL2SIL, SIL2SPEECH, NO_BEGIN, CHANGE_INVALID = 0, 1, 2, 3, 4, 5
# VadDetectMode
SINGLE_UTTERANCE, MULTIPLE_UTTERANCE = 0, 1


class _Linear(nn.Module):
    """``LinearTransform`` / ``AffineTransform``: one linear layer, kept under ``linear`` as the checkpoint names it."""

    def __init__(self, n_in: int, n_out: int, bias: bool, ops: Any) -> None:
        super().__init__()
        self.linear = ops.Linear(n_in, n_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Memory(nn.Module):
    """``FSMNBlock`` without a right context: each channel's ``lorder`` frames up to the current one (state from the
    previous chunk in front) through a depthwise convolution, plus the frame itself."""

    def __init__(self, dim: int, lorder: int, lstride: int, ops: Any) -> None:
        super().__init__()
        self.lorder, self.lstride = lorder, lstride
        self.conv_left = ops.Conv2d(dim, dim, (lorder, 1), dilation=(lstride, 1), groups=dim, bias=False)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` (batch, time, dim), ``cache`` (batch, dim, context, 1): the output and the next chunk's cache."""
        x_per = torch.unsqueeze(x, 1).permute(0, 3, 2, 1)  # batch, dim, time, 1
        y_left = torch.cat((cache.to(x_per.device), x_per), dim=2)
        cache = y_left[:, :, -(self.lorder - 1) * self.lstride :, :]
        out = x_per + self.conv_left(y_left)
        return out.permute(0, 3, 2, 1).squeeze(1), cache


class _Block(nn.Module):
    """``BasicBlock``: projection, memory block, affine layer, ReLU."""

    def __init__(self, linear_dim: int, proj_dim: int, lorder: int, lstride: int, ops: Any) -> None:
        super().__init__()
        self.linear = _Linear(linear_dim, proj_dim, False, ops)
        self.fsmn_block = _Memory(proj_dim, lorder, lstride, ops)
        self.affine = _Linear(proj_dim, linear_dim, True, ops)
        self.context = (lorder - 1) * lstride

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.linear(x)
        if cache is None:  # the first chunk: zeros before the file starts
            cache = torch.zeros(x1.shape[0], x1.shape[-1], self.context, 1, dtype=x1.dtype, device=x1.device)
        x2, cache = self.fsmn_block(x1, cache)
        return torch.relu(self.affine(x2)), cache


class Fsmn(nn.Module):
    """``FSMN`` (``encoder_conf`` of the config): the frame classifier, its softmax included."""

    def __init__(self, conf: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        if int(conf.get("rorder", 0)) != 0:
            raise ValueError("FSMN-VAD with a right context (rorder > 0) is not supported")
        linear_dim, lorder, lstride = int(conf["linear_dim"]), int(conf["lorder"]), int(conf["lstride"])
        self.in_linear1 = _Linear(int(conf["input_dim"]), int(conf["input_affine_dim"]), True, ops)
        self.in_linear2 = _Linear(int(conf["input_affine_dim"]), linear_dim, True, ops)
        self.fsmn = nn.ModuleList([_Block(linear_dim, int(conf["proj_dim"]), lorder, lstride, ops) for _ in range(int(conf["fsmn_layers"]))])
        self.out_linear1 = _Linear(linear_dim, int(conf["output_affine_dim"]), True, ops)
        self.out_linear2 = _Linear(int(conf["output_affine_dim"]), int(conf["output_dim"]), True, ops)

    def forward(self, feats: torch.Tensor, caches: list[torch.Tensor] | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``feats`` (batch, frames, input_dim) and the caches the previous chunk returned (None for the first): the
        class posteriors (batch, frames, output_dim) and the caches for the next chunk."""
        x = torch.relu(self.in_linear2(self.in_linear1(feats)))
        out_caches: list[torch.Tensor] = []
        for i, block in enumerate(self.fsmn):
            x, cache = block(x, None if caches is None else caches[i])
            out_caches.append(cache)
        return torch.softmax(self.out_linear2(self.out_linear1(x)), dim=-1), out_caches


@dataclass
class Options:
    """``VADXOptions``: the detector's settings (``model_conf`` of the config), with upstream's defaults."""

    sample_rate: int = 16000
    detect_mode: int = MULTIPLE_UTTERANCE
    max_end_silence_time: int = 800
    max_start_silence_time: int = 3000
    window_size_ms: int = 200
    sil_to_speech_time_thres: int = 150
    speech_to_sil_time_thres: int = 150
    speech_2_noise_ratio: float = 1.0
    do_extend: int = 1
    lookback_time_start_point: int = 200
    lookahead_time_end_point: int = 100
    max_single_segment_time: int = 60000
    snr_thres: float = -100.0
    noise_frame_num_used_for_snr: int = 100
    decibel_thres: float = -100.0
    speech_noise_thres: float = 0.6
    fe_prior_thres: float = 1e-4
    silence_pdf_num: int = 1
    sil_pdf_ids: tuple[int, ...] = (0,)
    frame_in_ms: int = 10
    frame_length_ms: int = 25

    @classmethod
    def from_config(cls, conf: dict[str, Any]) -> Options:
        """The values of ``conf`` as the config gives them (int or float, as upstream keeps them), the rest defaults."""
        values = {f.name: conf[f.name] for f in fields(cls) if f.name in conf}
        if "sil_pdf_ids" in values:
            values["sil_pdf_ids"] = tuple(int(i) for i in values["sil_pdf_ids"])
        return cls(**values)


class Vad(nn.Module):
    """``FsmnVADStreaming``'s weights (``encoder.*``, as the checkpoint names them) and its options."""

    def __init__(self, config: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.encoder = Fsmn(config.get("encoder_conf") or {}, ops)
        self.options = Options.from_config(config.get("model_conf") or {})


@dataclass
class FrontendCache:
    """``WavFrontendOnline``'s per-call cache: what one chunk leaves to the next."""

    input_cache: torch.Tensor | None = None
    """Samples after the last whole frame, put in front of the next chunk."""
    lfr_splice_cache: torch.Tensor | None = None
    """Filterbank frames (frames, n_mels) still waiting for their right context."""
    waveform_buffer: torch.Tensor | None = None
    """Samples from ``waveform_buffer_start`` on: those the next emitted frames cover."""
    waveform_buffer_start: int = 0
    emitted_lfr_frames: int = 0


def apply_lfr_online(inputs: torch.Tensor, lfr_m: int, lfr_n: int, is_final: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``WavFrontendOnline.apply_lfr``: the stacked frames whose right context ``inputs`` holds (all of them, the last
    padded with copies of the last frame, when ``is_final``) and the frames to keep for the next chunk."""
    t = inputs.shape[0]
    t_lfr = int(np.ceil((t - (lfr_m - 1) // 2) / lfr_n))
    splice_idx = t_lfr
    feat_dim = inputs.shape[-1]
    ori_inputs = inputs
    strides = (lfr_n * feat_dim, 1)
    sizes = (t_lfr, lfr_m * feat_dim)
    last_idx = (t - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (t - last_idx * lfr_n)
    if is_final:
        if num_padding > 0:
            # upstream's count, a float that can exceed what the strided view reads; the extra rows are never read
            padding = (2 * lfr_m - 2 * t + (t_lfr - 1 + last_idx) * lfr_n) / 2 * (t_lfr - last_idx)
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(padding))
    elif num_padding > 0:
        sizes = (last_idx, lfr_m * feat_dim)
        splice_idx = last_idx
    splice_idx = min(t - 1, splice_idx * lfr_n)
    outputs = inputs[:splice_idx].as_strided(sizes, strides)
    return outputs.clone().type(torch.float32), ori_inputs[splice_idx:, :]


class OnlineFrontend:
    """``WavFrontendOnline`` (``frontend_conf`` of the config) as the VAD calls it: chunk by chunk, with the samples
    behind each chunk's frames."""

    def __init__(self, conf: dict[str, Any], cmvn: torch.Tensor | None) -> None:
        self.fs = int(conf.get("fs", 16000))
        self.window = str(conf.get("window", "hamming"))
        self.n_mels = int(conf.get("n_mels", 80))
        self.frame_length = float(conf.get("frame_length", 25))
        self.frame_shift = float(conf.get("frame_shift", 10))
        self.frame_sample_length = int(self.frame_length * self.fs / 1000)
        self.frame_shift_sample_length = int(self.frame_shift * self.fs / 1000)
        self.lfr_m = int(conf.get("lfr_m", 1))
        self.lfr_n = int(conf.get("lfr_n", 1))
        self.dither = float(conf.get("dither", 1.0))
        self.cmvn = cmvn

    def compute_frame_num(self, sample_length: int) -> int:
        frame_num = int((sample_length - self.frame_sample_length) / self.frame_shift_sample_length + 1)
        return frame_num if frame_num >= 1 and sample_length >= self.frame_sample_length else 0

    def _lfr_cmvn(self, mat: torch.Tensor, is_final: bool, cache: FrontendCache) -> torch.Tensor:
        """``forward_lfr_cmvn`` for one item: LFR (updating the splice cache), then CMVN."""
        if self.lfr_m != 1 or self.lfr_n != 1:
            mat, cache.lfr_splice_cache = apply_lfr_online(mat, self.lfr_m, self.lfr_n, is_final)
        if self.cmvn is not None:
            mat = apply_cmvn(mat, self.cmvn)
        return mat

    def __call__(self, chunk: torch.Tensor, cache: FrontendCache, is_final: bool, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``chunk``: float32 samples in [-1, 1] at ``fs``, (samples,). The features of the frames this chunk completes
        (frames, n_mels × lfr_m; none when it completes none) and the samples those frames cover, or None. ``generator``
        feeds the dither (the VAD's config has none)."""
        frames = chunk.unsqueeze(0)
        if frames.shape[1] > 0:
            cache.waveform_buffer = frames.clone() if cache.waveform_buffer is None else torch.cat((cache.waveform_buffer, frames), dim=1)

        # forward_fbank: the left-over samples, then this chunk; whole frames only
        if cache.input_cache is not None:
            frames = torch.cat((cache.input_cache, frames), dim=1)
        frame_num = self.compute_frame_num(frames.shape[-1])
        cache.input_cache = frames[:, -(frames.shape[-1] - frame_num * self.frame_shift_sample_length) :]
        fbank: torch.Tensor | None = None
        if frame_num:
            fbank = kaldi.fbank(
                (frames[0] * (1 << 15)).unsqueeze(0),
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                generator=generator,
            )

        feats: torch.Tensor | None = None
        if fbank is not None:
            if cache.lfr_splice_cache is None:  # the first frame repeated as the left context
                cache.lfr_splice_cache = fbank[0, :].unsqueeze(dim=0).repeat((self.lfr_m - 1) // 2, 1)
            if is_final or fbank.shape[0] + cache.lfr_splice_cache.shape[0] >= self.lfr_m:
                feats = self._lfr_cmvn(torch.cat((cache.lfr_splice_cache, fbank), dim=0), is_final, cache)
            else:  # too few frames for one stacked frame: keep them all
                cache.lfr_splice_cache = torch.cat((cache.lfr_splice_cache, fbank), dim=0)
                return torch.empty(0), None
        elif is_final and cache.lfr_splice_cache is not None:  # nothing new: the waiting frames, padded
            feats = self._lfr_cmvn(cache.lfr_splice_cache.clone(), is_final, cache)

        if feats is None or feats.shape[0] == 0:
            return torch.empty(0) if feats is None else feats, None
        score_frame_shift = self.lfr_n * self.frame_shift_sample_length
        aligned_start = cache.emitted_lfr_frames * score_frame_shift
        aligned_end = aligned_start + (feats.shape[0] - 1) * score_frame_shift + self.frame_sample_length
        buffer = cache.waveform_buffer
        buffer_start = cache.waveform_buffer_start
        buffer_end = buffer_start + (0 if buffer is None else buffer.shape[1])
        if buffer is None or aligned_start < buffer_start or aligned_end > buffer_end:
            raise RuntimeError(f"Frontend emitted LFR frames without enough waveform samples: need [{aligned_start}, {aligned_end}), have [{buffer_start}, {buffer_end})")
        aligned = buffer[:, aligned_start - buffer_start : aligned_end - buffer_start].clone()
        cache.emitted_lfr_frames += feats.shape[0]
        next_buffer_start = min(cache.emitted_lfr_frames * score_frame_shift, buffer_end)
        cache.waveform_buffer = buffer[:, next_buffer_start - buffer_start :].clone()
        cache.waveform_buffer_start = next_buffer_start
        return feats, aligned[0]


@dataclass
class _Segment:
    """``E2EVadSpeechBufWithDoa`` without its samples."""

    start_ms: int
    end_ms: int
    contain_seg_start_point: bool = False
    contain_seg_end_point: bool = False


class _Detector:
    """``WindowDetector`` and ``Stats`` with ``FsmnVADStreaming``'s state machine over them, for one file."""

    def __init__(self, opts: Options) -> None:
        self.opts = opts
        # WindowDetector
        self.win_size_frame = int(opts.window_size_ms / opts.frame_in_ms)
        self.sil_to_speech_frmcnt_thres = int(opts.sil_to_speech_time_thres / opts.frame_in_ms)
        self.speech_to_sil_frmcnt_thres = int(opts.speech_to_sil_time_thres / opts.frame_in_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.cur_win_pos = 0
        self.pre_frame_state = FRAME_SIL
        # Stats
        self.data_buf_start_frame = 0
        self.frm_cnt = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.continous_silence_frame_count = 0
        self.vad_state_machine = START_NOT_DETECTED
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.number_end_time_detected = 0
        self.noise_average_decibel = -100.0
        self.max_end_sil_frame_cnt_thresh: float = opts.max_end_silence_time - opts.speech_to_sil_time_thres
        self.speech_noise_thres: float = opts.speech_noise_thres
        self.output_data_buf: list[_Segment] = []
        self.output_data_buf_offset = 0
        self.decibel: list[float] = []
        """Every frame's energy in dB, from the start of the file."""
        self.sil_score: list[float] = []
        """Every frame's silence posterior (summed over ``sil_pdf_ids``), from the start of the file."""

    # WindowDetector

    def reset_window(self) -> None:
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FRAME_SIL

    def window_detect(self, frame_state: int) -> int:
        if frame_state == FRAME_SPEECH:
            cur_frame_state = 1
        elif frame_state == FRAME_SIL:
            cur_frame_state = 0
        else:
            return CHANGE_INVALID
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame
        if self.pre_frame_state == FRAME_SIL and self.win_sum >= self.sil_to_speech_frmcnt_thres:
            self.pre_frame_state = FRAME_SPEECH
            return SIL2SPEECH
        if self.pre_frame_state == FRAME_SPEECH and self.win_sum <= self.speech_to_sil_frmcnt_thres:
            self.pre_frame_state = FRAME_SIL
            return SPEECH2SIL
        if self.pre_frame_state == FRAME_SIL:
            return SIL2SIL
        if self.pre_frame_state == FRAME_SPEECH:
            return SPEECH2SPEECH
        return CHANGE_INVALID

    # FsmnVADStreaming

    def add_chunk(self, scores: torch.Tensor, waveform: torch.Tensor) -> None:
        """``ComputeDecibel`` and ``ComputeScores``: a chunk's posteriors (1, frames, classes) and the samples its frames
        cover (exactly (frames - 1) shifts and one frame long)."""
        opts = self.opts
        frame_sample_length = int(opts.frame_length_ms * opts.sample_rate / 1000)
        frame_shift_length = int(opts.frame_in_ms * opts.sample_rate / 1000)
        frame_count = scores.shape[1]
        aligned_sample_count = (frame_count - 1) * frame_shift_length + frame_sample_length
        if waveform.numel() != aligned_sample_count:
            raise RuntimeError(f"VAD score frames and waveform samples are not aligned: expected {aligned_sample_count}, got {waveform.numel()}")
        # numpy in the waveform's dtype (float32), as upstream computes it: the sums' order decides the last bits
        waveform_numpy = waveform.detach().cpu().numpy()
        offsets = np.arange(0, waveform_numpy.shape[0] - frame_sample_length + 1, frame_shift_length)
        frames = waveform_numpy[offsets[:, np.newaxis] + np.arange(frame_sample_length)]
        decibel_numpy = 10 * np.log10(np.sum(np.square(frames), axis=1) + 0.000001)
        self.decibel.extend(decibel_numpy.tolist())

        if len(scores) != 1:
            raise ValueError("FSMN-VAD scores one item at a time")
        # float32 to Python floats, as upstream's ``.item()`` per frame
        columns = scores[0][:, list(opts.sil_pdf_ids)].tolist()
        if len(opts.sil_pdf_ids) > 1:
            self.sil_score.extend(sum(row) for row in columns)
        else:
            self.sil_score.extend(row[0] for row in columns)
        self.frm_cnt += frame_count

    def get_frame_state(self, t: int) -> int:
        """``GetFrameState`` (updating the noise level on non-speech frames)."""
        opts = self.opts
        if t >= len(self.decibel):
            return FRAME_SIL
        cur_decibel = self.decibel[t]
        cur_snr = cur_decibel - self.noise_average_decibel
        if cur_decibel < opts.decibel_thres:
            return FRAME_SIL
        sum_score = 0.0
        noise_prob = 0.0
        if len(opts.sil_pdf_ids) != opts.silence_pdf_num:
            raise ValueError("sil_pdf_ids and silence_pdf_num disagree")
        if len(opts.sil_pdf_ids) > 0:
            sum_score = self.sil_score[t]
            noise_prob = math.log(sum_score) * opts.speech_2_noise_ratio
            total_score = 1.0
            sum_score = total_score - sum_score
        speech_prob = math.log(sum_score)
        if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
            if cur_snr >= opts.snr_thres and cur_decibel >= opts.decibel_thres:
                return FRAME_SPEECH
            return FRAME_SIL
        if self.noise_average_decibel < -99.9:
            self.noise_average_decibel = cur_decibel
        else:
            self.noise_average_decibel = (cur_decibel + self.noise_average_decibel * (opts.noise_frame_num_used_for_snr - 1)) / opts.noise_frame_num_used_for_snr
        return FRAME_SIL

    def detect_frames(self, count: int, is_final: bool) -> None:
        """``DetectCommonFrames`` / ``DetectLastFrames``: the chunk's last ``count`` frames, the very last one final."""
        if self.vad_state_machine == END_DETECTED:
            return
        for i in range(count - 1, -1, -1):
            frame_state = self.get_frame_state(self.frm_cnt - 1 - i)
            self.detect_one_frame(frame_state, self.frm_cnt - 1 - i, is_final and i == 0)

    def latency(self) -> int:
        """``LatencyFrmNumAtStartPoint``: the window and the look-back before a confirmed start, in frames."""
        vad_latency = self.win_size_frame
        if self.opts.do_extend:
            vad_latency += int(self.opts.lookback_time_start_point / self.opts.frame_in_ms)
        return vad_latency

    def pop_till(self, frame_idx: int) -> None:
        """``PopDataBufTillFrame``: its buffer always holds the frames asked for, so the start frame moves to them."""
        self.data_buf_start_frame = max(self.data_buf_start_frame, frame_idx)

    def pop_to_output(self, start_frm: int, frm_cnt: int, first_frm_is_start_point: bool, last_frm_is_end_point: bool) -> None:
        """``PopDataToOutputBuf``: the segment bookkeeping (a new segment at a start point; the last one extended)."""
        self.pop_till(start_frm)
        if len(self.output_data_buf) == 0 or first_frm_is_start_point:
            start_ms = start_frm * self.opts.frame_in_ms
            self.output_data_buf.append(_Segment(start_ms, start_ms))
        cur_seg = self.output_data_buf[-1]
        self.data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def on_silence_detected(self, valid_frame: int) -> None:
        self.lastest_confirmed_silence_frame = valid_frame
        if self.vad_state_machine == START_NOT_DETECTED:
            self.pop_till(valid_frame)

    def on_voice_detected(self, valid_frame: int) -> None:
        self.latest_confirmed_speech_frame = valid_frame
        self.pop_to_output(valid_frame, 1, False, False)

    def on_voice_start(self, start_frame: int, fake_result: bool = False) -> None:
        if self.confirmed_start_frame == -1:  # upstream prints "not reset vad properly" otherwise
            self.confirmed_start_frame = start_frame
        if not fake_result and self.vad_state_machine == START_NOT_DETECTED:
            self.pop_to_output(self.confirmed_start_frame, 1, True, False)

    def on_voice_end(self, end_frame: int, fake_result: bool) -> None:
        for t in range(self.latest_confirmed_speech_frame + 1, end_frame):
            self.on_voice_detected(t)
        if self.confirmed_end_frame == -1:
            self.confirmed_end_frame = end_frame
        if not fake_result:
            self.pop_to_output(self.confirmed_end_frame, 1, False, True)
        self.number_end_time_detected += 1

    def maybe_on_voice_end_if_last_frame(self, is_final_frame: bool, cur_frm_idx: int) -> None:
        if is_final_frame:
            self.on_voice_end(cur_frm_idx, False)
            self.vad_state_machine = END_DETECTED

    def reset_detection(self) -> None:
        """``ResetDetection``: ready for the next utterance (the dropping of cached frames is not needed here)."""
        self.continous_silence_frame_count = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.vad_state_machine = START_NOT_DETECTED
        self.reset_window()
        if self.output_data_buf and not self.output_data_buf[-1].contain_seg_end_point:
            raise RuntimeError("FSMN-VAD reset with an open segment")

    def over_max_segment(self, cur_frm_idx: int) -> bool:
        return cur_frm_idx - self.confirmed_start_frame + 1 > self.opts.max_single_segment_time / self.opts.frame_in_ms

    def continue_speech(self, cur_frm_idx: int, is_final_frame: bool) -> None:
        """In a segment: end it past the longest segment, else extend it (or end it at the file's last frame)."""
        if self.over_max_segment(cur_frm_idx):
            self.on_voice_end(cur_frm_idx, False)
            self.vad_state_machine = END_DETECTED
        elif not is_final_frame:
            self.on_voice_detected(cur_frm_idx)
        else:
            self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)

    def detect_one_frame(self, cur_frm_state: int, cur_frm_idx: int, is_final_frame: bool) -> None:
        """``DetectOneFrame``."""
        opts = self.opts
        tmp_cur_frm_state = FRAME_INVALID
        if cur_frm_state == FRAME_SPEECH:
            tmp_cur_frm_state = FRAME_SPEECH if math.fabs(1.0) > opts.fe_prior_thres else FRAME_SIL
        elif cur_frm_state == FRAME_SIL:
            tmp_cur_frm_state = FRAME_SIL
        state_change = self.window_detect(tmp_cur_frm_state)
        frm_shift_in_ms = opts.frame_in_ms
        if state_change == SIL2SPEECH:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == START_NOT_DETECTED:
                start_frame = max(self.data_buf_start_frame, cur_frm_idx - self.latency())
                self.on_voice_start(start_frame)
                self.vad_state_machine = IN_SPEECH
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.on_voice_detected(t)
            elif self.vad_state_machine == IN_SPEECH:
                for t in range(self.latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.on_voice_detected(t)
                self.continue_speech(cur_frm_idx, is_final_frame)
        elif state_change in (SPEECH2SIL, SPEECH2SPEECH):
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == IN_SPEECH:
                self.continue_speech(cur_frm_idx, is_final_frame)
        elif state_change == SIL2SIL:
            self.continous_silence_frame_count += 1
            if self.vad_state_machine == START_NOT_DETECTED:
                # a silence timeout (single-utterance mode) or a file without speech: a zero-length decision
                if (opts.detect_mode == SINGLE_UTTERANCE and self.continous_silence_frame_count * frm_shift_in_ms > opts.max_start_silence_time) or (
                    is_final_frame and self.number_end_time_detected == 0
                ):
                    for t in range(self.lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.on_silence_detected(t)
                    self.on_voice_start(0, True)
                    self.on_voice_end(0, True)
                    self.vad_state_machine = END_DETECTED
                elif cur_frm_idx >= self.latency():
                    self.on_silence_detected(cur_frm_idx - self.latency())
            elif self.vad_state_machine == IN_SPEECH:
                if self.continous_silence_frame_count * frm_shift_in_ms >= self.max_end_sil_frame_cnt_thresh:
                    lookback_frame = int(self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms)
                    if opts.do_extend:
                        lookback_frame -= int(opts.lookahead_time_end_point / frm_shift_in_ms)
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.on_voice_end(cur_frm_idx - lookback_frame, False)
                    self.vad_state_machine = END_DETECTED
                elif self.over_max_segment(cur_frm_idx):
                    self.on_voice_end(cur_frm_idx, False)
                    self.vad_state_machine = END_DETECTED
                elif opts.do_extend and not is_final_frame:
                    if self.continous_silence_frame_count <= int(opts.lookahead_time_end_point / frm_shift_in_ms):
                        self.on_voice_detected(cur_frm_idx)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx)
        if self.vad_state_machine == END_DETECTED and opts.detect_mode == MULTIPLE_UTTERANCE:
            self.reset_detection()

    def take_segments(self, is_final: bool) -> list[list[int]]:
        """``forward``'s non-streaming output: the segments not yet returned that have both ends (all, when final)."""
        segments: list[list[int]] = []
        for i in range(self.output_data_buf_offset, len(self.output_data_buf)):
            seg = self.output_data_buf[i]
            if not is_final and (not seg.contain_seg_start_point or not seg.contain_seg_end_point):
                continue
            segments.append([seg.start_ms, seg.end_ms])
            self.output_data_buf_offset += 1
        return segments


def detect(model: Vad, frontend: OnlineFrontend, waveform: torch.Tensor, *, chunk_ms: int = CHUNK_MS, scores: list[torch.Tensor] | None = None) -> list[list[int]]:
    """FunASR's file inference (``AutoModel.inference`` with the VAD, ``FsmnVADStreaming.inference``): the speech
    segments of ``waveform`` as ``[start_ms, end_ms]``, in order (FunASR's ``value``).

    ``waveform``: float32 samples in [-1, 1] at ``frontend.fs`` (16 kHz), (samples,), as FunASR's
    ``load_audio_text_image_video`` gives them, on the CPU. The encoder runs on ``model``'s device and dtype; the
    frontend and the state machine on the CPU. ``chunk_ms``: the chunk length (upstream's ``chunk_size``; 15 s or more,
    below which upstream streams). ``scores``, when given, receives each chunk's encoder output (1, frames, classes).
    """
    if waveform.dim() != 1 or waveform.dtype != torch.float32:
        raise ValueError("FSMN-VAD takes float32 samples, (samples,)")
    if chunk_ms < 15000:
        raise ValueError("FSMN-VAD's file inference takes chunks of 15 s or more")
    if waveform.numel() == 0:
        return []
    opts = model.options
    detector = _Detector(opts)
    frontend_cache = FrontendCache()
    encoder_caches: list[torch.Tensor] | None = None
    param = next(model.encoder.parameters())
    chunk_stride_samples = int(chunk_ms * frontend.fs / 1000)
    n = int(len(waveform) // chunk_stride_samples + 1)  # a file is final: one chunk more, holding the remainder
    accumulated_ms = 0
    in_speech = False
    segments: list[list[int]] = []
    for i in range(n):
        is_final = i == n - 1
        # the dynamic silence schedule: speech since the last emitted segment sets the trailing silence that ends one
        if detector.vad_state_machine == IN_SPEECH or in_speech:
            accumulated_ms += chunk_ms
            in_speech = True
        for limit_ms, silence_ms in DEFAULT_SILENCE_SCHEDULE:
            if accumulated_ms <= limit_ms:
                detector.max_end_sil_frame_cnt_thresh = max(silence_ms - opts.speech_to_sil_time_thres, 0)
                detector.speech_noise_thres = 0.5
                break

        feats, aligned = frontend(waveform[i * chunk_stride_samples : (i + 1) * chunk_stride_samples], frontend_cache, is_final)
        if feats.numel() == 0:
            continue
        if aligned is None or aligned.numel() == 0:
            raise RuntimeError("The VAD frontend emitted frames without their samples")
        with torch.no_grad():
            chunk_scores, encoder_caches = model.encoder(feats.unsqueeze(0).to(device=param.device, dtype=param.dtype), encoder_caches)
        if chunk_scores.shape[1] != feats.shape[0]:
            raise RuntimeError("The shape between feats and scores does not match")
        if scores is not None:
            scores.append(chunk_scores)
        detector.add_chunk(chunk_scores, aligned)
        detector.detect_frames(chunk_scores.shape[1], is_final)
        emitted = detector.take_segments(is_final)
        if emitted:
            segments.extend(emitted)
            accumulated_ms = 0
            in_speech = False
    return segments


def load(folder: Path, ops: Any = default_ops.default) -> tuple[Vad, OnlineFrontend]:
    """The FSMN-VAD of a FunASR model folder (``config.yaml``, ``am.mvn``, ``model.pt``) on the CPU, and its frontend."""
    from mizuoto.runtime.loading import load_into, load_state_dict

    config = load_config(folder)
    model = Vad(config, ops)
    load_into(model, load_state_dict(folder / "model.pt"), what="FSMN-VAD")
    model.eval()
    return model, OnlineFrontend(config.get("frontend_conf") or {}, load_cmvn(folder / "am.mvn"))
