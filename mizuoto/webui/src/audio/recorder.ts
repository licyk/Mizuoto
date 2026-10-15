/**
 * Recording a clip in the browser: the microphone without the browser's own processing (it would
 * colour a voice reference), collected by an AudioWorklet at the context's rate, with levels ten
 * times a second, and written as a mono 16-bit WAV to upload.
 */

const PROCESSOR = 'mizuoto-recorder';

const WORKLET = `
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sum = 0;
    this.peak = 0;
    this.count = 0;
    this.block = Math.round(sampleRate / 10);
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.port.postMessage(ch.slice());
      for (let i = 0; i < ch.length; i++) {
        const v = ch[i];
        this.sum += v * v;
        const a = Math.abs(v);
        if (a > this.peak) this.peak = a;
      }
      this.count += ch.length;
      if (this.count >= this.block) {
        this.port.postMessage({ rms: Math.sqrt(this.sum / this.count), peak: this.peak });
        this.sum = 0;
        this.peak = 0;
        this.count = 0;
      }
    }
    return true;
  }
}
registerProcessor('${PROCESSOR}', Recorder);
`;

/** The length a voice reference should have, shown as a guide while recording. */
export const GUIDE_SECONDS = { min: 3, max: 30 } as const;

export type MicProblem = 'insecure' | 'unsupported';

/** Why this page cannot use the microphone, if it cannot: browsers allow it only over HTTPS or on localhost. */
export function micProblem(): MicProblem | null {
  if (typeof window === 'undefined') return 'unsupported';
  if (!window.isSecureContext) return 'insecure';
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') return 'unsupported';
  return null;
}

export const toDb = (x: number) => (x > 0 ? 20 * Math.log10(x) : -120);

/** Mono float samples → a 16-bit PCM WAV file. */
export function encodeWav(chunks: Float32Array[], sampleRate: number): ArrayBuffer {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, s: string) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      view.setInt16(offset, Math.round(Math.max(-1, Math.min(1, c[i]!)) * 32767), true);
      offset += 2;
    }
  }
  return buffer;
}

export interface RecorderLevels {
  rmsDb: number;
  peakDb: number;
  seconds: number;
}

export class Recorder {
  onLevels: ((levels: RecorderLevels) => void) | null = null;
  private ctx: AudioContext | null = null;
  private media: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private chunks: Float32Array[] = [];
  private samples = 0;

  get seconds(): number {
    return this.ctx ? this.samples / this.ctx.sampleRate : 0;
  }

  async start(): Promise<void> {
    this.chunks = [];
    this.samples = 0;
    try {
      this.media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
      const ctx = new AudioContext();
      this.ctx = ctx;
      const module = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      try {
        await ctx.audioWorklet.addModule(module);
      } finally {
        URL.revokeObjectURL(module);
      }
      const node = new AudioWorkletNode(ctx, PROCESSOR, { numberOfInputs: 1, numberOfOutputs: 0 });
      node.port.onmessage = (e: MessageEvent<Float32Array | { rms: number; peak: number }>) => {
        if (e.data instanceof Float32Array) {
          this.chunks.push(e.data);
          this.samples += e.data.length;
        } else this.onLevels?.({ rmsDb: toDb(e.data.rms), peakDb: toDb(e.data.peak), seconds: this.seconds });
      };
      ctx.createMediaStreamSource(this.media).connect(node);
      this.node = node;
      await ctx.resume();
    } catch (e) {
      this.cancel();
      throw e;
    }
  }

  /** Stop and return the recording as a WAV file, or null when nothing was recorded. */
  stop(name = `recording-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.wav`): File | null {
    const rate = this.ctx?.sampleRate ?? 48000;
    const chunks = this.chunks;
    this.cancel();
    if (!chunks.length) return null;
    return new File([encodeWav(chunks, rate)], name, { type: 'audio/wav' });
  }

  cancel() {
    this.node?.disconnect();
    this.node = null;
    this.media?.getTracks().forEach((t) => t.stop());
    this.media = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
    this.chunks = [];
    this.samples = 0;
  }
}
