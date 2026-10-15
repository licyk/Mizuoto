import { BASE_URL } from '@/api/baseUrl';
import type { SpeakRequest } from '@/api/types';
import { useAuthStore } from '@/stores/auth';

/**
 * Speak and hear it as it is made (``/api/v1/tts/stream``). The request goes over a WebSocket; the
 * mono 16-bit PCM that comes back plays through an AudioWorklet at the stream's own rate. Playback
 * starts once 150 ms are in, pauses (dropping nothing) when the model falls behind until 150 ms are in
 * again, and reports when the last sample has played. The full output still arrives in Results
 * from ``outputs_added``.
 */

const PROCESSOR = 'mizuoto-stream-player';
/** Audio buffered before playback starts, and again after the model fell behind. */
export const START_SECONDS = 0.15;

const WORKLET = `
class StreamPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunks = [];
    this.offset = 0;
    this.size = 0;
    this.target = options.processorOptions.target;
    this.playing = false;
    this.ended = false;
    this.drained = false;
    this.played = 0;
    this.quanta = 0;
    this.port.onmessage = (e) => {
      if (e.data === 'end') this.ended = true;
      else { this.chunks.push(e.data); this.size += e.data.length; }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0];
    const ch = out[0];
    const q = ch.length;
    if (!this.playing && this.size > 0 && (this.size >= this.target || this.ended)) this.playing = true;
    let i = 0;
    if (this.playing) {
      while (i < q && this.chunks.length) {
        const c = this.chunks[0];
        const n = Math.min(q - i, c.length - this.offset);
        ch.set(c.subarray(this.offset, this.offset + n), i);
        i += n;
        this.offset += n;
        this.size -= n;
        this.played += n;
        if (this.offset === c.length) { this.chunks.shift(); this.offset = 0; }
      }
      if (i < q && !this.ended) this.playing = false;
    }
    if (i < q) ch.fill(0, i);
    for (let c = 1; c < out.length; c++) out[c].set(ch);
    const drained = this.ended && this.size === 0;
    if (++this.quanta % 6 === 0 || (drained && !this.drained)) {
      this.drained = drained;
      this.port.postMessage({ played: this.played, buffered: this.size, drained });
    }
    return true;
  }
}
registerProcessor('${PROCESSOR}', StreamPlayer);
`;

export type StreamState = 'connecting' | 'queued' | 'playing' | 'finishing' | 'done' | 'failed' | 'stopped';

export interface StreamError {
  code: string;
  message: string;
  detail: Record<string, unknown>;
}

export interface StreamStatus {
  state: StreamState;
  jobId: string | null;
  sampleRate: number;
  granularity: string | null;
  /** Seconds received, played, and waiting to play. */
  received: number;
  played: number;
  buffered: number;
  outputIds: string[];
  error: StreamError | null;
}

/** A message from the stream: audio, or one of the server's JSON events. */
export type StreamMessage =
  | { type: 'audio'; samples: Float32Array }
  | { type: 'queued'; job_id: string }
  | { type: 'start'; job_id: string; sample_rate: number; granularity: string | null }
  | { type: 'segment'; index: number }
  | { type: 'end'; job_id: string; output_ids: string[]; output_id: string | null }
  | ({ type: 'error'; job_id?: string } & StreamError);

export function fromPcm16(data: ArrayBuffer): Float32Array {
  const pcm = new Int16Array(data, 0, data.byteLength >> 1);
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i]! / 32768;
  return out;
}

export function readStreamMessage(data: string | ArrayBuffer): StreamMessage | null {
  if (typeof data !== 'string') return { type: 'audio', samples: fromPcm16(data) };
  try {
    const m = JSON.parse(data) as StreamMessage;
    return m && typeof m === 'object' && 'type' in m ? m : null;
  } catch {
    return null;
  }
}

export function streamUrl(base = BASE_URL, token: string | null = useAuthStore().token): string {
  const url = `${base.replace(/^http/, 'ws')}/api/v1/tts/stream`;
  return token ? `${url}?token=${encodeURIComponent(token)}` : url;
}

const initial = (): StreamStatus => ({ state: 'connecting', jobId: null, sampleRate: 0, granularity: null, received: 0, played: 0, buffered: 0, outputIds: [], error: null });

/** Plays one stream; make a new one per request. */
export class StreamPlayer {
  onStatus: ((status: StreamStatus) => void) | null = null;
  status: StreamStatus = initial();
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private pending: (Float32Array | 'end')[] = [];
  private receivedSamples = 0;

  private patch(p: Partial<StreamStatus>) {
    this.status = { ...this.status, ...p };
    this.onStatus?.(this.status);
  }

  /** Connect and send the request; resolves once the server has queued the job (with its id). */
  start(req: SpeakRequest, url = streamUrl()): Promise<string> {
    this.patch(initial());
    return new Promise<string>((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;
      let queued = false;
      ws.onopen = () => ws.send(JSON.stringify(req));
      ws.onerror = () => {
        if (!queued) reject({ code: 'stream_failed', message: 'The server refused the stream', detail: {} } satisfies StreamError);
      };
      ws.onclose = () => {
        if (!queued) reject({ code: 'stream_failed', message: 'The stream closed before it started', detail: {} } satisfies StreamError);
        else if (['connecting', 'queued', 'playing'].includes(this.status.state) && !this.status.outputIds.length) this.fail({ code: 'stream_failed', message: 'The stream closed', detail: {} });
      };
      ws.onmessage = (e: MessageEvent<string | ArrayBuffer>) => {
        const m = readStreamMessage(e.data);
        if (!m) return;
        if (m.type === 'queued') {
          queued = true;
          this.patch({ state: 'queued', jobId: m.job_id });
          resolve(m.job_id);
        } else if (m.type === 'error' && !queued) {
          reject({ code: m.code, message: m.message, detail: m.detail ?? {} } satisfies StreamError);
          this.close();
        } else this.handle(m);
      };
    });
  }

  private handle(m: StreamMessage) {
    if (m.type === 'start') {
      this.patch({ state: 'playing', sampleRate: m.sample_rate, granularity: m.granularity });
      void this.openAudio(m.sample_rate);
    } else if (m.type === 'audio') {
      this.receivedSamples += m.samples.length;
      this.patch({ received: this.status.sampleRate ? this.receivedSamples / this.status.sampleRate : 0 });
      this.feed(m.samples);
    } else if (m.type === 'end') {
      this.patch({ state: 'finishing', outputIds: m.output_ids });
      this.feed('end');
      this.ws?.close();
      if (!this.receivedSamples) this.finish();
    } else if (m.type === 'error') this.fail({ code: m.code, message: m.message, detail: m.detail ?? {} });
  }

  private feed(item: Float32Array | 'end') {
    if (this.node) this.node.port.postMessage(item, item === 'end' ? [] : [item.buffer]);
    else this.pending.push(item);
  }

  private async openAudio(rate: number) {
    try {
      const ctx = new AudioContext({ sampleRate: rate, latencyHint: 'playback' });
      this.ctx = ctx;
      const module = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      try {
        await ctx.audioWorklet.addModule(module);
      } finally {
        URL.revokeObjectURL(module);
      }
      if (this.ctx !== ctx) return;
      const node = new AudioWorkletNode(ctx, PROCESSOR, { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2], processorOptions: { target: Math.round(rate * START_SECONDS) } });
      node.port.onmessage = (e: MessageEvent<{ played: number; buffered: number; drained: boolean }>) => {
        this.patch({ played: e.data.played / rate, buffered: e.data.buffered / rate });
        if (e.data.drained && this.status.state === 'finishing') this.finish();
      };
      node.connect(ctx.destination);
      this.node = node;
      for (const item of this.pending.splice(0)) this.feed(item);
      await ctx.resume();
    } catch (e) {
      this.fail({ code: 'stream_failed', message: e instanceof Error ? e.message : String(e), detail: {} });
    }
  }

  private finish() {
    this.patch({ state: 'done' });
    this.closeAudio();
  }

  private fail(error: StreamError) {
    if (['done', 'failed', 'stopped'].includes(this.status.state)) return;
    this.patch({ state: error.code === 'cancelled' ? 'stopped' : 'failed', error: error.code === 'cancelled' ? null : error });
    this.close();
  }

  /** Stop listening and cancel the job. */
  cancel() {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'cancel' }));
    if (!['done', 'failed'].includes(this.status.state)) this.patch({ state: 'stopped' });
    this.close();
  }

  /** Stop playing and close the connection (an unfinished job goes on and lands in Results). */
  close() {
    this.ws?.close();
    this.ws = null;
    this.closeAudio();
  }

  private closeAudio() {
    this.node?.disconnect();
    this.node = null;
    this.pending = [];
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}
