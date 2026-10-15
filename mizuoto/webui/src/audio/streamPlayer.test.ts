import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { StreamPlayer, fromPcm16, readStreamMessage, streamUrl } from '@/audio/streamPlayer';

/** A WebSocket stand-in: records what is sent and lets the test deliver messages. */
class FakeSocket {
  static last: FakeSocket | null = null;
  static OPEN = 1;
  readyState = 1;
  binaryType = '';
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: string | ArrayBuffer }) => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.last = this;
    queueMicrotask(() => this.onopen?.());
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
  }
  deliver(data: unknown) {
    this.onmessage?.({ data: data instanceof ArrayBuffer ? data : JSON.stringify(data) });
  }
}

describe('stream messages', () => {
  it('reads PCM16 and JSON', () => {
    const pcm = new Int16Array([0, 16384, -32768]).buffer;
    expect(Array.from(fromPcm16(pcm))).toEqual([0, 0.5, -1]);
    const audio = readStreamMessage(pcm);
    expect(audio?.type).toBe('audio');
    expect(readStreamMessage('{"type":"segment","index":2}')).toEqual({ type: 'segment', index: 2 });
    expect(readStreamMessage('not json')).toBeNull();
  });

  it('builds the URL with the token', () => {
    expect(streamUrl('http://h:7869', 'a b')).toBe('ws://h:7869/api/v1/tts/stream?token=a%20b');
    expect(streamUrl('https://h', null)).toBe('wss://h/api/v1/tts/stream');
  });
});

describe('StreamPlayer', () => {
  beforeEach(() => vi.stubGlobal('WebSocket', FakeSocket));
  afterEach(() => vi.unstubAllGlobals());

  it('sends the request, resolves with the job and ends', async () => {
    const player = new StreamPlayer();
    const states: string[] = [];
    player.onStatus = (s) => states.push(s.state);
    const started = player.start({ model: 'dummy/default', text: 'Hi.' }, 'ws://x/stream');
    await Promise.resolve();
    const ws = FakeSocket.last!;
    expect(JSON.parse(ws.sent[0]!)).toEqual({ model: 'dummy/default', text: 'Hi.' });
    ws.deliver({ type: 'queued', job_id: 'j1' });
    await expect(started).resolves.toBe('j1');
    ws.deliver({ type: 'end', job_id: 'j1', output_ids: ['o1'], output_id: 'o1' });
    expect(player.status.state).toBe('done');
    expect(player.status.outputIds).toEqual(['o1']);
    expect(states).toContain('queued');
  });

  it('rejects a refused request and reports a failed job', async () => {
    const player = new StreamPlayer();
    const refused = player.start({ model: 'm', text: '' }, 'ws://x/stream');
    await Promise.resolve();
    FakeSocket.last!.deliver({ type: 'error', code: 'invalid_input', message: 'There is no text to speak', detail: {} });
    await expect(refused).rejects.toMatchObject({ code: 'invalid_input' });

    const second = new StreamPlayer();
    const started = second.start({ model: 'm', text: 'x' }, 'ws://x/stream');
    await Promise.resolve();
    FakeSocket.last!.deliver({ type: 'queued', job_id: 'j2' });
    await started;
    FakeSocket.last!.deliver({ type: 'error', job_id: 'j2', code: 'engine_error', message: 'boom', detail: {} });
    expect(second.status.state).toBe('failed');
    expect(second.status.error?.message).toBe('boom');
  });

  it('cancels the job', async () => {
    const player = new StreamPlayer();
    const started = player.start({ model: 'm', text: 'x' }, 'ws://x/stream');
    await Promise.resolve();
    const ws = FakeSocket.last!;
    ws.deliver({ type: 'queued', job_id: 'j3' });
    await started;
    player.cancel();
    expect(JSON.parse(ws.sent[1]!)).toEqual({ type: 'cancel' });
    expect(player.status.state).toBe('stopped');
  });
});
