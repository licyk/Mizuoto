import { describe, expect, it } from 'vitest';
import { encodeWav, toDb } from '@/audio/recorder';

describe('recorder', () => {
  it('writes a mono 16-bit WAV', () => {
    const wav = encodeWav([new Float32Array([0, 1]), new Float32Array([-1, 2])], 16000);
    const view = new DataView(wav);
    const text = (o: number) => String.fromCharCode(...new Uint8Array(wav, o, 4));
    expect(text(0)).toBe('RIFF');
    expect(text(8)).toBe('WAVE');
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint32(40, true)).toBe(8);
    expect([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true))).toEqual([0, 32767, -32767, 32767]);
  });

  it('converts levels to dBFS', () => {
    expect(toDb(1)).toBe(0);
    expect(Math.round(toDb(0.5))).toBe(-6);
    expect(toDb(0)).toBe(-120);
  });
});
