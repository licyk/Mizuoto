import { describe, expect, it } from 'vitest';
import { ApiError } from '@/api/client';
import { errorReport, errorView } from '@/components/errors';
import en from '@/i18n/en';

function lookup(key: string): string | undefined {
  let node: unknown = en;
  for (const part of key.split('.')) node = (node as Record<string, unknown> | undefined)?.[part];
  return typeof node === 'string' ? node : undefined;
}
const t = (key: string) => lookup(key) ?? key;
const tOr = (key: string, fallback: string) => lookup(key) ?? fallback;

describe('error view', () => {
  it("shows the server's hint apart from the details", () => {
    const view = errorView(
      new ApiError(409, 'model_missing', 'Qwen3-TTS 1.7B Base is not downloaded', { models: [{ id: 'qwen3tts/1.7b-base', size_mb: 4300 }], hint: 'mizuoto model download qwen3tts/1.7b-base' }),
      t,
      tOr,
    );
    expect(view.title).toBe('The model is not downloaded');
    expect(view.hint).toBe('mizuoto model download qwen3tts/1.7b-base');
    expect([...view.fields, ...view.blocks].map((f) => f.key)).toEqual(['models']);
  });

  it('keeps tracebacks and process output readable, and copies everything', () => {
    const view = errorView(
      { code: 'internal_error', message: 'The qwen3tts worker stopped unexpectedly (killed by SIGSEGV (signal 11))', detail: { exit_code: -11, log: 'loading\nFatal Python error: Segmentation fault' } },
      t,
      tOr,
    );
    expect(view.title).toBe('A background process stopped unexpectedly');
    expect(view.fields).toEqual([{ key: 'exit_code', label: 'Exit code', value: '-11' }]);
    expect(view.blocks).toEqual([{ key: 'log', label: 'Process output', value: 'loading\nFatal Python error: Segmentation fault' }]);
    expect(errorReport(view, 'Error code')).toBe(
      [
        'A background process stopped unexpectedly',
        'The qwen3tts worker stopped unexpectedly (killed by SIGSEGV (signal 11))',
        '',
        'Error code: internal_error',
        'Exit code: -11',
        '',
        'Process output:',
        'loading\nFatal Python error: Segmentation fault',
      ].join('\n'),
    );
  });

  it('falls back to the code title and the raw key for unknown fields', () => {
    const view = errorView({ code: 'capability_not_supported', message: 'No preset speakers', detail: { capability: 'preset_speaker', other: 3 } }, t, tOr);
    expect(view.title).toBe('The model cannot do that');
    expect(view.fields).toEqual([
      { key: 'capability', label: 'Capability', value: 'preset_speaker' },
      { key: 'other', label: 'other', value: '3' },
    ]);
    expect(errorView({ code: 'something_new', message: 'x' }, t, tOr).title).toBe('Something went wrong');
  });
});
