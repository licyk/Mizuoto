import { describe, expect, it } from 'vitest';
import type { Capability, ModelParamSpecs, ParamField } from '@/api/types';
import { emotionForm, emptyEmotion, instructFit, speakRequest, speaks, type SpeakForm } from '@/components/speakForm';

const field = (name: string, kind: ParamField['kind'] = 'common'): ParamField => ({ name, kind, type: 'float', default: null, min: null, max: null, step: null, choices: null, group: 'basic', help: '', approximate: false, streaming: true });

function spec(model: string, capabilities: Capability[], extra: Partial<ModelParamSpecs> = {}): ModelParamSpecs {
  return {
    model,
    engine: model.split('/')[0],
    variant: model.split('/')[1],
    sample_rate: 24000,
    capabilities,
    voice_modes: [],
    languages: ['zh', 'en'],
    params: [field('temperature'), field('seed')],
    reference: { min_seconds: 0.5, max_seconds: 30, min_sample_rate: null, max_references: 1 },
    instruct_requires: [],
    design_attributes: null,
    inline_tags: [],
    emotions: [],
    speakers: [],
    stream_granularity: 'emulated',
    ...extra,
  };
}

// Shaped like the real models: what each takes of the form.
const INDEXTTS = spec('indextts/indextts2', ['clone', 'emotion_vector', 'emotion_audio', 'emotion_text'], { emotions: ['happy', 'angry', 'sad'], params: [field('temperature'), field('seed'), field('use_random', 'option')] });
const QWEN_BASE = spec('qwen3tts/1.7b-base', ['clone', 'clone_no_transcript', 'batch', 'seed'], { languages: ['zh', 'en', 'ja', 'yue'] });
const QWEN_DESIGN = spec('qwen3tts/1.7b-voicedesign', ['voice_design_freeform', 'instruct', 'batch']);
const OMNIVOICE = spec('omnivoice/default', ['clone', 'voice_design', 'auto_voice'], { languages: ['*'], design_attributes: { gender: ['female', 'male'], age: ['young', 'adult'] } });
const COSYVOICE = spec('cosyvoice/v2', ['clone', 'clone_no_transcript', 'instruct'], { instruct_requires: ['clone', 'clone_no_transcript'] });

/** A form with everything set, as after using every model in turn. */
function full(model: string, patch: Partial<SpeakForm> = {}): SpeakForm {
  return {
    model,
    text: 'one\ntwo',
    batch: true,
    language: 'yue',
    voiceKind: 'clip',
    voiceId: 'v1',
    libraryMode: null,
    clip: [{ key: 'k', name: 'a.wav', audioId: 'a1', status: 'ready' }],
    transcript: 'hello',
    speaker: 'alto',
    description: 'A warm voice',
    attributes: { gender: 'female', accent: 'british' },
    instruct: 'Cheerfully',
    emotion: { ...emptyEmotion(), weights: { happy: 0.6, calm: 0.4 } },
    params: { temperature: 0.7, top_k: 20 },
    options: { use_random: true, fused_projections: true },
    seed: 7,
    output: { format: 'wav' },
    ...patch,
  };
}

describe('speak request', () => {
  it('sends a model nothing the form keeps for another (IndexTTS2 → Qwen3-TTS Base)', () => {
    const req = speakRequest(QWEN_BASE, full(QWEN_BASE.model));
    expect(req.style).toBeNull();
    expect(req.engine_options).toEqual({});
    expect(req.params).toEqual({ temperature: 0.7, seed: 7 });
    expect(req.texts).toEqual(['one', 'two']);
    expect(req.language).toBe('yue');
  });

  it('sends the emotion a model takes, of the way shown, for its own emotions', () => {
    expect(speakRequest(INDEXTTS, full(INDEXTTS.model)).style).toEqual({ instruct: null, emotion: { vector: { happy: 0.6 } } });
    // The way chosen elsewhere is not one this model takes: its first way is shown, and sent.
    const text = { ...emptyEmotion(), way: 'text' as const, described: 'angry', weights: { sad: 0.5 } };
    expect(speakRequest(spec('x/y', ['emotion_vector'], { emotions: ['sad'] }), full('x/y', { emotion: text })).style?.emotion).toEqual({ vector: { sad: 0.5 } });
    expect(speakRequest(spec('x/y', ['emotion_vector']), full('x/y', { emotion: text })).style).toBeNull();
    expect(speakRequest(INDEXTTS, full(INDEXTTS.model, { emotion: text })).style?.emotion).toEqual({ text: 'angry' });
  });

  it('sends a batch, a language and engine options only to a model that takes them', () => {
    const req = speakRequest(INDEXTTS, full(INDEXTTS.model));
    expect(req.texts).toBeNull();
    expect(req.text).toBe('one\ntwo');
    expect(req.language).toBeNull();
    expect(req.engine_options).toEqual({ use_random: true });
    expect(req.params).toEqual({ temperature: 0.7, seed: 7 });
    expect(speakRequest(spec('x/y', ['clone'], { params: [field('temperature')] }), full('x/y')).params).toEqual({ temperature: 0.7 });
  });

  it('sends a description or attributes only where the model designs voices that way', () => {
    const design = (s: ModelParamSpecs) => speakRequest(s, full(s.model, { voiceKind: 'design' })).voice_spec;
    expect(design(QWEN_DESIGN)).toEqual({ description: 'A warm voice', attributes: null });
    expect(design(OMNIVOICE)).toEqual({ description: null, attributes: { gender: 'female' } });
  });

  it('sends an instruction only with a voice the model follows it with', () => {
    expect(speakRequest(QWEN_DESIGN, full(QWEN_DESIGN.model, { voiceKind: 'design' })).style?.instruct).toBe('Cheerfully');
    expect(speakRequest(COSYVOICE, full(COSYVOICE.model)).style).toEqual({ instruct: 'Cheerfully', emotion: null });
    expect(speakRequest(COSYVOICE, full(COSYVOICE.model, { voiceKind: 'preset' })).style).toBeNull();
    expect(speakRequest(COSYVOICE, full(COSYVOICE.model, { voiceKind: 'library', libraryMode: 'preset' })).style).toBeNull();
    // A library voice whose fit is not known yet: the server decides.
    expect(instructFit(COSYVOICE, null)).toEqual({ offered: true, taken: true, requires: ['clone', 'clone_no_transcript'] });
    expect(instructFit(QWEN_BASE, ['clone']).offered).toBe(false);
  });

  it('gives each kind of voice its own fields', () => {
    expect(speakRequest(QWEN_BASE, full(QWEN_BASE.model)).voice_spec).toEqual({ references: [{ audio_id: 'a1', transcript: 'hello' }] });
    const library = speakRequest(QWEN_BASE, full(QWEN_BASE.model, { voiceKind: 'library' }));
    expect([library.voice, library.voice_spec]).toEqual(['v1', undefined]);
    expect(speakRequest(QWEN_BASE, full(QWEN_BASE.model, { voiceKind: 'auto' })).voice_spec).toBeUndefined();
  });
});

describe('speak form', () => {
  it('reads languages as the server does', () => {
    expect(speaks(['*'], 'yue')).toBe(true);
    expect(speaks(['zh-*', 'en'], 'zh-sichuan')).toBe(true);
    expect(speaks(['zh', 'en'], 'yue')).toBe(false);
  });

  it('brings back the emotion an earlier request sent', () => {
    expect(emotionForm({ vector: { happy: 0.5 } })).toMatchObject({ way: 'vector', weights: { happy: 0.5 } });
    expect(emotionForm({ audio_id: 'a2', strength: 0.8 })).toMatchObject({ way: 'audio', strength: 0.8, clip: [{ audioId: 'a2', status: 'ready' }] });
    expect(emotionForm({ text: 'calm' })).toMatchObject({ way: 'text', described: 'calm' });
    expect(emotionForm(null)).toEqual(emptyEmotion());
  });
});
