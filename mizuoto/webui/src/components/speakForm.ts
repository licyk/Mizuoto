import type { EmotionInput, ModelParamSpecs, SpeakRequest, SpeakRequestOut, VoiceMode } from '@/api/types';
import { inputKey, type InputItem } from '@/components/inputs';

/**
 * What a model takes of the Speak form, and the request the form makes for it.
 *
 * The form keeps what was set even where the chosen model has no use for it, so that switching back
 * to a model brings it back. The request is therefore built here, from the parts the model takes, by
 * the same rules that decide what the form shows: a value the form does not show is never sent. The
 * server refuses what a model does not take (an explicit request is never trimmed), so a hidden value
 * that was sent would fail the synthesis.
 */

export const VOICE_KINDS = ['library', 'clip', 'preset', 'design', 'auto'] as const;
export type VoiceKind = (typeof VOICE_KINDS)[number];

// -- emotion ----------------------------------------------------------------------------------

export const EMOTION_WAYS = ['vector', 'audio', 'text'] as const;
export type EmotionWay = (typeof EMOTION_WAYS)[number];

/** Everything the emotion panel holds: each way's values, kept while another way is shown. */
export interface EmotionForm {
  way: EmotionWay;
  weights: Record<string, number>;
  clip: InputItem[];
  strength: number;
  described: string;
}
export const emptyEmotion = (): EmotionForm => ({ way: 'vector', weights: {}, clip: [], strength: 1, described: '' });

/** The ways the model takes an emotion, in the panel's order (weights only where it names its emotions). */
export function emotionWays(spec: ModelParamSpecs): EmotionWay[] {
  const caps = new Set(spec.capabilities);
  return EMOTION_WAYS.filter((w) => caps.has(`emotion_${w}`) && (w !== 'vector' || spec.emotions.length > 0));
}

/** The way shown: the one chosen when the model takes it, else its first. */
export function shownWay(form: EmotionForm, ways: EmotionWay[]): EmotionWay | null {
  return ways.includes(form.way) ? form.way : (ways[0] ?? null);
}

/** The emotion to send: the shown way's values (weights only for the model's own emotions), or null. */
export function emotionInput(spec: ModelParamSpecs, form: EmotionForm): EmotionInput | null {
  const way = shownWay(form, emotionWays(spec));
  if (way === 'vector') {
    const known = new Set(spec.emotions);
    const set = Object.fromEntries(Object.entries(form.weights).filter(([k, v]) => known.has(k) && v > 0));
    return Object.keys(set).length ? { vector: set } : null;
  }
  if (way === 'audio') {
    const ready = form.clip.find((c) => c.status === 'ready' && c.audioId);
    return ready ? { audio_id: ready.audioId, strength: form.strength } : null;
  }
  if (way === 'text') return form.described.trim() ? { text: form.described.trim() } : null;
  return null;
}

/** The panel's values for an emotion an earlier request sent ("Run again"). */
export function emotionForm(e: EmotionInput | null | undefined): EmotionForm {
  const form = emptyEmotion();
  if (e?.vector && Object.keys(e.vector).length) return { ...form, way: 'vector', weights: { ...e.vector } };
  if (e?.audio_id) return { ...form, way: 'audio', clip: [{ key: inputKey(), name: e.audio_id, audioId: e.audio_id, status: 'ready' }], strength: e.strength ?? 1 };
  if (e?.text) return { ...form, way: 'text', described: e.text };
  return form;
}

// -- instruction ------------------------------------------------------------------------------

/** The voice modes a kind of voice gives the model; null where it is not known yet (a library voice before its fit). */
export function voiceModes(kind: VoiceKind, libraryMode: VoiceMode | null | undefined): VoiceMode[] | null {
  if (kind === 'library') return libraryMode ? [libraryMode] : null;
  if (kind === 'clip') return ['clone', 'clone_no_transcript'];
  return [kind];
}

/**
 * Whether the model takes an instruction: ``taken`` with this voice, ``offered`` at all (the field is
 * shown, and says which voice it needs when it is not taken). A voice whose mode is not known yet
 * counts as fitting; the server decides then.
 */
export function instructFit(spec: ModelParamSpecs, modes: VoiceMode[] | null): { offered: boolean; taken: boolean; requires: VoiceMode[] } {
  const offered = spec.capabilities.includes('instruct');
  const requires = spec.instruct_requires ?? [];
  const taken = offered && (!requires.length || modes === null || modes.some((m) => requires.includes(m)));
  return { offered, taken, requires };
}

// -- language, voice design, parameters ------------------------------------------------------

/** Whether the model speaks ``code``, as the server decides it (``*`` covers everything; ``zh-*`` covers ``zh-sichuan``). */
export function speaks(languages: string[], code: string): boolean {
  if (code === 'auto' || !languages.length || languages.includes('*') || languages.includes(code)) return true;
  return languages.some((l) => l.endsWith('-*') && (code === l.slice(0, -2) || code.startsWith(l.slice(0, -1))));
}

/** The design to send: a description where the model takes one, attributes from its vocabulary. */
export function designFor(spec: ModelParamSpecs, description: string, attributes: Record<string, string>): { description: string | null; attributes: Record<string, string> | null } {
  const caps = new Set(spec.capabilities);
  const vocabulary = caps.has('voice_design') ? (spec.design_attributes ?? {}) : {};
  const known = Object.fromEntries(Object.entries(attributes).filter(([k, v]) => vocabulary[k]?.includes(v)));
  return { description: caps.has('voice_design_freeform') && description.trim() ? description.trim() : null, attributes: Object.keys(known).length ? known : null };
}

/** Whether the model takes a seed. */
export const takesSeed = (spec: ModelParamSpecs): boolean => spec.params.some((p) => p.name === 'seed');

/** The values of the model's own fields of ``kind`` (common parameters or engine options); others are dropped. */
export function fieldsFor(spec: ModelParamSpecs, kind: 'common' | 'option', values: Record<string, unknown>): Record<string, unknown> {
  const names = new Set(spec.params.filter((p) => p.kind === kind && p.name !== 'seed').map((p) => p.name));
  return Object.fromEntries(Object.entries(values).filter(([k, v]) => names.has(k) && v !== null && v !== undefined));
}

// -- the request ------------------------------------------------------------------------------

export interface SpeakForm {
  model: string;
  text: string;
  batch: boolean;
  language: string;
  voiceKind: VoiceKind;
  voiceId: string | null;
  /** The library voice's mode for this model, from its fit (null while it is not known). */
  libraryMode: VoiceMode | null;
  clip: InputItem[];
  transcript: string;
  speaker: string | null;
  description: string;
  attributes: Record<string, string>;
  instruct: string;
  emotion: EmotionForm;
  params: Record<string, unknown>;
  options: Record<string, unknown>;
  seed: number | null;
  output: SpeakRequest['output'];
}

/** The request for ``spec``'s model: every part of the form the model takes, and nothing else. */
export function speakRequest(spec: ModelParamSpecs, form: SpeakForm): SpeakRequest {
  const batch = form.batch && spec.capabilities.includes('batch');
  const instruct = instructFit(spec, voiceModes(form.voiceKind, form.libraryMode)).taken && form.instruct.trim() ? form.instruct.trim() : null;
  const emotion = emotionInput(spec, form.emotion);
  const req: SpeakRequest = {
    model: form.model,
    text: batch ? '' : form.text,
    texts: batch ? form.text.split('\n').map((l) => l.trim()).filter(Boolean) : null,
    language: form.language !== 'auto' && speaks(spec.languages, form.language) ? form.language : null,
    params: { ...fieldsFor(spec, 'common', form.params), ...(form.seed !== null && takesSeed(spec) ? { seed: form.seed } : {}) },
    engine_options: fieldsFor(spec, 'option', form.options),
    output: form.output,
    style: instruct || emotion ? { instruct, emotion } : null,
  };
  if (form.voiceKind === 'library') req.voice = form.voiceId;
  else if (form.voiceKind === 'clip') {
    const c = form.clip.find((i) => i.status === 'ready' && i.audioId);
    if (c) req.voice_spec = { references: [{ audio_id: c.audioId, transcript: form.transcript.trim() || null }] };
  } else if (form.voiceKind === 'preset') req.voice_spec = { preset: form.speaker };
  else if (form.voiceKind === 'design') req.voice_spec = designFor(spec, form.description, form.attributes);
  return req;
}

/** The parameters of an earlier request, without the seed (it has its own field) and the unset ones. */
export function rerunParams(r: SpeakRequestOut): { params: Record<string, unknown>; seed: number | null } {
  const { seed, ...rest } = (r.params ?? {}) as Record<string, unknown>;
  return { params: Object.fromEntries(Object.entries(rest).filter(([, v]) => v !== null && v !== undefined)), seed: typeof seed === 'number' ? seed : null };
}
