import type { Component } from 'vue';
import type { Capability } from '@/api/types';
import { icons } from '@/ui';

/** The capabilities worth showing as icons, in this order, with their icon. */
export const CAPABILITY_ICONS: [Capability, Component][] = [
  ['clone', icons.Copy],
  ['preset_speaker', icons.Users],
  ['voice_design_freeform', icons.WandSparkles],
  ['voice_design', icons.WandSparkles],
  ['instruct', icons.Sparkles],
  ['emotion_vector', icons.SlidersHorizontal],
  ['stream_audio', icons.Waves],
  ['batch', icons.Layers],
];

/** The model sizes a catalogue entry gives, in words: "4.4 GB · 6.3 GB GPU" ("26 MB" below a gigabyte). */
export function sizeText(sizeMb: number | null | undefined, vramMb: number | null | undefined, gpuLabel: (gb: string) => string): string {
  const parts: string[] = [];
  if (sizeMb) parts.push(sizeMb < 1000 ? `${Math.max(1, Math.round(sizeMb))} MB` : `${(sizeMb / 1024).toFixed(1)} GB`);
  if (vramMb) parts.push(gpuLabel((vramMb / 1024).toFixed(1)));
  return parts.join(' · ');
}

/** The languages a dataset can be marked with (the engines' training languages). */
export const TRAIN_LANGUAGES = ['zh', 'en', 'ja', 'ko', 'yue', 'de', 'fr', 'es', 'it', 'pt', 'ru'] as const;
