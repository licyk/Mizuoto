import { defineStore } from 'pinia';
import { ref } from 'vue';
import type { SpeakRequestOut } from '@/api/types';

/**
 * Hand-offs between screens: "Re-run with these settings" (History), "Use this voice" (Voices) and
 * "Use as reference" put something here, and Speak takes it when it opens.
 */
export const useHandoffStore = defineStore('handoff', () => {
  const request = ref<SpeakRequestOut | null>(null);
  const voice = ref<string | null>(null);
  const reference = ref<{ audioId: string; name: string; transcript?: string | null } | null>(null);

  function take<T>(r: { value: T | null }): T | null {
    const v = r.value;
    r.value = null;
    return v;
  }
  return {
    request,
    voice,
    reference,
    sendRequest: (r: SpeakRequestOut) => (request.value = r),
    sendVoice: (id: string) => (voice.value = id),
    sendReference: (audioId: string, name: string, transcript?: string | null) => (reference.value = { audioId, name, transcript }),
    takeRequest: () => take(request),
    takeVoice: () => take(voice),
    takeReference: () => take(reference),
  };
});
