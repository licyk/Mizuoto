import { useQuery } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { SpeakRequest } from '@/api/types';

export const planSpeech = (req: SpeakRequest) => unwrap(api.POST('/api/v1/tts/plan', { body: req }));

export const speak = (req: SpeakRequest) => unwrap(api.POST('/api/v1/tts/speak', { body: req }));

/** Where Mizuoto will cut the text for this model (debounced by the caller). */
export const useSegments = (text: MaybeRefOrGetter<string>, model: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.segments(toValue(text), toValue(model) ?? '')),
    queryFn: () => unwrap(api.POST('/api/v1/tts/segment', { body: { text: toValue(text), model: toValue(model) } })),
    enabled: computed(() => !!toValue(text).trim()),
    staleTime: Infinity,
  });
