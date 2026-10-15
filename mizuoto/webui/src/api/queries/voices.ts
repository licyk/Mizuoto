import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { BASE_URL } from '@/api/baseUrl';
import { keys } from '@/api/queries/keys';
import type { S, Voice } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const useVoices = (filter: MaybeRefOrGetter<{ q?: string; kind?: string; language?: string }> = {}) =>
  useQuery({
    queryKey: computed(() => keys.voiceList({ ...toValue(filter) })),
    queryFn: () => unwrap(api.GET('/api/v1/voices', { params: { query: { ...toValue(filter) } } })),
  });

export const useVoice = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.voice(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/voices/{voice_id}', { params: { path: { voice_id: toValue(id) ?? '' } } })),
    enabled: computed(() => !!toValue(id)),
  });

export const useVoiceFits = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.voiceFits(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/voices/{voice_id}/fits', { params: { path: { voice_id: toValue(id) ?? '' } } })),
    enabled: computed(() => !!toValue(id)),
  });

export const fetchVoiceFit = (id: string, model: string) => unwrap(api.GET('/api/v1/voices/{voice_id}/fit', { params: { path: { voice_id: id }, query: { model } } }));

export const voiceUrls = {
  reference: (voiceId: string, sha: string) => `${BASE_URL}/api/v1/voices/${encodeURIComponent(voiceId)}/references/${sha}/content`,
  export: (voiceId: string) => `${BASE_URL}/api/v1/voices/${encodeURIComponent(voiceId)}/export`,
};

export const importVoice = (file: File, name?: string) => uploadRaw<Voice>('/api/v1/voices/import', name ? { name } : {}, file);

export function useVoiceActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const refresh = () => qc.invalidateQueries({ queryKey: keys.voices });
  const create = useMutation({ mutationFn: (body: S['CreateVoice']) => unwrap(api.POST('/api/v1/voices', { body })), onSuccess: refresh });
  const fromOutput = useMutation({ mutationFn: (body: S['FromOutput']) => unwrap(api.POST('/api/v1/voices/from-output', { body })), onSuccess: refresh });
  const fromPreset = useMutation({ mutationFn: (body: S['FromPreset']) => unwrap(api.POST('/api/v1/voices/preset', { body })), onSuccess: refresh });
  const update = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: S['VoicePatch'] }) => unwrap(api.PATCH('/api/v1/voices/{voice_id}', { params: { path: { voice_id: id } }, body: patch })),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/voices/{voice_id}', { params: { path: { voice_id: id } } })), onSuccess: refresh });
  const addReference = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['AddReference'] }) => unwrap(api.POST('/api/v1/voices/{voice_id}/references', { params: { path: { voice_id: id } }, body })),
    onSuccess: refresh,
  });
  const updateReference = useMutation({
    mutationFn: ({ id, sha, patch }: { id: string; sha: string; patch: S['ReferencePatch'] }) =>
      unwrap(api.PATCH('/api/v1/voices/{voice_id}/references/{sha}', { params: { path: { voice_id: id, sha } }, body: patch })),
    onSuccess: refresh,
  });
  const removeReference = useMutation({
    mutationFn: ({ id, sha }: { id: string; sha: string }) => unwrap(api.DELETE('/api/v1/voices/{voice_id}/references/{sha}', { params: { path: { voice_id: id, sha } } })),
    onSuccess: refresh,
  });
  const retranscribe = useMutation({
    mutationFn: (id: string) => unwrap(api.POST('/api/v1/voices/{voice_id}/retranscribe', { params: { path: { voice_id: id } } })),
    onSuccess: refresh,
  });
  const prepare = useMutation({
    mutationFn: ({ id, model }: { id: string; model: string }) => unwrap(api.POST('/api/v1/voices/{voice_id}/prepare', { params: { path: { voice_id: id }, query: { model } } })),
    onSuccess: (job) => jobs.upsert(job, 'voices'),
  });
  return { create, fromOutput, fromPreset, update, remove, addReference, updateReference, removeReference, retranscribe, prepare };
}

/** How a model can use each library voice (the voice picker's badges). */
export const useModelVoiceFits = (model: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => ['voices', 'fits-for', toValue(model) ?? ''] as const),
    queryFn: () => {
      const [engine, ...rest] = (toValue(model) ?? '').split('/');
      return unwrap(api.GET('/api/v1/tts/models/{engine}/{name}/voice-fits', { params: { path: { engine, name: rest.join('/') } } }));
    },
    enabled: computed(() => !!toValue(model)),
  });
