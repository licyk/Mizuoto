import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import { useJobsStore } from '@/stores/jobs';

/** ``engine/name`` as the path parameters model routes take. */
export const modelPath = (id: string) => {
  const [engine, ...rest] = id.split('/');
  return { engine, name: rest.join('/') };
};

export const useModels = (filter: MaybeRefOrGetter<{ engine?: string; installed?: boolean }> = {}) =>
  useQuery({
    queryKey: computed(() => keys.modelList({ ...toValue(filter) })),
    queryFn: () => unwrap(api.GET('/api/v1/models', { params: { query: { ...toValue(filter) } } })),
  });

/** What a model takes: voice modes, languages, parameters (fixed for a model; refreshed when engines change). */
export const useModelSpec = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.modelSpec(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/tts/models/{engine}/{name}/spec', { params: { path: modelPath(toValue(id) ?? '') } })),
    enabled: computed(() => !!toValue(id)),
    staleTime: Infinity,
  });

export function useModelActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const refresh = () => qc.invalidateQueries({ queryKey: keys.models });
  const download = useMutation({
    mutationFn: (ids: string[]) => unwrap(api.POST('/api/v1/models/download', { body: { ids } })),
    onSuccess: (job) => {
      jobs.upsert(job, 'models');
      refresh();
    },
  });
  const verify = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/models/{engine}/{name}/verify', { params: { path: modelPath(id) } })), onSuccess: refresh });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/models/{engine}/{name}', { params: { path: modelPath(id) } })), onSuccess: refresh });
  const importFolder = useMutation({
    mutationFn: (body: { path: string; name?: string | null; move?: boolean }) => unwrap(api.POST('/api/v1/models/import', { body })),
    onSuccess: refresh,
  });
  return { download, verify, remove, importFolder };
}

/** Files several models of an engine share (encoders, text front-end data), downloaded once. */
export const useAssets = () => useQuery({ queryKey: keys.assets, queryFn: () => unwrap(api.GET('/api/v1/assets')) });

export function useAssetActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: keys.assets }), qc.invalidateQueries({ queryKey: keys.models })]);
  const download = useMutation({
    mutationFn: (ids: string[]) => unwrap(api.POST('/api/v1/assets/download', { body: { ids } })),
    onSuccess: (job) => {
      jobs.upsert(job, 'models');
      refresh();
    },
  });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/assets/{engine}/{name}', { params: { path: modelPath(id) } })), onSuccess: refresh });
  return { download, remove };
}
