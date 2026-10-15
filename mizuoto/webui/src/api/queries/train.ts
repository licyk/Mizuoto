import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { BASE_URL } from '@/api/baseUrl';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

/** ``<engine>/<name>`` → the path parameters of an experiment's routes. */
export const experimentPath = (key: string) => {
  const [engine, ...rest] = key.split('/');
  return { engine: engine ?? '', name: rest.join('/') };
};

// -- datasets ---------------------------------------------------------------------------------

export const useDatasets = () => useQuery({ queryKey: keys.datasets, queryFn: () => unwrap(api.GET('/api/v1/datasets')) });

export const useDataset = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.dataset(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/datasets/{dataset_id}', { params: { path: { dataset_id: toValue(id) ?? '' } } })),
    enabled: computed(() => !!toValue(id)),
  });

export const useDatasetRows = (id: MaybeRefOrGetter<string>, filter: MaybeRefOrGetter<{ offset: number; limit: number; q?: string; issue?: 'too_short' | 'too_long' | 'empty_text' | 'clipping' }>) =>
  useQuery({
    queryKey: computed(() => keys.datasetRows(toValue(id), { ...toValue(filter) })),
    queryFn: () => unwrap(api.GET('/api/v1/datasets/{dataset_id}/rows', { params: { path: { dataset_id: toValue(id) }, query: { ...toValue(filter) } } })),
    placeholderData: (previous) => previous,
  });

export const useDatasetReport = (id: MaybeRefOrGetter<string>) =>
  useQuery({
    queryKey: computed(() => keys.datasetReport(toValue(id))),
    queryFn: () => unwrap(api.GET('/api/v1/datasets/{dataset_id}/report', { params: { path: { dataset_id: toValue(id) } } })),
  });

export const datasetUrls = {
  rowAudio: (id: string, rowId: string) => `${BASE_URL}/api/v1/datasets/${encodeURIComponent(id)}/rows/${encodeURIComponent(rowId)}/audio`,
};

export function useDatasetActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const refresh = () => qc.invalidateQueries({ queryKey: keys.datasets });
  const path = (id: string) => ({ params: { path: { dataset_id: id } } });
  const create = useMutation({ mutationFn: (body: S['DatasetInput']) => unwrap(api.POST('/api/v1/datasets', { body })), onSuccess: refresh });
  const update = useMutation({ mutationFn: ({ id, patch }: { id: string; patch: S['DatasetPatch'] }) => unwrap(api.PATCH('/api/v1/datasets/{dataset_id}', { ...path(id), body: patch })), onSuccess: refresh });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/datasets/{dataset_id}', path(id))), onSuccess: refresh });
  const addAudio = useMutation({ mutationFn: ({ id, body }: { id: string; body: S['AudioIdsRequest'] }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/audio', { ...path(id), body })), onSuccess: refresh });
  const addVoice = useMutation({ mutationFn: ({ id, voice }: { id: string; voice: string }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/voice', { ...path(id), body: { voice } })), onSuccess: refresh });
  const addOutputs = useMutation({
    mutationFn: ({ id, ids }: { id: string; ids: string[] }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/outputs', { ...path(id), body: { output_ids: ids } })),
    onSuccess: refresh,
  });
  const importFolder = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['DatasetImportRequest'] }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/import', { ...path(id), body })),
    onSuccess: refresh,
  });
  const prepare = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['PrepareRequest'] }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/prepare', { ...path(id), body })),
    onSuccess: (job) => jobs.upsert(job, 'train'),
  });
  const updateRow = useMutation({
    mutationFn: ({ id, row, patch }: { id: string; row: string; patch: S['RowPatch'] }) =>
      unwrap(api.PATCH('/api/v1/datasets/{dataset_id}/rows/{row_id}', { params: { path: { dataset_id: id, row_id: row } }, body: patch })),
    onSuccess: refresh,
  });
  const deleteRows = useMutation({
    mutationFn: ({ id, ids }: { id: string; ids: string[] }) => unwrap(api.POST('/api/v1/datasets/{dataset_id}/rows/delete', { ...path(id), body: { ids } })),
    onSuccess: refresh,
  });
  return { create, update, remove, addAudio, addVoice, addOutputs, importFolder, prepare, updateRow, deleteRows };
}

// -- experiments ------------------------------------------------------------------------------

export const useExperiments = () => useQuery({ queryKey: keys.experiments, queryFn: () => unwrap(api.GET('/api/v1/train/experiments')) });

export const useExperiment = (key: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.experiment(toValue(key) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/train/experiments/{engine}/{name}', { params: { path: experimentPath(toValue(key) ?? '') } })),
    enabled: computed(() => !!toValue(key)),
  });

export const useTrainingMethods = (model: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.trainMethods(toValue(model) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/train/methods', { params: { query: { model: toValue(model) ?? '' } } })),
    enabled: computed(() => !!toValue(model)),
  });

/** Every metric point of an experiment; ``train_metrics`` events append to it. */
export const useExperimentMetrics = (key: MaybeRefOrGetter<string>) =>
  useQuery({
    queryKey: computed(() => keys.experimentMetrics(toValue(key))),
    queryFn: () => unwrap(api.GET('/api/v1/train/experiments/{engine}/{name}/metrics', { params: { path: experimentPath(toValue(key)) } })),
  });

export function useExperimentActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const refresh = () => qc.invalidateQueries({ queryKey: keys.experiments });
  const create = useMutation({ mutationFn: (body: S['ExperimentInput']) => unwrap(api.POST('/api/v1/train/experiments', { body })), onSuccess: refresh });
  const update = useMutation({
    mutationFn: ({ key, patch }: { key: string; patch: S['ExperimentPatch'] }) => unwrap(api.PATCH('/api/v1/train/experiments/{engine}/{name}', { params: { path: experimentPath(key) }, body: patch })),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: (key: string) => unwrap(api.DELETE('/api/v1/train/experiments/{engine}/{name}', { params: { path: experimentPath(key) } })), onSuccess: refresh });
  const run = useMutation({
    mutationFn: ({ key, body }: { key: string; body: S['RunRequest'] }) => unwrap(api.POST('/api/v1/train/experiments/{engine}/{name}/run', { params: { path: experimentPath(key) }, body })),
    onSuccess: (job) => {
      jobs.upsert(job, 'train');
      refresh();
      qc.invalidateQueries({ queryKey: keys.metrics });
    },
  });
  return { create, update, remove, run };
}
