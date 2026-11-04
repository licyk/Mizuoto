import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

export interface OutputFilter {
  job_id?: string;
  voice_id?: string;
  model?: string;
  engine?: string;
  kind?: string;
  q?: string;
  favourite?: boolean;
}

export const useOutputs = (filter: MaybeRefOrGetter<OutputFilter>, enabled: MaybeRefOrGetter<boolean> = true, limit: MaybeRefOrGetter<number> = 200) =>
  useQuery({
    queryKey: computed(() => keys.outputList({ ...toValue(filter), limit: String(toValue(limit)) })),
    queryFn: () => unwrap(api.GET('/api/v1/outputs', { params: { query: { ...toValue(filter), limit: toValue(limit) } } })),
    enabled: computed(() => toValue(enabled)),
  });

/** An output's spectrogram and pitch curve (computed by the server once, then cached); ``id`` null: nothing asked. */
export const useOutputAnalysis = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.outputAnalysis(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/outputs/{output_id}/analysis', { params: { path: { output_id: toValue(id) as string } } })),
    enabled: computed(() => !!toValue(id)),
    staleTime: Infinity,
  });

export const fetchOutput = (id: string) => unwrap(api.GET('/api/v1/outputs/{output_id}', { params: { path: { output_id: id } } }));

export function useDeleteOutputs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ids: string[]) => unwrap(api.POST('/api/v1/outputs/delete', { body: { ids } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.outputs }),
  });
}

export function useUpdateOutput() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name, favourite }: { id: string; name?: string; favourite?: boolean }) =>
      unwrap(api.PATCH('/api/v1/outputs/{output_id}', { params: { path: { output_id: id } }, body: { name, favourite } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.outputs }),
  });
}

export const revealOutput = (id: string) => unwrap(api.POST('/api/v1/outputs/{output_id}/reveal', { params: { path: { output_id: id } } }));

/** The request that made an output, to run it again. */
export const fetchRerun = (id: string) => unwrap(api.GET('/api/v1/outputs/{output_id}/rerun', { params: { path: { output_id: id } } }));
