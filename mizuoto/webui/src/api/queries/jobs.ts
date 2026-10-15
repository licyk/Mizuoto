import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { Job } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const fetchActiveJobs = () => unwrap(api.GET('/api/v1/jobs', { params: { query: { state: 'active', limit: 200 } } }));

export const useJobList = (filter: MaybeRefOrGetter<{ state?: string; kind?: string }>) =>
  useQuery({
    queryKey: computed(() => keys.jobList({ ...toValue(filter) })),
    queryFn: () => unwrap(api.GET('/api/v1/jobs', { params: { query: { ...toValue(filter), limit: 200 } } })),
  });

export const fetchJobLog = (id: string, offset: number) => unwrap(api.GET('/api/v1/jobs/{job_id}/log', { params: { path: { job_id: id }, query: { offset } } }));

export function useJobActions() {
  const store = useJobsStore();
  const qc = useQueryClient();
  const apply = (job: Job) => store.upsert(job);
  const cancel = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/jobs/{job_id}/cancel', { params: { path: { job_id: id } } })), onSuccess: apply });
  const retry = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/jobs/{job_id}/retry', { params: { path: { job_id: id } } })), onSuccess: (job) => store.upsert(job, 'jobs') });
  const runAnyway = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/jobs/{job_id}/run-anyway', { params: { path: { job_id: id } } })), onSuccess: apply });
  const remove = useMutation({
    mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/jobs/{job_id}', { params: { path: { job_id: id } } })),
    onSuccess: (_d, id) => {
      store.remove([id]);
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
  const clear = useMutation({
    mutationFn: () => unwrap(api.DELETE('/api/v1/jobs', { params: { query: { finished: true } } })),
    onSuccess: (data) => {
      store.remove(data.ids);
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
  return { cancel, retry, runAnyway, remove, clear };
}
