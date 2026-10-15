import { computed } from 'vue';
import { useOutputs } from '@/api/queries/outputs';
import type { Output } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

/** Outputs of the jobs started from ``screen`` in this session, newest first. */
export function useSessionOutputs(screen: string, kinds?: string[]) {
  const jobs = useJobsStore();
  const ids = computed(() => new Set(jobs.fromScreen(screen).map((j) => j.id)));
  const outputs = useOutputs({}, () => ids.value.size > 0);
  return computed<Output[]>(() => (outputs.data.value?.items ?? []).filter((o) => o.job_id && ids.value.has(o.job_id) && (!kinds || kinds.includes(o.kind))));
}
