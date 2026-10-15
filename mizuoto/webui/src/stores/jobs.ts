import { defineStore } from 'pinia';
import { computed, reactive } from 'vue';
import type { Job, JobState } from '@/api/types';

const FINISHED: ReadonlySet<JobState> = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
export const isFinished = (state: JobState) => FINISHED.has(state);

/**
 * How far along a job's life a state is. Events and REST snapshots can arrive out of order (a
 * socket event racing a fetch); a snapshot whose state is earlier than the one held is stale.
 * Queued and waiting share a rank: a job may move between them.
 */
export function stateRank(state: JobState): number {
  if (state === 'queued' || state === 'waiting') return 0;
  if (state === 'running') return 1;
  if (state === 'cancelling') return 2;
  return 3;
}

/** Whether ``next`` should replace ``prev``: never go back in a job's life, never lose a finished result. */
export function isNewer(prev: Job | undefined, next: Job): boolean {
  if (!prev) return true;
  const a = stateRank(prev.state);
  const b = stateRank(next.state);
  if (b !== a) return b > a;
  if (a === 3) return (next.finished_at ?? '') >= (prev.finished_at ?? '');
  if (a === 1 && prev.progress != null && next.progress != null && next.step === prev.step) return next.progress >= prev.progress;
  return true;
}

type FinishedListener = (job: Job, origin: string | null) => void;

/**
 * Jobs this session knows about: every active job, and finished ones that were active while the page
 * was open, so a job card stays where it was started. Progress updates arrive here, not in the
 * query cache: they are frequent.
 */
export const useJobsStore = defineStore('jobs', () => {
  const byId = reactive<Record<string, Job>>({});
  /** The screen each job was started from, for "a job started on another screen finished" notices. */
  const origin = reactive<Record<string, string>>({});
  const listeners = new Set<FinishedListener>();

  function upsert(job: Job, from?: string): boolean {
    if (from && !origin[job.id]) origin[job.id] = from;
    const prev = byId[job.id];
    if (!isNewer(prev, job)) return false;
    byId[job.id] = job;
    if (prev && !isFinished(prev.state) && isFinished(job.state)) for (const l of listeners) l(job, origin[job.id] ?? null);
    return true;
  }

  /** Merge a REST snapshot of the active jobs. Jobs no longer active keep their last known state. */
  function loadActive(jobs: Job[]) {
    for (const j of jobs) upsert(j);
  }

  function remove(ids: string[]) {
    for (const id of ids) {
      delete byId[id];
      delete origin[id];
    }
  }

  const all = computed(() => Object.values(byId).sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0)));
  const active = computed(() => all.value.filter((j) => !isFinished(j.state)));
  const fromScreen = (screen: string) => all.value.filter((j) => origin[j.id] === screen);
  const onFinished = (listener: FinishedListener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };

  return { byId, origin, all, active, upsert, loadActive, remove, fromScreen, onFinished };
});
