import type { QueryClient, QueryKey } from '@tanstack/vue-query';
import { io, type Socket } from 'socket.io-client';
import { BASE_PATH, BASE_URL } from '@/api/baseUrl';
import { fetchActiveJobs } from '@/api/queries/jobs';
import { keys } from '@/api/queries/keys';
import type { S, ServerEvents } from '@/api/types';
import { useAuthStore } from '@/stores/auth';
import { isFinished, useJobsStore } from '@/stores/jobs';

type Listeners = { [K in keyof ServerEvents]: (payload: ServerEvents[K]) => void };

let socket: Socket<Listeners> | null = null;

/** Invalidate a query key at most once per interval, so bursts of events cause one refetch. */
export function createInvalidator(qc: QueryClient, delay = 300) {
  const pending = new Map<string, QueryKey>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    timer = undefined;
    for (const key of pending.values()) qc.invalidateQueries({ queryKey: key });
    pending.clear();
  };
  return (key: QueryKey) => {
    pending.set(JSON.stringify(key), key);
    if (!timer) timer = setTimeout(flush, delay);
  };
}

/** Listeners for ``job_log`` lines, keyed by job id (the open log viewers). */
const logListeners = new Map<string, Set<(offset: number, lines: string[]) => void>>();
export function onJobLog(jobId: string, listener: (offset: number, lines: string[]) => void): () => void {
  const set = logListeners.get(jobId) ?? new Set();
  set.add(listener);
  logListeners.set(jobId, set);
  return () => set.delete(listener);
}

/** Load what REST knows now: the active jobs. Called on connect and reconnect. */
async function resync() {
  try {
    useJobsStore().loadActive((await fetchActiveJobs()).items);
  } catch {
    /* a token may be needed; the auth dialog handles it */
  }
}

/** The lists a finished job of each kind changes. */
const AFFECTS: Record<string, QueryKey[]> = {
  model_download: [keys.models, keys.assets],
  asset_download: [keys.assets, keys.models],
  model_import: [keys.models],
  voice_prepare: [keys.voices],
  transcribe: [keys.voices],
  dataset_prepare: [keys.datasets],
  train: [keys.experiments, keys.voices, keys.models, keys.outputs],
  project_render: [keys.projects],
  project_export: [keys.outputs],
};

/**
 * Connect once. REST stays the source of truth: events patch or invalidate the queries they affect,
 * and a reconnect refetches what is on screen.
 */
export function connectSocket(qc: QueryClient): Socket<Listeners> {
  if (socket) return socket;
  const jobs = useJobsStore();
  const invalidate = createInvalidator(qc);
  socket = io(new URL(BASE_URL).origin, {
    path: `${BASE_PATH}/ws/socket.io`,
    auth: (cb) => cb({ token: useAuthStore().token }),
    transports: ['websocket', 'polling'],
    reconnectionDelay: 1000,
    reconnectionDelayMax: 10_000,
  });

  socket.on('connect', () => resync());
  socket.on('job_updated', ({ job }) => {
    if (!jobs.upsert(job)) return;
    qc.setQueryData(keys.job(job.id), job);
    // A download shows its progress on its model rows.
    if (job.kind === 'model_download' && job.state === 'running') invalidate(keys.models);
    if (isFinished(job.state)) {
      invalidate(keys.jobs);
      for (const key of AFFECTS[job.kind] ?? []) invalidate(key);
    }
  });
  socket.on('job_log', (e) => {
    for (const l of logListeners.get(e.job_id) ?? []) l(e.offset, e.lines);
  });
  socket.on('jobs_cleared', (e) => {
    jobs.remove(e.ids);
    invalidate(keys.jobs);
  });
  socket.on('outputs_added', () => invalidate(keys.outputs));
  socket.on('outputs_removed', () => invalidate(keys.outputs));
  socket.on('voices_changed', (e) => {
    invalidate(keys.voices);
    for (const id of e.ids) invalidate(keys.voice(id));
  });
  socket.on('models_changed', () => {
    invalidate(keys.models);
    invalidate(keys.assets);
  });
  socket.on('presets_changed', () => invalidate(keys.presets));
  socket.on('datasets_changed', (e) => {
    invalidate(keys.datasets);
    for (const id of e.ids) invalidate(keys.dataset(id));
  });
  socket.on('projects_changed', (e) => {
    invalidate(keys.projects);
    for (const id of e.ids) invalidate(keys.project(id));
  });
  socket.on('experiments_changed', () => invalidate(keys.experiments));
  // Losses as they come: appended to the experiment's metrics (REST has them all on the next load).
  socket.on('train_metrics', (e) => {
    qc.setQueryData(keys.experimentMetrics(e.experiment), (old: S['MetricPoint'][] | undefined) => (old ? [...old, ...e.points] : old));
  });
  socket.on('engines_changed', () => {
    invalidate(keys.engines);
    invalidate(keys.models);
    invalidate(keys.meta);
  });
  socket.on('runtime_changed', ({ runtime }) => qc.setQueryData(keys.runtime, runtime));
  socket.on('compute_changed', ({ usage }) => qc.setQueryData(keys.computeUsage, usage));
  socket.on('settings_changed', () => {
    invalidate(keys.settings);
    invalidate(keys.meta);
  });
  socket.io.on('reconnect', () => {
    resync();
    for (const key of [keys.models, keys.voices, keys.outputs, keys.jobs, keys.engines, keys.runtime, keys.computeUsage, keys.datasets, keys.experiments, keys.metrics, keys.projects]) qc.invalidateQueries({ queryKey: key });
  });
  return socket;
}

export function disconnectSocket() {
  socket?.disconnect();
  socket = null;
}
