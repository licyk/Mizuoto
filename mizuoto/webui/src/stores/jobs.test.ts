import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, describe, expect, it } from 'vitest';
import type { Job } from '@/api/types';
import { isNewer, useJobsStore } from '@/stores/jobs';

const job = (p: Partial<Job>): Job => ({
  id: 'j1', kind: 'speak', title: 'Speak: hello', state: 'queued', waiting_for: null, progress: null, step: null, message: null, origin: null, steps: [], error: null, result: null,
  request: {}, created_at: '2026-10-04T00:00:00.000Z', started_at: null, finished_at: null, can_retry: false, can_run_anyway: false, transfer: null, ...p,
});

describe('jobs store event ordering', () => {
  beforeEach(() => setActivePinia(createPinia()));

  it('never goes back in a job life', () => {
    expect(isNewer(job({ state: 'running', progress: 0.5 }), job({ state: 'queued' }))).toBe(false);
    expect(isNewer(job({ state: 'completed', finished_at: 'b' }), job({ state: 'running', progress: 0.9 }))).toBe(false);
    expect(isNewer(job({ state: 'queued' }), job({ state: 'waiting' }))).toBe(true);
    expect(isNewer(job({ state: 'waiting' }), job({ state: 'queued' }))).toBe(true);
    expect(isNewer(job({ state: 'running', progress: 0.6, step: 'a' }), job({ state: 'running', progress: 0.4, step: 'a' }))).toBe(false);
    expect(isNewer(job({ state: 'running', progress: 0.6, step: 'a' }), job({ state: 'running', progress: 0.1, step: 'b' }))).toBe(true);
  });

  it('keeps the newest of an event racing a REST snapshot, and announces a finish once', () => {
    const store = useJobsStore();
    const finished: string[] = [];
    store.onFinished((j, origin) => finished.push(`${j.id}@${origin}`));
    expect(store.upsert(job({ state: 'queued' }), 'convert')).toBe(true);
    expect(store.upsert(job({ state: 'running', progress: 0.3 }))).toBe(true);
    // A REST list fetched before the job started arrives late.
    store.loadActive([job({ state: 'queued' })]);
    expect(store.byId.j1.state).toBe('running');
    expect(store.upsert(job({ state: 'completed', progress: 1, finished_at: '2026-10-04T00:00:05.000Z' }))).toBe(true);
    expect(store.upsert(job({ state: 'running', progress: 0.9 }))).toBe(false);
    store.upsert(job({ state: 'completed', progress: 1, finished_at: '2026-10-04T00:00:05.000Z' }));
    expect(finished).toEqual(['j1@convert']);
    expect(store.fromScreen('convert').map((j) => j.id)).toEqual(['j1']);
    expect(store.active).toEqual([]);
    store.remove(['j1']);
    expect(store.all).toEqual([]);
  });
});
