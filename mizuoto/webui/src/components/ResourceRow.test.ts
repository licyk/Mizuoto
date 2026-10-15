import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { describe, expect, it } from 'vitest';
import type { Job } from '@/api/types';
import ResourceRow from '@/components/ResourceRow.vue';
import { useJobsStore } from '@/stores/jobs';
import AppButton from '@/ui/AppButton.vue';
import IconButton from '@/ui/IconButton.vue';

const labels = (w: ReturnType<typeof mount>) => [...w.findAllComponents(AppButton).map((b) => b.text()), ...w.findAllComponents(IconButton).map((b) => b.props('label'))];

describe('ResourceRow', () => {
  it('offers Download when missing, and Verify and Delete once installed', () => {
    const pinia = createPinia();
    const missing = mount(ResourceRow, { props: { title: 'RMVPE', state: 'missing', deletable: true, verifiable: true }, global: { plugins: [pinia] } });
    expect(labels(missing)).toEqual(['Download']);
    const installed = mount(ResourceRow, { props: { title: 'RMVPE', state: 'installed', deletable: true, verifiable: true }, global: { plugins: [pinia] } });
    expect(labels(installed)).toEqual(['Verify', 'Delete']);
    installed.findAllComponents(IconButton)[0].vm.$emit('click', new MouseEvent('click'));
    expect(installed.emitted('delete')).toHaveLength(1);
  });

  it("shows a download's bytes, speed and time left", () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const job = {
      id: 'j1', kind: 'model_download', title: 'Download RMVPE', state: 'running', waiting_for: null, progress: 0.5, step: null, message: null, origin: null, steps: [], error: null, result: null, request: {},
      created_at: '', started_at: null, finished_at: null, can_retry: false, can_run_anyway: false,
      transfer: { done_bytes: 100 * 2 ** 20, total_bytes: 200 * 2 ** 20, bytes_per_second: 10 * 2 ** 20, eta_seconds: 10 },
    } as Job;
    useJobsStore().upsert(job);
    const w = mount(ResourceRow, { props: { title: 'RMVPE', state: 'downloading', jobId: 'j1', deletable: true }, global: { plugins: [pinia] } });
    expect(w.text()).toContain('100 MB of 200 MB · 10.0 MB/s · 10s left');
    expect(labels(w)).toEqual([]);
  });
});
