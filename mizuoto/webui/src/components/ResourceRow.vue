<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from '@/i18n';
import { formatBytes, formatEta } from '@/format';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, Badge, IconButton, ProgressBar, icons } from '@/ui';

/**
 * One downloadable or installed model, the same everywhere it is listed (Assets, Base models,
 * Separation models, demo voices): its name and details, its state, a download's progress with
 * speed and time left, and Download, Verify, Open and Delete as they apply.
 */
export type ResourceState = 'installed' | 'missing' | 'partial' | 'corrupt' | 'downloading' | 'builtin';

const props = defineProps<{
  title: string;
  details?: string;
  state: ResourceState;
  /** The download job while one runs. */
  jobId?: string | null;
  verifiable?: boolean;
  deletable?: boolean;
  /** A label for an extra action on an installed resource ("In library"). */
  openLabel?: string;
  /** The remote host it lives on: downloads and deletions happen there, not here. */
  remote?: string | null;
}>();
const emit = defineEmits<{ download: []; verify: []; delete: []; open: [] }>();
const { t, tOr } = useI18n();
const jobs = useJobsStore();

const job = computed(() => (props.jobId ? (jobs.byId[props.jobId] ?? null) : null));
const downloading = computed(() => props.state === 'downloading' || (!!job.value && ['queued', 'waiting', 'running'].includes(job.value.state)));
const installed = computed(() => props.state === 'installed' || props.state === 'corrupt' || props.state === 'builtin');
const tone = computed(() => (({ installed: 'primary', corrupt: 'error', partial: 'warning', downloading: 'primary' }) as Record<string, 'primary' | 'error' | 'warning'>)[props.state] ?? 'neutral');
const transfer = computed(() => {
  const x = job.value?.transfer;
  if (!x) return '';
  const parts = [t('jobs.transfer', { done: formatBytes(x.done_bytes), total: formatBytes(x.total_bytes) })];
  if (x.bytes_per_second) parts.push(t('jobs.speed', { speed: formatBytes(x.bytes_per_second) }));
  if (x.bytes_per_second) parts.push(t('jobs.eta', { time: formatEta(x.total_bytes - x.done_bytes, x.bytes_per_second) || '0s' }));
  return parts.join(' · ');
});
</script>

<template>
  <div class="resource" :class="{ busy: downloading }">
    <div class="text">
      <span class="type-body-large name">{{ title }}</span>
      <span v-if="details" class="type-body-small muted details">{{ details }}</span>
      <template v-if="downloading">
        <ProgressBar class="progress" :value="job?.state === 'running' ? (job?.progress ?? null) : null" :label="title" />
        <span class="type-body-small muted numbers">{{ transfer || tOr(`jobs.states.${job?.state ?? 'running'}`, t('resource.states.downloading')) }}</span>
      </template>
    </div>
    <Badge v-if="remote" :value="t('remote.onHost', { host: remote })" tone="neutral" />
    <Badge v-if="!downloading" :value="tOr(`resource.states.${state}`, state)" :tone="tone" />
    <div class="actions">
      <slot />
      <AppButton v-if="openLabel && installed" variant="text" @click="emit('open')">{{ openLabel }}</AppButton>
      <AppButton v-if="!installed && !downloading && !remote" variant="tonal" :icon="icons.Download" @click="emit('download')">{{ t('common.download') }}</AppButton>
      <AppButton v-if="verifiable && installed && !remote" variant="text" @click="emit('verify')">{{ t('resource.verify') }}</AppButton>
      <IconButton v-if="deletable && (installed || state === 'partial') && !downloading && !remote" :icon="icons.Trash2" :label="t('common.delete')" @click="emit('delete')" />
    </div>
  </div>
</template>

<style scoped>
.resource {
  display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3);
  min-height: 64px; padding: var(--app-space-2) var(--app-space-2) var(--app-space-2) var(--app-space-4);
  border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0;
}
.text { display: flex; flex-direction: column; gap: 2px; flex: 1 1 260px; min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.details { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.progress { margin-top: var(--app-space-1); }
.numbers { font-variant-numeric: tabular-nums; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); margin-left: auto; }
.muted { color: var(--md-sys-color-on-surface-variant); }
</style>
