<script setup lang="ts">
import { computed, ref } from 'vue';
import { useJobActions } from '@/api/queries/jobs';
import type { Job } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobLogView from '@/components/JobLogView.vue';
import { useI18n } from '@/i18n';
import { formatBytes, formatEta } from '@/format';
import { isFinished } from '@/stores/jobs';
import { AppButton, AppCard, Badge, IconButton, ProgressBar, TRANSITIONS, collapseHooks, icons } from '@/ui';

/** One long operation: title, steps, progress, cancel, retry and its log. */
const props = defineProps<{ job: Job; compact?: boolean }>();
const { t, tOr } = useI18n();
const actions = useJobActions();
const showLog = ref(false);
const finished = computed(() => isFinished(props.job.state));
const step = computed(() => props.job.steps.find((s) => s.id === props.job.step) ?? null);
const tone = computed(() => ({ completed: 'primary', failed: 'error', interrupted: 'warning', cancelled: 'neutral' })[props.job.state as string] ?? 'neutral');
// A waiting job names its resource: gpu:cuda:0, engine:qwen3tts:cuda:0, io, cpu.
const stateText = computed(() =>
  props.job.state === 'waiting' && props.job.waiting_for ? tOr(`jobs.waitingFor.${props.job.waiting_for.split(':')[0]}`, props.job.waiting_for) : tOr(`jobs.states.${props.job.state}`, props.job.state),
);
const doneSteps = computed(() => props.job.steps.filter((s) => s.state === 'done').length);
/** A download's bytes, speed and time left: "88 MB of 210 MB · 8.4 MB/s · 14s left". */
const transferText = computed(() => {
  const x = props.job.transfer;
  if (!x || finished.value) return '';
  const parts = [t('jobs.transfer', { done: formatBytes(x.done_bytes), total: formatBytes(x.total_bytes) })];
  if (x.bytes_per_second) parts.push(t('jobs.speed', { speed: formatBytes(x.bytes_per_second) }));
  if (x.bytes_per_second && x.eta_seconds != null) parts.push(t('jobs.eta', { time: formatEta(x.total_bytes - x.done_bytes, x.bytes_per_second) || '0s' }));
  return parts.join(' · ');
});
</script>

<template>
  <AppCard variant="outlined" class="job" :class="{ compact }">
    <div class="head">
      <div class="titles">
        <span class="type-title-small name">{{ job.title }}</span>
        <span class="type-body-small muted sub">
          {{ stateText }}<template v-if="job.message && !finished"> · {{ job.message }}</template><template v-else-if="step && !finished"> · {{ step.title }}</template><template v-if="job.steps.length > 1"> · {{ doneSteps }}/{{ job.steps.length }}</template>
        </span>
      </div>
      <Badge v-if="finished" :value="tOr(`jobs.states.${job.state}`, job.state)" :tone="tone as 'primary'" />
      <IconButton v-if="!finished" :icon="icons.X" :label="t('jobs.cancel')" :disabled="job.state === 'cancelling'" @click="actions.cancel.mutate(job.id)" />
      <IconButton v-if="job.can_retry" :icon="icons.RotateCcw" :label="t('jobs.retry')" @click="actions.retry.mutate(job.id)" />
      <IconButton :icon="icons.FileText" :label="showLog ? t('jobs.hideLog') : t('jobs.log')" @click="showLog = !showLog" />
    </div>
    <ProgressBar v-if="!finished" :value="job.state === 'running' ? job.progress : null" :label="job.title" />
    <span v-if="transferText" class="type-body-small muted transfer">{{ transferText }}</span>
    <div v-if="job.can_run_anyway" class="anyway">
      <span class="type-body-small muted">{{ t('jobs.runAnywayHint') }}</span>
      <AppButton variant="text" @click="actions.runAnyway.mutate(job.id)">{{ t('jobs.runAnyway') }}</AppButton>
    </div>
    <ErrorNotice v-if="job.error" :error="job.error" />
    <Transition :name="TRANSITIONS.collapse" v-bind="collapseHooks">
      <JobLogView v-if="showLog" :job-id="job.id" />
    </Transition>
  </AppCard>
</template>

<style scoped>
.job { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3) var(--app-space-2) var(--app-space-3) var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-1); min-width: 0; }
.titles { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.name, .sub { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.transfer { font-variant-numeric: tabular-nums; }
.anyway { display: flex; align-items: center; gap: var(--app-space-2); flex-wrap: wrap; }
</style>
