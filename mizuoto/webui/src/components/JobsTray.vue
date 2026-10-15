<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useJobActions } from '@/api/queries/jobs';
import type { Job } from '@/api/types';
import JobCard from '@/components/JobCard.vue';
import { useI18n } from '@/i18n';
import { AppIcon, IconButton, ProgressBar, TRANSITIONS, collapseHooks, icons, keepLeavingInPlace } from '@/ui';

/**
 * The jobs a page started, in its footer. One job shows as its card. Several fold into one line (how
 * many run and wait, their progress together, cancel all) that opens into their cards, which scroll
 * inside the footer instead of pushing the page out of view. Folded by default; the choice stays
 * while the page is open.
 */
const props = defineProps<{ jobs: Job[] }>();
const { t } = useI18n();
const actions = useJobActions();
const open = ref(false);
const running = computed(() => props.jobs.filter((j) => j.state === 'running' || j.state === 'cancelling'));
const waiting = computed(() => props.jobs.length - running.value.length);
// Together: each job counts as one; a queued one adds nothing yet, a running one its own share.
const progress = computed(() => (props.jobs.length ? props.jobs.reduce((n, j) => n + (j.state === 'running' ? (j.progress ?? 0) : 0), 0) / props.jobs.length : 0));
const summary = computed(() => t('jobs.trayRunning', { running: running.value.length, waiting: waiting.value }));
function cancelAll() {
  for (const j of props.jobs) if (j.state !== 'cancelling') actions.cancel.mutate(j.id);
}
// Back to one job: nothing to fold. The next batch starts folded.
watch(
  () => props.jobs.length,
  (n) => n < 2 && (open.value = false),
);
</script>

<template>
  <div v-if="jobs.length" class="tray">
    <JobCard v-if="jobs.length === 1" :key="jobs[0].id" :job="jobs[0]" />
    <template v-else>
      <div class="summary">
        <AppIcon :icon="icons.Loader2" :size="20" spin class="lead" />
        <div class="titles">
          <span class="type-title-small">{{ t('jobs.trayTitle', { n: jobs.length }) }}</span>
          <span class="type-body-small muted">{{ summary }}</span>
        </div>
        <IconButton :icon="icons.X" :label="t('jobs.cancelAll')" @click="cancelAll" />
        <IconButton :icon="open ? icons.ChevronDown : icons.ChevronUp" :label="open ? t('jobs.fold') : t('jobs.unfold')" :aria-expanded="open" @click="open = !open" />
      </div>
      <ProgressBar :value="running.length ? progress : null" :label="summary" />
      <Transition :name="TRANSITIONS.collapse" v-bind="collapseHooks">
        <div v-if="open" class="list-box">
          <TransitionGroup :name="TRANSITIONS.list" tag="div" class="list" v-bind="keepLeavingInPlace">
            <JobCard v-for="j in jobs" :key="j.id" :job="j" compact />
          </TransitionGroup>
        </div>
      </Transition>
    </template>
  </div>
</template>

<style scoped>
.tray { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.summary { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; padding-inline-start: var(--app-space-2); }
.lead { flex: none; color: var(--md-sys-color-primary); }
.titles { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.titles > span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* Several cards scroll here, within the footer; the page above stays in view. */
.list-box { min-height: 0; }
.list { position: relative; display: flex; flex-direction: column; gap: var(--app-space-2); max-height: min(40vh, 360px); overflow-y: auto; overscroll-behavior: contain; padding-inline-end: var(--app-space-1); }
</style>
