<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useJobList } from '@/api/queries/jobs';
import JobCard from '@/components/JobCard.vue';
import { useI18n } from '@/i18n';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, EmptyState, SideSheet, TRANSITIONS, icons, staggerStyle } from '@/ui';

/** Active and recent jobs, from every screen, in a side sheet. */
const open = defineModel<boolean>('open', { default: false });
const { t } = useI18n();
const jobs = useJobsStore();
const router = useRouter();
// This session's jobs (with live progress) first, then the recent history REST knows.
const recent = useJobList({});
const shown = computed(() => {
  const byId = new Map(jobs.all.map((j) => [j.id, j]));
  for (const j of recent.data.value?.items ?? []) if (!byId.has(j.id)) byId.set(j.id, j);
  return [...byId.values()].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).slice(0, 30);
});

function allJobs() {
  open.value = false;
  router.push('/jobs');
}
</script>

<template>
  <SideSheet v-model:open="open" :title="t('jobs.title')" :close-label="t('common.close')">
    <template #header-actions>
      <AppButton variant="text" @click="allJobs">{{ t('jobs.allJobs') }}</AppButton>
    </template>
    <EmptyState v-if="!shown.length" :icon="icons.ListChecks" :title="t('jobs.none')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="list">
      <JobCard v-for="(job, i) in shown" :key="job.id" :job="job" compact :style="staggerStyle(i)" />
    </TransitionGroup>
  </SideSheet>
</template>

<style scoped>
.list { position: relative; display: flex; flex-direction: column; gap: var(--app-space-2); }
</style>
