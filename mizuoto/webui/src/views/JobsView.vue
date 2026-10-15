<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useJobActions, useJobList } from '@/api/queries/jobs';
import JobCard from '@/components/JobCard.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { AppButton, EmptyState, SegmentedControl, TRANSITIONS, icons, staggerStyle } from '@/ui';

/** Every job, with history and logs. The list is REST; live progress comes from the store. */
type Filter = 'all' | 'active' | 'finished';
const { t } = useI18n();
const store = useJobsStore();
const actions = useJobActions();
const filter = ref<Filter>('all');
const list = useJobList(() => (filter.value === 'all' ? {} : { state: filter.value }));
const options = computed(() => (['all', 'active', 'finished'] as const).map((v) => ({ value: v, label: t(`jobs.filters.${v}`) })));
watch(
  () => list.data.value,
  (page) => page && store.loadActive(page.items.filter((j) => !isFinished(j.state))),
);
const jobs = computed(() => (list.data.value?.items ?? []).map((j) => store.byId[j.id] ?? j).filter((j) => filter.value === 'all' || (filter.value === 'active') === !isFinished(j.state)));
</script>

<template>
  <div class="jobs-view">
    <header class="bar">
      <SegmentedControl v-model="filter" :options="options" />
      <AppButton variant="text" :icon="icons.Trash2" @click="actions.clear.mutate()">{{ t('jobs.clear') }}</AppButton>
    </header>
    <ErrorNotice v-if="list.error.value" :error="list.error.value" />
    <EmptyState v-else-if="list.isSuccess.value && !jobs.length" :icon="icons.ListChecks" :title="t('jobs.none')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="list">
      <JobCard v-for="(j, i) in jobs" :key="j.id" :job="j" :style="staggerStyle(i)" />
    </TransitionGroup>
  </div>
</template>

<style scoped>
.jobs-view { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); min-width: 0; }
.bar { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); flex-wrap: wrap; }
.list { position: relative; display: flex; flex-direction: column; gap: var(--app-space-2); }
</style>
