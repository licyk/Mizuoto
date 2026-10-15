<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useDatasets, useExperiments } from '@/api/queries/train';
import type { Experiment } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import NewDatasetDialog from '@/components/NewDatasetDialog.vue';
import NewExperimentDialog from '@/components/NewExperimentDialog.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { AppButton, AppCard, AxisPanes, Badge, EmptyState, Tabs, TRANSITIONS, icons, morphStyle, staggerStyle, useSnackbar } from '@/ui';

/** Training: the experiments (a base model fine-tuned on a dataset, stage by stage) and the datasets they learn from. */
const { t, tOr } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const experiments = useExperiments();
const datasets = useDatasets();
const TABS = ['experiments', 'datasets'] as const;
type Tab = (typeof TABS)[number];
const tab = ref<Tab>('experiments');
const tabs = computed(() => TABS.map((v) => ({ value: v, label: t(`train.tabs.${v}`) })));
const newExperiment = ref(false);
const newDataset = ref(false);
const datasetName = (id: string) => (datasets.data.value ?? []).find((d) => d.id === id)?.name ?? id;

/** One line on an experiment's stages: running, failed, stale, done or not started. */
function summary(e: Experiment): { text: string; tone: 'primary' | 'warning' | 'error' | 'neutral' } {
  const done = e.stages.filter((s) => s.status === 'done').length;
  const running = e.stages.find((s) => s.status === 'running');
  const failed = e.stages.find((s) => s.status === 'failed');
  const stale = e.stages.find((s) => s.status === 'stale');
  if (running) return { text: t('train.stageRunning', { stage: running.title }), tone: 'primary' };
  if (failed) return { text: t('train.stageFailed', { stage: failed.title }), tone: 'error' };
  if (stale) return { text: t('train.stageStale', { stage: stale.title }), tone: 'warning' };
  if (done === e.stages.length) return { text: t('train.finished'), tone: 'primary' };
  return { text: t('train.stagesDone', { done, n: e.stages.length }), tone: 'neutral' };
}
</script>

<template>
  <div class="train">
    <header class="bar">
      <Tabs v-model="tab" :tabs="tabs" class="tabs" />
      <AppButton v-if="tab === 'experiments'" :icon="icons.Plus" @click="newExperiment = true">{{ t('train.newExperiment') }}</AppButton>
      <AppButton v-else :icon="icons.Plus" @click="newDataset = true">{{ t('train.newDataset') }}</AppButton>
    </header>
    <ErrorNotice v-if="experiments.error.value || datasets.error.value" :error="experiments.error.value ?? datasets.error.value" />

    <AxisPanes :value="tab" :order="TABS">
      <div v-if="tab === 'experiments'" key="experiments" class="pane">
        <EmptyState v-if="experiments.isSuccess.value && !experiments.data.value?.length" :icon="icons.GraduationCap" :title="t('train.noExperiments')" :text="t('train.noExperimentsHint')" />
        <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
          <AppCard
            v-for="(e, i) in experiments.data.value ?? []"
            :key="`${e.engine}/${e.name}`"
            variant="filled"
            interactive
            class="card"
            :style="[staggerStyle(i), morphStyle(`experiment:${e.engine}/${e.name}`)]"
            @activate="router.push(`/train/experiments/${e.engine}/${e.name}`)"
          >
            <div class="head">
              <span class="type-title-medium name">{{ e.name }}</span>
              <Badge :value="summary(e).text" :tone="summary(e).tone" />
            </div>
            <div class="facts type-body-small muted">
              <span>{{ e.base_model }}</span>
              <span>{{ tOr(`train.methods.${e.method}`, e.method) }}</span>
              <span>{{ datasetName(e.dataset) }}</span>
            </div>
            <div class="stages" aria-hidden="true">
              <span v-for="s in e.stages" :key="s.id" class="pip" :class="s.status" :title="s.title" />
            </div>
          </AppCard>
        </TransitionGroup>
      </div>

      <div v-else key="datasets" class="pane">
        <EmptyState v-if="datasets.isSuccess.value && !datasets.data.value?.length" :icon="icons.Layers" :title="t('train.noDatasets')" :text="t('train.noDatasetsHint')" />
        <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
          <AppCard v-for="(d, i) in datasets.data.value ?? []" :key="d.id" variant="filled" interactive class="card" :style="[staggerStyle(i), morphStyle(`dataset:${d.id}`)]" @activate="router.push(`/train/datasets/${d.id}`)">
            <div class="head">
              <span class="type-title-medium name">{{ d.name }}</span>
              <Badge v-if="d.language" :value="tOr(`languages.${d.language}`, d.language)" tone="neutral" />
            </div>
            <p v-if="d.description" class="type-body-small muted desc">{{ d.description }}</p>
            <div class="facts type-body-small muted">
              <span>{{ t('train.clips', { n: d.stats.rows }) }} · {{ formatDuration(d.stats.seconds) }}</span>
              <span v-if="d.stats.missing_text" class="warn">{{ t('train.missingText', { n: d.stats.missing_text }) }}</span>
              <span v-if="d.stats.speakers.length">{{ d.stats.speakers.join(', ') }}</span>
            </div>
          </AppCard>
        </TransitionGroup>
      </div>
    </AxisPanes>

    <NewExperimentDialog v-model:open="newExperiment" @created="(e) => (snackbar.show(t('train.created', { name: e.name })), router.push(`/train/experiments/${e.engine}/${e.name}`))" />
    <NewDatasetDialog v-model:open="newDataset" @created="(d) => (snackbar.show(t('train.created', { name: d.name })), router.push(`/train/datasets/${d.id}`))" />
  </div>
</template>

<style scoped>
.train { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.tabs { flex: 1; min-width: 240px; }
.pane { display: flex; flex-direction: column; min-width: 0; }
.grid { position: relative; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr)); gap: var(--app-space-3); }
.card { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.desc { margin: 0; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.facts { display: flex; flex-wrap: wrap; gap: var(--app-space-1) var(--app-space-3); }
.warn { color: var(--md-sys-color-tertiary); }
.stages { display: flex; gap: var(--app-space-1); margin-top: auto; }
.pip { flex: 1; height: 4px; border-radius: 2px; background: var(--md-sys-color-surface-container-highest); }
.pip.done { background: var(--md-sys-color-primary); }
.pip.running { background: var(--md-sys-color-primary-container); }
.pip.stale { background: var(--md-sys-color-tertiary); }
.pip.failed { background: var(--md-sys-color-error); }
@container app-content (max-width: 599px) {
  .train { padding: var(--app-space-3); }
}
</style>
