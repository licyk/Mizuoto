<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useOutputs } from '@/api/queries/outputs';
import { useDataset, useExperiment, useExperimentActions, useExperimentMetrics, useTrainingMethods } from '@/api/queries/train';
import type { StageState } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import HyperparamsForm from '@/components/HyperparamsForm.vue';
import JobCard from '@/components/JobCard.vue';
import ResultsList from '@/components/ResultsList.vue';
import { useI18n } from '@/i18n';
import { useHandoffStore } from '@/stores/handoff';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { AppButton, Badge, ConfirmDialog, DataList, ExpansionPanel, FilterChip, IconButton, LineChart, PageFooter, SegmentedButton, Stepper, Surface, icons, morphStyle, useSnackbar } from '@/ui';

/**
 * One experiment: its stages (what each did, whether its inputs changed since), running them, the
 * losses as they come, the previews it spoke, its settings (a change makes the stages that read
 * them stale) and what it exported.
 */
const props = defineProps<{ engine: string; name: string }>();
const key = computed(() => `${props.engine}/${props.name}`);
const { t, tOr } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const jobs = useJobsStore();
const handoff = useHandoffStore();
const expQuery = useExperiment(key);
const exp = computed(() => expQuery.data.value ?? null);
const dataset = useDataset(() => exp.value?.dataset ?? null);
const methods = useTrainingMethods(() => exp.value?.base_model ?? null);
const method = computed(() => (methods.data.value ?? []).find((m) => m.id === exp.value?.method) ?? null);
const metrics = useExperimentMetrics(key);
const actions = useExperimentActions();
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

// -- stages -----------------------------------------------------------------------------------
const selected = ref('');
watch(key, () => (selected.value = ''));
const steps = computed(() =>
  (exp.value?.stages ?? []).map((s) => ({ id: s.id, label: s.title, state: s.status, note: s.status === 'pending' ? '' : tOr(`train.status.${s.status}`, s.status) })),
);
watch(
  () => exp.value?.stages,
  (stages) => {
    if (!stages?.length || stages.some((s) => s.id === selected.value)) return;
    selected.value = (stages.find((s) => s.status === 'running' || s.status === 'failed' || s.status === 'stale' || s.status === 'pending') ?? stages[stages.length - 1]).id;
  },
  { immediate: true },
);
const stage = computed<StageState | null>(() => exp.value?.stages.find((s) => s.id === selected.value) ?? null);
const stageHelp = computed(() => method.value?.method.stages.find((s) => s.id === selected.value)?.help ?? '');
const stageResult = computed(() =>
  Object.entries(stage.value?.result ?? {})
    .filter(([, v]) => typeof v !== 'object' || v === null)
    .map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: String(v) })),
);
const job = computed(() => jobs.active.find((j) => j.kind === 'train' && (j.request as Record<string, unknown>).experiment === key.value) ?? null);
// While a run is going, the stage it is on is the one shown.
watch(
  () => exp.value?.stages.find((s) => s.status === 'running')?.id,
  (id) => {
    if (id) selected.value = id;
  },
);
const finished = computed(() => !!exp.value && exp.value.stages.every((s) => s.status === 'done'));
function run(stages: string[] | null, restart = false) {
  error.value = null;
  actions.run.mutate({ key: key.value, body: { stages, restart } }, { onError });
}
const confirmRestart = ref(false);

// -- losses -----------------------------------------------------------------------------------
const stagesWithMetrics = computed(() => {
  const seen = new Set((metrics.data.value ?? []).map((p) => p.stage));
  return (exp.value?.stages ?? []).filter((s) => seen.has(s.id));
});
const chartStage = ref('');
watch(
  [stagesWithMetrics, selected],
  ([list, sel]) => {
    if (list.some((s) => s.id === sel)) chartStage.value = sel;
    else if (!list.some((s) => s.id === chartStage.value)) chartStage.value = list[list.length - 1]?.id ?? '';
  },
  { immediate: true },
);
// A stage's metrics have different scales (a loss in the hundreds, an accuracy below one): the losses are
// shown at first, and chips pick the others.
const metricNames = computed(() => [...new Set((metrics.data.value ?? []).filter((p) => p.stage === chartStage.value).flatMap((p) => Object.keys(p.values)))]);
const hidden = ref<Set<string>>(new Set());
watch(
  [metricNames, chartStage],
  ([names], before) => {
    if (before && before[1] === chartStage.value && JSON.stringify(before[0]) === JSON.stringify(names)) return;
    const losses = names.filter((n) => n.includes('loss'));
    hidden.value = new Set(losses.length ? names.filter((n) => !losses.includes(n)) : []);
  },
  { immediate: true },
);
function toggleMetric(name: string, on: boolean) {
  const next = new Set(hidden.value);
  if (on) next.delete(name);
  else next.add(name);
  hidden.value = next;
}
const series = computed(() => {
  const by = new Map<string, [number, number][]>();
  for (const p of metrics.data.value ?? []) {
    if (p.stage !== chartStage.value) continue;
    for (const [name, value] of Object.entries(p.values)) {
      if (hidden.value.has(name)) continue;
      const line = by.get(name) ?? [];
      line.push([p.step, value]);
      by.set(name, line);
    }
  }
  return [...by.entries()].map(([label, points]) => ({ label, points }));
});

// -- previews, exports --------------------------------------------------------------------------
const samples = useOutputs(() => ({ kind: 'sample', engine: props.engine }), () => true, 100);
const ownSamples = computed(() => (samples.data.value?.items ?? []).filter((o) => (o.meta as Record<string, unknown>).experiment === key.value));
watch(
  () => job.value?.state,
  (state, before) => {
    if (before && (!state || isFinished(state))) samples.refetch();
  },
);
function useVoice(id: string) {
  handoff.sendVoice(id);
  router.push('/speak');
}

// -- settings ---------------------------------------------------------------------------------
const hyperparams = ref<Record<string, unknown>>({});
watch(
  exp,
  (e) => {
    if (e) hyperparams.value = { ...e.hyperparams };
  },
  { immediate: true },
);
const changed = computed(() => JSON.stringify(hyperparams.value) !== JSON.stringify(exp.value?.hyperparams ?? {}));
const advancedOpen = ref(false);
const hasAdvanced = computed(() => Object.values(method.value?.method.hyperparams ?? {}).some((s) => s.group === 'advanced'));
function saveSettings() {
  actions.update.mutate({ key: key.value, patch: { hyperparams: hyperparams.value } }, { onSuccess: () => snackbar.show(t('train.settingsSaved')), onError });
}

// -- delete -----------------------------------------------------------------------------------
const confirmDelete = ref(false);
function remove() {
  actions.remove.mutate(key.value, {
    onSuccess: () => {
      snackbar.show(t('train.deleted', { name: props.name }));
      router.push('/train');
    },
    onError,
  });
  confirmDelete.value = false;
}
</script>

<template>
  <div class="experiment" :style="morphStyle(`experiment:${engine}/${name}`)">
    <div v-if="exp" class="sections">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/train')" />
        <h1 class="type-headline-small title">{{ exp.name }}</h1>
        <Badge :value="method?.method.title ?? exp.method" tone="neutral" />
      </header>
      <DataList
        :rows="[
          { label: t('train.baseModel'), value: exp.base_model },
          { label: t('train.dataset'), value: dataset.data.value ? `${dataset.data.value.name} · ${t('train.clips', { n: dataset.data.value.stats.rows })}` : exp.dataset },
          { label: t('train.exportsTo'), value: method ? t(`train.exports.${method.method.export}`) : '' },
          { label: t('common.language'), value: exp.language ? tOr(`languages.${exp.language}`, exp.language) : '' },
        ]"
      />

      <Surface :level="0" shape="large" class="section">
        <div class="section-head">
          <h2 class="type-title-medium">{{ t('train.stages') }}</h2>
          <AppButton variant="text" :icon="icons.RotateCcw" :disabled="!!job" @click="confirmRestart = true">{{ t('train.restart') }}</AppButton>
          <AppButton :icon="icons.Play" :disabled="!!job || finished" :loading="actions.run.isPending.value" @click="run(null)">{{ finished ? t('train.finished') : t('train.run') }}</AppButton>
        </div>
        <Stepper v-model="selected" :steps="steps" />
        <JobCard v-if="job" :job="job" />
        <div v-if="stage" class="stage">
          <div class="stage-head">
            <h3 class="type-title-small">{{ stage.title }}</h3>
            <Badge :value="tOr(`train.status.${stage.status}`, stage.status)" :tone="stage.status === 'failed' ? 'error' : stage.status === 'stale' ? 'warning' : stage.status === 'done' ? 'primary' : 'neutral'" />
            <AppButton variant="tonal" :icon="icons.Play" :disabled="!!job" @click="run([stage.id])">{{ t('train.runStage') }}</AppButton>
          </div>
          <p v-if="stageHelp" class="type-body-medium muted">{{ stageHelp }}</p>
          <p v-if="stage.status === 'stale'" class="type-body-medium warn">{{ t('train.staleHint') }}</p>
          <ErrorNotice v-if="stage.status === 'failed' && stage.error" :error="{ message: stage.error }" />
          <DataList :rows="[{ label: t('train.finishedAt'), value: stage.finished_at ? new Date(stage.finished_at).toLocaleString() : '' }, ...stageResult]" />
        </div>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <div class="section-head">
          <h2 class="type-title-medium">{{ t('train.losses') }}</h2>
          <SegmentedButton v-if="stagesWithMetrics.length > 1" v-model="chartStage" :options="stagesWithMetrics.map((s) => ({ value: s.id, label: s.title }))" />
        </div>
        <div v-if="metricNames.length > 1" class="chips">
          <FilterChip v-for="n in metricNames" :key="n" :label="n.replace(/_/g, ' ')" :model-value="!hidden.has(n)" @update:model-value="toggleMetric(n, $event)" />
        </div>
        <LineChart :series="series" :x-label="t('train.step')" :empty-text="t('train.noMetrics')" />
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('train.settings') }}</h2>
        <p class="type-body-medium muted">{{ t('train.settingsHint') }}</p>
        <HyperparamsForm v-if="method" v-model="hyperparams" :specs="method.method.hyperparams" />
        <ExpansionPanel v-if="method && hasAdvanced" v-model:open="advancedOpen" :label="t('train.advanced')">
          <HyperparamsForm v-model="hyperparams" :specs="method.method.hyperparams" group="advanced" />
        </ExpansionPanel>
        <div class="row-end">
          <AppButton variant="tonal" :icon="icons.Save" :disabled="!changed || !!job" :loading="actions.update.isPending.value" @click="saveSettings">{{ t('common.save') }}</AppButton>
        </div>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <ResultsList v-if="ownSamples.length" :outputs="ownSamples" :title="t('train.previews')" />
        <template v-else>
          <h2 class="type-title-medium">{{ t('train.previews') }}</h2>
          <p class="type-body-medium muted">{{ t('train.noPreviews') }}</p>
        </template>
      </Surface>

      <Surface v-if="exp.artifacts.length" :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('train.exported') }}</h2>
        <ul class="artifacts">
          <li v-for="a in exp.artifacts" :key="`${a.kind}:${a.id}:${a.at}`" class="artifact">
            <Badge :value="t(`train.artifact.${a.kind}`)" tone="primary" />
            <span class="type-body-large grow">{{ a.title || a.id }}</span>
            <span class="type-body-small muted">{{ new Date(a.at).toLocaleString() }}</span>
            <template v-if="a.kind === 'voice'">
              <AppButton variant="text" @click="router.push(`/voices/${a.id}`)">{{ t('common.open') }}</AppButton>
              <AppButton variant="tonal" :icon="icons.Speech" @click="useVoice(a.id)">{{ t('voices.useInSpeak') }}</AppButton>
            </template>
            <AppButton v-else variant="text" @click="router.push('/models')">{{ t('train.seeModels') }}</AppButton>
          </li>
        </ul>
      </Surface>
    </div>
    <ErrorNotice v-else-if="expQuery.error.value" :error="expQuery.error.value" class="pad" />
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <div class="actions">
        <AppButton variant="text" :icon="icons.Trash2" :disabled="!exp || !!job" @click="confirmDelete = true">{{ t('common.delete') }}</AppButton>
      </div>
    </PageFooter>
    <ConfirmDialog
      v-model:open="confirmRestart"
      :title="t('train.restart')"
      :message="t('train.restartConfirm')"
      :confirm-label="t('train.restart')"
      :cancel-label="t('common.cancel')"
      @confirm="(confirmRestart = false), run(null, true)"
    />
    <ConfirmDialog
      v-model:open="confirmDelete"
      :title="t('common.delete')"
      :message="t('train.deleteExperimentConfirm', { name })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="remove"
    />
  </div>
</template>

<style scoped>
.experiment { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.section h2, .section h3, .section p { margin: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.section-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.section-head h2 { flex: 1; }
.stage { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); }
.stage-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.stage-head h3 { flex: 1; }
.warn { color: var(--md-sys-color-tertiary); }
.chips { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.artifacts { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.artifact { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); min-width: 0; }
.grow { flex: 1; min-width: 0; }
.row-end { display: flex; justify-content: flex-end; }
.actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--app-space-2); }
.pad { margin: var(--app-space-4) var(--app-space-6); }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
