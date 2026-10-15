<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useMeta } from '@/api/queries/app';
import { datasetUrls, useDataset, useDatasetActions, useDatasetReport, useDatasetRows } from '@/api/queries/train';
import { useVoices } from '@/api/queries/voices';
import type { DatasetRow, Output } from '@/api/types';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import NewExperimentDialog from '@/components/NewExperimentDialog.vue';
import OutputsPickerDialog from '@/components/OutputsPickerDialog.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { TRAIN_LANGUAGES } from '@/components/capabilities';
import { readyIds, type InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { useJobsStore } from '@/stores/jobs';
import { usePlayerStore } from '@/stores/player';
import { AppButton, Badge, ConfirmDialog, FieldRow, FilterChip, IconButton, PageFooter, SearchField, SelectField, Surface, Switch, TextField, icons, morphStyle, useSnackbar } from '@/ui';

/**
 * One dataset: add clips (files, recordings, a voice's references, earlier results, a server folder in a
 * known layout), prepare them (cut at silences, transcribe, normalise), read the report, and correct,
 * exclude or delete clips one by one.
 */
const props = defineProps<{ id: string }>();
const { t } = useI18n();
const router = useRouter();
const meta = useMeta();
const datasetQuery = useDataset(() => props.id);
const report = useDatasetReport(() => props.id);
const actions = useDatasetActions();
const voices = useVoices();
const jobs = useJobsStore();
const player = usePlayerStore();
const snackbar = useSnackbar();
const dataset = computed(() => datasetQuery.data.value ?? null);
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);
const added = (warnings: string[] | undefined, before: number) => {
  const n = (dataset.value?.stats.rows ?? before) - before;
  snackbar.show(t('train.clipsAdded', { n: Math.max(0, n) }));
  for (const w of warnings ?? []) snackbar.show(w);
};
const languageOptions = computed(() => [{ value: '', label: t('languages.any') }, ...TRAIN_LANGUAGES.map((c) => ({ value: c, label: t(`languages.${c}`) }))]);

// -- adding clips -----------------------------------------------------------------------------
const clips = ref<InputItem[]>([]);
const speaker = ref('');
function addClips() {
  const ids = readyIds(clips.value);
  if (!ids.length) return;
  const before = dataset.value?.stats.rows ?? 0;
  const texts = clips.value.filter((c) => c.status === 'ready' && c.audioId).map((c) => c.transcript ?? null);
  actions.addAudio.mutate(
    { id: props.id, body: { audio_ids: ids, texts, speaker: speaker.value.trim() } },
    {
      onSuccess: (r) => {
        clips.value = [];
        added(r.warnings, before);
      },
      onError,
    },
  );
}
const voiceId = ref<string | null>(null);
const voiceOptions = computed(() => (voices.data.value ?? []).filter((v) => v.references).map((v) => ({ value: v.id, label: v.name })));
function addVoice() {
  if (!voiceId.value) return;
  const before = dataset.value?.stats.rows ?? 0;
  actions.addVoice.mutate({ id: props.id, voice: voiceId.value }, { onSuccess: () => added([], before), onError });
}
// Earlier results keep their licence tags (a gated engine's output cannot train another engine).
const outputsOpen = ref(false);
function addOutputs(outputs: Output[]) {
  const before = dataset.value?.stats.rows ?? 0;
  actions.addOutputs.mutate({ id: props.id, ids: outputs.map((o) => o.id) }, { onSuccess: () => added([], before), onError });
}
const folderOpen = ref(false);
const layout = ref<'auto' | 'plain' | 'gptsovits_list' | 'ljspeech' | 'csv'>('auto');
const layoutOptions = computed(() => (['auto', 'gptsovits_list', 'ljspeech', 'csv', 'plain'] as const).map((v) => ({ value: v, label: t(`train.layouts.${v}`) })));
function importFolder(picked: { path: string }[]) {
  const path = picked[0]?.path;
  if (!path) return;
  const before = dataset.value?.stats.rows ?? 0;
  actions.importFolder.mutate({ id: props.id, body: { path, layout: layout.value } }, { onSuccess: (r) => added(r.warnings, before), onError });
}

// -- preparing --------------------------------------------------------------------------------
const prep = ref({ slice: true, max_seconds: 12, min_seconds: 1, threshold_db: -34, transcribe: true, normalize: true, language: '' });
const prepareJob = computed(() => jobs.active.find((j) => j.kind === 'dataset_prepare' && (j.request as Record<string, unknown>).dataset === props.id) ?? null);
function prepare() {
  const p = prep.value;
  actions.prepare.mutate(
    { id: props.id, body: { slice: p.slice, max_seconds: Number(p.max_seconds), min_seconds: Number(p.min_seconds), threshold_db: Number(p.threshold_db), transcribe: p.transcribe, normalize: p.normalize, language: p.language || null } },
    { onError },
  );
}
// The report and rows follow the dataset: refetch when it changes (a prepare, an edit, clips added).
watch(
  () => dataset.value?.fingerprint,
  () => report.refetch(),
);

// -- rows -------------------------------------------------------------------------------------
type Issue = 'too_short' | 'too_long' | 'empty_text' | 'clipping';
const PAGE = 50;
const q = ref('');
const issue = ref<Issue | null>(null);
const offset = ref(0);
watch([q, issue], () => (offset.value = 0));
const rows = useDatasetRows(() => props.id, () => ({ offset: offset.value, limit: PAGE, q: q.value.trim() || undefined, issue: issue.value ?? undefined }));
watch(
  () => dataset.value?.fingerprint,
  () => rows.refetch(),
);
const page = computed(() => rows.data.value ?? { items: [], total: 0 });
const edits = ref<Record<string, string>>({});
const rowKey = (r: DatasetRow) => `dataset:${props.id}:${r.id}`;
const play = (r: DatasetRow) => player.play({ key: rowKey(r), url: datasetUrls.rowAudio(props.id, r.id), label: r.text || r.id });
function saveText(r: DatasetRow) {
  const text = edits.value[r.id];
  if (text === undefined || text === r.text) return;
  actions.updateRow.mutate({ id: props.id, row: r.id, patch: { text } }, { onSuccess: () => delete edits.value[r.id], onError });
}
const setExcluded = (r: DatasetRow, excluded: boolean) => actions.updateRow.mutate({ id: props.id, row: r.id, patch: { excluded } }, { onError });
const removeRow = (r: DatasetRow) => actions.deleteRows.mutate({ id: props.id, ids: [r.id] }, { onError });

// -- details, delete ---------------------------------------------------------------------------
const name = ref('');
const language = ref('');
const description = ref('');
watch(
  dataset,
  (d) => {
    if (!d) return;
    name.value = d.name;
    language.value = d.language ?? '';
    description.value = d.description;
  },
  { immediate: true },
);
function saveDetails() {
  actions.update.mutate(
    { id: props.id, patch: { name: name.value, language: language.value, description: description.value } },
    { onSuccess: () => snackbar.show(t('settings.saved')), onError },
  );
}
const confirmDelete = ref(false);
function remove() {
  const d = dataset.value;
  if (!d) return;
  actions.remove.mutate(d.id, {
    onSuccess: () => {
      snackbar.show(t('train.deleted', { name: d.name }));
      router.push('/train');
    },
    onError,
  });
  confirmDelete.value = false;
}
const trainOpen = ref(false);
const histogramPeak = computed(() => Math.max(1, ...(report.data.value?.histogram ?? []).map((b) => b.count)));
const issues = computed(() => {
  const r = report.data.value;
  if (!r) return [];
  return (
    [
      ['too_short', r.too_short],
      ['too_long', r.too_long],
      ['empty_text', r.empty_text],
      ['clipping', r.clipping],
    ] as const
  ).filter(([, ids]) => ids.length);
});
</script>

<template>
  <div class="dataset" :style="morphStyle(`dataset:${id}`)">
    <div v-if="dataset" class="sections">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/train')" />
        <h1 class="type-headline-small title">{{ dataset.name }}</h1>
        <Badge :value="`${t('train.clips', { n: dataset.stats.rows })} · ${formatDuration(dataset.stats.seconds)}`" tone="neutral" />
      </header>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('train.addClips') }}</h2>
        <AudioSourceInput v-model="clips" :outputs="false" />
        <FieldRow>
          <TextField v-model="speaker" :label="t('train.speaker')" :supporting-text="t('train.speakerHint')" />
          <template #end>
            <AppButton :icon="icons.Plus" :disabled="!readyIds(clips).length" :loading="actions.addAudio.isPending.value" @click="addClips">{{ readyIds(clips).length ? t('train.addN', { n: readyIds(clips).length }) : t('common.add') }}</AppButton>
          </template>
        </FieldRow>
        <div class="line">
          <SelectField v-model="voiceId" :label="t('train.fromVoice')" :options="voiceOptions" class="grow" />
          <AppButton variant="tonal" :disabled="!voiceId" :loading="actions.addVoice.isPending.value" @click="addVoice">{{ t('common.add') }}</AppButton>
        </div>
        <div class="line wrap">
          <AppButton variant="tonal" :icon="icons.History" @click="outputsOpen = true">{{ t('train.fromOutputs') }}</AppButton>
          <template v-if="meta.data.value?.local">
            <SelectField v-model="layout" :label="t('train.layout')" :options="layoutOptions" class="layout" />
            <AppButton variant="tonal" :icon="icons.FolderInput" :loading="actions.importFolder.isPending.value" @click="folderOpen = true">{{ t('train.importFolder') }}</AppButton>
          </template>
        </div>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('train.prepare') }}</h2>
        <p class="type-body-medium muted">{{ t('train.prepareHint') }}</p>
        <div class="grid">
          <Switch v-model="prep.slice" :label="t('train.slice')" :supporting-text="t('train.sliceHint')" />
          <Switch v-model="prep.transcribe" :label="t('train.transcribe')" :supporting-text="t('train.transcribeHint')" />
          <Switch v-model="prep.normalize" :label="t('train.normalize')" :supporting-text="t('train.normalizeHint')" />
        </div>
        <div class="grid">
          <TextField v-model="prep.max_seconds" type="number" :min="2" :max="60" :label="t('train.maxSeconds')" :disabled="!prep.slice" />
          <TextField v-model="prep.min_seconds" type="number" :min="0.2" :max="10" :label="t('train.minSeconds')" />
          <TextField v-model="prep.threshold_db" type="number" :min="-80" :max="0" :label="t('train.threshold')" :disabled="!prep.slice" />
          <SelectField v-model="prep.language" :label="t('train.asrLanguage')" :options="languageOptions" :disabled="!prep.transcribe" />
        </div>
        <div class="row-end">
          <AppButton :icon="icons.Scissors" :disabled="!!prepareJob || !dataset.stats.rows" :loading="actions.prepare.isPending.value" @click="prepare">{{ t('train.prepareRun') }}</AppButton>
        </div>
        <JobCard v-if="prepareJob" :job="prepareJob" compact />
      </Surface>

      <Surface v-if="report.data.value && report.data.value.rows" :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('train.report') }}</h2>
        <p class="type-body-large">{{ t('train.usable', { usable: report.data.value.usable, rows: report.data.value.rows, time: formatDuration(report.data.value.seconds) }) }}</p>
        <div class="histogram" role="img" :aria-label="t('train.durations')">
          <div v-for="b in report.data.value.histogram" :key="b.seconds" class="bucket">
            <span class="bar" :style="{ height: `${(100 * b.count) / histogramPeak}%` }" />
            <span class="type-label-small muted">{{ b.seconds }}s</span>
          </div>
        </div>
        <ul class="issues">
          <li v-for="[kind, ids] in issues" :key="kind" class="type-body-medium">
            <Badge :value="ids.length" tone="warning" /> {{ t(`train.issues.${kind}`) }}
            <AppButton variant="text" @click="issue = kind">{{ t('train.showThem') }}</AppButton>
          </li>
        </ul>
        <p v-for="w in report.data.value.warnings" :key="w" class="type-body-medium warn">{{ w }}</p>
        <p v-if="report.data.value.licence_tags.length" class="type-body-small muted">{{ t('train.licenceTags', { tags: report.data.value.licence_tags.join(', ') }) }}</p>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <div class="section-head">
          <h2 class="type-title-medium">{{ t('train.rows') }}</h2>
          <FilterChip v-if="issue" :label="t(`train.issues.${issue}`)" :model-value="true" @update:model-value="issue = null" />
          <SearchField v-model="q" :label="t('train.searchRows')" :placeholder="t('train.searchRows')" class="search" />
        </div>
        <p v-if="rows.isSuccess.value && !page.items.length" class="type-body-medium muted">{{ t('train.noRows') }}</p>
        <ul class="rows">
          <li v-for="r in page.items" :key="r.id" class="row" :class="{ excluded: r.excluded }">
            <IconButton :icon="player.isPlaying(rowKey(r)) ? icons.Pause : icons.Play" :label="t('results.play')" @click="play(r)" />
            <span class="type-body-small muted dur">{{ r.duration.toFixed(1) }} s</span>
            <TextField
              :model-value="edits[r.id] ?? r.text"
              :label="r.speaker || t('train.text')"
              :error-text="!r.text && edits[r.id] === undefined ? t('train.noText') : ''"
              class="grow"
              @update:model-value="edits = { ...edits, [r.id]: String($event ?? '') }"
              @enter="saveText(r)"
            />
            <IconButton v-if="edits[r.id] !== undefined && edits[r.id] !== r.text" :icon="icons.Save" :label="t('common.save')" @click="saveText(r)" />
            <IconButton :icon="r.excluded ? icons.EyeOff : icons.Eye" :label="r.excluded ? t('train.include') : t('train.exclude')" @click="setExcluded(r, !r.excluded)" />
            <IconButton :icon="icons.Trash2" :label="t('common.remove')" @click="removeRow(r)" />
          </li>
        </ul>
        <div v-if="page.total > PAGE" class="pager">
          <IconButton :icon="icons.ChevronLeft" :label="t('train.previous')" :disabled="offset === 0" @click="offset = Math.max(0, offset - PAGE)" />
          <span class="type-body-small muted">{{ t('train.pageOf', { from: offset + 1, to: Math.min(offset + PAGE, page.total), total: page.total }) }}</span>
          <IconButton :icon="icons.ChevronRight" :label="t('common.next')" :disabled="offset + PAGE >= page.total" @click="offset += PAGE" />
        </div>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('voices.details') }}</h2>
        <div class="grid">
          <TextField v-model="name" :label="t('train.datasetName')" />
          <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" />
        </div>
        <TextField v-model="description" :label="t('common.description')" />
        <div class="row-end">
          <AppButton variant="tonal" :icon="icons.Save" :loading="actions.update.isPending.value" @click="saveDetails">{{ t('common.save') }}</AppButton>
        </div>
      </Surface>
    </div>
    <ErrorNotice v-else-if="datasetQuery.error.value" :error="datasetQuery.error.value" class="pad" />
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <div class="actions">
        <AppButton variant="text" :icon="icons.Trash2" :disabled="!dataset" @click="confirmDelete = true">{{ t('common.delete') }}</AppButton>
        <AppButton :icon="icons.GraduationCap" :disabled="!dataset?.stats.rows" @click="trainOpen = true">{{ t('train.trainOnIt') }}</AppButton>
      </div>
    </PageFooter>
    <OutputsPickerDialog v-model:open="outputsOpen" @select="addOutputs" />
    <ServerPathDialog v-model:open="folderOpen" folders :files="false" :title="t('train.importFolder')" @select="importFolder" />
    <NewExperimentDialog v-model:open="trainOpen" :dataset="id" @created="(e) => router.push(`/train/experiments/${e.engine}/${e.name}`)" />
    <ConfirmDialog
      v-model:open="confirmDelete"
      :title="t('common.delete')"
      :message="t('train.deleteDatasetConfirm', { name: dataset?.name ?? '' })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="remove"
    />
  </div>
</template>

<style scoped>
.dataset { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.section h2 { margin: 0; }
.section p { margin: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.section-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-4); }
.section-head h2 { flex: 1; }
.search { flex: 0 1 320px; min-width: 200px; }
.line { display: flex; align-items: center; gap: var(--app-space-3); min-width: 0; }
.line.wrap { flex-wrap: wrap; }
.layout { flex: 0 1 260px; }
.grow { flex: 1; min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: start; }
.row-end { display: flex; justify-content: flex-end; }
.histogram { display: flex; align-items: flex-end; gap: var(--app-space-2); height: 120px; padding-top: var(--app-space-2); }
.bucket { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; gap: var(--app-space-1); height: 100%; min-width: 0; }
.bar { width: 100%; min-height: 2px; border-radius: var(--md-sys-shape-corner-extra-small) var(--md-sys-shape-corner-extra-small) 0 0; background: var(--md-sys-color-primary); }
.issues { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.issues li { display: flex; align-items: center; gap: var(--app-space-2); }
.warn { color: var(--md-sys-color-tertiary); }
.rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.row { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0; }
.row.excluded { opacity: 0.55; }
.dur { flex: none; width: 4.5em; font-variant-numeric: tabular-nums; }
.pager { display: flex; align-items: center; justify-content: center; gap: var(--app-space-2); }
.actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--app-space-2); }
.pad { margin: var(--app-space-4) var(--app-space-6); }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
  .dur { display: none; }
}
</style>
