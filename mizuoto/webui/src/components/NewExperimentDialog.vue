<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useEngines } from '@/api/queries/engines';
import { useModels } from '@/api/queries/models';
import { useDatasets, useExperimentActions, useTrainingMethods } from '@/api/queries/train';
import { useVoices } from '@/api/queries/voices';
import type { Experiment } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import HyperparamsForm from '@/components/HyperparamsForm.vue';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, SelectField, TextField } from '@/ui';

/**
 * A new experiment: the model to start from, the way to train it (with why a way is not available),
 * the dataset, a name, and the method's main settings. Where it exports a voice's own weights, the
 * voice can be one of the library's (else a new voice is made from the dataset's clips).
 */
const props = withDefaults(defineProps<{ dataset?: string | null }>(), { dataset: null });
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ created: [Experiment] }>();
const { t } = useI18n();
const engines = useEngines();
const models = useModels(() => ({ installed: true }));
const datasets = useDatasets();
const voices = useVoices();
const actions = useExperimentActions();

// Models whose engine trains, of a variant some method starts from.
const trainable = computed(() => {
  const methods = new Map((engines.data.value ?? []).map((e) => [e.id, Object.values(e.training ?? {})]));
  return (models.data.value ?? []).filter((m) => (methods.get(m.engine) ?? []).some((variants) => variants === null || variants.includes(m.variant)));
});
const model = ref<string | null>(null);
const datasetId = ref<string | null>(null);
const method = ref<string | null>(null);
const name = ref('');
const voice = ref('');
const hyperparams = ref<Record<string, unknown>>({});
const error = ref<unknown>(null);
const methods = useTrainingMethods(model);
const chosen = computed(() => (methods.data.value ?? []).find((m) => m.id === method.value) ?? null);
/** A memory estimate (MB) in GB, one decimal. */
const gb = (mb: number | undefined) => (mb === undefined ? '?' : (mb / 1024).toFixed(1));

watch(open, (o) => {
  if (!o) return;
  error.value = null;
  hyperparams.value = {};
  datasetId.value = props.dataset ?? datasetId.value ?? datasets.data.value?.[0]?.id ?? null;
  if (!model.value) model.value = trainable.value[0]?.id ?? null;
});
watch(trainable, (list) => {
  if (!model.value || !list.some((m) => m.id === model.value)) model.value = list[0]?.id ?? null;
});
watch(
  () => datasets.data.value,
  (list) => {
    if (!datasetId.value) datasetId.value = list?.[0]?.id ?? null;
  },
);
// The first available method of the model, unless the one chosen still fits.
watch(
  () => methods.data.value,
  (list) => {
    if (!list?.some((m) => m.id === method.value)) method.value = (list?.find((m) => m.available) ?? list?.[0])?.id ?? null;
    hyperparams.value = {};
  },
);
// A name from the dataset and the model, until one is typed.
const suggested = computed(() => {
  const d = (datasets.data.value ?? []).find((x) => x.id === datasetId.value);
  const m = (model.value ?? '').split('/').pop() ?? '';
  return [d?.slug, m].filter(Boolean).join('-');
});
const typed = ref(false);
watch(suggested, (s) => {
  if (!typed.value) name.value = s;
});
watch(open, (o) => {
  if (o) {
    typed.value = false;
    name.value = suggested.value;
  }
});

const modelOptions = computed(() => trainable.value.map((m) => ({ value: m.id, label: m.title })));
const methodOptions = computed(() => (methods.data.value ?? []).map((m) => ({ value: m.id, label: m.available ? m.method.title : `${m.method.title} (${t('train.unavailable')})` })));
const datasetOptions = computed(() => (datasets.data.value ?? []).map((d) => ({ value: d.id, label: `${d.name} · ${t('train.clips', { n: d.stats.rows })}` })));
const voiceOptions = computed(() => [{ value: '', label: t('train.newVoice') }, ...(voices.data.value ?? []).filter((v) => v.kind !== 'preset').map((v) => ({ value: v.id, label: v.name }))]);
const ready = computed(() => !!model.value && !!datasetId.value && !!chosen.value?.available && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name.value.trim()));

function create() {
  if (!model.value || !datasetId.value || !chosen.value) return;
  error.value = null;
  actions.create.mutate(
    {
      name: name.value.trim(),
      base_model: model.value,
      dataset: datasetId.value,
      method: chosen.value.id,
      hyperparams: hyperparams.value,
      voice: chosen.value.method.export === 'voice_weights' && voice.value ? voice.value : null,
      language: null,
    },
    {
      onSuccess: (e) => {
        open.value = false;
        emit('created', e);
      },
      onError: (e) => (error.value = e),
    },
  );
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('train.newExperiment')" width="large" :close-label="t('common.close')">
    <div class="body">
      <p v-if="engines.isSuccess.value && models.isSuccess.value && !trainable.length" class="type-body-medium muted">{{ t('train.noTrainableModels') }}</p>
      <div class="grid">
        <SelectField v-model="model" :label="t('train.baseModel')" :options="modelOptions" />
        <SelectField v-model="method" :label="t('train.method')" :options="methodOptions" />
      </div>
      <template v-if="chosen">
        <p class="type-body-medium">{{ chosen.method.description }}</p>
        <p v-if="!chosen.available" class="type-body-medium warn">{{ chosen.reason }}</p>
        <p v-else-if="chosen.method.note" class="type-body-small muted">{{ chosen.method.note }}</p>
        <p class="type-body-small muted">
          {{ t('train.stagesList', { stages: chosen.method.stages.map((s) => s.title).join(' → ') }) }} ·
          {{ t(`train.exports.${chosen.method.export}`) }} ·
          {{ chosen.method.min_seconds < 60 ? t('train.minimumSeconds', { seconds: Math.round(chosen.method.min_seconds) }) : t('train.minimum', { minutes: Math.round(chosen.method.min_seconds / 60) }) }}
        </p>
        <p v-if="chosen.memory_mb" class="type-body-small muted">
          {{ t('train.memoryEstimate', { standard: gb(chosen.memory_mb.standard), low: gb(chosen.memory_mb.low), lowest: gb(chosen.memory_mb.lowest) }) }}
        </p>
      </template>
      <div class="grid">
        <SelectField v-model="datasetId" :label="t('train.dataset')" :options="datasetOptions" />
        <TextField v-model="name" :label="t('train.experimentName')" :supporting-text="t('train.nameHint')" @update:model-value="typed = true" />
        <SelectField v-if="chosen?.method.export === 'voice_weights'" v-model="voice" :label="t('train.voice')" :options="voiceOptions" :supporting-text="t('train.voiceHint')" />
      </div>
      <p v-if="datasets.isSuccess.value && !datasets.data.value?.length" class="type-body-medium muted">{{ t('train.noDatasetsHint') }}</p>
      <HyperparamsForm v-if="chosen" v-model="hyperparams" :specs="chosen.method.hyperparams" />
      <p v-if="chosen" class="type-body-small muted">{{ t('train.moreLater') }}</p>
      <ErrorNotice v-if="error" :error="error" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!ready" :loading="actions.create.isPending.value" @click="create">{{ t('train.create') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.body { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-4); align-items: start; }
.warn { color: var(--md-sys-color-error); }
p { margin: 0; }
</style>
