<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useSegments } from '@/api/queries/tts';
import { useI18n } from '@/i18n';
import { Chip, SelectField, Switch, TextArea } from '@/ui';

/**
 * The text to speak: a multi-line field with a counter, its language (the model's languages, or
 * detect), one result per line for models that take batches, and where Mizuoto will cut long text
 * (fetched a moment after typing stops).
 */
const props = withDefaults(defineProps<{ model: string | null; languages?: string[]; batchable?: boolean; maxLength?: number }>(), { languages: () => [], batchable: false, maxLength: 20000 });
const text = defineModel<string>({ default: '' });
const language = defineModel<string>('language', { default: 'auto' });
const batch = defineModel<boolean>('batch', { default: false });
const emit = defineEmits<{ submit: [] }>();
const { t, tOr } = useI18n();
const area = ref<InstanceType<typeof TextArea> | null>(null);

const settled = ref(text.value);
let timer: ReturnType<typeof setTimeout> | undefined;
watch(text, (v) => {
  clearTimeout(timer);
  timer = setTimeout(() => (settled.value = v), 600);
});
// The switch is kept while a model without batches is chosen (choosing one back brings it back), and off for it.
const batching = computed(() => props.batchable && batch.value);
const segments = useSegments(() => (batching.value ? '' : settled.value), () => props.model);
const pieces = computed(() => segments.data.value?.segments ?? []);

const languageOptions = computed(() => {
  const codes = props.languages.includes('*') || !props.languages.length ? ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'it', 'pt', 'ru'] : props.languages;
  return [{ value: 'auto', label: t('languages.auto') }, ...codes.filter((c) => c !== 'auto').map((c) => ({ value: c, label: tOr(`languages.${c}`, c) }))];
});
// Also when the editor appears: it is shown anew for a model whose spec was still loading.
watch(
  languageOptions,
  (opts) => {
    if (!opts.some((o) => o.value === language.value)) language.value = 'auto';
  },
  { immediate: true },
);
const lines = computed(() => text.value.split('\n').filter((l) => l.trim()).length);
defineExpose({ insert: (s: string) => area.value?.insert(s) });
</script>

<template>
  <div class="script">
    <TextArea ref="area" v-model="text" :label="t('speak.text')" :placeholder="t('speak.textPlaceholder')" :rows="5" :max-length="maxLength" @submit="emit('submit')" />
    <div class="row">
      <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" class="lang" />
      <Switch v-if="batchable" v-model="batch" :label="t('speak.batch')" />
    </div>
    <div v-if="batching && lines > 1" class="type-body-small muted">{{ t('speak.segments', { n: lines }) }}</div>
    <div v-else-if="pieces.length > 1" class="segments" :aria-label="t('speak.segments', { n: pieces.length })">
      <span class="type-label-medium muted">{{ t('speak.segments', { n: pieces.length }) }}</span>
      <Chip v-for="(p, i) in pieces" :key="i" :label="p.length > 32 ? `${p.slice(0, 32)}…` : p" />
    </div>
    <div v-else-if="segments.data.value?.segmented_by === 'engine' && settled.length > 200" class="type-body-small muted">{{ t('speak.segmentsEngine') }}</div>
  </div>
</template>

<style scoped>
.script { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-4); }
.lang { min-width: 200px; }
.segments { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
</style>
