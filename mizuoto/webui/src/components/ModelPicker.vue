<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useEngines } from '@/api/queries/engines';
import { useModels } from '@/api/queries/models';
import type { ModelInfo } from '@/api/types';
import { sizeText } from '@/components/capabilities';
import { useI18n } from '@/i18n';
import { AppButton, PickerMenu, icons, type PickerOption } from '@/ui';

/**
 * Choose a speech model: every catalogue and installed model of every synthesis engine, searchable,
 * with its engine, variant and size, and badges for what is missing (the engine, the download, the
 * licence). A model that is not ready can still be chosen: ``ModelGate`` then offers to get it.
 */
const props = withDefaults(defineProps<{ label?: string; installedOnly?: boolean; capability?: string; disabled?: boolean }>(), { label: '', installedOnly: false, capability: '' });
const model = defineModel<string | null>({ default: null });
const { t, tOr, tx } = useI18n();
const router = useRouter();
const models = useModels();
const engines = useEngines();

const engineById = computed(() => new Map((engines.data.value ?? []).map((e) => [e.id, e])));
const speechModels = computed(() => (models.data.value ?? []).filter((m) => !engineById.value.get(m.engine)?.internal));
const ready = (m: ModelInfo) => {
  const e = engineById.value.get(m.engine);
  return (m.state === 'installed' || m.state === 'builtin') && !!e && e.available && e.licence_accepted && e.enabled;
};
const shown = computed(() => speechModels.value.filter((m) => !props.installedOnly || ready(m)));

const options = computed<PickerOption[]>(() =>
  shown.value.map((m) => {
    const e = engineById.value.get(m.engine);
    const badges: PickerOption['badges'] = [];
    if (m.state === 'builtin') badges.push({ text: t('speak.demo'), tone: 'neutral' });
    if (e && !e.available) badges.push({ text: t('engines.unavailable'), tone: 'warning' });
    if (e && !e.licence_accepted) badges.push({ text: t('engines.licence'), tone: 'warning' });
    if (!['installed', 'builtin'].includes(m.state)) badges.push({ text: tOr(`resource.states.${m.state}`, m.state), tone: m.state === 'corrupt' ? 'error' : 'warning' });
    if (m.recommended && !ready(m)) badges.push({ text: t('models.recommended'), tone: 'primary' });
    const size = m.state === 'builtin' ? '' : sizeText(m.size_mb, m.vram_mb, (gb) => t('models.vram', { gb }));
    return { value: m.id, label: m.title, description: [e ? tx(e.title) : m.engine, m.variant, size].filter(Boolean).join(' · '), keywords: [m.id, m.engine, m.variant], badges };
  }),
);
</script>

<template>
  <div class="model-picker">
    <div v-if="models.isSuccess.value && !shown.length" class="none">
      <span class="type-body-medium">{{ t('speak.noModels') }}</span>
      <AppButton variant="tonal" :icon="icons.Library" @click="router.push('/models')">{{ t('speak.goModels') }}</AppButton>
    </div>
    <PickerMenu
      v-else
      v-model="model"
      :label="label || t('speak.model')"
      :options="options"
      :icon="icons.Boxes"
      :placeholder="models.isLoading.value ? t('common.loading') : t('speak.chooseModel')"
      :loading="models.isLoading.value"
      :disabled="disabled"
      :search-label="t('speak.searchModels')"
      :no-matches="t('common.noMatches')"
    />
  </div>
</template>

<style scoped>
.model-picker { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.none { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); }
</style>
