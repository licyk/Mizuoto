<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useModelVoiceFits, useVoices } from '@/api/queries/voices';
import type { VoiceFit } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppButton, PickerMenu, icons, type PickerOption } from '@/ui';

/**
 * Choose a library voice for a model. Each voice says how this model can use it: as a clone, with
 * its transcript made first, trimmed, as a preset speaker, or not at all (listed, greyed, with why).
 */
const props = withDefaults(defineProps<{ model: string | null; label?: string; disabled?: boolean }>(), { label: '' });
const voice = defineModel<string | null>({ default: null });
const { t, tOr } = useI18n();
const router = useRouter();
const voices = useVoices();
const fits = useModelVoiceFits(() => props.model);

const fitById = computed(() => new Map((fits.data.value ?? []).map((f) => [f.voice_id, f])));
const badges = (fit: VoiceFit | undefined): PickerOption['badges'] => {
  if (!fit) return [];
  if (!fit.usable) return [{ text: t('fit.notUsable'), tone: 'error' }];
  const out: PickerOption['badges'] = [{ text: tOr(`voiceModes.${fit.mode}`, fit.mode ?? ''), tone: 'primary' }];
  for (const n of fit.needs) out.push({ text: tOr(`fit.${n}`, n), tone: 'warning' });
  return out;
};
const options = computed<PickerOption[]>(() =>
  (voices.data.value ?? []).map((v) => {
    const fit = fitById.value.get(v.id);
    return {
      value: v.id,
      label: v.name,
      description: [tOr(`voices.kinds.${v.kind}`, v.kind), v.language ? tOr(`languages.${v.language}`, v.language) : '', fit && !fit.usable ? fit.reason : ''].filter(Boolean).join(' · '),
      keywords: [v.slug, ...v.tags],
      badges: badges(fit),
      unavailable: !!fit && !fit.usable,
    };
  }),
);
</script>

<template>
  <div class="voice-picker">
    <div v-if="voices.isSuccess.value && !options.length" class="none">
      <span class="type-body-medium">{{ t('voices.empty') }}</span>
      <AppButton variant="tonal" :icon="icons.UserPlus" @click="router.push('/voices')">{{ t('voices.new') }}</AppButton>
    </div>
    <PickerMenu
      v-else
      v-model="voice"
      :label="label || t('speak.voice')"
      :options="options"
      :icon="icons.Users"
      :placeholder="voices.isLoading.value ? t('common.loading') : t('common.choose')"
      :loading="voices.isLoading.value"
      :disabled="disabled"
      :search-label="t('voices.search')"
      :no-matches="t('common.noMatches')"
    />
  </div>
</template>

<style scoped>
.voice-picker { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.none { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); }
</style>
