<script setup lang="ts">
import type { SynthesisPlan } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppDialog, DataList } from '@/ui';

/** What a request will do, before it runs: the voice mode, transcriptions first, segments, parameters, notes. */
defineProps<{ plan: SynthesisPlan | null }>();
const open = defineModel<boolean>('open', { default: false });
const { t, tOr } = useI18n();
</script>

<template>
  <AppDialog v-model:open="open" :title="t('speak.planTitle')" width="medium" :close-label="t('common.close')">
    <div v-if="plan" class="plan">
      <DataList
        :rows="[
          { label: t('speak.model'), value: plan.model },
          { label: t('speak.planMode'), value: tOr(`voiceModes.${plan.voice_mode}`, plan.voice_mode) },
          { label: t('common.language'), value: plan.language ? tOr(`languages.${plan.language}`, plan.language) : t('languages.auto') },
          { label: t('speak.planDevice'), value: plan.device },
          { label: t('speak.planSegments'), value: String(plan.segments.length) },
          { label: t('speak.planParams'), value: Object.keys(plan.params).length ? Object.entries(plan.params).map(([k, v]) => `${k}=${v}`).join(', ') : '—' },
        ]"
      />
      <ul v-if="plan.references.length" class="refs type-body-medium">
        <li v-for="r in plan.references" :key="r.audio_id">
          {{ r.name }} · {{ r.duration.toFixed(1) }} s<template v-if="r.transcribe"> · {{ t('speak.planTranscribe') }}</template>
          <div v-if="r.transcript" class="type-body-small muted">{{ r.transcript }}</div>
        </li>
      </ul>
      <ol v-if="plan.segments.length > 1" class="segments type-body-small">
        <li v-for="(s, i) in plan.segments" :key="i">{{ s }}</li>
      </ol>
      <div v-if="plan.warnings.length" class="warnings">
        <h3 class="type-title-small">{{ t('speak.planWarnings') }}</h3>
        <p v-for="w in plan.warnings" :key="w" class="type-body-small">{{ w }}</p>
      </div>
    </div>
  </AppDialog>
</template>

<style scoped>
.plan { display: flex; flex-direction: column; gap: var(--app-space-3); }
.refs, .segments { margin: 0; padding-left: var(--app-space-5); display: flex; flex-direction: column; gap: var(--app-space-1); }
.warnings h3 { margin: 0 0 var(--app-space-1); }
.warnings p { margin: 0; color: var(--md-sys-color-tertiary); }
</style>
