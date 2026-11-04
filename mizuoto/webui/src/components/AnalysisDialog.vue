<script setup lang="ts">
import { computed } from 'vue';
import { useOutputAnalysis } from '@/api/queries/outputs';
import type { Output } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { AppDialog, DataList, LineChart, ProgressBar, Spectrogram, type Series } from '@/ui';

/**
 * A result's spectrogram and pitch curve, to judge it by more than ear: a voice's high-frequency
 * cut-off, noise and clicks, octave jumps, a monotone stretch, how high the voice sits.
 */
const props = defineProps<{ output: Output | null }>();
const open = defineModel<boolean>('open', { default: false });
const { t } = useI18n();
const analysis = useOutputAnalysis(() => (open.value ? (props.output?.id ?? null) : null));
const shown = computed(() => analysis.data.value ?? null);

// unvoiced frames break the line (NaN) rather than dropping it to zero
const series = computed<Series[]>(() => {
  const a = shown.value;
  if (!a?.pitch_median) return [];
  return [{ label: t('analysis.pitchLine'), points: a.pitch.map((hz, i) => [i * a.pitch_hop, hz > 0 ? hz : Number.NaN] as [number, number]) }];
});
const voiced = computed(() => {
  const p = shown.value?.pitch ?? [];
  return p.length ? p.filter((hz) => hz > 0).length / p.length : 0;
});
const rows = computed(() => {
  const a = shown.value;
  if (!a) return [];
  return [
    { label: t('analysis.duration'), value: formatDuration(a.duration) },
    { label: t('analysis.format'), value: `${a.sample_rate / 1000} kHz · ${a.channels === 1 ? t('analysis.mono') : t('analysis.channels', { n: a.channels })}${a.codec ? ` · ${a.codec}` : ''}` },
    { label: t('analysis.medianPitch'), value: a.pitch_median ? `${Math.round(a.pitch_median)} Hz` : t('analysis.noPitch') },
    { label: t('analysis.voiced'), value: `${Math.round(voiced.value * 100)} %` },
  ];
});
</script>

<template>
  <AppDialog v-model:open="open" :title="t('analysis.title', { name: output?.name ?? '' })" width="large" :close-label="t('common.close')">
    <ProgressBar v-if="analysis.isLoading.value" :label="t('analysis.working')" />
    <ErrorNotice v-else-if="analysis.error.value" :error="analysis.error.value" />
    <div v-else-if="shown" class="analysis">
      <Spectrogram :rows="shown.spectrogram.rows" :cols="shown.spectrogram.cols" :data="shown.spectrogram.data" :label="t('analysis.spectrogram')" :height="180" />
      <p class="type-body-small muted">{{ t('analysis.spectrogramHint', { khz: (shown.spectrogram.fmax / 1000).toFixed(1) }) }}</p>
      <DataList :rows="rows" />
      <h3 class="type-title-small sub">{{ t('analysis.pitch') }}</h3>
      <LineChart :series="series" :x-label="t('analysis.seconds')" :empty-text="t('analysis.noPitch')" :height="200" />
    </div>
  </AppDialog>
</template>

<style scoped>
.analysis { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.sub { margin: var(--app-space-2) 0 0; }
</style>
