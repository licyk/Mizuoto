<script setup lang="ts">
import { computed, ref } from 'vue';
import type { Job } from '@/api/types';
import JobCard from '@/components/JobCard.vue';
import { useI18n } from '@/i18n';
import { AppButton, Badge, DropZone, LevelMeter, LineChart, ParamSlider, ProgressBar, SegmentedControl, Stepper, Surface, Waveform, PickerMenu, icons, type PickerOption, type StepItem } from '@/ui';

/** Every shared component, for checking the theme and the motion in one place. */
const { t } = useI18n();
const peaks = computed(() => Array.from({ length: 512 }, (_, i) => Math.sin(i / 9) * 0.7 * Math.abs(Math.sin(i / 70))).flatMap((v) => [-Math.abs(v), Math.abs(v)]));
const progress = ref(0.35);
const value = ref(0);
const seg = ref<'a' | 'b'>('a');
const step = ref<'one' | 'two' | 'three'>('two');
const steps: StepItem<'one' | 'two' | 'three'>[] = [
  { id: 'one', label: 'Dataset', state: 'done' },
  { id: 'two', label: 'Run', state: 'running', note: 'Pitch' },
  { id: 'three', label: 'Results', state: 'stale' },
];
const devices: PickerOption[] = [
  { value: 'd', label: 'System default', description: 'Follows the system default device' },
  { value: 'mic', label: 'Microphone (USB Audio Device)', description: 'Windows WASAPI · 1 ch · 48 kHz', badges: [{ text: 'default', tone: 'primary' }] },
  { value: 'cable', label: 'CABLE Output (VB-Audio Virtual Cable)', description: 'Windows WASAPI · 2 ch · 48 kHz', badges: [{ text: 'virtual' }] },
  { value: 'gone', label: 'Old headset', badges: [{ text: 'not connected', tone: 'warning' }], unavailable: true },
];
const device = ref<string | null>('mic');
const series = [
  { label: 'g_total', points: Array.from({ length: 80 }, (_, i) => [i * 10, 40 / (1 + i / 10) + Math.random()] as [number, number]) },
  { label: 'd_total', points: Array.from({ length: 80 }, (_, i) => [i * 10, 3 + Math.sin(i / 5) * 0.3] as [number, number]) },
];
const job: Job = {
  id: 'demo', kind: 'speak', title: 'Speak: demo · Alto', state: 'running', waiting_for: null, progress: 0.42, step: 'conv:0', message: 'Loading the model: weights', origin: null,
  steps: [{ id: 'conv:0', title: 'demo.wav', state: 'running', progress: 0.42 }], error: null, result: null, request: {}, created_at: '', started_at: null, finished_at: null, can_retry: false, can_run_anyway: false,
  transfer: { done_bytes: 88_000_000, total_bytes: 210_000_000, bytes_per_second: 8_400_000, eta_seconds: 14.5 },
};
</script>

<template>
  <div class="dev">
    <Surface :level="0" class="panel">
      <h2 class="type-title-medium">{{ t('dev.title') }}</h2>
      <Waveform :peaks="peaks" :progress="progress" label="demo" @seek="progress = $event" />
      <LevelMeter label="in" :rms-db="-24" :peak-db="-12" />
      <LevelMeter label="hot" :rms-db="-6" :peak-db="-0.5" />
      <ParamSlider v-model="value" label="Pitch" help="Shift in semitones" :min="-24" :max="24" unit="st" :default-value="0" />
      <SegmentedControl v-model="seg" :options="[{ value: 'a', label: 'Result' }, { value: 'b', label: 'Source' }]" />
      <Stepper v-model="step" :steps="steps" />
      <PickerMenu v-model="device" label="Input" :icon="icons.Mic" :options="devices" />
      <DropZone label="Drop audio files here" hint="or click" />
      <ProgressBar :value="progress" label="progress" />
      <div class="row"><Badge value="default" tone="primary" /><Badge value="virtual" tone="neutral" /><Badge value="not connected" tone="warning" /></div>
      <LineChart :series="series" x-label="step" />
      <JobCard :job="job" />
      <AppButton :icon="icons.Mic">Start</AppButton>
    </Surface>
  </div>
</template>

<style scoped>
.dev { padding: var(--app-space-4); }
.panel { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); }
.row { display: flex; gap: var(--app-space-2); }
</style>
