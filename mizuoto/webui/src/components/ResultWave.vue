<script setup lang="ts">
import { computed } from 'vue';
import { usePeaks } from '@/api/queries/audio';
import type { Output } from '@/api/types';
import { Waveform } from '@/ui';

/** An output's waveform from its server peaks (fetched once per output). */
const props = defineProps<{ output: Output; progress: number }>();
defineEmits<{ seek: [number] }>();
const peaks = usePeaks(() => ({ kind: 'output', id: props.output.id }), 512);
const data = computed(() => peaks.data.value?.data ?? null);
</script>

<template>
  <Waveform :peaks="data" :progress="progress" :height="40" :label="output.name" @seek="$emit('seek', $event)" />
</template>
