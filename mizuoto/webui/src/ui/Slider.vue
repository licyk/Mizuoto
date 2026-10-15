<script setup lang="ts">
import '@material/web/slider/slider.js';

withDefaults(defineProps<{ label: string; min?: number; max?: number; step?: number; ticks?: boolean }>(), { min: 0, max: 100, step: 1 });
const model = defineModel<number>({ default: 0 });
</script>

<template>
  <label class="slider-row">
    <span class="type-body-large">{{ label }}</span>
    <md-slider
      labeled
      :min="min"
      :max="max"
      :step="step"
      :ticks.prop="ticks"
      :value.prop="model"
      :aria-label="label"
      @change="model = Number(($event.target as HTMLInputElement).value)"
    />
  </label>
</template>

<style scoped>
.slider-row { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-4); min-height: 56px; }
md-slider { flex: 0 1 clamp(260px, 30vw, 560px); }
</style>
