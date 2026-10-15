<script setup lang="ts">
import '@material/web/progress/circular-progress.js';

withDefaults(defineProps<{ value?: number | null; size?: number; label?: string }>(), { size: 40 });
</script>

<template>
  <md-circular-progress
    :value.prop="value ?? 0"
    :indeterminate.prop="value === null || value === undefined"
    :aria-label="label"
    :style="{ '--md-circular-progress-size': `${size}px` }"
  />
</template>
