<script setup lang="ts">
import '@material/web/chips/filter-chip.js';

defineProps<{ label: string; disabled?: boolean }>();
const model = defineModel<boolean>({ default: false });
</script>

<template>
  <md-filter-chip
    :label="label"
    :selected.prop="model"
    :disabled.prop="disabled"
    @click="model = ($event.target as HTMLElement & { selected: boolean }).selected"
  />
</template>
