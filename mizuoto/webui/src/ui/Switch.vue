<script setup lang="ts">
import '@material/web/switch/switch.js';

defineProps<{ label: string; supportingText?: string; disabled?: boolean }>();
const model = defineModel<boolean>({ default: false });
</script>

<template>
  <label class="switch-row">
    <span class="text">
      <span class="type-body-large">{{ label }}</span>
      <span v-if="supportingText" class="type-body-medium muted">{{ supportingText }}</span>
    </span>
    <md-switch :selected.prop="model" :disabled.prop="disabled" :aria-label="label" @change="model = ($event.target as HTMLInputElement & { selected: boolean }).selected" />
  </label>
</template>

<style scoped>
.switch-row { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-4); min-height: 56px; cursor: pointer; }
.text { display: flex; flex-direction: column; }
</style>
