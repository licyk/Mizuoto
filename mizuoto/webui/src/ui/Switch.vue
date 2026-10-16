<script setup lang="ts">
import '@material/web/switch/switch.js';

/** A labelled switch: a settings row, or with ``inline`` a compact label and switch for a toolbar. */
defineProps<{ label: string; supportingText?: string; disabled?: boolean; inline?: boolean }>();
const model = defineModel<boolean>({ default: false });
</script>

<template>
  <label class="switch-row" :class="{ inline }">
    <span class="text">
      <span :class="inline ? 'type-label-large' : 'type-body-large'">{{ label }}</span>
      <span v-if="supportingText" class="type-body-medium muted">{{ supportingText }}</span>
    </span>
    <md-switch :selected.prop="model" :disabled.prop="disabled" :aria-label="label" @change="model = ($event.target as HTMLInputElement & { selected: boolean }).selected" />
  </label>
</template>

<style scoped>
.switch-row { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-4); min-height: 56px; cursor: pointer; }
.text { display: flex; flex-direction: column; }
.switch-row.inline { display: inline-flex; justify-content: flex-start; gap: var(--app-space-2); min-height: 40px; padding: 0 var(--app-space-2); }
</style>
