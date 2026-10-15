<script setup lang="ts">
import '@material/web/checkbox/checkbox.js';

/** ``dense`` drops the 48px touch target, for rows and cards where it would dwarf the content. */
defineProps<{ label?: string; indeterminate?: boolean; disabled?: boolean; dense?: boolean }>();
const model = defineModel<boolean>({ default: false });
</script>

<template>
  <label class="checkbox" :class="{ dense }">
    <md-checkbox
      :touch-target="dense ? 'none' : 'wrapper'"
      :checked.prop="model"
      :indeterminate.prop="indeterminate"
      :disabled.prop="disabled"
      :aria-label="label"
      @change="model = ($event.target as HTMLInputElement).checked"
      @click.stop
    />
    <span v-if="label || $slots.default" class="type-body-medium"><slot>{{ label }}</slot></span>
  </label>
</template>

<style scoped>
.checkbox { display: inline-flex; align-items: center; gap: var(--app-space-1); cursor: pointer; }
.dense { --md-checkbox-container-size: 16px; --md-checkbox-icon-size: 14px; --md-checkbox-state-layer-size: 28px; }
.dense md-checkbox { display: block; }
</style>
