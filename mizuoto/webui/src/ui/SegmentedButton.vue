<script setup lang="ts" generic="T extends string">
import type { Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { Check } from '@/ui/icons';

/** Single-select segmented button, built from the tokens (@material/web has none). */
defineProps<{ options: { value: T; label?: string; icon?: Component; ariaLabel?: string }[] }>();
const model = defineModel<T>({ required: true });
</script>

<template>
  <div class="segmented" role="radiogroup">
    <button
      v-for="o in options"
      :key="o.value"
      type="button"
      role="radio"
      class="segment state-layer"
      :class="{ selected: model === o.value }"
      :aria-checked="model === o.value"
      :aria-label="o.ariaLabel ?? o.label"
      :title="o.ariaLabel ?? o.label"
      @click="model = o.value"
    >
      <AppIcon v-if="model === o.value && o.label" :icon="Check" :size="18" />
      <AppIcon v-else-if="o.icon" :icon="o.icon" :size="18" />
      <span v-if="o.label" class="type-label-large">{{ o.label }}</span>
    </button>
  </div>
</template>

<style scoped>
/* Its own width, even as a stretched flex or grid item: the outline belongs to the segments. */
.segmented { display: inline-flex; width: fit-content; max-width: 100%; border: 1px solid var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-full); overflow: hidden; height: 40px; }
.segment {
  display: inline-flex; align-items: center; gap: var(--app-space-2); padding: 0 var(--app-space-3);
  min-width: 48px; justify-content: center; border: 0; background: transparent; color: var(--md-sys-color-on-surface); cursor: pointer; font: inherit;
}
.segment + .segment { border-left: 1px solid var(--md-sys-color-outline); }
.segment.selected { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
</style>
