<script setup lang="ts">
import AppIcon from '@/ui/AppIcon.vue';
import { ChevronRight } from '@/ui/icons';

export interface Crumb {
  label: string;
  value: string;
}
defineProps<{ crumbs: Crumb[] }>();
defineEmits<{ navigate: [string] }>();
</script>

<template>
  <nav class="crumbs" aria-label="Breadcrumbs">
    <template v-for="(c, i) in crumbs" :key="c.value">
      <AppIcon v-if="i > 0" :icon="ChevronRight" :size="18" class="sep" />
      <button
        v-if="i < crumbs.length - 1"
        type="button"
        class="crumb state-layer type-label-large"
        @click="$emit('navigate', c.value)"
      >{{ c.label }}</button>
      <span v-else class="crumb current type-label-large" aria-current="location">{{ c.label }}</span>
    </template>
  </nav>
</template>

<style scoped>
.crumbs { display: flex; align-items: center; flex-wrap: wrap; gap: 2px; min-width: 0; }
.crumb { border: 0; background: transparent; color: var(--md-sys-color-on-surface-variant); padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); cursor: pointer; font: inherit; font-weight: 500; }
.current { color: var(--md-sys-color-on-surface); cursor: default; }
.sep { color: var(--md-sys-color-on-surface-variant); }
</style>
