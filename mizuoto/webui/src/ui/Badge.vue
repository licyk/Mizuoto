<script setup lang="ts">
defineProps<{ value?: number | string; tone?: 'error' | 'primary' | 'neutral' | 'warning' }>();
</script>

<template>
  <span class="badge type-label-small" :class="[tone ?? 'error', { dot: value === undefined }]">{{ value }}</span>
</template>

<style scoped>
.badge {
  display: inline-flex; align-items: center; justify-content: center; min-width: 16px; height: 16px; padding: 0 4px;
  border-radius: var(--md-sys-shape-corner-full); font-size: var(--md-sys-typescale-label-small-size); font-weight: 500;
}
.dot { min-width: 6px; width: 6px; height: 6px; padding: 0; }
.error { background: var(--md-sys-color-error); color: var(--md-sys-color-on-error); }
.primary { background: var(--md-sys-color-primary); color: var(--md-sys-color-on-primary); }
.neutral { background: var(--md-sys-color-surface-container-highest); color: var(--md-sys-color-on-surface-variant); }
.warning { background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
</style>
