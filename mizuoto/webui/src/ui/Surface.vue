<script setup lang="ts">
/** A container whose depth is shown by its surface tone first. */
withDefaults(defineProps<{ level?: 0 | 1 | 2 | 3 | 4; shape?: 'none' | 'small' | 'medium' | 'large' | 'extra-large'; as?: string }>(), { level: 1, shape: 'medium', as: 'div' });
</script>

<template>
  <component :is="as" class="surface" :class="[`level-${level}`, `shape-${shape}`]"><slot /></component>
</template>

<style scoped>
.surface { color: var(--md-sys-color-on-surface); }
.level-0 { background: var(--md-sys-color-surface); }
.level-1 { background: var(--md-sys-color-surface-container-low); }
.level-2 { background: var(--md-sys-color-surface-container); }
.level-3 { background: var(--md-sys-color-surface-container-high); }
.level-4 { background: var(--md-sys-color-surface-container-highest); }
.shape-none { border-radius: var(--md-sys-shape-corner-none); }
.shape-small { border-radius: var(--md-sys-shape-corner-small); }
.shape-medium { border-radius: var(--md-sys-shape-corner-medium); }
.shape-large { border-radius: var(--md-sys-shape-corner-large); }
.shape-extra-large { border-radius: var(--md-sys-shape-corner-extra-large); }
</style>
