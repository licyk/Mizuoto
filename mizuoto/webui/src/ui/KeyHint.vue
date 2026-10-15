<script setup lang="ts">
/** A key, as keyboard shortcuts show it. */
defineProps<{ keys: string }>();
const pretty = (key: string) => ({ ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', ' ': 'Space', Escape: 'Esc' })[key] ?? key;
</script>

<template>
  <kbd class="key type-label-small">{{ pretty(keys) }}</kbd>
</template>

<style scoped>
.key {
  display: inline-flex; align-items: center; justify-content: center; min-width: 22px; height: 22px; padding: 0 var(--app-space-1);
  border: 1px solid var(--md-sys-color-outline-variant); border-bottom-width: 2px; border-radius: var(--md-sys-shape-corner-extra-small);
  background: var(--md-sys-color-surface-container-highest); color: var(--md-sys-color-on-surface-variant); font-family: inherit;
}
</style>
