<script setup lang="ts">
/** A placeholder in the shape of the final layout. */
withDefaults(defineProps<{ width?: string; height?: string; shape?: 'small' | 'medium' | 'full' }>(), { width: '100%', height: '16px', shape: 'small' });
</script>

<template>
  <span class="skeleton" :class="shape" :style="{ width, height }" aria-hidden="true" />
</template>

<style scoped>
.skeleton { display: block; background: var(--md-sys-color-surface-container-highest); animation: skeleton-pulse 1.6s var(--md-sys-motion-easing-standard) infinite; }
.small { border-radius: var(--md-sys-shape-corner-small); }
.medium { border-radius: var(--md-sys-shape-corner-medium); }
.full { border-radius: var(--md-sys-shape-corner-full); }
</style>
