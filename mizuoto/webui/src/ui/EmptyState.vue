<script setup lang="ts">
import type { Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';

defineProps<{ icon: Component; title: string; text?: string }>();
</script>

<template>
  <div class="empty">
    <span class="icon-wrap"><AppIcon :icon="icon" :size="24" /></span>
    <h3 class="type-title-large title">{{ title }}</h3>
    <p v-if="text" class="type-body-medium muted text">{{ text }}</p>
    <div v-if="$slots.default" class="actions"><slot /></div>
  </div>
</template>

<style scoped>
/* Takes the space its container leaves (in a row or a column) and centres itself in it both ways. */
.empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: var(--app-space-2);
  flex: 1 1 auto; align-self: stretch; min-width: 0; box-sizing: border-box; padding: var(--app-space-8) var(--app-space-4);
}
.icon-wrap {
  display: grid; place-items: center; width: 64px; height: 64px; border-radius: var(--md-sys-shape-corner-large);
  background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); margin-bottom: var(--app-space-2);
}
.title, .text { margin: 0; max-width: var(--app-width-text); }
.actions { display: flex; gap: var(--app-space-2); margin-top: var(--app-space-3); flex-wrap: wrap; justify-content: center; }
</style>
