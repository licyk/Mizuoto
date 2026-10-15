<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <header class="top-bar">
    <slot name="leading" />
    <h1 class="type-title-large title">{{ title }}</h1>
    <div class="actions"><slot name="actions" /></div>
  </header>
</template>

<style scoped>
.top-bar { display: flex; align-items: center; gap: var(--app-space-1); height: 64px; padding: 0 var(--app-space-2) 0 var(--app-space-4); background: var(--md-sys-color-surface); }
.title { flex: 1; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.actions { display: flex; align-items: center; gap: var(--app-space-1); }
</style>
