<script setup lang="ts">
import { RouterLink } from 'vue-router';
import AppIcon from '@/ui/AppIcon.vue';
import type { NavItem } from '@/ui/NavigationRail.vue';

defineProps<{ items: NavItem[] }>();
</script>

<template>
  <nav class="bar" aria-label="Main">
    <RouterLink v-for="item in items" :key="item.label" v-slot="{ isActive, navigate, href }" :to="item.to" custom>
      <a :href="href" class="dest" :class="{ active: isActive }" :aria-current="isActive ? 'page' : undefined" @click="navigate">
        <span class="indicator state-layer"><AppIcon :icon="item.icon" :size="24" /></span>
        <span class="type-label-medium">{{ item.label }}</span>
      </a>
    </RouterLink>
  </nav>
</template>

<style scoped>
.bar { display: flex; justify-content: space-around; height: 80px; padding: var(--app-space-3) 0 var(--app-space-4); background: var(--md-sys-color-surface-container); }
.dest { flex: 1; display: flex; flex-direction: column; align-items: center; gap: var(--app-space-1); text-decoration: none; color: var(--md-sys-color-on-surface-variant); }
.indicator { display: grid; place-items: center; width: 64px; height: 32px; border-radius: var(--md-sys-shape-corner-full); }
.active { color: var(--md-sys-color-on-surface); }
.active .indicator { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
</style>
