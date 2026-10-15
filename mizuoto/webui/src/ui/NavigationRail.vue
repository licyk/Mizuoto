<script setup lang="ts">
import type { Component } from 'vue';
import { RouterLink } from 'vue-router';
import AppIcon from '@/ui/AppIcon.vue';

export interface NavItem {
  to: string;
  label: string;
  icon: Component;
  badge?: number;
}
/** ``items`` are the destinations; ``footer`` (Settings) stays in the rail's bottom corner. */
withDefaults(defineProps<{ items: NavItem[]; footer?: NavItem[] }>(), { footer: () => [] });
</script>

<template>
  <nav class="rail" aria-label="Main">
    <div class="top"><slot name="top" /></div>
    <!-- Only the destinations scroll when the window is too short for them all (a phone held sideways):
         the footer never leaves the bottom corner. -->
    <div class="dests">
      <RouterLink v-for="item in items" :key="item.label" v-slot="{ isActive, navigate, href }" :to="item.to" custom>
        <a :href="href" class="dest" :class="{ active: isActive }" :aria-current="isActive ? 'page' : undefined" @click="navigate">
          <span class="indicator state-layer"><AppIcon :icon="item.icon" :size="24" /><span v-if="item.badge" class="badge-dot" /></span>
          <span class="type-label-medium">{{ item.label }}</span>
        </a>
      </RouterLink>
    </div>
    <div v-if="footer.length" class="foot">
      <RouterLink v-for="item in footer" :key="item.label" v-slot="{ isActive, navigate, href }" :to="item.to" custom>
        <a :href="href" class="dest" :class="{ active: isActive }" :aria-current="isActive ? 'page' : undefined" @click="navigate">
          <span class="indicator state-layer"><AppIcon :icon="item.icon" :size="24" /></span>
          <span class="type-label-medium">{{ item.label }}</span>
        </a>
      </RouterLink>
    </div>
  </nav>
</template>

<style scoped>
.rail { display: flex; flex-direction: column; align-items: center; width: 88px; height: 100%; min-height: 0; padding: var(--app-space-3) 0; background: var(--md-sys-color-surface); }
.dests, .foot { display: flex; flex-direction: column; align-items: center; gap: var(--app-space-3); width: 100%; }
/* The space between takes the slack and scrolls when there is none; 2px keeps the focus ring inside it. */
.dests { flex: 1; min-height: 0; padding: 2px 0; overflow: hidden auto; scrollbar-width: none; }
.dests::-webkit-scrollbar { display: none; }
.foot { flex-shrink: 0; padding-top: var(--app-space-3); }
.badge-dot { position: absolute; top: 4px; right: 12px; width: 6px; height: 6px; border-radius: 50%; background: var(--md-sys-color-error); }
.indicator { position: relative; }
.top { flex-shrink: 0; min-height: 56px; display: grid; place-items: center; margin-bottom: var(--app-space-5); }
.dest { display: flex; flex-direction: column; align-items: center; gap: var(--app-space-1); width: 80px; text-decoration: none; color: var(--md-sys-color-on-surface-variant); }
.indicator {
  display: grid; place-items: center; width: 56px; height: 32px; border-radius: var(--md-sys-shape-corner-full);
  transition: background-color var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard);
}
.active { color: var(--md-sys-color-on-surface); }
.active .indicator { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.dest:focus-visible { outline: none; }
.dest:focus-visible .indicator { outline: 2px solid var(--md-sys-color-secondary); }
</style>
