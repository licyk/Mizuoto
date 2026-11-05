<script setup lang="ts">
import { computed } from 'vue';
import { RouterLink, useRoute, useRouter } from 'vue-router';
import AppIcon from '@/ui/AppIcon.vue';
import AppMenu from '@/ui/AppMenu.vue';
import { MoreVertical } from '@/ui/icons';
import type { NavItem } from '@/ui/NavigationRail.vue';

/**
 * The bottom bar of compact windows. Material 3 gives it three to five destinations: with more than ``max`` items, the
 * first ``max - 1`` stay on the bar and the rest open from a last destination, ``moreLabel``, which shows as current
 * while one of them is (and with the rail's dot while one of them has a badge). Each destination shrinks with the window, so the
 * bar never makes the page wider than it.
 */
const props = withDefaults(defineProps<{ items: NavItem[]; max?: number; moreLabel?: string }>(), { max: 5, moreLabel: 'More' });
const router = useRouter();
const route = useRoute();
const split = computed(() => props.items.length > props.max);
const shown = computed(() => (split.value ? props.items.slice(0, props.max - 1) : props.items));
const rest = computed(() => (split.value ? props.items.slice(props.max - 1) : []));
/** Whether ``to`` is the current page or one under it, as RouterLink's ``isActive`` decides. */
const current = (to: string) => {
  const target = router.resolve(to).path;
  return route.path === target || route.path.startsWith(`${target.replace(/\/$/, '')}/`);
};
const restActive = computed(() => rest.value.some((item) => current(item.to)));
const restBadge = computed(() => rest.value.some((item) => !!item.badge));
const restItems = computed(() => rest.value.map((item) => ({ id: item.to, label: item.label, icon: item.icon })));
</script>

<template>
  <nav class="bar" aria-label="Main">
    <RouterLink v-for="item in shown" :key="item.label" v-slot="{ isActive, navigate, href }" :to="item.to" custom>
      <a :href="href" class="dest" :class="{ active: isActive }" :aria-current="isActive ? 'page' : undefined" @click="navigate">
        <span class="indicator state-layer">
          <AppIcon :icon="item.icon" :size="24" />
          <span v-if="item.badge" class="badge-dot" />
        </span>
        <span class="label type-label-medium">{{ item.label }}</span>
      </a>
    </RouterLink>
    <AppMenu v-if="rest.length" class="more-root" :items="restItems" @select="router.push($event)">
      <template #default="{ toggle, open }">
        <button type="button" class="dest more" :class="{ active: restActive }" :aria-expanded="open" aria-haspopup="menu" @click="toggle">
          <span class="indicator state-layer">
            <AppIcon :icon="MoreVertical" :size="24" />
            <span v-if="restBadge" class="badge-dot" />
          </span>
          <span class="label type-label-medium">{{ moreLabel }}</span>
        </button>
      </template>
    </AppMenu>
  </nav>
</template>

<style scoped>
.bar { display: flex; justify-content: space-around; height: 80px; padding: var(--app-space-3) 0 var(--app-space-4); background: var(--md-sys-color-surface-container); }
.dest {
  flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; align-items: center; gap: var(--app-space-1); text-decoration: none;
  color: var(--md-sys-color-on-surface-variant);
}
.more { padding: 0; border: 0; background: transparent; font: inherit; cursor: pointer; }
.more-root { flex: 1 1 0; min-width: 0; display: flex; }
.more-root .dest { width: 100%; }
.indicator { position: relative; display: grid; place-items: center; width: min(64px, 100%); height: 32px; border-radius: var(--md-sys-shape-corner-full); }
.label { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge-dot { position: absolute; top: 4px; right: 12px; width: 6px; height: 6px; border-radius: 50%; background: var(--md-sys-color-error); }
.active { color: var(--md-sys-color-on-surface); }
.active .indicator { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
</style>
