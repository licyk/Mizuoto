<script setup lang="ts">
import { computed, ref } from 'vue';
import { useWindowClass } from '@/theme/breakpoints';
import { useLayerBase } from '@/ui/layers';
import NavigationBar from '@/ui/NavigationBar.vue';
import NavigationRail, { type NavItem } from '@/ui/NavigationRail.vue';
import Snackbar from '@/ui/Snackbar.vue';
import TopAppBar from '@/ui/TopAppBar.vue';

/** Navigation switches between a bottom bar (compact) and a rail (medium and wider). */
withDefaults(defineProps<{ items: NavItem[]; footer?: NavItem[]; title: string }>(), { footer: () => [] });
const windowClass = useWindowClass();
const compact = computed(() => windowClass.value === 'compact');
const content = ref<HTMLElement | null>(null);
const shell = ref<HTMLElement | null>(null);
// The page under every layer: inert while a dialog or sheet is open.
useLayerBase(shell);
defineExpose({ content });
</script>

<template>
  <div ref="shell" class="shell" :class="{ compact }">
    <NavigationRail v-if="!compact" :items="items" :footer="footer" class="rail"><template #top><slot name="rail-top" /></template></NavigationRail>
    <div class="main-column">
      <TopAppBar :title="title"><template #actions><slot name="actions" /></template></TopAppBar>
      <!-- The frame (background, corners) belongs to the shell and never animates; the scroller inside
           it is the page, the only thing a navigation fades through. -->
      <div class="frame"><main ref="content" class="content"><slot /></main></div>
    </div>
    <NavigationBar v-if="compact" :items="[...items, ...footer]" class="bottom" />
    <Snackbar />
  </div>
</template>

<style scoped>
/* One row the window's height: a rail taller than the window scrolls inside it rather than growing it. */
.shell { display: grid; grid-template-columns: auto 1fr; grid-template-rows: minmax(0, 1fr); height: 100%; background: var(--md-sys-color-surface); }
.shell.compact { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }
.main-column { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.frame {
  display: flex; flex-direction: column; flex: 1; min-height: 0; margin: 0 var(--app-space-4) var(--app-space-4) 0;
  background: var(--md-sys-color-surface-container-low); border-radius: var(--md-sys-shape-corner-large); overflow: hidden;
}
.content {
  position: relative; flex: 1; min-height: 0; overflow: auto;
  /* Reserve the scrollbar's width so content does not jump when it appears. */
  scrollbar-gutter: stable;
  /* Pages choose their columns by the space they have here (@container app-content), not by the
     window: the rail, a side sheet or the window size all change it. */
  container: app-content / inline-size;
  /* The page alone fades through on navigation; the rail, the top bar and the frame stay put. */
  view-transition-name: app-content;
}
.compact .frame { margin: 0; border-radius: 0; }
</style>
