<script setup lang="ts" generic="T extends string">
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';
import { computed, nextTick, onMounted, ref, watch } from 'vue';

const props = defineProps<{ tabs: { value: T; label: string }[] }>();
const model = defineModel<T>({ required: true });
const index = computed(() => Math.max(0, props.tabs.findIndex((t) => t.value === model.value)));
const el = ref<(HTMLElement & { activeTabIndex: number }) | null>(null);

function onChange(event: Event) {
  const i = (event.target as HTMLElement & { activeTabIndex: number }).activeTabIndex;
  const tab = props.tabs[i];
  if (tab) model.value = tab.value;
}

// md-tabs picks its active tab again when its children upgrade, after the property was set; set it
// once more when they are in place, so a remembered tab is the one shown as selected.
async function sync() {
  await nextTick();
  requestAnimationFrame(() => {
    if (el.value && el.value.activeTabIndex !== index.value) el.value.activeTabIndex = index.value;
  });
}
onMounted(sync);
watch(index, sync);
</script>

<template>
  <md-tabs ref="el" :active-tab-index.prop="index" @change="onChange">
    <md-primary-tab v-for="t in tabs" :key="t.value">{{ t.label }}</md-primary-tab>
  </md-tabs>
</template>

<style scoped>
/* A rounded bar, as wide as its row, on the same container tone as the app's other controls; its
   divider is dropped (the rounded box is the edge) and the indicator is a rounded pill. */
md-tabs {
  border-radius: var(--md-sys-shape-corner-large);
  overflow: hidden;
  background: var(--md-sys-color-surface-container);
  --md-divider-color: transparent;
  --md-primary-tab-container-color: transparent;
  --md-primary-tab-container-shape: var(--md-sys-shape-corner-large);
  --md-primary-tab-active-indicator-shape: var(--md-sys-shape-corner-full);
  --md-primary-tab-active-indicator-height: 3px;
}
/* md-primary-tab colours its label on the host, so the change between tabs can be eased here;
   the indicator's own slide is md-tabs'. */
md-primary-tab { transition: color var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard); }
</style>
