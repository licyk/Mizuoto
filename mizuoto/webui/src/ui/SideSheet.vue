<script setup lang="ts">
import { ref } from 'vue';
import IconButton from '@/ui/IconButton.vue';
import { X } from '@/ui/icons';
import { useLayer } from '@/ui/layers';

/** A modal side sheet that slides in from the right edge (the ``sheet`` transition). */
withDefaults(defineProps<{ title: string; closeLabel?: string }>(), { closeLabel: 'Close' });
const open = defineModel<boolean>('open', { default: false });
const sheet = ref<HTMLElement | null>(null);
const { zIndex } = useLayer({ kind: 'sheet', modal: true, open: () => open.value, close: () => (open.value = false), elements: () => [sheet.value], initialFocus: () => sheet.value });
</script>

<template>
  <Teleport to="body">
    <Transition name="scrim">
      <div v-if="open" class="scrim" :style="{ zIndex }" />
    </Transition>
    <Transition name="sheet">
      <aside v-if="open" ref="sheet" class="sheet" role="dialog" aria-modal="true" :aria-label="title" tabindex="-1" :style="{ zIndex }">
        <header class="head">
          <h2 class="type-title-large title">{{ title }}</h2>
          <slot name="header-actions" />
          <IconButton :icon="X" :label="closeLabel" @click="open = false" />
        </header>
        <div class="body"><slot /></div>
      </aside>
    </Transition>
  </Teleport>
</template>

<style scoped>
.scrim { position: fixed; inset: 0; background: color-mix(in srgb, var(--md-sys-color-scrim) 32%, transparent); }
.sheet {
  position: fixed; top: 0; right: 0; bottom: 0; width: min(var(--app-width-sheet), 100vw); display: flex; flex-direction: column; outline: none;
  background: var(--md-sys-color-surface-container-low); color: var(--md-sys-color-on-surface);
  border-radius: var(--md-sys-shape-corner-large) 0 0 var(--md-sys-shape-corner-large); box-shadow: var(--app-elevation-2);
}
.head { display: flex; align-items: center; gap: var(--app-space-1); padding: var(--app-space-3) var(--app-space-2) var(--app-space-2) var(--app-space-6); }
.title { flex: 1; margin: 0; }
.body { flex: 1; overflow: auto; padding: 0 var(--app-space-4) var(--app-space-4); }
</style>
