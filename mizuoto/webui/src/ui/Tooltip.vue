<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref } from 'vue';
import { tooltipZ } from '@/ui/layers';

/**
 * A plain tooltip shown on hover and keyboard focus of its content.
 *
 * The tip is rendered at the end of the document and placed against its content, inside the
 * window, so a sticky footer or a pane beside the navigation rail cannot clip or cover it. It
 * shows above its content, or below when there is no room above, and follows it while the page
 * scrolls.
 */
defineProps<{ text: string }>();
const shown = ref(false);
const root = ref<HTMLElement | null>(null);
const tip = ref<HTMLElement | null>(null);
// Hidden until measured, so the first frame never draws it in the wrong place.
const position = ref<Record<string, string>>({ visibility: 'hidden' });
const GAP = 4;
const MARGIN = 8;
let timer: ReturnType<typeof setTimeout> | undefined;

function place() {
  const anchor = root.value?.getBoundingClientRect();
  const box = tip.value?.getBoundingClientRect();
  if (!anchor || !box) return;
  const left = anchor.left + anchor.width / 2 - box.width / 2;
  const above = anchor.top - GAP - box.height;
  position.value = {
    left: `${Math.min(Math.max(MARGIN, left), Math.max(MARGIN, window.innerWidth - box.width - MARGIN))}px`,
    top: `${above >= MARGIN ? above : anchor.bottom + GAP}px`,
  };
}

function listen(on: boolean) {
  // Capture, so scrolling in any container keeps the tip on its content.
  if (on) {
    window.addEventListener('scroll', place, true);
    window.addEventListener('resize', place);
  } else {
    window.removeEventListener('scroll', place, true);
    window.removeEventListener('resize', place);
  }
}

const show = () => {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    position.value = { visibility: 'hidden' };
    shown.value = true;
    listen(true);
    await nextTick();
    place();
  }, 400);
};
const hide = () => {
  clearTimeout(timer);
  shown.value = false;
  listen(false);
};
onBeforeUnmount(hide);
</script>

<template>
  <span ref="root" class="tooltip-root" @pointerenter="show" @pointerleave="hide" @focusin="show" @focusout="hide">
    <slot />
    <Teleport to="body">
      <Transition name="snackbar">
        <span v-if="shown" ref="tip" role="tooltip" class="tooltip type-body-small" :style="{ ...position, zIndex: tooltipZ }">{{ text }}</span>
      </Transition>
    </Teleport>
  </span>
</template>

<style scoped>
.tooltip-root { display: inline-flex; }
.tooltip {
  position: fixed; width: max-content; max-width: min(var(--app-width-text), calc(100vw - 16px)); pointer-events: none;
  padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-extra-small);
  background: var(--md-sys-color-inverse-surface); color: var(--md-sys-color-inverse-on-surface);
}
</style>
