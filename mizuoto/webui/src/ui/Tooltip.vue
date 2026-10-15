<script setup lang="ts">
import { ref } from 'vue';

/** A plain tooltip shown on hover and keyboard focus of its content. */
defineProps<{ text: string }>();
const shown = ref(false);
let timer: ReturnType<typeof setTimeout> | undefined;
const show = () => {
  clearTimeout(timer);
  timer = setTimeout(() => (shown.value = true), 400);
};
const hide = () => {
  clearTimeout(timer);
  shown.value = false;
};
</script>

<template>
  <span class="tooltip-root" @pointerenter="show" @pointerleave="hide" @focusin="show" @focusout="hide">
    <slot />
    <Transition name="snackbar">
      <span v-if="shown" role="tooltip" class="tooltip type-body-small">{{ text }}</span>
    </Transition>
  </span>
</template>

<style scoped>
.tooltip-root { position: relative; display: inline-flex; }
.tooltip {
  position: absolute; bottom: calc(100% + 4px); left: 50%; translate: -50% 0; z-index: var(--app-z-tooltip); white-space: nowrap; pointer-events: none;
  padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-extra-small);
  background: var(--md-sys-color-inverse-surface); color: var(--md-sys-color-inverse-on-surface);
}
</style>
