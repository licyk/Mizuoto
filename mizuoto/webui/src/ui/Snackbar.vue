<script setup lang="ts">
import { computed, watch } from 'vue';
import IconButton from '@/ui/IconButton.vue';
import { X } from '@/ui/icons';
import { snackbarZ } from '@/ui/layers';
import { useSnackbar } from '@/ui/useSnackbar';

/**
 * Renders the head of the snackbar queue. Mount once, in the app shell. It sits beside the page,
 * not in it, so it stays live while a modal makes the page inert, and a press on it closes no layer.
 */
const { state, dismiss } = useSnackbar();
const current = computed(() => state.queue[0]);
let timer: ReturnType<typeof setTimeout> | undefined;

watch(
  current,
  (msg) => {
    clearTimeout(timer);
    if (msg) timer = setTimeout(() => dismiss(msg.id), msg.timeout);
  },
  { immediate: true },
);

function act() {
  const msg = current.value;
  if (!msg) return;
  msg.action?.();
  dismiss(msg.id);
}
</script>

<template>
  <Teleport to="body">
    <div class="snackbar-host" aria-live="polite" data-layer-ignore :style="{ zIndex: snackbarZ }">
      <Transition name="snackbar" mode="out-in">
        <div v-if="current" :key="current.id" class="snackbar" :class="{ error: current.error }" role="status">
          <span class="type-body-medium text">{{ current.text }}</span>
          <button v-if="current.actionLabel" type="button" class="action type-label-large state-layer" @click="act">{{ current.actionLabel }}</button>
          <IconButton :icon="X" label="Dismiss" class="close" @click="dismiss(current.id)" />
        </div>
      </Transition>
    </div>
  </Teleport>
</template>

<style scoped>
.snackbar-host { position: fixed; left: 50%; bottom: var(--app-space-4); translate: -50% 0; width: min(var(--app-width-snackbar), calc(100vw - 32px)); pointer-events: none; }
.snackbar {
  display: flex; align-items: center; gap: var(--app-space-2); min-height: 48px; padding: var(--app-space-1) var(--app-space-1) var(--app-space-1) var(--app-space-4);
  border-radius: var(--md-sys-shape-corner-extra-small); background: var(--md-sys-color-inverse-surface); color: var(--md-sys-color-inverse-on-surface);
  box-shadow: var(--app-elevation-3); pointer-events: auto;
}
.snackbar.error { background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); }
/* A message may name a file: let the name wrap instead of pushing the action and the close
   button out of the bar. */
.text { flex: 1; min-width: 0; padding: var(--app-space-2) 0; overflow-wrap: anywhere; }
.action { flex: none; border: 0; background: transparent; color: var(--md-sys-color-inverse-primary); padding: 0 var(--app-space-3); height: 40px; border-radius: var(--md-sys-shape-corner-full); cursor: pointer; font: inherit; font-weight: 500; }
.snackbar.error .action { color: var(--md-sys-color-on-error-container); }
.close { --md-icon-button-icon-color: currentColor; color: inherit; }
</style>
