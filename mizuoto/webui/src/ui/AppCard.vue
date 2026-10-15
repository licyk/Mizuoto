<script setup lang="ts">
import { ref } from 'vue';
import { useRipple } from '@/ui/motion/useRipple';

/** An elevated card built from the tokens. When ``interactive``, it has a state layer and a ripple. */
const props = withDefaults(defineProps<{ interactive?: boolean; selected?: boolean; variant?: 'elevated' | 'filled' | 'outlined' }>(), { variant: 'filled' });
defineEmits<{ activate: [MouseEvent | KeyboardEvent] }>();
const el = ref<HTMLElement | null>(null);
if (props.interactive) useRipple(el);
defineExpose({ el });
</script>

<template>
  <article
    ref="el"
    class="card"
    :class="[variant, { interactive, selected, 'state-layer': interactive }]"
    :tabindex="interactive ? 0 : undefined"
    @click="interactive && $emit('activate', $event)"
    @keydown.enter.self="interactive && $emit('activate', $event)"
  >
    <slot />
  </article>
</template>

<style scoped>
.card {
  position: relative; display: flex; flex-direction: column; overflow: hidden; border-radius: var(--md-sys-shape-corner-medium);
  color: var(--md-sys-color-on-surface); transition: box-shadow var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard);
}
.filled { background: var(--md-sys-color-surface-container-highest); }
.elevated { background: var(--md-sys-color-surface-container-low); box-shadow: var(--app-elevation-1); }
.outlined { background: var(--md-sys-color-surface); border: 1px solid var(--md-sys-color-outline-variant); }
.interactive { cursor: pointer; }
.interactive:hover { box-shadow: var(--app-elevation-2); }
.selected { outline: 3px solid var(--md-sys-color-primary); outline-offset: -3px; }
</style>
