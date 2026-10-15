<script setup lang="ts">
import { type Component, ref, useId } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { ChevronDown } from '@/ui/icons';
import { collapseHooks } from '@/ui/motion/transitions';
import { useRipple } from '@/ui/motion/useRipple';

/**
 * A labelled section that opens and closes on click, with the collapse transition and a
 * chevron that turns. The open state is a model, so a view can drive it or leave it alone.
 */
defineProps<{ label: string; icon?: Component; supportingText?: string; disabled?: boolean }>();
const open = defineModel<boolean>('open', { default: false });

const header = ref<HTMLElement | null>(null);
useRipple(header);
const id = useId();
</script>

<template>
  <div class="panel">
    <button
      :id="`${id}-header`"
      ref="header"
      type="button"
      class="header state-layer"
      :aria-expanded="open"
      :aria-controls="id"
      :disabled="disabled"
      @click="open = !open"
    >
      <AppIcon v-if="icon" :icon="icon" :size="20" />
      <span class="labels">
        <span class="type-title-small label">{{ label }}</span>
        <span v-if="supportingText" class="type-body-small muted">{{ supportingText }}</span>
      </span>
      <slot name="trailing" />
      <AppIcon :icon="ChevronDown" :size="20" class="chevron" :class="{ open }" />
    </button>
    <Transition name="collapse" v-bind="collapseHooks">
      <div v-if="open" :id="id" role="region" :aria-labelledby="`${id}-header`">
        <div class="content"><slot /></div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.panel { display: flex; flex-direction: column; }
.header {
  display: flex; align-items: center; gap: var(--app-space-2); width: 100%; padding: var(--app-space-2) var(--app-space-3);
  border: 0; border-radius: var(--md-sys-shape-corner-medium); background: transparent; color: var(--md-sys-color-on-surface);
  font: inherit; text-align: left; cursor: pointer;
}
.header:disabled { cursor: default; opacity: 0.38; }
.labels { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chevron { color: var(--md-sys-color-on-surface-variant); transition: transform var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard); }
.chevron.open { transform: rotate(180deg); }
/* The padding lives on an inner element: the animated height must be the content's own. */
.content { padding-top: var(--app-space-2); }
</style>
