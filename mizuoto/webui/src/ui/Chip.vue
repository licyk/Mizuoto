<script setup lang="ts">
import { ref, type Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { Check, X } from '@/ui/icons';
import { useRipple } from '@/ui/motion/useRipple';

/**
 * A compact chip built from the tokens: a tag, a filter value, a search preset. ``dot`` shows a
 * colour swatch before the label (a tag's colour); ``removable`` adds a close button.
 */
const props = defineProps<{ label: string; icon?: Component; selected?: boolean; removable?: boolean; dot?: string | null; count?: number | null; disabled?: boolean; title?: string }>();
const emit = defineEmits<{ click: [MouseEvent]; remove: [] }>();
const el = ref<HTMLElement | null>(null);
useRipple(el);
</script>

<template>
  <span ref="el" class="chip state-layer type-label-large" :class="{ selected, disabled }" :title="title ?? label" role="button" :tabindex="disabled ? -1 : 0" :aria-pressed="selected" @click="!disabled && emit('click', $event)" @keydown.enter="!disabled && emit('click', $event as unknown as MouseEvent)">
    <AppIcon v-if="selected && !icon" :icon="Check" :size="18" />
    <AppIcon v-else-if="icon" :icon="icon" :size="18" />
    <span v-if="dot" class="dot" :style="{ background: props.dot ?? undefined }" />
    <span class="label">{{ label }}</span>
    <span v-if="count !== null && count !== undefined" class="count">{{ count }}</span>
    <button v-if="removable" type="button" class="remove" :aria-label="`Remove ${label}`" @click.stop="emit('remove')"><AppIcon :icon="X" :size="18" /></button>
  </span>
</template>

<style scoped>
.chip {
  display: inline-flex; align-items: center; gap: var(--app-space-2); height: 32px; max-width: 100%; padding: 0 var(--app-space-3);
  border: 1px solid var(--md-sys-color-outline-variant); border-radius: var(--md-sys-shape-corner-small);
  color: var(--md-sys-color-on-surface-variant); background: transparent; cursor: pointer; user-select: none; white-space: nowrap;
}
.chip.selected { border-color: transparent; background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.chip.disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.label { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.count { color: var(--md-sys-color-on-surface-variant); font-weight: 400; }
.remove { display: grid; place-items: center; margin-right: calc(-1 * var(--app-space-2)); width: 24px; height: 24px; border: 0; padding: 0; border-radius: 50%; background: transparent; color: inherit; cursor: pointer; }
</style>
