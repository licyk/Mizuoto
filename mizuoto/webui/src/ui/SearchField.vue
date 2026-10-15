<script setup lang="ts">
import { ref } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { Search, X } from '@/ui/icons';

/** A pill-shaped search bar. Emits ``search`` on Enter; v-model follows every keystroke. */
defineProps<{ placeholder?: string; label?: string }>();
const model = defineModel<string>({ default: '' });
const emit = defineEmits<{ search: [string] }>();
const input = ref<HTMLInputElement | null>(null);

function clear() {
  model.value = '';
  emit('search', '');
  input.value?.focus();
}
defineExpose({ focus: () => input.value?.focus() });
</script>

<template>
  <label class="search state-layer">
    <AppIcon :icon="Search" :size="24" class="lead" />
    <input
      ref="input"
      v-model="model"
      type="search"
      class="type-body-large"
      :placeholder="placeholder"
      :aria-label="label ?? placeholder"
      @keydown.enter="emit('search', model)"
    />
    <button v-if="model" type="button" class="clear" :aria-label="'Clear'" @click="clear">
      <AppIcon :icon="X" :size="20" />
    </button>
  </label>
</template>

<style scoped>
.search {
  display: flex; align-items: center; gap: var(--app-space-2); height: 56px; padding: 0 var(--app-space-2) 0 var(--app-space-4);
  border-radius: var(--md-sys-shape-corner-full); background: var(--md-sys-color-surface-container-high); color: var(--md-sys-color-on-surface-variant);
  min-width: 0; cursor: text;
}
.search:focus-within { outline: 2px solid var(--md-sys-color-primary); outline-offset: -2px; }
input {
  flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; color: var(--md-sys-color-on-surface);
  font: inherit; font-size: var(--md-sys-typescale-body-large-size);
}
input::-webkit-search-cancel-button { display: none; }
/* An example, not a value: lighter than typed text, as in TextField. */
input::placeholder { color: var(--md-sys-color-outline); opacity: 1; }
.clear {
  display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border: 0; border-radius: 50%;
  background: transparent; color: inherit; cursor: pointer;
}
.clear:hover { background: color-mix(in srgb, currentColor calc(var(--md-sys-state-hover-state-layer-opacity) * 100%), transparent); }
</style>
