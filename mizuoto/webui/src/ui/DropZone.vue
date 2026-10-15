<script setup lang="ts">
import { ref } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { Upload } from '@/ui/icons';

/** A drop target that also opens the file picker on click or Enter. */
const props = withDefaults(defineProps<{ label: string; hint?: string; accept?: string; multiple?: boolean; disabled?: boolean; compact?: boolean }>(), { hint: '', accept: '', multiple: true });
const emit = defineEmits<{ files: [File[]] }>();
const over = ref(false);
const input = ref<HTMLInputElement | null>(null);
let depth = 0;

function enter(e: DragEvent) {
  if (props.disabled || !e.dataTransfer?.types.includes('Files')) return;
  depth++;
  over.value = true;
}
function leave() {
  depth = Math.max(0, depth - 1);
  if (!depth) over.value = false;
}
function drop(e: DragEvent) {
  depth = 0;
  over.value = false;
  if (props.disabled) return;
  const files = [...(e.dataTransfer?.files ?? [])];
  if (files.length) emit('files', props.multiple ? files : files.slice(0, 1));
}
function picked(e: Event) {
  const el = e.target as HTMLInputElement;
  const files = [...(el.files ?? [])];
  el.value = '';
  if (files.length) emit('files', files);
}
const open = () => !props.disabled && input.value?.click();
</script>

<template>
  <div
    class="drop state-layer"
    :class="{ over, compact, disabled }"
    role="button"
    tabindex="0"
    :aria-label="label"
    :data-dragging="over"
    @click="open"
    @keydown.enter.prevent="open"
    @keydown.space.prevent="open"
    @dragenter.prevent="enter"
    @dragover.prevent
    @dragleave="leave"
    @drop.prevent="drop"
  >
    <AppIcon :icon="Upload" :size="compact ? 20 : 24" />
    <span class="text">
      <span class="type-title-small">{{ label }}</span>
      <span v-if="hint" class="type-body-small muted">{{ hint }}</span>
    </span>
    <slot />
    <input ref="input" type="file" class="visually-hidden" tabindex="-1" :accept="accept" :multiple="multiple" @change="picked" />
  </div>
</template>

<style scoped>
.drop {
  display: flex; align-items: center; gap: var(--app-space-3); padding: var(--app-space-5); min-width: 0;
  border: 1px dashed var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-medium);
  color: var(--md-sys-color-on-surface-variant); cursor: pointer;
}
.drop.compact { padding: var(--app-space-3) var(--app-space-4); }
.drop.over { border-style: solid; border-color: var(--md-sys-color-primary); color: var(--md-sys-color-primary); background: color-mix(in srgb, var(--md-sys-color-primary) 8%, transparent); }
.drop.disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.text { display: flex; flex-direction: column; min-width: 0; }
</style>
