<script setup lang="ts">
import '@material/web/textfield/outlined-text-field.js';
import { computed, ref } from 'vue';

/** A multi-line outlined field with a character count. ``insert`` puts text at the cursor. */
const props = withDefaults(
  defineProps<{
    label?: string;
    placeholder?: string;
    rows?: number;
    maxLength?: number;
    supportingText?: string;
    errorText?: string;
    disabled?: boolean;
    /** Show the character count (off for compact rows of fields). */
    showCount?: boolean;
  }>(),
  { rows: 4, maxLength: 0, showCount: true },
);
const model = defineModel<string>({ default: '' });
const emit = defineEmits<{ submit: [] }>();
const field = ref<(HTMLElement & { selectionStart: number; selectionEnd: number; setSelectionRange: (a: number, b: number) => void; focus: () => void }) | null>(null);

const counter = computed(() => (props.maxLength ? `${model.value.length} / ${props.maxLength}` : `${model.value.length}`));

function onInput(event: Event) {
  model.value = (event.target as HTMLTextAreaElement).value;
}
function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    emit('submit');
  }
}

/** Insert ``text`` at the cursor (or replace the selection) and keep the cursor after it. */
function insert(text: string) {
  const el = field.value;
  const value = model.value;
  const start = el ? el.selectionStart : value.length;
  const end = el ? el.selectionEnd : value.length;
  model.value = value.slice(0, start) + text + value.slice(end);
  requestAnimationFrame(() => {
    el?.focus();
    el?.setSelectionRange(start + text.length, start + text.length);
  });
}
defineExpose({ insert });
</script>

<template>
  <div class="text-area">
    <md-outlined-text-field
      ref="field"
      class="field"
      type="textarea"
      :label="label ?? ''"
      :placeholder="placeholder ?? ''"
      :rows="rows"
      :value.prop="model"
      :maxlength="maxLength || undefined"
      :supporting-text="errorText || supportingText || ''"
      :error.prop="!!errorText"
      :error-text="errorText ?? ''"
      :disabled.prop="disabled"
      @input="onInput"
      @keydown="onKeydown"
    />
    <!-- With maxLength the field shows its own "n / max" counter. -->
    <span v-if="!maxLength && showCount" class="type-body-small counter" aria-hidden="true">{{ counter }}</span>
  </div>
</template>

<style scoped>
.text-area { display: flex; flex-direction: column; gap: var(--app-space-1); min-width: 0; }
.field { width: 100%; resize: vertical; }
.counter { align-self: flex-end; color: var(--md-sys-color-on-surface-variant); font-variant-numeric: tabular-nums; }
</style>
