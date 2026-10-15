<script setup lang="ts">
import '@material/web/textfield/outlined-text-field.js';
import type { Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';

withDefaults(
  defineProps<{
    label?: string;
    placeholder?: string;
    type?: 'text' | 'password' | 'number' | 'url' | 'search';
    supportingText?: string;
    errorText?: string;
    disabled?: boolean;
    icon?: Component;
    min?: number;
    max?: number;
    autocomplete?: string;
  }>(),
  { type: 'text' },
);
const model = defineModel<string | number | null>({ default: '' });
const emit = defineEmits<{ enter: []; change: [string] }>();

function onInput(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  model.value = typeof model.value === 'number' && value !== '' ? Number(value) : value;
}
</script>

<template>
  <md-outlined-text-field
    class="text-field"
    :label="label ?? ''"
    :placeholder="placeholder ?? ''"
    :type="type"
    :value.prop="model ?? ''"
    :supporting-text="errorText || supportingText || ''"
    :error.prop="!!errorText"
    :error-text="errorText ?? ''"
    :disabled.prop="disabled"
    :min="min"
    :max="max"
    :autocomplete="autocomplete"
    @input="onInput"
    @change="emit('change', ($event.target as HTMLInputElement).value)"
    @keydown.enter="emit('enter')"
  >
    <AppIcon v-if="icon" slot="leading-icon" :icon="icon" :size="20" />
    <slot name="trailing" />
  </md-outlined-text-field>
</template>

<style scoped>
.text-field {
  width: 100%;
  /* Its own height, even in a stretching grid or row: a neighbour's supporting text must not make
     this field's box taller, so fields side by side keep their boxes aligned. */
  height: fit-content;
  /* An example, not a value: the outline tone keeps it well apart from typed text (on-surface). */
  --md-outlined-text-field-input-text-placeholder-color: var(--md-sys-color-outline);
}
</style>
