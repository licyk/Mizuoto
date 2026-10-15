<script setup lang="ts">
import '@material/web/slider/slider.js';
import { computed } from 'vue';
import IconButton from '@/ui/IconButton.vue';
import { RotateCcw } from '@/ui/icons';

/**
 * One parameter: label, help, a slider with the range the server validates, the value, and a
 * reset to its default. ``input`` fires while dragging (Live applies it), ``update:modelValue`` too;
 * ``commit`` fires when the drag ends. A disabled slider says why.
 */
const props = withDefaults(
  defineProps<{ label: string; min: number; max: number; step?: number; help?: string; unit?: string; defaultValue?: number | null; disabled?: boolean; disabledReason?: string; resetLabel?: string; format?: (v: number) => string }>(),
  { step: 1, help: '', unit: '', defaultValue: null, disabledReason: '', resetLabel: 'Reset' },
);
const model = defineModel<number>({ required: true });
const emit = defineEmits<{ commit: [number] }>();
const shown = computed(() => (props.format ? props.format(model.value) : `${Number(model.value.toFixed(3))}${props.unit ? ` ${props.unit}` : ''}`));
const changed = computed(() => props.defaultValue !== null && model.value !== props.defaultValue);
const value = (e: Event) => Number((e.target as HTMLInputElement).value);
function reset() {
  if (props.defaultValue === null) return;
  model.value = props.defaultValue;
  emit('commit', props.defaultValue);
}
</script>

<template>
  <div class="param" :class="{ disabled }">
    <div class="head">
      <span class="type-body-large label">{{ label }}</span>
      <span class="type-label-large value">{{ shown }}</span>
      <IconButton v-if="changed && !disabled" :icon="RotateCcw" :label="resetLabel" class="reset" @click="reset" />
    </div>
    <md-slider
      labeled
      :min="min"
      :max="max"
      :step="step"
      :value.prop="model"
      :disabled.prop="disabled"
      :aria-label="label"
      @input="model = value($event)"
      @change="emit('commit', value($event))"
    />
    <span v-if="disabled && disabledReason" class="type-body-small note">{{ disabledReason }}</span>
    <span v-else-if="help" class="type-body-small muted help">{{ help }}</span>
  </div>
</template>

<style scoped>
.param { display: flex; flex-direction: column; gap: 0; min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-height: 40px; }
.label { flex: 1; min-width: 0; }
.value { font-variant-numeric: tabular-nums; color: var(--md-sys-color-primary); }
.disabled .value { color: var(--md-sys-color-on-surface-variant); }
md-slider { width: 100%; margin-inline: -8px; width: calc(100% + 16px); }
.help, .note { padding-inline: var(--app-space-1); }
.note { color: var(--md-sys-color-on-surface-variant); font-style: italic; }
</style>
