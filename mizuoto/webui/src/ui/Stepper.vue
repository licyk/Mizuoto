<script setup lang="ts" generic="T extends string">
import { Check, CircleX, Loader2, AlertTriangle } from '@/ui/icons';
import AppIcon from '@/ui/AppIcon.vue';

export interface StepItem<V extends string = string> {
  id: V;
  label: string;
  state?: 'pending' | 'running' | 'done' | 'stale' | 'failed' | 'skipped';
  note?: string;
}

/** A horizontal stepper: each step shows its state and can be opened. */
defineProps<{ steps: StepItem<T>[] }>();
const model = defineModel<T>({ required: true });
</script>

<template>
  <ol class="stepper" role="tablist">
    <li v-for="(step, i) in steps" :key="step.id" class="item">
      <button type="button" role="tab" class="step state-layer" :class="[step.state ?? 'pending', { current: model === step.id }]" :aria-selected="model === step.id" @click="model = step.id">
        <span class="dot">
          <AppIcon v-if="step.state === 'done'" :icon="Check" :size="18" />
          <AppIcon v-else-if="step.state === 'running'" :icon="Loader2" :size="18" spin />
          <AppIcon v-else-if="step.state === 'failed'" :icon="CircleX" :size="18" />
          <AppIcon v-else-if="step.state === 'stale'" :icon="AlertTriangle" :size="18" />
          <span v-else class="type-label-medium">{{ i + 1 }}</span>
        </span>
        <span class="text">
          <span class="type-label-large name">{{ step.label }}</span>
          <span v-if="step.note" class="type-body-small muted note">{{ step.note }}</span>
        </span>
      </button>
      <span v-if="i < steps.length - 1" class="line" aria-hidden="true" />
    </li>
  </ol>
</template>

<style scoped>
.stepper { display: flex; align-items: center; gap: 0; margin: 0; padding: 0; list-style: none; overflow-x: auto; }
.item { display: flex; align-items: center; flex: 1 1 0; min-width: 0; }
.item:last-child { flex: 0 1 auto; }
.step { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-2) var(--app-space-3); border: 0; background: transparent; color: var(--md-sys-color-on-surface-variant); border-radius: var(--md-sys-shape-corner-full); cursor: pointer; font: inherit; min-width: 0; }
.step.current { color: var(--md-sys-color-on-surface); background: var(--md-sys-color-secondary-container); }
.dot { flex: none; display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; background: var(--md-sys-color-surface-container-highest); color: var(--md-sys-color-on-surface-variant); }
.done .dot { background: var(--md-sys-color-primary); color: var(--md-sys-color-on-primary); }
.running .dot { background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
.failed .dot { background: var(--md-sys-color-error); color: var(--md-sys-color-on-error); }
.stale .dot { background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
.text { display: flex; flex-direction: column; align-items: flex-start; min-width: 0; }
.name, .note { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.line { flex: 1; min-width: 12px; height: 1px; background: var(--md-sys-color-outline-variant); }
</style>
