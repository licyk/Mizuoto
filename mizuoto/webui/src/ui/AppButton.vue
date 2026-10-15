<script setup lang="ts">
import '@material/web/button/filled-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';
import { computed, type Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';

/** One emphasis hierarchy: filled for the primary action of a region, tonal for secondary, outlined or text for the rest. */
const props = withDefaults(
  defineProps<{ variant?: 'filled' | 'tonal' | 'outlined' | 'text'; icon?: Component; disabled?: boolean; loading?: boolean; type?: 'button' | 'submit' }>(),
  { variant: 'filled', type: 'button' },
);
defineEmits<{ click: [MouseEvent] }>();

const tag = computed(() => ({ filled: 'md-filled-button', tonal: 'md-filled-tonal-button', outlined: 'md-outlined-button', text: 'md-text-button' })[props.variant]);
</script>

<template>
  <component :is="tag" :type="type" :disabled.prop="disabled || loading" :has-icon.prop="!!icon || loading" @click="$emit('click', $event)">
    <md-circular-progress v-if="loading" slot="icon" indeterminate class="spinner" />
    <AppIcon v-else-if="icon" slot="icon" :icon="icon" :size="18" />
    <slot />
  </component>
</template>

<style scoped>
.spinner { --md-circular-progress-size: 18px; --md-circular-progress-active-indicator-color: currentColor; }
</style>
