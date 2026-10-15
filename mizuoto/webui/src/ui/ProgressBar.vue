<script setup lang="ts">
import '@material/web/progress/linear-progress.js';

/**
 * ``value`` is 0 to 1; omit it (or null) for an indeterminate bar, Material's own animation.
 *
 * A determinate bar is drawn here. Material's eases in and out over 250 ms and starts slow again
 * on every update, so a value that changes several times a second (the Live load, an upload, a
 * job) moves in jerks; this one eases out, so it chases each new value smoothly. ``tone="error"``
 * marks a value that is too high (the Live load).
 */
withDefaults(defineProps<{ value?: number | null; label?: string; tone?: 'primary' | 'error' }>(), { value: null, label: '', tone: 'primary' });
</script>

<template>
  <md-linear-progress v-if="value === null || value === undefined" class="bar" :indeterminate.prop="true" :aria-label="label" />
  <div v-else class="track" :class="tone" role="progressbar" :aria-label="label" aria-valuemin="0" aria-valuemax="1" :aria-valuenow="value">
    <div class="fill" :style="{ '--value': Math.min(1, Math.max(0, value)) }" />
  </div>
</template>

<style scoped>
.bar { width: 100%; --md-linear-progress-track-shape: var(--md-sys-shape-corner-full); }
/* The same box and colours as md-linear-progress. */
.track { position: relative; width: 100%; height: 4px; border-radius: var(--md-sys-shape-corner-full); background: var(--md-sys-color-surface-container-highest); overflow: hidden; }
.fill {
  position: absolute; inset: 0; border-radius: inherit; background: var(--md-sys-color-primary);
  transform: translateX(calc((var(--value) - 1) * 100%));
  transition: transform var(--md-sys-motion-duration-medium2) var(--md-sys-motion-easing-emphasized-decelerate), background-color var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard);
}
.error .fill { background: var(--md-sys-color-error); }
</style>
