<script setup lang="ts">
import { computed } from 'vue';
import { useFalling } from '@/ui/motion/transitions';

/**
 * A level meter in dBFS: the bar is the RMS level, the tick the peak; the colour warns near
 * clipping. Between the stats (10 per second) the bar moves as audio meters do: it rises at once
 * and falls back smoothly, and the peak tick falls slowest.
 */
const props = withDefaults(defineProps<{ rmsDb?: number | null; peakDb?: number | null; label: string; floor?: number; showValue?: boolean }>(), { rmsDb: null, peakDb: null, floor: -60, showValue: true });
const fraction = (db: number | null) => (db === null ? 0 : Math.min(1, Math.max(0, (db - props.floor) / -props.floor)));
const level = computed(() => fraction(props.rmsDb));
const peak = computed(() => fraction(props.peakDb));
const levelFalling = useFalling(() => level.value);
const peakFalling = useFalling(() => peak.value);
const tone = computed(() => ((props.peakDb ?? -120) > -1 ? 'clip' : (props.peakDb ?? -120) > -6 ? 'hot' : 'ok'));
const text = computed(() => (props.peakDb === null || props.peakDb <= props.floor ? '—' : `${Math.round(props.peakDb)} dB`));
</script>

<template>
  <div class="meter" :class="tone" role="meter" :aria-label="label" :aria-valuemin="floor" aria-valuemax="0" :aria-valuenow="peakDb ?? floor">
    <div class="track">
      <div class="fill" :class="{ falling: levelFalling }" :style="{ '--level': level }" />
      <div class="peak-rail" :class="{ falling: peakFalling }" :style="{ '--peak': peak }"><div class="peak" /></div>
    </div>
    <span v-if="showValue" class="value type-label-medium">{{ text }}</span>
  </div>
</template>

<style scoped>
.meter { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.track { position: relative; flex: 1; min-width: 48px; height: 8px; border-radius: var(--md-sys-shape-corner-full); background: var(--md-sys-color-surface-container-highest); overflow: hidden; }
/* Moved with transform (no layout per update): the fill slides in from the left, the rail carries the peak tick. */
.fill {
  position: absolute; inset: 0; border-radius: inherit; background: var(--md-sys-color-primary);
  transform: translateX(calc((var(--level) - 1) * 100%));
  transition: transform var(--md-sys-motion-duration-short2) var(--md-sys-motion-easing-emphasized-decelerate), background-color var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard);
}
.fill.falling { transition-duration: var(--md-sys-motion-duration-medium2), var(--md-sys-motion-duration-short4); }
.peak-rail { position: absolute; inset: 0; transform: translateX(calc(var(--peak) * 100%)); transition: transform var(--md-sys-motion-duration-short2) var(--md-sys-motion-easing-emphasized-decelerate); pointer-events: none; }
.peak-rail.falling { transition-duration: var(--md-sys-motion-duration-long2); }
.peak { position: absolute; top: 0; bottom: 0; left: 0; width: 2px; margin-left: -1px; background: var(--md-sys-color-on-surface); }
.hot .fill { background: var(--md-sys-color-tertiary); }
.clip .fill { background: var(--md-sys-color-error); }
.value { flex: none; width: 52px; text-align: end; font-variant-numeric: tabular-nums; color: var(--md-sys-color-on-surface-variant); }
</style>
