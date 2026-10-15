<script setup lang="ts">
import { computed } from 'vue';

export interface Series {
  label: string;
  points: [number, number][];
}

/** A plain SVG line chart: a few series, shared axes, a legend. For loss curves. */
const props = withDefaults(defineProps<{ series: Series[]; xLabel?: string; yLabel?: string; height?: number; emptyText?: string }>(), { xLabel: '', yLabel: '', height: 240, emptyText: '' });
const W = 640;
const PAD = { l: 48, r: 12, t: 12, b: 28 };
const COLORS = ['primary', 'tertiary', 'secondary', 'error', 'outline'];

const bounds = computed(() => {
  const all = props.series.flatMap((s) => s.points);
  if (!all.length) return null;
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const [x, y] of all) {
    if (!Number.isFinite(y)) continue;
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;
  return { x0, x1, y0, y1 };
});

const sx = (x: number) => (bounds.value ? PAD.l + ((x - bounds.value.x0) / (bounds.value.x1 - bounds.value.x0)) * (W - PAD.l - PAD.r) : 0);
const sy = (y: number) => (bounds.value ? PAD.t + (1 - (y - bounds.value.y0) / (bounds.value.y1 - bounds.value.y0)) * (props.height - PAD.t - PAD.b) : 0);

/** At most 400 points per line: a long run is thinned by stride, keeping the last point. A point
 * whose value is not finite (NaN) breaks the line, so a gap is not drawn across. */
function path(points: [number, number][]): string {
  const stride = Math.max(1, Math.ceil(points.length / 400));
  const kept = points.filter((_, i) => i % stride === 0 || i === points.length - 1);
  let out = '';
  let pen = false;
  for (const [x, y] of kept) {
    if (!Number.isFinite(y)) {
      pen = false;
      continue;
    }
    out += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)} `;
    pen = true;
  }
  return out.trim();
}

const ticks = computed(() => {
  const b = bounds.value;
  if (!b) return { x: [], y: [] };
  const n = 4;
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3));
  return {
    x: Array.from({ length: n + 1 }, (_, i) => b.x0 + ((b.x1 - b.x0) * i) / n).map((v) => ({ pos: sx(v), text: v.toFixed(0) })),
    y: Array.from({ length: n + 1 }, (_, i) => b.y0 + ((b.y1 - b.y0) * i) / n).map((v) => ({ pos: sy(v), text: fmt(v) })),
  };
});
</script>

<template>
  <figure class="chart">
    <svg v-if="bounds" :viewBox="`0 0 ${W} ${height}`" preserveAspectRatio="none" role="img" :aria-label="series.map((s) => s.label).join(', ')">
      <g class="grid">
        <line v-for="t in ticks.y" :key="`y${t.pos}`" :x1="PAD.l" :x2="W - PAD.r" :y1="t.pos" :y2="t.pos" />
      </g>
      <g class="labels">
        <text v-for="t in ticks.y" :key="`yl${t.pos}`" :x="PAD.l - 6" :y="t.pos + 4" text-anchor="end">{{ t.text }}</text>
        <text v-for="t in ticks.x" :key="`xl${t.pos}`" :x="t.pos" :y="height - 8" text-anchor="middle">{{ t.text }}</text>
      </g>
      <path v-for="(s, i) in series" :key="s.label" :d="path(s.points)" class="line" :style="{ stroke: `var(--md-sys-color-${COLORS[i % COLORS.length]})` }" />
    </svg>
    <p v-else class="type-body-medium muted empty">{{ emptyText }}</p>
    <figcaption class="legend type-label-medium">
      <span v-for="(s, i) in series" :key="s.label" class="key"><span class="swatch" :style="{ background: `var(--md-sys-color-${COLORS[i % COLORS.length]})` }" />{{ s.label }}</span>
      <span v-if="xLabel" class="muted axis">{{ xLabel }}</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.chart { margin: 0; display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
svg { width: 100%; height: auto; aspect-ratio: 640 / 240; overflow: visible; }
.grid line { stroke: var(--md-sys-color-outline-variant); stroke-width: 1; vector-effect: non-scaling-stroke; }
.labels text { fill: var(--md-sys-color-on-surface-variant); font-size: 11px; }
.line { fill: none; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.legend { display: flex; flex-wrap: wrap; gap: var(--app-space-3); align-items: center; }
.key { display: inline-flex; align-items: center; gap: var(--app-space-1); }
.swatch { width: 12px; height: 3px; border-radius: 2px; }
.axis { margin-left: auto; }
.empty { padding: var(--app-space-6); text-align: center; }
</style>
