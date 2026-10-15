<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { tokenColor } from '@/ui/useTokenColor';

/**
 * A waveform drawn on a canvas from server peaks (interleaved min and max, -1..1), with a playback
 * cursor. The browser never decodes the file to draw it. Clicking seeks.
 */
const props = withDefaults(defineProps<{ peaks?: number[] | null; progress?: number; height?: number; label?: string; muted?: boolean }>(), { peaks: null, progress: 0, height: 48, label: '' });
const emit = defineEmits<{ seek: [fraction: number] }>();
const canvas = ref<HTMLCanvasElement | null>(null);
let observer: ResizeObserver | null = null;

function draw() {
  const c = canvas.value;
  if (!c) return;
  const ctx = c.getContext?.('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const width = c.clientWidth || 300;
  const height = props.height;
  if (c.width !== Math.round(width * dpr)) c.width = Math.round(width * dpr);
  if (c.height !== Math.round(height * dpr)) c.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const played = tokenColor(c, props.muted ? '--md-sys-color-outline' : '--md-sys-color-primary');
  const rest = tokenColor(c, '--md-sys-color-outline-variant');
  const peaks = props.peaks;
  const mid = height / 2;
  if (!peaks || peaks.length < 2) {
    ctx.fillStyle = rest;
    ctx.fillRect(0, mid - 0.5, width, 1);
    return;
  }
  const points = peaks.length / 2;
  const bar = 2;
  const gap = 1;
  const bars = Math.max(1, Math.floor(width / (bar + gap)));
  const cut = props.progress * bars;
  for (let i = 0; i < bars; i++) {
    const a = Math.floor((i / bars) * points);
    const b = Math.max(a + 1, Math.floor(((i + 1) / bars) * points));
    let lo = 0;
    let hi = 0;
    for (let p = a; p < b && p < points; p++) {
      lo = Math.min(lo, peaks[2 * p]);
      hi = Math.max(hi, peaks[2 * p + 1]);
    }
    const top = mid - Math.max(0.5, hi * mid);
    const h = Math.max(1, (hi - lo) * mid);
    const x = i * (bar + gap);
    // The bar under the cursor is split, so the cursor moves smoothly rather than a bar at a time.
    const done = Math.min(1, Math.max(0, cut - i));
    ctx.fillStyle = rest;
    if (done < 1) ctx.fillRect(x, top, bar, h);
    ctx.fillStyle = played;
    if (done > 0) ctx.fillRect(x, top, bar * done, h);
  }
}

function onClick(event: MouseEvent) {
  const c = canvas.value;
  if (!c) return;
  const rect = c.getBoundingClientRect();
  if (rect.width > 0) emit('seek', Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)));
}

function onKey(event: KeyboardEvent) {
  const step = event.key === 'ArrowRight' ? 0.05 : event.key === 'ArrowLeft' ? -0.05 : 0;
  if (step) emit('seek', Math.min(1, Math.max(0, props.progress + step)));
}

onMounted(() => {
  draw();
  if (typeof ResizeObserver !== 'undefined' && canvas.value) {
    observer = new ResizeObserver(() => draw());
    observer.observe(canvas.value);
  }
});
onBeforeUnmount(() => observer?.disconnect());
watch(() => [props.peaks, props.progress, props.muted, props.height], draw);
</script>

<template>
  <canvas
    ref="canvas"
    class="waveform"
    :style="{ height: `${height}px` }"
    role="slider"
    tabindex="0"
    :aria-label="label"
    aria-valuemin="0"
    aria-valuemax="100"
    :aria-valuenow="Math.round(progress * 100)"
    @click="onClick"
    @keydown="onKey"
  />
</template>

<style scoped>
.waveform { display: block; width: 100%; min-width: 0; cursor: pointer; border-radius: var(--md-sys-shape-corner-extra-small); }
</style>
